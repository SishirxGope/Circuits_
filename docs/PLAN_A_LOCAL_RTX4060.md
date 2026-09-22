# Plan A — Complete the project on the local PC (RTX 4060, 8 GB)

**Scope decision (2026-09-21):** this plan delivers a **complete, self-contained result on
Pythia-160M and Pythia-410M only**. Gemma-2-2B and Llama-3.2-1B are explicitly deferred to
Plan B (DGX Spark). This is not a reduced-quality result — it is the full Algorithm 1
(Stage A → B → freeze → C → D), the full null model, both comparison levels, the threshold
sweep and the chance floor, on the two models that fit in 8 GB.

**What this plan is not:** it is not a plan to run the primary models locally. Gemma-2-2B in
float32 needs roughly 25 GiB and Llama-3.2-1B roughly 39 GiB at Pythia's batch size
(*estimates*, `docs/BOTTLENECKS_AND_HARDWARE.md`). They do not fit, and bf16 workarounds would
break the pre-registered float32 decision.

**Companion:** [`PLAN_B_DGX_SPARK.md`](PLAN_B_DGX_SPARK.md) — the full grid on a DGX Spark, all
four models. Plan B also covers Pythia, so if the Spark arrives before this plan finishes you
can hand the remaining Pythia cells over. **If you do, read Plan B's freeze-ownership rule
first** — both machines freezing the same cell is the one mistake that cannot be undone.

**Just want the steps?** [`../EXECUTION_GUIDE.md`](../EXECUTION_GUIDE.md) has every command
for both machines in order. This document explains *why* each step is what it is.

**Executable version:** [`../deploy/plan_a_local_pc/`](../deploy/plan_a_local_pc/) holds a
numbered script per stage, the generated run queues, and a checklist. This document is the
reasoning; that folder is the doing. Start with `00_repair_and_verify.ps1`.

---

## Part 0 — The honest starting position

Read this before planning your time. Four things are true right now:

1. **Every stage runner exists** — `run_stage_a/b/c/d.py` and `freeze_stage_b.py` are
   written, tested, and work end-to-end on the mock model.
2. **No real-model compression exists.** All five compressors (`rtn.py`, `gptq.py`,
   `awq.py`, `magnitude_prune.py`, `wanda.py`) raise `NotImplementedError` the moment they
   are handed anything that is not a `MockModel`. The same is true of
   `MatchedMagnitudePerturber` in `src/science/matched_magnitude.py`, which is the null model
   itself.
3. **One pre-registered quantity is not implemented**: the chance-floor universe
   (`chance_floor_universe_implemented: false` in `configs/config.yaml` — Stage C still
   computes the ~10× too lenient `U*(U-1)`). It fails *silently*, which makes it the most
   dangerous item here. (Normalised L1 **is** implemented — see B4.)
4. **The compression grid configs now exist** — 11 cells in `configs/compression/`, generated
   2026-09-21 and verified to compose through Hydra.

So Plan A is: **write five things, close four decisions, then run four stages.** The running
is the cheap part.

---

## Part 1 — Environment and repository verification

Run all of this from `C:\Users\sishi\Circuits_Under_Compression` in PowerShell.

### 1.1 Repair the git index (do this first)

The index is currently empty — `git ls-files` returns 0 while HEAD holds 156 files. Your
files and commits are all intact; only the index was lost. Rebuild it from HEAD:

```powershell
git reset
git status --short          # expect: a clean tree, or only genuine edits
git ls-files | Measure-Object -Line    # expect: 156
```

### 1.2 Get the chat export out of the repo

`claude-chat-3ad0752e-2026-09-20-raw.jsonl` (5.4 MB) is sitting in the project root and must
not enter the AAAI artifact:

```powershell
Move-Item claude-chat-3ad0752e-2026-09-20-raw.jsonl $env:USERPROFILE\Documents\
```

### 1.3 Verify the environment

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv\Scripts\python.exe -c "import transformer_lens, transformers, hydra; print('deps ok')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Expected: CUDA `True`, an RTX 4060 with 8188 MiB.

### 1.4 The upstream forks are missing from this machine

**Verified 2026-09-21: neither fork is on disk.** `CLAUDE.md` §3 places both at the workspace
root beside the repo; they are absent, and `saediag` does not import.

| Fork | URL (verified, `HUMAN_DECISIONS.md` §3.3) | Pin | Needed for |
|---|---|---|---|
| circuit-tracer | `github.com/decoderesearch/circuit-tracer` | `8f1e2438…` | Pipeline A (Gemma/Llama) |
| sae-pruning-paper | `github.com/hecboar/sae-pruning-paper` | `261191804675…` ⚠️ inferred | `saediag.pruning`; the C5 cross-audit tables |

```powershell
.\deploy\shared\fetch_upstream_forks.ps1
```

This matters for B2: the magnitude and Wanda stubs say they wrap `saediag.pruning`, so decide
deliberately whether to wrap it or implement in torch.

### 1.5 The test suite is your regression gate

```powershell
.venv\Scripts\python.exe -m pytest -q
```

**Expected: 511 passed, 1 skipped.** The single skip is the Phase-1 exit gate, by design.
Re-run this after every change in Part 2. If the count drops, stop and fix before continuing —
this suite is the only thing standing between you and a silently wrong result.

---

## Part 2 — The six things that must be built

These are ordered by dependency. Items marked 🔒 are in the **Novelty Protection Zone**
(`AI_RULES.md` §3): an AI agent may not modify them without your explicit written approval,
and the approval belongs in `docs/HUMAN_DECISIONS.md`.

### B1 🔒 — Real-model matched-magnitude perturbation

**File:** `src/science/matched_magnitude.py`
**Why first:** this *is* the null model. Nothing downstream means anything without it.

Current state: `MatchedMagnitudePerturber.perturb()` handles `MockModel` (a numpy dict) and
raises `NotImplementedError` for everything else. The numpy maths in `generate_null_deltas`
and `frobenius_norm` is correct and tested — **do not rewrite it.** Add a torch path beside it.

Specification:

- Input: a loaded `HookedTransformer` (from `src/extraction/real_model.py::load_pinned_model`)
  and a `PerTensorFrobenius` mapping `{tensor_name: target_frobenius_norm}` produced by the
  matching compressor's `weight_delta()`.
- For each named weight tensor, draw `ΔW ~ N(0, I)` of the same shape, rescale it so that
  `||ΔW||_F` equals the target norm for that tensor exactly, and add it.
- **Never mutate the loaded model in place** (`ARCHITECTURE.md` §2). Deep-copy the state dict,
  perturb the copy, load it into a fresh model instance, and return that.
- The RNG must be seeded from the run's `seed` and the draw index `r`, so draw `r` is
  reproducible from the run name alone.
- Exclude non-weight tensors: layer-norm gains/biases, embeddings if the compressor did not
  touch them. **Rule: perturb exactly the tensor set the compressor reported in
  `weight_delta()`, no more and no less.** Matching on a different tensor set is the single
  easiest way to make the null meaningless.

Test to add (`tests/test_matched_magnitude_real.py`): on a small real model, assert
per-tensor `||ΔW||_F` matches the target to within floating-point tolerance, assert the
original model is unchanged, and assert two draws with the same `(seed, r)` are identical
while different `r` differ.

### B2 — Real-model compressors (all five)

**Files:** `src/compression/{rtn,gptq,awq,magnitude_prune,wanda}.py`

Each must satisfy the `Compressor` protocol in `src/interfaces.py`:

```python
def apply(self, model: Model, cfg: Any) -> CompressedModel: ...
def weight_delta(self, model: Model, cfg: Any) -> PerTensorFrobenius: ...
```

`weight_delta` is what feeds B1, so it must return the per-tensor Frobenius norm of the
*actual* `W_compressed − W_dense`, measured after compression, not predicted from theory.

Build in this order:

| Order | Family | Effort | Notes |
|---|---|---|---|
| 1 | **RTN** | Low | `rtn_quantize()` already exists and is tested. Wrap it: iterate the torch state dict, quantize per output channel, write back as float32 (simulated quantization — you are studying the *error*, not deploying INT4). Bit-widths 8, 6, 4. |
| 2 | **Magnitude pruning** | Low | Zero the smallest-|W| fraction per tensor. Sparsities 0.2, 0.4, 0.6. Pure torch; `magnitude_prune.py` already has 290 lines of tested numpy logic to mirror. |
| 3 | **Wanda** | Medium | Needs calibration activations: score = `|W| * ||X||_2` per input channel, pruned per output row. Requires a forward pass over the FineWeb-Edu calibration set (`configs/calibration/final.yaml`: 300,000 tokens, seed 7) with hooks capturing input norms. Reference: arXiv:2306.11695 — **verify against the paper before committing, do not code from memory.** |
| 4 | **GPTQ** | High | External dependency. See the warning below. |
| 5 | **AWQ** | High | External dependency. See the warning below. |

> **⚠️ GPTQ and AWQ — decide the dependency question before you write either.**
> You chose to cover all five families. On the local PC `auto-gptq` / `autoawq` will
> generally install on x86-64 + CUDA, but **on the DGX Spark (ARM64) they frequently fail to
> build** (Plan B §4). To keep both machines running the same code, prefer a
> **self-contained implementation in pure torch** over the upstream packages: GPTQ's core is
> a Hessian-based error-compensating round over columns, AWQ's is per-channel scaling chosen
> by activation magnitude. Both are implementable in ~150 lines and remove the entire ARM
> build risk. If you instead use the packages, `AI_RULES.md` §7 requires the
> `# Adapted from: <repo URL> @ <commit>, <license>` header and a copy of the license in
> `THIRD_PARTY_LICENSES/`.

Each compressor needs a test asserting that `weight_delta()` equals the measured norm of the
difference between the dense and compressed state dicts.

### B3 🔒 — The chance-floor universe

**File:** `src/science/chance_floor.py`, plus the Stage C call site.

`configs/config.yaml` records the decision (2026-09-14) and flags it unimplemented:

```yaml
chance_floor_universe: structural_edges_among_observed_nodes
chance_floor_universe_implemented: false   # run_stage_c still computes U*(U-1)
```

The current `U*(U-1)` counts pairs that can never be edges — inputs that never send, and
backwards flows — making it **8.7× too many on IOI and 6.3× on greater-than**, which lowers
the random-overlap floor by roughly 10×. A floor that is 10× too lenient will make noise look
like signal.

Fix: use `src/extraction/eap.py::candidate_edges(cfg)`, which already enumerates the
structurally possible pairs for the dense-node pipeline. Replace the `U*(U-1)` computation in
`run_stage_c.py`, then set the flag to `true` in the same commit.

### B4 🔒 — Normalised L1 — **ALREADY DONE**

**File:** `src/science/distances.py`

Q2 pre-registered three reported distances: L1 primary, Jensen-Shannon ablation, and
normalised L1 (L1 ÷ |edge union|) alongside.

**Correction (verified 2026-09-21):** this *is* implemented —
`distances.py::normalized_l1_distance`, a novelty-zone edit approved 2026-09-12. The summary
row in `docs/HUMAN_DECISIONS.md` Q2 that says otherwise is stale. Nothing to build; confirm
Stage C actually reports the column.

### B5 — The compression grid configs — **DONE 2026-09-21**

**Directory:** `configs/compression/` — 11 cells generated and verified to compose through
Hydra:

```
rtn_int8  rtn_int6  rtn_int4  gptq_int4  awq_int4
magnitude_20  magnitude_40  magnitude_60
wanda_20  wanda_40  wanda_60
```

**That is 11 cells.** Remember this number — it drives the entire compute budget, because the
matched-magnitude null is matched *per compression cell*, so Stage B must run R draws for
each one.

They are generated from one table in `deploy/shared/gen_cells.py`, which also emits the Stage
B and Stage C run queues, so a cell key cannot drift between the freeze and Stage C.

> **Hydra note.** These files populate `cfg.compression.*` only. They deliberately do **not**
> set the root keys `compression_family` / `compression_level` / `stage_c.cell`, because
> `configs/config.yaml` lists `_self_` **last** in its defaults, so its own
> `compression_family: dense` would override anything a group file set. The generated queue
> lines pass those roots explicitly. Verified by composition test, not assumed.

Configs are **immutable once referenced by a run** (`AI_RULES.md` 1.2); the generator refuses
to overwrite a changed one.

### B6 — The GPT-2 IOI exit gate

**Status: must pass before any result counts.** This is the Phase-1 gate currently skipped in
the test suite. Extract the IOI circuit from GPT-2 small with the real pipeline and check it
recovers the known published IOI heads. It validates the extraction stack against a published
reference; without it, a null result is indistinguishable from a broken extractor.

Note the caveat already recorded in `CLAUDE.md` §6: published reference circuits exist for
GPT-2 small, **not** for Pythia. The gate is a sanity check on the stack, not a claim about
Pythia.

---

## Part 3 — Decisions to close before `mode=scientific_run`

`src/common/config_guard.py` **refuses to run a scientific run while any gating question is
open.** Q1, Q2, Q4, Q5, Q6, Q7, Q10, Q11 are pre-registered. Outstanding:

| Item | What it is | Blocks |
|---|---|---|
| **C4** | The interaction-flag ratio — the threshold at which a component counts as interaction-dominated and is excluded from headline claims | Stage C reporting |
| **C5** | Grid alignment for the cross-audit | Stage D / `analysis/cross_audit.py` |
| **Attribution precision** | float32 confirmed for Pythia; the primary models are a Plan B question | Plan B only |
| **Docstring task** | Source verified, generator not ported to Pythia's tokenizer | Nothing — it is the third task and does not block the pilot |

> **⚠️ C5 carries a trap already documented in `docs/Run_Plan.md` §0.2.** The feature-level
> numbers the cross-audit correlates against were **corrected by their own authors**. The
> arXiv v1 abstract headlines ρ = −1.0; the authors' frozen record
> (`../sae-pruning-paper-main/results/E6/stat_tests.csv`) gives a range of **−0.540 to
> +0.062**. Correlate against the frozen CSV tables, never the PDF. Same for perplexities:
> Gemma-2-2B dense WikiText-2 PPL is **8.21**, not 410 and not ~11.

Record each closure in `docs/HUMAN_DECISIONS.md` with its date and reasoning, and commit
before the run that depends on it.

---

## Part 4 — Execution, stage by stage

All stage runners are Hydra entrypoints, invoked as `python experiments/run_stage_X.py
key=value ...` from the project root. `freeze_stage_b.py` is plain argparse.

### Stage A — the dense reference

Establishes the pre-compression inclusion-frequency vector every later distance is measured
against.

```powershell
.venv\Scripts\python.exe experiments\run_stage_a.py `
    mode=scientific_run model=pythia160m task=ioi `
    pipeline=dense-node ensemble=default nulls=default `
    stage=stageA setting=dense compression_family=dense `
    comparison_level=both seed=0
```

Repeat with `task=greater_than`. Then both again for `model=pythia410m`.

**Before the first scientific run, dry-run it:** the first invocation will try to download
model weights and datasets, which is gated by `mode.allow_model_download` and
`mode.allow_external_dataset_download`. Confirm the pinned revisions resolve
(`configs/model/pythia160m.yaml`) and that the loader's architecture check passes — that check
exists because the obvious TransformerLens call silently ignores the revision pin.

### Stage B — the null, then the freeze

**This is the pre-registration event. It is irreversible. Read this whole section before
running it.**

Run the null draws for each (model, task, compression cell):

```powershell
.venv\Scripts\python.exe experiments\run_stage_b.py `
    mode=scientific_run model=pythia160m task=ioi `
    compression=rtn_int4 pipeline=dense-node `
    ensemble=default nulls=default stage=stageB seed=0
```

That is **11 cells × 2 tasks × 2 models = 44 Stage B runs.** Script the loop; do not type them
by hand.

Then freeze. The freeze writer enforces four preconditions (`experiments/freeze_stage_b.py`):

1. `mode=scientific_run` — the freeze is not an engineering dry-run
2. `stage_b.freeze_approved=true` — explicit, no default, because freezing is irreversible
3. no gating questions open (Part 3)
4. `distance.pre_registered_for_stage_c=true` — because CSI is entirely downstream of D, a
   null frozen before D is chosen is a floor for nothing in particular

Always plan first:

```powershell
.venv\Scripts\python.exe experiments\freeze_stage_b.py --config <resolved_config.json> --dry-run
```

`stage_b.cells` must list every cell as `{model, task, cell, source_run_dir}`. Inspect the
printed plan, confirm all 44 cells are present and point at the right run directories, then
drop `--dry-run`.

**After this command, `frozen/` is append-only and may never be edited.** If you later
discover a bug in the null, you do not edit the freeze — you record the problem, freeze a new
version, and report both.

### Stage C — real compression

```powershell
.venv\Scripts\python.exe experiments\run_stage_c.py `
    mode=scientific_run model=pythia160m task=ioi `
    compression=rtn_int4 pipeline=dense-node `
    ensemble=default stage=stageC compression_family=rtn compression_level=4 `
    stage_c.cell=rtn_int4 null_frozen_hash=<hash from frozen/FREEZE_MANIFEST.json> seed=0
```

Stage C reads its frozen null from `frozen/{model}/{task}/{cell}` and refuses to run if the
hash does not match. Same 44 combinations.

### Stage D — aggregation, CSI table, FDR

```powershell
.venv\Scripts\python.exe experiments\run_stage_d.py `
    mode=scientific_run stage=stageD `
    stage_d.stage_c_dirs=[<list of Stage C run dirs>]
```

Produces the CSI table (`ARCHITECTURE.md` §2 schema) with bootstrap CIs resampled over B, S
and r, and Benjamini-Hochberg correction at q = 0.05 across the reported grid.

### Analysis

```powershell
.venv\Scripts\python.exe analysis\threshold_sweep.py    # does every conclusion survive the full threshold range?
.venv\Scripts\python.exe analysis\cross_audit.py        # Spearman(circuit damage, feature damage) — see the C5 trap
```

---

## Part 5 — Compute budget

From the measured pilot (`docs/PROGRESS_REPORT_2026-09-16.md` §5, RTX 4060, Pythia-160M):

| | IOI | Greater-than |
|---|---|---|
| One attribution pass | 42–48 s | 7.5–7.9 s |
| Peak GPU memory | 4.26 GiB | 2.64 GiB |

The cost driver is that **the matched-magnitude null is matched per compression cell**, so:

```
Stage B passes per (model, task) = 11 cells × R=20 draws × S=5 seeds = 1100 passes
Stage C passes per (model, task) = 11 cells × 2 (dense + compressed) × S=5 = 110 passes
Stage A passes per (model, task) = S=5
```

**Pythia-160M:**

| Stage | IOI | Greater-than |
|---|---|---|
| A | ~4 min | ~1 min |
| B | 1100 × 45 s ≈ **13.8 h** | 1100 × 7.7 s ≈ **2.4 h** |
| C | 110 × 45 s ≈ 1.4 h | 110 × 7.7 s ≈ 0.2 h |
| **Total** | **≈ 15.2 h** | **≈ 2.6 h** |

**Pythia-160M, both tasks: ≈ 18–19 GPU-hours**, which matches the figure already in the
progress report.

**Pythia-410M:** *estimate only.* It fits at reduced batch (about 4.8 GiB at batch 8 versus
about 14.5 GiB at batch 32), so expect **3–4× the Pythia-160M time: roughly 55–75 GPU-hours.**
Measure it with `experiments/time_attribution.py` before committing to a schedule rather than
trusting this range.

**Realistic wall-clock: 4–7 days of near-continuous GPU time** for both models and both tasks.
One GPU runs one cell at a time — an IOI pass peaks at 4.26 GiB, so two cells will not fit
side by side even though they are logically independent.

**Scope-cut order if you overrun** (fixed in `CLAUDE.md` §6, do not improvise): compression
levels and secondary models → task circuits → seeds. **Never cut the null.** Dropping R from
20 to 10 halves the dominant cost and is the correct first lever, but it must be decided and
recorded *before* the freeze, never after.

---

## Part 6 — Operating inside 8 GB

- **Batch size is the only real lever.** Peak is the activation cache, not the weights —
  4.26 GiB of cache against 0.61 GiB of weights on Pythia-160M.
- **Close everything using the GPU** before a run. Browsers and the VS Code GPU process can
  cost several hundred MiB.
- **On OOM:** halve the batch size, re-run. Never silently switch precision — float32 is
  pre-registered for Pythia, and changing it mid-grid makes cells incomparable.
- **Log peak memory per run** so Plan B's parallelism estimate rests on measurements.
- **Windows has no `CUDA_VISIBLE_DEVICES` subtlety here** — one GPU, index 0.

---

## Part 7 — Failure modes worth pre-empting

| Symptom | Likely cause | Response |
|---|---|---|
| Scientific run refuses to start | An open gating question | Read the guard's message; close the question in `HUMAN_DECISIONS.md`; commit |
| Freeze refuses | One of the four preconditions | Do not work around it — the precondition is the point |
| Stage C hash mismatch | Frozen null does not match the config | Confirm you are pointing at the right cell; never edit `frozen/` |
| Test count drops below 511 | A regression from Part 2 | Fix before running science |
| CSI ≈ 1 everywhere | Either the real finding, or a broken compressor | Check `weight_delta()` is non-zero and the compressed model's perplexity actually degraded |
| Results look too good | Chance floor still at `U*(U-1)` | Verify B3 landed and the flag is `true` |

The fifth row deserves emphasis. **CSI ≈ 1 is a publishable result** — it means compression
damage is indistinguishable from generic weight noise. Do not treat it as a bug and tune until
it goes away. That is precisely the failure mode the frozen null exists to prevent.

---

## Part 8 — Definition of done

Plan A is complete when all of these hold:

- [ ] 511+ tests pass; the GPT-2 IOI exit gate passes (no longer skipped)
- [ ] `frozen/` holds all 44 Pythia cells with hashes; `FREEZE_MANIFEST.json` verifies
- [ ] Stage C ran for all 44 cells against the frozen null
- [ ] The CSI table exists with bootstrap CIs over B, S, r and BH correction at q = 0.05
- [ ] Every conclusion is stated at **both** comparison levels (exact-edge and layer)
- [ ] The threshold sweep shows whether each conclusion survives the full threshold range
- [ ] The chance floor is attached to every overlap statistic, with B3 implemented
- [ ] Every number in the paper traces to a run directory in `runs/`
- [ ] `docs/HUMAN_DECISIONS.md` records C4 and C5 with dates preceding the runs that use them

---

## Part 9 — Suggested order of work

1. Part 1 (environment, index repair) — under an hour
2. B1 (null model) + its tests — the highest-value single item
3. B2 RTN and magnitude — the two easy compressors; enough to run a complete pilot
4. B3, B4, B5 — the pre-registered quantities and grid configs
5. **Run a full Pythia-160M / greater-than pilot end to end** (2.6 h, the cheap task) before
   building anything else. An end-to-end result on one model-task pair de-risks everything.
6. B2 Wanda, then GPTQ/AWQ
7. B6 exit gate
8. Close C4 and C5
9. Freeze, then the full grid

Step 5 is the important one. A complete cheap pass exposes integration problems that no amount
of unit testing will, and it does so before the freeze makes decisions permanent.
