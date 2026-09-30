# BLOCKERS — what must be implemented before any of these scripts can produce science

**Verified by running `python deploy/shared/preflight_blockers.py` (last: 2026-09-29, Windows).**
That script is the authority, not this document. Re-run it whenever you want the truth.

Every stage runner in this repository works today — **on the mock model.** Hand any of them
a real model and it raises `NotImplementedError`. The deployment scripts in
`plan_a_local_pc/` and `plan_b_dgx_spark/` therefore call the preflight first and refuse to
start, rather than failing four hours into a queue at night.

Current state (2026-09-30): **all engineering is done.** On
the Spark the preflight reads **10 of 10 OK** (2026-09-30, after B6 passed; its pass record
is committed, so a fresh clone reads the same once its calibration caches match). On a machine without
the calibration caches Q7 also reads BLOCKED: it is data, and reads OK only where the caches
match `deploy/shared/calibration_fingerprints.json`.

Run `python deploy/shared/preflight_blockers.py` for the live status; this table is a
summary and can drift.

| ID | Item | Status | Zone |
|---|---|---|---|
| B1 | Matched-magnitude null on real models | **DONE** — `matched_magnitude.py` delegates to `compression/torch_weights.py`; PI-approved 2026-09-27 | 🔒 novelty |
| B2 | RTN compressor | **DONE** — `rtn.py::rtn_quantize_torch`, per-output-channel, 2026-09-27 | engineering |
| B2 | Magnitude pruner | **DONE** — `magnitude_prune.py::prune_magnitude_torch`, 2026-09-27 | engineering |
| B2 | GPTQ compressor | **DONE** — `gptq.py`, port of IST-DASLab/gptq @ `2d65066e`, agrees with its `fasterquant`, 2026-09-29 | engineering |
| B2 | AWQ compressor | **DONE** — `awq.py`, port of mit-han-lab/llm-awq @ `d6e797a4`, agrees with its quantizer, clip and scale search, 2026-09-29 | engineering |
| B2 | Wanda pruner | **DONE** (code) — `magnitude_prune.py::prune_wanda_torch`, masks identical to upstream's own function, 2026-09-29 | engineering |
| Q7 | Calibration caches (Wanda, GPTQ, AWQ) | **DONE on the Spark** 2026-09-29 — fineweb-edu @ `87f09149…`, fingerprints recorded in `deploy/shared/calibration_fingerprints.json` | data |
| B3 | Chance-floor universe | **DONE** — `run_stage_c.py::_chance_floor_universes`, structural and per-level, 2026-09-27 | 🔒 novelty |
| B4 | Normalised L1 | **DONE** — `distances.py::normalized_l1_distance`, approved 2026-09-12 | 🔒 novelty |
| B6 | GPT-2 IOI exit gate | **PASSED** 2026-09-30 on seeds 5-9: precision 0.725 (>= 0.3), p 1.27e-45 (<= 0.001). The earlier Jaccard gate failed (0.028, size-capped) and was replaced post-hoc (HUMAN_DECISIONS.md Step 7) | - |

**Wanda is unblocked on the Spark: code and data.** The four caches were built 2026-09-29
(fineweb-edu `sample-10BT` @ `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, 300k tokens,
**seed 7**, 1,171 x 256 windows each) with

    python experiments/build_calibration_cache.py mode=scientific_run model=<name>

Their fingerprints are tracked in `deploy/shared/calibration_fingerprints.json`, and preflight
blocks any cache that differs from that record - including a rebuild whose own sidecar is
self-consistent. Pythia-160m and -410m share a tokenizer and their two independent builds
came out byte-identical, so the seeded streaming shuffle is deterministic at a fixed
dataset commit. The first Wanda cell collects E[x^2] once per model and every later cell,
in either stage, reuses the same file.

**GPTQ and AWQ are unblocked on the Spark: code and data.** Both are torch ports of the
authors' reference code at pinned commits (no `auto-gptq`/`autoawq`, which do not build on
aarch64), calibrated on the same Q7 cache as Wanda, forward in bf16 as Q7 pre-registers.
`tests/test_gptq_awq_torch.py` runs verbatim excerpts of the reference code
(`tests/reference_gptq.py`, `tests/reference_awq.py`) beside the ports and requires them to
agree. The grid pre-registers only `bits: 4`; every other setting is the reference's own:

- **GPTQ** (`llama.py` run without flags): per-row, asymmetric, min-max grid, no groups,
  no act-order, blocksize 128, percdamp 0.01; layers in order, each calibrated on the
  output of the already-quantized layers before it.
- **AWQ** (README / paper INT4 setting): zero-point, groups of 128 input channels, 20-point
  scale search, clipping searched on 512 sampled tokens and skipped for Q and K (and for V
  on Pythia, whose fused QKV the reference skips by name). Scaling groups follow the
  reference per architecture; the V -> O group never applies to our four models (GQA, or
  GPT-NeoX). Scales are folded into the quantized projections themselves, so - as for every
  family - only projection matrices change and the null is matched to that same set.

**Note the two methods do not share a grid**: GPTQ quantizes per row, AWQ per group of 128.
That is each method as published; a common grid would be a pre-registration change.

Each cell's compressed weights are computed once per model and cached under
`calibration_cache/compressed/`, so Stage B's `weight_delta` and Stage C's `apply` read the
same compression. Budget disk for it: one copy of the projection weights per method per
model, float32 - roughly 8 GB each for Gemma-2-2B, 4 GB for Llama-3.2-1B, 1-2 GB for the
Pythias. The first GPTQ/AWQ cell of each model does the computation; later cells reuse it.

**The collector does not read `ln1/ln2.hook_normalized`.** TransformerLens fires that hook
*before* the norm's affine `* w + b`, so it equals the projection input only for untrained
norms. Our models load with `fold_ln=False`, so reading it would have been wrong on every real
model without raising. The collector hooks the attention/MLP module inputs instead; see
`src/calibration/second_moment.py`.

**B3 was the silent one and its fix changed two numbers.** The universe is now the structural
EAP universe intersected with the observed nodes, and there is one N *per comparison level*.
Measured effect: the exact-edge N fell ~10x (10.70x pythia-160m, 10.52x pythia-410m, 11.03x
llama-3.2-1b, 9.95x gemma-2-2b) and the routing-head N fell 2400x-8300x, because
`project_to_routing_heads` maps onto heads (144 for pythia-160m) and not onto edges. The
routing-head chance floor moves from ~1e-5 to ~0.025 — that is the level claim C2 is stated at.

🔒 = Novelty Protection Zone (`AI_RULES.md` §3). **No AI agent may modify these without your
explicit written approval**, recorded in `docs/HUMAN_DECISIONS.md`.

---

## B1 — Matched-magnitude null on real models 🔒

**File:** `src/science/matched_magnitude.py`
**Probe result:** `NotImplementedError: MatchedMagnitudePerturber currently supports MockModel`

This is the null model. It is the contribution. Nothing downstream means anything without it,
and it is the single highest-value item in this list.

The numpy maths (`generate_null_deltas`, `frobenius_norm`) is correct and tested — **do not
rewrite it.** Add a torch path beside it:

- Input: a `HookedTransformer` from `real_model.py::load_pinned_model`, plus the
  `PerTensorFrobenius` map returned by the matching compressor's `weight_delta()`.
- Draw `ΔW ~ N(0, I)` per tensor, rescale so `||ΔW||_F` **exactly** equals that tensor's
  target, add it.
- **Never mutate the loaded model.** Deep-copy the state dict, perturb the copy, load into a
  fresh instance (`ARCHITECTURE.md` §2).
- Seed the RNG from `(run seed, draw index r)` so draw `r` is reproducible from the run name.
- **Perturb exactly the tensor set the compressor reported — no more, no less.** Matching on a
  different tensor set is the easiest way to silently make the null meaningless.

**Test:** per-tensor norms match to tolerance; original model unchanged; same `(seed, r)`
reproduces, different `r` differs.

---

## B2 — The five compressors on real models

**Files:** `src/compression/{rtn,gptq,awq,magnitude_prune,wanda}.py`
**Probe result:** all five raise `NotImplementedError` on a non-mock model.

Protocol (`src/interfaces.py`):

```python
def apply(self, model: Model, cfg: Any) -> CompressedModel: ...
def weight_delta(self, model: Model, cfg: Any) -> PerTensorFrobenius: ...
```

`weight_delta` feeds B1, so it must be the **measured** Frobenius norm of
`W_compressed − W_dense` per tensor, not a theoretical prediction.

Build order, easiest first:

1. **RTN** — `rtn_quantize()` already exists and is tested. Iterate the torch state dict,
   quantize per output channel, write back as float32. You are studying quantization *error*,
   not deploying INT4, so simulated quantization is correct. Bits 8, 6, 4.
2. **Magnitude** — zero the smallest-|W| fraction per tensor. Sparsities 0.2, 0.4, 0.6.
3. **Wanda** — DONE 2026-09-29. Score `|W| · sqrt(E[x^2])`, thresholded **per matrix**, as
   upstream's revision does (`reprune.py`: "per-matrix scores"). The Wanda paper
   (arXiv:2306.11695) and upstream's notebook compare within each output row instead; the
   revision is what produced the results C5 correlates against, so we follow it and report
   the deviation from the paper.
4. **GPTQ**, 5. **AWQ** — DONE 2026-09-29, ported in torch (see the note below and the
   settings above).

> **Magnitude and Wanda are torch reimplementations of `saediag.pruning`**, not wrappers:
> TransformerLens has no `nn.Linear` modules for upstream's functions to act on. The fork is
> at `../sae-pruning-paper-main` (read-only), and `tests/test_wanda_torch.py` runs upstream's
> own `prune_wanda_style_inplace` on the equivalent HF layout and requires identical masks.

> **GPTQ/AWQ and ARM.** `auto-gptq` and `autoawq` ship x86-64 wheels and commonly fail to
> build on the Spark's aarch64. **Recommendation: implement both directly in torch** (~150
> lines each) so both machines run identical code and the build risk disappears. GPTQ's core
> is a Hessian-based error-compensating round over columns; AWQ's is per-channel scaling
> chosen by activation magnitude.

---

## B3 — Chance-floor universe 🔒 (the silent one)

**Files:** `src/science/chance_floor.py` and the Stage C call site.
**Probe result:** `chance_floor_universe_implemented: false` in `configs/config.yaml`.

**This one does not crash. It produces results that look better than they are.**

The decision (recorded 2026-09-14) is
`chance_floor_universe: structural_edges_among_observed_nodes`. Stage C still computes
`U*(U-1)`, which counts pairs that can never be edges — inputs that never send, and backwards
flows. That is **8.7× too many on IOI and 6.3× on greater-than**, lowering the random-overlap
floor roughly 10×.

Fix: use `src/extraction/eap.py::candidate_edges(cfg)`, which already enumerates the
structurally possible pairs. Replace the computation in `run_stage_c.py` and flip the flag to
`true` in the same commit.

---

## B6 — GPT-2 IOI exit gate

**File:** `tests/test_regression_ioi_gpt2_small.py`.

Until it passes, **no result counts.** It extracts the IOI circuit from GPT-2 small and
checks it recovers the published circuit - validating the extraction stack against a known
reference. Without it, a null result is indistinguishable from a broken extractor.

**What opens B6 (changed 2026-09-29).** The preflight used to search the test file for the
word "skip", which the file always contains, so B6 could never have read OK. It now reads
the record the gate writes when it passes (`data/reference/ioi_gpt2_small_gate_pass.json`)
and checks it against the reference file's current sha256 and your criterion.

**Done (engineering, 2026-09-30):**

- **Reference edges, derived, never typed.** `src/tasks/ioi_reference.py` ports ACDC's
  `get_ioi_true_edges` (acdc/ioi/utils.py @ `bc99ace8`): the Wang et al. circuit (26 heads,
  7 classes, their Q/K/V connections) as the edge-level ground truth ACDC and EAP are both
  scored against, in our dense-node ids. `tests/test_ioi_reference.py` runs ACDC's own code
  verbatim and requires the two sets to be equal: **963 edges**, the same set edge for edge,
  over the same 32,491-edge universe.
- **Extraction.** `configs/model/gpt2_small.yaml` at the pin (`607a30d7…`, architecture
  verified against its config.json). The gate runs Stage A exactly as every grid cell does
  (`GATE_OVERRIDES` = `_run_one.sh`'s overrides + model + task, held equal by a test):
  dense-node EAP, B=16 x S=5, 300 prompts. The extracted set is that run's **core band**.

**Criterion (PI-owned, HUMAN_DECISIONS.md Step 7).** The first criterion, Jaccard >= 0.2
on seeds 0-4, FAILED at 0.028: the core band had 37 edges (27 in the reference), and 37
edges cannot exceed a Jaccard of 37/963. The PI replaced it the same day, post-hoc and
disclosed: **precision >= 0.3 AND hypergeometric p <= 0.001, on fresh seeds 5-9**. The
file carries it:

    python deploy/shared/make_ioi_reference.py --min-precision 0.3 --max-p-value 0.001 \
        --seed 5 --decided-on 2026-09-30

The script has no defaults and will not overwrite an existing file. A file in the old
Jaccard format is refused by the gate.

**What it still needs:** on the Spark,
`RUN_IOI_GPT2_REFERENCE=1 python -m pytest tests/test_regression_ioi_gpt2_small.py -s`.
It runs Stage A on seeds 5-9 under `runs/`, prints precision, p, core size, shared edges and
Jaccard, and on a pass writes the record that opens B6. Each gate run deletes the previous
record before it extracts anything, so a later failing run closes B6 again.

Fixed on the way: the test looked for the reference file one directory *above* the repo,
so a file placed in `data/reference/` would never have been found.

Note the caveat in `CLAUDE.md` §6: published reference circuits exist for GPT-2 small, **not**
for Pythia, Gemma or Llama. The gate is a sanity check on the stack, not a claim about those
models.

---

## Missing dependencies (not code — files that are absent)

**Both upstream forks are present on the Windows machine** (verified 2026-09-29):
`circuit-tracer-0.5.2/` and `sae-pruning-paper-main/` sit at the workspace root beside the repo,
per `CLAUDE.md` §3. The C5 cross-audit input exists:
`sae-pruning-paper-main/results/E6/stat_tests.csv`. Their revisions have NOT been verified
against the pins below, and neither fork has been confirmed present on the Spark.

| Fork | URL (verified, `HUMAN_DECISIONS.md` §3.3) | Pin | Needed for |
|---|---|---|---|
| circuit-tracer | `github.com/decoderesearch/circuit-tracer` | `8f1e2438…` | Pipeline A (Gemma/Llama attribution graphs) |
| sae-pruning-paper | `github.com/hecboar/sae-pruning-paper` | `261191804675…` ⚠️ **inferred** | `saediag.pruning`; the C5 cross-audit frozen tables |

Fetch with `deploy/shared/fetch_upstream_forks.ps1` (Windows) or `.sh` (Spark).

> `safety-research/circuit-tracer` 301-redirects to the `decoderesearch` URL — the latter is
> canonical, per the BibTeX, verified 2026-09-12.

> **The cross-audit depends on `sae-pruning-paper/results/E6/stat_tests.csv`, not the arXiv
> PDF.** The authors corrected their headline: ρ = −1.0 became a range of **−0.540 to +0.062**,
> and Gemma-2-2B's dense WikiText-2 perplexity went from 410 to **8.21**. Quoting the PDF puts
> a walked-back number in your paper, and a reviewer who opens the repo will find it.
> (`docs/Run_Plan.md` §0.2.)

---

## Decisions still open (these block `mode=scientific_run`)

`src/common/config_guard.py` refuses a scientific run while any gating question is open.
Q1, Q2, Q4, Q5, Q6, Q7, Q10, Q11 are pre-registered. Outstanding:

- **C4** — the interaction-flag ratio (when a component counts as interaction-dominated and
  is excluded from headline claims)
- **C5** — grid alignment for the cross-audit
- **Attribution precision for Gemma-2 / Llama-3.2** (float32 is settled for Pythia)
- **Freeze ownership** — which machine owns which model's freeze, if both are running
  (Plan B, top). Record it *before* either machine runs Stage B.

Record each in `docs/HUMAN_DECISIONS.md` with a date preceding the run that depends on it.
