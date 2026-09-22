# EXECUTION GUIDE

**The single operational document.** Every step to take this project from where it is today to
a finished result, on both machines, in order. If you only open one file, open this one.

- **Reasoning behind the plans:** [`docs/PLAN_A_LOCAL_RTX4060.md`](docs/PLAN_A_LOCAL_RTX4060.md) · [`docs/PLAN_B_DGX_SPARK.md`](docs/PLAN_B_DGX_SPARK.md)
- **What isn't built yet:** [`deploy/BLOCKERS.md`](deploy/BLOCKERS.md)
- **Scripts:** [`deploy/`](deploy/)

Last verified against the repository: **2026-09-21**.

---

## Contents

- [Part 0 — Status in one paragraph](#part-0--status-in-one-paragraph)
- [Part 1 — Do these today](#part-1--do-these-today-they-have-lead-times)
- [Part 2 — Build the six blockers](#part-2--build-the-six-blockers)
- [Part 3 — Run it on the PC](#part-3--run-it-on-the-pc-rtx-4060)
- [Part 4 — Run it on the DGX Spark](#part-4--run-it-on-the-dgx-spark)
- [Part 5 — Command reference](#part-5--command-reference)
- [Part 6 — When things go wrong](#part-6--when-things-go-wrong)
- [Part 7 — Rules that never bend](#part-7--rules-that-never-bend)

---

## Part 0 — Status in one paragraph

Every stage runner, config, queue and orchestration script is written and tested. **The science
cannot run yet** because the real-model code paths are stubs: the five compressors and the
matched-magnitude null raise `NotImplementedError` on anything that is not a mock model, and
the chance-floor universe is still the too-lenient `U*(U-1)`. Check the current truth any time:

```powershell
.venv\Scripts\python.exe deploy\shared\preflight_blockers.py
```

As of 2026-09-21 it reports **8 of 9 blocked**. Once it exits 0, one command runs the whole
pipeline.

---

## Part 1 — Do these today (they have lead times)

### 1.1 Accept the model licences — days of waiting, start now

- Gemma-2-2B: https://huggingface.co/google/gemma-2-2b
- Llama-3.2-1B: https://huggingface.co/meta-llama/Llama-3.2-1B

Create a read token at https://huggingface.co/settings/tokens, then:

```powershell
.venv\Scripts\python.exe -c "from huggingface_hub import HfApi; HfApi().model_info('google/gemma-2-2b'); print('gemma ok')"
.venv\Scripts\python.exe -c "from huggingface_hub import HfApi; HfApi().model_info('meta-llama/Llama-3.2-1B'); print('llama ok')"
```

`GatedRepoError` means approval hasn't arrived. Nothing works around it. **Pythia needs no
licence**, so all of Plan A can proceed while you wait.

### 1.2 Fetch the two missing upstream forks

Verified absent from this machine on 2026-09-21, though `CLAUDE.md` assumes them:

```powershell
.\deploy\shared\fetch_upstream_forks.ps1
```

Clones `circuit-tracer` (`decoderesearch`, pin `8f1e2438…`) and `sae-pruning-paper`
(`hecboar`, pin `261191804675…` ⚠️ inferred) beside the repo. Needed for pipeline A, for
`saediag.pruning`, and for the cross-audit's frozen tables.

### 1.3 Tidy the repo

```powershell
git reset                                                    # if git ls-files returns 0
Move-Item claude-chat-*.jsonl $env:USERPROFILE\Documents\     # 5 MB, keep it out of the artifact
```

### 1.4 Close the open decisions

`mode=scientific_run` **refuses to start** while any gating question is open. Outstanding:

| Decision | What it is |
|---|---|
| **C4** | Interaction-flag ratio — when a component is interaction-dominated and excluded from headline claims |
| **C5** | Grid alignment for the cross-audit |
| **Precision** | Attribution dtype for Gemma-2 / Llama-3.2 (float32 settled for Pythia) |
| **Freeze ownership** | Which machine owns which model's freeze, if both run Pythia |

Record each in `docs/HUMAN_DECISIONS.md` with a date **preceding** the run that uses it.

---

## Part 2 — Build the six blockers

Full specifications in [`deploy/BLOCKERS.md`](deploy/BLOCKERS.md). Order matters — this is
dependency order, and it front-loads the item everything else depends on.

| # | Item | File | Zone |
|---|---|---|---|
| 1 | Matched-magnitude null on real models | `src/science/matched_magnitude.py` | 🔒 needs your approval |
| 2 | RTN quantizer | `src/compression/rtn.py` | engineering |
| 3 | Magnitude pruner | `src/compression/magnitude_prune.py` | engineering |
| 4 | Chance-floor universe | `src/science/chance_floor.py` + Stage C | 🔒 needs your approval |
| 5 | Wanda pruner | `src/compression/wanda.py` | engineering |
| 6 | GPTQ + AWQ | `src/compression/{gptq,awq}.py` | engineering |
| 7 | GPT-2 IOI exit gate | `tests/test_regression_ioi_gpt2_small.py` | engineering |

🔒 = Novelty Protection Zone (`AI_RULES.md` §3): no AI modification without your written
approval in `docs/HUMAN_DECISIONS.md`.

**Normalised L1 is already done** — `distances.py::normalized_l1_distance`, approved
2026-09-12. The summary row in `HUMAN_DECISIONS.md` Q2 saying otherwise is stale.

After items 1–3, **stop and run a complete Pythia-160M / greater-than pass end to end**
(~2.6 h, the cheap task). A full cheap pass exposes integration problems that unit tests never
will, and it does so before the freeze makes anything permanent.

**Recommendation for GPTQ/AWQ:** implement both directly in torch (~150 lines each) rather
than using `auto-gptq`/`autoawq`. Those packages ship x86-64 wheels and commonly fail to build
on the Spark's ARM64. Your own implementation makes both machines run identical code and
removes the build risk entirely.

After each item:

```powershell
.venv\Scripts\python.exe -m pytest -q                        # must stay green
.venv\Scripts\python.exe deploy\shared\preflight_blockers.py  # watch items flip to OK
```

---

## Part 3 — Run it on the PC (RTX 4060)

**Scope:** Pythia-160M + Pythia-410M, both tasks, 11 compression cells, Stage A → D.
Gemma and Llama do not fit in 8 GB.

### The one command

```powershell
.\deploy\plan_a_local_pc\RUN_ALL.ps1
```

Runs everything in order, transcribes to `logs/run_all/`, and is **resumable** — re-run after
any crash, OOM or reboot and finished work is skipped. It pauses once, at the freeze.

Flags: `-SkipFetch`, `-StopBefore freeze`.

### Or step by step

```powershell
.\deploy\plan_a_local_pc\00_repair_and_verify.ps1    # ~2 min   index, GPU, tests, blockers
.\deploy\plan_a_local_pc\01_fetch_assets.ps1         # ~20 min  pinned weights + corpora + forks
.\deploy\plan_a_local_pc\02_stage_a.ps1              # ~5 min   dense reference, 4 pairs
.\deploy\plan_a_local_pc\03_stage_b.ps1              # ~16 h    44 null cells  ← the long pole
.\deploy\plan_a_local_pc\04_freeze.ps1               # ~1 min   IRREVERSIBLE, asks you to type FREEZE
.\deploy\plan_a_local_pc\05_stage_c.ps1              # ~2 h     44 compression cells
.\deploy\plan_a_local_pc\06_stage_d_analysis.ps1     # ~10 min  CSI table, sweep, cross-audit
```

With Pythia-410M included, budget **4–7 days of near-continuous GPU time**. 410M is an
*estimated* 3–4× slower — measure it with `experiments/time_attribution.py` before scheduling.

### After Stage A, before you go further

Check the two CIRCUS grid diagnostics. They are evidence the ensemble works, not decoration:

- **Mean pairwise Jaccard** across the B views — near 1.0 means the configs are near-duplicates
- **Match rate** (consensus equals a single view) — ~100% is the nesting artifact; stop and fix

Dense-node nesting (82–116 of 120 view pairs) is a known, pre-registered outcome — report it,
don't try to fix it.

### Watch the memory

Peak is the activation cache, not the weights: 4.26 GiB (IOI) against 0.61 GiB of weights.
One cell at a time on 8 GB. On OOM, halve the batch size — **never change precision**, because
float32 is pre-registered for Pythia and a mid-grid change makes cells incomparable.

### Finish with the checklist

[`deploy/plan_a_local_pc/CHECKLIST.md`](deploy/plan_a_local_pc/CHECKLIST.md)

---

## Part 4 — Run it on the DGX Spark

**Scope:** the whole grid — all four models, both tasks, 11 cells, 88 cells per stage.

### 4.1 Before you travel (on the PC)

```powershell
.venv\Scripts\python.exe deploy\shared\fetch_assets.py --plan b --datasets

.venv\Scripts\python.exe -m experiments.time_attribution --model pythia160m --tasks ioi greater_than --seed 0 | Tee-Object docs\baseline_pc.txt
.venv\Scripts\python.exe -m pytest -q | Tee-Object docs\baseline_tests.txt

.\deploy\shared\make_transfer_bundle.ps1
```

That produces `%USERPROFILE%\transfer\` containing the repo as a git bundle, the HF cache,
`frozen/`, the upstream forks and the baselines. **Copy the whole folder** — nothing is ever
pushed to a remote, so the project travels as files.

Also settle **freeze ownership** now, in `docs/HUMAN_DECISIONS.md`, if both machines will run
Pythia.

### 4.2 At the Spark — the one command

```bash
tmux new -s cuc
./deploy/plan_b_dgx_spark/RUN_ALL.sh 1                  # careful first pass
```

Then scale, **per model**, because peak memory differs by an order of magnitude:

```bash
./deploy/plan_b_dgx_spark/RUN_ALL.sh 12 pythia160m
./deploy/plan_b_dgx_spark/RUN_ALL.sh 4  gemma2_2b
```

It pauses twice: after timing (to confirm you re-applied the Q3 rule) and at the freeze.

### 4.3 Or step by step

```bash
bash deploy/plan_b_dgx_spark/00_inspect_hardware.sh     # install NOTHING before this
bash deploy/plan_b_dgx_spark/01_setup_env.sh            # torch first, verify CUDA, then the rest
bash deploy/plan_b_dgx_spark/02_restore_repo.sh         # repo + cache + frozen/ + forks
bash deploy/plan_b_dgx_spark/03_verify_gate.sh          # THE GATE - do not skip
bash deploy/plan_b_dgx_spark/04_timing.sh               # re-measure, then re-apply Q3
./deploy/plan_b_dgx_spark/05_run_queue.sh stageB 4      # 88 null cells
bash deploy/plan_b_dgx_spark/06_freeze.sh               # IRREVERSIBLE
./deploy/plan_b_dgx_spark/05_run_queue.sh stageC 4      # 88 compression cells
bash deploy/plan_b_dgx_spark/07_stage_d_analysis.sh
bash deploy/plan_b_dgx_spark/08_bundle_results.sh       # commit locally + bundle for home
```

### 4.4 Four things that will bite you

**ARM64.** If `uname -m` is `aarch64`, use the NGC PyTorch container rather than pip:

```bash
docker pull nvcr.io/nvidia/pytorch:25.01-py3
docker run --gpus all -it --rm \
    -v $HOME/Circuits_Under_Compression:/workspace/cuc \
    -v $HOME/hf_cache:/hf_cache -e HF_HOME=/hf_cache \
    --shm-size=16g nvcr.io/nvidia/pytorch:25.01-py3 bash
```

If `torch.cuda.is_available()` is `False`, **stop and fix that first** — every later failure
is a confusing symptom of it.

**GPTQ/AWQ probably won't install.** Step 01 reports it. Options in order: your own torch
implementation → source build with `TORCH_CUDA_ARCH_LIST` → CPU kernels → cut the cells and
record the cut in `HUMAN_DECISIONS.md`.

**Grouped-query attention.** Check on day one:

```bash
python -c "
from transformers import AutoConfig
for m in ['google/gemma-2-2b','meta-llama/Llama-3.2-1B']:
    c = AutoConfig.from_pretrained(m)
    print(m, 'heads:', c.num_attention_heads, 'kv heads:', getattr(c,'num_key_value_heads',None))
"
```

If `kv heads` < `heads`, the query/key/value split in `src/extraction/eap.py` needs a
head-group mapping written and tested before Stage A means anything. **Pythia is unaffected.**

**Run Pythia first**, even though it is not a primary model. Cheapest complete pass, exercises
every stage and both analysis scripts, needs no new extraction code.

### 4.5 Finish with the checklist

[`deploy/plan_b_dgx_spark/CHECKLIST.md`](deploy/plan_b_dgx_spark/CHECKLIST.md)

---

## Part 5 — Command reference

### Status and diagnostics

```powershell
.venv\Scripts\python.exe deploy\shared\preflight_blockers.py   # what is still a stub
.venv\Scripts\python.exe -m pytest -q                          # regression gate
git ls-files | Measure-Object -Line                            # 156 = index healthy
Get-Content frozen\FREEZE_MANIFEST.json                        # the pre-registration record
```

### Regenerating the grid

```powershell
.venv\Scripts\python.exe deploy\shared\gen_cells.py all
```

One table in `deploy/shared/gen_cells.py` produces `configs/compression/*.yaml` **and** both
run queues, so a cell key cannot drift between the freeze and Stage C. It refuses to overwrite
a changed config, because configs are immutable once a run references them.

### The grid itself

11 cells × 2 tasks × models:

```
rtn_int8  rtn_int6  rtn_int4  gptq_int4  awq_int4
magnitude_20  magnitude_40  magnitude_60
wanda_20  wanda_40  wanda_60
```

### Budget arithmetic

```
Stage B passes per (model, task) = 11 cells × R × S       = 1100 at R=20, S=5
Stage C passes per (model, task) = 11 cells × 2 × S       = 110
```

| Grid | Pairs | Stage B | Stage C |
|---|---|---|---|
| Plan A (Pythia ×2) | 4 | 4,400 | 440 |
| Plan B (all four models) | 8 | 8,800 | 880 |

The null is matched **per compression cell** — that is why Stage B dominates everything.

---

## Part 6 — When things go wrong

| Symptom | Cause | What to do |
|---|---|---|
| `REFUSING TO RUN SCIENCE` | Blockers unimplemented | Part 2. Not a config problem |
| Scientific run refuses to start | An open gating question | Close it in `HUMAN_DECISIONS.md`, commit |
| Freeze refuses | One of four preconditions unmet | The precondition is the point — don't work around it |
| Stage C hash mismatch | Frozen null doesn't match the config | Check the cell key. **Never edit `frozen/`** |
| Licence tests fail | Empty git index | `git reset` |
| `GatedRepoError` | Licence not approved | Wait. Nothing works around it |
| OOM on the 4060 | Batch too large | Halve the batch. Never change precision |
| OOM on the Spark | Too many concurrent cells | Lower `-P`; re-measure peak per cell |
| Spark scores ≠ PC scores | Precision/kernel/revision differs | **Stop.** Mismatched cells cannot be pooled |
| Architecture mismatch on load | Pinned revision vs latest config | The check is deliberate. Don't bypass it |
| `auto-gptq` build fails on ARM | No aarch64 wheel | §4.4 |
| Results look too good | Chance floor still `U*(U-1)` | Verify blocker 4 landed and the flag is `true` |
| **CSI ≈ 1 everywhere** | Possibly the real finding | See below |

**On that last row.** CSI ≈ 1 means compression damage is indistinguishable from generic
weight noise. **That is a result, and it is publishable.** Do not treat it as a bug and tune
until it disappears — that is precisely the failure the frozen null exists to prevent. Check
the compressor actually changed the weights (`weight_delta()` non-zero, perplexity degraded),
and if it did, report what you found.

---

## Part 7 — Rules that never bend

1. **`frozen/` is append-only.** After a freeze it is never edited, regenerated, or tuned to
   match a result you prefer. If a run fails afterwards, re-run the run — not the null.
2. **The freeze is a human act.** No script does it unattended. `stage_b.freeze_approved` has
   no default on purpose.
3. **Decisions are recorded before the runs that use them**, with dates, in
   `docs/HUMAN_DECISIONS.md`.
4. **Configs are immutable once a run references them** (`AI_RULES.md` 1.2).
5. **Nothing is ever pushed to a remote.** Local commits only; results travel as git bundles.
6. **Never cite from memory.** Every citation is verified against the live source before it
   lands in code or paper (`AI_RULES.md` §2).
7. **The cross-audit uses the frozen CSV tables, never the arXiv PDF.** The authors corrected
   ρ = −1.0 to a range of **−0.540 to +0.062**, and Gemma-2-2B's dense WikiText-2 perplexity
   from 410 to **8.21**. Quoting the PDF puts a walked-back number in your paper.
8. **No model weights in the repo or the artifact.** Gemma and Llama restrict redistribution;
   ship configs, hashes and scripts.
9. **Two machines never freeze the same cell.** Decide ownership first. If it somehow happens,
   keep and report both nulls — never pick the one you prefer after seeing results.
10. **Every claim is stated at both comparison levels**, exact-edge and layer. They can
    disagree, and that disagreement is part of the contribution.
