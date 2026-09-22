# Plan B — DGX Spark

**Written to be followed without an AI agent.** Every step is a script; every script says what
it expects, what it changes, and what to do when it fails.

**Scope:** the entire grid — Pythia-160M, Pythia-410M, Gemma-2-2B, Llama-3.2-1B, both tasks,
all 11 compression cells. 88 cells per stage.

Full reasoning: [`../../docs/PLAN_B_DGX_SPARK.md`](../../docs/PLAN_B_DGX_SPARK.md)

All scripts are bash, run from the **repo root**.

---

## Before you travel (do this on the PC)

1. **Accept the Gemma and Llama licences** — approval takes days and blocks everything:
   - https://huggingface.co/google/gemma-2-2b
   - https://huggingface.co/meta-llama/Llama-3.2-1B
2. **Cache the weights and corpora:** `.venv\Scripts\python.exe deploy\shared\fetch_assets.py --plan b --datasets`
3. **Record the baselines** you will verify against:
   ```powershell
   .venv\Scripts\python.exe -m experiments.time_attribution --model pythia160m --tasks ioi greater_than --seeds 2 | Tee-Object docs\baseline_pc.txt
   .venv\Scripts\python.exe -m pytest -q | Tee-Object docs\baseline_tests.txt
   ```
4. **Pack the bundle:** `.\deploy\shared\make_transfer_bundle.ps1`
   → produces `%USERPROFILE%\transfer\` with the repo bundle, HF cache, `frozen/`, the
   upstream forks and the baselines. Copy the whole folder.

Nothing is pushed to a remote — that is a standing project rule, so the project travels as
files.

---

## One command

```bash
tmux new -s cuc
./deploy/plan_b_dgx_spark/RUN_ALL.sh 1                 # careful first pass
./deploy/plan_b_dgx_spark/RUN_ALL.sh 12 pythia160m     # then scale, per model
```

Runs every step below in order and is **resumable** — re-run after any interruption and
finished cells are skipped. Everything is tee'd to `logs/run_all/`.

It pauses twice, both times because a human must decide:

1. **After timing** — to confirm you re-applied the Q3 rule to the measured numbers and
   recorded any change to B/S/R *before* the freeze.
2. **At the freeze** — the pre-registration event, where it asks you to type `FREEZE`.

The steps below are the same things it runs; call them individually when you want control.

## The sequence, at the Spark

| Step | Script | What it does |
|---|---|---|
| 0 | `00_inspect_hardware.sh` | Records the real hardware into `docs/spark_environment.md`. **Install nothing before this.** |
| 1 | `01_setup_env.sh` | Verifies torch+CUDA, installs the project, attempts GPTQ/AWQ |
| 2 | `02_restore_repo.sh` | Restores repo, HF cache, `frozen/` and the forks from the bundle |
| 3 | `03_verify_gate.sh` | **The gate.** Tests, loader, Pythia reproduction, manifest |
| 4 | `04_timing.sh` | Re-measures per-pass time for all four models, then re-apply Q3 |
| 5 | `05_run_queue.sh stageB <P> [model]` | Stage B null draws, parallel and resumable |
| 6 | `06_freeze.sh` | **THE FREEZE** — irreversible, asks you to type `FREEZE` |
| 7 | `05_run_queue.sh stageC <P> [model]` | Stage C real compression against the frozen null |
| 8 | `07_stage_d_analysis.sh` | Stage D CSI table, threshold sweep, cross-audit |
| 9 | `08_bundle_results.sh` | Commits locally and bundles results for the trip home |

---

## Four things that will bite you

**1. ARM64.** If `uname -m` says `aarch64`, use the **NGC PyTorch container** rather than pip
(`docs/PLAN_B_DGX_SPARK.md` §3 Route A). Building torch for aarch64+CUDA yourself is a lost
day. If `torch.cuda.is_available()` is `False`, stop and fix it before installing anything
else — every later failure is a confusing symptom of that one.

**2. GPTQ and AWQ probably will not install.** `auto-gptq`/`autoawq` ship x86-64 wheels. Step 1
tells you if they failed and lists the four options in order. The recommended one is to use
in-repo torch implementations so both machines run identical code.

**3. Concurrency is per-model.** Peak memory differs by an order of magnitude between
Pythia-160M (4.26 GiB measured) and Gemma-2-2B (~25 GiB estimated). **Always start with
`-P 1`**, confirm one cell completes, then scale — and filter by model:

```bash
./05_run_queue.sh stageB 1                    # prove one cell works
./05_run_queue.sh stageB 12 pythia160m        # then pack the small model densely
./05_run_queue.sh stageB 4  gemma2_2b         # and the large model sparsely
```

Watch it: `nvidia-smi --query-gpu=memory.used --format=csv -l 5`. Use `tmux` so a dropped SSH
session does not kill a two-day run.

**4. Grouped-query attention.** Gemma-2 and Llama-3.2 are expected to use GQA, which the
extraction pipeline does not support yet. Check on **day one**:

```bash
python -c "
from transformers import AutoConfig
for m in ['google/gemma-2-2b','meta-llama/Llama-3.2-1B']:
    c = AutoConfig.from_pretrained(m)
    print(m, 'heads:', c.num_attention_heads, 'kv heads:', getattr(c,'num_key_value_heads',None))
"
```

If `kv heads` < `heads`, the query/key/value edge split in `src/extraction/eap.py` needs a
head-group mapping written and tested before Stage A means anything. **Pythia is unaffected** —
standard multi-head attention, already proven (commit `fdd7717`).

---

## Run Pythia first

Even though it is not a primary model. It is the cheapest complete pass, it exercises every
stage, the freeze, the hash checks and both analysis scripts, and it needs no new extraction
code. Integration problems surface there for a fraction of the compute — and *before* the
freeze makes anything permanent.

---

## Freeze ownership — decide before any Stage B run

If the PC is also running Pythia, **exactly one machine owns each model's freeze.** Write the
table into `docs/HUMAN_DECISIONS.md` first. The non-owning machine may read a frozen cell and
run Stage C against it; the hash check confirms it is the same null.

If both machines somehow freeze the same cell key, **do not pick the one you prefer.** Keep
both, report both. Choosing after seeing results is the re-tuning the frozen null exists to
prevent.

---

## Nothing runs yet

Steps 5–8 refuse until `python deploy/shared/preflight_blockers.py` passes. As of 2026-09-21,
8 of 9 checks are blocked — the real-model compressors and the null perturber are stubs. See
[`../BLOCKERS.md`](../BLOCKERS.md) for what each one needs.
