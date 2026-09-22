# Plan B — Complete the project on a DGX Spark

**Written to be followed without an AI agent.** Every step is a command you can type, with the
expected output and what to do when it differs. Where a number is an estimate rather than a
measurement, it says so.

**Scope: the entire grid — all four models.** Pythia-160M, Pythia-410M, Gemma-2-2B and
Llama-3.2-1B, across both tasks and all five compression families. The Spark can own the whole
project end to end; it is not restricted to the models that fail to fit on the 4060.

**Why Pythia is here too.** Two reasons beyond completeness. First, Pythia is the verification
vehicle — reproducing a Pythia run is how you prove the Spark and the PC measure the same
thing (Part 5). Second, Pythia's ~19 GPU-hours on the 4060 are **serial** hours; on the Spark
the same cells run several at a time, so the secondary models stop being a multi-day
commitment. If the Spark arrives early, running Pythia here frees the 4060 entirely.

**Companion:** [`PLAN_A_LOCAL_RTX4060.md`](PLAN_A_LOCAL_RTX4060.md) — the same Pythia pipeline
on the local PC, for when the Spark is not available. **Read Plan A Part 2 first.** The six
missing pieces it describes are missing on the Spark too. This plan does not repeat those
specifications; it covers getting them running on different hardware.

> **⚠️ Decide which machine owns Pythia's freeze, and decide it before either machine
> freezes.** The freeze is the pre-registration event and `frozen/` is append-only. If both
> machines independently freeze Pythia cells, you have two null distributions for the same
> cell key and no principled way to choose between them after the fact — which is exactly the
> re-tuning the whole design forbids. Pick one machine, record the choice in
> `docs/HUMAN_DECISIONS.md`, and let the other machine only *read* that freeze. See Part 9.

**Just want the steps?** [`../EXECUTION_GUIDE.md`](../EXECUTION_GUIDE.md) has every command
for both machines in order. This document explains *why* each step is what it is.

**Executable version:** [`../deploy/plan_b_dgx_spark/`](../deploy/plan_b_dgx_spark/) holds a
numbered script per step, the 88-cell run queues, a parallel runner and a printable checklist.
This document is the reasoning; that folder is the doing. Start with `00_inspect_hardware.sh`.

> **Missing on the PC as of 2026-09-21: both upstream forks.** `circuit-tracer-0.5.2/` and
> `sae-pruning-paper-main/` are in the documented topology but absent from disk, and
> `saediag` does not import. Pipeline A needs the first; the magnitude/Wanda stubs and the C5
> cross-audit tables need the second. Fetch them **before you travel**:
> `deploy/shared/fetch_upstream_forks.ps1` (PC) or `.sh` (Spark).

---

## Part 0 — What the Spark buys you, and what it does not

**Buys you:**

1. **Everything fits, in full precision.** 128 GB unified memory against the 4060's 8 GB.
   Gemma-2-2B needs about 25 GiB and Llama-3.2-1B about 39 GiB at Pythia's batch size
   (*estimates*). Both fit in float32, so the pre-registered precision decision survives.
2. **The circuit-tracer pipeline becomes possible** — the model plus its transcoder sets
   loaded together, which will not fit in 8 GB.
3. **Cells run in parallel.** The grid is model × task × compression × seed with no shared
   state. This is the real time saving, larger than any per-pass speedup.
4. **Room for compression itself** — GPTQ/AWQ calibration, and holding a dense model next to
   its compressed copy.
5. **Pythia stops being slow.** Pythia-160M peaks at 4.26 GiB on IOI, so the 4060 runs exactly
   one cell at a time and the model costs ~19 serial GPU-hours. The Spark fits a dozen such
   cells simultaneously, turning the secondary models from a multi-day job into a background
   one that runs while you work on the primary models.

**Does not buy you:**

- **Not necessarily faster per pass.** The Spark is high-memory and moderate-bandwidth. Its
  advantage is capacity and parallelism, not single-pass speed. A single attribution pass may
  well be *slower* than on the 4060. This is unmeasured — Part 6 measures it.
- **Not a different codebase.** Same repo, same pins, same frozen nulls.

---

## Part 1 — Pre-arrival checklist (do all of this on the PC, before the Spark exists)

These items have lead times measured in days. Starting them after the hardware arrives wastes
the hardware.

### 1.1 Accept the model licenses — do this today

Both primary models are gated on HuggingFace and **manual approval can take days**:

- Gemma-2-2B: https://huggingface.co/google/gemma-2-2b — accept the Gemma license
- Llama-3.2-1B: https://huggingface.co/meta-llama/Llama-3.2-1B — accept the Llama community license

Then create a read token at https://huggingface.co/settings/tokens and verify access:

```powershell
.venv\Scripts\python.exe -c "from huggingface_hub import HfApi; HfApi().model_info('google/gemma-2-2b'); print('gemma ok')"
.venv\Scripts\python.exe -c "from huggingface_hub import HfApi; HfApi().model_info('meta-llama/Llama-3.2-1B'); print('llama ok')"
```

A `GatedRepoError` means approval has not come through yet. Nothing in Part 5 can run until
both print `ok`.

> **Licensing note for the artifact:** the AAAI artifact ships **configs, hashes and scripts —
> never weights.** Gemma and Llama both restrict redistribution. Do not put model files, or
> compressed copies of them, in the repo or the submission zip.

### 1.2 Pre-download every weight and dataset on the PC

Downloads on a new machine behind an unfamiliar network are a bad first task. Cache
everything now, then copy the cache.

```powershell
$env:HF_HOME = "$env:USERPROFILE\hf_cache"
.venv\Scripts\python.exe -c "
from huggingface_hub import snapshot_download
for repo, rev in [('google/gemma-2-2b', '<pinned-rev>'),
                  ('meta-llama/Llama-3.2-1B', '<pinned-rev>'),
                  ('EleutherAI/pythia-160m', '<pinned-rev>'),
                  ('EleutherAI/pythia-410m', '<pinned-rev>')]:
    snapshot_download(repo, revision=rev)
    print('cached', repo)
"
```

Take the pinned revisions from `configs/model/gemma2_2b.yaml`,
`configs/model/llama32_1b.yaml`, `configs/model/pythia160m.yaml` and
`configs/model/pythia410m.yaml` — **use the pin, not `main`.** The whole reproducibility
story rests on those pins, and Q5 exists because they were not always honoured. The Pythia
weights are small and are already cached on the PC; copy them across rather than
re-downloading, so the Spark verifies against byte-identical files.

Also cache the corpora from `configs/calibration/final.yaml`: **FineWeb-Edu** (300,000 tokens,
seed 7) for calibration and **WikiText-2 test** for perplexity.

### 1.3 Record the baseline you will verify against

On the PC, capture a known-good result to reproduce on the Spark:

```powershell
.venv\Scripts\python.exe -m experiments.time_attribution --model pythia160m --tasks ioi greater_than --seed 0 | Tee-Object docs\baseline_pc.txt
.venv\Scripts\python.exe -m pytest -q | Tee-Object docs\baseline_tests.txt
```

Keep both. Part 4's verification gate compares against them.

### 1.4 Prepare the transfer bundle

**You have a standing rule that nothing gets pushed to a remote.** So the repo moves as a
file, not through GitHub. A git bundle carries the full history in one file:

```powershell
git bundle create $env:USERPROFILE\cuc-repo.bundle --all
git bundle verify $env:USERPROFILE\cuc-repo.bundle
```

Copy these four things to a USB drive or direct network copy:

| Item | Source | Approx. size |
|---|---|---|
| `cuc-repo.bundle` | above | tens of MB |
| `hf_cache/` | §1.2 | ~15–25 GB |
| `frozen/` | repo (if any Pythia cells are already frozen) | small |
| `runs/` | repo (if you want Plan A's results on the Spark) | varies |

**`frozen/` is append-only and its hashes are the pre-registration record.** Copy it; never
regenerate it on the new machine to "make it match."

---

## Part 2 — First hour on the Spark: find out what you actually have

Do not install anything yet. Run these and **write the output into
`docs/spark_environment.md`** — later steps and every estimate in this plan depend on them.

```bash
uname -m                                    # expect: aarch64
cat /etc/os-release                         # DGX OS / Ubuntu version
nvidia-smi                                  # GPU name, driver, memory
nvidia-smi --query-gpu=memory.total --format=csv
free -g                                     # unified memory total
nproc                                       # CPU cores
df -h ~                                     # free disk — you need 100 GB+
python3 --version
which docker && docker --version            # is the container path available?
```

**Decision point.** If `uname -m` returns `aarch64`, you are on ARM and §3 and §4 apply in
full. If it returns `x86_64`, the environment build is far easier and you can largely follow
Plan A's setup instead.

---

## Part 3 — Build the environment

Two routes. **Route A (container) is strongly recommended on ARM** because NVIDIA ships
PyTorch built for aarch64 + CUDA, and building it yourself is a day you will not get back.

### Route A — NGC PyTorch container (recommended)

```bash
docker pull nvcr.io/nvidia/pytorch:25.01-py3      # use the current tag; check the NGC catalog
docker run --gpus all -it --rm \
    -v $HOME/Circuits_Under_Compression:/workspace/cuc \
    -v $HOME/hf_cache:/hf_cache \
    -e HF_HOME=/hf_cache \
    --shm-size=16g \
    nvcr.io/nvidia/pytorch:25.01-py3 bash
```

Inside the container:

```bash
cd /workspace/cuc
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pip install -e .          # or: pip install -r requirements.txt
```

### Route B — native venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install torch FIRST, from the index that carries aarch64+CUDA wheels, and verify before anything else:
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**If `torch.cuda.is_available()` is `False`, stop and fix it before installing anything
else.** Every later failure will be a confusing downstream symptom of this one.

Then:

```bash
pip install -e .
```

### Restore the repository

```bash
git clone ~/cuc-repo.bundle Circuits_Under_Compression
cd Circuits_Under_Compression
git log --oneline -5        # confirm history arrived
cp -r ~/frozen_backup/* frozen/    # if you brought frozen cells
```

---

## Part 4 — The ARM dependency problem (GPTQ and AWQ)

You chose to cover all five compression families. Three of them — RTN, magnitude and Wanda —
are pure PyTorch and will work anywhere torch works. **GPTQ and AWQ are the risk.**

`auto-gptq` and `autoawq` ship prebuilt CUDA wheels for x86-64. On aarch64 they commonly have
no wheel, fall back to a source build, and fail on CUDA extension compilation. Budget for this
rather than being surprised by it.

**Try first:**

```bash
pip install auto-gptq autoawq
python -c "import auto_gptq, awq; print('both ok')"
```

**If it fails, in order of preference:**

1. **Use your own implementation.** Plan A §B2 recommends implementing GPTQ and AWQ directly
   in torch (~150 lines each) precisely so both machines run identical code with no build
   risk. If you took that advice, this entire section is moot — which is the point.
2. **Build from source with the right toolchain:**
   ```bash
   export TORCH_CUDA_ARCH_LIST="9.0"     # set to your GPU's compute capability from nvidia-smi
   pip install --no-build-isolation git+https://github.com/AutoGPTQ/AutoGPTQ.git
   ```
3. **Drop to CPU-kernel mode.** Both libraries have slower pure-PyTorch paths. You are
   measuring *quantization error*, not deployment speed, so a slow kernel is scientifically
   equivalent.
4. **Cut the cells.** `CLAUDE.md` §6 fixes the scope-cut order: compression levels and
   secondary models first. GPTQ/AWQ are compression levels — they are the first legitimate
   cut. **Record the cut in `docs/HUMAN_DECISIONS.md` with its reason.** A documented cut is
   fine; a silently missing grid cell is not.

---

## Part 5 — The verification gate

**Do not run any science until all four of these pass.** This gate is what lets you claim
results from two machines are comparable.

```bash
# 1. Test suite — must match the PC baseline
pytest -q                      # expect: 511 passed, 1 skipped (or 512 passed once the exit gate lands)

# 2. Pinned loading works, including the architecture check
python -c "
from src.extraction.real_model import load_pinned_model
print('loader import ok')"

# 3. Reproduce a Pythia run from the PC and compare numbers
python -m experiments.time_attribution --model pythia160m --tasks ioi greater_than --seed 0

# 4. Frozen manifest still verifies after the transfer
python -c "
import json,pathlib
print(json.loads(pathlib.Path('frozen/FREEZE_MANIFEST.json').read_text()))"
```

For step 3, the **attribution scores** must match the PC to floating-point tolerance. The
**timings** will differ — that is expected and is the subject of Part 6. If the scores differ
materially, something in the environment differs (precision, kernel, model revision) and you
must find it before proceeding. Different scores mean the two machines are measuring different
things, and the grid cannot be pooled.

---

## Part 6 — Re-measure timing, then re-apply the Q3 rule

Every compute figure in this project was measured on an RTX 4060. **None of them transfer.**
The Spark's per-pass speed is genuinely unknown and may be slower.

```bash
python -m experiments.time_attribution --model pythia160m  --tasks ioi greater_than --seed 0
python -m experiments.time_attribution --model pythia410m  --tasks ioi greater_than --seed 0
python -m experiments.time_attribution --model gemma2_2b   --tasks ioi greater_than --seed 0
python -m experiments.time_attribution --model llama32_1b  --tasks ioi greater_than --seed 0
```

Record: seconds per attribution pass, and peak memory per pass, per model and task. This
output is **marked non-evidence by construction** — it is a planning measurement, not a result.

The Pythia rows do double duty: they are the planning measurement *and* the direct
apples-to-apples comparison against the 4060 baseline in `docs/baseline_pc.txt`. If the Spark
turns out to be slower per pass on Pythia-160M than the 4060's 42–48 s (IOI) and 7.5–7.9 s
(greater-than), that is expected — high memory, moderate bandwidth — and it tells you the
Spark's value is parallelism, so plan Part 7 accordingly.

Then re-apply the **Q3 rule** (the pre-stated rule in `docs/Run_Plan.md` that converts measured
timing into ensemble sizes) to these numbers. Q3 gave B = 16, S = 5, R = 20 for Pythia. The
rule, not the answer, is what carries over. If the Spark is slower per pass, the rule may
legitimately yield a smaller R — **and that is a decision to make and record before the
freeze, never after.**

### Budget arithmetic

The matched-magnitude null is matched **per compression cell**, which is what makes this
expensive:

```
Stage B passes per (model, task) = 11 cells × R draws × S seeds
Stage C passes per (model, task) = 11 cells × 2 × S seeds
```

At B = 16, S = 5, R = 20 that is **1100 Stage B passes and 110 Stage C passes** per model-task
pair. The full grid is **4 models × 2 tasks = 8 pairs**, so:

| Models | Pairs | Stage B passes | Stage C passes |
|---|---|---|---|
| Gemma-2-2B + Llama-3.2-1B | 4 | 4,400 | 440 |
| Pythia-160M + Pythia-410M | 4 | 4,400 | 440 |
| **Whole grid** | **8** | **8,800** | **880** |

Multiply each model's passes by *its own* measured seconds-per-pass — do not use a single
average, because Pythia-160M and Gemma-2-2B differ by an order of magnitude.
*Illustrative only:* if the primary models average 60 s per pass and Pythia averages 25 s,
the grid is roughly 81 + 34 ≈ **115 GPU-hours serial**, which Part 7's parallelism divides
by your concurrency factor. **Measure before you schedule.**

**If Plan A already completed Pythia on the PC,** you do not re-run those cells here. Copy
`frozen/` and `runs/` across, use them as-is, and the Spark's job shrinks to the top row of
that table. Re-running Pythia on the Spark is for the case where the Spark arrives before the
PC finishes, or where you want one machine to own the entire record.

---

## Part 7 — Parallelism, the actual advantage

Cells are fully independent — no shared state — so the Spark's memory lets you run several at
once. This, not clock speed, is where the time goes.

**How many?** `floor(available_memory / peak_memory_per_cell)`, minus headroom. With 128 GB and
a measured peak of, say, 25 GiB per Gemma cell, four concurrent cells is reasonable. **Use your
Part 6 measurement, not this example.**

**Concurrency is per-model, because peak memory is per-model.** Pythia-160M peaked at 4.26 GiB
(IOI) and 2.64 GiB (greater-than) on the 4060, so a dozen or more Pythia cells fit alongside
each other — which is precisely why the model's 19 serial GPU-hours collapse here. Do not
apply the Gemma concurrency number to a Pythia queue, or you will leave most of the machine
idle. Run separate queues with separate `-P` values, or sort the queue so same-model cells
batch together.

A simple work queue, which is all you need:

```bash
# cells.txt — one Hydra override line per cell
# model=gemma2_2b task=ioi compression=rtn_int4 stage=stageB seed=0
# model=gemma2_2b task=ioi compression=rtn_int6 stage=stageB seed=0
# ...

cat cells.txt | xargs -P 4 -I {} sh -c \
  'python experiments/run_stage_b.py mode=scientific_run pipeline=dense-node ensemble=default nulls=default {} \
   > logs/$(echo "{}" | tr " =" "__").log 2>&1'
```

`-P 4` is the concurrency. Rules:

- **Start with `-P 1`** and confirm one cell completes correctly before scaling up.
- **Watch memory** with `nvidia-smi --query-gpu=memory.used --format=csv -l 5` during the first
  parallel batch.
- **Never run Stage B and Stage C for the same cell concurrently.** Stage C reads the frozen
  null that Stage B produces.
- **Log every cell to its own file.** When something fails at 3 a.m. across 4,400 passes, the
  log is the only forensic record.
- Use `tmux` or `screen` so a dropped SSH session does not kill a two-day run.

---

## Part 8 — Execution order

Same Algorithm 1 as Plan A. The commands are identical apart from the model name, which takes
one of:

```
model=pythia160m   model=pythia410m   model=gemma2_2b   model=llama32_1b
```

**Run Pythia-160M first, all the way through Stage D**, even though it is not a primary model.
It is the cheapest complete pass and it exercises every stage, the freeze, the hash checks and
the analysis scripts. Integration problems surface there for a fraction of the compute, and
they surface *before* the freeze makes anything permanent. Only then start the primary models.

The examples below use `gemma2_2b`; substitute freely.

```bash
# Stage A — dense reference
python experiments/run_stage_a.py mode=scientific_run model=gemma2_2b task=ioi \
    pipeline=dense-node ensemble=default nulls=default \
    stage=stageA setting=dense compression_family=dense comparison_level=both seed=0

# Stage B — null draws, every cell (parallelise per Part 7)
python experiments/run_stage_b.py mode=scientific_run model=gemma2_2b task=ioi \
    compression=rtn_int4 pipeline=dense-node ensemble=default nulls=default stage=stageB seed=0

# FREEZE — plan first, always
python experiments/freeze_stage_b.py --config <resolved_config.json> --dry-run
python experiments/freeze_stage_b.py --config <resolved_config.json>

# Stage C — real compression, against the frozen null
python experiments/run_stage_c.py mode=scientific_run model=gemma2_2b task=ioi \
    compression=rtn_int4 pipeline=dense-node ensemble=default stage=stageC \
    compression_family=rtn compression_level=4 stage_c.cell=rtn_int4 \
    null_frozen_hash=<hash> seed=0

# Stage D — CSI table, bootstrap CIs, BH correction
python experiments/run_stage_d.py mode=scientific_run stage=stageD \
    stage_d.stage_c_dirs=[<stage C run dirs>]

# Analysis
python analysis/threshold_sweep.py
python analysis/cross_audit.py
```

The freeze preconditions are enforced in code and are not negotiable: `mode=scientific_run`,
an explicit `stage_b.freeze_approved=true`, no open gating questions, and
`distance.pre_registered_for_stage_c=true`.

### Grouped-query attention — expect to write code here

**Gemma-2 and Llama-3.2 are expected to use grouped-query attention, and the extraction
pipeline does not support it yet.** This is listed as outstanding work in
`docs/PROGRESS_REPORT_2026-09-16.md` §7. Under GQA several query heads share one key/value
head, so the query/key/value edge split in `src/extraction/eap.py` needs a head-group mapping
rather than a one-to-one assumption.

This affects the primary models only. **Pythia uses standard multi-head attention and its
extraction path is already proven** — real dense-node extraction ran on Pythia-160M on the PC
(commit `fdd7717`), with every attribution score matching an exact finite-difference check. So
the Pythia half of the grid needs no new extraction code on the Spark, which is another reason
to run it first.

**Check this early — on day one, not the night before a run:**

```bash
python -c "
from transformers import AutoConfig
for m in ['google/gemma-2-2b','meta-llama/Llama-3.2-1B']:
    c = AutoConfig.from_pretrained(m)
    print(m, 'heads:', c.num_attention_heads, 'kv heads:', getattr(c,'num_key_value_heads',None))
"
```

If `kv heads` is smaller than `heads`, GQA is in use and the mapping must be written and
tested before Stage A means anything.

---

## Part 9 — Getting results back

```bash
# On the Spark: commit locally (never push — standing rule)
git add runs/ frozen/ docs/
git commit -m "results: full grid (Pythia-160M/410M, Gemma-2-2B, Llama-3.2-1B), stages A-D"
git bundle create ~/cuc-results.bundle --all
```

Copy `cuc-results.bundle` plus `runs/` and `frozen/` back to the PC, then:

```powershell
git pull ~/cuc-results.bundle main        # or: git fetch ~/cuc-results.bundle
```

**Never merge two different `frozen/` directories.** This is the one irreversible way to ruin
the project, and running Pythia on both machines is exactly how it would happen.

The rule, decided in advance (see the warning at the top of this document):

- **One machine owns each model's freeze.** Write the ownership table into
  `docs/HUMAN_DECISIONS.md` before either machine runs Stage B.
- The non-owning machine may **read** a frozen cell and run Stage C against it — that is
  normal and correct, and the hash check will confirm it is the same null.
- If both machines have somehow frozen the same cell key, **do not pick the one you prefer.**
  Keep both, report both, and say in the paper that two nulls exist for that cell. Choosing
  after seeing the results is the re-tuning this design exists to prevent.

A suggested split, if the Spark arrives while the PC is mid-grid: the PC owns whatever it has
already frozen, the Spark owns everything else.

---

## Part 10 — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is False | Wrong wheel for aarch64, or driver mismatch | Use the NGC container (§3 Route A) |
| `auto-gptq` build fails | No aarch64 wheel | §4 — own implementation, source build, CPU kernel, or documented cut |
| `GatedRepoError` | License not approved | §1.1 — wait for approval; nothing works around it |
| Architecture mismatch on load | The pinned revision disagrees with the latest config | This check is deliberate. Do not bypass it — it is why the loader exists |
| OOM despite 128 GB | Too many concurrent cells | Lower `-P`; re-measure peak per cell |
| Scores differ from the PC | Precision, kernel or revision differs | Stop. Find it. Pooling mismatched cells invalidates the grid |
| Scientific run refuses | Open gating question | Close it in `HUMAN_DECISIONS.md`, commit, re-run |
| Freeze refuses | One of four preconditions unmet | The precondition is the point — do not work around it |

---

## Part 11 — Definition of done

- [ ] `docs/spark_environment.md` records the real hardware and software versions
- [ ] Verification gate (§5) passed: tests match, Pythia scores reproduce
- [ ] Timing re-measured; Q3 rule re-applied and the outcome recorded before the freeze
- [ ] GQA support written and tested, if the primary models use it
- [ ] All five compression families working, or cuts documented in `HUMAN_DECISIONS.md`
- [ ] Freeze-ownership table recorded in `HUMAN_DECISIONS.md` before any Stage B run
- [ ] Pythia-160M completed end to end first, as the integration pass
- [ ] Stage A → B → freeze → C → D complete for **all four models** (Pythia-160M, Pythia-410M,
      Gemma-2-2B, Llama-3.2-1B) across both tasks — or inherited from Plan A for the Pythia
      half, with the source machine recorded
- [ ] CSI table with bootstrap CIs over B, S, r and BH at q = 0.05
- [ ] Every conclusion stated at both comparison levels
- [ ] Cross-audit run against the **frozen CSV tables** of the feature-level paper, not the
      arXiv PDF (`docs/Run_Plan.md` §0.2 — the authors corrected ρ = −1.0 to a range of
      −0.540 to +0.062, and Gemma-2's dense PPL from 410 to 8.21)
- [ ] Results bundled back to the PC; no weights in the repo

---

## Part 12 — The one thing not to get wrong

The contribution of this project is the **null model**, and its scientific force comes entirely
from the fact that it was frozen *before* any real compression was measured.

Everything in this plan — the four freeze preconditions, the append-only `frozen/` directory,
the hash checks in Stage C, the insistence that the Q3 rule is re-applied before rather than
after the freeze — exists to protect that one property.

If a run fails after the freeze, you re-run it. **You do not adjust the null.** If the answer
comes out as CSI ≈ 1 — compression damage indistinguishable from random weight noise — that is
a result, and it is publishable. The honest null is the paper.
