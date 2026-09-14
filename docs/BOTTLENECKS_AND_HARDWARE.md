# Bottlenecks, and what each tier of hardware can achieve

Written 2026-09-14. **Every number marked _measured_ comes from the non-evidence pilots run on
my RTX 4060** (Pythia-160M at its pinned sha `50f5173d`, float32, 300 prompts per task, batch
size 32). **Every number marked _estimate_ is a linear scaling of those measurements** with the
assumptions stated beside it; it has not been run. Re-measure on any new machine with:

```powershell
python -m experiments.time_attribution --model <model> --tasks ioi greater_than --seeds 2
```

---

## Part 1 — What is actually slow or blocking

### 1. Compute: the attribution pass is the whole cost

| Measured on the RTX 4060 (Pythia-160M) | IOI | Greater-than |
|---|---|---|
| One edge-attribution-patching pass (2 forwards + 1 backward per batch) | **42–48 s** (4 runs) | **7.5–7.9 s** (3 runs) |
| Peak VRAM during the pass | **4.26 GiB** | **2.64 GiB** |
| Batches for 300 prompts | 12–13 (7 token lengths, 14–20) | 10 (all 10 tokens) |
| Re-pruning the cached scores under one threshold view | 10–16 ms | 10–12 ms |
| Drawing and tokenizing the prompts | 0.06–0.35 s | 0.08–0.35 s |
| Loading the model at the pinned sha | 14 s | — |

One earlier IOI pass measured 25.8 s. It was never reproduced in four later runs and is
unexplained, so it is not used here.

**What this means:**

- **The threshold grid is free.** Sixteen views cost ~0.2 s against a 45 s pass, which confirms
  CIRCUS's observation that the ensemble is near-free once raw attribution exists. B is not a
  budget lever.
- **IOI costs ~6× greater-than** because its prompts are longer and spread over more batches.
  Each batch also runs 601 hook-pair tensor contractions in a Python loop (576 into head and MLP
  inputs, 25 into the logits) — likely a real share of the time, **not yet profiled**.
- **The multiplier is S × (1 + R).** Each (model, task, compression) cell repeats the pass for S
  seeds on the dense model and on each of R null draws.

**Per-cell cost** (Run_Plan's formula, from the measured passes):

| S × R | IOI cell | Greater-than cell |
|---|---|---|
| **5 × 20 (adopted, Q3)** | **77–82 min** | **14 min** |
| 5 × 10 | 40–43 min | 7.3–7.6 min |
| 3 × 10 | 24–26 min | 4.4–4.5 min |

With the repo's working estimate of ~12 compression cells per (model, task), **Pythia-160M's two
tasks at S = 5, R = 20 come to roughly 19 GPU-hours on the 4060** (_estimate_ from measured
passes). Docstring, Pythia-410M and both primaries are on top of that and unmeasured.

### 2. Memory: the attribution cache, not the weights

On Pythia-160M the weights take **0.61 GiB** (_measured_) but the IOI pass peaks at **4.26 GiB**.
The difference is the attribution cache: clean and corrupted activations for every head and MLP
output, plus gradients at every head's query, key and value inputs, all held for a whole batch.
That cache grows with **layers × heads × model width × batch size × sequence length** — so it,
not the parameter count, sets the batch size a card can run.

### 3. Engineering not yet built

These block runs on any hardware:

| Missing piece | Blocks |
|---|---|
| Stage B: matched-magnitude perturbation of a real TransformerLens model | the null distribution, so everything after |
| Stage C: RTN / GPTQ / AWQ / magnitude / Wanda applied to a real model | real compression |
| Chance-floor N implemented as "possible edges among observed nodes" (decided) | Stage C chance floors |
| Grouped-query attention in the EAP code (it currently refuses) | the dense-node C8 control on Gemma-2 and Llama-3.2, both of which are **expected** to use grouped-query attention — to verify once their configs are readable |
| Pipeline A: the real circuit-tracer wrapper | the primary-model circuits |
| The docstring prompt generator | the third task |
| The Phase-1 exit gate (published GPT-2 IOI circuit) | trusting any measurement |
| Perplexity evaluation and the calibration-corpus downloads (Q7) | matched-perplexity control, Stage C calibration |

### 4. Waiting on people or outside services

- **Gemma-2 and Llama-3.2 licences** are `gated: manual` on HuggingFace. Approval can take days and
  nothing on the primaries can start until it lands. **This is the longest pole.**
- **Pythia-410M** has not been downloaded or timed; its download was not part of the approval.
- **Decisions still open:** the C4 interaction-flag ratio (before Stage D), C5 grid alignment
  (before Stage D), and the attribution precision for the primaries (Pythia's float32 decision
  does not transfer automatically).

### 5. Method limitations (not fixable with hardware)

- **The dense-node threshold grid gives largely nested views** (82–116 of 120 view pairs nested),
  and a pre-registered sweep found no edge range that fixes it. Kept as pre-registered and
  reported — see `docs/preregistration/`.
- **Edge attribution patching is poorly calibrated** even where its ranking works (R² = 0.27
  against true patching in its own paper, arXiv:2310.10348), and its error grows with downstream
  non-linearity (arXiv:2606.09899) — which compression changes.
- **Seeds 0 and 1** were used to diagnose the grid and will also be Stage A seeds (disclosed).

### 6. Platform friction on this Windows machine

- The HuggingFace cache cannot use symlinks here, so duplicate files are stored in full. Enabling
  Windows Developer Mode fixes it. Disk is not tight: **169 GB free** (_measured_).
- TransformerLens 3.9.0 marks `HookedTransformer` as deprecated, to be removed in 4.0. The
  `models` extra pins exactly `transformer-lens==3.9.0`, so this cannot break silently.
- Git converts line endings on checkout (`core.autocrlf=true`); any script that patches files
  by exact text must allow for it.

---

## Part 2 — What each tier of hardware can achieve

### How the estimates are made

For the dense-node attribution pass on IOI (300 prompts, sequences ≤ 20 tokens):

- **Weights** ≈ parameters × 4 bytes (float32) or × 2 bytes (bfloat16).
- **Attribution cache** ≈ 3.65 GiB × (layers × heads × width ÷ 110,592) × (batch ÷ 32). The
  3.65 GiB is the _measured_ Pythia-160M peak minus its weights; 110,592 = 12 × 12 × 768 is
  Pythia-160M's layers × heads × width.

This ignores vocabulary size (Gemma-2's is large, so its logits need more memory than this
formula counts), memory fragmentation, and anything specific to circuit-tracer. **Treat every
estimate as ±50%**, and re-measure with the timing pilot before committing to a batch size.

| Model | Layers × heads × width | Cache relative to Pythia-160M | Weights fp32 / bf16 | Architecture source |
|---|---|---|---|---|
| Pythia-160M | 12 × 12 × 768 | 1.0× | 0.61 GiB _measured_ | verified against pinned config |
| Pythia-410M | 24 × 16 × 1024 | 3.6× | ~1.5 / 0.8 GiB | verified against pinned config |
| Llama-3.2-1B | 16 × 32 × 2048 | 9.5× | ~4.6 / 2.3 GiB | **placeholder** — config gated |
| Gemma-2-2B | 26 × 8 × 2304 | 4.3× | ~9.7 / 4.8 GiB | **placeholder** — config gated |

**Estimated peak memory for one IOI attribution pass:**

| Model and precision | batch 4 | batch 8 | batch 16 | batch 32 |
|---|---|---|---|---|
| Pythia-160M fp32 | — | — | — | **4.3 GiB _measured_** |
| Pythia-410M fp32 | ~3.1 | ~4.8 | ~8.0 | ~14.5 |
| Llama-3.2-1B bf16 | ~4.5 | ~6.6 | ~10.9 | ~19.6 |
| Llama-3.2-1B fp32 | ~8.9 | ~13.2 | ~21.9 | ~39.2 |
| Gemma-2-2B bf16 | ~5.8 | ~6.8 | ~8.8 | ~12.7 |
| Gemma-2-2B fp32 | ~11.7 | ~13.7 | ~17.6 | ~25.5 |

The bf16 rows halve both weights and cache — an approximation. The primaries' attribution
precision is **not yet decided**.

### The tiers

#### Tier 0 — RTX 4060, 8 GB (my current machine; the only measured tier)

- **Can do:** everything on **Pythia-160M** at full batch 32 — Stage A, and Stages B and C once
  built — for IOI and greater-than. About **19 GPU-hours** at S = 5, R = 20.
- **Can do, slowly:** Pythia-410M at batch ~8 (~4.8 GiB _estimate_), so roughly 4× the batches
  and a correspondingly longer pass.
- **Cannot do:** either primary. Gemma-2-2B's float32 weights alone (~9.7 GiB) exceed the card,
  and Llama-3.2-1B fits only at tiny bf16 batches, if at all. Pipeline A on the primaries is out
  of reach.
- **Best role:** developing and debugging every stage on Pythia-160M, which is what the plan
  already assigns it.

#### Tier 1 — 12–16 GB (e.g. RTX 4070 Super 12 GB; RTX 4060 Ti 16 GB, RTX 4080 16 GB)

- **Adds:** Pythia-410M at batch 16 (~8 GiB _estimate_) on a 12 GB card, batch ~24 on 16 GB.
- **Adds:** the dense-node C8 control on both primaries in **bf16 at small batches** — Gemma-2-2B
  at batch 8–16 (~6.8–8.8 GiB), Llama-3.2-1B at batch 4–8 (~4.5–6.6 GiB). All _estimates_, and
  subject to the grouped-query-attention support not yet built.
- **Still limited:** float32 on the primaries, and two cells at once.

#### Tier 2 — 24–32 GB (e.g. RTX 3090 / RTX 4090, 24 GB; RTX 5090, 32 GB)

- **Adds:** the primaries in **float32** at batch 8–16 (Gemma-2-2B ~13.7–17.6 GiB, Llama-3.2-1B
  ~13.2 GiB at batch 8). _Estimates._
- **Adds:** two or three Pythia-160M cells in parallel (4.3 GiB each, _measured_) — close to a
  proportional cut in wall-clock time, because cells are independent.
- **Probably adds:** circuit-tracer on the primaries (pipeline A). Its memory use is **not
  measured**.

#### Tier 3 — 40–80 GB datacenter GPUs (e.g. A100 40 / 80 GB, H100 80 GB)

- **Adds:** every model, float32, batch 32 (Llama-3.2-1B ~39 GiB is the largest _estimate_), with
  room to spare on 80 GB.
- **Adds:** many cells in parallel, which is where these cards pay off; the grid is embarrassingly
  parallel across (model, task, compression, seed).
- **Changes the plan:** at this tier the S = 5, R = 20 budget stops binding for Pythia, and the
  scope-cut order in CLAUDE.md §6 is unlikely to be needed.

#### Tier 4 — DGX Spark, 128 GB unified memory (the project's planned machine)

- **Adds:** all four models loaded at once, and several primary-model cells concurrently — memory
  stops being the constraint.
- **Trade-off:** the proposal describes it as a high-memory, **moderate-bandwidth** box, so a single
  attribution pass is not necessarily faster than on a high-end discrete GPU. It wins on capacity
  and parallelism, not per-pass speed. Unmeasured here.
- **Best role:** the primaries and the parallel cell grid, with the 4060 continuing Pythia work
  alongside — exactly the split in the proposal.

#### Multiple GPUs of any tier

Cells share nothing during extraction, so **N GPUs give roughly N× throughput** on the cell grid.
The one serialisation point is the freeze: `frozen/` has a single writer and is verified after
every append.

---

## Part 3 — What moves the project fastest, in order

1. **Accept the Gemma-2 and Llama-3.2 licences now.** It costs minutes and its latency is days.
2. **Build Stage B and Stage C for real models on the 4060**, using Pythia-160M. No hardware
   upgrade helps until they exist.
3. **Profile the attribution pass** (the Python loop over 601 hook pairs is the obvious suspect)
   before buying compute; a faster pass multiplies through every cell.
4. **Add grouped-query-attention support and run the timing pilot on the primaries** as soon as
   their configs are readable — that replaces the Tier 1–4 estimates with measurements.
5. **Only then choose hardware**, with measured per-pass times in hand.
