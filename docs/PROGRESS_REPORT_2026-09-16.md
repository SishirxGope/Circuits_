# Progress report — Do Circuits Survive Compression?

**Date:** 2026-09-16 · **Branch:** `main` (local, not pushed) · **Tests:** 511 passed, 1 skipped

Companion documents: [`PROGRESS.md`](PROGRESS.md) (running status),
[`HUMAN_DECISIONS.md`](HUMAN_DECISIONS.md) (every decision and its reasoning),
[`BOTTLENECKS_AND_HARDWARE.md`](BOTTLENECKS_AND_HARDWARE.md) (full hardware analysis).

---

## 1. Where things stand

| | |
|---|---|
| **Evidence measured** | None yet. Only non-evidence pilots have run (timing and threshold-grid checks). |
| **Decisions** | Every decision needed before the first real extraction is made and recorded. |
| **Real models** | Pythia-160M loads at its pinned revision, and circuits extract from it. |
| **Missing before real results** | Stage B (the null) and Stage C (real compression) for real models; the Gemma-2 and Llama-3.2 licences. |
| **Tests** | 511 passed, 1 skipped (the Phase-1 exit gate, by design). |

---

## 2. Work completed, in order

| Commit | Work |
|---|---|
| `1d5d456` | Restored the upstream licence files, which had been deleted and gitignored, so every clone was breaking the MIT terms. Added tests so it cannot recur. |
| `050390f` | Q9: pinned and verified the two upstream code commits. |
| `ce23965` | Q5: pinned all eight HuggingFace model revisions. |
| `eb77356` | Q1, Q2, Q4, Q10, Q11 pre-registered. Caught that the planned "crossed 4×4 grid, B = 16" could not be non-nested. |
| `60b0abf` | Q6 and Q7 pre-registered. Caught greater-than being mis-described as a day-of-month task, and a mix-up between the calibration and perplexity corpora. |
| `06b59c8` | Confidence intervals resampled over B, S and r; FDR, normalised L1 and tracker settled; three silent bugs fixed. |
| `d7c2e58` → `9798bf9` | Threshold-grid pass criterion committed **before** the sweep, then the sweep results. |
| `fdd7717` | Real circuit extraction on Pythia-160M. |
| `f3b6e21` | Q3 and the chance-floor rule recorded. |
| `40ad0d2` | Repository organised; every document brought up to date. |

---

## 3. Decisions

- **Measurement:**
  - L1 distance, with Jensen–Shannon as a check and normalised L1 reported alongside.
  - CIRCUS core/contingent/noise bands (core s = 1, noise s < 0.5).
  - An anti-diagonal threshold grid with B = 16.
  - Layer-level coarse comparison, with a pre-registered falsification condition.
  - Edges aggregated over token positions.
- **Tasks:** IOI (30 templates) and greater-than (120 nouns), 300 prompts each, no BOS token.
- **Data:** FineWeb-Edu for calibration (300,000 tokens, seed 7); WikiText-2 test for perplexity.
- **Statistics:** bootstrap over B, S and r; FDR q = 0.05; the chance floor counted over structurally possible edges only.
- **Pythia pipeline:**
  - Edge attribution patching, with query/key/value inputs split.
  - Scores summed over prompts, then the absolute value taken once.
  - float32.
- **Ensemble size (Q3):** B = 16, S = 5, R = 20 — the plan's pre-stated rule applied to the measured timing.

---

## 4. What was built

- **A loader that really pins the model version.** The obvious TransformerLens call silently ignores
  the pin, and it builds the architecture from the latest configuration rather than the pinned one.
  The loader works around the first and refuses to run if the second produces a mismatch. It also
  enforces download approval, which was previously only written in the documentation.
- **Real IOI and greater-than prompt sets**, re-derived for Pythia's tokenizer. Only 88 of the 99
  original IOI names are single tokens, and the `"01"` corruption token is 520 rather than GPT-2's
  486.
- **Circuit extraction** that computes attribution once per seed and re-thresholds it for all 16
  views in milliseconds.
- **A timing tool** whose output is marked non-evidence by construction.

---

## 5. What was measured (RTX 4060, Pythia-160M — not evidence)

| | IOI | Greater-than |
|---|---|---|
| One attribution pass | 42–48 s | 7.5–7.9 s |
| Peak GPU memory | 4.26 GiB | 2.64 GiB |
| Task signal, clean → corrupted | +4.56 → −0.27 | +0.763 → −0.640 |

- **Correctness:** every attribution score matches an exact finite-difference check.
- **Grid check:** 82–116 of 120 view pairs came out nested. A pre-registered fix found no passing
  option, so the grid was kept and both results will be reported.

---

## 6. Mistakes caught and corrected

Two errors in the project records, now corrected everywhere:

- Greater-than has **120 nouns, not 26** — the earlier count was of lines, not nouns.
- The **IOI template source was wrongly given as TransformerLens**; it is Easy-Transformer, via ACDC.

Bugs and plan errors caught before they could affect results:

- Stage C would have reported fake agreement on Pythia's graph.
- Empty ensemble views were silently dropped, which inflated scores.
- The chance floor was about **10× too lenient**.

---

## 7. Not done yet

1. Stage B (the null) and Stage C (real compression) for real models.
2. Code for the chance-floor rule.
3. The docstring task generator.
4. Support for grouped-query attention, which Gemma-2 and Llama-3.2 are expected to use.
5. The circuit-tracer pipeline for Gemma-2 and Llama-3.2.
6. The GPT-2 IOI exit gate, which must pass before any result counts.
7. The null freeze, then the actual results.

**Waiting on decisions or action:**

- Accept the Gemma-2 and Llama-3.2 licences on HuggingFace (manual approval can take days).
- Decide C4 (the interaction-flag ratio) and C5 (grid alignment for the cross-audit).
- Choose the attribution precision for Gemma-2 and Llama-3.2.
- Settle whether "change vs. the published circuit" is claimed only on models that have one.

---

## 8. Why the work could not be completed on the RTX 4060 (8 GB)

**Most of the remaining work is not blocked by the RTX 4060.** It is blocked by the missing Stage B/C
code, the unaccepted licences, and a deliberate pause. **Pythia-160M can be completed entirely on the
RTX 4060** once Stage B/C exist.

Where the RTX 4060 genuinely falls short:

1. **The primary models do not fit in 8 GB.** Gemma-2-2B's weights alone are about 9.7 GiB in
   float32. Attribution also needs a cache several times larger than the weights — 4.26 GiB against
   0.61 GiB of weights on the smallest model. *Estimates:* Gemma-2 needs about 25 GiB and Llama-3.2
   about 39 GiB at the batch size Pythia uses.
2. **Time.** Pythia-160M alone is about **19 GPU-hours** at S = 5, R = 20, because each experiment cell
   repeats the attribution pass 105 times (S × (1 + R)). Pythia-410M and both primary models come on
   top of that.
3. **One GPU runs one cell at a time.** An IOI pass peaks at 4.26 GiB, so even two Pythia cells cannot
   run side by side, although the cells are fully independent.
4. **Pythia-410M only fits at small batches.** *Estimate:* about 4.8 GiB at batch 8 against about
   14.5 GiB at batch 32, making it several times slower.
5. **The circuit-tracer pipeline** loads the model plus large transcoder sets. Its memory use is not
   measured, but it is clearly beyond 8 GB.

---

## 9. Why the DGX Spark is useful

The project's documents describe it as having **128 GB of unified memory** — 16× the RTX 4060.

1. **Everything fits at once, in full precision.** Both primary models in float32 at full batch size
   (*estimates* top out around 39 GiB), with no need to fall back to bf16.
2. **The circuit-tracer pipeline becomes possible:** the model and its transcoder sets loaded together.
3. **Cells run in parallel.** The grid is model × task × compression × seed with no shared state, so
   several cells can run simultaneously — the real source of time savings.
4. **Room for compression itself.** GPTQ/AWQ calibration, and holding a dense model next to its
   compressed copies, need memory the RTX 4060 lacks.
5. **Large-vocabulary models.** Gemma-2's large vocabulary makes its output layer expensive, which the
   unified memory absorbs.

**Caveats:**

- **Not necessarily faster per pass.** The proposal describes it as "high-memory / moderate-bandwidth",
  so its advantage is capacity and parallelism, not single-pass speed. This is unmeasured.
- **Re-run the timing tool on the Spark**, and re-apply the Q3 rule to the primary models' measured
  numbers. Every figure above 8 GB in this report is an estimate.
- **It is expected to be an ARM-based system.** Confirm that the pinned PyTorch/CUDA versions install
  on it before relying on them.

**Division of work:** the RTX 4060 handles all Pythia development and runs; the DGX Spark handles
Gemma-2, Llama-3.2 and the parallel grid — the split the proposal already plans.

---

## 10. Next steps

1. Accept the Gemma-2 and Llama-3.2 licences.
2. Build Stage B and Stage C on Pythia-160M.
3. Run the timing tool on the DGX Spark.
