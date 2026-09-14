# Progress — Do Circuits Survive Compression?

Last updated: **2026-09-14**. This is my running record of where the project stands. The binding
decision record is [`HUMAN_DECISIONS.md`](HUMAN_DECISIONS.md); the plan is
[`Run_Plan.md`](Run_Plan.md); what limits speed, and what hardware would change it, is in
[`BOTTLENECKS_AND_HARDWARE.md`](BOTTLENECKS_AND_HARDWARE.md). AI-assisted work is logged
file-by-file in [`implementation_log.md`](implementation_log.md), as AI_RULES §7 requires.

---

## Where things stand

| | |
|---|---|
| **Evidence measured** | None yet. Only non-evidence pilots have run (timing, grid diagnostics). |
| **Decisions** | All pre-Stage-A decisions are made, including Q3 (B = 16, S = 5, R = 20). |
| **Real models** | Pythia-160M loads at its pinned revision, and dense-node circuits extract from it. |
| **Tests** | 510 passed, 1 skipped (the Phase-1 exit gate, by design), with the `models` extra installed. |
| **`frozen/`** | Empty. The null freeze has not happened. |
| **Next real step** | Build Stage B and Stage C for real models, on Pythia-160M. |

---

## What I have done

### 1. Repository hygiene

- Restored the upstream licence files, which had been deleted and then gitignored, so every clone
  was missing notices that the MIT licences require. Added tests that fail if they are deleted
  **or** untracked.
- Pinned exact library versions for model work (`torch 2.14.0`, `transformers 5.17.0`,
  `transformer-lens 3.9.0`, `huggingface-hub 1.31.0`).

### 2. Pins

- **Q9** — upstream commits: circuit-tracer `8f1e2438` (tag v0.5.2), sae-pruning-paper `26119180`
  (inferred from the fork's contents and recorded as inferred).
- **Q5** — all eight HuggingFace revisions, including both transcoder sets; each is asserted by
  value in the tests.

### 3. Pre-registration

| Area | What I fixed in advance |
|---|---|
| Distance (Q2) | L1 primary, Jensen–Shannon ablation, normalised L1 reported alongside |
| Bands (Q1) | CIRCUS's taxonomy: core s = 1, contingent 0.5 ≤ s < 1, noise s < 0.5 (strict) |
| Threshold grid (Q4) | Anti-diagonal, B = 16, node 0.6 → 0.9 against edge 0.99 → 0.95 |
| Coarse level (Q10) | Layer, with a pre-registered falsification condition |
| Positions (Q11) | Aggregated |
| Tasks (Q6) | IOI (30 templates) and greater-than (120 nouns), 300 prompts each, no BOS |
| Corpora (Q7) | FineWeb-Edu for calibration (300,000 tokens, seed 7), WikiText-2 test for perplexity |
| Statistics | CSI bootstrap over B, S and r; FDR q = 0.05; W&B as tracker |
| Ensemble sizes (Q3) | B = 16, **S = 5, R = 20** — Run_Plan's pre-stated rule applied to the measured timing |
| Chance-floor N | Structurally possible edges among observed nodes |

### 4. Real dense-node extraction on Pythia-160M

Before this, the real-model loader, Pythia's only extractor, and every real task loader were
stubs. Now built and tested:

- **Pinned loader** (`src/extraction/real_model.py`): loads through `transformers` at the pinned
  sha, asserts the architecture against the pinned config, and enforces the download gate in code.
- **Task prompts** (`src/tasks/`): IOI with ABC corruption, and greater-than with year-suffix
  corruption, ported from ACDC and re-derived for Pythia's tokenizer.
- **Edge attribution patching** (`src/extraction/eap.py`): split Q/K/V inputs, |sum| aggregation,
  and no phantom same-layer attention → MLP edges on parallel-residual models.
- **Threshold pruning** (`src/extraction/dense_prune.py`): circuit-tracer's cumulative-fraction
  semantics, applied to total-effect scores.
- **Extractor with caching** (`src/extraction/dense_node_variant.py`): one attribution pass per
  (model, task, seed), re-pruned for every threshold view.
- **Timing pilot** (`experiments/time_attribution.py`): non-evidence by construction.

### 5. Checks that passed

- Weights loaded at the pinned sha are bit-identical to the pinned checkpoint.
- Both tasks run on the dense model, and the corruptions remove the signal: IOI logit difference
  +4.56 → −0.27, greater-than probability difference +0.763 → −0.640.
- Every EAP edge score matches a finite difference of an exact edge patch.
- Pythia-160M has 32,347 candidate edges, matching a hand count.

### 6. A finding I am reporting rather than hiding

The pre-registered grid produces **largely nested views** on dense-node scores (82–116 of 120
view pairs nested). I pre-registered a fix — criterion, candidate list and selection rule —
committed it before looking, ran it exactly as stated, and **no candidate passed**. I kept the
grid as pre-registered and will report both results. The full record is in
[`preregistration/`](preregistration/).

### 7. Corrections I made to my own records

- Greater-than has **120** nouns, not 26 — the earlier figure counted lines, not nouns.
- The 30 IOI templates come from Easy-Transformer via ACDC, **not** TransformerLens, whose dataset
  has only 2.
- Greater-than is a **year-span** task, not day-of-month.

---

## Measured so far (RTX 4060, Pythia-160M, non-evidence)

| | IOI | Greater-than |
|---|---|---|
| Attribution pass | 42–48 s | 7.5–7.9 s |
| Peak VRAM | 4.26 GiB | 2.64 GiB |
| Cost per cell at S = 5, R = 20 | ~80 min | ~14 min |

That is roughly **19 GPU-hours** for Pythia-160M's two tasks across ~12 compression cells.

---

## What comes next, in order

1. **Accept the Gemma-2 and Llama-3.2 licences** on HuggingFace — manual approval takes days.
2. **Stage B for real models:** matched-magnitude perturbation of a TransformerLens model.
3. **Stage C for real models:** RTN, GPTQ, AWQ, magnitude and Wanda.
4. **Implement the chance-floor N rule** in Stage C.
5. **Port the docstring generator** and check it under Pythia's tokenizer.
6. **Approve and time Pythia-410M.**
7. **Grouped-query-attention support** in the EAP code, then time the primaries.
8. **Pipeline A** (circuit-tracer) for the primaries.
9. **The Phase-1 exit gate:** reproduce the published GPT-2 IOI circuit.
10. **Stage B on every cell, then the freeze.** After that, Stage C, Stage D, and the paper.

Before Stage D I still need to decide the C4 interaction-flag ratio and the C5 grid alignment,
and the attribution precision for the primaries before they run.
