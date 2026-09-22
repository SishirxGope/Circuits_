# BLOCKERS — what must be implemented before any of these scripts can produce science

**Verified by running `python deploy/shared/preflight_blockers.py` on 2026-09-21.**
That script is the authority, not this document. Re-run it whenever you want the truth.

Every stage runner in this repository works today — **on the mock model.** Hand any of them
a real model and it raises `NotImplementedError`. The deployment scripts in
`plan_a_local_pc/` and `plan_b_dgx_spark/` therefore call the preflight first and refuse to
start, rather than failing four hours into a queue at night.

Current state: **8 of 9 checks blocked.**

| ID | Item | Status | Zone |
|---|---|---|---|
| B1 | Matched-magnitude null on real models | BLOCKED | 🔒 novelty |
| B2 | RTN compressor | BLOCKED | engineering |
| B2 | GPTQ compressor | BLOCKED | engineering |
| B2 | AWQ compressor | BLOCKED | engineering |
| B2 | Magnitude pruner | BLOCKED | engineering |
| B2 | Wanda pruner | BLOCKED | engineering |
| B3 | Chance-floor universe | BLOCKED (**silent**) | 🔒 novelty |
| B4 | Normalised L1 | **DONE** — `distances.py::normalized_l1_distance`, approved 2026-09-12 | 🔒 novelty |
| B6 | GPT-2 IOI exit gate | BLOCKED (test exists, skipped) | engineering |

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
3. **Wanda** — score `|W| · ‖X‖₂` per input channel, pruned per output row. Needs calibration
   activations (FineWeb-Edu, 300k tokens, seed 7 — `configs/calibration/final.yaml`).
   Reference arXiv:2306.11695 — **verify against the paper, never code from memory.**
4. **GPTQ**, 5. **AWQ** — see the dependency warning below.

> **The existing stubs say magnitude and Wanda "wrap `saediag.pruning`" from
> `sae-pruning-paper`.** That package is **not installed and the fork is not on this machine**
> (see "Missing dependencies" below). Decide deliberately: wrap the upstream package once it
> is available, or implement in torch. Wrapping requires the `# Adapted from: <url> @ <commit>,
> <license>` header (`AI_RULES.md` §7) and the license in `THIRD_PARTY_LICENSES/`.

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

**File:** `tests/test_regression_ioi_gpt2_small.py` (exists, `skipif`-gated).

Until this passes, **no result counts.** It extracts the IOI circuit from GPT-2 small and
checks it recovers the published heads — validating the extraction stack against a known
reference. Without it, a null result is indistinguishable from a broken extractor.

Note the caveat in `CLAUDE.md` §6: published reference circuits exist for GPT-2 small, **not**
for Pythia, Gemma or Llama. The gate is a sanity check on the stack, not a claim about those
models.

---

## Missing dependencies (not code — files that are absent)

**Neither upstream fork is on this machine.** `CLAUDE.md` §3 places both at the workspace
root, beside the repo. Searched 2026-09-21: absent.

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
