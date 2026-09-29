# BLOCKERS — what must be implemented before any of these scripts can produce science

**Verified by running `python deploy/shared/preflight_blockers.py` on 2026-09-21.**
That script is the authority, not this document. Re-run it whenever you want the truth.

Every stage runner in this repository works today — **on the mock model.** Hand any of them
a real model and it raises `NotImplementedError`. The deployment scripts in
`plan_a_local_pc/` and `plan_b_dgx_spark/` therefore call the preflight first and refuse to
start, rather than failing four hours into a queue at night.

Current state: **4 of 10 checks blocked** (was 8 of 9 on 2026-09-21). The tenth check,
Q7 calibration caches, was added 2026-09-29: it is missing *data*, not missing code.

Run `python deploy/shared/preflight_blockers.py` for the live status; this table is a
summary and can drift.

| ID | Item | Status | Zone |
|---|---|---|---|
| B1 | Matched-magnitude null on real models | **DONE** — `matched_magnitude.py` delegates to `compression/torch_weights.py`; PI-approved 2026-09-27 | 🔒 novelty |
| B2 | RTN compressor | **DONE** — `rtn.py::rtn_quantize_torch`, per-output-channel, 2026-09-27 | engineering |
| B2 | Magnitude pruner | **DONE** — `magnitude_prune.py::prune_magnitude_torch`, 2026-09-27 | engineering |
| B2 | GPTQ compressor | BLOCKED — needs the Q7 calibration cache | engineering |
| B2 | AWQ compressor | BLOCKED — needs the Q7 calibration cache | engineering |
| B2 | Wanda pruner | **DONE** (code) — `magnitude_prune.py::prune_wanda_torch`, masks identical to upstream's own function, 2026-09-29 | engineering |
| Q7 | Calibration caches | BLOCKED — not built; `experiments/build_calibration_cache.py`, needs `allow_external_dataset_download` | data |
| B3 | Chance-floor universe | **DONE** — `run_stage_c.py::_chance_floor_universes`, structural and per-level, 2026-09-27 | 🔒 novelty |
| B4 | Normalised L1 | **DONE** — `distances.py::normalized_l1_distance`, approved 2026-09-12 | 🔒 novelty |
| B6 | GPT-2 IOI exit gate | BLOCKED (test exists, skipped) — needs a PI-supplied reference edge list + tolerance | engineering |

**Wanda is waiting on data, not code.** Its real path, the token-cache builder and the
E[x^2] collector all exist and are tested (`src/calibration/`). What is missing is the cache
itself: fineweb-edu, 300k tokens, **seed 7**, bf16 forward (`configs/calibration/final.yaml`),
one per tokenizer, built on the Spark with

    python experiments/build_calibration_cache.py mode=scientific_run model=<name>

for each of `pythia160m pythia410m gemma2_2b llama32_1b`. That needs
`mode.allow_external_dataset_download`. The first Wanda cell then collects E[x^2] once per
model and every later cell, in either stage, reuses the same file.

GPTQ and AWQ will calibrate on the same cache, but they still need code, and GPTQ needs more
than Wanda's statistic: the full input Hessian X^T X per layer, not just its diagonal.

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
4. **GPTQ**, 5. **AWQ** — see the dependency warning below.

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

**File:** `tests/test_regression_ioi_gpt2_small.py` (exists, `skipif`-gated).

Until this passes, **no result counts.** It extracts the IOI circuit from GPT-2 small and
checks it recovers the published heads — validating the extraction stack against a known
reference. Without it, a null result is indistinguishable from a broken extractor.

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
