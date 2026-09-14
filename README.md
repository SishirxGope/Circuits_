# Do Circuits Survive Compression?

**A Null-Model Audit of Circuit Stability Under Quantization and Pruning**
Target venue: AAAI Undergraduate Consortium

---

## The question

Interpretability findings are produced on full-precision models and consumed on
compressed ones. Recent audits have asked whether that transfer holds, but they study
*representations* — SAE features, linear probes. None studies *computation*. Circuits
are the structural unit the field actually reasons with ("the IOI circuit", "the
refusal pathway"), and their behaviour under quantization or weight pruning is
essentially unmeasured.

The difficulty is that circuit measurement carries two documented pathologies: circuits
reflect the analyst's pruning threshold as much as the model, and activation patching
estimates a quantity that mixes a component's own effect with interaction effects
(NIE = PIE + INT). So any claim that compression broke a circuit needs a **null model**
before it means anything.

> **The one sentence to internalise:** the question is not *"did the circuit change"*
> but *"did it change more than nothing-in-particular changes it"* — and the null
> distribution that defines "nothing in particular" is **frozen before any real
> compression run** and may never be re-tuned afterwards.

**Headline metric — Circuit Survival Index:**

```
CSI(c, T) = D(c) / median(D_null(c))          with a bootstrap CI over B, S and r
```

`CSI ≈ 1` → compression damages the circuit no more than generic weight noise of the
same magnitude, and the observed "damage" is not evidence about compression at all.
`CSI ≫ 1` → compression is structurally selective. `CSI < 1` → compression is *gentler*
than random noise, which would itself be a publishable surprise. **Every outcome is
pre-committed as publishable**; only an untrusted floor is a failure.

---

## Status

| | |
|---|---|
| Pipeline | All four Algorithm-1 stages + the freeze run end to end **on a synthetic stack** |
| Real models | **Pythia-160M dense-node extraction works** — pinned revision, edge attribution patching, checked against exact edge patches. Stage B and Stage C for real models are not built yet |
| Scientific decisions | **All pre-Stage-A decisions are made** (Q1–Q11 and the statistical settings, 2026-09-12/14) |
| Tests | 510 passing, 1 skipped (the Phase-1 exit gate, by design), with the `models` extra installed |
| `frozen/` | Empty. Decisions are pre-registered in `configs/` and `docs/preregistration/`; the null freeze has not happened |

**Nothing has been measured as evidence.** Only non-evidence pilots have run (timing and
threshold-grid diagnostics). Start here:

- [`docs/PROGRESS.md`](docs/PROGRESS.md) — where the project stands and what comes next
- [`docs/HUMAN_DECISIONS.md`](docs/HUMAN_DECISIONS.md) — every decision, with its reasoning
- [`docs/BOTTLENECKS_AND_HARDWARE.md`](docs/BOTTLENECKS_AND_HARDWARE.md) — what is slow, and what each tier of hardware can achieve
- [`docs/Run_Plan.md`](docs/Run_Plan.md) — the ordered runbook

---

## Quick start

```bash
pip install -e .
python -m pytest                        # synthetic stack only
python -m experiments.dry_run_stage_a 0  # synthetic Stage A: no model, no GPU
```

For real models, install PyTorch for your CUDA version first, then the pinned model stack:

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[models]"
python -m pytest                        # 510 passed, 1 skipped
```

Test tiers live in the filename, not in directories:

```bash
pytest -k "not integration and not regression"   # unit
pytest -k "integration"                          # integration
pytest -k "regression"                           # regression (before any tag / the freeze)
```

---

## Algorithm 1

```
Stage A  dense reference     extract B configs x S seeds -> inclusion frequencies s(e)
Stage B  null distribution   R matched-magnitude perturbations -> D_null
  FREEZE                     <-- pre-registration. Irreversible. frozen/ becomes append-only
Stage C  real compression    compress -> re-extract -> D(c) -> CSI, at BOTH levels
Stage D  causal + cross-audit damage ranking, Spearman vs published feature damage
```

The ordering is enforced in code, not by convention: `src/common/stage_guard.py`
refuses any Stage C cell whose null is not frozen and hash-matched, and
`experiments/freeze_stage_b.py` refuses to freeze while any decision is open or the
distance function is not pre-registered.

```bash
python -m experiments.freeze_stage_b --config <resolved.json> --dry-run   # always first
python -m experiments.freeze_stage_b --config <resolved.json>             # irreversible
```

---

## Layout

```
src/science/        🔒 THE NOVELTY ZONE — the entire contribution, one package.
                       nulls, CSI, distances, inclusion frequencies, decomposition,
                       two-level comparison, chance floors, FDR, patch diagnostic.
                       AI agents may read and propose, never modify without approval.
src/common/         schemas, seeding, hashing, run naming, guards, freeze, CSI table
src/compression/    RTN, GPTQ, AWQ, magnitude, Wanda
src/extraction/     pipelines (attribution graph, edge pruning, dense-node, mock), the pinned
                       model loader, edge attribution patching, dense-node threshold pruning
src/tasks/          real prompt sets: IOI and greater-than (clean/corrupted pairs + metrics)
src/synthetic/      mock model + synthetic tasks (dry-runs only, never evidence)
configs/            Hydra groups; IMMUTABLE once a run references them
experiments/        one entrypoint per stage, the freeze, and the Q3 timing pilot (non-evidence)
frozen/             the frozen null store — append-only after the freeze commit
analysis/           🔒 cross_audit.py, threshold_sweep.py, figure scripts
data/               dataset and calibration-corpus provenance
docs/               PROGRESS, HUMAN_DECISIONS, Run_Plan, BOTTLENECKS_AND_HARDWARE,
                       implementation_log, project_history
docs/preregistration/  criteria committed before the analyses they govern
docs/reports/       the proposal, group report and earlier progress report
docs/archive/       superseded files kept for the record
THIRD_PARTY_LICENSES/  notices for every upstream whose code is adapted
runs/               run directories and pilot reports (gitignored)
```

Governing documents live one level up and are not included in this repository:
`PRD.md` (claims → experiments contract), `AI_RULES.md` (integrity guardrails,
non-negotiable) and `ARCHITECTURE.md` (system blueprint). `CLAUDE.md` (research identity +
paper-to-code map) is included.

---

## The rules that shape the code

These are not style preferences; they are why the code refuses things.

- **Null before effect.** Stage C cannot run for a cell whose Stage B null is not
  frozen. Every CSI verifies the `null_frozen_hash` it divides by and fails hard on
  mismatch.
- **Nothing binary.** No circuit is ever reported as an edge list. The reported object
  is always the inclusion-frequency vector `s(e)` over the ensemble.
- **Two levels or it does not count.** Every claim is stated at both the exact-edge and
  routing-head level. The CSI table writer refuses a table carrying only one.
- **Floors attached.** Every overlap statistic carries its random top-k chance floor;
  every circuit-change statistic carries its frozen null.
- **Intervals, not points.** A number without a CI does not enter a figure.
- **Append-only history.** `frozen/` is never edited. Runs are never overwritten.
  Broken results are marked invalid, never deleted.
- **Negative results are results.** `CSI ≈ 1` everywhere is pre-committed as
  publishable. Iterating on analysis choices until an effect appears is forbidden.

---

## Reproducibility

Every random draw takes an explicit seed from config and is logged. Every run writes
its resolved config, that config's hash, and the full tag set to `run_meta.json`. Every
frozen null carries a SHA-256 that is re-verified on read. `verify_frozen_store()`
re-hashes the entire store and reports any drift — run it before any Stage C batch.

Real models load only at the commit sha pinned in `configs/model/*.yaml`, and the loader
refuses if the architecture it builds does not match that checkpoint. The model libraries
are pinned to exact versions in `pyproject.toml`.

## Licensing

The artifact is MIT-licensed (`LICENSE`). Model weights are never redistributed (Gemma-2
and Llama-3.2 licences); the artifact ships configs, scripts and hashes. Upstream licences
are in `THIRD_PARTY_LICENSES/`, and a test fails if any of them goes missing or leaves git.
No human-subjects data, no PII, no IRB.

## AI assistance

Substantial portions of this codebase were AI-generated under the constraints in
`AI_RULES.md`. Every such file carries an `[AI-GEN]` header and is logged in
`docs/implementation_log.md`, which is the source of truth for the paper's
reproducibility statement. Files marked `reviewed-by: PENDING` have not been approved
as scientific logic.
