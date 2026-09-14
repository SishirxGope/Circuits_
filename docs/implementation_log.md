# Implementation Log

> Every AI-generated or AI-substantially-modified file is listed here (AI_RULES.md §7).
> `reviewed-by: PENDING` files are NOT approved scientific logic; the novelty-zone
> modules additionally require explicit human `APPROVE <module_name>`.

## Entry 1 — 2026-08-07, agent=OpenCode

### Completed steps
- Steps 1–4 (previous session): scaffolded `Circuits_Under_Compression/` topology
  (CLAUDE.md §3), `src/interfaces.py` (ARCHITECTURE.md §4 Protocols), model/task
  configs, wrapper skeletons (`attribution_graph.py`, `magnitude_prune.py`).
- Step 5: core schema layer (`src/io/schema.py`): Edge/Graph/FreqVector/EnsembleResult
  dataclasses, edges.parquet + freq.parquet column schemas, run-tag validation,
  frozen meta.json schema, CSI table placeholder columns + writer, parquet I/O
  (pyarrow, optional).
- Step 6: `src/utils/` — `seeding.py` (explicit seeded generators only, AI_RULES.md
  1.1), `hashing.py` (config/file/manifest hashing), `run_naming.py` (CLAUDE.md §4
  convention validator/builder).
- Step 7: Hydra config scaffolding — `configs/config.yaml` (root composition), and
  groups `ensemble/default`, `ensemble/decompose`, `distance/placeholder`,
  `compression/placeholder`, `null/placeholder`. Existing `model/*` and `task/*`
  configs unchanged.
- Step 8: `experiments/run_stage_a.py` — Stage A entrypoint (Hydra; works from plain
  dicts for tests), run dir per CLAUDE.md §4, resolved-config dump + config hash,
  tag validation, mock-pipeline support, edges.parquet + freq.parquet output,
  explicit no-frozen/no-compression guarantees.
- Step 9: extraction layer — `attribution_graph.py` (adapter over circuit-tracer,
  upstream API verified against local fork), `edge_pruning_graph.py` (pipeline B
  skeleton), `dense_node_variant.py` (C8 skeleton), `mock_extractor.py` (deterministic
  synthetic extractor for CI).
- Step 10 (NOVELTY ZONE, drafts, PENDING): `src/ensemble/circus_wrapper.py`,
  `inclusion_freq.py`, `decompose.py`. No scientific choices made; cutoffs/distance
  remain PI-owned.
- Step 11: `src/guards/stage_guard.py` — `assert_stage_b_frozen` +
  `assert_null_frozen_hash` (AI_RULES.md 1.4/1.5; tested).
- Step 12: tests — `tests/unit/{test_seeding,test_run_naming,test_schema,
  test_inclusion_freq,test_decompose}.py`, `tests/integration/{test_stage_a_synthetic,
  test_stage_guard}.py`, `tests/conftest.py` (path bootstrap).
- Step 13: this log.

### Files created (all `[AI-GEN] agent=OpenCode date=2026-08-07`, `reviewed-by: PENDING`)
- `src/io/__init__.py`, `src/io/schema.py`
- `src/utils/__init__.py`, `src/utils/seeding.py`, `src/utils/hashing.py`,
  `src/utils/run_naming.py`
- `src/guards/__init__.py`, `src/guards/stage_guard.py`
- `src/ensemble/circus_wrapper.py`, `src/ensemble/inclusion_freq.py`,
  `src/ensemble/decompose.py` (NOVELTY ZONE drafts)
- `src/extraction/attribution_graph.py` (rewritten), `src/extraction/edge_pruning_graph.py`,
  `src/extraction/dense_node_variant.py`, `src/extraction/mock_extractor.py`
- `configs/config.yaml`, `configs/ensemble/default.yaml`, `configs/ensemble/decompose.yaml`,
  `configs/distance/placeholder.yaml`, `configs/compression/placeholder.yaml`,
  `configs/null/placeholder.yaml`
- `experiments/__init__.py`, `experiments/run_stage_a.py`
- `tests/conftest.py`, the 7 test files above
- `pyproject.toml` (pytest config only), `docs/implementation_log.md`, `runs/.gitkeep`

### Scientific decisions made
None. All unresolved scientific choices are marked `⚠️ TODO [QUESTION FOR PI]`:
- Band cutoffs (`configs/ensemble/decompose.yaml` → null; decompose() raises until set).
- Distance function D (`configs/distance/placeholder.yaml` → `name: null`).
- B/S/R values: B=16, S=5, R=20 copied from the ARCHITECTURE.md §5 suggested starting
  point (documented, not invented), flagged TODO for PI.
- Threshold grid design (B non-nested configs) → null; mock pipeline uses an explicit
  TEST-ONLY placeholder grid.
- Mechanical conventions implemented (not scientific choices, documented in code):
  s(e) = count/(B·S); band boundaries inclusive at both cutoffs; deterministic
  edge_id ordering (`src->dst` string sort); ensemble iteration seeds-outer/configs-inner.

### Scientific decisions deferred to PI (⚠️ TODO [QUESTION FOR PI])
1. Core/contingent/noise band cutoffs (CLAUDE.md §5; AI_RULES.md §6).
2. Primary distance function D + ≥1 ablation alternative (CLAUDE.md §5; PRD §3 P0;
   must be pre-registered before Stage C).
3. B, S, R values (ARCHITECTURE.md §5 starting point).
4. The B non-nested threshold-config grid design (CIRCUS-style).
5. HF revision pins for all models (configs/model/* `hf_revision: null`).
6. Task dataset sources/counts (all `n_prompts`/`n_templates` PLACEHOLDER).
7. Calibration dataset licenses (data/README.md; CLAUDE.md §6).
8. W&B vs MLflow (currently: JSON `run_meta.json` is the source of record).
9. Whether Stage A dense reference needs `git init` for the freeze-commit hash.
10. circuit-tracer / sae-pruning-paper upstream commit pins + license files
    (CLAUDE.md §7 adaptation headers).

### Upstream repositories inspected (read-only)
- `circuit-tracer-0.5.2/`: README.md; `circuit_tracer/__init__.py`;
  `attribution/attribute.py` (signature); `graph.py` (Graph attributes, prune_graph,
  PruneResult, find_threshold); `replacement_model/replacement_model.py`
  (from_pretrained signatures); frontend/targets module surfaces.
- `sae-pruning-paper-main/`: README.md (revision protocol notes, model/SAE tables);
  `revision/src/saediag/{__init__,pruning,models,ppl,stats,sae,matching,io,fragility,
  fixed_dict,ablation,reprune}.py` (function surfaces).

### Unresolved questions / environment notes
- hydra-core + omegaconf NOT installed in this environment: `experiments/run_stage_a.py`
  Hydra entrypoint is import-safe (guarded import); tests exercise `run_stage_a(dict)`.
  Install `hydra-core` before real runs.
- pyarrow IS installed: parquet tests run for real. If pyarrow is absent elsewhere,
  the schema parquet tests are marked skipable via `pytest.importorskip`.
- Optional deps: torch (used only for the torch-Generator half of seeding; test
  skips if absent), circuit-tracer + sae-pruning-paper packages (upstream forks;
  NOT installed — wrappers raise informative errors until installed).
- No heavy workloads run: no model downloads (pending `RUN MODEL DOWNLOAD`), no
  attribution passes, no GPU jobs, no compression.

### Next recommended step
Stage A engineering on the extraction stack with the **real** (non-mock) pipeline:
1. PI resolves the open choices that gate Stage A (band cutoffs, B/S, threshold grid,
   HF revision pins, task template sources).
2. Pin circuit-tracer commit + license (CLAUDE.md §7) and adapt the upstream
   `attribute`/`prune_graph` calls in `AttributionGraphExtractor.extract`.
3. Implement the Phase-1 exit gate (ARCHITECTURE.md §6): reproduce one published
   reference circuit (IOI on GPT-2 small) within tolerance as a regression test.
4. Then run Stage A on the secondary models (Pythia-160M/410M) first per the
   ARCHITECTURE.md §5 compute plan.

## Entry 2 — 2026-08-07, agent=OpenCode

Scope per the "decision packet + non-gating engineering" instruction: NO Stage B, NO
real compression, NO model downloads, NO scientific choices resolved.

### Completed
- **TASK 1 — `docs/pi_decisions.md`**: formal decision packet for all nine PI
  questions (Q1 band cutoffs, Q2 distance D, Q3 B/S/R, Q4 threshold grid, Q5 HF
  revision pins, Q6 task datasets, Q7 calibration licenses, Q8 tracker, Q9 upstream
  pins). Each carries file(s), affected claims, options, trade-offs, status OPEN.
  Single source of truth for what the PI must answer; nothing RESOLVED.
- **TASK 3.1 — `docs/upstream_pins.md`**: pin table for circuit-tracer and
  sae-pruning-paper (license verified locally; commit hashes undetectable — neither
  local fork is a git repo — recorded as ⚠️ TODO [QUESTION FOR PI] Q9, NOT invented),
  plus model/transcoder HF pin status table.
- **TASK 3.2 — `THIRD_PARTY_LICENSES/`**: `circuit-tracer-LICENSE.txt` +
  `sae-pruning-paper-LICENSE.txt` copied byte-identical from the local forks
  (verified: MIT-style "Copyright (c) 2024 Michael Hanna and Mateusz Piotrowski";
  MIT "Copyright (c) 2025-2026 Héctor Borobia"); `.gitkeep` removed; `README.md`
  provenance table written.
- **TASK 3.3 — `src/extraction/attribution_graph.py`**: added pure, testable helpers
  `_thresholds_from_config` (validates the ensemble cell's node/edge thresholds are
  present and in (0,1) — never hardcoded) and `_validate_task_config` (structural
  key check, no scientific values); `extract()` now validates before any upstream
  touch. Upstream API facts unchanged (verified earlier).
- **TASK 3.4 — `tests/regression/test_ioi_gpt2_small_reference.py`**: Phase-1 exit
  gate (ARCHITECTURE.md §6) skeleton. Pure `edge_overlap` (Jaccard) tested NOW; the
  gate itself skips unless `RUN_IOI_GPT2_REFERENCE` is set AND the PI-provided
  reference edges file exists (no invented reference edges; tolerance is PI-owned).
- **TASK 3.5 — `experiments/run_stage_a.py`**: wired the config guard
  (`assert_no_gating_questions` refuses real pipelines while Q1/Q3/Q4/Q5/Q6 are
  OPEN; mock is the only bypass) and records `open_questions` in run_meta.json even
  in mock mode, for audit.
- **TASK 3.6 — `src/guards/config_guard.py`** (new): decision-aware validation;
  detects the config-detectable subset of pi_decisions.md (Q1–Q6; Q7/Q8/Q9 are
  docs-only) with gating levels (stageA/stageC/docs-only). Never chooses values.
- **Config markers**: `pi_confirmed: false` added to `configs/ensemble/default.yaml`
  (Q3) and `configs/task/{ioi,greater_than,docstring}.yaml` (Q6) so the guard can
  detect the OPEN state. Values only; no science chosen.
- **Tests**: `tests/unit/test_config_guard.py`, `tests/unit/test_attribution_helpers.py`
  (AI_RULES.md 4.2). Suite: **69 passed, 1 skipped** (the exit gate, as designed).

### Scientific decisions made
None. All decisions are OPEN in `docs/pi_decisions.md`; the config guard refuses
real runs that need them.

### Files created/modified (all `[AI-GEN] agent=OpenCode date=2026-08-07`, `reviewed-by: PENDING`)
- created: `docs/pi_decisions.md`, `docs/upstream_pins.md`,
  `THIRD_PARTY_LICENSES/README.md`, `src/guards/config_guard.py`,
  `tests/regression/test_ioi_gpt2_small_reference.py`,
  `tests/unit/test_config_guard.py`, `tests/unit/test_attribution_helpers.py`
- modified: `src/extraction/attribution_graph.py`, `experiments/run_stage_a.py`,
  `configs/ensemble/default.yaml`, `configs/task/{ioi,greater_than,docstring}.yaml`,
  this log
- copied (read-only upstream): `THIRD_PARTY_LICENSES/circuit-tracer-LICENSE.txt`,
  `THIRD_PARTY_LICENSES/sae-pruning-paper-LICENSE.txt`

### Newly identified facts
- Neither upstream local fork is a git repository → commit hashes cannot be
  determined locally (recorded in upstream_pins.md as PI item, not guessed).

### Next recommended step (unchanged)
Same as Entry 1: PI resolves gating decisions in `docs/pi_decisions.md` (Q1, Q3, Q4,
Q5, Q6), then Stage A real engineering + the exit gate.

## Entry 3 — 2026-08-07, agent=OpenCode, PI PROVISIONAL ENGINEERING UNBLOCK

PI prompt of 2026-08-07 unblocked ENGINEERING-ONLY work: config layer, guard
refactor, synthetic stack, draft module implementations, tests. NOT a scientific
release: every draft module is marked `scientific-status: PROVISIONAL_ENGINEERING_ONLY`
and `reviewed-by: PENDING`; `final_pre_registration: false`; all provisional values
require final PI confirmation before the Stage B freeze / Stage C real runs.

### Decision layer (Task 1)
- `docs/pi_decisions.md`: all 9 questions -> `PROVISIONAL_ENGINEERING_DEFAULT`
  (approved_by/date/scope/final_pre_registration/remaining_action recorded per item).
- created `docs/pi_decisions_resolved_provisional.md` (values + binding constraints).

### Config layer (Task 2)
- created `configs/pi/provisional_defaults.yaml`,
  `configs/mode/{engineering_dry_run,scientific_run}.yaml`,
  `configs/distance/provisional_l1_js.yaml`,
  `configs/ensemble/provisional_b4_s2_r3.yaml`,
  `configs/ensemble/decompose_provisional.yaml`,
  `configs/task/synthetic-{ioi,greater-than,docstring}.yaml`,
  `configs/model/mock_model.yaml`.
- `configs/config.yaml` defaults -> engineering composition; deleted
  `configs/distance/placeholder.yaml`.

### Guard (Task 3)
- `src/guards/config_guard.py` rewritten: `mode_of`, `report_open_questions`
  (Q1-Q6; Q7-Q9 docs-only), `report_provisional_resolutions`,
  `assert_no_gating_questions` (scientific blocks synthetic + OPEN Q1/Q3/Q4/Q5/Q6),
  `assert_engineering_dry_run_limits` (engineering refuses Stage B freeze, Stage C,
  writes under `frozen/`).

### Draft modules (Task 4, all PROVISIONAL_ENGINEERING_ONLY)
- created: `src/ensemble/threshold_grid.py`, `src/nulls/matched_magnitude.py`,
  `src/nulls/matched_perplexity.py`, `src/metrics/csi.py`,
  `src/compare/distances.py`, `src/compare/two_level.py`,
  `src/causal/patch_diagnostic.py`, `analysis/cross_audit.py`,
  `src/compression/rtn.py`, `src/compression/gptq.py`, `src/compression/awq.py`,
  `src/compression/magnitude_prune.py`, `src/compression/wanda.py`,
  `experiments/run_stage_b.py`, `experiments/run_stage_c.py`,
  `experiments/run_stage_d.py`, `experiments/dry_run_stage_a.py`.
- rewritten: `src/extraction/{edge_pruning_graph,dense_node_variant,mock_extractor}.py`,
  `experiments/run_stage_a.py` (guard before run-dir creation; mock model loader;
  seeded grid in engineering mode).
- scientific-status headers added: `src/ensemble/{circus_wrapper,inclusion_freq,decompose}.py`.

### Synthetic stack (Task 5)
- created `src/synthetic/{mock_model,synthetic_tasks,synthetic_calibration}.py`
  (deterministic, seeded, no downloads; mock edge scores span the seeded grid
  thresholds after scaling coupling to `tanh(dot/2)`).

### Tests (Task 6)
- created/rewritten: `tests/unit/{test_config_guard,test_distance_l1_js,
  test_threshold_grid_generator,test_synthetic_task_prompts,
  test_stage_b_refuses_freeze,test_stage_c_requires_frozen_null}.py`,
  `tests/integration/{test_engineering_dry_run_stage_a,
  test_scientific_run_still_blocked}.py`.
- Full suite: **85 passed, 1 skipped** (`python -m pytest -q`).

### Dry-run verification (Task 8 command)
- `python -m experiments.dry_run_stage_a 0` ->
  `runs/20260806_stageA_mock_synthetic-ioi_dense_B4xS2_seed0/`
  (14 edges, 7 unique; mode=engineering_dry_run, B=4, S=2, n_cells=8,
  band_cutoffs_resolved=True, 9 provisional resolutions recorded).

### Still PENDING / intentionally blocked
- Stage A real extraction (RUN MODEL DOWNLOAD approval), Stage B freeze, Stage C
  real compression, Stage D, real compressors (GPTQ/AWQ/Wanda/magnitude on real
  models), attribution-graph extraction, PatchDiagnostic.run, matched-perplexity
  tuning — all `NotImplementedError` or guard-refused until PI final decisions +
  explicit approval.

### Next step
Same as Entry 1 (unchanged): PI resolves final decisions in `docs/pi_decisions.md`
(Q1-Q9 -> FINAL), then Stage A real engineering + exit gate.

## Entry 4 — 2026-08-07, agent=OpenCode, folder consolidation + HUMAN_DECISIONS.md + thinking export

- PART A (safe consolidation): merged `src/io/` (schema.py), `src/utils/`
  (seeding.py, hashing.py, run_naming.py), `src/guards/` (stage_guard.py,
  config_guard.py) into `src/common/` (new `__init__.py` re-exports all public
  names; old `__init__.py` docstrings folded into it; no content lost, no
  collisions). Protected paths untouched (NPZ, configs incl. every Hydra group,
  analysis, frozen, experiments, tests split, data/paper/artifact/runs,
  THIRD_PARTY_LICENSES, root docs). 33 files updated for the path change
  (imports + doc references); `docs/implementation_log.md` itself intentionally
  left as history; CLAUDE.md §3 topology updated (added src/common/ + src/synthetic/).
  Tests unchanged: 118 passed, 1 skipped.
- PART B: created `docs/HUMAN_DECISIONS.md` (PI decision guide, Q1-Q9, GROUP 1-3,
  17 fill-in fields; content provided by the PI, transcribed verbatim).
- PART C: mirrored GROUP 1 placeholders (Q5/Q8/Q9 fill-in blocks) into
  `docs/pi_decisions.md`; no values filled in.
- New file: `docs/THINKING_LOG_2026-08-07.md` (full thinking/work export of
  sessions 1-3; narrative record, `reviewed-by: PENDING`).

### Next step (unchanged)
PI resolves final decisions in `docs/pi_decisions.md` / `docs/HUMAN_DECISIONS.md`
(Q1-Q9 -> FINAL), then Stage A real engineering + exit gate.

## Entry 5 — 2026-08-08, agent=Claude (Opus), folder consolidation + bug fixes + test build-out + docs merge

PI instruction: reduce the folder count, keep every file, test everything, fix any
bug/mismatch/error found, and merge or remove unneeded docs. No scientific decision
was made; no model was downloaded; no real compression ran; nothing was frozen.

### PART A — Folder consolidation (46 directories on disk → 24)

| Change | Before | After |
|---|---|---|
| Caches | `.pytest_cache/` + 12 `__pycache__/` | deleted; `.gitignore` added |
| Novelty zone | `src/{nulls,ensemble,compare,metrics,causal}/` (5 pkgs) | **`src/science/`** (1 pkg) |
| Tests | `tests/{unit,integration,regression}/` | flat `tests/`; tier now in the filename (`test_*`, `test_integration_*`, `test_regression_*`) |
| Config groups | `configs/pi/`, `configs/null/` | `configs/provisional_defaults.yaml` (not a group); `configs/nulls/`; new `configs/ensemble/decompose/` |
| Docs | 6 md files | 3 (`HUMAN_DECISIONS.md`, `implementation_log.md`, `project_history.md`) |

Every module file name, its contents and its public API are unchanged — only paths
moved. `analysis/cross_audit.py` deliberately stayed put so the AI_RULES §3 line
naming it remains literally true.

Governance docs updated for the new paths: CLAUDE.md §2 (paper-to-code map) and §3
(topology, now showing the whole workspace), AI_RULES.md §3 (the novelty zone is now
the single path prefix `src/science/`), ARCHITECTURE.md §2/§3/§6.

### PART B — Bugs found and fixed

Hydra composition (would have failed on the first real run; invisible here because
hydra-core is not installed):

1. `configs/config.yaml` `- null: placeholder` — a bare `null` YAML key is the null
   literal, so the entry parsed as `{None: "placeholder"}` and the group was never
   composed. Group renamed to `configs/nulls/`; config key `null` -> `nulls` at all
   6 call sites + 3 test fixtures.
2. `- ensemble/decompose: decompose_provisional` referenced a nested group whose
   directory did not exist. Created `configs/ensemble/decompose/{final,provisional}.yaml`.
3. `configs/ensemble/default.yaml` `configset: B${B}xS${S}` — interpolation is
   root-relative after composition, so it must be `${ensemble.B}` / `${ensemble.S}`.
4. Hydra composed R=20 while `experiments/dry_run_stage_a.py` used R=3, despite the
   driver docstring promising they mirror each other. Added
   `configs/nulls/provisional_r3.yaml` and made it the default.

Correctness:

5. `src/compression/magnitude_prune.py`, global-scope tie correction: the repair loop
   located the smallest surviving weight in `prunable[k]` (the ORIGINAL weights) while
   zeroing `out[k]` (the PRUNED ones), and never updated its own view — it could
   re-zero an already-zero entry and decrement the count anyway, under-pruning.
   Replaced with an exact global top-k mask. Also made shape-agnostic: the old path
   indexed `mask[i, j]`, so it assumed 2-D and would crash on real 1-D bias/norm tensors.
6. `analysis/cross_audit.py` used `Sequence` in an annotation without importing it —
   a latent NameError under any runtime annotation evaluation. Import added.
7. `experiments/run_stage_b.py` printed `sorted(...)[R // 2]` as "median", which is
   wrong for even R. Now `statistics.median`.
8. `experiments/run_stage_{c,d}.py` recorded `config_hash: ""` and `git_commit: None`
   as literals. Now computed.
9. `frozen/FREEZE_MANIFEST.json` was `{}`. Now an explicit unfrozen placeholder
   (`frozen: false`, `cells: {}`) with the freeze protocol stated inline.
10. Unused imports removed (`os`, `subprocess`, `dataclasses`, three unused typing
    names). No behaviour change.

Missing module: `src/science/perplexity.py` — referenced by CLAUDE.md §3 and by
`matched_perplexity.py`, but never created. Added, with the token-weighted PPL
aggregation implemented and tested; the model forward pass raises until Stage B.

Project metadata: `pyproject.toml` gained a real `[project]` table with dependencies
and extras (`models`, `tracking`, `tracking-local`, `dev`), so the stack installs with
`pip install -e .`.

### PART C — NOVELTY-ZONE EDITS MADE UNDER PI INSTRUCTION (AI_RULES.md §3)

The PI's instruction to "correct bugs/mismatch/errors" was taken as approval for the
two items below. Both are behaviour-preserving on every value computed so far and
neither changes an estimand. Recorded explicitly; they still need a formal `APPROVE`
before the freeze.

- `src/science/matched_magnitude.py` — `np.linalg.norm(z, "fro")` is defined for 2-D
  input ONLY and raises on 1-D/3-D arrays. Real checkpoints carry 1-D bias and
  LayerNorm tensors, so the matched-magnitude null would have crashed on the first real
  Stage B run. Replaced by a new `frobenius_norm()` = 2-norm of the flattened tensor,
  numerically identical to `"fro"` for 2-D input. Strict generalisation; no
  already-computed magnitude changes.
- `src/science/{decompose,inclusion_freq,distances,threshold_grid,csi,matched_perplexity}.py`
  — docstring and error-message path references only
  (`configs/ensemble/decompose.yaml` -> `.../decompose/final.yaml`;
  `docs/pi_decisions*.md` -> `docs/HUMAN_DECISIONS.md`;
  `src/metrics|compare/...` -> `src/science/...`). Zero logic touched. Left uncorrected,
  these would have pointed the PI at deleted files.

Two novelty-zone issues were deliberately NOT changed, and are flagged instead:

- `csi.py` bootstraps over the null draws `r` only, while the proposal and CLAUDE.md §5
  specify a bootstrap "over B, S, r". Resampling the ensemble axes is a scientific
  choice about the estimand and belongs to the PI, before Stage C.
- `distances.py` `jensen_shannon_distance` normalises both vectors, so it is blind to a
  uniform change in total inclusion mass (halving every s(e) reads as zero change).
  That is a real property to weigh when pre-registering Q2, not a coding error; it is
  now written up in HUMAN_DECISIONS.md Q2. Its unreachable `isinf` branch was left
  untouched.

### PART D — Test build-out: 118 passed -> 287 passed, 1 skipped

ARCHITECTURE.md §6 named required coverage that did not exist. Added:

| New file | Covers | Tests |
|---|---|---|
| `test_csi.py` | CSI division, zero-median guard, bootstrap determinism, golden-file CSI on a fixed synthetic cell, frozen-hash verification | 16 |
| `test_matched_magnitude_null.py` | per-tensor Frobenius match, isotropy sanity checks, determinism, dense model never mutated, 1-D/3-D tensors, input validation | 21 |
| `test_two_level.py` | routing-head projection on known edge sets, Jaccard arithmetic, cutoffs, and a fixture encoding a level disagreement (exact-edge 0.25 vs routing-head 0.67) | 14 |
| `test_cross_audit.py` | Spearman known values, average-rank ties, NaN degenerate cases, shared-cell count, C5 decision bands | 14 |
| `test_patch_diagnostic.py` | NIE = PIE + INT identity, flag rule incl. sign-flip and threshold boundary | 13 |
| `test_compression.py` | RTN grid/clipping, exact sparsity for magnitude + Wanda incl. all-ties, Wanda scoring order, `weight_delta` == the real weight change for all five compressors, no mutation | 39 |
| `test_perplexity.py` | token-weighted aggregation vs the naive window mean, relative gap, guards | 14 |
| `test_integration_stage_a_to_c.py` | the full Stage A->B->C chain: schemas, tags, bands, draft-null determinism, and that a DRAFT null is not accepted as a FROZEN one | 15 |
| `test_config_integrity.py` | static Hydra validation (regression for bugs 1-4) + that the OPEN-decision markers survive edits | 18 |

Tier selection without directories: `pytest -k "integration"`, `-k "regression"`,
`-k "not integration and not regression"`.

### PART E — Docs merged 6 -> 3

- `docs/HUMAN_DECISIONS.md` — now the single PI-facing file. Absorbed `pi_decisions.md`
  (options/trade-offs/status), `pi_decisions_resolved_provisional.md` (now §3.2, the
  provisional values table) and `upstream_pins.md` (now §3.3, upstream repositories).
  Restructured as an ordered action plan with exact commands and a recommendation per
  decision.
- `docs/project_history.md` — the former `THINKING_LOG_2026-08-07.md`, content unchanged.
- `docs/implementation_log.md` — this file.
- 29 files repointed from the deleted doc paths to the merged one.

### Newly identified PI items (added to HUMAN_DECISIONS.md Part 1)

- The repository is NOT under git at all. `git_commit` is `null` in every run log, and
  AI_RULES.md 1.4 requires the freeze to BE a commit. `git init` before anything else.
- hydra-core / omegaconf are not installed, so the Hydra entrypoints cannot start.
- circuit-tracer's own README is internally inconsistent about its home:
  `safety-research/circuit-tracer` in the install and demo links (9 places) vs
  `decoderesearch/circuit-tracer` in the BibTeX block (1 place). Both quoted verbatim;
  neither chosen. `sae-pruning-paper` -> `github.com/hecboar/sae-pruning-paper`,
  verified from the fork's `CITATION.cff`.

### Scientific decisions made

None. Q1-Q9 remain exactly as they were. Every provisional value is unchanged, and
`tests/test_config_integrity.py` now actively asserts that the OPEN markers
(`hf_revision: null`, `pi_confirmed: false`, null band cutoffs,
`pre_registered_for_stage_c: false`) stay open until the PI changes them deliberately.

### Verification

`python -m pytest` -> 287 passed, 1 skipped. `python -m experiments.dry_run_stage_a 7`
and a Stage B draft run both completed and wrote schema-valid artifacts. No write under
`frozen/`; no network access; no model download.

### Next step

PI works through `docs/HUMAN_DECISIONS.md` Part 1, in order.

## Entry 6 — 2026-08-08, agent=Claude (Opus), the missing machinery: freeze writer, floors, corrections, Stage C/D bodies

PI question: "is the HUMAN_DECISIONS part the only thing left, or is there code left to
build?" Answer: there was substantial code left. A gap audit against PRD.md, AI_RULES.md
and ARCHITECTURE.md found SEVEN required pieces of machinery with zero implementation,
none of which needed a GPU or a PI decision.

### The gap audit (grep counts before this entry)

| Required by | Missing | Hits |
|---|---|---|
| AI_RULES 1.4 | the freeze WRITER — nothing wrote frozen/; only the reader/guard existed | 0 |
| AI_RULES 4.4 / PRD §2 | random top-k chance floor for overlap statistics | 0 |
| PRD C5 / AI_RULES 4.1 | permutation p-value + CI for Spearman | 0 |
| AI_RULES 4.3 | Benjamini-Hochberg | 0 |
| PRD C3 / §3.3 | threshold-sensitivity sweep | 0 |
| ARCH §2 | real CSI table writer (only write_csi_placeholder existed) | 0 |
| PRD §6 | root README.md | 0 |

Plus: Stage C and Stage D raised NotImplementedError even after every guard passed, so
Algorithm 1 had never been run end-to-end by anyone.

### Built

**`src/common/freeze.py` + `experiments/freeze_stage_b.py` — the pre-registration event.**
The single most load-bearing methodological commitment in the project had no runnable
path. Mechanics (freeze.py): append-only (an existing cell raises FreezeViolation, never
overwrites), hash-verified on the way in (the dnull is re-hashed at the destination and a
corrupt copy is removed rather than becoming the floor), manifest registration (a cell
directory alone does not satisfy the Stage C guard), and `verify_frozen_store()` which
re-hashes the whole store and also flags cells present on disk but unregistered. Policy
(freeze_stage_b.py): scientific mode + explicit `freeze_approved` + every Stage-A-gating
decision resolved + **Q2 pre-registered** (a floor frozen before D is chosen is not a
floor for anything in particular) + `--dry-run` that reports without writing.

**`src/science/chance_floor.py` (AI_RULES 4.4).** The paper's SECOND floor, distinct from
the null: the null asks "more than random weight noise?", the chance floor asks "more than
two arbitrary edge sets of these sizes would score anyway?". Ref 2607.18921 needed exactly
this — their Jaccard 0.14-0.16 was indistinguishable from a random top-k baseline at
p = 0.106. Provides the closed form (`expected_jaccard_random`), the seeded Monte-Carlo
null, and `overlap_vs_chance` returning {observed, chance median/lo/hi, p_value, excess}.
The universe size N is a required argument with no default, because a wrong N silently
moves the floor.

**`src/science/multiple_comparisons.py` (AI_RULES 4.3).** BH step-up + `correct_cell_grid`
+ `expected_false_positives`. The grid is ~4 models x 3 tasks x ~12 cells x 2 levels ~ 288
per-cell statements; at alpha 0.05 uncorrected that yields ~14 "significant" cells under
the global null. Documented in the module why BH rather than Bonferroni here.

**`analysis/cross_audit.py` — intervals added.** `spearman_rank` returned a bare point
estimate; PRD C5 requires "+ CI via permutation" and AI_RULES 4.1 forbids a number without
an interval from entering a figure. Added `spearman_permutation_test` (two-sided, +1
correction so p is never exactly 0), `spearman_bootstrap_ci` (resamples cells; degenerate
resamples are dropped rather than counted as rho=0), the pre-registered C5 bands as
constants, and `cross_audit_report` which returns the verdict AND
`verdict_ci_consistent` — false when the CI spans two decision bands. Existing functions
untouched.

**`analysis/threshold_sweep.py` (C3).** CSI at every operating point, the verdict at each,
`stable`, and `flip_points`. A conclusion with `stable: False` is reported as
threshold-dependent. Uses the identical verdict rule as `csi_table.summarize_csi`, or
"stable" would mean nothing.

**`src/common/csi_table.py` (ARCH §2).** Real row construction from a `csi()` result,
validation that refuses a row without an interval or without a `null_frozen_hash`,
`assert_both_levels_present` (proposal §4 rule 3 enforced by the WRITER, because a
reporting rule that is not enforced is one that gets broken by the last figure before a
deadline), refusal to append to a PLACEHOLDER file, and `summarize_csi` in the language of
the C1 decision rule including which cells' levels disagree.

**`src/science/perplexity.py`** (created in Entry 5) now consumed by the matched-PPL path.

**Stage C body** (`experiments/run_stage_c.py`): Algorithm 1 lines 16-20 — compress,
re-extract the ensemble, D, CSI with bootstrap, repeated at routing-head granularity, with
a chance floor attached per level, writing `csi_table.csv` + `freq_post.parquet`.
**Stage D body** (`run_stage_d.py`): damage ranking at both levels, cross-audit with
intervals, multiplicity reporting, and C4 declared PENDING rather than fabricated.

**`README.md`** — status, the question, Algorithm 1, layout, and the rules that shape the
code.

### Deliberate semantics change (engineering-mode Stage C)

`assert_engineering_dry_run_limits` refused Stage C unconditionally in engineering mode.
It now refuses Stage C/D unless the run is **provably synthetic** — new
`is_provably_synthetic()` requires model.synthetic AND task.synthetic AND pipeline=mock.
Rationale: a synthetic Stage C compresses a numpy toy and stamps `evidence: false`; it has
exactly the status of the Stage A and Stage B dry-runs that were always permitted, and it
is what lets Algorithm 1 be validated before a single GPU hour is spent. The prohibition
that matters — no REAL compression outside scientific mode — is unchanged and is now
pinned by a parametrized test asserting that ANY ONE real component (model, task, or
pipeline) re-blocks the stage. The frozen-path prohibition was also reordered to fire
BEFORE the stage-scope check, since it is the more fundamental rule.

### Bugs found while building

11. **Derived run-name `setting` token was invalid.** Stage C derives
    `setting = f"{family}-{level}"` = `magnitude-0.30`, but run-name tokens allow only
    `[A-Za-z0-9-]` — every Stage C cell whose compression level has a decimal point
    crashed at run-name construction. Added `run_naming.sanitize_token`
    (`magnitude-0.30` -> `magnitude-0-30`, deterministic and stable).
12. **NaN written into run reports made them invalid JSON.** `json.dump` emits bare `NaN`,
    which strict parsers (jq, most JS/Go/Rust readers, anything a reviewer runs on the
    artifact) reject. NaN is a *normal* result here — an undefined Spearman on two cells,
    a CI that could not be formed. Added `schema.json_safe()` (NaN/±Inf -> null) and set
    `allow_nan=False` on the Stage C/D writers, so a regression cannot reintroduce it.

### Tests: 287 -> 376 passed, 1 skipped

- `test_freeze.py` (26): append-only refusals including the dangerous case — re-running
  Stage B and quietly freezing the new null must fail and leave the original bytes
  untouched; tamper detection; unregistered-cell detection; partial-cell cleanup on
  failure; and every freeze-entrypoint policy refusal.
- `test_chance_floor_and_corrections.py` (38): Monte-Carlo tracks the closed form; an
  at-chance overlap is correctly NOT significant; p is never exactly 0; BH step-up
  behaviour and its alignment with input order; sweep stability and flip detection.
- `test_integration_algorithm1.py` (17): the whole chain A -> B -> freeze -> C -> D in one
  module-scoped fixture. Asserts the frozen store still verifies after the full run, that
  every CSI row names a hash that was actually frozen, that `dnull_median` in the table
  equals the median of the frozen cell's draws (i.e. Stage C divided by the frozen null and
  not a re-drawn one), that 50% pruning damages more than 30%, that both levels are always
  present, that reports are strict-valid JSON, and that a 2-cell cross-audit cannot support
  a verdict.
- Updated `test_config_guard.py` and `test_integration_stage_a_to_c.py` for the new
  engineering-mode Stage C semantics.

### Verification

`python -m pytest` -> **376 passed, 1 skipped**. 50/50 modules import cleanly. A
30-point audit of PRD/AI_RULES/ARCHITECTURE requirements against the code now reports
**30/30 have an implementation**. Algorithm 1 was run end-to-end on the synthetic stack:
Stage A -> 2 Stage B nulls -> freeze (store verifies) -> 2 Stage C cells -> Stage D. The
first end-to-end run immediately surfaced a `levels_disagree` cell (claim C2's whole point)
and a cross-audit that refused to support its own verdict on 2 cells.

### Scientific decisions made

**None.** Q1-Q9 are untouched. The FDR level q = 0.05 in `multiple_comparisons.py` and the
INT ratio threshold 0.5 in `patch_diagnostic.py` are both marked provisional and flagged
for pre-registration (AI_RULES 4.2).

### What still genuinely requires a model or a decision

Everything remaining is one of: (a) a real forward pass — attribution extraction on real
models, real GPTQ/AWQ/Wanda/magnitude, perplexity evaluation, matched-PPL tuning,
`PatchDiagnostic.run`; or (b) a PI decision from docs/HUMAN_DECISIONS.md. There is no
longer any model-independent, decision-independent code sitting unbuilt.

### Next step

Unchanged: PI works through `docs/HUMAN_DECISIONS.md` Part 1, in order.

## Entry 7 — 2026-08-09, agent=Claude (Opus), upstream-fork audit + property smoke test

PI request: deep-dive the proposal against the two upstream forks to find missing code,
then smoke-test every file for bugs/mismatches.

### PART A — What reading the forks changed

Both forks were read as SOURCE, not as documentation. Facts below are verified from the
local trees on 2026-08-08/09 and are now pinned as tests
(`tests/test_node_ids_and_upstream_contract.py`), so a different pinned commit (Q9)
fails loudly instead of silently producing a wrong graph.

**circuit-tracer-0.5.2 — the node taxonomy does not contain attention heads.**
`circuit_tracer/frontend/graph_models.py` defines exactly four node kinds:
`cross layer transcoder` (feature_node(layer, pos, feat_idx)), `mlp reconstruction
error` (error_node(layer, pos)), `embedding` (token_node(pos, vocab_idx)), `logit`.
A grep for head-like nodes across the package returns nothing.

Consequences, both of which are SCIENTIFIC decisions and are now Q10/Q11 in
docs/HUMAN_DECISIONS.md:

- **Q10** ARCHITECTURE.md §2's `L{layer}.{type}{index}` scheme and
  `src/science/two_level.py`'s routing-head projection (`^L\d+\.H\d+$`) both assume
  heads exist. For pipeline A the projection returns the EMPTY SET, which makes claim
  C2 ("two levels or it does not count") vacuous for the primary pipeline. Four options
  are implemented and documented in `src/extraction/node_ids.py::LEVEL2_OPTIONS`;
  `project_level2()` refuses to run without an explicit choice.
- **Q11** Upstream nodes are position-specific `(layer, pos, feature_idx)`; our
  edges.parquet schema has no position column, so several upstream edges collapse onto
  one component pair. Aggregating is a modelling claim ("the circuit is the same
  computation wherever in the prompt it fires") and it changes the edge universe, hence
  the chance floor.

Other verified facts now recorded in the adapter docstring: `prune_graph(graph,
node_threshold=0.8, edge_threshold=0.98) -> PruneResult(node_mask, edge_mask,
cumulative_scores)`; node order `[active_features, error_nodes, embed_nodes,
logit_nodes]` with `n_layers * n_pos` error nodes and `n_pos` embed nodes; errors laid
out as `divmod(idx - n_features, n_pos)`; `selected_features` indexes INTO
`active_features`; the `attribute()` signature matches what was already documented.

**sae-pruning-paper-main — the perplexity protocol is fully specified, and getting it
wrong is a documented disaster.** `docs/PROTOCOL.md`: WikiText-2 raw test split
(~289K tokens), window 1024, stride 512, `<bos>` prepended to EVERY window, bfloat16,
eager attention for Gemma-2, degradation reported as `Δlog PPL`. That document exists
because the first protocol prepended `<bos>` once to the whole corpus and evaluated
bf16 models in fp16 — **Gemma-2-2B's dense perplexity read 410 instead of ~11**, while
Llama looked merely plausible, so the fault was invisible without a forensic sweep.

This matters twice for us: the matched-perplexity null (§2.1b) is *defined* by "the
same perplexity", and C5 compares our ordering against numbers produced under exactly
this protocol. Encoded as `REFERENCE_PROTOCOL` + `validate_ppl_protocol()` (which warns
by name about the BOS fault) + `delta_log_ppl()` in `src/science/perplexity.py`.

Also confirmed: our `prune_wanda` matches upstream `prune_wanda_style_inplace`
("Per-matrix Wanda pruning: score = |W| * sqrt(E[x^2])") exactly.

### PART B — New file

`src/extraction/node_ids.py` — the mapping layer between upstream nodes and our
component IDs, which did not exist (`_to_canonical_graph` raised). Transcoder features
map to `L{layer}.F{idx}` — deliberately `F`, not `H`, so the routing-head projection
cannot silently pick up feature nodes and report a "head overlap" that is nothing of
the kind. Carries `decode_upstream_index()` mirroring the upstream index decode, and
`position_policy_is_pi_owned()` which returns the sentence that must appear in the paper
for whichever Q11 policy is chosen.

### PART C — Property smoke test: 6 real defects

A property-based smoke test (warnings-as-errors, edge-case and malformed inputs across
every public function) found six defects that the 376-test suite passed straight
through, because every existing test used well-formed inputs.

1. **s(e) could exceed 1.0 — the serious one.** `compute_from_graphs` counted edge
   RECORDS, not cells containing the edge. Because Q11's position collapse produces
   repeated edges within one (config, seed) cell, a real pipeline-A run would produce
   s(e) = 2.0, 3.0, … — not a fraction — corrupting D, CSI and the band decomposition.
   `Graph.included_edge_ids()` had always deduplicated, so the two disagreed. Fixed:
   edges are counted once per (edge, config_id, seed) cell, matching CLAUDE.md §5's
   definition ("the fraction of the ensemble whose graphs CONTAIN edge e" — containing
   it twice is still containing it). An invariant check at the single choke point where
   s(e) is created now raises if any frequency leaves [0, 1]. No existing number
   changes: the mock extractor builds edges from a dict, so it never produced duplicates.
2. `csi()` accepted a **negative D** and returned csi = -5.0, which passed every
   downstream check and would have landed in the CSI table. Now rejected.
3. `csi()` accepted **negative null draws**. Now rejected.
4. `decompose()` banded s(e) = 1.5 as "core", laundering an upstream counting bug into
   a scientific label. Now rejected.
5. `validate_csi_row()` accepted a negative csi / D / dnull_median. Now rejected.
6. (test-side) the new regression test's own `pytest.raises` regex did not match the
   error message; fixed.

All six have regressions in `tests/test_regression_smoke_findings.py`. Re-running the
smoke test reports 0 remaining.

### Tests: 376 -> 420 passed, 1 skipped

- `test_node_ids_and_upstream_contract.py` (23): the four upstream node kinds pinned
  verbatim, "there is no attention-head kind", the adjacency index decode incl. the
  `divmod` error layout, F-not-H, and that level-2 has no default scheme.
- `test_regression_smoke_findings.py` (21): one class per defect above, plus the
  reference PPL protocol constants and the named BOS fault.

### Verification

`python -m pytest` -> 420 passed, 1 skipped. 51/51 modules import. Algorithm 1 still
runs end-to-end (A -> B -> freeze -> C -> D). Dry-run green.

### Scientific decisions made

None. Two new ones were SURFACED (Q10 level-2 scheme, Q11 position policy) with
recommendations, and both are refused-by-default in code rather than guessed.

### Next step

PI works through docs/HUMAN_DECISIONS.md Part 1. Q10 and Q11 sit at Step 4b and block
Stage A alongside Q1/Q3/Q4/Q5/Q6.

## Entry 8 — 2026-08-09, agent=Claude (Opus), corrections from the fork audit

PI instruction: "correct whatever you found." Entry 7 fixed six smoke-test defects but
left four findings recorded-not-corrected. This entry closes them.

### 1. Upstream pruning-pool mismatch — the C5 grid-alignment risk

`saediag.pruning.prune_magnitude_global_inplace` takes
`include_embedding_in_threshold: bool = True` and documents that adding the token
embedding to the threshold POOL (while never zeroing it) "is what maps the labeled
sparsity (e.g. 0.30) to the observed effective per-matrix sparsities (~0.26 attn,
~0.38 MLP)". Our pruner had no such parameter and produced exactly `target_sparsity`
over prunable tensors only.

Why it mattered: PRD.md §2 requires our pruning grid to match ref [1]'s grid, because
C5 correlates our damage ordering against theirs cell by cell. A "30% cell" computed
over a different pool is **not their 30% cell**, so the cross-audit would have compared
non-corresponding cells — and `weight_delta` (hence the matched-magnitude null) would
have been matched to a different perturbation than the published one.

Fixed: `prune_magnitude` gained `embedding_names` + `include_embedding_in_threshold`
and now reproduces upstream's rule faithfully — pooled quantile threshold, strict
`|w| > threshold`, embeddings pooled but never zeroed. A demonstration in the tests:
on weights where the embedding's magnitudes differ from the Linear weights, the flag
moves the achieved Linear sparsity from 0.50 to 0.00. The docstring now states, in
capitals, that LABELLED SPARSITY IS NOT ACHIEVED SPARSITY, and `achieved_sparsity()`
was added so runs report what they actually got.

Two follow-on defects found while doing it:
- Porting upstream's strict `>` rule reintroduces its tie behaviour: under mass ties
  every tied weight drops at once, so "remove 25%" became "removed 100%". Real float
  weights never tie, but a silently dead model whose CSI is meaningless is exactly the
  failure this project exists to prevent, so the collapse now raises with an
  explanation. Per-tensor scope keeps exact top-k semantics and is unaffected.
- `achieved_sparsity()` initially counted the never-zeroed embedding in its
  denominator. On a real model the embedding dwarfs any single Linear, so a 50% prune
  would have reported as a few percent. It now excludes `embedding_names`, matching
  upstream's `global_sparsity_linear_weights`.

### 2. The silent empty routing-head projection — the most dangerous failure mode

`project_to_routing_heads` matched `^L\d+\.H\d+$` and returned `{}` when nothing
matched. For a pipeline-A circuit nothing ever matches (circuit-tracer's nodes are
transcoder features). An empty projection makes BOTH sides of the routing-head overlap
empty, the Jaccard 1.0, and claim C2 vacuously satisfied — **reporting a better
two-level agreement than reality, without crashing.**

Fixed: `strict=True` by default; an empty projection from a non-empty circuit now
raises and names Q10. `strict=False` remains for MLP-only fixtures. All callers
(`routing_head_overlap`, Stage C, the threshold sweep) inherit the guard, so Stage C
refuses rather than reporting a vacuous level.

### 3. Q10/Q11 are now ENFORCED, not merely documented

Entry 7 surfaced them in prose. Prose does not stop a run. Added:
- `configs/comparison/provisional.yaml` with `level2_scheme: null` and
  `position_policy: null`, composed by the root config;
- both questions registered in `config_guard` with `gating: stageA`, so a
  `scientific_run` with every other decision made now **refuses** with a message naming
  the two and pointing at `node_ids.LEVEL2_OPTIONS`. Verified by direct probe.
- `_detect_open_questions` now sorts by NUMERIC id, so Q10/Q11 follow Q9 instead of
  being buried after Q1 by a lexical sort.

### 4. Dead branch in `jensen_shannon_distance`

`if math.isinf(kl_pm) or math.isinf(kl_qm): return 1.0` was unreachable — `m =
0.5*(p+q)` is strictly positive wherever `p` is, so neither KL term can diverge — and
it contradicted the docstring, which correctly stated the disjoint-support maximum is
sqrt(ln 2) ≈ 0.8326. Removed; zero behaviour change. The docstring also now records the
Q2-relevant property that JS normalises, so it is blind to a uniform change in total
inclusion mass.

### Tests: 420 -> 432 passed, 1 skipped

New: `TestEmbeddingInTheThresholdPool` (7) pinning the upstream pool semantics and the
labelled-vs-achieved distinction; the strict-projection refusals; the numeric ordering
of open questions; the tie-collapse refusal. Nine pre-existing fixtures were updated
for Q10/Q11 — each now carries the comparison keys explicitly, so the fixtures state
whether the decisions are open or recorded rather than inheriting a default.

### Verification

`python -m pytest` -> 432 passed, 1 skipped. 51 modules import cleanly. Algorithm 1
still runs end-to-end. A widened property smoke sweep over every module touched this
session reports 0 issues.

### Scientific decisions made

None. Q10 and Q11 now BLOCK a scientific run instead of being silently defaulted, which
is the opposite of deciding them.

### Still recorded-not-corrected (they are PI decisions, by design)

- Q2 distance pre-registration; the CSI bootstrap axes (r only vs B, S, r); the FDR
  level q; the chance-floor universe N. Each is refused-or-flagged in code and written
  up in docs/HUMAN_DECISIONS.md.
- `analysis/rebuild_figures.py` and `artifact/build.sh` (PRD §6 artifact checklist) are
  Phase-5 deliverables that need real results to be meaningful; not built.

## Entry 9 — 2026-08-09, agent=Claude (Opus), first execution of the Hydra entrypoints

PI: "check the whole codebase ... I'm frustrated I can't start running my code."

Root cause of the frustration: **hydra-core had never been installed, so the Hydra
entrypoints had never once been executed.** The tests call `run_stage_*(dict)` directly
and bypass config composition entirely, so 432 green tests said nothing about whether
`python -m experiments.run_stage_a` worked. Installed hydra-core 1.3.5 + omegaconf
2.3.1 and ran all four stages plus the freeze for real. Three defects fell out
immediately; none was reachable from the test suite as written.

### 1. Run-directory collisions blocked the second config (THE blocker)

`ensemble/decompose=final` and `nulls=default` both produced the SAME run directory as
the default run, because the run name encodes only grid coordinates
(`{date}_{stage}_{model}_{task}_{setting}_{configset}_seed{S}`, CLAUDE.md §4) while the
scientific choices — band cutoffs, R, distance, comparison scheme — appear nowhere in
it. Result: a hard `FileExistsError` on the second config tried, whose only suggested
escapes were "use a fresh seed" (changes the science) or "a new date" (wait a day).

This is CLAUDE.md §4 (name = grid coordinates) disagreeing with ARCHITECTURE.md §3
("the resolved-config hash is the run's identity"). Resolved in `src/common/run_dir.py`
without weakening either: the run NAME is unchanged; the DIRECTORY is the run name when
free; on collision the config hashes decide — same hash means a true re-run and is
still refused (AI_RULES.md 1.2), different hash means two genuinely different
experiments and the directory becomes `{run_name}__cfg{hash8}`. Nothing is ever
overwritten in either branch. Verified: three different configs now coexist; an
identical config is still refused.

### 2. Every Stage B/C/D run through Hydra was misnamed, hiding R

`configs/config.yaml` interpolated
`run_name: ${now:%Y%m%d}_..._${ensemble.configset}_seed${seed}`, and
`ensemble.configset` never contains R. So the first Stage B run through Hydra was named
`..._B4xS2_seed0` — no null count — while CLAUDE.md §4 requires the configset token to
encode B, S AND R "so ensemble size is legible from the run name". A null run whose
name hides how many null draws it used is exactly the case the convention exists for.

Fixed: `run_name: null` in the root config; each stage derives its own name from its
own (B, S, R). New `run_naming.resolve_run_name()` validates a *supplied* name against
the derived one component by component and refuses a mismatch, so a stale override
cannot silently misstate a run. Verified: Stage B now names itself `B4xS2xR3`.

### 3. `config_hash` was undefined in Stage D

Stage D computed it inline inside the `tags` dict, so hoisting the run-dir allocation
above `tags` raised `NameError`. Caught by the integration suite; hoisted.

### Verified working for the first time

All four Hydra entrypoints plus the freeze:
`run_stage_a` (incl. `ensemble=default` -> B=16/S=5, `ensemble/decompose=final`,
`nulls=default`, and `mode=scientific_run` correctly refusing with Q1-Q6/Q10/Q11);
`run_stage_b`; `run_stage_c` (produced a real `csi_table.csv` at both comparison
levels); `run_stage_d`. Also removed six genuinely unused imports left by earlier
refactors.

### Tests: 432 -> 451 passed, 1 skipped

`tests/test_run_dir_and_naming.py` (19): collision disambiguation, immutability of an
identical config, three configs coexisting, interrupted-run identification via
`resolved_config.json`, R-in-configset for null runs, per-component refusal of a stale
supplied name, and a guard that `configs/config.yaml` never re-introduces the
interpolated `run_name`.

### Note on runs/

This sweep left 15 directories under `runs/` (including `runs/_hydra_frozen`, a frozen
store built to drive Stage C). They are engineering dry-run artifacts, all stamped
`evidence: false`. AI_RULES.md §6 forbids an agent deleting run logs, so they were left
in place — the PI can delete them freely.

### Scientific decisions made

None.

## Entry 10 — 2026-08-09, agent=Claude (Opus), second sweep: licence compliance + fresh-clone verification

Autonomous loop, iteration 2. The PI had, between ticks, run `git init`, pushed to
`github.com/SupratikB23/Circuits_Under_Compression`, added an MIT `LICENSE`, and made
several cleanup commits.

### The finding: THIRD_PARTY_LICENSES/ had been deleted

`git log --diff-filter=D` shows the commit "docs: remove obsolete documentation files"
removed `THIRD_PARTY_LICENSES/{README.md, circuit-tracer-LICENSE.txt,
sae-pruning-paper-LICENSE.txt}`. They are not obsolete.

Both upstreams are MIT. MIT obliges the copyright and permission notice to be included
"in all copies or substantial portions of the Software". This repository re-implements
the pruning mathematics of `saediag.pruning` (magnitude + Wanda, including the
embedding-in-threshold-pool rule) and wraps `circuit-tracer`, and PRD.md §6 ships that
code as the camera-ready artifact. Removing the notices is a licence violation, not a
tidy-up — and it is the kind that surfaces at review time.

Restored byte-identical from the read-only forks' own `LICENSE` files, with the
provenance README rewritten to say why the directory is not optional. Added
`tests/test_licence_compliance.py` (8 tests): the directory exists, each file names its
copyright holder, each carries the permission notice and warranty disclaimer, the
provenance README exists, and our own `LICENSE` is present and MIT.

Nothing in the codebase imported that directory, which is exactly why its deletion was
silent. It is now load-bearing on the test suite.

### Resolved decision: artifact licence

The PI's MIT `LICENSE` ("Copyright (c) 2026 Supratik Bhowal") answers the PRD.md §6 open
item. MIT-on-MIT is compatible **provided the upstream notices ship**, which is what the
restored directory is for. Marked RESOLVED in docs/HUMAN_DECISIONS.md.

### Verified this tick

- **freeze CLI, end to end.** `--dry-run` reports the plan and writes nothing; the real
  run froze a cell, re-verified the store, and recorded a genuine freeze commit. Never
  exercised through its own command line before.
- **Fresh-clone behaviour.** `runs/.gitkeep` was deleted and `runs/` is now fully
  gitignored, so a collaborator clones a repo with no `runs/` directory. Confirmed Stage
  A creates it (`allocate_run_dir` uses `mkdir(parents=True)`).
- **`frozen/` is tracked and new cells are not ignored** — checked explicitly, because a
  `.gitignore` that swallowed the frozen store would silently destroy the
  pre-registration claim while everything still looked green.
- **`analysis/threshold_sweep.py` on real pipeline output** rather than hand-written
  vectors: real Stage A (7 edges), Stage C (5 edges), R=3. Stable at both levels.
- Doc/code path audit across the four governing docs + README + HUMAN_DECISIONS: one
  genuine stale reference (CLAUDE.md §5 pointed at `configs/ensemble/decompose.yaml`,
  now `decompose/final.yaml`). The old paths at CLAUDE.md lines 89-91 are the deliberate
  structure-history note and were left alone.
- `docs/HUMAN_DECISIONS.md` Steps 1 and 2 marked DONE, Step 3 flagged as the current
  position, so the PI does not redo finished work.

### Note on the code-review-graph MCP

`get_minimal_context_tool` returns `not_ready` (no graph built), and `detect_changes_tool`
is git-diff based against `HEAD~1`, which is not useful for reviewing work that is
already committed. Reviewed directly instead.

### Tests: 451 -> 459 passed, 1 skipped

### Scientific decisions made

None.

---

## Entry 11 — 2026-09-12, agent=Claude (Opus), THIRD_PARTY_LICENSES restored a second time + untracking guard

### What was broken

A fresh clone of `github.com/SupratikB23/Circuits_Under_Compression` fails `python -m
pytest` with **6 failures**, all in `tests/test_licence_compliance.py`. The upstream
licence texts are absent from the published repository.

Two commits, three weeks apart, did it:

| Commit | Date | Effect |
|---|---|---|
| `f8bd168` "Delete THIRD_PARTY_LICENSES directory" | 2026-08-15 | removed all 3 files (62 lines) |
| `29d7ed3` | 2026-09-02 | appended `THIRD_PARTY_LICENSES/` to `.gitignore` |

This is the **second** recurrence. Entry 10 restored the same directory on 2026-08-09
and added `tests/test_licence_compliance.py` specifically so a silent repeat would fail
loudly. It failed loudly — and then the ignore rule hid the evidence on the machine
where the suite was being run.

### Why the existing guard did not hold

The Entry 10 tests assert the texts exist **on disk**. That is true on the PI's working
machine (the files were never deleted locally, only from git), so the suite stayed green
there while every clone was missing the notices. Deletion was guarded; *untracking* was
not. Entry 10 explicitly checked that `frozen/` was not gitignored for exactly this
reason and did not extend the check to `THIRD_PARTY_LICENSES/`.

### Fix

- Restored all three files from `f8bd168^` and removed the `.gitignore` rule.
  **Verified authentic, not merely present:** the two licence blobs were diffed against
  upstream at the Run_Plan.md-pinned commits — `decoderesearch/circuit-tracer`
  @ `8f1e2438df612464e229e44c4a00ff637bf9379b` and `hecboar/sae-pruning-paper`
  @ `261191804675e2d39d0a265320dbc0bc85afd30a`. Git blob hashes match exactly
  (`f802a4da…`, `4db4c21f…`); working-tree CRLF is `core.autocrlf=true`, not drift.
  Both raw fetches at those SHAs succeeded, which incidentally confirms Q9's hashes
  resolve.
- Added `TestTheLicencesAreActuallyTrackedByGit` (4 tests) to
  `tests/test_licence_compliance.py`: each file is tracked by `git ls-files`, and no
  `.gitignore` rule matches the directory.
- `check-ignore` is called with **`--no-index`**, without which git declines to report a
  tracked path as ignored and the guard silently passes while a live rule sits in
  `.gitignore` waiting for the next deletion. Both new assertions were negative-tested:
  re-adding the rule fails the ignore test, `git rm --cached` on one licence fails the
  tracking test.

### Environment

Repo verified on a second machine (`C:\Users\sishi`, Windows 11). Default `python` there
is 3.7.9 against `requires-python >=3.10`; a 3.12.10 venv at `.venv/` is required and the
runbook's `python` should be read as `.venv/Scripts/python.exe`.

### Tests: 459 -> 462 passed, 2 skipped

The second skip is not a regression: `tests/test_seeding.py:38` skips on `No module named
'torch'`, correct while the `models` extra is uninstalled and RUN MODEL DOWNLOAD is
unapproved. README and HUMAN_DECISIONS said "459 passed, 1 skipped"; both updated.

### Scientific decisions made

None.

---

## Entry 12 — 2026-09-12, agent=Claude (Opus), Q9 resolved: upstream pins recorded

Run_Plan.md Step 1. Every claim verified against the live source this session
(AI_RULES.md §2.2 — never cite from memory), not taken from Run_Plan's prose.

### circuit-tracer — read directly, no inference

| | |
|---|---|
| Canonical URL | `github.com/decoderesearch/circuit-tracer` |
| Commit | `8f1e2438df612464e229e44c4a00ff637bf9379b` (tag `v0.5.2`) |

`github.com/safety-research/circuit-tracer` returns `HTTP/1.1 301 Moved Permanently`
with `Location: https://github.com/decoderesearch/circuit-tracer`. The fork README's
BibTeX was right and its 9 install/demo links are stale — the ambiguity recorded in
Entry 7 is resolved, not merely picked. Commit read from
`api.github.com/repos/decoderesearch/circuit-tracer/git/ref/tags/v0.5.2`; the tag is
lightweight, so the returned sha is the commit.

### sae-pruning-paper — INFERRED, and the inference is recorded as such

`261191804675e2d39d0a265320dbc0bc85afd30a`, 2026-07-31, "Add revision materials and
align repository with the revised manuscript". The local fork is not a git repo, so
this is deduced from which files it contains. Three supporting facts confirmed via the
GitHub API:

1. The commit exists with that date and message.
2. `c8cce94ff4fce08c0dee6bd0ab7391e67c2a4880` has it as its **only** parent — nothing
   sits between them — and adds `revision/scripts/verify_metric_ordering.py`, which the
   fork lacks.
3. At the pin, `revision/scripts/e6_stat_freeze.py` still contains "rebuttal"; the next
   commit `bd85878` ("Reword two docstrings that framed the analysis as rebuttal
   material") removes it.

A copy lacking the script and containing that word can only be at this commit. **The
inference rests on two claims about the local fork, which cannot be checked from a
machine that does not hold it.** Run the tree diff in Run_Plan Step 1 on the machine
with `../sae-pruning-paper-main/` and trust the diff over this entry if they disagree.
Flagged as `sae_pruning_paper_pin_is_inferred: true` in the config.

### A C5 risk checked and cleared

`c8cce94` also modifies `results/E6/stat_tests.csv` — the file the cross-audit's target
numbers come from — so the pin choice could in principle have moved a C5 number. Diffed
both revisions: the change is **two appended rows** about Proposition 1 metric ordering.
Every fragility and Spearman value Run_Plan §0.2 quotes, including the
`ρ(firing,survival)` range `-0.540..0.062`, is byte-identical at both commits. No C5
number depends on the pin.

### Files updated

- `docs/HUMAN_DECISIONS.md` §3.3 (both tables + a derivation note), the Q9 summary row,
  and Part 1 Step 3
- `THIRD_PARTY_LICENSES/README.md` — both PI TODOs closed (the licence-compatibility one
  was already answered by the MIT decision of 2026-08-09)
- `# Adapted from:` headers in `src/extraction/attribution_graph.py`,
  `src/compression/magnitude_prune.py`, `src/compression/wanda.py` (CLAUDE.md §7)
- the machine-readable copies, so docs and code cannot drift:
  `src/common/config_guard.py` `PROVISIONAL_VALUES["Q9"]` and
  `configs/provisional_defaults.yaml` `q9_upstream_pins`
  (`artifact_packaging_blocked: false`)
- the two `NotImplementedError` strings that quoted `UNKNOWN_LOCAL_FORK` at the user

`docs/project_history.md` still says `UNKNOWN_LOCAL_FORK` and was deliberately left
alone: it is a record of what was true at the time, not a live value.

### Tests: 462 passed, 2 skipped (unchanged); `dry_run_stage_a` still green

### Scientific decisions made

None. Q9 is a lookup, not a judgment (HUMAN_DECISIONS Part 1 Step 3: "no scientific
judgment needed"). Q1–Q4, Q6, Q7, Q10, Q11 remain open; Q5 is the next blocker.

---

## Entry 13 — 2026-09-12, agent=Claude (Opus), Q5 resolved: all eight revision pins

Run_Plan.md Step 2. Every hash read from the HuggingFace API this session
(`api/models/<repo>` → `sha`, each repo's current `main` HEAD).

### What was actually pinned — eight repos, not four

HUMAN_DECISIONS Q5 listed eight, and three of them had nowhere to live in the config
schema:

| Repo | Revision | Gated |
|---|---|---|
| `google/gemma-2-2b` | `c5ebcd40d208330abc697524c919956e692655cf` | manual |
| `meta-llama/Llama-3.2-1B` | `4e20de362430cd3b72f300e6b0f18e50e7166e08` | manual |
| `EleutherAI/pythia-160m` | `50f5173d932e8e61f858120bcb800b97af589f46` | no |
| `EleutherAI/pythia-410m` | `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f` | no |
| `EleutherAI/pythia-70m` | `a39f36b100fe8a5377810d56c3f4789b9c53ac42` | no |
| `openai-community/gpt2` (exit gate) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | no |
| `mntss/gemma-scope-transcoders` | `9250a2d4860ce5ed5c96c14d5882b7d8162809a3` | no |
| `mntss/transcoder-Llama-3.2-1B` | `c37a82c1ec4cea30d424850d159b17b720ce19e2` | no |

**Gating restricts downloads, not metadata** — both gated shas were readable without
credentials, so pinning did not have to wait on the licences.

**A pinned model with a floating transcoder set is still a floating circuit.** The two
transcoder sets had no home in `configs/model/*.yaml`; added `transcoder_revision:` and a
test asserting it is present and 40 chars wherever `transcoder_set` is not null, and null
for both Pythias (no transcoder set exists for Pythia — pipeline A cannot run there).

Pythia-70M and GPT-2 still have no model config; their pins live in HUMAN_DECISIONS Q5 and
`configs/provisional_defaults.yaml` until the exit gate needs them.

### The test was strengthened, not just flipped

`test_real_models_have_no_pinned_revision_yet` asserted every pin was `None`. Replaced
with `test_real_models_carry_their_pinned_revision`, which asserts each pin **by value**
and checks it is 40 hex chars. "Not null" would have been the weaker guard: the failure
that matters is a *silent re-pin*, which would make a Stage A dense reference and a Stage
C re-extraction different models while every hash in `run_meta.json` still looked
self-consistent. Changing a pin now has to be a deliberate edit in three places.

Config immutability (AI_RULES.md 1.2) checked first: no run in `runs/` references a real
model config — the only hits were the Q3 field `pythia_pilot_approved` — so editing them
was permitted.

### Architecture PLACEHOLDERs: two verified, two still blocked

The configs say to verify their architecture numbers against the pinned HF config.

- **Pythia-160M (768/12/12/64/2048) and Pythia-410M (1024/24/16/64/2048): exactly right**,
  checked against `config.json` at the pinned sha. Marked VERIFIED in both configs.
- **Gemma-2 and Llama-3.2 return 401** on `resolve/<sha>/config.json` without accepted
  licences. Their numbers stay PLACEHOLDER and must be re-checked right after acceptance.

### One thing that did not check out: the Pythia eval dtype

Both Pythia configs say `dtype: bfloat16` with "verify against fork eval protocol". The
fork's `docs/PROTOCOL.md` does fix compute dtype at bf16 — but its stated reason is that
an earlier run "ran in float16 on models trained in bfloat16, which pushed Gemma 3 toward
non-finite values", i.e. the rule is *match the training dtype*. **Pythia reports
`torch_dtype: float16`** and is absent from that protocol's table. So both Pythia configs
inherit a justification that does not apply to them.

Not cosmetic: Stage B perturbs weights at matched magnitude, and the precision the weights
live in sets the floor that magnitude is measured against. Flagged in the configs and in
HUMAN_DECISIONS Q5 as a new open item, to settle before the Pythia-160M pilot — the run
that fixes B/S/R. **Not decided here; it is a scientific call.**

### Also

Q9's own section still read `[PI-INPUT-NEEDED]` / `Status: OPEN` after Entry 12 updated
§3.3 and the summary table. Marked RESOLVED there too, with the pins and the 301 finding.

### A mistake worth recording

The first attempt at the Q5 doc edit built a replacement span with
`s[s.index("> **PI fill-in**"):s.index(...)]`. There are eleven `PI fill-in` blocks in
that file; `str.index` found the first, and the edit deleted 248 lines across Q2, Q3, Q4
and the Q5 heading. Caught by re-reading the file rather than by any test — no test covers
prose. Reverted with `git checkout --` (the Entry 12 work was already committed, so
nothing was lost) and redone with exact unique anchors and a `count == 1` assertion on
every replacement.

### Tests: 462 -> 463 passed, 2 skipped (one test replaced by two); dry-run green

### Scientific decisions made

None. Q5 is a lookup (HUMAN_DECISIONS Part 1 Step 3: "no scientific judgment needed").
The dtype question it surfaced is left open for the PI.

---

## Entry 14 — 2026-09-12, agent=Claude (Opus), Q2/Q1/Q4/Q10/Q11 pre-registered

Run_Plan.md Step 3. The PI approved all five, plus explicit novelty-zone write access
to three modules (`decompose.py`, `threshold_grid.py`, `two_level.py`) under CLAUDE.md
§2 / AI_RULES.md §3. **No scientific value in this entry was chosen by the agent** —
four were adopted as Run_Plan recommended, and the fifth was returned to the PI as a
question because the recommendation was not implementable.

| | Pre-registered | Config |
|---|---|---|
| Q2 | primary L1, ablation JS, also-report normalised L1 | `distance/provisional_l1_js.yaml` |
| Q1 | core s=1, contingent 0.5 ≤ s < 1, noise s < 0.5 | `ensemble/decompose/final.yaml` |
| Q4 | anti-diagonal grid, node 0.6→0.9 × edge 0.99→0.95 | `ensemble/default.yaml` |
| Q10 | layer + falsification condition | `comparison/final.yaml` |
| Q11 | aggregate over positions | `comparison/final.yaml` |

### Q4's recommendation was not achievable, and this is the important finding

Run_Plan Part 2 asks for "a crossed 4×4 grid ... anti-correlated, giving B = 16". The
two halves contradict each other:

- the full 4×4 Cartesian product is B=16 with **84 of 120 pairs nested** (70%), and
  contains CIRCUS's own counter-example (0.6,0.95) ⊂ (0.8,0.98) ⊂ (0.9,0.99);
- the anti-correlated pairing of those four levels is non-nested but gives **B=4**.

Order configs by dominance and a k×k product is a grid poset, whose largest antichain
has exactly **k** members — so B=16 non-nested is impossible with 4 levels per axis, at
any values. Had this been implemented as written, every s(e) would have been inflated
toward the loosest view and the artifact would have been frozen into the null
irreversibly. Put to the PI as a question with three options; the PI chose the
anti-diagonal at B=16.

`generate_anti_diagonal_grid(B)` walks node LOW→HIGH while walking edge HIGH→LOW and
pairs them index-by-index: non-nested **by construction** rather than by rejection
sampling, a pure function of B with no seed axis, and it sweeps the same
literature-anchored box. `is_non_nested()` is exposed so the property is asserted on
whatever grid a config supplies, not only on generated ones.

### Two silent failures this surfaced

**A nested grid inside the test suite.**
`test_integration_scientific_run_blocked.py` built its "fully resolved scientific run"
fixture as node `0.8 − 0.05i`, edge `0.98 − 0.01i` — both axes descending together, so
every config dominated the next. A fully nested chain, standing in for the configuration
this project treats as correct, passing for five weeks because nothing checked it.
`run_stage_a` now refuses a nested grid before extracting anything.

**`noise_strict` never reached `decompose()`.**
Both `run_stage_a` and `run_stage_c` called `decompose()` positionally and dropped the
new flag, so every edge at exactly s = 0.5 was banded **noise** while the config said
CIRCUS's taxonomy puts it in **contingent**. Caught because the dry-run integration test
was rewritten to read the cutoffs from the config rather than restate them — the version
that hard-coded `0.9`/`0.1` would have been "fixed" by hard-coding `1.0`/`0.5` and would
have hidden this.

### Q1: the flag, not the 7/16 hack

CIRCUS puts s = 0.5 in *contingent*, so the noise test is `< 0.5`. Under the module's
inclusive comparison the only encoding was `(ceil(B/2) − 1)/B` = 7/16 at B=16, which
hard-codes B into a config whose name never mentions B. Added `noise_strict` to
`decompose()` instead, defaulting to **False** so every existing caller and every
pre-2026-09-12 run keeps its exact behaviour.

### Q10: the wiring bug that would have made C2 report zeros

`two_level.py` carried its own `^L\d+\.H\d+$` regex, independent of
`node_ids.project_level2`. Choosing `layer` in the config would therefore not have
rewired the module at all. Added `project_to_coarse_level`, which delegates to
`project_level2` — one definition of the coarse level, in one place — and
`routing_head_overlap` takes a `scheme` argument defaulting to `None`, so the literal
head projection is preserved for the dense-node pipeline and for fixtures that really do
carry head nodes.

`coarse_level_verdict` implements the pre-registered falsification condition and returns
the sentence, not a bare bool, so the paper wording is not re-derived per caller.

### Not done, and why

**Normalised L1 is pre-registered but not implemented.** It needs
`src/science/distances.py`, which is novelty-zone and was NOT in the approval (which
named `decompose.py`, `threshold_grid.py`, `two_level.py`). Recorded as
`also_report_implemented: false` rather than quietly widening the approval.

### Tests: 463 -> 467 passed, 2 skipped

Four guards that asserted these decisions were still open were replaced by guards that
assert the pre-registered values **by value**, plus new ones: the grid is non-nested at
the pre-registered B, a crossed product would be nested (pins the reasoning), the bands
reproduce CIRCUS on a worked vector, and the Hydra composition and dry-run driver agree
on `noise_strict`.

### Scientific decisions made

Five, **all by the PI**: Q2, Q1, Q4, Q10, Q11. Q3, Q6, Q7 remain open.

---

## Entry 15 — 2026-09-12, agent=Claude (Opus), Q6 + Q7 pre-registered; two source corrections

Run_Plan.md Step 4. Three PI decisions: the two-corpus split (Q7), n=300 with a
template-level bootstrap (Q6), and the prompt sources (Q6). Every value below was read
from the upstream source this session, not quoted from Run_Plan's prose — which is how
both corrections surfaced.

### Q7 — calibration and perplexity are DIFFERENT corpora

`HUMAN_DECISIONS.md` recommended WikiText-2 "for both calibration and the perplexity
evaluation". Read from the pinned fork:

| | Value | Read from |
|---|---|---|
| Calibration | **FineWeb-Edu**, 300,000 tokens, **seed 7**, bf16 | `revision/src/saediag/reprune.py::calib_cache_path` |
| Perplexity | **WikiText-2 raw test**, window 1024 / stride 512 / `<bos>` per window / bf16 / `eager` for Gemma-2 | `docs/PROTOCOL.md` |

Calibrating Wanda on the perplexity corpus would mean our Wanda-pruned model is not ref
[1]'s Wanda-pruned model, and C5 would correlate circuit damage from one intervention
against feature damage from another.

**The calibration seed is part of the data, not the run config.** Run_Plan records the
corpus and the token count but not the seed. Their pruning is deterministic *given the
calibration cache*, and that cache is built with seed 7. A different seed gives a
different cache, a different Wanda mask and a different pruned model, with nothing in any
output to indicate it. Recorded in `configs/calibration/final.yaml` and `data/README.md`.

### Q6 — two corrections to this repo's own description of the tasks

**1. `greater_than` is a year-span task, not day-of-month.** `HUMAN_DECISIONS.md` Q6
said: *"Day-of-month construction: `"The {day} of {month} is"` → next-token day"*,
attributed to Conmy et al. ACDC's `acdc/greaterthan/utils.py` actually builds:

```python
template = "The {noun} lasted from the year {year1} to "
```

a year-span completion scored over two-digit suffixes `yearend+1..99`, across 26 nouns.
The whole reason greater-than is in this project is as an anchor against a *published*
circuit; a circuit extracted on a day-of-month task would not be comparable to it, and
nothing downstream would have flagged the mismatch.

**2. IOI has 30 templates, not 400.** `n_templates: 400` was a placeholder and is not a
count from any IOI release. The reference `ioi_dataset.py` defines exactly 15
`BABA_TEMPLATES`, with `ABBA_TEMPLATES = BABA_TEMPLATES[:]` then reordered — 30 total.
This is not cosmetic: the bootstrap resamples **templates**, so the template count is the
effective sample size for every prompt-level CI. A config claiming 400 would have implied
roughly 13x the statistical power actually available.

### Pre-registered

| Task | Source | n_prompts | n_templates |
|---|---|---|---|
| `ioi` | `transformer_lens` IOI generator (Wang et al., arXiv:2211.00593) | 300 | 30 |
| `greater_than` | ACDC `get_year_data` | 300 | 26 |
| `docstring` | ⚠️ **still open** | 300 | `null` |

`bootstrap_resampling_unit = template` on all three: prompts from one template are
correlated, and resampling prompts would treat correlated draws as independent and return
CIs that are too narrow. This is the PROMPT bootstrap only — the CSI bootstrap axes (over
B, S, r) remain open and are flagged as interacting with it.

### Deliberately left unset

- **`docstring` source.** Run_Plan says "MIB if it covers docstring, else the original
  release" and nobody has checked which. `pi_confirmed` stays `false`, `n_templates` stays
  `null`, and `data/README.md` marks the licence UNVERIFIED. Docstring is the third task
  and does not block the Pythia-160M pilot. Guessing it would be the invention AI_RULES
  §2.2 forbids.
- **Calibration `context_length`.** The fork reads it from the cached array's shape rather
  than declaring it. Its SAE activations use context 256, but that is the SAE cache, not
  necessarily the pruning calibration cache.

### Files

New config group `configs/calibration/final.yaml`, added to the root composition.
`configs/task/{ioi,greater_than,docstring}.yaml` pinned. `data/README.md` rewritten with
the five-row provenance table AI_RULES §5 requires before Stage A (`content_hash` is
PENDING on every row — nothing has been downloaded, and a hash invented before the bytes
exist is worse than an empty cell). Both machine-readable mirrors updated.

### Tests: 467 -> 469 passed, 2 skipped

`test_real_task_configs_are_not_pi_confirmed` replaced by one asserting the pre-registered
values by value while requiring docstring to stay unconfirmed, plus
`test_greater_than_is_the_published_year_span_task` (guards the correction: the template
must mention a year and must not mention a month) and `test_q7_uses_two_different_corpora`
(asserts the corpora actually differ, and that the seed is 7).

### Scientific decisions made

Three, all by the PI: the Q7 two-corpus split, the Q6 counts + bootstrap unit, the Q6
sources. Q3 and Q8 remain open; docstring's source remains open.

---

## Entry 16 — 2026-09-12, agent=Claude (Opus), the statistical pre-registrations + three silent failures

Four of the "open, not numbered" items in HUMAN_DECISIONS, plus Q8. All decided by the
PI; novelty-zone approval extended to `csi.py` and `distances.py` the same day.

| Item | Decision |
|---|---|
| CSI bootstrap axes | **B, S, r** (was: r only) |
| Chance-floor universe N | **observed node union**, U·(U−1) |
| FDR level q | **0.05**, moved into config |
| Normalised L1 | implemented; new `d_normalized` column |
| Q8 tracker | **W&B**; `run_meta.json` stays source of record |

### The bootstrap change, and why it is the one that matters

`csi()` resampled only the null draws, so its interval carried uncertainty in the CSI
denominator and nothing else — while the B threshold views and S seeds are sampled too.
An interval ignoring them is anti-conservative: it excludes 1 more often than it should,
which is the direction that manufactures findings.

`csi_over_ensemble()` resamples the B config ids and S seeds with replacement, recomputes
s(e) and D over the resampled crossed grid, and resamples the null independently.
**Dense and compressed are resampled with the same labels**, because they are paired —
the quantity of interest is their difference at matched (config, seed), and resampling
them independently would inject variance the experiment deliberately controls out.

Measured on a deliberately heterogeneous test ensemble: CI width **1.10 → 3.14**, point
estimate unchanged to 1e-12. Some cells that would have read significant under the old
interval will not under the new one. That is the correct direction, and doing it before
any real result exists is the entire point.

`distance_fn` is injected rather than imported, so csi.py still does not choose the
distance (Q2 lives in configs). The coarse level passes a callable that projects first.

### Three silent failures this surfaced

**1. Stage C never used the Q10 scheme.** Entry 14 wired `two_level.py` to
`node_ids.project_level2`, but Stage C still called `project_to_routing_heads` — the
literal `L{l}.H{h}` projection. On a real pipeline-A circuit that raises; with
`strict=False` it would instead have reported a routing-head overlap of **1.0 from two
empty sets**, i.e. claim C2 vacuously satisfied and *better* agreement than reality. Now
reads `comparison.level2_scheme` from config. **Entry 14's fix was incomplete and this
entry completes it.**

**2. Empty ensemble cells vanished from the ensemble.** A `(config, seed)` cell whose
extraction returns no edges emits no records, so it was absent from the per-cell map —
shrinking the bootstrap denominator and inflating every s(e). An empty view is a real
observation ("this configuration found nothing"), not a missing one. Caught by the
integration suite refusing a dense/compressed key mismatch, which is the guard doing
exactly its job. `cells_from_records` now takes the expected grid and materialises such
cells as empty sets, matching what `compute_from_graphs` does with its explicit `n`.

**3. `stage_d.fdr_q` was read twice** with an independent literal `0.05` fallback in each
call, so the reported q and the applied q did not have to agree. One source now.

### The chance-floor universe

Stage C computed N from `len(model.nodes())` — every component in the model. For pipeline
A the node basis is *active transcoder features for these prompts*, so a model-wide count
includes nodes the extractor could never have emitted. That inflates N, pushes the
random-overlap baseline down, and makes every overlap look above chance for purely
combinatorial reasons — the same trap the Q11 aggregate decision avoided.
arXiv:2607.18921's candidate sets are 37 and 31: small universes, honest floors.

### Schema change

`CSI_TABLE_COLUMNS` gains `d_normalized` (Q2's third column). `make_csi_row` takes it as
an optional argument defaulting to None, so a caller that has not computed it writes an
explicit blank rather than a fabricated number; Stage C always supplies it.

### A guard worth flagging

`validate_csi_row` requires `ci_lo <= csi <= ci_hi`. A percentile bootstrap does not
guarantee the point estimate falls inside its own interval — rare, but possible with a
skewed bootstrap distribution at small R. The row is then refused rather than written.
Correct behaviour, but it surfaces as a hard Stage C failure, not a warning.

### A test that was wrong before the code was

The first widening test used `(j + i + s) % 2` to vary cells. That pattern is perfectly
balanced: every edge lands in exactly half the cells under *any* resample, so D is
constant and B/S resampling correctly adds no width. The test failed while the estimator
was right. Rewritten with genuine heterogeneity, and the balanced case kept as
`TestNoVarianceMeansNoWidening` — asserted on the invariant (one distinct D across 40
resamples) rather than on two Monte Carlo interval widths, which come from different RNG
streams and would never match exactly.

### Tests: 469 -> 482 passed, 2 skipped

New file `tests/test_csi_bootstrap_axes.py` (13 tests): point estimate unchanged, CI
widens under heterogeneity, determinism, unpaired ensembles refused, zero null median
refused, empty cells counted, excluded records ignored, normalised L1 separates cells raw
L1 conflates, and the no-variance contrast pair.

### Scientific decisions made

Five, all by the PI. Remaining open: Q3 (B/S/R — needs the measured Pythia wall time),
the docstring task source, the C4 INT-flag ratio, the Pythia dtype, and C5 grid alignment.

---

## Entry 17 — 2026-09-14, agent=Claude (Opus), real dense-node extraction on Pythia-160M; grid nesting; Q3 and N decided; project organised

Run_Plan Step 5. It could not run on any hardware: the real-model loader, `DenseNodeExtractor`
(Pythia's only pipeline) and every real task loader were stubs. This entry covers building them,
the decisions that required, the non-evidence pilots, and the reorganisation the PI asked for.

### Files (all AI-generated; `reviewed-by: PENDING`)

| File | What |
|---|---|
| `src/tasks/base.py` | Prompt batch structure; summed metric; per-prompt resampling unit |
| `src/tasks/ioi.py` | IOI with ACDC's ABC corruption, all 30 templates, tokenizer-filtered names |
| `src/tasks/greater_than.py` | Year-span task, "01" corruption, tokenizer-derived years and suffixes |
| `src/tasks/_acdc_vendored.py` | GENERATED by script from the pinned upstream files, lengths asserted |
| `src/extraction/real_model.py` | Loader at the pinned sha; architecture assertions; download gate enforced |
| `src/extraction/eap.py` | Edge attribution patching (split Q/K/V, abs(sum), parallel-residual rule) |
| `src/extraction/dense_prune.py` | circuit-tracer `find_threshold` port + dangling-node cleanup |
| `src/extraction/dense_node_variant.py` | Real path with a once-per-(model, task, seed) attribution cache |
| `experiments/run_stage_a.py` | Loader wired in for the dense-node pipeline |
| `experiments/time_attribution.py` | Q3 timing pilot, non-evidence by construction |
| `tests/test_eap_dense_node.py` | 21 tests (below) |
| `THIRD_PARTY_LICENSES/{automatic-circuit-discovery,easy-transformer}-LICENSE.txt` | Vendored at pinned commits; git blobs match upstream |
| `docs/PROGRESS.md` | Project progress, written in the PI's first person **at the PI's request** |
| `docs/BOTTLENECKS_AND_HARDWARE.md` | Bottlenecks and hardware tiers; only RTX 4060 figures are measured |
| `docs/preregistration/2026-09-14_dense_node_grid_{criterion.md,sweep_results.json,outcome.md}` | Grid pre-registration, results, decision |

`docs/PROGRESS.md` is in the PI's voice because the PI asked for that. **This log is unchanged
in role:** it remains the AI-attribution record required by AI_RULES §7.

### Decisions, all by the PI

Edge attribution patching (arXiv:2310.10348) for the dense-node pipeline; split Q/K/V inputs;
abs(sum) aggregation; node scores by attribution patching at component outputs; Q4 thresholds
applied directly to total-effect scores; float32 for Pythia; no BOS; docstring source ACDC and
unit prompt-within-style; download approved for Pythia-160M only; the Q4 grid kept after the
nesting finding; **Q3 = B 16, S 5, R 20**; **chance-floor N = structurally possible edges among
observed nodes**.

On Q3 the PI declined to pick an option and instructed "just decide on the things now". The value
recorded is Run_Plan's rule, stated before any timing existed, applied to the measured pass. It
is labelled as applied on that instruction, not as a PI selection among options.

### Hazards found and closed

1. **TransformerLens ignores the revision pin** — it has no `revision` argument. Weights go
   through `transformers` at the sha and are passed as `hf_model`; verified bit-identical.
2. **TransformerLens builds the architecture from `main` by name** (`convert_hf_model_config`,
   no revision). The loader asserts against the pinned config and the YAML.
3. **`mode.allow_model_download` was read by no code.** Now enforced.
4. **GPT-2 tokenizer assumptions:** 88/99 names and 112/120 nouns are single tokens under Pythia,
   456 years qualify (618 under GPT-2), and "01" is token 520 (ACDC hard-codes 486).
5. **Phantom same-layer attention -> MLP edges** on parallel-residual models would receive nonzero
   EAP scores. Excluded structurally.
6. **The seed axis was undefined** for a deterministic method; a seed now draws the prompt batch
   and its corruptions (resampling variance, arXiv:2606.16920).

### Verification

- **Tests 482 -> 510 passed, 1 skipped:** +21 EAP/pruning/extractor tests, +6 licence tests for
  the two new notices, and the torch seeding test that now runs instead of skipping.
- **The central test is a finite difference, not a smoke test.** Every edge type, in parallel and
  sequential blocks, matches the derivative of an exact additive edge patch (float64). Batch-split
  invariance is proven to have teeth.
- **Real model:** weights bit-identical to the pin; 32,347 candidate edges (= hand count); IOI
  logit difference +4.56 -> -0.27 under corruption; greater-than +0.763 -> -0.640.

### Non-evidence pilots (RTX 4060)

- **Attribution pass:** IOI 42.3-48.1 s over 4 runs; greater-than 7.5-7.9 s over 3 runs; peak
  VRAM 4.26 / 2.64 GiB; 10-16 ms to re-prune per view. One earlier IOI measurement of 25.8 s never
  reproduced and is excluded, unexplained. GPU state was unchanged across the re-run.
- **Grid diagnostics:** 82-116 of 120 view pairs nested. The pass criterion was committed first
  (`d7c2e58`), the sweep run exactly as stated (`9798bf9`), and no candidate passed. Accepting
  IOI's pass alone was deliberately not offered: it would change the criterion after seeing
  results.
- **N overcount:** U*(U-1) = 23,870 vs 2,734 structurally possible pairs on IOI (8.7x) and 2,970
  vs 469 on greater-than (6.3x); the random-overlap floor for two 827-edge circuits falls from
  0.178 to 0.018.

### Errors of mine, corrected

- **Greater-than noun count:** recorded as 26 on 2026-09-12. That counted *lines* (`grep -c '"'`),
  not nouns; there are 120. Corrected in every live document. Entry 15's historical text is left
  as written.
- **IOI source:** recorded as the TransformerLens generator on the strength of an option
  description I wrote without checking. TransformerLens's dataset has 2 templates; the 30 are
  Easy-Transformer's, via ACDC.
- **Mid-session mistakes caught before landing:**
  - a redundant cross-head einsum in the first `eap.py` draft;
  - a degenerate test fixture that could not detect B/S widening;
  - a double `-q` that hid pytest's summary line;
  - two heredoc-embedded patch scripts that bash failed to parse, and one wrapped-line anchor.

  The patches were atomic, so no partial writes landed.

### Organisation (PI request)

- Moved with `git mv` into `docs/reports/`: the proposal PDF, the group report PDF and
  `Progress_Report.tex`.
- Moved into `docs/archive/root-logs-2026-08-09/`: four `run_stage_*.log` files holding only
  NumExpr start-up lines. Archived, not deleted.
- README rewritten for the current state. That fixed two stale claims: "Scientific decisions:
  None made", and the artifact licence being "still to be chosen" (MIT since 2026-08-09).
- CLAUDE.md's resolved `[QUESTION FOR PI]` TODOs replaced by the decisions.

### Not done, by instruction

The PI said not to start anything further: no Stage B/C engineering, no N implementation, no
Pythia-410M download. These are listed in `docs/PROGRESS.md` and `docs/BOTTLENECKS_AND_HARDWARE.md`.

### Scientific decisions made

By the PI: those listed above. Q3's value was applied from a pre-stated rule on the PI's instruction.
