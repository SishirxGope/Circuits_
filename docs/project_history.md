# [AI-GEN] agent=OpenCode date=2026-08-07 task=Full thinking/work export of sessions 1-3 (scaffold, provisional unblock, cleanup)
# reviewed-by: PENDING

# THINKING LOG — Circuits Under Compression (2026-08-07)

A complete narrative export of every piece of thinking, decision, and action taken so far
across three work sessions: (1) repository scaffold, (2) PI PROVISIONAL ENGINEERING
UNBLOCK, (3) folder consolidation + HUMAN_DECISIONS.md. Everything below is what was
actually done, with the reasoning that led to each choice. Machine-readable artifact logs
live in `docs/implementation_log.md` (Entries 1-3); this file is the thinking-level record.

---

## 0. Project Identity

- **Title:** *Do Circuits Survive Compression? A Null-Model Audit of Circuit Stability Under Quantization and Pruning*
- **Target venue:** AAAI Undergraduate Consortium (AAAI UC)
- **Repo:** `D:\Users\SUPRATIK\AAAI-UC\Standalone Project\Circuits_Under_Compression\`
- **Upstream forks (READ-ONLY):** `../circuit-tracer-0.5.2/`, `../sae-pruning-paper-main/`
- **Governing docs (workspace root, parent folder):** `CLAUDE.md`, `PRD.md`, `AI_RULES.md`, `ARCHITECTURE.md`
  plus the proposal PDF `Circuits_Under_Compression_AAAI_UC_Proposal.pdf`.

### The one-sentence thesis (internalized throughout all work)
> The question is not "did the circuit change" but "did it change more than
> nothing-in-particular changes it" — and the null distribution that defines "nothing in
> particular" is **frozen before any real compression run** and may never be re-tuned afterward.

### The gap being filled
1. Interpretability findings (circuits) are produced on full-precision models, but
   deployed models are quantized/pruned; whether circuits survive is essentially unmeasured.
2. Existing compression audits cover SAE features and linear probes (representations),
   not computation.
3. The one prior circuit claim under compression (VLMs, arXiv:2603.25035) is qualitative
   and was made without a noise floor — despite two documented measurement pathologies:
   threshold dependence (CIRCUS, arXiv:2603.00523; level-of-comparison, arXiv:2607.18921)
   and hidden mediator interactions in activation patching (NIE = PIE + INT, arXiv:2606.27510).

### Our novel contribution
- Build and **freeze a null model first** (matched-magnitude random perturbation,
  matched-perplexity control, seed/threshold ensembles), then measure circuit change under
  real compression against that floor.
- **Circuit Survival Index:** CSI(c,T) = D(c) / median(D_null(c)), with bootstrap CIs.
- Report **edge inclusion frequencies** s(e) instead of binary membership.
- Test every conclusion at **two levels of description** (exact-edge overlap AND
  routing-head-set overlap — these can disagree, Jaccard@10 0.14–0.16 vs 0.55–0.67, arXiv:2607.18921 §4).
- Test whether circuit-level damage rankings agree with published feature-level damage
  rankings on the same models (cross-audit, Spearman).

---

## 1. Governing Documents and the Rules They Enforce (as understood by the agent)

| Document | What it governs | Key constraints I must never break |
|---|---|---|
| `AI_RULES.md` | Integrity guardrails (non-negotiable) | §1.1 seeded randomness only; §1.2 configs/runs immutable once started (never overwrite a run dir); §1.4 `frozen/` append-only with hashes; §1.5 null-before-effect (Stage B freeze precedes any Stage C cell); §2.2 never invent numbers (hashes, licenses, arXiv numbers) — cite only what was verified; §3 Novelty Protection Zone (NPZ) — no AI-initiated modification without explicit human approval; §4.2 pre-register before measurement; §5 no PII; §7 every AI-gen file logged in `docs/implementation_log.md` |
| `CLAUDE.md` | Research identity + paper-to-code map + topology | §2 maps proposal sections to modules; §3 repository topology (source of truth for structure); §4 run-naming convention `{YYYYMMDD}_{stage}_{model}_{task}_{compression-or-null}_{configset}_seed{S}`; §5 glossary (circuit, s(e), core/contingent/noise, D(c), CSI, two levels, NIE/PIE/INT, null freeze, matched-perplexity control); §6 compute constraints + scope-cut order (null model and two-level reporting are NOT cuttable); §7 citation/attribution protocol (adapted code headers carry `# Adapted from: <repo> @ <commit>, <license>`) |
| `ARCHITECTURE.md` | System blueprint | §1 Algorithm 1 stages A-D; §2 schemas (edges.parquet, freq.parquet, run tags, frozen meta.json); §3 tracking (tags, Hydra); §4 protocols (Extractor/Compressor/Perturber interfaces); §5 compute plan (DGX Spark 128 GB unified memory, RTX 4060 for Pythia sweeps, all models ≤ 2B); §6 exit gate |
| `PRD.md` | Claims → experiments contract | Claims C1-C8; every experiment maps to a claim; artifact checklist |

### Novelty Protection Zone (AI_RULES.md §3, CLAUDE.md §2) — "🔒 Novelty zone" modules
`src/nulls/`, `src/metrics/`, `src/ensemble/`, `src/compare/`, `src/causal/`,
`analysis/cross_audit.py`. Readable and proposed-on by the agent; **never modified without
explicit human approval**. (During the sessions below, the PI granted a *provisional
engineering* scope that permitted DRAFT implementations of these modules with explicit
`# scientific-status: PROVISIONAL_ENGINEERING_ONLY` headers and `reviewed-by: PENDING` —
they remain unapproved scientific logic.)

### The 9 PI decisions (Q1–Q9) — the human-input surface
- **Q1** band cutoffs core/contingent/noise (affects C3, C1)
- **Q2** distance function D (affects C7 — CSI is downstream of D) ⭐ most important
- **Q3** ensemble sizes B, S, R (C1, C3, C7)
- **Q4** B non-nested threshold-grid design (C3)
- **Q5** HuggingFace revision pins (reproducibility, artifact)
- **Q6** task prompt datasets — source/counts/license (C1, C2, C3)
- **Q7** calibration dataset licenses for GPTQ/AWQ
- **Q8** experiment tracker (W&B vs MLflow vs local JSON)
- **Q9** upstream commit hashes + licenses for the two adapted forks

---

## 2. Session 1 — Repository Scaffold (previous sessions; implementation_log Entries 1-2)

Goal: a submission-ready skeleton with no fake science — every scientific value either
placeholder or PI-gated.

### 2.1 What was built
- **Topology** per CLAUDE.md §3: `src/`, `configs/`, `experiments/`, `frozen/`,
  `analysis/`, `tests/` (unit/integration/regression), `data/`, `paper/`, `artifact/`,
  `THIRD_PARTY_LICENSES/`, `docs/`.
- **`src/interfaces.py`** — Protocol definitions (ARCHITECTURE.md §4): Extractor,
  Compressor, Perturber, and friends. All concrete classes implement these; the mock
  pipeline is also a full Protocol citizen.
- **`src/io/schema.py`** — the schema layer: dataclasses `Edge`, `Graph`, `FreqVector`,
  `EnsembleResult`, `RunMetadata`, `FrozenMeta`, `CSIRow`; column specs
  `EDGES_COLUMNS = (src_component, dst_component, config_id, seed, included)` and
  `FREQ_COLUMNS = (edge_id, s_e, band)`; `validate_run_tags`; frozen `meta.json` schema;
  parquet I/O via pyarrow (optional dependency).
- **`src/utils/`** — `seeding.py` (explicit seeded generators only: `create_seed_generator`,
  `derive_child_seed` — AI_RULES §1.1), `hashing.py` (`hash_config`, `hash_file`,
  `manifest_entry`, `generate_manifest` — AI_RULES §1.4), `run_naming.py`
  (`build_configset`, `build_run_name`, `validate_run_name` — CLAUDE.md §4).
- **`src/guards/`** — `stage_guard.py` (`assert_stage_b_frozen`, `assert_null_frozen_hash`
  — the "null before effect" gate), `config_guard.py` (detects OPEN decisions Q1-Q6 from
  the resolved config; Stage-A gating).
- **Hydra configs** — `configs/config.yaml` root composition; groups `ensemble/default`
  (B16xS5), `ensemble/decompose`, `distance/placeholder`, `compression/placeholder`,
  `null/placeholder`; `model/{gemma2_2b,llama32_1b,pythia160m,pythia410m}.yaml`;
  `task/{ioi,greater_than,docstring}.yaml` (all with `pi_confirmed: false` and
  PLACEHOLDER dataset fields).
- **`experiments/run_stage_a.py`** — Stage A entrypoint: run-identity validation
  (run name, config hash), immutable run dir (FileExistsError on second identical run),
  resolved-config dump, tags, mock-pipeline support, edges.parquet + freq.parquet output,
  explicit no-frozen/no-compression guarantees.
- **Extraction layer** — `attribution_graph.py` (adapter over circuit-tracer, upstream API
  verified against the local fork), `edge_pruning_graph.py`, `dense_node_variant.py`
  (C8 skeleton), `mock_extractor.py` (deterministic CI stand-in).
- **Ensemble layer** — `circus_wrapper.py` (CIRCUS-style runner), `inclusion_freq.py`
  (s(e)), `decompose.py` (core/contingent/noise bands).
- **`docs/pi_decisions.md`** — machine-readable Q1-Q9 tracker, all OPEN.
- **`docs/upstream_pins.md`** + **`THIRD_PARTY_LICENSES/`** — both license texts copied
  and verified locally.

### 2.2 Key finding of Session 1
Neither upstream local fork is a git repository → commit hashes are undetectable locally.
Per AI_RULES §2.2 (never invent), this was recorded as `UNKNOWN_LOCAL_FORK` in
`docs/upstream_pins.md` and surfaced to the PI as **Q9** — never guessed.

### 2.3 Session-1 exit state
69 tests passed, 1 skipped (the skip is the pre-existing pyarrow/HF-optional regression
test). Everything scientific blocked on Q1, Q3, Q4, Q5, Q6.

---

## 3. Session 2 — PI PROVISIONAL ENGINEERING UNBLOCK (2026-08-07)

### 3.1 The PI's mandate (8 tasks) and its 6 binding constraints
The PI unblocked ENGINEERING-ONLY work with 6 explicit numbered constraints:
1. **Scope:** engineering dry-run + synthetic tests only.
2. **Stage B freeze gated by explicit flags** — `freeze_approved` / `freeze`, never implicit.
3. **Stage C real compression refused in engineering mode.**
4. **`frozen/` writable only in scientific mode with a verified `null_frozen_hash`.**
5. **Q9 = UNKNOWN_LOCAL_FORK** — no invented commit hashes, licenses, or arXiv numbers.
6. **After the final output: STOP.** No model download, GPU work, freeze, or real
   compression without a later PI instruction.

### 3.2 The mode system (the conceptual heart of the unblock)
Two execution modes selected per run from `configs/mode/`:
- **`engineering_dry_run`** (DEFAULT): provisional defaults allowed; Q5/Q9 degrade to
  warnings, not blocks; but **still refuses** Stage B freeze, Stage C real compression,
  and any write under `frozen/`. Every run records the 9 active provisional resolutions +
  the still-open questions in `run_meta.json` for audit.
- **`scientific_run`** (explicit opt-in): strict. Blocks while ANY final decision is OPEN
  (Q1/Q3/Q4/Q5/Q6 for Stage A), refuses synthetic/mock components outright, and does NOT
  accept provisional values as finals.

### 3.3 Task 1 — Decision layer (statuses + provisional values)
`docs/pi_decisions.md`: all 9 rows → `PROVISIONAL_ENGINEERING_DEFAULT` with the audit block
(approved_by: PI prompt, date: 2026-08-07, scope: engineering dry-run and synthetic tests
only, final_pre_registration: **false**, remaining_action: final PI confirmation before
Stage B freeze / Stage C real runs).

New file `docs/pi_decisions_resolved_provisional.md` — the single source of truth for the
provisional values:
- **Q1:** core ≥ **0.90**, noise ≤ **0.10** (cutoffs live in `configs/ensemble/decompose_provisional.yaml`)
- **Q2:** primary = **L1**, alternative = **jensen_shannon** (pre_registered_for_stage_c: false)
- **Q3:** synthetic tests **B=4, S=2, R=3** (`provisional_b4_s2_r3.yaml`); Pythia pilot
  B16xS5xR20 stays PROPOSED / NOT approved — it is a compute-budget decision for the PI on a GPU
- **Q4:** **deterministic seeded non-nested grid** generated from the run seed
  (`src/ensemble/threshold_grid.py`)
- **Q5:** revision null + `allow_local_or_mock_model: true` + `pin_required_before_real_run: true`
- **Q6/Q7:** synthetic seeded prompts / synthetic calibration only — no external downloads
- **Q8:** **local JSON** `run_meta.json` as source of record
- **Q9:** `UNKNOWN_LOCAL_FORK` for both forks (never invented)
Plus 6 binding constraints (freeze flags, frozen/hash rules, engineering limits, etc.).

### 3.4 Task 2 — Config layer
Created: `configs/pi/provisional_defaults.yaml` (machine-readable mirror of the above),
`configs/mode/{engineering_dry_run,scientific_run}.yaml`, `configs/distance/provisional_l1_js.yaml`,
`configs/ensemble/provisional_b4_s2_r3.yaml`, `configs/ensemble/decompose_provisional.yaml`,
`configs/task/synthetic-{ioi,greater-than,docstring}.yaml`, `configs/model/mock_model.yaml`.
`configs/config.yaml` defaults rewritten to the engineering composition (mock model +
synthetic-ioi + B4xS2 + provisional cutoffs + L1/JS distance); `distance/placeholder.yaml` deleted.

### 3.5 Task 3 — Mode-aware guard rewrite (`src/guards/config_guard.py`)
Public API: `mode_of(cfg)`, `report_open_questions(cfg, pipeline)` (Q1-Q6 detectable;
Q7/Q8/Q9 docs-only), `report_provisional_resolutions(cfg)` (9 entries in engineering,
`[]` in scientific), `assert_no_gating_questions(cfg, pipeline)` (scientific blocks
synthetic + OPEN gating), `assert_engineering_dry_run_limits(resolved, stage)`
(engineering refuses stageB freeze / stageC / frozen paths). Constants
`MODE_ENGINEERING`, `MODE_SCIENTIFIC`, `PI_DECISIONS_DOC`.

### 3.6 Task 4 — The 19 draft modules (all `# scientific-status: PROVISIONAL_ENGINEERING_ONLY`)
Every one is a DRAFT implementation of a paper object, marked `reviewed-by: PENDING`,
with real-model paths raising `NotImplementedError`. Details:

| Module | Design decisions made (thinking) |
|---|---|
| `src/ensemble/threshold_grid.py` | `generate_seeded_non_nested_grid(B, seed, node_range=(0.55,0.95), edge_range=(0.90,0.995))`; config ids `seedgrid-{i}`; rejection sampling guarantees pairwise non-nested (no config dominates another on BOTH axes); deterministic per seed; B=0 raises |
| `src/nulls/matched_magnitude.py` | `generate_null_deltas(magnitudes, shapes, R, seed)` returns R delta-sets, each scaled so per-tensor Frobenius norm EXACTLY equals the compression-induced magnitude (proposal §2.1(a)); `MatchedMagnitudePerturber.apply` supports MockModel only |
| `src/nulls/matched_perplexity.py` | `MatchedPerplexityControl.tune` → NotImplementedError; `provisional_placeholder(target_ppl)` returns a clearly-marked placeholder; `MATCHED_PPL_TOLERANCE_DEFAULT = 0.05` |
| `src/metrics/csi.py` | `csi(d_c, d_null, n_boot=1000, seed=0)` → `{csi, ci_lo, ci_hi, D, dnull_median}`; 2.5/97.5 percentile bootstrap over null resamples; refuses median=0 |
| `src/compare/distances.py` | `l1_distance` over the union of keys (missing → 0); `jensen_shannon_distance` over normalized vectors; disjoint support → `sqrt(ln2)` ≈ 0.8326; 0·log0 := 0 |
| `src/compare/two_level.py` | `exact_edge_overlap`, `routing_head_overlap` (head inclusion = max s(e) over incident edges; provisional `HEAD_NODE_RE ^L\d+\.H\d+$`), `compare_at_both_levels` — every claim stated at both levels per arXiv:2607.18921 |
| `src/causal/patch_diagnostic.py` | `nie_pie_int(pie, nie)`; `interaction_dominated(pie, nie, ratio_threshold=0.5)`: |INT| > 0.5·max(|NIE|,|PIE|,eps); `run` → NotImplementedError (ref arXiv:2606.27510) |
| `analysis/cross_audit.py` | `spearman_rank` (average ranks, numpy-only, NaN for <2 pairs or zero variance), `circuit_vs_feature_damage_rank_agreement` — the refs [1,2] agreement test |
| `src/compression/rtn.py` | symmetric RTN: scale = max|w|/(2^(bits-1)-1), zero_point 0; MockModel path live, real path blocked |
| `src/compression/gptq.py` / `awq.py` | MockModel dry-run stand-ins (explicitly RTN-helper, NOT GPTQ/AWQ); real implementations NotImplementedError — no fake calibration curves |
| `src/compression/magnitude_prune.py` | rewritten; adapted-header `UNKNOWN_LOCAL_FORK` + MIT; `prune_magnitude` (global scope with exact-count correction; per-tensor via mask), `prune_wanda` (score = |w|·sqrt(s2)); `MagnitudePruner` MockModel path; lm_head excluded |
| `src/compression/wanda.py` | `WandaPruner`; `synthetic_second_moments` seeded from `derive_child_seed(seed, "synthetic-second-moments")` |
| `src/extraction/edge_pruning_graph.py` / `dense_node_variant.py` | rewritten with MockModel paths (edge_scores ≥ edge_threshold); real extraction NotImplementedError |
| `src/extraction/mock_extractor.py` | rewritten: synthetic-model path (uses model.edge_scores) + legacy None path (threshold-based seeded RNG) |
| `src/ensemble/{circus_wrapper,inclusion_freq,decompose}.py` | scientific-status headers added (NPZ modules; logic untouched) |
| `experiments/run_stage_b.py` | draft null pipeline: dense inclusion frequencies → R null draws via `derive_child_seed(seed, "null-draw", r)` → `dnull.parquet` with columns `(r, distance_l1, distance_jensen_shannon)`; meta.json `frozen: false` + 64-hex `null_frozen_hash` + run_meta note "DRAFT null distribution (engineering dry-run); NOT a Stage B freeze."; refuses freeze in engineering |
| `experiments/run_stage_c.py` | `frozen_cell_path` → `frozen/{model}/{task}/{cell}`; `assert_stage_b_frozen` + `assert_null_frozen_hash` BEFORE anything else (null-before-effect); then engineering refusal; then NotImplementedError |
| `experiments/run_stage_d.py` | requires `stage_d.stage_c_run_dir` to exist; then limits; then NotImplementedError |
| `experiments/dry_run_stage_a.py` | `engineering_stage_a_cfg(seed=0, run_root="runs")` mirrors config.yaml defaults; prints DRY-RUN OK |

### 3.7 Task 5 — Synthetic stack (`src/synthetic/`)
- `mock_model.py` — `MockModel(seed, n_layers=4, n_heads=2, d_model=8)`; weights keyed
  `L{l}.{type}.W` (N(0,1)); `nodes()` = heads + MLP per layer; `clone()` /
  `apply_weight_delta` (never mutates — Perturber protocol); `weight_delta_frobenius`;
  `edge_scores(task_cfg, seed) = clip(0.5·base + 0.5·coupling, 0, 1)` where base is a
  seeded per-(task,edge) uniform draw and coupling = `0.5 + 0.5·tanh(<w_src, w_dst>/2)`.
- `synthetic_tasks.py` — families `indirect-object-identification-synthetic`,
  `greater-than-synthetic`, `docstring-completion-synthetic`; 24/24/12 prompts;
  `load_synthetic_prompts` REFUSES non-synthetic tasks ("synthetic tasks" error).
- `synthetic_calibration.py` — deterministic int64 tokens, seeded provenance.

### 3.8 Task 6 — Tests (8 files; all unit/integration)
`test_config_guard.py` (15), `test_distance_l1_js.py` (10), `test_threshold_grid_generator.py`
(6), `test_synthetic_task_prompts.py` (7), `test_stage_b_refuses_freeze.py` (4),
`test_stage_c_requires_frozen_null.py` (5), `test_engineering_dry_run_stage_a.py` (4),
`test_scientific_run_still_blocked.py` (4).

### 3.9 Verification loop — every failure found and why (the "thinking" trail)
1. **`NameError: mode`** — `mode` was referenced in the tags dict before the guard block
   defined it. Fix: compute `mode = mode_of(resolved)` immediately after `_resolve`.
2. **Scientific-blocked tests failed on second call with `FileExistsError`** — the guard
   ran AFTER run-dir creation, so a refused run still created a directory. Fix: move the
   whole PI-decision guard BEFORE any filesystem write. Consequence: a refused run never
   leaves artifacts. (This is the right behavior for a guard.)
3. **`run_stage_c` gating order** — `assert_no_gating_questions` fired before the
   frozen-null guard in scientific mode. Fix: frozen-null checks first (AI_RULES §1.5
   null-before-effect is the most fundamental gate).
4. **Zero edges in the dry run** — MockModel scores maxed at ≈0.83 because coupling used
   `tanh(dot/8)`, while seeded grid edge thresholds go to ~0.995. Fix: coupling →
   `tanh(dot/2)`, which puts tanh in the saturating regime (dot ~ N(0,√8) for d_model=8),
   so the mock score distribution actually spans the grid. Mechanical, documented,
   explicitly NOT a scientific scoring rule.
5. **Float precision** — `0.6000000000000001 != 0.6` in the L1 test → `pytest.approx`.
6. **`_resolved_cfg` mangling** — duplicate `mode` kwarg + broken indentation from an
   earlier edit → module-level function, fixed.
7. **Stage C tests hit the CWD-relative `frozen/`** — `frozen_cell_path` defaults to
   `frozen/` relative to CWD, but tests wrote under `tmp_path`. Fix: tests set
   `frozen_root=str(tmp_path/"frozen")`.
8. **`threshold grid has 1 configs but ensemble.B=4`** — the fully-resolved scientific
   test supplied 1 grid entry; fixed to 4.

### 3.10 Session-2 exit state
- **Full suite: 118 passed, 1 skipped** (my interim "85" claim was an unverified miscount;
  the reconciled arithmetic: 69+1 baseline − 6 replaced old config-guard tests + 55 new =
  119 items).
- **Dry run:** `python -m experiments.dry_run_stage_a 0` →
  `runs/20260806_stageA_mock_synthetic-ioi_dense_B4xS2_seed0/` — 14 edges, 7 unique,
  mode=engineering_dry_run, B=4, S=2, n_cells=8, band_cutoffs_resolved=True,
  9 provisional resolutions in run_meta, still-open = Q3/Q4/Q5/Q6.
- `frozen/` untouched; no downloads, no GPU.

---

## 4. Session 3 — Folder Consolidation + HUMAN_DECISIONS.md (2026-08-07)

### 4.1 PART A — Safe folder-count reduction
**Before tree** (dirs | file counts, pycache excluded): analysis|3, artifact|1,
configs|1 + 8 Hydra groups (compression 1, distance 1, ensemble 4, mode 2, model 5,
null 1, pi 1, task 6), data|1, docs|4, experiments|7, frozen|1, paper|1,
runs|1 (+1 run dir|4), src|2 (causal 2, compare 3, compression 6, ensemble 5, extraction 5,
guards 3, io 2, metrics 2, nulls 3, synthetic 4, utils 4), tests|1 (integration 5,
regression 2, unit 13), THIRD_PARTY_LICENSES|3.

**Protected list** (untouched): all NPZ folders, `analysis/`, `frozen/`, `experiments/`,
`tests/` (split kept per ARCH §6), `configs/` + every Hydra group (composition depends on
group names — NOT merged even at 1 file), `data/ paper/ artifact/ runs/`,
`THIRD_PARTY_LICENSES/`, the four root docs, `docs/`.

**The merge:** `src/io/` (schema.py), `src/utils/` (seeding.py, hashing.py, run_naming.py),
`src/guards/` (stage_guard.py, config_guard.py) → **`src/common/`**, each file moved intact
as its own module; new `src/common/__init__.py` re-exports every public name (modules +
their `__all__`/module-level API: Edge, Graph, FreqVector, EnsembleResult, RunMetadata,
FrozenMeta, CSIRow, EDGES_COLUMNS, validate_run_tags, edge_id, parquet readers/writers,
create_seed_generator, derive_child_seed, hash_config/file, manifest_entry,
generate_manifest, build_configset/run_name, validate_run_name, assert_stage_b_frozen,
assert_null_frozen_hash, mode_of, report_open_questions, report_provisional_resolutions,
assert_no_gating_questions, assert_engineering_dry_run_limits, MODE_ENGINEERING,
MODE_SCIENTIFIC, PI_DECISIONS_DOC). The three old `__init__.py` docstrings were folded
into the new package docstring (content preserved). No `.gitkeep`-only folders removed —
every such folder (`artifact`, `paper`, `data`, `runs`) is planned AAAI structure, so
none were "kept (unsure)" — all kept knowingly. `src/synthetic/` stayed separate.

**Reference fixup (A5):** a Python relocation script rewrote 33 files
(`src/guards/`→`src/common/`, `src.guards`→`src.common`, `src/io/`→`src/common/`,
`src.io`→`src.common`, `src/utils/`→`src/common/`, `src.utils`→`src.common`,
`..guards`→`..common`, `..io`→`..common`, `..utils`→`..common`; longest-first, UTF-8 safe):
- 16 Python files with imports: `experiments/run_stage_{a,b,c,d}.py`,
  `experiments/__init__.py`, 12 test files.
- 11 `src/` files with relative imports (`..io.schema` / `..utils.seeding`) inside
  `metrics, compression, synthetic, extraction, ensemble, nulls` — these are NPZ modules;
  only their import paths changed, zero logic touched.
- 5 docs/configs: `configs/config.yaml` comments, `docs/pi_decisions.md`,
  `docs/pi_decisions_resolved_provisional.md`, docstrings in `run_stage_a/b/c.py`,
  `src/common/stage_guard.py`.
- **`docs/implementation_log.md` deliberately left untouched** — it is an append-only
  historical record; the old paths there describe what existed at that time.
- **CLAUDE.md §3 topology** — added `src/common/` and `src/synthetic/` lines to the
  topology tree ONLY (root docs otherwise untouched).

**Verification (A6):** `python -m pytest` → **118 passed, 1 skipped** — identical to
pre-move; `--collect-only` confirmed the same 17 test files with the same per-file counts
(no tests added/removed/duplicated by the move).

### 4.2 PART B — `docs/HUMAN_DECISIONS.md`
Created verbatim from the PI-provided content: statuses (OPEN / PROVISIONAL /
PI-INPUT-NEEDED), grouping (GROUP 1 = mechanical lookups Q5/Q8/Q9; GROUP 2 = scientific
judgment Q1-Q4; GROUP 3 = data/licensing Q6/Q7), per-question sections with
What/Why/Options/Recommendation/Status/PI fill-in fields (17 `____` fields), the
"still blocked" map (Stage A → Q5+Q6+Q3+Q2/Q1; Stage B freeze → all finals; Stage C →
Q2 pre-registered + Q7 + Stage B frozen; Stage D → Stage C + INT rule; artifact → Q9 +
license), and the "also still open" list (artifact license MIT vs Apache-2.0, INT-flag
quantitative rule, DVC vs hashed-manifest).

### 4.3 PART C — GROUP 1 placeholders mirrored
`docs/pi_decisions.md` Q5/Q8/Q9 sections now carry explicit `**PI fill-in (GROUP 1,
mirrors docs/HUMAN_DECISIONS.md ...)**` blocks with the same `____` fields (5 HF revisions;
tracker = [W&B / MLflow]; 2 upstream commit+license pairs). **No value was filled in** —
all remain PI-INPUT-NEEDED. Statuses in pi_decisions.md untouched.

### 4.4 After tree
Identical to before except: `src/common|7` added; `src/guards|3`, `src/io|2`,
`src/utils|4` gone; `docs|5` (HUMAN_DECISIONS.md added). `frozen/` still contains only the
scaffold `FREEZE_MANIFEST.json` (`{}`, dated 2026-08-06) — never written by any run.

---

## 5. Current State Snapshot (end of Session 3)

- **Tests:** 118 passed, 1 skipped. Command: `python -m pytest` (inside the repo).
- **Dry-run entrypoint:** `python -m experiments.dry_run_stage_a 0` (note: `-m`, not a
  bare script path — `python experiments/dry_run_stage_a.py 0` fails with
  `ModuleNotFoundError: experiments`).
- **Existing run:** `runs/20260806_stageA_mock_synthetic-ioi_dense_B4xS2_seed0/` with
  resolved_config.json, run_meta.json, edges.parquet, freq.parquet.
- **Mode system active:** config.yaml defaults to `mode=engineering_dry_run` +
  mock/synthetic; `mode=scientific_run` is strict and blocks.
- **Decision docs:** `docs/pi_decisions.md` (statuses + mirrored GROUP 1 fill-ins),
  `docs/pi_decisions_resolved_provisional.md` (provisional values),
  `docs/HUMAN_DECISIONS.md` (PI guide).
- **Docs:** `docs/implementation_log.md` Entries 1-3 (+ this file's entry 4),
  `docs/upstream_pins.md`.

## 6. Blocked / Intentionally Refused (all `NotImplementedError` or guard-raised)
- Stage A real extraction (needs `RUN MODEL DOWNLOAD` approval + Q5 pins + Q6 datasets)
- Stage B **freeze** (needs all finals + explicit `freeze_approved`)
- Stage C real compression (engineering refuses outright)
- Stage D reporting
- Real compressors: GPTQ, AWQ, Wanda, magnitude pruning on real models
- Attribution-graph extraction (pipeline A), `PatchDiagnostic.run`,
  matched-perplexity tuning
- Any write under `frozen/`, any model download, any GPU work

## 7. Next Steps (pending PI)
1. PI fills Q1-Q9 in `docs/pi_decisions.md` / `docs/HUMAN_DECISIONS.md`
   (Q5/Q8/Q9 are pure lookups; Q1-Q4 scientific choices; Q6/Q7 data/licensing).
2. Final confirmation → statuses become FINAL (pre-registration at the Stage B freeze).
3. Stage A real engineering (pinned HF revisions, upstream packages, download approval).
4. Exit gate per ARCHITECTURE.md §6 before any scientific claim.

## 8. Principles exercised throughout (and to keep exercising)
- **Never invent:** hashes (Q9 = UNKNOWN_LOCAL_FORK), licenses, arXiv numbers, or
  scientific values — every number used is either PI-approved provisional or seeded-synthetic.
- **Null before effect:** the frozen-null guard precedes every Stage C check.
- **Refusal is a feature:** guards raise early (before any artifact is written), making
  violations impossible rather than merely visible.
- **Append-only history:** logs and `frozen/` are never rewritten; corrections become new entries.
- **NPZ discipline:** novelty modules were drafted under explicit provisional approval with
  clear headers, and their scientific logic remains `reviewed-by: PENDING`.
