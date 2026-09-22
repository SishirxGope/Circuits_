# CLAUDE.md — AI Agent Research Context

> Project: **Do Circuits Survive Compression? A Null-Model Audit of Circuit Stability Under Quantization and Pruning**
> Target venue: AAAI Undergraduate Consortium (AAAI UC).
> This file is persistent context for AI coding agents. Read it fully before touching any code. The scientific goal, not the code structure, is the thing you must not break.

---

## 1. Research Identity

1. **Problem:** Interpretability findings (circuits) are produced on full-precision models but deployed models are quantized or pruned, and whether circuits survive that transfer is essentially unmeasured; existing compression audits cover SAE features and linear probes (representations), not computation.
2. **Gap:** The one prior circuit claim under compression (VLMs, arXiv:2603.25035) is qualitative and was made without a noise floor, even though circuit measurement has two documented pathologies: threshold dependence (CIRCUS, arXiv:2603.00523; level-of-comparison dependence, arXiv:2607.18921) and hidden mediator interactions in activation patching (NIE = PIE + INT, arXiv:2606.27510).
3. **Our novel contribution:** We build and **freeze a null model first** (matched-magnitude random perturbation, matched-perplexity control, seed/threshold ensembles), then measure circuit change under real compression against that floor via the **Circuit Survival Index (CSI)**, reporting edge inclusion frequencies instead of binary membership, testing every conclusion at two levels of description, and testing whether circuit-level damage rankings agree with published feature-level damage rankings on the same models.

**The one-sentence version an agent must internalize:** the question is not "did the circuit change" but "did it change more than nothing-in-particular changes it," and the null distribution that defines "nothing in particular" is frozen before any real compression run and may never be re-tuned afterward.

---

## 2. Paper-to-Code Mapping

| Proposal / paper section | Scientific object | Code module | Status |
|---|---|---|---|
| §2.1 Null model (a): matched-magnitude perturbation | Random ΔW with per-tensor Frobenius norm matched to compression-induced ΔW | `src/science/matched_magnitude.py` | 🔒 Novelty zone |
| §2.1 Null model (b): matched-perplexity control | Alternate compression family tuned to same PPL (per Duan 2606.03002) | `src/science/matched_perplexity.py`, `src/science/perplexity.py` | 🔒 Novelty zone |
| §2.1 Null model (c): seed + threshold variation | CIRCUS-style ensemble of B non-nested pruning configs × S seeds | `src/science/circus_wrapper.py`, `src/science/threshold_grid.py` | 🔒 Novelty zone |
| §2.2 Circuit extraction, pipeline A | Attribution-patching graph | `src/extraction/attribution_graph.py` | Engineering (wraps public code) |
| §2.2 / §2.4 Node identity | Upstream feature/error/embed/logit nodes → our component IDs; carries the two open schema decisions | `src/extraction/node_ids.py` | Engineering + ⚠️ 2 PI decisions |
| §2.2 Circuit extraction, pipeline B | Edge-pruning graph | `src/extraction/edge_pruning_graph.py` | Engineering (wraps public code) |
| §2.3 Stability scores | Edge inclusion frequency s(e) ∈ [0,1]; core/contingent/noise decomposition | `src/science/inclusion_freq.py`, `src/science/decompose.py` | 🔒 Novelty zone |
| §2.4 Two-level comparison | Exact-edge overlap AND routing-head-set overlap, always paired | `src/science/two_level.py`, `src/science/distances.py` | 🔒 Novelty zone |
| §2.5 Interaction-aware patching | NIE + grouped-vs-single INT diagnostic | `src/science/patch_diagnostic.py` | 🔒 Novelty zone |
| §2.6 Cross-audit agreement | Spearman(circuit damage rank, feature damage rank from refs [1,2]) | `analysis/cross_audit.py` | 🔒 Novelty zone |
| §2.1 headline metric | CSI(c,T) = D(c) / median(D_null(c)) with bootstrap CI | `src/science/csi.py` | 🔒 Novelty zone |
| §3.2 Compression grid | RTN INT8→INT4, GPTQ, AWQ; magnitude + Wanda 0→60% | `src/compression/` | Engineering |
| §4 rule 1 Null freeze | The pre-registration event: append-only frozen store + hash verification | `src/common/freeze.py`, `experiments/freeze_stage_b.py` | 🔒 Novelty zone |
| §3.3 / C3 Threshold sweep | Does each conclusion survive the full threshold range? | `analysis/threshold_sweep.py` | 🔒 Novelty zone |
| AI_RULES 4.4 Chance floor | Random top-k baseline attached to every overlap statistic | `src/science/chance_floor.py` | 🔒 Novelty zone |
| AI_RULES 4.3 Multiplicity | Benjamini-Hochberg across the reported grid | `src/science/multiple_comparisons.py` | 🔒 Novelty zone |
| §3.3 Reported quantities | The CSI table (ARCHITECTURE.md §2 schema) | `src/common/csi_table.py` | Engineering |
| Algorithm 1, Stages A–D | Orchestration | `experiments/run_stage_{a,b,c,d}.py` | Engineering |

"🔒 Novelty zone" modules fall under the Novelty Protection Zone in `AI_RULES.md`: no AI-initiated modification without explicit human approval. **Every one of them lives in `src/science/`** (plus `analysis/cross_audit.py` and `frozen/`), so the zone is one directory, not five (consolidated 2026-08-08; module file names unchanged).

---

## 3. Repository Topology

Structured for AAAI artifact submission from day one.

```
AAAI - Standalone Project/           # workspace root
├── CLAUDE.md  PRD.md  AI_RULES.md  ARCHITECTURE.md   # the four governing docs
├── Circuits_Under_Compression_AAAI_UC_Proposal.pdf
├── circuit-tracer-0.5.2/            # upstream fork, READ-ONLY
├── sae-pruning-paper-main/          # upstream fork, READ-ONLY
└── Circuits_Under_Compression/      # the project repo
    ├── README.md
    ├── src/
    │   ├── science/     # 🔒 THE NOVELTY ZONE, one package (AI_RULES.md §3):
    │   │                #   matched_magnitude.py, matched_perplexity.py, perplexity.py,
    │   │                #   circus_wrapper.py, threshold_grid.py, inclusion_freq.py,
    │   │                #   decompose.py, distances.py, two_level.py, csi.py,
    │   │                #   patch_diagnostic.py, chance_floor.py, multiple_comparisons.py
    │   ├── compression/ # rtn.py, gptq.py, awq.py, magnitude_prune.py, wanda.py
    │   ├── extraction/  # attribution_graph.py, edge_pruning_graph.py, dense_node_variant.py, mock_extractor.py,
    │   │                #   real_model.py (pinned loader), eap.py (edge attribution patching),
    │   │                #   dense_prune.py (Q4 thresholds on dense-node scores), node_ids.py
    │   ├── common/      # schema.py, seeding.py, hashing.py, run_naming.py, stage_guard.py,
    │   │                #   config_guard.py, freeze.py (🔒 the freeze writer), csi_table.py
    │   ├── synthetic/   # mock_model.py, synthetic_tasks.py, synthetic_calibration.py (engineering dry-runs)
    │   ├── tasks/       # REAL prompt sets: base.py, ioi.py, greater_than.py, _acdc_vendored.py (generated)
    │   └── interfaces.py
    ├── configs/         # Hydra groups; IMMUTABLE once an experiment references them
    │   ├── model/  task/  compression/  distance/  mode/
    │   ├── ensemble/    # + ensemble/decompose/ (band cutoffs: final.yaml | provisional.yaml)
    │   ├── nulls/       # NOT "null/": a bare `null` YAML key is the null literal
    │   └── provisional_defaults.yaml
    ├── experiments/     # run_stage_{a,b,c,d}.py + freeze_stage_b.py + dry_run_stage_a.py
    │                    #   + time_attribution.py (Q3 timing pilot; NON-EVIDENCE)
    ├── frozen/          # FROZEN null distributions + hashes; append-only, never edited
    ├── analysis/        # 🔒 cross_audit.py, threshold_sweep.py; + figure scripts (Phase 5)
    ├── tests/           # flat; tier is in the filename:
    │                    #   test_*.py = unit, test_integration_*.py, test_regression_*.py
    ├── docs/            # PROGRESS.md, HUMAN_DECISIONS.md, Run_Plan.md, BOTTLENECKS_AND_HARDWARE.md,
    │                    #   implementation_log.md, project_history.md
    │                    #   + preregistration/ (criteria committed before their analyses),
    │                    #     reports/ (proposal, reports), archive/ (superseded files)
    ├── data/            # dataset + calibration provenance log (hash-versioned)
    ├── runs/            # run directories, {run_root}/{run_name}; immutable, never overwritten
    ├── paper/           # AAAI UC draft, figures, poster
    ├── artifact/        # packaging scripts for camera-ready zip
    └── THIRD_PARTY_LICENSES/
```

Structure history: `src/io` + `src/utils` + `src/guards` → `src/common` (2026-08-07);
`src/nulls` + `src/ensemble` + `src/compare` + `src/metrics` + `src/causal` → `src/science`,
`tests/{unit,integration,regression}` → flat `tests/`, `configs/pi/` → `configs/`,
`configs/null/` → `configs/nulls/` (2026-08-08). Every file was preserved; only paths changed.

---

## 4. Experiment Naming Convention (mandatory)

```
{YYYYMMDD}_{stage}_{model}_{task}_{compression-or-null}_{configset}_seed{S}
```

Examples:
- `20261012_stageB_pythia160m_ioi_null-matchedmag-int4_B16xS5xR20_seed0`
- `20261103_stageC_gemma2-2b_greaterthan_rtn-int6_B16xS5_seed3`

Rules:
- `stage` ∈ {stageA, stageB, stageC, stageD} per Algorithm 1. A run whose name says stageB (null) can never write into a stageC (real compression) results directory, and vice versa.
- The config set string must encode |B| (pruning configs), |S| (seeds), and R (null draws) so ensemble size is legible from the run name.
- Every W&B/MLflow run carries tags: `stage`, `model`, `task`, `compression_family`, `level` (exact-edge | routing-head), `null_frozen_hash`.

---

## 5. Key Definitions Glossary (paper-facing; use these exact terms)

- **Circuit:** a computation graph over model components (heads/MLPs as nodes, information flow as edges) extracted for a task, NOT a set of feature directions.
- **Edge inclusion frequency s(e):** fraction of the (B configs × S seeds) extraction ensemble whose graphs contain edge e. The reported circuit object is always the vector of s(e), never a binary edge list.
- **Core / contingent / noise decomposition:** partition of edges by s(e) bands (per CIRCUS): core ≈ retained across nearly all configs; contingent ≈ retained in some; noise ≈ rarely retained. PRE-REGISTERED 2026-09-12: CIRCUS's taxonomy — core s(e) = 1, contingent 0.5 ≤ s(e) < 1, noise s(e) < 0.5 (strict) — in `configs/ensemble/decompose/final.yaml`.
- **D(c):** distance between pre- and post-compression inclusion-frequency vectors for setting c. PRE-REGISTERED 2026-09-12: L1 primary, Jensen-Shannon ablation, and normalised L1 (L1 / |edge union|) reported alongside — `configs/distance/provisional_l1_js.yaml`. CSI is entirely downstream of this choice, which is why it was fixed before any run.
- **D_null(c):** distribution of the same distance under matched-magnitude random perturbation (R draws).
- **CSI (Circuit Survival Index):** CSI(c,T) = D(c) / median(D_null(c)), with bootstrap CI over B, S, r. CSI ≈ 1: compression damage indistinguishable from generic weight noise. CSI ≫ 1: compression is structurally selective. CSI < 1: compression gentler than random noise (surprising, publishable).
- **Two comparison levels:** (i) exact component-to-component edge overlap; (ii) routing-head-set overlap.
  ⚠️ **VERIFIED 2026-08-08: circuit-tracer emits NO attention-head nodes** (its four node kinds are
  `cross layer transcoder`, `mlp reconstruction error`, `embedding`, `logit`), so level (ii) is currently
  UNDEFINED for pipeline A and its projection would return the empty set. PRE-REGISTERED 2026-09-12:
  level (ii) is **layer** routing, with a falsification condition attached
  (`src/science/two_level.py::coarse_level_verdict`), and edges **aggregate over token positions** (Q11). Ref 2607.18921 shows these can disagree (Jaccard@10 0.14–0.16 vs 0.55–0.67); every claim is stated at both.
- **NIE / PIE / INT:** natural indirect effect = pure indirect effect + interaction term (ref 2606.27510). A component is **interaction-dominated** when the grouped-vs-single patching diagnostic shows its effect depends on sibling states; such components are excluded from headline stability claims and reported separately.
- **Null freeze:** the event after which `frozen/` contents and their hashes may not change. Pre-registration point. See AI_RULES.md.
- **Matched-perplexity control:** a different compression family tuned to the same perplexity as the setting under test (precedent: Duan 2606.03002).

---

## 6. Known Constraints

- **Compute:** DGX Spark, 128 GB unified memory (inference + attribution workload, no training). RTX 4060 runs Pythia sweeps in parallel. All models ≤ 2B params.
- **Dominant cost:** the ensemble (|B| × |S| × R per cell), not the models. CIRCUS-style pruning reuses one attribution run under many configs, so ensembles are near-free once raw attribution exists. Budget is controlled by shrinking R and B, **never by dropping the null**.
- **Timeline:** ~2 months to AAAI UC deadline. Scope-cut order if overrun (fixed, in order): compression levels and secondary models → task circuits → seeds. The null model and two-level reporting are NOT cuttable; they are the contribution.
- **Models:** Primary Gemma-2-2B, Llama-3.2-1B (overlap with CIRCUS and the pruning feature audit, enabling the cross-audit test). Secondary Pythia-160M/410M for dense threshold sweeps and seed replication; Pythia-70M as contact point with ref [2].
- **Reference circuits caveat:** IOI, greater-than, and docstring reference circuits were published on GPT-2 small / small attention-only models, not on Gemma-2-2B or Llama-3.2-1B. ⚠️ TODO: [QUESTION FOR PI] decide whether "change vs published reference" is claimed only on models where a published circuit exists, with Gemma/Llama circuits anchored instead to our own Stage-A dense ensemble. Current default assumption: published circuits serve as sanity anchors for the extraction stack; all compression deltas are measured against our own dense-reference ensemble on the same model.
- **Licensing:** Gemma-2 (Gemma license), Llama-3.2 (Llama community license), Pythia (Apache-2.0). Weight redistribution restrictions apply to Gemma/Llama; artifact ships configs + hashes + scripts, not compressed weights. No human-subjects data; no IRB required. Calibration corpus FineWeb-Edu (ODC-BY; 300,000 tokens, seed 7) and perplexity corpus WikiText-2 test (CC BY-SA 3.0) — Q7, pre-registered 2026-09-12 (`configs/calibration/final.yaml`).

---

## 7. Citation & Attribution Protocol

- Any code block adapted from a public release (refs [1] Borobia, [5] CIRCUS, [6] Sheng & Fu, TransformerLens, nnsight, SAELens) carries a header comment: `# Adapted from: <repo URL> @ <commit hash>, <license>`. Upstream licenses are copied into `THIRD_PARTY_LICENSES/`.
- Numbers quoted from prior work in code comments or docstrings carry the arXiv ID inline (e.g., `# Jaccard@10 0.14–0.16, arXiv:2607.18921 §4`).
- AI-generated code blocks are logged per the mandatory format in `AI_RULES.md` §7.
- Never cite from memory: every citation added to code or paper must be verified against the actual arXiv page / DOI before commit (AI_RULES.md §2).

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes_tool` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review — token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.
