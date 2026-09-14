# [AI-GEN] agent=OpenCode date=2026-08-07 task=Mode-aware config guard (engineering_dry_run vs scientific_run; PI provisional unblock)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY

"""Decision-aware config validation (HUMAN_DECISIONS.md; PI provisional unblock 2026-08-07).

Two execution modes (configs/mode/*.yaml):

1. ``engineering_dry_run`` (DEFAULT while the project is in engineering phase):
   - PROVISIONAL defaults from docs/HUMAN_DECISIONS.md §3.2 are allowed;
   - synthetic tasks and mock models are allowed;
   - unknown upstream commit hashes (Q9) -> warning, never blocking;
   - null HF revisions (Q5) -> warning, never blocking;
   - STILL REFUSES: Stage B freeze, Stage C real compression, any write to frozen/.
   No result produced in this mode is scientific evidence.

2. ``scientific_run`` (must be EXPLICIT): keeps the strict blocking behavior:
   - requires final PI decisions (Q1-Q6 detectable from config values);
   - requires pinned HF revisions; requires verified (non-synthetic) task datasets;
   - requires verified calibration data; requires final distance pre-registration;
   - requires exact upstream commit hashes before artifact packaging (Q9);
   - synthetic/mock components are refused outright.

The guard NEVER chooses scientific values; it only detects OPEN state and mode
limits (AI_RULES.md §6).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PI_DECISIONS_DOC = "docs/HUMAN_DECISIONS.md"

MODE_ENGINEERING = "engineering_dry_run"
MODE_SCIENTIFIC = "scientific_run"
KNOWN_MODES = (MODE_ENGINEERING, MODE_SCIENTIFIC)
DEFAULT_MODE = MODE_ENGINEERING  # project is in the engineering phase (PI unblock)

_QUESTION_TEMPLATE = {
    "Q1": {"title": "band cutoffs (core/contingent/noise)", "file": "configs/ensemble/decompose/*.yaml", "claims": "C3, C1", "gating": "stageA"},
    "Q2": {"title": "primary distance function D (pre-registered)", "file": "configs/distance/*.yaml", "claims": "C1, C2, C3, C5, C6", "gating": "stageC"},
    "Q3": {"title": "ensemble sizes B, S, R confirmed by PI", "file": "configs/ensemble/*.yaml, configs/nulls/*.yaml", "claims": "C1, C3, C7", "gating": "stageA"},
    "Q4": {"title": "B non-nested threshold-grid design", "file": "configs/ensemble/*.yaml, src/science/threshold_grid.py", "claims": "C3", "gating": "stageA"},
    "Q5": {"title": "HuggingFace revision pins", "file": "configs/model/*.yaml", "claims": "reproducibility, artifact", "gating": "stageA"},
    "Q6": {"title": "task prompt datasets (source/counts/license)", "file": "configs/task/*.yaml", "claims": "C1, C2, C3", "gating": "stageA"},
    "Q7": {"title": "calibration dataset licenses", "file": "configs/compression/*.yaml, src/synthetic/synthetic_calibration.py", "claims": "GPTQ/AWQ cells, artifact", "gating": "stageC"},
    "Q8": {"title": "experiment tracker choice", "file": "experiments/run_stage_a.py", "claims": "run tags", "gating": "docs-only"},
    "Q9": {"title": "upstream commit pins + licenses", "file": "docs/HUMAN_DECISIONS.md §3.3", "claims": "AI_RULES §7, artifact", "gating": "docs-only"},
    # Q10/Q11 were forced into the open on 2026-08-08 by reading circuit-tracer:
    # its graph has no attention-head nodes, and its nodes are position-specific.
    "Q10": {"title": "coarse comparison-level scheme (circuit-tracer has NO attention heads)", "file": "configs/comparison/*.yaml, src/extraction/node_ids.py", "claims": "C2", "gating": "stageA"},
    "Q11": {"title": "edge position policy (upstream nodes carry pos; our edge id does not)", "file": "configs/comparison/*.yaml", "claims": "C1, C2, chance floor", "gating": "stageA"},
}

# Provisional values (docs/HUMAN_DECISIONS.md §3.2) — mirrors
# configs/provisional_defaults.yaml; documented here so the guard is auditable.
PROVISIONAL_VALUES: dict[str, Any] = {
    # Q1/Q2/Q4/Q10/Q11 pre-registered 2026-09-12 (PI-approved). These are no longer
    # engineering placeholders; the authoritative copies are the config files named
    # in each entry and asserted by value in tests/test_config_integrity.py.
    "Q1": {"core": 1.0, "noise": 0.5, "noise_strict": True,
           "source": "CIRCUS arXiv:2603.00523 §3.2"},
    "Q2": {"primary": "l1", "alternative": "jensen_shannon",
           "also_report": "normalized_l1", "pre_registered_for_stage_c": True},
    # Q3 confirmed 2026-09-14 by the pre-stated Run_Plan rule on the measured Pythia-160M timing.
    "Q3": {"synthetic_tests": {"B": 4, "S": 2, "R": 3}, "confirmed": {"B": 16, "S": 5, "R": 20},
           "pythia_pilot_approved": True},
    "Q4": {"design": "anti_diagonal", "node_range": (0.6, 0.9), "edge_range": (0.99, 0.95),
           "generated_from": "B alone (no seed)"},
    # Q5 resolved 2026-09-12: all pins live in configs/model/*.yaml and are asserted
    # by value in tests/test_config_integrity.py. Gated downloads still need approval.
    "Q5": {"revision": "pinned-in-model-configs", "pin_required_before_real_run": True},
    "Q6": {"use": "template-generated", "n_prompts": 300,
           "bootstrap_resampling_unit": "template", "external_download": True,
           "docstring_source_still_open": True},
    "Q7": {"calibration": "fineweb-edu", "calibration_tokens": 300_000,
           "calibration_seed": 7, "perplexity": "wikitext-2-raw-test",
           "external_download": True},
    "Q8": {"tracker": "wandb", "source_of_record": "local-json-run_meta",
           "tracker_is_source_of_record": False},
    # Q9 resolved 2026-09-12; sae_pruning pin is inferred, see HUMAN_DECISIONS.md §3.3
    "Q9": {"circuit_tracer": "8f1e2438df612464e229e44c4a00ff637bf9379b", "sae_pruning_paper": "261191804675e2d39d0a265320dbc0bc85afd30a"},
    "Q10": {"level2_scheme": "layer",
            "falsification": "coarse_jaccard_within_exact_edge_bootstrap_ci"},
    "Q11": {"position_policy": "aggregate", "matches_current_edges_schema": True,
            "pre_registered": True},
}

_KNOWN_IDS = frozenset(_QUESTION_TEMPLATE)


def mode_of(cfg: Mapping[str, Any]) -> str:
    """The execution mode from the resolved config (default: engineering_dry_run)."""
    mode = cfg.get("mode") or {}
    name = mode.get("name") if isinstance(mode, Mapping) else None
    if name in KNOWN_MODES:
        return str(name)
    return DEFAULT_MODE


def _q(question_id: str) -> dict[str, Any]:
    return {"id": question_id, "status": "OPEN", **_QUESTION_TEMPLATE[question_id]}


def _detect_open_questions(cfg: Mapping[str, Any], pipeline: str) -> list[dict[str, Any]]:
    """Config-detectable OPEN questions (Q1-Q6); Q7/Q8/Q9 are docs-only."""
    found: list[dict[str, Any]] = []

    ensemble = cfg.get("ensemble") or {}
    decompose = ensemble.get("decompose") or {}
    if decompose.get("core_threshold") is None or decompose.get("noise_threshold") is None:
        found.append(_q("Q1"))
    if (cfg.get("distance") or {}).get("name") in (None, "", "null"):
        found.append(_q("Q2"))
    if not ensemble.get("pi_confirmed", False):
        found.append(_q("Q3"))
    if ensemble.get("threshold_grid") in (None, []):
        found.append(_q("Q4"))
    if (cfg.get("model") or {}).get("hf_revision") in (None, ""):
        found.append(_q("Q5"))
    if not (cfg.get("task") or {}).get("pi_confirmed", False):
        found.append(_q("Q6"))
    comparison = cfg.get("comparison") or {}
    if comparison.get("level2_scheme") in (None, "", "null"):
        found.append(_q("Q10"))
    if comparison.get("position_policy") in (None, "", "null"):
        found.append(_q("Q11"))
    # sort by the NUMERIC id so Q10/Q11 follow Q9 rather than Q1
    return sorted(found, key=lambda q: int(q["id"][1:]))


def report_open_questions(cfg: Mapping[str, Any], pipeline: str) -> list[dict[str, Any]]:
    """All OPEN questions detectable from ``cfg`` (pipeline-independent)."""
    return _detect_open_questions(cfg, pipeline)


def report_provisional_resolutions(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The provisional engineering resolutions active for this run (audit trail).

    In engineering mode: Q1-Q9 are carried as PROVISIONAL_ENGINEERING_DEFAULT
    (HUMAN_DECISIONS.md §3.2). In scientific mode: nothing is
    provisionally resolved — the list is empty.
    """
    if mode_of(cfg) != MODE_ENGINEERING:
        return []
    return [
        {
            "id": qid,
            "status": "PROVISIONAL_ENGINEERING_DEFAULT",
            "approved_by": "PI prompt",
            "date": "2026-08-07",
            "scope": "engineering dry-run and synthetic tests only",
            "final_pre_registration": False,
            "value": PROVISIONAL_VALUES[qid],
        }
        for qid in sorted(_KNOWN_IDS)
    ]


def assert_no_gating_questions(cfg: Mapping[str, Any], pipeline: str) -> None:
    """Refuse runs that need decisions the current mode does not provide.

    - scientific_run: blocks while any Stage-A-gating question (Q1/Q3/Q4/Q5/Q6) is
      OPEN and refuses synthetic/mock components outright.
    - engineering_dry_run: no blocking (provisional defaults apply); the limits on
      freeze/compression/frozen-writes are enforced by
      ``assert_engineering_dry_run_limits`` at the stage runners.
    """
    if mode_of(cfg) != MODE_SCIENTIFIC:
        return

    model_cfg = cfg.get("model") or {}
    task_cfg = cfg.get("task") or {}
    if bool(model_cfg.get("synthetic")) or bool(task_cfg.get("synthetic")) or pipeline == "mock":
        raise ValueError(
            "scientific_run mode requires REAL (non-synthetic) components: "
            "mock/synthetic models, synthetic tasks and the mock pipeline are refused. "
            "Engineering dry-runs must use mode=engineering_dry_run."
        )

    open_qs = _detect_open_questions(cfg, pipeline)
    gating = [q for q in open_qs if q["gating"] == "stageA"]
    if not gating:
        return
    lines = [
        f"scientific_run blocked: {len(gating)} final PI decision(s) still required:"
    ]
    for q in gating:
        lines.append(f"  {q['id']} {q['title']} ({q['file']}) — affects {q['claims']}")
    lines.append(f"Final decisions are PI-owned; provisional engineering values are NOT "
                 f"accepted in scientific mode. See {PI_DECISIONS_DOC}.")
    raise ValueError("\n".join(lines))


def is_provably_synthetic(resolved: Mapping[str, Any], pipeline: str | None = None) -> bool:
    """True only when NOTHING in this run can touch a real model, dataset or corpus.

    All three must hold: the model config declares ``synthetic``, the task config
    declares ``synthetic``, and the pipeline is the mock extractor. Any one of them
    false means a real component could be reached, and the engineering-mode Stage C
    allowance below does not apply.
    """
    model_cfg = resolved.get("model") or {}
    task_cfg = resolved.get("task") or {}
    pipe = str(pipeline if pipeline is not None else resolved.get("pipeline", ""))
    return bool(model_cfg.get("synthetic")) and bool(task_cfg.get("synthetic")) and pipe == "mock"


def assert_engineering_dry_run_limits(resolved: Mapping[str, Any], stage: str) -> None:
    """Engineering-mode hard limits (invoked by the stage runners).

    Still refuses, in engineering mode:
    - the Stage B **freeze** (frozen/ is append-only, AI_RULES.md 1.4; the freeze is a
      pre-registration event requiring scientific mode + explicit approval);
    - Stage C on anything that is not provably synthetic — i.e. any run that could
      reach a real model, a real task dataset or a real compressor;
    - any run whose run_root (or explicit frozen path) lies under ``frozen/``.

    Stage C on a **provably synthetic** stack (mock model + synthetic task + mock
    extractor) IS allowed, and is how Algorithm 1 gets exercised end-to-end before a
    single GPU hour is spent. Such a run compresses a numpy toy, produces a CSI table
    stamped ``evidence: false``, and is no more a scientific result than the Stage A and
    Stage B dry-runs that were always permitted. The prohibition that matters — no real
    compression outside scientific mode — is unchanged and is what
    ``is_provably_synthetic`` pins down.
    """
    if mode_of(resolved) != MODE_ENGINEERING:
        return
    stage_b = resolved.get("stage_b") or {}
    freeze_requested = bool(stage_b.get("freeze_approved")) or bool(resolved.get("freeze"))
    if stage == "stageB" and freeze_requested:
        raise ValueError(
            "engineering dry-run mode REFUSES the Stage B freeze: frozen/ is "
            "append-only and the freeze is the pre-registration event (AI_RULES.md "
            "1.4). Freezing requires mode=scientific_run AND explicit freeze approval."
        )
    # The frozen-path prohibition is checked BEFORE the stage-scope prohibition: it is
    # the more fundamental rule (frozen/ is append-only, AI_RULES.md 1.4) and applies at
    # every stage, so it should be the message a caller sees when both would fire.
    run_root = str(resolved.get("run_root", "") or "")
    frozen_paths = [
        k for k in ("frozen_root", "frozen_path") if str(resolved.get(k, "") or "").startswith("frozen")
    ]
    if run_root.startswith("frozen") or frozen_paths:
        raise ValueError(
            "engineering dry-run mode REFUSES writes under frozen/: frozen/ is "
            "append-only (AI_RULES.md 1.4). Draft nulls belong under run_root."
        )

    if stage in ("stageC", "stageD") and not is_provably_synthetic(resolved):
        human = "Stage C" if stage == "stageC" else "Stage D"
        raise ValueError(
            f"engineering dry-run mode REFUSES {human} real compression on non-synthetic "
            "components: no real compression cell is valid evidence in this mode "
            "(docs/HUMAN_DECISIONS.md §3.2; Q7). Real runs require mode=scientific_run + "
            "final decisions + explicit Stage C approval. A fully synthetic dry-run "
            "(model.synthetic AND task.synthetic AND pipeline=mock) IS allowed and "
            "produces artifacts stamped evidence=false."
        )


__all__ = [
    "mode_of",
    "is_provably_synthetic",
    "report_open_questions",
    "report_provisional_resolutions",
    "assert_no_gating_questions",
    "assert_engineering_dry_run_limits",
    "PI_DECISIONS_DOC",
    "MODE_ENGINEERING",
    "MODE_SCIENTIFIC",
]
