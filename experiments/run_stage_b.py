# [AI-GEN] agent=OpenCode date=2026-08-07 task=Stage B entrypoint: null distributions (draft; freeze-guarded)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY

"""Algorithm 1, Stage B: null distributions (proposal §2.1; ARCHITECTURE.md §1).

Orchestration skeleton with the required guards:
1. Stage tag/tags validated (R > 0 required for Stage B).
2. PI-decision guard (mode-aware, src/common/config_guard.py).
3. **Freeze guard**: engineering_dry_run mode REFUSES the freeze (frozen/ is
   append-only, AI_RULES.md 1.4; freezing is the pre-registration event). Draft null
   distributions are written under ``run_root`` with ``frozen: false`` and a
   ``null_frozen_hash`` of the draft dnull.parquet, so Stage C's hash check is
   exercisable without freezing anything.
4. Real-model nulls: NotImplementedError (needs Stage B engineering + RUN MODEL
   DOWNLOAD). Engineering dry-run path: MockModel + matched-magnitude perturbation
   (src/science/matched_magnitude.py) against magnitudes from the magnitude-pruner
   dry-run cell (Q3/Q7 provisional).
"""

from __future__ import annotations

import datetime
import json
import statistics
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from hydra import main as hydra_main  # type: ignore[import-not-found]
    from omegaconf import DictConfig  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    hydra_main = None  # type: ignore[assignment]

from experiments.run_stage_a import PIPELINE_REGISTRY, _build_threshold_configs, _git_commit, _resolve
from src.compression.magnitude_prune import MagnitudePruner
from src.science.circus_wrapper import CircusEnsembleRunner
from src.extraction.mock_extractor import MockExtractor
from src.common.config_guard import (
    MODE_ENGINEERING,
    assert_engineering_dry_run_limits,
    assert_no_gating_questions,
    mode_of,
    report_open_questions,
    report_provisional_resolutions,
)
from src.common import schema
from src.science.matched_magnitude import generate_null_deltas
from src.science.distances import jensen_shannon_distance, l1_distance
from src.common.hashing import hash_config, hash_file
from src.common.run_dir import allocate_run_dir
from src.common.run_naming import build_configset, resolve_run_name
from src.common.seeding import derive_child_seed

DNULL_COLUMNS = ("r", "distance_l1", "distance_jensen_shannon")


def _load_model_guard(resolved: Mapping[str, Any]) -> Any:
    """Model loader shared with Stage A semantics (engineering: MockModel only)."""
    model_cfg = dict(resolved.get("model", {}))
    if mode_of(resolved) == MODE_ENGINEERING and model_cfg.get("synthetic"):
        from src.synthetic.mock_model import MockModel

        return MockModel(
            seed=int(resolved.get("seed", 0) or 0),
            n_layers=int(model_cfg.get("n_layers", 4)),
            n_heads=int(model_cfg.get("n_heads", 2)),
            d_model=int(model_cfg.get("d_model", 8)),
        )
    raise NotImplementedError(
        "Stage B real nulls are not available yet: they need pinned HF revisions, "
        "upstream packages and an explicit RUN MODEL DOWNLOAD approval. "
        "Engineering dry-runs use a MockModel (mode=engineering_dry_run)."
    )


def _synthetic_dnull_rows(resolved: Mapping[str, Any], model: Any, f_pre: Any) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Engineering-only D_null computation (matched-magnitude draws on MockModel).

    Magnitudes come from the dry-run magnitude-prune cell (Q7 stand-in); each of the
    R draws perturbs a COPY of the model and re-extracts the ensemble, then measures
    L1 and Jensen-Shannon distances vs the dense reference. Deterministic given
    (config, seed): every draw is seeded (AI_RULES.md 1.1).
    """
    from src.synthetic.mock_model import MockModel

    if not isinstance(model, MockModel):
        raise NotImplementedError("_synthetic_dnull_rows supports MockModel only")

    B, S = int(resolved["ensemble"]["B"]), int(resolved["ensemble"]["S"])
    R = int(resolved["nulls"]["R"])
    seed = int(resolved.get("seed", 0))
    configs = _build_threshold_configs(resolved)
    seeds = [seed + i for i in range(S)]

    pruner = MagnitudePruner(
        sparsity=float(resolved.get("stage_b", {}).get("sparsity", 0.3) or 0.3),
        global_scope=bool(resolved.get("stage_b", {}).get("global_scope", True)),
    )
    magnitudes = pruner.weight_delta(model, resolved)
    shapes = model.tensor_shapes()

    runner = CircusEnsembleRunner(MockExtractor())
    f_pre_dict = f_pre.frequencies if hasattr(f_pre, "frequencies") else dict(f_pre)

    rows: list[dict[str, Any]] = []
    for r in range(R):
        deltas = generate_null_deltas(
            magnitudes, shapes, 1, seed=derive_child_seed(seed, "null-draw", r)
        )[0]
        perturbed = model.apply_weight_delta(deltas)
        result = runner.run(model=perturbed, task=resolved["task"], configs=configs, seeds=seeds)
        f_post = result.freq.frequencies
        rows.append(
            {
                "r": r,
                "distance_l1": l1_distance(f_pre_dict, f_post),
                "distance_jensen_shannon": jensen_shannon_distance(f_pre_dict, f_post),
            }
        )
    return rows, magnitudes


def run_stage_b(cfg: Any) -> Path:
    """Run Stage B (draft null distributions) for a resolved config; returns run dir.

    Raises on: malformed run name, missing tags (R required), pre-existing run dir,
    engineering-mode freeze requests, unresolved scientific mode, and everything
    real-model related (NotImplementedError).
    """
    resolved = _resolve(cfg)
    if resolved["stage"] != "stageB":
        raise ValueError(f"run_stage_b is the Stage B entrypoint; got stage={resolved['stage']!r}")

    model_cfg, task_cfg = dict(resolved["model"]), dict(resolved["task"])
    ensemble = dict(resolved.get("ensemble", {}))
    B, S = int(ensemble.get("B", 0)), int(ensemble.get("S", 0))
    R = int((resolved.get("nulls") or {}).get("R", 0))
    seed = int(resolved.get("seed", 0))
    setting = str(resolved.get("setting", "null-matchedmag"))
    configset = build_configset(B, S, R=R)

    pipeline = str(resolved.get("pipeline", "mock"))
    if pipeline not in PIPELINE_REGISTRY:
        raise ValueError(f"unknown pipeline {pipeline!r}; expected one of {sorted(PIPELINE_REGISTRY)}")

    mode = mode_of(resolved)
    assert_no_gating_questions(resolved, pipeline)
    assert_engineering_dry_run_limits(resolved, stage="stageB")  # refuses freeze in engineering

    config_hash = hash_config(resolved)
    run_root = Path(resolved.get("run_root", "runs"))
    run_name = resolve_run_name(
        resolved.get("run_name"), datetime.date.today().strftime("%Y%m%d"), "stageB",
        model_cfg["name"], task_cfg["name"], setting, configset, seed,
    )
    run_dir = allocate_run_dir(run_root, run_name, config_hash)

    with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2, sort_keys=True, default=str)

    tags = {
        "stage": "stageB",
        "model": model_cfg["name"],
        "task": task_cfg["name"],
        "compression_family": resolved.get("compression_family", "null"),
        "compression_level": resolved.get("compression_level"),
        "comparison_level": resolved.get("comparison_level", "both"),
        "pipeline": pipeline,
        "mode": mode,
        "B": B,
        "S": S,
        "R": R,
        "seed": seed,
        "null_frozen_hash": None,
        "git_commit": _git_commit(),
        "config_hash": config_hash,
    }
    schema.validate_run_tags(tags, stage="stageB")

    model = _load_model_guard(resolved)

    # --- dense reference frequency vector (same ensemble machinery as Stage A) ----
    configs = _build_threshold_configs(resolved)
    seeds = [seed + i for i in range(S)]
    runner = CircusEnsembleRunner(MockExtractor())
    dense_result = runner.run(model=model, task=task_cfg, configs=configs, seeds=seeds)

    # --- null draws (engineering path) ---------------------------------------------
    rows, magnitudes = _synthetic_dnull_rows(resolved, model, dense_result.freq)

    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({c: [row[c] for row in rows] for c in DNULL_COLUMNS})
    pq.write_table(table, str(run_dir / "dnull.parquet"))

    meta = {
        "config_hash": config_hash,
        "seeds": seeds,
        "R": R,
        "frobenius_magnitudes": {k: float(v) for k, v in sorted(magnitudes.items())},
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "null_frozen_hash": hash_file(run_dir / "dnull.parquet"),
        "dnull_file": "dnull.parquet",
        "frozen": False,
        "mode": mode,
        "provisional": mode == MODE_ENGINEERING,
    }
    schema.validate_frozen_meta(meta)
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    run_meta = {
        **tags,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "dnull_hash": meta["null_frozen_hash"],
        "frozen": False,
        "open_questions": [q["id"] for q in report_open_questions(resolved, pipeline)],
        "provisional_resolutions": report_provisional_resolutions(resolved),
        "note": "DRAFT null distribution (engineering dry-run); NOT a Stage B freeze.",
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, sort_keys=True)

    print(
        f"[stageB-draft] run_name={run_name} R={R} dnull_hash={meta['null_frozen_hash'][:12]} "
        f"median_l1={statistics.median(r['distance_l1'] for r in rows):.4f} -> {run_dir}"
    )
    return run_dir


def _require_pyarrow():
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyarrow is required for dnull.parquet (ARCHITECTURE.md §2)") from exc


def main() -> None:
    if hydra_main is None:  # pragma: no cover
        print("hydra-core + omegaconf are required to run the entrypoint (optional dep for tests).", file=sys.stderr)
        sys.exit(1)

    @hydra_main(version_base=None, config_path="../configs", config_name="config")
    def _run(cfg: DictConfig) -> None:
        run_stage_b(cfg)

    _run()


if __name__ == "__main__":
    main()
