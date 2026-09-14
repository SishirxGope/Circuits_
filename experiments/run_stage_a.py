# [AI-GEN] agent=OpenCode date=2026-08-07 task=Stage A entrypoint: dense-reference ensemble extraction (mode-aware)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY

"""Algorithm 1, Stage A: dense reference circuit extraction + inclusion-frequency
ensemble + reporting protocol (PRD.md Phase 1; ARCHITECTURE.md §1).

Responsibilities (Step 8 contract):
- load model/task/ensemble configs (Hydra composition, ARCHITECTURE.md §3);
- create a run directory under ``run_root`` named per CLAUDE.md §4 and dump the
  resolved config + config hash there;
- validate required tracking tags (ARCHITECTURE.md §3) and log them to
  run_meta.json (W&B/MLflow integration is a later decision — HUMAN_DECISIONS.md Q8;
  the JSON log is the source of record for now);
- enforce PI decisions via src/common/config_guard.py: scientific_run refuses while
  final decisions are OPEN; engineering_dry_run (DEFAULT) allows the provisional
  engineering defaults (docs/HUMAN_DECISIONS.md §3.2) and records the
  active provisional resolutions + open questions in run_meta.json, for audit;
- call CircuitExtractor through the Protocol and EnsembleRunner (circus_wrapper);
- produce edges.parquet + freq.parquet (ARCHITECTURE.md §2 schemas);
- NEVER write to frozen/, NEVER compress, NEVER call Stage B/C logic.

Engineering dry-run (current phase): ``pipeline: mock`` + a synthetic mock model
(configs/model/mock_model.yaml) runs the full stack deterministically — no model
download, no GPU. Real pipelines (attr / edgeprune / dense-node) require Stage A
engineering + pinned HF revisions + explicit ``RUN MODEL DOWNLOAD`` approval.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:  # hydra-core is an optional dependency until real runs begin
    from hydra import main as hydra_main  # type: ignore[import-not-found]
    from omegaconf import DictConfig  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - CI/test environment
    hydra_main = None  # type: ignore[assignment]

from src.science.circus_wrapper import CircusEnsembleRunner
from src.science.decompose import decompose
from src.science.threshold_grid import (
    generate_anti_diagonal_grid,
    generate_seeded_non_nested_grid,
    is_non_nested,
)
from src.extraction.attribution_graph import AttributionGraphExtractor
from src.extraction.dense_node_variant import DenseNodeExtractor
from src.extraction.edge_pruning_graph import EdgePruningExtractor
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
from src.synthetic.mock_model import MockModel
from src.common.hashing import hash_config
from src.common.run_dir import allocate_run_dir
from src.common.run_naming import build_configset, resolve_run_name

PIPELINE_REGISTRY: dict[str, Any] = {
    "attr": AttributionGraphExtractor,
    "edgeprune": EdgePruningExtractor,
    "dense-node": DenseNodeExtractor,
    "mock": MockExtractor,
}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None


def _resolve(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, "to_container"):  # OmegaConf DictConfig
        return cfg.to_container(resolve=True)  # type: ignore[attr-defined]
    raise TypeError(f"expected Mapping or OmegaConf config, got {type(cfg).__name__}")


def _build_threshold_configs(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The B threshold configs for the ensemble (configs/ensemble/*.yaml).

    - If ``threshold_grid`` is a DESIGN (a mapping carrying ``design``), build the grid
      from it. ``anti_diagonal`` is the Q4 pre-registration (2026-09-12): the config
      records the design and the ranges, and the generator produces the B configs, so
      the config file never has to restate B configs that would then have to be kept
      in sync with B.
    - If ``threshold_grid`` is an explicit LIST, use it (each item must carry an ``id``).
    - engineering_dry_run mode: generate the Q4 PROVISIONAL seeded non-nested grid
      deterministically from the run seed. Retained only to reproduce pre-registration
      dry-runs; it adds a second seed axis for no scientific gain.
    - Otherwise refuse: the grid is a pre-registration item, never invented here.

    Whatever the source, the result is checked for non-nestedness before it is returned.
    A nested grid collapses consensus onto the loosest view and inflates every s(e)
    (CIRCUS arXiv:2603.00523 §3.2), and it would do so silently.
    """
    ensemble = dict(resolved.get("ensemble", {}))
    B = int(ensemble.get("B", 0))
    if B <= 0:
        raise ValueError(f"ensemble.B must be > 0, got {B}")

    grid = ensemble.get("threshold_grid")
    configs = _resolve_grid(grid, B, resolved)

    if not is_non_nested(configs):
        raise ValueError(
            f"the threshold grid for B={B} is NESTED: at least one config dominates "
            "another on both axes, so consensus collapses onto the loosest view and "
            "every s(e) is inflated toward it (CIRCUS arXiv:2603.00523 §3.2). This is "
            "the artifact the ensemble exists to avoid; refusing rather than measuring."
        )
    return configs


def _resolve_grid(
    grid: Any, B: int, resolved: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Grid source selection; see ``_build_threshold_configs`` for the contract."""
    if isinstance(grid, Mapping):
        design = grid.get("design")
        if design == "anti_diagonal":
            node_range = tuple(grid.get("node_range", (0.6, 0.9)))
            edge_range = tuple(grid.get("edge_range", (0.99, 0.95)))
            return generate_anti_diagonal_grid(B, node_range, edge_range)
        raise ValueError(
            f"unknown threshold_grid design {design!r}; expected 'anti_diagonal' (the "
            "Q4 pre-registration) or an explicit list of B configs"
        )

    if grid:
        return [dict(c, id=str(c.get("id", f"config{i}"))) for i, c in enumerate(grid)]

    if mode_of(resolved) == MODE_ENGINEERING:
        return generate_seeded_non_nested_grid(B, int(resolved.get("seed", 0)))

    raise ValueError(
        "ensemble.threshold_grid is null but real extraction requires the B non-nested "
        "threshold configs (CIRCUS, arXiv:2603.00523). The Q4 pre-registration is "
        "{design: anti_diagonal, node_range: [0.6, 0.9], edge_range: [0.99, 0.95]} — "
        "see configs/ensemble/default.yaml."
    )


def _load_model(resolved: Mapping[str, Any]):
    """Load the dense reference model.

    engineering_dry_run + ``model.synthetic`` -> a MockModel (no download, no GPU).
    ``pipeline: mock`` without a synthetic model -> None (legacy CI path).
    Real pipelines -> NotImplementedError until Stage A engineering (pinned HF
    revisions + upstream packages) and explicit ``RUN MODEL DOWNLOAD`` approval.
    """
    model_cfg = dict(resolved.get("model", {}))
    pipeline = str(resolved.get("pipeline", ""))
    if pipeline == "mock" and not model_cfg.get("synthetic"):
        return None
    if mode_of(resolved) == MODE_ENGINEERING and model_cfg.get("synthetic"):
        return MockModel(
            seed=int(resolved.get("seed", 0) or 0),
            n_layers=int(model_cfg.get("n_layers", 4)),
            n_heads=int(model_cfg.get("n_heads", 2)),
            d_model=int(model_cfg.get("d_model", 8)),
        )
    if pipeline == "dense-node" and not model_cfg.get("synthetic"):
        # Real dense-node extraction (2026-09-14): pinned revision, download gate and
        # architecture checks are all enforced inside load_pinned_model.
        from src.extraction.real_model import load_pinned_model

        return load_pinned_model(resolved)
    raise NotImplementedError(
        "Stage A real extraction is not available yet: model loading needs pinned HF "
        "revisions + upstream packages (TransformerLens/nnsight, circuit-tracer). "
        "Await RUN MODEL DOWNLOAD approval and Stage A engineering."
    )


def run_stage_a(cfg: Any) -> Path:
    """Run Stage A for a resolved config; returns the run directory path.

    Accepts a plain Mapping (tests) or an OmegaConf DictConfig (Hydra entrypoint).
    Raises on: malformed run name, missing required tags, pre-existing run dir
    (a run directory is immutable: never overwrite), unresolved threshold grid for
    real pipelines.
    """
    resolved = _resolve(cfg)
    mode = mode_of(resolved)

    # --- pipeline must be known before anything else ---------------------------------
    pipeline = str(resolved.get("pipeline", ""))
    if pipeline not in PIPELINE_REGISTRY:
        raise ValueError(f"unknown pipeline {pipeline!r}; expected one of {sorted(PIPELINE_REGISTRY)}")

    # --- PI-decision guard (mode-aware) ----------------------------------------------
    # engineering_dry_run (default): provisional defaults allowed; still refuses
    # Stage B freeze / Stage C compression / frozen writes (limits below); the
    # provisional resolutions + still-open questions are recorded in run_meta for
    # audit. scientific_run: refuses while any final decision is OPEN and refuses
    # synthetic components. Raised BEFORE any run directory is created, so a
    # refused run never leaves artifacts.
    open_questions = report_open_questions(resolved, pipeline)
    assert_no_gating_questions(resolved, pipeline)
    assert_engineering_dry_run_limits(resolved, stage="stageA")
    provisional_resolutions = report_provisional_resolutions(resolved)

    # --- run identity -----------------------------------------------------------
    stage = resolved["stage"]
    if stage != "stageA":
        raise ValueError(f"run_stage_a is the Stage A entrypoint; got stage={stage!r}")

    model_cfg = dict(resolved["model"])
    task_cfg = dict(resolved["task"])
    ensemble = dict(resolved.get("ensemble", {}))
    B, S = int(ensemble.get("B", 0)), int(ensemble.get("S", 0))
    seed = int(resolved.get("seed", 0))
    setting = str(resolved.get("setting", "dense"))
    configset = build_configset(B, S, R=None)
    declared = ensemble.get("configset")
    if declared not in (None, configset):
        raise ValueError(f"ensemble.configset {declared!r} != derived {configset!r} (B={B}, S={S})")

    run_name = resolve_run_name(
        resolved.get("run_name"), datetime.date.today().strftime("%Y%m%d"), stage,
        model_cfg["name"], task_cfg["name"], setting, configset, seed,
    )

    config_hash = hash_config(resolved)
    run_dir = allocate_run_dir(resolved.get("run_root", "runs"), run_name, config_hash)

    # --- resolved config dump (the run's identity) ------------------------------
    with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2, sort_keys=True, default=str)

    # --- tags (ARCHITECTURE.md §3) ----------------------------------------------
    tags = {
        "stage": stage,
        "model": model_cfg["name"],
        "task": task_cfg["name"],
        "compression_family": resolved.get("compression_family", "dense"),
        "compression_level": resolved.get("compression_level"),
        "comparison_level": resolved.get("comparison_level", "both"),
        "pipeline": resolved.get("pipeline", "mock"),
        "mode": mode,
        "B": B,
        "S": S,
        "R": 0,  # Stage A draws no nulls
        "seed": seed,
        "null_frozen_hash": None,  # Stage C/D only (AI_RULES.md 1.4)
        "git_commit": _git_commit(),
        "config_hash": config_hash,
    }
    schema.validate_run_tags(tags, stage=stage)  # raises ValueError listing gaps

    # --- ensemble configs + extractor (through the Protocol) --------------------
    configs = _build_threshold_configs(resolved)
    if len(configs) != B:
        raise ValueError(f"threshold grid has {len(configs)} configs but ensemble.B={B}")
    seeds = [seed + i for i in range(S)]

    extractor = PIPELINE_REGISTRY[pipeline](**(resolved.get("extractor_kwargs", {}) or {}))

    model = _load_model(resolved)

    # --- extraction ensemble ----------------------------------------------------
    runner = CircusEnsembleRunner(extractor)
    result = runner.run(model=model, task=task_cfg, configs=configs, seeds=seeds)

    # --- outputs (ARCHITECTURE.md §2 schemas) ------------------------------------
    schema.write_edges_parquet(str(run_dir / "edges.parquet"), result.records)

    cutoffs = dict(ensemble.get("decompose", {}))
    core_thr, noise_thr = cutoffs.get("core_threshold"), cutoffs.get("noise_threshold")
    bands_resolved = core_thr is not None and noise_thr is not None
    if bands_resolved:
        # noise_strict is part of the Q1 pre-registration, not a detail: it decides
        # whether s(e) = noise_threshold is contingent (CIRCUS §3.2) or noise. Dropping
        # it here would band every boundary edge against the pre-registered taxonomy
        # while the config still said 0.5.
        bands = decompose(
            result.freq.frequencies,
            float(core_thr),
            float(noise_thr),
            noise_strict=bool(cutoffs.get("noise_strict", False)),
        )
        result.freq.bands = bands
    freq_rows = [
        {"edge_id": eid, "s_e": s, "band": result.freq.band_for(eid)}
        for eid, s in result.freq.items_sorted()
    ]
    schema.write_freq_parquet(str(run_dir / "freq.parquet"), freq_rows)

    run_meta = {
        **tags,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "seeds": seeds,
        "n_cells": result.n_cells,
        "band_cutoffs_resolved": bands_resolved,
        "open_questions": [q["id"] for q in open_questions],
        "provisional_resolutions": provisional_resolutions,
        "outputs": {"edges.parquet": str(run_dir / "edges.parquet"), "freq.parquet": str(run_dir / "freq.parquet")},
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, sort_keys=True)

    print(
        f"[stageA] run_name={run_name} config_hash={config_hash} "
        f"B={B} S={S} seed={seed} edges={len(result.records)} unique_edges={len(result.freq.frequencies)} "
        f"-> {run_dir}"
    )
    return run_dir


def main() -> None:
    """Hydra entrypoint: ``python experiments/run_stage_a.py`` (from project root)."""
    if hydra_main is None:  # pragma: no cover
        print("hydra-core + omegaconf are required to run the entrypoint (optional dep for tests).", file=sys.stderr)
        sys.exit(1)

    @hydra_main(version_base=None, config_path="../configs", config_name="config")
    def _run(cfg: DictConfig) -> None:
        run_stage_a(cfg)

    _run()


if __name__ == "__main__":
    main()
