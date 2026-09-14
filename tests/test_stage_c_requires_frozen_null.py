# [AI-GEN] agent=OpenCode date=2026-08-07 task=Unit tests: Stage C refuses without frozen Stage B artifacts
# reviewed-by: PENDING

import pytest

from experiments.run_stage_c import frozen_cell_path, run_stage_c
from src.common.config_guard import MODE_ENGINEERING, MODE_SCIENTIFIC


def _cfg(tmp_path, **overrides):
    cfg = {
        "mode": {"name": MODE_ENGINEERING},
        "stage": "stageC",
        "seed": 0,
        "run_root": str(tmp_path / "runs"),
        "frozen_root": str(tmp_path / "frozen"),
        "pipeline": "mock",
        "comparison_level": "both",
        "compression_family": "magnitude",
        "compression_level": "0.30",
        "model": {"name": "pythia160m"},
        "task": {"name": "ioi"},
        "ensemble": {"B": 4, "S": 2, "configset": "B4xS2", "threshold_grid": None,
                     "pi_confirmed": False,
                     "decompose": {"core_threshold": None, "noise_threshold": None}},
        "distance": {"name": None},
        "comparison": {"level2_scheme": None, "position_policy": None},
        "nulls": {"R": 3},
        "null_frozen_hash": "f" * 64,
        "compression": {"name": "null"},
    }
    cfg.update(overrides)
    return cfg


def test_refuses_without_frozen_artifacts(tmp_path):
    with pytest.raises(RuntimeError, match="not frozen"):
        run_stage_c(_cfg(tmp_path))


def test_refuses_when_null_frozen_hash_tag_missing(tmp_path):
    cfg = _cfg(tmp_path)
    del cfg["null_frozen_hash"]
    with pytest.raises(ValueError, match="null_frozen_hash"):
        run_stage_c(cfg)


def test_refuses_when_hash_mismatch_with_frozen_cell(tmp_path):
    # a frozen store exists but the declared hash does not match the cell's
    cell = _write_frozen_cell(tmp_path, "pythia160m/ioi/magnitude-0.30", hash_="a" * 64)
    cfg = _cfg(tmp_path, null_frozen_hash="b" * 64)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        run_stage_c(cfg)


def test_scientific_mode_also_refuses_without_frozen_artifacts(tmp_path):
    cfg = _cfg(tmp_path, mode={"name": MODE_SCIENTIFIC}, pipeline="attr")
    # the frozen-null guard fires before anything else
    with pytest.raises(RuntimeError, match="not frozen"):
        run_stage_c(cfg)


def test_frozen_cell_path_layout(tmp_path):
    cfg = _cfg(tmp_path)
    path = frozen_cell_path(cfg)
    assert path.parts[-4:] == ("frozen", "pythia160m", "ioi", "magnitude-0.30")


def _write_frozen_cell(root, cell_rel, hash_):
    import json

    cell = root / "frozen" / cell_rel
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "dnull.parquet").write_bytes(b"frozen")
    (cell / "meta.json").write_text(
        json.dumps({"config_hash": "c" * 64, "seeds": [0], "R": 3,
                    "null_frozen_hash": hash_, "dnull_file": "dnull.parquet"}),
        encoding="utf-8",
    )
    (root / "frozen" / "FREEZE_MANIFEST.json").write_text(
        json.dumps({"cells": {cell_rel: {"null_frozen_hash": hash_}}}), encoding="utf-8"
    )
    return cell
