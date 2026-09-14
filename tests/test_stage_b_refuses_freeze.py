# [AI-GEN] agent=OpenCode date=2026-08-07 task=Unit tests: Stage B refuses freeze in engineering mode; draft runs OK
# reviewed-by: PENDING

import json

import pytest

from experiments.run_stage_b import run_stage_b
from src.common.config_guard import MODE_ENGINEERING


def _cfg(tmp_path, **overrides):
    cfg = {
        "mode": {"name": MODE_ENGINEERING},
        "stage": "stageB",
        "seed": 0,
        "run_root": str(tmp_path / "runs"),
        "pipeline": "mock",
        "setting": "null-matchedmag",
        "comparison_level": "both",
        "compression_family": "null",
        "compression_level": "matchedmag-0.30",
        "model": {"name": "mock", "synthetic": True, "n_layers": 4, "n_heads": 2, "d_model": 8},
        "task": {"name": "synthetic-ioi", "family": "indirect-object-identification-synthetic",
                 "synthetic": True, "n_prompts": 8, "max_seq_len": 32, "seed": 0},
        "ensemble": {"B": 4, "S": 2, "configset": "B4xS2", "threshold_grid": None,
                     "pi_confirmed": False,
                     "decompose": {"core_threshold": 0.9, "noise_threshold": 0.1}},
        "distance": {"name": "l1", "alternative": "jensen_shannon"},
        "comparison": {"level2_scheme": None, "position_policy": None},
        "nulls": {"R": 3},
        "compression": {"name": "null"},
    }
    cfg.update(overrides)
    return cfg


def test_stage_b_refuses_freeze_approval_in_engineering_mode(tmp_path):
    cfg = _cfg(tmp_path, stage_b={"freeze_approved": True})
    with pytest.raises(ValueError, match="REFUSES the Stage B freeze"):
        run_stage_b(cfg)


def test_stage_b_refuses_freeze_flag_in_engineering_mode(tmp_path):
    cfg = _cfg(tmp_path, freeze=True)
    with pytest.raises(ValueError, match="REFUSES the Stage B freeze"):
        run_stage_b(cfg)


def test_stage_b_draft_run_produces_no_freeze(tmp_path):
    run_dir = run_stage_b(_cfg(tmp_path))

    assert (run_dir / "dnull.parquet").exists()
    assert (run_dir / "meta.json").exists()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["frozen"] is False
    assert meta["mode"] == MODE_ENGINEERING
    assert meta["R"] == 3
    assert len(meta["null_frozen_hash"]) == 64
    assert meta["frobenius_magnitudes"], "expected non-empty magnitudes from the dry-run pruner cell"

    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["frozen"] is False
    assert "DRAFT null distribution" in run_meta["note"]

    # never wrote into frozen/
    assert not (tmp_path / "frozen").exists()


def test_stage_b_draft_is_deterministic_and_immutable(tmp_path):
    run_dir = run_stage_b(_cfg(tmp_path))
    # runs are immutable: a second identical call must refuse to overwrite
    with pytest.raises(FileExistsError):
        run_stage_b(_cfg(tmp_path))
