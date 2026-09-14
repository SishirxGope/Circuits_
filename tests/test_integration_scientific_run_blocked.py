# [AI-GEN] agent=OpenCode date=2026-08-07 task=Integration: scientific_run mode remains blocked while finals are OPEN
# reviewed-by: PENDING

"""Scientific mode strictness (configs/mode/scientific_run.yaml).

Asserts: synthetic/mock components are refused outright; OPEN final decisions block;
a fully-resolved scientific config passes the guard and then fails ONLY at the
real-model boundary (NotImplementedError) — i.e., the guard never invents values.
"""

import pytest

from experiments.run_stage_a import run_stage_a
from src.common.config_guard import MODE_SCIENTIFIC


def _sci_cfg(tmp_path, **overrides):
    cfg = {
        "mode": {"name": MODE_SCIENTIFIC},
        "stage": "stageA",
        "seed": 0,
        "run_root": str(tmp_path / "runs"),
        "pipeline": "attr",
        "setting": "dense",
        "comparison_level": "both",
        "compression_family": "dense",
        "compression_level": None,
        "model": {"name": "pythia160m", "hf_revision": None},
        "task": {"name": "ioi", "pi_confirmed": False},
        "ensemble": {"B": 4, "S": 2, "configset": "B4xS2", "threshold_grid": None,
                     "pi_confirmed": False,
                     "decompose": {"core_threshold": None, "noise_threshold": None}},
        "distance": {"name": None},
        "comparison": {"level2_scheme": None, "position_policy": None},
        "nulls": {"R": 3},
        "compression": {"name": "null"},
    }
    cfg.update(overrides)
    return cfg


def test_scientific_refuses_synthetic_components(tmp_path):
    cfg = _sci_cfg(tmp_path, pipeline="mock", model={"name": "mock", "synthetic": True})
    with pytest.raises(ValueError, match="synthetic"):
        run_stage_a(cfg)


def test_scientific_blocked_while_finals_open(tmp_path):
    cfg = _sci_cfg(tmp_path)
    with pytest.raises(ValueError) as exc:
        run_stage_a(cfg)
    message = str(exc.value)
    assert "scientific_run blocked" in message
    for qid in ("Q1", "Q3", "Q4", "Q5", "Q6"):
        assert qid in message


def test_scientific_blocks_on_partially_resolved(tmp_path):
    # Q5/Q6 still open -> still blocked
    cfg = _sci_cfg(
        tmp_path,
        ensemble={"B": 4, "S": 2, "configset": "B4xS2",
                  "threshold_grid": [{"id": "c1", "node_threshold": 0.8, "edge_threshold": 0.98}],
                  "pi_confirmed": True,
                  "decompose": {"core_threshold": 0.9, "noise_threshold": 0.1}},
        distance={"name": "l1"},
    )
    with pytest.raises(ValueError, match="Q5"):
        run_stage_a(cfg)
    with pytest.raises(ValueError, match="Q6"):
        run_stage_a(cfg)


def test_scientific_fully_resolved_reaches_only_the_model_boundary(tmp_path):
    cfg = _sci_cfg(
        tmp_path,
        model={"name": "pythia160m", "hf_revision": "abc123"},
        task={"name": "ioi", "pi_confirmed": True},
        ensemble={"B": 4, "S": 2, "configset": "B4xS2",
                  # Anti-correlated: node RISES as edge FALLS, so no config dominates
                  # another. The original fixture moved both axes down together, which
                  # is a fully NESTED chain — the artifact CIRCUS §3.2 exists to
                  # prevent, and run_stage_a now refuses it (caught 2026-09-12).
                  "threshold_grid": [
                      {"id": f"c{i}", "node_threshold": 0.65 + 0.05 * i, "edge_threshold": 0.98 - 0.01 * i}
                      for i in range(4)],
                  "pi_confirmed": True,
                  "decompose": {"core_threshold": 0.9, "noise_threshold": 0.1}},
        distance={"name": "l1"},
        comparison={"level2_scheme": "layer", "position_policy": "aggregate"},
    )
    # the config guard passes; only the real-model boundary refuses (no download)
    with pytest.raises(NotImplementedError, match="RUN MODEL DOWNLOAD"):
        run_stage_a(cfg)
