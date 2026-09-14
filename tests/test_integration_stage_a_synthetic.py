# [AI-GEN] agent=OpenCode date=2026-08-07 task=Integration test: Stage A with mock extractor (no model, no GPU)
# reviewed-by: PENDING

"""End-to-end Stage A with the synthetic extractor.

Asserts the Step 8 contract: run dir under run_root, resolved config + config hash
written, required tags present, edges.parquet + freq.parquet written with valid
schemas, no frozen/ output, no compression.
"""

import json

import pytest

from experiments.run_stage_a import run_stage_a
from src.common import schema
from src.common.hashing import hash_config


def _cfg(tmp_path, **overrides):
    cfg = {
        "stage": "stageA",
        "seed": 0,
        "run_root": str(tmp_path / "runs"),
        "pipeline": "mock",
        "setting": "dense",
        "comparison_level": "both",
        "compression_family": "dense",
        "compression_level": None,
        "model": {"name": "pythia160m"},
        "task": {"name": "ioi"},
        "ensemble": {
            "B": 3,
            "S": 2,
            "configset": "B3xS2",
            "threshold_grid": None,
            "decompose": {"core_threshold": None, "noise_threshold": None},
        },
    }
    cfg.update(overrides)
    return cfg


def test_stage_a_synthetic_produces_schema_valid_outputs(tmp_path):
    run_dir = run_stage_a(_cfg(tmp_path))

    assert run_dir.exists()
    for name in ("resolved_config.json", "run_meta.json", "edges.parquet", "freq.parquet"):
        assert (run_dir / name).exists(), f"missing {name}"

    edges = schema.read_edges_parquet(str(run_dir / "edges.parquet"))
    assert len(edges) > 0
    assert set(edges[0]) == set(schema.EDGES_COLUMNS)

    freq = schema.read_freq_parquet(str(run_dir / "freq.parquet"))
    assert set(freq[0]) == set(schema.FREQ_COLUMNS)
    assert all(0.0 <= row["s_e"] <= 1.0 for row in freq)
    # cutoffs unresolved -> band column is None (never invented)
    assert all(row["band"] is None for row in freq)


def test_stage_a_writes_config_hash_and_tags(tmp_path):
    cfg = _cfg(tmp_path)
    run_dir = run_stage_a(cfg)

    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert hash_config(resolved) == hash_config(cfg)

    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    for key in ("stage", "model", "task", "pipeline", "comparison_level", "B", "S", "seed", "config_hash", "run_name"):
        assert key in meta, f"missing tag {key}"
    assert meta["stage"] == "stageA"
    assert meta["B"] == 3 and meta["S"] == 2
    assert meta["config_hash"] == hash_config(cfg)
    assert meta["n_cells"] == 6  # B x S
    assert meta["band_cutoffs_resolved"] is False

    # run name follows CLAUDE.md §4
    from src.common.run_naming import validate_run_name

    parsed = validate_run_name(meta["run_name"])
    assert parsed["stage"] == "stageA" and parsed["model"] == "pythia160m" and parsed["task"] == "ioi"


def test_stage_a_never_touches_frozen(tmp_path):
    run_dir = run_stage_a(_cfg(tmp_path))
    assert not (run_dir.parent / "frozen").exists()
    assert "frozen" not in str(run_dir)


def test_stage_a_with_resolved_band_cutoffs(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["ensemble"]["decompose"] = {"core_threshold": 0.9, "noise_threshold": 0.1}
    run_dir = run_stage_a(cfg)
    freq = schema.read_freq_parquet(str(run_dir / "freq.parquet"))
    assert freq, "expected non-empty freq table"
    for row in freq:
        s, band = row["s_e"], row["band"]
        assert band in ("core", "contingent", "noise")
        assert (s >= 0.9) == (band == "core")
        assert (s <= 0.1) == (band == "noise")


def test_stage_a_refuses_to_overwrite_run(tmp_path):
    cfg = _cfg(tmp_path)
    run_stage_a(cfg)
    with pytest.raises(FileExistsError):
        run_stage_a(cfg)


def test_stage_a_refuses_unknown_pipeline(tmp_path):
    cfg = _cfg(tmp_path, pipeline="bogus")
    with pytest.raises(ValueError, match="unknown pipeline"):
        run_stage_a(cfg)
