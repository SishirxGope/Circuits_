# [AI-GEN] agent=OpenCode date=2026-08-07 task=Integration test: Stage C refuses to run without frozen Stage B (AI_RULES.md 1.4/1.5)
# reviewed-by: PENDING

"""Stage-ordering guard tests (ARCHITECTURE.md §1 dotted edge; AI_RULES.md 1.4/1.5).

The ordering guard is TESTED, not trusted (ARCHITECTURE.md §6): Stage C must refuse
to run when Stage B artifacts are missing, unregistered, or hash-mismatched, and
pass only when the frozen hash matches.
"""

import json

import pytest

from src.common.stage_guard import assert_null_frozen_hash, assert_stage_b_frozen

FROZEN_HASH = "a" * 64


def _write_frozen_cell(root, cell_rel="pythia160m/ioi/cell-mag-30", hash_=FROZEN_HASH, manifest_hash=None, with_manifest=True):
    cell = root / cell_rel
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "dnull.parquet").write_bytes(b"placeholder-frozen-dnull")
    (cell / "meta.json").write_text(
        json.dumps(
            {
                "config_hash": "b" * 64,
                "seeds": [0, 1, 2],
                "R": 20,
                "frobenius_magnitudes": {"W1": 1.23},
                "timestamp": "2026-08-07T00:00:00Z",
                "git_commit": "deadbeef",
                "null_frozen_hash": hash_,
                "dnull_file": "dnull.parquet",
            }
        ),
        encoding="utf-8",
    )
    if with_manifest:
        entry_hash = hash_ if manifest_hash is None else manifest_hash
        (root / "FREEZE_MANIFEST.json").write_text(
            json.dumps(
                {
                    "freeze_commit": "deadbeef",
                    "frozen_at": "2026-08-07T00:00:00Z",
                    "cells": {cell_rel: {"null_frozen_hash": entry_hash, "sha256": "c" * 64}},
                }
            ),
            encoding="utf-8",
        )
    return cell


def test_refuses_when_artifacts_missing(tmp_path):
    cell = _write_frozen_cell(tmp_path, with_manifest=True)
    (cell / "dnull.parquet").unlink()
    with pytest.raises(RuntimeError, match="not frozen"):
        assert_stage_b_frozen(cell)


def test_refuses_when_meta_missing_hash(tmp_path):
    cell = _write_frozen_cell(tmp_path, with_manifest=True)
    meta = json.loads((cell / "meta.json").read_text(encoding="utf-8"))
    del meta["null_frozen_hash"]
    (cell / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="null_frozen_hash"):
        assert_stage_b_frozen(cell)


def test_refuses_when_manifest_missing(tmp_path):
    cell = _write_frozen_cell(tmp_path, with_manifest=False)
    with pytest.raises(RuntimeError, match="FREEZE_MANIFEST"):
        assert_stage_b_frozen(cell)


def test_refuses_when_cell_unregistered(tmp_path):
    _write_frozen_cell(tmp_path, cell_rel="pythia160m/ioi/cell-a", with_manifest=True)
    cell_b = _write_frozen_cell(tmp_path, cell_rel="pythia160m/ioi/cell-b", with_manifest=False)
    with pytest.raises(RuntimeError, match="not registered"):
        assert_stage_b_frozen(cell_b)
    # register cell-b, then it passes
    manifest = tmp_path / "FREEZE_MANIFEST.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["cells"]["pythia160m/ioi/cell-b"] = {"null_frozen_hash": FROZEN_HASH}
    manifest.write_text(json.dumps(data), encoding="utf-8")
    assert_stage_b_frozen(cell_b)  # now passes


def test_refuses_on_manifest_hash_mismatch(tmp_path):
    cell = _write_frozen_cell(tmp_path, hash_="x" * 64, manifest_hash=FROZEN_HASH, with_manifest=True)
    # meta says "x"*64, manifest says FROZEN_HASH -> mismatch
    with pytest.raises(RuntimeError, match="hash mismatch"):
        assert_stage_b_frozen(cell)


def test_passes_when_frozen_and_matching(tmp_path):
    cell = _write_frozen_cell(tmp_path)
    assert_stage_b_frozen(cell)  # must not raise
    assert_null_frozen_hash(cell / "meta.json", FROZEN_HASH)  # must not raise


def test_null_hash_mismatch_fails_hard(tmp_path):
    cell = _write_frozen_cell(tmp_path)
    with pytest.raises(RuntimeError, match="null_frozen_hash mismatch"):
        assert_null_frozen_hash(cell / "meta.json", "f" * 64)
