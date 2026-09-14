# [AI-GEN] agent=Claude date=2026-08-08 task=Tests for the freeze writer (the pre-registration event had no implementation and no tests)
# reviewed-by: PENDING

"""The freeze is the project's central methodological commitment; these tests are the
enforcement (AI_RULES.md 1.4).

If any test in this file can be made to fail by a plausible sequence of operations,
the paper's claim that "the null was frozen before the effect was measured" is not
backed by anything.
"""

import json

import pytest

from src.common.freeze import (
    FreezeViolation,
    cell_key,
    freeze_cell,
    read_manifest,
    verify_frozen_store,
)
from src.common.hashing import hash_file
from src.common.stage_guard import assert_stage_b_frozen


def _draft(tmp_path, name="draft", payload=b"dnull-bytes"):
    """A Stage B draft run directory: dnull.parquet + meta.json with a matching hash."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "dnull.parquet").write_bytes(payload)
    meta = {
        "config_hash": "c" * 64,
        "seeds": [0, 1],
        "R": 5,
        "frobenius_magnitudes": {"L0.H0.W": 1.25},
        "null_frozen_hash": hash_file(d / "dnull.parquet"),
        "dnull_file": "dnull.parquet",
        "frozen": False,
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


class TestFreezingACell:
    def test_writes_cell_and_registers_it(self, tmp_path):
        draft = _draft(tmp_path)
        frozen_root = tmp_path / "frozen"
        meta = freeze_cell(frozen_root, "pythia-160m", "ioi", "magnitude-0.30", draft,
                           freeze_commit="abc123")

        cell = frozen_root / "pythia-160m" / "ioi" / "magnitude-0.30"
        assert (cell / "dnull.parquet").exists()
        assert (cell / "meta.json").exists()
        assert meta["frozen"] is True
        assert meta["freeze_commit"] == "abc123"

        manifest = read_manifest(frozen_root)
        assert manifest["frozen"] is True
        assert manifest["cells"]["pythia-160m/ioi/magnitude-0.30"]["null_frozen_hash"] == meta["null_frozen_hash"]

    def test_the_frozen_cell_satisfies_the_stage_c_guard(self, tmp_path):
        """The writer and the reader must agree, or Stage C can never run."""
        draft = _draft(tmp_path)
        frozen_root = tmp_path / "frozen"
        freeze_cell(frozen_root, "m", "t", "cell-1", draft)
        assert_stage_b_frozen(frozen_root / "m" / "t" / "cell-1")  # must not raise

    def test_records_the_source_run(self, tmp_path):
        draft = _draft(tmp_path)
        meta = freeze_cell(tmp_path / "frozen", "m", "t", "c", draft)
        assert str(draft) in meta["source_run_dir"]


class TestAppendOnly:
    def test_refuses_to_refreeze_an_existing_cell(self, tmp_path):
        """AI_RULES.md 1.4: a frozen null is never regenerated or corrected."""
        draft = _draft(tmp_path)
        frozen_root = tmp_path / "frozen"
        freeze_cell(frozen_root, "m", "t", "cell-1", draft)
        with pytest.raises(FreezeViolation, match="ALREADY frozen"):
            freeze_cell(frozen_root, "m", "t", "cell-1", draft)

    def test_a_second_different_draft_cannot_replace_the_first(self, tmp_path):
        """The dangerous case: re-running Stage B and quietly freezing the new null."""
        frozen_root = tmp_path / "frozen"
        first = _draft(tmp_path, "first", b"original-null")
        freeze_cell(frozen_root, "m", "t", "cell-1", first)
        original = hash_file(frozen_root / "m" / "t" / "cell-1" / "dnull.parquet")

        retuned = _draft(tmp_path, "retuned", b"re-tuned-null-that-fits-better")
        with pytest.raises(FreezeViolation, match="append-only"):
            freeze_cell(frozen_root, "m", "t", "cell-1", retuned)
        assert hash_file(frozen_root / "m" / "t" / "cell-1" / "dnull.parquet") == original

    def test_new_cells_may_be_appended(self, tmp_path):
        frozen_root = tmp_path / "frozen"
        freeze_cell(frozen_root, "m", "t", "cell-1", _draft(tmp_path, "a", b"aaa"))
        freeze_cell(frozen_root, "m", "t", "cell-2", _draft(tmp_path, "b", b"bbb"))
        assert set(read_manifest(frozen_root)["cells"]) == {"m/t/cell-1", "m/t/cell-2"}

    def test_appending_preserves_the_original_freeze_commit(self, tmp_path):
        frozen_root = tmp_path / "frozen"
        freeze_cell(frozen_root, "m", "t", "c1", _draft(tmp_path, "a", b"aaa"), freeze_commit="first")
        freeze_cell(frozen_root, "m", "t", "c2", _draft(tmp_path, "b", b"bbb"), freeze_commit="later")
        assert read_manifest(frozen_root)["freeze_commit"] == "first"


class TestIntegrityOnTheWayIn:
    def test_refuses_a_draft_whose_hash_no_longer_matches(self, tmp_path):
        """A draft edited after Stage B wrote it must never become the floor."""
        draft = _draft(tmp_path)
        (draft / "dnull.parquet").write_bytes(b"tampered-after-the-fact")
        with pytest.raises(FreezeViolation, match="modified after it was written"):
            freeze_cell(tmp_path / "frozen", "m", "t", "c", draft)

    def test_refuses_an_incomplete_draft(self, tmp_path):
        draft = tmp_path / "incomplete"
        draft.mkdir()
        (draft / "meta.json").write_text("{}", encoding="utf-8")
        with pytest.raises(FreezeViolation, match="missing"):
            freeze_cell(tmp_path / "frozen", "m", "t", "c", draft)

    def test_refuses_a_draft_with_no_hash_recorded(self, tmp_path):
        draft = _draft(tmp_path)
        meta = json.loads((draft / "meta.json").read_text(encoding="utf-8"))
        del meta["null_frozen_hash"]
        (draft / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="null_frozen_hash"):
            freeze_cell(tmp_path / "frozen", "m", "t", "c", draft)

    def test_a_partial_cell_is_not_left_behind_on_failure(self, tmp_path):
        draft = _draft(tmp_path)
        (draft / "dnull.parquet").write_bytes(b"changed")
        frozen_root = tmp_path / "frozen"
        with pytest.raises(FreezeViolation):
            freeze_cell(frozen_root, "m", "t", "c", draft)
        assert not (frozen_root / "m" / "t" / "c").exists()


class TestStoreVerification:
    def test_clean_store_verifies(self, tmp_path):
        frozen_root = tmp_path / "frozen"
        freeze_cell(frozen_root, "m", "t", "c1", _draft(tmp_path, "a", b"aaa"))
        freeze_cell(frozen_root, "m", "t", "c2", _draft(tmp_path, "b", b"bbb"))
        result = verify_frozen_store(frozen_root)
        assert result["ok"] is True and result["n_cells"] == 2 and result["problems"] == []

    def test_detects_a_tampered_frozen_null(self, tmp_path):
        """The scenario the whole mechanism exists to catch."""
        frozen_root = tmp_path / "frozen"
        freeze_cell(frozen_root, "m", "t", "c1", _draft(tmp_path, "a", b"aaa"))
        (frozen_root / "m" / "t" / "c1" / "dnull.parquet").write_bytes(b"a-friendlier-null")
        result = verify_frozen_store(frozen_root)
        assert result["ok"] is False
        assert any("hash" in p for p in result["problems"])

    def test_detects_an_unregistered_cell_on_disk(self, tmp_path):
        frozen_root = tmp_path / "frozen"
        freeze_cell(frozen_root, "m", "t", "c1", _draft(tmp_path, "a", b"aaa"))
        rogue = frozen_root / "m" / "t" / "smuggled"
        rogue.mkdir(parents=True)
        (rogue / "dnull.parquet").write_bytes(b"xxx")
        (rogue / "meta.json").write_text(json.dumps({"null_frozen_hash": "z" * 64}), encoding="utf-8")
        result = verify_frozen_store(frozen_root)
        assert result["ok"] is False
        assert any("NOT registered" in p for p in result["problems"])

    def test_detects_a_registered_cell_with_missing_files(self, tmp_path):
        import shutil

        frozen_root = tmp_path / "frozen"
        freeze_cell(frozen_root, "m", "t", "c1", _draft(tmp_path, "a", b"aaa"))
        shutil.rmtree(frozen_root / "m" / "t" / "c1")
        result = verify_frozen_store(frozen_root)
        assert result["ok"] is False
        assert any("missing" in p for p in result["problems"])

    def test_empty_store_is_vacuously_ok(self, tmp_path):
        assert verify_frozen_store(tmp_path / "nothing-here")["ok"] is True


class TestManifest:
    def test_absent_manifest_reads_as_unfrozen(self, tmp_path):
        m = read_manifest(tmp_path / "frozen")
        assert m["frozen"] is False and m["cells"] == {}

    def test_cell_key_format(self):
        assert cell_key("gemma-2-2b", "ioi", "rtn-int4") == "gemma-2-2b/ioi/rtn-int4"


class TestFreezeEntrypointPolicy:
    """experiments/freeze_stage_b.py owns WHEN to freeze; these are its refusals."""

    def _cfg(self, tmp_path, **over):
        cfg = {
            "mode": {"name": "scientific_run"},
            "frozen_root": str(tmp_path / "frozen"),
            "pipeline": "attr",
            "model": {"name": "pythia-160m", "hf_revision": "abc"},
            "task": {"name": "ioi", "pi_confirmed": True},
            "ensemble": {"B": 4, "S": 2, "pi_confirmed": True,
                         "threshold_grid": [{"id": "g0", "node_threshold": 0.8, "edge_threshold": 0.98}],
                         "decompose": {"core_threshold": 0.9, "noise_threshold": 0.1}},
            "distance": {"name": "l1", "pre_registered_for_stage_c": True},
            "comparison": {"level2_scheme": "layer", "position_policy": "aggregate"},
            "stage_b": {"freeze_approved": True,
                        "cells": [{"model": "pythia-160m", "task": "ioi", "cell": "magnitude-0.30",
                                   "source_run_dir": str(_draft(tmp_path, "src", b"nnn"))}]},
        }
        cfg.update(over)
        return cfg

    def test_refuses_in_engineering_mode(self, tmp_path):
        from experiments.freeze_stage_b import run_freeze

        cfg = self._cfg(tmp_path, mode={"name": "engineering_dry_run"})
        with pytest.raises(FreezeViolation, match="mode=scientific_run"):
            run_freeze(cfg)

    def test_refuses_without_explicit_approval(self, tmp_path):
        from experiments.freeze_stage_b import run_freeze

        cfg = self._cfg(tmp_path)
        cfg["stage_b"] = {**cfg["stage_b"], "freeze_approved": False}
        with pytest.raises(FreezeViolation, match="EXPLICIT approval"):
            run_freeze(cfg)

    def test_refuses_while_a_gating_decision_is_open(self, tmp_path):
        from experiments.freeze_stage_b import run_freeze

        cfg = self._cfg(tmp_path)
        cfg["model"] = {**cfg["model"], "hf_revision": None}  # Q5 reopened
        with pytest.raises(ValueError, match="Q5"):
            run_freeze(cfg)

    def test_refuses_while_the_distance_is_not_pre_registered(self, tmp_path):
        """Q2: a floor frozen before D is chosen is not a floor for anything."""
        from experiments.freeze_stage_b import run_freeze

        cfg = self._cfg(tmp_path)
        cfg["distance"] = {"name": "l1", "pre_registered_for_stage_c": False}
        with pytest.raises(FreezeViolation, match="not pre-registered"):
            run_freeze(cfg)

    def test_refuses_with_no_cells_listed(self, tmp_path):
        from experiments.freeze_stage_b import run_freeze

        cfg = self._cfg(tmp_path)
        cfg["stage_b"] = {**cfg["stage_b"], "cells": []}
        with pytest.raises(FreezeViolation, match="nothing to freeze"):
            run_freeze(cfg)

    def test_dry_run_writes_nothing(self, tmp_path):
        from experiments.freeze_stage_b import run_freeze

        report = run_freeze(self._cfg(tmp_path), dry_run=True)
        assert report["dry_run"] is True
        assert len(report["would_freeze"]) == 1
        assert not (tmp_path / "frozen" / "pythia-160m").exists()

    def test_dry_run_reports_blocking_problems_instead_of_raising(self, tmp_path):
        from experiments.freeze_stage_b import run_freeze

        cfg = self._cfg(tmp_path, mode={"name": "engineering_dry_run"})
        report = run_freeze(cfg, dry_run=True)
        assert report["blocking_problems"]

    def test_a_fully_approved_config_freezes(self, tmp_path):
        from experiments.freeze_stage_b import run_freeze

        report = run_freeze(self._cfg(tmp_path))
        assert report["dry_run"] is False
        assert len(report["frozen_cells"]) == 1
        assert report["store_verification"]["ok"] is True
        assert_stage_b_frozen(tmp_path / "frozen" / "pythia-160m" / "ioi" / "magnitude-0.30")
