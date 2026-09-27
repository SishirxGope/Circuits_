# [AI-GEN] agent=Claude date=2026-09-27 task=Regression tests: the null must be matched to the cell it is the denominator for
# reviewed-by: PENDING

"""The null/compression pairing (ARCHITECTURE.md §4; AI_RULES.md 1.4, 1.5).

CSI(c) = D(c) / median(D_null(c)) is only meaningful if D_null(c) was drawn against the
magnitudes of compression *c*. Until 2026-09-27 Stage B hardcoded
``MagnitudePruner(sparsity=0.3)`` for every cell while faithfully recording the cell's
real family in its run tags, so all 88 Stage B cells in
``deploy/plan_b_dgx_spark/cells_stageb.txt`` — rtn_int4 through wanda_60 — would have
shared one denominator drawn against a sparsity that is not even in the grid. The
existing ``null_frozen_hash`` check could not catch it: the wrong null hashes perfectly.

These tests pin both halves of the fix — Stage B drawing from the cell's own compressor,
and Stage C refusing a null that was matched to something else.
"""

from __future__ import annotations

import json

import pytest

from experiments.run_stage_b import run_stage_b
from src.common.config_guard import MODE_ENGINEERING
from src.common.stage_guard import assert_null_matched_to_cell
from src.compression.registry import compression_provenance, compressor_for


def _cfg(tmp_path, **overrides):
    cfg = {
        "mode": {"name": MODE_ENGINEERING},
        "stage": "stageB",
        "seed": 0,
        "run_root": str(tmp_path / "runs"),
        "pipeline": "mock",
        "setting": "null-matchedmag",
        "comparison_level": "both",
        "compression_family": "magnitude",
        "compression_level": "0.30",
        "stage_c": {"cell": "magnitude-0.30", "compressor_kwargs": {"sparsity": 0.3}},
        "model": {"name": "mock", "synthetic": True, "n_layers": 4, "n_heads": 2, "d_model": 8},
        "task": {"name": "synthetic-ioi", "family": "indirect-object-identification-synthetic",
                 "synthetic": True, "n_prompts": 8, "max_seq_len": 32, "seed": 0},
        "ensemble": {"B": 4, "S": 2, "configset": "B4xS2", "threshold_grid": None,
                     "pi_confirmed": False,
                     "decompose": {"core_threshold": 0.9, "noise_threshold": 0.1}},
        "distance": {"name": "l1", "alternative": "jensen_shannon"},
        "comparison": {"level2_scheme": None, "position_policy": None},
        "nulls": {"R": 3},
        "compression": {"name": "magnitude_30"},
    }
    cfg.update(overrides)
    return cfg


def _meta(run_dir):
    with open(run_dir / "meta.json", encoding="utf-8") as f:
        return json.load(f)


class TestStageBDrawsFromTheCellsOwnCompressor:
    def test_different_cells_get_different_magnitudes(self, tmp_path):
        """The regression itself: these two used to produce identical magnitudes."""
        rtn = _meta(run_stage_b(_cfg(
            tmp_path / "a", setting="null-rtn4", compression_family="rtn", compression_level="4",
            stage_c={"cell": "rtn_int4", "compressor_kwargs": {"bits": 4}},
        )))["frobenius_magnitudes"]
        mag = _meta(run_stage_b(_cfg(
            tmp_path / "b", setting="null-mag60", compression_family="magnitude",
            compression_level="0.60",
            stage_c={"cell": "magnitude-0.60", "compressor_kwargs": {"sparsity": 0.6}},
        )))["frobenius_magnitudes"]

        assert rtn != mag, (
            "an rtn_int4 null and a magnitude-0.60 null must not share magnitudes; "
            "if they do, Stage B is ignoring the cell again"
        )

    def test_the_level_within_a_family_changes_the_magnitudes(self, tmp_path):
        """Heavier compression must move the null further, or levels are indistinguishable."""
        light = sum(_meta(run_stage_b(_cfg(
            tmp_path / "int8", setting="null-rtn8", compression_family="rtn",
            compression_level="8", stage_c={"cell": "rtn_int8", "compressor_kwargs": {"bits": 8}},
        )))["frobenius_magnitudes"].values())
        heavy = sum(_meta(run_stage_b(_cfg(
            tmp_path / "int4", setting="null-rtn4", compression_family="rtn",
            compression_level="4", stage_c={"cell": "rtn_int4", "compressor_kwargs": {"bits": 4}},
        )))["frobenius_magnitudes"].values())
        assert heavy > light

    def test_the_frozen_meta_records_which_compression_it_matched(self, tmp_path):
        meta = _meta(run_stage_b(_cfg(tmp_path)))
        assert meta["compression_family"] == "magnitude"
        assert meta["compressor_kwargs"] == {"sparsity": 0.3}

    @pytest.mark.parametrize("family", ["dense", "null", "none", ""])
    def test_a_cell_that_names_no_compression_is_refused(self, tmp_path, family):
        """A null matched to "no compression" would be a spike at zero — CSI's denominator."""
        with pytest.raises(ValueError, match="names no compression"):
            run_stage_b(_cfg(tmp_path, compression_family=family))

    def test_an_unknown_family_is_refused_rather_than_guessed(self, tmp_path):
        with pytest.raises(ValueError, match="unknown compression_family"):
            run_stage_b(_cfg(tmp_path, compression_family="int3_magic"))

    def test_kwargs_the_family_cannot_accept_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="does not accept"):
            run_stage_b(_cfg(
                tmp_path, compression_family="rtn",
                stage_c={"cell": "rtn_int4", "compressor_kwargs": {"sparsity": 0.3}},
            ))

    def test_a_bad_cell_leaves_no_half_built_run_behind(self, tmp_path):
        """Validated before the run dir is allocated (run immutability, AI_RULES.md 1.2)."""
        with pytest.raises(ValueError):
            run_stage_b(_cfg(tmp_path, compression_family="int3_magic"))
        runs = tmp_path / "runs"
        assert not runs.exists() or not any(runs.iterdir())


class TestBothStagesBuildTheSameCompressor:
    """The drift this registry exists to prevent."""

    def test_stage_b_and_stage_c_agree_for_one_config(self, tmp_path):
        cfg = _cfg(tmp_path, compression_family="rtn",
                   stage_c={"cell": "rtn_int8", "compressor_kwargs": {"bits": 8}})
        b = compressor_for(cfg, stage="stageB")
        c = compressor_for(cfg, stage="stageC")
        assert type(b) is type(c)
        assert b.bits == c.bits == 8

    def test_kwargs_come_from_the_key_gen_cells_actually_emits(self, tmp_path):
        """gen_cells.py emits +stage_c.compressor_kwargs.bits=8 for Stage B AND Stage C.

        Reading the identically-named block in configs/compression/*.yaml instead would
        silently fall back to the default, quantizing every rtn_int8 cell at 4 bits.
        """
        cfg = _cfg(tmp_path, compression_family="rtn",
                   compression={"name": "rtn_int8", "compressor_kwargs": {"bits": 8}},
                   stage_c={"cell": "rtn_int8", "compressor_kwargs": {"bits": 6}})
        assert compressor_for(cfg, stage="stageB").bits == 6


class TestStageCRefusesAMismatchedNull:
    """``null_frozen_hash`` proves the null is intact; this proves it is the right one."""

    def _frozen(self, tmp_path, **overrides):
        meta = {"config_hash": "x" * 64, "seeds": [0], "R": 3, "null_frozen_hash": "y" * 64,
                "compression_family": "rtn", "compression_level": 8,
                "compressor_kwargs": {"bits": 8}}
        meta.update(overrides)
        path = tmp_path / "meta.json"
        path.write_text(json.dumps(meta), encoding="utf-8")
        return path

    def _want(self, **overrides):
        base = {"compression_family": "rtn", "compression_level": 8,
                "compressor_kwargs": {"bits": 8}}
        base.update(overrides)
        return base

    def test_a_matching_null_passes(self, tmp_path):
        assert_null_matched_to_cell(self._frozen(tmp_path), self._want())

    def test_hydra_string_coercion_is_not_a_spurious_mismatch(self, tmp_path):
        """+stage_c.compressor_kwargs.bits=8 arrives as "8" on some paths, 8 on others."""
        assert_null_matched_to_cell(
            self._frozen(tmp_path, compression_level="8", compressor_kwargs={"bits": "8"}),
            self._want(),
        )

    def test_a_different_family_is_refused(self, tmp_path):
        with pytest.raises(RuntimeError, match="matched to a different compression"):
            assert_null_matched_to_cell(
                self._frozen(tmp_path), self._want(compression_family="magnitude")
            )

    def test_a_different_level_within_the_family_is_refused(self, tmp_path):
        """rtn_int8's null is not rtn_int4's null, though both are rtn."""
        with pytest.raises(RuntimeError, match="matched to a different compression"):
            assert_null_matched_to_cell(
                self._frozen(tmp_path),
                self._want(compression_level=4, compressor_kwargs={"bits": 4}),
            )

    def test_a_null_predating_provenance_is_refused_not_trusted(self, tmp_path):
        """Every null written before the fix was matched to the hardcoded stand-in."""
        meta = {"config_hash": "x" * 64, "seeds": [0], "R": 3, "null_frozen_hash": "y" * 64}
        path = tmp_path / "meta.json"
        path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(RuntimeError, match="records no compression provenance"):
            assert_null_matched_to_cell(path, self._want())

    def test_the_guard_reads_what_stage_b_wrote(self, tmp_path):
        """End to end: Stage B's own meta.json satisfies the Stage C guard for its cell."""
        cfg = _cfg(tmp_path, compression_family="rtn", compression_level="4",
                   stage_c={"cell": "rtn_int4", "compressor_kwargs": {"bits": 4}})
        run_dir = run_stage_b(cfg)
        assert_null_matched_to_cell(run_dir / "meta.json", compression_provenance(cfg))

        with pytest.raises(RuntimeError, match="matched to a different compression"):
            assert_null_matched_to_cell(
                run_dir / "meta.json",
                compression_provenance(_cfg(
                    tmp_path, compression_family="rtn", compression_level="8",
                    stage_c={"cell": "rtn_int8", "compressor_kwargs": {"bits": 8}},
                )),
            )
