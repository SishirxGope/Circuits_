# [AI-GEN] agent=Claude date=2026-08-08 task=Full Stage A->B->C chain integration test (ARCHITECTURE.md §6 required coverage that was missing)
# reviewed-by: PENDING

"""Full Stage A -> B -> C chain on the synthetic stack (ARCHITECTURE.md §6).

ARCHITECTURE.md §6 requires: "full Stage A->C pipeline ... with tiny B/S/R,
asserting output schemas, tag completeness, and that Stage C refuses to run without
frozen Stage B (the ordering guard is TESTED, not trusted)."

The existing suite tested each stage in isolation. This file tests the CHAIN: the
Stage A dense reference feeds Stage B, Stage B's draft null produces a real hash,
and Stage C's guard accepts exactly that hash and nothing else. It also exercises
the arithmetic that will run for real at Stage C — distances, CSI, and the two-level
comparison — over genuine Stage A/B ensemble output rather than hand-written vectors.

Nothing here is scientific evidence: the model, the task and the extractor are all
synthetic (AI_RULES.md 2.3).
"""

import json
from pathlib import Path

import pytest

from experiments.run_stage_a import run_stage_a
from experiments.run_stage_b import run_stage_b
from experiments.run_stage_c import frozen_cell_path, run_stage_c
from src.common import schema
from src.common.config_guard import MODE_ENGINEERING
from src.common.csi_table import read_csi_table
from src.science.csi import csi
from src.science.distances import jensen_shannon_distance, l1_distance
from src.science.two_level import compare_at_both_levels


def _reject_constant(name):  # pragma: no cover - only called on malformed output
    raise AssertionError(f"run_meta.json contains the non-standard JSON constant {name!r}")

B, S, R = 4, 2, 3
CELL = "magnitude-0.30"

_MODEL = {"name": "mock", "synthetic": True, "n_layers": 4, "n_heads": 2, "d_model": 8, "hf_revision": None}
_TASK = {"name": "synthetic-ioi", "family": "indirect-object-identification-synthetic",
         "synthetic": True, "n_prompts": 24, "max_seq_len": 32, "seed": 0}
_ENSEMBLE = {"B": B, "S": S, "configset": f"B{B}xS{S}", "threshold_grid": None,
             "pi_confirmed": False,
             "decompose": {"core_threshold": 0.90, "noise_threshold": 0.10}}


def _base(tmp_path, stage, **overrides):
    cfg = {
        "mode": {"name": MODE_ENGINEERING},
        "stage": stage,
        "seed": 0,
        "run_root": str(tmp_path / "runs"),
        "pipeline": "mock",
        "comparison_level": "both",
        "model": dict(_MODEL),
        "task": dict(_TASK),
        "ensemble": {**_ENSEMBLE, "decompose": dict(_ENSEMBLE["decompose"])},
        "distance": {"name": "l1", "alternative": "jensen_shannon"},
        "comparison": {"level2_scheme": None, "position_policy": None},
        "nulls": {"R": R},
        "compression": {"name": "null"},
    }
    cfg.update(overrides)
    return cfg


def _stage_a(tmp_path):
    return run_stage_a(_base(tmp_path, "stageA", setting="dense",
                             compression_family="dense", compression_level=None))


def _stage_b(tmp_path):
    return run_stage_b(_base(tmp_path, "stageB", setting="null-matchedmag",
                             compression_family="null", compression_level="matchedmag-0.30",
                             stage_b={"sparsity": 0.30, "global_scope": True}))


def _freq(run_dir: Path) -> dict[str, float]:
    return {r["edge_id"]: r["s_e"] for r in schema.read_freq_parquet(str(run_dir / "freq.parquet"))}


class TestStageAOutputs:
    def test_schemas_and_tags_are_complete(self, tmp_path):
        run_dir = _stage_a(tmp_path)

        edges = schema.read_edges_parquet(str(run_dir / "edges.parquet"))
        freq = schema.read_freq_parquet(str(run_dir / "freq.parquet"))
        schema.validate_edges_records(edges)
        schema.validate_freq_records(freq)

        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        schema.validate_run_tags(meta, stage="stageA")
        assert meta["n_cells"] == B * S
        assert meta["band_cutoffs_resolved"] is True

    def test_inclusion_frequencies_are_valid_probabilities(self, tmp_path):
        for s_e in _freq(_stage_a(tmp_path)).values():
            assert 0.0 < s_e <= 1.0

    def test_every_edge_carries_a_band(self, tmp_path):
        run_dir = _stage_a(tmp_path)
        bands = {r["band"] for r in schema.read_freq_parquet(str(run_dir / "freq.parquet"))}
        assert bands <= {"core", "contingent", "noise"}
        assert None not in bands, "band cutoffs were resolved, so no edge may be unlabelled"

    def test_stage_a_never_creates_a_frozen_store(self, tmp_path):
        _stage_a(tmp_path)
        assert not (tmp_path / "frozen").exists()


class TestStageBFollowsStageA:
    def test_draft_null_has_R_draws_and_a_real_hash(self, tmp_path):
        run_dir = _stage_b(tmp_path)
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        schema.validate_frozen_meta(meta)
        assert meta["R"] == R
        assert meta["frozen"] is False
        assert len(meta["null_frozen_hash"]) == 64

        import pyarrow.parquet as pq

        dnull = pq.read_table(str(run_dir / "dnull.parquet")).to_pylist()
        assert len(dnull) == R
        assert all(row["distance_l1"] >= 0.0 for row in dnull)

    def test_magnitudes_come_from_the_compressor_not_the_null(self, tmp_path):
        """ARCHITECTURE.md §4: Compressor.weight_delta is the single source of truth."""
        meta = json.loads((_stage_b(tmp_path) / "meta.json").read_text(encoding="utf-8"))
        mags = meta["frobenius_magnitudes"]
        assert mags and all(v >= 0.0 for v in mags.values())
        assert any(v > 0.0 for v in mags.values()), "a 30% prune must move some tensor"

    def test_draft_null_is_deterministic_across_run_roots(self, tmp_path):
        a = json.loads((_stage_b(tmp_path / "one") / "meta.json").read_text(encoding="utf-8"))
        b = json.loads((_stage_b(tmp_path / "two") / "meta.json").read_text(encoding="utf-8"))
        assert a["null_frozen_hash"] == b["null_frozen_hash"], (
            "the same seed must regenerate a byte-identical null (AI_RULES.md 1.1)"
        )


class TestStageCOrdering:
    """Null before effect (AI_RULES.md 1.5) — enforced in code, not by convention."""

    def _cfg(self, tmp_path, **overrides):
        defaults = {"compression_family": "magnitude", "compression_level": "0.30",
                    "frozen_root": str(tmp_path / "frozen"), "stage_c": {"cell": CELL},
                    "null_frozen_hash": "0" * 64}
        return _base(tmp_path, "stageC", **{**defaults, **overrides})

    def test_stage_c_refuses_when_no_null_was_ever_frozen(self, tmp_path):
        _stage_a(tmp_path)
        _stage_b(tmp_path)  # a DRAFT null exists under runs/ — that must not count
        with pytest.raises(RuntimeError, match="not frozen"):
            run_stage_c(self._cfg(tmp_path))

    def test_a_draft_null_under_runs_is_not_a_frozen_null(self, tmp_path):
        """The Stage B draft carries a hash; Stage C must still refuse it."""
        b_dir = _stage_b(tmp_path)
        draft_hash = json.loads((b_dir / "meta.json").read_text(encoding="utf-8"))["null_frozen_hash"]
        with pytest.raises(RuntimeError, match="not frozen"):
            run_stage_c(self._cfg(tmp_path, null_frozen_hash=draft_hash))

    def test_stage_c_accepts_only_the_registered_hash(self, tmp_path):
        cell_hash = self._promote_draft_to_frozen(tmp_path)

        with pytest.raises(RuntimeError, match="mismatch"):
            run_stage_c(self._cfg(tmp_path, null_frozen_hash="9" * 64))

        # correct hash -> every ordering guard passes and Stage C runs to completion
        run_dir = run_stage_c(self._cfg(tmp_path, null_frozen_hash=cell_hash))
        assert (run_dir / "csi_table.csv").exists()

    def test_stage_c_runs_on_the_synthetic_stack_and_stamps_it_non_evidence(self, tmp_path):
        """Algorithm 1 Stage C, end to end, with nothing real anywhere near it."""
        cell_hash = self._promote_draft_to_frozen(tmp_path)
        run_dir = run_stage_c(self._cfg(tmp_path, null_frozen_hash=cell_hash))

        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["evidence"] is False
        assert meta["synthetic"] is True
        assert "NOT scientific evidence" in meta["note"]
        assert meta["null_frozen_hash"] == cell_hash

        rows = read_csi_table(run_dir / "csi_table.csv")
        assert {r["comparison_level"] for r in rows} == {"exact_edge", "routing_head"}
        for row in rows:
            assert row["ci_lo"] <= row["csi"] <= row["ci_hi"]
            assert row["null_frozen_hash"] == cell_hash

    def test_stage_c_attaches_a_chance_floor_to_every_level(self, tmp_path):
        """AI_RULES.md 4.4: no overlap statistic is reported without its floor."""
        cell_hash = self._promote_draft_to_frozen(tmp_path)
        run_dir = run_stage_c(self._cfg(tmp_path, null_frozen_hash=cell_hash))
        floors = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))["chance_floors"]
        assert set(floors) == {"exact_edge", "routing_head"}
        for f in floors.values():
            assert {"observed", "chance_median", "p_value"} <= set(f)

    def test_stage_c_run_meta_is_strict_valid_json(self, tmp_path):
        """NaN is a normal result here; bare NaN in JSON is not parseable by strict readers."""
        cell_hash = self._promote_draft_to_frozen(tmp_path)
        run_dir = run_stage_c(self._cfg(tmp_path, null_frozen_hash=cell_hash))
        text = (run_dir / "run_meta.json").read_text(encoding="utf-8")
        json.loads(text, parse_constant=_reject_constant)

    def test_engineering_mode_still_refuses_stage_c_on_a_real_model(self, tmp_path):
        """The prohibition that matters is unchanged: nothing real, ever, in this mode."""
        cell_hash = self._promote_draft_to_frozen(tmp_path)
        cfg = self._cfg(tmp_path, null_frozen_hash=cell_hash)
        cfg["model"] = {**cfg["model"], "synthetic": False, "name": "mock"}
        with pytest.raises(ValueError, match="REFUSES Stage C"):
            run_stage_c(cfg)

    def _promote_draft_to_frozen(self, tmp_path) -> str:
        """Stand in for the (PI-approved, scientific-mode) freeze event."""
        b_dir = _stage_b(tmp_path)
        meta = json.loads((b_dir / "meta.json").read_text(encoding="utf-8"))
        cell_rel = f"{_MODEL['name']}/{_TASK['name']}/{CELL}"
        cell = tmp_path / "frozen" / cell_rel
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "dnull.parquet").write_bytes((b_dir / "dnull.parquet").read_bytes())
        (cell / "meta.json").write_text(json.dumps({**meta, "frozen": True}), encoding="utf-8")
        (tmp_path / "frozen" / "FREEZE_MANIFEST.json").write_text(
            json.dumps({"frozen": True, "freeze_commit": "test",
                        "cells": {cell_rel: {"null_frozen_hash": meta["null_frozen_hash"]}}}),
            encoding="utf-8",
        )
        return meta["null_frozen_hash"]

    def test_frozen_cell_path_matches_the_architecture_layout(self, tmp_path):
        path = frozen_cell_path(self._cfg(tmp_path))
        assert path.parts[-3:] == (_MODEL["name"], _TASK["name"], CELL)


class TestMetricsRunOverRealEnsembleOutput:
    """The Stage C arithmetic, exercised on genuine Stage A/B output."""

    def test_distance_of_a_reference_against_itself_is_zero(self, tmp_path):
        f = _freq(_stage_a(tmp_path))
        assert l1_distance(f, f) == pytest.approx(0.0)
        assert jensen_shannon_distance(f, f) == pytest.approx(0.0)

    def test_csi_is_computable_from_a_stage_b_draft(self, tmp_path):
        import pyarrow.parquet as pq

        f_dense = _freq(_stage_a(tmp_path))
        b_dir = _stage_b(tmp_path)
        d_null = [r["distance_l1"] for r in pq.read_table(str(b_dir / "dnull.parquet")).to_pylist()]
        if all(d == 0.0 for d in d_null):
            pytest.skip("degenerate synthetic null: perturbation moved no edge at this seed")

        # a fabricated "post-compression" vector: halve every inclusion frequency
        f_post = {k: v / 2 for k, v in f_dense.items()}
        out = csi(l1_distance(f_dense, f_post), d_null, n_boot=100, seed=0)
        assert out["csi"] > 0.0
        assert out["ci_lo"] <= out["csi"] <= out["ci_hi"]

    def test_both_comparison_levels_are_reported_together(self, tmp_path):
        f = _freq(_stage_a(tmp_path))
        levels = compare_at_both_levels(f, {k: v for i, (k, v) in enumerate(f.items()) if i % 2 == 0})
        assert set(levels) == {"exact_edge", "routing_head"}
        assert all(0.0 <= v <= 1.0 for v in levels.values())
