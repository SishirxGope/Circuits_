# [AI-GEN] agent=Claude date=2026-08-08 task=End-to-end Algorithm 1 (A -> B -> freeze -> C -> D) on the synthetic stack
# reviewed-by: PENDING

"""Algorithm 1, all four stages plus the freeze, in one test.

Every other test checks a stage or a guard in isolation. This one checks that the
stages COMPOSE: that Stage A's reference feeds Stage B, that Stage B's draft can be
frozen, that Stage C divides by exactly that frozen null and by nothing else, and that
Stage D consumes Stage C's table. A pipeline whose parts each pass but which cannot be
run start to finish is not a pipeline.

Everything is synthetic (mock model, synthetic task, mock extractor). No scientific
evidence is produced and every artifact is stamped ``evidence: false``.
"""

import copy
import json
from pathlib import Path

import pytest

from experiments.run_stage_a import run_stage_a
from experiments.run_stage_b import run_stage_b
from experiments.run_stage_c import run_stage_c
from experiments.run_stage_d import run_stage_d
from src.common.csi_table import read_csi_table
from src.common.freeze import freeze_cell, verify_frozen_store

B, S, R = 4, 2, 5
CELLS = [("magnitude", "0.30", 0.30), ("magnitude", "0.50", 0.50)]


def _base(tmp_path):
    return {
        "mode": {"name": "engineering_dry_run"}, "seed": 0,
        "run_root": str(tmp_path / "runs"), "pipeline": "mock", "comparison_level": "both",
        "model": {"name": "mock", "synthetic": True, "n_layers": 4, "n_heads": 2, "d_model": 8},
        "task": {"name": "synthetic-ioi", "family": "indirect-object-identification-synthetic",
                 "synthetic": True, "n_prompts": 24, "max_seq_len": 32, "seed": 0},
        "ensemble": {"B": B, "S": S, "configset": f"B{B}xS{S}", "threshold_grid": None,
                     "pi_confirmed": False,
                     "decompose": {"core_threshold": 0.90, "noise_threshold": 0.10}},
        "distance": {"name": "l1", "alternative": "jensen_shannon"},
        "comparison": {"level2_scheme": None, "position_policy": None},
        "nulls": {"R": R}, "compression": {"name": "null"},
    }


def _cfg(tmp_path, **kw):
    cfg = copy.deepcopy(_base(tmp_path))
    cfg.update(kw)
    return cfg


@pytest.fixture(scope="module")
def algorithm1(tmp_path_factory):
    """Run the whole of Algorithm 1 once; every test below inspects the result."""
    tmp_path = tmp_path_factory.mktemp("alg1")
    frozen_root = tmp_path / "frozen"

    stage_a = run_stage_a(_cfg(tmp_path, stage="stageA", setting="dense",
                               compression_family="dense", compression_level=None))

    frozen, stage_b_dirs = {}, {}
    for family, level, sparsity in CELLS:
        cell = f"{family}-{level}"
        b_dir = run_stage_b(_cfg(
            tmp_path, stage="stageB", setting=f"null-matchedmag-{level.replace('.', '')}",
            compression_family="null", compression_level=f"matchedmag-{level}",
            stage_b={"sparsity": sparsity, "global_scope": True},
        ))
        stage_b_dirs[cell] = b_dir
        meta = freeze_cell(frozen_root, "mock", "synthetic-ioi", cell, b_dir,
                           freeze_commit="TEST-FREEZE-COMMIT")
        frozen[cell] = meta["null_frozen_hash"]

    stage_c_dirs = []
    for family, level, sparsity in CELLS:
        cell = f"{family}-{level}"
        stage_c_dirs.append(str(run_stage_c(_cfg(
            tmp_path, stage="stageC", compression_family=family, compression_level=level,
            frozen_root=str(frozen_root), null_frozen_hash=frozen[cell],
            stage_c={"cell": cell, "n_boot": 200, "compressor_kwargs": {"sparsity": sparsity}},
        ))))

    stage_d = run_stage_d(_cfg(
        tmp_path, stage="stageD", setting="crossaudit", compression_family="magnitude",
        compression_level="grid", null_frozen_hash=next(iter(frozen.values())),
        stage_d={"stage_c_run_dirs": stage_c_dirs, "n_permutations": 500, "n_boot": 200,
                 "feature_damage": {"mock/synthetic-ioi/magnitude/0.30": 0.4,
                                    "mock/synthetic-ioi/magnitude/0.50": 0.9}},
    ))

    return {"tmp": tmp_path, "frozen_root": frozen_root, "stage_a": stage_a,
            "stage_b_dirs": stage_b_dirs, "frozen": frozen,
            "stage_c_dirs": stage_c_dirs, "stage_d": stage_d}


class TestTheChainCompletes:
    def test_all_stages_produced_their_artifacts(self, algorithm1):
        assert (algorithm1["stage_a"] / "freq.parquet").exists()
        for b in algorithm1["stage_b_dirs"].values():
            assert (b / "dnull.parquet").exists()
        for c in algorithm1["stage_c_dirs"]:
            assert (Path(c) / "csi_table.csv").exists()
        assert (algorithm1["stage_d"] / "stage_d_report.json").exists()

    def test_the_frozen_store_verifies_after_the_whole_run(self, algorithm1):
        result = verify_frozen_store(algorithm1["frozen_root"])
        assert result["ok"] is True, result["problems"]
        assert result["n_cells"] == len(CELLS)


class TestStageCDividesByTheFrozenNull:
    def test_every_csi_row_names_the_frozen_hash_it_used(self, algorithm1):
        hashes = set(algorithm1["frozen"].values())
        for c in algorithm1["stage_c_dirs"]:
            for row in read_csi_table(Path(c) / "csi_table.csv"):
                assert row["null_frozen_hash"] in hashes

    def test_dnull_median_matches_the_frozen_cell(self, algorithm1):
        """The denominator in the table must be the frozen null's, not a re-drawn one."""
        import statistics

        import pyarrow.parquet as pq

        for c in algorithm1["stage_c_dirs"]:
            meta = json.loads((Path(c) / "run_meta.json").read_text(encoding="utf-8"))
            frozen_cell = Path(meta["frozen_cell"])
            draws = [r["distance_l1"] for r in pq.read_table(str(frozen_cell / "dnull.parquet")).to_pylist()]
            rows = read_csi_table(Path(c) / "csi_table.csv")
            assert all(r["dnull_median"] == pytest.approx(statistics.median(draws)) for r in rows)

    def test_heavier_compression_moves_the_circuit_further(self, algorithm1):
        """A sanity property of the whole chain: 50% pruning must damage more than 30%."""
        by_level = {}
        for c in algorithm1["stage_c_dirs"]:
            for row in read_csi_table(Path(c) / "csi_table.csv"):
                by_level.setdefault(row["comparison_level"], {})[row["level_param"]] = row["D"]
        assert by_level["exact_edge"]["0.50"] > by_level["exact_edge"]["0.30"]


class TestBothLevelsAlways:
    def test_every_stage_c_cell_reports_both_levels(self, algorithm1):
        for c in algorithm1["stage_c_dirs"]:
            levels = {r["comparison_level"] for r in read_csi_table(Path(c) / "csi_table.csv")}
            assert levels == {"exact_edge", "routing_head"}

    def test_stage_d_ranks_damage_at_both_levels(self, algorithm1):
        report = json.loads((algorithm1["stage_d"] / "stage_d_report.json").read_text(encoding="utf-8"))
        assert set(report["damage_ranking"]) == {"exact_edge", "routing_head"}
        assert len(report["damage_ranking"]["exact_edge"]) == len(CELLS)

    def test_level_disagreements_are_surfaced_not_averaged(self, algorithm1):
        """Claim C2: where the levels disagree, the report must say so."""
        report = json.loads((algorithm1["stage_d"] / "stage_d_report.json").read_text(encoding="utf-8"))
        assert "levels_disagree" in report["csi_summary"]


class TestStageDReporting:
    def test_report_is_strict_valid_json(self, algorithm1):
        def reject(name):  # pragma: no cover - only on malformed output
            raise AssertionError(f"stage_d_report.json contains bare {name!r}")

        text = (algorithm1["stage_d"] / "stage_d_report.json").read_text(encoding="utf-8")
        json.loads(text, parse_constant=reject)

    def test_cross_audit_ran_and_carries_an_interval(self, algorithm1):
        report = json.loads((algorithm1["stage_d"] / "stage_d_report.json").read_text(encoding="utf-8"))
        c5 = report["cross_audit_c5"]["exact_edge"]
        assert {"spearman", "ci_lo", "ci_hi", "p_value", "verdict", "verdict_ci_consistent"} <= set(c5)
        assert c5["n_shared_cells"] == len(CELLS)

    def test_a_two_cell_cross_audit_cannot_support_a_verdict(self, algorithm1):
        """The honest brake: two cells is not evidence however good rho looks."""
        report = json.loads((algorithm1["stage_d"] / "stage_d_report.json").read_text(encoding="utf-8"))
        assert report["cross_audit_c5"]["exact_edge"]["verdict_ci_consistent"] is False

    def test_grid_size_is_reported_so_multiplicity_is_visible(self, algorithm1):
        report = json.loads((algorithm1["stage_d"] / "stage_d_report.json").read_text(encoding="utf-8"))
        assert report["multiple_comparisons"]["applied"] is False
        assert "expected_false_positives_if_uncorrected_at_0.05" in report["multiple_comparisons"]

    def test_c4_is_declared_pending_not_fabricated(self, algorithm1):
        report = json.loads((algorithm1["stage_d"] / "stage_d_report.json").read_text(encoding="utf-8"))
        assert report["interaction_diagnostic_c4"]["status"] == "PENDING"


class TestNothingHereIsEvidence:
    def test_every_stage_c_and_d_artifact_is_stamped_non_evidence(self, algorithm1):
        for c in algorithm1["stage_c_dirs"]:
            meta = json.loads((Path(c) / "run_meta.json").read_text(encoding="utf-8"))
            assert meta["evidence"] is False and meta["synthetic"] is True
        report = json.loads((algorithm1["stage_d"] / "stage_d_report.json").read_text(encoding="utf-8"))
        assert report["evidence"] is False


class TestStageDRefusals:
    def test_refuses_without_stage_c_outputs(self, tmp_path):
        cfg = _cfg(tmp_path, stage="stageD", setting="crossaudit",
                   compression_family="magnitude", compression_level="grid",
                   null_frozen_hash="a" * 64, stage_d={"stage_c_run_dirs": []})
        with pytest.raises(RuntimeError, match="requires Stage C outputs"):
            run_stage_d(cfg)

    def test_refuses_a_stage_c_dir_with_no_csi_table(self, tmp_path):
        empty = tmp_path / "not-a-stage-c-run"
        empty.mkdir()
        cfg = _cfg(tmp_path, stage="stageD", setting="crossaudit",
                   compression_family="magnitude", compression_level="grid",
                   null_frozen_hash="a" * 64, stage_d={"stage_c_run_dirs": [str(empty)]})
        with pytest.raises(RuntimeError, match="no csi_table.csv"):
            run_stage_d(cfg)

    def test_cross_audit_is_pending_rather_than_invented_without_pi_numbers(self, algorithm1, tmp_path):
        """AI_RULES.md 2.2: the comparison target is read from the papers, never guessed."""
        cfg = _cfg(tmp_path, stage="stageD", setting="crossaudit-nofeature",
                   compression_family="magnitude", compression_level="grid",
                   null_frozen_hash=next(iter(algorithm1["frozen"].values())),
                   stage_d={"stage_c_run_dirs": algorithm1["stage_c_dirs"]})
        run_dir = run_stage_d(cfg)
        report = json.loads((run_dir / "stage_d_report.json").read_text(encoding="utf-8"))
        assert report["cross_audit_c5"]["status"] == "PENDING"
        assert "AI_RULES.md 2.2" in report["cross_audit_c5"]["reason"]
