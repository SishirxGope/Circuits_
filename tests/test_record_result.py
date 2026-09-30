# [AI-GEN] agent=Claude date=2026-09-30 task=RESULTS.md run log - one timestamped row per finished or failed cell
# reviewed-by: PENDING

"""deploy/shared/record_result.py: the rows it appends are read from the run's own files."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_record_result_under_test", REPO / "deploy" / "shared" / "record_result.py")
record_result = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(record_result)

WHEN = re.compile(r"^\| \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2} \|")


def _rows(results: Path) -> list[str]:
    return [line for line in results.read_text(encoding="utf-8").splitlines()
            if line.startswith("| 2")]


def _stage_a_run(root: Path) -> Path:
    pytest.importorskip("pyarrow")
    from src.common.schema import write_freq_parquet

    run = root / "20260930_stageA_pythia-160m_ioi_dense_B16xS5_seed0"
    run.mkdir(parents=True)
    write_freq_parquet(str(run / "freq.parquet"), [
        {"edge_id": "EMB->LOGIT", "s_e": 1.0, "band": "core"},
        {"edge_id": "L0.H0->LOGIT", "s_e": 0.5, "band": "contingent"},
        {"edge_id": "L0.MLP->LOGIT", "s_e": 0.25, "band": "noise"},
        {"edge_id": "L1.MLP->LOGIT", "s_e": 0.125, "band": "noise"},
    ])
    (run / "run_meta.json").write_text(json.dumps({
        "stage": "stageA", "model": "pythia-160m", "task": "ioi", "B": 16, "S": 5,
        "compression_family": "dense", "compression_level": None,
    }), encoding="utf-8")
    return run


def _stage_b_run(root: Path) -> Path:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    run = root / "20260930_stageB_pythia-160m_ioi_null-matchedmag-rtn-int8_B16xS5xR20_seed0"
    run.mkdir(parents=True)
    pq.write_table(pa.table({"r": [0, 1, 2], "distance_l1": [1.0, 3.0, 2.0],
                             "distance_jensen_shannon": [0.1, 0.3, 0.2]}), str(run / "dnull.parquet"))
    (run / "run_meta.json").write_text(json.dumps({
        "stage": "stageB", "model": "pythia-160m", "task": "ioi", "compression_family": "rtn",
        "compression_level": 8, "dnull_hash": "abcdef0123456789",
    }), encoding="utf-8")
    return run


class TestRows:
    def test_a_stage_a_row_counts_the_bands(self, tmp_path):
        results = tmp_path / "RESULTS.md"
        run = _stage_a_run(tmp_path)
        assert record_result.main(["--run-dir", str(run)], results=results) == 0
        text = results.read_text(encoding="utf-8")
        assert "## Run log" in text and "| When (local) |" in text
        (row,) = _rows(results)
        assert WHEN.match(row)
        assert "| stageA | pythia-160m | ioi | dense |" in row
        assert "edges with s>0: 4; core 1, contingent 1, noise 2" in row

    def test_a_stage_b_row_reports_the_draft_null(self, tmp_path):
        results = tmp_path / "RESULTS.md"
        run = _stage_b_run(tmp_path)
        assert record_result.main(["--run-dir", str(run)], results=results) == 0
        (row,) = _rows(results)
        assert "| stageB | pythia-160m | ioi | rtn-8 |" in row
        assert "R=3; D_null L1 median 2 [min 1, max 3]; JS median 0.2" in row
        assert "NOT frozen" in row and "abcdef012345" in row

    def test_rows_append_and_never_rewrite(self, tmp_path):
        results = tmp_path / "RESULTS.md"
        a, b = _stage_a_run(tmp_path), _stage_b_run(tmp_path)
        record_result.main(["--run-dir", str(a)], results=results)
        before = results.read_text(encoding="utf-8")
        record_result.main(["--run-dir", str(b)], results=results)
        after = results.read_text(encoding="utf-8")
        assert after.startswith(before) and len(_rows(results)) == 2

    def test_the_run_dir_is_read_from_the_cell_log(self, tmp_path):
        results = tmp_path / "RESULTS.md"
        run = _stage_b_run(tmp_path)
        log = tmp_path / "cell.log"
        log.write_text(f"noise\n[stageB-draft] run_name=x R=3 dnull_hash=abc median_l1=2 -> {run}\n"
                       "CELL_COMPLETE 2026-09-30T12:00:00+05:30\n", encoding="utf-8")
        assert record_result.main(["--log", str(log)], results=results) == 0
        assert len(_rows(results)) == 1

    def test_a_failed_cell_is_logged_as_failed(self, tmp_path):
        results = tmp_path / "RESULTS.md"
        cell = "stage=stageB model=gemma2_2b task=ioi setting=null-matchedmag-awq-int4 compression=awq_int4"
        assert record_result.main(["--failed", "--cell", cell, "--log", "logs/x.log"], results=results) == 0
        (row,) = _rows(results)
        assert "| stageB | gemma2_2b | ioi | awq_int4 | **FAILED** - see `logs/x.log` |" in row

    def test_no_run_dir_records_nothing(self, tmp_path):
        results = tmp_path / "RESULTS.md"
        log = tmp_path / "cell.log"
        log.write_text("Traceback ...\n  run_dir = allocate_run_dir(\n", encoding="utf-8")
        assert record_result.main(["--log", str(log)], results=results) == 1
        assert not results.exists()


class TestTheCellRunner:
    def test_done_means_the_marker_not_a_traceback(self):
        script = (REPO / "deploy" / "plan_b_dgx_spark" / "_run_one.sh").read_text(encoding="utf-8")
        assert 'grep -q "^$DONE_MARK"' in script
        assert 'grep -qE "run_dir' not in script
        assert "record_result.py --log" in script and "record_result.py --failed" in script
