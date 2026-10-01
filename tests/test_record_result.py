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


class TestScriptsLogThemselves:
    """CLAUDE.md §8: a result without a stage run directory still gets its row."""

    @pytest.fixture
    def results(self, tmp_path, monkeypatch):
        import functools
        import sys
        import types

        results = tmp_path / "RESULTS.md"
        fake = types.SimpleNamespace(append_row=functools.partial(record_result.append_row, results=results))
        monkeypatch.setitem(sys.modules, "record_result", fake)
        return results

    def _load(self, path: Path, name: str):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_behaviour_check_writes_one_row_per_model_and_task(self, results, tmp_path):
        check = self._load(REPO / "deploy" / "shared" / "check_task_behaviour.py", "_check_under_test")
        report = {"seed": 0, "results": [
            {"model": "gemma2_2b", "task": "ioi", "prepend_bos": False, "pre_registered": True,
             "clean_mean": -4.326, "corrupt_mean": -0.164, "n_prompts": 300, "verdict": "DOES NOT do the task"},
            {"model": "gemma2_2b", "task": "ioi", "prepend_bos": True, "pre_registered": False,
             "clean_mean": 5.311, "corrupt_mean": -0.170, "n_prompts": 300, "verdict": "does the task"},
        ]}
        check._record_in_results(report, tmp_path / "x_behaviour.json")
        (row,) = _rows(results)
        assert WHEN.match(row)
        assert "| pilot-behaviour | gemma2_2b | ioi | dense |" in row
        assert "BOS False (pre-registered): clean -4.326, corrupt -0.164 -> DOES NOT do the task" in row
        assert "BOS True (flipped): clean 5.311" in row

    def test_the_timing_pilot_writes_one_row_per_task(self, results, tmp_path):
        timing = self._load(REPO / "experiments" / "time_attribution.py", "_timing_under_test")
        run = {"peak_vram_gib": 37.5, "clean_metric_mean": -4.33}
        report = {"model": {"name": "gemma-2-2b", "dtype": "float32"}, "tasks": {"ioi": {
            "runs": [run, {**run, "clean_metric_mean": -4.69}], "t_attribution_s_mean": 290.3,
            "run_plan_rule_applied_to_t_attribution": "S=5, R=10"}}}
        timing._record_in_results(report, tmp_path / "x_timing.json")
        (row,) = _rows(results)
        assert "| pilot-timing | gemma-2-2b | ioi | float32 |" in row
        assert "290.3 s per attribution pass (2 seeds), peak 37.5 GiB; rule -> S=5, R=10" in row
        assert "clean metric per seed -4.33 / -4.69" in row

    def test_a_logging_failure_never_fails_the_script(self, tmp_path, monkeypatch, capsys):
        import sys

        monkeypatch.setitem(sys.modules, "record_result", None)  # import raises
        check = self._load(REPO / "deploy" / "shared" / "check_task_behaviour.py", "_check_under_test2")
        check._record_in_results({"seed": 0, "results": []}, tmp_path / "x.json")
        assert "could not add RESULTS.md rows" in capsys.readouterr().out


class TestTheCellRunner:
    def test_done_means_the_marker_not_a_traceback(self):
        script = (REPO / "deploy" / "plan_b_dgx_spark" / "_run_one.sh").read_text(encoding="utf-8")
        assert 'grep -q "^$DONE_MARK"' in script
        assert 'grep -qE "run_dir' not in script
        assert "record_result.py --log" in script and "record_result.py --failed" in script

    def test_no_script_is_executed_without_bash(self):
        """Every .sh is committed from Windows as mode 100644, so a script run directly (as
        xargs ran _run_one.sh) dies with "Permission denied" on Linux. That killed the first
        Stage A on the Spark, 2026-10-01. Scripts must be invoked as `bash <script>.sh`."""
        direct = re.compile(r'(?<!bash )(?<!source )(?<!bash ")(?<!source ")\$HERE/\w+\.sh')
        offenders = []
        for path in sorted((REPO / "deploy" / "plan_b_dgx_spark").glob("*.sh")):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                code = line.split("#", 1)[0] if not line.lstrip().startswith("#") else ""
                if direct.search(code):
                    offenders.append(f"{path.name}:{n}: {line.strip()}")
        assert not offenders, offenders
