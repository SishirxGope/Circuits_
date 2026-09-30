# [AI-GEN] agent=Claude date=2026-09-30 task=Append one timestamped row per finished run to RESULTS.md (the run log the PI asked for)
# reviewed-by: PENDING
#
# WHAT THIS DOES
# --------------
# Appends ONE row to the run-log table at the end of RESULTS.md (repo root) for a finished
# run directory, with the local date and time, the stage, model, task, cell, the headline
# numbers read from the run's own output files, the run directory and the git commit.
# _run_one.sh calls it after every cell, success or failure, so the log fills itself.
#
# It only READS run outputs; it never changes a run directory (runs are immutable,
# AI_RULES.md 1.2). Numbers are copied from the files the run wrote, never recomputed or
# rounded into a different claim (AI_RULES.md 2.2).
#
# Usage (repo root):
#   python deploy/shared/record_result.py --run-dir runs/<run_name>
#   python deploy/shared/record_result.py --log logs/stageB/<cell>.log       # run dir from the log
#   python deploy/shared/record_result.py --failed --cell "<overrides>" --log logs/stageB/<cell>.log
#
# Appends are one os.write on an O_APPEND descriptor, so cells finishing at the same time
# under `xargs -P` do not interleave their rows.

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "RESULTS.md"
TABLE_HEADER = (
    "| When (local) | Stage | Model | Task | Cell | Result | Run dir | Commit |\n"
    "|---|---|---|---|---|---|---|---|\n"
)
_RUN_DIR_IN_LOG = re.compile(r"-> (\S+)\s*$")


def now_local() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def git_commit() -> str:
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=False).stdout.strip()
    return out or "?"


def run_dir_from_log(log: Path) -> Path | None:
    """The run directory a stage runner printed on its last ``... -> <run_dir>`` line."""
    if not log.exists():
        return None
    for line in reversed(log.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith("[stage"):
            match = _RUN_DIR_IN_LOG.search(line)
            if match:
                return Path(match.group(1))
    return None


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4g}"


def summarize(run_dir: Path) -> dict[str, str]:
    """Stage, model, task, cell and a one-line headline, from the run's own files."""
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    stage = str(meta.get("stage", "?"))
    level = meta.get("compression_level")
    cell = "dense" if stage == "stageA" else f"{meta.get('compression_family', '?')}-{level}"
    row = {"stage": stage, "model": str(meta.get("model", "?")), "task": str(meta.get("task", "?")),
           "cell": cell, "result": ""}

    if stage == "stageA":
        from src.common.schema import read_freq_parquet

        rows = read_freq_parquet(str(run_dir / "freq.parquet"))
        bands = {b: sum(1 for r in rows if r.get("band") == b) for b in ("core", "contingent", "noise")}
        row["result"] = (f"B={meta.get('B')} S={meta.get('S')}; edges with s>0: {len(rows)}; "
                         f"core {bands['core']}, contingent {bands['contingent']}, noise {bands['noise']}")
    elif stage == "stageB":
        import pyarrow.parquet as pq

        dnull = pq.read_table(str(run_dir / "dnull.parquet")).to_pylist()
        l1 = [r["distance_l1"] for r in dnull]
        js = [r["distance_jensen_shannon"] for r in dnull]
        row["result"] = (f"R={len(dnull)}; D_null L1 median {_fmt(statistics.median(l1))} "
                         f"[min {_fmt(min(l1))}, max {_fmt(max(l1))}]; JS median {_fmt(statistics.median(js))}; "
                         f"dnull sha {str(meta.get('dnull_hash', ''))[:12]} (NOT frozen)")
    elif stage == "stageC":
        with open(run_dir / "csi_table.csv", encoding="utf-8", newline="") as f:
            table = [r for r in csv.DictReader(f)]
        parts = [f"CSI {r['comparison_level']} {_fmt(float(r['csi']))} "
                 f"[{_fmt(float(r['ci_lo']))}, {_fmt(float(r['ci_hi']))}]" for r in table]
        row["result"] = "; ".join(parts) + f"; D_null median {_fmt(meta.get('dnull_median'))}"
    elif stage == "stageD":
        report = json.loads((run_dir / "stage_d_report.json").read_text(encoding="utf-8"))
        summary = report.get("csi_summary", {})
        c5 = report.get("cross_audit_c5", {})
        row["result"] = (f"cells {summary.get('n_cells')}; selective {summary.get('structurally_selective')}; "
                         f"indistinguishable {summary.get('indistinguishable_from_noise')}; "
                         f"C5 {c5.get('status', 'computed')}")
    return row


def table_row(row: dict[str, str], run_dir: str, commit: str, when: str) -> str:
    cells = [when, row["stage"], row["model"], row["task"], row["cell"], row["result"], f"`{run_dir}`", commit]
    return "| " + " | ".join(c.replace("|", "/").replace("\n", " ") for c in cells) + " |\n"


def append(line: str, results: Path = RESULTS) -> None:
    if not results.exists():
        results.write_text("# RESULTS\n\n## Run log\n\n" + TABLE_HEADER, encoding="utf-8", newline="\n")
    fd = os.open(str(results), os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0))
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def _cell_field(cell: str, key: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(key)}=(\S+)", cell)
    return match.group(1) if match else "?"


def main(argv: list[str] | None = None, results: Path = RESULTS) -> int:
    ap = argparse.ArgumentParser(description="Append a timestamped run-log row to RESULTS.md.")
    ap.add_argument("--run-dir", help="the finished run directory")
    ap.add_argument("--log", help="the cell log; the run dir is read from its last line")
    ap.add_argument("--failed", action="store_true", help="record a FAILED cell (needs --cell)")
    ap.add_argument("--cell", default="", help="the cell's override string, for --failed")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    when, commit = now_local(), git_commit()

    if args.failed:
        row = {"stage": _cell_field(args.cell, "stage"), "model": _cell_field(args.cell, "model"),
               "task": _cell_field(args.cell, "task"), "cell": _cell_field(args.cell, "compression"),
               "result": f"**FAILED** - see `{args.log or '?'}`"}
        if row["cell"] == "?":
            row["cell"] = "dense" if row["stage"] == "stageA" else "?"
        append(table_row(row, "-", commit, when), results)
        return 0

    run_dir = Path(args.run_dir) if args.run_dir else (run_dir_from_log(Path(args.log)) if args.log else None)
    if run_dir is None or not (run_dir / "run_meta.json").exists():
        print(f"record_result: no finished run directory found ({run_dir or args.log}); nothing recorded")
        return 1
    try:
        row = summarize(run_dir)
    except Exception as exc:  # noqa: BLE001 - a log row must never be what fails a finished run
        row = {"stage": "?", "model": "?", "task": "?", "cell": "?",
               "result": f"finished, but its summary could not be read: {type(exc).__name__}: {exc}"}
    try:
        shown = run_dir.resolve().relative_to(REPO).as_posix()
    except ValueError:
        shown = run_dir.as_posix()
    append(table_row(row, shown, commit, when), results)
    print(f"recorded in {results.name}: {row['stage']} {row['model']} {row['task']} {row['cell']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
