# [AI-GEN] agent=Claude date=2026-09-21 task=Build the resolved freeze config from completed Stage B runs
# reviewed-by: PENDING
#
# experiments/freeze_stage_b.py takes --config <resolved config JSON> and reads
# stage_b.cells = [{model, task, cell, source_run_dir}, ...]. Assembling that list by hand
# across 44 (Plan A) or 88 (Plan B) cells is where a mistake would be both easy and
# invisible: a cell pointing at the wrong run directory freezes the wrong null, and Stage C
# would happily compare against it.
#
# So this script DISCOVERS the Stage B runs on disk, matches them to the grid table in
# gen_cells.py, and refuses to emit a config if anything is missing or ambiguous.
#
# It deliberately does NOT set stage_b.freeze_approved. That flag is the human act of
# pre-registration and must be typed by a person (freeze_stage_b.py has no default for it).
#
# Usage:
#   python deploy/shared/make_freeze_config.py --plan a --out freeze_config.json
#   python deploy/shared/make_freeze_config.py --plan a --out f.json --approve   # sets the flag

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_cells import CELLS, PLAN_A_MODELS, PLAN_B_MODELS, TASKS  # noqa: E402

# Run-name model tokens differ from config file names (configs/model/pythia160m.yaml
# carries `name: pythia-160m`), so map explicitly rather than guessing.
MODEL_NAME = {
    "pythia160m": "pythia-160m",
    "pythia410m": "pythia-410m",
    "gemma2_2b": "gemma2-2b",
    "llama32_1b": "llama32-1b",
}


def find_stage_b_runs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    return sorted(p for p in run_root.iterdir() if p.is_dir() and "_stageB_" in p.name)


def match_run(runs: list[Path], model_cfg: str, task: str, cell: str) -> list[Path]:
    """Every Stage B run dir whose name carries this model, task and cell."""
    model_tok = MODEL_NAME.get(model_cfg, model_cfg)
    task_tok = task.replace("_", "")
    out = []
    for r in runs:
        n = r.name.lower()
        if model_tok.lower() in n and (task_tok in n.replace("_", "")) and cell.lower() in n:
            out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["a", "b"], default="a")
    ap.add_argument("--out", required=True)
    ap.add_argument("--approve", action="store_true",
                    help="set stage_b.freeze_approved=true (the pre-registration act)")
    ap.add_argument("--run-root", default="runs")
    args = ap.parse_args()

    models = PLAN_A_MODELS if args.plan == "a" else PLAN_B_MODELS
    runs = find_stage_b_runs(REPO / args.run_root)
    print(f"found {len(runs)} Stage B run directories under {args.run_root}/")

    cells_out, missing, ambiguous = [], [], []
    for model in models:
        for task in TASKS:
            for c in CELLS:
                hits = match_run(runs, model, task, c["cell"])
                if not hits:
                    missing.append(f"{model}/{task}/{c['cell']}")
                elif len(hits) > 1:
                    ambiguous.append(f"{model}/{task}/{c['cell']} -> {[h.name for h in hits]}")
                else:
                    cells_out.append({
                        "model": MODEL_NAME.get(model, model),
                        "task": task,
                        "cell": c["cell"],
                        "source_run_dir": str(hits[0]),
                    })

    expected = len(models) * len(TASKS) * len(CELLS)
    print(f"matched {len(cells_out)} / {expected} cells")

    if missing:
        print(f"\nMISSING ({len(missing)}) - Stage B has not been run for these:")
        for m in missing[:15]:
            print(f"  {m}")
        if len(missing) > 15:
            print(f"  ... and {len(missing) - 15} more")
    if ambiguous:
        print(f"\nAMBIGUOUS ({len(ambiguous)}) - more than one run matches; resolve by hand:")
        for a in ambiguous[:10]:
            print(f"  {a}")

    if missing or ambiguous:
        print("\nREFUSING to write a partial freeze config.")
        print("A freeze is irreversible and append-only; freeze the complete grid or")
        print("deliberately narrow --plan, never a silently partial one.")
        return 1

    cfg = {
        "mode": {"name": "scientific_run"},
        "stage_b": {"cells": cells_out, "freeze_approved": bool(args.approve)},
        "distance": {"pre_registered_for_stage_c": True},
        "frozen_root": "frozen",
    }
    out = Path(args.out)
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({len(cells_out)} cells, approved={bool(args.approve)})")
    if not args.approve:
        print("freeze_approved is FALSE - freeze_stage_b.py will refuse until you pass --approve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
