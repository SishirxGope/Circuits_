# [AI-GEN] agent=Claude date=2026-09-21 task=Build the resolved freeze config from completed Stage B runs
# modified: [AI-GEN] agent=Claude date=2026-09-30 task=viable pairs only, model names from configs, --models for a deliberate per-model freeze
# reviewed-by: PENDING
#
# experiments/freeze_stage_b.py takes --config <resolved config JSON> and reads
# stage_b.cells = [{model, task, cell, source_run_dir}, ...]. Assembling that list by hand
# across 44 (Plan A) or 66 (Plan B) cells is where a mistake would be both easy and
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
#   python deploy/shared/make_freeze_config.py --plan b --models pythia160m --out f.json
#       # freeze ONE model's cells. Protocol rule 1 is per cell (a cell's null is frozen
#       # before THAT cell's Stage C runs), so freezing model by model is allowed - but it
#       # is a deliberate, named narrowing, never a silent partial grid.

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_cells import CELLS, PLAN_A_MODELS, PLAN_B_MODELS, viable_pairs


def model_name(model_cfg: str) -> str:
    """The model's ``name:`` from configs/model/<model_cfg>.yaml.

    Stage C looks for its null at frozen/{model.name}/{task}/{cell}, so the freeze must
    file it under exactly that name. This was a hand-typed map that said "gemma2-2b" and
    "llama32-1b" where the configs say "gemma-2-2b" and "llama-3.2-1b": both primaries'
    nulls would have been frozen where Stage C never looks.
    """
    path = REPO / "configs" / "model" / f"{model_cfg}.yaml"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].split("#", 1)[0].strip()
    raise ValueError(f"{path} has no top-level name:")


MODEL_NAME = {m: model_name(m) for m in PLAN_B_MODELS}


def find_stage_b_runs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    return sorted(p for p in run_root.iterdir() if p.is_dir() and "_stageB_" in p.name)


def _norm(value: str) -> str:
    """Strip every separator so run names and config values compare on content alone.

    Run names are built by run_naming.sanitize_token, which turns illegal characters
    into "-": task "greater_than" becomes "greater-than", cell "rtn_int8" becomes
    "rtn-int8". This matcher previously stripped "_" only, so it looked for
    "greaterthan" and "rtn_int8" and matched NEITHER - every greater_than and every
    compression cell would have been reported MISSING, and a partial freeze config
    refused, with nothing pointing at the cause. Normalising both sides to
    alphanumerics makes the comparison independent of which separator was used.
    """
    return re.sub(r"[^a-z0-9]", "", value.lower())


def match_run(runs: list[Path], model_cfg: str, task: str, cell: str) -> list[Path]:
    """Every Stage B run dir whose name carries this model, task and cell."""
    model_tok = _norm(MODEL_NAME.get(model_cfg, model_cfg))
    task_tok = _norm(task)
    cell_tok = _norm(cell)
    out = []
    for r in runs:
        n = _norm(r.name)
        if model_tok in n and task_tok in n and cell_tok in n:
            out.append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["a", "b"], default="a")
    ap.add_argument("--out", required=True)
    ap.add_argument("--approve", action="store_true",
                    help="set stage_b.freeze_approved=true (the pre-registration act)")
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--models", nargs="+", default=None,
                    help="freeze only these models' cells (a deliberate per-model freeze)")
    args = ap.parse_args(argv)

    models = PLAN_A_MODELS if args.plan == "a" else PLAN_B_MODELS
    if args.models:
        unknown = sorted(set(args.models) - set(models))
        if unknown:
            print(f"--models {unknown} are not in plan {args.plan}: {models}")
            return 2
        models = [m for m in models if m in args.models]
    runs = find_stage_b_runs(REPO / args.run_root)
    print(f"found {len(runs)} Stage B run directories under {args.run_root}/")

    cells_out, missing, ambiguous = [], [], []
    # Viable pairs only: the greater_than cells Gemma-2 and Llama-3.2 cannot run are never
    # queued, so counting them made every Plan B freeze config "MISSING" 22 cells.
    pairs = viable_pairs(models)
    for model, task in pairs:
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

    expected = len(pairs) * len(CELLS)
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
