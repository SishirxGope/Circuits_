# [AI-GEN] agent=Claude date=2026-09-14 task=Q3 timing pilot for the dense-node pipeline (NON-EVIDENCE)
# reviewed-by: PENDING

"""Q3 timing pilot: how long one attribution pass takes, so B/S/R can be fixed from a number.

**This is not a Stage A run and produces no evidence.** It exists because of a deliberate
circularity: ``scientific_run`` refuses Stage A while Q3 (B, S, R) is open, and Q3 is meant
to be fixed from the measured wall time of one attribution pass (docs/Run_Plan.md Step 5).
So the timing is taken here, outside the stage runners, and the output is stamped
``non_evidence``. Nothing is written under a stage run directory, and nothing it measures
may enter a CSI table or a figure.

What it measures, per task and seed, on the pinned model:

- ``prompt_build_s``: drawing and tokenizing the clean/corrupted prompt pairs;
- ``attribution_s``: one edge-attribution-patching pass (two forwards + one backward per
  batch) - Run_Plan's ``t_attribution``;
- ``reprune_s_per_view``: re-thresholding the cached scores under ONE of the B views.

The per-cell cost model follows Run_Plan: a cell runs S seeds on the dense model and on
each of R null draws, so ``S * (1 + R)`` attribution passes, each followed by B re-prunes.

The script REPORTS Run_Plan's pre-stated decision rule; it does not choose B, S or R. Q3 is
a PI decision.

Usage:
    python -m experiments.time_attribution --model pythia160m --tasks ioi greater_than --seeds 2
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata as md
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUILDERS = {
    "ioi": ("src.tasks.ioi", "build_ioi_prompts"),
    "greater_than": ("src.tasks.greater_than", "build_greater_than_prompts"),
}
# (S, R) options and the rule, verbatim from docs/Run_Plan.md "Q3 - B, S, R".
S_R_OPTIONS = [(5, 20), (5, 10), (3, 10)]
RUN_PLAN_RULE = (
    "t <= 2 min -> S=5, R=20;  2 min < t <= 6 min -> R=10, S=5;  "
    "t > 6 min -> R=10, S=3, then drop secondary compression levels (never below S=3)"
)


def _load(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, cwd=ROOT)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def run(model_name: str, tasks: list[str], n_seeds: int, max_batch_size: int, out_dir: Path) -> Path:
    # n_seeds is a COUNT (seeds 0..n_seeds-1), not a seed value. A zero here used to fall
    # through to statistics.fmean() on an empty list, which fails as "fmean requires at
    # least one data point" long after the model has loaded - see the --seed/--seeds note
    # in main(). Refuse early and say what was actually wrong.
    if n_seeds < 1:
        raise SystemExit(
            f"--seeds is the NUMBER of seeds to time (0..n-1), got {n_seeds}. "
            "Pass --seeds 1 to time seed 0 only; --seeds 2 is the default."
        )

    import importlib

    import torch

    from src.extraction.dense_prune import prune_dense_graph
    from src.extraction.eap import compute_eap
    from src.extraction.real_model import load_pinned_model
    from src.science.threshold_grid import generate_anti_diagonal_grid

    model_cfg = _load(ROOT / "configs" / "model" / f"{model_name}.yaml")
    ensemble = _load(ROOT / "configs" / "ensemble" / "default.yaml")
    mode = _load(ROOT / "configs" / "mode" / "scientific_run.yaml")  # the download gate only
    B = int(ensemble["B"])
    grid_cfg = ensemble["threshold_grid"]
    grid = generate_anti_diagonal_grid(B, tuple(grid_cfg["node_range"]), tuple(grid_cfg["edge_range"]))

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{stamp}_timing_{model_name}.json"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; timing reports are never overwritten")

    t0 = time.perf_counter()
    model = load_pinned_model({"model": model_cfg, "mode": mode})
    load_s = time.perf_counter() - t0
    device = next(model.parameters()).device

    per_task: dict[str, Any] = {}
    for task_name in tasks:
        if task_name not in BUILDERS:
            raise ValueError(f"no real prompt generator for {task_name!r}; have {sorted(BUILDERS)}")
        module, fn = BUILDERS[task_name]
        build = getattr(importlib.import_module(module), fn)
        task_cfg = _load(ROOT / "configs" / "task" / f"{task_name}.yaml")
        runs = []
        for seed in range(n_seeds):
            t = time.perf_counter()
            prompts = build(model.tokenizer, task_cfg, seed, max_batch_size=max_batch_size)
            prompt_build_s = time.perf_counter() - t
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            scores = compute_eap(model, prompts)
            peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
            t = time.perf_counter()
            views = [prune_dense_graph(scores.edge_scores, scores.node_scores, c["node_threshold"], c["edge_threshold"]) for c in grid]
            reprune_s = (time.perf_counter() - t) / len(grid)
            runs.append({
                "seed": seed,
                "n_prompts": scores.n_prompts,
                "n_batches": scores.metadata["n_batches"],
                "prompt_build_s": prompt_build_s,
                "attribution_s": scores.metadata["seconds"],
                "reprune_s_per_view": reprune_s,
                "peak_vram_gib": peak,
                "clean_metric_mean": scores.clean_metric_mean,
                "corrupt_metric_mean": scores.corrupt_metric_mean,
                "n_candidate_edges": len(scores.edge_scores),
                "edges_per_view_min": min(len(v[1]) for v in views),
                "edges_per_view_max": max(len(v[1]) for v in views),
                "distinct_edge_sets": len({v[1] for v in views}),
            })
        t_attr = statistics.fmean(r["attribution_s"] for r in runs)
        t_reprune = statistics.fmean(r["reprune_s_per_view"] for r in runs)
        t_build = statistics.fmean(r["prompt_build_s"] for r in runs)
        per_pass = t_build + t_attr + B * t_reprune
        per_task[task_name] = {
            "runs": runs,
            "t_attribution_s_mean": t_attr,
            "t_pass_s_mean": per_pass,
            "per_cell_seconds": {f"S{S}xR{R}": S * (1 + R) * per_pass for S, R in S_R_OPTIONS},
            "run_plan_rule_applied_to_t_attribution": (
                "S=5, R=20" if t_attr <= 120 else "S=5, R=10" if t_attr <= 360 else "S=3, R=10 (+ scope cuts)"
            ),
        }

    report = {
        "non_evidence": True,
        "purpose": "Q3 timing pilot (docs/Run_Plan.md Step 5). Not a Stage A run; must never enter a CSI table or figure.",
        "created": stamp,
        "git_commit": _git_commit(),
        "model": {k: model_cfg.get(k) for k in ("name", "hf_id", "hf_revision", "dtype")},
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "libraries": {p: md.version(p) for p in ("torch", "transformers", "transformer-lens", "huggingface-hub")},
        "B": B,
        "max_batch_size": max_batch_size,
        "model_load_s": load_s,
        "run_plan_rule": RUN_PLAN_RULE,
        "tasks": per_task,
        "note": "B, S and R are a PI decision (Q3). This file reports the pre-stated rule; it decides nothing.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> None:
    # allow_abbrev=False: argparse's prefix matching otherwise accepts `--seed 0` as
    # `--seeds=0`, which silently timed zero seeds. Every deploy script and doc carried
    # that typo. Now it is a loud "unrecognized argument" instead.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--model", default="pythia160m")
    parser.add_argument("--tasks", nargs="+", default=["ioi", "greater_than"])
    parser.add_argument("--seeds", type=int, default=2, help="number of seeds to time (0..n-1)")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--out", default=str(ROOT / "runs" / "pilot_timing"))
    args = parser.parse_args(argv)
    path = run(args.model, args.tasks, args.seeds, args.max_batch_size, Path(args.out))
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"NON-EVIDENCE timing report -> {path}")
    for task, r in report["tasks"].items():
        cells = ", ".join(f"{k} {v/60:.1f} min" for k, v in r["per_cell_seconds"].items())
        print(f"  {task}: t_attribution {r['t_attribution_s_mean']:.2f}s | per cell: {cells} | Run_Plan rule -> {r['run_plan_rule_applied_to_t_attribution']}")


if __name__ == "__main__":
    main()
