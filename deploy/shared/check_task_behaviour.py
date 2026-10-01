# [AI-GEN] agent=Claude date=2026-10-01 task=Does each model DO each task on the pre-registered prompts? Forward pass only, with and without BOS
# modified: [AI-GEN] agent=Claude date=2026-10-01 task=append its verdicts to RESULTS.md (project rule, CLAUDE.md §8)
# reviewed-by: PENDING
#
# WHY THIS EXISTS
# ---------------
# The Spark timing pilot (2026-10-01) printed a clean IOI metric of -4.33 / -4.69 for
# Gemma-2-2B (logit(IO) - logit(S), so NEGATIVE means it prefers the wrong name), against
# +4.97 for Llama-3.2-1B and +4.56 for Pythia-160M. The IOI prompts are pre-registered
# WITHOUT a BOS token (configs/task/ioi.yaml, decided on Pythia), and Gemma-2 is known to
# degrade badly without one (configs/calibration/final.yaml: per-window <bos> alone moves
# its perplexity 192 -> 12). A circuit extracted from a model that does not perform the
# task is a circuit for something else, so this is checked before any Gemma run.
#
# WHAT IT DOES
# ------------
# For each (model, task) it builds the task's prompts exactly as the pipeline does, with
# prepend_bos as pre-registered AND flipped, runs clean and corrupted prompts forward (no
# gradients, no attribution), and reports the mean task metric on each. Minutes, not hours.
#
# NON-EVIDENCE. It changes no config and decides nothing: whether BOS changes for a model
# is a PI decision, recorded in docs/HUMAN_DECISIONS.md before that model's first run.
#
# Usage (repo root):
#   python deploy/shared/check_task_behaviour.py                       # every viable (model, task)
#   python deploy/shared/check_task_behaviour.py --models gemma2_2b    # one model

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_cells import PLAN_B_MODELS, viable_pairs

BUILDERS = {
    "ioi": ("src.tasks.ioi", "build_ioi_prompts"),
    "greater_than": ("src.tasks.greater_than", "build_greater_than_prompts"),
}


def _load(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def measure(model: Any, prompts: Any) -> dict[str, float]:
    """Mean task metric on clean and on corrupted prompts (forward only)."""
    import torch

    clean_sum = corrupt_sum = 0.0
    n = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in prompts.batches:
            clean_sum += float(batch.metric_sum(model(batch.clean.to(device))))
            corrupt_sum += float(batch.metric_sum(model(batch.corrupt.to(device))))
            n += batch.n
    return {"clean_mean": clean_sum / n, "corrupt_mean": corrupt_sum / n, "n_prompts": n}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Task behaviour check with and without BOS (non-evidence).")
    ap.add_argument("--models", nargs="+", default=PLAN_B_MODELS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batch-size", type=int, default=32)
    ap.add_argument("--out", default=str(REPO / "runs" / "pilot_behaviour"))
    args = ap.parse_args(argv)

    import importlib

    from src.extraction.real_model import load_pinned_model

    mode = _load(REPO / "configs" / "mode" / "scientific_run.yaml")  # the download gate only
    report: dict[str, Any] = {
        "non_evidence": True,
        "purpose": "Does the model perform the task on the pre-registered prompts? Decides nothing.",
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                     capture_output=True, text=True, check=False).stdout.strip() or None,
        "seed": args.seed,
        "results": [],
    }
    print(f"{'model':<12} {'task':<13} {'BOS':<22} {'clean':>9} {'corrupt':>9}  verdict")
    for model_key in args.models:
        tasks = [t for m, t in viable_pairs([model_key])]
        model_cfg = _load(REPO / "configs" / "model" / f"{model_key}.yaml")
        model = load_pinned_model({"model": model_cfg, "mode": mode})
        for task_name in tasks:
            module, fn = BUILDERS[task_name]
            build = getattr(importlib.import_module(module), fn)
            task_cfg = _load(REPO / "configs" / "task" / f"{task_name}.yaml")
            registered = bool(task_cfg["prepend_bos"])
            for bos in (registered, not registered):
                cfg = {**task_cfg, "prepend_bos": bos}
                prompts = build(model.tokenizer, cfg, args.seed, max_batch_size=args.max_batch_size)
                m = measure(model, prompts)
                verdict = "does the task" if m["clean_mean"] > 0 and m["clean_mean"] > m["corrupt_mean"] else "DOES NOT do the task"
                label = f"{bos} ({'pre-registered' if bos == registered else 'flipped'})"
                print(f"{model_key:<12} {task_name:<13} {label:<22} {m['clean_mean']:>9.3f} {m['corrupt_mean']:>9.3f}  {verdict}")
                report["results"].append({"model": model_key, "task": task_name, "prepend_bos": bos,
                                          "pre_registered": bos == registered, **m, "verdict": verdict})
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110 - best effort between models
            pass

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{stamp}_behaviour.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nNON-EVIDENCE report -> {path}")
    _record_in_results(report, path)
    return 0


def _record_in_results(report: dict[str, Any], path: Path) -> None:
    """Project rule (CLAUDE.md §8): one timestamped RESULTS.md row per (model, task)."""
    try:
        from record_result import append_row

        try:
            where = f"`{path.resolve().relative_to(REPO).as_posix()}`"
        except ValueError:
            where = f"`{path}`"
        by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in report["results"]:
            by_pair.setdefault((r["model"], r["task"]), []).append(r)
        for (model_key, task_name), rs in by_pair.items():
            text = "; ".join(
                f"BOS {r['prepend_bos']} ({'pre-registered' if r['pre_registered'] else 'flipped'}): "
                f"clean {r['clean_mean']:.3f}, corrupt {r['corrupt_mean']:.3f} -> {r['verdict']}"
                for r in rs
            )
            append_row("pilot-behaviour", model_key, task_name, "dense",
                       f"NON-EVIDENCE, seed {report['seed']}, n={rs[0]['n_prompts']}: {text}", where)
        print("rows added to RESULTS.md")
    except Exception as exc:  # noqa: BLE001 - logging must never fail the check
        print(f"(could not add RESULTS.md rows: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    sys.exit(main())
