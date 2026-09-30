# [AI-GEN] agent=Claude date=2026-09-21 task=Generate the compression grid configs and the Stage B/C run queues from one cell table
# modified: [AI-GEN] agent=Claude date=2026-09-30 task=per-cell run-name setting (freeze discovery) + Stage A queue
# modified: [AI-GEN] agent=Claude date=2026-10-01 task=per-model R from the Q3 rule on Spark timings
# reviewed-by: PENDING
#
# WHY THIS EXISTS
# ---------------
# The compression grid appears in four places: configs/compression/*.yaml, the Stage B
# queue, the Stage C queue, and the freeze cell list. Typing it four times guarantees a
# mismatch, and a mismatch between the frozen null's cell key and Stage C's cell key is
# not a typo you find quickly - Stage C refuses with a hash error and you go looking in
# the wrong place. So the table lives here once and everything else is generated.
#
# Regenerating is safe and idempotent EXCEPT for configs that a run has already
# referenced: those are IMMUTABLE (AI_RULES.md 1.2). The script refuses to overwrite a
# config whose content would change; delete it deliberately if you really mean to.
#
# Usage:
#   python deploy/shared/gen_cells.py configs      # write configs/compression/*.yaml
#   python deploy/shared/gen_cells.py queues       # write deploy/*/cells_*.txt
#   python deploy/shared/gen_cells.py all

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# --- THE GRID -------------------------------------------------------------------------
# 11 cells (proposal 3.2: RTN INT8->INT4, GPTQ, AWQ; magnitude + Wanda 0-60%).
# `cell` is the key used by frozen/{model}/{task}/{cell} and by stage_c.cell - it is the
# identity that ties a Stage C run to its frozen null. Never rename one after a freeze.
CELLS: list[dict] = [
    {"cell": "rtn_int8",     "family": "rtn",       "level": 8,    "kwargs": {"bits": 8}},
    {"cell": "rtn_int6",     "family": "rtn",       "level": 6,    "kwargs": {"bits": 6}},
    {"cell": "rtn_int4",     "family": "rtn",       "level": 4,    "kwargs": {"bits": 4}},
    {"cell": "gptq_int4",    "family": "gptq",      "level": 4,    "kwargs": {"bits": 4}},
    {"cell": "awq_int4",     "family": "awq",       "level": 4,    "kwargs": {"bits": 4}},
    {"cell": "magnitude_20", "family": "magnitude", "level": 0.2,  "kwargs": {"sparsity": 0.2}},
    {"cell": "magnitude_40", "family": "magnitude", "level": 0.4,  "kwargs": {"sparsity": 0.4}},
    {"cell": "magnitude_60", "family": "magnitude", "level": 0.6,  "kwargs": {"sparsity": 0.6}},
    {"cell": "wanda_20",     "family": "wanda",     "level": 0.2,  "kwargs": {"sparsity": 0.2}},
    {"cell": "wanda_40",     "family": "wanda",     "level": 0.4,  "kwargs": {"sparsity": 0.4}},
    {"cell": "wanda_60",     "family": "wanda",     "level": 0.6,  "kwargs": {"sparsity": 0.6}},
]

# Model/task coverage. Plan A = pythia only; Plan B = everything.
PLAN_A_MODELS = ["pythia160m", "pythia410m"]
PLAN_B_MODELS = ["pythia160m", "pythia410m", "gemma2_2b", "llama32_1b"]

# Null draws R where the Q3 rule, applied to the measured Spark timings, departs from
# configs/nulls/default.yaml (R = 20). docs/HUMAN_DECISIONS.md Q3, 2026-10-01: one IOI
# attribution pass on Gemma-2-2B takes 290 s (2 min < t <= 6 min -> S = 5, R = 10); every
# other model is under 2 min (R = 20). Emitted as `nulls.R=<R>` on that model's Stage B AND
# Stage C lines, so the run name (B16xS5xR10) and the Stage C tags carry the R of the frozen
# null. Changing this after a model's freeze means re-running its Stage B (AI_RULES.md 1.3).
MODEL_NULL_R: dict[str, int] = {"gemma2_2b": 10}
TASKS = ["ioi", "greater_than"]

# (model, task) pairs that CANNOT run, with the reason. These are properties of the
# pinned tokenizers, not preferences: greater_than compares a year at its century
# boundary, so it needs " CCYY" to tokenize as exactly [" CC", "YY"] and needs all 100
# two-digit strings "00".."99" to be single tokens. Verified 2026-09-27 against the
# pinned tokenizers; src/tasks/greater_than.py enforces both conditions and raises
# "empty pool under this tokenizer" when they fail.
#
# Emitting them anyway cost 22 of 88 queue entries that fail one at a time, mid-run,
# after the model has been loaded. The grid is 66 cells, and the paper has to say so:
# greater_than is a Pythia-only task here, which weakens the task-generality claim and
# must be reported rather than quietly dropped.
IMPOSSIBLE_CELLS: dict[tuple[str, str], str] = {
    ("gemma2_2b", "greater_than"):
        "Gemma-2 has NO single-token two-digit string and tokenizes ' 1799' as 5 pieces",
    ("llama32_1b", "greater_than"):
        "Llama-3.2 tokenizes ' 1799' as [' ', '179', '9'] - no century/year split",
}


def viable_pairs(models: list[str]) -> list[tuple[str, str]]:
    """(model, task) pairs that can actually run, in queue order."""
    return [
        (model, task)
        for model in models
        for task in TASKS
        if (model, task) not in IMPOSSIBLE_CELLS
    ]

CONFIG_HEADER = """\
# [AI-GEN] agent=Claude date=2026-09-21 task=Compression grid cell (generated by deploy/shared/gen_cells.py)
# reviewed-by: PENDING
#
# Compression grid cell for Stage C (proposal 3.2). IMMUTABLE once referenced by a run
# (AI_RULES.md 1.2): if a Stage B freeze has been taken against this cell, changing any
# value here silently invalidates the frozen null that Stage C will compare against.
#
# NOTE ON HYDRA PACKAGING. This file populates cfg.compression.*. It deliberately does
# NOT set the root keys compression_family / compression_level / stage_c.cell, because
# configs/config.yaml lists `_self_` LAST in its defaults, so its own root values
# (compression_family: dense) would override anything a group file set. The run scripts
# therefore pass those root keys explicitly on the command line, generated from this same
# table so they cannot drift. See deploy/shared/gen_cells.py.
"""


def _fmt_kwargs(kwargs: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in kwargs.items())


def write_configs() -> None:
    out_dir = REPO / "configs" / "compression"
    out_dir.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0
    for c in CELLS:
        path = out_dir / f"{c['cell']}.yaml"
        body = (
            f"{CONFIG_HEADER}\n"
            f"name: {c['cell']}\n"
            f"family: {c['family']}\n"
            f"level: {c['level']}\n"
            f"cell: {c['cell']}\n"
            f"compressor_kwargs:\n{_fmt_kwargs(c['kwargs'])}\n"
        )
        if path.exists() and path.read_text(encoding="utf-8") != body:
            print(f"  REFUSING to overwrite changed immutable config: {path.name}")
            skipped += 1
            continue
        path.write_text(body, encoding="utf-8")
        written += 1
    print(f"configs: {written} written, {skipped} refused, in {out_dir}")


def setting_for(stage: str, cell: str) -> str:
    """The run-name setting token for one cell (CLAUDE.md §4 naming).

    Without it every run inherited ``setting: dense`` from configs/config.yaml, so all 11
    Stage B nulls of a (model, task) shared one run name and differed only by a
    ``__cfg<hash>`` suffix. make_freeze_config.py finds a cell's null by the cell token in
    the run name, so it would have found none of them and refused to build the freeze.
    """
    token = cell.replace("_", "-")
    return f"null-matchedmag-{token}" if stage == "stageB" else token


def _overrides(model: str, task: str, c: dict, stage: str) -> str:
    """The exact Hydra override string for one cell. One source of truth."""
    common = (
        f"model={model} task={task} setting={setting_for(stage, c['cell'])} "
        f"compression={c['cell']} "
        f"compression_family={c['family']} compression_level={c['level']} "
        f"+stage_c.cell={c['cell']} "
        + " ".join(f"+stage_c.compressor_kwargs.{k}={v}" for k, v in c["kwargs"].items())
    )
    if model in MODEL_NULL_R:
        common += f" nulls.R={MODEL_NULL_R[model]}"
    return f"stage={stage} {common}"


def write_stage_a_queues() -> None:
    """One Stage A dense reference per viable (model, task): the reported pre-compression
    circuit (s(e), core/contingent/noise) and the CIRCUS view diagnostics."""
    plans = {
        REPO / "deploy" / "plan_a_local_pc": PLAN_A_MODELS,
        REPO / "deploy" / "plan_b_dgx_spark": PLAN_B_MODELS,
    }
    for plan_dir, models in plans.items():
        pairs = viable_pairs(models)
        lines = [
            "# Generated by deploy/shared/gen_cells.py - do not hand-edit.",
            "# One Stage A dense reference per viable (model, task) pair.",
            f"# {len(pairs)} viable (model, task) pairs = {len(pairs)} cells",
            "",
            *(f"stage=stageA model={model} task={task} setting=dense" for model, task in pairs),
        ]
        path = plan_dir / "cells_stagea.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"queue: {path.relative_to(REPO)}  ({len(pairs)} cells)")


def write_queues() -> None:
    plans = {
        REPO / "deploy" / "plan_a_local_pc": PLAN_A_MODELS,
        REPO / "deploy" / "plan_b_dgx_spark": PLAN_B_MODELS,
    }
    for plan_dir, models in plans.items():
        plan_dir.mkdir(parents=True, exist_ok=True)
        pairs = viable_pairs(models)
        excluded = [
            (model, task)
            for model in models
            for task in TASKS
            if (model, task) in IMPOSSIBLE_CELLS
        ]
        for stage in ("stageB", "stageC"):
            lines = [
                "# Generated by deploy/shared/gen_cells.py - do not hand-edit.",
                "# One cell per line; feed to the runner script. Comments and blanks are skipped.",
                (f"# {len(pairs)} viable (model, task) pairs x {len(CELLS)} cells "
                 f"= {len(pairs) * len(CELLS)} cells"),
            ]
            if excluded:
                full = len(models) * len(TASKS) * len(CELLS)
                lines.append(
                    f"# {full - len(pairs) * len(CELLS)} of {full} cells are EXCLUDED as "
                    "structurally impossible under the pinned tokenizer:"
                )
                for model, task in excluded:
                    lines.append(
                        f"#   {model} {task}: {IMPOSSIBLE_CELLS[(model, task)]} "
                        f"({len(CELLS)} cells)"
                    )
            lines.append("")
            for model, task in pairs:
                for c in CELLS:
                    lines.append(_overrides(model, task, c, stage))
            path = plan_dir / f"cells_{stage.lower()}.txt"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            n = len(pairs) * len(CELLS)
            print(f"queue: {path.relative_to(REPO)}  ({n} cells)")


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("configs", "all"):
        write_configs()
    if what in ("queues", "all"):
        write_queues()
        write_stage_a_queues()
    if what not in ("configs", "queues", "all"):
        print(__doc__ or "usage: gen_cells.py [configs|queues|all]")
        sys.exit(2)


if __name__ == "__main__":
    main()
