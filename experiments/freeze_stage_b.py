# [AI-GEN] agent=Claude date=2026-08-08 task=Stage B freeze entrypoint (the pre-registration event had no runnable path)
# reviewed-by: PENDING

"""THE FREEZE — Algorithm 1's pre-registration point (proposal §4 rule 1; AI_RULES.md 1.4).

This is the most consequential command in the project. After it runs, the null
distributions it registers may never be re-tuned, re-drawn, re-parameterised or
"corrected" for any cell whose real-compression counterpart has been extracted. That
irreversibility is the point: it is what lets the paper say the floor was fixed before
the effect was measured.

Policy enforced here (mechanics live in src/common/freeze.py):

1. ``mode=scientific_run``. The freeze is not an engineering operation. Engineering
   mode refuses it (config_guard), and so does this entrypoint.
2. ``stage_b.freeze_approved: true`` must be set EXPLICITLY. There is no default and
   no flag that turns it on implicitly.
3. Every Stage-A-gating decision must be resolved (Q1/Q3/Q4/Q5/Q6). Freezing a null
   built on provisional values would pre-register a placeholder.
4. Q2 (the distance D) must be pre-registered. CSI is entirely downstream of D, so a
   floor frozen before D is chosen is not a floor for anything in particular.
5. Each cell is appended, never overwritten, and re-hashed at the destination.
6. The whole store is re-verified after the append; a store that fails verification
   aborts the freeze.

Usage::

    python -m experiments.freeze_stage_b --dry-run    # report what WOULD be frozen
    python -m experiments.freeze_stage_b              # requires an approved config

Run the dry-run first, read what it says, then commit the result and tag it. The tag
is your pre-registration reference (PRD.md §6 artifact checklist).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from experiments.run_stage_a import _git_commit, _resolve
from src.common.config_guard import MODE_SCIENTIFIC, assert_no_gating_questions, mode_of
from src.common.freeze import FreezeViolation, cell_key, freeze_cell, read_manifest, verify_frozen_store


def _require_freeze_preconditions(resolved: dict[str, Any]) -> None:
    """Every condition that must hold before anything is written to frozen/."""
    if mode_of(resolved) != MODE_SCIENTIFIC:
        raise FreezeViolation(
            "the freeze requires mode=scientific_run. The freeze is the pre-registration "
            "event, not an engineering dry-run (AI_RULES.md 1.4)."
        )

    if not bool((resolved.get("stage_b") or {}).get("freeze_approved")):
        raise FreezeViolation(
            "the freeze requires an EXPLICIT approval: set stage_b.freeze_approved=true. "
            "There is deliberately no default — freezing is irreversible."
        )

    # All Stage-A-gating decisions must be final (raises with the list if not).
    assert_no_gating_questions(resolved, str(resolved.get("pipeline", "")))

    distance = resolved.get("distance") or {}
    if not bool(distance.get("pre_registered_for_stage_c")):
        raise FreezeViolation(
            "the distance function D is not pre-registered "
            "(configs/distance/*.yaml: pre_registered_for_stage_c is false). CSI is "
            "entirely downstream of D, so a null frozen before D is chosen is not a "
            "floor for anything in particular. See docs/HUMAN_DECISIONS.md Q2."
        )


def plan_freeze(resolved: dict[str, Any]) -> list[dict[str, str]]:
    """The cells this config would freeze: [{cell_key, source_run_dir}, ...].

    ``stage_b.cells`` is a list of ``{model, task, cell, source_run_dir}`` entries —
    one per (model, task, compression cell) whose Stage B draft has been produced.
    """
    cells = (resolved.get("stage_b") or {}).get("cells") or []
    if not cells:
        raise FreezeViolation(
            "nothing to freeze: stage_b.cells is empty. List the Stage B draft runs to "
            "freeze as {model, task, cell, source_run_dir} entries."
        )
    plan = []
    for c in cells:
        for field in ("model", "task", "cell", "source_run_dir"):
            if not c.get(field):
                raise FreezeViolation(f"stage_b.cells entry missing {field!r}: {c}")
        plan.append({
            "cell_key": cell_key(c["model"], c["task"], c["cell"]),
            "source_run_dir": str(c["source_run_dir"]),
            "model": c["model"], "task": c["task"], "cell": c["cell"],
        })
    return plan


def run_freeze(cfg: Any, dry_run: bool = False) -> dict[str, Any]:
    """Freeze every cell listed in ``stage_b.cells``. Returns a freeze report.

    With ``dry_run=True`` nothing is written: preconditions are checked, the plan is
    reported, and the store is verified as-is. Always do this first.
    """
    resolved = _resolve(cfg)
    frozen_root = Path(resolved.get("frozen_root", "frozen"))

    if dry_run:
        problems: list[str] = []
        try:
            _require_freeze_preconditions(resolved)
        except (FreezeViolation, ValueError) as exc:
            problems.append(str(exc))
        try:
            plan = plan_freeze(resolved)
        except FreezeViolation as exc:
            plan, _ = [], problems.append(str(exc))
        already = set(read_manifest(frozen_root).get("cells", {}))
        return {
            "dry_run": True,
            "would_freeze": [p for p in plan if p["cell_key"] not in already],
            "already_frozen": sorted(already),
            "blocking_problems": problems,
            "store_verification": verify_frozen_store(frozen_root),
        }

    _require_freeze_preconditions(resolved)
    plan = plan_freeze(resolved)
    commit = _git_commit()
    if commit is None:
        print(
            "[freeze] WARNING: no git commit available. The freeze is supposed to BE a "
            "commit (AI_RULES.md 1.4) — `git init` and commit before freezing, or the "
            "pre-registration has no anchor.",
            file=sys.stderr,
        )

    frozen: list[dict[str, Any]] = []
    for entry in plan:
        meta = freeze_cell(
            frozen_root, entry["model"], entry["task"], entry["cell"],
            entry["source_run_dir"], freeze_commit=commit,
            extra_meta={"config_hash_at_freeze": resolved.get("config_hash")},
        )
        frozen.append({"cell": entry["cell_key"], "null_frozen_hash": meta["null_frozen_hash"]})
        print(f"[freeze] {entry['cell_key']} -> {meta['null_frozen_hash'][:12]}")

    verification = verify_frozen_store(frozen_root)
    if not verification["ok"]:
        raise FreezeViolation(
            f"the frozen store failed verification immediately after the freeze: "
            f"{verification['problems']}. Do NOT run Stage C against it."
        )

    report = {
        "dry_run": False,
        "freeze_commit": commit,
        "frozen_cells": frozen,
        "store_verification": verification,
    }
    print(
        f"[freeze] FROZEN {len(frozen)} cell(s) at commit {commit or 'UNKNOWN'}. "
        f"frozen/ is now append-only for these cells. Tag this commit — it is your "
        f"pre-registration reference."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Stage B null distributions (irreversible).")
    parser.add_argument("--config", required=True, help="path to a resolved config JSON")
    parser.add_argument("--dry-run", action="store_true", help="report the plan; write nothing")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    report = run_freeze(cfg, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
