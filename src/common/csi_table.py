# [AI-GEN] agent=Claude date=2026-08-08 task=Real CSI table writer (only write_csi_placeholder existed; nothing could emit a real row)
# reviewed-by: PENDING

"""The CSI table — the paper's result object (ARCHITECTURE.md §2 schema).

Schema (fixed in ARCHITECTURE.md §2)::

    (model, task, family, level_param, comparison_level,
     csi, ci_lo, ci_hi, D, dnull_median, null_frozen_hash)

Three rules are enforced on the way in, because they are the ones that would otherwise
be violated silently by an analysis script:

1. **Every row carries its interval.** ``csi`` without ``ci_lo``/``ci_hi`` is rejected
   (AI_RULES.md 4.1: "A number without an interval does not enter a figure").
2. **Every row carries its frozen-null hash.** A CSI is a division by a specific frozen
   null; a row that cannot name which one is unauditable (AI_RULES.md 1.4).
3. **Both comparison levels or neither.** ``assert_both_levels_present`` refuses a
   table that reports a cell at exact-edge but not routing-head, or vice versa
   (proposal §4 rule 3: "Two levels or it does not count").

Placeholder rows can never share a file with real rows (AI_RULES.md 2.3): the
placeholder writer lives in ``schema.write_csi_placeholder`` and writes a file that is
visibly marked, and this module refuses to append to it.
"""

from __future__ import annotations

import csv as _csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .schema import CSI_TABLE_COLUMNS

COMPARISON_LEVELS: tuple[str, ...] = ("exact_edge", "routing_head")
_PLACEHOLDER_MARKER = "# PLACEHOLDER"


def make_csi_row(
    model: str,
    task: str,
    family: str,
    level_param: str,
    comparison_level: str,
    csi_result: Mapping[str, float],
    null_frozen_hash: str,
    d_normalized: float | None = None,
) -> dict[str, Any]:
    """Build one validated CSI row from a ``science.csi.csi()`` result.

    ``csi_result`` is the dict returned by ``csi()``: {csi, ci_lo, ci_hi, D,
    dnull_median}. Keeping this the only construction path means the table can never
    drift from the metric that produced it (ARCHITECTURE.md §3: metrics are centralized,
    never re-implemented inline).
    """
    if comparison_level not in COMPARISON_LEVELS:
        raise ValueError(
            f"comparison_level must be one of {COMPARISON_LEVELS}, got {comparison_level!r}"
        )
    row = {
        "model": str(model),
        "task": str(task),
        "family": str(family),
        "level_param": str(level_param),
        "comparison_level": comparison_level,
        "csi": float(csi_result["csi"]),
        "ci_lo": float(csi_result["ci_lo"]),
        "ci_hi": float(csi_result["ci_hi"]),
        "D": float(csi_result["D"]),
        # Q2's third column. Optional so a caller that has not computed it writes an
        # explicit blank rather than a fabricated number; Stage C always supplies it.
        "d_normalized": None if d_normalized is None else float(d_normalized),
        "dnull_median": float(csi_result["dnull_median"]),
        "null_frozen_hash": str(null_frozen_hash),
    }
    validate_csi_row(row)
    return row


def validate_csi_row(row: Mapping[str, Any]) -> None:
    """Raise ValueError unless the row is complete, interval-bearing and hash-anchored."""
    missing = [c for c in CSI_TABLE_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"CSI row missing columns: {missing}")

    for key in ("csi", "ci_lo", "ci_hi", "D", "dnull_median"):
        value = row[key]
        if value is None or not isinstance(value, (int, float)):
            raise ValueError(f"CSI row: {key} must be numeric, got {value!r} (AI_RULES.md 4.1)")

    if not (row["ci_lo"] <= row["csi"] <= row["ci_hi"]):
        raise ValueError(
            f"CSI row: interval [{row['ci_lo']}, {row['ci_hi']}] does not bracket the point "
            f"estimate {row['csi']} — the bootstrap and the point estimate disagree"
        )
    if row["csi"] < 0 or row["D"] < 0 or row["dnull_median"] < 0:
        raise ValueError(
            f"CSI row: csi={row['csi']}, D={row['D']}, dnull_median={row['dnull_median']} — "
            f"CSI is a ratio of distances and none of these may be negative"
        )
    if row["dnull_median"] == 0:
        raise ValueError("CSI row: dnull_median is 0; the division is undefined (see science/csi.py)")
    if not row.get("null_frozen_hash"):
        raise ValueError(
            "CSI row: null_frozen_hash is empty. Every CSI must name the frozen null it "
            "divided by, or it cannot be audited (AI_RULES.md 1.4)."
        )
    if row["comparison_level"] not in COMPARISON_LEVELS:
        raise ValueError(f"CSI row: bad comparison_level {row['comparison_level']!r}")


def cell_identity(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """The (model, task, family, level_param) key a cell is reported under."""
    return (str(row["model"]), str(row["task"]), str(row["family"]), str(row["level_param"]))


def assert_both_levels_present(rows: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a table that reports any cell at only one comparison level.

    Proposal §4 rule 3 ("Two levels or it does not count") is a reporting rule, and a
    reporting rule that is not enforced by the writer is a reporting rule that will be
    broken by the last figure produced before a deadline.
    """
    seen: dict[tuple[str, str, str, str], set[str]] = {}
    for row in rows:
        seen.setdefault(cell_identity(row), set()).add(str(row["comparison_level"]))
    incomplete = {k: sorted(v) for k, v in seen.items() if set(v) != set(COMPARISON_LEVELS)}
    if incomplete:
        lines = [f"  {'/'.join(k)}: only {v}" for k, v in sorted(incomplete.items())]
        raise ValueError(
            "CSI table reports cells at only one comparison level:\n"
            + "\n".join(lines)
            + "\nEvery claim is stated at BOTH the exact-edge and routing-head level "
              "(proposal §4 rule 3; claim C2)."
        )


def write_csi_table(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    require_both_levels: bool = True,
) -> Path:
    """Write a validated CSI table as CSV (PRD.md §6 ships CSI grids as CSV).

    Refuses to overwrite a placeholder file (AI_RULES.md 2.3: placeholder values can
    never share a file with real results) and refuses an empty table, which is nearly
    always a silent upstream failure rather than a real result.
    """
    out = Path(path)
    if not rows:
        raise ValueError("refusing to write an empty CSI table; check the upstream stage")
    for row in rows:
        validate_csi_row(row)
    if require_both_levels:
        assert_both_levels_present(rows)

    if out.exists():
        first = out.read_text(encoding="utf-8").splitlines()[:1]
        if first and first[0].startswith(_PLACEHOLDER_MARKER):
            raise ValueError(
                f"{out} is a PLACEHOLDER file; real results may never share a file with "
                f"placeholders (AI_RULES.md 2.3). Write to a new path."
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(CSI_TABLE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in CSI_TABLE_COLUMNS})
    return out


def read_csi_table(path: str | Path) -> list[dict[str, Any]]:
    """Read a CSI table back, coercing the numeric columns."""
    numeric = {"csi", "ci_lo", "ci_hi", "D", "dnull_median"}
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = [
            {k: (float(v) if k in numeric else v) for k, v in row.items()}
            for row in _csv.DictReader(f)
            if not (row.get("model", "") or "").startswith("#")
        ]
    return rows


def summarize_csi(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Grid-level summary, in the language of the paper's decision rule (PRD.md §1 C1).

    A cell is "structurally selective" when its CI excludes 1 from above, "gentler than
    noise" when it excludes 1 from below, and "indistinguishable from noise" when the CI
    contains 1 — the pre-committed publishable negative (AI_RULES.md 4.5).
    """
    selective = [r for r in rows if r["ci_lo"] > 1.0]
    gentler = [r for r in rows if r["ci_hi"] < 1.0]
    indistinguishable = [r for r in rows if r["ci_lo"] <= 1.0 <= r["ci_hi"]]
    return {
        "n_rows": len(rows),
        "n_cells": len({cell_identity(r) for r in rows}),
        "structurally_selective": len(selective),
        "gentler_than_noise": len(gentler),
        "indistinguishable_from_noise": len(indistinguishable),
        "levels_disagree": sorted(
            "/".join(k) for k, v in _by_cell(rows).items()
            if len({_verdict(r) for r in v}) > 1
        ),
    }


def _by_cell(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], list[Mapping[str, Any]]]:
    out: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        out.setdefault(cell_identity(row), []).append(row)
    return out


def _verdict(row: Mapping[str, Any]) -> str:
    if row["ci_lo"] > 1.0:
        return "selective"
    if row["ci_hi"] < 1.0:
        return "gentler"
    return "indistinguishable"


__all__ = [
    "COMPARISON_LEVELS",
    "make_csi_row",
    "validate_csi_row",
    "assert_both_levels_present",
    "write_csi_table",
    "read_csi_table",
    "summarize_csi",
    "cell_identity",
]
