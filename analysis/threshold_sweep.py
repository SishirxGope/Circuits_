# [AI-GEN] agent=Claude date=2026-08-08 task=Threshold-sensitivity sweep (claim C3 / PRD §3.3 required it; it did not exist)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY

"""Threshold-sensitivity sweep (claim C3; proposal §3.3; PRD.md §1).

PRD.md §1 C3 decision rule, verbatim: *"A headline conclusion must survive the full
threshold sweep or be flagged threshold-dependent."* Proposal §3.3 requires
*"sensitivity of every headline conclusion to the pruning threshold, reported as an
explicit sweep rather than a single operating point."*

This is not a robustness footnote — it is the paper's own thesis applied to itself.
CIRCUS's finding is that two pruning thresholds yield two different circuits with no
principled way to choose between them. A paper that says so and then reports its own
CSI at one threshold has not taken its own point seriously. The sweep is what converts
"here is the number" into "here is the range of numbers an analyst could have obtained,
and here is whether the conclusion is stable across it."

What the sweep produces per conclusion:

- the CSI (and its CI) at every operating point in the ensemble's threshold range;
- the **verdict** at each point (selective / gentler / indistinguishable);
- ``stable``: whether the verdict is the same at every point;
- ``flip_points``: where it changes, if it does.

A conclusion with ``stable: False`` is reported as threshold-dependent in the paper.
That is a finding, not a failure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.science.csi import csi
from src.science.distances import l1_distance
from src.science.two_level import project_to_routing_heads


def verdict_for(ci_lo: float, ci_hi: float) -> str:
    """The C1 decision rule applied to one interval (PRD.md §1).

    Identical wording to ``csi_table.summarize_csi`` on purpose: the sweep must judge
    stability with the exact rule the headline table uses, or "stable" means nothing.
    """
    if ci_lo > 1.0:
        return "structurally-selective"
    if ci_hi < 1.0:
        return "gentler-than-noise"
    return "indistinguishable-from-noise"


def sweep_csi_over_cutoffs(
    freq_dense: Mapping[str, float],
    freq_post: Mapping[str, float],
    d_null: Sequence[float],
    cutoffs: Sequence[float],
    comparison_level: str = "exact_edge",
    n_boot: int = 500,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """CSI at each inclusion-frequency cutoff, at one comparison level.

    The cutoff filters which edges count as present before the distance is taken, so
    sweeping it is the reporting-side analogue of sweeping the extraction threshold:
    both ask "does the conclusion depend on where the analyst drew the line?"
    """
    if comparison_level not in ("exact_edge", "routing_head"):
        raise ValueError(f"unknown comparison_level {comparison_level!r}")
    pre_v, post_v = (
        (dict(freq_dense), dict(freq_post))
        if comparison_level == "exact_edge"
        else (project_to_routing_heads(freq_dense), project_to_routing_heads(freq_post))
    )

    out: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        pre = {k: v for k, v in pre_v.items() if v > cutoff}
        post = {k: v for k, v in post_v.items() if v > cutoff}
        d_c = l1_distance(pre, post)
        try:
            result = csi(d_c, list(d_null), n_boot=n_boot, seed=seed)
        except ValueError:  # degenerate null at this operating point
            out.append({"cutoff": float(cutoff), "comparison_level": comparison_level,
                        "csi": None, "ci_lo": None, "ci_hi": None, "D": d_c,
                        "verdict": "undefined", "n_edges_pre": len(pre), "n_edges_post": len(post)})
            continue
        out.append({
            "cutoff": float(cutoff), "comparison_level": comparison_level,
            "csi": result["csi"], "ci_lo": result["ci_lo"], "ci_hi": result["ci_hi"],
            "D": result["D"],
            "verdict": verdict_for(result["ci_lo"], result["ci_hi"]),
            "n_edges_pre": len(pre), "n_edges_post": len(post),
        })
    return out


def conclusion_survives_sweep(sweep_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Does one conclusion hold at every operating point? (PRD.md §1 C3 decision rule.)

    Returns {stable, verdicts, flip_points, csi_min, csi_max, n_points}. ``undefined``
    points (degenerate null) are excluded from the stability judgement but reported, so
    a sweep that is mostly undefined cannot masquerade as stable.
    """
    defined = [r for r in sweep_rows if r.get("verdict") not in (None, "undefined")]
    verdicts = [str(r["verdict"]) for r in defined]
    unique = sorted(set(verdicts))
    flips = [
        {"from_cutoff": defined[i - 1]["cutoff"], "to_cutoff": defined[i]["cutoff"],
         "from": verdicts[i - 1], "to": verdicts[i]}
        for i in range(1, len(defined)) if verdicts[i] != verdicts[i - 1]
    ]
    csis = [float(r["csi"]) for r in defined if r.get("csi") is not None]
    return {
        "stable": len(unique) <= 1 and len(defined) > 0,
        "verdicts": unique,
        "flip_points": flips,
        "csi_min": min(csis) if csis else None,
        "csi_max": max(csis) if csis else None,
        "n_points": len(sweep_rows),
        "n_defined": len(defined),
        "report_as": (
            "threshold-independent" if len(unique) <= 1 and defined
            else "THRESHOLD-DEPENDENT — must be flagged as such (PRD.md §1 C3)"
        ),
    }


def sweep_report(
    freq_dense: Mapping[str, float],
    freq_post: Mapping[str, float],
    d_null: Sequence[float],
    cutoffs: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9),
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """The full C3 object: the sweep at BOTH comparison levels, plus stability verdicts."""
    out: dict[str, Any] = {"cutoffs": [float(c) for c in cutoffs], "levels": {}}
    for level in ("exact_edge", "routing_head"):
        rows = sweep_csi_over_cutoffs(freq_dense, freq_post, d_null, cutoffs,
                                      comparison_level=level, n_boot=n_boot, seed=seed)
        out["levels"][level] = {"sweep": rows, "stability": conclusion_survives_sweep(rows)}
    both_stable = all(v["stability"]["stable"] for v in out["levels"].values())
    out["headline_survives_sweep"] = both_stable
    out["note"] = (
        "Conclusion is stable across the reported threshold range at both levels."
        if both_stable else
        "Conclusion is THRESHOLD-DEPENDENT at one or both levels and must be reported "
        "as such (PRD.md §1 C3). This is a finding about circuit measurement, not a bug."
    )
    return out


__all__ = ["verdict_for", "sweep_csi_over_cutoffs", "conclusion_survives_sweep", "sweep_report"]
