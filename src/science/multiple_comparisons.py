# [AI-GEN] agent=Claude date=2026-08-08 task=Benjamini-Hochberg FDR control (AI_RULES.md 4.3 named it as the default; it did not exist)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY
#
# NOVELTY PROTECTION ZONE (AI_RULES.md §3): draft for review only; do not treat as
# approved scientific logic until human approval (APPROVE multiple_comparisons).

"""Multiple-comparison control (AI_RULES.md 4.3).

AI_RULES.md 4.3, verbatim: *"The grid is large (models × tasks × methods × levels × 2
comparison levels). Headline claims aggregate across cells rather than cherry-picking
cells; any per-cell significance statement uses a stated correction
(Benjamini-Hochberg by default). Never scan the grid for the most dramatic cell and
lead with it."*

Why this is not optional here. The grid is roughly
4 models × 3 tasks × ~12 compression cells × 2 comparison levels ≈ 288 per-cell
statements. At α = 0.05 with no correction you expect **~14 "significant" cells even if
compression does nothing at all**. Finding a handful of dramatic cells in a grid that
size is the *expected* outcome under the null, not evidence. Benjamini-Hochberg
controls the false discovery rate: of the cells you call significant, at most a
q-fraction are expected to be false positives.

BH is the right family here rather than Bonferroni: the cells are not independent
(the same model and task recur across compression settings), the question is
"which cells moved" rather than "did anything move anywhere", and Bonferroni over ~288
tests would be so conservative that a real effect in a single cell could never surface.
BH under positive dependence is the standard choice for exactly this shape.

⚠️ TODO [QUESTION FOR PI]: fix the FDR level q before Stage C and record it with the
other pre-registered decisions (AI_RULES.md 4.2). q = 0.05 is the default below, but
it is a decision, not a fact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

DEFAULT_FDR_Q = 0.05  # ⚠️ provisional; pre-register before Stage C (AI_RULES.md 4.2)


def benjamini_hochberg(p_values: Sequence[float], q: float = DEFAULT_FDR_Q) -> dict[str, object]:
    """Benjamini-Hochberg step-up procedure at FDR level ``q``.

    Returns ``{"rejected": [bool, ...], "adjusted": [float, ...], "n_rejected": int,
    "threshold": float, "q": float}``, aligned to the input order.

    ``adjusted`` are BH-adjusted p-values (q-values), made monotone by the standard
    running minimum from the largest p downward and clipped to 1. ``threshold`` is the
    largest raw p-value that was rejected (0.0 if none), i.e. the effective cutoff.

    An empty input returns empty results rather than raising: a grid with nothing to
    test is a legitimate state (e.g. every cell was cut under the scope-cut order).
    """
    if not (0.0 < q < 1.0):
        raise ValueError(f"FDR level q must be in (0, 1), got {q}")
    p = np.asarray(list(p_values), dtype=np.float64)
    if p.size == 0:
        return {"rejected": [], "adjusted": [], "n_rejected": 0, "threshold": 0.0, "q": float(q)}
    if np.any(~np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p-values must all be finite and within [0, 1]")

    n = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)

    # step-up: largest i with p_(i) <= (i/n)·q; everything up to i is rejected
    passes = ranked <= (ranks / n) * q
    k = int(np.max(np.nonzero(passes)[0]) + 1) if np.any(passes) else 0

    rejected_sorted = np.zeros(n, dtype=bool)
    rejected_sorted[:k] = True

    # BH-adjusted p-values, monotone from the top down
    adjusted_sorted = np.minimum.accumulate((ranked * n / ranks)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    rejected = np.empty(n, dtype=bool)
    adjusted = np.empty(n, dtype=np.float64)
    rejected[order] = rejected_sorted
    adjusted[order] = adjusted_sorted

    return {
        "rejected": rejected.tolist(),
        "adjusted": adjusted.tolist(),
        "n_rejected": int(k),
        "threshold": float(ranked[k - 1]) if k > 0 else 0.0,
        "q": float(q),
    }


def correct_cell_grid(
    cell_p_values: Mapping[str, float], q: float = DEFAULT_FDR_Q
) -> dict[str, dict[str, object]]:
    """Apply BH across a {cell_id: p_value} grid; return {cell_id: {p, q_value, significant}}.

    Use this for the whole reported grid at once. Correcting per-model or per-task
    subsets and then reporting the union is the same cherry-picking AI_RULES.md 4.3
    forbids, just spread over more steps.
    """
    keys = sorted(cell_p_values)
    result = benjamini_hochberg([cell_p_values[k] for k in keys], q=q)
    return {
        key: {
            "p": float(cell_p_values[key]),
            "q_value": float(result["adjusted"][i]),      # type: ignore[index]
            "significant": bool(result["rejected"][i]),   # type: ignore[index]
        }
        for i, key in enumerate(keys)
    }


def expected_false_positives(n_tests: int, alpha: float = 0.05) -> float:
    """How many "significant" cells an uncorrected grid yields under the global null.

    The number to put next to any uncorrected count so a reader can see immediately
    whether the finding exceeds what noise alone supplies.
    """
    if n_tests < 0:
        raise ValueError(f"n_tests must be >= 0, got {n_tests}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return float(n_tests) * float(alpha)


__all__ = [
    "DEFAULT_FDR_Q",
    "benjamini_hochberg",
    "correct_cell_grid",
    "expected_false_positives",
]
