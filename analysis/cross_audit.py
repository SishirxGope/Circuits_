# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft cross-audit rank agreement (proposal §2.6)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY
#
# NOVELTY PROTECTION ZONE (AI_RULES.md §3): draft for review only; do not treat as
# approved scientific logic until human approval (APPROVE cross_audit).

"""Cross-audit agreement (proposal §2.6; claim C5).

Spearman rank correlation between the circuit-level damage ranking (our Stage C
cells, ranked by D(c) or CSI) and the feature-level damage ranking from the
published audits of the same models (refs [1]/[2]: SAE feature survival under
pruning). Implemented without scipy (numpy only, average ranks for ties) so the
analysis runs in the same environment as everything else.

The published feature-damage rankings themselves are PI-provided inputs (they must
be verified against the actual papers, AI_RULES.md 2.2); this module only implements
the mechanical agreement statistic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (standard Spearman tie handling)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rank(xs: Mapping[str, float] | Sequence[float], ys: Mapping[str, float] | Sequence[float]) -> float:
    """Spearman rho between two value collections.

    Accepts either two aligned sequences, or two mappings whose intersection of keys
    defines the aligned pairs (used for cross-audit: same models/cells in both
    rankings). Returns NaN if fewer than 2 aligned pairs.
    """
    if isinstance(xs, Mapping) and isinstance(ys, Mapping):
        keys = sorted(set(xs) & set(ys))
        x_vals = np.asarray([float(xs[k]) for k in keys], dtype=np.float64)
        y_vals = np.asarray([float(ys[k]) for k in keys], dtype=np.float64)
    else:
        x_vals = np.asarray(list(xs), dtype=np.float64)
        y_vals = np.asarray(list(ys), dtype=np.float64)
    if x_vals.size < 2 or x_vals.size != y_vals.size:
        return float("nan")
    rx, ry = _average_ranks(x_vals), _average_ranks(y_vals)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def circuit_vs_feature_damage_rank_agreement(
    circuit_damage: Mapping[str, float],
    feature_damage: Mapping[str, float],
) -> dict[str, float]:
    """Spearman(circuit damage rank, feature damage rank) over shared cells.

    Keys identify cells (e.g. "pythia160m/ioi/rtn-int4"); the mapping values are the
    damage magnitudes per cell in each audit. The agreement is reported together with
    the number of shared cells so a thin overlap is visible (no invented numbers).

    POINT ESTIMATE ONLY — AI_RULES.md 4.1 forbids a number without an interval from
    entering a figure. Use ``cross_audit_report`` for the reportable object.
    """
    shared = sorted(set(circuit_damage) & set(feature_damage))
    rho = spearman_rank(
        {k: circuit_damage[k] for k in shared},
        {k: feature_damage[k] for k in shared},
    )
    return {"spearman": rho, "n_shared_cells": len(shared)}


# -----------------------------------------------------------------------------------
# Interval estimation (PRD.md §1 C5: "Spearman rho (+ CI via permutation)";
# AI_RULES.md 4.1: "A number without an interval does not enter a figure.")
# -----------------------------------------------------------------------------------
#
# Two different questions, two different procedures — they are routinely conflated:
#
#   permutation test -> "could this rho have arisen with NO association at all?"
#                       Shuffle one ranking, rebuild the null, read off a p-value.
#   bootstrap CI     -> "how precisely is rho pinned down by this many cells?"
#                       Resample cells with replacement, read off percentiles.
#
# C5's decision rule (rho >= 0.8 proxy-valid / rho <= 0.4 audits-disagree) is a
# statement about the VALUE of rho, so it needs the CI. The p-value is what stops a
# thin overlap of 4 cells being read as agreement. Report both.

def spearman_permutation_test(
    xs: Mapping[str, float],
    ys: Mapping[str, float],
    n_permutations: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """Two-sided permutation test for Spearman rho. Deterministic given ``seed``.

    Returns {rho, p_value, n_permutations, n_pairs}. The p-value is
    ``(1 + #{|rho_perm| >= |rho_obs|}) / (1 + n_permutations)``; the +1 keeps it
    strictly positive, because a finite simulation can never justify p = 0.

    With few shared cells this test is the honest brake on over-reading C5: with 4
    cells there are only 24 distinct permutations, so p cannot fall below ~0.04 no
    matter how perfect the correlation looks.
    """
    from src.common.seeding import create_seed_generator, derive_child_seed

    keys = sorted(set(xs) & set(ys))
    x = np.asarray([float(xs[k]) for k in keys], dtype=np.float64)
    y = np.asarray([float(ys[k]) for k in keys], dtype=np.float64)
    n = x.size
    observed = spearman_rank(x, y)
    if n < 3 or not np.isfinite(observed):
        return {"rho": observed, "p_value": float("nan"), "n_permutations": 0.0, "n_pairs": float(n)}

    rng, _ = create_seed_generator(derive_child_seed(int(seed), "spearman-permutation", n))
    count = 0
    for _ in range(int(n_permutations)):
        permuted = spearman_rank(x, rng.permutation(y))
        if np.isfinite(permuted) and abs(permuted) >= abs(observed) - 1e-12:
            count += 1
    return {
        "rho": float(observed),
        "p_value": float((1 + count) / (1 + int(n_permutations))),
        "n_permutations": float(int(n_permutations)),
        "n_pairs": float(n),
    }


def spearman_bootstrap_ci(
    xs: Mapping[str, float],
    ys: Mapping[str, float],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for Spearman rho, resampling CELLS with replacement.

    Returns {rho, ci_lo, ci_hi, n_boot, n_pairs}. Resamples that degenerate (all one
    cell, hence zero rank variance) are dropped rather than counted as rho = 0, which
    would drag the interval toward the middle and make a thin overlap look better
    determined than it is.
    """
    from src.common.seeding import create_seed_generator, derive_child_seed

    keys = sorted(set(xs) & set(ys))
    x = np.asarray([float(xs[k]) for k in keys], dtype=np.float64)
    y = np.asarray([float(ys[k]) for k in keys], dtype=np.float64)
    n = x.size
    observed = spearman_rank(x, y)
    if n < 3 or not np.isfinite(observed):
        return {"rho": observed, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_boot": 0.0, "n_pairs": float(n)}

    rng, _ = create_seed_generator(derive_child_seed(int(seed), "spearman-bootstrap", n))
    draws = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        r = spearman_rank(x[idx], y[idx])
        if np.isfinite(r):
            draws.append(r)
    if len(draws) < 2:
        return {"rho": float(observed), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_boot": float(len(draws)), "n_pairs": float(n)}
    lo, hi = np.percentile(np.asarray(draws, dtype=np.float64), [2.5, 97.5])
    return {"rho": float(observed), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_boot": float(len(draws)), "n_pairs": float(n)}


# PRD.md §1 C5 pre-registered decision bands.
C5_PROXY_VALID_RHO = 0.8      # rho >= this  -> "the feature audit is a valid proxy"
C5_AUDITS_DIFFER_RHO = 0.4    # rho <= this  -> "the audits license different decisions"


def c5_verdict(rho: float) -> str:
    """The pre-registered C5 reading of a rho (PRD.md §1). Never re-derive this inline."""
    if not np.isfinite(rho):
        return "undetermined"
    if rho >= C5_PROXY_VALID_RHO:
        return "feature-audit-is-valid-proxy"
    if rho <= C5_AUDITS_DIFFER_RHO:
        return "audits-license-different-decisions"
    return "inconclusive"


def cross_audit_report(
    circuit_damage: Mapping[str, float],
    feature_damage: Mapping[str, float],
    n_permutations: int = 10000,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """THE reportable C5 object: rho, its CI, its permutation p, and the verdict.

    The verdict is applied to the point estimate but is reported alongside the CI on
    purpose: if the CI spans two bands (e.g. 0.35 to 0.85) the honest conclusion is
    "inconclusive on this many cells", regardless of where the point estimate landed.
    ``verdict_ci_consistent`` says whether the whole interval agrees with the verdict.
    """
    shared = sorted(set(circuit_damage) & set(feature_damage))
    c = {k: circuit_damage[k] for k in shared}
    f = {k: feature_damage[k] for k in shared}

    perm = spearman_permutation_test(c, f, n_permutations=n_permutations, seed=seed)
    boot = spearman_bootstrap_ci(c, f, n_boot=n_boot, seed=seed)
    rho = perm["rho"]
    verdict = c5_verdict(rho)
    consistent = (
        np.isfinite(boot["ci_lo"]) and np.isfinite(boot["ci_hi"])
        and c5_verdict(boot["ci_lo"]) == verdict and c5_verdict(boot["ci_hi"]) == verdict
    )
    return {
        "spearman": rho,
        "ci_lo": boot["ci_lo"],
        "ci_hi": boot["ci_hi"],
        "p_value": perm["p_value"],
        "n_shared_cells": len(shared),
        "verdict": verdict,
        "verdict_ci_consistent": bool(consistent),
        "cells": shared,
    }


__all__ = [
    "spearman_rank",
    "circuit_vs_feature_damage_rank_agreement",
    "spearman_permutation_test",
    "spearman_bootstrap_ci",
    "cross_audit_report",
    "c5_verdict",
    "C5_PROXY_VALID_RHO",
    "C5_AUDITS_DIFFER_RHO",
]
