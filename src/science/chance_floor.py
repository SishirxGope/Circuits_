# [AI-GEN] agent=Claude date=2026-08-08 task=Random top-k chance floor for overlap statistics (AI_RULES.md 4.4 / PRD.md §2 required it; it did not exist)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY
#
# NOVELTY PROTECTION ZONE (AI_RULES.md §3): draft for review only; do not treat as
# approved scientific logic until human approval (APPROVE chance_floor).

"""Random top-k baselines for overlap statistics (AI_RULES.md 4.4; PRD.md §2).

AI_RULES.md 4.4, verbatim: *"Every overlap statistic is reported next to its random
top-k baseline (per ref 2607.18921); every circuit-change statistic next to its frozen
null. Numbers without floors are meaningless in this project by construction."*

This is the second floor in the paper, and it is a different floor from the null.

- The **null** (`matched_magnitude.py`) answers: *did compression change the circuit
  more than random weight noise of the same size?*
- The **chance floor** here answers: *is this overlap number distinguishable from what
  two arbitrary edge sets of these sizes would score anyway?*

They are not interchangeable. Ref arXiv:2607.18921 needed exactly this one: it reports
exact-edge Jaccard@10 of 0.14–0.16 which, on one split, was **statistically
indistinguishable from a random top-k baseline (p = 0.106)**. Without the chance floor,
0.15 reads as "some structure survived". With it, that particular 0.15 reads as
"nothing measurable survived". A Jaccard reported bare is not interpretable.

Two ways to get the floor, both provided because they answer different questions:

- ``expected_jaccard_random`` — the closed form. Cheap, exact, no seed. Answers "what
  is the expected overlap?" Use it in tables next to the observed value.
- ``random_topk_null`` / ``overlap_vs_chance`` — the Monte-Carlo null distribution.
  Answers "how surprising is the observed overlap?" and yields the p-value and
  percentile interval that AI_RULES.md 4.1 requires. Seeded (AI_RULES.md 1.1).

The universe size N is a modelling choice you must state: it is the number of
*candidate* edges the extractor could have returned, not the number it did return.
For an attribution graph over C components with directed edges and no self-loops that
is C·(C−1). Passing the wrong N silently moves the floor, so it is a required argument
with no default.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..common.seeding import create_seed_generator, derive_child_seed


def candidate_edge_count(n_components: int, directed: bool = True, self_loops: bool = False) -> int:
    """Size of the candidate-edge universe N for ``n_components`` components.

    Directed without self-loops (our default graph object, ARCHITECTURE.md §2):
    N = C·(C−1). Stated explicitly rather than inferred, because N sets the floor.
    """
    if n_components < 0:
        raise ValueError(f"n_components must be >= 0, got {n_components}")
    c = int(n_components)
    n = c * c if self_loops else c * (c - 1)
    return n if directed else n // 2


def expected_jaccard_random(k_a: int, k_b: int, n_universe: int) -> float:
    """Expected Jaccard of two independent uniformly-random subsets of sizes k_a, k_b.

    Under independence E|A ∩ B| = k_a·k_b/N, and |A ∪ B| = k_a + k_b − |A ∩ B|, so the
    ratio-of-expectations approximation is

        E[J] ≈ (k_a·k_b/N) / (k_a + k_b − k_a·k_b/N).

    This is the *ratio of expectations*, not the expectation of the ratio — they differ
    at small k. For anything that enters a figure use ``random_topk_null``, which makes
    no such approximation; this closed form is for quick sanity checks and table
    annotations.
    """
    if n_universe <= 0:
        raise ValueError(f"n_universe must be > 0, got {n_universe}")
    if not (0 <= k_a <= n_universe and 0 <= k_b <= n_universe):
        raise ValueError(f"set sizes must lie in [0, {n_universe}], got {k_a} and {k_b}")
    if k_a == 0 and k_b == 0:
        return 1.0  # both empty: trivially identical, matching two_level._jaccard
    expected_intersection = (k_a * k_b) / n_universe
    union = k_a + k_b - expected_intersection
    return float(expected_intersection / union) if union > 0 else 0.0


def random_topk_null(
    k_a: int,
    k_b: int,
    n_universe: int,
    n_draws: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Monte-Carlo null distribution of Jaccard for random sets of sizes k_a, k_b.

    Draws ``n_draws`` independent pairs of uniformly-random subsets of the universe and
    returns their Jaccard values. Deterministic given ``seed`` (AI_RULES.md 1.1).
    """
    if n_universe <= 0:
        raise ValueError(f"n_universe must be > 0, got {n_universe}")
    if not (0 <= k_a <= n_universe and 0 <= k_b <= n_universe):
        raise ValueError(f"set sizes must lie in [0, {n_universe}], got {k_a} and {k_b}")
    if n_draws <= 0:
        raise ValueError(f"n_draws must be > 0, got {n_draws}")

    rng, _ = create_seed_generator(derive_child_seed(int(seed), "random-topk-null", k_a, k_b, n_universe))
    out = np.empty(int(n_draws), dtype=np.float64)
    for i in range(int(n_draws)):
        a = rng.choice(n_universe, size=k_a, replace=False)
        b = rng.choice(n_universe, size=k_b, replace=False)
        inter = np.intersect1d(a, b, assume_unique=True).size
        union = k_a + k_b - inter
        out[i] = 1.0 if union == 0 else inter / union
    return out


def overlap_vs_chance(
    observed_jaccard: float,
    k_a: int,
    k_b: int,
    n_universe: int,
    n_draws: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """The reportable object: an overlap with its chance floor attached (AI_RULES.md 4.4).

    Returns ``{observed, chance_median, chance_lo, chance_hi, chance_expected, p_value,
    excess, n_draws}`` where:

    - ``chance_lo/hi`` are the 2.5/97.5 percentiles of the random top-k null;
    - ``p_value`` is the one-sided Monte-Carlo p-value
      ``(1 + #{null >= observed}) / (1 + n_draws)`` — the +1 correction keeps p > 0,
      which matters because a p reported as exactly 0 is never true of a finite
      simulation;
    - ``excess`` is ``observed − chance_median``, the part of the overlap that chance
      does not explain.

    A large ``p_value`` is the ref-2607.18921 situation (their p = 0.106): the overlap
    is not distinguishable from chance, and the honest report says so.
    """
    null = random_topk_null(k_a, k_b, n_universe, n_draws=n_draws, seed=seed)
    obs = float(observed_jaccard)
    lo, hi = np.percentile(null, [2.5, 97.5])
    n_ge = int(np.sum(null >= obs))
    return {
        "observed": obs,
        "chance_median": float(np.median(null)),
        "chance_lo": float(lo),
        "chance_hi": float(hi),
        "chance_expected": expected_jaccard_random(k_a, k_b, n_universe),
        "p_value": float((1 + n_ge) / (1 + len(null))),
        "excess": obs - float(np.median(null)),
        "n_draws": float(len(null)),
    }


def overlap_vs_chance_for_freqs(
    f_pre: Mapping[str, float],
    f_post: Mapping[str, float],
    n_universe: int,
    cutoff: float = 0.0,
    n_draws: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """``overlap_vs_chance`` computed directly from two inclusion-frequency vectors.

    Set sizes are taken from the vectors at the given cutoff, so the floor is matched to
    the sets actually compared rather than to a nominal top-k.
    """
    from .two_level import exact_edge_overlap

    a = {k for k, v in f_pre.items() if v > cutoff}
    b = {k for k, v in f_post.items() if v > cutoff}
    observed = exact_edge_overlap(f_pre, f_post, cutoff=cutoff)
    return overlap_vs_chance(observed, len(a), len(b), n_universe, n_draws=n_draws, seed=seed)


def summarize_floor(result: Mapping[str, float]) -> str:
    """One-line human summary — the sentence that belongs in a figure caption."""
    verdict = (
        "indistinguishable from chance"
        if result["p_value"] > 0.05
        else "above chance"
    )
    return (
        f"Jaccard {result['observed']:.3f} vs random top-k floor "
        f"{result['chance_median']:.3f} [{result['chance_lo']:.3f}, {result['chance_hi']:.3f}], "
        f"p = {result['p_value']:.3f} — {verdict}"
    )


__all__ = [
    "candidate_edge_count",
    "expected_jaccard_random",
    "random_topk_null",
    "overlap_vs_chance",
    "overlap_vs_chance_for_freqs",
    "summarize_floor",
]
