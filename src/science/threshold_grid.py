# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft deterministic seeded non-nested threshold grid (Q4 provisional)
# modified: [AI-GEN] agent=Claude date=2026-09-12 task=Q4 pre-registration: add the CIRCUS anti-diagonal grid
# reviewed-by: PENDING
# scientific-status: Q4 PRE-REGISTERED 2026-09-12 (PI-approved). The seeded sampler below
#   is retained for the historical dry-runs only and is no longer the Stage A path.

"""Deterministic seeded non-nested threshold grid (proposal §2.1c; Q4).

PROVISIONAL ENGINEERING DEFAULT (docs/HUMAN_DECISIONS.md §3.2 Q4):
for B=4 dry-runs, four seeded threshold configs generated from the run seed.
Engineering placeholder for the CIRCUS-style non-nested config set (arXiv:2603.00523);
the final grid design remains PI-dependent before real Stage A.

Definition used here (mechanical, unit-tested):
- A config is a (node_threshold, edge_threshold) pair, both in (0, 1).
- Config A dominates config B if A's thresholds are >= B's on BOTH axes (A's edge
  set is then a subset of B's under monotone thresholding — nested).
- The grid is non-nested iff NO pair of configs dominates each other.
- Generation: seeded rejection sampling over the axis ranges; deterministic given
  (B, seed). The generator never inspects any data — it is a pure function of B,
  seed, and the axis ranges.

Q4 (PRE-REGISTERED 2026-09-12): ``generate_anti_diagonal_grid`` is the Stage A path.
``generate_seeded_non_nested_grid`` is kept for reproducing the pre-registration
dry-runs and must not be used for evidence: it adds a second seed axis on top of the
seed axis the design already has, for no scientific gain.

**Why an anti-diagonal and not a crossed product.** CIRCUS §3.2 requires non-nested
views ("crossing node and edge thresholds in opposite directions so that each view
prunes a different part of the graph") and gives the counter-example
``(0.6,0.95) ⊂ (0.8,0.98) ⊂ (0.9,0.99)``. A full ``k × k`` Cartesian product of
threshold levels *contains* that counter-example: order the configs by domination and
it is a grid poset, whose largest antichain has exactly ``k`` members. So a 4×4
product cannot yield 16 mutually non-nested views — it yields 16 views of which 84 of
the 120 pairs are nested, and nesting collapses consensus onto the loosest view,
inflating every s(e). B non-nested views therefore require B distinct levels per
axis, walked in opposite directions. That is the anti-diagonal.
"""

from __future__ import annotations

from typing import Any

from ..common.seeding import create_seed_generator, derive_child_seed

DEFAULT_NODE_RANGE = (0.55, 0.95)
DEFAULT_EDGE_RANGE = (0.90, 0.995)
_MAX_ATTEMPTS = 10000

#: Q4 pre-registered axis ranges (2026-09-12). Anchored on published values, not
#: invented: circuit-tracer's CLI defaults are node 0.8 / edge 0.98, and CIRCUS's
#: published non-nested examples are (0.6, 0.99) and (0.9, 0.95). The swept box is the
#: convex hull of those, so the upstream default lies inside it.
PREREGISTERED_NODE_RANGE = (0.6, 0.9)    # walked LOW -> HIGH
PREREGISTERED_EDGE_RANGE = (0.99, 0.95)  # walked HIGH -> LOW (opposite direction)


def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """True if config ``a`` dominates ``b`` on both threshold axes (nested)."""
    return (
        a["node_threshold"] >= b["node_threshold"]
        and a["edge_threshold"] >= b["edge_threshold"]
    )


def _non_nested(candidate: dict[str, float], existing: list[dict[str, float]]) -> bool:
    return all(
        not _dominates(candidate, c) and not _dominates(c, candidate) for c in existing
    )


def generate_seeded_non_nested_grid(
    B: int,
    seed: int,
    node_range: tuple[float, float] = DEFAULT_NODE_RANGE,
    edge_range: tuple[float, float] = DEFAULT_EDGE_RANGE,
    max_attempts: int = _MAX_ATTEMPTS,
) -> list[dict[str, Any]]:
    """Return B non-nested (node_threshold, edge_threshold) configs (deterministic).

    Each config carries ``id`` (``seedgrid-{i}``), ``node_threshold``,
    ``edge_threshold``. Same (B, seed) -> identical grid (tested).
    """
    if not (isinstance(B, int) and B > 0):
        raise ValueError(f"B must be a positive int, got {B!r}")
    rng, _ = create_seed_generator(derive_child_seed(int(seed), "threshold-grid"))
    configs: list[dict[str, float]] = []
    for i in range(B):
        for _ in range(max_attempts):
            candidate = {
                "node_threshold": float(rng.uniform(*node_range)),
                "edge_threshold": float(rng.uniform(*edge_range)),
            }
            if _non_nested(candidate, configs):
                configs.append(candidate)
                break
        else:  # pragma: no cover - ranges are wide enough; defensive
            raise RuntimeError(
                f"could not find a non-nested threshold config after {max_attempts} attempts "
                f"(B={B}, seed={seed}); widen the axis ranges"
            )
    return [
        {"id": f"seedgrid-{i}", **c} for i, c in enumerate(configs)
    ]


def generate_anti_diagonal_grid(
    B: int,
    node_range: tuple[float, float] = PREREGISTERED_NODE_RANGE,
    edge_range: tuple[float, float] = PREREGISTERED_EDGE_RANGE,
    ndigits: int = 4,
) -> list[dict[str, Any]]:
    """The Q4 pre-registered grid: B non-nested views, no seed, no data inspection.

    Walks the two axes in OPPOSITE directions and pairs them index-by-index, so
    config ``i`` has a strictly higher node threshold and a strictly lower edge
    threshold than config ``i-1``. Neither can dominate the other, so the set is
    non-nested by construction rather than by rejection sampling — which also means
    it is provable, not merely tested.

    Each config carries ``id`` (``antidiag-{i}``), ``node_threshold``,
    ``edge_threshold``. A pure function of ``B`` and the two ranges: the same B gives
    the same grid on any machine, in any run, forever. That is what makes it
    pre-registerable.
    """
    if not (isinstance(B, int) and B > 0):
        raise ValueError(f"B must be a positive int, got {B!r}")
    n_lo, n_hi = node_range
    e_lo, e_hi = edge_range
    if n_lo >= n_hi:
        raise ValueError(f"node_range must be (low, high) with low < high, got {node_range}")
    if e_lo <= e_hi:
        raise ValueError(
            f"edge_range must be (high, low) — walked opposite to node_range — got {edge_range}. "
            "Walking both axes in the same direction produces a NESTED chain, which is the "
            "artifact CIRCUS §3.2 exists to prevent."
        )
    if B == 1:
        return [{"id": "antidiag-0", "node_threshold": round(n_lo, ndigits),
                 "edge_threshold": round(e_lo, ndigits)}]
    out: list[dict[str, Any]] = []
    for i in range(B):
        t = i / (B - 1)
        out.append({
            "id": f"antidiag-{i}",
            "node_threshold": round(n_lo + (n_hi - n_lo) * t, ndigits),
            "edge_threshold": round(e_lo + (e_hi - e_lo) * t, ndigits),
        })
    return out


def is_non_nested(configs: list[dict[str, Any]]) -> bool:
    """True iff no config in ``configs`` dominates another (CIRCUS §3.2).

    Exposed so the property can be asserted at run time on whatever grid a config
    file supplies, not only on the one this module generates.
    """
    return all(
        _non_nested(c, [o for j, o in enumerate(configs) if j != i])
        for i, c in enumerate(configs)
    )


__all__ = [
    "generate_seeded_non_nested_grid",
    "generate_anti_diagonal_grid",
    "is_non_nested",
    "PREREGISTERED_NODE_RANGE",
    "PREREGISTERED_EDGE_RANGE",
]
