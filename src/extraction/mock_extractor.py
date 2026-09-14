# [AI-GEN] agent=OpenCode date=2026-08-07 task=Mock CircuitExtractor (dry-run; MockModel-aware, node/edge thresholds)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY

"""Synthetic, deterministic CircuitExtractor for dry-runs (no model, no GPU).

NOT a scientific object: it generates fake graphs over a fixed component universe to
exercise the ensemble/inclusion-frequency/reporting stack in CI (AI_RULES.md 2.3 —
never mistaken for evidence). Discipline is honored:

- deterministic given (task, config, seed): every draw uses a seeded generator
  derived from the seed AND the config id (AI_RULES.md 1.1);
- component IDs follow the L{layer}.{type}{index} scheme (ARCHITECTURE.md §2);
- one graph per (config, seed) cell, threshold-sensitive so B configs produce
  distinct inclusion patterns.

Two paths:
1. ``model`` is a MockModel (src/synthetic/mock_model.py): edges are included iff the
   model's weight-dependent score >= config.edge_threshold (Q4 provisional seeded
   non-nested grid supplies node/edge thresholds). Weight perturbation (null draws,
   Stage B) therefore shifts graphs.
2. ``model`` is None: legacy probabilistic rule over the fixed node universe using
   config.threshold (backwards-compatible with the original CI-only behavior).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..common.schema import Edge, Graph
from ..common.seeding import create_seed_generator, derive_child_seed

_NODE_UNIVERSE: tuple[str, ...] = (
    "L0.H0",
    "L0.H1",
    "L1.H0",
    "L1.H1",
    "L2.MLP",
    "L3.MLP",
)


def _candidate_edges() -> tuple[tuple[str, str], ...]:
    return tuple((src, dst) for src in _NODE_UNIVERSE for dst in _NODE_UNIVERSE if src != dst)


class MockExtractor:
    """Synthetic extractor: ``pipeline: mock`` (dry-runs / CI only)."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def extract(self, model: Any, task: Mapping[str, Any], config: Mapping[str, Any], seed: int) -> Graph:
        """Return a deterministic fake graph for this (config, seed) cell."""
        config_id = str(config.get("id", "config"))
        edge_thr = float(config.get("edge_threshold", config.get("threshold", 0.5)))
        node_thr = float(config.get("node_threshold", config.get("threshold", 0.5)))

        if model is not None and getattr(model, "is_synthetic", False):
            scores = model.edge_scores(task, int(seed))
            edges = tuple(
                Edge(src_component=e.split("->")[0], dst_component=e.split("->")[1],
                     config_id=config_id, seed=int(seed))
                for e, s in sorted(scores.items())
                if s >= edge_thr
            )
            return Graph(edges=edges, nodes=tuple(model.nodes()),
                         metadata={"mock": True, "synthetic_model": True,
                                   "node_threshold": node_thr, "edge_threshold": edge_thr})

        np_gen, _ = create_seed_generator(derive_child_seed(int(seed), config_id))
        p = 0.1 + 0.6 * min(max(edge_thr, 0.0), 1.0)
        edges = tuple(
            Edge(src_component=src, dst_component=dst, config_id=config_id, seed=int(seed))
            for src, dst in _candidate_edges()
            if np_gen.random() < p
        )
        return Graph(edges=edges, nodes=_NODE_UNIVERSE,
                     metadata={"mock": True, "threshold": edge_thr})


__all__ = ["MockExtractor", "_NODE_UNIVERSE"]
