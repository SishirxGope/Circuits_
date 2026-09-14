# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft edge-pruning graph extractor (pipeline B, engineering path)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY
#
# Reference for pipeline B: edge-pruning circuit discovery on attention-only models
# (Conmy et al., NeurIPS 2023, "Towards Automated Circuit Discovery"; cited in PRD.md
# §2 as the greater-than reference circuit source). Our wrapping follows the same
# EAP-style edge-pruning approach; upstream code to be pinned at Stage A engineering
# (CLAUDE.md §7 adaptation header required).

"""Pipeline B: edge-pruning graph extraction (proposal §2.2; CLAUDE.md §2.2).

Implements the ``CircuitExtractor`` protocol for the edge-pruning family
(``pipeline: edgeprune``). Science requirements (same as pipeline A):
- thresholds are ensemble axes from configs/ensemble/ (never hardcoded);
- output uses the ARCHITECTURE.md §2 edge schema with L{layer}.{type}{index} IDs;
- deterministic given (model, task, config, seed).

Engineering path (PROVISIONAL): with a MockModel the extractor scores candidate
edges from the mock's weight-dependent edge scores and applies the ensemble cell's
node/edge thresholds — the full ensemble/inclusion-frequency stack runs end-to-end
without real models. Real wrapping (EAP on real models) is Stage A engineering.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..common.schema import Edge, Graph
from ..synthetic.mock_model import MockModel


class EdgePruningExtractor:
    """Pipeline B extractor: edge-pruning graph (swappable via ``pipeline: edgeprune``)."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def extract(self, model: Any, task: Mapping[str, Any], config: Mapping[str, Any], seed: int) -> Graph:
        """One edge-pruning graph for this (config, seed) ensemble cell.

        MockModel path: include edge e iff score(e) >= config.edge_threshold
        (scores are weight-dependent, deterministic, seeded). Deterministic given
        (model, task, config, seed) (AI_RULES.md 1.1).
        """
        config_id = str(config.get("id", "config"))
        node_thr = float(config.get("node_threshold", config.get("threshold", 0.5)))
        edge_thr = float(config.get("edge_threshold", config.get("threshold", 0.5)))

        if isinstance(model, MockModel):
            scores = model.edge_scores(task, int(seed))
            edges = tuple(
                Edge(src_component=e.split("->")[0], dst_component=e.split("->")[1],
                     config_id=config_id, seed=int(seed))
                for e, s in sorted(scores.items())
                if s >= edge_thr
            )
            return Graph(edges=edges, nodes=model.nodes(),
                         metadata={"pipeline": "edgeprune", "synthetic": True,
                                   "node_threshold": node_thr, "edge_threshold": edge_thr})
        raise NotImplementedError(
            "Stage A engineering: wrap the edge-pruning pipeline (pipeline B) on real "
            "models (EAP, Conmy et al. 2023); requires pinned upstream code + "
            "RUN MODEL DOWNLOAD (engineering mode only supports MockModel)"
        )
