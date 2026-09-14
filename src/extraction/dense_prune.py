# [AI-GEN] agent=Claude date=2026-09-14 task=Apply the Q4 threshold grid to EAP scores with circuit-tracer's semantics
# reviewed-by: PENDING
#
# Adapted from: https://github.com/decoderesearch/circuit-tracer @ 8f1e2438df612464e229e44c4a00ff637bf9379b, MIT
#   circuit_tracer/graph.py: ``find_threshold`` (ported exactly) and the iterative removal of
#   nodes left without incoming/outgoing edges in ``prune_graph`` (adapted to a dense
#   component graph). Licence: THIRD_PARTY_LICENSES/circuit-tracer-LICENSE.txt.

"""Pruning a dense-node attribution graph under one (node_threshold, edge_threshold) view.

The Q4 grid's thresholds are CUMULATIVE-INFLUENCE FRACTIONS in circuit-tracer: keep the
highest-scoring items until they account for that fraction of the total. They are applied
with exactly that meaning here, so the same grid means the same thing on Pythia as on the
primaries.

**The one deliberate difference from ``prune_graph`` (PI decision, 2026-09-14).**
circuit-tracer's adjacency entries are DIRECT effects, which is why it propagates influence
(A + A^2 + ...) before thresholding. EAP scores are already TOTAL effects: the gradient at an
edge's endpoint includes every downstream path. Propagating again would double-count, so the
thresholds are applied to the EAP node and edge scores directly.

Pipeline, mirroring ``prune_graph``'s order:

1. Keep nodes whose score clears ``find_threshold(node_scores, node_threshold)``; the
   embedding and the logit node are always kept, as circuit-tracer always keeps tokens and
   logits.
2. Among edges whose source is kept and whose destination COMPONENT is kept, keep those
   clearing ``find_threshold(candidate_edge_scores, edge_threshold)``. circuit-tracer
   thresholds the flattened pruned matrix, whose zeros for removed nodes do not change the
   cumulative sum, so thresholding the candidates is equivalent for every threshold below 1.
3. Iteratively remove heads and MLPs left with no kept incoming OR no kept outgoing edge, and
   the edges touching them, until nothing changes. Heads and MLPs are computed from their
   inputs, so they need both - circuit-tracer's requirement for feature nodes.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .eap import EMB, LOGIT

ALWAYS_KEPT = frozenset({EMB, LOGIT})


def find_threshold(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Exact port of circuit-tracer's ``find_threshold``.

    Returns the score value such that keeping every score ``>=`` it retains the top items
    accounting for ``threshold`` of the total. Left-sided ``searchsorted``, clamped to the
    last index (which only binds at ``threshold = 1.0``).
    """
    sorted_scores = torch.sort(scores, descending=True).values
    cumulative_score = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    threshold_index: int = int(torch.searchsorted(cumulative_score, threshold).item())
    threshold_index = min(threshold_index, len(cumulative_score) - 1)
    return sorted_scores[threshold_index]


def owner(node_id: str) -> str:
    """The component a destination node belongs to: ``L3.H0.Q`` -> ``L3.H0``."""
    parts = node_id.split(".")
    if len(parts) == 3 and parts[2] in ("Q", "K", "V"):
        return ".".join(parts[:2])
    return node_id


def _checked_threshold(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")
    return value


def _positive_total(values: torch.Tensor, what: str) -> None:
    if values.numel() == 0 or float(values.sum()) <= 0.0:
        raise ValueError(
            f"{what} sum to zero, so a cumulative-influence fraction is undefined "
            "(circuit-tracer would divide by zero and produce NaN). Refusing rather than "
            "returning an arbitrary circuit - check that the corruption actually changes the task."
        )


def prune_dense_graph(
    edge_scores: Mapping[str, float],
    node_scores: Mapping[str, float],
    node_threshold: float,
    edge_threshold: float,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(kept_nodes, kept_edges)`` for one threshold view. Deterministic."""
    node_threshold = _checked_threshold(node_threshold, "node_threshold")
    edge_threshold = _checked_threshold(edge_threshold, "edge_threshold")

    names = sorted(node_scores)
    values = torch.tensor([float(node_scores[n]) for n in names], dtype=torch.float64)
    _positive_total(values, "node scores")
    cut = find_threshold(values, node_threshold)
    kept_nodes = {n for n, v in zip(names, values.tolist()) if v >= float(cut)} | set(ALWAYS_KEPT)

    edges = sorted(edge_scores)
    candidates = [e for e in edges if _src(e) in kept_nodes and owner(_dst(e)) in kept_nodes]
    cand_values = torch.tensor([float(edge_scores[e]) for e in candidates], dtype=torch.float64)
    _positive_total(cand_values, "candidate edge scores after node pruning")
    edge_cut = float(find_threshold(cand_values, edge_threshold))
    kept_edges = {e for e, v in zip(candidates, cand_values.tolist()) if v >= edge_cut}

    while True:
        has_out = {_src(e) for e in kept_edges}
        has_in = {owner(_dst(e)) for e in kept_edges}
        dangling = {n for n in kept_nodes if n not in ALWAYS_KEPT and (n not in has_out or n not in has_in)}
        if not dangling:
            break
        kept_nodes -= dangling
        kept_edges = {e for e in kept_edges if _src(e) in kept_nodes and owner(_dst(e)) in kept_nodes}

    return frozenset(kept_nodes), frozenset(kept_edges)


def _src(edge_id: str) -> str:
    return edge_id.split("->", 1)[0]


def _dst(edge_id: str) -> str:
    return edge_id.split("->", 1)[1]


__all__ = ["ALWAYS_KEPT", "find_threshold", "owner", "prune_dense_graph"]
