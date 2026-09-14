# [AI-GEN] agent=Claude date=2026-08-08 task=Node-ID mapping layer: circuit-tracer emits feature/error/embed/logit nodes, NOT the L{l}.H{h} heads our schema assumes
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY — the SCHEME below is a PI decision
#
# Facts verified against the local fork circuit-tracer-0.5.2 (2026-08-08):
#   circuit_tracer/frontend/graph_models.py  Node.feature_node / error_node /
#       token_node / logit_node, feature_type in {"cross layer transcoder",
#       "mlp reconstruction error", "embedding", "logit"}
#   circuit_tracer/graph.py  Graph docstring: node order
#       [active_features, error_nodes, embed_nodes, logit_nodes];
#       rows = target nodes, columns = source nodes;
#       active_features rows are (layer, pos, feature_idx) triples;
#       n error nodes = cfg.n_layers * len(input_tokens); n embed nodes = len(input_tokens)
#   circuit_tracer/utils/create_graph_files.py  the index -> node-kind decode
#   tests/test_node_id_format.py  error nodes use feature slot -1: f"{layer}_-1_{pos}"

"""Mapping upstream attribution nodes onto our component ID scheme.

⚠️ **THIS FILE DOCUMENTS A SCHEMA MISMATCH THAT IS A SCIENTIFIC DECISION, NOT A BUG.**

ARCHITECTURE.md §2 fixes our component IDs as ``L{layer}.{type}{index}`` — e.g.
``L11.H3`` (attention head 3 of layer 11), ``L7.MLP``. The two-level comparison
(proposal §2.4) then projects edges onto *routing heads* via that ``H`` marker.

**circuit-tracer does not produce attention-head nodes.** Its graph is over transcoder
features. Its four node kinds are, verbatim from the fork:

    feature_type = "cross layer transcoder"   Node.feature_node(layer, pos, feat_idx)
    feature_type = "mlp reconstruction error" Node.error_node(layer, pos)
    feature_type = "embedding"                Node.token_node(pos, vocab_idx)
    feature_type = "logit"                    Node.logit_node(...)

There is no head anywhere in the taxonomy. Two consequences the PI must resolve
BEFORE Stage A, because both change what the paper measures:

**(1) What is a "routing head" for pipeline A?**
    If we keep the ``L{l}.H{h}`` scheme, the routing-head projection returns the EMPTY
    SET for every pipeline-A circuit, and claim C2 ("two levels or it does not count")
    is vacuous for the primary pipeline. Options in ``LEVEL2_OPTIONS`` below.

**(2) Do we aggregate over token positions?**
    Upstream nodes are position-specific: ``(layer, pos, feature_idx)``. Our edge id is
    ``src->dst`` with NO position field (ARCHITECTURE.md §2 edges.parquet schema). So
    several distinct upstream edges collapse onto one component pair. That collapse is
    a modelling choice: it says "the circuit is the same computation wherever in the
    prompt it fires". Defensible, and probably what you want for a task circuit — but
    it must be stated, because it changes the edge universe (and therefore the chance
    floor, ``chance_floor.candidate_edge_count``).

Until the PI decides, ``to_component_id`` implements the *mechanical* mapping only and
``project_level2`` refuses rather than guessing. Nothing here invents a scheme.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Upstream node kinds, verbatim from circuit_tracer/frontend/graph_models.py.
FEATURE = "cross layer transcoder"
ERROR = "mlp reconstruction error"
EMBED = "embedding"
LOGIT = "logit"
UPSTREAM_NODE_KINDS: tuple[str, ...] = (FEATURE, ERROR, EMBED, LOGIT)

#: The options for what the routing-head level means when the graph has no heads.
#: ⚠️ TODO [QUESTION FOR PI] — pick one and pre-register it (docs/HUMAN_DECISIONS.md).
LEVEL2_OPTIONS: dict[str, str] = {
    "layer": (
        "Collapse every node to its LAYER (L7.F123 -> L7). The coarse level becomes "
        "'which layers route the computation'. Cheapest, always defined, and preserves "
        "the spirit of ref 2607.18921 (a coarser description that can disagree with the "
        "exact-edge one). Loses the 'head' reading of the proposal's wording."
    ),
    "feature_family": (
        "Collapse features that share a transcoder feature index across positions "
        "(L7.F123@pos3, L7.F123@pos9 -> L7.F123). This is position aggregation, not a "
        "second LEVEL — it does not give you a coarser description, so it does not "
        "satisfy C2 on its own."
    ),
    "dense_node_heads": (
        "Use the dense-node pipeline (src/extraction/dense_node_variant.py) as the "
        "carrier of the routing-head level, since a dictionary-free attribution over "
        "heads/MLPs DOES have heads. Two levels then come from two pipelines. Most "
        "faithful to the proposal's wording, most work, and it entangles C2 with C8."
    ),
    "node_kind": (
        "Collapse to the upstream node kind (feature / error / embed / logit). Trivially "
        "defined but almost certainly too coarse to be informative."
    ),
}


def to_component_id(
    kind: str,
    layer: int | None = None,
    feature_idx: int | None = None,
    pos: int | None = None,
    include_position: bool = False,
) -> str:
    """One upstream node -> one component ID string (mechanical; no scheme invented).

    Produces, in the ``L{layer}.{type}{index}`` spirit of ARCHITECTURE.md §2:

        feature -> ``L{layer}.F{feature_idx}``     (``.P{pos}`` appended if requested)
        error   -> ``L{layer}.ERR``               (``.P{pos}`` appended if requested)
        embed   -> ``EMB.P{pos}``
        logit   -> ``LOGIT.P{pos}``

    ``F`` (transcoder feature) is deliberately NOT ``H``: these are not attention heads,
    and naming them ``H`` would make the routing-head projection silently pick up
    feature nodes and report a "head overlap" that is nothing of the kind.

    ``include_position`` controls the aggregation decision described in the module
    docstring. It defaults to False (aggregate over positions) because our edges.parquet
    schema has no position column — but see ``position_policy_is_pi_owned``.
    """
    if kind not in UPSTREAM_NODE_KINDS:
        raise ValueError(f"unknown upstream node kind {kind!r}; expected one of {UPSTREAM_NODE_KINDS}")

    if kind == EMBED:
        _require(pos is not None, "embedding node needs pos")
        return f"EMB.P{pos}"
    if kind == LOGIT:
        _require(pos is not None, "logit node needs pos")
        return f"LOGIT.P{pos}"

    _require(layer is not None, f"{kind} node needs layer")
    if kind == ERROR:
        base = f"L{layer}.ERR"
    else:  # FEATURE
        _require(feature_idx is not None, "feature node needs feature_idx")
        base = f"L{layer}.F{feature_idx}"
    if include_position:
        _require(pos is not None, f"{kind} node needs pos when include_position=True")
        return f"{base}.P{pos}"
    return base


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def decode_upstream_index(
    node_idx: int,
    n_selected_features: int,
    n_layers: int,
    n_pos: int,
) -> dict[str, Any]:
    """Decode an adjacency-matrix index into its node kind + coordinates.

    Mirrors the decode in ``circuit_tracer/utils/create_graph_files.py``: the node order
    is ``[features, errors, embeds, logits]``, with ``n_layers * n_pos`` error nodes and
    ``n_pos`` embed nodes, and errors laid out as ``divmod(idx - n_features, n_pos)``.

    Returns ``{kind, layer, pos, feature_slot}``. ``feature_slot`` is the index INTO
    ``graph.selected_features`` for feature nodes (the caller resolves it against
    ``graph.active_features`` to get the real ``(layer, pos, feature_idx)`` triple),
    and None otherwise.
    """
    if node_idx < 0:
        raise ValueError(f"node_idx must be >= 0, got {node_idx}")
    error_end = n_selected_features + n_layers * n_pos
    embed_end = error_end + n_pos

    if node_idx < n_selected_features:
        return {"kind": FEATURE, "layer": None, "pos": None, "feature_slot": node_idx}
    if node_idx < error_end:
        layer, pos = divmod(node_idx - n_selected_features, n_pos)
        return {"kind": ERROR, "layer": layer, "pos": pos, "feature_slot": None}
    if node_idx < embed_end:
        return {"kind": EMBED, "layer": None, "pos": node_idx - error_end, "feature_slot": None}
    return {"kind": LOGIT, "layer": None, "pos": node_idx - embed_end, "feature_slot": None}


def position_policy_is_pi_owned(include_position: bool) -> str:
    """The sentence that must appear in the paper for whichever policy is chosen."""
    if include_position:
        return (
            "Edges are position-specific: the same feature pair at two token positions "
            "is two edges. The edge universe grows by a factor of n_pos, which widens "
            "the chance floor and shrinks every overlap. Requires adding a position "
            "column to the edges.parquet schema (ARCHITECTURE.md §2)."
        )
    return (
        "Edges are aggregated over token positions: the same feature pair firing at any "
        "position is one edge. This asserts the circuit is the same computation wherever "
        "in the prompt it fires. Matches the current edges.parquet schema."
    )


def project_level2(component_ids: Sequence[str], scheme: str) -> dict[str, str]:
    """Project component IDs onto the coarse comparison level under ``scheme``.

    ``scheme`` must be a key of ``LEVEL2_OPTIONS``. There is no default: the choice
    changes what claim C2 states, so this function refuses to pick one.
    """
    if scheme not in LEVEL2_OPTIONS:
        raise ValueError(
            f"unknown level-2 scheme {scheme!r}. The routing-head level is UNDEFINED for "
            f"pipeline A because circuit-tracer emits no attention-head nodes; pick one "
            f"of {sorted(LEVEL2_OPTIONS)} and pre-register it "
            f"(docs/HUMAN_DECISIONS.md). Options are documented in LEVEL2_OPTIONS."
        )
    out: dict[str, str] = {}
    for cid in component_ids:
        if scheme == "layer":
            out[cid] = cid.split(".")[0]
        elif scheme == "feature_family":
            parts = cid.split(".")
            out[cid] = ".".join(parts[:2]) if len(parts) > 2 else cid
        elif scheme == "node_kind":
            out[cid] = (
                "embed" if cid.startswith("EMB") else
                "logit" if cid.startswith("LOGIT") else
                "error" if cid.endswith(".ERR") or ".ERR." in cid else
                "feature"
            )
        else:  # dense_node_heads
            raise ValueError(
                "scheme 'dense_node_heads' is not a projection of pipeline-A IDs: it "
                "means the routing-head level is carried by the dense-node pipeline "
                "instead. Run src/extraction/dense_node_variant.py and compare pipelines."
            )
    return out


__all__ = [
    "FEATURE", "ERROR", "EMBED", "LOGIT", "UPSTREAM_NODE_KINDS",
    "LEVEL2_OPTIONS",
    "to_component_id",
    "decode_upstream_index",
    "project_level2",
    "position_policy_is_pi_owned",
]
