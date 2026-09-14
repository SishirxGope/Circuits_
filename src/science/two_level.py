# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft two-level comparison (exact-edge + routing-head, proposal §2.4)
# modified: [AI-GEN] agent=Claude date=2026-09-12 task=Q10: wire the coarse level to node_ids.project_level2 + falsification condition
# reviewed-by: PENDING
# scientific-status: Q10 PRE-REGISTERED 2026-09-12 (PI-approved): level2_scheme = layer,
#   with the falsification condition in ``coarse_level_verdict`` pre-registered alongside.
#
# NOVELTY PROTECTION ZONE (AI_RULES.md §3): draft for review only; do not treat as
# approved scientific logic until human approval (APPROVE two_level).

"""Two-level comparison (proposal §2.4; CLAUDE.md §5 claim C2).

Every claim is stated at BOTH levels of description:
1. exact-edge: component-to-component edge overlap;
2. routing-head: routing-head-set overlap.

Ref arXiv:2607.18921 shows these levels can disagree (exact-edge Jaccard@10
0.14-0.16 vs routing-head 0.55-0.67); a conclusion holding at only one level is
reported as level-specific, never generalized.

Projection definition: a coarse node's inclusion is the MAX s(e) over edges touching
it (src or dst). Mechanical, deterministic.

**Q10 (pre-registered 2026-09-12): the coarse level is ``layer``.** circuit-tracer's
node basis is cross-layer-transcoder features, so head-level routing -- the coarse
level used by arXiv:2607.18921 -- does not exist in pipeline A at all. We substitute
layer routing, and the scheme is applied by ``src.extraction.node_ids.project_level2``
so there is exactly ONE definition of the coarse level in the codebase. Before this,
``project_to_routing_heads`` carried its own head regex, which meant choosing a scheme
in the config did not rewire this module at all and C2 would have reported zeros.

**The falsification condition, pre-registered with the choice** (see
``coarse_level_verdict``): arXiv:2607.18921 reports that *semantic* coarsening was
numerically indistinguishable from structural (Jaccard@10 0.163 vs 0.163) and only
routing-head separated. A coarsening that does not cut along a functional boundary
buys nothing. So if the coarse-level Jaccard lies within the bootstrap CI of the
exact-edge Jaccard, the second level is declared VACUOUS for pipeline A and reported
as a negative methodological result -- not presented as a second level that agrees.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from ..extraction.node_ids import project_level2

HEAD_NODE_RE = re.compile(r"^L\d+\.H\d+$")


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0  # both empty at this level: trivially identical
    return len(a & b) / len(union)


def exact_edge_overlap(
    f_pre: Mapping[str, float], f_post: Mapping[str, float], cutoff: float = 0.0
) -> float:
    """Jaccard overlap of the edge sets at the exact-edge level.

    Edges are included when s(e) > cutoff (default: any observed edge). The cutoff
    is an argument, never hardcoded (Q1 band cutoffs remain PI-owned; this is the
    mechanical level projection, not the core/noise partition).
    """
    a = {k for k, v in f_pre.items() if v > cutoff}
    b = {k for k, v in f_post.items() if v > cutoff}
    return _jaccard(a, b)


def project_to_routing_heads(
    f: Mapping[str, float], strict: bool = True
) -> dict[str, float]:
    """Project a frequency vector to the routing-head level (PROVISIONAL definition).

    Head inclusion = max s(e) over edges touching the head (src or dst). MLP and
    other non-head components are dropped. Deterministic.

    ⚠️ **``strict=True`` refuses to return an empty projection from a non-empty input.**
    circuit-tracer's node basis is transcoder features, not attention heads (verified
    2026-08-08; see src/extraction/node_ids.py), so a pipeline-A frequency vector
    contains NO ``L{l}.H{h}`` nodes and this projection would silently return ``{}``.
    An empty routing-head set makes every routing-head overlap trivially 1.0 (both
    sides empty) and claim C2 — "two levels or it does not count" — vacuously
    satisfied. That is the single most dangerous silent failure in the pipeline: it
    would not crash, and it would report a *better* two-level agreement than reality.

    Pass ``strict=False`` only when an empty result is genuinely expected (e.g. an
    MLP-only test fixture).
    """
    out: dict[str, float] = {}
    for edge_id_, s in f.items():
        src, dst = edge_id_.split("->", 1)
        for node in (src, dst):
            if HEAD_NODE_RE.match(node):
                out[node] = max(out.get(node, 0.0), float(s))

    if strict and f and not out:
        sample = sorted(f)[:3]
        raise ValueError(
            "routing-head projection is EMPTY for a non-empty circuit: none of the "
            f"components match {HEAD_NODE_RE.pattern} (e.g. {sample}). If this is a "
            "pipeline-A (circuit-tracer) circuit, its nodes are transcoder features and "
            "there are no attention heads at all — the coarse comparison level is "
            "UNDEFINED until the scheme is chosen (docs/HUMAN_DECISIONS.md Q10; options "
            "in src/extraction/node_ids.py::LEVEL2_OPTIONS). Returning {} here would "
            "make every routing-head overlap 1.0 and claim C2 vacuously true."
        )
    return out


def project_to_coarse_level(
    f: Mapping[str, float], scheme: str, strict: bool = True
) -> dict[str, float]:
    """Project a frequency vector onto the coarse comparison level under ``scheme``.

    ``scheme`` is delegated to ``node_ids.project_level2`` -- the single definition of
    what the coarse level means -- so choosing a scheme in ``configs/comparison/*.yaml``
    actually rewires this module. A coarse node's inclusion is the max s(e) over edges
    touching it, matching the routing-head definition it generalises.

    ``strict`` keeps the original guarantee: an empty projection from a non-empty
    circuit raises rather than returning ``{}``, because an empty coarse set makes every
    coarse overlap trivially 1.0 and claim C2 vacuously satisfied.
    """
    nodes: list[str] = []
    for edge_id_ in f:
        src, dst = edge_id_.split("->", 1)
        nodes.extend((src, dst))
    mapping = project_level2(nodes, scheme)

    out: dict[str, float] = {}
    for edge_id_, s in f.items():
        src, dst = edge_id_.split("->", 1)
        for node in (src, dst):
            coarse = mapping[node]
            out[coarse] = max(out.get(coarse, 0.0), float(s))

    if strict and f and not out:
        raise ValueError(
            f"coarse-level projection under scheme {scheme!r} is EMPTY for a non-empty "
            f"circuit (e.g. {sorted(f)[:3]}). Returning an empty set would make every "
            "coarse overlap 1.0 and claim C2 vacuously true."
        )
    return out


def routing_head_overlap(
    f_pre: Mapping[str, float],
    f_post: Mapping[str, float],
    cutoff: float = 0.0,
    scheme: str | None = None,
) -> float:
    """Jaccard overlap of the coarse-level sets (claim C2's second level).

    ``scheme=None`` keeps the original literal head projection, for the dense-node
    pipeline and for fixtures that really do carry ``L{l}.H{h}`` nodes. Pass the
    pre-registered scheme (``"layer"``) for pipeline A, where no head nodes exist.
    """
    if scheme is None:
        a = {h for h, v in project_to_routing_heads(f_pre).items() if v > cutoff}
        b = {h for h, v in project_to_routing_heads(f_post).items() if v > cutoff}
    else:
        a = {h for h, v in project_to_coarse_level(f_pre, scheme).items() if v > cutoff}
        b = {h for h, v in project_to_coarse_level(f_post, scheme).items() if v > cutoff}
    return _jaccard(a, b)


def coarse_level_verdict(
    coarse_jaccard: float,
    exact_edge_ci: Sequence[float],
) -> dict[str, object]:
    """The Q10 falsification condition, pre-registered 2026-09-12.

    If the coarse-level Jaccard falls inside the bootstrap CI of the exact-edge
    Jaccard, the two levels are not distinguishable and the coarse level is declared
    VACUOUS for this pipeline -- reported as a negative methodological result rather
    than as a second level that happens to agree.

    Returns ``{"vacuous": bool, "statement": str}``. Deliberately not a bare bool: the
    sentence is what goes in the paper, and it should not be re-derived by each caller.
    """
    lo, hi = float(exact_edge_ci[0]), float(exact_edge_ci[1])
    if lo > hi:
        raise ValueError(f"exact_edge_ci must be (lo, hi), got {(lo, hi)}")
    vacuous = lo <= float(coarse_jaccard) <= hi
    if vacuous:
        statement = (
            f"The coarse (layer-routing) Jaccard {coarse_jaccard:.3f} lies within the "
            f"bootstrap CI of the exact-edge Jaccard [{lo:.3f}, {hi:.3f}]. Per the "
            "pre-registered falsification condition, the second level is VACUOUS for "
            "this pipeline: circuit-tracer's node basis has no routing structure to "
            "coarsen onto, so the two levels are one measurement reported twice. "
            "Reported as a negative methodological result."
        )
    else:
        statement = (
            f"The coarse (layer-routing) Jaccard {coarse_jaccard:.3f} lies outside the "
            f"bootstrap CI of the exact-edge Jaccard [{lo:.3f}, {hi:.3f}], so the two "
            "levels are distinguishable and claim C2 is stated at both."
        )
    return {"vacuous": vacuous, "statement": statement}


def compare_at_both_levels(
    f_pre: Mapping[str, float],
    f_post: Mapping[str, float],
    cutoff: float = 0.0,
    scheme: str | None = None,
) -> dict[str, float]:
    """The always-paired two-level comparison (claim C2).

    Returns {"exact_edge": ..., "routing_head": ...} so every caller states both
    levels together; the paper's rule (CLAUDE.md §5) is that a claim may only be
    made at a level if it holds there.
    """
    return {
        "exact_edge": exact_edge_overlap(f_pre, f_post, cutoff=cutoff),
        "routing_head": routing_head_overlap(f_pre, f_post, cutoff=cutoff, scheme=scheme),
    }


__all__ = [
    "exact_edge_overlap",
    "routing_head_overlap",
    "project_to_routing_heads",
    "project_to_coarse_level",
    "coarse_level_verdict",
    "compare_at_both_levels",
]
