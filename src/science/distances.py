# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft distance functions D (Q2 provisional: L1 primary, JS alternative)
# modified: [AI-GEN] agent=Claude date=2026-09-12 task=Q2: add the pre-registered normalised-L1 reporting column
# reviewed-by: PENDING
# scientific-status: Q2 PRE-REGISTERED 2026-09-12 (PI-approved): L1 primary, JS ablation,
#   normalised L1 reported alongside. Novelty-zone edit approved 2026-09-12.
#
# NOVELTY PROTECTION ZONE (AI_RULES.md §3): draft for review only; do not treat as
# approved scientific logic until human approval (APPROVE distances).

"""Distance functions D(c) between pre/post-compression frequency vectors.

PROVISIONAL ENGINEERING DEFAULT (docs/HUMAN_DECISIONS.md §3.2 Q2):
- primary: L1 over inclusion-frequency vectors
- alternative: Jensen-Shannon over normalized vectors (ablation sensitivity)

NOT pre-registered for Stage C until the PI confirms. Both implementations are pure
functions over {edge_id: s(e)} mappings so they are trivially testable and usable
with any FreqVector source (schema.FreqVector.frequencies or plain dicts).

Mechanical definitions (documented for review):
- L1: sum over the union of edge ids of |s_pre(e) - s_post(e)| (missing key -> 0).
- JS: Jensen-Shannon divergence over the normalized vectors (0 if both empty);
  sqrt of the mean of the two KL terms, with 0 log 0 := 0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


def _union_keys(a: Mapping[str, float], b: Mapping[str, float]) -> list[str]:
    return sorted(set(a) | set(b))


def l1_distance(f_pre: Mapping[str, float], f_post: Mapping[str, float]) -> float:
    """L1 distance over the union of edge ids (0 iff identical on all edges)."""
    total = 0.0
    for k in _union_keys(f_pre, f_post):
        total += abs(float(f_pre.get(k, 0.0)) - float(f_post.get(k, 0.0)))
    return total


def normalized_l1_distance(f_pre: Mapping[str, float], f_post: Mapping[str, float]) -> float:
    """L1 divided by the size of the edge union (Q2's pre-registered third column).

    Raw L1 scales with circuit size, and cells WILL differ in edge count because
    compression changes how many edges survive, so an L1 of 8.0 means something
    different in a 20-edge cell than in a 400-edge cell. Dividing by |union| puts every
    cell on a per-edge scale and pre-empts the obvious reviewer question.

    This is a **monotone rescaling of the primary within a cell**, not a third decision:
    it cannot reorder cells that share an edge union, and it is reported next to the raw
    number, never instead of it. The unnormalised L1 is the one that means "the circuit
    changed by this much".

    Returns 0.0 for two empty vectors, matching ``l1_distance`` (nothing changed),
    rather than dividing by zero.
    """
    union = _union_keys(f_pre, f_post)
    if not union:
        return 0.0
    return l1_distance(f_pre, f_post) / len(union)


def _normalize(f: Mapping[str, float]) -> dict[str, float]:
    total = sum(float(v) for v in f.values())
    if total <= 0.0:
        return {}
    return {k: float(v) / total for k, v in f.items()}


def _kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    out = 0.0
    for k, pv in p.items():
        qv = q.get(k, 0.0)
        if pv > 0.0:
            out += pv * math.log(pv / qv) if qv > 0.0 else math.inf
    return out


def jensen_shannon_distance(f_pre: Mapping[str, float], f_post: Mapping[str, float]) -> float:
    """JS distance over NORMALIZED vectors (Q2 provisional alternative).

    Identical vectors -> 0.0; disjoint support -> sqrt(ln 2) ~ 0.8326 (we average the
    two KL terms and take the square root, so the maximum is sqrt(ln 2), not 1.0).

    NOTE for Q2: this NORMALIZES both vectors, so it is blind to a uniform change in
    total inclusion mass — halving every s(e) reads as zero change. That is a real
    property to weigh when pre-registering the distance, not a defect.
    """
    p = _normalize(f_pre)
    q = _normalize(f_post)
    if not p and not q:
        return 0.0
    keys = _union_keys(p, q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    # No infinity guard is needed: m = 0.5*(p + q) is strictly positive wherever p or q
    # is, so neither KL term can diverge. Disjoint support therefore gives
    # sqrt(0.5 * (ln2 + ln2)) = sqrt(ln 2) ~ 0.8326, which is the documented maximum.
    # (An unreachable `if isinf(...): return 1.0` branch was removed 2026-08-09; it
    # contradicted the docstring and could never fire.)
    return math.sqrt(0.5 * (_kl(p, m) + _kl(q, m)))


__all__ = ["l1_distance", "normalized_l1_distance", "jensen_shannon_distance"]
