# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft inclusion-frequency computation s(e) (proposal §2.3)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY
#
# NOVELTY PROTECTION ZONE (AI_RULES.md §3): draft for review only; do not treat as
# approved scientific logic until human approval (APPROVE inclusion_freq).

"""Edge inclusion frequency s(e) (proposal §2.3; CLAUDE.md §5).

s(e) = fraction of the (B configs x S seeds) extraction ensemble whose graphs
contain edge e. The reported circuit object is always the vector of s(e), never a
binary edge list.

Mechanical definition only: no scientific choices are made here. Band cutoffs
(core/contingent/noise) are a separate, PI-approved decision in
configs/ensemble/decompose/final.yaml (CLAUDE.md §5; AI_RULES.md §6).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ..common.schema import EnsembleResult, FreqVector, Graph, edge_id


def compute_from_graphs(
    graphs: Sequence[Graph],
    n: int | None = None,
) -> EnsembleResult:
    """Compute s(e) over the ensemble of graphs.

    Args:
        graphs: one Graph per (config, seed) cell.
        n: number of ensemble cells (defaults to len(graphs)). Passed explicitly when
            the caller knows the full grid even if some cells were skipped.

    Returns:
        EnsembleResult with:
        - ``records``: edges.parquet rows (src_component, dst_component, config_id,
          seed, included=True), deterministically ordered by (edge_id, config_id,
          seed).
        - ``freq``: FreqVector with s(e) for every edge seen in >=1 graph.
        - ``n_cells``: the denominator used.

    An empty ensemble yields an empty EnsembleResult (safe; no division by zero).
    """
    if not graphs:
        return EnsembleResult(records=(), freq=FreqVector(), n_cells=0)

    # s(e) counts CELLS THAT CONTAIN e, not edge records. An extractor may legitimately
    # emit the same component pair more than once within one (config, seed) cell —
    # circuit-tracer's nodes are (layer, pos, feature_idx), so several position-specific
    # upstream edges collapse onto one component-to-component pair once position is
    # dropped from the edge id. Counting records would then push s(e) above 1, which is
    # not a fraction and breaks every downstream quantity (D, CSI, bands). Deduplicating
    # per cell is what "the fraction of the ensemble whose graphs CONTAIN edge e"
    # (CLAUDE.md §5) says, and it matches Graph.included_edge_ids(), which already
    # deduplicates. Fixed 2026-08-08.
    counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for g in graphs:
        seen_in_cell: set[tuple[str, str, int]] = set()
        for e in g.edges:
            if not e.included:
                continue
            key = e.edge_id()
            cell_key = (key, str(e.config_id), int(e.seed))
            if cell_key in seen_in_cell:
                continue  # already counted for this (edge, config, seed) cell
            seen_in_cell.add(cell_key)
            counts[key] += 1
            records.append(
                {
                    "src_component": e.src_component,
                    "dst_component": e.dst_component,
                    "config_id": e.config_id,
                    "seed": e.seed,
                    "included": True,
                }
            )

    n_cells = len(graphs) if n is None else n
    if n_cells <= 0:
        raise ValueError(f"ensemble size n must be > 0 when provided, got {n_cells}")

    frequencies = {k: c / n_cells for k, c in sorted(counts.items())}
    # The single choke point where s(e) is created: assert the invariant here so any
    # future counting change cannot silently produce a non-fraction.
    out_of_range = {k: v for k, v in frequencies.items() if not (0.0 <= v <= 1.0)}
    if out_of_range:
        raise ValueError(
            f"inclusion frequencies outside [0, 1]: {dict(sorted(out_of_range.items())[:5])} "
            f"(n_cells={n_cells}). s(e) is a fraction of ensemble cells; a value above 1 "
            f"means an edge was counted more times than there are cells."
        )
    records.sort(key=lambda r: (edge_id(r["src_component"], r["dst_component"]), r["config_id"], r["seed"]))
    return EnsembleResult(records=tuple(records), freq=FreqVector(frequencies=frequencies), n_cells=n_cells)


def frequencies_from_inclusion_counts(counts: Mapping[str, int], n: int) -> FreqVector:
    """Build a FreqVector from raw inclusion counts (helper for tests / pipelines)."""
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    return FreqVector(frequencies={k: c / n for k, c in sorted(counts.items())})


__all__ = ["compute_from_graphs", "frequencies_from_inclusion_counts"]
