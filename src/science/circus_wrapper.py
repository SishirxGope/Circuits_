# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft EnsembleRunner (CIRCUS wrapper, proposal §2.1c)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY
#
# NOVELTY PROTECTION ZONE (AI_RULES.md §3): draft for review only; do not treat as
# approved scientific logic until human approval (APPROVE circus_wrapper).

"""CIRCUS-style ensemble runner (proposal §2.1c; CLAUDE.md §2.1(c)).

Runs B non-nested threshold/pruning configs x S seeds through a CircuitExtractor and
returns the inclusion-frequency vector s(e) over the ensemble — the ONLY reported
circuit object (CLAUDE.md §5; never a binary edge list).

Design note (ARCHITECTURE.md §4, compute plan §5): the CIRCUS observation is that the
ensemble is near-free after one attribution pass — raw attribution is computed once
per (model, task, seed) and re-pruned under all B configs. That optimization is
INTENTIONALLY not implemented here yet (do not optimize prematurely): the naive
per-cell loop below is the reference implementation, and the reuse optimization lands
only after the extractor API exposes cached raw attribution per seed. The reuse
comment must survive into that optimization so the compute plan stays honest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..interfaces import CircuitExtractor, Model
from ..common.schema import EnsembleResult, Graph
from .inclusion_freq import compute_from_graphs


class CircusEnsembleRunner:
    """EnsembleRunner implementation for B threshold configs x S seeds.

    Deterministic given (extractor, model, task, configs, seeds): iteration order is
    seeds outer, configs inner; every extractor call receives an explicit seed
    (AI_RULES.md 1.1).
    """

    def __init__(self, extractor: CircuitExtractor) -> None:
        self._extractor = extractor

    def run(
        self,
        model: Model,
        task: Mapping[str, Any],
        configs: Sequence[Mapping[str, Any]],
        seeds: Sequence[int],
    ) -> EnsembleResult:
        """Extract one graph per (seed, config) cell and aggregate to s(e).

        Returns EnsembleResult: per-cell edge records (edges.parquet rows) + the
        FreqVector. Empty configs/seeds yield an empty result (safe, no exception).
        """
        graphs: list[Graph] = []
        for seed in seeds:
            for ci, config in enumerate(configs):
                config_id = str(config.get("id", f"config{ci}"))
                # TODO(Stage A engineering): reuse one raw attribution per
                # (model, task, seed) across all B configs (CIRCUS, arXiv:2603.00523;
                # ARCHITECTURE.md §4). Naive loop is the reference until then.
                graphs.append(self._extractor.extract(model, task, config, seed))
        n_cells = len(configs) * len(seeds)
        return compute_from_graphs(graphs, n=n_cells if n_cells else None)


__all__ = ["CircusEnsembleRunner"]
