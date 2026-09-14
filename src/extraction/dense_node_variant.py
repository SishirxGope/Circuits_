# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft dense-node extraction variant (claim C8, engineering path)
# modified: [AI-GEN] agent=Claude date=2026-09-14 task=Real path: EAP scores cached per seed, pruned per threshold view
# reviewed-by: PENDING
# scientific-status: method pre-registered 2026-09-14 (PI): EAP, split Q/K/V, |sum| aggregation,
#   thresholds applied to total-effect scores. See src/extraction/eap.py and dense_prune.py.

"""Dense-node extraction variant (claim C8, PRD.md §1; CLAUDE.md §2).

C8 requires measuring the dictionary/attribution-basis drift confound: run a
dictionary-free extraction variant alongside the dictionary-based pipeline and report the
divergence between the two as a quantity. This variant attributes over dense components
(heads, MLPs) directly - no transcoder set needed, so it is the Pythia-compatible pipeline.
Swappable via ``pipeline: dense-node``.

**Real models.** For one (model, task, seed) the extractor draws the task's prompt pairs,
runs edge attribution patching once, and caches the scores. Every threshold view in the B
grid then only re-prunes those cached scores - the CIRCUS observation that the ensemble is
near-free once raw attribution exists (arXiv:2603.00523). ``CircusEnsembleRunner`` loops
seeds outer and configs inner, so a single-entry cache is enough. The cache is keyed on the
model OBJECT (identity, not equality): a perturbed (Stage B) or compressed (Stage C) model
is a different object and is always re-attributed.

**MockModel** keeps the original engineering-only path, which thresholds raw synthetic
scores and ignores ``node_threshold``. It exercises plumbing only and is never evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..common.schema import Edge, Graph
from ..synthetic.mock_model import MockModel


class DenseNodeExtractor:
    """Dictionary-free extraction variant (C8 basis-drift control)."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._cache_model: Any = None
        self._cache_key: tuple | None = None
        self._cache_scores: Any = None
        #: How many times EAP actually ran. Should equal the number of (model, seed) pairs.
        self.n_attribution_runs = 0

    def extract(self, model: Any, task: Mapping[str, Any], config: Mapping[str, Any], seed: int) -> Graph:
        config_id = str(config.get("id", "config"))

        if isinstance(model, MockModel):
            edge_thr = float(config.get("edge_threshold", config.get("threshold", 0.5)))
            scores = model.edge_scores(task, int(seed))
            edges = tuple(
                Edge(src_component=e.split("->")[0], dst_component=e.split("->")[1],
                     config_id=config_id, seed=int(seed))
                for e, s in sorted(scores.items())
                if s >= edge_thr
            )
            return Graph(edges=edges, nodes=model.nodes(),
                         metadata={"pipeline": "dense-node", "synthetic": True,
                                   "edge_threshold": edge_thr})

        if not hasattr(model, "run_with_hooks") or not hasattr(model, "cfg"):
            raise NotImplementedError(
                "the dense-node pipeline needs a hooked TransformerLens model "
                "(src/extraction/real_model.load_pinned_model); got "
                f"{type(model).__name__}"
            )

        from .dense_prune import prune_dense_graph

        node_thr = float(config["node_threshold"])
        edge_thr = float(config["edge_threshold"])
        scores = self._scores(model, task, int(seed))
        kept_nodes, kept_edges = prune_dense_graph(scores.edge_scores, scores.node_scores, node_thr, edge_thr)
        edges = tuple(
            Edge(src_component=e.split("->", 1)[0], dst_component=e.split("->", 1)[1],
                 config_id=config_id, seed=int(seed))
            for e in sorted(kept_edges)
        )
        return Graph(
            edges=edges,
            nodes=tuple(sorted(kept_nodes)),
            metadata={
                "pipeline": "dense-node",
                "synthetic": False,
                "node_threshold": node_thr,
                "edge_threshold": edge_thr,
                "n_candidate_edges": len(scores.edge_scores),
                "clean_metric_mean": scores.clean_metric_mean,
                "corrupt_metric_mean": scores.corrupt_metric_mean,
                "attribution_seconds": scores.metadata.get("seconds"),
                "eap": {k: v for k, v in scores.metadata.items() if k != "task_metadata"},
                "task_metadata": scores.metadata.get("task_metadata", {}),
            },
        )

    def _scores(self, model: Any, task: Mapping[str, Any], seed: int):
        key = (str(task.get("name")), int(task.get("n_prompts", 0)), task.get("prepend_bos"), seed)
        if self._cache_model is model and self._cache_key == key:
            return self._cache_scores

        from .eap import compute_eap

        prompts = self._build_prompts(model, task, seed)
        scores = compute_eap(model, prompts)
        self._cache_model, self._cache_key, self._cache_scores = model, key, scores
        self.n_attribution_runs += 1
        return scores

    def _build_prompts(self, model: Any, task: Mapping[str, Any], seed: int):
        """The task's clean/corrupted prompt pairs for ``seed``, tokenized for ``model``."""
        name = str(task.get("name"))
        max_batch_size = int(self._kwargs.get("max_batch_size", 32))
        if name == "ioi":
            from ..tasks.ioi import build_ioi_prompts

            return build_ioi_prompts(model.tokenizer, task, seed, max_batch_size=max_batch_size)
        if name == "greater_than":
            from ..tasks.greater_than import build_greater_than_prompts

            return build_greater_than_prompts(model.tokenizer, task, seed, max_batch_size=max_batch_size)
        raise NotImplementedError(
            f"no real prompt generator for task {name!r}. docstring's source and resampling unit "
            "are decided but its generator is not yet ported or checked under this tokenizer."
        )
