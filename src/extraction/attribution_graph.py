# [AI-GEN] agent=OpenCode date=2026-08-07 task=Implement attribution-graph extractor adapter over circuit-tracer (pipeline A)
# reviewed-by: PENDING
#
# Adapted from: https://github.com/decoderesearch/circuit-tracer @ 8f1e2438df612464e229e44c4a00ff637bf9379b, MIT-style
#   (tag v0.5.2, 2026-07-18; licence text in THIRD_PARTY_LICENSES/)
#   - Q9 RESOLVED 2026-09-12. The upstream README's own links were ambiguous — install and
#     demo links point at github.com/safety-research/circuit-tracer (9 places), the BibTeX
#     at github.com/decoderesearch (1 place). The BibTeX is correct: the safety-research
#     URL returns HTTP 301 to decoderesearch (verified by fetch, not by memory). Commit
#     read from the GitHub API ref for tag v0.5.2.
#   - upstream public interfaces verified against the local fork (circuit-tracer-0.5.2)
#     on 2026-08-08; the contract is pinned as tests in
#     tests/test_node_ids_and_upstream_contract.py so a different pinned commit fails
#     loudly instead of silently producing a wrong graph.

"""Pipeline A: attribution-patching graph extraction (proposal §2.2).

Wraps the ``circuit-tracer`` library. Upstream public interface (verified against the
local fork):

    from circuit_tracer import ReplacementModel, Graph, attribute
    model = ReplacementModel.from_pretrained(
        model_name, transcoders=transcoder_set, backend="transformerlens" | "nnsight")
    graph = attribute(prompt, model, *,
                      attribution_targets=None, max_n_logits=10,
                      desired_logit_prob=0.95, batch_size=512, max_feature_nodes=None,
                      offload=None, verbose=False)          # -> Graph (unpruned adjacency)
    pruned = prune_graph(graph, node_threshold=..., edge_threshold=..., ...)  # -> PruneResult

Upstream ``Graph`` (graph.py), verified 2026-08-08: ``adjacency_matrix`` rows = target
nodes, columns = source nodes; node order ``[active_features, error_nodes, embed_nodes,
logit_nodes]`` with ``cfg.n_layers * len(input_tokens)`` error nodes, ``len(input_tokens)``
embed nodes and ``len(logit_targets)`` logit nodes; ``active_features`` = (layer, pos,
feature_idx) triples; ``selected_features`` indexes INTO ``active_features``.
``prune_graph(graph, node_threshold=0.8, edge_threshold=0.98) -> PruneResult(node_mask,
edge_mask, cumulative_scores)``. Those two thresholds are ENSEMBLE axes in our design
(CIRCUS, arXiv:2603.00523), never hardcoded (CLAUDE.md §2.2 / §5).

⚠️ **NODE-TAXONOMY MISMATCH — read src/extraction/node_ids.py before implementing.**
circuit-tracer emits four node kinds — ``cross layer transcoder``, ``mlp reconstruction
error``, ``embedding``, ``logit`` — and **no attention heads**. ARCHITECTURE.md §2's
``L{layer}.{type}{index}`` scheme and the routing-head projection in
src/science/two_level.py both assume heads exist. Two PI decisions are therefore
outstanding before this extractor can be finished: what the coarse comparison level
means for pipeline A (``node_ids.LEVEL2_OPTIONS``), and whether edges aggregate over
token positions (upstream nodes carry ``pos``; our edge schema does not).

SCIENCE REQUIREMENTS (the Stage A implementation must preserve):
1. Threshold config comes from configs/ensemble/, one cell of the B x S grid.
2. Output uses the ARCHITECTURE.md §2 edge schema with model-agnostic component IDs
   L{layer}.{type}{index} (e.g. L11.H3, L7.MLP).
3. Deterministic given (model, task, config, seed); every random draw seeded from
   ``seed`` and logged (AI_RULES.md 1.1).
4. The dense checkpoint is never mutated (ARCHITECTURE.md §2).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..interfaces import Graph as CanonicalGraph
from ..interfaces import Model
from ..common.schema import Edge


class AttributionGraphExtractor:
    """Pipeline A extractor: attribution-patching graph via circuit-tracer.

    Implements the ``CircuitExtractor`` protocol (src/interfaces.py) for the
    attribution-patching family (CLAUDE.md §2.2, pipeline A).

    One ``extract`` call = one graph for one (threshold config, seed) ensemble cell.
    The B x S ensemble is orchestrated by src/science/circus_wrapper.py, not here.
    """

    def __init__(
        self,
        backend: str = "transformerlens",
        transcoder_set: str | None = None,
        offload: str | None = None,
        batch_size: int = 512,
        max_feature_nodes: int | None = None,
    ) -> None:
        """Configure the upstream circuit-tracer call.

        backend: 'transformerlens' (default) or 'nnsight' (experimental fallback for
            models TransformerLens does not support; slower / less memory-efficient
            upstream, fork README).
        transcoder_set: HuggingFace repo ID (e.g. mntss/gemma-scope-transcoders),
            'gemma'/'llama' shortcut, or a local path. For Pythia (no transcoder set
            exists) this pipeline must be replaced by the dense-node variant.
        offload: memory optimization ('cpu', 'disk', or None) passed through upstream.
        batch_size: backward-pass batch size passed through upstream.
        """
        self.backend = backend
        self.transcoder_set = transcoder_set
        self.offload = offload
        self.batch_size = batch_size
        self.max_feature_nodes = max_feature_nodes

    def _load_replacement_model(self, model_cfg: Mapping[str, Any]):
        """Load a frozen circuit-tracer ReplacementModel (lazy import).

        ⚠️ TODO: verify exact ``from_pretrained`` kwargs for the pinned circuit-tracer
        commit before implementation (fork replacement_model.py exposes
        ``from_pretrained(model_name, transcoders=..., backend=...)``).
        """
        try:
            from circuit_tracer import ReplacementModel  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - upstream package absent
            raise RuntimeError(
                "circuit-tracer is not installed; this extractor needs the upstream "
                "package (local fork: circuit-tracer-0.5.2, `pip install .`)"
            ) from exc

        # TODO(Stage A engineering): instantiate ReplacementModel.from_pretrained with
        # model_cfg["hf_id"], self.transcoder_set, backend=self.backend; freeze it.
        _ = (model_cfg, ReplacementModel)
        raise NotImplementedError("Stage A engineering: load + freeze ReplacementModel")

    def extract(self, model: Model, task: Mapping[str, Any], config: Mapping[str, Any], seed: int) -> CanonicalGraph:
        """Compute one attribution graph for (model, task) under ``config``.

        This method validates the ensemble cell + task config *before* touching any
        upstream package, so config errors are caught cheaply in CI:
        1. ``_validate_task_config(task)`` - required task keys present.
        2. ``_thresholds_from_config(config)`` - node/edge thresholds are in (0, 1).

        Stage A engineering sequence (after the above checks pass):
        1. Load ``ReplacementModel`` (transcoder set from the model config).
        2. Build attribution targets from the task config (attribution_target,
           max_seq_len, n_prompts).
        3. Call ``circuit_tracer.attribute(...)`` -> unpruned upstream Graph.
        4. Apply this ensemble cell's node/edge thresholds via ``prune_graph``.
        5. ``_to_canonical_graph``: upstream Graph + PruneResult -> our Graph with
           component IDs L{layer}.{type}{index} and ARCHITECTURE.md §2 edge schema.

        All randomness seeded from ``seed`` (AI_RULES.md 1.1).
        """
        _validate_task_config(task)
        node_thr, edge_thr = _thresholds_from_config(config)
        # ⚠️ TODO: verify circuit-tracer API before implementation (see module docstring)
        _ = (model, node_thr, edge_thr, seed)
        raise NotImplementedError(
            "Stage A engineering: wrap circuit_tracer.attribute + prune_graph "
            "(needs upstream package + pinned transcoder set; requires RUN MODEL DOWNLOAD)"
        )

    def _to_canonical_graph(
        self,
        upstream_graph: Any,
        pruned: Any,
        config: Mapping[str, Any],
        seed: int,
    ) -> CanonicalGraph:
        """Convert an upstream circuit-tracer Graph + PruneResult to our schema.

        Upstream node ordering (graph.py docstring): [active_features, error_nodes,
        embed_nodes, logit_nodes]; adjacency rows = targets, cols = sources.

        ⚠️ TODO: verify prune_graph/PruneResult field names at implementation time;
        exact node-index -> component-ID mapping (feature nodes -> L{layer}.H{head}
        style IDs per ARCHITECTURE.md §2) must be pinned in Stage A engineering.
        """
        _ = (upstream_graph, pruned, config, seed)
        raise NotImplementedError(
            "Stage A engineering: convert circuit-tracer Graph to canonical edge schema"
        )

    def _to_edges(self, srcs: Any, dsts: Any, config_id: str, seed: int) -> tuple[Edge, ...]:
        """Internal helper for the conversion above (kept for Stage A engineering)."""
        _ = (srcs, dsts, config_id, seed)
        raise NotImplementedError


def _thresholds_from_config(config: Mapping[str, Any]) -> tuple[float, float]:
    """Extract and validate the ensemble cell's node/edge thresholds.

    Thresholds are ensemble axes (CIRCUS, arXiv:2603.00523) driven by
    configs/ensemble/, never hardcoded. Both must be present and in (0, 1).
    Pure function (no upstream imports) so CI validates it without circuit-tracer.
    """
    node_thr = config.get("node_threshold")
    edge_thr = config.get("edge_threshold")
    if node_thr is None or edge_thr is None:
        raise ValueError(
            "ensemble cell config must carry node_threshold and edge_threshold "
            f"(got node_threshold={node_thr!r}, edge_threshold={edge_thr!r}); "
            "the B threshold configs come from configs/ensemble/default.yaml (Q4)"
        )
    node_thr, edge_thr = float(node_thr), float(edge_thr)
    if not (0.0 < node_thr < 1.0) or not (0.0 < edge_thr < 1.0):
        raise ValueError(
            f"thresholds must be in (0, 1): node_threshold={node_thr}, edge_threshold={edge_thr}"
        )
    return node_thr, edge_thr


def _validate_task_config(task: Mapping[str, Any]) -> None:
    """Structural validation of a task config (no scientific values chosen).

    Only checks that the keys the Stage A implementation will read exist, so config
    errors surface in CI instead of at the model boundary. Values are PI-owned
    (HUMAN_DECISIONS.md Q6); nothing here validates their magnitude.
    """
    required = ("dataset", "n_prompts", "max_seq_len", "attribution_target")
    missing = [k for k in required if task.get(k) in (None, "")]
    if missing:
        raise ValueError(
            f"task config missing required key(s) {missing}; fix configs/task/*.yaml "
            "(values remain PI-owned, HUMAN_DECISIONS.md Q6)"
        )
