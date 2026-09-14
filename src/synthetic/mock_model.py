# [AI-GEN] agent=OpenCode date=2026-08-07 task=Mock model for synthetic dry-runs (Q5 provisional)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY

"""Deterministic mock model for engineering dry-runs (PI provisional unblock).

NOT a scientific object: no real weights, no downloads, no claims about real models
(docs/HUMAN_DECISIONS.md §3.2 Q5/Q6). It exists so the full stack —
ensemble runner, inclusion frequencies, band decomposition, null perturbation,
distance metrics — can be exercised end-to-end with deterministic, seeded behavior.

Design (all mechanical, documented for review):
- Weights: per layer ``L{l}.H{h}.W`` and ``L{l}.MLP.W``, shape (d_model, d_model).
- Node universe: ``L{l}.H{h}`` heads + ``L{l}.MLP`` per layer (ARCHITECTURE.md §2
  component ID scheme).
- ``edge_scores(task_cfg, seed)``: per candidate edge a score in [0,1] mixing a
  seeded per-edge base draw with a weight-coupling term, so that weight perturbation
  (null draws, Stage B) actually changes scores and therefore graphs.
- Perturbation: ``apply_weight_delta`` returns a NEW model (never mutates), matching
  the Perturber protocol contract (interfaces.py).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..common.seeding import create_seed_generator, derive_child_seed


class MockModel:
    """Seeded synthetic model with a weight registry and deterministic edge scores."""

    def __init__(
        self,
        seed: int = 0,
        n_layers: int = 4,
        n_heads: int = 2,
        d_model: int = 8,
        weights: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.seed = int(seed)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.d_model = int(d_model)
        self.weights: dict[str, np.ndarray] = {}
        if weights is not None:
            self.weights = {k: np.array(v, dtype=np.float64) for k, v in weights.items()}
        else:
            rng, _ = create_seed_generator(derive_child_seed(self.seed, "mock-weights"))
            for layer in range(self.n_layers):
                for head in range(self.n_heads):
                    self.weights[self._weight_name(layer, f"H{head}")] = rng.standard_normal(
                        (self.d_model, self.d_model)
                    )
                self.weights[self._weight_name(layer, "MLP")] = rng.standard_normal(
                    (self.d_model, self.d_model)
                )

    @property
    def is_synthetic(self) -> bool:
        return True

    def _weight_name(self, layer: int, node_type_index: str) -> str:
        return f"L{layer}.{node_type_index}.W"

    def nodes(self) -> tuple[str, ...]:
        """Model-agnostic component IDs (ARCHITECTURE.md §2): L{l}.H{h} + L{l}.MLP."""
        out: list[str] = []
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                out.append(f"L{layer}.H{head}")
            out.append(f"L{layer}.MLP")
        return tuple(out)

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(w.shape) for name, w in self.weights.items()}

    def clone(self) -> "MockModel":
        """Deep copy (perturbation and pruning always operate on copies)."""
        return MockModel(
            seed=self.seed,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            weights={k: v.copy() for k, v in self.weights.items()},
        )

    def apply_weight_delta(self, deltas: Mapping[str, np.ndarray]) -> "MockModel":
        """Return a new model with ``deltas`` added to its weights (never mutates)."""
        copy = self.clone()
        for name, delta in deltas.items():
            if name not in copy.weights:
                raise KeyError(f"unknown tensor {name!r}; known: {sorted(copy.weights)}")
            if tuple(delta.shape) != tuple(copy.weights[name].shape):
                raise ValueError(
                    f"delta shape {delta.shape} for {name} != weight shape {copy.weights[name].shape}"
                )
            copy.weights[name] = copy.weights[name] + np.asarray(delta, dtype=np.float64)
        return copy

    def weight_delta_frobenius(self, other: "MockModel") -> dict[str, float]:
        """Per-tensor ||W_self - W_other||_F (Compressor.weight_delta contract).

        Uses the rank-agnostic Frobenius helper so the magnitudes fed to the null
        are computed the same way the null consumes them (src/science/matched_magnitude.py).
        """
        if set(other.weights) != set(self.weights):
            raise ValueError("weight registries differ; cannot compute deltas")
        from ..science.matched_magnitude import frobenius_norm

        return {name: frobenius_norm(self.weights[name] - other.weights[name])
                for name in sorted(self.weights)}

    def edge_scores(self, task_cfg: Mapping[str, Any], seed: int) -> dict[str, float]:
        """Deterministic per-edge score in [0, 1]; perturbing weights shifts scores.

        score = clip(0.5 * base + 0.5 * coupling, 0, 1), where:
        - base: seeded per-(task, edge) uniform draw (derive_child_seed);
        - coupling: 0.5 + 0.5 * tanh(<w_src, w_dst> / 2) in [0, 1], scaled so the
          mock score distribution actually spans the seeded grid's edge thresholds
          (~0.90-0.995, threshold_grid.py): row vectors are N(0,1) in d_model=8,
          so <w_src, w_dst> ~ N(0, sqrt(8)) and the /2 denominator puts tanh in the
          saturating regime without collapsing to 0.5.
        Mechanical only; explicitly NOT a scientific scoring rule.
        """
        task_name = str(task_cfg.get("name", "task"))
        nodes = self.nodes()
        scores: dict[str, float] = {}
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                edge_id_ = f"{src}->{dst}"
                rng, _ = create_seed_generator(
                    derive_child_seed(int(seed), task_name, "edge-base", edge_id_)
                )
                base = float(rng.uniform(0.0, 1.0))
                w_src = self._vector_for(src)
                w_dst = self._vector_for(dst)
                coupling = 0.5 + 0.5 * math.tanh(float(w_src @ w_dst) / 2.0)
                scores[edge_id_] = min(max(0.5 * base + 0.5 * coupling, 0.0), 1.0)
        return scores

    def _vector_for(self, node_id: str) -> np.ndarray:
        """Weight row vector for a node: L{l}.H{h} -> L{l}.H{h}.W[0], L{l}.MLP -> L{l}.MLP.W[0]."""
        layer = int(node_id[1:].split(".")[0])
        node_type = node_id.split(".")[1]
        return self.weights[self._weight_name(layer, node_type)][0]


__all__ = ["MockModel"]
