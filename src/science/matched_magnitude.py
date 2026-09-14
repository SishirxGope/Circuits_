# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft matched-magnitude null model (proposal §2.1a)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY
#
# NOVELTY PROTECTION ZONE (AI_RULES.md §3): draft for review only; do not treat as
# approved scientific logic until human approval (APPROVE matched_magnitude).

"""Null model (a): matched-magnitude random perturbation (proposal §2.1a).

Draws random Delta W whose per-tensor Frobenius norms equal the magnitudes produced
by ``Compressor.weight_delta`` (the single source of truth for the null,
ARCHITECTURE.md §4 / interfaces.py) and applies them to a COPY of the model. The R
draws per cell become D_null and are frozen at Stage B (frozen/); the null may never
be re-tuned afterward (AI_RULES.md 1.4).

Engineering implementation (PROVISIONAL):
- Works on numpy tensor registries (MockModel, src/synthetic/mock_model.py) so the
  null pipeline is exercised end-to-end without real models.
- Real-model support lands in Stage B engineering (requires RUN MODEL DOWNLOAD +
  upstream packages); it raises an informative error until then.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..interfaces import Model, PerTensorFrobenius, Rng
from ..synthetic.mock_model import MockModel
from ..common.seeding import create_seed_generator, derive_child_seed


def frobenius_norm(tensor: np.ndarray) -> float:
    """||T||_F for a tensor of ANY rank = the 2-norm of its flattened entries.

    ``np.linalg.norm(x, "fro")`` is defined for 2-D input only and raises on 1-D or
    3-D arrays; real checkpoints carry 1-D bias / LayerNorm tensors alongside 2-D
    weight matrices, so the null would have crashed at Stage B on real models.
    Flattening is numerically identical to "fro" for 2-D input, so no already-computed
    magnitude changes (generalised 2026-08-08; behaviour-preserving).
    """
    return float(np.linalg.norm(np.asarray(tensor, dtype=np.float64).ravel()))


def _delta_for(name: str, norm: float, shape: tuple[int, ...], rng: Rng) -> np.ndarray:
    """One random Delta W tensor with ||Delta W||_F == norm (or zeros for norm==0)."""
    if norm == 0.0:
        return np.zeros(shape, dtype=np.float64)
    z = np.asarray(rng.standard_normal(shape), dtype=np.float64)
    scale = norm / frobenius_norm(z)
    return z * scale


def generate_null_deltas(
    magnitudes: Mapping[str, float],
    shapes: Mapping[str, tuple[int, ...]],
    R: int,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    """Return R deterministic matched-magnitude delta sets.

    Deterministic given (magnitudes, shapes, R, seed) — every draw comes from a
    seeded generator derived from ``seed`` (AI_RULES.md 1.1). The per-tensor
    Frobenius norm of each returned delta equals ``magnitudes[name]`` exactly.
    """
    if not (isinstance(R, int) and R > 0):
        raise ValueError(f"R must be a positive int, got {R!r}")
    unknown = sorted(set(magnitudes) - set(shapes))
    if unknown:
        raise ValueError(f"magnitudes for unknown tensors: {unknown} (no shapes provided)")
    missing = sorted(set(shapes) - set(magnitudes))
    if missing:
        raise ValueError(f"no magnitude provided for tensors: {missing}")
    rng, _ = create_seed_generator(derive_child_seed(int(seed), "matched-magnitude"))
    deltas: list[dict[str, np.ndarray]] = []
    for _ in range(R):
        deltas.append(
            {
                name: _delta_for(name, float(magnitudes[name]), shapes[name], rng)
                for name in sorted(shapes)
            }
        )
    return deltas


class MatchedMagnitudePerturber:
    """Perturber protocol implementation (interfaces.py) for engineering dry-runs.

    ``apply(model, magnitudes, rng)`` perturbs a COPY of the model; the dense model
    is never mutated (ARCHITECTURE.md §2). Works with MockModel (numpy registry);
    real models raise NotImplementedError until Stage B engineering.
    """

    def apply(self, model: Model, magnitudes: PerTensorFrobenius, rng: Rng) -> Model:
        if isinstance(model, MockModel):
            deltas = {
                name: _delta_for(name, float(magnitudes[name]), shape, rng)
                for name, shape in model.tensor_shapes().items()
                if name in magnitudes
            }
            if set(deltas) != set(magnitudes):
                raise ValueError(
                    f"magnitude keys {sorted(magnitudes)} do not match model tensors "
                    f"{sorted(model.tensor_shapes())}"
                )
            return model.apply_weight_delta(deltas)
        raise NotImplementedError(
            "MatchedMagnitudePerturber currently supports MockModel (engineering dry-run "
            "only, Q3/Q9). Real models require Stage B engineering + pinned upstream "
            "packages and an explicit RUN MODEL DOWNLOAD approval."
        )


__all__ = ["generate_null_deltas", "MatchedMagnitudePerturber", "frobenius_norm"]
