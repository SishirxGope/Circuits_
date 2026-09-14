# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft Wanda pruner wrapper (compression grid cell)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY
#
# Adapted from: https://github.com/hecboar/sae-pruning-paper @ 261191804675e2d39d0a265320dbc0bc85afd30a, MIT
#   (arXiv:2603.25325 code release, 2026-07-31)
#   - local fork: sae-pruning-paper-main, revision/src/saediag/pruning.py (read-only).
#   - Q9 RESOLVED 2026-09-12. The pin is INFERRED from the fork's contents, not read off a
#     URL — derivation and the outstanding check are in docs/HUMAN_DECISIONS.md §3.3.
#   - license: THIRD_PARTY_LICENSES/sae-pruning-paper-LICENSE.txt (verified 2026-08-07).

"""Wanda pruning (compression grid, proposal §3.2; matched-sparsity C6 test).

Technique reference: Sun et al. 2023, "A Simple and Effective Pruning Approach for
Large Language Models" (arXiv:2306.11695) — prune the lowest |w| * ||x||_2 scores.
Upstream interface (names verified against the fork)::

    prune_wanda_style_inplace(model, ex2_by_linear, target_sparsity, ...)
    collect_linear_input_second_moment_from_cache(model, token_cache, ...)

Engineering scope: pure numpy scoring/implementation in
src/compression/magnitude_prune.py (prune_wanda); synthetic calibration supplies
the second-moment stand-in for dry-runs (Q7). Real models: NotImplementedError.
"""

from __future__ import annotations

from typing import Any

from ..interfaces import CompressedModel, Model, PerTensorFrobenius
from ..synthetic.mock_model import MockModel
from .magnitude_prune import prune_wanda

TECHNIQUE_CITE = "Wanda: Sun et al. 2023 (arXiv:2306.11695)"


def synthetic_second_moments(model: MockModel, seed: int = 0) -> dict[str, Any]:
    """Deterministic synthetic second-moment stand-ins for dry-runs (Q7).

    PROVISIONAL: this is NOT activation statistics from a real calibration corpus
    (none is approved). Shape-matched positive arrays so the Wanda math is exercised.
    """
    import numpy as np

    from ..common.seeding import create_seed_generator, derive_child_seed

    rng, _ = create_seed_generator(derive_child_seed(int(seed), "synthetic-second-moments"))
    return {
        name: np.maximum(rng.standard_normal(shape) ** 2, 1e-6) + 1.0
        for name, shape in model.tensor_shapes().items()
    }


class WandaPruner:
    """Wanda pruner (Compressor protocol) — engineering draft."""

    def __init__(self, sparsity: float = 0.3, calibration: dict[str, Any] | None = None) -> None:
        self.sparsity = float(sparsity)
        self.calibration = calibration  # synthetic-calibration dict for dry-runs (Q7)

    def apply(self, model: Model, cfg: Any) -> CompressedModel:
        if isinstance(model, MockModel):
            second_moments = synthetic_second_moments(model, seed=int((cfg or {}).get("seed", 0)))
            pruned = prune_wanda(model.weights, second_moments, self.sparsity, exclude=("lm_head",))
            return MockModel(
                seed=model.seed,
                n_layers=model.n_layers,
                n_heads=model.n_heads,
                d_model=model.d_model,
                weights=pruned,
            )
        raise NotImplementedError(
            f"{TECHNIQUE_CITE}; real-model Wanda wraps saediag.pruning "
            "(sae-pruning-paper @ 261191804675…) and requires Stage C approval "
            "+ verified calibration corpus (Q7) + RUN MODEL DOWNLOAD"
        )

    def weight_delta(self, model: Model, cfg: Any) -> PerTensorFrobenius:
        if isinstance(model, MockModel):
            return model.weight_delta_frobenius(self.apply(model, cfg))
        raise NotImplementedError(
            "real-model Wanda weight_delta requires Stage C approval; "
            "use the engineering path (MockModel) for dry-runs"
        )


__all__ = ["WandaPruner", "synthetic_second_moments"]
