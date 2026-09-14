# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft GPTQ wrapper (compression grid cell)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY

"""GPTQ quantization wrapper (compression grid, proposal §3.2).

Technique reference: Frantar et al. 2022, "GPTQ: Accurate Post-Training
Quantization for Generative Pre-trained Transformers" (arXiv:2210.17323). The real
implementation is delegated to the pinned quantization library at Stage C; no
algorithm code is re-implemented here and no upstream repo is adapted yet
(CLAUDE.md §7 adaptation header will be added at Stage C when the library is pinned).

Calibration data requirement (Q7): real GPTQ needs a calibration corpus. NOT
approved yet — dry-runs use synthetic calibration (src/synthetic/synthetic_calibration.py).
"""

from __future__ import annotations

from typing import Any

from ..interfaces import CompressedModel, Model, PerTensorFrobenius
from ..synthetic.mock_model import MockModel

TECHNIQUE_CITE = "GPTQ: Frantar et al. 2022 (arXiv:2210.17323); delegated to pinned lib at Stage C"


class GptqCompressor:
    """GPTQ compressor (Compressor protocol) — interface-only draft.

    Engineering scope: the dry-run path exists so the grid surface is testable with
    synthetic calibration; it is NOT GPTQ. Real GPTQ runs require Stage C approval,
    a pinned quantization library, verified calibration-data licenses (Q7), and
    RUN MODEL DOWNLOAD.
    """

    def __init__(self, bits: int = 4, calibration: dict[str, Any] | None = None) -> None:
        self.bits = int(bits)
        self.calibration = calibration  # synthetic-calibration dict for dry-runs (Q7)

    def apply(self, model: Model, cfg: Any) -> CompressedModel:
        if isinstance(model, MockModel):
            # Dry-run stand-in ONLY (provisional): quantize per-tensor with the
            # RTN helper so the compression grid surface is exercisable. This is
            # explicitly NOT GPTQ and produces no scientific evidence.
            from .rtn import rtn_quantize

            quantized = {name: rtn_quantize(w, self.bits)[0] for name, w in model.weights.items()}
            return MockModel(
                seed=model.seed,
                n_layers=model.n_layers,
                n_heads=model.n_heads,
                d_model=model.d_model,
                weights=quantized,
            )
        raise NotImplementedError(
            f"{TECHNIQUE_CITE}; real-model GPTQ requires Stage C approval + pinned "
            "library + verified calibration corpus (Q7) + RUN MODEL DOWNLOAD"
        )

    def weight_delta(self, model: Model, cfg: Any) -> PerTensorFrobenius:
        if isinstance(model, MockModel):
            return model.weight_delta_frobenius(self.apply(model, cfg))
        raise NotImplementedError(
            "real-model GPTQ weight_delta requires Stage C approval; "
            "use the engineering path (MockModel) for dry-runs"
        )


__all__ = ["GptqCompressor"]
