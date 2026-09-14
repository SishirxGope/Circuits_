# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft RTN quantizer wrapper (compression grid cell)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY

"""RTN (round-to-nearest) quantization (compression grid, proposal §3.2).

Technique reference: RTN is the round-to-nearest baseline described in the GPTQ
paper (Frantar et al. 2022, "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers", arXiv:2210.17323). Standard technique — no
upstream code adapted; no repository pin required (CLAUDE.md §7 applies only to
adapted code).

Engineering scope (PROVISIONAL):
- ``rtn_quantize`` is a pure numpy round-to-nearest uniform quantizer (testable now).
- The Compressor protocol methods work on MockModel (numpy registry) for dry-runs;
  real-model quantization raises NotImplementedError (needs Stage C approval +
  RUN MODEL DOWNLOAD; never run real compression in engineering mode).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..interfaces import CompressedModel, Model, PerTensorFrobenius
from ..synthetic.mock_model import MockModel

TECHNIQUE_CITE = "RTN: round-to-nearest baseline, Frantar et al. 2022 (arXiv:2210.17323)"


def rtn_quantize(weights: np.ndarray, bits: int = 4) -> tuple[np.ndarray, dict[str, Any]]:
    """Uniform symmetric round-to-nearest quantization of a weight tensor.

    Returns (quantized_weights, {scale, zero_point}) with scale =
    max(|w|) / (2^(bits-1) - 1) and zero_point = 0 (symmetric). Pure numpy,
    deterministic; engineering helper only — the real pipeline uses the pinned
    quantization library at Stage C.
    """
    if not (isinstance(bits, int) and 2 <= bits <= 16):
        raise ValueError(f"bits must be an int in [2, 16], got {bits!r}")
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        raise ValueError("empty weight tensor")
    qmax = 2 ** (bits - 1) - 1
    scale = float(np.max(np.abs(w))) / qmax
    if scale == 0.0:
        return np.zeros_like(w), {"scale": 0.0, "zero_point": 0}
    q = np.clip(np.round(w / scale), -qmax, qmax)
    return q * scale, {"scale": scale, "zero_point": 0}


class RtnQuantizer:
    """RTN compressor (Compressor protocol, interfaces.py) — engineering draft."""

    def __init__(self, bits: int = 4) -> None:
        self.bits = int(bits)

    def apply(self, model: Model, cfg: Any) -> CompressedModel:
        if isinstance(model, MockModel):
            quantized = {
                name: rtn_quantize(w, self.bits)[0] for name, w in model.weights.items()
            }
            return MockModel(
                seed=model.seed,
                n_layers=model.n_layers,
                n_heads=model.n_heads,
                d_model=model.d_model,
                weights=quantized,
            )
        raise NotImplementedError(
            f"{TECHNIQUE_CITE}; real-model RTN requires Stage C approval + pinned "
            "quantization tooling + RUN MODEL DOWNLOAD (engineering mode never "
            "compresses real models)"
        )

    def weight_delta(self, model: Model, cfg: Any) -> PerTensorFrobenius:
        if isinstance(model, MockModel):
            return model.weight_delta_frobenius(self.apply(model, cfg))
        raise NotImplementedError(
            "real-model RTN weight_delta requires Stage C approval; "
            "use the engineering path (MockModel) for dry-runs"
        )


__all__ = ["RtnQuantizer", "rtn_quantize"]
