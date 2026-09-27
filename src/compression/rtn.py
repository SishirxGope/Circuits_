# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft RTN quantizer wrapper (compression grid cell)
# reviewed-by: PENDING
# modified: [AI-GEN] agent=Claude date=2026-09-27 task=B2.1 real-model RTN (deploy/BLOCKERS.md)
# scientific-status: PROVISIONAL_ENGINEERING_ONLY (mock path); real path REVIEW PENDING

"""RTN (round-to-nearest) quantization (compression grid, proposal §3.2).

Technique reference: RTN is the round-to-nearest baseline described in the GPTQ
paper (Frantar et al. 2022, "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers", arXiv:2210.17323). Standard technique — no
upstream code adapted; no repository pin required (CLAUDE.md §7 applies only to
adapted code).

Two paths, deliberately separate:

- ``rtn_quantize`` — pure numpy, per-TENSOR symmetric, used by the MockModel dry-run.
- ``rtn_quantize_torch`` — per-OUTPUT-CHANNEL symmetric, used on real models, per
  deploy/BLOCKERS.md B2: "quantize per output channel, write back as float32".

**Simulated quantization.** Weights are quantized and immediately dequantized back to
the model's own dtype. We study quantization *error*, not INT4 deployment, so the
round-trip is the point: the resulting model runs in its original precision with
exactly the error a real INT-k quantizer would have introduced.

**Which tensors.** Attention and MLP projection matrices only - the standard PTQ
surface (GPTQ §4, AWQ §3). Embeddings, unembedding, layer norms and biases are left
alone; quantizing them is a different experiment and would change what the compression
grid means.

**Per output channel, reducing over dim=-2.** TransformerLens stores weights as
``[..., d_in, d_out]`` (``W_Q`` is ``[n_heads, d_model, d_head]``, ``W_out`` is
``[d_mlp, d_model]``), so the input dimension is always -2. Reducing there gives one
scale per output channel *within each head*, which is what per-channel RTN means.
Reducing over the last dim instead would share a scale across heads - a different and
wrong quantizer.

**GQA.** ``named_parameters()`` yields the COMPACT ``_W_K``/``_W_V``
(``[n_key_value_heads, ...]``), never the expanded ``W_K``/``W_V`` properties. That is
correct and load-bearing: quantizing the property would quantize
``n_heads/n_key_value_heads`` duplicate copies and inflate the Frobenius norms this
function reports by the same factor, silently mis-scaling the matched-magnitude null.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..interfaces import CompressedModel, Model, PerTensorFrobenius
from ..synthetic.mock_model import MockModel

# Parameter-name suffixes that name a projection matrix. Substring match, so "W_K"
# also catches the GQA parameter "_W_K" - which is exactly the tensor we want.
QUANTIZED_SUFFIXES: tuple[str, ...] = ("W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_gate")

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



def rtn_quantize_torch(w: Any, bits: int = 4) -> Any:
    """Per-output-channel symmetric RTN on a torch tensor, dequantized in place.

    Scales are computed over dim=-2 (the input dimension in TransformerLens's
    ``[..., d_in, d_out]`` layout), giving one scale per output channel per head.
    Arithmetic runs in float32 regardless of the model dtype, then casts back, so a
    bf16 model does not compound bf16 rounding into the quantization error we measure.
    All-zero channels keep a scale of 1 and quantize to zero rather than dividing by 0.
    """
    import torch

    if not (isinstance(bits, int) and 2 <= bits <= 16):
        raise ValueError(f"bits must be an int in [2, 16], got {bits!r}")
    if w.ndim < 2:
        raise ValueError(f"expected a matrix with an input dimension, got shape {tuple(w.shape)}")
    qmax = 2 ** (bits - 1) - 1
    work = w.detach().to(torch.float32)
    scale = work.abs().amax(dim=-2, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(work / scale), -qmax, qmax)
    return (q * scale).to(w.dtype)


def _is_torch_model(model: Any) -> bool:
    return hasattr(model, "named_parameters") and hasattr(model, "state_dict")


def quantizable_parameters(model: Any) -> list[str]:
    """Names of the projection matrices this compressor touches, in model order."""
    return [
        name
        for name, param in model.named_parameters()
        if param.ndim >= 2 and any(suffix in name.split(".")[-1] for suffix in QUANTIZED_SUFFIXES)
    ]


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
        if _is_torch_model(model):
            import copy

            import torch

            # deepcopy, not in-place: the Compressor protocol requires apply() to be
            # side-effect-free w.r.t. the dense model, which Stage B re-reads for every
            # null draw. Costs one model's worth of memory for the duration.
            out = copy.deepcopy(model)
            params = dict(out.named_parameters())
            with torch.no_grad():
                for name in quantizable_parameters(out):
                    params[name].copy_(rtn_quantize_torch(params[name], self.bits))
            return out
        raise NotImplementedError(
            f"{TECHNIQUE_CITE}; model type {type(model).__name__} is neither MockModel "
            "nor a torch module exposing named_parameters()"
        )

    def weight_delta(self, model: Model, cfg: Any) -> PerTensorFrobenius:
        if isinstance(model, MockModel):
            return model.weight_delta_frobenius(self.apply(model, cfg))
        if _is_torch_model(model):
            import torch

            # MEASURED, never predicted (deploy/BLOCKERS.md B2): the null matches these
            # numbers exactly, so an analytic estimate of quantization error would make
            # D_null answer a different question than D. Computed tensor by tensor
            # without copying the model, so this is cheap enough to call on its own.
            out: PerTensorFrobenius = {}
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name not in set(quantizable_parameters(model)):
                        out[name] = 0.0
                        continue
                    delta = rtn_quantize_torch(param, self.bits).to(torch.float32) - param.detach().to(torch.float32)
                    out[name] = float(torch.linalg.vector_norm(delta).item())
            return out
        raise NotImplementedError(
            f"{TECHNIQUE_CITE}; model type {type(model).__name__} is neither MockModel "
            "nor a torch module exposing named_parameters()"
        )


__all__ = ["QUANTIZED_SUFFIXES", "RtnQuantizer", "quantizable_parameters", "rtn_quantize", "rtn_quantize_torch"]
