# [AI-GEN] agent=Claude date=2026-09-27 task=B1 - torch adapter so the existing matched-magnitude null works on real models
# reviewed-by: PENDING

"""Weight plumbing that lets the matched-magnitude null run on a real model.

**Why this module exists.** ``src/science/matched_magnitude.py`` is already
model-agnostic: ``generate_null_deltas(magnitudes, shapes, R, seed)`` takes two plain
dicts and returns numpy arrays. What it cannot do is read shapes off a real model or
write deltas back into one - ``tensor_shapes()`` and ``apply_weight_delta()`` are
MockModel methods. This module supplies both for torch models, so the frozen null
generator itself is used **unchanged**. That matters: the null is the paper's
pre-registered contribution and lives in the Novelty Protection Zone (AI_RULES.md §3).
Perturbation semantics are not reimplemented here, only plumbed.

**Which tensors get perturbed.** MockModel's ``tensor_shapes()`` returns exactly the
projection matrices it stores, so the mock null already perturbs only the tensors the
compressor touched. ``null_draw_inputs`` reproduces that on a real model by restricting
both dicts to the tensors whose measured magnitude is nonzero. Two reasons:

1. It keeps the real and mock paths semantically identical rather than perturbing
   embeddings and layer norms that no compression cell moves.
2. ``generate_null_deltas`` allocates float64, i.e. 8 bytes per parameter per draw.
   Passing every parameter would allocate zero-filled deltas for the embedding and
   unembedding matrices - the largest tensors in a small model - for no effect.

A tensor the compressor selected but whose delta measured exactly 0.0 is dropped, which
is equivalent: ``_delta_for`` returns zeros for a zero magnitude anyway.

**Precision caveat.** A perturbation is applied in float32 and cast back to the
parameter's own dtype. For a float32 checkpoint the realized ``||dW||_F`` equals the
requested magnitude. For a **bfloat16** model the cast rounds, so the realized norm only
approximates it - the null is matched to the precision the model is stored in. This is
one more reason the Gemma dtype question (config says bfloat16, checkpoint says float32)
has to be settled before Gemma cells run, not after.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np


def is_torch_model(model: Any) -> bool:
    return hasattr(model, "named_parameters") and hasattr(model, "state_dict")


def tensor_shapes(model: Any) -> dict[str, tuple[int, ...]]:
    """Shapes of every perturbable tensor, keyed exactly as ``named_parameters()`` is.

    Delegates to MockModel's own method when present so the two paths cannot diverge.
    Under GQA the keys are the compact ``_W_K``/``_W_V`` parameters, matching what
    ``RtnQuantizer.weight_delta`` measures - never the expanded ``W_K``/``W_V``
    properties, whose norms are inflated by ``n_heads // n_key_value_heads``.
    """
    if hasattr(model, "tensor_shapes"):
        return dict(model.tensor_shapes())
    if not is_torch_model(model):
        raise TypeError(
            f"cannot read tensor shapes from {type(model).__name__}: expected a MockModel "
            "or a torch module exposing named_parameters()"
        )
    return {name: tuple(param.shape) for name, param in model.named_parameters()}


def null_draw_inputs(
    model: Any, magnitudes: Mapping[str, float]
) -> tuple[dict[str, float], dict[str, tuple[int, ...]]]:
    """``(magnitudes, shapes)`` for ``generate_null_deltas``, restricted to what moved.

    ``generate_null_deltas`` requires the two key sets to be equal and raises otherwise,
    so they are built together here rather than by two independent callers.
    """
    shapes = tensor_shapes(model)
    touched = {
        name: float(value) for name, value in magnitudes.items() if float(value) > 0.0
    }
    unknown = sorted(set(touched) - set(shapes))
    if unknown:
        raise ValueError(
            f"compressor reported a nonzero magnitude for tensors the model does not "
            f"expose: {unknown}. The compressor and the null are reading different "
            "parameter names, so the null would be matched to the wrong tensors."
        )
    if not touched:
        raise ValueError(
            "no tensor has a nonzero weight delta, so every null draw would be a zero "
            "perturbation and D_null would collapse to a spike at 0"
        )
    return touched, {name: shapes[name] for name in touched}


def apply_weight_delta(model: Any, deltas: Mapping[str, np.ndarray]) -> Any:
    """Return a COPY of ``model`` with ``deltas`` added to the named parameters.

    Side-effect-free with respect to ``model``: Stage B re-reads the dense model for
    every one of the R draws, so an in-place add would compound across draws and
    silently corrupt R-1 of them (ARCHITECTURE.md §2, §4).
    """
    if hasattr(model, "apply_weight_delta"):
        return model.apply_weight_delta(deltas)
    if not is_torch_model(model):
        raise TypeError(
            f"cannot apply a weight delta to {type(model).__name__}: expected a MockModel "
            "or a torch module exposing named_parameters()"
        )

    import torch

    out = copy.deepcopy(model)
    params = dict(out.named_parameters())

    unknown = sorted(set(deltas) - set(params))
    if unknown:
        raise ValueError(f"delta names absent from the model: {unknown}")

    with torch.no_grad():
        for name, delta in deltas.items():
            param = params[name]
            array = np.asarray(delta)
            if tuple(array.shape) != tuple(param.shape):
                raise ValueError(
                    f"delta for {name} has shape {tuple(array.shape)} but the parameter "
                    f"is {tuple(param.shape)}"
                )
            # float32 arithmetic regardless of the model dtype, then cast back, so a
            # bf16 model does not compound bf16 rounding into the perturbation.
            shift = torch.as_tensor(array, dtype=torch.float32, device=param.device)
            param.copy_((param.detach().to(torch.float32) + shift).to(param.dtype))
    return out


__all__ = ["apply_weight_delta", "is_torch_model", "null_draw_inputs", "tensor_shapes"]
