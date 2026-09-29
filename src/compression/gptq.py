# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft GPTQ wrapper (compression grid cell)
# modified: [AI-GEN] agent=Claude date=2026-09-29 task=B2 real-model GPTQ in torch (deploy/BLOCKERS.md)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY (mock path); real path REVIEW PENDING
#
# Adapted from: https://github.com/IST-DASLab/gptq @ 2d65066eeb06a5c9ff5184d8cebdf33662c67faf, Apache-2.0
#   - gptq.py::GPTQ.add_batch / fasterquant  -> gptq_quantize_matrix, the Hessian accumulation
#   - quant.py::Quantizer.find_params / quantize (perchannel=True, mse=False) -> _row_grid, _quantize
#   - llama.py::llama_sequential (the layer loop; true_sequential off, the default)
#   - Changed from upstream: ported from nn.Linear to TransformerLens einsum weights, and
#     from fp16 to the pre-registered bf16 calibration forward. Licence text:
#     THIRD_PARTY_LICENSES/gptq-LICENSE.txt

"""GPTQ quantization (compression grid, proposal §3.2).

Technique reference: Frantar et al. 2022, "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers" (arXiv:2210.17323).

**Why a port and not the package.** ``auto-gptq`` ships x86-64 wheels and does not build
on the Spark's aarch64 (deploy/BLOCKERS.md), and it acts on ``nn.Linear`` modules, which
TransformerLens does not have. The algorithm is short; it is ported line for line from
the authors' reference code at the pinned commit, so both machines run identical code.

**Settings are the reference code's defaults, not choices made here.** The grid
pre-registers only ``bits: 4`` (configs/compression/gptq_int4.yaml). Everything else is
what ``llama.py`` does when run without flags:

    per-row (perchannel=True), asymmetric (--sym off), min-max grid (mse=False),
    no grouping (--groupsize -1), no act-order, blocksize 128, percdamp 0.01,
    layers quantized in order, each fed the output of the already-quantized layers.

Each is a constructor argument, so a different pre-registered setting is a config change.
The calibration tokens are the Q7 cache (fineweb-edu, 300k tokens, seed 7) rather than the
reference's 128 x 2048 C4 samples: Q7 fixed one corpus for Wanda, GPTQ and AWQ.

**Simulated quantization**, as for RTN: weights are quantized and dequantized back to the
model's dtype, so ``weight_delta`` is the measured error the null is matched to.
"""

from __future__ import annotations

import copy
from typing import Any

from ..interfaces import CompressedModel, Model, PerTensorFrobenius
from ..synthetic.mock_model import MockModel
from .layerwise import (
    as_matrix,
    block_source,
    cast_copy,
    embed_batches,
    exact_float32_matmul,
    from_matrix,
    injected_tokens,
    load_or_compute,
    measured_delta,
    projections_by_layer,
    run_block,
    with_tensors,
)
from .torch_weights import is_torch_model

TECHNIQUE_CITE = "GPTQ: Frantar et al. 2022 (arXiv:2210.17323); IST-DASLab/gptq @ 2d65066e"


def _row_grid(w: Any, bits: int, sym: bool) -> tuple[Any, Any, int]:
    """quant.py::Quantizer.find_params with perchannel=True, weight=True, mse=False.

    The range always includes 0 (``xmin <= 0 <= xmax``); an all-zero row gets [-1, 1].
    Returns (scale [rows, 1], zero [rows, 1], maxq).
    """
    import torch

    maxq = 2**bits - 1
    zero_row = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
    xmin = torch.minimum(w.min(1).values, zero_row)
    xmax = torch.maximum(w.max(1).values, zero_row)
    if sym:
        xmax = torch.maximum(xmin.abs(), xmax)
        xmin = torch.where(xmin < 0, -xmax, xmin)
    empty = (xmin == 0) & (xmax == 0)
    xmin = torch.where(empty, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(empty, torch.full_like(xmax, 1.0), xmax)
    scale = (xmax - xmin) / maxq
    zero = torch.full_like(scale, (maxq + 1) / 2) if sym else torch.round(-xmin / scale)
    return scale.unsqueeze(1), zero.unsqueeze(1), maxq


def _quantize(x: Any, scale: Any, zero: Any, maxq: int) -> Any:
    """quant.py::quantize."""
    import torch

    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def gptq_quantize_matrix(
    w: Any,
    hessian: Any,
    *,
    bits: int = 4,
    sym: bool = False,
    group_size: int = -1,
    percdamp: float = 0.01,
    blocksize: int = 128,
) -> Any:
    """gptq.py::GPTQ.fasterquant on one ``[rows, columns]`` matrix; returns the quantized matrix.

    ``hessian`` is ``2/N * X^T X`` over the matrix's inputs (``[columns, columns]``). Its
    overall scale does not matter - the damping is relative to its mean diagonal - but its
    shape of course does. act-order and static groups are not ported: the reference runs
    without them by default and no cell asks for them.
    """
    import torch

    if not (isinstance(bits, int) and 2 <= bits <= 16):
        raise ValueError(f"bits must be an int in [2, 16], got {bits!r}")
    columns = w.shape[1]
    if tuple(hessian.shape) != (columns, columns):
        raise ValueError(f"Hessian {tuple(hessian.shape)} does not match {columns} input columns")

    with exact_float32_matmul():
        W = w.detach().clone().to(torch.float32)
        H = hessian.detach().clone().to(device=W.device, dtype=torch.float32)
        scale, zero, maxq = _row_grid(W, bits, sym)  # before dead columns are zeroed, as upstream

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=W.device)
        H[diag, diag] += damp
        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            Hinv = torch.linalg.cholesky(H, upper=True)
        except RuntimeError as exc:  # torch.linalg.LinAlgError subclasses RuntimeError
            raise RuntimeError(
                f"GPTQ: the damped Hessian is not positive definite ({exc}). The reference "
                "code fails here too; more calibration tokens or a larger percdamp is the "
                "remedy, and either is a pre-registration change."
            ) from exc

        Q = torch.zeros_like(W)
        for i1 in range(0, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(i2 - i1):
                col = W1[:, i]
                d = Hinv1[i, i]
                if group_size > 0 and (i1 + i) % group_size == 0:
                    # upstream reads the OUTER W here: updated by earlier blocks, not by
                    # the in-block updates to W1. Kept as-is.
                    scale, zero, _ = _row_grid(W[:, i1 + i:i1 + i + group_size], bits, sym)
                q = _quantize(col.unsqueeze(1), scale, zero, maxq).flatten()
                Q1[:, i] = q
                err1 = (col - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    return Q


def input_hessians(block: Any, batches: list) -> dict[str, Any]:
    """``2/N * X^T X`` for every projection input of ``block`` (gptq.py::GPTQ.add_batch).

    Upstream keeps a running mean over batches of sequences; the sum here, normalised
    once by the token count, is the same matrix up to a constant factor, which GPTQ's
    relative damping makes irrelevant. Accumulated in float32 without TF32, as upstream.
    """
    import torch

    sums: dict[str, Any] = {}
    counts: dict[str, int] = {}

    def accumulate(source: str, x: Any) -> None:
        x2 = x.reshape(-1, x.shape[-1]).to(torch.float32)
        with exact_float32_matmul():
            gram = x2.t().matmul(x2)
        sums[source] = gram if source not in sums else sums[source] + gram
        counts[source] = counts.get(source, 0) + int(x2.shape[0])

    run_block(block, batches, on_input=accumulate)
    return {source: total * (2.0 / counts[source]) for source, total in sums.items()}


def gptq_quantize_model(
    model: Any,
    tokens: Any,
    *,
    n_seq: int,
    bits: int,
    sym: bool,
    group_size: int,
    percdamp: float,
    blocksize: int,
    batch_size: int,
    dtype: Any,
) -> dict[str, Any]:
    """llama.py::llama_sequential for a HookedTransformer; returns ``{param name: quantized}``.

    Per layer: Hessians from the layer's inputs (the output of the quantized layers
    before it), every projection quantized against its own input's Hessian, then the
    quantized layer run to produce the next layer's inputs. ``model`` is not modified.
    """
    import torch

    layers = projections_by_layer(model)
    batches = embed_batches(model, tokens, n_seq=n_seq, batch_size=batch_size, dtype=dtype)
    out: dict[str, Any] = {}
    for index, block in enumerate(model.blocks):
        kinds = layers[index]
        if not kinds:
            continue
        work = copy.deepcopy(block)  # quantized in place; the dense block is untouched
        hessians = input_hessians(cast_copy(work, dtype), batches)
        params = dict(work.named_parameters())
        with torch.no_grad():
            for kind, local in kinds.items():
                source = block_source(kind)
                if source not in hessians:
                    raise RuntimeError(f"layer {index}: no input reached {source} for {local}")
                param = params[local]
                quantized = gptq_quantize_matrix(
                    as_matrix(kind, param.detach().to(torch.float32)),
                    hessians[source],
                    bits=bits,
                    sym=sym,
                    group_size=group_size,
                    percdamp=percdamp,
                    blocksize=blocksize,
                )
                param.copy_(from_matrix(kind, quantized, tuple(param.shape)).to(param.dtype))
                out[f"blocks.{index}.{local}"] = param.detach().clone()
        del hessians
        batches = run_block(cast_copy(work, dtype), batches)
    return out


class GptqCompressor:
    """GPTQ compressor (Compressor protocol).

    ``calibration`` injects a token array directly (tests, the preflight probe). Left as
    None, a real model calibrates on the Q7 cache named by the cell's config, and the
    result is cached on disk so Stage B and Stage C share one compression.
    """

    def __init__(
        self,
        bits: int = 4,
        calibration: Any = None,
        *,
        sym: bool = False,
        group_size: int = -1,
        percdamp: float = 0.01,
        blocksize: int = 128,
        batch_size: int = 4,
    ) -> None:
        self.bits = int(bits)
        self.calibration = calibration
        self.sym = bool(sym)
        self.group_size = int(group_size)
        self.percdamp = float(percdamp)
        self.blocksize = int(blocksize)
        self.batch_size = int(batch_size)

    def settings(self) -> dict[str, Any]:
        return {
            "bits": self.bits, "sym": self.sym, "group_size": self.group_size,
            "percdamp": self.percdamp, "blocksize": self.blocksize,
            "batch_size": self.batch_size, "reference": "IST-DASLab/gptq@2d65066e",
        }

    def _quantized(self, model: Any, cfg: Any) -> dict[str, Any]:
        def compute(tokens: Any, n_seq: int, dtype: Any) -> dict[str, Any]:
            return gptq_quantize_model(
                model, tokens, n_seq=n_seq, bits=self.bits, sym=self.sym,
                group_size=self.group_size, percdamp=self.percdamp,
                blocksize=self.blocksize, batch_size=self.batch_size, dtype=dtype,
            )

        return calibrated_tensors(self, model, cfg, "gptq", compute)

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
        if is_torch_model(model):
            return with_tensors(model, self._quantized(model, cfg))
        raise NotImplementedError(
            f"{TECHNIQUE_CITE}: GptqCompressor supports MockModel and TransformerLens "
            f"models; got {type(model).__name__}."
        )

    def weight_delta(self, model: Model, cfg: Any) -> PerTensorFrobenius:
        if isinstance(model, MockModel):
            return model.weight_delta_frobenius(self.apply(model, cfg))
        if is_torch_model(model):
            return measured_delta(model, self.apply(model, cfg))
        raise NotImplementedError(
            "gptq weight_delta supports MockModel and TransformerLens models; "
            f"got {type(model).__name__}."
        )


def calibrated_tensors(compressor: Any, model: Any, cfg: Any, method: str, compute: Any) -> dict[str, Any]:
    """Shared by GPTQ and AWQ: injected tokens, or the Q7 cache plus the on-disk result.

    ``compute(tokens, n_seq, dtype)`` runs the method. Injected tokens are never cached:
    nothing identifies them. Without either, the cell cannot calibrate and says so.
    """
    from ..calibration.token_cache import CalibrationUnavailable, calibration_spec

    if not hasattr(model, "cfg") or not hasattr(model, "blocks"):
        raise TypeError(
            f"{method}: needs a TransformerLens HookedTransformer (cfg and blocks); "
            f"got {type(model).__name__}"
        )
    if compressor.calibration is not None:
        tokens, n_seq = injected_tokens(compressor.calibration, int(model.cfg.d_vocab))
        # the cell's calibration dtype when there is a cell, else the model's own
        dtype = calibration_spec(cfg)["dtype"] if cfg and cfg.get("calibration") else None
        return compute(tokens, n_seq, dtype)
    if not cfg:
        raise CalibrationUnavailable(
            f"{method} needs the cell's resolved config to locate its calibration cache"
        )
    dtype = calibration_spec(cfg)["dtype"]
    return load_or_compute(
        model, cfg, method=method, settings=compressor.settings(),
        compute=lambda tokens, n_seq: compute(tokens, n_seq, dtype),
    )


__all__ = [
    "GptqCompressor",
    "calibrated_tensors",
    "gptq_quantize_matrix",
    "gptq_quantize_model",
    "input_hessians",
]
