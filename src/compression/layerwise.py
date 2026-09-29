# [AI-GEN] agent=Claude date=2026-09-29 task=B2 GPTQ/AWQ - block-by-block calibration walk over TransformerLens
# reviewed-by: PENDING

"""The calibration walk GPTQ and AWQ share: embed once, then run the model block by block.

Both reference implementations quantize one decoder layer at a time. They catch the
input to layer 0 with a "Catcher" module, then feed each layer the previous layer's
output (IST-DASLab/gptq ``llama.py::llama_sequential``; mit-han-lab/llm-awq
``pre_quant.py::run_awq``). This module is that loop for TransformerLens:

- :func:`embed_batches` is the Catcher. A pre-hook on ``blocks[0]`` records the residual
  *and the keyword arguments the model passes to its blocks* (attention mask, shortformer
  embedding), then stops the forward. Replaying blocks with those exact kwargs reproduces
  the model's own residual stream; the tests check this bit for bit.
- :func:`run_block` runs one block over every batch and hands each projection's input to
  a callback. Inputs are read where ``src/calibration/second_moment.py`` reads them (its
  ``INPUT_SOURCES`` is the single table), so GPTQ's Hessians, AWQ's activation scales and
  Wanda's E[x^2] all see the same activations.

**What the two methods do differently with the walk.** GPTQ feeds layer l+1 the output
of the *quantized* layer l (``llama.py`` recomputes ``outs`` after ``fasterquant``). AWQ
feeds it the output of the *dense* layer l (``run_awq`` computes the next ``inps`` before
searching scales, and the scaling itself preserves the function). Each caller chooses.

**Calibration dtype.** The forward runs in the pre-registered calibration dtype
(``configs/calibration/final.yaml``: bf16, as for Wanda) on cast *copies* of each block.
Weight arithmetic stays in float32 on the model's own weights, and the result is written
back in the model's dtype. The dense model is never modified.

**Matrix views.** GPTQ and AWQ are defined on an ``nn.Linear`` weight ``[out, in]``.
TransformerLens stores ``W_Q`` as ``[heads, d_model, d_head]`` and ``W_O`` as
``[heads, d_head, d_model]``; :func:`as_matrix` / :func:`from_matrix` convert exactly,
with the input index of ``W_O`` ordered ``head * d_head + j`` as in HF's ``o_proj``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ..calibration.cache_lock import cache_lock, partial_path
from ..calibration.second_moment import INPUT_SOURCES
from ..calibration.token_cache import (
    cache_path,
    calibration_root,
    calibration_spec,
    load_token_cache,
    validate_token_cache,
)
from .torch_weights import projection_parameters

Batch = tuple[Any, dict[str, Any]]  # (residual [b, p, d_model], kwargs the model passes to blocks)


class _Stop(Exception):
    """Raised by the Catcher to end the forward once block 0's input is recorded."""


def projection_kind(name: str) -> str:
    """``"W_Q"`` for ``blocks.3.attn.W_Q`` and for the GQA parameter ``blocks.3.attn._W_K``."""
    kind = name.split(".")[-1].lstrip("_")
    if kind not in INPUT_SOURCES:
        raise ValueError(f"{name!r} is not a projection parameter ({sorted(INPUT_SOURCES)})")
    return kind


def block_source(kind: str) -> str:
    """Where a projection's input is read, relative to its block (``"attn:query_input"``)."""
    template = INPUT_SOURCES[kind][0]
    prefix = "blocks.{layer}."
    if not template.startswith(prefix):
        raise ValueError(f"input source {template!r} is not inside a block")
    return template[len(prefix):]


def as_matrix(kind: str, w: Any) -> Any:
    """The ``[out, in]`` matrix an ``nn.Linear`` would hold for this parameter (a view or copy)."""
    if kind in ("W_Q", "W_K", "W_V"):  # [h, d_model, d_head] -> [h * d_head, d_model]
        h, d, dh = w.shape
        return w.permute(0, 2, 1).reshape(h * dh, d)
    if kind == "W_O":  # [h, d_head, d_model] -> [d_model, h * d_head]
        h, dh, d = w.shape
        return w.reshape(h * dh, d).t()
    if kind in ("W_in", "W_gate", "W_out"):  # [d_in, d_out] -> [d_out, d_in]
        return w.t()
    raise ValueError(f"no matrix view for {kind!r}")


def from_matrix(kind: str, m: Any, shape: tuple[int, ...]) -> Any:
    """Inverse of :func:`as_matrix`, returning a contiguous tensor of ``shape``."""
    if kind in ("W_Q", "W_K", "W_V"):
        h, d, dh = shape
        return m.reshape(h, dh, d).permute(0, 2, 1).contiguous()
    if kind == "W_O":
        h, dh, d = shape
        return m.t().reshape(h, dh, d).contiguous()
    if kind in ("W_in", "W_gate", "W_out"):
        return m.t().contiguous()
    raise ValueError(f"no matrix view for {kind!r}")


def projections_by_layer(model: Any) -> list[dict[str, str]]:
    """Per layer, ``{kind: block-local parameter name}`` (``{"W_Q": "attn.W_Q", ...}``)."""
    layers: list[dict[str, str]] = [{} for _ in range(int(model.cfg.n_layers))]
    for name in projection_parameters(model):
        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "blocks" or not parts[1].isdigit():
            raise ValueError(f"{name!r} is a projection outside the blocks; GPTQ/AWQ cannot place it")
        kind = projection_kind(name)
        local = ".".join(parts[2:])
        if kind in layers[int(parts[1])]:
            raise ValueError(f"layer {parts[1]} has two {kind} parameters")
        layers[int(parts[1])][kind] = local
    return layers


def _dtype(value: Any) -> Any:
    import torch

    return getattr(torch, value) if isinstance(value, str) else value


def cast_copy(module: Any, dtype: Any) -> Any:
    """``module`` itself when already in ``dtype``, else a cast deep copy (never in place)."""
    target = _dtype(dtype)
    if target is None or next(module.parameters()).dtype == target:
        return module
    out = copy.deepcopy(module).to(target)
    # TransformerLens norms compute in float32 and cast back to cfg.dtype, so a block cast
    # without its config would hand float32 activations to bf16 weights.
    # HookedTransformer.to() updates the config itself; a bare block does not. The copy
    # owns a deep-copied cfg shared by all its submodules, so the model's is untouched.
    if hasattr(getattr(out, "cfg", None), "dtype"):
        out.cfg.dtype = target
    return out


@contextmanager
def exact_float32_matmul():
    """Disable TF32 for the duration, as IST-DASLab/gptq does at import time.

    TF32 keeps 10 mantissa bits; a Hessian or a Cholesky factor computed that way is not
    the float32 quantity the algorithm is specified in.
    """
    import torch

    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def sequences_needed(tokens: np.ndarray, n_tokens: int | None) -> int:
    """Upstream's rule (reprune.py), shared with Wanda: ``min(ceil(n_tokens / ctx), n_seq)``."""
    n_seq, ctx = tokens.shape
    return n_seq if n_tokens is None else min(math.ceil(int(n_tokens) / ctx), n_seq)


def embed_batches(
    model: Any, tokens: np.ndarray, *, n_seq: int, batch_size: int, dtype: Any = None
) -> list[Batch]:
    """The Catcher: block 0's input for every calibration batch, in the calibration dtype."""
    import torch

    run_model = cast_copy(model, dtype)
    device = next(run_model.parameters()).device
    captured: dict[str, Any] = {}

    def catch(module: Any, args: tuple, kwargs: dict) -> None:
        captured["resid"] = args[0] if args else kwargs["resid_pre"]
        captured["kwargs"] = {
            key: kwargs[key]
            for key in ("shortformer_pos_embed", "attention_mask")
            if kwargs.get(key) is not None
        }
        raise _Stop

    batches: list[Batch] = []
    handle = run_model.blocks[0].register_forward_pre_hook(catch, with_kwargs=True)
    try:
        with torch.no_grad():
            for start in range(0, int(n_seq), int(batch_size)):
                batch = torch.as_tensor(
                    tokens[start:min(start + int(batch_size), int(n_seq))],
                    dtype=torch.long,
                    device=device,
                )
                captured.clear()
                try:
                    run_model(batch, return_type=None)
                except _Stop:
                    pass
                if "resid" not in captured:
                    raise RuntimeError("the forward never reached blocks[0]")
                batches.append((captured["resid"].detach(), dict(captured["kwargs"])))
    finally:
        handle.remove()
    return batches


ATTN_INPUTS = ("query_input", "key_input", "value_input")


def run_block(
    block: Any,
    batches: list[Batch],
    *,
    on_input: Callable[[str, Any], None] | None = None,
    layout: dict[str, Any] | None = None,
) -> list[Batch]:
    """Run ``block`` over every batch; return its outputs with the same kwargs.

    ``on_input(source, x)`` receives each projection input as ``[batch, pos, features]``
    (``hook_z`` with its head axes flattened, W_O's input order). Under split-qkv the
    attention inputs carry a head axis of identical copies (TransformerBlock repeats the
    residual before ln1); one copy is passed on, after checking they really are equal.
    ``layout``, if given, is filled with each attention input's head count (None when the
    block passes no head axis), so a caller can replay the attention module.

    ``x`` may be a view into a larger tensor (the head-repeated attention input): a
    callback that keeps it past the call must keep a ``clone()``, not the view.
    """
    import torch

    handles = []
    if on_input is not None or layout is not None:

        def attn_pre(module: Any, args: tuple, kwargs: dict) -> None:
            for key in ATTN_INPUTS:
                if key not in kwargs:
                    raise RuntimeError(
                        f"attention was called without {key!r}; TransformerLens changed "
                        "how blocks call attention"
                    )
                x = kwargs[key]
                heads = int(x.shape[-2]) if x.ndim == 4 else None
                if layout is not None:
                    layout[key] = heads
                if heads is not None:
                    if not torch.equal(x[..., 0, :], x[..., -1, :]):
                        raise RuntimeError(f"{key} differs across heads; cannot read one projection input")
                    x = x[..., 0, :]
                if on_input is not None:
                    on_input(f"attn:{key}", x)

        handles.append(block.attn.register_forward_pre_hook(attn_pre, with_kwargs=True))

    if on_input is not None:

        def mlp_pre(module: Any, args: tuple) -> None:
            if not args:
                raise RuntimeError("the MLP was called without a positional input")
            on_input("mlp:input", args[0])

        def hook_z(module: Any, args: tuple, output: Any) -> None:
            on_input("attn.hook_z", output.flatten(-2))

        def hook_post(module: Any, args: tuple, output: Any) -> None:
            on_input("mlp.hook_post", output)

        handles.append(block.mlp.register_forward_pre_hook(mlp_pre))
        handles.append(block.attn.hook_z.register_forward_hook(hook_z))
        handles.append(block.mlp.hook_post.register_forward_hook(hook_post))

    outputs: list[Batch] = []
    try:
        with torch.no_grad():
            for resid, kwargs in batches:
                outputs.append((block(resid, **kwargs), kwargs))
    finally:
        for handle in handles:
            handle.remove()
    return outputs


# ------------------------------------------------------------------ disk cache of results

def calibration_tokens(resolved: Mapping[str, Any], vocab_size: int) -> tuple[np.ndarray, dict]:
    """The Q7 token cache named by the cell's config, verified against its sidecar."""
    return load_token_cache(cache_path(resolved), vocab_size=int(vocab_size))


def code_digest(method: str) -> str:
    """Hash of the code that computes ``method``'s tensors (this module + ``<method>.py``).

    Part of the cache identity, so a result computed by older code is never reused after
    a fix. Line endings are normalised: a Windows and a Linux checkout hash the same.
    """
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("layerwise.py", f"{method}.py"):
        digest.update(name.encode())
        digest.update((here / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:16]


def _identity(
    resolved: Mapping[str, Any],
    method: str,
    settings: Mapping[str, Any],
    fingerprint: str,
    *,
    model_dtype: str,
) -> dict:
    spec = calibration_spec(resolved)
    model_cfg = dict(resolved.get("model") or {})
    return {
        "method": method,
        "settings": dict(sorted(settings.items())),
        "model": str(model_cfg.get("hf_id") or model_cfg.get("name") or ""),
        "model_revision": str(model_cfg.get("hf_revision") or ""),
        # the weights the method reads and the dtype its result is written back in
        "model_dtype": model_dtype,
        "calibration_dtype": spec["dtype"],
        "token_cache_fingerprint": fingerprint,
        "n_tokens": spec["n_tokens"],
        "code": code_digest(method),
    }


def compressed_path(identity: Mapping[str, Any], resolved: Mapping[str, Any]) -> Path:
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    stem = (
        f"{identity['method']}_{identity['model'].replace('/', '_')}_"
        f"{identity['model_revision'][:8] or 'unpinned'}_{digest}"
    )
    return calibration_root(resolved) / "compressed" / f"{stem}.pt"


def load_or_compute(
    model: Any,
    resolved: Mapping[str, Any],
    *,
    method: str,
    settings: Mapping[str, Any],
    compute: Callable[[np.ndarray, int], dict[str, Any]],
) -> dict[str, Any]:
    """The compressed projection tensors for this cell, computed once and reused.

    Stage B measures ``weight_delta`` and Stage C applies the compressor in separate
    processes. GPTQ's propagation and AWQ's searches run bf16 forwards whose GPU results
    can differ run to run, so recomputing in each stage could hand Stage C a slightly
    different compression than the null was matched to (ARCHITECTURE.md §4). Both stages
    read this file instead. ``compute(tokens, n_seq)`` returns ``{name: tensor}``.
    """
    import torch

    tokens, meta = calibration_tokens(resolved, int(model.cfg.d_vocab))
    model_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    identity = _identity(resolved, method, settings, meta["fingerprint"], model_dtype=model_dtype)
    path = compressed_path(identity, resolved)

    def stored() -> dict[str, Any]:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if json.loads(data["meta"]) != identity:
            raise ValueError(f"{path} was computed for {data['meta']}, not {identity}")
        return dict(data["tensors"])

    if path.exists():
        return stored()
    # parallel cells of one model and method all want this file: one computes, the rest
    # wait and load it (and none of them writes over another's half-written file)
    with cache_lock(path):
        if path.exists():
            return stored()
        tensors = compute(tokens, sequences_needed(tokens, calibration_spec(resolved)["n_tokens"]))
        tmp = partial_path(path, ".pt")
        try:
            torch.save(
                {"meta": json.dumps(identity, sort_keys=True), "tensors": {k: v.cpu() for k, v in tensors.items()}},
                tmp,
            )
            os.replace(tmp, path)  # atomic: a crash never leaves a half-written file under the real name
        finally:
            tmp.unlink(missing_ok=True)
    return tensors


def injected_tokens(tokens: Any, vocab_size: int) -> tuple[np.ndarray, int]:
    """Validate a token array handed in directly (tests, the preflight probe)."""
    arr = validate_token_cache(np.asarray(tokens), vocab_size=int(vocab_size))
    return arr, arr.shape[0]


def _check_tensors(params: Mapping[str, Any], tensors: Mapping[str, Any]) -> None:
    unknown = sorted(set(tensors) - set(params))
    if unknown:
        raise ValueError(f"compressed tensors name parameters the model lacks: {unknown}")
    for name, value in tensors.items():
        if tuple(value.shape) != tuple(params[name].shape):
            raise ValueError(f"{name}: compressed shape {tuple(value.shape)} != {tuple(params[name].shape)}")


def with_tensors(model: Any, tensors: Mapping[str, Any]) -> Any:
    """A deep copy of ``model`` with the named parameters replaced (dense model untouched)."""
    import torch

    out = copy.deepcopy(model)
    params = dict(out.named_parameters())
    _check_tensors(params, tensors)
    with torch.no_grad():
        for name, value in tensors.items():
            params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))
    return out


def tensors_delta(model: Any, tensors: Mapping[str, Any]) -> dict[str, float]:
    """:func:`measured_delta` of ``with_tensors(model, tensors)``, without copying the model.

    Each replacement is cast to its parameter's dtype first, exactly as :func:`with_tensors`
    writes it, so the two agree to the bit; untouched parameters report 0.
    """
    import torch

    params = dict(model.named_parameters())
    _check_tensors(params, tensors)
    out: dict[str, float] = {}
    with torch.no_grad():
        for name, param in params.items():
            if name not in tensors:
                out[name] = 0.0
                continue
            written = tensors[name].to(device=param.device, dtype=param.dtype)
            delta = written.to(torch.float32) - param.detach().to(torch.float32)
            out[name] = float(torch.linalg.vector_norm(delta).item())
    return out


def measured_delta(model: Any, compressed: Any) -> dict[str, float]:
    """Frobenius norm of ``W_compressed - W_dense`` for every parameter, in float32."""
    import torch

    after = dict(compressed.named_parameters())
    out: dict[str, float] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            delta = after[name].detach().to(torch.float32) - param.detach().to(torch.float32)
            out[name] = float(torch.linalg.vector_norm(delta).item())
    return out


__all__ = [
    "ATTN_INPUTS",
    "as_matrix",
    "block_source",
    "calibration_tokens",
    "cast_copy",
    "code_digest",
    "compressed_path",
    "embed_batches",
    "exact_float32_matmul",
    "from_matrix",
    "injected_tokens",
    "load_or_compute",
    "measured_delta",
    "projection_kind",
    "projections_by_layer",
    "run_block",
    "sequences_needed",
    "tensors_delta",
    "with_tensors",
]
