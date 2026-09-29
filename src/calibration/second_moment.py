# [AI-GEN] agent=Claude date=2026-09-29 task=Q7 per-projection input second moments (Wanda score input)
# reviewed-by: PENDING
#
# Adapted from: https://github.com/hecboar/sae-pruning-paper @ 261191804675e2d39d0a265320dbc0bc85afd30a, MIT
#   - revision/src/saediag/pruning.py::collect_linear_input_second_moment_from_cache
#   - revision/src/saediag/reprune.py::build_pruned_model (bf16 forward, non-finite guard)
#   - license: THIRD_PARTY_LICENSES/sae-pruning-paper-LICENSE.txt

"""E[x^2] over the input of every projection matrix, from a fixed calibration cache.

**Where each input is read.** Upstream hooks the input of every ``nn.Linear``.
TransformerLens has no Linear modules for attention - its projections are parameters
used in einsums - so each input is read at the nearest point that carries exactly the
tensor the projection consumes:

    W_Q / W_K / W_V  <- attention module's query_input / key_input / value_input
    W_in / W_gate    <- MLP module's input
    W_O              <- blocks.{l}.attn.hook_z        (keeps [n_heads, d_head])
    W_out            <- blocks.{l}.mlp.hook_post

The first two are PyTorch forward pre-hooks on the attention and MLP modules, **not**
TransformerLens's ``ln1/ln2.hook_normalized``. That hook fires on ``x / scale`` *before*
the norm's affine ``* w + b`` (components/layer_norm.py, rms_norm.py), so it equals the
projection input only when w = 1 and b = 0. Our models load with ``fold_ln=False``, so
their norms are live and ``hook_normalized`` would measure the wrong activation on every
real model - with no error. A fixture with randomised norm weights caught this; the
default-initialised one could not. The module inputs are also taken after the cast to
the model dtype, and they follow the block's own wiring (parallel attention, Gemma-2
post-norms, OLMo-style post-norm MLPs) without a case for each.

Each statistic is the mean of x^2 over every leading axis. With the extraction flags on
(``enable_extraction_hooks``) the attention inputs carry a head axis of identical copies,
which cancel in the mean; ``hook_z`` keeps its head axis because W_O's input genuinely is
(head, d_head). tests/test_second_moment.py re-derives every input independently and
checks it against the model's own downstream activations.

**Upstream fidelity.** Same statistic (``sum(x^2)/count`` per input feature), same token
budget rule (``ceil(n_tokens / ctx)`` sequences, capped at the cache), same batching
slice, same non-finite refusal, and the forward runs in the pre-registered calibration
dtype (bf16; upstream: "Gemma activations overflow fp16"). Two deliberate differences,
neither of which moves a mask beyond upstream's own "cross-stack activation numerics"
tolerance: accumulation is float64 rather than float32, and a projection with no
statistic is an error - upstream silently falls back to ``|W|`` (``scale = ones``),
which would quietly turn a Wanda cell into a magnitude cell.

**Why statistics are cached on disk.** Stage B measures Wanda's weight delta and Stage C
applies Wanda, in separate processes. Recomputing E[x^2] in each lets GPU nondeterminism
flip mask entries at the threshold, so the null would be matched to a slightly different
compression than the one Stage C applies - the drift ARCHITECTURE.md 4 forbids. Both
stages read one file instead.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .cache_lock import cache_lock, partial_path
from .token_cache import (
    cache_path,
    calibration_root,
    calibration_spec,
    load_token_cache,
    validate_token_cache,
)

# projection suffix -> (input source template, trailing axes the statistic keeps).
# "module:argument" is a forward pre-hook on that module; anything else is a
# TransformerLens hook point.
INPUT_SOURCES: dict[str, tuple[str, int]] = {
    "W_Q": ("blocks.{layer}.attn:query_input", 1),
    "W_K": ("blocks.{layer}.attn:key_input", 1),
    "W_V": ("blocks.{layer}.attn:value_input", 1),
    "W_O": ("blocks.{layer}.attn.hook_z", 2),
    "W_in": ("blocks.{layer}.mlp:input", 1),
    "W_gate": ("blocks.{layer}.mlp:input", 1),
    "W_out": ("blocks.{layer}.mlp.hook_post", 1),
}


def input_source_for(param_name: str) -> tuple[str, int]:
    """``(source, trailing axes kept)`` for a projection parameter.

    Accepts the compact GQA names (``_W_K``/``_W_V``): their input is the same.
    """
    parts = param_name.split(".")
    suffix = parts[-1].lstrip("_")
    if len(parts) < 4 or parts[0] != "blocks" or not parts[1].isdigit() or suffix not in INPUT_SOURCES:
        raise ValueError(
            f"{param_name!r} is not a TransformerLens projection parameter "
            f"(blocks.<layer>.<attn|mlp>.<{'|'.join(INPUT_SOURCES)}>)"
        )
    template, keep = INPUT_SOURCES[suffix]
    return template.format(layer=int(parts[1])), keep


def statistic_shape(param_shape: tuple[int, ...], keep: int) -> tuple[int, ...]:
    """The parameter's input axes: the ``keep`` axes ending just before the output axis."""
    return tuple(param_shape[-1 - keep:-1])


def broadcast_shape(param_shape: tuple[int, ...], keep: int) -> tuple[int, ...]:
    """The statistic reshaped to broadcast against the parameter (output axes set to 1).

    W_Q [H, D, dh] -> (1, D, 1);  W_O [H, dh, D] -> (H, dh, 1);  W_in [D, M] -> (D, 1).
    """
    return (1,) * (len(param_shape) - 1 - keep) + statistic_shape(param_shape, keep) + (1,)


def collect_second_moments(
    model: Any,
    tokens: np.ndarray,
    *,
    n_tokens: int | None = None,
    batch_size: int = 2,
    dtype: Any = None,
    param_names: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """E[x^2] of each projection's input, shaped to broadcast against the parameter.

    ``dtype`` runs the forward on a cast copy (the model itself is untouched), as upstream
    loads the model in bf16 for calibration. ``n_tokens=None`` uses the whole cache.
    Returns float64 arrays keyed by ``named_parameters()`` names.
    """
    import torch

    from ..compression.torch_weights import projection_parameters

    names = list(param_names) if param_names is not None else projection_parameters(model)
    if not names:
        raise ValueError("no projection parameters to collect statistics for")
    params = dict(model.named_parameters())
    plan: dict[str, tuple[str, int]] = {}
    keep_by_source: dict[str, int] = {}
    for name in names:
        source, keep = input_source_for(name)
        if keep_by_source.setdefault(source, keep) != keep:
            raise ValueError(f"{source} is read with two different reductions")
        plan[name] = (source, keep)

    arr = validate_token_cache(tokens, vocab_size=int(model.cfg.d_vocab))
    n_seq, ctx = arr.shape
    # upstream: seq_needed = min(ceil(n_tokens / ctx), n_seq)
    seq_needed = n_seq if n_tokens is None else min(math.ceil(n_tokens / ctx), n_seq)

    run_model = model
    if dtype is not None:
        target = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if next(model.parameters()).dtype != target:
            run_model = copy.deepcopy(model).to(target)
    device = next(run_model.parameters()).device

    sums: dict[str, Any] = {}
    counts: dict[str, int] = {}

    def record(source: str, activation: Any) -> None:
        keep = keep_by_source[source]
        x = activation.detach().to(torch.float64)
        leading = tuple(range(x.ndim - keep))
        squared = (x * x).sum(dim=leading).cpu()
        sums[source] = squared if source not in sums else sums[source] + squared
        counts[source] = counts.get(source, 0) + int(np.prod(x.shape[: x.ndim - keep]))

    def tl_hook(activation: Any, hook: Any) -> None:
        record(hook.name, activation)

    def pre_hook_for(module_path: str, arguments: list[str]):
        def pre_hook(module: Any, args: tuple, kwargs: dict) -> None:
            for argument in arguments:
                if argument == "input":
                    if not args:
                        raise RuntimeError(f"{module_path} was called without a positional input")
                    value = args[0]
                else:
                    if argument not in kwargs:
                        raise RuntimeError(
                            f"{module_path} was called without the keyword {argument!r}; "
                            "TransformerLens changed how blocks call attention"
                        )
                    value = kwargs[argument]
                record(f"{module_path}:{argument}", value)

        return pre_hook

    by_module: dict[str, list[str]] = {}
    for source in keep_by_source:
        if ":" in source:
            module_path, argument = source.split(":", 1)
            by_module.setdefault(module_path, []).append(argument)
    fwd_hooks = [(source, tl_hook) for source in keep_by_source if ":" not in source]

    run_model.eval()
    handles = [
        run_model.get_submodule(path).register_forward_pre_hook(pre_hook_for(path, args), with_kwargs=True)
        for path, args in by_module.items()
    ]
    try:
        with torch.no_grad():
            # upstream's slice: token_cache[i:i + batch_size] for i in range(0, seq_needed, bs)
            for start in range(0, seq_needed, int(batch_size)):
                batch = torch.as_tensor(
                    arr[start:start + int(batch_size)], dtype=torch.long, device=device
                )
                run_model.run_with_hooks(
                    batch, stop_at_layer=run_model.cfg.n_layers, fwd_hooks=fwd_hooks
                )
    finally:
        for handle in handles:
            handle.remove()

    silent = sorted(source for source in keep_by_source if not counts.get(source))
    if silent:
        raise RuntimeError(
            f"input sources never fired: {silent}. This architecture feeds its projections "
            "differently; refusing rather than fall back to |W| for those tensors"
        )
    out: dict[str, np.ndarray] = {}
    for name, (source, keep) in plan.items():
        stat = (sums[source] / counts[source]).numpy()
        want = statistic_shape(tuple(params[name].shape), keep)
        if tuple(stat.shape) != want:
            raise RuntimeError(f"{name}: statistic from {source} has shape {stat.shape}, expected {want}")
        out[name] = stat.reshape(broadcast_shape(tuple(params[name].shape), keep))
    bad = sorted(name for name, stat in out.items() if not np.isfinite(stat).all())
    if bad:  # reprune.py: "refusing to build a corrupted mask"
        raise RuntimeError(
            f"non-finite E[x^2] for {len(bad)} tensors (first: {bad[0]}); the calibration "
            "forward overflowed. Refusing to build a corrupted mask."
        )
    return out


# ---------------------------------------------------------------------------- disk cache

def _model_identity(resolved: Mapping[str, Any]) -> tuple[str, str]:
    model_cfg = dict(resolved.get("model") or {})
    return (
        str(model_cfg.get("hf_id") or model_cfg.get("name") or ""),
        str(model_cfg.get("hf_revision") or ""),
    )


def second_moments_path(resolved: Mapping[str, Any], cache_fingerprint: str) -> Path:
    spec = calibration_spec(resolved)
    hf_id, revision = _model_identity(resolved)
    stem = (
        f"{hf_id.replace('/', '_')}_{revision[:8] or 'unpinned'}_{spec['dtype']}_"
        f"{cache_fingerprint[:16]}"
    )
    return calibration_root(resolved) / "second_moments" / f"{stem}.npz"


def load_or_collect_second_moments(
    model: Any, resolved: Mapping[str, Any], *, batch_size: int = 2
) -> dict[str, np.ndarray]:
    """The cell's E[x^2], computed once per (model, revision, dtype, cache) and reused.

    Raises :class:`~src.calibration.token_cache.CalibrationUnavailable` when the token
    cache has not been built on this machine.
    """
    spec = calibration_spec(resolved)
    tokens, cache_meta = load_token_cache(cache_path(resolved), vocab_size=int(model.cfg.d_vocab))
    path = second_moments_path(resolved, cache_meta["fingerprint"])
    hf_id, revision = _model_identity(resolved)
    identity = {
        "model": hf_id,
        "model_revision": revision,
        "calibration_dtype": spec["dtype"],
        "token_cache_fingerprint": cache_meta["fingerprint"],
        "n_tokens": spec["n_tokens"],
    }
    def stored() -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"]))
            if meta != identity:
                raise ValueError(f"{path} was collected for {meta}, not {identity}")
            return {key: data[key] for key in data.files if key != "__meta__"}

    if path.exists():
        return stored()
    # parallel cells of one model all want this file: one collects, the rest wait and load
    with cache_lock(path):
        if path.exists():
            return stored()
        stats = collect_second_moments(
            model, tokens, n_tokens=spec["n_tokens"], batch_size=batch_size, dtype=spec["dtype"]
        )
        tmp = partial_path(path, ".npz")
        try:
            np.savez(tmp, __meta__=np.array(json.dumps(identity, sort_keys=True)), **stats)
            os.replace(tmp, path)  # atomic: a crash never leaves a half-written file under the real name
        finally:
            tmp.unlink(missing_ok=True)
    return stats


__all__ = [
    "INPUT_SOURCES",
    "broadcast_shape",
    "collect_second_moments",
    "input_source_for",
    "load_or_collect_second_moments",
    "second_moments_path",
    "statistic_shape",
]
