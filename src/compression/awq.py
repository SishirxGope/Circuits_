# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft AWQ wrapper (compression grid cell)
# modified: [AI-GEN] agent=Claude date=2026-09-29 task=B2 real-model AWQ in torch (deploy/BLOCKERS.md)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY (mock path); real path REVIEW PENDING
#
# Adapted from: https://github.com/mit-han-lab/llm-awq @ d6e797a42b9ef7778de8ee2352116e0f48a78d61, MIT
#   - awq/quantize/quantizer.py::pseudo_quantize_tensor (zero_point=True) -> awq_pseudo_quantize
#   - awq/quantize/auto_scale.py::auto_scale_block / _search_module_scale / apply_scale
#   - awq/quantize/auto_clip.py::auto_clip_layer / auto_clip_block / apply_clip
#   - awq/quantize/pre_quant.py::run_awq (the layer loop)
#   - Gemma-2's scaling groups are not in llm-awq; they follow AutoAWQ
#     (github.com/casper-hansen/AutoAWQ awq/models/gemma2.py, MIT), which uses the same four
#     groups as llm-awq's Llama. Licence text: THIRD_PARTY_LICENSES/llm-awq-LICENSE.txt

"""AWQ quantization (compression grid, proposal §3.2).

Technique reference: Lin et al. 2023, "AWQ: Activation-aware Weight Quantization for LLM
Compression and Acceleration" (arXiv:2306.00978).

**Why a port and not the package.** As for GPTQ: ``autoawq`` does not build on the
Spark's aarch64 and acts on ``nn.Linear`` modules TransformerLens does not have.

**Settings are the authors' published ones.** The grid pre-registers only ``bits: 4``.
The rest is llm-awq at the pinned commit: zero-point (asymmetric) quantization in groups
of 128 input channels (``--q_group_size 128``, the setting of every command in its README
and of the paper's INT4 results), a 20-point search over the scale exponent, and a
10-step clipping search on 512 sampled tokens, clipping every projection except Q and K.

**No weights outside the projections change.** llm-awq folds each scale into the
preceding op (a norm, ``v_proj``, ``up_proj`` or the activation) and quantizes
``W * s``. With simulated quantization that is the same function as keeping the
preceding op and storing ``W_eff = Q(clip(diag(1/s_out) W diag(s_in))) scaled back``,
which is what this port writes. So ``weight_delta`` covers exactly the projection
matrices, as for every other family, and the null is matched to the same tensor set.

**Faithful to the reference, including where it is architecture-specific:**

- Pythia (GPT-NeoX): groups ln1 -> QKV (loss: attention output), ln2 -> W_in (loss: MLP
  output), activation -> W_out. No W_O group. Clipping skips the *fused* query_key_value,
  so V is not clipped either - a side effect of fusion in the reference, kept.
- Llama / Gemma-2: ln1 -> QKV, V -> O only when ``v_proj`` and ``o_proj`` have the same
  shape (never under GQA, so never for Llama-3.2-1B or Gemma-2-2B), ln2 -> gate+up (loss:
  MLP output), up -> down. Clipping skips Q and K.

Every scale is searched against the dense layer (the reference restores the weights
after each trial and applies all scales afterwards), and layer l+1 is fed the dense
output of layer l. The calibration tokens are the Q7 cache, not pile-val (Q7 fixed one
corpus for Wanda, GPTQ and AWQ). One guard differs: when a layer sees fewer than 512
tokens the clipping sample uses them all, where the reference would slice with step 0
and crash.
"""

from __future__ import annotations

import copy
from typing import Any

from ..interfaces import CompressedModel, Model, PerTensorFrobenius
from ..synthetic.mock_model import MockModel
from .gptq import calibrated_tensors
from .layerwise import (
    as_matrix,
    block_source,
    cast_copy,
    embed_batches,
    from_matrix,
    measured_delta,
    projections_by_layer,
    run_block,
    with_tensors,
)
from .torch_weights import is_torch_model

TECHNIQUE_CITE = "AWQ: Lin et al. 2023 (arXiv:2306.00978); mit-han-lab/llm-awq @ d6e797a4"

# original_architecture -> the reference's per-architecture rules.
#   v_to_o:     search a V -> O scale when v_proj and o_proj have the same shape
#   clip_skip:  projections auto_clip_block leaves unclipped (by name: q_, k_, query, key)
ARCH_RULES: dict[str, dict[str, Any]] = {
    "GPTNeoXForCausalLM": {"v_to_o": False, "clip_skip": ("W_Q", "W_K", "W_V")},
    "LlamaForCausalLM": {"v_to_o": True, "clip_skip": ("W_Q", "W_K")},
    "Gemma2ForCausalLM": {"v_to_o": True, "clip_skip": ("W_Q", "W_K")},
}


def arch_rules(cfg: Any) -> dict[str, Any]:
    arch = getattr(cfg, "original_architecture", None)
    if arch not in ARCH_RULES:
        raise ValueError(
            f"AWQ: no scaling rules for architecture {arch!r} (have {sorted(ARCH_RULES)}); "
            "the reference raises for unsupported models too"
        )
    rules = dict(ARCH_RULES[arch])
    gated = bool(getattr(cfg, "gated_mlp", False))
    if gated == (arch == "GPTNeoXForCausalLM"):
        raise ValueError(f"AWQ: {arch} with gated_mlp={gated} does not match the reference's layout")
    return rules


def awq_pseudo_quantize(w: Any, bits: int, group_size: int) -> Any:
    """quantizer.py::pseudo_quantize_tensor with zero_point=True on ``[rows, columns]``.

    Unlike GPTQ's grid, the range is the group's own [min, max] (0 need not be inside).
    """
    import torch

    columns = w.shape[-1]
    g = columns if group_size <= 0 else int(group_size)
    if columns % g:
        raise ValueError(f"AWQ: {columns} input channels are not divisible by group size {g}")
    x = w.reshape(-1, g)
    max_val = x.amax(dim=1, keepdim=True)
    min_val = x.amin(dim=1, keepdim=True)
    max_int = 2**bits - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    q = (torch.clamp(torch.round(x / scales) + zeros, 0, max_int) - zeros) * scales
    return q.reshape(w.shape)


def _normalised_scales(x_mean_abs: Any, ratio: float) -> Any:
    """auto_scale.py: ``s = x_max^ratio`` clamped at 1e-4, divided by sqrt(max * min)."""
    s = x_mean_abs.pow(ratio).clamp(min=1e-4).view(-1)
    return s / (s.max() * s.min()).sqrt()


def awq_clip_max(
    w: Any,
    x: Any,
    *,
    bits: int,
    group_size: int,
    n_grid: int = 20,
    max_shrink: float = 0.5,
    n_sample_token: int = 512,
    row_chunk: int = 64,
) -> Any:
    """auto_clip.py::auto_clip_layer. ``w`` [out, in], ``x`` [tokens, in] (already scaled).

    Returns the per-(row, group) clip bound, shape ``[out, n_group, 1]``. The reference
    batches rows by 256 or 64 to save memory; batching does not change the result.
    """
    import torch

    rows, columns = w.shape
    g = columns if group_size <= 0 else int(group_size)
    feats = x.reshape(-1, columns).to(torch.float32)
    feats = feats.reshape(1, feats.shape[0], -1, g)
    step = max(1, feats.shape[1] // n_sample_token)
    feats = feats[:, 0::step]
    grouped = w.to(torch.float32).reshape(rows, 1, -1, g)

    best_all = []
    for r0 in range(0, rows, row_chunk):
        wb = grouped[r0:r0 + row_chunk]
        org_max = wb.abs().amax(dim=-1, keepdim=True)  # [c, 1, n_group, 1]
        best = org_max.clone()
        min_errs = torch.full_like(org_max, float("inf"))
        org_out = (feats * wb).sum(dim=-1)  # [c, n_token, n_group]
        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max * (1 - i_s / n_grid)
            cur = torch.clamp(wb, -max_val, max_val)
            q = awq_pseudo_quantize(cur.reshape(cur.shape[0], -1), bits, g).reshape(cur.shape)
            cur_out = (feats * q).sum(dim=-1)
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            better = err < min_errs
            min_errs[better] = err[better]
            best[better] = max_val[better]
        best_all.append(best)
    return torch.cat(best_all, dim=0).squeeze(1)


class _LayerFeatures:
    """Every projection input of one dense layer, kept per batch in the calibration dtype."""

    def __init__(self) -> None:
        self.batches: dict[str, list[Any]] = {}
        self.abs_sum: dict[str, Any] = {}
        self.count: dict[str, int] = {}

    def __call__(self, source: str, x: Any) -> None:
        import torch

        self.batches.setdefault(source, []).append(x.detach())
        flat = x.detach().reshape(-1, x.shape[-1]).to(torch.float32)
        total = flat.abs().sum(dim=0)
        self.abs_sum[source] = total if source not in self.abs_sum else self.abs_sum[source] + total
        self.count[source] = self.count.get(source, 0) + int(flat.shape[0])

    def mean_abs(self, source: str) -> Any:
        """auto_scale.py::get_act_scale - mean |x| per input channel."""
        return self.abs_sum[source] / self.count[source]

    def sample(self, source: str, n_sample_token: int) -> Any:
        """auto_clip_layer's token sample - every ``(T // n)``-th of all T tokens.

        Taken batch by batch, so the full activation (5.5 GB of Gemma-2 hook_post in
        bf16) is never concatenated or upcast just to keep ~512 rows. The rows are the
        ones the reference keeps; awq_clip_max's own sampling then keeps all of them.
        """
        import torch

        step = max(1, self.count[source] // n_sample_token)
        rows, offset = [], 0
        for x in self.batches[source]:
            flat = x.reshape(-1, x.shape[-1])
            rows.append(flat[(-offset) % step::step])
            offset += int(flat.shape[0])
        return torch.cat(rows, dim=0)


def _awq_layer(
    cfg: Any,
    index: int,
    block: Any,
    kinds: dict[str, str],
    batches: list,
    rules: dict[str, Any],
    *,
    bits: int,
    group_size: int,
    n_grid: int,
    clip: bool,
    clip_n_grid: int,
    clip_max_shrink: float,
    clip_n_sample_token: int,
    dtype: Any,
    record: dict[int, dict[str, Any]] | None,
) -> tuple[dict[str, Any], list]:
    """One layer of run_awq: search scales, clip, quantize. Returns (tensors, next inputs)."""
    import torch

    out: dict[str, Any] = {}
    dense = cast_copy(block, dtype)
    if dense is block:
        dense = copy.deepcopy(block)  # trial weights are written into it; never the model's
    run_dtype = next(dense.parameters()).dtype
    feats = _LayerFeatures()
    layout: dict[str, Any] = {}
    next_batches = run_block(dense, batches, on_input=feats, layout=layout)

    params = dict(block.named_parameters())
    trial = dict(dense.named_parameters())
    mats = {kind: as_matrix(kind, params[local].detach().to(torch.float32)) for kind, local in kinds.items()}
    s_in: dict[str, Any] = {}
    s_out: dict[str, Any] = {}

    def set_trial(assignments: dict[str, Any]) -> None:
        with torch.no_grad():
            for kind, matrix in assignments.items():
                local = kinds[kind]
                trial[local].copy_(from_matrix(kind, matrix, tuple(trial[local].shape)).to(run_dtype))

    def attn_replay(x: Any, kwargs: dict[str, Any]) -> Any:
        if kwargs.get("shortformer_pos_embed") is not None:
            raise ValueError("AWQ attention replay does not support shortformer embeddings")

        def expand(key: str) -> Any:
            heads = layout[key]
            return x if heads is None else x.unsqueeze(-2).expand(*x.shape[:-1], heads, x.shape[-1])

        return dense.attn(
            query_input=expand("query_input"),
            key_input=expand("key_input"),
            value_input=expand("value_input"),
            attention_mask=kwargs.get("attention_mask"),
        )

    def search(group: list[str], source: str, forward: Any) -> Any:
        """auto_scale.py::_search_module_scale for the tensors in ``group``."""
        x_mean_abs = feats.mean_abs(source)
        xs = feats.batches[source]
        kwargs = [kw for _, kw in batches]
        with torch.no_grad():
            org = [forward(x, kw) for x, kw in zip(xs, kwargs, strict=True)]
            best_error, best_scales, history = float("inf"), None, []
            for step in range(n_grid):
                ratio = step / n_grid
                scales = _normalised_scales(x_mean_abs, ratio)
                set_trial({
                    kind: awq_pseudo_quantize(mats[kind] * scales.view(1, -1), bits, group_size)
                    / scales.view(1, -1)
                    for kind in group
                })
                total, numel = 0.0, 0
                for x, kw, reference in zip(xs, kwargs, org, strict=True):
                    diff = (reference - forward(x, kw)).float()
                    total += float(diff.pow(2).sum().item())
                    numel += diff.numel()
                loss = total / numel
                history.append(loss)
                if loss < best_error:
                    best_error, best_scales = loss, scales
            set_trial({kind: mats[kind] for kind in group})
        if best_scales is None:
            raise RuntimeError(f"AWQ layer {index}: no finite loss in the scale search {history}")
        return best_scales.detach()

    def linear(kind: str) -> Any:
        local = kinds[kind]
        return lambda x, kw: x.matmul(as_matrix(kind, trial[local]).t())

    # --- auto_scale_block: every search against the dense layer ------------------
    qkv = [k for k in ("W_Q", "W_K", "W_V") if k in kinds]
    s_qkv = search(qkv, block_source("W_Q"), attn_replay)
    for kind in qkv:
        s_in[kind] = s_qkv

    n_heads = int(cfg.n_heads)
    n_kv = int(getattr(cfg, "n_key_value_heads", None) or n_heads)
    d_model, d_head = int(cfg.d_model), int(cfg.d_head)
    # llm-awq: `v_proj.weight.shape == o_proj.weight.shape` ([n_kv*dh, d] vs [d, n*dh])
    if rules["v_to_o"] and n_kv * d_head == d_model and n_heads * d_head == d_model:
        s_o = search(["W_O"], block_source("W_O"), linear("W_O"))
        s_in["W_O"] = s_o
        s_out["W_V"] = s_o

    mlp_in = [k for k in ("W_gate", "W_in") if k in kinds]
    s_mlp = search(mlp_in, block_source("W_in"), lambda x, kw: dense.mlp(x))
    for kind in mlp_in:
        s_in[kind] = s_mlp

    s_down = search(["W_out"], block_source("W_out"), linear("W_out"))
    s_in["W_out"] = s_down
    if getattr(cfg, "gated_mlp", False):
        s_out["W_in"] = s_down  # prev_op = up_proj: its output rows absorb 1/s
    # GPT-NeoX: prev_op = the activation (ScaledActivation) - no weight absorbs it

    if record is not None:
        record[index] = {"qkv": s_qkv, "mlp": s_mlp, "down": s_down, "clip": {}}
        if "W_O" in s_in:
            record[index]["o"] = s_in["W_O"]

    # --- apply_scale, auto_clip_block, pseudo-quantize -------------------------
    with torch.no_grad():
        for kind, local in kinds.items():
            matrix = mats[kind]
            if kind in s_out:
                matrix = matrix / s_out[kind].view(-1, 1)
            if kind in s_in:
                matrix = matrix * s_in[kind].view(1, -1)
            if clip and kind not in rules["clip_skip"]:
                x = feats.sample(block_source(kind), clip_n_sample_token).to(torch.float32)
                if kind in s_in:
                    x = x / s_in[kind].view(1, -1)
                bound = awq_clip_max(
                    matrix, x, bits=bits, group_size=group_size, n_grid=clip_n_grid,
                    max_shrink=clip_max_shrink, n_sample_token=clip_n_sample_token,
                )
                if record is not None:
                    record[index]["clip"][kind] = bound
                rows, columns = matrix.shape
                grouped = matrix.reshape(rows, bound.shape[1], -1)
                matrix = torch.clamp(grouped, -bound, bound).reshape(rows, columns)
            quantized = awq_pseudo_quantize(matrix, bits, group_size)
            if kind in s_in:
                quantized = quantized / s_in[kind].view(1, -1)
            if kind in s_out:
                quantized = quantized * s_out[kind].view(-1, 1)
            param = params[local]
            out[f"blocks.{index}.{local}"] = (
                from_matrix(kind, quantized, tuple(param.shape)).to(param.dtype).detach().clone()
            )
    return out, next_batches  # run_awq: the next layer sees the DENSE output


def awq_quantize_model(
    model: Any,
    tokens: Any,
    *,
    n_seq: int,
    bits: int,
    group_size: int,
    n_grid: int,
    clip: bool,
    clip_n_grid: int,
    clip_max_shrink: float,
    clip_n_sample_token: int,
    batch_size: int,
    dtype: Any,
    record: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """pre_quant.py::run_awq, then pseudo-quantization; returns ``{param name: quantized}``.

    ``record``, if given, receives what llm-awq returns as ``awq_results``: per layer, the
    chosen scale of each group (``qkv``, ``o``, ``mlp``, ``down``) and, under ``clip``,
    the clip bound of each clipped projection.
    """
    rules = arch_rules(model.cfg)
    layers = projections_by_layer(model)
    batches = embed_batches(model, tokens, n_seq=n_seq, batch_size=batch_size, dtype=dtype)
    out: dict[str, Any] = {}
    for index, block in enumerate(model.blocks):
        kinds = layers[index]
        if not kinds:
            continue
        quantized, batches = _awq_layer(
            model.cfg, index, block, kinds, batches, rules, bits=bits, group_size=group_size,
            n_grid=n_grid, clip=clip, clip_n_grid=clip_n_grid, clip_max_shrink=clip_max_shrink,
            clip_n_sample_token=clip_n_sample_token, dtype=dtype, record=record,
        )
        out.update(quantized)
    return out


class AwqCompressor:
    """AWQ compressor (Compressor protocol).

    ``calibration`` injects a token array directly (tests, the preflight probe). Left as
    None, a real model calibrates on the Q7 cache named by the cell's config, and the
    result is cached on disk so Stage B and Stage C share one compression.
    """

    def __init__(
        self,
        bits: int = 4,
        calibration: Any = None,
        *,
        group_size: int = 128,
        n_grid: int = 20,
        clip: bool = True,
        clip_n_grid: int = 20,
        clip_max_shrink: float = 0.5,
        clip_n_sample_token: int = 512,
        batch_size: int = 4,
    ) -> None:
        self.bits = int(bits)
        self.calibration = calibration
        self.group_size = int(group_size)
        self.n_grid = int(n_grid)
        self.clip = bool(clip)
        self.clip_n_grid = int(clip_n_grid)
        self.clip_max_shrink = float(clip_max_shrink)
        self.clip_n_sample_token = int(clip_n_sample_token)
        self.batch_size = int(batch_size)

    def settings(self) -> dict[str, Any]:
        return {
            "bits": self.bits, "group_size": self.group_size, "zero_point": True,
            "n_grid": self.n_grid, "clip": self.clip, "clip_n_grid": self.clip_n_grid,
            "clip_max_shrink": self.clip_max_shrink,
            "clip_n_sample_token": self.clip_n_sample_token,
            "batch_size": self.batch_size, "reference": "mit-han-lab/llm-awq@d6e797a4",
        }

    def _quantized(self, model: Any, cfg: Any) -> dict[str, Any]:
        def compute(tokens: Any, n_seq: int, dtype: Any) -> dict[str, Any]:
            return awq_quantize_model(
                model, tokens, n_seq=n_seq, bits=self.bits, group_size=self.group_size,
                n_grid=self.n_grid, clip=self.clip, clip_n_grid=self.clip_n_grid,
                clip_max_shrink=self.clip_max_shrink,
                clip_n_sample_token=self.clip_n_sample_token,
                batch_size=self.batch_size, dtype=dtype,
            )

        return calibrated_tensors(self, model, cfg, "awq", compute)

    def apply(self, model: Model, cfg: Any) -> CompressedModel:
        if isinstance(model, MockModel):
            # Dry-run stand-in ONLY (provisional): quantize per-tensor with the
            # RTN helper so the compression grid surface is exercisable. This is
            # explicitly NOT AWQ and produces no scientific evidence.
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
            f"{TECHNIQUE_CITE}: AwqCompressor supports MockModel and TransformerLens "
            f"models; got {type(model).__name__}."
        )

    def weight_delta(self, model: Model, cfg: Any) -> PerTensorFrobenius:
        if isinstance(model, MockModel):
            return model.weight_delta_frobenius(self.apply(model, cfg))
        if is_torch_model(model):
            return measured_delta(model, self.apply(model, cfg))
        raise NotImplementedError(
            "awq weight_delta supports MockModel and TransformerLens models; "
            f"got {type(model).__name__}."
        )


__all__ = [
    "ARCH_RULES",
    "AwqCompressor",
    "arch_rules",
    "awq_clip_max",
    "awq_pseudo_quantize",
    "awq_quantize_model",
]
