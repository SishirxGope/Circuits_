# [AI-GEN] agent=Claude date=2026-09-14 task=Edge attribution patching for the dense-node pipeline (Stage A engineering)
# reviewed-by: PENDING
#
# Method: Syed, Rager & Conmy, "Attribution Patching Outperforms Automated Circuit
# Discovery", arXiv:2310.10348. Aggregation and hook placement follow the reference
# implementation the paper points to (github.com/Aaquib111/acdcpp, utils/prune_utils.py),
# re-implemented here against TransformerLens 3.9.0 rather than copied.

"""Edge attribution patching (EAP) over heads and MLPs, dictionary-free (claim C8).

Pre-registered choices, all PI decisions of 2026-09-14:

- **Method: EAP.** Linear attribution, like circuit-tracer's, so a divergence between the
  dense-node pipeline and pipeline A can be attributed to the BASIS rather than to a change
  of attribution method (C8).
- **Q/K/V split.** An upstream component sends separate edges into a head's query, key and
  value inputs, exactly as the reference implementation scores them. Head input node ids
  are ``L{l}.H{h}.Q`` / ``.K`` / ``.V``; head outputs are ``L{l}.H{h}``. The graph is
  component outputs -> component inputs only; the reference code's internal
  Q/K/V-input -> projection edges are not part of it.
- **Aggregation: |sum|.** For edge u->v the score is ``| sum over prompts and positions of
  dL/d(input_v) . (a_u^clean - a_u^corrupt) |`` - the reference code sums over the batch
  and THEN takes the absolute value. An edge whose effect changes sign across prompts
  cancels and scores low. Signed contributions are summed across ALL batches and the
  absolute value is taken ONCE at the end; taking it per batch would silently compute a
  different quantity.
- **Node scores** for ``node_threshold``: attribution patching at each component's OUTPUT,
  ``| sum dL/d(a_u) . (a_u^clean - a_u^corrupt) |`` - the reference code's node mode, which
  covers attention heads only; applying the identical formula to MLP outputs and the
  embedding is an extension, labelled as such.
- **Positions aggregated** (Q11): contributions are summed over every position.
- **Precision**: run in the model's dtype (float32 for Pythia); accumulated in float64.

The one structural rule that is NOT in the reference code, and matters for Pythia:
**parallel residual models have no same-layer attention -> MLP edge.** In a parallel block
both sublayers read ``resid_pre``, so layer ``l``'s attention output never reaches layer
``l``'s MLP input. EAP's gradient-at-the-endpoint trick would still assign that non-edge a
NONZERO score (the gradient at ``mlp_in`` and the attention output's delta are both
nonzero), so it must be excluded structurally, not by its score.

Why the metric is divided by the TOTAL prompt count before backward: ``metric_sum`` sums
over a batch, and the pre-registered quantity is the mean over the whole prompt set, so
each batch's gradient is of its share of that mean.

Validity caveat recorded for the paper: EAP's linear approximation is poorly calibrated
even where its ranking works (R^2 = 0.27 against activation patching on docstring,
arXiv:2310.10348 §5.1), and its dominant error comes from downstream non-linearity, growing
with perturbation size (arXiv:2606.09899). Compression changes that downstream network, so
EAP's error could differ between dense and compressed models; the matched-magnitude null
perturbs the model too and partly absorbs this, but not provably.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from ..tasks.base import TaskPrompts

EMB = "EMB"
LOGIT = "LOGIT"


@dataclass(frozen=True)
class _Hook:
    name: str
    layer: int          # EMB = -1, LOGIT = n_layers
    kind: str           # "emb" | "head" | "mlp" | "q" | "k" | "v" | "mlp_in" | "logit"


def upstream_hooks(cfg: Any) -> list[_Hook]:
    """Component OUTPUTS that send edges."""
    hooks = [_Hook("blocks.0.hook_resid_pre", -1, "emb")]
    for layer in range(cfg.n_layers):
        hooks.append(_Hook(f"blocks.{layer}.attn.hook_result", layer, "head"))
        hooks.append(_Hook(f"blocks.{layer}.hook_mlp_out", layer, "mlp"))
    return hooks


def downstream_hooks(cfg: Any) -> list[_Hook]:
    """Component INPUTS that receive edges (pre-LayerNorm, so an additive patch is exact)."""
    hooks = []
    for layer in range(cfg.n_layers):
        for x in ("q", "k", "v"):
            hooks.append(_Hook(f"blocks.{layer}.hook_{x}_input", layer, x))
        hooks.append(_Hook(f"blocks.{layer}.hook_mlp_in", layer, "mlp_in"))
    hooks.append(_Hook(f"blocks.{cfg.n_layers - 1}.hook_resid_post", cfg.n_layers, "logit"))
    return hooks


def edge_pair_exists(up: _Hook, down: _Hook, parallel_attn_mlp: bool) -> bool:
    """Whether upstream hook ``up`` can causally reach downstream hook ``down``."""
    if up.layer < down.layer:
        return True
    if up.layer == down.layer and up.kind == "head" and down.kind == "mlp_in":
        # Sequential block: attention writes resid_mid, which the MLP reads.
        # Parallel block (GPT-NeoX / Pythia): both read resid_pre - no path.
        return not parallel_attn_mlp
    return False


def _up_ids(hook: _Hook, n_heads: int) -> list[str]:
    if hook.kind == "emb":
        return [EMB]
    if hook.kind == "head":
        return [f"L{hook.layer}.H{h}" for h in range(n_heads)]
    return [f"L{hook.layer}.MLP"]


def _down_ids(hook: _Hook, n_heads: int) -> list[str]:
    if hook.kind in ("q", "k", "v"):
        return [f"L{hook.layer}.H{h}.{hook.kind.upper()}" for h in range(n_heads)]
    if hook.kind == "mlp_in":
        return [f"L{hook.layer}.MLP"]
    return [LOGIT]


def candidate_edges(cfg: Any) -> list[str]:
    """Every structurally possible edge id ``src->dst`` - the dense-node edge universe."""
    parallel = bool(cfg.parallel_attn_mlp)
    out = []
    for down in downstream_hooks(cfg):
        for up in upstream_hooks(cfg):
            if edge_pair_exists(up, down, parallel):
                for dst in _down_ids(down, cfg.n_heads):
                    for src in _up_ids(up, cfg.n_heads):
                        out.append(f"{src}->{dst}")
    return out


def _check_model(model: Any) -> None:
    cfg = model.cfg
    missing = [f for f in ("use_attn_result", "use_split_qkv_input", "use_hook_mlp_in") if not getattr(cfg, f, False)]
    if missing:
        raise ValueError(f"EAP needs these TransformerLens hooks enabled: {missing} (see real_model.load_pinned_model)")
    if getattr(cfg, "attn_only", False):
        raise NotImplementedError("attention-only models are not supported by this EAP implementation")
    kv = getattr(cfg, "n_key_value_heads", None)
    if kv is not None and int(kv) != int(cfg.n_heads):
        raise NotImplementedError(
            "grouped-query attention: key/value inputs have n_key_value_heads < n_heads, so K/V "
            "input node ids need a mapping that has not been specified. Not guessed here."
        )


@dataclass
class AttributionScores:
    """Non-negative EAP edge scores and node scores for one (model, task, seed)."""

    edge_scores: dict[str, float]
    node_scores: dict[str, float]
    n_prompts: int
    clean_metric_mean: float
    corrupt_metric_mean: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _einsum_spec(g_has_head: bool, d_has_head: bool) -> str:
    return {
        (True, True): "bphd,bpkd->hk",
        (True, False): "bphd,bpd->h",
        (False, True): "bpd,bpkd->k",
        (False, False): "bpd,bpd->",
    }[(g_has_head, d_has_head)]


def compute_eap(model: Any, prompts: TaskPrompts) -> AttributionScores:
    """Edge and node attribution scores for ``prompts`` on ``model`` (two forwards + one backward per batch)."""
    _check_model(model)
    cfg = model.cfg
    n_heads = int(cfg.n_heads)
    parallel = bool(cfg.parallel_attn_mlp)
    ups, downs = upstream_hooks(cfg), downstream_hooks(cfg)
    pairs = [(d, u) for d in downs for u in ups if edge_pair_exists(u, d, parallel)]
    n_total = prompts.n_prompts
    if n_total == 0:
        raise ValueError("no prompts: EAP needs at least one clean/corrupted pair")
    device = next(model.parameters()).device
    head_kinds = {"head", "q", "k", "v"}

    edge_acc: dict[tuple[str, str], torch.Tensor] = {}
    node_acc: dict[str, torch.Tensor] = {}
    clean_total = corrupt_total = 0.0

    saved_requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    try:
        for batch in prompts.batches:
            corrupt_cache: dict[str, torch.Tensor] = {}

            def cache_corrupt(act, hook):
                corrupt_cache[hook.name] = act.detach()

            with torch.no_grad():
                logits = model.run_with_hooks(
                    batch.corrupt.to(device), fwd_hooks=[(u.name, cache_corrupt) for u in ups]
                )
                corrupt_total += float(batch.metric_sum(logits))
                del logits

            clean: dict[str, torch.Tensor] = {}

            def keep(act, hook):
                if hook.name == "blocks.0.hook_resid_pre":
                    act = act.detach().requires_grad_(True)   # the leaf gradients flow from
                else:
                    act.retain_grad()
                clean[hook.name] = act
                return act

            logits = model.run_with_hooks(
                batch.clean.to(device),
                fwd_hooks=[(h.name, keep) for h in ups + downs],
            )
            metric = batch.metric_sum(logits)
            clean_total += float(metric.detach())
            (metric / n_total).backward()
            del logits, metric

            with torch.no_grad():
                deltas = {u.name: (clean[u.name].detach() - corrupt_cache[u.name]) for u in ups}
                for u in ups:
                    # Node score: gradient and delta at the SAME output, per head where there is one.
                    spec = "bphd,bphd->h" if u.kind in head_kinds else "bpd,bpd->"
                    contrib = torch.einsum(spec, clean[u.name].grad, deltas[u.name]).double()
                    node_acc[u.name] = node_acc.get(u.name, 0) + contrib
                for d, u in pairs:
                    contrib = torch.einsum(
                        _einsum_spec(d.kind in head_kinds, u.kind in head_kinds),
                        clean[d.name].grad, deltas[u.name],
                    ).double()
                    key = (d.name, u.name)
                    edge_acc[key] = edge_acc.get(key, 0) + contrib
            del clean, corrupt_cache, deltas
    finally:
        for p, flag in zip(model.parameters(), saved_requires_grad):
            p.requires_grad_(flag)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - t0

    hook_by_name = {h.name: h for h in ups + downs}
    edge_scores: dict[str, float] = {}
    for (d_name, u_name), acc in edge_acc.items():
        d, u = hook_by_name[d_name], hook_by_name[u_name]
        values = acc.abs().cpu()
        dst_ids, src_ids = _down_ids(d, n_heads), _up_ids(u, n_heads)
        grid = values.reshape(len(dst_ids), len(src_ids))
        for i, dst in enumerate(dst_ids):
            for j, src in enumerate(src_ids):
                edge_scores[f"{src}->{dst}"] = float(grid[i, j])

    node_scores: dict[str, float] = {}
    for u_name, acc in node_acc.items():
        u = hook_by_name[u_name]
        values = acc.abs().cpu().reshape(-1)
        for j, node in enumerate(_up_ids(u, n_heads)):
            node_scores[node] = float(values[j])

    return AttributionScores(
        edge_scores=edge_scores,
        node_scores=node_scores,
        n_prompts=n_total,
        clean_metric_mean=clean_total / n_total,
        corrupt_metric_mean=corrupt_total / n_total,
        metadata={
            "method": "edge attribution patching (arXiv:2310.10348)",
            "aggregation": "abs(sum over prompts and positions), taken once after all batches",
            "qkv": "split",
            "parallel_attn_mlp": parallel,
            "n_batches": len(prompts.batches),
            "n_candidate_edges": len(edge_scores),
            "dtype": str(cfg.dtype),
            "seconds": seconds,
            "task": prompts.task,
            "seed": prompts.seed,
            "task_metadata": dict(prompts.metadata),
        },
    )


__all__ = [
    "EMB",
    "LOGIT",
    "AttributionScores",
    "candidate_edges",
    "compute_eap",
    "downstream_hooks",
    "edge_pair_exists",
    "upstream_hooks",
]
