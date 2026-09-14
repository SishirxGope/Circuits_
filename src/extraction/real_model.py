# [AI-GEN] agent=Claude date=2026-09-14 task=Load a real model at its pinned HF revision into TransformerLens (Stage A engineering)
# reviewed-by: PENDING

"""Loading a real model exactly at its Q5-pinned revision.

Three failure modes this module exists to close, each verified against the installed
libraries on 2026-09-14 (transformers 5.17.0, transformer-lens 3.9.0):

1. **TransformerLens has no ``revision`` argument.** ``HookedTransformer.from_pretrained``'s
   ``checkpoint_value`` maps to Pythia TRAINING-STEP branches, not commit shas, so the
   obvious call silently loads ``main`` and ignores the pin. The weights are therefore
   loaded through ``transformers`` at the pinned sha and handed over as ``hf_model``.
2. **The architecture is still fetched by name.** Even with ``hf_model``, TransformerLens
   builds its config via ``convert_hf_model_config(model_name)``, which calls
   ``AutoConfig.from_pretrained`` by name with no revision; the pinned config only overrides
   a few fields. If the repo's ``main`` config ever changes, TransformerLens would build a
   different architecture around pinned weights. So the constructed config is asserted
   against BOTH the pinned HF config and the VERIFIED numbers in configs/model/*.yaml, and
   loading refuses on any mismatch.
3. **The download gate was prose, not code.** ``mode.allow_model_download`` was read by no
   Python; only stub ``raise``s stopped a download. It is enforced here.

No reparameterisation: ``fold_ln``, ``center_writing_weights`` and ``center_unembed`` are all
off, so the weights that Stage B perturbs and Stage C compresses are exactly the HF weights,
and hook activations are the model's own.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_DTYPES = {"float32": "float32", "bfloat16": "bfloat16", "float16": "float16"}
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _require_download_allowed(resolved: Mapping[str, Any]) -> None:
    mode = resolved.get("mode") or {}
    if mode.get("allow_model_download") is not True:
        raise PermissionError(
            f"mode {mode.get('name')!r} does not allow model downloads "
            "(mode.allow_model_download is not true). Real models are loaded only under an "
            "explicitly approved mode - RUN MODEL DOWNLOAD (docs/HUMAN_DECISIONS.md)."
        )


def _torch_dtype(model_cfg: Mapping[str, Any]):
    import torch

    name = str(model_cfg.get("dtype", ""))
    if name not in _DTYPES:
        raise ValueError(f"model {model_cfg.get('name')!r}: dtype {name!r} is not one of {sorted(_DTYPES)}")
    return getattr(torch, _DTYPES[name])


def architecture_mismatches(tl_cfg: Any, hf_config: Any, model_cfg: Mapping[str, Any]) -> list[str]:
    """Every disagreement between TransformerLens, the pinned HF config and our YAML."""
    n_heads = int(hf_config.num_attention_heads)
    hf = {
        "d_model": int(hf_config.hidden_size),
        "n_layers": int(hf_config.num_hidden_layers),
        "n_heads": n_heads,
        "head_dim": int(getattr(hf_config, "head_dim", None) or hf_config.hidden_size // n_heads),
        "context_length": int(hf_config.max_position_embeddings),
    }
    tl = {
        "d_model": int(tl_cfg.d_model),
        "n_layers": int(tl_cfg.n_layers),
        "n_heads": int(tl_cfg.n_heads),
        "head_dim": int(tl_cfg.d_head),
        "context_length": int(tl_cfg.n_ctx),
    }
    problems = []
    for key in hf:
        yaml_value = model_cfg.get(key)
        if tl[key] != hf[key]:
            problems.append(f"{key}: TransformerLens {tl[key]} != pinned HF config {hf[key]}")
        if yaml_value is not None and int(yaml_value) != hf[key]:
            problems.append(f"{key}: configs/model yaml {yaml_value} != pinned HF config {hf[key]}")
    if hasattr(hf_config, "intermediate_size") and int(tl_cfg.d_mlp) != int(hf_config.intermediate_size):
        problems.append(f"d_mlp: TransformerLens {tl_cfg.d_mlp} != pinned HF config {hf_config.intermediate_size}")
    if hasattr(hf_config, "use_parallel_residual") and bool(tl_cfg.parallel_attn_mlp) != bool(hf_config.use_parallel_residual):
        problems.append(
            f"parallel residual: TransformerLens {tl_cfg.parallel_attn_mlp} != pinned HF config "
            f"{hf_config.use_parallel_residual} - this changes which edges EXIST"
        )
    return problems


def load_pinned_model(resolved: Mapping[str, Any], device: str | None = None):
    """Load ``resolved['model']`` at its pinned revision as a hooked TransformerLens model.

    Enables the hooks edge attribution patching needs: per-head attention results, split
    Q/K/V inputs, and the MLP input hook. Returns the model in eval mode.
    """
    import torch
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _require_download_allowed(resolved)
    model_cfg = dict(resolved.get("model") or {})
    hf_id = str(model_cfg.get("hf_id") or "")
    revision = str(model_cfg.get("hf_revision") or "")
    if not hf_id:
        raise ValueError(f"model {model_cfg.get('name')!r} has no hf_id")
    if not _SHA_RE.match(revision):
        raise ValueError(
            f"model {model_cfg.get('name')!r}: hf_revision {revision!r} is not a 40-hex commit sha; "
            "a branch name or null would load whatever the repo holds today (Q5)"
        )
    dtype = _torch_dtype(model_cfg)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    hf_model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision, torch_dtype=dtype)
    got = getattr(hf_model.config, "_commit_hash", None)
    if got != revision:
        raise RuntimeError(
            f"{hf_id}: requested revision {revision} but the loaded config reports commit {got!r}; "
            "refusing rather than extracting from an unverified checkpoint"
        )
    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)

    model = HookedTransformer.from_pretrained(
        hf_id,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    problems = architecture_mismatches(model.cfg, hf_model.config, model_cfg)
    if problems:
        raise RuntimeError(
            f"{hf_id} @ {revision[:8]}: TransformerLens built a model that does not match the "
            "pinned checkpoint:\n  " + "\n  ".join(problems)
        )
    del hf_model

    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)
    model.eval()
    return model


__all__ = ["load_pinned_model", "architecture_mismatches"]
