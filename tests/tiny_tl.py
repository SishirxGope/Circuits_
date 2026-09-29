# [AI-GEN] agent=Claude date=2026-09-29 task=Tiny random TransformerLens models for calibration/Wanda tests
# reviewed-by: PENDING

"""Tiny random TransformerLens models, built from a config: no download, no GPU.

One per architecture family in the grid, so a hook-name or layout assumption that holds
for one family but not another fails here rather than on the Spark:

- ``pythia``: LayerNorm, GELU, parallel attention/MLP (GPT-NeoX)
- ``llama``:  RMSNorm, gated SiLU MLP, grouped-query attention
- ``gemma2``: RMSNorm, gated GELU MLP, grouped-query attention, post-norms

``extraction_hooks=True`` applies ``enable_extraction_hooks`` - the same call
``load_pinned_model`` makes - so tests see the model exactly as the pipeline does.

**Norm weights and biases are randomised.** TransformerLens initialises every norm to
w=1, b=0 and every bias to 0. In a parallel-attention block (Pythia) ln1 and ln2 then
normalise the same residual identically, so a test could not tell a projection that
reads ln1 from one that reads ln2 - a mutation check found exactly that blind spot.
A trained checkpoint has distinct norms; the fixture has to as well.

``original_architecture`` is set as ``from_pretrained`` sets it: AWQ's scaling and
clipping rules are per architecture (src/compression/awq.py::ARCH_RULES). TransformerLens
itself branches on it only for OLMo, so it changes nothing else here.
"""

from __future__ import annotations

BASE = {
    "n_layers": 2, "d_model": 16, "n_ctx": 16, "d_head": 4, "n_heads": 4, "d_mlp": 32,
    "d_vocab": 50, "positional_embedding_type": "rotary", "rotary_dim": 4,
}
ARCHS: dict[str, dict] = {
    "pythia": {
        "act_fn": "gelu", "normalization_type": "LN", "parallel_attn_mlp": True,
        "original_architecture": "GPTNeoXForCausalLM",
    },
    "llama": {
        "act_fn": "silu", "normalization_type": "RMS", "gated_mlp": True, "n_key_value_heads": 2,
        "original_architecture": "LlamaForCausalLM",
    },
    "gemma2": {
        "act_fn": "gelu_pytorch_tanh", "normalization_type": "RMS", "gated_mlp": True,
        "n_key_value_heads": 2, "use_normalization_before_and_after": True,
        "original_architecture": "Gemma2ForCausalLM",
    },
}


def tiny_model(arch: str, *, extraction_hooks: bool = True, seed: int = 0):
    import torch
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    from src.extraction.real_model import enable_extraction_hooks

    torch.manual_seed(seed)
    model = HookedTransformer(HookedTransformerConfig(**BASE, **ARCHS[arch], seed=seed)).eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            leaf = name.split(".")[-1].lstrip("_")
            if leaf == "w":  # norm scale
                param.copy_(1.0 + 0.5 * torch.randn(param.shape, generator=generator))
            elif leaf == "b" or leaf.startswith("b_"):  # norm shift and projection biases
                param.copy_(0.1 * torch.randn(param.shape, generator=generator))
    if extraction_hooks:
        enable_extraction_hooks(model)
    return model
