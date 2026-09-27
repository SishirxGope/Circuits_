# [AI-GEN] agent=Claude date=2026-09-27 task=Real-model RTN tests (B2.1; the mock path had coverage, the torch path had none)
# reviewed-by: PENDING

"""Real-model RTN tests (deploy/BLOCKERS.md B2, proposal §3.2).

Three things are load-bearing here and each has a test:

1. **weight_delta is MEASURED, not predicted.** It is the single source of truth for
   the magnitudes the matched-magnitude null matches (ARCHITECTURE.md §4). An analytic
   estimate of quantization error would make D_null answer a different question than
   D, and CSI would divide by the wrong floor.
2. **apply() does not mutate the dense model.** Stage B re-reads it for every null
   draw; in-place quantization would compound across draws and silently corrupt R-1
   of them.
3. **GQA quantizes the compact ``_W_K``/``_W_V`` parameters, never the expanded
   ``W_K``/``W_V`` properties.** The properties are
   ``torch.repeat_interleave(_W_K, repeats=n_heads//n_key_value_heads)``, so
   quantizing them would report Frobenius norms inflated by that factor and mis-scale
   the null on exactly the two primary models (Gemma-2-2b 8/4, Llama-3.2-1B 32/8).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformer_lens import HookedTransformer, HookedTransformerConfig

from src.compression.rtn import (
    RtnQuantizer,
    quantizable_parameters,
    rtn_quantize_torch,
)


def _model(n_key_value_heads: int | None = None) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_heads=4, d_head=4, d_mlp=32, d_vocab=40, n_ctx=16,
        act_fn="gelu", normalization_type="LN", positional_embedding_type="rotary",
        rotary_dim=4, n_key_value_heads=n_key_value_heads,
        dtype=torch.float32, seed=0, device="cpu",
    )
    model = HookedTransformer(cfg)
    model.eval()
    return model


def test_selects_projections_and_leaves_embeddings_and_norms_alone():
    """Quantizing embeddings or norms would be a different experiment (GPTQ §4)."""
    model = _model()
    selected = quantizable_parameters(model)

    assert "blocks.0.attn.W_Q" in selected
    assert "blocks.0.mlp.W_out" in selected
    assert len(selected) == 12  # (W_Q, W_K, W_V, W_O, W_in, W_out) x 2 layers

    for name in ("embed.W_E", "unembed.W_U", "blocks.0.ln1.w", "blocks.0.attn.b_Q"):
        assert name not in selected


def test_quantization_error_grows_as_bits_shrink():
    """A quantizer whose error did not order by bit-width is not quantizing."""
    w = torch.randn(8, 16, 4, generator=torch.Generator().manual_seed(0))
    errors = [
        torch.linalg.vector_norm(rtn_quantize_torch(w, bits) - w).item()
        for bits in (8, 6, 4, 2)
    ]
    assert errors == sorted(errors), f"error must increase as bits fall, got {errors}"
    assert errors[0] > 0.0


def test_scale_is_per_output_channel_not_per_tensor():
    """Per-channel is the B2 specification; per-tensor would be a different quantizer.

    One output channel is given a much larger range than the others. Under per-channel
    scaling the small channels keep their own fine scale; under a single per-tensor
    scale the large channel's range would swamp them and they would collapse to zero.
    """
    w = torch.zeros(1, 8, 3)
    w[..., 0] = 1000.0          # loud channel
    w[..., 1] = 0.001           # quiet channel
    w[..., 2] = 0.002
    q = rtn_quantize_torch(w, bits=4)
    assert q[..., 1].abs().sum() > 0.0, "quiet channel was destroyed - scale is not per-channel"
    assert torch.allclose(q[..., 1], w[..., 1], rtol=0.5)


def test_all_zero_channel_does_not_divide_by_zero():
    w = torch.zeros(1, 4, 2)
    w[..., 0] = 0.5
    q = rtn_quantize_torch(w, bits=4)
    assert torch.isfinite(q).all()
    assert torch.equal(q[..., 1], torch.zeros_like(q[..., 1]))


def test_apply_does_not_mutate_the_dense_model():
    """Stage B re-reads the dense model for every one of the R null draws."""
    model = _model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    compressed = RtnQuantizer(bits=4).apply(model, None)

    for name, param in model.named_parameters():
        assert torch.equal(before[name], param), f"apply() mutated {name}"
    changed = [
        n for n, p in compressed.named_parameters() if not torch.equal(before[n], p)
    ]
    assert set(changed) == set(quantizable_parameters(model))


def test_weight_delta_equals_the_measured_frobenius_norm():
    """The contract: measured ||W_q - W||_F per tensor, never a predicted magnitude."""
    model = _model()
    quantizer = RtnQuantizer(bits=4)

    delta = quantizer.weight_delta(model, None)
    compressed = dict(quantizer.apply(model, None).named_parameters())
    dense = dict(model.named_parameters())

    assert set(delta) == set(dense), "every tensor must be reported, zeros included"
    for name in quantizable_parameters(model):
        measured = torch.linalg.vector_norm(compressed[name] - dense[name]).item()
        assert delta[name] == pytest.approx(measured, rel=1e-6, abs=1e-9)
    for name in set(dense) - set(quantizable_parameters(model)):
        assert delta[name] == 0.0


def test_gqa_quantizes_the_compact_kv_parameters_not_the_expanded_properties():
    """The trap: W_K is repeat_interleave(_W_K), so its norm is inflated by the group size."""
    model = _model(n_key_value_heads=2)   # 4 query heads, 2 kv heads -> group size 2
    selected = quantizable_parameters(model)

    assert "blocks.0.attn._W_K" in selected
    assert "blocks.0.attn._W_V" in selected

    compact = dict(model.named_parameters())["blocks.0.attn._W_K"]
    expanded = model.blocks[0].attn.W_K
    assert compact.shape[0] == 2, "named_parameters() must yield the kv-width tensor"
    assert expanded.shape[0] == 4, "the property is expanded to n_heads"

    delta = RtnQuantizer(bits=4).weight_delta(model, None)
    inflated = torch.linalg.vector_norm(
        rtn_quantize_torch(expanded, 4) - expanded
    ).item()
    assert delta["blocks.0.attn._W_K"] < inflated, (
        "K/V magnitude matches the expanded property - the null would be mis-scaled "
        "by n_heads/n_key_value_heads on both primary models"
    )


def test_dtype_is_preserved_so_the_model_still_runs():
    """Simulated quantization: dequantized back into the model's own precision."""
    model = _model()
    compressed = RtnQuantizer(bits=8).apply(model, None)
    for name, param in compressed.named_parameters():
        assert param.dtype == torch.float32, name
    logits = compressed(torch.zeros(1, 4, dtype=torch.long))
    assert torch.isfinite(logits).all()
