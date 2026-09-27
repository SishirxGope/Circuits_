# [AI-GEN] agent=Claude date=2026-09-27 task=Tests for the torch adapter that carries the null onto real models (B1)
# reviewed-by: PENDING

"""The torch side of the matched-magnitude null (deploy/BLOCKERS.md B1).

The null generator in ``src/science/matched_magnitude.py`` is used unchanged; these tests
cover the adapter that feeds it a real model and writes its draws back. The property that
matters is end-to-end: after a draw is applied, the realized per-tensor
``||W_perturbed - W||_F`` must equal the magnitude the compressor measured. That equality
is what makes D_null a fair floor for D(c) — if it drifts, CSI divides by the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

from transformer_lens import HookedTransformer, HookedTransformerConfig

from src.compression.rtn import RtnQuantizer, quantizable_parameters
from src.compression.torch_weights import (
    apply_weight_delta,
    null_draw_inputs,
    tensor_shapes,
)
from src.science.matched_magnitude import generate_null_deltas
from src.synthetic.mock_model import MockModel


def _model(n_key_value_heads: int | None = None, dtype=torch.float32) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_heads=4, d_head=4, d_mlp=32, d_vocab=40, n_ctx=16,
        act_fn="gelu", normalization_type="LN", positional_embedding_type="rotary",
        rotary_dim=4, n_key_value_heads=n_key_value_heads,
        dtype=dtype, seed=0, device="cpu",
    )
    model = HookedTransformer(cfg)
    model.eval()
    return model


def _frob(t) -> float:
    return float(torch.linalg.vector_norm(t.detach().to(torch.float32)).item())


class TestShapesAreReadFromTheModel:
    def test_shapes_match_named_parameters(self):
        model = _model()
        shapes = tensor_shapes(model)
        assert shapes == {n: tuple(p.shape) for n, p in model.named_parameters()}

    def test_a_mock_model_still_uses_its_own_method(self):
        """Delegated, not reimplemented, so the two paths cannot diverge."""
        mock = MockModel(seed=0, n_layers=2, n_heads=2, d_model=8)
        assert tensor_shapes(mock) == dict(mock.tensor_shapes())

    def test_gqa_reports_the_compact_kv_parameters(self):
        """Must agree with what RtnQuantizer.weight_delta measures, or the null mis-scales."""
        model = _model(n_key_value_heads=2)
        shapes = tensor_shapes(model)
        assert shapes["blocks.0.attn._W_K"][0] == 2       # kv width, not n_heads
        assert model.blocks[0].attn.W_K.shape[0] == 4     # the expanded property
        assert "blocks.0.attn.W_K" not in shapes

    def test_something_that_is_not_a_model_is_refused(self):
        with pytest.raises(TypeError, match="cannot read tensor shapes"):
            tensor_shapes(object())


class TestOnlyTheCompressedTensorsArePerturbed:
    def test_untouched_tensors_are_excluded(self):
        """Embeddings and norms are not compressed, so the null must not perturb them."""
        model = _model()
        magnitudes = RtnQuantizer(bits=4).weight_delta(model, None)
        touched, shapes = null_draw_inputs(model, magnitudes)

        assert set(touched) == set(quantizable_parameters(model))
        assert set(shapes) == set(touched), "generate_null_deltas requires equal key sets"
        assert "embed.W_E" not in touched
        assert not any(name.endswith((".w", ".b")) for name in touched)

    def test_excluding_them_saves_the_float64_allocation(self):
        """8 bytes per parameter per draw; the embeddings dominate a small model."""
        model = _model()
        magnitudes = RtnQuantizer(bits=4).weight_delta(model, None)
        _, shapes = null_draw_inputs(model, magnitudes)
        all_params = sum(int(np.prod(s)) for s in tensor_shapes(model).values())
        perturbed = sum(int(np.prod(s)) for s in shapes.values())
        assert perturbed < all_params

    def test_an_all_zero_delta_is_refused_rather_than_drawn(self):
        model = _model()
        zeros = {name: 0.0 for name, _ in model.named_parameters()}
        with pytest.raises(ValueError, match="collapse to a spike at 0"):
            null_draw_inputs(model, zeros)

    def test_a_magnitude_for_an_unknown_tensor_is_refused(self):
        """Names drifting between compressor and null is exactly the GQA failure mode."""
        model = _model()
        with pytest.raises(ValueError, match="does not expose"):
            null_draw_inputs(model, {"blocks.0.attn.W_K_expanded": 1.0})


class TestTheDrawIsAppliedFaithfully:
    def test_the_realized_norm_equals_the_measured_magnitude(self):
        """The load-bearing property: matched magnitude, measured after application."""
        model = _model()
        magnitudes = RtnQuantizer(bits=4).weight_delta(model, None)
        touched, shapes = null_draw_inputs(model, magnitudes)
        deltas = generate_null_deltas(touched, shapes, 1, seed=0)[0]

        perturbed = apply_weight_delta(model, deltas)
        before = dict(model.named_parameters())
        after = dict(perturbed.named_parameters())

        for name, wanted in touched.items():
            realized = _frob(after[name] - before[name])
            assert realized == pytest.approx(wanted, rel=1e-5), name

    def test_tensors_outside_the_draw_are_untouched(self):
        model = _model()
        magnitudes = RtnQuantizer(bits=4).weight_delta(model, None)
        touched, shapes = null_draw_inputs(model, magnitudes)
        deltas = generate_null_deltas(touched, shapes, 1, seed=0)[0]

        perturbed = apply_weight_delta(model, deltas)
        after = dict(perturbed.named_parameters())
        for name, param in model.named_parameters():
            if name not in touched:
                assert torch.equal(param, after[name]), f"{name} was perturbed but should not be"

    def test_the_dense_model_is_never_mutated(self):
        """Stage B re-reads the dense model for each of the R draws."""
        model = _model()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        magnitudes = RtnQuantizer(bits=4).weight_delta(model, None)
        touched, shapes = null_draw_inputs(model, magnitudes)

        for delta in generate_null_deltas(touched, shapes, 3, seed=0):
            apply_weight_delta(model, delta)

        for name, param in model.named_parameters():
            assert torch.equal(before[name], param), f"apply_weight_delta mutated {name}"

    def test_draws_differ_from_each_other_but_are_reproducible(self):
        model = _model()
        magnitudes = RtnQuantizer(bits=4).weight_delta(model, None)
        touched, shapes = null_draw_inputs(model, magnitudes)

        a, b = generate_null_deltas(touched, shapes, 2, seed=0)
        name = next(iter(touched))
        assert not np.allclose(a[name], b[name]), "two null draws must not be identical"

        again = generate_null_deltas(touched, shapes, 2, seed=0)
        assert np.array_equal(a[name], again[0][name]), "same seed must reproduce the draw"

    def test_the_perturbed_model_still_runs(self):
        model = _model()
        magnitudes = RtnQuantizer(bits=4).weight_delta(model, None)
        touched, shapes = null_draw_inputs(model, magnitudes)
        deltas = generate_null_deltas(touched, shapes, 1, seed=0)[0]
        perturbed = apply_weight_delta(model, deltas)
        logits = perturbed(torch.zeros(1, 4, dtype=torch.long))
        assert torch.isfinite(logits).all()

    def test_a_shape_mismatch_is_refused(self):
        model = _model()
        with pytest.raises(ValueError, match="has shape"):
            apply_weight_delta(model, {"blocks.0.attn.W_Q": np.zeros((3, 3))})

    def test_a_name_the_model_does_not_have_is_refused(self):
        model = _model()
        with pytest.raises(ValueError, match="absent from the model"):
            apply_weight_delta(model, {"blocks.9.attn.W_Q": np.zeros((4, 16, 4))})

    def test_a_mock_model_still_uses_its_own_method(self):
        mock = MockModel(seed=0, n_layers=2, n_heads=2, d_model=8)
        magnitudes = RtnQuantizer(bits=4).weight_delta(mock, {})
        touched, shapes = null_draw_inputs(mock, magnitudes)
        deltas = generate_null_deltas(touched, shapes, 1, seed=0)[0]
        assert isinstance(apply_weight_delta(mock, deltas), MockModel)


class TestPrecision:
    def test_bfloat16_rounding_is_visible_rather_than_silent(self):
        """The null is matched only to the precision the model is stored in.

        Not a bug to fix here - a fact that has to be known before the Gemma cells run,
        since Gemma's config says bfloat16 while its checkpoint says float32.
        """
        model = _model(dtype=torch.bfloat16)
        magnitudes = RtnQuantizer(bits=4).weight_delta(model, None)
        touched, shapes = null_draw_inputs(model, magnitudes)
        deltas = generate_null_deltas(touched, shapes, 1, seed=0)[0]

        perturbed = apply_weight_delta(model, deltas)
        before = dict(model.named_parameters())
        after = dict(perturbed.named_parameters())

        name = next(iter(touched))
        realized = _frob(after[name] - before[name])
        # Close, but NOT exact the way float32 is: bf16 has ~3 decimal digits.
        assert realized == pytest.approx(touched[name], rel=0.10)
