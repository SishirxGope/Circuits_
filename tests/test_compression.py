# [AI-GEN] agent=Claude date=2026-08-08 task=Compressor tests (the compression grid had no coverage; weight_delta feeds the null)
# reviewed-by: PENDING

"""Compression-grid tests (proposal §3.2).

The load-bearing contract (ARCHITECTURE.md §4): ``Compressor.weight_delta`` is the
SINGLE SOURCE OF TRUTH for the magnitudes the matched-magnitude null must match, so
the null can never drift from the compression it is matched to. If weight_delta is
wrong, CSI divides by the wrong floor and every number in the paper is wrong.

Also covered: sparsity is exact (a pruner that quietly removes 29.6% instead of 30%
would break the matched-sparsity Wanda-vs-magnitude comparison, claim C6), and the
dense model is never mutated.
"""

import numpy as np
import pytest

from src.compression.awq import AwqCompressor
from src.compression.gptq import GptqCompressor
from src.compression.magnitude_prune import (
    MagnitudePruner,
    achieved_sparsity,
    prune_magnitude,
    prune_wanda,
)
from src.compression.rtn import RtnQuantizer, rtn_quantize
from src.compression.wanda import WandaPruner, synthetic_second_moments
from src.science.matched_magnitude import frobenius_norm
from src.synthetic.mock_model import MockModel


def _model():
    return MockModel(seed=0, n_layers=3, n_heads=2, d_model=6)


def _sparsity(weights):
    total = sum(v.size for v in weights.values())
    zeros = sum(int((v == 0).sum()) for v in weights.values())
    return zeros / total


class TestRtnQuantize:
    def test_output_lands_on_the_quantization_grid(self):
        w = np.linspace(-1.0, 1.0, 64).reshape(8, 8)
        q, meta = rtn_quantize(w, bits=4)
        levels = np.round(q / meta["scale"])
        assert np.allclose(q / meta["scale"], levels), "values are not on the integer grid"

    def test_fewer_bits_means_more_error(self):
        w = np.linspace(-1.0, 1.0, 64).reshape(8, 8)
        err8 = frobenius_norm(rtn_quantize(w, 8)[0] - w)
        err4 = frobenius_norm(rtn_quantize(w, 4)[0] - w)
        assert err4 > err8, "INT4 must be a coarser grid than INT8"

    def test_quantization_is_symmetric_with_zero_point_zero(self):
        _, meta = rtn_quantize(np.array([[-3.0, 2.0]]), bits=8)
        assert meta["zero_point"] == 0

    def test_clips_within_the_representable_range(self):
        w = np.array([[-4.0, 4.0, 0.0]])
        q, meta = rtn_quantize(w, bits=4)
        qmax = 2 ** (4 - 1) - 1
        assert np.all(np.abs(q / meta["scale"]) <= qmax + 1e-9)

    def test_all_zero_tensor_is_handled(self):
        q, meta = rtn_quantize(np.zeros((3, 3)), bits=4)
        assert np.array_equal(q, np.zeros((3, 3)))
        assert meta["scale"] == 0.0

    def test_invalid_bits_rejected(self):
        with pytest.raises(ValueError, match="bits must be an int"):
            rtn_quantize(np.ones((2, 2)), bits=1)

    def test_empty_tensor_rejected(self):
        with pytest.raises(ValueError, match="empty weight tensor"):
            rtn_quantize(np.array([]), bits=4)


class TestMagnitudePruning:
    def test_per_tensor_sparsity_is_exact(self):
        weights = {"w": np.arange(1, 101, dtype=np.float64).reshape(10, 10)}
        out = prune_magnitude(weights, target_sparsity=0.3, global_scope=False)
        assert _sparsity(out) == pytest.approx(0.30)

    def test_global_sparsity_is_exact(self):
        weights = {"a": np.arange(1, 51, dtype=np.float64).reshape(5, 10),
                   "b": np.arange(51, 101, dtype=np.float64).reshape(5, 10)}
        out = prune_magnitude(weights, target_sparsity=0.4, global_scope=True)
        assert _sparsity(out) == pytest.approx(0.40)

    def test_global_scope_removes_the_globally_smallest_weights(self):
        """Global pruning must strip the small tensor harder than the large one."""
        weights = {"small": np.full((4, 4), 0.01), "big": np.full((4, 4), 10.0)}
        out = prune_magnitude(weights, target_sparsity=0.5, global_scope=True)
        assert np.all(out["small"] == 0.0)
        assert np.all(out["big"] == 10.0)

    def test_per_tensor_scope_hits_the_exact_count_under_ties(self):
        """Per-tensor scope uses an exact top-k, so ties cannot move the count."""
        out = prune_magnitude({"w": np.ones((10, 10))}, target_sparsity=0.25, global_scope=False)
        assert _sparsity(out) == pytest.approx(0.25)

    def test_global_scope_refuses_to_return_a_dead_model_under_mass_ties(self):
        """Global scope reproduces upstream's strict `|w| > threshold` rule.

        Under exact ties that rule drops every tied weight at once, so "remove 25%"
        becomes "removed everything". Real float weights never tie, but the collapse
        must be loud rather than a silently dead model with a meaningless CSI.
        """
        with pytest.raises(ValueError, match="zeroed EVERY prunable weight"):
            prune_magnitude({"w": np.ones((10, 10))}, target_sparsity=0.25, global_scope=True)

    def test_global_scope_on_realistic_weights_lands_near_the_target(self):
        rng = np.random.default_rng(0)
        weights = {"a": rng.standard_normal((40, 40)), "b": rng.standard_normal((40, 40))}
        out = prune_magnitude(weights, target_sparsity=0.30, global_scope=True)
        assert _sparsity(out) == pytest.approx(0.30, abs=0.01)

    def test_survivors_keep_their_exact_values(self):
        weights = {"w": np.arange(1, 17, dtype=np.float64).reshape(4, 4)}
        out = prune_magnitude(weights, target_sparsity=0.5, global_scope=False)
        kept = out["w"][out["w"] != 0]
        assert set(kept.tolist()) == set(range(9, 17))

    def test_zero_sparsity_is_a_no_op(self):
        weights = {"w": np.arange(1, 17, dtype=np.float64).reshape(4, 4)}
        assert np.array_equal(prune_magnitude(weights, 0.0, global_scope=True)["w"], weights["w"])

    def test_full_sparsity_zeroes_everything(self):
        weights = {"w": np.ones((4, 4))}
        assert np.all(prune_magnitude(weights, 1.0, global_scope=True)["w"] == 0.0)

    def test_excluded_tensors_are_untouched(self):
        """lm_head has the smallest magnitudes in the model and still survives intact."""
        weights = {"w": np.arange(1, 17, dtype=np.float64).reshape(4, 4),
                   "lm_head": np.full((4, 4), 1e-9)}
        out = prune_magnitude(weights, 0.5, global_scope=True)
        assert np.array_equal(out["lm_head"], weights["lm_head"])
        assert (out["w"] == 0).any(), "the prunable tensor should still have been pruned"

    def test_inputs_are_never_mutated(self):
        weights = {"w": np.arange(1, 17, dtype=np.float64).reshape(4, 4)}
        before = weights["w"].copy()
        prune_magnitude(weights, 0.5, global_scope=True)
        prune_magnitude(weights, 0.5, global_scope=False)
        assert np.array_equal(weights["w"], before)

    def test_out_of_range_sparsity_rejected(self):
        with pytest.raises(ValueError, match="target_sparsity"):
            prune_magnitude({"w": np.ones((2, 2))}, 1.5)

    def test_handles_1d_tensors(self):
        """Real checkpoints carry 1-D tensors; index-pair logic used to assume 2-D."""
        out = prune_magnitude({"bias": np.arange(1, 11, dtype=np.float64)}, 0.5, global_scope=False)
        assert _sparsity(out) == pytest.approx(0.5)


class TestEmbeddingInTheThresholdPool:
    """Reproduces saediag.pruning.prune_magnitude_global_inplace (verified 2026-08-08).

    Upstream adds the token embedding to the threshold POOL but never zeroes it, and
    documents that this "is what maps the labeled sparsity (e.g. 0.30) to the observed
    effective per-matrix sparsities (~0.26 attn, ~0.38 MLP)". If our pool differs from
    theirs, our "30% cell" is not their "30% cell" and the C5 cross-audit compares
    cells that are not the same cell (PRD.md §2).
    """

    # An embedding whose magnitudes differ from the Linear weights — which is the whole
    # reason the pool choice moves the threshold.
    WEIGHTS = {"linear": np.arange(1, 17, dtype=np.float64).reshape(4, 4),
               "embed": np.full((8, 8), 0.01)}

    def test_the_embedding_is_never_zeroed(self):
        out = prune_magnitude(self.WEIGHTS, 0.5, embedding_names=("embed",))
        assert np.array_equal(out["embed"], self.WEIGHTS["embed"])

    def test_including_the_embedding_changes_the_threshold(self):
        """The documented upstream effect, reproduced."""
        with_embed = prune_magnitude(self.WEIGHTS, 0.5, embedding_names=("embed",),
                                     include_embedding_in_threshold=True)
        without = prune_magnitude(self.WEIGHTS, 0.5, embedding_names=("embed",),
                                  include_embedding_in_threshold=False)
        assert int((with_embed["linear"] == 0).sum()) != int((without["linear"] == 0).sum())

    def test_labelled_sparsity_is_not_achieved_sparsity(self):
        """The fact that must be stated in the paper rather than assumed away."""
        out = prune_magnitude(self.WEIGHTS, 0.5, embedding_names=("embed",),
                              include_embedding_in_threshold=True)
        achieved = achieved_sparsity(out, embedding_names=("embed",))["__overall__"]
        assert achieved != pytest.approx(0.5)

    def test_achieved_sparsity_excludes_the_never_zeroed_embedding(self):
        """Counting the embedding in the denominator would dilute a 50% prune to a few
        percent on a real model, where the embedding matrix dwarfs any single Linear."""
        out = prune_magnitude(self.WEIGHTS, 0.5, embedding_names=("embed",),
                              include_embedding_in_threshold=False)
        correct = achieved_sparsity(out, embedding_names=("embed",))["__overall__"]
        diluted = achieved_sparsity(out)["__overall__"]  # embedding wrongly counted
        assert correct == pytest.approx(0.5)
        assert diluted < correct

    def test_unknown_embedding_name_is_rejected(self):
        with pytest.raises(ValueError, match="embedding_names not present"):
            prune_magnitude(self.WEIGHTS, 0.5, embedding_names=("not-a-tensor",))

    def test_achieved_sparsity_reports_per_tensor_and_overall(self):
        out = prune_magnitude(self.WEIGHTS, 0.5, embedding_names=("embed",),
                              include_embedding_in_threshold=False)
        report = achieved_sparsity(out, embedding_names=("embed",))
        assert "linear" in report and "__overall__" in report
        assert "embed" not in report


class TestWandaPruning:
    def test_score_is_magnitude_times_root_second_moment(self):
        """A large weight with tiny activations must lose to a small weight with large ones."""
        weights = {"w": np.array([[10.0, 1.0]])}
        s2 = {"w": np.array([[1e-8, 100.0]])}
        out = prune_wanda(weights, s2, target_sparsity=0.5)
        assert out["w"][0, 0] == 0.0
        assert out["w"][0, 1] == 1.0

    def test_sparsity_is_exact(self):
        weights = {"w": np.arange(1, 101, dtype=np.float64).reshape(10, 10)}
        s2 = {"w": np.ones((10, 10))}
        assert _sparsity(prune_wanda(weights, s2, 0.6)) == pytest.approx(0.6)

    def test_missing_second_moment_is_an_error_not_a_silent_skip(self):
        with pytest.raises(ValueError, match="missing second moment"):
            prune_wanda({"w": np.ones((2, 2))}, {}, 0.5)

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="second-moment shape"):
            prune_wanda({"w": np.ones((2, 2))}, {"w": np.ones((3, 3))}, 0.5)

    def test_synthetic_second_moments_are_positive_and_shape_matched(self):
        m = _model()
        s2 = synthetic_second_moments(m, seed=1)
        assert set(s2) == set(m.tensor_shapes())
        assert all(np.all(v > 0) for v in s2.values())

    def test_synthetic_second_moments_are_deterministic(self):
        m = _model()
        a, b = synthetic_second_moments(m, seed=3), synthetic_second_moments(m, seed=3)
        assert all(np.array_equal(a[k], b[k]) for k in a)


class TestWeightDeltaFeedsTheNull:
    """ARCHITECTURE.md §4: weight_delta is the single source of truth for the null."""

    @pytest.mark.parametrize(
        "compressor",
        [RtnQuantizer(bits=4), MagnitudePruner(sparsity=0.3), WandaPruner(sparsity=0.3),
         GptqCompressor(bits=4), AwqCompressor(bits=4)],
        ids=["rtn", "magnitude", "wanda", "gptq-stub", "awq-stub"],
    )
    def test_delta_matches_the_actual_weight_change(self, compressor):
        model = _model()
        compressed = compressor.apply(model, {})
        reported = compressor.weight_delta(model, {})
        for name in model.tensor_shapes():
            actual = frobenius_norm(model.weights[name] - compressed.weights[name])
            assert reported[name] == pytest.approx(actual, rel=1e-12), (
                f"weight_delta disagrees with the real change for {name}: the null "
                "would be matched to a magnitude the compression never produced"
            )

    @pytest.mark.parametrize(
        "compressor",
        [RtnQuantizer(bits=4), MagnitudePruner(sparsity=0.3), WandaPruner(sparsity=0.3)],
        ids=["rtn", "magnitude", "wanda"],
    )
    def test_dense_model_is_never_mutated(self, compressor):
        model = _model()
        before = {k: v.copy() for k, v in model.weights.items()}
        compressor.apply(model, {})
        compressor.weight_delta(model, {})
        assert all(np.array_equal(model.weights[k], v) for k, v in before.items())

    def test_delta_is_deterministic(self):
        model = _model()
        p = MagnitudePruner(sparsity=0.4)
        assert p.weight_delta(model, {}) == p.weight_delta(model, {})

    def test_more_sparsity_means_a_larger_delta(self):
        model = _model()
        light = sum(MagnitudePruner(sparsity=0.1).weight_delta(model, {}).values())
        heavy = sum(MagnitudePruner(sparsity=0.6).weight_delta(model, {}).values())
        assert heavy > light

    def test_fewer_bits_means_a_larger_delta(self):
        model = _model()
        int8 = sum(RtnQuantizer(bits=8).weight_delta(model, {}).values())
        int4 = sum(RtnQuantizer(bits=4).weight_delta(model, {}).values())
        assert int4 > int8


class TestRealModelsAreStillRefused:
    @pytest.mark.parametrize(
        "compressor",
        [RtnQuantizer(), MagnitudePruner(), WandaPruner(), GptqCompressor(), AwqCompressor()],
        ids=["rtn", "magnitude", "wanda", "gptq", "awq"],
    )
    def test_apply_refuses_non_mock_models(self, compressor):
        with pytest.raises(NotImplementedError):
            compressor.apply(object(), {})

    @pytest.mark.parametrize(
        "compressor",
        [RtnQuantizer(), MagnitudePruner(), WandaPruner(), GptqCompressor(), AwqCompressor()],
        ids=["rtn", "magnitude", "wanda", "gptq", "awq"],
    )
    def test_weight_delta_refuses_non_mock_models(self, compressor):
        with pytest.raises(NotImplementedError, match="Stage C approval"):
            compressor.weight_delta(object(), {})
