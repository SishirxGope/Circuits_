# [AI-GEN] agent=Claude date=2026-08-08 task=Perturber tests (ARCHITECTURE.md §6 required coverage that was missing)
# reviewed-by: PENDING

"""Matched-magnitude null tests (proposal §2.1a).

ARCHITECTURE.md §6 requires, verbatim: "Perturber: per-tensor Frobenius of output
delta matches requested magnitude within tolerance; direction distribution passes an
isotropy sanity check."

This is the module the whole paper rests on: if the null's magnitude does not match
the compression it is matched to, CSI is measuring nothing.
"""

import numpy as np
import pytest

from src.common.seeding import create_seed_generator
from src.science.matched_magnitude import (
    MatchedMagnitudePerturber,
    frobenius_norm,
    generate_null_deltas,
)
from src.synthetic.mock_model import MockModel

MAGNITUDES = {"a": 2.5, "b": 0.0, "c": 1.0}
SHAPES = {"a": (4, 3), "b": (2, 2), "c": (7,)}  # deliberately includes a 1-D tensor


class TestFrobeniusNorm:
    def test_matches_numpy_fro_on_2d(self):
        m = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert frobenius_norm(m) == pytest.approx(float(np.linalg.norm(m, "fro")))

    def test_handles_1d_tensors(self):
        """Real checkpoints carry 1-D bias / LayerNorm tensors; np.linalg.norm(x,'fro') raises on those."""
        v = np.array([3.0, 4.0])
        assert frobenius_norm(v) == pytest.approx(5.0)

    def test_handles_3d_tensors(self):
        t = np.ones((2, 3, 4))
        assert frobenius_norm(t) == pytest.approx(np.sqrt(24.0))


class TestMagnitudeMatching:
    def test_per_tensor_norm_matches_the_request_exactly(self):
        deltas = generate_null_deltas(MAGNITUDES, SHAPES, R=4, seed=11)
        for draw in deltas:
            for name, target in MAGNITUDES.items():
                assert frobenius_norm(draw[name]) == pytest.approx(target, rel=1e-12), (
                    f"null magnitude drifted for {name}: the null must match the "
                    "compression it is matched to (ARCHITECTURE.md §4)"
                )

    def test_zero_magnitude_gives_exactly_zeros(self):
        draw = generate_null_deltas(MAGNITUDES, SHAPES, R=1, seed=0)[0]
        assert np.array_equal(draw["b"], np.zeros(SHAPES["b"]))

    def test_shapes_are_preserved(self):
        draw = generate_null_deltas(MAGNITUDES, SHAPES, R=1, seed=0)[0]
        assert {k: v.shape for k, v in draw.items()} == SHAPES

    def test_returns_exactly_R_draws(self):
        assert len(generate_null_deltas(MAGNITUDES, SHAPES, R=6, seed=0)) == 6

    def test_draws_differ_from_each_other(self):
        draws = generate_null_deltas(MAGNITUDES, SHAPES, R=3, seed=5)
        assert not np.allclose(draws[0]["a"], draws[1]["a"])


class TestDeterminism:
    """AI_RULES.md 1.1 — the frozen null must be exactly regenerable from its seed."""

    def test_same_seed_gives_identical_draws(self):
        a = generate_null_deltas(MAGNITUDES, SHAPES, R=3, seed=11)
        b = generate_null_deltas(MAGNITUDES, SHAPES, R=3, seed=11)
        for da, db in zip(a, b):
            for name in MAGNITUDES:
                assert np.array_equal(da[name], db[name])

    def test_different_seed_gives_different_draws(self):
        a = generate_null_deltas(MAGNITUDES, SHAPES, R=1, seed=11)[0]
        b = generate_null_deltas(MAGNITUDES, SHAPES, R=1, seed=12)[0]
        assert not np.allclose(a["a"], b["a"])


class TestIsotropy:
    """Direction distribution sanity check (ARCHITECTURE.md §6).

    The null's direction must be random, not aligned with any particular axis: a
    perturbation that quietly preferred large-magnitude coordinates would be a
    magnitude-pruning-like null, not a random one, and would deflate CSI.
    """

    def test_mean_direction_is_near_zero(self):
        deltas = generate_null_deltas({"w": 1.0}, {"w": (200,)}, R=200, seed=3)
        directions = np.stack([d["w"] / frobenius_norm(d["w"]) for d in deltas])
        mean_dir = directions.mean(axis=0)
        assert frobenius_norm(mean_dir) < 0.15, "draws are not centred: direction is biased"

    def test_coordinate_variance_is_uniform_across_axes(self):
        deltas = generate_null_deltas({"w": 1.0}, {"w": (60,)}, R=400, seed=4)
        directions = np.stack([d["w"] / frobenius_norm(d["w"]) for d in deltas])
        per_axis = directions.var(axis=0)
        # isotropic => every coordinate has the same variance (1/n for a unit sphere)
        assert per_axis.max() / per_axis.min() < 3.0, "direction distribution is anisotropic"

    def test_no_preferred_sign(self):
        deltas = generate_null_deltas({"w": 1.0}, {"w": (50,)}, R=200, seed=5)
        signs = np.stack([np.sign(d["w"]) for d in deltas])
        assert abs(signs.mean()) < 0.1


class TestPerturberContract:
    def test_dense_model_is_never_mutated(self):
        """ARCHITECTURE.md §2: raw checkpoints are read-only; perturbation copies."""
        model = MockModel(seed=0, n_layers=2, n_heads=1, d_model=4)
        before = {k: v.copy() for k, v in model.weights.items()}
        magnitudes = {k: 0.5 for k in model.tensor_shapes()}
        rng, _ = create_seed_generator(9)
        perturbed = MatchedMagnitudePerturber().apply(model, magnitudes, rng)

        for k, v in before.items():
            assert np.array_equal(model.weights[k], v), "the dense model was mutated"
        assert perturbed is not model
        assert not np.allclose(perturbed.weights[next(iter(before))], before[next(iter(before))])

    def test_applied_delta_has_the_requested_magnitude(self):
        model = MockModel(seed=0, n_layers=2, n_heads=1, d_model=4)
        magnitudes = {k: 1.25 for k in model.tensor_shapes()}
        rng, _ = create_seed_generator(9)
        perturbed = MatchedMagnitudePerturber().apply(model, magnitudes, rng)
        for name, target in magnitudes.items():
            actual = frobenius_norm(perturbed.weights[name] - model.weights[name])
            assert actual == pytest.approx(target, rel=1e-12)

    def test_real_models_refuse_until_stage_b_engineering(self):
        rng, _ = create_seed_generator(0)
        with pytest.raises(NotImplementedError, match="RUN MODEL DOWNLOAD"):
            MatchedMagnitudePerturber().apply(object(), {"w": 1.0}, rng)

    def test_magnitude_key_mismatch_is_rejected(self):
        model = MockModel(seed=0, n_layers=1, n_heads=1, d_model=3)
        rng, _ = create_seed_generator(0)
        with pytest.raises(ValueError, match="do not match model tensors"):
            MatchedMagnitudePerturber().apply(model, {"not-a-tensor": 1.0}, rng)


class TestInputValidation:
    def test_non_positive_R_rejected(self):
        with pytest.raises(ValueError, match="R must be a positive int"):
            generate_null_deltas(MAGNITUDES, SHAPES, R=0, seed=0)

    def test_magnitude_without_shape_rejected(self):
        with pytest.raises(ValueError, match="unknown tensors"):
            generate_null_deltas({"ghost": 1.0}, SHAPES, R=1, seed=0)

    def test_shape_without_magnitude_rejected(self):
        """Silently skipping a tensor would under-perturb the null."""
        with pytest.raises(ValueError, match="no magnitude provided"):
            generate_null_deltas({"a": 1.0}, SHAPES, R=1, seed=0)
