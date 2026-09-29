# [AI-GEN] agent=Claude date=2026-09-29 task=Tests for real-model Wanda (per-matrix, upstream-equivalent)
# reviewed-by: PENDING

"""Real-model Wanda (magnitude_prune.prune_wanda_torch, WandaPruner).

The central claim is that our masks are upstream's masks. So the first class runs
**upstream's own function** - ``saediag.pruning.prune_wanda_style_inplace`` from the
pinned fork, loaded read-only by path - on each TransformerLens projection laid out as
the HF ``nn.Linear`` it corresponds to, and requires the identical mask. It skips on a
machine without the fork rather than pass vacuously.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

from tiny_tl import ARCHS, tiny_model

from src.calibration import second_moment
from src.calibration.second_moment import broadcast_shape, input_source_for
from src.calibration.token_cache import (
    CalibrationUnavailable,
    cache_path,
    save_token_cache,
    synthetic_token_cache,
)
from src.compression.magnitude_prune import prune_wanda_torch
from src.compression.torch_weights import null_draw_inputs, projection_parameters
from src.compression.wanda import WandaPruner

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO.parent / "sae-pruning-paper-main" / "revision" / "src" / "saediag" / "pruning.py"
LEVELS = (0.2, 0.4, 0.6)  # the grid's Wanda cells (deploy/shared/gen_cells.py)


@pytest.fixture(scope="module", params=sorted(ARCHS))
def model(request):
    return tiny_model(request.param)


@pytest.fixture(scope="module")
def upstream():
    if not UPSTREAM.exists():
        pytest.skip(f"upstream fork not present at {UPSTREAM}")
    spec = importlib.util.spec_from_file_location("_upstream_saediag_pruning", UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _moments(model, seed=0):
    """Positive, deliberately non-uniform statistics, so Wanda is not rescaled |W|."""
    rng = np.random.default_rng(seed)
    params = dict(model.named_parameters())
    return {
        name: rng.uniform(0.1, 10.0, size=broadcast_shape(tuple(params[name].shape), input_source_for(name)[1]))
        for name in projection_parameters(model)
    }


def _as_hf_linear(name, weight, s2):
    """A TL projection as the HF nn.Linear it corresponds to: (weight [out, in], E[x^2]
    over `in`, and a map taking an [out, in] mask back to the TL layout)."""
    suffix = name.split(".")[-1].lstrip("_")
    s2 = torch.as_tensor(np.asarray(s2))
    if suffix in ("W_Q", "W_K", "W_V"):  # TL [H, D, dh]  ->  HF q/k/v_proj [H*dh, D]
        h, d, dh = weight.shape
        return (weight.permute(0, 2, 1).reshape(h * dh, d), s2.reshape(d),
                lambda m: m.reshape(h, dh, d).permute(0, 2, 1))
    if suffix == "W_O":                  # TL [H, dh, D]  ->  HF o_proj [D, H*dh]
        h, dh, d = weight.shape
        return (weight.reshape(h * dh, d).T, s2.reshape(h * dh),
                lambda m: m.T.reshape(h, dh, d))
    return weight.T, s2.reshape(weight.shape[0]), lambda m: m.T  # TL [in, out]


class TestMasksAreUpstreams:
    @pytest.mark.parametrize("sparsity", LEVELS)
    def test_identical_to_prune_wanda_style_inplace(self, model, upstream, sparsity):
        s2 = _moments(model)
        ours = dict(prune_wanda_torch(model, s2, sparsity).named_parameters())
        dense = dict(model.named_parameters())
        for name in projection_parameters(model):
            hf_weight, ex2, back = _as_hf_linear(name, dense[name].detach(), s2[name])
            holder = torch.nn.Module()
            holder.proj = torch.nn.Linear(hf_weight.shape[1], hf_weight.shape[0], bias=False)
            with torch.no_grad():
                holder.proj.weight.copy_(hf_weight)
            upstream.prune_wanda_style_inplace(holder, {"proj": ex2.numpy()}, sparsity)
            theirs = back(holder.proj.weight.detach() != 0)
            assert torch.equal(ours[name].detach() != 0, theirs), f"{name} at {sparsity}"

    def test_the_comparison_is_not_vacuous(self, model, upstream):
        """The HF layout must actually be different from TL's for the check to mean
        anything, and a wrong statistic must change upstream's mask."""
        s2 = _moments(model)
        name = "blocks.0.attn.W_O"
        weight = dict(model.named_parameters())[name].detach()
        hf_weight, ex2, back = _as_hf_linear(name, weight, s2[name])
        assert hf_weight.shape != weight.shape
        masks = []
        for stat in (ex2, ex2.flip(0)):
            holder = torch.nn.Module()
            holder.proj = torch.nn.Linear(hf_weight.shape[1], hf_weight.shape[0], bias=False)
            with torch.no_grad():
                holder.proj.weight.copy_(hf_weight)
            upstream.prune_wanda_style_inplace(holder, {"proj": stat.numpy()}, 0.4)
            masks.append(back(holder.proj.weight.detach() != 0))
        assert not torch.equal(masks[0], masks[1])


class TestTheRule:
    @pytest.mark.parametrize("sparsity", LEVELS)
    def test_each_matrix_hits_its_own_target(self, model, sparsity):
        """Per matrix: exactly int(numel * sparsity) zeros when there are no ties."""
        pruned = dict(prune_wanda_torch(model, _moments(model), sparsity).named_parameters())
        for name in projection_parameters(model):
            p = pruned[name]
            assert int((p == 0).sum()) == int(p.numel() * sparsity), name

    def test_a_constant_statistic_reduces_to_per_matrix_magnitude(self, model):
        flat = {name: np.ones_like(s) for name, s in _moments(model).items()}
        pruned = dict(prune_wanda_torch(model, flat, 0.4).named_parameters())
        for name, w in model.named_parameters():
            if name not in flat:
                continue
            a = w.detach().abs().flatten()
            threshold = torch.kthvalue(a, int(a.numel() * 0.4)).values
            assert torch.equal(pruned[name].detach() != 0, w.detach().abs() > threshold), name

    def test_the_statistic_changes_the_mask(self, model):
        s2 = _moments(model)
        flat = {name: np.ones_like(s) for name, s in s2.items()}
        a = dict(prune_wanda_torch(model, s2, 0.4).named_parameters())
        b = dict(prune_wanda_torch(model, flat, 0.4).named_parameters())
        assert any(not torch.equal(a[n] != 0, b[n] != 0) for n in s2)

    def test_only_projections_change_and_the_original_is_untouched(self, model):
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        pruned = dict(prune_wanda_torch(model, _moments(model), 0.6).named_parameters())
        projections = set(projection_parameters(model))
        for name, value in before.items():
            assert torch.equal(dict(model.named_parameters())[name].detach(), value), name
            if name not in projections:
                assert torch.equal(pruned[name].detach(), value), name

    def test_zero_sparsity_prunes_nothing(self, model):
        pruned = dict(prune_wanda_torch(model, _moments(model), 0.0).named_parameters())
        for name, w in model.named_parameters():
            assert torch.equal(pruned[name].detach(), w.detach()), name


class TestRefusals:
    def test_a_missing_statistic_is_an_error_not_a_magnitude_fallback(self, model):
        s2 = _moments(model)
        s2.pop("blocks.1.mlp.W_out")
        with pytest.raises(ValueError, match="refusing to score them by"):
            prune_wanda_torch(model, s2, 0.4)

    def test_a_statistic_that_does_not_broadcast_is_refused(self, model):
        s2 = _moments(model)
        s2["blocks.0.attn.W_O"] = np.ones((3, 1))
        with pytest.raises((ValueError, RuntimeError)):
            prune_wanda_torch(model, s2, 0.4)

    @pytest.mark.parametrize("sparsity", [-0.1, 1.5])
    def test_sparsity_out_of_range(self, model, sparsity):
        with pytest.raises(ValueError, match="target_sparsity"):
            prune_wanda_torch(model, _moments(model), sparsity)


class TestTheCompressor:
    def test_apply_uses_injected_statistics(self, model):
        s2 = _moments(model)
        via_class = dict(WandaPruner(0.4, calibration=s2).apply(model, None).named_parameters())
        direct = dict(prune_wanda_torch(model, s2, 0.4).named_parameters())
        for name, value in direct.items():
            assert torch.equal(via_class[name], value), name

    def test_weight_delta_is_measured_and_feeds_the_null(self, model):
        """Nonzero exactly on the projections, and accepted by the B1 null's inputs."""
        delta = WandaPruner(0.4, calibration=_moments(model)).weight_delta(model, None)
        assert set(delta) == {n for n, _ in model.named_parameters()}
        projections = set(projection_parameters(model))
        assert {n for n, v in delta.items() if v > 0} == projections
        magnitudes, shapes = null_draw_inputs(model, delta)
        assert set(magnitudes) == set(shapes) == projections

    def test_no_statistics_and_no_config_is_missing_data(self, model):
        with pytest.raises(CalibrationUnavailable):
            WandaPruner(0.4).apply(model, None)

    def test_the_grid_cells_construct_through_their_kwargs(self):
        for level in LEVELS:
            assert WandaPruner(sparsity=level).sparsity == level


class TestStageBAndStageCShareOneStatistic:
    """weight_delta (Stage B) and apply (Stage C) read the same file, so the null is
    matched to exactly the mask Stage C applies."""

    def _resolved(self, tmp_path):
        return {
            "calibration": {"calibration": {
                "hf_id": "HuggingFaceFW/fineweb-edu", "n_tokens": 64, "seed": 7,
                "dtype": "bfloat16", "context_length": 16,
            }},
            "model": {"hf_id": "tiny/llama", "hf_revision": "b" * 40},
            "calibration_root": str(tmp_path),
        }

    def test_one_collection_serves_both_stages(self, tmp_path, monkeypatch):
        model = tiny_model("llama")
        resolved = self._resolved(tmp_path)
        save_token_cache(cache_path(resolved), synthetic_token_cache(4, 16, 50, seed=2), {})
        calls = []
        real = second_moment.collect_second_moments
        monkeypatch.setattr(
            second_moment, "collect_second_moments",
            lambda *a, **k: calls.append(1) or real(*a, **k),
        )
        stage_c = dict(WandaPruner(0.4).apply(model, resolved).named_parameters())
        delta = WandaPruner(0.4).weight_delta(model, resolved)
        stage_c_again = dict(WandaPruner(0.4).apply(model, resolved).named_parameters())
        assert len(calls) == 1
        for name, value in stage_c.items():
            assert torch.equal(value, stage_c_again[name]), name
        dense = dict(model.named_parameters())
        for name in projection_parameters(model):
            measured = float(torch.linalg.vector_norm(stage_c[name].detach() - dense[name].detach()))
            assert delta[name] == pytest.approx(measured, rel=1e-6), name

    def test_without_the_token_cache_it_is_missing_data(self, tmp_path):
        with pytest.raises(CalibrationUnavailable):
            WandaPruner(0.4).apply(tiny_model("llama"), self._resolved(tmp_path))
