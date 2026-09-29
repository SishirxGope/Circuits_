# [AI-GEN] agent=Claude date=2026-09-29 task=Tests for the per-projection input second-moment collector
# reviewed-by: PENDING

"""The E[x^2] collector (src/calibration/second_moment.py).

Wanda's score is only as right as the statistic, and the statistic is only as right as
the place it is read from. So these tests do not trust the collector's source table.
For every projection they derive the input **independently** - by calling the block's
own norm module on TransformerLens's pre-norm hooks - prove that input is what the
projection consumes by reproducing the model's own downstream activation, and only
then require the collector's statistic to equal its mean square.

This found a real error during development: ``ln1/ln2.hook_normalized`` fires before
the norm's affine ``* w + b``, so it is the projection input only for untrained norms.
``TestTheTrap`` pins that down.

All models carry the extraction hook flags, because that is the model the pipeline
hands to a compressor (``load_pinned_model`` -> ``enable_extraction_hooks``), and
randomised norms (tests/tiny_tl.py), because a trained checkpoint has them.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

from tiny_tl import ARCHS, tiny_model

from src.calibration import second_moment
from src.calibration.second_moment import (
    broadcast_shape,
    collect_second_moments,
    input_source_for,
    load_or_collect_second_moments,
    second_moments_path,
    statistic_shape,
)
from src.calibration.token_cache import (
    CalibrationUnavailable,
    cache_path,
    save_token_cache,
    synthetic_token_cache,
)
from src.compression.torch_weights import projection_parameters


@pytest.fixture(scope="module", params=sorted(ARCHS))
def model(request):
    return tiny_model(request.param)


@pytest.fixture(scope="module")
def tokens():
    return synthetic_token_cache(5, 16, 50, seed=1)


def _cache(model, tokens):
    with torch.no_grad():
        _, cache = model.run_with_cache(torch.as_tensor(tokens))
    return cache


def _parse(pname):
    parts = pname.split(".")
    return int(parts[1]), parts[-1].lstrip("_")


def _param(params, prefix, name):
    """Compact GQA parameters are stored as _W_K/_b_K; plain ones as W_K/b_K."""
    return params.get(f"{prefix}_{name}", params.get(f"{prefix}{name}"))


def _true_input(model, cache, pname):
    """What the projection consumes, derived without the collector: the block's own norm
    applied to TransformerLens's PRE-norm hooks (split Q/K/V input and hook_mlp_in are
    both on under the extraction flags)."""
    layer, suffix = _parse(pname)
    block = model.blocks[layer]
    with torch.no_grad():
        if suffix in ("W_Q", "W_K", "W_V"):
            return block.ln1(cache[f"blocks.{layer}.hook_{suffix[-1].lower()}_input"])
        if suffix == "W_O":
            return cache[f"blocks.{layer}.attn.hook_z"]
        if suffix in ("W_in", "W_gate"):
            return block.ln2(cache[f"blocks.{layer}.hook_mlp_in"])
        if suffix == "W_out":
            return cache[f"blocks.{layer}.mlp.hook_post"]
    raise AssertionError(pname)


def _apply(model, cache, pname, x):
    """(the projection applied to x, the model's own activation it should reproduce)."""
    params = dict(model.named_parameters())
    layer, suffix = _parse(pname)
    w = params[pname]
    attn, mlp = f"blocks.{layer}.attn.", f"blocks.{layer}.mlp."
    with torch.no_grad():
        if suffix in ("W_Q", "W_K", "W_V"):
            letter = suffix[-1]
            got = torch.einsum("bphd,hde->bphe", x[:, :, : w.shape[0]], w) + _param(params, attn, f"b_{letter}")
            want = cache[f"{attn}hook_{letter.lower()}"]
            if got.shape != want.shape:  # compact K/V against the ungrouped hook
                got = got.repeat_interleave(want.shape[2] // got.shape[2], dim=2)
            return got, want
        if suffix == "W_O":
            return torch.einsum("bphe,hed->bphd", x, w), cache[f"{attn}hook_result"]
        if suffix == "W_gate":
            return x @ w, cache[f"{mlp}hook_pre"]
        if suffix == "W_in":
            if model.cfg.gated_mlp:  # TL's GatedMLP adds b_in after the gate multiply
                return x @ w, cache[f"{mlp}hook_pre_linear"]
            return x @ w + params[f"{mlp}b_in"], cache[f"{mlp}hook_pre"]
        if suffix == "W_out":
            out = x @ w + params[f"{mlp}b_out"]
            if model.cfg.use_normalization_before_and_after:
                # Gemma-2 post-norm. hook_mlp_out fires after ln2_post's affine;
                # ln2_post.hook_normalized fires before it (the same trap as ln1/ln2).
                out = model.blocks[layer].ln2_post(out)
            return out, cache[f"blocks.{layer}.hook_mlp_out"]
    raise AssertionError(pname)


class TestTheInputIsWhatTheProjectionConsumes:
    def test_the_derived_input_reproduces_every_projection(self, model, tokens):
        cache = _cache(model, tokens)
        names = projection_parameters(model)
        assert len(names) == model.cfg.n_layers * (7 if model.cfg.gated_mlp else 6)
        for pname in names:
            got, want = _apply(model, cache, pname, _true_input(model, cache, pname))
            assert got.shape == want.shape, pname
            assert torch.allclose(got, want, atol=1e-5), (
                f"{pname}: max diff {float((got - want).abs().max()):.2e}"
            )

    def test_the_statistic_is_the_mean_square_of_that_input(self, model, tokens):
        stats = collect_second_moments(model, tokens, batch_size=2)
        cache = _cache(model, tokens)
        for pname in projection_parameters(model):
            _, keep = input_source_for(pname)
            x = _true_input(model, cache, pname).to(torch.float64)
            want = (x * x).mean(dim=tuple(range(x.ndim - keep))).cpu().numpy()
            np.testing.assert_allclose(stats[pname].reshape(want.shape), want, rtol=1e-5, err_msg=pname)

    def test_split_qkv_input_repeats_identical_copies(self, model, tokens):
        """Why a mean over every leading axis is right under the extraction flags."""
        x = _true_input(model, _cache(model, tokens), "blocks.0.attn.W_Q")
        assert x.ndim == 4 and x.shape[2] == model.cfg.n_heads
        assert torch.equal(x, x[:, :, :1, :].expand_as(x))


class TestTheTrap:
    """``hook_normalized`` is the norm output BEFORE ``* w + b``. With trained norms it is
    not what the projections consume, and a collector reading it would be wrong on every
    real model without raising."""

    def test_hook_normalized_does_not_reproduce_the_query(self, model, tokens):
        cache = _cache(model, tokens)
        pre_affine = cache["blocks.0.ln1.hook_normalized"]
        got, want = _apply(model, cache, "blocks.0.attn.W_Q", pre_affine)
        assert not torch.allclose(got, want, atol=1e-3)

    def test_the_statistic_is_not_the_pre_affine_one(self, model, tokens):
        stats = collect_second_moments(model, tokens)
        x = _cache(model, tokens)["blocks.0.ln2.hook_normalized"].to(torch.float64)
        pre_affine = (x * x).mean(dim=tuple(range(x.ndim - 1))).cpu().numpy()
        assert not np.allclose(stats["blocks.0.mlp.W_in"].ravel(), pre_affine, rtol=1e-3)


class TestTheStatistic:
    def test_every_projection_gets_one_including_the_last_layer(self, model, tokens):
        stats = collect_second_moments(model, tokens)
        assert set(stats) == set(projection_parameters(model))
        last = f"blocks.{model.cfg.n_layers - 1}."
        assert any(name.startswith(last) for name in stats)

    def test_each_broadcasts_against_its_parameter(self, model, tokens):
        stats = collect_second_moments(model, tokens)
        params = dict(model.named_parameters())
        for name, stat in stats.items():
            shape = tuple(params[name].shape)
            assert stat.shape == broadcast_shape(shape, input_source_for(name)[1]), name
            assert np.broadcast_shapes(stat.shape, shape) == shape, name
            assert stat.dtype == np.float64 and (stat > 0).all(), name

    def test_batch_size_does_not_change_it(self, model, tokens):
        a = collect_second_moments(model, tokens, batch_size=1)
        b = collect_second_moments(model, tokens, batch_size=5)
        for name in a:
            np.testing.assert_allclose(a[name], b[name], rtol=1e-5, err_msg=name)

    @pytest.mark.parametrize(("n_tokens", "n_seq"), [(32, 2), (33, 3), (10_000, 5)])
    def test_the_token_budget_is_upstreams_ceiling_rule(self, model, tokens, n_tokens, n_seq):
        """ceil(n_tokens / ctx) sequences, capped at the cache (reprune.py)."""
        budget = collect_second_moments(model, tokens, n_tokens=n_tokens, batch_size=1)
        direct = collect_second_moments(model, tokens[:n_seq], batch_size=1)
        for name in budget:
            np.testing.assert_allclose(budget[name], direct[name], rtol=1e-12, err_msg=name)

    def test_the_calibration_dtype_is_used_and_the_model_is_untouched(self, model, tokens):
        fp32 = collect_second_moments(model, tokens)
        bf16 = collect_second_moments(model, tokens, dtype="bfloat16")
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert model.cfg.dtype == torch.float32
        name = "blocks.0.attn.W_O"
        assert not np.array_equal(fp32[name], bf16[name]), "the bf16 forward did not run"
        np.testing.assert_allclose(fp32[name], bf16[name], rtol=0.1)

    def test_no_hook_is_left_behind(self, model, tokens):
        """The pre-hooks are PyTorch hooks, which TransformerLens's reset does not remove."""
        collect_second_moments(model, tokens)
        for block in model.blocks:
            assert not block.attn._forward_pre_hooks
            assert not block.mlp._forward_pre_hooks


class TestRefusals:
    def test_a_non_finite_statistic_is_refused(self, tokens):
        """reprune.py: 'refusing to build a corrupted mask'."""
        broken = copy.deepcopy(tiny_model("gemma2"))
        with torch.no_grad():
            broken.blocks[0].ln1.w.fill_(float("inf"))
        with pytest.raises(RuntimeError, match="corrupted mask"):
            collect_second_moments(broken, tokens)

    @pytest.mark.parametrize("source", ["ln_final.hook_normalized", "ln_final:input"])
    def test_a_source_that_never_fires_is_an_error_not_a_fallback(self, tokens, monkeypatch, source):
        """Upstream scores a tensor with no statistic by |W| alone, which would quietly
        turn a Wanda cell into a magnitude cell. ln_final never runs below the stop layer,
        so it stands in for a hook or module TransformerLens renamed."""
        monkeypatch.setitem(second_moment.INPUT_SOURCES, "W_in", (source, 1))
        with pytest.raises(RuntimeError, match="never fired"):
            collect_second_moments(tiny_model("pythia"), tokens)

    def test_out_of_vocabulary_tokens_are_refused(self):
        bad = np.full((2, 16), 60, dtype=np.int64)
        with pytest.raises(ValueError, match="different tokenizer"):
            collect_second_moments(tiny_model("pythia"), bad)

    @pytest.mark.parametrize(
        "name", ["embed.W_E", "unembed.W_U", "blocks.0.ln1.w", "blocks.x.attn.W_Q", "blocks.0.attn.b_Q"]
    )
    def test_non_projection_names_are_refused(self, name):
        with pytest.raises(ValueError, match="not a TransformerLens projection"):
            input_source_for(name)


class TestShapes:
    @pytest.mark.parametrize(
        ("name", "source", "keep"),
        [
            ("blocks.3.attn.W_Q", "blocks.3.attn:query_input", 1),
            ("blocks.3.attn._W_K", "blocks.3.attn:key_input", 1),
            ("blocks.3.attn._W_V", "blocks.3.attn:value_input", 1),
            ("blocks.3.attn.W_O", "blocks.3.attn.hook_z", 2),
            ("blocks.3.mlp.W_in", "blocks.3.mlp:input", 1),
            ("blocks.3.mlp.W_gate", "blocks.3.mlp:input", 1),
            ("blocks.3.mlp.W_out", "blocks.3.mlp.hook_post", 1),
        ],
    )
    def test_source_for_each_projection(self, name, source, keep):
        assert input_source_for(name) == (source, keep)

    @pytest.mark.parametrize(
        ("shape", "keep", "stat", "bcast"),
        [
            ((4, 16, 4), 1, (16,), (1, 16, 1)),   # W_Q [H, D, dh]
            ((2, 16, 4), 1, (16,), (1, 16, 1)),   # compact _W_K [n_kv, D, dh]
            ((4, 4, 16), 2, (4, 4), (4, 4, 1)),   # W_O [H, dh, D]
            ((16, 32), 1, (16,), (16, 1)),        # W_in [D, M]
            ((32, 16), 1, (32,), (32, 1)),        # W_out [M, D]
        ],
    )
    def test_statistic_and_broadcast_shapes(self, shape, keep, stat, bcast):
        assert statistic_shape(shape, keep) == stat
        assert broadcast_shape(shape, keep) == bcast


# ---------------------------------------------------------------------------- disk cache

def _resolved(tmp_path, *, n_tokens=80):
    return {
        "calibration": {"calibration": {
            "hf_id": "HuggingFaceFW/fineweb-edu", "n_tokens": n_tokens, "seed": 7,
            "dtype": "float32", "context_length": 16,
        }},
        "model": {"hf_id": "tiny/pythia", "hf_revision": "a" * 40},
        "calibration_root": str(tmp_path),
    }


@pytest.fixture
def counted_collect(monkeypatch):
    calls = []
    real = second_moment.collect_second_moments

    def wrapper(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(second_moment, "collect_second_moments", wrapper)
    return calls


class TestTheDiskCache:
    """Stage B and Stage C run in separate processes; both must see one statistic."""

    def test_collected_once_then_reused(self, tmp_path, tokens, counted_collect):
        resolved = _resolved(tmp_path)
        save_token_cache(cache_path(resolved), tokens, {})
        model = tiny_model("pythia")
        first = load_or_collect_second_moments(model, resolved)
        second = load_or_collect_second_moments(model, resolved)
        assert len(counted_collect) == 1
        assert set(first) == set(second)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_the_file_is_named_by_model_revision_dtype_and_cache(self, tmp_path, tokens):
        resolved = _resolved(tmp_path)
        save_token_cache(cache_path(resolved), tokens, {})
        load_or_collect_second_moments(tiny_model("pythia"), resolved)
        (path,) = (tmp_path / "second_moments").glob("*.npz")
        assert path.name.startswith("tiny_pythia_aaaaaaaa_float32_")
        assert not list((tmp_path / "second_moments").glob("*.partial*"))

    def test_a_statistic_for_a_different_budget_is_not_silently_reused(self, tmp_path, tokens):
        resolved = _resolved(tmp_path, n_tokens=80)
        save_token_cache(cache_path(resolved), tokens, {})
        load_or_collect_second_moments(tiny_model("pythia"), resolved)
        other = _resolved(tmp_path, n_tokens=48)
        # n_tokens is in the token-cache filename, so give the other budget the same tokens
        save_token_cache(cache_path(other), tokens, {})
        fp = second_moment.load_token_cache(cache_path(resolved))[1]["fingerprint"]
        assert second_moments_path(resolved, fp) == second_moments_path(other, fp)
        with pytest.raises(ValueError, match="was collected for"):
            load_or_collect_second_moments(tiny_model("pythia"), other)

    def test_no_token_cache_means_missing_data(self, tmp_path):
        with pytest.raises(CalibrationUnavailable):
            load_or_collect_second_moments(tiny_model("pythia"), _resolved(tmp_path))
