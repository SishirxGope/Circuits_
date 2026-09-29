# [AI-GEN] agent=Claude date=2026-09-29 task=Tests for real-model GPTQ and AWQ (torch ports of the reference code)
# modified: [AI-GEN] agent=Claude date=2026-09-29 task=review fixes - AWQ feature memory, cache identity and lock, delta without a copy
# reviewed-by: PENDING

"""Real-model GPTQ and AWQ (src/compression/{layerwise,gptq,awq}.py).

The central claim is that each port computes what the authors' code computes. So the
cores are run side by side with **verbatim excerpts of the reference implementations**
(tests/reference_gptq.py, tests/reference_awq.py, pinned commits, licences shipped) and
must agree. The rest checks the TransformerLens plumbing the references do not have: the
block walk, the matrix views, which activations feed which projection, GPTQ's sequential
propagation, AWQ's per-architecture rules, and the on-disk result both stages share.
"""

from __future__ import annotations

import copy
import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

import reference_awq
import reference_gptq
from tiny_tl import ARCHS, tiny_model

from src.calibration.cache_lock import cache_lock, partial_path
from src.calibration.second_moment import collect_second_moments
from src.calibration.token_cache import (
    CalibrationUnavailable,
    cache_path,
    save_token_cache,
    synthetic_token_cache,
)
from src.compression import awq as awq_module
from src.compression import gptq as gptq_module
from src.compression import layerwise as layerwise_module
from src.compression.awq import (
    AwqCompressor,
    awq_clip_max,
    awq_pseudo_quantize,
    awq_quantize_model,
)
from src.compression.gptq import (
    GptqCompressor,
    _quantize,
    _row_grid,
    gptq_quantize_matrix,
    gptq_quantize_model,
)
from src.compression.layerwise import (
    as_matrix,
    block_source,
    code_digest,
    embed_batches,
    from_matrix,
    measured_delta,
    projection_kind,
    projections_by_layer,
    run_block,
    tensors_delta,
    with_tensors,
)
from src.compression.torch_weights import null_draw_inputs, projection_parameters

VOCAB = 50


@pytest.fixture(scope="module", params=sorted(ARCHS))
def model(request):
    return tiny_model(request.param)


@pytest.fixture(scope="module")
def tokens():
    return synthetic_token_cache(6, 16, VOCAB, seed=3)


def _awq(**kwargs):
    kwargs.setdefault("group_size", 8)  # the tiny models have 16 input channels
    return AwqCompressor(**kwargs)


def _correlated(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) @ torch.randn(d, d, generator=g)


# ================================================================ the block walk

class TestMatrixViews:
    def test_round_trip_for_every_projection(self, model):
        for name, param in model.named_parameters():
            if name not in projection_parameters(model):
                continue
            kind = projection_kind(name)
            assert torch.equal(from_matrix(kind, as_matrix(kind, param), tuple(param.shape)), param)

    def test_the_view_is_the_linear_map_the_model_applies(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(3, 5, 16, generator=g)
        w_q = torch.randn(4, 16, 4, generator=g)
        expect = torch.einsum("bpd,hdk->bphk", x, w_q).flatten(-2)
        assert torch.allclose(x @ as_matrix("W_Q", w_q).t(), expect, atol=1e-5)
        z = torch.randn(3, 5, 4, 4, generator=g)
        w_o = torch.randn(4, 4, 16, generator=g)
        expect = torch.einsum("bphk,hkd->bpd", z, w_o)
        assert torch.allclose(z.flatten(-2) @ as_matrix("W_O", w_o).t(), expect, atol=1e-5)
        w_in = torch.randn(16, 32, generator=g)
        assert torch.allclose(x @ as_matrix("W_in", w_in).t(), x @ w_in, atol=1e-5)


class TestTheWalk:
    def test_embed_then_blocks_reproduces_the_model_exactly(self, model, tokens):
        batches = embed_batches(model, tokens, n_seq=6, batch_size=4, dtype=None)
        for block in model.blocks:
            batches = run_block(block, batches)
        walked = torch.cat([resid for resid, _ in batches])
        with torch.no_grad():
            direct = model(torch.as_tensor(tokens), stop_at_layer=model.cfg.n_layers)
        assert torch.equal(walked, direct)

    def test_the_calibration_dtype_leaves_the_model_alone(self, model, tokens):
        before = {k: v.clone() for k, v in model.state_dict().items()}
        batches = embed_batches(model, tokens, n_seq=2, batch_size=2, dtype="bfloat16")
        assert batches[0][0].dtype == torch.bfloat16
        assert next(model.parameters()).dtype == torch.float32
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[key]), key

    def test_projection_inputs_are_the_ones_wanda_reads(self, model, tokens):
        """Same activations as the collector already verified against the model."""
        batches = embed_batches(model, tokens, n_seq=6, batch_size=4, dtype=None)
        sums, counts = {}, {}

        def record(source, x):
            x2 = x.reshape(-1, x.shape[-1]).double()
            sums[source] = sums.get(source, 0) + (x2 * x2).sum(0)
            counts[source] = counts.get(source, 0) + x2.shape[0]

        for block in model.blocks[:1]:
            run_block(block, batches, on_input=record)
        wanda = collect_second_moments(model, tokens, batch_size=4)
        for name in projection_parameters(model):
            if not name.startswith("blocks.0."):
                continue
            source = block_source(projection_kind(name))
            mine = (sums[source] / counts[source]).cpu().numpy()  # TL puts models on cuda:0 when it can
            assert np.allclose(mine, wanda[name].reshape(-1), rtol=1e-6), name

    def test_no_hook_is_left_behind(self, model, tokens):
        batches = embed_batches(model, tokens, n_seq=2, batch_size=2, dtype=None)
        run_block(model.blocks[0], batches, on_input=lambda s, x: None, layout={})
        for module in model.modules():
            assert not module._forward_pre_hooks and not module._forward_hooks


# ================================================================ GPTQ core vs reference

def _reference_gptq(w, x, *, bits, sym, groupsize):
    layer = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
    layer.weight.data = w.clone()
    engine = reference_gptq.GPTQ(layer)
    engine.quantizer = reference_gptq.Quantizer()
    engine.quantizer.configure(bits, perchannel=True, sym=sym, mse=False)
    engine.add_batch(x.unsqueeze(0), None)
    hessian = engine.H.clone()
    engine.fasterquant(groupsize=groupsize)
    return layer.weight.data.clone(), hessian


class TestGptqAgainstTheReference:
    @pytest.mark.parametrize("sym", [False, True])
    @pytest.mark.parametrize("groupsize", [-1, 8])
    def test_identical_to_fasterquant(self, sym, groupsize):
        w = torch.randn(24, 40, generator=torch.Generator().manual_seed(1))
        x = _correlated(300, 40, seed=2)
        expect, hessian = _reference_gptq(w, x, bits=4, sym=sym, groupsize=groupsize)
        got = gptq_quantize_matrix(w, hessian, bits=4, sym=sym, group_size=groupsize)
        assert torch.allclose(got, expect, atol=1e-6), (got - expect).abs().max()

    def test_the_comparison_is_not_vacuous(self):
        """The reference output differs from plain rounding, so agreeing with it means something."""
        w = torch.randn(24, 40, generator=torch.Generator().manual_seed(1))
        expect, _ = _reference_gptq(w, _correlated(300, 40, seed=2), bits=4, sym=False, groupsize=-1)
        scale, zero, maxq = _row_grid(w, 4, False)
        assert not torch.allclose(expect, _quantize(w, scale, zero, maxq), atol=1e-3)

    def test_our_hessian_scaling_does_not_matter(self):
        w = torch.randn(24, 40, generator=torch.Generator().manual_seed(1))
        x = _correlated(300, 40, seed=2)
        a = gptq_quantize_matrix(w, 2 * x.t() @ x / 300)
        b = gptq_quantize_matrix(w, 17.0 * x.t() @ x)
        assert torch.allclose(a, b, atol=1e-5)


class TestGptqProperties:
    def test_an_identity_hessian_is_plain_rounding_on_the_grid(self):
        w = torch.randn(24, 40, generator=torch.Generator().manual_seed(4))
        scale, zero, maxq = _row_grid(w, 4, False)
        assert torch.equal(gptq_quantize_matrix(w, torch.eye(40)), _quantize(w, scale, zero, maxq))

    @pytest.mark.parametrize("blocksize", [1, 7, 128])
    def test_the_block_size_is_only_an_optimisation(self, blocksize):
        w = torch.randn(24, 40, generator=torch.Generator().manual_seed(5))
        x = _correlated(300, 40, seed=6)
        h = 2 * x.t() @ x / 300
        assert torch.allclose(gptq_quantize_matrix(w, h, blocksize=blocksize), gptq_quantize_matrix(w, h), atol=1e-5)

    def test_it_beats_rounding_on_the_data_it_saw(self):
        w = torch.randn(24, 40, generator=torch.Generator().manual_seed(7))
        x = _correlated(300, 40, seed=8)
        q = gptq_quantize_matrix(w, 2 * x.t() @ x / 300)
        scale, zero, maxq = _row_grid(w, 4, False)
        rtn = _quantize(w, scale, zero, maxq)
        assert ((x @ (q - w).t()) ** 2).sum() < 0.8 * ((x @ (rtn - w).t()) ** 2).sum()

    def test_every_row_stays_on_its_4_bit_grid(self):
        w = torch.randn(24, 40, generator=torch.Generator().manual_seed(9))
        q = gptq_quantize_matrix(w, 2 * _correlated(300, 40, 10).t() @ _correlated(300, 40, 10))
        assert all(len(torch.unique(row)) <= 16 for row in q)

    def test_a_dead_input_zeroes_its_column(self):
        w = torch.randn(8, 12, generator=torch.Generator().manual_seed(11))
        x = _correlated(100, 12, seed=12)
        x[:, 3] = 0
        assert torch.all(gptq_quantize_matrix(w, x.t() @ x)[:, 3] == 0)

    def test_a_hessian_of_the_wrong_size_is_refused(self):
        with pytest.raises(ValueError, match="Hessian"):
            gptq_quantize_matrix(torch.randn(4, 6), torch.eye(5))


class TestGptqOnAModel:
    def test_only_projections_change_and_the_model_is_untouched(self, model, tokens):
        before = {k: v.clone() for k, v in model.state_dict().items()}
        out = GptqCompressor(calibration=tokens).apply(model, None)
        dense = dict(model.named_parameters())
        moved = {
            name for name, param in out.named_parameters()
            if not torch.equal(param, dense[name])
        }
        assert moved == set(projection_parameters(model))
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[key]), key

    def test_layer_one_is_calibrated_on_quantized_layer_zero(self, tokens):
        """llama.py feeds each layer the output of the layers already quantized."""
        model = tiny_model("llama")
        result = gptq_quantize_model(
            model, tokens, n_seq=6, bits=4, sym=False, group_size=-1, percdamp=0.01,
            blocksize=128, batch_size=4, dtype=None,
        )
        layer0 = copy.deepcopy(model.blocks[0])
        params0 = dict(layer0.named_parameters())
        with torch.no_grad():
            for name, value in result.items():
                if name.startswith("blocks.0."):
                    params0[name.split(".", 2)[2]].copy_(value)

        def expected_w_in(first_block):
            batches = run_block(first_block, embed_batches(model, tokens, n_seq=6, batch_size=4, dtype=None))
            xs = []
            run_block(model.blocks[1], batches, on_input=lambda s, x: xs.append(x) if s == "mlp:input" else None)
            x2 = torch.cat([x.reshape(-1, x.shape[-1]) for x in xs])
            w = as_matrix("W_in", model.blocks[1].mlp.W_in.detach())
            return from_matrix("W_in", gptq_quantize_matrix(w, 2 * x2.t() @ x2 / x2.shape[0]), (16, 32))

        got = result["blocks.1.mlp.W_in"]
        assert torch.allclose(got, expected_w_in(layer0), atol=1e-6)
        assert not torch.allclose(got, expected_w_in(model.blocks[0]), atol=1e-6)

    def test_weight_delta_is_measured_and_feeds_the_null(self, model, tokens):
        compressor = GptqCompressor(calibration=tokens)
        delta = compressor.weight_delta(model, None)
        out = dict(compressor.apply(model, None).named_parameters())
        dense = dict(model.named_parameters())
        for name in projection_parameters(model):
            measured = float(torch.linalg.vector_norm(out[name].detach() - dense[name].detach()))
            assert delta[name] == pytest.approx(measured, rel=1e-6) and delta[name] > 0
        magnitudes, _ = null_draw_inputs(model, delta)
        assert set(magnitudes) == set(projection_parameters(model))


# ================================================================ AWQ cores vs reference

class TestAwqAgainstTheReference:
    @pytest.mark.parametrize("group", [-1, 8])
    def test_pseudo_quantize_is_the_reference(self, group):
        w = torch.randn(24, 32, generator=torch.Generator().manual_seed(13))
        expect = reference_awq.pseudo_quantize_tensor(w.clone(), n_bit=4, zero_point=True, q_group_size=group)
        assert torch.equal(awq_pseudo_quantize(w, 4, group), expect)

    def test_clip_search_is_the_reference(self):
        w = torch.randn(128, 32, generator=torch.Generator().manual_seed(14))
        x = _correlated(2048, 32, seed=15)
        expect = reference_awq.auto_clip_layer(
            w.clone(), x.clone(), n_bit=4, q_config={"zero_point": True, "q_group_size": 8}
        )
        got = awq_clip_max(w, x, bits=4, group_size=8)
        assert torch.equal(got, expect)
        assert not torch.equal(got, w.reshape(128, 4, 8).abs().amax(-1, keepdim=True)), "vacuous: nothing clipped"

    def test_the_down_projection_scale_is_the_references(self, model, tokens):
        """_search_module_scale on an nn.Linear, fed the same activations, picks the same s."""
        record = {}
        awq_quantize_model(
            model, tokens, n_seq=6, bits=4, group_size=8, n_grid=20, clip=True, clip_n_grid=20,
            clip_max_shrink=0.5, clip_n_sample_token=512, batch_size=4, dtype=None, record=record,
        )
        search = reference_awq.make_search(4, {"zero_point": True, "q_group_size": 8})
        batches = embed_batches(model, tokens, n_seq=6, batch_size=4, dtype=None)
        for index, block in enumerate(model.blocks):
            posts = []
            next_batches = run_block(
                block, batches, on_input=lambda s, x, posts=posts: posts.append(x) if s == "mlp.hook_post" else None
            )
            x = torch.cat([p.reshape(-1, p.shape[-1]) for p in posts])
            fc = torch.nn.Linear(x.shape[1], model.cfg.d_model)
            fc.weight.data = as_matrix("W_out", block.mlp.W_out.detach()).clone()
            fc.bias.data = block.mlp.b_out.detach().clone()
            with torch.no_grad():  # auto_scale_block is @torch.no_grad(); the closure relies on it
                expect = search(fc, [fc], x)
            assert torch.allclose(record[index]["down"], expect, rtol=1e-5), index
            batches = next_batches
        assert any(not torch.allclose(r["down"], torch.ones_like(r["down"])) for r in record.values()), (
            "vacuous: every searched scale was 1"
        )


class TestAwqPlumbing:
    @pytest.mark.parametrize("n_sample", [1, 5, 20, 84, 512])
    def test_the_clip_sample_is_the_references_without_concatenating(self, n_sample):
        """_LayerFeatures.sample must keep exactly auto_clip_layer's rows: every
        (T // n)-th token of all T, counted across batch boundaries."""
        from src.compression.awq import _LayerFeatures

        feats = _LayerFeatures({"src"})
        g = torch.Generator().manual_seed(0)
        xs = [torch.randn(3, 7, 5, generator=g) for _ in range(4)]  # 84 tokens, batches of 21
        for x in xs:
            feats("src", x)
        full = torch.cat([x.reshape(-1, 5) for x in xs])
        assert torch.equal(feats.sample("src", n_sample), full[0::max(1, 84 // n_sample)])

    def test_the_down_projection_end_to_end(self, model, tokens):
        """W_out = Q(clip(W * s)) / s, clipped against the SCALED inputs x / s.

        apply_scale divides the cached input features by s before auto_clip_block runs;
        clipping against the unscaled features would pick bounds for a different product.
        """
        record = {}
        result = awq_quantize_model(
            model, tokens, n_seq=6, bits=4, group_size=8, n_grid=20, clip=True, clip_n_grid=20,
            clip_max_shrink=0.5, clip_n_sample_token=512, batch_size=4, dtype=None, record=record,
        )
        batches = embed_batches(model, tokens, n_seq=6, batch_size=4, dtype=None)
        for index, block in enumerate(model.blocks):
            posts = []
            next_batches = run_block(
                block, batches, on_input=lambda s, x, posts=posts: posts.append(x) if s == "mlp.hook_post" else None
            )
            x = torch.cat([p.reshape(-1, p.shape[-1]) for p in posts])
            scale = record[index]["down"].view(1, -1)
            w = as_matrix("W_out", block.mlp.W_out.detach()) * scale
            bound = awq_clip_max(w, x / scale, bits=4, group_size=8)
            assert torch.equal(record[index]["clip"]["W_out"], bound), index
            rows, cols = w.shape
            clipped = torch.clamp(w.reshape(rows, bound.shape[1], -1), -bound, bound).reshape(rows, cols)
            expect = from_matrix("W_out", awq_pseudo_quantize(clipped, 4, 8) / scale, tuple(block.mlp.W_out.shape))
            assert torch.allclose(result[f"blocks.{index}.mlp.W_out"], expect, atol=1e-6), index
            batches = next_batches

    def test_no_scaling_and_no_clipping_is_plain_group_quantization(self, model, tokens):
        """n_grid=1 searches only ratio 0 (s = 1); then AWQ is pseudo_quantize_tensor."""
        out = dict(_awq(calibration=tokens, n_grid=1, clip=False).apply(model, None).named_parameters())
        for name, param in model.named_parameters():
            if name in projection_parameters(model):
                kind = projection_kind(name)
                plain = from_matrix(kind, awq_pseudo_quantize(as_matrix(kind, param.detach()), 4, 8), tuple(param.shape))
                assert torch.allclose(out[name], plain, atol=1e-6), name

    def test_the_scaling_is_undone_exactly(self, model, tokens, monkeypatch):
        """With rounding stubbed out, every s (inputs) and 1/s (outputs) must cancel.

        Fixed, far-from-1 scales make each group's s_in / s_out pairing visible: a scale
        applied to the wrong axis, or undone on the wrong tensor, leaves a residue.
        """
        monkeypatch.setattr(awq_module, "awq_pseudo_quantize", lambda w, bits, g: w.clone())
        monkeypatch.setattr(
            awq_module, "_normalised_scales",
            lambda x, ratio: torch.linspace(0.25, 4.0, x.numel(), dtype=x.dtype, device=x.device),
        )
        out = dict(_awq(calibration=tokens).apply(model, None).named_parameters())
        dense = dict(model.named_parameters())
        for name in projection_parameters(model):
            assert torch.allclose(out[name], dense[name].detach(), rtol=1e-5, atol=1e-6), name

    def test_q_and_k_are_never_clipped(self, model, tokens):
        """With s = 1 (n_grid=1) an unclipped tensor is plain group quantization."""
        out = dict(_awq(calibration=tokens, n_grid=1).apply(model, None).named_parameters())
        params = dict(model.named_parameters())
        checked = 0
        for name in projection_parameters(model):
            kind = projection_kind(name)
            if kind not in ("W_Q", "W_K"):
                continue
            dense = params[name].detach()
            plain = from_matrix(kind, awq_pseudo_quantize(as_matrix(kind, dense), 4, 8), tuple(dense.shape))
            assert torch.allclose(out[name], plain, atol=1e-6), name
            checked += 1
        assert checked == 2 * model.cfg.n_layers

    def test_pythia_leaves_v_unclipped_like_the_fused_reference(self, tokens):
        model = tiny_model("pythia")
        out = dict(_awq(calibration=tokens, n_grid=1).apply(model, None).named_parameters())
        w_v = model.blocks[0].attn.W_V.detach()
        plain = from_matrix("W_V", awq_pseudo_quantize(as_matrix("W_V", w_v), 4, 8), tuple(w_v.shape))
        assert torch.allclose(out["blocks.0.attn.W_V"], plain, atol=1e-6)

    @pytest.mark.parametrize("arch,expect_o", [("pythia", False), ("llama", False), ("llama_mha", True)])
    def test_the_v_to_o_scale_only_where_the_reference_searches_it(self, arch, expect_o, tokens):
        if arch == "llama_mha":
            model = _mha_llama()
        else:
            model = tiny_model(arch)
        record = {}
        awq_quantize_model(
            model, tokens, n_seq=6, bits=4, group_size=8, n_grid=20, clip=False, clip_n_grid=20,
            clip_max_shrink=0.5, clip_n_sample_token=512, batch_size=4, dtype=None, record=record,
        )
        assert all(("o" in layer) == expect_o for layer in record.values())

    def test_only_projections_change_and_the_model_is_untouched(self, model, tokens):
        before = {k: v.clone() for k, v in model.state_dict().items()}
        out = _awq(calibration=tokens).apply(model, None)
        dense = dict(model.named_parameters())
        moved = {name for name, p in out.named_parameters() if not torch.equal(p, dense[name])}
        assert moved == set(projection_parameters(model))
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[key]), key

    def test_an_unknown_architecture_is_refused(self, tokens):
        model = tiny_model("llama")
        model.cfg.original_architecture = "MistralForCausalLM"
        with pytest.raises(ValueError, match="no scaling rules"):
            _awq(calibration=tokens).apply(model, None)
        model.cfg.original_architecture = "LlamaForCausalLM"

    def test_a_group_that_does_not_divide_the_inputs_is_refused(self, tokens):
        with pytest.raises(ValueError, match="divisible"):
            AwqCompressor(calibration=tokens, group_size=128).apply(tiny_model("llama"), None)

    def test_the_bf16_calibration_forward_runs(self, tokens):
        resolved = {"calibration": {"calibration": {
            "hf_id": "HuggingFaceFW/fineweb-edu", "seed": 7, "dtype": "bfloat16", "n_tokens": 96,
        }}}
        out = _awq(calibration=tokens).apply(tiny_model("gemma2"), resolved)
        assert next(out.parameters()).dtype == torch.float32


class TestAwqKeepsOnlyWhatItReads:
    """Under split-qkv the attention inputs arrive as one slice of a [b, p, heads, d] tensor;
    keeping the slice kept every head's copy (32x on Llama-3.2-1B). Now: clones, and only
    the sources a search or a clip reads."""

    def test_each_kept_input_owns_only_its_own_storage(self, model, tokens):
        rules = awq_module.arch_rules(model.cfg)
        kinds = projections_by_layer(model)[0]
        feats = awq_module._LayerFeatures(awq_module.awq_sources(model.cfg, kinds, rules, clip=True))
        layout = {}
        batches = embed_batches(model, tokens, n_seq=6, batch_size=4, dtype=None)
        run_block(model.blocks[0], batches, on_input=feats, layout=layout)
        assert any(heads is not None for heads in layout.values())  # the case that leaked
        assert set(feats.batches) == feats.wanted
        for source, xs in feats.batches.items():
            for x in xs:
                assert x.untyped_storage().nbytes() == x.numel() * x.element_size(), source

    @pytest.mark.parametrize("clip", [True, False])
    def test_the_key_input_is_never_kept(self, model, clip):
        rules = awq_module.arch_rules(model.cfg)
        kinds = projections_by_layer(model)[0]
        wanted = awq_module.awq_sources(model.cfg, kinds, rules, clip=clip)
        assert block_source("W_K") not in wanted
        base = {block_source("W_Q"), block_source("W_in"), block_source("W_out")}
        assert base <= wanted
        if not clip:
            extra = {block_source("W_O")} if awq_module._searches_v_to_o(model.cfg, rules) else set()
            assert wanted == base | extra


def _mha_llama():
    import transformer_lens

    from src.extraction.real_model import enable_extraction_hooks

    cfg = dict(ARCHS["llama"])
    cfg["n_key_value_heads"] = 4  # MHA: v_proj and o_proj both 16 x 16
    from tiny_tl import BASE

    torch.manual_seed(0)
    model = transformer_lens.HookedTransformer(transformer_lens.HookedTransformerConfig(**BASE, **cfg)).eval()
    return enable_extraction_hooks(model)


# ================================================================ both compressors

@pytest.mark.parametrize("make", [GptqCompressor, _awq], ids=["gptq", "awq"])
class TestTheCompressorContract:
    def test_the_mock_path_is_unchanged(self, make):
        from src.synthetic.mock_model import MockModel

        mock = MockModel(seed=0, n_layers=2, n_heads=2, d_model=8)
        assert make().apply(mock, None).weights.keys() == mock.weights.keys()

    def test_no_tokens_and_no_config_is_missing_data(self, make):
        with pytest.raises(CalibrationUnavailable):
            make().apply(tiny_model("llama"), None)

    def test_a_module_that_is_not_transformerlens_is_refused(self, make, tokens):
        with pytest.raises(TypeError, match="HookedTransformer"):
            make(calibration=tokens).apply(torch.nn.Linear(4, 4), None)

    def test_bits_arrive_as_strings_from_hydra(self, make):
        assert make(bits="4").bits == 4


class TestStageBAndStageCShareOneCompression:
    """weight_delta (Stage B) and apply (Stage C) read one file, so the null is matched to
    exactly the compression Stage C applies."""

    def _resolved(self, tmp_path):
        return {
            "calibration": {"calibration": {
                "hf_id": "HuggingFaceFW/fineweb-edu", "n_tokens": 64, "seed": 7,
                "dtype": "bfloat16", "context_length": 16,
            }},
            "model": {"hf_id": "tiny/llama", "hf_revision": "c" * 40},
            "calibration_root": str(tmp_path),
        }

    @pytest.mark.parametrize("module,attr,make", [
        (gptq_module, "gptq_quantize_model", GptqCompressor),
        (awq_module, "awq_quantize_model", _awq),
    ], ids=["gptq", "awq"])
    def test_one_computation_serves_both_stages(self, tmp_path, monkeypatch, module, attr, make):
        model = tiny_model("llama")
        resolved = self._resolved(tmp_path)
        save_token_cache(cache_path(resolved), synthetic_token_cache(4, 16, VOCAB, seed=2), {})
        calls = []
        real = getattr(module, attr)
        monkeypatch.setattr(module, attr, lambda *a, **k: calls.append(k) or real(*a, **k))
        stage_c = dict(make().apply(model, resolved).named_parameters())
        delta = make().weight_delta(model, resolved)
        again = dict(make().apply(model, resolved).named_parameters())
        assert len(calls) == 1
        assert calls[0]["dtype"] == "bfloat16" and calls[0]["n_seq"] == 4
        for name, value in stage_c.items():
            assert torch.equal(value, again[name]), name
        dense = dict(model.named_parameters())
        for name in projection_parameters(model):
            measured = float(torch.linalg.vector_norm(stage_c[name].detach() - dense[name].detach()))
            assert delta[name] == pytest.approx(measured, rel=1e-6), name

    def test_different_settings_are_a_different_file(self, tmp_path):
        model = tiny_model("llama")
        resolved = self._resolved(tmp_path)
        save_token_cache(cache_path(resolved), synthetic_token_cache(4, 16, VOCAB, seed=2), {})
        a = dict(GptqCompressor().apply(model, resolved).named_parameters())
        b = dict(GptqCompressor(percdamp=0.5).apply(model, resolved).named_parameters())
        assert len(list((tmp_path / "compressed").glob("gptq_*.pt"))) == 2
        assert any(not torch.equal(a[n], b[n]) for n in projection_parameters(model))

    def test_without_the_token_cache_it_is_missing_data(self, tmp_path):
        with pytest.raises(CalibrationUnavailable):
            GptqCompressor().apply(tiny_model("llama"), self._resolved(tmp_path))

    def test_the_model_dtype_is_part_of_the_identity(self, tmp_path):
        model = tiny_model("llama")
        resolved = self._resolved(tmp_path)
        save_token_cache(cache_path(resolved), synthetic_token_cache(4, 16, VOCAB, seed=2), {})
        GptqCompressor().apply(model, resolved)
        GptqCompressor().apply(copy.deepcopy(model).to(torch.bfloat16), resolved)
        assert len(list((tmp_path / "compressed").glob("gptq_*.pt"))) == 2

    def test_a_code_change_is_a_different_file(self, tmp_path, monkeypatch):
        model = tiny_model("llama")
        resolved = self._resolved(tmp_path)
        save_token_cache(cache_path(resolved), synthetic_token_cache(4, 16, VOCAB, seed=2), {})
        GptqCompressor().apply(model, resolved)
        monkeypatch.setattr(layerwise_module, "code_digest", lambda method: "0" * 16)
        GptqCompressor().apply(model, resolved)
        assert len(list((tmp_path / "compressed").glob("gptq_*.pt"))) == 2

    def test_the_code_digest_is_per_method_and_stable(self):
        assert code_digest("gptq") == code_digest("gptq") != code_digest("awq")

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        model = tiny_model("llama")
        resolved = self._resolved(tmp_path)
        save_token_cache(cache_path(resolved), synthetic_token_cache(4, 16, VOCAB, seed=2), {})
        GptqCompressor().apply(model, resolved)
        assert not list((tmp_path / "compressed").glob("*.partial*"))

    def test_a_second_process_waits_for_the_first(self, tmp_path):
        path, order = tmp_path / "x.pt", []

        def waiter():
            with cache_lock(path):
                order.append("waiter")

        with cache_lock(path):
            thread = threading.Thread(target=waiter)
            thread.start()
            thread.join(timeout=0.5)
            assert thread.is_alive()  # blocked while the lock is held
            order.append("holder")
        thread.join(timeout=30)
        assert order == ["holder", "waiter"]

    def test_temporary_names_never_collide(self, tmp_path):
        assert partial_path(tmp_path / "x.pt", ".pt") != partial_path(tmp_path / "x.pt", ".pt")


class TestTheDeltaNeedsNoModelCopy:
    def test_it_equals_the_delta_of_the_compressed_model(self, model, tokens):
        tensors = GptqCompressor(calibration=tokens)._quantized(model, None)
        assert tensors_delta(model, tensors) == measured_delta(model, with_tensors(model, tensors))

    def test_unknown_or_misshapen_tensors_are_refused(self, model):
        name, param = next(iter(model.named_parameters()))
        with pytest.raises(ValueError, match="lacks"):
            tensors_delta(model, {"blocks.99.nope": param})
        with pytest.raises(ValueError, match="shape"):
            tensors_delta(model, {name: param.reshape(-1)[:1]})


class TestTheGpu:
    """The Spark runs these on cuda:0; the ports must not assume CPU tensors.

    Deliberately NOT a CPU-vs-GPU equality test: different kernels round differently, and
    a near-tie in AWQ's scale search may then pick a neighbouring ratio. That is why each
    cell's result is cached on disk and shared by Stage B and Stage C."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.parametrize("make", [GptqCompressor, _awq], ids=["gptq", "awq"])
    @pytest.mark.parametrize("dtype", [None, "bfloat16"])
    def test_runs_on_the_gpu(self, make, dtype, tokens):
        model = tiny_model("gemma2").to("cuda")
        resolved = None if dtype is None else {"calibration": {"calibration": {
            "hf_id": "HuggingFaceFW/fineweb-edu", "seed": 7, "dtype": dtype, "n_tokens": 96,
        }}}
        out = dict(make(calibration=tokens).apply(model, resolved).named_parameters())
        dense = dict(model.named_parameters())
        moved = {name for name, p in out.items() if not torch.equal(p, dense[name])}
        assert moved == set(projection_parameters(model))
        for name in moved:
            assert out[name].device.type == "cuda" and torch.isfinite(out[name]).all(), name
