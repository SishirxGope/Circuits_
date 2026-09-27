# [AI-GEN] agent=Claude date=2026-09-27 task=B2 - real-model magnitude pruning (torch path)
# reviewed-by: PENDING

"""Real-model magnitude pruning (deploy/BLOCKERS.md B2; proposal §3.2).

The torch path must reproduce the numpy rule in this module tensor for tensor, because
that rule was verified against upstream (``saediag.pruning.prune_magnitude_global_inplace``,
2026-08-08) and PRD.md §2 requires our pruning grid to be *the same grid* as ref [1]'s.
If a cell labelled 30% is computed over a different threshold pool than theirs, the C5
cross-audit correlates circuit damage from one intervention against feature damage from
another.

Two upstream asymmetries are therefore load-bearing and each has a test:

* the token embedding joins the threshold pool but is **never zeroed**;
* the keep rule is a strict ``|w| > threshold``, so magnitudes tied at the threshold all
  fall on the drop side.

And one property the paper depends on: **the labelled sparsity is not the achieved
sparsity**, so ``weight_delta`` must measure the real perturbation rather than predict it
from the label.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

from transformer_lens import HookedTransformer, HookedTransformerConfig

from src.compression.magnitude_prune import (
    MagnitudePruner,
    achieved_sparsity_torch,
    prune_magnitude,
    prune_magnitude_torch,
)
from src.compression.torch_weights import projection_parameters


def _model(dtype=torch.float32) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_heads=4, d_head=4, d_mlp=32, d_vocab=40, n_ctx=16,
        act_fn="gelu", normalization_type="LN", positional_embedding_type="rotary",
        rotary_dim=4, dtype=dtype, seed=0, device="cpu",
    )
    model = HookedTransformer(cfg)
    model.eval()
    return model


class TestItPrunesTheRightTensors:
    def test_only_projection_matrices_are_zeroed(self):
        model = _model()
        pruned = prune_magnitude_torch(model, 0.5)
        before, after = dict(model.named_parameters()), dict(pruned.named_parameters())
        changed = {n for n in before if not torch.equal(before[n], after[n])}
        assert changed == set(projection_parameters(model))

    def test_the_embedding_is_pooled_but_never_zeroed(self):
        """Upstream's asymmetry: embed sets the threshold, embed is not pruned."""
        model = _model()
        pruned = prune_magnitude_torch(model, 0.6)
        before, after = dict(model.named_parameters()), dict(pruned.named_parameters())
        assert torch.equal(before["embed.W_E"], after["embed.W_E"])

    def test_including_the_embedding_in_the_pool_changes_the_threshold(self):
        """If it did not, the flag would be decorative and our pool could silently differ
        from ref [1]'s while appearing configured."""
        model = _model()
        with_embed = achieved_sparsity_torch(
            prune_magnitude_torch(model, 0.5, include_embedding_in_threshold=True)
        )["__overall__"]
        without = achieved_sparsity_torch(
            prune_magnitude_torch(model, 0.5, include_embedding_in_threshold=False)
        )["__overall__"]
        assert with_embed != without

    def test_a_missing_embedding_name_is_refused_not_ignored(self):
        model = _model()
        with pytest.raises(ValueError, match="not present in the model"):
            prune_magnitude_torch(model, 0.3, embedding_names=("transformer.wte.weight",))


class TestTheSparsityItActuallyAchieves:
    @pytest.mark.parametrize("target", [0.2, 0.4, 0.6])
    def test_achieved_is_close_to_the_label(self, target):
        pruned = prune_magnitude_torch(_model(), target)
        achieved = achieved_sparsity_torch(pruned)["__overall__"]
        assert achieved == pytest.approx(target, abs=0.05)

    def test_achieved_is_reported_per_tensor_and_overall(self):
        model = _model()
        achieved = achieved_sparsity_torch(prune_magnitude_torch(model, 0.5))
        assert "__overall__" in achieved
        for name in projection_parameters(model):
            assert 0.0 <= achieved[name] <= 1.0

    def test_the_embedding_is_excluded_from_the_denominator(self):
        """Counting it would dilute a 50% prune to a few percent on a real model."""
        achieved = achieved_sparsity_torch(prune_magnitude_torch(_model(), 0.5))
        assert "embed.W_E" not in achieved
        assert "unembed.W_U" not in achieved

    def test_global_scope_does_not_prune_every_tensor_equally(self):
        """One shared threshold means per-matrix rates diverge - which is why the cell
        label must not be described as a per-matrix sparsity."""
        achieved = achieved_sparsity_torch(prune_magnitude_torch(_model(), 0.4))
        rates = [v for k, v in achieved.items() if k != "__overall__"]
        assert len(set(round(r, 3) for r in rates)) > 1

    def test_per_tensor_scope_hits_each_tensor_exactly(self):
        model = _model()
        pruned = prune_magnitude_torch(model, 0.5, global_scope=False)
        achieved = achieved_sparsity_torch(pruned)
        for name in projection_parameters(model):
            assert achieved[name] == pytest.approx(0.5, abs=0.02), name


class TestTheTorchPathMatchesTheVerifiedNumpyRule:
    def test_same_mask_as_prune_magnitude_on_identical_weights(self):
        """The numpy rule is the one verified against upstream, so it is the reference."""
        model = _model()
        names = projection_parameters(model)
        params = dict(model.named_parameters())
        weights = {n: params[n].detach().numpy().astype(np.float64) for n in names}

        np_pruned = prune_magnitude(
            weights, 0.5, global_scope=True, exclude=(), embedding_names=(),
            include_embedding_in_threshold=False,
        )
        torch_pruned = dict(
            prune_magnitude_torch(
                model, 0.5, exclude=(), embedding_names=(),
                include_embedding_in_threshold=False,
            ).named_parameters()
        )
        for name in names:
            np_mask = np_pruned[name] != 0
            torch_mask = (torch_pruned[name].detach().numpy() != 0)
            assert np.array_equal(np_mask, torch_mask), f"masks differ for {name}"

    def test_surviving_weights_keep_their_exact_values(self):
        """Magnitude pruning zeroes; it never rescales what it keeps."""
        model = _model()
        pruned = prune_magnitude_torch(model, 0.5)
        before, after = dict(model.named_parameters()), dict(pruned.named_parameters())
        for name in projection_parameters(model):
            kept = after[name] != 0
            assert torch.equal(after[name][kept], before[name][kept]), name


class TestEdgeCases:
    def test_zero_sparsity_is_a_no_op(self):
        model = _model()
        pruned = prune_magnitude_torch(model, 0.0)
        for name, param in model.named_parameters():
            assert torch.equal(param, dict(pruned.named_parameters())[name])

    def test_full_sparsity_zeroes_every_prunable_tensor(self):
        model = _model()
        pruned = dict(prune_magnitude_torch(model, 1.0).named_parameters())
        for name in projection_parameters(model):
            assert torch.count_nonzero(pruned[name]) == 0
        assert torch.equal(
            dict(model.named_parameters())["embed.W_E"], pruned["embed.W_E"]
        )

    def test_an_out_of_range_sparsity_is_refused(self):
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError, match="target_sparsity must be in"):
                prune_magnitude_torch(_model(), bad)

    def test_a_dead_model_fails_loudly_rather_than_silently(self):
        """All-equal magnitudes: a strict `>` rule drops everything. That is upstream's
        rule, but a dead model's CSI is meaningless, so it must raise."""
        model = _model()
        with torch.no_grad():
            for name in projection_parameters(model):
                dict(model.named_parameters())[name].fill_(1.0)
            dict(model.named_parameters())["embed.W_E"].fill_(1.0)
        with pytest.raises(ValueError, match="zeroed EVERY prunable weight"):
            prune_magnitude_torch(model, 0.5)


class TestTheCompressorContract:
    def test_apply_does_not_mutate_the_dense_model(self):
        model = _model()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        MagnitudePruner(sparsity=0.5).apply(model, None)
        for name, param in model.named_parameters():
            assert torch.equal(before[name], param), f"apply() mutated {name}"

    def test_weight_delta_is_measured_not_predicted(self):
        model = _model()
        pruner = MagnitudePruner(sparsity=0.4)
        delta = pruner.weight_delta(model, None)
        pruned = dict(pruner.apply(model, None).named_parameters())
        dense = dict(model.named_parameters())

        assert set(delta) == set(dense), "every tensor must be reported, zeros included"
        for name, value in delta.items():
            measured = torch.linalg.vector_norm(
                pruned[name].to(torch.float32) - dense[name].to(torch.float32)
            ).item()
            assert value == pytest.approx(measured, rel=1e-6, abs=1e-9), name

    def test_untouched_tensors_report_zero(self):
        model = _model()
        delta = MagnitudePruner(sparsity=0.5).weight_delta(model, None)
        assert delta["embed.W_E"] == 0.0

    def test_heavier_pruning_means_a_larger_delta(self):
        model = _model()
        light = sum(MagnitudePruner(sparsity=0.2).weight_delta(model, None).values())
        heavy = sum(MagnitudePruner(sparsity=0.6).weight_delta(model, None).values())
        assert heavy > light

    def test_it_still_refuses_what_it_cannot_handle(self):
        for method in ("apply", "weight_delta"):
            with pytest.raises(NotImplementedError, match="supports MockModel"):
                getattr(MagnitudePruner(), method)(object(), {})

    def test_the_pruned_model_still_runs(self):
        pruned = MagnitudePruner(sparsity=0.5).apply(_model(), None)
        logits = pruned(torch.zeros(1, 4, dtype=torch.long))
        assert torch.isfinite(logits).all()


class TestItComposesWithTheNull:
    def test_the_null_can_be_matched_to_a_pruning_cell(self):
        """End to end: the magnitudes a pruning cell measures must seed a matched null."""
        from src.common.seeding import create_seed_generator
        from src.science.matched_magnitude import (
            MatchedMagnitudePerturber,
            frobenius_norm,
        )

        model = _model()
        magnitudes = MagnitudePruner(sparsity=0.4).weight_delta(model, None)
        rng, _ = create_seed_generator(5)
        perturbed = MatchedMagnitudePerturber().apply(model, magnitudes, rng)

        before, after = dict(model.named_parameters()), dict(perturbed.named_parameters())
        checked = 0
        for name, target in magnitudes.items():
            if target == 0.0:
                continue
            realized = frobenius_norm((after[name] - before[name]).detach().numpy())
            assert realized == pytest.approx(target, rel=1e-5), name
            checked += 1
        assert checked > 0
