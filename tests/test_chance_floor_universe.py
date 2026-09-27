# [AI-GEN] agent=Claude date=2026-09-27 task=B3 - per-level structural chance-floor universe
# reviewed-by: PENDING

"""The chance-floor universe N (deploy/BLOCKERS.md B3; AI_RULES.md 4.4).

B3 is the blocker that **fails silently**: the wrong N does not crash, it just makes every
overlap look further above chance than it is. Two errors were live:

1. ``candidate_edge_count`` returns ``C*(C-1)`` — all ordered pairs. EAP's universe is
   layer-ordered and type-restricted, so all-pairs is about 10x too large (measured
   10.70x pythia-160m, 10.52x pythia-410m, 11.03x llama-3.2-1b, 9.95x gemma-2-2b).
2. One N was shared by both comparison levels, but ``project_to_routing_heads`` projects
   onto HEADS, not coarse edges. The routing-head universe is 144 for pythia-160m, not
   32,491 — the shared N understated that floor by 2400x-8300x, giving a chance floor of
   ~1e-5 where the honest value is ~0.025. That is the level claim C2 is stated at.

``configs/config.yaml`` pre-registers the intended rule as
``chance_floor_universe: structural_edges_among_observed_nodes``; these tests pin that
the implementation now matches it.
"""

from __future__ import annotations

import pytest

from experiments.run_stage_c import _chance_floor_universes, _structural_edge_universe
from src.science.chance_floor import candidate_edge_count, expected_jaccard_random
from src.synthetic.mock_model import MockModel


def _tl_cfg(n_layers: int = 12, n_heads: int = 12):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=n_layers, d_model=64, n_heads=n_heads, d_head=4, d_mlp=128, d_vocab=40,
        n_ctx=16, act_fn="gelu", normalization_type="LN",
        positional_embedding_type="rotary", rotary_dim=4, parallel_attn_mlp=False,
        dtype=torch.float32, seed=0, device="cpu",
    )
    model = HookedTransformer(cfg)
    model.eval()
    return model


def _observed(model, stride: int = 1) -> dict[str, float]:
    return {e: 1.0 for e in _structural_edge_universe(model)[::stride]}


class TestTheStructuralUniverse:
    def test_a_real_model_uses_the_eap_edge_universe(self):
        model = _tl_cfg()
        from src.extraction.eap import candidate_edges

        assert _structural_edge_universe(model) == candidate_edges(model.cfg)

    def test_it_is_about_ten_times_smaller_than_all_ordered_pairs(self):
        """The heart of B3: all-pairs is not the universe EAP could have returned."""
        model = _tl_cfg()
        universe = _structural_edge_universe(model)
        components = {c for e in universe for c in e.split("->")}
        all_pairs = candidate_edge_count(len(components))
        ratio = all_pairs / len(universe)
        assert 8.0 < ratio < 13.0, f"expected ~10x, got {ratio:.2f}x"

    def test_a_mock_model_uses_all_ordered_pairs_over_its_own_nodes(self):
        """Correct for the mock: MockExtractor really does enumerate every ordered pair."""
        mock = MockModel(seed=0, n_layers=4, n_heads=2, d_model=8)
        universe = _structural_edge_universe(mock)
        nodes = mock.nodes()
        assert len(universe) == len(nodes) * (len(nodes) - 1)
        assert "L0.H0->L0.H0" not in universe, "no self-loops"

    def test_an_undescribable_universe_returns_none_rather_than_a_guess(self):
        """Pipeline A's basis is active transcoder features; candidate_edges does not
        describe it, and inventing a number there is how B3 happened."""
        assert _structural_edge_universe(object()) is None


class TestOneUniversePerLevel:
    def test_the_two_levels_get_different_sizes(self):
        model = _tl_cfg()
        obs = _observed(model, stride=3)
        u = _chance_floor_universes(model, obs, obs, None)
        assert u["exact_edge"] != u["routing_head"]

    def test_the_routing_head_universe_counts_heads_not_edges(self):
        """project_to_routing_heads maps onto L{l}.H{h}, so N is a head count."""
        model = _tl_cfg(n_layers=12, n_heads=12)
        obs = _observed(model)
        u = _chance_floor_universes(model, obs, obs, None)
        assert u["routing_head"] == 144, f"12 layers x 12 heads, got {u['routing_head']}"
        assert u["exact_edge"] > 10_000

    def test_sharing_the_exact_edge_n_would_understate_the_coarse_floor_by_thousands(self):
        """The magnitude of the bug, asserted rather than described."""
        model = _tl_cfg()
        obs = _observed(model)
        u = _chance_floor_universes(model, obs, obs, None)
        k = max(1, u["routing_head"] // 20)
        honest = expected_jaccard_random(k, k, u["routing_head"])
        if_shared = expected_jaccard_random(k, k, u["exact_edge"])
        assert honest / if_shared > 100.0, (
            f"honest floor {honest:.6f} vs shared-N floor {if_shared:.6f}"
        )

    @pytest.mark.parametrize("n_layers,n_heads", [(12, 12), (24, 16), (16, 32), (26, 8)])
    def test_the_head_count_is_right_across_the_grid(self, n_layers, n_heads):
        model = _tl_cfg(n_layers, n_heads)
        obs = _observed(model)
        u = _chance_floor_universes(model, obs, obs, None)
        assert u["routing_head"] == n_layers * n_heads


class TestTheObservedNodeRestrictionSurvives:
    def test_unobserved_nodes_are_excluded(self):
        """The original rationale is kept: N counts only edges whose endpoints the
        extractor actually emitted, so a model-wide count cannot include impossible nodes."""
        model = _tl_cfg()
        full = _chance_floor_universes(model, _observed(model), _observed(model), None)
        narrow_obs = {e: 1.0 for e in _structural_edge_universe(model)[:50]}
        narrow = _chance_floor_universes(model, narrow_obs, narrow_obs, None)
        assert narrow["exact_edge"] < full["exact_edge"]
        assert narrow["routing_head"] <= full["routing_head"]

    def test_both_vectors_contribute_to_the_observed_set(self):
        model = _tl_cfg()
        universe = _structural_edge_universe(model)
        a = {e: 1.0 for e in universe[:200]}
        b = {e: 1.0 for e in universe[200:400]}
        union = _chance_floor_universes(model, a, b, None)
        only_a = _chance_floor_universes(model, a, a, None)
        assert union["exact_edge"] >= only_a["exact_edge"]


class TestItNeverProducesADegenerateFloor:
    def test_neither_level_is_ever_zero(self):
        """N=0 makes expected_jaccard_random raise; N must stay >= 1."""
        model = _tl_cfg()
        u = _chance_floor_universes(model, {}, {}, None)
        assert u["exact_edge"] >= 1
        assert u["routing_head"] >= 1

    def test_an_mlp_only_circuit_does_not_fall_back_to_the_edge_count(self):
        """No heads to project onto. The coarse N must not silently become the exact-edge
        N - that is the 2400x error, and it would not crash."""
        model = _tl_cfg()
        mlp_only = {
            e: 1.0 for e in _structural_edge_universe(model)
            if "MLP" in e and ".H" not in e
        }
        if not mlp_only:
            pytest.skip("no MLP-only edges in this configuration")
        u = _chance_floor_universes(model, mlp_only, mlp_only, None)
        assert u["routing_head"] < u["exact_edge"]


class TestTheMockPathIsUnchanged:
    """Existing engineering artifacts must not shift at the exact-edge level."""

    def test_mock_exact_edge_n_equals_the_old_all_pairs_count(self):
        mock = MockModel(seed=0, n_layers=4, n_heads=2, d_model=8)
        obs = {e: 1.0 for e in _structural_edge_universe(mock)}
        u = _chance_floor_universes(mock, obs, obs, None)
        observed_nodes = {n for e in obs for n in e.split("->", 1)}
        assert u["exact_edge"] == candidate_edge_count(len(observed_nodes))

    def test_mock_routing_head_n_is_the_head_count(self):
        mock = MockModel(seed=0, n_layers=4, n_heads=2, d_model=8)
        obs = {e: 1.0 for e in _structural_edge_universe(mock)}
        u = _chance_floor_universes(mock, obs, obs, None)
        assert u["routing_head"] == 4 * 2
