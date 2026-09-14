# [AI-GEN] agent=Claude date=2026-08-08 task=Tests for the chance floor (AI_RULES 4.4), FDR correction (4.3) and threshold sweep (C3)
# reviewed-by: PENDING

"""Floors and corrections — the machinery that stops a number from being over-read.

AI_RULES.md 4.4: "Numbers without floors are meaningless in this project by
construction." AI_RULES.md 4.3: any per-cell significance statement uses a stated
correction. PRD.md §1 C3: a headline conclusion must survive the full threshold sweep
or be flagged threshold-dependent.
"""

import pytest

from analysis.threshold_sweep import conclusion_survives_sweep, sweep_report, verdict_for
from src.science.chance_floor import (
    candidate_edge_count,
    expected_jaccard_random,
    overlap_vs_chance,
    overlap_vs_chance_for_freqs,
    random_topk_null,
    summarize_floor,
)
from src.science.multiple_comparisons import (
    benjamini_hochberg,
    correct_cell_grid,
    expected_false_positives,
)


class TestCandidateUniverse:
    def test_directed_without_self_loops(self):
        assert candidate_edge_count(5) == 20  # 5 * 4

    def test_self_loops_included(self):
        assert candidate_edge_count(5, self_loops=True) == 25

    def test_undirected_halves_it(self):
        assert candidate_edge_count(5, directed=False) == 10

    def test_degenerate_sizes(self):
        assert candidate_edge_count(0) == 0
        assert candidate_edge_count(1) == 0

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            candidate_edge_count(-1)


class TestExpectedJaccard:
    def test_tiny_sets_in_a_large_universe_barely_overlap(self):
        assert expected_jaccard_random(10, 10, 1000) < 0.06

    def test_full_sets_overlap_completely(self):
        assert expected_jaccard_random(100, 100, 100) == pytest.approx(1.0)

    def test_grows_with_set_size(self):
        assert expected_jaccard_random(50, 50, 500) > expected_jaccard_random(5, 5, 500)

    def test_both_empty_is_one(self):
        assert expected_jaccard_random(0, 0, 100) == pytest.approx(1.0)

    def test_one_empty_is_zero(self):
        assert expected_jaccard_random(0, 10, 100) == pytest.approx(0.0)

    def test_rejects_impossible_sizes(self):
        with pytest.raises(ValueError):
            expected_jaccard_random(200, 10, 100)


class TestMonteCarloNull:
    def test_null_is_deterministic_given_seed(self):
        a = random_topk_null(10, 10, 200, n_draws=50, seed=3)
        b = random_topk_null(10, 10, 200, n_draws=50, seed=3)
        assert (a == b).all()

    def test_different_seeds_differ(self):
        a = random_topk_null(10, 10, 200, n_draws=50, seed=3)
        b = random_topk_null(10, 10, 200, n_draws=50, seed=4)
        assert not (a == b).all()

    def test_values_are_valid_jaccards(self):
        null = random_topk_null(8, 12, 100, n_draws=100, seed=0)
        assert null.min() >= 0.0 and null.max() <= 1.0

    def test_monte_carlo_tracks_the_closed_form(self):
        null = random_topk_null(20, 20, 200, n_draws=800, seed=1)
        assert null.mean() == pytest.approx(expected_jaccard_random(20, 20, 200), abs=0.03)

    def test_identical_full_sets_always_overlap_fully(self):
        assert (random_topk_null(50, 50, 50, n_draws=20, seed=0) == 1.0).all()


class TestOverlapVsChance:
    def test_an_at_chance_overlap_is_not_significant(self):
        """The ref-2607.18921 situation: a Jaccard that looks like structure but isn't."""
        n = 200
        chance = expected_jaccard_random(20, 20, n)
        result = overlap_vs_chance(chance, 20, 20, n, n_draws=500, seed=0)
        assert result["p_value"] > 0.05
        assert "indistinguishable from chance" in summarize_floor(result)

    def test_a_high_overlap_beats_chance(self):
        result = overlap_vs_chance(0.90, 20, 20, 200, n_draws=500, seed=0)
        assert result["p_value"] < 0.05
        assert result["excess"] > 0

    def test_p_value_is_never_exactly_zero(self):
        """A finite simulation can never justify p = 0."""
        result = overlap_vs_chance(1.0, 5, 5, 10000, n_draws=100, seed=0)
        assert result["p_value"] > 0.0

    def test_reports_an_interval_not_just_a_point(self):
        result = overlap_vs_chance(0.3, 10, 10, 200, n_draws=300, seed=0)
        assert result["chance_lo"] <= result["chance_median"] <= result["chance_hi"]

    def test_works_directly_from_frequency_vectors(self):
        f_pre = {"L0.H0->L1.H1": 0.9, "L1.H1->L2.MLP": 0.8}
        f_post = {"L0.H0->L1.H1": 0.9, "L3.H0->L2.MLP": 0.7}
        result = overlap_vs_chance_for_freqs(f_pre, f_post, n_universe=200, n_draws=200, seed=0)
        assert result["observed"] == pytest.approx(1 / 3)
        assert 0.0 <= result["p_value"] <= 1.0


class TestBenjaminiHochberg:
    def test_all_null_rejects_almost_nothing(self):
        p = [i / 100 for i in range(1, 101)]  # uniform under H0
        result = benjamini_hochberg(p, q=0.05)
        assert result["n_rejected"] <= 5

    def test_strong_signals_are_rejected(self):
        result = benjamini_hochberg([1e-9, 1e-8, 1e-7, 0.5, 0.6, 0.7], q=0.05)
        assert result["n_rejected"] == 3
        assert result["rejected"][:3] == [True, True, True]

    def test_step_up_rejects_a_borderline_run(self):
        """BH's defining behaviour: a p above its own line is still rejected if a
        larger p passes, because rejection is a step-UP procedure."""
        result = benjamini_hochberg([0.01, 0.02, 0.03, 0.04], q=0.05)
        assert result["n_rejected"] == 4

    def test_is_less_conservative_than_bonferroni(self):
        p = [0.001, 0.009, 0.02, 0.5, 0.9]
        bh = benjamini_hochberg(p, q=0.05)["n_rejected"]
        bonferroni = sum(1 for x in p if x <= 0.05 / len(p))
        assert bh >= bonferroni

    def test_adjusted_values_are_monotone_and_bounded(self):
        adj = benjamini_hochberg([0.001, 0.04, 0.03, 0.9, 0.5], q=0.05)["adjusted"]
        assert all(0.0 <= a <= 1.0 for a in adj)
        order = sorted(range(5), key=lambda i: [0.001, 0.04, 0.03, 0.9, 0.5][i])
        ranked = [adj[i] for i in order]
        assert ranked == sorted(ranked)

    def test_results_align_with_input_order(self):
        result = benjamini_hochberg([0.9, 1e-9, 0.5], q=0.05)
        assert result["rejected"] == [False, True, False]

    def test_empty_input_is_a_valid_state(self):
        result = benjamini_hochberg([], q=0.05)
        assert result["n_rejected"] == 0 and result["rejected"] == []

    def test_rejects_bad_q(self):
        with pytest.raises(ValueError):
            benjamini_hochberg([0.1], q=1.5)

    def test_rejects_out_of_range_p(self):
        with pytest.raises(ValueError):
            benjamini_hochberg([0.5, 1.7], q=0.05)

    def test_cell_grid_correction_keys_and_flags(self):
        grid = {"cell-b": 0.9, "cell-a": 1e-9, "cell-c": 0.5}
        out = correct_cell_grid(grid, q=0.05)
        assert set(out) == set(grid)
        assert out["cell-a"]["significant"] is True
        assert out["cell-b"]["significant"] is False
        assert out["cell-a"]["q_value"] >= out["cell-a"]["p"]

    def test_expected_false_positives_makes_the_grid_size_visible(self):
        """288 cells at alpha 0.05 yields ~14 'significant' cells under the global null."""
        assert expected_false_positives(288, 0.05) == pytest.approx(14.4)


class TestThresholdSweep:
    def test_verdict_bands(self):
        assert verdict_for(1.2, 3.0) == "structurally-selective"
        assert verdict_for(0.1, 0.8) == "gentler-than-noise"
        assert verdict_for(0.5, 2.0) == "indistinguishable-from-noise"

    def test_stable_conclusion_is_reported_as_threshold_independent(self):
        rows = [{"cutoff": c, "csi": 3.0, "verdict": "structurally-selective"} for c in (0.0, 0.5, 0.9)]
        out = conclusion_survives_sweep(rows)
        assert out["stable"] is True
        assert out["report_as"] == "threshold-independent"
        assert out["flip_points"] == []

    def test_a_flip_is_caught_and_flagged(self):
        rows = [
            {"cutoff": 0.0, "csi": 3.0, "verdict": "structurally-selective"},
            {"cutoff": 0.5, "csi": 1.1, "verdict": "indistinguishable-from-noise"},
        ]
        out = conclusion_survives_sweep(rows)
        assert out["stable"] is False
        assert "THRESHOLD-DEPENDENT" in out["report_as"]
        assert out["flip_points"][0]["from_cutoff"] == 0.0

    def test_undefined_points_do_not_masquerade_as_stable(self):
        rows = [{"cutoff": c, "csi": None, "verdict": "undefined"} for c in (0.0, 0.5)]
        out = conclusion_survives_sweep(rows)
        assert out["stable"] is False
        assert out["n_defined"] == 0

    def test_full_sweep_report_covers_both_levels(self):
        f_dense = {"L0.H0->L1.H1": 1.0, "L1.H1->L2.MLP": 0.6, "L0.H0->L2.MLP": 0.2}
        f_post = {"L0.H0->L1.H1": 0.9, "L3.H0->L2.MLP": 0.5}
        report = sweep_report(f_dense, f_post, d_null=[0.2, 0.3, 0.4, 0.5, 0.6],
                              cutoffs=(0.0, 0.25, 0.5), n_boot=100, seed=0)
        assert set(report["levels"]) == {"exact_edge", "routing_head"}
        for level in report["levels"].values():
            assert len(level["sweep"]) == 3
            assert "stability" in level
        assert isinstance(report["headline_survives_sweep"], bool)

    def test_sweep_rejects_an_unknown_level(self):
        from analysis.threshold_sweep import sweep_csi_over_cutoffs

        with pytest.raises(ValueError, match="unknown comparison_level"):
            sweep_csi_over_cutoffs({}, {}, [0.1], [0.0], comparison_level="nonsense")
