# [AI-GEN] agent=Claude date=2026-08-08 task=Cross-audit rank-agreement tests (claim C5 had no coverage)
# reviewed-by: PENDING

"""Cross-audit agreement tests (proposal §2.6; claim C5).

PRD.md §1 C5 pre-registers the decision rule: rho >= 0.8 => "the feature audit is a
valid proxy"; rho <= 0.4 => "the audits license different deployment decisions";
in between => inconclusive, reported as such. These tests pin the statistic that
rule is applied to, including its tie handling and its degenerate cases — a silent
NaN or a wrong tie correction would flip a pre-registered conclusion.

The published feature-damage rankings themselves are PI-provided and must be
verified against the actual papers (AI_RULES.md 2.2); nothing here invents one.
"""

import math

import pytest

from analysis.cross_audit import circuit_vs_feature_damage_rank_agreement, spearman_rank


class TestSpearmanKnownValues:
    def test_perfect_positive_monotone(self):
        assert spearman_rank([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_perfect_negative_monotone(self):
        assert spearman_rank([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_monotone_but_nonlinear_is_still_one(self):
        """Spearman is rank-based: a nonlinear but order-preserving map is rho = 1."""
        assert spearman_rank([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)

    def test_ties_use_average_ranks(self):
        # ranks x = [1, 2.5, 2.5, 4]; ranks y = [1, 2, 3, 4]
        assert spearman_rank([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832980505138)

    def test_all_tied_input_is_undefined_not_zero(self):
        """Zero variance means no ranking exists; reporting 0.0 would be a false 'no agreement'."""
        assert math.isnan(spearman_rank([5, 5, 5, 5], [1, 2, 3, 4]))


class TestSpearmanGuards:
    def test_fewer_than_two_pairs_is_nan(self):
        assert math.isnan(spearman_rank([1.0], [2.0]))

    def test_empty_is_nan(self):
        assert math.isnan(spearman_rank([], []))

    def test_length_mismatch_is_nan_not_a_crash(self):
        assert math.isnan(spearman_rank([1, 2, 3], [1, 2]))


class TestMappingAlignment:
    def test_mappings_align_on_shared_keys_only(self):
        circuit = {"cell-a": 0.1, "cell-b": 0.2, "cell-c": 0.3}
        feature = {"cell-a": 1.0, "cell-b": 2.0, "cell-z": 9.0}
        assert spearman_rank(circuit, feature) == pytest.approx(1.0)

    def test_key_order_does_not_matter(self):
        a = {"x": 1.0, "y": 2.0, "z": 3.0}
        b = {"z": 30.0, "y": 20.0, "x": 10.0}
        assert spearman_rank(a, b) == pytest.approx(1.0)


class TestAgreementReport:
    def test_reports_rho_and_shared_cell_count(self):
        circuit = {"pythia160m/ioi/rtn-int4": 0.9, "pythia160m/ioi/rtn-int6": 0.4,
                   "pythia160m/ioi/magnitude-30": 0.6}
        feature = {"pythia160m/ioi/rtn-int4": 0.95, "pythia160m/ioi/rtn-int6": 0.30,
                   "pythia160m/ioi/magnitude-30": 0.55}
        out = circuit_vs_feature_damage_rank_agreement(circuit, feature)
        assert out["spearman"] == pytest.approx(1.0)
        assert out["n_shared_cells"] == 3

    def test_thin_overlap_is_visible_rather_than_hidden(self):
        """A rho computed on 2 cells must not look like a rho computed on 20."""
        out = circuit_vs_feature_damage_rank_agreement({"a": 1.0, "b": 2.0, "c": 3.0},
                                                       {"a": 1.0, "b": 2.0})
        assert out["n_shared_cells"] == 2

    def test_no_shared_cells_gives_nan_and_zero_count(self):
        out = circuit_vs_feature_damage_rank_agreement({"a": 1.0}, {"b": 2.0})
        assert math.isnan(out["spearman"])
        assert out["n_shared_cells"] == 0

    def test_disagreement_is_representable(self):
        """C5's informative outcome: the two audits order the grid differently."""
        circuit = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
        feature = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
        out = circuit_vs_feature_damage_rank_agreement(circuit, feature)
        assert out["spearman"] == pytest.approx(-1.0)
        assert out["spearman"] <= 0.4  # PRD.md §1 C5 "different deployment decisions" band
