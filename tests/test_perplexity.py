# [AI-GEN] agent=Claude date=2026-08-08 task=Perplexity aggregation tests (matched-perplexity null depends on this arithmetic)
# reviewed-by: PENDING

"""Perplexity tests (proposal §2.1b; PRD.md §2).

The matched-perplexity null is *defined* by "same PPL, different family". If the PPL
aggregation is wrong, the second null is matched to the wrong target and claim C7
(the three nulls may disagree) becomes an artefact of arithmetic rather than a
finding. The classic error — averaging per-window perplexities instead of
token-weighting the NLL — is pinned below.
"""

import math

import pytest

from src.science.perplexity import (
    aggregate_perplexity,
    evaluate_perplexity,
    perplexity_from_nll,
    relative_ppl_gap,
)


class TestPerplexityFromNll:
    def test_unit_nll_per_token_gives_e(self):
        assert perplexity_from_nll(10.0, 10) == pytest.approx(math.e)

    def test_zero_nll_gives_one(self):
        assert perplexity_from_nll(0.0, 5) == pytest.approx(1.0)

    def test_zero_tokens_rejected(self):
        with pytest.raises(ValueError, match="total_tokens must be > 0"):
            perplexity_from_nll(1.0, 0)

    def test_non_finite_nll_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            perplexity_from_nll(float("inf"), 10)


class TestAggregation:
    def test_token_weighted_not_window_averaged(self):
        """One long low-loss window must dominate one short high-loss window."""
        nlls, tokens = [10.0, 9.0], [100, 3]
        token_weighted = aggregate_perplexity(nlls, tokens)
        window_mean = (math.exp(10.0 / 100) + math.exp(9.0 / 3)) / 2
        assert token_weighted == pytest.approx(math.exp(19.0 / 103))
        assert token_weighted != pytest.approx(window_mean)

    def test_equal_windows_reduce_to_the_simple_case(self):
        assert aggregate_perplexity([4.0, 4.0], [4, 4]) == pytest.approx(math.e)

    def test_single_window(self):
        assert aggregate_perplexity([6.0], [3]) == pytest.approx(math.exp(2.0))

    def test_misaligned_inputs_rejected(self):
        with pytest.raises(ValueError, match="must align"):
            aggregate_perplexity([1.0, 2.0], [10])

    def test_no_windows_rejected(self):
        with pytest.raises(ValueError, match="undefined"):
            aggregate_perplexity([], [])


class TestRelativeGap:
    def test_identical_perplexities_have_zero_gap(self):
        assert relative_ppl_gap(12.5, 12.5) == pytest.approx(0.0)

    def test_gap_is_relative_to_the_reference(self):
        assert relative_ppl_gap(11.0, 10.0) == pytest.approx(0.1)

    def test_gap_is_symmetric_in_magnitude(self):
        assert relative_ppl_gap(9.0, 10.0) == pytest.approx(0.1)

    def test_non_positive_reference_rejected(self):
        with pytest.raises(ValueError, match="must be > 0"):
            relative_ppl_gap(10.0, 0.0)


class TestEvaluationIsStillGated:
    def test_real_evaluation_refuses_until_the_corpus_is_pinned(self):
        with pytest.raises(NotImplementedError, match="RUN MODEL DOWNLOAD"):
            evaluate_perplexity(object(), corpus=None)
