# [AI-GEN] agent=OpenCode date=2026-08-07 task=Unit tests: provisional distance functions (Q2: L1 primary, JS alternative)
# reviewed-by: PENDING

import math

import pytest

from src.science.distances import jensen_shannon_distance, l1_distance


class TestL1Distance:
    def test_identical_vectors_zero(self):
        f = {"a->b": 0.5, "b->c": 1.0}
        assert l1_distance(f, dict(f)) == 0.0

    def test_simple_shift(self):
        assert l1_distance({"a->b": 1.0}, {"a->b": 0.0}) == pytest.approx(1.0)
        assert l1_distance({"a->b": 0.8}, {"a->b": 0.2}) == pytest.approx(0.6)

    def test_missing_key_counts_as_zero(self):
        assert l1_distance({"a->b": 0.5}, {}) == 0.5
        assert l1_distance({}, {"a->b": 0.5}) == 0.5

    def test_union_over_disjoint_edges(self):
        assert l1_distance({"a->b": 1.0}, {"c->d": 1.0}) == 2.0

    def test_empty_both_zero(self):
        assert l1_distance({}, {}) == 0.0


class TestJensenShannonDistance:
    def test_identical_vectors_zero(self):
        f = {"a->b": 0.5, "b->c": 0.5}
        assert jensen_shannon_distance(f, dict(f)) == 0.0

    def test_empty_both_zero(self):
        assert jensen_shannon_distance({}, {}) == 0.0

    def test_disjoint_support_sqrt_ln2(self):
        # p = {a: 1}, q = {b: 1} -> JS = sqrt(ln 2) ~ 0.8326 (normalized KL convention)
        d = jensen_shannon_distance({"a->b": 1.0}, {"c->d": 1.0})
        assert d == pytest.approx(math.sqrt(math.log(2)), abs=1e-9)

    def test_symmetric(self):
        a, b = {"x->y": 0.9, "y->z": 0.1}, {"x->y": 0.3, "y->z": 0.7}
        assert jensen_shannon_distance(a, b) == pytest.approx(jensen_shannon_distance(b, a))

    def test_normalization_invariance(self):
        # scaling both vectors by a constant must not change JS
        a, b = {"x->y": 1.0, "y->z": 2.0}, {"x->y": 3.0, "y->z": 1.0}
        d = jensen_shannon_distance(a, b)
        assert d == pytest.approx(jensen_shannon_distance({k: 10 * v for k, v in a.items()},
                                                          {k: 10 * v for k, v in b.items()}))
