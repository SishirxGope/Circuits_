# [AI-GEN] agent=Claude date=2026-09-12 task=Guard the pre-registered CSI bootstrap axes, normalised L1, and the chance-floor universe
# reviewed-by: PENDING

"""The 2026-09-12 statistical pre-registrations (B/S/r bootstrap, normalised L1, N).

These guard properties, not outputs. Each one corresponds to a decision that is fixed
before any real result exists, and each would fail silently rather than loudly if it
regressed — which is why they are asserted rather than left to review.
"""

import numpy as np
import pytest

from src.science.csi import cells_from_records, csi, csi_over_ensemble
from src.science.distances import l1_distance, normalized_l1_distance


def _paired_ensemble(n_cfg=4, n_seed=3):
    """A dense ensemble and a compressed one that loses edges unevenly across cells.

    Uneven loss is the point: config ``i`` at seed ``s`` drops its first ``i + s``
    edges, so resampling configs or seeds genuinely changes the aggregate s(e).

    A balanced pattern would NOT do: with ``(j + i + s) % 2`` every edge lands in
    exactly half the cells and every resample reproduces s(e) = 0.5 identically, so the
    B/S axes contribute no variance and a widening assertion would fail even though the
    estimator is correct. See ``TestNoVarianceMeansNoWidening``.
    """
    cfgs = [f"c{i}" for i in range(n_cfg)]
    seeds = list(range(n_seed))
    edges = [f"n{k}->n{k + 1}" for k in range(8)]
    pre = {(c, s): frozenset(edges) for c in cfgs for s in seeds}
    post = {
        (c, s): frozenset(edges[min(i + s, len(edges)):])
        for i, c in enumerate(cfgs)
        for s in seeds
    }
    return pre, post


def _aggregate(cells):
    n = len(cells)
    counts = {}
    for v in cells.values():
        for e in v:
            counts[e] = counts.get(e, 0) + 1
    return {e: c / n for e, c in counts.items()}


class TestBootstrapAxes:
    """proposal §2.1c / CLAUDE.md §5: the CI is over B, S AND r."""

    def test_the_point_estimate_is_unchanged(self):
        """Widening the interval must not move the number the interval is around."""
        pre, post = _paired_ensemble()
        d_null = list(np.linspace(0.5, 1.5, 20))
        d_c = l1_distance(_aggregate(pre), _aggregate(post))

        r_only = csi(d_c, d_null, n_boot=500, seed=0)
        bsr = csi_over_ensemble(pre, post, d_null, l1_distance, n_boot=500, seed=0)
        assert bsr["csi"] == pytest.approx(r_only["csi"], rel=1e-12)

    def test_resampling_the_ensemble_axes_widens_the_interval(self):
        """The whole reason the decision matters: it changes what excludes 1."""
        pre, post = _paired_ensemble()
        d_null = list(np.linspace(0.5, 1.5, 20))
        d_c = l1_distance(_aggregate(pre), _aggregate(post))

        r_only = csi(d_c, d_null, n_boot=2000, seed=0)
        bsr = csi_over_ensemble(pre, post, d_null, l1_distance, n_boot=2000, seed=0)
        assert (bsr["ci_hi"] - bsr["ci_lo"]) > (r_only["ci_hi"] - r_only["ci_lo"]), (
            "B/S resampling added no width, so the ensemble axes are not being "
            "resampled — the CI would be anti-conservative exactly as before"
        )

    def test_it_is_deterministic_given_the_seed(self):
        pre, post = _paired_ensemble()
        d_null = list(np.linspace(0.5, 1.5, 20))
        a = csi_over_ensemble(pre, post, d_null, l1_distance, n_boot=200, seed=3)
        b = csi_over_ensemble(pre, post, d_null, l1_distance, n_boot=200, seed=3)
        assert a == b  # AI_RULES.md 1.1

    def test_unpaired_ensembles_are_refused(self):
        """Dense and compressed must be resampled at matched (config, seed)."""
        pre, post = _paired_ensemble()
        post = {k: v for k, v in post.items() if k != ("c0", 0)}
        with pytest.raises(ValueError, match="SAME"):
            csi_over_ensemble(pre, post, [1.0], l1_distance)

    def test_a_zero_null_median_is_refused_not_reported(self):
        pre, post = _paired_ensemble()
        with pytest.raises(ValueError, match="undefined"):
            csi_over_ensemble(pre, post, [0.0, 0.0, 0.0], l1_distance)


class TestEmptyCellsStillCount:
    """A view that found no edges is an observation, not a missing row."""

    def test_a_cell_with_no_edges_is_absent_without_the_key_list(self):
        records = [
            {"src_component": "a", "dst_component": "b", "config_id": "c0", "seed": 0,
             "included": True},
        ]
        assert cells_from_records(records) == {("c0", 0): frozenset({"a->b"})}

    def test_supplying_the_grid_keeps_empty_cells_in_the_denominator(self):
        records = [
            {"src_component": "a", "dst_component": "b", "config_id": "c0", "seed": 0,
             "included": True},
        ]
        keys = [("c0", 0), ("c1", 0)]
        cells = cells_from_records(records, keys=keys)
        assert cells == {("c0", 0): frozenset({"a->b"}), ("c1", 0): frozenset()}

        # and the inflation it prevents: s(e) is 1/2, not 1/1
        pre = cells
        post = dict(cells)
        out = csi_over_ensemble(pre, post, [1.0, 1.0], l1_distance, n_boot=10, seed=0)
        assert out["D"] == 0.0  # identical ensembles
        assert out["n_configs"] == 2

    def test_excluded_records_are_ignored(self):
        records = [
            {"src_component": "a", "dst_component": "b", "config_id": "c0", "seed": 0,
             "included": False},
        ]
        assert cells_from_records(records, keys=[("c0", 0)]) == {("c0", 0): frozenset()}


class TestNormalizedL1:
    """Q2's pre-registered third column."""

    def test_it_is_l1_over_the_union_size(self):
        pre, post = {"a": 1.0, "b": 1.0}, {"a": 0.5}
        # |union| = 2; L1 = 0.5 + 1.0 = 1.5
        assert normalized_l1_distance(pre, post) == pytest.approx(0.75)

    def test_two_empty_vectors_are_zero_not_a_division_by_zero(self):
        assert normalized_l1_distance({}, {}) == 0.0

    def test_it_separates_cells_that_raw_l1_conflates(self):
        """The reason it exists: identical raw L1, very different per-edge change.

        A 1-edge circuit wholly destroyed and a 10-edge circuit that lost one edge both
        score L1 = 1.0. Reported bare, they look like equal damage. Cells in this grid
        WILL differ in edge count, because compression changes how many edges survive.
        """
        small_pre, small_post = {"a": 1.0}, {"a": 0.0}
        big_pre = {f"e{i}": 1.0 for i in range(10)}
        big_post = {f"e{i}": (0.0 if i == 0 else 1.0) for i in range(10)}

        assert l1_distance(small_pre, small_post) == pytest.approx(
            l1_distance(big_pre, big_post)
        ), "fixture is wrong: the two cells must have EQUAL raw L1 for this to mean anything"

        assert normalized_l1_distance(small_pre, small_post) == pytest.approx(1.0)
        assert normalized_l1_distance(big_pre, big_post) == pytest.approx(0.1)


class TestNoVarianceMeansNoWidening:
    """A guard against reading the widening test as "B/S always adds width".

    If the ensemble is perfectly balanced — every edge in exactly the same fraction of
    cells no matter which cells you draw — then resampling B and S reproduces the same
    s(e) and the two estimators agree. That is correct, not a failure: there is no
    between-cell uncertainty to propagate. Pinning it here means a future change that
    manufactures width out of nothing fails loudly.
    """

    def test_a_balanced_ensemble_has_no_between_cell_variance_to_propagate(self):
        """Asserted on the invariant itself, not on two noisy interval widths.

        ``csi`` and ``csi_over_ensemble`` draw from different RNG streams, so even when
        they estimate the identical quantity their Monte Carlo interval widths differ by
        a few percent. Comparing those widths would be a flaky test of a real property.
        The property is that under a balanced ensemble the recomputed D is the SAME for
        every resampled draw of configs and seeds — so there is nothing for the B/S axes
        to add. That is exact and can be asserted exactly.
        """
        from src.science.csi import _frequencies_from_cells

        cfgs = [f"c{i}" for i in range(4)]
        seeds = [0, 1, 2]
        edges = [f"n{k}->n{k + 1}" for k in range(8)]
        pre = {(c, s): frozenset(edges) for c in cfgs for s in seeds}
        # (j + i + s) % 2: every edge lands in exactly half the cells, symmetrically,
        # so the aggregate is 0.5 for every edge under any draw.
        post = {
            (c, s): frozenset(e for j, e in enumerate(edges) if (j + i + s) % 2)
            for i, c in enumerate(cfgs)
            for s in seeds
        }

        rng = np.random.default_rng(0)
        distances = set()
        for _ in range(40):
            cfg_draw = [cfgs[i] for i in rng.integers(0, len(cfgs), size=len(cfgs))]
            seed_draw = [seeds[i] for i in rng.integers(0, len(seeds), size=len(seeds))]
            keys = [(c, s) for c in cfg_draw for s in seed_draw]
            distances.add(round(l1_distance(
                _frequencies_from_cells(pre, keys),
                _frequencies_from_cells(post, keys),
            ), 12))
        assert len(distances) == 1, (
            f"a balanced ensemble should give one D under every resample, got {distances}"
        )

    def test_a_heterogeneous_ensemble_does_vary(self):
        """The contrast case, so the test above cannot pass by measuring nothing."""
        from src.science.csi import _frequencies_from_cells

        pre, post = _paired_ensemble()
        cfgs = sorted({c for c, _ in pre})
        seeds = sorted({s for _, s in pre})
        rng = np.random.default_rng(0)
        distances = set()
        for _ in range(40):
            cfg_draw = [cfgs[i] for i in rng.integers(0, len(cfgs), size=len(cfgs))]
            seed_draw = [seeds[i] for i in rng.integers(0, len(seeds), size=len(seeds))]
            keys = [(c, s) for c in cfg_draw for s in seed_draw]
            distances.add(round(l1_distance(
                _frequencies_from_cells(pre, keys),
                _frequencies_from_cells(post, keys),
            ), 12))
        assert len(distances) > 1
