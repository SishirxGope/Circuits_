# [AI-GEN] agent=OpenCode date=2026-08-07 task=Unit tests: seeded non-nested threshold grid generator (Q4 provisional)
# reviewed-by: PENDING

import pytest

from src.science.threshold_grid import generate_seeded_non_nested_grid


def _dominates(a, b):
    return a["node_threshold"] >= b["node_threshold"] and a["edge_threshold"] >= b["edge_threshold"]


def test_deterministic_given_seed():
    g1 = generate_seeded_non_nested_grid(4, seed=7)
    g2 = generate_seeded_non_nested_grid(4, seed=7)
    assert g1 == g2


def test_different_seeds_differ():
    g1 = generate_seeded_non_nested_grid(4, seed=7)
    g2 = generate_seeded_non_nested_grid(4, seed=8)
    assert g1 != g2


def test_count_and_ids():
    grid = generate_seeded_non_nested_grid(4, seed=0)
    assert len(grid) == 4
    assert len({c["id"] for c in grid}) == 4
    assert [c["id"] for c in grid] == ["seedgrid-0", "seedgrid-1", "seedgrid-2", "seedgrid-3"]


def test_thresholds_within_ranges():
    for c in generate_seeded_non_nested_grid(8, seed=3):
        assert 0.55 < c["node_threshold"] < 0.95
        assert 0.90 < c["edge_threshold"] < 0.995


def test_grid_is_pairwise_non_nested():
    grid = generate_seeded_non_nested_grid(4, seed=0)
    for i, a in enumerate(grid):
        for b in grid[i + 1:]:
            assert not _dominates(a, b), f"{a} dominates {b}"
            assert not _dominates(b, a), f"{b} dominates {a}"


def test_rejects_bad_B():
    with pytest.raises(ValueError):
        generate_seeded_non_nested_grid(0, seed=0)
