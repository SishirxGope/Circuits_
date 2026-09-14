# [AI-GEN] agent=OpenCode date=2026-08-07 task=Unit tests: inclusion frequency s(e) (proposal §2.3)
# reviewed-by: PENDING

import pytest

from src.science.inclusion_freq import compute_from_graphs, frequencies_from_inclusion_counts
from src.common.schema import Edge, FreqVector, Graph


def _g(edges, config_id="c0", seed=0):
    return Graph(edges=tuple(Edge(src_component=s, dst_component=d, config_id=config_id, seed=seed) for s, d in edges))


def test_hand_built_three_graph_ensemble():
    # G1: A->B, A->C   G2: A->B, D->E   G3: A->B
    graphs = [_g([("A", "B"), ("A", "C")], "c0", 0), _g([("A", "B"), ("D", "E")], "c1", 0), _g([("A", "B")], "c2", 1)]
    result = compute_from_graphs(graphs, n=3)
    freq = result.freq.frequencies
    assert freq["A->B"] == pytest.approx(1.0)
    assert freq["A->C"] == pytest.approx(1 / 3)
    assert freq["D->E"] == pytest.approx(1 / 3)
    assert set(freq) == {"A->B", "A->C", "D->E"}
    assert result.n_cells == 3


def test_empty_ensemble_is_safe():
    result = compute_from_graphs([])
    assert result.records == ()
    assert result.freq.frequencies == {}
    assert result.n_cells == 0


def test_deterministic_ordering():
    graphs = [_g([("B", "A")]), _g([("A", "B"), ("B", "A")]), _g([("B", "A")])]
    r1 = compute_from_graphs(graphs, n=3)
    r2 = compute_from_graphs(graphs, n=3)
    assert r1.freq.items_sorted() == r2.freq.items_sorted()
    assert [x["src_component"] for x in r1.records] == [x["src_component"] for x in r2.records]
    # records ordered by (edge_id, config_id, seed)
    keys = [(x["src_component"] + "->" + x["dst_component"], x["config_id"], x["seed"]) for x in r1.records]
    assert keys == sorted(keys)


def test_records_carry_cell_identity():
    graphs = [_g([("A", "B")], config_id="c0", seed=0)]
    result = compute_from_graphs(graphs, n=1)
    assert result.records[0] == {"src_component": "A", "dst_component": "B", "config_id": "c0", "seed": 0, "included": True}


def test_non_included_edges_are_not_counted():
    g = Graph(edges=(Edge("A", "B", "c0", 0, included=True), Edge("C", "D", "c0", 0, included=False)))
    result = compute_from_graphs([g], n=1)
    assert result.freq.frequencies == {"A->B": 1.0}


def test_bad_denominator_raises():
    with pytest.raises(ValueError):
        compute_from_graphs([_g([("A", "B")])], n=0)


def test_frequencies_from_counts_helper():
    fv = frequencies_from_inclusion_counts({"e1": 2, "e2": 1}, n=4)
    assert isinstance(fv, FreqVector)
    assert fv.frequencies == {"e1": 0.5, "e2": 0.25}
