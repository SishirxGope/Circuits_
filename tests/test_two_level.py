# [AI-GEN] agent=Claude date=2026-08-08 task=Routing-head projector + two-level comparison tests (ARCHITECTURE.md §6 required coverage that was missing)
# reviewed-by: PENDING

"""Two-level comparison tests (proposal §2.4; claim C2).

ARCHITECTURE.md §6 requires: "Routing-head projector: known edge sets project to
known head sets."

Protocol rule 3 (proposal §4): "Two levels or it does not count." The levels can
disagree — ref arXiv:2607.18921 reports exact-edge Jaccard@10 0.14-0.16 while the
routing-head set overlaps at 0.55-0.67 on the same comparison. The tests below
encode that a disagreement is representable and is reported, never averaged away.
"""

import pytest

from src.science.two_level import (
    compare_at_both_levels,
    exact_edge_overlap,
    project_to_routing_heads,
    routing_head_overlap,
)


class TestRoutingHeadProjection:
    def test_known_edge_set_projects_to_known_head_set(self):
        f = {"L0.H0->L1.H1": 0.9, "L1.H1->L2.MLP": 0.4}
        assert project_to_routing_heads(f) == {"L0.H0": 0.9, "L1.H1": 0.9}

    def test_mlp_nodes_are_dropped_at_the_routing_head_level(self):
        assert project_to_routing_heads({"L2.MLP->L3.MLP": 1.0}, strict=False) == {}

    def test_an_all_empty_projection_is_refused_by_default(self):
        """The most dangerous silent failure in the pipeline.

        circuit-tracer's nodes are transcoder features, so a pipeline-A circuit has no
        `L{l}.H{h}` at all. Silently returning {} makes both sides of the routing-head
        overlap empty, the Jaccard 1.0, and claim C2 vacuously satisfied — reporting a
        BETTER two-level agreement than reality, without crashing.
        """
        with pytest.raises(ValueError, match="projection is EMPTY"):
            project_to_routing_heads({"L0.F12->L1.F7": 0.9})

    def test_the_refusal_names_the_open_decision(self):
        with pytest.raises(ValueError, match="Q10"):
            project_to_routing_heads({"L2.MLP->L3.MLP": 1.0})

    def test_an_empty_input_is_not_an_error(self):
        assert project_to_routing_heads({}) == {}

    def test_head_inclusion_is_the_max_over_touching_edges(self):
        f = {"L0.H0->L1.H1": 0.2, "L5.MLP->L0.H0": 0.8, "L0.H0->L9.MLP": 0.5}
        assert project_to_routing_heads(f)["L0.H0"] == pytest.approx(0.8)

    def test_both_endpoints_are_credited(self):
        assert set(project_to_routing_heads({"L4.H2->L7.H3": 0.6})) == {"L4.H2", "L7.H3"}

    def test_empty_vector_projects_to_empty(self):
        assert project_to_routing_heads({}) == {}

    def test_projection_is_deterministic(self):
        f = {"L0.H0->L1.H1": 0.9, "L1.H1->L0.H0": 0.3, "L2.MLP->L1.H1": 0.7}
        assert project_to_routing_heads(f) == project_to_routing_heads(dict(reversed(list(f.items()))))


class TestExactEdgeOverlap:
    def test_identical_vectors_overlap_fully(self):
        f = {"L0.H0->L1.H1": 0.5}
        assert exact_edge_overlap(f, f) == pytest.approx(1.0)

    def test_disjoint_edge_sets_overlap_zero(self):
        assert exact_edge_overlap({"L0.H0->L1.H1": 1.0}, {"L2.H0->L3.H1": 1.0}) == pytest.approx(0.0)

    def test_jaccard_arithmetic(self):
        pre = {"a->b": 1.0, "b->c": 1.0, "c->d": 1.0}
        post = {"a->b": 1.0, "b->c": 1.0, "x->y": 1.0}
        assert exact_edge_overlap(pre, post) == pytest.approx(2 / 4)

    def test_cutoff_excludes_low_frequency_edges(self):
        pre = {"L0.H0->L1.H1": 0.9, "L0.H0->L2.MLP": 0.05}
        post = {"L0.H0->L1.H1": 0.9}
        assert exact_edge_overlap(pre, post, cutoff=0.0) == pytest.approx(0.5)
        assert exact_edge_overlap(pre, post, cutoff=0.1) == pytest.approx(1.0)

    def test_both_empty_is_trivially_identical(self):
        assert exact_edge_overlap({}, {}) == pytest.approx(1.0)


class TestLevelsCanDisagree:
    """The motivating finding of ref arXiv:2607.18921, reproduced structurally."""

    PRE = {"L0.H0->L1.H1": 1.0, "L1.H1->L2.MLP": 0.8, "L0.H0->L2.MLP": 0.5}
    POST = {"L0.H0->L1.H1": 1.0, "L3.H0->L2.MLP": 0.9}

    def test_exact_edge_collapses_while_routing_head_holds_up(self):
        exact = exact_edge_overlap(self.PRE, self.POST)
        heads = routing_head_overlap(self.PRE, self.POST)
        assert exact == pytest.approx(0.25)
        assert heads == pytest.approx(2 / 3)
        assert heads > exact, "this fixture exists to encode a level DISAGREEMENT"

    def test_compare_at_both_levels_returns_both_together(self):
        out = compare_at_both_levels(self.PRE, self.POST)
        assert set(out) == {"exact_edge", "routing_head"}
        assert out["exact_edge"] == pytest.approx(exact_edge_overlap(self.PRE, self.POST))
        assert out["routing_head"] == pytest.approx(routing_head_overlap(self.PRE, self.POST))

    def test_a_claim_can_never_be_read_off_one_level_alone(self):
        """Both keys are always present, so no caller can report one and omit the other."""
        out = compare_at_both_levels({}, {})
        assert "exact_edge" in out and "routing_head" in out
