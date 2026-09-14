# [AI-GEN] agent=Claude date=2026-08-08 task=Regressions for the six defects the property smoke test found
# reviewed-by: PENDING

"""Regressions for defects found by property-based smoke testing on 2026-08-08.

Each test below corresponds to a real defect that the 376-test suite passed straight
through, because every existing test used well-formed inputs. These use the inputs a
real pipeline actually produces.
"""

import pytest

from src.common.csi_table import validate_csi_row
from src.common.schema import Edge, Graph
from src.science.csi import csi
from src.science.decompose import decompose
from src.science.inclusion_freq import compute_from_graphs
from src.science.perplexity import (
    REFERENCE_PROTOCOL,
    delta_log_ppl,
    validate_ppl_protocol,
)


class TestInclusionFrequencyIsAFraction:
    """THE serious one: s(e) could exceed 1.0, which is not a fraction.

    circuit-tracer's nodes are (layer, pos, feature_idx). Our edge id carries no
    position, so several position-specific upstream edges collapse onto ONE component
    pair within a single (config, seed) cell. Counting edge RECORDS instead of cells
    that CONTAIN the edge then pushed s(e) above 1 and corrupted D, CSI and the bands.
    """

    def test_repeated_edge_in_one_cell_counts_once(self):
        g = Graph(edges=(Edge("L0.F12", "L1.F7", "cfg0", 0),
                         Edge("L0.F12", "L1.F7", "cfg0", 0),
                         Edge("L0.F12", "L1.F7", "cfg0", 0)))
        result = compute_from_graphs([g])
        assert result.freq.frequencies["L0.F12->L1.F7"] == pytest.approx(1.0)

    def test_it_agrees_with_Graph_included_edge_ids(self):
        """Graph.included_edge_ids() always deduplicated; the counter did not."""
        g = Graph(edges=(Edge("A", "B", "c0", 0), Edge("A", "B", "c0", 0)))
        assert set(compute_from_graphs([g]).freq.frequencies) == set(g.included_edge_ids())

    def test_records_are_deduplicated_too(self):
        """edges.parquet must not carry the duplicate rows either."""
        g = Graph(edges=(Edge("A", "B", "c0", 0), Edge("A", "B", "c0", 0)))
        assert len(compute_from_graphs([g]).records) == 1

    def test_same_edge_in_different_cells_still_counts_separately(self):
        """The fix must not collapse across cells — that is what s(e) measures."""
        g0 = Graph(edges=(Edge("A", "B", "c0", 0), Edge("A", "B", "c0", 0)))
        g1 = Graph(edges=(Edge("A", "B", "c1", 0),))
        assert compute_from_graphs([g0, g1]).freq.frequencies["A->B"] == pytest.approx(1.0)

    def test_partial_presence_still_gives_a_partial_fraction(self):
        g0 = Graph(edges=(Edge("A", "B", "c0", 0), Edge("A", "B", "c0", 0)))
        g1 = Graph(edges=(Edge("C", "D", "c1", 0),))
        freq = compute_from_graphs([g0, g1]).freq.frequencies
        assert freq["A->B"] == pytest.approx(0.5)
        assert freq["C->D"] == pytest.approx(0.5)

    def test_every_frequency_lands_in_the_unit_interval(self):
        graphs = [
            Graph(edges=tuple(Edge("A", "B", f"c{c}", s) for _ in range(4)))
            for c in range(3) for s in range(2)
        ]
        assert all(0.0 <= v <= 1.0 for v in compute_from_graphs(graphs).freq.frequencies.values())


class TestCsiRejectsImpossibleDistances:
    def test_negative_D_is_rejected(self):
        """A sign error upstream produced csi = -5.0, which passed every later check."""
        with pytest.raises(ValueError, match="must be >= 0"):
            csi(-1.0, [0.1, 0.2, 0.3], n_boot=10, seed=0)

    def test_negative_null_draws_are_rejected(self):
        with pytest.raises(ValueError, match="negative distances"):
            csi(0.5, [-0.1, 0.2, 0.3], n_boot=10, seed=0)

    def test_zero_distance_is_still_legal(self):
        """D = 0 means the circuit did not move at all — a real, reportable outcome."""
        assert csi(0.0, [0.1, 0.2, 0.3], n_boot=10, seed=0)["csi"] == pytest.approx(0.0)


class TestDecomposeRejectsNonFractions:
    def test_out_of_range_frequency_is_rejected(self):
        """Banding s(e)=1.5 as 'core' would launder an upstream counting bug."""
        with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
            decompose({"a->b": 1.5}, 0.9, 0.1)

    def test_negative_frequency_is_rejected(self):
        with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
            decompose({"a->b": -0.1}, 0.9, 0.1)

    def test_the_boundaries_themselves_remain_legal(self):
        assert decompose({"a->b": 1.0, "c->d": 0.0}, 0.9, 0.1) == {"a->b": "core", "c->d": "noise"}


class TestCsiTableRejectsNegativeRatios:
    def _row(self, **over):
        row = {"model": "m", "task": "t", "family": "f", "level_param": "0.3",
               "comparison_level": "exact_edge", "csi": 1.0, "ci_lo": 0.5, "ci_hi": 1.5,
               "D": 0.3, "d_normalized": 0.15, "dnull_median": 0.3,
               "null_frozen_hash": "h"}
        row.update(over)
        return row

    def test_negative_csi_cannot_enter_the_table(self):
        with pytest.raises(ValueError, match="none of these may be negative"):
            validate_csi_row(self._row(csi=-1.0, ci_lo=-2.0, ci_hi=0.0))

    def test_negative_D_cannot_enter_the_table(self):
        with pytest.raises(ValueError, match="none of these may be negative"):
            validate_csi_row(self._row(D=-0.3))

    def test_a_well_formed_row_still_passes(self):
        validate_csi_row(self._row())


class TestPerplexityProtocolMatchesTheReference:
    """Transcribed from sae-pruning-paper-main/docs/PROTOCOL.md, verified 2026-08-08."""

    def test_reference_constants(self):
        assert REFERENCE_PROTOCOL["window"] == 1024
        assert REFERENCE_PROTOCOL["stride"] == 512
        assert REFERENCE_PROTOCOL["bos_policy"] == "per_window"
        assert REFERENCE_PROTOCOL["dtype"] == "bfloat16"

    def test_matching_protocol_reports_no_deviation(self):
        assert validate_ppl_protocol(dict(REFERENCE_PROTOCOL)) == []

    def test_the_bos_fault_that_broke_gemma_is_called_out_by_name(self):
        """Wrong BOS policy made the reference's Gemma-2 read 410 instead of ~11."""
        deviations = validate_ppl_protocol({**REFERENCE_PROTOCOL, "bos_policy": "single_leading"})
        assert any("410" in d for d in deviations)

    def test_window_deviation_is_reported(self):
        deviations = validate_ppl_protocol({**REFERENCE_PROTOCOL, "window": 512})
        assert any("window" in d for d in deviations)

    def test_delta_log_ppl_is_the_reported_quantity(self):
        import math

        assert delta_log_ppl(20.0, 10.0) == pytest.approx(math.log(2.0))
        assert delta_log_ppl(10.0, 10.0) == pytest.approx(0.0)

    def test_delta_log_ppl_rejects_non_positive_perplexity(self):
        with pytest.raises(ValueError, match="must be > 0"):
            delta_log_ppl(0.0, 10.0)
