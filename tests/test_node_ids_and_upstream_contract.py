# [AI-GEN] agent=Claude date=2026-08-08 task=Tests pinning the upstream circuit-tracer contract our adapter depends on
# reviewed-by: PENDING

"""The upstream contract, pinned as tests.

Every fact asserted here was read from the local fork `circuit-tracer-0.5.2` on
2026-08-08. They are pinned as tests so that when the PI installs a *different*
circuit-tracer commit (Q9), a changed contract fails loudly here instead of silently
producing a wrong graph. That is the whole point of pinning a commit: without a test,
a pin is just a number in a document.
"""

import pytest

from src.extraction.node_ids import (
    EMBED,
    ERROR,
    FEATURE,
    LEVEL2_OPTIONS,
    LOGIT,
    UPSTREAM_NODE_KINDS,
    decode_upstream_index,
    position_policy_is_pi_owned,
    project_level2,
    to_component_id,
)


class TestUpstreamTaxonomy:
    """circuit_tracer/frontend/graph_models.py — the four node kinds, verbatim."""

    def test_the_four_node_kinds(self):
        assert UPSTREAM_NODE_KINDS == (
            "cross layer transcoder",
            "mlp reconstruction error",
            "embedding",
            "logit",
        )

    def test_there_is_no_attention_head_kind(self):
        """The finding that makes the routing-head level undefined for pipeline A."""
        assert not any("head" in kind.lower() for kind in UPSTREAM_NODE_KINDS)


class TestComponentIdMapping:
    def test_feature_node(self):
        assert to_component_id(FEATURE, layer=7, feature_idx=123) == "L7.F123"

    def test_feature_nodes_are_F_not_H(self):
        """Naming a transcoder feature 'H' would make the routing-head projection
        silently report a 'head overlap' that is nothing of the kind."""
        assert "H" not in to_component_id(FEATURE, layer=7, feature_idx=123)

    def test_error_node(self):
        assert to_component_id(ERROR, layer=3) == "L3.ERR"

    def test_embed_and_logit_nodes(self):
        assert to_component_id(EMBED, pos=4) == "EMB.P4"
        assert to_component_id(LOGIT, pos=0) == "LOGIT.P0"

    def test_position_is_appended_only_when_asked(self):
        assert to_component_id(FEATURE, layer=7, feature_idx=1, pos=9) == "L7.F1"
        assert to_component_id(FEATURE, layer=7, feature_idx=1, pos=9, include_position=True) == "L7.F1.P9"

    def test_missing_coordinates_are_rejected(self):
        with pytest.raises(ValueError, match="needs layer"):
            to_component_id(FEATURE, feature_idx=1)
        with pytest.raises(ValueError, match="needs feature_idx"):
            to_component_id(FEATURE, layer=1)
        with pytest.raises(ValueError, match="needs pos"):
            to_component_id(EMBED)

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError, match="unknown upstream node kind"):
            to_component_id("attention head")


class TestAdjacencyIndexDecode:
    """circuit_tracer/utils/create_graph_files.py — node order [features, errors, embeds, logits]."""

    N_FEAT, N_LAYERS, N_POS = 5, 3, 4  # -> errors 5..16, embeds 17..20, logits 21+

    def _decode(self, idx):
        return decode_upstream_index(idx, self.N_FEAT, self.N_LAYERS, self.N_POS)

    def test_feature_range(self):
        assert self._decode(0)["kind"] == FEATURE
        assert self._decode(4)["kind"] == FEATURE
        assert self._decode(4)["feature_slot"] == 4

    def test_error_range_uses_divmod_layout(self):
        """Upstream: layer, pos = divmod(idx - n_features, n_pos)."""
        assert self._decode(5) == {"kind": ERROR, "layer": 0, "pos": 0, "feature_slot": None}
        assert self._decode(9) == {"kind": ERROR, "layer": 1, "pos": 0, "feature_slot": None}
        assert self._decode(16) == {"kind": ERROR, "layer": 2, "pos": 3, "feature_slot": None}

    def test_error_block_is_n_layers_times_n_pos(self):
        kinds = [self._decode(i)["kind"] for i in range(self.N_FEAT, self.N_FEAT + self.N_LAYERS * self.N_POS)]
        assert set(kinds) == {ERROR}

    def test_embed_range(self):
        assert self._decode(17) == {"kind": EMBED, "layer": None, "pos": 0, "feature_slot": None}
        assert self._decode(20)["pos"] == 3

    def test_logit_range(self):
        assert self._decode(21) == {"kind": LOGIT, "layer": None, "pos": 0, "feature_slot": None}

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError):
            self._decode(-1)


class TestLevel2IsUndecided:
    """C2 needs a coarse level; pipeline A has no heads, so the scheme is a PI decision."""

    def test_there_is_no_default_scheme(self):
        with pytest.raises(ValueError, match="UNDEFINED for pipeline A"):
            project_level2(["L7.F1"], scheme="whatever-seems-reasonable")

    def test_every_option_is_documented(self):
        assert set(LEVEL2_OPTIONS) == {"layer", "feature_family", "dense_node_heads", "node_kind"}
        assert all(len(v) > 80 for v in LEVEL2_OPTIONS.values())

    def test_layer_scheme_collapses_to_the_layer(self):
        assert project_level2(["L7.F1", "L7.F2", "L3.ERR"], "layer") == {
            "L7.F1": "L7", "L7.F2": "L7", "L3.ERR": "L3"
        }

    def test_node_kind_scheme(self):
        out = project_level2(["L7.F1", "L3.ERR", "EMB.P0", "LOGIT.P0"], "node_kind")
        assert out == {"L7.F1": "feature", "L3.ERR": "error", "EMB.P0": "embed", "LOGIT.P0": "logit"}

    def test_feature_family_keeps_the_feature_drops_position(self):
        assert project_level2(["L7.F1.P3", "L7.F1.P9"], "feature_family") == {
            "L7.F1.P3": "L7.F1", "L7.F1.P9": "L7.F1"
        }

    def test_dense_node_scheme_refuses_to_be_a_projection(self):
        """It means 'use the other pipeline', not 'collapse these IDs'."""
        with pytest.raises(ValueError, match="not a projection"):
            project_level2(["L7.F1"], "dense_node_heads")


class TestPositionPolicyIsStated:
    def test_both_policies_produce_a_paper_sentence(self):
        assert "aggregated over token positions" in position_policy_is_pi_owned(False)
        assert "position-specific" in position_policy_is_pi_owned(True)

    def test_the_aggregating_policy_matches_the_current_schema(self):
        from src.common.schema import EDGES_COLUMNS

        assert "pos" not in EDGES_COLUMNS, (
            "edges.parquet gained a position column — the position policy changed and "
            "position_policy_is_pi_owned() plus the chance-floor universe must change too"
        )
