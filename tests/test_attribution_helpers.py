# [AI-GEN] agent=OpenCode date=2026-08-07 task=Unit tests: attribution-graph adapter pure helpers
# reviewed-by: PENDING

import pytest

from src.extraction.attribution_graph import (
    _thresholds_from_config,
    _validate_task_config,
)


class TestThresholdsFromConfig:
    def test_valid_thresholds_returned(self):
        assert _thresholds_from_config({"node_threshold": 0.8, "edge_threshold": 0.98}) == (
            0.8,
            0.98,
        )

    def test_missing_thresholds_raise(self):
        with pytest.raises(ValueError, match="node_threshold and edge_threshold"):
            _thresholds_from_config({})

    @pytest.mark.parametrize(
        "bad", [{"node_threshold": 0.0, "edge_threshold": 0.5}, {"node_threshold": 1.0, "edge_threshold": 0.5}, {"node_threshold": -0.1, "edge_threshold": 0.5}]
    )
    def test_out_of_range_raises(self, bad):
        with pytest.raises(ValueError, match=r"in \(0, 1\)"):
            _thresholds_from_config(bad)


class TestValidateTaskConfig:
    def test_complete_task_config_passes(self):
        _validate_task_config(
            {"dataset": "x", "n_prompts": 100, "max_seq_len": 64, "attribution_target": "y"}
        )

    def test_missing_keys_raise(self):
        with pytest.raises(ValueError, match="missing required key"):
            _validate_task_config({"dataset": "x"})
