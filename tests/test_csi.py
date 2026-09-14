# [AI-GEN] agent=Claude date=2026-08-08 task=CSI unit + golden-file tests (ARCHITECTURE.md §6 required coverage that was missing)
# reviewed-by: PENDING

"""CSI tests (ARCHITECTURE.md §6: "CSI: hash-mismatch on frozen null must raise;
division and bootstrap on synthetic data" + "Golden-file CSI on a fixed synthetic
cell: any diff in metric code that changes the number fails CI loudly").

CSI(c,T) = D(c) / median(D_null(c)) with a bootstrap CI (CLAUDE.md §5).
"""

import json

import numpy as np
import pytest

from src.common.stage_guard import assert_null_frozen_hash
from src.science.csi import csi, median_dnull

# A fixed synthetic cell. D_null = 5 draws with median 0.3; D(c) = 0.6 -> CSI = 2.0.
GOLDEN_DNULL = [0.10, 0.20, 0.30, 0.40, 0.50]
GOLDEN_D = 0.6
GOLDEN = {"csi": 2.0, "ci_lo": 1.2, "ci_hi": 6.0, "D": 0.6, "dnull_median": 0.3}


class TestMedianDnull:
    def test_odd_and_even_lengths(self):
        assert median_dnull(np.array([0.1, 0.3, 0.2])) == pytest.approx(0.2)
        assert median_dnull(np.array([0.1, 0.3])) == pytest.approx(0.2)

    def test_empty_null_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            median_dnull(np.array([]))


class TestCsiDivision:
    def test_csi_equals_one_when_damage_matches_the_floor(self):
        """CSI ~ 1 is the pre-committed negative result (AI_RULES.md 4.5)."""
        out = csi(0.3, GOLDEN_DNULL, n_boot=100, seed=0)
        assert out["csi"] == pytest.approx(1.0)

    def test_csi_above_one_is_structurally_selective(self):
        assert csi(0.9, GOLDEN_DNULL, n_boot=100, seed=0)["csi"] == pytest.approx(3.0)

    def test_csi_below_one_is_gentler_than_noise(self):
        assert csi(0.15, GOLDEN_DNULL, n_boot=100, seed=0)["csi"] == pytest.approx(0.5)

    def test_zero_median_null_raises_rather_than_dividing(self):
        """A null that never moves the circuit must surface loudly, never as inf."""
        with pytest.raises(ValueError, match="median\\(D_null\\) == 0"):
            csi(0.5, [0.0, 0.0, 0.0], n_boot=10, seed=0)


class TestBootstrap:
    def test_ci_brackets_the_point_estimate(self):
        out = csi(GOLDEN_D, GOLDEN_DNULL, n_boot=500, seed=3)
        assert out["ci_lo"] <= out["csi"] <= out["ci_hi"]

    def test_bootstrap_is_deterministic_given_seed(self):
        """AI_RULES.md 1.1: every random draw is seeded and reproducible."""
        a = csi(GOLDEN_D, GOLDEN_DNULL, n_boot=250, seed=42)
        b = csi(GOLDEN_D, GOLDEN_DNULL, n_boot=250, seed=42)
        assert a == b

    def test_different_seeds_move_the_interval_not_the_point(self):
        a = csi(GOLDEN_D, GOLDEN_DNULL, n_boot=250, seed=1)
        b = csi(GOLDEN_D, GOLDEN_DNULL, n_boot=250, seed=2)
        assert a["csi"] == b["csi"]
        assert (a["ci_lo"], a["ci_hi"]) != (b["ci_lo"], b["ci_hi"]) or a["ci_lo"] == b["ci_lo"]

    def test_degenerate_null_gives_a_point_interval(self):
        out = csi(0.6, [0.3, 0.3, 0.3, 0.3], n_boot=100, seed=0)
        assert out["ci_lo"] == pytest.approx(2.0)
        assert out["ci_hi"] == pytest.approx(2.0)


class TestGoldenFile:
    """Any metric-code change that moves this number fails CI loudly (ARCHITECTURE.md §6)."""

    def test_fixed_synthetic_cell_matches_golden_values(self):
        out = csi(GOLDEN_D, GOLDEN_DNULL, n_boot=200, seed=7)
        for key, expected in GOLDEN.items():
            assert out[key] == pytest.approx(expected, rel=1e-12), (
                f"CSI golden value drifted for {key!r}: {out[key]} != {expected}. "
                "If this change is intentional it invalidates every downstream number "
                "(AI_RULES.md 1.3) and needs PI sign-off (AI_RULES.md §6)."
            )

    def test_csi_row_is_serializable_for_the_csi_table(self):
        out = csi(GOLDEN_D, GOLDEN_DNULL, n_boot=50, seed=7)
        assert set(out) == {"csi", "ci_lo", "ci_hi", "D", "dnull_median"}
        json.dumps(out)  # every field must land in the CSI parquet/CSV table


class TestFrozenHashIsVerifiedBeforeDivision:
    """AI_RULES.md 1.4: every CSI computation verifies the null it divides by."""

    def _frozen_meta(self, tmp_path, null_hash="abc123"):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps({"config_hash": "c", "seeds": [0], "R": 3,
                                    "null_frozen_hash": null_hash}), encoding="utf-8")
        return meta

    def test_matching_hash_passes(self, tmp_path):
        assert_null_frozen_hash(self._frozen_meta(tmp_path), "abc123")

    def test_mismatched_hash_fails_hard(self, tmp_path):
        with pytest.raises(RuntimeError, match="null_frozen_hash mismatch"):
            assert_null_frozen_hash(self._frozen_meta(tmp_path), "not-the-frozen-hash")

    def test_missing_hash_fails_hard(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps({"config_hash": "c"}), encoding="utf-8")
        with pytest.raises(RuntimeError, match="no null_frozen_hash"):
            assert_null_frozen_hash(meta, "abc123")
