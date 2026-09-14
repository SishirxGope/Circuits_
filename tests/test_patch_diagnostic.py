# [AI-GEN] agent=Claude date=2026-08-08 task=NIE/PIE/INT decomposition + interaction-flag tests (claim C4 had no coverage)
# reviewed-by: PENDING

"""Interaction-aware patching diagnostic tests (proposal §2.5; claim C4).

NIE = PIE + INT (ref arXiv:2606.27510). Protocol rule 4 (proposal §4): no NIE is
reported without the grouped-versus-single diagnostic attached. Components flagged
interaction-dominated are excluded from the headline stability claim and reported
separately.

The flag RULE (currently |INT| > ratio * max(|NIE|, |PIE|), ratio = 0.5) is
PROVISIONAL and must be pre-registered before Stage D (AI_RULES.md 4.2). These tests
pin the arithmetic and the boundary behaviour of whatever rule is registered, so a
later change to the threshold is visible rather than silent.
"""

import pytest

from src.science.patch_diagnostic import (
    INT_RATIO_THRESHOLD_DEFAULT,
    PatchDiagnostic,
    interaction_dominated,
    nie_pie_int,
)


class TestDecompositionIdentity:
    def test_int_is_exactly_nie_minus_pie(self):
        out = nie_pie_int(pie=0.3, nie=0.5)
        assert out == {"nie": 0.5, "pie": 0.3, "int": pytest.approx(0.2)}

    def test_identity_holds_for_arbitrary_values(self):
        for pie, nie in [(0.0, 0.0), (-1.0, 2.0), (0.7, 0.7), (1e-9, -1e-9)]:
            out = nie_pie_int(pie=pie, nie=nie)
            assert out["pie"] + out["int"] == pytest.approx(out["nie"]), "NIE = PIE + INT violated"

    def test_zero_interaction_when_grouping_changes_nothing(self):
        assert nie_pie_int(pie=0.42, nie=0.42)["int"] == pytest.approx(0.0)


class TestInteractionFlag:
    def test_no_interaction_is_not_flagged(self):
        assert interaction_dominated(pie=0.5, nie=0.5) is False

    def test_dominant_interaction_is_flagged(self):
        """INT (0.9) is far larger than half of max(|NIE|, |PIE|) = 0.5."""
        assert interaction_dominated(pie=0.1, nie=1.0) is True

    def test_sign_flip_under_grouping_is_flagged(self):
        """A component whose effect reverses when siblings are grouped is the paradigm case."""
        assert interaction_dominated(pie=0.5, nie=-0.5) is True

    def test_flag_is_scale_invariant(self):
        assert interaction_dominated(pie=0.1, nie=1.0) == interaction_dominated(pie=10.0, nie=100.0)

    def test_threshold_boundary_is_strict_inequality(self):
        # |INT| = 0.5, scale = max(1.0, 0.5) = 1.0, ratio*scale = 0.5 -> NOT flagged
        assert interaction_dominated(pie=0.5, nie=1.0, ratio_threshold=0.5) is False
        # nudge INT above the boundary -> flagged
        assert interaction_dominated(pie=0.49, nie=1.0, ratio_threshold=0.5) is True

    def test_stricter_threshold_flags_more(self):
        assert interaction_dominated(pie=0.8, nie=1.0, ratio_threshold=0.5) is False
        assert interaction_dominated(pie=0.8, nie=1.0, ratio_threshold=0.1) is True

    def test_both_effects_zero_is_not_flagged(self):
        """eps guard: a component with no measurable effect must not be flagged by 0/0."""
        assert interaction_dominated(pie=0.0, nie=0.0) is False

    def test_default_threshold_is_the_documented_provisional_value(self):
        assert INT_RATIO_THRESHOLD_DEFAULT == 0.5


class TestOrchestrationIsStillGated:
    def test_run_refuses_until_stage_d_engineering(self):
        with pytest.raises(NotImplementedError, match="Stage D"):
            PatchDiagnostic().run(model=object(), edge="L0.H0->L1.H1", siblings=[])

    def test_diagnostic_declares_itself_provisional(self):
        """The flag rule is not pre-registered yet; nothing may treat it as final."""
        assert PatchDiagnostic().provisional is True

    def test_custom_threshold_is_carried(self):
        assert PatchDiagnostic(ratio_threshold=0.25).ratio_threshold == pytest.approx(0.25)
