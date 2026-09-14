# [AI-GEN] agent=OpenCode date=2026-08-07 task=Unit tests: mode-aware config guard (engineering vs scientific; PI provisional unblock)
# reviewed-by: PENDING

import pytest

from src.common.config_guard import (
    MODE_ENGINEERING,
    MODE_SCIENTIFIC,
    PI_DECISIONS_DOC,
    assert_engineering_dry_run_limits,
    assert_no_gating_questions,
    mode_of,
    report_open_questions,
    report_provisional_resolutions,
)


def _cfg(**overrides):
    """A config with every Stage-A-gating question OPEN (values are flags, not science)."""
    cfg = {
        "mode": {"name": MODE_ENGINEERING},
        "pipeline": "mock",
        "model": {"name": "pythia160m", "hf_revision": None},
        "task": {"name": "ioi", "pi_confirmed": False},
        "ensemble": {
            "pi_confirmed": False,
            "threshold_grid": None,
            "decompose": {"core_threshold": None, "noise_threshold": None},
        },
        "distance": {"name": None},
        "comparison": {"level2_scheme": None, "position_policy": None},  # Q10/Q11 OPEN
    }
    cfg.update(overrides)
    return cfg


def _resolved_cfg(**overrides):
    """All Stage-A-gating questions resolved (synthetic stand-ins for values only)."""
    return _cfg(
        pipeline="attr",
        model={"name": "pythia160m", "hf_revision": "abc123"},
        task={"name": "ioi", "pi_confirmed": True},
        ensemble={
            "pi_confirmed": True,
            "threshold_grid": [{"id": "c1", "node_threshold": 0.8, "edge_threshold": 0.98}],
            "decompose": {"core_threshold": 0.9, "noise_threshold": 0.1},
        },
        distance={"name": "l1"},
        comparison={"level2_scheme": "layer", "position_policy": "aggregate"},
        **overrides,
    )


def test_mode_of_defaults_to_engineering():
    assert mode_of({}) == MODE_ENGINEERING
    assert mode_of({"mode": {"name": MODE_SCIENTIFIC}}) == MODE_SCIENTIFIC
    assert mode_of({"mode": {"name": "bogus"}}) == MODE_ENGINEERING


def test_all_detectable_open_questions_reported():
    ids = {q["id"] for q in report_open_questions(_cfg(), pipeline="mock")}
    assert ids == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q10", "Q11"}


def test_open_questions_are_ordered_numerically_not_lexically():
    """Q10 must follow Q9, not Q1 — a lexical sort buries the newest decisions."""
    ids = [q["id"] for q in report_open_questions(_cfg(), pipeline="mock")]
    assert ids == sorted(ids, key=lambda q: int(q[1:]))
    assert ids[-2:] == ["Q10", "Q11"]


def test_docs_only_questions_never_detectable_from_config():
    ids = {q["id"] for q in report_open_questions(_cfg(), pipeline="mock")}
    assert "Q7" not in ids and "Q8" not in ids and "Q9" not in ids


def test_resolved_config_reports_nothing():
    assert report_open_questions(_resolved_cfg(), pipeline="attr") == []


def test_engineering_mode_allows_provisional_defaults():
    # OPEN finals are fine in engineering mode (provisional defaults apply).
    assert_no_gating_questions(_cfg(), pipeline="mock")
    assert_no_gating_questions(_cfg(pipeline="attr"), pipeline="attr")


def test_engineering_mode_reports_every_provisional_resolution():
    res = report_provisional_resolutions(_cfg())
    assert len(res) == 11
    assert all(r["status"] == "PROVISIONAL_ENGINEERING_DEFAULT" for r in res)
    assert {r["id"] for r in res} == {f"Q{i}" for i in range(1, 12)}


def test_scientific_mode_reports_no_provisional_resolutions():
    assert report_provisional_resolutions(_cfg(mode={"name": MODE_SCIENTIFIC})) == []


def test_scientific_mode_refuses_synthetic_components():
    cfg = _cfg(mode={"name": MODE_SCIENTIFIC})
    with pytest.raises(ValueError, match="synthetic"):
        assert_no_gating_questions(cfg, pipeline="mock")
    cfg = _cfg(mode={"name": MODE_SCIENTIFIC}, pipeline="attr",
               model={"name": "mock", "synthetic": True, "hf_revision": "abc"})
    with pytest.raises(ValueError, match="synthetic"):
        assert_no_gating_questions(cfg, pipeline="attr")


def test_scientific_mode_blocked_while_questions_open():
    cfg = _cfg(mode={"name": MODE_SCIENTIFIC}, pipeline="attr")
    with pytest.raises(ValueError) as exc:
        assert_no_gating_questions(cfg, pipeline="attr")
    message = str(exc.value)
    assert "Q1" in message and "Q5" in message and "Q6" in message
    assert PI_DECISIONS_DOC in message


def test_scientific_mode_passes_when_all_gating_questions_resolved():
    cfg = _resolved_cfg(mode={"name": MODE_SCIENTIFIC})
    assert_no_gating_questions(cfg, pipeline="attr")  # must not raise


def test_engineering_limits_refuse_stage_b_freeze():
    cfg = _cfg(**{"stage_b": {"freeze_approved": True}})
    with pytest.raises(ValueError, match="REFUSES the Stage B freeze"):
        assert_engineering_dry_run_limits(cfg, stage="stageB")
    with pytest.raises(ValueError, match="REFUSES the Stage B freeze"):
        assert_engineering_dry_run_limits(_cfg(freeze=True), stage="stageB")


def test_engineering_limits_allow_draft_stage_b():
    assert_engineering_dry_run_limits(_cfg(), stage="stageB")  # no freeze requested


def test_engineering_limits_refuse_stage_c_on_non_synthetic():
    """A real model in engineering mode can never reach Stage C."""
    with pytest.raises(ValueError, match="REFUSES Stage C real compression"):
        assert_engineering_dry_run_limits(_cfg(), stage="stageC")


def test_engineering_limits_refuse_stage_d_on_non_synthetic():
    with pytest.raises(ValueError, match="REFUSES Stage D real compression"):
        assert_engineering_dry_run_limits(_cfg(), stage="stageD")


def _synthetic_cfg(**overrides):
    """A stack that provably cannot touch a real model, dataset or corpus."""
    base = {
        "pipeline": "mock",
        "model": {"name": "mock", "synthetic": True, "hf_revision": None},
        "task": {"name": "synthetic-ioi", "synthetic": True, "pi_confirmed": False},
    }
    return _cfg(**{**base, **overrides})


def test_engineering_limits_allow_stage_c_on_a_provably_synthetic_stack():
    """Algorithm 1 must be exercisable end-to-end before any GPU hour is spent.

    A synthetic Stage C compresses a numpy toy and stamps its outputs evidence=false;
    it has exactly the status of the Stage A and Stage B dry-runs that were always
    permitted. The prohibition that matters — no REAL compression outside scientific
    mode — is what the non-synthetic tests above pin down.
    """
    assert_engineering_dry_run_limits(_synthetic_cfg(), stage="stageC")  # must not raise
    assert_engineering_dry_run_limits(_synthetic_cfg(), stage="stageD")


@pytest.mark.parametrize(
    "override",
    [{"model": {"name": "pythia160m", "synthetic": False}},
     {"task": {"name": "ioi", "synthetic": False}},
     {"pipeline": "attr"}],
    ids=["real-model", "real-task", "real-pipeline"],
)
def test_any_single_real_component_re_blocks_stage_c(override):
    """All three must be synthetic; one real component is enough to refuse."""
    with pytest.raises(ValueError, match="REFUSES Stage C"):
        assert_engineering_dry_run_limits(_synthetic_cfg(**override), stage="stageC")


def test_engineering_limits_refuse_frozen_writes():
    with pytest.raises(ValueError, match="REFUSES writes under frozen"):
        assert_engineering_dry_run_limits(_cfg(run_root="frozen/pythia160m"), stage="stageB")
    with pytest.raises(ValueError, match="REFUSES writes under frozen"):
        assert_engineering_dry_run_limits(_cfg(frozen_root="frozen/pythia160m"), stage="stageD")


def test_engineering_limits_noop_in_scientific_mode():
    cfg = _cfg(mode={"name": MODE_SCIENTIFIC}, **{"stage_b": {"freeze_approved": True}})
    assert_engineering_dry_run_limits(cfg, stage="stageB")  # scientific handles its own guards
