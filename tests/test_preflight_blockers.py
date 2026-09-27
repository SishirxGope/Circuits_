# [AI-GEN] agent=Claude date=2026-09-27 task=Tests for the preflight gate that decides whether science may run
# reviewed-by: PENDING

"""The preflight gate (deploy/shared/preflight_blockers.py, deploy/BLOCKERS.md).

``_common.sh::assert_preflight`` refuses to start the science queue while this script
reports any blocker unimplemented, so both failure directions are expensive:

- a false **BLOCKED** keeps the gate shut after the work is genuinely done (this happened:
  the compressor probe passed a bare sentinel object, which a correct compressor refuses
  with ``NotImplementedError`` — the same signal the probe uses to mean "still a stub", so
  RTN read as BLOCKED after its real path landed);
- a false **OK** lets a queue start against a stub and waste hours of Spark time.

The script lives under ``deploy/`` rather than in a package, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "deploy" / "shared" / "preflight_blockers.py"


@pytest.fixture(scope="module")
def preflight():
    spec = importlib.util.spec_from_file_location("_preflight_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_compressor_probe_is_handed_a_real_torch_module(preflight):
    """A bare sentinel cannot distinguish "refuses junk" from "not implemented"."""
    torch = pytest.importorskip("torch")
    probe = preflight._probe_model()
    assert isinstance(probe, torch.nn.Module), (
        "the probe must be a real torch module, or an implemented compressor will "
        "correctly refuse it and be misreported as a stub"
    )

    from src.compression.rtn import quantizable_parameters

    assert quantizable_parameters(probe), (
        "the probe carries no projection-shaped parameter, so a compressor would have "
        "nothing to act on and could pass without doing anything"
    )


def test_b2_status_matches_which_families_are_actually_implemented(preflight):
    """RTN and magnitude landed 2026-09-27. Wanda, GPTQ and AWQ all need the Q7
    calibration cache, so they must keep gating the queue."""
    pytest.importorskip("torch")
    status = {
        label.replace("B2 compressor: ", ""): ok
        for label, ok, _ in preflight.check_compressors()
    }
    for family in ("rtn", "magnitude"):
        assert status[family] is True, f"{family} is implemented; a BLOCKED keeps the gate shut"
    for family in ("gptq", "awq", "wanda"):
        assert status[family] is False, f"{family} still needs calibration data (Q7)"


def test_an_implemented_family_reads_ok_by_succeeding_not_by_erroring(preflight):
    """Guards the false-OK class of bug directly.

    ``_probe`` scores any non-NotImplementedError as "real path entered", so a compressor
    that blew up on the probe's inputs would read as implemented. Magnitude pruning did
    exactly that when the probe carried no ``embed.W_E``: it raised ValueError and scored
    OK while pruning nothing.
    """
    pytest.importorskip("torch")
    details = {
        label.replace("B2 compressor: ", ""): detail
        for label, ok, detail in preflight.check_compressors()
        if ok
    }
    for family, detail in details.items():
        assert detail == "returned without error", (
            f"{family} reads OK via {detail!r} rather than by completing; the probe inputs "
            "are wrong for it and the OK is not evidence the path works"
        )


def test_the_probe_model_carries_the_names_the_pruners_special_case(preflight):
    """embed.W_E joins the threshold pool and unembed.W_U is excluded, so a probe missing
    them cannot exercise the pruning rule at all."""
    pytest.importorskip("torch")
    names = {n for n, _ in preflight._probe_model().named_parameters()}
    assert "embed.W_E" in names
    assert "unembed.W_U" in names


def test_a_family_needs_both_apply_and_weight_delta(preflight):
    """Stage C needs apply(); Stage B needs weight_delta() for the null magnitudes.

    Either one alone is unusable, so the probe must require both. Verified through the
    real reporting path with a compressor that implements only half the protocol.
    """
    pytest.importorskip("torch")

    class _HalfDone:
        def apply(self, model, cfg):
            return model

        def weight_delta(self, model, cfg):
            raise NotImplementedError("stub")

    inst = _HalfDone()
    apply_ok, _ = preflight._probe(inst.apply, preflight._probe_model(), None)
    delta_ok, detail = preflight._probe(inst.weight_delta, preflight._probe_model(), None)
    assert apply_ok and not delta_ok
    assert "NotImplementedError" in detail


def test_the_null_perturber_reads_implemented(preflight):
    """B1 closed 2026-09-27 (PI-approved): the Perturber now handles torch modules.

    The probe must hand it a real module - with a bare sentinel a correct perturber
    raises NotImplementedError and would be misreported as a stub, exactly as RTN was.
    """
    pytest.importorskip("torch")
    _, ok, detail = preflight.check_null_perturber()
    assert ok is True, f"B1 reads blocked: {detail}"


def test_the_b1_probe_actually_perturbs_something(preflight):
    """Guards against a false OK: a perturber that returned the model untouched would
    also "return without error" and read as implemented."""
    torch = pytest.importorskip("torch")
    from src.common.seeding import create_seed_generator
    from src.science.matched_magnitude import MatchedMagnitudePerturber

    model = preflight._probe_model()
    magnitudes = {name: 1.0 for name, _ in model.named_parameters()}
    rng, _ = create_seed_generator(0)
    perturbed = MatchedMagnitudePerturber().apply(model, magnitudes, rng)

    before = dict(model.named_parameters())
    moved = [n for n, p in perturbed.named_parameters() if not torch.equal(before[n], p)]
    assert set(moved) == set(magnitudes), "the probe model was not actually perturbed"


def test_the_gate_is_still_shut_overall(preflight):
    """Whole-script verdict: science must not be clear to run yet."""
    pytest.importorskip("torch")
    results = [preflight.check_null_perturber(), *preflight.check_compressors(),
               preflight.check_chance_floor()]
    assert not all(ok for _, ok, _ in results), (
        "the preflight gate reads clear; if that is intended, the remaining blockers in "
        "deploy/BLOCKERS.md should have been closed first"
    )
