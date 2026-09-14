# [AI-GEN] agent=OpenCode date=2026-08-07 task=Unit tests: core/contingent/noise decomposition skeleton
# reviewed-by: PENDING

import pytest

from src.science.decompose import CORE, CONTINGENT, NOISE, decompose


def test_respects_config_cutoffs():
    bands = decompose({"a": 1.0, "b": 0.9, "c": 0.5, "d": 0.1, "e": 0.05}, core_threshold=0.9, noise_threshold=0.1)
    assert bands == {"a": CORE, "b": CORE, "c": CONTINGENT, "d": NOISE, "e": NOISE}


def test_boundaries_are_inclusive():
    # s == core_threshold -> core ; s == noise_threshold -> noise
    bands = decompose({"x": 0.9, "y": 0.1}, core_threshold=0.9, noise_threshold=0.1)
    assert bands["x"] == CORE
    assert bands["y"] == NOISE


def test_unresolved_cutoffs_raise_informative_error():
    with pytest.raises(ValueError, match="core_threshold|noise_threshold"):
        decompose({"a": 0.5}, core_threshold=None, noise_threshold=None)
    with pytest.raises(ValueError, match="QUESTION FOR PI"):
        decompose({"a": 0.5}, core_threshold=None, noise_threshold=None)
    with pytest.raises(ValueError):
        decompose({"a": 0.5}, core_threshold=0.9, noise_threshold=None)


def test_degenerate_cutoffs_raise():
    with pytest.raises(ValueError):
        decompose({"a": 0.5}, core_threshold=0.3, noise_threshold=0.5)
    with pytest.raises(ValueError):
        decompose({"a": 0.5}, core_threshold=0.9, noise_threshold=0.9)


def test_deterministic_sorted_output():
    f = {"z": 0.2, "a": 0.8, "m": 0.5}
    r1 = decompose(f, core_threshold=0.7, noise_threshold=0.3)
    r2 = decompose(f, core_threshold=0.7, noise_threshold=0.3)
    assert list(r1.items()) == list(r2.items()) == sorted(r1.items())


def test_empty_input():
    assert decompose({}, core_threshold=0.9, noise_threshold=0.1) == {}
