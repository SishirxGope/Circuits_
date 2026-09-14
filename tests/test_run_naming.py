# [AI-GEN] agent=OpenCode date=2026-08-07 task=Unit tests: run-naming convention (CLAUDE.md §4)
# reviewed-by: PENDING

import pytest

from src.common.run_naming import build_configset, build_run_name, validate_run_name

VALID = [
    "20261012_stageB_pythia160m_ioi_null-matchedmag-int4_B16xS5xR20_seed0",
    "20261103_stageC_gemma2-2b_greaterthan_rtn-int6_B16xS5_seed3",
    "20261012_stageA_pythia160m_ioi_dense_B16xS5_seed0",
    "20260901_stageD_llama32-1b_docstring_null-seedthr_B8xS3_seed2",
]


@pytest.mark.parametrize("name", VALID)
def test_valid_names_pass(name):
    parsed = validate_run_name(name)
    assert parsed["seed"] == int(name.rsplit("_", 1)[1][4:])


def test_parse_returns_fields():
    parsed = validate_run_name(VALID[0])
    assert parsed == {
        "date": "20261012",
        "stage": "stageB",
        "model": "pythia160m",
        "task": "ioi",
        "setting": "null-matchedmag-int4",
        "configset": "B16xS5xR20",
        "seed": 0,
    }


@pytest.mark.parametrize(
    "name",
    [
        "20261012_stageX_pythia160m_ioi_dense_B16xS5_seed0",  # bad stage
        "20261012_stageA_pythia160m_ioi_dense_B16xS5",  # missing seed
        "20261012_stageA_pythia160m_ioi_dense_B16xS5_seedX",  # non-integer seed
        "20261012_stageA_pythia160m_ioi_dense_B16xR20_seed0",  # configset missing S
        "20261012_stageA_pythia160m_ioi_dense_B16xS5_seed0_extra",  # 8 tokens
        "20261012_stageA_pythia160m_ioi_dense_B16xS5_seed-1",  # negative seed
        "20261012_stageA_pythia_160m_ioi_dense_B16xS5_seed0",  # underscore in model token
        "not-a-run-name",
    ],
)
def test_invalid_names_fail(name):
    with pytest.raises(ValueError):
        validate_run_name(name)


def test_build_configset():
    assert build_configset(16, 5) == "B16xS5"
    assert build_configset(16, 5, R=20) == "B16xS5xR20"
    with pytest.raises(ValueError):
        build_configset(0, 5)
    with pytest.raises(ValueError):
        build_configset(16, 5, R=0)


def test_build_run_name_roundtrip():
    name = build_run_name("20261012", "stageA", "pythia160m", "ioi", "dense", "B16xS5", 0)
    assert validate_run_name(name)["seed"] == 0
    with pytest.raises(ValueError):
        build_run_name("20261012", "stageZ", "pythia160m", "ioi", "dense", "B16xS5", 0)
