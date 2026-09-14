# [AI-GEN] agent=Claude date=2026-08-09 task=Regressions for the two defects that blocked running the code
# reviewed-by: PENDING

"""Run-directory allocation and run-name resolution.

Both defects here were found by actually executing the Hydra entrypoints for the first
time (hydra-core had never been installed, so that whole surface was unverified):

1. Two scientifically different configs that share grid coordinates — swapping the band
   cutoffs, R, the distance, the comparison scheme — produced the SAME run directory
   and a hard FileExistsError whose only suggested escapes were "use a fresh seed"
   (which changes the science) or "a new date" (which means waiting a day).
2. Every Stage B/C/D run launched through Hydra was named `B4xS2` instead of
   `B4xS2xR3`, because the root config interpolated `${ensemble.configset}`, which
   never contains R. CLAUDE.md §4 requires R in the configset token precisely so the
   null count is legible from a null run's name.
"""

import json

import pytest

from src.common.run_dir import RunDirExists, allocate_run_dir
from src.common.run_naming import build_configset, build_run_name, resolve_run_name

NAME = "20260809_stageA_mock_synthetic-ioi_dense_B4xS2_seed0"


def _finish_run(run_dir, config_hash):
    """Write what a completed run leaves behind."""
    (run_dir / "run_meta.json").write_text(json.dumps({"config_hash": config_hash}), encoding="utf-8")


class TestRunDirAllocation:
    def test_first_run_gets_the_plain_name(self, tmp_path):
        d = allocate_run_dir(tmp_path, NAME, "a" * 64)
        assert d.name == NAME and d.is_dir()

    def test_a_different_config_gets_a_disambiguated_directory(self, tmp_path):
        """The case that blocked the PI: same grid coordinates, different science."""
        first = allocate_run_dir(tmp_path, NAME, "a" * 64)
        _finish_run(first, "a" * 64)

        second = allocate_run_dir(tmp_path, NAME, "b" * 64)
        assert second != first
        assert second.name.startswith(NAME) and second.name.endswith("__cfgbbbbbbbb")
        assert first.is_dir() and second.is_dir(), "neither run may be destroyed"

    def test_an_identical_config_is_still_refused(self, tmp_path):
        """Immutability is preserved: a true re-run never overwrites (AI_RULES.md 1.2)."""
        first = allocate_run_dir(tmp_path, NAME, "a" * 64)
        _finish_run(first, "a" * 64)
        with pytest.raises(RunDirExists, match="exact config already exists"):
            allocate_run_dir(tmp_path, NAME, "a" * 64)

    def test_three_different_configs_coexist(self, tmp_path):
        dirs = []
        for h in ("a" * 64, "b" * 64, "c" * 64):
            d = allocate_run_dir(tmp_path, NAME, h)
            _finish_run(d, h)
            dirs.append(d)
        assert len({d.name for d in dirs}) == 3
        assert all(d.is_dir() for d in dirs)

    def test_repeating_a_disambiguated_config_is_refused(self, tmp_path):
        _finish_run(allocate_run_dir(tmp_path, NAME, "a" * 64), "a" * 64)
        _finish_run(allocate_run_dir(tmp_path, NAME, "b" * 64), "b" * 64)
        with pytest.raises(RunDirExists, match="exact config already exists"):
            allocate_run_dir(tmp_path, NAME, "b" * 64)

    def test_an_interrupted_run_is_still_identified_by_its_config_dump(self, tmp_path):
        """A run that died before writing run_meta.json still has resolved_config.json."""
        from src.common.hashing import hash_dict

        cfg = {"seed": 0, "model": {"name": "mock"}}
        d = allocate_run_dir(tmp_path, NAME, hash_dict(cfg))
        (d / "resolved_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        with pytest.raises(RunDirExists):
            allocate_run_dir(tmp_path, NAME, hash_dict(cfg))

    def test_an_unidentifiable_directory_does_not_block_a_new_run(self, tmp_path):
        """An empty leftover directory must not wedge the pipeline."""
        (tmp_path / NAME).mkdir(parents=True)
        d = allocate_run_dir(tmp_path, NAME, "a" * 64)
        assert d.name.endswith("__cfgaaaaaaaa")


class TestRunNameResolution:
    ARGS = ("20260809", "stageB", "mock", "synthetic-ioi", "null-matchedmag")

    def test_derived_name_encodes_R_for_null_runs(self):
        """CLAUDE.md §4: the configset token must encode B, S AND R."""
        name = resolve_run_name(None, *self.ARGS, build_configset(4, 2, R=3), 0)
        assert name.endswith("_B4xS2xR3_seed0")

    def test_stage_a_derived_name_has_no_R(self):
        name = resolve_run_name(None, "20260809", "stageA", "mock", "synthetic-ioi",
                                "dense", build_configset(4, 2), 0)
        assert "_B4xS2_seed0" in name

    def test_a_supplied_name_hiding_R_is_refused(self):
        """The exact bug: Hydra supplied `B4xS2` for a run with R=3 nulls."""
        stale = build_run_name(*self.ARGS, "B4xS2", 0)
        with pytest.raises(ValueError, match="configset"):
            resolve_run_name(stale, *self.ARGS, build_configset(4, 2, R=3), 0)

    def test_the_refusal_shows_the_correct_name(self):
        stale = build_run_name(*self.ARGS, "B4xS2", 0)
        with pytest.raises(ValueError, match="B4xS2xR3"):
            resolve_run_name(stale, *self.ARGS, build_configset(4, 2, R=3), 0)

    def test_a_matching_supplied_name_is_accepted(self):
        good = build_run_name(*self.ARGS, "B4xS2xR3", 0)
        assert resolve_run_name(good, *self.ARGS, build_configset(4, 2, R=3), 0) == good

    @pytest.mark.parametrize(
        "bad_name,field",
        [("20260809_stageC_mock_synthetic-ioi_null-matchedmag_B4xS2xR3_seed0", "stage"),
         ("20260809_stageB_gemma_synthetic-ioi_null-matchedmag_B4xS2xR3_seed0", "model"),
         ("20260809_stageB_mock_ioi_null-matchedmag_B4xS2xR3_seed0", "task"),
         ("20260809_stageB_mock_synthetic-ioi_dense_B4xS2xR3_seed0", "setting"),
         ("20260809_stageB_mock_synthetic-ioi_null-matchedmag_B4xS2xR3_seed7", "seed")],
    )
    def test_every_component_is_checked(self, bad_name, field):
        with pytest.raises(ValueError, match=field):
            resolve_run_name(bad_name, *self.ARGS, build_configset(4, 2, R=3), 0)

    def test_a_malformed_supplied_name_is_still_rejected(self):
        with pytest.raises(ValueError, match="malformed run name"):
            resolve_run_name("not-a-run-name", *self.ARGS, build_configset(4, 2, R=3), 0)


class TestTheRootConfigNoLongerHardcodesTheName:
    def test_run_name_is_derived_not_interpolated(self):
        """`${ensemble.configset}` can never contain R, so it cannot name a null run."""
        import pathlib

        import yaml

        cfg = yaml.safe_load(
            (pathlib.Path(__file__).resolve().parents[1] / "configs" / "config.yaml").read_text(encoding="utf-8")
        )
        assert cfg["run_name"] is None, (
            "configs/config.yaml sets run_name again — if it interpolates "
            "${ensemble.configset}, every Stage B/C/D run is misnamed without R"
        )
