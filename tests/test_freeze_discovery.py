# [AI-GEN] agent=Claude date=2026-09-30 task=The Stage B queue -> run name -> freeze config chain, end to end without a model
# reviewed-by: PENDING

"""Can the freeze find the nulls the Stage B queue produces?

Three links, each of which was broken before 2026-09-30:

1. Every Stage B run inherited ``setting: dense``, so the 11 nulls of a (model, task)
   shared one run name and ``make_freeze_config.py`` - which finds a cell by the cell
   token in the run name - matched none of them.
2. ``make_freeze_config.py`` counted the structurally impossible greater_than pairs, so a
   Plan B config always had 22 cells "MISSING" and was refused.
3. It filed Gemma and Llama under ``gemma2-2b`` / ``llama32-1b``; Stage C looks under the
   config ``name:`` (``gemma-2-2b`` / ``llama-3.2-1b``) and would never have found them.

The tests compose every queue line with Hydra exactly as ``_run_one.sh`` does, derive the
run name exactly as ``run_stage_b`` does, and require the freeze config to come out whole.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHARED = REPO / "deploy" / "shared"
QUEUE_B = REPO / "deploy" / "plan_b_dgx_spark" / "cells_stageb.txt"
QUEUE_A = REPO / "deploy" / "plan_b_dgx_spark" / "cells_stagea.txt"
RUN_ONE = REPO / "deploy" / "plan_b_dgx_spark" / "_run_one.sh"


def _load(name: str):
    import sys

    sys.path.insert(0, str(SHARED))
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", SHARED / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def freeze_cfg():
    return _load("make_freeze_config")


def _rows(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _cell_overrides() -> list[str]:
    """The fixed overrides _run_one.sh puts around every cell."""
    script = RUN_ONE.read_text(encoding="utf-8")
    block = script.split('"$PY" "$RUNNER"', 1)[1].split("then", 1)[0]
    return [tok for tok in block.replace("\\", " ").split() if "=" in tok and not tok.startswith(("$", ">"))]


def _compose(line: str) -> dict:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = [*_cell_overrides(), *line.split()]
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        return OmegaConf.to_container(compose("config", overrides=overrides), resolve=True)


def _stage_b_run_name(resolved: dict) -> str:
    """Exactly run_stage_b's naming."""
    from src.common.run_naming import build_configset, resolve_run_name

    return resolve_run_name(
        # the same local-date call run_stage_b makes, so the names match what it writes
        resolved.get("run_name"), datetime.date.today().strftime("%Y%m%d"), "stageB",  # noqa: DTZ011
        resolved["model"]["name"], resolved["task"]["name"],
        str(resolved.get("setting", "null-matchedmag")),
        build_configset(int(resolved["ensemble"]["B"]), int(resolved["ensemble"]["S"]),
                        R=int(resolved["nulls"]["R"])),
        int(resolved.get("seed", 0)),
    )


class TestModelNames:
    def test_names_come_from_the_model_configs(self, freeze_cfg):
        assert freeze_cfg.MODEL_NAME == {
            "pythia160m": "pythia-160m", "pythia410m": "pythia-410m",
            "gemma2_2b": "gemma-2-2b", "llama32_1b": "llama-3.2-1b",
        }


class TestTheStageBQueue:
    def test_every_line_composes_to_a_distinct_run_name_that_names_its_cell(self, freeze_cfg):
        pytest.importorskip("hydra")
        names = {}
        for line in _rows(QUEUE_B):
            resolved = _compose(line)
            assert resolved["stage"] == "stageB" and resolved["nulls"]["R"] > 0
            name = _stage_b_run_name(resolved)
            cell = resolved["stage_c"]["cell"]
            assert freeze_cfg._norm(cell) in freeze_cfg._norm(name), (name, cell)
            names[name] = line
        assert len(names) == len(_rows(QUEUE_B)) == 66, "two cells would share one run name"

    def test_each_model_gets_the_r_the_q3_rule_gave_it(self, freeze_cfg):
        """HUMAN_DECISIONS.md Q3, 2026-10-01: Gemma-2-2B R = 10, every other model R = 20."""
        pytest.importorskip("hydra")
        for line in _rows(QUEUE_B):
            resolved = _compose(line)
            expected = 10 if "model=gemma2_2b " in line else 20
            assert resolved["nulls"]["R"] == expected, line
            assert f"xR{expected}_" in _stage_b_run_name(resolved)

    def test_the_stage_a_queue_is_one_dense_reference_per_viable_pair(self):
        pytest.importorskip("hydra")
        rows = _rows(QUEUE_A)
        assert len(rows) == 6
        for line in rows:
            resolved = _compose(line)
            assert resolved["stage"] == "stageA" and resolved["setting"] == "dense"
            assert resolved["compression_family"] == "dense"


class TestTheFreezeConfig:
    def _fake_stage_b_runs(self, root: Path, lines: list[str]) -> None:
        for line in lines:
            (root / _stage_b_run_name(_compose(line))).mkdir(parents=True)

    def test_the_whole_plan_b_grid_is_found(self, freeze_cfg, tmp_path, monkeypatch):
        pytest.importorskip("hydra")
        runs = tmp_path / "runs"
        self._fake_stage_b_runs(runs, _rows(QUEUE_B))
        out = tmp_path / "freeze.json"
        monkeypatch.setattr(freeze_cfg, "REPO", tmp_path)
        assert freeze_cfg.main(["--plan", "b", "--out", str(out)]) == 0
        cfg = json.loads(out.read_text(encoding="utf-8"))
        cells = cfg["stage_b"]["cells"]
        assert len(cells) == 66 and cfg["stage_b"]["freeze_approved"] is False
        assert {c["model"] for c in cells} == {"pythia-160m", "pythia-410m", "gemma-2-2b", "llama-3.2-1b"}
        assert not any(c["task"] == "greater_than" and c["model"] in ("gemma-2-2b", "llama-3.2-1b")
                       for c in cells)
        assert len({(c["model"], c["task"], c["cell"]) for c in cells}) == 66

    def test_one_model_can_be_frozen_deliberately(self, freeze_cfg, tmp_path, monkeypatch):
        pytest.importorskip("hydra")
        runs = tmp_path / "runs"
        self._fake_stage_b_runs(runs, [r for r in _rows(QUEUE_B) if "model=pythia160m " in r])
        out = tmp_path / "freeze.json"
        monkeypatch.setattr(freeze_cfg, "REPO", tmp_path)
        assert freeze_cfg.main(["--plan", "b", "--models", "pythia160m", "--out", str(out)]) == 0
        cells = json.loads(out.read_text(encoding="utf-8"))["stage_b"]["cells"]
        assert len(cells) == 22 and {c["model"] for c in cells} == {"pythia-160m"}

    def test_a_partial_grid_is_still_refused_without_models(self, freeze_cfg, tmp_path, monkeypatch):
        pytest.importorskip("hydra")
        runs = tmp_path / "runs"
        self._fake_stage_b_runs(runs, [r for r in _rows(QUEUE_B) if "model=pythia160m " in r])
        out = tmp_path / "freeze.json"
        monkeypatch.setattr(freeze_cfg, "REPO", tmp_path)
        assert freeze_cfg.main(["--plan", "b", "--out", str(out)]) == 1
        assert not out.exists()

    def test_an_unknown_model_is_refused(self, freeze_cfg, tmp_path, monkeypatch):
        monkeypatch.setattr(freeze_cfg, "REPO", tmp_path)
        assert freeze_cfg.main(["--plan", "a", "--models", "gemma2_2b", "--out", str(tmp_path / "f.json")]) == 2
