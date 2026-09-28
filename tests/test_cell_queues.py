# [AI-GEN] agent=Claude date=2026-09-29 task=Guard the generated run queues (gen_cells.py)
# reviewed-by: PENDING

"""The generated cell queues (``deploy/shared/gen_cells.py``).

These files are what the runner scripts iterate, so an error here is an error in every
run. Two things they must get right:

1. **No structurally impossible cell is emitted.** ``greater_than`` compares a year at its
   century boundary, so it needs ``" CCYY"`` to tokenize as exactly ``[" CC", "YY"]`` and
   needs all 100 two-digit strings to be single tokens. Neither holds for Gemma-2 or
   Llama-3.2. Emitting them anyway put 22 of 88 entries in the queue that fail one at a
   time, mid-run, after the model has been loaded.

2. **Every emitted line carries the overrides both stages read.** In particular
   ``+stage_c.compressor_kwargs.*``, which is the key ``compressor_for`` reads for Stage B
   *and* Stage C; without it every cell would silently fall back to the compressor's
   default (e.g. an rtn_int8 cell quantizing at 4 bits).

The reduced grid is a reportable fact, not a tidy-up: greater_than is Pythia-only here,
which weakens the task-generality claim and has to appear in the paper.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "deploy" / "shared" / "gen_cells.py"
QUEUES = [
    REPO / "deploy" / "plan_a_local_pc" / "cells_stageb.txt",
    REPO / "deploy" / "plan_a_local_pc" / "cells_stagec.txt",
    REPO / "deploy" / "plan_b_dgx_spark" / "cells_stageb.txt",
    REPO / "deploy" / "plan_b_dgx_spark" / "cells_stagec.txt",
]


@pytest.fixture(scope="module")
def gen_cells():
    spec = importlib.util.spec_from_file_location("_gen_cells_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize("queue", QUEUES, ids=lambda p: f"{p.parent.name}/{p.name}")
class TestEveryQueue:
    def test_no_impossible_model_task_pair_is_emitted(self, queue, gen_cells):
        for line in _rows(queue):
            for (model, task), reason in gen_cells.IMPOSSIBLE_CELLS.items():
                assert not (f"model={model} " in line and f"task={task} " in line), (
                    f"{queue.name} still queues {model}/{task}, which cannot run: {reason}"
                )

    def test_every_row_carries_the_compressor_kwargs_both_stages_read(self, queue):
        for line in _rows(queue):
            assert "+stage_c.compressor_kwargs." in line, (
                f"missing compressor_kwargs, so this cell would use the compressor's "
                f"default instead of its own level: {line}"
            )

    def test_every_row_names_a_family_and_a_level(self, queue):
        for line in _rows(queue):
            assert "compression_family=" in line
            assert "compression_level=" in line

    def test_the_stage_matches_the_filename(self, queue):
        expected = "stageB" if queue.name.endswith("stageb.txt") else "stageC"
        for line in _rows(queue):
            assert line.startswith(f"stage={expected} "), line

    def test_the_header_count_matches_the_rows(self, queue):
        text = queue.read_text(encoding="utf-8")
        declared = next(
            int(part) for line in text.splitlines() if line.startswith("#")
            for part in line.replace("=", " ").split() if part.isdigit() and f"= {part} cells" in line
        )
        assert declared == len(_rows(queue)), (
            f"{queue.name} declares {declared} cells but contains {len(_rows(queue))}"
        )


class TestTheGridSize:
    def test_plan_b_is_sixty_six_cells_not_eighty_eight(self, gen_cells):
        """4 models x 2 tasks x 11 cells = 88, minus the 22 impossible greater_than cells."""
        pairs = gen_cells.viable_pairs(gen_cells.PLAN_B_MODELS)
        assert len(pairs) == 6
        assert len(pairs) * len(gen_cells.CELLS) == 66

    def test_plan_a_is_unaffected_because_pythia_can_do_both_tasks(self, gen_cells):
        pairs = gen_cells.viable_pairs(gen_cells.PLAN_A_MODELS)
        assert len(pairs) == len(gen_cells.PLAN_A_MODELS) * len(gen_cells.TASKS)

    def test_greater_than_survives_only_on_pythia(self, gen_cells):
        pairs = gen_cells.viable_pairs(gen_cells.PLAN_B_MODELS)
        models = {model for model, task in pairs if task == "greater_than"}
        assert models == {"pythia160m", "pythia410m"}

    def test_ioi_survives_on_every_model(self, gen_cells):
        pairs = gen_cells.viable_pairs(gen_cells.PLAN_B_MODELS)
        models = {model for model, task in pairs if task == "ioi"}
        assert models == set(gen_cells.PLAN_B_MODELS)

    def test_each_exclusion_records_why(self, gen_cells):
        """A silently dropped cell is indistinguishable from a bug in the generator."""
        assert gen_cells.IMPOSSIBLE_CELLS
        for (model, task), reason in gen_cells.IMPOSSIBLE_CELLS.items():
            assert task in gen_cells.TASKS
            assert model in gen_cells.PLAN_B_MODELS
            assert len(reason) > 30, f"{model}/{task} exclusion is not explained"


class TestTheQueuesOnDiskAreCurrent:
    def test_regenerating_would_not_change_them(self, gen_cells, tmp_path, monkeypatch):
        """Hand-edited or stale queues would run a different grid than the generator
        describes, and the files say 'do not hand-edit'."""
        before = {q: q.read_text(encoding="utf-8") for q in QUEUES}
        gen_cells.write_queues()
        try:
            for q in QUEUES:
                assert q.read_text(encoding="utf-8") == before[q], (
                    f"{q.name} on disk differs from what gen_cells.py generates; "
                    "re-run `python deploy/shared/gen_cells.py queues`"
                )
        finally:
            for q, text in before.items():
                q.write_text(text, encoding="utf-8")

    def test_the_exclusions_are_documented_in_the_file_itself(self):
        """Whoever reads the queue should see why it is 66 and not 88."""
        for queue in QUEUES:
            text = queue.read_text(encoding="utf-8")
            if "gemma2_2b" in text:
                assert "EXCLUDED as structurally impossible" in text, queue.name
