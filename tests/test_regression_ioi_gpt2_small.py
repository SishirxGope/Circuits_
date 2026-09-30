# [AI-GEN] agent=OpenCode date=2026-08-07 task=Phase-1 exit-gate regression test (IOI on GPT-2 small)
# modified: [AI-GEN] agent=Claude date=2026-09-29 task=B6 - PI file carries edges AND tolerance; a pass leaves a record preflight can verify
# modified: [AI-GEN] agent=Claude date=2026-09-29 task=review fix - a gate run removes the previous pass record first
# modified: [AI-GEN] agent=Claude date=2026-09-30 task=B6 - extraction wired: Stage A dense-node on GPT-2 small, core band vs ACDC's IOI ground truth
# modified: [AI-GEN] agent=Claude date=2026-09-30 task=B6 - PI replaced the Jaccard criterion (size-capped) with precision + enrichment on fresh seeds
# reviewed-by: PENDING

"""Phase-1 exit gate (ARCHITECTURE.md §6): reproduce one published reference circuit
before any real Stage A run is trusted.

Gate contract:
- Target: IOI circuit on GPT-2 small (Wang et al., ICLR 2023) at its pinned checkpoint
  (configs/model/gpt2_small.yaml), extracted by the pipeline every grid cell runs:
  Stage A, ``pipeline=dense-node`` (edge attribution patching), the pre-registered
  B=16 x S=5 ensemble, 300 IOI prompts. The extracted set is that run's CORE band
  (s(e) = 1). ``GATE_OVERRIDES`` is the cells' override list plus model and task, and a
  test keeps it equal to deploy/plan_b_dgx_spark/_run_one.sh, so the gate cannot drift
  from what the science runs. Only the base seed differs: the PI's file names it.
- Reference: the Wang et al. circuit as ACDC's edge-level ground truth
  (src/tasks/ioi_reference.py; proven equal to ACDC's own code in
  tests/test_ioi_reference.py), written into the PI's file by
  deploy/shared/make_ioi_reference.py.
- Criterion (PI-owned, AI_RULES.md 2.2), in the same file as the edges so neither can
  drift from the other::

      {"edges": [...], "source": "...", "decided_on": "YYYY-MM-DD",
       "criterion": {"min_precision": <share of core edges in the reference>,
                     "max_p_value": <hypergeometric P(overlap this large by chance)>,
                     "seed": <base seed of the gate's Stage A run>}}

  **History.** The first criterion (2026-09-30) was Jaccard >= 0.2 on seeds 0-4. It
  FAILED at 0.028: the core band had 37 edges, 27 of them in the 963-edge reference, and
  a 37-edge set cannot exceed a Jaccard of 37/963 = 0.038 however correct it is. The PI
  replaced it the same day, after seeing that result, with a size-robust criterion on
  FRESH seeds (5-9). docs/HUMAN_DECISIONS.md Step 7 records both; the paper must too.
- The gate is a REGRESSION test, not a scientific claim: failing it blocks Stage A
  real extraction; passing it does not validate any compression claim.

Execution state:
- The metrics and the reference-file validation are pure and tested NOW.
- The gate itself is SKIPPED until BOTH the environment variable
  RUN_IOI_GPT2_REFERENCE is set (human intent: pinned weights, a GPU) AND the PI's
  reference file exists. Then it runs Stage A on GPT-2 small (a real run directory under
  runs/) and compares.
- A pass writes ``PASS_RECORD``, which the preflight (deploy/shared/preflight_blockers.py,
  check B6) verifies against the reference file's hash. That record - not the presence
  of this file, and not the word "skip" in it - is what opens B6.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import pytest

REPO = Path(__file__).resolve().parents[1]
REFERENCE_EDGES_ENV = "RUN_IOI_GPT2_REFERENCE"
# Was os.path.join(dirname(__file__), "..", "..", "data", ...) - one ".." too many, which
# pointed OUTSIDE the repository, so a PI file placed in data/reference/ was never found.
REFERENCE_FILE = REPO / "data" / "reference" / "ioi_gpt2_small_edges.json"
PASS_RECORD = REPO / "data" / "reference" / "ioi_gpt2_small_gate_pass.json"
REFERENCE_EDGES_FILE = str(REFERENCE_FILE)  # kept for callers of the old name


def edge_overlap(extracted: set[str], reference: set[str]) -> float:
    """Jaccard overlap between extracted and reference edge IDs (pure). Reported, not gated."""
    if not reference:
        raise ValueError("reference edge set must be non-empty (gate is meaningless otherwise)")
    union = extracted | reference
    if not union:
        return 0.0
    return len(extracted & reference) / len(union)


def precision(extracted: set[str], reference: set[str]) -> float:
    """Share of the extracted edges that are in the reference; 0 for an empty extraction."""
    return len(extracted & reference) / len(extracted) if extracted else 0.0


def enrichment_p_value(shared: int, n_extracted: int, n_reference: int, n_universe: int) -> float:
    """P(X >= shared) for X ~ Hypergeometric(n_universe, n_reference, n_extracted), exact.

    The chance that ``n_extracted`` edges drawn at random from the ``n_universe`` possible
    edges would share at least ``shared`` with the reference. Exact integer arithmetic,
    so it stays correct far into the tail where floating-point sums would not.
    """
    if not 0 <= shared <= n_extracted <= n_universe or not 0 < n_reference <= n_universe:
        raise ValueError(f"impossible counts: shared={shared} extracted={n_extracted} "
                         f"reference={n_reference} universe={n_universe}")
    top = min(n_extracted, n_reference)
    tail = sum(
        math.comb(n_reference, k) * math.comb(n_universe - n_reference, n_extracted - k)
        for k in range(shared, top + 1)
    )
    return tail / math.comb(n_universe, n_extracted)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(path: Path = REFERENCE_FILE) -> tuple[set[str], dict[str, Any]]:
    """The PI's reference edges and gate criterion, validated. Raises ValueError on any gap."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):  # a malformed FILE, not a bad argument: ValueError
        raise ValueError(  # noqa: TRY004
            f"{path}: expected an object with 'edges' and 'criterion'; a bare edge list "
            "carries no criterion and the gate may not invent one"
        )
    if "tolerance" in data and "criterion" not in data:
        raise ValueError(
            f"{path}: this is the superseded Jaccard format. The PI replaced that criterion on "
            "2026-09-30 (docs/HUMAN_DECISIONS.md Step 7); regenerate the file with "
            "deploy/shared/make_ioi_reference.py"
        )
    missing = [key for key in ("edges", "criterion", "source", "decided_on") if key not in data]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    edges = data["edges"]
    if not isinstance(edges, list) or not edges or not all(isinstance(e, str) and "->" in e for e in edges):
        raise ValueError(f"{path}: 'edges' must be a non-empty list of 'src->dst' strings")
    if len(set(edges)) != len(edges):
        raise ValueError(f"{path}: 'edges' contains duplicates")
    crit = data["criterion"]
    if not isinstance(crit, dict):
        raise ValueError(f"{path}: 'criterion' must be an object")  # noqa: TRY004

    def number(key: str) -> float:
        value = crit.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: criterion '{key}' must be a number, got {value!r}")  # noqa: TRY004
        return float(value)

    min_precision, max_p = number("min_precision"), number("max_p_value")
    if not 0 < min_precision <= 1:
        raise ValueError(f"{path}: criterion 'min_precision' must be in (0, 1], got {min_precision}")
    if not 0 < max_p < 1:
        raise ValueError(f"{path}: criterion 'max_p_value' must be in (0, 1), got {max_p}")
    seed = crit.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"{path}: criterion 'seed' must be a non-negative integer, got {seed!r}")
    datetime.date.fromisoformat(str(data["decided_on"]))
    return set(edges), {"min_precision": min_precision, "max_p_value": max_p, "seed": seed}


# The grid cells' overrides (deploy/plan_b_dgx_spark/_run_one.sh) plus the gate's model and task.
GATE_OVERRIDES: tuple[str, ...] = (
    "mode=scientific_run",
    "pipeline=dense-node",
    "ensemble=default",
    "ensemble/decompose=final",
    "nulls=default",
    "comparison=final",
    "comparison_level=both",
    "seed=0",
    "model=gpt2_small",
    "task=ioi",
)


def gate_overrides(seed: int) -> tuple[str, ...]:
    """``GATE_OVERRIDES`` with the base seed the PI's criterion names."""
    return (*(o for o in GATE_OVERRIDES if not o.startswith("seed=")), f"seed={int(seed)}")


def extract_gate_edges(
    run_root: Path = REPO / "runs", overrides: tuple[str, ...] = GATE_OVERRIDES
) -> tuple[set[str], Path]:
    """The core band of a Stage A run composed from ``overrides``; returns (edges, run dir).

    Composed with Hydra from configs/ and run through ``run_stage_a`` itself - the same
    entrypoint, guards and outputs as any cell - then read back from its freq.parquet.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from experiments.run_stage_a import run_stage_a

    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose("config", overrides=[*overrides, f"run_root={Path(run_root).as_posix()}"])
    run_dir = Path(run_stage_a(OmegaConf.to_container(cfg, resolve=True)))
    return core_band(run_dir), run_dir


def core_band(run_dir: Path) -> set[str]:
    """Edge ids in the CORE band of a Stage A run's freq.parquet (s(e) = 1, CIRCUS C_1)."""
    from src.common.schema import read_freq_parquet

    meta = json.loads((Path(run_dir) / "run_meta.json").read_text(encoding="utf-8"))
    if meta.get("band_cutoffs_resolved") is not True:
        raise RuntimeError(f"{run_dir}: the band cutoffs were not resolved, so there is no core band")
    rows = read_freq_parquet(str(Path(run_dir) / "freq.parquet"))
    return {row["edge_id"] for row in rows if row["band"] == "core"}


def write_pass_record(result: dict[str, Any], criterion: dict[str, Any], run_dir: Path | None = None) -> None:
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip() or None
    PASS_RECORD.write_text(
        json.dumps({
            "passed": True,
            **result,
            "criterion": criterion,
            "reference_file": REFERENCE_FILE.relative_to(REPO).as_posix(),
            "reference_sha256": file_sha256(REFERENCE_FILE),
            "git_commit": commit,
            "stage_a_run_dir": None if run_dir is None else Path(run_dir).as_posix(),
            "date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_edge_overlap_pure_logic():
    ref = {"L0.H1->L2.H3", "L2.H3->L5.MLP"}
    assert edge_overlap({"L0.H1->L2.H3", "L2.H3->L5.MLP"}, ref) == pytest.approx(1.0)
    assert edge_overlap({"L0.H1->L2.H3"}, ref) == pytest.approx(1 / 2)
    assert edge_overlap(set(), ref) == 0.0
    with pytest.raises(ValueError):
        edge_overlap({"a->b"}, set())


class TestTheMetrics:
    def test_precision(self):
        ref = {"a->b", "b->c"}
        assert precision({"a->b", "x->y"}, ref) == 0.5
        assert precision(set(), ref) == 0.0  # an empty core can never pass

    def test_the_p_value_matches_a_hand_count(self):
        # 5 edges, 2 in the reference, draw 2: P(both reference) = 1 / C(5,2) = 0.1
        assert enrichment_p_value(2, 2, 2, 5) == pytest.approx(0.1)
        # P(at least one) = 1 - C(3,2)/C(5,2) = 0.7
        assert enrichment_p_value(1, 2, 2, 5) == pytest.approx(0.7)
        assert enrichment_p_value(0, 2, 2, 5) == pytest.approx(1.0)

    def test_it_stays_exact_deep_in_the_tail(self):
        # the failed Jaccard run's counts: 27 of 37 core edges in a 963-of-32,491 reference
        p = enrichment_p_value(27, 37, 963, 32491)
        assert 0 < p < 1e-20

    def test_one_lucky_edge_is_not_significant(self):
        assert enrichment_p_value(1, 1, 963, 32491) == pytest.approx(963 / 32491)

    def test_impossible_counts_are_refused(self):
        with pytest.raises(ValueError):
            enrichment_p_value(5, 3, 963, 32491)


class TestTheReferenceFile:
    """The PI's file is validated before any model runs, so a malformed one fails fast."""

    GOOD: ClassVar[dict] = {
        "edges": ["a->b", "b->c"], "source": "x", "decided_on": "2026-10-01",
        "criterion": {"min_precision": 0.3, "max_p_value": 0.001, "seed": 5},
    }

    def _write(self, tmp_path, data):
        path = tmp_path / "ref.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_a_complete_file_loads(self, tmp_path):
        edges, crit = load_reference(self._write(tmp_path, self.GOOD))
        assert edges == {"a->b", "b->c"}
        assert crit == {"min_precision": 0.3, "max_p_value": 0.001, "seed": 5}

    def test_a_bare_edge_list_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="criterion"):
            load_reference(self._write(tmp_path, ["a->b"]))

    def test_the_superseded_jaccard_file_is_refused(self, tmp_path):
        old = {"edges": ["a->b"], "tolerance": 0.2, "source": "x", "decided_on": "2026-09-30"}
        with pytest.raises(ValueError, match="superseded"):
            load_reference(self._write(tmp_path, old))

    @pytest.mark.parametrize("key", ["edges", "criterion", "source", "decided_on"])
    def test_every_field_is_required(self, tmp_path, key):
        data = {k: v for k, v in self.GOOD.items() if k != key}
        with pytest.raises(ValueError, match=key):
            load_reference(self._write(tmp_path, data))

    @pytest.mark.parametrize("key,value", [
        ("min_precision", 0), ("min_precision", 1.5), ("min_precision", True), ("min_precision", "0.3"),
        ("max_p_value", 0), ("max_p_value", 1), ("seed", -1), ("seed", 1.5), ("seed", True),
    ])
    def test_the_criterion_is_validated(self, tmp_path, key, value):
        data = {**self.GOOD, "criterion": {**self.GOOD["criterion"], key: value}}
        with pytest.raises(ValueError, match=key):
            load_reference(self._write(tmp_path, data))

    @pytest.mark.parametrize("edges", [[], ["ab"], ["a->b", "a->b"], "a->b"])
    def test_the_edges_must_be_distinct_src_dst_strings(self, tmp_path, edges):
        with pytest.raises(ValueError, match="edges"):
            load_reference(self._write(tmp_path, {**self.GOOD, "edges": edges}))

    def test_the_reference_path_is_inside_the_repository(self):
        assert REFERENCE_FILE.resolve().is_relative_to(REPO)


class TestThePassRecord:
    def test_a_run_that_fails_removes_the_previous_pass(self, tmp_path, monkeypatch):
        """Otherwise a regression after one pass would leave B6 open on an old record."""
        reference = tmp_path / "ref.json"
        reference.write_text(json.dumps(TestTheReferenceFile.GOOD), encoding="utf-8")
        record = tmp_path / "pass.json"
        record.write_text("{}", encoding="utf-8")
        monkeypatch.setitem(globals(), "REFERENCE_FILE", reference)
        monkeypatch.setitem(globals(), "PASS_RECORD", record)

        def extraction_fails(*args, **kwargs):
            raise RuntimeError("extraction failed")

        monkeypatch.setitem(globals(), "extract_gate_edges", extraction_fails)
        with pytest.raises(RuntimeError, match="extraction failed"):
            test_ioi_gpt2_small_reference_gate()
        assert not record.exists()

    def test_the_gate_runs_on_the_criterions_seed(self, tmp_path, monkeypatch):
        reference = tmp_path / "ref.json"
        reference.write_text(json.dumps(TestTheReferenceFile.GOOD), encoding="utf-8")
        monkeypatch.setitem(globals(), "REFERENCE_FILE", reference)
        monkeypatch.setitem(globals(), "PASS_RECORD", tmp_path / "pass.json")
        seen = {}

        def capture(*args, overrides=GATE_OVERRIDES, **kwargs):
            seen["overrides"] = overrides
            raise RuntimeError("stop")

        monkeypatch.setitem(globals(), "extract_gate_edges", capture)
        with pytest.raises(RuntimeError, match="stop"):
            test_ioi_gpt2_small_reference_gate()
        assert "seed=5" in seen["overrides"] and "seed=0" not in seen["overrides"]
        assert set(seen["overrides"]) - {"seed=5"} == set(GATE_OVERRIDES) - {"seed=0"}


class TestTheGateRunsWhatTheCellsRun:
    def test_the_overrides_are_the_cells_plus_model_and_task(self):
        """_run_one.sh is what every grid cell runs; the gate must not quietly differ."""
        script = (REPO / "deploy" / "plan_b_dgx_spark" / "_run_one.sh").read_text(encoding="utf-8")
        block = script.split('"$PY" "$RUNNER"', 1)[1].split("then", 1)[0]
        cell = {tok for tok in block.replace("\\", " ").split() if "=" in tok and not tok.startswith(("$", ">"))}
        assert cell, "could not read the cell overrides from _run_one.sh"
        gate = set(GATE_OVERRIDES)
        assert cell <= gate, f"the cells run {sorted(cell - gate)} and the gate does not"
        assert gate - cell == {"model=gpt2_small", "task=ioi"}

    def test_the_gate_config_clears_the_scientific_guards(self):
        pytest.importorskip("hydra")
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        from src.common.config_guard import assert_engineering_dry_run_limits, assert_no_gating_questions

        with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
            resolved = OmegaConf.to_container(compose("config", overrides=list(gate_overrides(5))), resolve=True)
        assert_no_gating_questions(resolved, "dense-node")
        assert_engineering_dry_run_limits(resolved, stage="stageA")
        assert resolved["seed"] == 5
        assert resolved["model"]["hf_revision"] == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
        assert resolved["mode"]["allow_model_download"] is True

    def test_the_core_band_is_read_from_the_run(self, tmp_path):
        """The extraction plumbing end to end, on the synthetic engineering path."""
        pytest.importorskip("hydra")
        pytest.importorskip("pyarrow")
        from src.common.schema import read_freq_parquet

        overrides = ("mode=engineering_dry_run", "pipeline=dense-node", "seed=0")
        core, run_dir = extract_gate_edges(tmp_path, overrides)
        rows = read_freq_parquet(str(run_dir / "freq.parquet"))
        assert run_dir.parent == tmp_path and rows
        assert core == {r["edge_id"] for r in rows if r["band"] == "core"}

    def test_only_core_edges_are_extracted(self, tmp_path):
        pytest.importorskip("pyarrow")
        from src.common.schema import write_freq_parquet

        write_freq_parquet(str(tmp_path / "freq.parquet"), [
            {"edge_id": "EMB->LOGIT", "s_e": 1.0, "band": "core"},
            {"edge_id": "L0.H0->LOGIT", "s_e": 0.5, "band": "contingent"},
            {"edge_id": "L0.MLP->LOGIT", "s_e": 0.1, "band": "noise"},
        ])
        (tmp_path / "run_meta.json").write_text(json.dumps({"band_cutoffs_resolved": True}), encoding="utf-8")
        assert core_band(tmp_path) == {"EMB->LOGIT"}

    def test_a_run_without_bands_is_refused(self, tmp_path):
        (tmp_path / "run_meta.json").write_text(json.dumps({"band_cutoffs_resolved": False}), encoding="utf-8")
        with pytest.raises(RuntimeError, match="core band"):
            core_band(tmp_path)


@pytest.mark.skipif(
    not os.environ.get(REFERENCE_EDGES_ENV),
    reason=(
        f"exit gate not runnable: set {REFERENCE_EDGES_ENV}=1 only once Stage A "
        "engineering is done (pinned GPT-2-small + upstream packages + RUN MODEL "
        "DOWNLOAD approval)"
    ),
)
def test_ioi_gpt2_small_reference_gate():
    """THE gate: the IOI core band on GPT-2 small must be the published circuit, not chance.

    Skipped unless RUN_IOI_GPT2_REFERENCE is set AND the reference file exists.
    """
    if not REFERENCE_FILE.exists():
        pytest.skip(
            f"reference file missing: {REFERENCE_FILE} "
            "(PI must provide edges + criterion and commit it; HUMAN_DECISIONS.md Step 7)"
        )
    reference, crit = load_reference(REFERENCE_FILE)
    # A record from an earlier pass must not outlive a run that fails: only THIS run's
    # pass may open B6, so the old record goes before anything can fail.
    PASS_RECORD.unlink(missing_ok=True)
    extracted, run_dir = extract_gate_edges(overrides=gate_overrides(crit["seed"]))

    from src.tasks.ioi_reference import _gpt2_small_edges

    universe = _gpt2_small_edges()
    outside = extracted - universe
    assert not outside, f"core edges outside GPT-2 small's edge universe: {sorted(outside)[:5]}"
    shared = len(extracted & reference)
    result = {
        "precision": precision(extracted, reference),
        "p_value": enrichment_p_value(shared, len(extracted), len(reference), len(universe)),
        "jaccard": edge_overlap(extracted, reference),
        "n_extracted_edges": len(extracted),
        "n_reference_edges": len(reference),
        "n_shared_edges": shared,
        "n_universe_edges": len(universe),
        "seed": crit["seed"],
    }
    summary = (
        f"precision {result['precision']:.3f} (min {crit['min_precision']}), "
        f"p {result['p_value']:.3g} (max {crit['max_p_value']}); core band {len(extracted)} edges, "
        f"shared {shared}, reference {len(reference)}; Jaccard {result['jaccard']:.3f} (reported only); "
        f"seeds {crit['seed']}-{crit['seed'] + 4}; Stage A run {run_dir}"
    )
    print(f"\nexit gate: {summary}")
    passed = result["precision"] >= crit["min_precision"] and result["p_value"] <= crit["max_p_value"]
    assert passed, f"exit gate FAILED: {summary}. Blocking real Stage A runs."
    write_pass_record(result, crit, run_dir)
