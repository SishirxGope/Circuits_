# [AI-GEN] agent=OpenCode date=2026-08-07 task=Integration: engineering dry-run Stage A end-to-end (PI provisional unblock)
# reviewed-by: PENDING

"""Engineering dry-run Stage A end-to-end (no internet, no GPU, no model download).

Asserts the dry-run contract: synthetic/mock components run the full stack and produce
schema-valid outputs; run_meta records mode + provisional resolutions; frozen/ is never
touched; runs are immutable.

Updated 2026-09-12: Q1/Q2/Q4/Q10/Q11 are pre-registered, so the dry run now exercises
the PRE-REGISTERED band cutoffs and the pre-registered anti-diagonal grid rather than
the retired placeholders. Q3 (B=4/S=2 here) is still provisional — it needs the
measured Pythia wall time.
"""

import json

import pytest

from experiments.dry_run_stage_a import engineering_stage_a_cfg
from experiments.run_stage_a import run_stage_a
from src.common import schema


def test_engineering_dry_run_stage_a_end_to_end(tmp_path):
    cfg = engineering_stage_a_cfg(seed=0, run_root=str(tmp_path / "runs"))
    run_dir = run_stage_a(cfg)

    for name in ("resolved_config.json", "run_meta.json", "edges.parquet", "freq.parquet"):
        assert (run_dir / name).exists(), f"missing {name}"

    edges = schema.read_edges_parquet(str(run_dir / "edges.parquet"))
    assert edges and set(edges[0]) == set(schema.EDGES_COLUMNS)

    freq = schema.read_freq_parquet(str(run_dir / "freq.parquet"))
    assert freq
    for row in freq:
        assert 0.0 <= row["s_e"] <= 1.0
        # provisional cutoffs resolved -> real band labels, never None
        assert row["band"] in ("core", "contingent", "noise")
        # Cutoffs are read from the config, never restated here: a test that
        # hard-codes them passes while silently disagreeing with what ran.
        bands_cfg = cfg["ensemble"]["decompose"]
        core_t, noise_t = bands_cfg["core_threshold"], bands_cfg["noise_threshold"]
        assert (row["s_e"] >= core_t) == (row["band"] == "core")
        if bands_cfg.get("noise_strict"):
            assert (row["s_e"] < noise_t) == (row["band"] == "noise")
        else:
            assert (row["s_e"] <= noise_t) == (row["band"] == "noise")

    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "engineering_dry_run"
    assert meta["B"] == 4 and meta["S"] == 2 and meta["n_cells"] == 8
    assert meta["band_cutoffs_resolved"] is True
    assert len(meta["provisional_resolutions"]) == 11
    assert all(r["status"] == "PROVISIONAL_ENGINEERING_DEFAULT" for r in meta["provisional_resolutions"])
    # Q3 (B/S/R) needs the measured Pythia wall time; Q5/Q6 need the PI's downloads.
    assert "Q3" in meta["open_questions"]
    # Pre-registered 2026-09-12, so no longer reported open: Q1, Q2, Q4, Q10, Q11.
    for closed in ("Q4", "Q10", "Q11"):
        assert closed not in meta["open_questions"], (
            f"{closed} was pre-registered on 2026-09-12 but is still reported OPEN"
        )

    # never touched frozen/
    assert not (tmp_path / "frozen").exists()


def test_dry_run_uses_the_preregistered_anti_diagonal_grid(tmp_path):
    cfg = engineering_stage_a_cfg(seed=0, run_root=str(tmp_path / "runs"))
    run_dir = run_stage_a(cfg)
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    # Q4 pre-registered 2026-09-12: the config records the DESIGN and the generator
    # builds the B configs, so the grid is reproducible from B alone with no seed.
    grid = resolved["ensemble"]["threshold_grid"]
    assert grid["design"] == "anti_diagonal"
    assert grid["node_range"] == [0.6, 0.9] and grid["edge_range"] == [0.99, 0.95]
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    q4 = next(r for r in meta["provisional_resolutions"] if r["id"] == "Q4")
    assert q4["value"]["design"] == "anti_diagonal"


def test_dry_run_runs_are_immutable(tmp_path):
    cfg = engineering_stage_a_cfg(seed=0, run_root=str(tmp_path / "runs"))
    run_stage_a(cfg)
    with pytest.raises(FileExistsError):
        run_stage_a(cfg)


def test_dry_run_deterministic_with_same_seed(tmp_path):
    d1 = run_stage_a(engineering_stage_a_cfg(seed=0, run_root=str(tmp_path / "runs")))
    # identical config -> identical run name -> immutable refusal (never a second run)
    with pytest.raises(FileExistsError):
        run_stage_a(engineering_stage_a_cfg(seed=0, run_root=str(tmp_path / "runs")))
    assert d1.exists()
