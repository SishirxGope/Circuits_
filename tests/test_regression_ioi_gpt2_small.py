# [AI-GEN] agent=OpenCode date=2026-08-07 task=Phase-1 exit-gate regression test (IOI on GPT-2 small)
# modified: [AI-GEN] agent=Claude date=2026-09-29 task=B6 - PI file carries edges AND tolerance; a pass leaves a record preflight can verify
# modified: [AI-GEN] agent=Claude date=2026-09-29 task=review fix - a gate run removes the previous pass record first
# reviewed-by: PENDING

"""Phase-1 exit gate (ARCHITECTURE.md §6): reproduce one published reference circuit
within tolerance before any real Stage A run is trusted.

Gate contract:
- Target: IOI circuit on GPT-2 small (Wang et al., ICLR 2023), pipeline A
  (attribution-patching graph) with a pinned checkpoint.
- Tolerance: edge-set overlap vs the published reference edge list. Both the edge list
  and the tolerance are PI-owned (AI_RULES.md 2.2 - no invented numbers), and both live
  in ONE file the PI writes and commits (``REFERENCE_FILE``)::

      {"edges": ["<src>-><dst>", ...],   # our edge_id convention (ARCHITECTURE.md §2)
       "tolerance": <Jaccard the gate must reach>,
       "source": "<where the edge list comes from>",
       "decided_on": "YYYY-MM-DD"}      # set BEFORE the first gate run

  One file so the tolerance cannot drift from the edges it was chosen for.
- The gate is a REGRESSION test, not a scientific claim: failing it blocks Stage A
  real extraction; passing it does not validate any compression claim.

Execution state:
- ``edge_overlap`` and the reference-file validation are pure and tested NOW.
- The gate itself is SKIPPED until BOTH the environment variable
  RUN_IOI_GPT2_REFERENCE is set (human intent: pinned weights, a GPU) AND the PI's
  reference file exists. Then it FAILS, loudly, until GPT-2 extraction is wired
  (``extract_gate_edges``) - it never reports an overlap it did not measure.
- A pass writes ``PASS_RECORD``, which the preflight (deploy/shared/preflight_blockers.py,
  check B6) verifies against the reference file's hash. That record - not the presence
  of this file, and not the word "skip" in it - is what opens B6.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import ClassVar

import pytest

REPO = Path(__file__).resolve().parents[1]
REFERENCE_EDGES_ENV = "RUN_IOI_GPT2_REFERENCE"
# Was os.path.join(dirname(__file__), "..", "..", "data", ...) - one ".." too many, which
# pointed OUTSIDE the repository, so a PI file placed in data/reference/ was never found.
REFERENCE_FILE = REPO / "data" / "reference" / "ioi_gpt2_small_edges.json"
PASS_RECORD = REPO / "data" / "reference" / "ioi_gpt2_small_gate_pass.json"
REFERENCE_EDGES_FILE = str(REFERENCE_FILE)  # kept for callers of the old name


def edge_overlap(extracted: set[str], reference: set[str]) -> float:
    """Jaccard overlap between extracted and reference edge IDs (pure).

    Edge IDs follow the ARCHITECTURE.md §2 edge_id convention ("src->dst", exact
    string). Computed over the UNION of both sets so a missing pipeline edge and a
    spurious pipeline edge both count against the gate.
    """
    if not reference:
        raise ValueError("reference edge set must be non-empty (gate is meaningless otherwise)")
    union = extracted | reference
    if not union:
        return 0.0
    return len(extracted & reference) / len(union)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(path: Path = REFERENCE_FILE) -> tuple[set[str], float]:
    """The PI's reference edges and tolerance, validated. Raises ValueError on any gap.

    A bare list (the old format) is refused: it has no tolerance, and the gate may not
    supply one (AI_RULES.md 2.2).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):  # a malformed FILE, not a bad argument: ValueError
        raise ValueError(  # noqa: TRY004
            f"{path}: expected an object with 'edges' and 'tolerance'; a bare edge list "
            "carries no tolerance and the gate may not invent one"
        )
    missing = [key for key in ("edges", "tolerance", "source", "decided_on") if key not in data]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    edges = data["edges"]
    if not isinstance(edges, list) or not edges or not all(isinstance(e, str) and "->" in e for e in edges):
        raise ValueError(f"{path}: 'edges' must be a non-empty list of 'src->dst' strings")
    if len(set(edges)) != len(edges):
        raise ValueError(f"{path}: 'edges' contains duplicates")
    tolerance = data["tolerance"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 < tolerance <= 1:
        raise ValueError(f"{path}: 'tolerance' must be a Jaccard in (0, 1], got {tolerance!r}")
    datetime.date.fromisoformat(str(data["decided_on"]))
    return set(edges), float(tolerance)


def extract_gate_edges() -> set[str]:
    """Core edges of the dense IOI ensemble on GPT-2 small (Stage A, pinned revision).

    Not wired yet. It needs configs/model/gpt2_small.yaml at the pinned revision
    (``607a30d7...``, docs/HUMAN_DECISIONS.md) with its architecture VERIFIED, a Stage A
    run in a mode that allows the download, and the core band of that run's freq.parquet.
    Raising here keeps the gate from ever reporting an overlap it did not measure.
    """
    raise NotImplementedError(
        "GPT-2 small extraction is not wired into the exit gate yet "
        "(configs/model/gpt2_small.yaml + a Stage A run + its core edges)"
    )


def write_pass_record(overlap: float, tolerance: float, n_extracted: int, n_reference: int) -> None:
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip() or None
    PASS_RECORD.write_text(
        json.dumps({
            "passed": True,
            "jaccard": overlap,
            "tolerance": tolerance,
            "reference_file": REFERENCE_FILE.relative_to(REPO).as_posix(),
            "reference_sha256": file_sha256(REFERENCE_FILE),
            "n_extracted_edges": n_extracted,
            "n_reference_edges": n_reference,
            "git_commit": commit,
            "date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_edge_overlap_pure_logic():
    """The gate's tolerance math is testable without any model."""
    ref = {"L0.H1->L2.H3", "L2.H3->L5.MLP"}
    assert edge_overlap({"L0.H1->L2.H3", "L2.H3->L5.MLP"}, ref) == pytest.approx(1.0)
    assert edge_overlap({"L0.H1->L2.H3"}, ref) == pytest.approx(1 / 2)
    assert edge_overlap(set(), ref) == 0.0
    with pytest.raises(ValueError):
        edge_overlap({"a->b"}, set())


class TestTheReferenceFile:
    """The PI's file is validated before any model runs, so a malformed one fails fast."""

    GOOD: ClassVar[dict] = {"edges": ["a->b", "b->c"], "tolerance": 0.5, "source": "x", "decided_on": "2026-10-01"}

    def _write(self, tmp_path, data):
        path = tmp_path / "ref.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_a_complete_file_loads(self, tmp_path):
        assert load_reference(self._write(tmp_path, self.GOOD)) == ({"a->b", "b->c"}, 0.5)

    def test_a_bare_edge_list_is_refused_it_has_no_tolerance(self, tmp_path):
        with pytest.raises(ValueError, match="tolerance"):
            load_reference(self._write(tmp_path, ["a->b"]))

    @pytest.mark.parametrize("key", ["edges", "tolerance", "source", "decided_on"])
    def test_every_field_is_required(self, tmp_path, key):
        data = {k: v for k, v in self.GOOD.items() if k != key}
        with pytest.raises(ValueError, match=key):
            load_reference(self._write(tmp_path, data))

    @pytest.mark.parametrize("tolerance", [0, -0.1, 1.5, True, "0.7"])
    def test_the_tolerance_must_be_a_jaccard(self, tmp_path, tolerance):
        with pytest.raises(ValueError, match="tolerance"):
            load_reference(self._write(tmp_path, {**self.GOOD, "tolerance": tolerance}))

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
        with pytest.raises(NotImplementedError):
            test_ioi_gpt2_small_reference_gate()
        assert not record.exists()


@pytest.mark.skipif(
    not os.environ.get(REFERENCE_EDGES_ENV),
    reason=(
        f"exit gate not runnable: set {REFERENCE_EDGES_ENV}=1 only once Stage A "
        "engineering is done (pinned GPT-2-small + upstream packages + RUN MODEL "
        "DOWNLOAD approval)"
    ),
)
def test_ioi_gpt2_small_reference_gate():
    """THE gate: extracted IOI edges on GPT-2 small must overlap the published circuit.

    Skipped unless RUN_IOI_GPT2_REFERENCE is set AND the reference file exists.
    """
    if not REFERENCE_FILE.exists():
        pytest.skip(
            f"reference file missing: {REFERENCE_FILE} "
            "(PI must provide edges + tolerance and commit it; HUMAN_DECISIONS.md Step 7)"
        )
    reference, tolerance = load_reference(REFERENCE_FILE)
    # A record from an earlier pass must not outlive a run that fails: only THIS run's
    # pass may open B6, so the old record goes before anything can fail.
    PASS_RECORD.unlink(missing_ok=True)
    extracted = extract_gate_edges()

    overlap = edge_overlap(extracted, reference)
    assert overlap >= tolerance, (
        f"exit gate FAILED: extracted-vs-reference Jaccard {overlap:.3f} < PI tolerance "
        f"{tolerance}. Blocking real Stage A runs."
    )
    write_pass_record(overlap, tolerance, len(extracted), len(reference))
