# [AI-GEN] agent=OpenCode date=2026-08-07 task=Phase-1 exit-gate regression test (IOI on GPT-2 small)
# reviewed-by: PENDING

"""Phase-1 exit gate (ARCHITECTURE.md §6): reproduce one published reference circuit
within tolerance before any real Stage A run is trusted.

Gate contract:
- Target: IOI circuit on GPT-2 small (Wang et al., ICLR 2023), pipeline A
  (attribution-patching graph) with a pinned checkpoint.
- Tolerance: edge-set overlap vs the published reference edge list. Exact tolerance
  value is PI-owned (AI_RULES.md 2.2 — no invented numbers); the reference edge list
  must be provided by the PI as a JSON file and pinned (data/reference/).
- The gate is a REGRESSION test, not a scientific claim: failing it blocks Stage A
  real extraction; passing it does not validate any compression claim.

Execution state:
- The overlap helper (edge_overlap) is pure and tested NOW.
- The gate itself is SKIPPED until BOTH:
  1. the environment variable RUN_IOI_GPT2_REFERENCE is set (human intent; the gate
     needs pinned weights + upstream packages + a GPU), AND
  2. the PI-provided reference edges file exists (no invented reference edges).
"""

import json
import os

import pytest

REFERENCE_EDGES_ENV = "RUN_IOI_GPT2_REFERENCE"
REFERENCE_EDGES_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "reference", "ioi_gpt2_small_edges.json"
)

# ⚠️ TODO [QUESTION FOR PI]: fixed tolerance (e.g. overlap >= 0.7) + exact reference
# edge list + pinned GPT-2-small revision (HUMAN_DECISIONS.md Q5/Q6). Nothing here may
# invent them (AI_RULES.md 2.2).


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


def test_edge_overlap_pure_logic():
    """The gate's tolerance math is testable without any model."""
    ref = {"L0.H1->L2.H3", "L2.H3->L5.MLP"}
    assert edge_overlap({"L0.H1->L2.H3", "L2.H3->L5.MLP"}, ref) == pytest.approx(1.0)
    assert edge_overlap({"L0.H1->L2.H3"}, ref) == pytest.approx(1 / 2)
    assert edge_overlap(set(), ref) == 0.0
    with pytest.raises(ValueError):
        edge_overlap({"a->b"}, set())


@pytest.mark.skipif(
    not os.environ.get(REFERENCE_EDGES_ENV),
    reason=(
        f"exit gate not runnable: set {REFERENCE_EDGES_ENV}=1 only once Stage A "
        "engineering is done (pinned GPT-2-small + upstream packages + RUN MODEL "
        "DOWNLOAD approval)"
    ),
)
def test_ioi_gpt2_small_reference_gate(tmp_path):
    """THE gate: extracted IOI edges on GPT-2 small must overlap the published circuit.

    Skipped unless RUN_IOI_GPT2_REFERENCE is set AND the reference edges file exists.
    """
    if not os.path.exists(REFERENCE_EDGES_FILE):
        pytest.skip(
            f"reference edges file missing: {REFERENCE_EDGES_FILE} "
            "(PI must provide + pin it; HUMAN_DECISIONS.md Q6)"
        )

    # Stage A engineering TODO: run pipeline A on GPT-2 small (pinned revision,
    # dense reference ensemble), then build the extracted edge-ID set from the
    # ensemble core edges. Placeholder below keeps the gate structurally honest
    # until then (it is skipped in CI, never silently green).
    with open(REFERENCE_EDGES_FILE, "r", encoding="utf-8") as f:
        reference: set[str] = set(json.load(f))

    extracted: set[str] = set()  # TODO(Stage A engineering): real extraction output

    overlap = edge_overlap(extracted, reference)
    assert overlap >= 1.0, (
        f"exit gate FAILED: extracted-vs-reference Jaccard {overlap:.3f} "
        f"(expected >= 1.0; tolerance TBD by PI). Blocking real Stage A runs."
    )
