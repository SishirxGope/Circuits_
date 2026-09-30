# [AI-GEN] agent=Claude date=2026-09-30 task=B6 - write the PI's exit-gate reference file (edges derived, tolerance supplied by the PI)
# reviewed-by: PENDING
#
# WHAT THIS WRITES
# ----------------
# data/reference/ioi_gpt2_small_edges.json, the file the exit gate reads
# (tests/test_regression_ioi_gpt2_small.py, deploy/BLOCKERS.md B6):
#
#   edges       the published IOI circuit as dense-node edge ids, DERIVED - never typed:
#               ACDC's get_ioi_true_edges ported in src/tasks/ioi_reference.py, and
#               tests/test_ioi_reference.py proves it equals ACDC's own code edge for edge.
#   tolerance   the Jaccard the gate must reach. PI-OWNED (AI_RULES.md 2.2): this script
#               has no default and invents none. Choose it BEFORE the gate has ever run
#               (docs/HUMAN_DECISIONS.md Step 7) - a tolerance picked after seeing the
#               number is not a test.
#   decided_on  the date the PI fixed the tolerance.
#
# It refuses to overwrite an existing file: the gate record is tied to this file's sha256,
# so replacing it is a deliberate act (delete it by hand, and say why in HUMAN_DECISIONS.md).
#
# Usage (repo root):
#   python deploy/shared/make_ioi_reference.py --tolerance <0..1> --decided-on YYYY-MM-DD

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "data" / "reference" / "ioi_gpt2_small_edges.json"


def main(argv: list[str] | None = None, out: Path = OUT) -> int:
    parser = argparse.ArgumentParser(description="Write the PI's IOI exit-gate reference file.")
    parser.add_argument("--tolerance", type=float, required=True, help="Jaccard in (0, 1] - PI-owned")
    parser.add_argument("--decided-on", required=True, help="YYYY-MM-DD the tolerance was fixed")
    args = parser.parse_args(argv)

    if not 0 < args.tolerance <= 1:
        print(f"--tolerance must be a Jaccard in (0, 1], got {args.tolerance}")
        return 1
    try:
        datetime.date.fromisoformat(args.decided_on)
    except ValueError:
        print(f"--decided-on must be YYYY-MM-DD, got {args.decided_on!r}")
        return 1
    if out.exists():
        print(f"REFUSING: {out} already exists. The gate is tied to its sha256; replace it only "
              "deliberately (delete it by hand and record why in docs/HUMAN_DECISIONS.md).")
        return 1

    from src.tasks.ioi_reference import IOI_CIRCUIT, SOURCE, ioi_reference_edges

    edges = sorted(ioi_reference_edges())
    data = {
        "edges": edges,
        "tolerance": args.tolerance,
        "source": SOURCE,
        "decided_on": args.decided_on,
        "n_edges": len(edges),
        "circuit_heads": {group: [f"L{l}.H{h}" for l, h in heads] for group, heads in IOI_CIRCUIT.items()},
        "compared_with": (
            "the core band (s(e) = 1) of the dense-node Stage A ensemble on GPT-2 small at its pin, "
            "run exactly as the grid cells run (tests/test_regression_ioi_gpt2_small.py::GATE_OVERRIDES)"
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}: {len(edges)} edges, tolerance {args.tolerance}, decided on {args.decided_on}")
    print("Commit it BEFORE running the gate: the commit is the pre-registration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
