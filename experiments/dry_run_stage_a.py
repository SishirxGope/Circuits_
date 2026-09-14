# [AI-GEN] agent=OpenCode date=2026-08-07 task=Engineering dry-run Stage A driver (no hydra dependency)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY

"""Engineering dry-run driver for Stage A (PI provisional unblock, 2026-08-07).

Runs the full Stage A stack end-to-end with synthetic components only (no model
download, no GPU, no external data): MockModel + synthetic IOI-like task + the Q4
provisional seeded threshold grid (B=4, S=2) + provisional L1/JS distance config +
provisional band cutoffs (0.90/0.10).

This driver mirrors the engineering composition of configs/config.yaml (Hydra
defaults) so both entrypoints agree; it exists because hydra-core is an optional
dependency in this environment. NO scientific evidence is produced (AI_RULES.md 2.3).

Usage::

    python experiments/dry_run_stage_a.py [seed]

Output: prints the run directory under runs/ (CLAUDE.md §4 naming).
"""

from __future__ import annotations

import sys

from experiments.run_stage_a import run_stage_a


def engineering_stage_a_cfg(seed: int = 0, run_root: str = "runs") -> dict:
    """The engineering dry-run Stage A config (mirrors configs/config.yaml defaults)."""
    return {
        "mode": {"name": "engineering_dry_run"},
        "stage": "stageA",
        "seed": int(seed),
        "run_root": run_root,
        "pipeline": "mock",
        "setting": "dense",
        "comparison_level": "both",
        "compression_family": "dense",
        "compression_level": None,
        "model": {"name": "mock", "synthetic": True, "n_layers": 4, "n_heads": 2, "d_model": 8, "hf_revision": None},
        "task": {"name": "synthetic-ioi", "family": "indirect-object-identification-synthetic",
                 "synthetic": True, "n_prompts": 24, "max_seq_len": 32, "seed": 0},
        "ensemble": {
            "B": 4,
            "S": 2,
            "configset": "B4xS2",
            # Q4 pre-registered 2026-09-12: the anti-diagonal grid. The dry run uses the
            # SAME generator at B=4, so the code path exercised here is the one Stage A
            # will take — a dry run that exercises a retired sampler tests nothing.
            "threshold_grid": {
                "design": "anti_diagonal",
                "node_range": [0.6, 0.9],
                "edge_range": [0.99, 0.95],
            },
            "pi_confirmed": False,  # Q3 (B/S/R) still needs the Pythia wall-time pilot
            # Q1 pre-registered 2026-09-12 (CIRCUS section 3.2). Mirrors
            # configs/ensemble/decompose/final.yaml; test_config_integrity asserts the
            # Hydra composition and this driver never drift apart.
            "decompose": {
                "core_threshold": 1.0,
                "noise_threshold": 0.5,
                "noise_strict": True,
            },
        },
        "distance": {"name": "l1", "alternative": "jensen_shannon"},
        # Q10/Q11 pre-registered 2026-09-12; mirrors configs/comparison/final.yaml
        "comparison": {"level2_scheme": "layer", "position_policy": "aggregate"},

        "nulls": {"R": 3},
        "compression": {"name": "null"},
    }


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_dir = run_stage_a(engineering_stage_a_cfg(seed=seed))
    print(f"DRY-RUN OK -> {run_dir}")


if __name__ == "__main__":
    main()
