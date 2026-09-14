# [AI-GEN] agent=OpenCode date=2026-08-07 task=Stage D entrypoint: cross-audit + reporting (guard skeleton)
# modified: [AI-GEN] agent=Claude date=2026-08-08 task=Implement the Stage D body: damage ranking, BH correction, cross-audit with intervals
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY

"""Algorithm 1, Stage D: causal + cross-audit checks (proposal §2.6; ARCHITECTURE.md §1).

Algorithm 1 lines 22-26:

    for each edge with large |freq_dense - freq_c|:
        NIE, INT_flag <- PatchSingle(e), PatchGrouped(e, siblings)
        if INT_flag: move e to the interaction-dominated set (excluded from headline)
    rank_circuit  <- order C by D(c)
    rank_feature  <- feature-damage order from refs [1,2]
    report Spearman(rank_circuit, rank_feature) + the threshold-sensitivity sweep

What runs here now:

- **Damage ranking** over the Stage C cells, by D(c), at BOTH comparison levels.
- **Cross-audit (C5)** with a permutation p-value and a bootstrap CI, and the
  pre-registered verdict band — but ONLY against PI-provided, verified feature-damage
  numbers. There is no fallback and no default table: inventing the comparison target
  would be the exact failure AI_RULES.md 2.2 exists to prevent, so a Stage D run
  without ``stage_d.feature_damage`` reports the circuit ranking and states plainly
  that C5 is pending.
- **Multiple-comparison control** across the reported grid (AI_RULES.md 4.3).
- **Interaction diagnostic (C4)**: the largest inclusion-frequency movers are
  identified here, but ``PatchDiagnostic.run`` needs real models, so the mover list is
  emitted for Stage-D engineering and the INT columns stay empty rather than fabricated.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any

try:
    from hydra import main as hydra_main  # type: ignore[import-not-found]
    from omegaconf import DictConfig  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    hydra_main = None  # type: ignore[assignment]

from analysis.cross_audit import cross_audit_report
from experiments.run_stage_a import _git_commit, _resolve
from src.common import schema
from src.common.config_guard import (
    MODE_ENGINEERING,
    assert_engineering_dry_run_limits,
    assert_no_gating_questions,
    is_provably_synthetic,
    mode_of,
)
from src.common.csi_table import cell_identity, read_csi_table, summarize_csi
from src.common.hashing import hash_config
from src.common.run_dir import allocate_run_dir
from src.common.run_naming import build_configset, resolve_run_name, sanitize_token
from src.science.multiple_comparisons import (
    DEFAULT_FDR_Q,
    correct_cell_grid,
    expected_false_positives,
)


def _collect_stage_c_rows(stage_c_dirs: list[str]) -> list[dict[str, Any]]:
    """Read every Stage C run's csi_table.csv into one grid."""
    rows: list[dict[str, Any]] = []
    for d in stage_c_dirs:
        table = Path(d) / "csi_table.csv"
        if not table.exists():
            raise RuntimeError(
                f"Stage C run {d} has no csi_table.csv — Stage D is downstream of Stage C "
                f"output, not of a Stage C attempt (ARCHITECTURE.md §1)."
            )
        rows.extend(read_csi_table(table))
    if not rows:
        raise RuntimeError("no CSI rows found in the supplied Stage C run directories")
    return rows


def damage_ranking(rows: list[dict[str, Any]], comparison_level: str) -> dict[str, float]:
    """{cell_id: D(c)} at one comparison level — the circuit-damage ordering (line 25)."""
    return {
        "/".join(cell_identity(r)): float(r["D"])
        for r in rows
        if r["comparison_level"] == comparison_level
    }


def largest_movers(
    freq_pre: dict[str, float], freq_post: dict[str, float], top_n: int = 20
) -> list[dict[str, Any]]:
    """Edges with the largest |s_pre - s_post| — the C4 patching candidates (line 22)."""
    keys = set(freq_pre) | set(freq_post)
    movers = [
        {
            "edge_id": k,
            "s_pre": float(freq_pre.get(k, 0.0)),
            "s_post": float(freq_post.get(k, 0.0)),
            "delta": float(freq_post.get(k, 0.0) - freq_pre.get(k, 0.0)),
        }
        for k in sorted(keys)
    ]
    movers.sort(key=lambda m: (-abs(m["delta"]), m["edge_id"]))
    return movers[: int(top_n)]


def run_stage_d(cfg: Any) -> Path:
    """Run Stage D over a set of Stage C runs; returns the run directory."""
    resolved = _resolve(cfg)
    if resolved["stage"] != "stageD":
        raise ValueError(f"run_stage_d is the Stage D entrypoint; got stage={resolved['stage']!r}")

    pipeline = str(resolved.get("pipeline", "mock"))
    ensemble = dict(resolved.get("ensemble", {}))
    B, S = int(ensemble.get("B", 0)), int(ensemble.get("S", 0))
    R = int((resolved.get("nulls") or {}).get("R", 0))
    seed = int(resolved.get("seed", 0))
    null_hash = str(resolved.get("null_frozen_hash") or "")
    mode = mode_of(resolved)
    config_hash = hash_config(resolved)

    tags = {
        "stage": "stageD",
        "model": resolved["model"]["name"],
        "task": resolved["task"]["name"],
        "compression_family": resolved.get("compression_family", ""),
        "compression_level": resolved.get("compression_level"),
        "comparison_level": resolved.get("comparison_level", "both"),
        "pipeline": pipeline, "mode": mode,
        "B": B, "S": S, "R": R, "seed": seed,
        "null_frozen_hash": null_hash or None,
        "git_commit": _git_commit(),
        "config_hash": config_hash,
    }
    schema.validate_run_tags(tags, stage="stageD")

    # --- downstream-of-nothing guard ------------------------------------------------
    stage_d = dict(resolved.get("stage_d") or {})
    stage_c_dirs = [str(d) for d in (stage_d.get("stage_c_run_dirs") or ([stage_d["stage_c_run_dir"]] if stage_d.get("stage_c_run_dir") else []))]
    missing = [d for d in stage_c_dirs if not Path(d).exists()]
    if not stage_c_dirs or missing:
        raise RuntimeError(
            "Stage D requires Stage C outputs: stage_d.stage_c_run_dirs must list existing "
            f"Stage C run directories (missing: {missing or 'none supplied'}). "
            "No downstream-of-nothing (ARCHITECTURE.md §1)."
        )

    assert_no_gating_questions(resolved, pipeline)
    assert_engineering_dry_run_limits(resolved, stage="stageD")

    rows = _collect_stage_c_rows(stage_c_dirs)

    run_name = resolve_run_name(
        resolved.get("run_name"), datetime.date.today().strftime("%Y%m%d"), "stageD",
        tags["model"], tags["task"], sanitize_token(resolved.get("setting") or "crossaudit"),
        build_configset(B, S, R=R), seed,
    )
    run_dir = allocate_run_dir(resolved.get("run_root", "runs"), run_name, config_hash)

    # --- line 25: circuit damage ranking at BOTH levels -----------------------------
    rankings = {lvl: damage_ranking(rows, lvl) for lvl in ("exact_edge", "routing_head")}

    # --- line 26: cross-audit, ONLY against PI-provided verified numbers -------------
    feature_damage = stage_d.get("feature_damage") or {}
    if feature_damage:
        cross_audit = {
            lvl: cross_audit_report(
                rankings[lvl], {str(k): float(v) for k, v in feature_damage.items()},
                n_permutations=int(stage_d.get("n_permutations", 10000)),
                n_boot=int(stage_d.get("n_boot", 2000)),
                seed=seed,
            )
            for lvl in rankings
        }
    else:
        cross_audit = {
            "status": "PENDING",
            "reason": (
                "no stage_d.feature_damage supplied. C5 compares against the PUBLISHED "
                "feature-level damage ranking from refs [1]/[2]; those numbers must be "
                "read from the papers and entered by the PI (AI_RULES.md 2.2 — no "
                "invented numbers). The circuit-side ranking below is complete and is "
                "what C5 will be computed against."
            ),
        }

    # --- AI_RULES.md 4.3: multiple-comparison control across the reported grid --------
    # A CSI CI that excludes 1 is the per-cell "significant" statement; we convert the
    # interval to a crude per-cell p-proxy only when the config supplies real p-values.
    cell_p = stage_d.get("cell_p_values") or {}
    # One source for q. It was read twice with a literal 0.05 fallback in each call,
    # so a config that set fdr_q could in principle be applied to one and not the
    # other, and the reported q would not have to match the q actually used.
    fdr_q = float(stage_d.get("fdr_q", DEFAULT_FDR_Q))
    correction = (
        {"applied": True, "q": fdr_q,
         "cells": correct_cell_grid({str(k): float(v) for k, v in cell_p.items()},
                                    q=fdr_q)}
        if cell_p else
        {"applied": False,
         "reason": "no per-cell p-values supplied; CSI intervals are the per-cell statement",
         "n_reported_cells": len({cell_identity(r) for r in rows}),
         "expected_false_positives_if_uncorrected_at_0.05":
             expected_false_positives(len({cell_identity(r) for r in rows}))}
    )

    evidence = not (mode == MODE_ENGINEERING or is_provably_synthetic(resolved, pipeline))
    report = {
        **tags,
        "run_name": run_name, "run_dir": str(run_dir),
        "stage_c_run_dirs": stage_c_dirs,
        "csi_summary": summarize_csi(rows),
        "damage_ranking": rankings,
        "cross_audit_c5": cross_audit,
        "multiple_comparisons": correction,
        "interaction_diagnostic_c4": {
            "status": "PENDING",
            "reason": (
                "PatchDiagnostic.run requires real models + patching APIs (Stage D "
                "engineering). The NIE/PIE/INT arithmetic and the flag rule are "
                "implemented and tested in src/science/patch_diagnostic.py; the "
                "quantitative INT-flag rule is still to be pre-registered "
                "(AI_RULES.md 4.2)."
            ),
        },
        "evidence": evidence,
        "synthetic": is_provably_synthetic(resolved, pipeline),
    }
    with open(run_dir / "stage_d_report.json", "w", encoding="utf-8") as f:
        json.dump(schema.json_safe(report), f, indent=2, sort_keys=True, default=str, allow_nan=False)

    print(
        f"[stageD{'' if evidence else '-synthetic'}] {run_name} "
        f"cells={report['csi_summary']['n_cells']} "
        f"selective={report['csi_summary']['structurally_selective']} "
        f"indistinguishable={report['csi_summary']['indistinguishable_from_noise']} "
        f"C5={cross_audit.get('status', 'computed')} -> {run_dir}"
    )
    return run_dir


def main() -> None:
    if hydra_main is None:  # pragma: no cover
        print("hydra-core + omegaconf are required to run the entrypoint (optional dep for tests).", file=sys.stderr)
        sys.exit(1)

    @hydra_main(version_base=None, config_path="../configs", config_name="config")
    def _run(cfg: DictConfig) -> None:
        run_stage_d(cfg)

    _run()


if __name__ == "__main__":
    main()
