# [AI-GEN] agent=OpenCode date=2026-08-07 task=Stage C entrypoint: CSI measurement (guard skeleton)
# modified: [AI-GEN] agent=Claude date=2026-08-08 task=Implement the Stage C body; it raised even after every guard passed, so Algorithm 1 had never run end-to-end
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY

"""Algorithm 1, Stage C: real-compression cells + CSI (proposal §2.1/§3.2).

Guards, enforced in this order (each is tested):

1. Stage tag + required tags (Stage C/D require ``R > 0`` and a non-null
   ``null_frozen_hash``).
2. **Null before effect** (AI_RULES.md 1.5): the frozen Stage B artifacts must exist,
   be registered in the freeze manifest, the declared ``null_frozen_hash`` must match
   the frozen cell's, and the frozen null must have been matched to THIS cell's
   compression — ``assert_stage_b_frozen`` + ``assert_null_frozen_hash`` +
   ``assert_null_matched_to_cell``. These fire FIRST among the substantive checks: a
   Stage C run that cannot verify the null it divides by fails hard, before anything
   else is even considered.
3. PI-decision guard: scientific_run blocks while final decisions are OPEN.
4. Mode guard: engineering mode permits Stage C **only** on a provably synthetic stack
   (mock model + synthetic task + mock extractor) and refuses anything that could reach
   a real model or corpus.

What the body does (Algorithm 1 lines 15-20):

    M_c    <- Compress(M, c)
    freq_c <- ExtractCircuitEnsemble(M_c, T, B, S)
    D(c)   <- Dist(freq_dense, freq_c)
    CSI(c) <- D(c) / median(D_null(c))          [bootstrap CI]
    repeat at routing-head granularity

Both comparison levels are computed unconditionally, and the CSI table writer refuses
a table that carries only one of them (proposal §4 rule 3). Every overlap statistic
gets its random top-k chance floor attached (AI_RULES.md 4.4).

Real compression on real models still raises: it needs scientific mode, final
decisions, pinned compressors, verified calibration data (Q7) and explicit approval.
"""

from __future__ import annotations

import datetime
import json
import statistics
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from hydra import main as hydra_main  # type: ignore[import-not-found]
    from omegaconf import DictConfig  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    hydra_main = None  # type: ignore[assignment]

from experiments.run_stage_a import (
    PIPELINE_REGISTRY,
    _build_threshold_configs,
    _git_commit,
    _resolve,
)
from src.common import schema
from src.common.config_guard import (
    MODE_ENGINEERING,
    assert_engineering_dry_run_limits,
    assert_no_gating_questions,
    is_provably_synthetic,
    mode_of,
)
from src.common.csi_table import make_csi_row, summarize_csi, write_csi_table
from src.common.hashing import hash_config
from src.common.run_dir import allocate_run_dir
from src.common.run_naming import build_configset, resolve_run_name, sanitize_token
from src.common.stage_guard import (
    assert_null_frozen_hash,
    assert_null_matched_to_cell,
    assert_stage_b_frozen,
)
from src.science.chance_floor import candidate_edge_count, overlap_vs_chance_for_freqs
from src.science.csi import cells_from_records, csi_over_ensemble
from src.science.circus_wrapper import CircusEnsembleRunner
from src.compression.registry import compression_provenance, compressor_for, compressor_registry
from src.science.csi import csi
from src.science.decompose import decompose
from src.science.distances import jensen_shannon_distance, l1_distance, normalized_l1_distance
from src.science.two_level import project_to_coarse_level, project_to_routing_heads

# Kept as a module-level alias: the registry now lives in src/compression/registry.py so
# Stage B and Stage C cannot drift to different compressors for the same cell.
_compressor_registry = compressor_registry


def frozen_cell_path(resolved: dict[str, Any]) -> Path:
    """``frozen/{model}/{task}/{cell}`` for the compression cell under test."""
    cell = str(
        resolved.get("stage_c", {}).get("cell")
        or f"{resolved.get('compression_family', 'family')}-{resolved.get('compression_level', 'level')}"
    )
    return Path(resolved.get("frozen_root", "frozen")) / resolved["model"]["name"] / resolved["task"]["name"] / cell


def _load_dnull(cell: Path, metric: str = "distance_l1") -> list[float]:
    """The frozen D_null draws for this cell — read ONLY through the frozen store."""
    import pyarrow.parquet as pq

    rows = pq.read_table(str(cell / "dnull.parquet")).to_pylist()
    if not rows:
        raise RuntimeError(f"frozen dnull at {cell} is empty; CSI has no denominator")
    if metric not in rows[0]:
        raise RuntimeError(
            f"frozen dnull at {cell} has no column {metric!r} (has {sorted(rows[0])}); "
            f"the distance used at Stage C must be one the frozen null recorded"
        )
    return [float(r[metric]) for r in rows]


def _build_compressor(resolved: dict[str, Any]):
    """Instantiate the compressor for this cell from config; never guess a family.

    Same call Stage B makes, so the compression measured here is the compression the
    frozen null was matched to.
    """
    return compressor_for(resolved, stage="stageC")


def _structural_edge_universe(model: Any) -> list[str] | None:
    """Every edge the extractor COULD have scored, or None if that is not derivable.

    For the dense-node pipeline this is ``eap.candidate_edges``: EAP's universe is
    layer-ordered and type-restricted (head outputs feed Q/K/V and MLP inputs, never
    arbitrary components), so it is ~10x SMALLER than all ordered pairs of components.
    For a MockModel it is all ordered pairs over the model's declared node universe,
    which is exactly what MockExtractor enumerates.

    Returns None for pipeline A, whose node basis is active transcoder features rather
    than heads - there ``candidate_edges`` does not describe the universe at all.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is not None and hasattr(cfg, "n_heads") and hasattr(cfg, "n_layers"):
        from src.extraction.eap import candidate_edges

        return candidate_edges(cfg)
    nodes = getattr(model, "nodes", None)
    if callable(nodes):
        universe = tuple(nodes())
        return [f"{a}->{b}" for a in universe for b in universe if a != b]
    return None


def _chance_floor_universes(
    model: Any, f_dense: Mapping[str, float], f_post: Mapping[str, float],
    level2_scheme: Any,
) -> dict[str, int]:
    """Candidate-universe size N per comparison level (AI_RULES.md 4.4).

    Two corrections over the single N this used to compute:

    1. **The universe is structural, not all-pairs.** ``candidate_edge_count`` returns
       C*(C-1), which assumes every ordered pair of components is a possible edge. EAP's
       universe is layer-ordered and type-restricted, making C*(C-1) about 10x too large
       (measured: 10.70x pythia-160m, 10.52x pythia-410m, 11.03x llama-3.2-1b, 9.95x
       gemma-2-2b). Too large an N pushes the random-overlap baseline DOWN and makes every
       overlap look above chance for combinatorial reasons.

    2. **Each level has its own universe.** The same N was used for both levels, but
       ``project_to_routing_heads`` projects onto HEADS, not onto coarse edges - so the
       routing-head universe is the number of heads (144 for pythia-160m), not the number
       of edges. Sharing the exact-edge N understated that floor by 2400x-8300x, driving
       the routing-head chance floor to ~1e-5 where the honest value is ~0.025. That is
       the level the two-level stability claim is stated at (claim C2).

    The observed-node restriction from the original implementation is kept: N counts only
    edges whose endpoints the extractor actually emitted for these prompts, so a
    model-wide count cannot include nodes that could never have appeared.
    """
    observed_nodes = {
        node for edge_id_ in set(f_dense) | set(f_post)
        for node in edge_id_.split("->", 1)
    }
    structural = _structural_edge_universe(model)

    if structural is None:
        # Pipeline A: no structural description of the universe, so fall back to the
        # all-pairs count over observed nodes and say so rather than inventing a number.
        n_exact = candidate_edge_count(len(observed_nodes))
        reachable = ()
    else:
        reachable = tuple(
            e for e in structural
            if all(part in observed_nodes for part in e.split("->", 1))
        )
        n_exact = len(reachable)

    if reachable:
        ones = {e: 1.0 for e in reachable}
        coarse = (
            project_to_coarse_level(ones, level2_scheme, strict=False)
            if level2_scheme
            else project_to_routing_heads(ones, strict=False)
        )
        n_coarse = len(coarse)
    else:
        n_coarse = 0

    if n_coarse <= 0:
        # An empty coarse universe would make the routing-head floor undefined; fall back
        # to the coarse projection of what was actually observed, and never to n_exact -
        # that is the 2400x error this function exists to remove.
        observed_coarse = (
            project_to_coarse_level({e: 1.0 for e in set(f_dense) | set(f_post)},
                                    level2_scheme, strict=False)
            if level2_scheme
            else project_to_routing_heads({e: 1.0 for e in set(f_dense) | set(f_post)},
                                          strict=False)
        )
        n_coarse = max(len(observed_coarse), 1)

    return {"exact_edge": max(n_exact, 1), "routing_head": n_coarse}


def _load_model(resolved: dict[str, Any]):
    """Engineering: MockModel only. Real models raise (Stage C approval required)."""
    model_cfg = dict(resolved.get("model", {}))
    if mode_of(resolved) == MODE_ENGINEERING and model_cfg.get("synthetic"):
        from src.synthetic.mock_model import MockModel

        return MockModel(
            seed=int(resolved.get("seed", 0) or 0),
            n_layers=int(model_cfg.get("n_layers", 4)),
            n_heads=int(model_cfg.get("n_heads", 2)),
            d_model=int(model_cfg.get("d_model", 8)),
        )
    raise NotImplementedError(
        "Stage C real compression is not approved yet: it requires mode=scientific_run "
        "with all final decisions (docs/HUMAN_DECISIONS.md), pinned compressors, "
        "verified calibration data (Q7), and explicit PI approval for real runs. "
        "The frozen-null guards ran and passed before this point."
    )


def run_stage_c(cfg: Any) -> Path:
    """Run one Stage C cell; returns the run directory.

    Raises unless every guard passes. On a provably synthetic stack this executes the
    full Algorithm-1 Stage C and writes ``csi_table.csv`` + ``freq_post.parquet``.
    """
    resolved = _resolve(cfg)
    if resolved["stage"] != "stageC":
        raise ValueError(f"run_stage_c is the Stage C entrypoint; got stage={resolved['stage']!r}")

    pipeline = str(resolved.get("pipeline", "mock"))
    if pipeline not in PIPELINE_REGISTRY:
        raise ValueError(f"unknown pipeline {pipeline!r}; expected one of {sorted(PIPELINE_REGISTRY)}")

    ensemble = dict(resolved.get("ensemble", {}))
    B, S = int(ensemble.get("B", 0)), int(ensemble.get("S", 0))
    R = int((resolved.get("nulls") or {}).get("R", 0))
    seed = int(resolved.get("seed", 0))
    null_hash = str(resolved.get("null_frozen_hash") or "")
    mode = mode_of(resolved)
    config_hash = hash_config(resolved)

    tags = {
        "stage": "stageC",
        "model": resolved["model"]["name"],
        "task": resolved["task"]["name"],
        "compression_family": resolved.get("compression_family", ""),
        "compression_level": resolved.get("compression_level"),
        "comparison_level": resolved.get("comparison_level", "both"),
        "pipeline": pipeline,
        "mode": mode,
        "B": B, "S": S, "R": R, "seed": seed,
        "null_frozen_hash": null_hash or None,
        "git_commit": _git_commit(),
        "config_hash": config_hash,
    }
    schema.validate_run_tags(tags, stage="stageC")  # requires R > 0 + null_frozen_hash

    # --- Null before effect (AI_RULES.md 1.5) -------------------------------------
    # The most fundamental gate: it fires before decision gating and before any model
    # is touched. A Stage C cell whose null is not frozen and hash-matched cannot run.
    cell = frozen_cell_path(resolved)
    assert_stage_b_frozen(cell)
    assert_null_frozen_hash(cell / "meta.json", null_hash)
    # ...and that the intact null is the RIGHT null: matched to this cell's compression,
    # not merely present and unmodified.
    assert_null_matched_to_cell(cell / "meta.json", compression_provenance(resolved))

    assert_no_gating_questions(resolved, pipeline)
    assert_engineering_dry_run_limits(resolved, stage="stageC")

    # --- run identity --------------------------------------------------------------
    setting = sanitize_token(
        resolved.get("setting") or f"{tags['compression_family']}-{tags['compression_level']}"
    )
    run_name = resolve_run_name(
        resolved.get("run_name"), datetime.date.today().strftime("%Y%m%d"), "stageC",
        tags["model"], tags["task"], setting, build_configset(B, S, R=R), seed,
    )
    run_dir = allocate_run_dir(resolved.get("run_root", "runs"), run_name, config_hash)
    with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2, sort_keys=True, default=str)

    # --- Algorithm 1 lines 16-17: compress, then re-extract the ensemble ------------
    model = _load_model(resolved)
    compressor = _build_compressor(resolved)
    compressed = compressor.apply(model, resolved)
    magnitudes = compressor.weight_delta(model, resolved)

    configs = _build_threshold_configs(resolved)
    seeds = [seed + i for i in range(S)]
    extractor = PIPELINE_REGISTRY[pipeline](**(resolved.get("extractor_kwargs", {}) or {}))
    runner = CircusEnsembleRunner(extractor)

    dense = runner.run(model=model, task=resolved["task"], configs=configs, seeds=seeds)
    post = runner.run(model=compressed, task=resolved["task"], configs=configs, seeds=seeds)
    f_dense, f_post = dense.freq.frequencies, post.freq.frequencies

    # --- Algorithm 1 lines 18-20: D, CSI, then repeat at routing-head granularity ---
    d_null = _load_dnull(cell)
    n_boot = int((resolved.get("stage_c") or {}).get("n_boot", 1000))

    # --- the chance-floor universe N (AI_RULES.md 4.4), pre-registered 2026-09-12 ----
    # N is the number of edges the extractor COULD have returned, over the nodes it
    # actually saw for these prompts — not over every component in the model. For
    # pipeline A the node basis is ACTIVE transcoder features, so a model-wide count
    # includes nodes the extractor could never have emitted; that inflates N, pushes
    # the random-overlap baseline down, and makes every overlap look above chance for
    # purely combinatorial reasons. Same trap the Q11 aggregate decision avoided.
    # arXiv:2607.18921's candidate sets are 37 and 31 — small universes, honest floors.
    #
    # ONE N PER LEVEL, and structural rather than all-pairs — see
    # _chance_floor_universes for the two errors this replaces (~10x at the exact-edge
    # level, 2400x-8300x at the routing-head level). Computed after the coarse scheme is
    # resolved, since the coarse universe depends on it.

    # --- the coarse comparison level (Q10, pre-registered 2026-09-12) ----------------
    # The scheme comes from configs/comparison/*.yaml. Passing scheme=None would use
    # the literal L{l}.H{h} head projection, which for a pipeline-A circuit matches
    # nothing and raises — the head basis does not exist in circuit-tracer.
    level2_scheme = (resolved.get("comparison") or {}).get("level2_scheme")
    if level2_scheme:
        coarse_pre = project_to_coarse_level(f_dense, level2_scheme)
        coarse_post = project_to_coarse_level(f_post, level2_scheme)
    else:
        coarse_pre = project_to_routing_heads(f_dense)
        coarse_post = project_to_routing_heads(f_post)

    n_universes = _chance_floor_universes(model, f_dense, f_post, level2_scheme)

    # --- per-cell membership, so the CI can be bootstrapped over B, S and r ----------
    # The full grid is passed explicitly: a (config, seed) cell that extracted no edges
    # emits no records, and dropping it would shrink the denominator and inflate s(e).
    grid_keys = [(str(c["id"]), int(s)) for c in configs for s in seeds]
    cells_dense = cells_from_records(dense.records, keys=grid_keys)
    cells_post = cells_from_records(post.records, keys=grid_keys)

    def _coarse(f):
        if level2_scheme:
            return project_to_coarse_level(f, level2_scheme, strict=False)
        return project_to_routing_heads(f, strict=False)

    levels = {
        "exact_edge": ((f_dense, f_post), l1_distance),
        "routing_head": (
            (coarse_pre, coarse_post),
            lambda a, b: l1_distance(_coarse(a), _coarse(b)),
        ),
    }
    rows, floors = [], {}
    for level_name, ((pre, post_vec), distance_fn) in levels.items():
        result = csi_over_ensemble(
            cells_dense, cells_post, d_null, distance_fn, n_boot=n_boot, seed=seed
        )
        rows.append(make_csi_row(
            model=tags["model"], task=tags["task"], family=tags["compression_family"],
            level_param=str(tags["compression_level"]), comparison_level=level_name,
            csi_result=result, null_frozen_hash=null_hash,
            d_normalized=normalized_l1_distance(pre, post_vec),
        ))
        # AI_RULES.md 4.4: every overlap statistic carries its random top-k floor.
        floors[level_name] = overlap_vs_chance_for_freqs(
            pre, post_vec, n_universe=n_universes[level_name], n_draws=500, seed=seed
        )
        floors[level_name]["n_universe"] = n_universes[level_name]

    csi_path = write_csi_table(run_dir / "csi_table.csv", rows)

    # --- post-compression inclusion frequencies + bands (C3 object of record) -------
    cutoffs = dict(ensemble.get("decompose", {}))
    core_thr, noise_thr = cutoffs.get("core_threshold"), cutoffs.get("noise_threshold")
    # noise_strict must match Stage A exactly: banding the post-compression vector
    # under a different boundary than the dense reference would make the band columns
    # incomparable, which is the whole object of record for claim C3.
    bands = (
        decompose(
            f_post,
            float(core_thr),
            float(noise_thr),
            noise_strict=bool(cutoffs.get("noise_strict", False)),
        )
        if core_thr is not None and noise_thr is not None else {}
    )
    schema.write_freq_parquet(
        str(run_dir / "freq_post.parquet"),
        [{"edge_id": eid, "s_e": s, "band": bands.get(eid)} for eid, s in sorted(f_post.items())],
    )

    evidence = not (mode == MODE_ENGINEERING or is_provably_synthetic(resolved, pipeline))
    run_meta = {
        **tags,
        "run_name": run_name, "run_dir": str(run_dir), "seeds": seeds,
        "n_cells": post.n_cells,
        "frozen_cell": str(cell),
        "dnull_median": statistics.median(d_null),
        "d_alternative_jensen_shannon": jensen_shannon_distance(f_dense, f_post),
        "chance_floors": floors,
        "csi_summary": summarize_csi(rows),
        "frobenius_magnitudes": {k: float(v) for k, v in sorted(magnitudes.items())},
        "evidence": evidence,
        "synthetic": is_provably_synthetic(resolved, pipeline),
        "note": (
            "SYNTHETIC DRY-RUN — mock model, synthetic task, mock extractor. Exercises "
            "Algorithm 1 Stage C end-to-end; NOT scientific evidence (AI_RULES.md 2.3)."
            if not evidence else "Stage C measurement."
        ),
        "outputs": {"csi_table.csv": str(csi_path), "freq_post.parquet": str(run_dir / "freq_post.parquet")},
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(schema.json_safe(run_meta), f, indent=2, sort_keys=True, default=str, allow_nan=False)

    exact = next(r for r in rows if r["comparison_level"] == "exact_edge")
    head = next(r for r in rows if r["comparison_level"] == "routing_head")
    print(
        f"[stageC{'' if evidence else '-synthetic'}] {run_name} "
        f"CSI(exact-edge)={exact['csi']:.3f} [{exact['ci_lo']:.3f}, {exact['ci_hi']:.3f}]  "
        f"CSI(routing-head)={head['csi']:.3f} [{head['ci_lo']:.3f}, {head['ci_hi']:.3f}]  "
        f"-> {run_dir}"
    )
    return run_dir


def main() -> None:
    if hydra_main is None:  # pragma: no cover
        print("hydra-core + omegaconf are required to run the entrypoint (optional dep for tests).", file=sys.stderr)
        sys.exit(1)

    @hydra_main(version_base=None, config_path="../configs", config_name="config")
    def _run(cfg: DictConfig) -> None:
        run_stage_c(cfg)

    _run()


if __name__ == "__main__":
    main()
