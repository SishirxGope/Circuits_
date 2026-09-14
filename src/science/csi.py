# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft CSI computation (proposal §2.1 headline metric)
# modified: [AI-GEN] agent=Claude date=2026-09-12 task=Bootstrap over B, S and r as the proposal specifies
# reviewed-by: PENDING
# scientific-status: bootstrap axes PRE-REGISTERED 2026-09-12 (PI-approved) as B, S, r.
#   Novelty-zone edit approved 2026-09-12.
#
# NOVELTY PROTECTION ZONE (AI_RULES.md §3): draft for review only; do not treat as
# approved scientific logic until human approval (APPROVE csi).

"""Circuit Survival Index (CLAUDE.md §5; proposal §2.1 headline metric).

    CSI(c,T) = D(c) / median(D_null(c)), with bootstrap CI over B, S, r.

Interpretation:
- CSI ≈ 1: compression damage indistinguishable from generic weight noise.
- CSI ≫ 1: compression is structurally selective.
- CSI < 1: compression gentler than random noise (surprising, publishable).

Dependencies are injected as plain numbers/arrays: this module does not choose the
distance (src/science/distances.py, Q2 provisional L1/JS) nor the band cutoffs
(Q1). The bootstrap is deterministic given (n_boot, seed) (AI_RULES.md 1.1).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from ..common.seeding import create_seed_generator, derive_child_seed

#: A (config_id, seed) ensemble cell -> the set of edge ids present in that cell.
Cells = Mapping[tuple[str, int], frozenset[str]]


def median_dnull(d_null: np.ndarray) -> float:
    """Median of the null distance distribution (the CSI denominator)."""
    arr = np.asarray(d_null, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("D_null must be non-empty (R >= 1)")
    return float(np.median(arr))


def csi(
    d_c: float,
    d_null: np.ndarray | list[float],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """CSI point estimate + percentile bootstrap CI over the null draws.

    Returns {csi, ci_lo, ci_hi, D, dnull_median}. Raises ValueError if the null
    median is 0 (division undefined) — the caller must surface that as a guard
    failure, never silently.

    The bootstrap resamples D_null (with replacement) and recomputes the median,
    i.e. it propagates uncertainty in the CSI denominator; the percentile CI is
    [2.5%, 97.5%] of the bootstrapped CSI values.
    """
    d_null_arr = np.asarray(d_null, dtype=np.float64)
    # A distance cannot be negative. Without this, a sign error upstream produces a
    # negative CSI that is meaningless but passes every downstream check and lands in
    # the CSI table (added 2026-08-08).
    if float(d_c) < 0.0:
        raise ValueError(f"D(c) must be >= 0 (it is a distance), got {d_c}")
    if np.any(d_null_arr < 0.0):
        raise ValueError(
            f"D_null contains negative distances (min={float(d_null_arr.min())}); "
            f"the frozen null is malformed and CSI must not divide by it"
        )
    med = median_dnull(d_null_arr)
    if med == 0.0:
        raise ValueError(
            "median(D_null) == 0; CSI is undefined. A null with all-zero distances "
            "indicates the perturbation did not move the circuit at all — investigate "
            "the perturbation magnitudes before computing CSI (AI_RULES.md 2.3)."
        )
    d_c_f = float(d_c)
    point = d_c_f / med

    rng, _ = create_seed_generator(derive_child_seed(int(seed), "csi-bootstrap"))
    boot_medians = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        resample = d_null_arr[rng.integers(0, d_null_arr.size, size=d_null_arr.size)]
        boot_medians[b] = np.median(resample)
    lo, hi = np.percentile(d_c_f / boot_medians, [2.5, 97.5])
    return {
        "csi": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "D": d_c_f,
        "dnull_median": med,
    }


def _frequencies_from_cells(
    cells: Cells,
    keys: Sequence[tuple[str, int]],
) -> dict[str, float]:
    """s(e) over an explicit (possibly resampled, possibly repeated) list of cells.

    ``keys`` may contain duplicates — that is the point under bootstrap resampling, and
    a cell drawn twice must count twice, so this counts occurrences rather than
    deduplicating. The denominator is ``len(keys)``, matching
    ``inclusion_freq.compute_from_graphs``.
    """
    n = len(keys)
    if n == 0:
        return {}
    counts: dict[str, int] = {}
    for key in keys:
        for edge in cells.get(key, ()):
            counts[edge] = counts.get(edge, 0) + 1
    return {edge: c / n for edge, c in counts.items()}


def csi_over_ensemble(
    cells_pre: Cells,
    cells_post: Cells,
    d_null: np.ndarray | list[float],
    distance_fn: Callable[[Mapping[str, float], Mapping[str, float]], float],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """CSI with the CI bootstrapped over **B, S and r** (proposal §2.1c, CLAUDE.md §5).

    ``csi()`` resamples only the null draws, so its interval carries the uncertainty in
    the CSI *denominator* and nothing else. But the B threshold views and the S seeds
    are sampled too: a different draw of either gives a different s(e), hence a
    different D(c), hence a different CSI. An interval that ignores them is
    anti-conservative, and the proposal says B, S, r.

    Resampling scheme, pre-registered 2026-09-12:

    - resample the **B config ids** with replacement (B draws),
    - resample the **S seeds** with replacement (S draws),
    - take the crossed B×S grid of the resampled labels, recompute s(e) for the dense
      and compressed ensembles over exactly those cells, and recompute D,
    - independently resample ``d_null`` with replacement and take its median,
    - CSI_b = D_b / median_b; the CI is the [2.5, 97.5] percentile of the CSI_b.

    Both axes are resampled independently because the design is crossed: every config
    is run at every seed, so config effects and seed effects are separately estimable.
    The dense and compressed ensembles are resampled with the **same** labels, because
    they are paired — the whole quantity of interest is the difference between them at
    matched (config, seed), and resampling them independently would inject variance
    that the experiment deliberately controls out.

    ``distance_fn`` is injected rather than imported so this module still does not
    choose the distance (Q2 lives in configs/distance/*.yaml). For the coarse level,
    pass a callable that projects before measuring.

    Returns the same keys as ``csi()`` plus ``n_configs``/``n_seeds``, so a CSI row is
    interchangeable regardless of which estimator produced it.
    """
    if not cells_pre or not cells_post:
        raise ValueError(
            "cells_pre and cells_post must be non-empty: the B x S ensemble is what "
            "the bootstrap resamples, and an empty ensemble has no CSI to report"
        )
    if set(cells_pre) != set(cells_post):
        missing = sorted(set(cells_pre) ^ set(cells_post))[:5]
        raise ValueError(
            "the dense and compressed ensembles must cover the SAME (config, seed) "
            f"cells — they are paired and resampled together; differing at {missing}"
        )

    d_null_arr = np.asarray(d_null, dtype=np.float64)
    if d_null_arr.size == 0:
        raise ValueError("D_null must be non-empty (R >= 1)")
    if np.any(d_null_arr < 0.0):
        raise ValueError(
            f"D_null contains negative distances (min={float(d_null_arr.min())}); "
            "the frozen null is malformed and CSI must not divide by it"
        )

    config_ids = sorted({c for c, _ in cells_pre})
    seeds = sorted({s for _, s in cells_pre})
    all_keys = [(c, s) for c in config_ids for s in seeds]

    point_pre = _frequencies_from_cells(cells_pre, all_keys)
    point_post = _frequencies_from_cells(cells_post, all_keys)
    d_c = float(distance_fn(point_pre, point_post))
    if d_c < 0.0:
        raise ValueError(f"D(c) must be >= 0 (it is a distance), got {d_c}")

    med = median_dnull(d_null_arr)
    if med == 0.0:
        raise ValueError(
            "median(D_null) == 0; CSI is undefined. A null with all-zero distances "
            "indicates the perturbation did not move the circuit at all — investigate "
            "the perturbation magnitudes before computing CSI (AI_RULES.md 2.3)."
        )

    rng, _ = create_seed_generator(derive_child_seed(int(seed), "csi-bootstrap-bsr"))
    boot = np.empty(int(n_boot), dtype=np.float64)
    n_cfg, n_seed = len(config_ids), len(seeds)
    for b in range(int(n_boot)):
        cfg_draw = [config_ids[i] for i in rng.integers(0, n_cfg, size=n_cfg)]
        seed_draw = [seeds[i] for i in rng.integers(0, n_seed, size=n_seed)]
        keys = [(c, s) for c in cfg_draw for s in seed_draw]
        d_b = float(distance_fn(
            _frequencies_from_cells(cells_pre, keys),
            _frequencies_from_cells(cells_post, keys),
        ))
        med_b = float(np.median(
            d_null_arr[rng.integers(0, d_null_arr.size, size=d_null_arr.size)]
        ))
        # A resampled null median of 0 is possible at small R even when the full null
        # median is positive. Dropping those draws would bias the CI; NaN keeps the
        # draw visible and nanpercentile below reports on what is defined.
        boot[b] = d_b / med_b if med_b > 0.0 else np.nan

    finite = boot[np.isfinite(boot)]
    if finite.size == 0:
        raise ValueError(
            "every bootstrap draw had a zero null median; R is too small for a CI at "
            "these magnitudes. Increase R rather than reporting a point without one."
        )
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {
        "csi": d_c / med,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "D": d_c,
        "dnull_median": med,
        "n_configs": n_cfg,
        "n_seeds": n_seed,
        "n_boot_usable": int(finite.size),
    }


def cells_from_records(
    records,
    keys: Sequence[tuple[str, int]] | None = None,
) -> dict[tuple[str, int], frozenset[str]]:
    """Build the (config_id, seed) -> edge-id-set map from edges.parquet records.

    Records are the rows written by ``inclusion_freq.compute_from_graphs``:
    ``src_component``, ``dst_component``, ``config_id``, ``seed``, ``included``. Rows
    with ``included`` false are ignored; an extractor may legitimately emit the same
    pair twice in one cell, and a set collapses that, matching how s(e) counts cells
    rather than records.

    ``keys`` is the full expected (config_id, seed) grid, and passing it matters. A cell
    whose extraction returned NO edges emits no records at all, so it would simply be
    absent here — silently shrinking the ensemble and inflating every s(e), because the
    denominator is the number of cells. An empty view is a real observation ("this
    threshold configuration found nothing"), not a missing one. With ``keys`` supplied,
    such cells are present with an empty edge set and counted in the denominator, which
    is what ``inclusion_freq.compute_from_graphs`` does via its explicit ``n``.
    """
    cells: dict[tuple[str, int], set[str]] = {}
    if keys is not None:
        for key in keys:
            cells[(str(key[0]), int(key[1]))] = set()
    for r in records:
        if not r.get("included", True):
            continue
        key = (str(r["config_id"]), int(r["seed"]))
        cells.setdefault(key, set()).add(f"{r['src_component']}->{r['dst_component']}")
    return {k: frozenset(v) for k, v in cells.items()}


__all__ = ["csi", "csi_over_ensemble", "cells_from_records", "median_dnull"]
