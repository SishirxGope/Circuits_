# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft magnitude/Wanda pruner (adaptation of saediag.pruning surface)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY
#
# Adapted from: https://github.com/hecboar/sae-pruning-paper @ 261191804675e2d39d0a265320dbc0bc85afd30a, MIT
#   (arXiv:2603.25325 code release, 2026-07-31; licence in THIRD_PARTY_LICENSES/)
#   - local fork: sae-pruning-paper-main, revision/src/saediag/pruning.py (read-only).
#     Upstream function names verified against the fork; this module re-implements the
#     pruning MATH on numpy registries for engineering (deterministic, no models), it
#     does not import the upstream package.
#   - Q9 RESOLVED 2026-09-12, but note the pin is INFERRED from the fork's file contents,
#     not read off a URL — the fork is not a git repo. The successor commit c8cce94 adds a
#     file the fork lacks and has this commit as its only parent; bd85878 (one later)
#     removes a word the fork still contains. Derivation and the one outstanding check are
#     in docs/HUMAN_DECISIONS.md §3.3.
#   - license: THIRD_PARTY_LICENSES/sae-pruning-paper-LICENSE.txt (verified 2026-08-07).

"""Unstructured weight pruning (compression grid: magnitude + Wanda, 0-60% sparsity,
proposal §3.2; the ref [1] feature audit uses magnitude + Wanda 0-60% on the same
primary models, which is what makes C5 comparable).

Upstream interface surface (names verified against the fork; signatures confirmed at
Stage A)::

    prune_magnitude_inplace(model, target_sparsity, ...)
    prune_magnitude_permatrix_inplace(model, target_sparsity, exclude=...)
    prune_magnitude_global_inplace(model, target_sparsity, n_iter=...)
    prune_wanda_style_inplace(model, ex2_by_linear, ...)
    collect_linear_input_second_moment_from_cache(model, token_cache, ...)
    iter_linear_modules(model); global_sparsity_linear_weights(model, exclude=("lm_head",))

SCIENCE REQUIREMENTS (preserved from the earlier skeleton):
1. The dense reference is NEVER mutated: every engineering function below operates on
   copies (MockModel.apply_weight_delta / clone semantics, ARCHITECTURE.md §2).
2. ``weight_delta`` remains the single source of truth for the per-tensor ||Delta
   W||_F magnitudes the matched-magnitude null must match (ARCHITECTURE.md §4).
3. All randomness is seeded from config and logged (AI_RULES.md 1.1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..interfaces import CompressedModel, Model, PerTensorFrobenius
from ..synthetic.mock_model import MockModel


def _topk_keep_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Boolean keep-mask selecting exactly the ``k`` largest ``scores``.

    Ties are broken by flat index order, so the mask is deterministic (no RNG) and
    the retained count is exactly ``k`` even when many entries share a magnitude.
    Shape-agnostic: works for 1-D bias/norm tensors as well as 2-D weight matrices
    (real checkpoints have both).
    """
    flat = np.asarray(scores, dtype=np.float64).ravel()
    if k <= 0:
        return np.zeros(scores.shape, dtype=bool)
    if k >= flat.size:
        return np.ones(scores.shape, dtype=bool)
    # argsort on (-score, index) via a stable sort of the negated scores: equal
    # scores keep ascending index order, so the choice among ties is deterministic.
    keep_idx = np.argsort(-flat, kind="stable")[:k]
    mask = np.zeros(flat.size, dtype=bool)
    mask[keep_idx] = True
    return mask.reshape(scores.shape)


def _magnitude_mask(weight: np.ndarray, keep_fraction: float) -> np.ndarray:
    """Keep mask for unstructured magnitude pruning of one tensor (deterministic)."""
    w = np.abs(np.asarray(weight, dtype=np.float64))
    return _topk_keep_mask(w, int(np.floor(keep_fraction * w.size)))


def _wanda_score(weight: np.ndarray, second_moment: np.ndarray) -> np.ndarray:
    """Wanda score |w| * sqrt(s2) (per-element); s2 = input activation 2nd moment."""
    w = np.abs(np.asarray(weight, dtype=np.float64))
    s2 = np.asarray(second_moment, dtype=np.float64)
    if w.shape != s2.shape:
        raise ValueError(f"second-moment shape {s2.shape} != weight shape {w.shape}")
    return w * np.sqrt(np.maximum(s2, 0.0))


def achieved_sparsity(
    weights: Mapping[str, np.ndarray],
    exclude: Sequence[str] = ("lm_head",),
    embedding_names: Sequence[str] = (),
) -> dict[str, float]:
    """Per-tensor and overall fraction of zeros among the ZEROABLE tensors.

    The LABELLED sparsity of a grid cell is not the sparsity any given matrix ends up
    with (see ``prune_magnitude``), so every run must report what it actually achieved
    rather than what it asked for.

    ``embedding_names`` must be passed whenever it was passed to ``prune_magnitude``.
    Embeddings join the threshold pool but are never zeroed, so counting them in the
    denominator dilutes the reported sparsity by however large the embedding matrix is
    — which for a real model is enormous, and would make a 50% prune report as a few
    percent. Upstream measures over Linear weights only
    (``saediag.pruning.global_sparsity_linear_weights``).
    """
    skip = set(exclude) | set(embedding_names)
    zeroable = {k: np.asarray(v) for k, v in weights.items() if k not in skip}
    out = {k: float((v == 0).sum()) / float(v.size) for k, v in sorted(zeroable.items())}
    total = sum(v.size for v in zeroable.values())
    zeros = sum(int((v == 0).sum()) for v in zeroable.values())
    out["__overall__"] = (zeros / total) if total else 0.0
    return out


def prune_magnitude(
    weights: Mapping[str, np.ndarray],
    target_sparsity: float,
    global_scope: bool = True,
    exclude: Sequence[str] = ("lm_head",),
    embedding_names: Sequence[str] = (),
    include_embedding_in_threshold: bool = True,
) -> dict[str, np.ndarray]:
    """Prune to ``target_sparsity`` (fraction removed), return pruned copies.

    Global scope: one global threshold across the pooled magnitudes; every prunable
    tensor is then zeroed below it. Per-tensor scope: each tensor keeps (1 - sparsity)
    of its own weights. Excluded tensors are returned unchanged. Deterministic (no
    RNG), never mutates inputs.

    ``embedding_names`` + ``include_embedding_in_threshold`` reproduce the upstream
    reference behaviour (``saediag.pruning.prune_magnitude_global_inplace``, verified
    2026-08-08): the **token-embedding matrix joins the threshold pool but is never
    zeroed**. That asymmetry is not incidental — upstream documents that it "is what
    maps the labeled sparsity (e.g. 0.30) to the observed effective per-matrix
    sparsities (~0.26 attn, ~0.38 MLP)".

    ⚠️ **LABELLED SPARSITY IS NOT ACHIEVED SPARSITY.** Under global scope a "30% cell"
    does not remove 30% of every matrix, and if the embedding joins the pool it does not
    remove 30% overall either. Two consequences for this project:

    1. ``weight_delta`` (hence the matched-magnitude null) is matched to the ACHIEVED
       perturbation, which is correct — but the cell LABEL is not a per-matrix sparsity
       and must not be described as one in the paper.
    2. PRD.md §2 requires our pruning grid to match ref [1]'s grid. If our "30%" is
       computed over a different pool than theirs, the C5 cross-audit compares cells
       that are not the same cell. Use ``embedding_names`` so the pools agree.

    Call ``achieved_sparsity()`` on the result and record it in the run log.
    """
    if not (0.0 <= target_sparsity <= 1.0):
        raise ValueError(f"target_sparsity must be in [0, 1], got {target_sparsity}")
    exclude_set = set(exclude)
    embed_set = set(embedding_names)
    unknown_embed = sorted(embed_set - set(weights))
    if unknown_embed:
        raise ValueError(f"embedding_names not present in weights: {unknown_embed}")
    # Embeddings are never zeroed: upstream excludes them from zeroing because embed is
    # tied to the excluded lm_head. They only ever contribute to the threshold pool.
    prunable = {
        k: np.asarray(v, dtype=np.float64)
        for k, v in weights.items()
        if k not in exclude_set and k not in embed_set
    }
    frozen = {
        k: np.asarray(v, dtype=np.float64)
        for k, v in weights.items()
        if k in exclude_set or k in embed_set
    }
    out: dict[str, np.ndarray] = dict(frozen)

    if global_scope:
        # One global keep-set across all prunable tensors: concatenate magnitudes,
        # take the exact global top-k, then scatter the mask back per tensor. This
        # replaces a threshold-plus-tie-repair loop that read the ORIGINAL weights
        # while writing the PRUNED ones, so it could zero an already-zero entry and
        # under-prune (fixed 2026-08-08).
        names = sorted(prunable)  # deterministic concatenation order
        if target_sparsity >= 1.0:
            for k in names:
                out[k] = np.zeros_like(prunable[k])
            return out
        if target_sparsity <= 0.0:
            out.update(prunable)
            return out

        # The THRESHOLD POOL may be larger than the set of tensors that get zeroed:
        # upstream adds the token embedding to the pool but never zeroes it. The
        # threshold is the target_sparsity-quantile of |w| over the pool; every
        # prunable tensor is then zeroed below that threshold. This is why the achieved
        # per-matrix sparsity differs from the labelled one.
        pool_names = list(names)
        if include_embedding_in_threshold:
            pool_names += sorted(embed_set)
        pool = (
            np.concatenate([np.abs(np.asarray(weights[k], dtype=np.float64)).ravel() for k in pool_names])
            if pool_names else np.empty(0)
        )
        n_drop_from_pool = int(np.floor(target_sparsity * pool.size))
        if n_drop_from_pool <= 0:
            out.update(prunable)
            return out
        # threshold = the n_drop-th smallest magnitude in the pool
        threshold = float(np.partition(pool, n_drop_from_pool - 1)[n_drop_from_pool - 1])
        for k in names:
            v = prunable[k]
            out[k] = np.where(np.abs(v) > threshold, v, 0.0)

        # A strict `>` threshold cannot hit an exact count when many magnitudes tie at
        # the threshold value: every tied weight falls on the same side and the prune
        # overshoots. On real float weights exact ties are vanishingly unlikely, but a
        # silent jump from "remove 25%" to "removed everything" would produce a dead
        # model whose CSI is meaningless, so it fails loudly instead.
        kept = sum(int((out[k] != 0).sum()) for k in names)
        if kept == 0 and target_sparsity < 1.0:
            n_at_threshold = int((pool == threshold).sum())
            raise ValueError(
                f"global magnitude pruning at target_sparsity={target_sparsity} zeroed EVERY "
                f"prunable weight: {n_at_threshold} of {pool.size} pooled magnitudes tie at the "
                f"threshold {threshold!r}, and a strict `|w| > threshold` rule drops all of them. "
                f"This reproduces the upstream rule (saediag.pruning.prune_magnitude_global_inplace) "
                f"but the result is a dead model. Use global_scope=False for exact per-tensor "
                f"counts, or check for a degenerate weight distribution."
            )
    else:
        for k, v in prunable.items():
            mask = _magnitude_mask(v, 1.0 - target_sparsity)
            out[k] = np.where(mask, v, 0.0)
    return out


def prune_wanda(
    weights: Mapping[str, np.ndarray],
    second_moments: Mapping[str, np.ndarray],
    target_sparsity: float,
    exclude: Sequence[str] = ("lm_head",),
) -> dict[str, np.ndarray]:
    """Per-tensor Wanda pruning: zero the lowest |w|*sqrt(s2) entries."""
    if not (0.0 <= target_sparsity <= 1.0):
        raise ValueError(f"target_sparsity must be in [0, 1], got {target_sparsity}")
    out: dict[str, np.ndarray] = {}
    for k, v in weights.items():
        if k in set(exclude):
            out[k] = np.asarray(v, dtype=np.float64)
            continue
        if k not in second_moments:
            raise ValueError(f"missing second moment for tensor {k}")
        score = _wanda_score(v, second_moments[k])
        mask = _magnitude_mask(score, 1.0 - target_sparsity)
        out[k] = np.where(mask, np.asarray(v, dtype=np.float64), 0.0)
    return out


class MagnitudePruner:
    """Magnitude pruner (Compressor protocol) — engineering draft.

    Real models: NotImplementedError (Stage C approval + RUN MODEL DOWNLOAD +
    upstream saediag package). MockModel (numpy registry): the pure helpers above.
    """

    def __init__(self, sparsity: float = 0.3, global_scope: bool = True, exclude: tuple[str, ...] = ("lm_head",)) -> None:
        self.sparsity = float(sparsity)
        self.global_scope = bool(global_scope)
        self.exclude = tuple(exclude)

    def apply(self, model: Model, cfg: Any) -> CompressedModel:
        if isinstance(model, MockModel):
            pruned_weights = prune_magnitude(
                model.weights, self.sparsity, global_scope=self.global_scope, exclude=self.exclude
            )
            return MockModel(
                seed=model.seed,
                n_layers=model.n_layers,
                n_heads=model.n_heads,
                d_model=model.d_model,
                weights=pruned_weights,
            )
        raise NotImplementedError(
            "real-model magnitude pruning wraps saediag.pruning (sae-pruning-paper @ "
            "261191804675…) and requires Stage C approval + RUN MODEL DOWNLOAD"
        )

    def weight_delta(self, model: Model, cfg: Any) -> PerTensorFrobenius:
        if isinstance(model, MockModel):
            return model.weight_delta_frobenius(self.apply(model, cfg))
        raise NotImplementedError(
            "real-model magnitude weight_delta requires Stage C approval; "
            "use the engineering path (MockModel) for dry-runs"
        )


__all__ = ["MagnitudePruner", "prune_magnitude", "prune_wanda"]
