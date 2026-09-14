# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft core/contingent/noise decomposition skeleton (proposal §2.3)
# modified: [AI-GEN] agent=Claude date=2026-09-12 task=Q1 pre-registration: strict noise boundary so the cutoff is not B-dependent
# reviewed-by: PENDING
# scientific-status: Q1 PRE-REGISTERED 2026-09-12 (PI-approved): CIRCUS §3.2 taxonomy
#
# NOVELTY PROTECTION ZONE (AI_RULES.md §3): draft for review only; do not treat as
# approved scientific logic until human approval (APPROVE decompose).

"""Core / contingent / noise decomposition (proposal §2.3; CLAUDE.md §5).

Partitions edges by their inclusion frequency s(e) into three bands, per CIRCUS
(arXiv:2603.00523): core ~ retained across nearly all configs; contingent ~ retained
in some; noise ~ rarely retained.

THE CUTOFFS ARE A SCIENTIFIC DECISION THAT BELONGS TO THE PI. They are read from
configs/ensemble/decompose/final.yaml and were pre-registered on 2026-09-12 as CIRCUS's
own taxonomy (arXiv:2603.00523 §3.2 "Circuit taxonomy"):

    core        s(e) = 1        present in EVERY view (CIRCUS's C_1)
    contingent  0.5 <= s(e) < 1
    noise       s(e) < 0.5      "flagged for rejection"

Boundary semantics, which are testable and unambiguous:
- s(e) >= core_threshold -> core
- s(e) <  noise_threshold -> noise      when ``noise_strict`` (the CIRCUS reading)
- s(e) <= noise_threshold -> noise      when not ``noise_strict`` (the original reading)
- otherwise -> contingent

**Why the strict flag exists.** CIRCUS puts s = 0.5 in the *contingent* band, so the
boundary is ``< 0.5``, not ``<= 0.5``. Under the inclusive comparison the only way to
encode that is to write the largest achievable s strictly below 0.5, which is
``(ceil(B/2) - 1)/B`` — 7/16 at B=16. That silently hard-codes B into a config whose
name never mentions B, and it changes meaning if B ever changes. The flag lets the
config say the honest ``0.5`` and keeps the cutoff independent of B.

Raising on unresolved cutoffs is deliberate: a run must never silently produce band
labels that were never approved.
"""

from __future__ import annotations

from collections.abc import Mapping

CORE = "core"
CONTINGENT = "contingent"
NOISE = "noise"


def decompose(
    frequencies: Mapping[str, float],
    core_threshold: float | None,
    noise_threshold: float | None,
    noise_strict: bool = False,
) -> dict[str, str]:
    """Assign a band to every edge given (core_threshold, noise_threshold).

    ``noise_strict`` selects the comparison at the noise boundary: ``s < noise_threshold``
    when True (CIRCUS §3.2, which places s = noise_threshold in the contingent band),
    ``s <= noise_threshold`` when False. It defaults to False so every existing caller
    and every pre-2026-09-12 run keeps its exact behaviour; the pre-registered Stage A
    path passes True from configs/ensemble/decompose/final.yaml.

    Raises ValueError with an informative message if either cutoff is unresolved
    (None) or if the cutoffs are degenerate (noise_threshold >= core_threshold).

    Returns a deterministic {edge_id: band} mapping (sorted by edge_id).
    """
    if core_threshold is None or noise_threshold is None:
        raise ValueError(
            "core/contingent/noise band cutoffs are unresolved: "
            "set core_threshold and noise_threshold in configs/ensemble/decompose/final.yaml. "
            "⚠️ TODO [QUESTION FOR PI]: cutoffs require PI approval (CLAUDE.md §5; AI_RULES.md §6)."
        )
    if not (0.0 <= noise_threshold < core_threshold <= 1.0):
        raise ValueError(
            f"degenerate band cutoffs: core_threshold={core_threshold}, "
            f"noise_threshold={noise_threshold}; need 0.0 <= noise < core <= 1.0"
        )

    # s(e) is a fraction of ensemble cells. A value outside [0, 1] means the ensemble
    # counting is broken upstream, and banding it would silently launder that into a
    # "core" label (added 2026-08-08).
    bad = {k: v for k, v in frequencies.items() if not (0.0 <= float(v) <= 1.0)}
    if bad:
        raise ValueError(
            f"inclusion frequencies outside [0, 1]: {dict(sorted(bad.items())[:5])}. "
            f"s(e) must be a fraction; fix the ensemble counting before banding."
        )

    bands: dict[str, str] = {}
    for edge_id_, s in sorted(frequencies.items()):
        if s >= core_threshold:
            bands[edge_id_] = CORE
        elif (s < noise_threshold) if noise_strict else (s <= noise_threshold):
            bands[edge_id_] = NOISE
        else:
            bands[edge_id_] = CONTINGENT
    return bands


__all__ = ["decompose", "CORE", "CONTINGENT", "NOISE"]
