# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft interaction-aware patching diagnostic (proposal §2.5)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY
#
# NOVELTY PROTECTION ZONE (AI_RULES.md §3): draft for review only; do not treat as
# approved scientific logic until human approval (APPROVE patch_diagnostic).

"""Interaction-aware patching diagnostic (proposal §2.5; claim C4).

NIE = PIE + INT (ref arXiv:2606.27510): the natural indirect effect decomposes into
the pure indirect effect plus an interaction term. A component is
**interaction-dominated** when the grouped-vs-single patching diagnostic shows its
effect depends on sibling states; such components are EXCLUDED from headline
stability claims and reported separately (CLAUDE.md §5).

Definitions (PROVISIONAL until final pre-registration at Stage D, AI_RULES.md 4.2):
- NIE  := effect of patching the edge while siblings keep their grouped state.
- PIE  := effect of patching the edge with siblings fixed at their clean state.
- INT  := NIE - PIE (the decomposition identity).
- interaction-dominated flag: |INT| > ratio_threshold * max(|NIE|, |PIE|, eps)
  with ratio_threshold = 0.5 provisional (config-driven, never hardcoded in the
  science: the value belongs in configs/causal/*.yaml once that group exists).

The real patching machinery (model intervention loops) is Stage D engineering; this
module defines the pure arithmetic + flag rule that is testable now.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..interfaces import Model

INT_RATIO_THRESHOLD_DEFAULT = 0.5  # provisional; final pre-registration at Stage D


def nie_pie_int(pie: float, nie: float) -> dict[str, float]:
    """Decompose NIE = PIE + INT given the two measured effects."""
    return {"nie": float(nie), "pie": float(pie), "int": float(nie) - float(pie)}


def interaction_dominated(
    pie: float,
    nie: float,
    ratio_threshold: float = INT_RATIO_THRESHOLD_DEFAULT,
) -> bool:
    """Flag rule (PROVISIONAL): |INT| > ratio_threshold * max(|NIE|, |PIE|, eps)."""
    eps = 1e-12
    int_term = abs(float(nie) - float(pie))
    scale = max(abs(float(nie)), abs(float(pie)), eps)
    return int_term > float(ratio_threshold) * scale


class PatchDiagnostic:
    """Orchestration skeleton: run grouped-vs-single patching for an edge.

    ``run(model, edge, siblings)`` raises NotImplementedError for real models: the
    intervention machinery is Stage D engineering (requires real models + upstream
    patching APIs). The pure arithmetic above is the tested part.
    """

    def __init__(self, ratio_threshold: float = INT_RATIO_THRESHOLD_DEFAULT) -> None:
        self.ratio_threshold = float(ratio_threshold)
        self.provisional = True  # flag rule is provisional (Stage D pre-registration)

    def run(self, model: Model, edge: Any, siblings: Sequence[Any]) -> dict[str, Any]:
        _ = (model, edge, siblings)
        raise NotImplementedError(
            "PatchDiagnostic.run requires real models + patching APIs (Stage D "
            "engineering) and an explicit PI approval for interaction measurement. "
            "The NIE/PIE/INT arithmetic and the flag rule are implemented in this "
            "module for review (interaction_dominated)."
        )


__all__ = ["PatchDiagnostic", "nie_pie_int", "interaction_dominated", "INT_RATIO_THRESHOLD_DEFAULT"]
