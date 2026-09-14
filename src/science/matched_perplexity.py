# [AI-GEN] agent=OpenCode date=2026-08-07 task=Draft matched-perplexity null control skeleton (proposal §2.1b)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY
#
# NOVELTY PROTECTION ZONE (AI_RULES.md §3): draft for review only; do not treat as
# approved scientific logic until human approval (APPROVE matched_perplexity).

"""Null model (b): matched-perplexity control (proposal §2.1b).

An alternate compression family tuned to the same perplexity as the setting under
test (precedent: Duan et al., arXiv:2606.03002). Used to ask: is the circuit change
specific to HOW the model was compressed, or shared by ANY compression at the same
model quality?

Engineering skeleton (PROVISIONAL):
- The tuning loop needs real models + perplexity evaluation (src/science/perplexity.py,
  not yet implemented) — it raises NotImplementedError for real models.
- The dry-run stub returns a clearly-marked provisional placeholder object so the
  Stage B orchestration skeleton can be wired and tested without models.
- Real perplexity evaluation + family tuning land in Stage B engineering; the PPL
  protocol follows the sae-pruning-paper fork revision notes (docs/PROTOCOL.md,
  sae-pruning-paper README "What changed in the revision").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..interfaces import Model

MATCHED_PPL_TOLERANCE_DEFAULT = 0.05  # provisional; PI-owned before Stage B (Q2 adjacent)


class MatchedPerplexityControl:
    """Tune an alternate compression family to a target perplexity.

    Engineering stub only: real tuning requires real models, a perplexity evaluator
    and a compression family; all of those are Stage B engineering + PI-gated.
    """

    def tune(
        self,
        model: Model,
        target_ppl: float,
        families: Sequence[str] = ("rtn", "gptq"),
        tolerance: float = MATCHED_PPL_TOLERANCE_DEFAULT,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Return a matched-perplexity configuration descriptor.

        Raises NotImplementedError for real models. For dry-runs this method is not
        callable either — use ``provisional_placeholder`` (below) instead, which the
        Stage B skeleton consumes with an explicit provisional marker.
        """
        _ = (model, target_ppl, families, tolerance, seed)
        raise NotImplementedError(
            "matched-perplexity tuning requires real models + a perplexity evaluator "
            "(Stage B engineering) and the PPL protocol from the sae-pruning-paper "
            "fork docs/PROTOCOL.md. PROVISIONAL until PI final confirmation."
        )

    @staticmethod
    def provisional_placeholder(target_ppl: float) -> dict[str, Any]:
        """Clearly-marked placeholder for dry-run orchestration (NOT a real match)."""
        return {
            "control": "matched-perplexity",
            "target_ppl": float(target_ppl),
            "matched": False,
            "status": "PROVISIONAL_ENGINEERING_ONLY",
            "note": "placeholder for Stage B skeleton wiring; real tuning is Stage B engineering",
        }


def is_placeholder(match: Mapping[str, Any]) -> bool:
    """True if a matched-perplexity result is the provisional placeholder."""
    return bool(match.get("status") == "PROVISIONAL_ENGINEERING_ONLY" or match.get("matched") is False)


__all__ = ["MatchedPerplexityControl", "is_placeholder"]
