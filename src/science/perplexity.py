# [AI-GEN] agent=Claude date=2026-08-08 task=Create the perplexity module that CLAUDE.md §3 and matched_perplexity.py reference but that did not exist
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY
#
# NOVELTY PROTECTION ZONE (AI_RULES.md §3): draft for review only; do not treat as
# approved scientific logic until human approval (APPROVE perplexity).

"""Perplexity evaluation (proposal §2.1b; PRD.md §2 baseline requirements).

Two consumers:

1. **Matched-perplexity null** (src/science/matched_perplexity.py): the alternate
   compression family is tuned until its PPL matches the setting under test. Both
   sides MUST be evaluated on the SAME corpus with the SAME protocol, or the "match"
   is meaningless (precedent: Duan, arXiv:2606.03002; the sae-pruning-paper fork's
   revision notes exist precisely because an earlier PPL protocol was wrong).
2. **The conventional acceptance criterion** the paper argues is insufficient:
   PPL / task accuracy of every compressed model is reported alongside CSI, so a
   reader can see that a cell passed the usual bar and still moved the circuit.

Implemented here: the pure aggregation arithmetic (token-weighted mean NLL ->
perplexity), which is testable now and is the piece most often gotten wrong (naive
averaging of per-batch perplexities is NOT perplexity). The model forward pass is
Stage B engineering and requires real models + RUN MODEL DOWNLOAD approval.

THE REFERENCE PROTOCOL (verified against the fork, not invented). The
sae-pruning-paper fork's ``docs/PROTOCOL.md`` documents the corrected WikiText-2
protocol its published numbers use, and `revision/src/saediag/ppl.py::windowed_ppl`
implements it. The constants below are transcribed from that table.

That document exists because an earlier protocol was WRONG in a way that is worth
internalising: it prepended ``<bos>`` once to the whole concatenated corpus instead of
to every window, and evaluated bf16-trained models in fp16. Gemma-2-2B's dense
perplexity came out as **410 instead of ~11**. Llama looked merely plausible, so the
fault was invisible without a forensic sweep.

Two consequences for this project:

1. Our matched-perplexity null (§2.1b) tunes an alternate family to "the same
   perplexity". If our PPL protocol differs from the reference's, "the same" is
   meaningless and claim C7 measures our protocol difference, not the models.
2. The cross-audit (C5) compares our damage ordering against *their published
   numbers*. Those numbers were produced under exactly this protocol.

⚠️ TODO [QUESTION FOR PI]: adopting the reference protocol verbatim is strongly
recommended and is what these defaults encode, but it is a decision that must be
recorded before the Stage B freeze (docs/HUMAN_DECISIONS.md Q7).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ..interfaces import Model

# Transcribed from sae-pruning-paper-main/docs/PROTOCOL.md ("The corrected protocol").
# [VERIFIED: read from the local fork, 2026-08-08]
REFERENCE_PROTOCOL: dict[str, Any] = {
    "corpus": "wikitext-2-raw-v1, full test split (~289K tokens)",
    "window": 1024,
    "stride": 512,
    "bos_policy": "per_window",   # <bos> prepended to EVERY window — see module docstring
    "dtype": "bfloat16",
    "attn_implementation_gemma2": "eager",  # Gemma-2 attention-logit softcapping
    "reported_quantity": "delta_log_ppl",   # log-scaled: pruned PPL spans orders of magnitude
    "source": "sae-pruning-paper-main/docs/PROTOCOL.md + revision/src/saediag/ppl.py",
}
BOS_POLICIES: tuple[str, ...] = ("per_window", "single_leading", "none")


def perplexity_from_nll(total_nll: float, total_tokens: int) -> float:
    """Perplexity from a summed negative log-likelihood over ``total_tokens``.

    ``exp(sum_nll / n_tokens)``. Token-weighted by construction: this is the only
    correct aggregation across unequal-length windows.
    """
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be > 0, got {total_tokens}")
    if not math.isfinite(total_nll):
        raise ValueError(f"total_nll must be finite, got {total_nll}")
    return math.exp(float(total_nll) / int(total_tokens))


def aggregate_perplexity(window_nlls: Sequence[float], window_tokens: Sequence[int]) -> float:
    """Token-weighted perplexity over per-window summed NLLs.

    NOT the mean of per-window perplexities: windows differ in token count, and
    averaging perplexities silently over-weights short windows.
    """
    if len(window_nlls) != len(window_tokens):
        raise ValueError(
            f"window_nlls ({len(window_nlls)}) and window_tokens ({len(window_tokens)}) must align"
        )
    if not window_nlls:
        raise ValueError("no evaluation windows: perplexity is undefined")
    return perplexity_from_nll(float(sum(window_nlls)), int(sum(window_tokens)))


def relative_ppl_gap(ppl_a: float, ppl_b: float) -> float:
    """|ppl_a - ppl_b| / ppl_b — the quantity the matched-PPL tolerance is applied to."""
    if ppl_b <= 0.0:
        raise ValueError(f"reference perplexity must be > 0, got {ppl_b}")
    return abs(float(ppl_a) - float(ppl_b)) / float(ppl_b)


def delta_log_ppl(ppl_compressed: float, ppl_dense: float) -> float:
    """``log PPL_compressed − log PPL_dense`` — the reference's reported degradation.

    The fork reports output degradation on a log scale because pruned perplexities span
    several orders of magnitude (docs/PROTOCOL.md). Reporting a raw difference would let
    one catastrophic cell dominate every aggregate. Our cross-audit (C5) compares
    against numbers on this scale, so we must produce them on this scale.
    """
    if ppl_compressed <= 0.0 or ppl_dense <= 0.0:
        raise ValueError(
            f"perplexities must be > 0, got compressed={ppl_compressed}, dense={ppl_dense}"
        )
    return math.log(float(ppl_compressed)) - math.log(float(ppl_dense))


def validate_ppl_protocol(protocol: dict[str, Any]) -> list[str]:
    """Warn about deviations from the verified reference protocol.

    Returns a list of human-readable deviation strings (empty = matches the reference).
    Deviating is allowed — it is a PI decision — but it must be a *decision*, and the
    matched-perplexity null and the C5 comparison both depend on it, so a silent
    deviation is the single easiest way to produce numbers that cannot be compared with
    the published ones.
    """
    deviations = []
    for key in ("window", "stride", "bos_policy", "dtype"):
        expected = REFERENCE_PROTOCOL[key]
        actual = protocol.get(key)
        if actual is not None and actual != expected:
            deviations.append(
                f"{key}: {actual!r} != reference {expected!r} "
                f"(sae-pruning-paper docs/PROTOCOL.md)"
            )
    if protocol.get("bos_policy") in ("single_leading", "none"):
        deviations.append(
            "bos_policy is NOT per_window: this is the exact fault that made the "
            "reference's Gemma-2-2B perplexity read 410 instead of ~11. Gemma models are "
            "highly sensitive to a missing leading <bos>; Llama is not, so the symptom is "
            "a plausible-looking Llama number next to a broken Gemma one."
        )
    return deviations


def evaluate_perplexity(model: Model, corpus: Any, window: int = 1024, stride: int = 512,
                        bos_policy: str = "per_window") -> float:
    """Perplexity of ``model`` on ``corpus`` under the pinned PPL protocol.

    Defaults are the verified reference protocol (REFERENCE_PROTOCOL). Raises
    NotImplementedError: the forward pass needs real models, the pinned evaluation
    corpus (Q7) and RUN MODEL DOWNLOAD approval. The arithmetic above is the tested
    part; the upstream implementation to wrap is
    ``sae-pruning-paper-main/revision/src/saediag/ppl.py::windowed_ppl``.
    """
    if bos_policy not in BOS_POLICIES:
        raise ValueError(f"bos_policy must be one of {BOS_POLICIES}, got {bos_policy!r}")
    _ = (model, corpus, window, stride)
    raise NotImplementedError(
        "perplexity evaluation requires real models + the pinned evaluation corpus "
        "(docs/HUMAN_DECISIONS.md Q7) and an explicit RUN MODEL DOWNLOAD approval. "
        "Wrap sae-pruning-paper revision/src/saediag/ppl.py::windowed_ppl(model, "
        "content_ids, window, stride, bos_policy, tokenizer, ...) so our numbers are "
        "produced by the same code path as the published ones. "
        "Use perplexity_from_nll / aggregate_perplexity for the arithmetic."
    )


__all__ = [
    "REFERENCE_PROTOCOL",
    "BOS_POLICIES",
    "perplexity_from_nll",
    "aggregate_perplexity",
    "relative_ppl_gap",
    "delta_log_ppl",
    "validate_ppl_protocol",
    "evaluate_perplexity",
]
