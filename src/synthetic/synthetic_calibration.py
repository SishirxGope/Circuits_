# [AI-GEN] agent=OpenCode date=2026-08-07 task=Synthetic calibration data (Q7 provisional)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY

"""Synthetic calibration data for dry-run compression wrappers (Q7 provisional).

PROVISIONAL (docs/HUMAN_DECISIONS.md §3.2 Q7): synthetic calibration
tokens only; no real GPTQ/AWQ calibration corpus is approved yet. Real calibration
datasets remain OPEN until their licenses are verified (data/README.md log).

The returned token stream is deterministic given (seed, n_tokens, vocab_size) and
mimics the *shape* of a token-level calibration cache (int64 token ids), which is
all the dry-run wrappers consume. It is NOT a language corpus of any kind.
"""

from __future__ import annotations

import numpy as np

from ..common.seeding import create_seed_generator, derive_child_seed


def synthetic_calibration_tokens(
    seed: int = 0, n_tokens: int = 512, vocab_size: int = 256
) -> np.ndarray:
    """Deterministic int64 token-id stream (engineering dry-runs only)."""
    if n_tokens <= 0 or vocab_size <= 1:
        raise ValueError(f"n_tokens and vocab_size must be positive (got {n_tokens}, {vocab_size})")
    rng, _ = create_seed_generator(derive_child_seed(int(seed), "synthetic-calibration"))
    return rng.integers(0, int(vocab_size), size=int(n_tokens), dtype=np.int64)


def load_synthetic_calibration(cfg) -> dict:
    """Load the synthetic calibration set described by a (compression) config.

    cfg keys read: ``calibration`` -> {seed, n_tokens, vocab_size} or None.
    Returns a dict with ``tokens`` (ndarray) and provenance metadata; always
    marks the set as synthetic so no downstream code mistakes it for a real corpus.
    """
    cal = (cfg or {}).get("calibration") or {}
    return {
        "tokens": synthetic_calibration_tokens(
            seed=int(cal.get("seed", 0) or 0),
            n_tokens=int(cal.get("n_tokens", 512) or 512),
            vocab_size=int(cal.get("vocab_size", 256) or 256),
        ),
        "synthetic": True,
        "provenance": "src/synthetic/synthetic_calibration.py (Q7 provisional; no real corpus approved)",
    }


__all__ = ["synthetic_calibration_tokens", "load_synthetic_calibration"]
