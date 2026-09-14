# [AI-GEN] agent=OpenCode date=2026-08-07 task=Seeded RNG utilities (AI_RULES.md 1.1)
# reviewed-by: PENDING

"""Seeded randomness utilities (AI_RULES.md 1.1).

Rule 1.1: all randomness is seeded and logged; no ``np.random.*`` / ``torch.rand*``
call without a seeded generator object. These helpers are the ONLY sanctioned way to
obtain generators:

- ``create_seed_generator(seed)`` -> (numpy Generator, torch Generator or None).
  Pure construction: no global RNG state is touched, so behavior never depends on
  module-level state.
- ``derive_child_seed(base_seed, *parts)`` -> a deterministic derived seed for nested
  loops (e.g. per (seed, config_id) draws), so sub-runs are reproducible in isolation.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

TORCH_SEED_BITS: int = 63  # keep derived seeds inside torch's supported range


def create_seed_generator(seed: int) -> tuple[np.random.Generator, Any | None]:
    """Return an explicit, seeded (numpy Generator, torch Generator|None) pair.

    Neither generator is drawn from global RNG state; two calls with the same seed
    yield identical sequences (tested in tests/unit/test_seeding.py). The torch
    generator is None when torch is not installed.
    """
    np_gen = np.random.default_rng(seed=int(seed))
    torch_gen = None
    try:
        import torch

        torch_gen = torch.Generator()
        torch_gen.manual_seed(int(seed))
    except ImportError:  # pragma: no cover - torch optional in some CI
        pass
    return np_gen, torch_gen


def derive_child_seed(base_seed: int, *parts: str | int) -> int:
    """Deterministic child seed in [0, 2**63) derived from (base_seed, *parts).

    Example: ``derive_child_seed(0, "config-7")`` always returns the same int, so a
    per-(seed, config) RNG can be recreated anywhere with just the logged seed and
    the config id (AI_RULES.md 1.1: seeds are logged, never hidden).
    """
    joined = "|".join([str(base_seed), *(str(p) for p in parts)])
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest, 16) & ((1 << TORCH_SEED_BITS) - 1)
