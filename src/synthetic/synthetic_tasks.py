# [AI-GEN] agent=OpenCode date=2026-08-07 task=Synthetic seeded task prompts (Q6 provisional)
# reviewed-by: PENDING
# scientific-status: PROVISIONAL_ENGINEERING_ONLY

"""Deterministic seeded synthetic task prompts for engineering dry-runs.

PROVISIONAL (docs/HUMAN_DECISIONS.md §3.2 Q6): synthetic prompts only,
no external dataset download. These prompts do NOT reproduce published
IOI/greater-than/docstring behavior (AI_RULES.md 2.2 — no claim of that kind).

The prompt families mirror the task families of the real project (configs/task/*.yaml)
so the synthetic layer exercises the same config surface:
- indirect-object-identification-synthetic (IOI-like: fill-in sentence)
- greater-than-synthetic (day-of-month style)
- docstring-completion-synthetic (docstring line completion)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..common.seeding import create_seed_generator, derive_child_seed

_NAMES = ("Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry")

_SENTENCE = (
    "{name1} and {name2} went to the park. {name2} gave a bottle to {name3}. "
    "{name3} said thank you and drank."
)
_DAY_MONTH = "The {day} of {month} is"
_DOCSTRING = (
    'def helper_{i}():\n    """Returns the result for {word}.\n'
    '    {word} is processed deterministically."""\n    return'
)


def _rng_for(task_cfg: Mapping[str, Any], seed: int):
    task_name = str(task_cfg.get("name", "task"))
    base_seed = int(task_cfg.get("seed", 0) or 0)
    return create_seed_generator(derive_child_seed(base_seed, seed, task_name))[0]


def synthetic_ioi_prompts(task_cfg: Mapping[str, Any], n: int, seed: int) -> list[str]:
    rng = _rng_for(task_cfg, seed)
    names = rng.choice(_NAMES, size=(n, 3), replace=True)
    return [_SENTENCE.format(name1=a, name2=b, name3=c) for a, b, c in names]


def synthetic_greater_than_prompts(task_cfg: Mapping[str, Any], n: int, seed: int) -> list[str]:
    rng = _rng_for(task_cfg, seed)
    days = rng.integers(1, 29, size=n)
    months = rng.choice(
        ("January", "February", "March", "April", "May", "June"), size=n, replace=True
    )
    return [_DAY_MONTH.format(day=int(d), month=m) for d, m in zip(days, months)]


def synthetic_docstring_prompts(task_cfg: Mapping[str, Any], n: int, seed: int) -> list[str]:
    rng = _rng_for(task_cfg, seed)
    words = rng.choice(("alpha", "beta", "gamma", "delta"), size=n, replace=True)
    return [_DOCSTRING.format(i=i, word=w) for i, w in enumerate(words)]


def load_synthetic_prompts(task_cfg: Mapping[str, Any], seed: int | None = None) -> list[str]:
    """Deterministic prompt list for a synthetic task config (no downloads).

    The extraction seed comes from the caller (ensemble cell seed); the task config
    seed anchors the prompt content. Same (task_cfg, seed) -> identical prompts.
    """
    family = str(task_cfg.get("family", ""))
    n = int(task_cfg.get("n_prompts", 0))
    if n <= 0:
        raise ValueError(f"synthetic task n_prompts must be > 0, got {n}")
    if not task_cfg.get("synthetic", False):
        raise ValueError(
            "load_synthetic_prompts is only for synthetic tasks "
            "(task.synthetic=True); real datasets remain OPEN (HUMAN_DECISIONS.md Q6)"
        )
    seed = int(task_cfg.get("seed", 0) or 0) if seed is None else int(seed)
    if family == "indirect-object-identification-synthetic":
        return synthetic_ioi_prompts(task_cfg, n, seed)
    if family == "greater-than-synthetic":
        return synthetic_greater_than_prompts(task_cfg, n, seed)
    if family == "docstring-completion-synthetic":
        return synthetic_docstring_prompts(task_cfg, n, seed)
    raise ValueError(f"unknown synthetic family {family!r}")


__all__ = [
    "load_synthetic_prompts",
    "synthetic_ioi_prompts",
    "synthetic_greater_than_prompts",
    "synthetic_docstring_prompts",
]
