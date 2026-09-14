# [AI-GEN] agent=Claude date=2026-09-14 task=Real task prompt sets for the dense-node pipeline (Stage A engineering)
# reviewed-by: PENDING

"""Shared structure for real task prompt sets.

A task materialises, for ONE ensemble seed, a set of clean/corrupted prompt pairs and a
metric. Edge attribution patching needs a positionwise difference between clean and
corrupted activations, so every clean prompt and its corruption must have the same token
length; prompts are therefore grouped into same-length batches.

Two things about batching that are easy to get wrong, and are handled here once:

- ``metric_sum`` returns the metric SUMMED over a batch's prompts, not averaged. The
  pre-registered task quantity is the mean over all prompts in the set, so the caller
  divides by ``TaskPrompts.n_prompts`` once. Averaging per batch and then summing batches
  would weight a batch of 3 prompts the same as a batch of 60.
- Batches are an efficiency device, NOT the bootstrap resampling unit. Each prompt carries
  its own unit id (IOI template, greater-than noun) in ``units``, so any prompt-level
  resampling can be done correctly regardless of how prompts were batched.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class PromptBatch:
    """Same-length clean/corrupted token batches and their summed metric."""

    clean: torch.Tensor                                   # LongTensor [n, seq]
    corrupt: torch.Tensor                                 # LongTensor [n, seq]
    metric_sum: Callable[[torch.Tensor], torch.Tensor]    # logits [n, seq, vocab] -> 0-dim
    units: tuple[str, ...]                                # resampling-unit id per prompt

    def __post_init__(self) -> None:
        if self.clean.dim() != 2 or tuple(self.clean.shape) != tuple(self.corrupt.shape):
            raise ValueError(
                f"clean {tuple(self.clean.shape)} and corrupt {tuple(self.corrupt.shape)} must "
                "be equal-shape [n, seq]: attribution patching takes a positionwise difference"
            )
        if self.clean.shape[0] != len(self.units):
            raise ValueError(f"{self.clean.shape[0]} prompts but {len(self.units)} unit ids")

    @property
    def n(self) -> int:
        return int(self.clean.shape[0])


@dataclass(frozen=True)
class TaskPrompts:
    """Every prompt pair a task produced for one seed, batched by length."""

    task: str
    seed: int
    batches: tuple[PromptBatch, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_prompts(self) -> int:
        return sum(b.n for b in self.batches)


def batch_by_length(
    items: Sequence[tuple[list[int], list[int], str, Any]],
    make_metric: Callable[[list[Any]], Callable[[torch.Tensor], torch.Tensor]],
    max_batch_size: int,
) -> tuple[PromptBatch, ...]:
    """Group ``(clean_ids, corrupt_ids, unit, metric_payload)`` items into batches.

    Deterministic: lengths ascending, generation order preserved within a length, then
    chunked to ``max_batch_size`` to bound memory. ``make_metric`` receives the payloads of
    one chunk, in order, and returns that chunk's ``metric_sum``.
    """
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
    by_len: dict[int, list[tuple[list[int], list[int], str, Any]]] = {}
    for clean_ids, corrupt_ids, unit, payload in items:
        if len(clean_ids) != len(corrupt_ids):
            raise ValueError(
                f"clean and corrupted prompts differ in token length ({len(clean_ids)} vs "
                f"{len(corrupt_ids)}) for unit {unit!r}; the corruption changed tokenization"
            )
        by_len.setdefault(len(clean_ids), []).append((clean_ids, corrupt_ids, unit, payload))

    batches: list[PromptBatch] = []
    for length in sorted(by_len):
        group = by_len[length]
        for start in range(0, len(group), max_batch_size):
            chunk = group[start:start + max_batch_size]
            batches.append(PromptBatch(
                clean=torch.tensor([c for c, _, _, _ in chunk], dtype=torch.long),
                corrupt=torch.tensor([k for _, k, _, _ in chunk], dtype=torch.long),
                metric_sum=make_metric([p for _, _, _, p in chunk]),
                units=tuple(u for _, _, u, _ in chunk),
            ))
    return tuple(batches)


def single_token_words(words: Sequence[str], tokenizer: Any) -> list[str]:
    """Words that encode to exactly one token WITH a leading space, in upstream order.

    The leading space matters: mid-sentence words are tokenized as " word", and a word can
    be one token with the space and two without (or vice versa).
    """
    return [w for w in words if len(tokenizer.encode(" " + w, add_special_tokens=False)) == 1]


def encode(tokenizer: Any, text: str, prepend_bos: bool) -> list[int]:
    """Token ids for ``text``, with the model's BOS prepended iff ``prepend_bos``."""
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if prepend_bos:
        if tokenizer.bos_token_id is None:
            raise ValueError("prepend_bos=True but the tokenizer has no bos_token_id")
        ids = [int(tokenizer.bos_token_id)] + ids
    return ids


def require_prepend_bos(task_cfg: dict[str, Any]) -> bool:
    """``prepend_bos`` is a pre-registered protocol choice with no default."""
    if "prepend_bos" not in task_cfg or task_cfg["prepend_bos"] is None:
        raise ValueError(
            f"task {task_cfg.get('name')!r} has no prepend_bos. It is a pre-registered prompt "
            "protocol choice (docs/HUMAN_DECISIONS.md), so it is never defaulted here."
        )
    return bool(task_cfg["prepend_bos"])


__all__ = [
    "PromptBatch",
    "TaskPrompts",
    "batch_by_length",
    "single_token_words",
    "encode",
    "require_prepend_bos",
]
