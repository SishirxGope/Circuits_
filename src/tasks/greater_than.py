# [AI-GEN] agent=Claude date=2026-09-14 task=Real greater-than prompt set with "01" corruption (Stage A engineering)
# reviewed-by: PENDING
#
# Adapted from: https://github.com/ArthurConmy/Automatic-Circuit-Discovery @ bc99ace817974b5584b7ee203d596a8e2bbcd399, MIT
#   acdc/greaterthan/utils.py (GreaterThanConstants year filter, get_year_data,
#   greaterthan_metric, get_all_greaterthan_things). Licence in THIRD_PARTY_LICENSES/.

"""Greater-than prompts (Hanna et al. 2023, arXiv:2305.00586, as operationalised in ACDC).

``"The {noun} lasted from the year {CCYY} to {CC}"`` -> the model should put probability on
two-digit suffixes greater than ``YY``.

ACDC's implementation is tied to GPT-2's tokenizer in three places, each re-derived here
from the tokenizer of the model under test rather than copied:

1. **The year set.** ACDC keeps years whose " CCYY" tokenizes as exactly [" CC", "YY"],
   trimming the first and last success per century. The filter is ported exactly but run
   with the model's tokenizer: 618 years qualify under GPT-2, 456 under Pythia (442 after
   the per-century trim).
2. **The corruption.** ACDC writes token id 486 - GPT-2's "01" - at a hard-coded position
   7. Under Pythia "01" is token 520, and the position is only 7 when the noun is a single
   token. Here the "01" id comes from the tokenizer and the suffix position is located per
   prompt and verified.
3. **The noun pool.** Filtered to nouns that are one token, so every prompt has the same
   length and the suffix position is fixed. Under Pythia this keeps 112 of 120.

Metric: ACDC's ``greaterthan_metric`` as a utility, P(suffix > YY) - P(suffix <= YY), with
probabilities taken over the FULL vocabulary and summed over the 100 two-digit suffixes
(ACDC returns the negation as a loss). Resampling unit per prompt: its noun.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import torch

from ..common.seeding import derive_child_seed
from ._acdc_vendored import ACDC_COMMIT, NOUNS
from .base import TaskPrompts, batch_by_length, encode, require_prepend_bos, single_token_words

TASK_NAME = "greater_than"
CORRUPT_SUFFIX = "01"


def valid_years(tokenizer: Any) -> list[str]:
    """ACDC's year filter, run with ``tokenizer``: " CCYY" must be exactly [" CC", "YY"]."""
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
    years: list[str] = []
    for century in range(11, 18):
        ok = []
        for year in range(century * 100 + 2, century * 100 + 99):
            y = str(year)
            if enc(f" {y}") == [enc(f" {y[:2]}")[0], enc(y[2:])[0]]:
                ok.append(y)
        years.extend(ok[1:-1])
    return years


def suffix_token_ids(tokenizer: Any) -> list[int]:
    """Token id of each two-digit suffix "00".."99"; every one must be a single token."""
    ids = []
    for i in range(100):
        toks = tokenizer.encode(f"{i:02d}", add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"suffix {i:02d} is {len(toks)} tokens under this tokenizer; the metric needs 1")
        ids.append(toks[0])
    return ids


def _prob_diff_sum(suffix_ids: list[int]):
    suffix = torch.tensor(suffix_ids, dtype=torch.long)

    def make(payloads: list[int]):
        yy = torch.tensor(payloads, dtype=torch.long)

        def metric_sum(logits: torch.Tensor) -> torch.Tensor:
            probs = torch.softmax(logits[:, -1, :], dim=-1)[:, suffix.to(logits.device)]
            csum = torch.cumsum(probs, dim=-1)
            rows = torch.arange(probs.shape[0], device=logits.device)
            positive = csum[:, -1]                              # P(any two-digit suffix)
            negative = csum[rows, yy.to(logits.device)]         # P(suffix <= YY)
            return (positive - 2 * negative).sum()              # = P(> YY) - P(<= YY)

        return metric_sum

    return make


def build_greater_than_prompts(
    tokenizer: Any,
    task_cfg: Mapping[str, Any],
    seed: int,
    max_batch_size: int = 64,
) -> TaskPrompts:
    """Draw ``task_cfg['n_prompts']`` greater-than prompts and their "01" corruptions."""
    n_prompts = int(task_cfg["n_prompts"])
    prepend_bos = require_prepend_bos(dict(task_cfg))
    nouns = single_token_words(NOUNS, tokenizer)
    years = valid_years(tokenizer)
    suffix_ids = suffix_token_ids(tokenizer)
    corrupt_id = tokenizer.encode(CORRUPT_SUFFIX, add_special_tokens=False)
    if len(corrupt_id) != 1:
        raise ValueError(f"{CORRUPT_SUFFIX!r} is not a single token under this tokenizer")
    if not nouns or not years:
        raise ValueError(f"empty pool under this tokenizer: {len(nouns)} nouns, {len(years)} years")

    rng = random.Random(derive_child_seed(int(seed), "greater-than-prompts"))
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
    bos_len = 1 if prepend_bos else 0

    items = []
    for _ in range(n_prompts):
        noun, year = rng.choice(nouns), rng.choice(years)
        yy = int(year[2:])
        if yy <= 0:
            raise ValueError(f"year {year}: ACDC's metric special-cases YY == 0, which the filter excludes")
        text = f"The {noun} lasted from the year {year} to {year[:2]}"
        clean_ids = encode(tokenizer, text, prepend_bos)

        # Locate the year's "YY" token from the pieces, and verify the pieces tokenize the
        # same way as the whole prompt - BPE is not guaranteed to be compositional.
        head = enc(f"The {noun} lasted from the year")
        year_toks = enc(f" {year}")
        tail = enc(f" to {year[:2]}")
        if head + year_toks + tail != clean_ids[bos_len:] or len(year_toks) != 2:
            raise ValueError(f"prompt for {noun!r}/{year} does not tokenize compositionally")
        yy_pos = bos_len + len(head) + 1

        corrupt_ids = list(clean_ids)
        corrupt_ids[yy_pos] = corrupt_id[0]
        items.append((clean_ids, corrupt_ids, noun, yy))

    return TaskPrompts(
        task=TASK_NAME,
        seed=int(seed),
        batches=batch_by_length(items, _prob_diff_sum(suffix_ids), max_batch_size),
        metadata={
            "source": f"ACDC acdc/greaterthan @ {ACDC_COMMIT} (Hanna et al. 2023, arXiv:2305.00586)",
            "nouns_upstream": len(NOUNS),
            "nouns_single_token": len(nouns),
            "years_usable": len(years),
            "corrupt_suffix_token_id": corrupt_id[0],
            "prepend_bos": prepend_bos,
            "corruption": f"year YY token -> {CORRUPT_SUFFIX!r}",
            "metric": "P(suffix > YY) - P(suffix <= YY), full-vocab probabilities",
            "resampling_unit": "noun",
        },
    )


__all__ = ["TASK_NAME", "valid_years", "suffix_token_ids", "build_greater_than_prompts"]
