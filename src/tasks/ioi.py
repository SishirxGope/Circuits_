# [AI-GEN] agent=Claude date=2026-09-14 task=Real IOI prompt set with ABC corruption (Stage A engineering)
# reviewed-by: PENDING
#
# Adapted from: https://github.com/ArthurConmy/Automatic-Circuit-Discovery @ bc99ace817974b5584b7ee203d596a8e2bbcd399, MIT
#   acdc/ioi/ioi_dataset.py (gen_prompt_uniform, gen_flipped_prompts, the ABBA derivation) and
#   acdc/ioi/utils.py (the ABC corrupted set: IO->RAND, S->RAND, S1->RAND). That file is a
#   seeded edit of https://github.com/redwoodresearch/Easy-Transformer @
#   ea15315dd24481e9e2ac5c3ef335d82907a1dc34, MIT. Licences in THIRD_PARTY_LICENSES/.

"""Indirect-object identification prompts (Wang et al., ICLR 2023) with ABC corruption.

What is ported faithfully, and what deliberately differs from ACDC's setup:

- **Templates**: all 30 (15 BABA + 15 ABBA). ACDC's own IOI benchmark uses ONE template
  (``prompt_type="ABBA", nb_templates=1``), which the EAP paper inherits. Q6 pre-registered
  the template as the prompt-bootstrap unit, and one template would give n = 1, so the
  full set is required rather than chosen.
- **ABBA derivation**: an exact port of ACDC's character loop (swap the first-clause names).
- **Names**: the upstream 99, filtered to those that are ONE token for the model under
  test. The logit-difference metric compares single next-token logits, so a multi-token
  name has no well-defined logit. Under Pythia this leaves 88.
- **Corruption (ABC)**: an exact port of the three chained flips ACDC applies, including a
  property worth knowing - the S->RAND and S1->RAND draws have NO distinctness check
  upstream, so rarely a corrupted name coincides with another name in the prompt. Kept as
  published rather than silently fixed.
- **Seeds**: ACDC fixes the three corruption seeds at 1, 2, 3 regardless of the dataset
  seed. Here the ensemble seed determines BOTH which prompts are drawn AND their
  corruptions, because the pre-registered seed axis is resampling of the probe batch
  ("resampling variance", arXiv:2606.16920). Every draw derives from ``seed`` via
  ``derive_child_seed`` (AI_RULES.md 1.1).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from ..common.seeding import derive_child_seed
from ._acdc_vendored import ACDC_COMMIT, BABA_TEMPLATES, NAMES, OBJECTS, PLACES
from .base import TaskPrompts, batch_by_length, encode, require_prepend_bos, single_token_words

TASK_NAME = "ioi"


def abba_from_baba(template: str) -> str:
    """Exact port of ACDC's ABBA derivation: swap the names of the FIRST clause only.

    ``"Then, [B] and [A] went ... [B] gave ... to [A]"`` -> ``"Then, [A] and [B] went ...
    [B] gave ... to [A]"``. Character-level, mutating as it scans, exactly as upstream.
    """
    t = template
    first_clause = True
    for j in range(1, len(t) - 1):
        if t[j - 1:j + 2] == "[B]" and first_clause:
            t = t[:j] + "A" + t[j + 1:]
        elif t[j - 1:j + 2] == "[A]" and first_clause:
            first_clause = False
            t = t[:j] + "B" + t[j + 1:]
    return t


def all_templates() -> list[tuple[str, str]]:
    """The 30 templates with stable ids: ``BABA-00``..``BABA-14``, ``ABBA-00``..``ABBA-14``."""
    return (
        [(f"BABA-{i:02d}", t) for i, t in enumerate(BABA_TEMPLATES)]
        + [(f"ABBA-{i:02d}", abba_from_baba(t)) for i, t in enumerate(BABA_TEMPLATES)]
    )


def _numpy_stream(seed: int, label: str) -> np.random.RandomState:
    return np.random.RandomState(derive_child_seed(int(seed), label) % (2 ** 32))


def abc_corrupt(
    words: list[str],
    io: str,
    s: str,
    names: list[str],
    r_io: np.random.RandomState,
    r_s: np.random.RandomState,
    r_s1: np.random.RandomState,
) -> list[str]:
    """ACDC's ABC corruption, applied as its three chained flips (word-level, on " " splits).

    1. IO -> RAND: a name distinct from IO and S replaces the first TWO occurrences of IO
       (the first-clause IO and the answer).
    2. S -> RAND: a name (no distinctness check) replaces the first and the last occurrence
       of S.
    3. S1 -> RAND: a name (no distinctness check) replaces the first occurrence of the
       CURRENT S, so S1 and S2 end up as different random names.
    """
    t = list(words)

    new_io = names[r_io.randint(len(names))]
    while new_io == io or new_io == s:
        new_io = names[r_io.randint(len(names))]
    t[t.index(io)] = new_io
    t[t.index(io)] = new_io

    new_s = names[r_s.randint(len(names))]
    t[t.index(s)] = new_s
    t[len(t) - t[::-1].index(s) - 1] = new_s

    new_s1 = names[r_s1.randint(len(names))]
    t[t.index(new_s)] = new_s1
    return t


def _logit_diff_sum(payloads: list[tuple[int, int]]):
    io_ids = torch.tensor([p[0] for p in payloads], dtype=torch.long)
    s_ids = torch.tensor([p[1] for p in payloads], dtype=torch.long)

    def metric_sum(logits: torch.Tensor) -> torch.Tensor:
        last = logits[:, -1, :]
        rows = torch.arange(last.shape[0], device=last.device)
        return (last[rows, io_ids.to(last.device)] - last[rows, s_ids.to(last.device)]).sum()

    return metric_sum


def build_ioi_prompts(
    tokenizer: Any,
    task_cfg: Mapping[str, Any],
    seed: int,
    max_batch_size: int = 64,
) -> TaskPrompts:
    """Draw ``task_cfg['n_prompts']`` IOI prompts and their ABC corruptions for ``seed``.

    Metric: logit(IO) - logit(S) at the final position, summed per batch (the ACDC
    ``logit_diff`` task metric). Resampling unit per prompt: its template id.
    """
    n_prompts = int(task_cfg["n_prompts"])
    prepend_bos = require_prepend_bos(dict(task_cfg))
    names = single_token_words(NAMES, tokenizer)
    places = single_token_words(PLACES, tokenizer)
    objects = single_token_words(OBJECTS, tokenizer)
    if len(names) < 3:
        raise ValueError(f"only {len(names)} single-token names for this tokenizer; need >= 3")
    templates = all_templates()

    rng = random.Random(derive_child_seed(int(seed), "ioi-prompts"))
    r_io = _numpy_stream(seed, "ioi-corrupt-io")
    r_s = _numpy_stream(seed, "ioi-corrupt-s")
    r_s1 = _numpy_stream(seed, "ioi-corrupt-s1")

    items = []
    for _ in range(n_prompts):
        template_id, template = rng.choice(templates)
        a = b = c = ""
        while len({a, b, c}) < 3:
            a, b, c = rng.choice(names), rng.choice(names), rng.choice(names)
        filled = template.replace("[PLACE]", rng.choice(places)).replace("[OBJECT]", rng.choice(objects))
        words = filled.replace("[A]", a).replace("[B]", b).split(" ")

        # The corruption relies on exact whole-word matches; verify rather than assume.
        if words.count(a) != 2 or words.count(b) != 2 or words[-1] != a:
            raise ValueError(
                f"template {template_id} did not yield IO twice (ending the prompt) and S twice "
                f"as whole words: {' '.join(words)!r}"
            )
        corrupted = abc_corrupt(words, a, b, names, r_io, r_s, r_s1)

        clean_ids = encode(tokenizer, " ".join(words[:-1]), prepend_bos)
        corrupt_ids = encode(tokenizer, " ".join(corrupted[:-1]), prepend_bos)
        io_id = tokenizer.encode(" " + a, add_special_tokens=False)[0]
        s_id = tokenizer.encode(" " + b, add_special_tokens=False)[0]
        items.append((clean_ids, corrupt_ids, template_id, (io_id, s_id)))

    return TaskPrompts(
        task=TASK_NAME,
        seed=int(seed),
        batches=batch_by_length(items, _logit_diff_sum, max_batch_size),
        metadata={
            "source": f"ACDC acdc/ioi @ {ACDC_COMMIT} (Wang et al. templates via Easy-Transformer)",
            "n_templates": len(templates),
            "names_upstream": len(NAMES),
            "names_single_token": len(names),
            "places_single_token": len(places),
            "objects_single_token": len(objects),
            "prepend_bos": prepend_bos,
            "corruption": "ABC: IO->RAND, S->RAND, S1->RAND (ACDC chained flips)",
            "metric": "logit_diff(IO - S) at final position",
            "resampling_unit": "template",
        },
    )


__all__ = ["TASK_NAME", "abba_from_baba", "all_templates", "abc_corrupt", "build_ioi_prompts"]
