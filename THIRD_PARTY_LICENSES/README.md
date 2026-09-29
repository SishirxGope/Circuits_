# THIRD_PARTY_LICENSES/

Verbatim copies of upstream license texts for code we adapt or wrap. License texts
are byte-identical to the local fork files (never edited); provenance is recorded
here and in `docs/HUMAN_DECISIONS.md §3.3`. Adaptation headers in our code reference these
files (CLAUDE.md §7).

| File | Upstream repo | License | Verified | Copy date |
|---|---|---|---|---|
| `circuit-tracer-LICENSE.txt` | circuit-tracer (local fork: `circuit-tracer-0.5.2/`) | MIT-style ("Copyright (c) 2024 Michael Hanna and Mateusz Piotrowski") | Yes — text read from local fork file | 2026-08-07 |
| `sae-pruning-paper-LICENSE.txt` | sae-pruning-paper (local fork: `sae-pruning-paper-main/`) | MIT ("Copyright (c) 2025-2026 Héctor Borobia") | Yes — text read from local fork file | 2026-08-07 |
| `automatic-circuit-discovery-LICENSE.txt` | ACDC, `github.com/ArthurConmy/Automatic-Circuit-Discovery` @ `bc99ace817974b5584b7ee203d596a8e2bbcd399` | MIT ("Copyright 2023 Arthur Conmy, Adrià Garriga-Alonso") | Yes — fetched at the pinned commit; git blob `a01cfb07` matches upstream | 2026-09-14 |
| `easy-transformer-LICENSE.txt` | Easy-Transformer, `github.com/redwoodresearch/Easy-Transformer` @ `ea15315dd24481e9e2ac5c3ef335d82907a1dc34` | MIT ("Copyright (c) 2022 neelnanda-io") | Yes — fetched at the pinned commit; git blob `a6be6f03` matches upstream | 2026-09-14 |
| `gptq-LICENSE.txt` | GPTQ, `github.com/IST-DASLab/gptq` @ `2d65066eeb06a5c9ff5184d8cebdf33662c67faf` | Apache-2.0 (upstream ships the unfilled template; no named holder) | Yes — fetched at the pinned commit; git blob `261eeb9e` matches upstream | 2026-09-29 |
| `llm-awq-LICENSE.txt` | llm-awq, `github.com/mit-han-lab/llm-awq` @ `d6e797a42b9ef7778de8ee2352116e0f48a78d61` | MIT ("Copyright (c) 2023 MIT HAN Lab") | Yes — fetched at the pinned commit; git blob `aca327a5` matches upstream | 2026-09-29 |

Notes / open items:

- ✅ RESOLVED 2026-09-12 (HUMAN_DECISIONS.md Q9) — upstream commit pins:
  - circuit-tracer: `github.com/decoderesearch/circuit-tracer` @ `8f1e2438df612464e229e44c4a00ff637bf9379b`
    (tag `v0.5.2`, 2026-07-18). The `safety-research` URL in the fork's README is stale
    and HTTP-301s here.
  - sae-pruning-paper: `github.com/hecboar/sae-pruning-paper` @ `261191804675e2d39d0a265320dbc0bc85afd30a`
    (2026-07-31). **Inferred** from the fork's contents, not read off a URL; derivation
    and the one outstanding check are in HUMAN_DECISIONS.md §3.3.
- ✅ RESOLVED 2026-08-09: the artifact is MIT ("Copyright (c) 2026 Supratik Bhowal").
  MIT-on-MIT is compatible **provided these notices ship**, which is what this directory
  is for; `tests/test_licence_compliance.py` now fails if it goes missing or leaves git.
- Any future third-party dependency (e.g., GPTQ/AWQ implementations, calibration
  corpora) must add its license here before use (AI_RULES.md §5).

- 2026-09-29: GPTQ and llm-awq added. `src/compression/gptq.py` and `awq.py` port their
  algorithms to TransformerLens in torch (no package dependency: `auto-gptq`/`autoawq`
  do not build on the Spark's aarch64). Apache-2.0 §4 requires the licence to travel with
  a derivative work and modified files to say they were changed; both files' headers do.
- 2026-09-14: ACDC and Easy-Transformer added. `src/tasks/` ports ACDC's IOI and greater-than generators (and vendors their word lists by script in `src/tasks/_acdc_vendored.py`); ACDC's IOI file states it is "a very slightly edited version of" Easy-Transformer's, so both notices ship.
