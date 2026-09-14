# data/ — task datasets and calibration corpora

Schema for recording data provenance (AI_RULES.md §5; CLAUDE.md §6 licensing TODO).

Every dataset used in this project MUST have a row in the table below BEFORE Stage A:
- `name` — short identifier matching the config (configs/task/*.yaml, configs/calibration/*.yaml)
- `source` — canonical URL or generator script path (scripts live in `scripts/` or
  `data/`; large corpora are downloaded, never committed)
- `license` — exact license permitting research use (GPTQ/AWQ calibration corpora must
  permit research use; record the license here)
- `content_hash` — sha256 of the (download + preprocessing) result
- `contact` — human who verified the license entry

## Registered datasets

Pre-registered 2026-09-12 (Q6, Q7). `content_hash` is filled at first materialisation:
nothing has been downloaded or generated yet, and a hash invented before the bytes exist
would be worse than an empty cell.

| name | source | license | content_hash | contact |
|------|--------|---------|--------------|---------|
| `ioi` | ACDC `acdc/ioi/ioi_dataset.py` @ `bc99ace8`, a seeded edit of Easy-Transformer @ `ea15315d`; templates of Wang et al., ICLR 2023 (arXiv:2211.00593) | MIT (generator); prompts are synthetic template text | PENDING — not yet generated | PI |
| `greater_than` | ACDC `acdc/greaterthan/utils.py::get_year_data` (Conmy et al., NeurIPS 2023) | MIT (generator); synthetic template text | PENDING — not yet generated | PI |
| `docstring` | ACDC `acdc/docstring/prompts.py` @ `bc99ace8` (MIB has no docstring task — verified 2026-09-14) | MIT (generator); synthetic template code text | PENDING — generator not yet ported | PI |
| `fineweb-edu-calib` | `HuggingFaceFW/fineweb-edu`, 300,000 tokens, seed 7 | ODC-BY | PENDING — not yet downloaded | PI |
| `wikitext-2-raw` | WikiText-2 raw, **test** split (~289K tokens) | CC BY-SA 3.0 | PENDING — not yet downloaded | PI |

No human-subjects data anywhere in the pipeline; no PII (AI_RULES.md §5). All three task
prompt sets are generated from templates, so no corpus is redistributed.

## Two things about these rows that are easy to get wrong

**Calibration and perplexity are DIFFERENT corpora.** The earlier recommendation was
WikiText-2 for both. The reference work calibrates pruning on **FineWeb-Edu** and
evaluates perplexity on **WikiText-2 test**. Calibrating Wanda on WikiText-2 while ref
[1] calibrated on FineWeb-Edu means our Wanda-pruned model is not their Wanda-pruned
model, and the C5 cross-audit would correlate circuit damage from one intervention
against feature damage from another.

**The calibration seed is part of the data, not part of the run config.** The fork's
pruning is deterministic *given the same calibration token cache*, and that cache is
built with **seed 7** (`revision/src/saediag/reprune.py::calib_cache_path`). A different
seed gives a different cache, a different Wanda mask, and a different pruned model, with
nothing in any output to indicate it.

## Corrections made while registering these (2026-09-12)

**`greater_than` is a year-span task, not day-of-month.** `docs/HUMAN_DECISIONS.md` Q6
described it as `"The {day} of {month} is"` → next-token day, attributed to Conmy et al.
ACDC's actual construction, read from source, is:

```python
template = "The {noun} lasted from the year {year1} to "
```

The model must emit a two-digit year suffix greater than the given one, scored over
suffixes `yearend+1..99`. A circuit extracted on a day-of-month task is not comparable
to the published greater-than circuit, which is the only reason to use it as an anchor.

**`ioi` has 30 templates, not 400.** The config's `n_templates: 400` was a placeholder,
not a count from any IOI release. The reference `ioi_dataset.py` defines exactly 15
`BABA_TEMPLATES`, with `ABBA_TEMPLATES = BABA_TEMPLATES[:]` then reordered — 30 in total.
This matters directly: the prompt bootstrap resamples **templates**, so the template
count is the effective sample size for every prompt-level CI.

## Still open

- **`docstring` generator is not yet ported.** Its source is verified (2026-09-14: MIB has no
  docstring task, so it is ACDC's release) and its resampling unit is decided (prompt within
  style), but the generator has not been ported or checked under Pythia's tokenizer, so
  `pi_confirmed` stays `false`. It does not block the Pythia-160M pilot.
- **Calibration `context_length`.** The fork reads it from the cached array's shape
  rather than declaring it; its SAE activations use context 256, but that is the SAE
  cache, not necessarily the pruning calibration cache. Confirm when the cache is built.
- Downloading either corpus requires `mode.allow_external_dataset_download`, which is
  `false` in every current mode.

## Corrections made 2026-09-14

**The IOI source was recorded wrongly.** The `ioi` row said "`transformer_lens` IOI
generator". TransformerLens's `evals.IOIDataset` has **two** templates. The 30 Wang et al.
templates are in Easy-Transformer's `ioi_dataset.py`, and ACDC ships a seeded edit of that
file; `src/tasks/ioi.py` ports ACDC's version, and both licences are in
`THIRD_PARTY_LICENSES/`. The error originated in an option description written without
checking the TransformerLens source.

**The greater-than noun count was recorded wrongly.** Earlier text gave 26 nouns. That was a
count of *lines* in the upstream list, which puts several nouns per line. There are **120**.
Because the noun is the prompt-bootstrap resampling unit, the error understated the number of
resampling units about 4.6×. The lists are now extracted from the pinned upstream files by
script and length-asserted (`src/tasks/_acdc_vendored.py`), not transcribed.

**What the Pythia tokenizer changes** (all computed at runtime from the tokenizer of the model
under test and recorded in each prompt set's metadata, never hard-coded):

| | GPT-2 (what ACDC assumed) | Pythia-160M @ `50f5173d` |
|---|---|---|
| IOI names usable as single tokens | 99 | **88** |
| greater-than nouns usable as single tokens | — | **112** of 120 |
| years whose " CCYY" is exactly [" CC", "YY"] | 618 | **456** (442 after ACDC's per-century trim) |
| token id of the corruption suffix "01" | 486 | **520** |

**Prompt protocol decided 2026-09-14:** no BOS token (matching the ACDC/EAP setups),
float32, and docstring resampled by prompt within style. The docstring source is now verified
— MIB has no docstring task — so it is ACDC's `acdc/docstring/prompts.py`; its generator is not
yet ported, so its row stays unconfirmed.
