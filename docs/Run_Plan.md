# Run_Plan.md — what to run, in what order, and what I think you should decide

> Companion to `docs/HUMAN_DECISIONS.md`. That file asks the questions. **This file answers
> the ones that can be answered by reading the literature, and says plainly which ones are
> still yours.**
>
> Written 2026-09-11. Every paper claim below was fetched and verified this session, not
> recalled (AI_RULES.md §2.2). arXiv IDs, commit hashes and table values are quoted from the
> live sources named inline.

---

## Part 0 — Read this first: the research changed five answers

I read CIRCUS, the comparison-levels paper, the pruning feature audit, the quantization audit,
and both upstream repositories. Five things came back that **contradict what is currently
written down in this repo.** These are the highest-value findings in this document.

### 0.1 CIRCUS *does* fix the band cutoffs — and they are not the ones in our config

`HUMAN_DECISIONS.md` Q1 offers "Option D: whatever CIRCUS uses — only if you open the paper and
verify it." I opened it. CIRCUS §3.2 ("Circuit taxonomy") fixes them explicitly:

| Band | CIRCUS definition (arXiv:2603.00523 §3.2) | Our provisional value |
|---|---|---|
| Core | **s(e) = 1** (present in *every* view), forming `C₁` | s(e) ≥ 0.90 |
| Contingent | **0.5 ≤ s(e) < 1** | 0.10 < s(e) < 0.90 |
| Noise | **s(e) < 0.5** — "flagged for rejection" | s(e) ≤ 0.10 |

Our provisional `core ≥ 0.90, noise ≤ 0.10` is **not** CIRCUS's taxonomy. Adopting theirs is
free, removes a reviewer objection, and is exactly what CLAUDE.md §5 asked you to check.

**There is a code consequence you must not skip.** `src/science/decompose.py` uses *inclusive*
boundaries: `s <= noise_threshold → noise`. CIRCUS uses a *strict* one: `s < 0.5 → noise`, with
s = 0.5 being **contingent**. Setting `noise_threshold: 0.5` would therefore mislabel every
edge sitting at exactly half the views. With B = 16 the achievable s values are multiples of
1/16, so the exact encoding of CIRCUS is:

```yaml
core_threshold: 1.0      # s >= 1.0  <=>  s == 1  (CIRCUS C_1, strict consensus)
noise_threshold: 0.4375  # = 7/16, the largest achievable s strictly below 0.5 at B=16
```

`noise_threshold` is **B-dependent** under this encoding: it is `(ceil(B/2) − 1)/B`. Change B
and you must recompute it, which is a footgun. See Q1 for how I would handle that.

### 0.2 The feature-level numbers our cross-audit (C5) compares against were **corrected by their own authors**

This is the big one. `CLAUDE.md` §2 and `PRD.md` §2 plan to correlate our circuit-damage
ranking against the published feature-damage ranking from ref [1] (arXiv:2603.25325). The
arXiv v1 abstract headlines **"Spearman ρ = −1.0 in 11 of 17 conditions."**

The authors' own frozen statistical record, in the fork already on your disk at
`../sae-pruning-paper-main/results/E6/stat_tests.csv`, says:

```
Fragility Spearman ρ(firing,survival) range, across conditions @τ0.7, n=18, -0.540..0.062
```

**The range is −0.540 to +0.062, not −1.0.** The revision commit message states why: *"The
README led with Q1/Q5 survival ratios, which are unstable when S(Q5) is near zero. The primary
fragility effect is now S(Q1)−S(Q5) with cluster-bootstrap confidence intervals."*

Three more numbers in the arXiv v1 tables are superseded by that revision:

| Quantity | arXiv v1 | Corrected (frozen tables) |
|---|---|---|
| Gemma-2-2B dense WikiText-2 PPL | 410 | **8.21** |
| Gemma-3-1B dense PPL | 64.2 | **9.99** |
| Llama-3.2-1B dense PPL | 17.7 | **9.28** |
| Gemma-3 transferability FVU | 0.011 / L0 11.35 | **0.0064 / L0 20.68** (L13-on-L13) |

Note `HUMAN_DECISIONS.md` says Gemma-2's corrected perplexity is "~11". That is the
**intermediate** rung S2 of their forensic ladder (11.97, at window 512); the final
protocol value at window 1024 is **8.21**. Use 8.21.

**Consequence: C5 must correlate against the frozen CSV tables, never against the arXiv PDF.**
If you quote ρ = −1.0 you will be quoting a number the authors walked back, and a reviewer who
opens the repo will find it. Corrected C5 design is Part 4.

### 0.3 Q9's "which URL is canonical" question is answered

`github.com/safety-research/circuit-tracer` **HTTP-redirects to**
`github.com/decoderesearch/circuit-tracer` (verified by fetching the former and receiving the
latter). So the fork README's *BibTeX* was right and its *install links* are stale. Hashes:

| Repo | Canonical URL | Commit | Licence |
|---|---|---|---|
| circuit-tracer | `github.com/decoderesearch/circuit-tracer` | `8f1e2438df612464e229e44c4a00ff637bf9379b` (tag `v0.5.2`, 2026-07-18) | MIT |
| sae-pruning-paper | `github.com/hecboar/sae-pruning-paper` | `261191804675e2d39d0a265320dbc0bc85afd30a` (2026-07-31) | MIT |

The sae-pruning pin is **inferred from the local fork's contents, not read off a URL**, and the
inference is tight: your copy contains `revision/`, `results/E6/` and `docs/PROTOCOL.md` (all
added in `261191…`), but lacks the metric-ordering script added in the next commit `c8cce94…`
and still contains the word "rebuttal" in `revision/scripts/e6_stat_freeze.py`, which the
commit after that (`bd85878…`) removed. `c8cce94…`'s parent *is* `261191…`, so there is no
commit in between. **Verify before committing** with the command in Step 1 — if the tree diff
disagrees, trust the diff, not this paragraph.

### 0.4 Q4's threshold ranges are pinned by the upstream defaults

circuit-tracer's CLI defaults are `--node_threshold 0.8` and `--edge_threshold 0.98`. CIRCUS's
published non-nested examples are `(0.6, 0.99)` and `(0.9, 0.95)`, and its nested
*counter*-example is `(0.6,0.95) ⊂ (0.8,0.98) ⊂ (0.9,0.99)`. So the grid axes are not a free
choice: node ∈ [0.6, 0.9], edge ∈ [0.95, 0.99], with the upstream default at the centre.

### 0.5 There is a published prior that predicts your result, at both levels

arXiv:2607.18921 compared dense vs **75% weight-sparse** checkpoints — i.e. pruning, your
Stage C — and measured:

| Level | Jaccard@10 | Random baseline | Above chance? |
|---|---|---|---|
| Exact edges (success split) | 0.163 | 0.126 | barely, p = 0.040 |
| Exact edges (near-miss split) | 0.142 | 0.117 | **no, p = 0.106** |
| Routing-head sets (success) | 0.666 | 0.199 | yes, p < 0.001 |
| Routing-head sets (near-miss) | 0.554 | 0.185 | yes, p < 0.001 |

And their dense **cross-seed** exact-edge Jaccard is only **0.363** — same model, same method,
different seed, and two-thirds of the edge list changes. That is your null model's thesis,
already in print, from someone else. Cite it in the introduction: it is the strongest available
evidence that a naked before/after circuit comparison is uninterpretable.

It also gives you a **pre-registerable directional prediction**: exact-edge CSI near or above
the noise floor, coarse-level CSI well below it. Write that down *before* Stage C — a correct
pre-registered prediction is worth far more than the same number found post hoc.

---

## Part 1 — The runbook, in order

⚠️ = blocks everything after it.

### Step 0 — Confirm the repo is green on this machine **[5 min]**

```powershell
cd "D:\Users\SUPRATIK\AAAI-UC\AAAI - Standalone Project\Circuits_Under_Compression"
pip install -e .
python -m pytest                      # expect: 462 passed, 2 skipped
python -m experiments.dry_run_stage_a 0
```

The first skip is `tests/test_regression_ioi_gpt2_small.py`, the Phase-1 exit gate, skipped
by design until you supply a reference circuit (Step 6). The second is `tests/test_seeding.py:38`,
which needs `torch` and is correct to skip until the `models` extra is installed. Any *failure*
stops everything.

On a machine whose default `python` predates 3.10 (`requires-python >=3.10`), create the venv
first and read every `python` below as `.venv/Scripts/python.exe`:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install -e . pytest
```

### Step 1 — Record the verified upstream pins (Q9) **[15 min]**

Verify my claims rather than trusting them, then write them down:

```powershell
# 1. Confirm the canonical URL redirect yourself
curl -sIL https://github.com/safety-research/circuit-tracer | Select-String -Pattern "^location:"

# 2. Confirm the circuit-tracer v0.5.2 hash
curl -s https://api.github.com/repos/decoderesearch/circuit-tracer/git/ref/tags/v0.5.2

# 3. Confirm the sae-pruning-paper pin by comparing trees, not by trusting prose
git clone https://github.com/hecboar/sae-pruning-paper "$env:TEMP\sp"
cd "$env:TEMP\sp"; git checkout 261191804675e2d39d0a265320dbc0bc85afd30a
git --no-index diff --stat HEAD "D:\Users\SUPRATIK\AAAI-UC\AAAI - Standalone Project\sae-pruning-paper-main"
```

Then put both hashes in `docs/HUMAN_DECISIONS.md` §3.3 and in the
`# Adapted from: <repo> @ <commit>, <license>` headers of
`src/extraction/attribution_graph.py` and `src/compression/magnitude_prune.py` (CLAUDE.md §7).

### Step 2 — Pin the model revisions (Q5) ⚠️ **[30 min]**

Blocks every real run. Accept the gated licences **now**, not on run day.

```powershell
huggingface-cli login
# for each repo: https://huggingface.co/<repo> -> "Files and versions" -> copy the full 40-char hash
```

Write each into `hf_revision:` in the matching `configs/model/*.yaml`, and update the test that
asserts they are all `null` **in the same commit** so the two cannot drift.

```powershell
python -m pytest -k "revision or model_config"
```

### Step 3 — Make the decisions (Q1, Q2, Q4, Q10, Q11) ⚠️ **[90 min, not 2–3 h]**

The literature has collapsed most of this. Order: **Q2 → Q1 → Q4 → Q10 → Q11**. Details in
Part 2. After editing, the guard tests must flip from "refuses" to "passes":

```powershell
python -m pytest -k "decompose or distance or two_level or node_ids or threshold_grid"
python -m experiments.dry_run_stage_a 0      # must still run end to end
```

### Step 4 — Source the data (Q6, Q7) ⚠️ **[3–4 h]** — see Part 3

### Step 5 — First real Stage A on Pythia-160M, then fix B/S/R (Q3) **[1 day]**

```powershell
python -m experiments.run_stage_a mode=scientific_run model=pythia160m task=ioi
```

**Pythia goes through the dense-node pipeline** — no transcoder set exists for it, so pipeline
A cannot run there at all.

### Step 6 — The Phase-1 exit gate ⚠️ **[1–2 days]**

```powershell
python -m pytest tests/test_regression_ioi_gpt2_small.py -v
```

Supply the reference edge list **and the Jaccard tolerance, chosen before you see your own
number**. Given that 2607.18921 measures cross-seed exact-edge Jaccard at 0.363 on a *fixed*
model, do not set this tolerance high at the exact-edge level — "Jaccard ≥ 0.8 vs the published
GPT-2 IOI circuit" is likely unachievable by *any* correct implementation. Set the gate at the
coarse level, or at a defensibly low exact-edge value, and say which.

### Step 7 — Stage B everywhere, then FREEZE ⚠️ **THE POINT OF NO RETURN**

```powershell
python -m experiments.run_stage_b mode=scientific_run model=<m> task=<t> nulls=<n>
python -m experiments.freeze_stage_b --config <resolved.json> --dry-run   # ALWAYS first
python -m experiments.freeze_stage_b --config <resolved.json>             # irreversible
git tag -a null-freeze -m "Pre-registration: null distributions frozen"
```

### Step 8 — Stage C / D, then the paper

```powershell
# before EVERY Stage C batch, cheapest possible safety check:
python -c "from src.common.freeze import verify_frozen_store; print(verify_frozen_store())"

python -m experiments.run_stage_c mode=scientific_run model=<m> task=<t> compression=<c>
python -m experiments.run_stage_d mode=scientific_run model=<m> task=<t>
python -m analysis.threshold_sweep
python -m analysis.cross_audit
```

---

## Part 2 — The decisions: what I found, what I think, what to run

### Q2 — The distance function D ⭐ decide first

**What I think: keep the recommendation. Primary L1, ablation Jensen–Shannon.** I went looking
for a reason to overturn it and did not find one. Two findings *strengthen* it:

- Heavy pruning genuinely does shrink total inclusion mass — the pruning audit shows fixed-dict
  Jaccard collapsing from 0.591 (Wanda s=0.3) to 0.048 (magnitude s=0.5) on matched models. A
  metric blind to uniform mass loss, which is exactly JS's weakness, would report "nothing
  happened" in the cells where the most happens. L1 must be primary.
- JS is still a *real* second lens rather than a box-tick, because it isolates shape from mass.
  That is what an ablation is for.

**One thing I would add, and it is cheap.** L1 scales with circuit size, so cells with
different edge counts are not directly comparable — and your cells *will* differ in edge count,
because pruning changes how many edges survive. Report **normalised L1** (L1 ÷ size of the edge
union) as a third column throughout. One line of code, not a new decision (it is a monotone
rescaling of the primary), and it pre-empts the obvious reviewer question. Do *not* make it
primary: the unnormalised number is the one that means "the circuit changed by this much".

> **Fill in:** `primary_D = L1` · `ablation_D = Jensen-Shannon` · `also_report = L1/|union|`
> justification: "L1 is total inclusion-frequency change with no hidden assumption; JS tests
> whether conclusions depend on shape rather than mass; both implemented and tested."

**Run:**
```powershell
# edit configs/distance/provisional_l1_js.yaml: status: FINAL, pre_registered_for_stage_c: true
python -m pytest tests/ -k distance -v      # 14 tests
```

---

### Q1 — Band cutoffs: **resolved by CIRCUS, use theirs**

See §0.1. CIRCUS §3.2: core `s(e) = 1`, contingent `0.5 ≤ s(e) < 1`, noise `s(e) < 0.5`.

**What I think: adopt CIRCUS verbatim and cite it.** Strictly better than our provisional
0.9/0.1 — identical work, for a citation instead of an argument.

Know what you are buying. At B = 3, CIRCUS reports **2.5% of union edges at s = 1 and 73% below
0.5** — a core defined as strict consensus is a *small* set. It is also a high-quality one:
edges surviving all 25 configs carry **~70× the mean influence** of single-config edges. A
small, heavily-loaded core is the intended behaviour, not a bug. But it means your core band
may hold few enough edges that per-cell CIs on "core size" are wide. Budget for that in Q3 and
do not be alarmed when the core is ~2–3% of the union.

**The one place I would deviate from a literal reading.** Encoding "noise is strictly below 0.5"
as `noise_threshold: 0.4375` silently hard-codes B = 16 into a config whose name never mentions
B. That is a latent bug worth 20 minutes: add a `noise_strict: true` flag to `decompose()` so
the cutoff can be written as the honest `0.5` with a strict comparison, and the config stops
being B-dependent. If you would rather not touch a 🔒 novelty-zone module before the freeze — a
defensible call — then write `0.4375` **with `# = 7/16; valid only for B=16` on the same line**,
plus a test that fails if B ≠ 16.

> **Fill in:** `core_threshold = 1.0` · `noise_threshold = 0.4375 (= 7/16, strict-<0.5 at B=16)`
> `source = CIRCUS arXiv:2603.00523 §3.2 "Circuit taxonomy"`

**Run:**
```powershell
# edit configs/ensemble/decompose/final.yaml with the two values above
# then switch the root config default: ensemble/decompose: provisional -> final
python -m pytest -k "decompose" -v
python -m experiments.dry_run_stage_a 0
```

---

### Q4 — How the B configurations are generated: **CIRCUS's construction exists, use it**

CIRCUS ships no config-generation *function* I could locate (no code release surfaced; the
paper is on OpenReview as `M3wusJ6Mqo`), **but it specifies the construction in prose precisely
enough to reimplement, and that is enough to cite.** §3.2, "Non-nesting principle":

> *"We therefore require non-nested configurations, achieved by crossing node and edge
> thresholds in opposite directions so that each view prunes a different part of the graph."*

Plus worked examples: `(0.6, 0.99)` keeps few nodes but many edges, `(0.9, 0.95)` the opposite.
And the counter-example to avoid: `(0.6,0.95) ⊂ (0.8,0.98) ⊂ (0.9,0.99)` is nested and makes
consensus collapse onto the loosest view (Match = 50/50 prompts).

**What I think: a crossed 4×4 grid, node ∈ {0.6, 0.7, 0.8, 0.9} × edge ∈ {0.99, 0.98, 0.97,
0.95}, anti-correlated, giving B = 16.** Rationale:

- It is CIRCUS's construction, describable in one sentence, anchored on the *upstream defaults*
  (0.8 / 0.98) rather than on numbers we invented.
- It is a grid, so trivially pre-registerable and reproducible with **no seed dependence** —
  strictly better than our current provisional placeholder (a *seeded random* non-nested sampler
  in `src/science/threshold_grid.py`), which adds a second seed axis on top of the seed axis you
  already have, for no scientific gain.
- B = 16 sits between CIRCUS's B = 9 (main evaluation) and B = 25 (stability distributions), and
  CIRCUS measured the whole B-prune stage at **0.19 s / 5.5% overhead** on top of one
  attribution run. B is effectively free.

**CIRCUS also hands you two diagnostics that turn Q4 from an assertion into a check.** Run both
on Stage A output and report them — they are the evidence your grid works:

| Diagnostic | CIRCUS's value | What it tells you |
|---|---|---|
| Mean pairwise Jaccard across the B views | 0.33 (Gemma), 0.35 (Llama) | genuine view diversity; near 1.0 means your configs are near-duplicates |
| "Match" = how often consensus equals a single view | 8/50 Gemma, 0/20 Llama at B=9 | Match ≈ B/B is the **nesting artifact**; your grid is broken |

If your grid returns Match ≈ 100% of prompts, **stop**: the grid is nested and every s(e) is
inflated toward the loosest threshold. That is the exact failure CIRCUS warns about, and it is
cheap to detect before it contaminates a frozen null.

> **Fill in:** `grid_design = crossed anti-correlated grid, CIRCUS arXiv:2603.00523 §3.2` ·
> `node_threshold = {0.6, 0.7, 0.8, 0.9}` · `edge_threshold = {0.99, 0.98, 0.97, 0.95}` (crossed) · `B = 16`

**Run:**
```powershell
# put threshold_grid under configs/ensemble/default.yaml
python -m pytest -k "threshold_grid" -v     # 6 tests
# after the first real Stage A, check both diagnostics BEFORE freezing:
python -m analysis.threshold_sweep
```

---

### Q10 — What the coarse level is: **`layer`, but with a falsification test attached**

The fork reading was right: circuit-tracer emits four node kinds and none is an attention head,
so a literal routing-head projection returns ∅. Confirmed in source (`node_ids.py:60-64`).

**What I think: `layer`, as recommended — but the recommendation as written is incomplete in a
way that matters, and I would not pre-register it without the fix.**

The problem I found: 2607.18921 does not report two levels, it reports **three** — structural,
semantic, routing. And its own result is that *semantic* coarsening is **"numerically
indistinguishable from structural"**: Jaccard@10 0.163 vs 0.163, identical to three decimals, in
every split and every supplementary row. Only the *routing-head* level separated (0.666 vs
0.163).

So that paper's actual lesson is sharper than "pick a coarser level": **a coarsening that does
not cut along a functional boundary buys you literally nothing.** Their structural bundle ID
already groups by layer among other things — which puts "group by layer" on the side of the
ledger that came out indistinguishable, not the side that separated.

`layer` is still the right choice: the only option always defined, cheap, and honest, and
`dense_node_heads` entangles C2 with C8 (basis drift) for a Phase-4 luxury. **But pre-register a
falsification condition alongside it:**

> If layer-level Jaccard lies within the bootstrap CI of exact-edge Jaccard, the second level is
> declared **vacuous for pipeline A** and reported as a negative methodological result, rather
> than presented as a second level that happens to agree.

That costs nothing if the levels separate, and saves you from "your two levels are the same
measurement twice" if they do not. It is also a publishable negative finding for an AAAI UC
paper: *the two-level reporting practice 2607.18921 recommends cannot be instantiated in the
dominant attribution-graph tool, because its node basis has no routing structure to coarsen
onto.* Real contribution, and honest.

**Paper wording:** "We report every claim at two levels of description: exact
component-to-component edge overlap, and layer-routing overlap. circuit-tracer's node basis is
cross-layer-transcoder features, so head-level routing (the coarse level used by
arXiv:2607.18921) is not available in pipeline A; we substitute layer routing and test
explicitly whether it separates from the exact level."

**One code gap to close.** `two_level.py::project_to_routing_heads` has its own regex
`^L\d+\.H\d+$`, independent of `node_ids.project_level2`. Choosing `layer` does **not**
automatically rewire `routing_head_overlap` — two_level.py will still match heads and still
return ∅. Connect them, or C2 silently reports zeros forever.

> **Fill in:** `level2_scheme = layer` + the falsification condition above, pre-registered.

**Run:**
```powershell
python -m pytest -k "two_level or node_ids" -v
# verify the wiring actually changed — this must NOT return an empty set:
python -c "from src.extraction.node_ids import project_level2; print(project_level2(['L7.F123','L7.F99','L9.ERR'],'layer'))"
```

---

### Q11 — Position aggregation: **aggregate. Agreed, and here is the number that decides it**

**What I think: aggregate, as recommended.** No reason to deviate, and one finding makes it
nearly forced.

The usual argument — "it matches the schema and how people talk about circuits" — is true but
weak; schemas can change. The stronger argument is the **chance floor**.
`chance_floor.candidate_edge_count` requires N explicitly, and going position-specific
multiplies the edge universe by `n_pos`. For IOI prompts `n_pos` is ~15–20. A ~15–20× larger
universe pushes the random-overlap baseline down, making *every* overlap look impressively above
chance for purely combinatorial reasons. 2607.18921's candidate sets are **37 (structural) and
31 (routing)** — small universes, honest baselines. Inflating the universe is the single easiest
way to accidentally manufacture a significant result, and your whole paper is an argument
against accidentally manufacturing significant results.

Aggregating also keeps s(e) interpretable: with positions collapsed, "this edge appeared in 12
of 16 views" is a statement about the computation. Position-specific, it is partly a statement
about prompt length.

State in the paper the sentence `node_ids.position_policy_is_pi_owned(False)` returns.

> **Fill in:** `position_policy = aggregate over positions`

**Run:**
```powershell
python -m pytest -k "node_ids or inclusion_freq or chance_floor" -v
python -c "from src.extraction.node_ids import position_policy_is_pi_owned; print(position_policy_is_pi_owned(False))"
```

---

### Q3 — B, S, R: **B = 16 now, S and R after one measurement**

B is settled by Q4 at **16**, and CIRCUS's 5.5% / 0.19 s overhead means B costs essentially
nothing — the B configs re-prune one already-computed attribution graph. **So B is not a budget
lever and you should stop treating it as one.** `HUMAN_DECISIONS.md` says "halve R first
(20→10), then B (16→8)"; halving B saves ~0.1 s per cell. Drop that step.

The real cost is `S` attribution passes per cell, and Stage B multiplies by `R`. Budget equation:
`cells × S × (1 + R) × t_attribution`.

**What I think:** start S = 5, R = 20 on Pythia per ARCHITECTURE.md §5, measure, then:

- `t ≤ 2 min` → keep S = 5, R = 20 on the primaries too.
- `2 min < t ≤ 6 min` → **R = 10, S = 5**. Cut R first: R buys resolution on the *median* of
  D_null, and a median is cheap to estimate. S buys the seed axis, which is a *named part of your
  contribution* (proposal §2.1c) and cannot be cut to 1 without deleting a claim.
- `t > 6 min` → R = 10, S = 3, then drop secondary compression levels per CLAUDE.md §6's fixed
  scope-cut order. **Never below S = 3.**

One warning the pruning audit makes concrete: **the top of your sparsity grid may be
scientifically empty.** Under the corrected protocol, Gemma-3-1B magnitude at 50% sparsity gives
PPL **386,224** and at 60% gives 92,636; Llama at magnitude 50% gives 2,412. A model at PPL 10⁵
does not have a circuit for IOI — it does not do IOI. Extracting one and reporting its CSI is
measuring noise and calling it damage. **Pre-register a behavioural floor** (e.g. "cells whose
task accuracy falls below chance are reported as behaviourally dead and excluded from CSI
claims") and you convert ~6 wasted cells into a principled scope statement — and save the
compute.

> **Fill in:** `B = 16` (fixed by Q4) · `measured s/pass = ____` · `Pythia S = __ R = __` ·
> `Primaries S = __ R = __` · `behavioural floor = ____`

**Run:**
```powershell
Measure-Command { python -m experiments.run_stage_a mode=scientific_run model=pythia160m task=ioi }
# then set ensemble sizes in configs/ensemble/default.yaml; confirm the run name encodes them
python -m pytest -k "run_naming or ensemble" -v
```

---

### The unnumbered items — my positions, briefly

| Item | My position | Why |
|---|---|---|
| **FDR level q** | `q = 0.05`, Benjamini–Hochberg, across the whole reported grid | Keep it. ~288 statements uncorrected at α=0.05 yields ~14 false "significant" cells. BH at 0.05 is the conventional default; deviating invites questions you gain nothing by answering. |
| **Bootstrap axes for CSI** | **Resample B, S *and* r** — change `csi.py` from r-only | `csi.py` bootstraps r only; CLAUDE.md §5 and the proposal both say "over B, S, r". r-only CIs are too narrow, and a too-narrow CI that excludes 1 is exactly the failure mode your paper exists to prevent. Fix **before** the freeze. This widens every CI and may turn a "significant" cell into a null — accept that; it is the honest number. |
| **Chance-floor N** | `candidate_edge_count(n_components)` from the *dense Stage-A union*, positions aggregated (Q11), recorded per cell | Must be the number of edges the extractor *could* have returned. 2607.18921 reports theirs (37, 31); so should you. |
| **INT-flag ratio** | Keep provisional `\|INT\| > 0.5 · max(\|NIE\|,\|PIE\|)` | arXiv:2606.27510 proves INT is *inevitable*, not a bug, and gives no canonical cutoff. 0.5 is a defensible round number; declare it a reporting convention and show sensitivity at 0.3 / 0.7 in the appendix. |
| **DVC** | Skip | Single machine; hashed manifests suffice. |
| **C5 grid alignment** | **Restrict to the intersection — it is 7 cells and I have enumerated them.** | Re-running their code on our grid is weeks of SAE training. The intersection is free. See Part 4. |

---

## Part 3 — Sourcing (Q6, Q7)

### Q7 — Calibration and perplexity: **mostly pre-decided, and one correction**

**Copy `../sae-pruning-paper-main/docs/PROTOCOL.md` verbatim.** Verified on disk:

| Parameter | Value |
|---|---|
| Corpus | WikiText-2 raw **test** split (~289K tokens) |
| Window / Stride | 1024 / 512 |
| BOS | `<bos>` prepended to **every** window |
| dtype | bfloat16 |
| Attention | `eager` for Gemma-2 (logit softcapping) |
| Reported | `Δlog PPL = log PPL_pruned − log PPL_dense` |

That document exists because the original protocol prepended `<bos>` once to the whole corpus
and evaluated bf16 models in fp16. Their forensic ladder shows restoring per-window `<bos>`
**alone** moves Gemma-2 from 192 → 12 while moving Llama by +0.31 — Llama is the negative control
that made the fault invisible. Wrap `saediag/ppl.py::windowed_ppl` rather than writing your own,
so your numbers come off the same code path as the ones you compare against.

**The correction to `HUMAN_DECISIONS.md`'s recommendation.** It recommends WikiText-2 "for both
calibration and the perplexity evaluation." Those are two different corpora in the reference
work, and conflating them breaks the cross-audit:

- **Perplexity evaluation: WikiText-2** test split. ✅ as recommended.
- **Calibration (Wanda / GPTQ / AWQ): FineWeb-Edu.** The pruning audit computes Wanda's
  activation norms from **300K calibration tokens of FineWeb-Edu**, and trains its SAEs on 1.5M
  FineWeb-Edu tokens. Calibrate Wanda on WikiText-2 while they calibrated on FineWeb-Edu and your
  Wanda-pruned model is not their Wanda-pruned model — C5 would compare two different
  interventions.

FineWeb-Edu is ODC-BY; WikiText-2 is CC BY-SA 3.0. Both permit research use. Record both in
`data/README.md` before use (AI_RULES.md §5).

> **Fill in:** `calibration = FineWeb-Edu, ODC-BY, 300K tokens (matching ref [1])` ·
> `perplexity = WikiText-2 raw test, CC BY-SA 3.0` · `protocol = reference verbatim` ·
> `n_calibration_samples = ____ seq_len = ____`

**Run:**
```powershell
python -m pytest -k "perplexity or ppl" -v
python -c "from src.science.perplexity import validate_ppl_protocol; print(validate_ppl_protocol(window=1024, stride=512, bos_per_window=True, dtype='bfloat16'))"
```

### Q6 — Task prompts

Unchanged from `HUMAN_DECISIONS.md`; nothing I read alters it. Generate from templates, don't
hard-code sentences. **n = 200–500 per task**, not the placeholder 1000.

The one thing to get right: prompts from one template are correlated, so **the bootstrap must
resample templates, not prompts**. Write that down now (AI_RULES.md 4.2 wants the bootstrap
scheme in advance) — it interacts with the Q3 bootstrap-axes decision, so settle both in one
sitting.

A scope note worth stating in the paper: CIRCUS validated on **short factoid completions**
(capitals, arithmetic, trivia), not IOI or greater-than. You are applying their ensemble
machinery to a different prompt family than it was demonstrated on. Fine and normal, but say it,
because their own limitations section flags that "the stability–coverage tradeoff will shift
under different families."

> **Fill in:** `ioi source/n_templates/n_prompts = ____` · `greater_than = ____` ·
> `docstring = ____` · `bootstrap resampling unit = template`

---

## Part 4 — The C5 cross-audit, rebuilt on the corrected numbers

`HUMAN_DECISIONS.md` leaves C5 as "reports PENDING rather than inventing the published numbers
it compares against." It no longer has to. The numbers exist, on your disk, frozen, with CIs.

**Target:** `../sae-pruning-paper-main/results/E4/diag_perf_correlations.csv`
**Frozen record:** `../sae-pruning-paper-main/results/E6/stat_tests.csv`

### The published feature-level damage correlations (vs Δlog PPL, n = 17 pruned conditions)

| Feature-level diagnostic | Spearman ρ | 95% CI |
|---|---|---|
| `fixed_dict_jaccard` (Gemma only) | **−0.975** | [−1.000, −0.873] |
| `downstream_macro_dacc` (n=16) | −0.906 | [−0.991, −0.632] |
| `mnn_survival_tau0.7` | **−0.684** | [−0.889, −0.276] |
| `fixed_dict_fvu` (Gemma only) | +0.556 | [0.030, 0.918] |
| `fragility_q1_minus_q5` | **−0.032** | [−0.473, 0.514] ← null |

**This is a far better C5 than "do the rankings agree".** It is a *calibrated* comparison:
compute `Spearman(circuit damage D, Δlog PPL)` on the same conditions and ask where your
circuit-level diagnostic lands in that table. Three outcomes, all publishable, all
pre-registerable:

1. ρ near **−0.975** → circuit damage tracks output damage as tightly as the best feature
   diagnostic; circuits add no information beyond perplexity.
2. ρ near **−0.684** → circuit damage behaves like feature survival: related to output damage
   but not reducible to it.
3. ρ near **−0.032** → **circuit damage is orthogonal to perplexity.** The strongest result
   available to you: you cannot predict circuit damage from the metric practitioners actually use
   to accept a compressed model — which is precisely the argument for circuit-level audits.

### The intersection grid — 7 cells, enumerated

Only conditions in both their frozen tables and your model list (Gemma-2-2B, Llama-3.2-1B) can
be paired. From `stat_tests.csv`:

| Model | Method | Sparsity | Their MNN survival | Their fixed-dict Jaccard | Their corrected PPL |
|---|---|---|---|---|---|
| Gemma-2-2B | magnitude | 0.3 | 0.210 (SD 0.0024) | 0.317 | 13.08 |
| Gemma-2-2B | magnitude | 0.5 | 0.002 (SD 0.0004) | 0.140 | 377.03 |
| Gemma-2-2B | Wanda | 0.3 | 0.794 (SD 0.0011) | 0.591 | 8.82 |
| Gemma-2-2B | Wanda | 0.5 | 0.148 (SD 0.0027) | 0.314 | 13.47 |
| Llama-3.2-1B | magnitude | 0.5 | 0.511 (SD 0.0005) | — | 2412.44 |
| Llama-3.2-1B | Wanda | 0.3 | 0.850 (SD 0.0015) | — | 10.65 |
| Llama-3.2-1B | Wanda | 0.5 | 0.709 (SD 0.0020) | — | 33.35 |

(Dense references: Gemma-2-2B **8.21**, Llama-3.2-1B **9.28**.)

**Make `configs/compression/` contain exactly these 7 cells**, or C5 has nothing to pair. Two
caveats for the paper:

- `fixed_dict_jaccard` is **Gemma-only** — no official Gemma Scope SAE fit was used for Llama. So
  the strongest correlation (−0.975) is available on one of your two primaries.
- Gemma-2-2B magnitude s=0.5 (PPL 377) and Llama magnitude s=0.5 (PPL 2412) are candidates for
  the Q3 behavioural floor. Decide *before* you see your CSI whether they count.

### Quantization has a feature-level reference too — but watch the model

- **arXiv:2606.03002 (Duan)** — RTN INT8→INT4 on **Pythia-70M and Gemma-2-2B**: your models, your
  bit-widths. Reports INT7 *improving* perplexity while degrading 18.7% of active Gemma features;
  INT6 improving perplexity with only 51.3% of features surviving. And the direct precedent for
  your null (b): RTN and **matched-perplexity magnitude pruning damage strongly overlapping
  feature sets — Jaccard 0.79–0.86, damage-score Spearman 0.98.**
- **The pruning audit's quant pilot** — `results/E5/quant_pilot.csv`: INT8 Jaccard 0.830 (PPL
  10.10), INT4 Jaccard 0.603 (PPL 10.87). **But on Gemma-3-1B, not one of your models.** Sanity
  anchor only, not a C5 pair.

Duan's Spearman-0.98 gives a second pre-registerable prediction: *if circuits behave like
features, RTN and matched-PPL pruning should damage the same circuit edges.* If they do not, you
have found a place where computation and representation come apart — the whole thesis of measuring
circuits rather than features.

### Two precedents to cite, because they legitimise your method

- **A published noise floor.** `stat_tests.csv`: "seed-to-seed MNN floor (descriptive),
  same-model seed pairs @τ0.7, n=22, mean 0.0273 [0.0113, 0.0393]" — and they report
  cross-condition effects as "18–29× above this seed-variability floor." Prior work already
  accepted that a floor is required; yours is simply a *better* floor (matched-magnitude, not just
  seed variation). Frame it that way in related work — a much easier sell than claiming nobody
  thought of it.
- **A threshold-robustness statistic.** `stat_tests.csv`: "Kendall's W concordance, condition
  rankings across τ∈{.6,.7,.8}, n=17, W = 0.9416." Exactly the form your C3 threshold sweep
  should take. Report Kendall's W across your threshold grid and you speak the same statistical
  language as the work you compare against.

**Run:**
```powershell
python -m analysis.cross_audit       # should stop reporting PENDING once the 7 cells exist
python -m pytest -k "cross_audit" -v
```

---

## Part 5 — Every command, in one place

```powershell
cd "D:\Users\SUPRATIK\AAAI-UC\AAAI - Standalone Project\Circuits_Under_Compression"

# --- health ---
pip install -e .
python -m pytest                                             # 462 passed, 2 skipped
python -m pytest -k "not integration and not regression"      # fast unit tier
python -m pytest -k "integration"
python -m experiments.dry_run_stage_a 0                       # synthetic, no model, no GPU

# --- per-decision guards (flip from refuse -> pass as you decide) ---
python -m pytest -k "decompose" -v                # Q1
python -m pytest -k "distance" -v                 # Q2  (14 tests)
python -m pytest -k "threshold_grid" -v           # Q4  (6 tests)
python -m pytest -k "two_level or node_ids" -v    # Q10, Q11
python -m pytest -k "perplexity or ppl" -v        # Q7
python -m pytest tests/test_licence_compliance.py -v

# --- real pipeline (blocked until decisions + Q5 + Q6) ---
python -m experiments.run_stage_a mode=scientific_run model=pythia160m task=ioi
python -m experiments.run_stage_b mode=scientific_run model=pythia160m task=ioi nulls=matched_magnitude

# --- the freeze ---
python -m experiments.freeze_stage_b --config <resolved.json> --dry-run   # ALWAYS first
python -m experiments.freeze_stage_b --config <resolved.json>             # irreversible
git tag -a null-freeze -m "Pre-registration: null distributions frozen"

# --- before EVERY Stage C batch ---
python -c "from src.common.freeze import verify_frozen_store; print(verify_frozen_store())"

# --- Stage C / D / analysis ---
python -m experiments.run_stage_c mode=scientific_run model=<m> task=<t> compression=<c>
python -m experiments.run_stage_d mode=scientific_run model=<m> task=<t>
python -m analysis.threshold_sweep
python -m analysis.cross_audit

# --- exit gate ---
python -m pytest tests/test_regression_ioi_gpt2_small.py -v
```

Note the `-m` everywhere. `python experiments/dry_run_stage_a.py` fails with
`ModuleNotFoundError: experiments`.

---

## Part 6 — What I would actually do next, in order

1. **Step 0** — `pytest`. Five minutes, and everything below assumes it.
2. **Q1 + Q4 today.** No longer judgment calls; CIRCUS decided them and the quotes are in §0.1
   and Q4. Thirty minutes of config editing converts two open scientific decisions into two
   citations. Biggest value-per-minute in the project.
3. **Q2.** Confirm L1 / JS, add normalised L1, flip `pre_registered_for_stage_c: true`.
4. **Q10 + Q11.** Take `layer` and `aggregate`, but write the Q10 falsification condition into
   the config comment, and fix the `two_level.py` ↔ `project_level2` wiring gap or C2 reports
   zeros forever.
5. **Fix the CSI bootstrap axes before the freeze.** r-only is narrower than what the proposal
   promises, and after the freeze it is expensive to change.
6. **Q5** — HF hashes and gated licences. Clerical, blocks everything.
7. **Q7** — copy the protocol verbatim; calibrate on FineWeb-Edu, evaluate on WikiText-2.
8. **Align `configs/compression/` to the 7 intersection cells** (Part 4) so C5 is alive rather
   than PENDING.
9. **Pre-register the two directional predictions** (exact-edge vs coarse, §0.5; RTN vs
   matched-PPL pruning, Part 4) in the freeze commit message.

The one thing I would not do: start Stage B on the primaries before the exit gate passes. A stack
that cannot find the IOI circuit in GPT-2 cannot be believed when it says one moved, and a frozen
null computed with a broken extractor is worse than no null — it is a broken yardstick with a
hash on it.

---

## Appendix — sources, all verified 2026-09-11

| Ref | What | Used for |
|---|---|---|
| arXiv:2603.00523 | CIRCUS (Parekh, Intuit). §3.2 taxonomy + non-nesting; B=9/25; 5.5% overhead; Jaccard 0.33/0.35; Match | Q1, Q3, Q4 |
| arXiv:2607.18921 | Circuit Claims Depend on What Is Extracted (Sheng & Fu). Tables 11, 12, 18 | Q10, §0.5 |
| arXiv:2603.25325 | How Pruning Reshapes Features (Borobia et al.) — **v1 tables superseded** | C5 (via repo) |
| arXiv:2606.03002 | Perplexity Can Miss SAE Feature Damage Under Quantization (Duan) | C5 quant arm, null (b) |
| arXiv:2606.27510 | Curse of Multiple Mediators (Vaidyanathan et al.) | INT-flag rule |
| arXiv:2603.25035 | Mechanistically Interpreting Compression in VLMs | the qualitative claim being tested |
| `decoderesearch/circuit-tracer` @ `8f1e2438…` | v0.5.2; CLI defaults node 0.8 / edge 0.98; MIT | Q4, Q9 |
| `hecboar/sae-pruning-paper` @ `261191…` | `docs/PROTOCOL.md`, `results/E4/`, `results/E6/` | Q7, C5 |

**Not found:** a CIRCUS code release. The paper is on OpenReview (`M3wusJ6Mqo`); no repository
surfaced. Q4 therefore reimplements the prose construction rather than reusing their code — state
that in the paper ("following the construction of …, which we reimplement, as no reference
implementation is available").
