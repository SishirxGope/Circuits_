# HUMAN_DECISIONS.md — everything only YOU can do

> **Eleven decisions, not nine.** Q10 and Q11 were forced into the open on 2026-08-08 by
> reading the circuit-tracer fork instead of trusting the plan; both block Stage A.

> ⚠️ **2026-09-11: see `docs/Run_Plan.md` before answering Q1, Q4, Q7, Q9 or C5.** A literature
> pass verified that CIRCUS fixes the band cutoffs (Q1: core s=1, noise s<0.5 — *not* our
> provisional 0.9/0.1) and the config construction (Q4), recovered both upstream hashes (Q9),
> corrected the calibration corpus (Q7: FineWeb-Edu for calibration, WikiText-2 for eval), and
> found that the feature-level numbers C5 compares against were corrected by their own authors
> (ρ = −1.0 → −0.540..0.062; Gemma-2 dense PPL 410 → 8.21). Run_Plan.md supersedes the
> recommendations in this file wherever the two disagree.

This is the **single** file for human decisions and human actions on this project.
It replaces the old `pi_decisions.md`, `pi_decisions_resolved_provisional.md` and
`upstream_pins.md` (merged here on 2026-08-08 — no content was lost).

Read Part 1 first. It is an ordered to-do list. Part 2 explains each decision. Part 3
is reference material you only need when you are actually filling something in.

---

## Part 0 — Where the project actually stands

| | Status |
|---|---|
| Code | Complete for everything that can be built without a model or a decision. 510 tests pass, 1 skipped (the exit gate, by design) with the `models` extra installed. A 30-point audit of PRD/AI_RULES/ARCHITECTURE requirements reports 30/30 implemented. |
| Stage A (dense reference) | Runs end-to-end on a synthetic mock model. **Real dense-node extraction on Pythia-160M works** (2026-09-14: pinned loader, edge attribution patching, threshold port, verified by finite-difference tests). Stage B/C for real models are not built yet. |
| Stage B (null) | Produces a **draft** null on the mock model. The *freeze* is blocked — deliberately. |
| Stage C (real compression) | Refuses on real models. Runs end-to-end on the synthetic stack, so Algorithm 1 is validated plumbing, not untested plumbing. |
| Stage D (causal + cross-audit) | Same. Cross-audit reports PENDING rather than inventing the published numbers it compares against. |
| The freeze | Has a runnable, tested path (`experiments/freeze_stage_b.py`). It refuses until your decisions are final. |
| Environment | hydra installed, git initialised + pushed, MIT `LICENSE` added, all four entrypoints executed |
| The science | **Nothing has been decided.** Every scientific value in the repo is a placeholder marked PROVISIONAL. |

**Nothing here is a bug.** The pipeline refuses to run because you have not made the
decisions yet, and the code is built so that it cannot quietly proceed without them.
That refusal *is* the contribution — it is what "the null is frozen before the effect
is measured" looks like in code.

### The one thing to understand before you decide anything

Your paper's claim is not "compression changes circuits." It is: **"we measured
whether compression changes circuits more than random noise of the same size does,
and we fixed the yardstick before we looked."** Every decision below is a piece of
that yardstick. Once you run Stage B and freeze, **you can never change them** for
the affected cells without invalidating and re-running everything downstream
(AI_RULES.md 1.3, 1.4). That is why they are all being asked now, together, before
anything real runs.

A reviewer's job is to ask "did you pick that threshold *after* seeing the result?"
The freeze is your answer. So: decide, write it down, freeze, then run.

---

## Part 1 — Do these in this order

Items marked **[15 min]** etc. are rough. Items marked ⚠️ block everything after them.

### ~~Step 1 — Install the two missing packages~~ DONE (2026-08-09)

`hydra-core` 1.3.5 + `omegaconf` 2.3.1 are installed and **all four Hydra entrypoints
plus the freeze CLI have now been executed for real**. That surface had never once been
run before — the tests call `run_stage_*(dict)` directly and bypass config composition,
so a green suite said nothing about it. Running it surfaced three defects (run-directory
collisions, Stage B/C/D run names hiding R, a `NameError` in Stage D), all fixed.

```bash
cd "D:\Users\SUPRATIK\AAAI-UC\AAAI - Standalone Project\Circuits_Under_Compression"
pip install -e .
python -m pytest                       # expect: 510 passed, 1 skipped (models extra installed)
python -m experiments.dry_run_stage_a 0
```

*Why:* without Hydra the config system — the thing that makes every ablation a
one-key override (ARCHITECTURE.md §4) — is inert.

### ~~Step 2 — Put the project under git~~ DONE (2026-08-09)

`git init` is done and the repo is pushed to
`github.com/SupratikB23/Circuits_Under_Compression`. Runs now record a real
`git_commit`, and the freeze has a commit to anchor to. Why it mattered:

- AI_RULES.md 1.4 requires the null freeze to be a **commit**. Without git there is
  nothing to freeze *at*.
- Every run records `git_commit` — currently `null` in every run log.
- PRD.md §6 requires the artifact to ship a freeze-commit reference.
- Without version control, an accidental edit to a config after a run is undetectable,
  which is exactly the failure mode AI_RULES.md 1.2 exists to prevent.

**One thing the cleanup commits cost you, now restored:**
`THIRD_PARTY_LICENSES/` was deleted in "docs: remove obsolete documentation files". It
is not obsolete — both upstreams are MIT, and MIT *requires* their copyright and
permission notices to travel with any substantial portion of the code. This repo adapts
the pruning maths from `saediag.pruning` and wraps `circuit-tracer`, and your artifact
ships that code, so removing the notices is a licence violation rather than a tidy-up.
Restored byte-identical from the forks, and `tests/test_licence_compliance.py` now fails
if it ever goes missing again.

Also gone: `runs/.gitkeep` and `artifact/.gitkeep`. Harmless — `runs/` auto-creates on a
fresh clone (verified), and `artifact/` is a Phase-5 placeholder.

### Step 3 — Answer the three lookups  <- YOU ARE HERE ⚠️ **[45 min]** → Q5, Q8, Q9

No scientific judgment needed. These are "go and copy a value." Details in Part 2.

- ~~**Q5** — HuggingFace revision hashes.~~ **DONE 2026-09-12** — all 8 pinned (4 model
  configs, GPT-2 exit gate, Pythia-70M, 2 transcoder sets). ⚠️ The two gated licences are
  still unaccepted and still block real runs on the primaries.
- **Q8** — Weights & Biases or MLflow. Pick one; W&B recommended.
- ~~**Q9** — the two upstream commit hashes.~~ **DONE 2026-09-12** — both recorded in
  §3.3 and in the `# Adapted from:` headers. One residual check is flagged there.

### ~~Step 4 — Make the four scientific calls~~ **DONE 2026-09-12** → Q2, Q1, Q4 (Q3 open)

In this order, because Q2 is the one everything else hangs off. Details in Part 2.

- **Q2** — the distance function D. **Do this one first and most carefully.**
- **Q1** — the core/contingent/noise band cutoffs.
- **Q4** — how the B threshold configurations are generated.
- **Q3** — B, S, R. Requires Step 6 (measure wall time) to finalise.

### ~~Step 4b — Two decisions the upstream fork forced into the open~~ **DONE 2026-09-12**

Found on 2026-08-08 by reading `circuit-tracer-0.5.2` rather than trusting the plan.
Both block Stage A. Full detail in Part 2.

- **Q10 — what is the "routing-head" level for pipeline A?** circuit-tracer emits **no
  attention-head nodes at all.** Without a decision here, claim C2 is vacuous for your
  primary pipeline.
- **Q11 — do edges aggregate over token positions?** Upstream nodes are
  `(layer, pos, feature_idx)`; your edge id has no position.

### ~~Step 5 — Source the data~~ **DONE 2026-09-12** → Q6, Q7 (docstring source still open)

- **Q6** — the IOI / greater-than / docstring prompt sets.
- **Q7** — the GPTQ/AWQ calibration corpus and its licence.

### Step 6 — First real Stage A run, then finalise Q3 **[1 day]**

Get onto the DGX Spark / 4060, run Stage A on **Pythia-160M first** (cheapest, and
ARCHITECTURE.md §5 says debug the null pipeline on Pythia before the primaries), time
one attribution pass, then set B/S/R from the measured number rather than from a guess.

### Step 7 — The Phase-1 exit gate ⚠️ **[1–2 days]**

`tests/test_regression_ioi_gpt2_small.py` is the gate and it currently **skips**. It
reproduces a published circuit (IOI on GPT-2 small) through your own extraction stack.
Until it passes, no measurement from this pipeline is trustworthy — a stack that cannot
find a circuit everyone agrees exists cannot be believed when it says one moved.

You must supply two things: the reference edge list, and the tolerance (what Jaccard
counts as "reproduced"). Set the tolerance **before** you see your own number.

### Step 8 — The freeze ⚠️ **THE POINT OF NO RETURN**

Run Stage B for every cell, then freeze. After this commit, `frozen/` is append-only
and no null may be re-tuned, ever, for any cell whose real-compression counterpart has
been run. Tag the commit. This is your pre-registration.

```bash
python -m experiments.freeze_stage_b --config <resolved.json> --dry-run   # ALWAYS first
python -m experiments.freeze_stage_b --config <resolved.json>             # irreversible
```

The dry-run writes nothing and reports exactly what would be frozen and what is
blocking. The real command refuses unless mode is `scientific_run`, `stage_b.
freeze_approved` is explicitly true, every Stage-A-gating decision is resolved, **and
Q2 is pre-registered** — a floor frozen before D is chosen is not a floor for anything
in particular. It re-hashes each null at its destination and re-verifies the whole
store afterwards; a store that fails verification aborts the freeze.

Afterwards, `verify_frozen_store()` re-hashes everything and reports any drift. Run it
before every Stage C batch — it is the cheapest possible check that the floor has not
moved under your results.

### Step 9 — Stage C, Stage D, paper

Only now does real compression run. Note that **Algorithm 1 already runs end-to-end on
the synthetic stack** (`tests/test_integration_algorithm1.py` runs A → B → freeze → C →
D on a mock model), so what you switch on here is real models, not untested plumbing.

### Also open, not numbered (decide before they bite)

| Item | Where | When it bites |
|---|---|---|
| ~~Artifact licence: MIT or Apache-2.0~~ **RESOLVED 2026-08-09: MIT** | PRD.md §6 | Done — `LICENSE` (MIT, © 2026 Supratik Bhowal). MIT-on-MIT is compatible with both upstreams **provided their notices ship with the artifact**, which is what `THIRD_PARTY_LICENSES/` is for; `tests/test_licence_compliance.py` now enforces it. |
| The quantitative INT-flag rule for C4 | AI_RULES.md 4.2 | Before Stage D. Currently provisional: `\|INT\| > 0.5 · max(\|NIE\|, \|PIE\|)`. Implemented and tested in `src/science/patch_diagnostic.py`; you are choosing the ratio, not the mechanism. **Still open.** |
| ~~**FDR level q for the grid**~~ | AI_RULES.md 4.3 | **RESOLVED 2026-09-12: q = 0.05**, now in `configs/config.yaml` under `stage_d.fdr_q` rather than as a module default, so every run records it in its resolved config and its hash. Fixing this also removed a duplicated literal `0.05` fallback in `run_stage_d`, which was read **twice** — the reported q and the q actually applied did not have to agree. |
| ~~**Bootstrap resampling axes for CSI**~~ | proposal §2.1 | **RESOLVED 2026-09-12: B, S, r**, as the proposal specifies. `csi_over_ensemble()` resamples the B config ids and the S seeds with replacement, recomputes s(e) and D over the resampled grid, and resamples the null independently. Dense and compressed are resampled with the **same** labels because they are paired. Measured effect on a heterogeneous test ensemble: CI width 1.10 → 3.14, same point estimate. |
| ~~**Candidate-edge universe N for the chance floor**~~ | AI_RULES.md 4.4 | **RESOLVED 2026-09-12: the observed node union.** N = U·(U−1) over the nodes appearing anywhere in the cell's dense + compressed circuits. Stage C previously used every component in the model, which for pipeline A counts nodes the extractor could never have emitted — inflating N, pushing the random-overlap baseline down, and making every overlap look above chance for combinatorial reasons. |
| DVC vs plain hashed manifests | ARCHITECTURE.md §2 | Probably never — for a single-machine project hashed manifests are enough. Recommend: skip DVC. |
| C5 grid alignment: restrict to the intersection, or re-run the published feature-audit code on your grid | PRD.md §2 | Before Stage D. |

---

## Part 2 — The nine decisions

Statuses: **OPEN** (nothing chosen) · **PROVISIONAL** (engineering placeholder in use,
not a pre-registration) · **PI-INPUT-NEEDED** (only you can supply the value).

---

### Q2 — The distance function D  **PRE-REGISTERED 2026-09-12**

**What.** `CSI(c,T) = D(c) / median(D_null(c))`. `D` measures how far the
post-compression edge-inclusion-frequency vector `s(e)` sits from the dense one. You
must name a **primary** D and at least **one alternative** for a sensitivity check.

**Why it matters.** CSI is your headline number and it is *entirely* downstream of D.
Choose a different D and every CSI in the paper changes. If you choose it after seeing
results, that is p-hacking (AI_RULES.md 4.5) and a reviewer will say so. It must be
pre-registered before Stage C.

**Options.**

| Option | What it does | Trade-off |
|---|---|---|
| **L1** | `Σ \|s_pre(e) − s_post(e)\|` over the union of edges | Simplest. "Total circuit change" in the most literal sense. Its null distribution is easy to read. Scales with circuit size, so cells with different edge counts are not directly comparable — report normalised alongside if that comes up. |
| **Jensen–Shannon** | JS divergence over the **normalised** vectors | Respects the simplex structure of s(e). **But**: normalisation makes it blind to a uniform change in total inclusion mass — if compression halves *every* s(e), JS says nothing happened. That is a real scenario under heavy pruning. |
| **Wasserstein** | Earth-mover over binned/ranked s(e) | Sensitive to distribution *shape*, possibly more stable to threshold noise. Harder to bootstrap, harder to explain in a 6-page paper. |
| **Cosine** | Angle between the vectors | Ignores magnitude entirely — same blindness as JS, with less principle behind it. |

**Recommendation: primary = L1, alternative = Jensen–Shannon.** L1 is what the
sentence "the circuit changed by this much" actually means, it has no hidden
assumptions, and its bootstrap is trivial. JS as the ablation is a genuinely different
lens (relative shape vs absolute mass), which is what an ablation is for. PRD.md §3
lists exactly this pair as the P0 ablation row.

**How to record it.** Write the values in the box below, then edit
`configs/distance/provisional_l1_js.yaml`: set `pre_registered_for_stage_c: true` and
`status: FINAL`. Both functions are already implemented and tested in
`src/science/distances.py` (14 tests).

> **PI fill-in**
> ```
> primary_D   = ____________________
> ablation_D  = ____________________
> justification (one line) = ______________________________________________
> ```

**Current provisional value:** primary L1, alternative Jensen–Shannon,
`pre_registered_for_stage_c: false`. Status: **PROVISIONAL**.

---

### Q10 — What "routing-head level" means for pipeline A  **PRE-REGISTERED 2026-09-12**

**What.** Proposal §2.4 requires every claim at two levels: exact component-to-component
edge overlap, **and routing-head-set overlap**. Protocol rule 3: "two levels or it does
not count."

**The problem, verified from the fork on 2026-08-08.** circuit-tracer's graph has
exactly four node kinds — and none of them is an attention head:

| Upstream node kind | Constructor |
|---|---|
| `cross layer transcoder` | `Node.feature_node(layer, pos, feat_idx)` |
| `mlp reconstruction error` | `Node.error_node(layer, pos)` |
| `embedding` | `Node.token_node(pos, vocab_idx)` |
| `logit` | `Node.logit_node(...)` |

Its graph is over **transcoder features**, not heads. So `two_level.py`'s routing-head
projection (which matches `^L\d+\.H\d+$`) returns the **empty set** for every
pipeline-A circuit, and claim C2 becomes vacuous for your primary pipeline. This is
not a coding bug — it is the plan meeting the tool.

**Options** (implemented and documented in `src/extraction/node_ids.py::LEVEL2_OPTIONS`):

| Option | What the coarse level becomes | Trade-off |
|---|---|---|
| **`layer`** | "which layers route the computation" | Always defined, cheap, and keeps the *spirit* of ref 2607.18921 — a coarser description that can disagree with the fine one. Loses the literal word "head". |
| `dense_node_heads` | the routing-head level is carried by the **dense-node pipeline**, which does have heads | Most faithful to the proposal's wording. Most work, and it entangles C2 with C8 (basis drift), so a disagreement becomes hard to attribute. |
| `node_kind` | feature / error / embed / logit | Trivially defined, almost certainly too coarse to say anything. |
| `feature_family` | collapse positions of the same feature | **Not a second level at all** — it is position aggregation (that is Q11). Does not satisfy C2. |

**Recommendation: `layer`, and say so plainly in the paper.** Rewrite the two levels as
"exact edge" and "layer-routing", state in one sentence that circuit-tracer's node
basis is transcoder features so head-level routing is not available in pipeline A, and
cite that as a limitation. This keeps C2 real, keeps it cheap, and is honest. If you
have time later, `dense_node_heads` is the stronger version — but it is a Phase-4
luxury, not a Phase-1 requirement.

> **PI fill-in**
> ```
> level2_scheme = [ layer / dense_node_heads / node_kind / other: ______ ]
> paper wording for the two levels = ______________________________________
> ```

**Status: PRE-REGISTERED 2026-09-12** as `level2_scheme = layer`, **with** the
falsification condition in `src/science/two_level.py::coarse_level_verdict`. Recorded
in `configs/comparison/final.yaml`. `project_level2()` still refuses an unknown scheme.

---

### Q11 — Do edges aggregate over token positions?  **PRE-REGISTERED 2026-09-12**

**What.** Upstream nodes are position-specific: `(layer, pos, feature_idx)`. Our
`edges.parquet` schema is `(src_component, dst_component, config_id, seed, included)` —
**no position column**. So several distinct upstream edges collapse onto one component
pair.

**Why it matters.** This is a modelling claim, not a detail. Aggregating says *"the
circuit is the same computation wherever in the prompt it fires."* For a task circuit
that is usually what you want — but it changes the size of the edge universe, which
changes the random top-k chance floor (`chance_floor.candidate_edge_count`), which
changes whether an overlap is "above chance".

It also already bit the code: because the collapse produces repeated edges within one
ensemble cell, `s(e)` was being computed above 1.0 — not a fraction. That is fixed
(edges are now counted once per cell), but the *policy* is still yours.

**Recommendation: aggregate (keep the current schema).** It matches the existing
schema, keeps the edge universe small enough for the ensemble to be meaningful, and
matches how people talk about circuits. State the sentence
`node_ids.position_policy_is_pi_owned(False)` returns, in the paper.

Choosing position-specific instead means adding a `pos` column to `edges.parquet`
(ARCHITECTURE.md §2), multiplying the edge universe by `n_pos`, and re-deriving the
chance floor.

> **PI fill-in**
> ```
> position_policy = [ aggregate over positions / position-specific ]
> ```

**Status: PRE-REGISTERED 2026-09-12** as `position_policy = aggregate`, recorded in
`configs/comparison/final.yaml`. The code already aggregated; what changed is that this
is now a stated modelling claim rather than a schema side-effect, and the sentence
`node_ids.position_policy_is_pi_owned(False)` returns must appear in the paper.

---

### Q1 — Band cutoffs  **PRE-REGISTERED 2026-09-12** (CIRCUS §3.2, adopted verbatim)

**What.** Two numbers that split edges by inclusion frequency `s(e)`:
`s(e) ≥ core_threshold` → core; `s(e) ≤ noise_threshold` → noise; everything else →
contingent. (Boundaries are inclusive; that convention is fixed and tested.)

**Why it matters.** This is the C3 reporting object: "binary edge lists are unstable;
inclusion-frequency reporting is the valid object." The cutoffs decide which edges are
called structural, so they decide every "the core shrank under INT4" sentence in the
paper. CLAUDE.md §5 asks for them fixed **before Stage A**.

**Options.**
- **A.** `core ≥ 0.9`, `noise ≤ 0.1` — the example in CLAUDE.md §5.
- **B.** `core ≥ 0.8`, `noise ≤ 0.2` — a wider core, fewer contingent edges.
- **C.** Terciles of the observed Stage-A distribution — adapts to the data, but adds a
  data dependence that must itself be frozen before Stage C, and makes cells harder to
  compare against each other.
- **D.** Whatever CIRCUS (arXiv:2603.00523) uses — **only if you open the paper and
  verify it**. Do not adopt a cutoff you have not read (AI_RULES.md 2.1/2.2).

**Recommendation: check D first, fall back to A.** Spend 20 minutes on the CIRCUS
paper. If it fixes cutoffs, use theirs and cite them — that removes an entire
reviewer objection at zero cost. If it does not, use `core ≥ 0.9, noise ≤ 0.1` and say
plainly in the paper that the cutoffs are a reporting convention, that the threshold
sweep (C3) shows which conclusions depend on them, and move on.

**How to record it.** Fill the box, then set both values in
`configs/ensemble/decompose/final.yaml` (they are `null` there now — deliberately; a
test enforces that they stay null until you decide). Switch the root config's
`ensemble/decompose` default from `provisional` to `final`.

> **PI fill-in**
> ```
> core_threshold   ≥ ______
> noise_threshold  ≤ ______
> source = [ CIRCUS §___ / my choice ]
> ```

**Current provisional value:** core ≥ 0.90, noise ≤ 0.10. Status: **PROVISIONAL**.

---

### Q4 — The B non-nested threshold configs  **PRE-REGISTERED 2026-09-12**

**What.** The ensemble runs B different pruning-threshold configurations. "Non-nested"
means no configuration's edge set is a strict subset of another's — otherwise you are
counting near-duplicates and `s(e)` is inflated toward whatever the loosest threshold
found.

**Why it matters.** This is the CIRCUS mechanism (proposal §2.1c): it is what separates
*analyst degrees of freedom* from *model structure*. The design changes what `s(e)`
means.

**Options.**
- **A.** Uniform grid over the threshold axes — transparent, trivially pre-registerable.
- **B.** Quantile-based, anchored on the Stage-A score distribution — tracks the data,
  adds a dependence that must be frozen.
- **C.** CIRCUS's own construction, taken from their release — **strongest if it exists
  and you verify it.**
- **D.** Seeded random non-nested sampling — a valid pre-registered randomisation, but
  adds seed sensitivity on top of the seed axis you already have.

**Recommendation: C if it exists, otherwise A.** Look for a config-generation function
in the CIRCUS release. Reusing it is less work *and* more defensible — "we used the
ensemble construction from the paper that introduced the method." If you cannot find
one, use a uniform grid: it is the easiest thing to describe in one sentence of a
6-page paper, and describability matters here.

Whatever you choose goes in `configs/ensemble/default.yaml` under `threshold_grid`.
The code refuses to invent a grid in scientific mode.

> **PI fill-in**
> ```
> grid_design = [ CIRCUS construction (source: ______) / uniform grid / other: ______ ]
> node_threshold range = ______ to ______
> edge_threshold range = ______ to ______
> ```

**Current provisional value:** deterministic seeded non-nested grid generated from the
run seed (`src/science/threshold_grid.py`, 6 tests) — an engineering placeholder only.
Status: **PROVISIONAL**.

---

### Q3 — Ensemble sizes B, S, R

**What.** B pruning configs × S seeds per cell, and R null draws per cell.

**Why it matters.** This is a compute-budget decision, not a scientific one — but it
sets the width of every confidence interval you will report. Too small and every CI
straddles 1 and you conclude nothing. The ensemble, not the model, is the dominant cost
(CLAUDE.md §6). Critically: **budget is controlled by shrinking R and B, never by
dropping the null.**

**You cannot finalise this from a desk.** You need one measured number: how long one
attribution pass takes on Pythia-160M on your hardware. Per cell you need roughly
`S` attribution passes (CIRCUS's insight is that the B configs re-prune one
already-computed attribution run, so B is nearly free), and Stage B multiplies that by
R.

**Recommendation.** Start from ARCHITECTURE.md §5: **B=16, S=5, R=20 on Pythia**. Run
Stage A once on Pythia-160M, time it, then:
- if one attribution pass ≤ ~2 min → keep those numbers for the primaries too;
- if it is much slower on Gemma/Llama → halve **R first** (20→10), then **B** (16→8),
  and keep S ≥ 3 so the seed axis still exists.

> **PI fill-in**
> ```
> measured seconds per attribution pass (Pythia-160M) = ______
> Pythia     B = ____  S = ____  R = ____
> Primaries  B = ____  S = ____  R = ____
> ```

**Current provisional value:** synthetic/dry-run B=4, S=2, R=3. The B=16/S=5/R=20
pilot is **proposed, not approved**. Status: **PROVISIONAL**.

---

### Q5 — HuggingFace revision pins  **RESOLVED 2026-09-12** (one PI action left)

**What.** The exact commit hash of each model checkpoint.

**Why it matters.** Model repos get updated. If you extract your dense reference in
week 3 and someone re-uploads the weights in week 6, your "before" and "after" are
different models and your entire measurement is noise you cannot detect. Pinning costs
nothing and removes the risk permanently.

**Done — all eight pins recorded.** Read from the HuggingFace API rather than the web UI
(`api/models/<repo>` → `sha`, i.e. each repo's current `main` HEAD). Gating restricts
**downloads, not metadata**, so the two gated shas were readable without credentials.

> ### ⚠️ The one thing still on you
> Pinning is done; **accepting the gated licences is not**, and it still blocks every real
> run on the two primaries. `google/gemma-2-2b` and `meta-llama/Llama-3.2-1B` are both
> `gated: manual` — a human request approved by the owner, which can take hours or days.
> Do it now, not on run day.
>
> ```bash
> huggingface-cli login          # then accept the licences on both model pages
> ```
>
> Tracked as `gated_licences_accepted: false` in `configs/provisional_defaults.yaml`.
> Pythia and GPT-2 are ungated, so the Pythia-160M pilot (Run_Plan Step 5) and the GPT-2
> exit gate (Step 6) are **not** blocked by this.

| Repo | Pinned revision | Gated | Recorded in |
|---|---|---|---|
| `google/gemma-2-2b` | `c5ebcd40d208330abc697524c919956e692655cf` | ⚠️ manual | `configs/model/gemma2_2b.yaml` |
| `meta-llama/Llama-3.2-1B` | `4e20de362430cd3b72f300e6b0f18e50e7166e08` | ⚠️ manual | `configs/model/llama32_1b.yaml` |
| `EleutherAI/pythia-160m` | `50f5173d932e8e61f858120bcb800b97af589f46` | no | `configs/model/pythia160m.yaml` |
| `EleutherAI/pythia-410m` | `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f` | no | `configs/model/pythia410m.yaml` |
| `EleutherAI/pythia-70m` | `a39f36b100fe8a5377810d56c3f4789b9c53ac42` | no | no config yet |
| `openai-community/gpt2` (exit gate) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | no | no config yet |
| `mntss/gemma-scope-transcoders` | `9250a2d4860ce5ed5c96c14d5882b7d8162809a3` | no | `transcoder_revision:` in `gemma2_2b.yaml` |
| `mntss/transcoder-Llama-3.2-1B` | `c37a82c1ec4cea30d424850d159b17b720ce19e2` | no | `transcoder_revision:` in `llama32_1b.yaml` |

All are asserted **by value** in `tests/test_config_integrity.py`, not merely as "not
null" — a silent re-pin would make a Stage A dense reference and a Stage C re-extraction
different models while every hash in `run_meta.json` still looked self-consistent.

**A pinned model with a floating transcoder set is still a floating circuit**, so the two
transcoder sets got their own `transcoder_revision:` field and their own test. There was
no such field in the config schema before.

**Pythia-70M and GPT-2 have no `configs/model/*.yaml` yet** — their pins are recorded here
and in `configs/provisional_defaults.yaml`, ready for when the exit gate (Step 6) needs
them.

The last two are the transcoder sets pipeline A needs (from the circuit-tracer fork's
README). **Note:** no transcoder set exists for Pythia, so pipeline A cannot run on
Pythia at all — Pythia goes through the dense-node pipeline. That is already recorded
in `configs/model/pythia*.yaml` and is a limitation to state in the paper, not a bug.

### Verified while pinning — and one thing that did not check out

The model configs carry architecture numbers marked `PLACEHOLDER - verify against pinned
HF config`. For the two ungated models that verification is now done, against
`config.json` at the pinned sha: **Pythia-160M (768/12/12/64/2048) and Pythia-410M
(1024/24/16/64/2048) are both exactly right**, and are marked VERIFIED in the configs.
Gemma-2 and Llama-3.2 return `401` on `resolve/<sha>/config.json` without accepted
licences, so their numbers stay PLACEHOLDER — **re-check them immediately after you
accept**, before Stage A.

> ⚠️ **New open item: the Pythia eval dtype.** Both Pythia configs say `dtype: bfloat16`,
> "verify against fork eval protocol". The fork's `docs/PROTOCOL.md` does fix compute
> dtype at **bf16** — but its stated reason is that an earlier run "ran in float16 on
> models trained in bfloat16, which pushed Gemma 3 toward non-finite values", i.e. the
> rule is *match the training dtype*. **Pythia reports `torch_dtype: float16`** and does
> not appear in that protocol's table at all. The two Pythia configs currently inherit a
> justification that does not apply to them.
>
> Not cosmetic: Stage B perturbs weights at a matched magnitude, and the precision the
> weights live in sets the floor that magnitude is measured against. Settle it before the
> Pythia-160M pilot (Run_Plan Step 5) — that is the run which fixes B/S/R.

**Status: RESOLVED** for the pins. Outstanding: accept the two gated licences; verify the
Gemma/Llama architecture numbers once their configs are readable; settle the Pythia
dtype.

---

### Q8 — Experiment tracker  **RESOLVED 2026-09-12: Weights & Biases**

**What.** Where run metrics and tags are recorded, in addition to the local
`run_meta.json` (which stays the source of record either way).

**Recommendation: Weights & Biases.** Free academic tier, works from any machine, and
because your runs happen on a different box than the one you write on, a web dashboard
genuinely helps. MLflow is the alternative if you would rather keep everything local
and offline. This decision is cheap and reversible — a tracker is an extra sink, not a
dependency.

```bash
pip install -e ".[tracking]"    # W&B      (or ".[tracking-local]" for MLflow)
wandb login
```

> **PI fill-in**
> ```
> tracker = [ W&B / MLflow / JSON only ]
> ```

**Status: RESOLVED 2026-09-12 — Weights & Biases**, with `run_meta.json` remaining the
**source of record**. The tracker is an extra sink, not a dependency: if W&B is
unreachable the run still completes and is still reproducible from `run_meta.json` plus
the resolved-config hash. Recorded as `q8_tracker` in
`configs/provisional_defaults.yaml` (`tracker_is_source_of_record: false`).

---

### Q9 — Upstream commit pins  **RESOLVED 2026-09-12**

**What.** The exact git commit of the two upstream repositories you adapt code from.

**Why it matters.** CLAUDE.md §7 and AI_RULES.md §7 require every adapted block to
carry `# Adapted from: <repo> @ <commit>, <license>`. "Version 0.5.2" is not a pin —
it does not identify a tree. Reviewers of an artifact check this.

**The problem, and how it was solved.** Neither local folder is a git repository, so the
hash could not be read off disk. circuit-tracer was recovered exactly (a tag ref);
sae-pruning-paper was recovered by *inference* from the fork's contents, and is recorded
as inferred. Neither was guessed (AI_RULES.md 2.2). Full derivation in §3.3.

**How, exactly.** Re-clone each from source and read the hash:

```bash
git clone https://github.com/safety-research/circuit-tracer /tmp/ct
cd /tmp/ct && git log -1 --format=%H
# if you need the exact 0.5.2 tree:  git rev-list -n 1 v0.5.2

git clone https://github.com/hecboar/sae-pruning-paper /tmp/sp
cd /tmp/sp && git log -1 --format=%H
```

⚠️ **One thing to check for circuit-tracer.** Its own README is internally
inconsistent about its home: the install and demo links point at
`github.com/safety-research/circuit-tracer` (9 places), while the BibTeX block cites
`github.com/decoderesearch/circuit-tracer` (1 place). Both are quoted verbatim from
the local fork; neither was chosen for you. Open both URLs, see which is canonical
(one probably redirects to the other after a rename), and record the canonical one
with its hash. Your paper's citation should match whatever the repo says to cite.

| Repo | Canonical URL | Commit | Licence |
|---|---|---|---|
| circuit-tracer | `github.com/decoderesearch/circuit-tracer` | `8f1e2438df612464e229e44c4a00ff637bf9379b` (tag `v0.5.2`) | MIT-style, verified |
| sae-pruning-paper | `github.com/hecboar/sae-pruning-paper` | `261191804675e2d39d0a265320dbc0bc85afd30a` ⚠️ inferred | MIT, verified |

The README ambiguity above is settled: `safety-research` returns **HTTP 301** to
`decoderesearch`, so the BibTeX was right and the install links are stale. Cite
`decoderesearch`.

**Status: RESOLVED.** Licence texts were already verified and copied verbatim into
`THIRD_PARTY_LICENSES/`. One residual check remains on the sae-pruning pin — see §3.3;
it needs the machine that holds the fork.

---

### Q6 — Task prompt datasets  **PRE-REGISTERED 2026-09-12** (docstring still open)

**What.** The actual prompt sets for IOI, greater-than and docstring, plus how many
prompts each uses.

**Why it matters.** The prompts *are* the input distribution — they determine which
circuit gets extracted at all. And the counts set your statistical power: more prompts
narrow every CI but cost attribution passes, which is your dominant compute cost.

**Where each comes from.**

| Task | Source | How to get it |
|---|---|---|
| IOI | Wang et al., ICLR 2023 (arXiv:2211.00593) | **PINNED**: ACDC's seeded edit of Easy-Transformer's `ioi_dataset.py` (⚠️ CORRECTED 2026-09-14: this said the `transformer_lens` IOI generator, whose dataset has only **2** templates), **30 templates** (15 `BABA` + 15 `ABBA`, the latter being `BABA[:]` reordered — counted from the reference `ioi_dataset.py` on 2026-09-12). The old `n_templates: 400` was a placeholder, not a count from any release. |
| greater-than | Conmy et al., NeurIPS 2023 (ACDC) | ⚠️ **CORRECTED 2026-09-12.** This row used to say "day-of-month construction: `\"The {day} of {month} is\"` → next-token day". That is **not the published task.** ACDC's `acdc/greaterthan/utils.py` builds `"The {noun} lasted from the year {year1} to "` — a **year-span** completion scored over two-digit suffixes `yearend+1..99`, over **120 nouns** (⚠️ CORRECTED 2026-09-14: this said 26 — a count of *lines* in the upstream list, not of nouns). Read from source, not memory. |
| docstring | MIB benchmark (Mueller et al., ICML 2025) as the harness | Use the MIB harness if it covers docstring; otherwise the original docstring-circuit release. |

All three are **synthetic template text** — no PII, no licence problem
(AI_RULES.md §5). Record each source and count in `data/README.md` (the table is
already there and empty).

**Recommendation on counts.** Start at **n_prompts = 200–500** per task, not the
placeholder 1000. Attribution cost is linear in prompts and you have two months. Note
that prompts generated from the same template are correlated, so your bootstrap should
resample **templates**, not individual prompts — decide that when you fix the counts,
and write it down (AI_RULES.md 4.2 requires the bootstrap scheme in advance).

> **PI fill-in**
> ```
> ioi           source = ______________________  n_templates = ____  n_prompts = ____
> greater_than  source = ______________________  n_prompts = ____
> docstring     source = ______________________  n_prompts = ____
> bootstrap resampling unit = [ template / prompt ]
> ```

### Pre-registered 2026-09-12

| Task | Source | n_prompts | n_templates | Confirmed |
|---|---|---|---|---|
| `ioi` | ACDC / Easy-Transformer `ioi_dataset.py` (Wang et al.) | 300 | 30 | ✅ |
| `greater_than` | ACDC `get_year_data` (year spans) | 300 | 120 nouns (112 single-token under Pythia) | ✅ |
| `docstring` | ACDC `acdc/docstring/prompts.py` (MIB has no docstring task — verified 2026-09-14) | 300 | `null` — 2 styles; resampled by prompt within style | ❌ generator not yet ported |

`bootstrap_resampling_unit = template` for all three. Prompts from one template are
correlated, so resampling prompts would treat correlated draws as independent and return
CIs that are too narrow. **The template count is therefore the effective sample size for
every prompt-level CI** — which is why the IOI correction from "400" to the real 30
matters rather than being cosmetic.

> This is the PROMPT bootstrap only. The **CSI bootstrap axes** (over B, S, r) are still
> open — see "Also open, not numbered" in Part 1. Run_Plan flags that the two interact;
> settle the CSI axes before Stage C.

**Status: PRE-REGISTERED for `ioi` and `greater_than`.** `docstring`'s source is verified (2026-09-14: MIB has no docstring task, so it is ACDC's release) and its resampling unit is decided (prompt within style, because the generator has only two styles), but its generator is not yet ported or checked under Pythia's tokenizer, so it stays unconfirmed. Docstring is the third task and does not block the Pythia-160M pilot.

---

### Q7 — Calibration corpus + perplexity protocol  **PRE-REGISTERED 2026-09-12**

**What.** The text corpus GPTQ and AWQ use to estimate activation statistics — and its
licence.

**Why it matters twice over.** (1) Licensing: AI_RULES.md §5 requires a licence
permitting research use, recorded in `data/README.md` before use. (2) **Science**: the
calibration corpus shapes the quantisation error, which shapes D, which shapes CSI. It
is not merely a legal checkbox. And the matched-perplexity control (null b) must be
evaluated on the **same** corpus as the setting it is matched to, or the "match" is
meaningless.

**Options.**

| Corpus | Licence | Note |
|---|---|---|
| **WikiText-2** | CC BY-SA 3.0 | Small, standard, trivial to download, universally used for LM perplexity. |
| C4 (subset) | ODC-BY | What GPTQ/AWQ papers typically use; large. |
| The Pile (subset) | mixed per-component | Licence is per-subset — more homework. |

**Recommendation: WikiText-2** for both calibration and the perplexity evaluation, on
the grounds that it is small, unambiguous, and lets you state one corpus for the whole
paper. If a reviewer expects C4 because that is what GPTQ used, switching later is a
config change — but it invalidates the affected cells, so decide now.

Also fix here, because the matched-perplexity null depends on all of it: evaluation
sequence length, stride, and whether the final partial window counts. The
sae-pruning-paper fork's `docs/PROTOCOL.md` is your reference — its revision notes
exist *because* an earlier perplexity protocol was wrong, so read it before inventing
your own.

**The reference protocol is already documented — copy it, do not invent one.**
`sae-pruning-paper-main/docs/PROTOCOL.md` specifies exactly what produced the published
numbers you will compare against in C5:

| Parameter | Value |
|---|---|
| Corpus | WikiText-2 raw **test** split (~289K tokens) |
| Window | 1024 |
| Stride | 512 |
| BOS policy | `<bos>` prepended to **every** window |
| dtype | bfloat16 |
| Attention | `eager` for Gemma-2 (attention-logit softcapping) |
| Reported quantity | `Δlog PPL = log PPL_pruned − log PPL_dense` |

That document exists because the *first* protocol was wrong: it prepended `<bos>` once
to the whole corpus and evaluated bf16 models in fp16. **Gemma-2-2B's dense perplexity
came out as 410 instead of ~11.** Llama looked merely plausible, so the fault was
invisible without a forensic sweep. `src/science/perplexity.py::validate_ppl_protocol`
now warns by name if you deviate.

Strong recommendation: adopt it verbatim, and wrap
`saediag/ppl.py::windowed_ppl` rather than writing your own, so your numbers come off
the same code path as theirs.

> **PI fill-in**
> ```
> calibration_dataset = ______________________  licence = ______________
> n_calibration_samples = ______  seq_len = ______
> perplexity protocol = [ reference verbatim / deviation: ______________ ]
> ```

### Pre-registered 2026-09-12 — and the recommendation in this file was wrong

**Two corpora, not one.** This file previously recommended WikiText-2 "for both
calibration and the perplexity evaluation". Those are two different corpora in the
reference work, and conflating them breaks the C5 cross-audit: calibrate Wanda on
WikiText-2 while ref [1] calibrated on FineWeb-Edu and our Wanda-pruned model is not
their Wanda-pruned model, so C5 would correlate circuit damage from one intervention
against feature damage from another.

| | Value | Source |
|---|---|---|
| Calibration corpus | **FineWeb-Edu**, ODC-BY | `reprune.py::calib_cache_path` |
| Calibration tokens | **300,000** | same, default argument |
| Calibration **seed** | **7** | same — "deterministic given the same calibration token cache (seed 7)" |
| Calibration dtype | bfloat16 | `reprune.py` — Gemma activations overflow in fp16 |
| Perplexity corpus | **WikiText-2 raw test** (~289K tokens), CC BY-SA 3.0 | `docs/PROTOCOL.md` |
| Window / stride | 1024 / 512 | same |
| BOS | prepended to **every** window | same |
| dtype / attention | bfloat16 / `eager` for Gemma-2 | same |
| Reported | `Δlog PPL = log PPL_pruned − log PPL_dense` | same |

All read from the pinned fork on 2026-09-12, not quoted from prose.

**The seed is part of the data, not the run config.** Run_Plan records the corpus and the
token count but not the seed. Their pruning is deterministic *given the cache*, and the
cache is built with seed 7 — a different seed gives a different cache, a different Wanda
mask and a different pruned model, with nothing in any output to indicate it.

**Wrap `saediag/ppl.py::windowed_ppl`; do not reimplement.** `PROTOCOL.md` exists because
the original protocol prepended `<bos>` once to the whole corpus and evaluated bf16
models in fp16. Their forensic ladder shows restoring per-window `<bos>` **alone** moves
Gemma-2 from 192 to 12 while moving Llama by +0.31 — Llama is the negative control that
kept the fault invisible for a submission cycle. Coming off the same code path as the
numbers we compare against is the whole point.

**Status: PRE-REGISTERED.** Recorded in `configs/calibration/final.yaml` and
`data/README.md`. One value deliberately left `null`: the calibration **context length**,
which the fork reads from the cached array's shape rather than declaring. Its SAE
activations use context 256, but that is the SAE cache, not necessarily the pruning
calibration cache — confirm when the cache is built. Downloading either corpus still
requires `mode.allow_external_dataset_download`.

---

---

## The five decisions of 2026-09-12 — what was pre-registered, and the one correction

Q2, Q1, Q4, Q10 and Q11 were decided together on 2026-09-12 on the recommendations in
`docs/Run_Plan.md` Part 2. Four were adopted as recommended. **One had to be corrected
before it could be implemented**, and the correction is the important part of this entry.

| | Pre-registered value | Where it lives |
|---|---|---|
| **Q2** distance | primary **L1**, ablation **Jensen–Shannon**, also-report **normalised L1** | `configs/distance/provisional_l1_js.yaml` |
| **Q1** bands | core **s(e) = 1**, contingent **0.5 ≤ s < 1**, noise **s < 0.5** | `configs/ensemble/decompose/final.yaml` |
| **Q4** grid | **anti-diagonal**, node 0.6→0.9 while edge 0.99→0.95, B configs from B alone | `configs/ensemble/default.yaml` |
| **Q10** coarse level | **layer**, with a pre-registered falsification condition | `configs/comparison/final.yaml` |
| **Q11** positions | **aggregate** over token positions | `configs/comparison/final.yaml` |

### The correction: Run_Plan's Q4 grid was not achievable

Run_Plan Part 2 recommends "a crossed 4×4 grid, node ∈ {0.6, 0.7, 0.8, 0.9} × edge ∈
{0.99, 0.98, 0.97, 0.95}, anti-correlated, giving B = 16". **Those requirements are
mutually exclusive**, and the check is worth keeping written down because it is not
obvious:

- The full 4×4 Cartesian product is B = 16, and **84 of its 120 pairs are nested** — 70%.
  It contains CIRCUS's own counter-example verbatim: (0.6,0.95) ⊂ (0.8,0.98) ⊂ (0.9,0.99).
- The anti-correlated *pairing* of those same four levels is non-nested but gives **B = 4**.
- Order threshold configs by dominance and a k×k product is a grid poset, whose largest
  antichain has exactly **k** members. So **B = 16 mutually non-nested configs is
  impossible with 4 levels per axis**, at any choice of values.

The resolution keeps everything Run_Plan actually wanted: **B distinct levels per axis,
walked in opposite directions and paired index-by-index.** Non-nested by construction
rather than by rejection sampling — provable, not merely tested — reproducible from B
alone with no seed, and sweeping the same literature-anchored box (circuit-tracer's CLI
defaults 0.8/0.98 sit inside it; CIRCUS's published examples (0.6,0.99) and (0.9,0.95)
are its corners).

`tests/test_config_integrity.py::test_a_crossed_product_would_be_nested` pins the
reasoning so nobody "simplifies" the grid back into a product.

### Two silent failures this surfaced

**1. A nested grid in the test suite.**
`test_integration_scientific_run_blocked.py` built its "fully resolved scientific run"
fixture as `node 0.8 - 0.05i`, `edge 0.98 - 0.01i` — both axes descending together, so
every config dominated the next. A fully nested chain, standing in for the configuration
this project treats as correct. It passed for five weeks because nothing checked.
`run_stage_a` now refuses any nested grid before extracting anything.

**2. `noise_strict` was not reaching `decompose()`.**
Wiring the config through revealed that both `run_stage_a` and `run_stage_c` called
`decompose()` positionally and dropped the flag, so every edge at exactly s = 0.5 was
banded **noise** while the config said CIRCUS's taxonomy (contingent). Caught by an
integration test that reads the cutoffs from the config instead of restating them.

### The coarse level is a choice *and* a falsification condition (Q10)

The coarse level is `layer`, but arXiv:2607.18921's own numbers say a coarsening that
does not cut along a functional boundary buys nothing: its *semantic* level scored
Jaccard@10 0.163 against structural's 0.163 — identical to three decimals — while only
routing-head separated (0.666). "Group by layer" sits on the indistinguishable side of
that ledger, so the choice is pre-registered **together with the condition that would
falsify it**:

> If the coarse-level Jaccard lies within the bootstrap CI of the exact-edge Jaccard, the
> second level is declared VACUOUS for pipeline A and reported as a negative
> methodological result — not presented as a second level that happens to agree.

Implemented in `src/science/two_level.py::coarse_level_verdict`. If the levels do
separate this costs nothing; if they do not, it is a publishable finding in its own
right — the two-level reporting practice arXiv:2607.18921 recommends cannot be
instantiated in the dominant attribution-graph tool.

This also closed a live wiring bug: `two_level.py` carried its own `^L\d+\.H\d+$` regex
independent of `node_ids.project_level2`, so choosing a scheme in the config would not
have rewired the module and C2 would have reported zeros forever. Both now go through
`project_level2` — one definition of the coarse level, in one place.

### Still open after this

- **Q3** (B, S, R) — B is settled at the grid size, but S and R need the measured Pythia
  wall time. Run_Plan Step 5.
- **Q6, Q7** — the datasets.
- **Normalised L1 is pre-registered but not implemented**: it needs
  `src/science/distances.py`, which is novelty-zone and was not part of the 2026-09-12
  approval (which covered `decompose.py`, `threshold_grid.py`, `two_level.py`).
  `also_report_implemented: false` records this.
- The Pythia **dtype** question raised by Q5.

## The statistical pre-registrations of 2026-09-12

Four items that were "open, not numbered" are now fixed, plus Q8. All five were decided
by the PI; two required novelty-zone approval (`csi.py`, `distances.py`), granted the
same day.

| Item | Decision | Lives in |
|---|---|---|
| CSI bootstrap axes | **B, S, r** | `csi.py::csi_over_ensemble`, `stage_c.bootstrap_axes` |
| Chance-floor universe N | **observed node union**, U·(U−1) | `run_stage_c` |
| FDR level q | **0.05** | `configs/config.yaml` `stage_d.fdr_q` |
| Normalised L1 | **reported as a third column** | `distances.py`, `d_normalized` in the CSI table |
| Q8 tracker | **W&B**, `run_meta.json` stays source of record | `provisional_defaults.yaml` |

### Why the bootstrap change is the one that matters

`csi()` resampled only the null draws, so its interval carried uncertainty in the CSI
*denominator* and nothing else. But the B threshold views and the S seeds are sampled
too: a different draw of either gives a different s(e), hence a different D(c), hence a
different CSI. An interval that ignores them is anti-conservative — it will exclude 1
more often than it should, which is precisely the direction that manufactures findings.

On a deliberately heterogeneous test ensemble the CI width went from **1.10 to 3.14**
with the point estimate unchanged. Some cells that would have read as significant under
the old interval will not under the new one. **That is the correct direction, and fixing
it before any real result exists is the entire purpose of pre-registration.**

### Three silent failures this work surfaced

**1. Stage C never used the Q10 scheme.** It still called `project_to_routing_heads`,
the literal `L{l}.H{h}` projection. The Q10 pre-registration wired `two_level.py` but not
the call site, so on a real pipeline-A circuit Stage C would have raised — or, with
`strict=False`, silently reported a routing-head overlap of 1.0 from two empty sets. It
now reads `comparison.level2_scheme` from config.

**2. Empty ensemble cells vanished.** A `(config, seed)` cell whose extraction returned
no edges emits no rows, so it was simply absent from the per-cell map — shrinking the
bootstrap denominator and inflating every s(e). An empty view is a real observation
("this threshold configuration found nothing"), not a missing one. Stage C now passes the
full B×S grid explicitly and such cells count as empty sets.

**3. `stage_d.fdr_q` was read twice with independent literal fallbacks**, so a config
setting one could in principle have been applied to the correction but not to the
reported value.

### A guard worth knowing about

`validate_csi_row` requires `ci_lo <= csi <= ci_hi`. A percentile bootstrap does **not**
guarantee the point estimate lies inside its own interval — it is rare, but possible with
a skewed bootstrap distribution at small R. If that happens, the row is refused rather
than written. That is the right behaviour (refuse rather than report something
incoherent), but it will surface as a hard failure at Stage C rather than a warning, so
it is worth recognising if it appears.

---

## Step 5 engineering, 2026-09-14 — decisions, findings, corrections

Step 5 (the first real Stage A on Pythia-160M) could not run even with a GPU: the real-model
loader, `DenseNodeExtractor` (Pythia's only pipeline) and every real task loader were stubs.
They are now built and tested. Nothing measured on the way is evidence.

### Decisions

| Decision | Chosen | Where |
|---|---|---|
| Dense-node attribution method | **Edge attribution patching** (arXiv:2310.10348) — score-then-threshold, and linear like circuit-tracer's, so C8 compares bases rather than methods | `src/extraction/eap.py` |
| Head-input representation (proposal §4 rule 5) | **Split Q / K / V** inputs, as the EAP reference code scores them | `eap.py` |
| Aggregation over prompt pairs | **\|sum\|** — the reference code sums, *then* takes the absolute value, once, after all batches | `eap.py` |
| Node scores for `node_threshold` | Attribution patching at each component's output (the reference node mode covers heads; extending it to MLPs and the embedding is labelled as an extension) | `eap.py` |
| How the Q4 thresholds apply | **Directly to total-effect scores**, with circuit-tracer's `find_threshold` and dangling-node cleanup; no influence propagation, because EAP scores already include every downstream path | `src/extraction/dense_prune.py` |
| Pythia precision | **float32** for attribution and weights (Pythia is fp16-trained, so the bf16 rationale never applied; the Q7 perplexity protocol stays bf16) | `configs/model/pythia*.yaml` |
| BOS on task prompts | **None**, matching the ACDC/EAP setups — decided on principle, since a dense-model check showed the two tasks moving in opposite directions | `configs/task/*.yaml` |
| Docstring | Source = ACDC (MIB has none); resampling unit = **prompt within style** | `configs/task/docstring.yaml` |
| RUN MODEL DOWNLOAD | **Approved** for Pythia-160M on this machine (RTX 4060, 8 GB) | — |
| Q4 grid after the nesting finding | **Kept as pre-registered for every pipeline; both findings reported** | `docs/preregistration/` |
| **Q3 — B, S, R** | **B = 16, S = 5, R = 20.** Run_Plan's rule, stated before any timing existed, applied to the measured pass (IOI 42–48 s, greater-than 7.5–7.9 s, both under the 2-minute line). Applied on the PI's instruction to settle the remaining decisions now. Re-apply the same rule to measured primary-model timings before their runs. | `configs/ensemble/default.yaml`, `configs/nulls/default.yaml` |
| **Chance-floor universe N** | **Structurally possible edges among observed nodes.** The earlier U·(U−1) over node strings counts pairs that can never be edges (inputs never send; nothing flows backwards): 8.7× too many on IOI, 6.3× on greater-than, lowering the random-overlap floor ~10× (0.178 → 0.018 for two 827-edge circuits). Pipeline A needs its own structural rule before it runs. | Stage C; not yet implemented |

### Hazards found and closed

- **TransformerLens silently ignores the revision pin.** It has no `revision` argument, so weights
  are loaded through `transformers` at the pinned sha and handed over as `hf_model` (verified
  bit-identical).
- **TransformerLens builds the architecture from `main`, by name.** `convert_hf_model_config` calls
  `AutoConfig.from_pretrained` without a revision. The loader now hard-asserts the result against
  the pinned HF config and the VERIFIED YAML numbers.
- **The download gate was prose, not code.** `mode.allow_model_download` was read by no Python. It
  is enforced in `src/extraction/real_model.py`.
- **GPT-2 tokenizer assumptions break under Pythia:** 88 of 99 IOI names are single tokens, 112 of
  120 nouns, 456 years qualify (618 under GPT-2), and `"01"` is token 520, not ACDC's hard-coded
  486. All are now computed from the tokenizer of the model under test.
- **Parallel residual has no same-layer attention → MLP edge**, yet EAP's gradient-at-endpoint
  trick would give it a nonzero score. It is excluded structurally.
- **The S seed axis was undefined.** EAP is deterministic given its prompts, so a seed now draws the
  prompt batch and its corruptions — "resampling variance" in arXiv:2606.16920.

### Verified

- **Corruptions remove the task signal:** IOI logit difference +4.56 clean → −0.27 corrupted;
  greater-than probability difference +0.763 → −0.640.
- **EAP is correct to first order:** every edge type in both parallel and sequential blocks matches a
  finite difference of an exact additive edge patch (float64, `tests/test_eap_dense_node.py`).
- **Candidate edges for Pythia-160M: 32,347**, equal to a hand count.

### The grid-nesting finding

On dense-node scores the Q4 grid yields largely nested views (82–116 of 120 view pairs nested;
consensus 90–100% of the smallest view), though the config's own stop rule is not triggered. A sweep
of edge ranges was pre-registered and committed **before** evaluation (`d7c2e58`), run exactly as
stated (`9798bf9`), and selected nothing. The PI kept the grid and requires both findings to be
reported. Full record: `docs/preregistration/2026-09-14_dense_node_grid_*`.

### Corrections to this document (both errors were the agent's)

- **Greater-than has 120 nouns, not 26.** The 26 was a count of lines, not nouns; it understated the
  resampling units about 4.6×. The lists are now extracted by script and length-asserted.
- **The IOI source was not TransformerLens**, whose dataset has 2 templates. The 30 templates are
  Easy-Transformer's, via ACDC's seeded edit; both licences now ship.

### Still open

- **Stage B and Stage C for real models** are not built (perturbation and compression of a
  TransformerLens model).
- **The chance-floor N rule** is decided but not implemented.
- **Grouped-query attention** is refused by the EAP code; both primaries are expected to need it.
- **Pythia-410M** has not been downloaded or timed.
- **Validity caveat to report:** EAP's linear approximation is poorly calibrated even where its
  ranking works (R² = 0.27 against activation patching, arXiv:2310.10348 §5.1), and its error grows
  with downstream non-linearity (arXiv:2606.09899), which compression changes.
- **ABC corruption property:** the upstream S/S1 draws have no distinctness check, so about 2% of IOI
  corruptions change 2 tokens instead of 3 (6/300 at seed 0). Kept as published.
- Bottlenecks and a hardware plan: [`BOTTLENECKS_AND_HARDWARE.md`](BOTTLENECKS_AND_HARDWARE.md).

## Part 3 — Reference

### 3.1 What is blocked on what

```
Q5 pins ─┐
Q6 tasks ─┼──► Stage A real extraction ──► Phase-1 exit gate ──┐
Q3 B/S/R ─┤                                                     │
Q1 bands ─┘                                                     │
                                                                ▼
Q2 pre-registered ──────────────────────────────────► STAGE B FREEZE  ◄── the pre-registration point
                                                                │
Q7 calibration ─────────────────────────────────────► Stage C real compression
                                                                │
INT-flag rule ──────────────────────────────────────► Stage D causal + cross-audit
                                                                │
Q9 hashes + artifact licence ───────────────────────► Artifact packaging
```

### 3.2 Provisional values currently in force

Every one is an **engineering placeholder**, approved 2026-08-07 for dry-runs and
synthetic tests only. `final_pre_registration: false` on all of them. Machine-readable
copy: `configs/provisional_defaults.yaml`.

| ID | Provisional value | Binding constraint |
|---|---|---|
| Q1 | **PRE-REGISTERED 2026-09-12** — core s=1, noise s<0.5 (CIRCUS §3.2) | `configs/ensemble/decompose/final.yaml`, `noise_strict: true` |
| Q2 | **PRE-REGISTERED 2026-09-12** — primary L1, ablation JS, also-report normalised L1 | normalised L1 not yet implemented (needs `distances.py`, novelty zone) |
| Q3 | B=4, S=2, R=3 (synthetic) | B=16/S=5/R=20 pilot proposed, not approved |
| Q4 | **PRE-REGISTERED 2026-09-12** — anti-diagonal grid, node 0.6→0.9 × edge 0.99→0.95 | non-nested by construction; no seed axis |
| Q5 | **RESOLVED 2026-09-12** — all 8 revisions pinned (see Q5) | ⚠️ gated licences for Gemma-2 + Llama-3.2 still unaccepted |
| Q6 | **PRE-REGISTERED 2026-09-12** — ioi (300 prompts × 30 templates), greater_than (300 prompts × 120 nouns); no BOS | docstring source verified 2026-09-14, generator not yet ported; resampling units: template / noun / prompt within style |
| Q7 | **PRE-REGISTERED 2026-09-12** — calibration FineWeb-Edu 300K seed 7; PPL WikiText-2 test | download still gated by `mode.allow_external_dataset_download` |
| Q8 | local JSON `run_meta.json` | no network logging |
| Q9 | **RESOLVED 2026-09-12** — circuit-tracer `8f1e2438…` (tag v0.5.2), sae-pruning-paper `26119180…` (inferred; see §3.3) | recorded in §3.3 and in both adaptation headers |

### 3.3 Upstream repositories

**circuit-tracer** — local fork `../circuit-tracer-0.5.2/` (READ-ONLY)

| Field | Value |
|---|---|
| Canonical URL | `github.com/decoderesearch/circuit-tracer` — **RESOLVED 2026-09-12**. The README's install/demo links are stale: `github.com/safety-research/circuit-tracer` returns `HTTP 301 Moved Permanently` to the decoderesearch URL (verified by fetching it). The BibTeX was right. |
| Commit | `8f1e2438df612464e229e44c4a00ff637bf9379b` — **RESOLVED 2026-09-12**, read from the GitHub API: `git/ref/tags/v0.5.2` → this sha (lightweight tag, so the sha *is* the commit), 2026-07-18 |
| Licence | MIT-style, "Copyright (c) 2024 Michael Hanna and Mateusz Piotrowski" — verified, copied to `THIRD_PARTY_LICENSES/circuit-tracer-LICENSE.txt` |
| Files inspected | `README.md`, `circuit_tracer/__init__.py`, `attribution/attribute.py`, `graph.py`, `replacement_model/replacement_model.py`, `attribution/targets.py`, `frontend/*` |
| Adapted in our repo | `src/extraction/attribution_graph.py` (pipeline A adapter) |

Open adaptation questions: exact `prune_graph` / `PruneResult` / `find_threshold`
argument names (verify against the pinned commit at Stage A); whether the `nnsight`
backend is needed for Llama-3.2-1B (the fork's README notes the Llama demo is not
Colab-supported); **Pythia has no transcoder set, so pipeline A is unavailable there.**

**sae-pruning-paper** — local fork `../sae-pruning-paper-main/` (READ-ONLY)

| Field | Value |
|---|---|
| URL | `github.com/hecboar/sae-pruning-paper` (verified from `CITATION.cff`) |
| Paper | arXiv:2603.25325 |
| Commit | `261191804675e2d39d0a265320dbc0bc85afd30a` — **INFERRED, not read off a URL.** 2026-07-31, "Add revision materials and align repository with the revised manuscript". See the verification note below; ⚠️ one check still outstanding. |
| Licence | MIT, "Copyright (c) 2025-2026 Héctor Borobia" — verified, copied to `THIRD_PARTY_LICENSES/sae-pruning-paper-LICENSE.txt` |
| Files inspected | `README.md`, `revision/src/saediag/{__init__,io,models,ppl,stats,pruning,sae,matching,fragility,fixed_dict,ablation,reprune}.py` |
| Adapted in our repo | `src/compression/magnitude_prune.py` (pruning maths re-implemented on numpy for engineering; upstream not imported) |

Open adaptation questions: exact signatures of `prune_magnitude_inplace`,
`prune_wanda_style_inplace`, `collect_linear_input_second_moment_from_cache`; the
calibration-cache format in `reprune.py` and whether to reuse it; **the fork's
corrected perplexity protocol (`docs/PROTOCOL.md`) is your reference for both the
matched-perplexity control and the perplexity eval** — read it before Q7.

> **How the sae-pruning pin was established (2026-09-12), and what is still unverified.**
> The local fork is not a git repo, so the commit was inferred from which files it contains.
> Three facts were confirmed against the GitHub API this session:
>
> 1. `261191804675e2d39d0a265320dbc0bc85afd30a` exists, dated 2026-07-31.
> 2. The next commit `c8cce94ff4fce08c0dee6bd0ab7391e67c2a4880` has `26119180…` as its
>    **only parent**, so nothing sits between them. It adds
>    `revision/scripts/verify_metric_ordering.py` — a file the fork lacks.
> 3. At the pin, `revision/scripts/e6_stat_freeze.py` still contains the word "rebuttal",
>    which the commit after that (`bd85878`, "Reword two docstrings that framed the analysis
>    as rebuttal material") removes.
>
> A copy that **lacks** the metric-ordering script and **contains** "rebuttal" can therefore
> only be at this commit. That inference rests on two claims about the local fork's contents
> which **cannot be checked from a machine that does not hold the fork**. Run the tree diff in
> `docs/Run_Plan.md` Step 1 on the machine with `../sae-pruning-paper-main/` and trust the
> diff over this note if they disagree.
>
> Checked while here: `c8cce94` also modifies `results/E6/stat_tests.csv`, the file C5's
> target numbers come from. The diff is **two appended rows** about Proposition 1 metric
> ordering; every fragility and Spearman value Run_Plan §0.2 quotes is byte-identical at both
> commits. The pin choice does not move any C5 number.

**Model / transcoder checkpoints** — all 8 pinned 2026-09-12, table in Q5 (Part 2).
⚠️ The Gemma-2 and Llama-3.2 gated licences are still unaccepted, which blocks downloads
but not the pins.

### 3.4 Commands you will actually use

```bash
cd "D:\Users\SUPRATIK\AAAI-UC\AAAI - Standalone Project\Circuits_Under_Compression"

python -m pytest                                  # full suite: 510 passed, 1 skipped
python -m pytest -k "not integration and not regression"   # fast unit tier
python -m pytest -k "integration"                 # integration tier
python -m experiments.dry_run_stage_a 0           # synthetic Stage A, no model, no GPU

# real runs (blocked until the decisions above are made):
python -m experiments.run_stage_a mode=scientific_run model=pythia160m task=ioi
```

Note the `-m`. `python experiments/dry_run_stage_a.py` fails with
`ModuleNotFoundError: experiments`.

### 3.5 The rules that constrain all of this

- **AI_RULES.md 1.2** — a config referenced by a completed run may never be edited.
  Changes mean a *new file with a new name*. That is why there are `_provisional`
  variants rather than edits.
- **AI_RULES.md 1.3** — change anything upstream (extraction, distance, compression)
  and every downstream number is invalidated and recomputed from Stage A. No patching
  of result tables.
- **AI_RULES.md 1.4** — after the freeze, `frozen/` is append-only. If you find a bug
  in a null *after* the freeze, you re-run from Stage B, **disclose it in the paper**,
  and leave the old frozen values in the repo for the record.
- **AI_RULES.md 4.5** — CSI ≈ 1 everywhere is pre-committed as a publishable result.
  If that is what you get, that is your paper. Do not go looking for an analysis choice
  that makes an effect appear.
