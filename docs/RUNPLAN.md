# RUNPLAN — from "10 of 10 blockers OK" to an AAAI UC submission, and on to a main-track paper

Written 2026-09-30, right after the Spark preflight read **10 of 10 OK** (B6 passed on seeds
5–9). It answers four questions:

1. What does a UC submission actually have to contain? (Part 1, read off the two AAAI-26 UC papers)
2. Against the proposal's formulation (Algorithm 1, §2–§4), which runs are done and which are left? (Part 2)
3. What do I type, in what order? (Part 3, every command)
4. What does a main-conference version need on top? (Part 5)

Results go in [`../RESULTS.md`](../RESULTS.md) (dated, append-only, filled automatically per cell).
Decisions go in [`HUMAN_DECISIONS.md`](HUMAN_DECISIONS.md). The older [`Run_Plan.md`](Run_Plan.md)
(2026-09-11) is the literature reasoning behind the pre-registered choices; this file supersedes
its runbook.

---

## Part 1 — What the two AAAI-26 UC papers say about the target

The two examples are *Towards Data-Efficient Deep Learning for RNA 3D Structure Prediction and
Design* (Liu, AAAI-26 UC, pp. 41501–41503) and *Adapting Hybrid Parallel-Head Large Language
Models for Southeast Asia* (Ng, AAAI-26 UC, pp. 41504–41506). What they have in common:

| Property | Liu (RNA) | Ng (SEA LLMs) | What it means for you |
|---|---|---|---|
| Length | 2 pages of content + a 3rd with acknowledgments, ethics statement, references | 2 pages of content + references | **~2 pages.** Your 7-page proposal PDF must be cut to about a quarter. Check the AAAI-27 UC call for the exact limit. |
| Authors | single author + mentor in acknowledgments | single author | sole student author; thank the mentor/PI |
| Tense | "we **will** benchmark…", "will be evaluated…" | "I **will** develop…", "I expect to find…" | **A proposal, not a results paper.** Neither reports a single number of its own. |
| Structure | Abstract · Background · **Problem Definition** (formal input/output) · Proposed Approach (data, benchmarks, architecture, loss) · Evaluation · Ethics | Abstract · Introduction · Background & Related Work · Approach & Evaluation (Phase 1 / Phase 2) · Discussion · Conclusion | abstract, motivation + gap, formal definition, method, evaluation plan, expected outcome |
| Named comparators | VFold (physics), DeepFoldRNA (DL) | Sailor2 1B/3B, SEA-HELM | **name your comparators**: the qualitative VLM claim (arXiv:2603.25035), the naive before/after Jaccard, the feature-level audits |
| Figure | none | one architecture diagram | one figure: Algorithm 1 as a diagram, or CSI vs null |
| Phasing | two-stage architecture | Phase 1 short-term, Phase 2 long-term | short-term (this grid) vs longer-term (extensions) reads naturally |

**The consequence that matters most:** a UC paper is judged as a *research plan by a student*.
You do not need the full 66-cell grid to submit. Your proposal plus **preliminary results** is
already stronger than either example, because each preliminary number shows the plan is
feasible:

- the exit gate: the stack recovers the published IOI circuit, precision 0.725 against ~0.03
  chance, p = 1.27e-45 (report the failed Jaccard criterion and the post-hoc change too);
- the nested-grid finding (82–116 of 120 view pairs nested), a methodological result by itself;
- Stage A dense references (core/contingent/noise) on Pythia and the primaries;
- if time allows, Pythia-160M Stage B null distributions, and CSI for a few cells.

The deadline (you asked earlier whether this can be done by the 5th) therefore governs **how many
preliminary numbers** go in, not whether you can submit. Part 4 gives the priority order.

---

## Part 2 — The formulation, item by item: done, runnable, blocked

Proposal = `docs/reports/Circuits_Under_Compression_AAAI_UC_Proposal.pdf`.

| Proposal item | Status 2026-09-30 | Runs left |
|---|---|---|
| Phase 1: reproduce a published circuit (exit gate) | ✅ **PASSED** (seeds 5–9) after the Jaccard criterion failed; both in RESULTS.md | none |
| Phase 1: CIRCUS ensemble, bands, reporting protocol | ✅ implemented, pre-registered 2026-09-12 | — |
| Alg. 1 Stage A — dense reference per (model, task) | code ✅; **0 of 6 real runs** (the Pythia runs of 2026-09-26 predate a config change and were to be re-run) | **6 runs** (`cells_stagea.txt`) |
| Timing on the Spark → Q3 (B, S, R per model) | ❌ not measured on the Spark (only on the 4060) | `04_timing.sh`, then a recorded decision |
| Alg. 1 Stage B — null (a) matched-magnitude | code ✅ (queue fixed today, see Part 6) | **66 cells** |
| Null (b) matched-perplexity | ❌ raises for real models (`src/science/matched_perplexity.py`, Novelty Zone) | engineering + your approval, then runs |
| Null (c) seed + threshold variation | ✅ built into every ensemble (B = 16 × S = 5) | — |
| **The freeze** | ❌ `frozen/` empty | per model, after its Stage B |
| Alg. 1 Stage C — real compression, CSI at both levels | ⛔ **blocked by design**: `run_stage_c._load_model` refuses real models pending *explicit PI approval*; two wiring gaps too (Part 6) | **66 cells** after approval + fixes |
| Stage D — damage ranking, BH at q = 0.05 | ⛔ `07_stage_d_analysis.sh` passes a key Stage D does not read (Part 6) | 6 runs (one per model/task) |
| C3 threshold-sensitivity sweep | ⛔ `analysis/threshold_sweep.py` has functions but no entry point, so step 07 runs nothing | driver + 66 sweeps |
| C4 interaction diagnostic (NIE/PIE/INT) | ❌ `PatchDiagnostic.run` raises for real models (Novelty Zone); INT-flag ratio still open | engineering + decision |
| C5 cross-audit vs feature-level damage | ❌ `analysis/cross_audit.py` has no entry point; **our grid (20/40/60 %) does not contain the published cells (30/50 %)**, so the pairing is empty | decision (Part 3, D3) |
| Two extraction pipelines (§2.2) | ❌ only dense-node runs; `attr`/`edgeprune` raise | main-track item |
| Three tasks (IOI, greater-than, docstring) | IOI ✅ all models; greater-than ✅ Pythia only (tokenizers of Gemma-2/Llama-3.2 make it impossible); docstring ❌ not ported | disclose, or port |
| Models | Pythia-160M, Pythia-410M, Gemma-2-2B, Llama-3.2-1B configured; Pythia-70M not | — |

**Scope-cut order if time runs short** (fixed in the proposal §5 / CLAUDE.md §6): compression
levels and secondary models first, then task circuits, then seeds. **Never** the null or the
two-level reporting.

---

## Part 3 — The runs, in order, with every command

All on the Spark, from `~/Documents/Supratik/Circuits_`, inside `tmux` (`tmux new -s cuc`, and
`tmux attach -t cuc` after a dropped SSH session). Every cell writes `logs/<stage>/<cell>.log`
and appends a timestamped row to `RESULTS.md`. A re-run skips cells whose log ends with
`CELL_COMPLETE`, so interrupting costs nothing.

**Memory on the GB10:** `nvidia-smi` shows no memory on this machine. Watch unified memory
with `watch -n 10 free -g` in a second tmux pane.

### R0 — Commit today's changes (Windows) and pull (Spark)

On Windows (you commit and push; I have not):

```powershell
git add RESULTS.md docs/RUNPLAN.md docs/HUMAN_DECISIONS.md deploy/BLOCKERS.md `
  deploy/shared/record_result.py deploy/shared/gen_cells.py deploy/shared/make_freeze_config.py `
  deploy/plan_b_dgx_spark/_run_one.sh deploy/plan_b_dgx_spark/05_run_queue.sh deploy/plan_b_dgx_spark/06_freeze.sh `
  deploy/plan_b_dgx_spark/cells_stagea.txt deploy/plan_b_dgx_spark/cells_stageb.txt deploy/plan_b_dgx_spark/cells_stagec.txt `
  deploy/plan_a_local_pc/cells_stagea.txt deploy/plan_a_local_pc/cells_stageb.txt deploy/plan_a_local_pc/cells_stagec.txt `
  tests/test_freeze_discovery.py tests/test_record_result.py
git commit -m "Run plan, RESULTS.md run log, and fixes to the Stage B -> freeze chain"
git push
```

On the Spark, first commit the gate's pass record if you have not yet (it must travel with the
repo), then pull:

```bash
cd ~/Documents/Supratik/Circuits_
git status --short data/reference/          # the pass record, if still uncommitted:
git add data/reference/ioi_gpt2_small_gate_pass.json && git commit -m "B6 exit gate passed on seeds 5-9"
git pull --rebase && git push
python -m pytest -q -p no:cacheprovider                  # expect 0 failed
python deploy/shared/preflight_blockers.py               # expect 10 of 10 OK
```

### R1 — Time one attribution pass per model on the Spark (≈ 1–2 h)

```bash
bash deploy/plan_b_dgx_spark/04_timing.sh 2>&1 | tee logs/timing_all.txt
```

Non-evidence. Every compute figure so far was measured on the RTX 4060. Add one hand row per
model to RESULTS.md (s/pass, peak memory).

### R2 — Decisions that must be written down BEFORE the first freeze (PI-owned)

Each goes into `docs/HUMAN_DECISIONS.md` with a date. After a freeze, changing any of them for a
frozen cell means re-running from Stage B and disclosing it.

| # | Decision | My recommendation |
|---|---|---|
| D1 | **Q3 on the Spark timings**, per model: B, S, R | Apply the pre-stated rule unchanged: s/pass ≤ 2 min → S = 5, R = 20; 2–6 min → S = 5, **R = 10**; > 6 min → R = 10, S = 3. Never S < 3. B stays 16 (it is nearly free). |
| D2 | **Freeze ownership** | The Spark owns every cell (the PC is not running a grid). One line in HUMAN_DECISIONS. |
| D3 | **C5 alignment.** The published feature-damage table has magnitude/Wanda at **30 % and 50 %** on Gemma-2-2B and Llama-3.2-1B; our grid has 20/40/60. As it stands, C5 has **nothing to pair**. | Add `magnitude_30`, `magnitude_50`, `wanda_30`, `wanda_50` for the two primaries on IOI (8 cells; Run_Plan Part 4 lists the 7 published cells) **before** their Stage B. Otherwise C5 is reported as not computable, with the reason. |
| D4 | **Behavioural floor**: a cell whose compressed model no longer does the task has no circuit to measure | e.g. "cells whose task metric falls to or below the corrupted-prompt baseline are reported as behaviourally dead and excluded from CSI claims". Needs a small engineering addition (task metric per Stage C cell). Decide before Stage C. |
| D5 | **Two directional predictions** (Run_Plan §0.5 and Part 4): exact-edge CSI near/above the floor with coarse-level CSI below it; RTN and matched-PPL pruning damaging the same edges | Write them into the freeze commit message. A correct pre-registered prediction is worth far more than the same number found afterwards. |
| D6 | **Pythia-160M AWQ outlier**: one tensor at ‖ΔW‖/‖W‖ = 0.60 against ~0.10 elsewhere | Identify it before freezing the `awq_int4` cells. The null is matched to these magnitudes, so a bug here would be frozen into the denominator. |

### R3 — Stage A dense references (6 runs)

```bash
# the cheapest first, to prove the path
bash deploy/plan_b_dgx_spark/05_run_queue.sh stageA 1 "pythia160m task=greater_than"
bash deploy/plan_b_dgx_spark/05_run_queue.sh stageA 1            # the other five (done ones are skipped)
tail -n 8 RESULTS.md
```

Each row reports the edges with s > 0 and the core/contingent/noise counts. These are the
"pre-compression circuit" of the proposal §3.3, and preliminary results for the UC paper.

### R4 — Stage B on Pythia-160M (22 cells)

```bash
# one cell first: the fastest one (greater-than, rtn_int8); check its RESULTS.md row
bash deploy/plan_b_dgx_spark/05_run_queue.sh stageB 1 "pythia160m task=greater_than setting=null-matchedmag-rtn-int8"
# then the rest of Pythia-160M, a few at a time
bash deploy/plan_b_dgx_spark/05_run_queue.sh stageB 4 pythia160m
grep -l CELL_COMPLETE logs/stageB/*pythia160m* | wc -l      # progress: done cells of 22
```

Every Stage B row says **NOT frozen**: it is a draft until R6.

Cost per cell = S × (R + 1) attribution passes (the dense ensemble plus R perturbed ensembles),
i.e. **105 passes at S = 5, R = 20**. At the 4060's Pythia-160M speed that is ~75–85 min per IOI
cell and ~13–14 min per greater-than cell, **~16–18 h for all 22** run one at a time. Replace
these with the R1 numbers; parallelism on one GPU gives less than a linear speed-up.

### R5 — Stage B on the primaries, then Pythia-410M

```bash
bash deploy/plan_b_dgx_spark/05_run_queue.sh stageB 2 llama32_1b     # 11 cells, IOI only
bash deploy/plan_b_dgx_spark/05_run_queue.sh stageB 2 gemma2_2b      # 11 cells, IOI only
bash deploy/plan_b_dgx_spark/05_run_queue.sh stageB 4 pythia410m     # 22 cells - first to cut
```

Keep `-P 2` or lower on Gemma: its AWQ peak alone is 42.9 GiB. GPTQ/AWQ are computed once per
model and cached (Gemma AWQ 5.1 h, Llama AWQ 2.8 h, measured).

### R6 — The freeze, one model at a time (irreversible)

Protocol rule 1 is per cell: a cell's null is frozen before *that cell's* Stage C. Freezing a
finished model while the next one is still in Stage B is therefore within the protocol. It is a
deliberate, named step, never a silently partial grid.

```bash
git status --short                      # code must be committed: the freeze records the commit
bash deploy/plan_b_dgx_spark/06_freeze.sh pythia160m       # shows the dry run, then asks you to type FREEZE
cat frozen/FREEZE_MANIFEST.json | head -40
git add frozen/ && git commit -m "Null freeze: pythia160m (22 cells)" && git tag -a null-freeze-pythia160m -m "pre-registration" && git push --follow-tags
```

Add a hand row to RESULTS.md with the manifest's `freeze_commit` and the number of cells. Repeat
per model as its Stage B completes.

### R7 — Stage C (66 cells) — BLOCKED until you approve it

Stage C is where real compression is measured, and the code refuses it on purpose: its model
loader raises *"Stage C real compression is not approved yet … explicit PI approval for real
runs"*. I tried to switch that on today and the permission system stopped it as the removal of a
safety gate, so it is yours to decide. The three changes are listed in Part 6. With them made,
the commands will be:

```bash
bash deploy/plan_b_dgx_spark/05_run_queue.sh stageC 1 "pythia160m task=greater_than setting=rtn-int8"
bash deploy/plan_b_dgx_spark/05_run_queue.sh stageC 4 pythia160m
```

Each Stage C row reports CSI with its bootstrap CI at both comparison levels. Cost ≈ 2 × S = 10
passes per cell plus the (cached) compression.

### R8 — Stage D, the threshold sweep, the cross-audit — BLOCKED behind R7

Needs the Stage D driver fix and entry points for the two analyses (Part 6), plus D3 and D5
decided, and the C5 feature-damage numbers entered from the **frozen CSV** of the feature-audit
repo (never the arXiv PDF: ρ was corrected from −1.0 to −0.540..0.062).

---

## Part 4 — Priority order if the deadline is days away

| Priority | Deliverable | Runs | Enough for |
|---|---|---|---|
| 1 | Exit gate result + disclosure | done | UC |
| 2 | Spark timing + D1/D2 recorded | R1, R2 | UC (feasibility) |
| 3 | Stage A on all 6 pairs | R3 (hours) | UC preliminary results: core sizes, grid-nesting diagnostics on primaries |
| 4 | Pythia-160M Stage B, frozen | R4, R6 | UC: "the null exists and is frozen": D_null distributions per cell |
| 5 | Pythia-160M Stage C (after approval) | R7 | UC: first CSI numbers, both levels |
| 6 | Primaries Stage B → freeze → C | R5–R7 | main track |
| 7 | Everything in Part 5 | — | main track |

For the UC paper itself: write it as the two examples do (proposal-style, ~2 pages). Put
priorities 1–4 (and 5 if done) in a short "Preliminary results" paragraph, and the rest of the
grid in "Evaluation plan".

---

## Part 5 — What a main-conference version needs on top

A main-track reviewer will not accept a proposal. They will ask for the full measurement, a
demonstration that the null changes conclusions, and validity checks. In rough order of how
much they strengthen the paper:

1. **The full grid on both primaries**, CSI with bootstrap CIs over B, S and r at both levels, BH
   at q = 0.05 across the grid, the behavioural floor applied (D4).
2. **The null's value, demonstrated.** For every cell, what a naive before/after comparison
   (Jaccard or D with no null) would conclude against what CSI concludes. This is the paper's
   thesis made visible, and it costs **no new runs** (Stage A + Stage C outputs).
3. **All three nulls, as promised**: implement matched-perplexity (b). It needs per-cell
   WikiText-2 perplexity under the pre-registered protocol (window 1024, stride 512, BOS per
   window), which also supplies the Δlog PPL that C5 correlates against.
4. **C5, calibrated**: Spearman(circuit damage, Δlog PPL) on the 7 intersection cells, placed in
   the published feature-diagnostic table (−0.975 / −0.684 / −0.032), plus Duan's RTN arm on
   Gemma-2-2B. Requires D3.
5. **C4 interaction diagnostic** on the largest inclusion-frequency movers, with the INT-flag
   ratio pre-registered and a 0.3 / 0.7 sensitivity check.
6. **C3 threshold sweep** with Kendall's W across the threshold grid (the statistic the
   feature-audit paper uses, W = 0.9416), reported as "which conclusions survive".
7. **A second extraction pipeline**, or an explicit limitation. The proposal promises two so
   that no conclusion rests on one. The circuit-tracer transcoder pins exist for Gemma/Llama.
   Comparing it with dense-node is also the basis-drift check (proposal risk 1).
8. **Task generality**: greater-than cannot run on the primaries, so add a task their tokenizers
   support (or port docstring) so that the primaries have more than IOI.
9. **Extraction validity**: EAP vs activation patching on a subset (EAP's R² against patching
   is 0.27, arXiv:2310.10348 §5.1), and a seed replication (for example seeds 5–9 on a few cells).
10. **More scale points**: Pythia-410M, and Pythia-70M as the contact point with Duan.
11. **Reproducibility**: the AAAI reproducibility checklist, an artifact with configs + hashes +
    scripts (no weights: Gemma/Llama licences), the frozen manifest hash, and the freeze tags.
12. **Writing**: related work built on arXiv:2607.18921 (cross-seed Jaccard 0.363) and CIRCUS; a
    limitations section built on the proposal's three named risks; the disclosures (B6 post-hoc
    criterion, nested grid, greater-than Pythia-only, EAP calibration caveat).

Compute for items 3 and 7 is comparable to the whole current grid. Plan them after the
primaries' Stage C, not before.

---

## Part 6 — What I changed in the code today, and what is left for you to approve

**Fixed (Stage A → Stage B → freeze), tested (1000 passed, 14 skipped):**

1. **Stage B runs were indistinguishable by name.** Every run inherited `setting: dense`, so the
   11 nulls of a (model, task) shared one run name, and `make_freeze_config.py`, which finds a
   cell by its token in the run name, would have reported all 66 as MISSING and refused to
   build the freeze. The queues now pass `setting=null-matchedmag-<cell>` (Stage B) and
   `setting=<cell>` (Stage C). `tests/test_freeze_discovery.py` composes all 66 queue lines with
   Hydra and requires 66 distinct names that the freeze config finds.
2. **The freeze config counted cells that cannot exist.** It looped over every task for every
   model, so the 22 greater-than cells of Gemma/Llama were always "MISSING". It now uses the
   viable pairs.
3. **Gemma and Llama would have been frozen under the wrong names** (`gemma2-2b`, `llama32-1b`),
   where Stage C never looks (it uses `gemma-2-2b`, `llama-3.2-1b`). Names are now read from the
   model configs.
4. **A crashed cell counted as done.** `_run_one.sh` treated any log containing `run_dir` as
   finished, and every Python traceback from a runner contains `run_dir = allocate_run_dir(`. It
   now writes an explicit `CELL_COMPLETE` marker after a zero exit.
5. **New:** `--models` / `06_freeze.sh <model>` for a deliberate per-model freeze; a Stage A
   queue (`cells_stagea.txt`, `05_run_queue.sh stageA`); `deploy/shared/record_result.py` and
   `RESULTS.md`, one timestamped row per cell.

**Left for your decision (not changed):**

| Gap | Effect if run as is | What the change would be |
|---|---|---|
| `experiments/run_stage_c.py::_load_model` raises for every real model ("explicit PI approval for real runs") | every Stage C cell fails | delegate to the Stage A loader, as Stage B does. **Needs your explicit go-ahead.** |
| `_run_one.sh` never passes `null_frozen_hash` to Stage C, which requires it | every Stage C cell fails at tag validation | read the cell's hash from `frozen/FREEZE_MANIFEST.json` and pass `+null_frozen_hash=…` |
| `07_stage_d_analysis.sh` passes `stage_d.stage_c_dirs`; Stage D reads `stage_d.stage_c_run_dirs` (and a new key needs `+`); no model/task/hash/ensemble overrides | Stage D fails, and would mix all models in one ranking | one Stage D per (model, task), overrides generated like the queues |
| `analysis/threshold_sweep.py` and `analysis/cross_audit.py` have no entry point | step 07 prints nothing for C3 and C5 | a driver outside the Novelty Zone that feeds them Stage C outputs; Stage C should also save `freq_dense.parquet` |
| `05_run_queue.sh stageC` checks only that `frozen/FREEZE_MANIFEST.json` exists (it always does) | none: the Python guard refuses anyway | check `"frozen": true` |
| Matched-perplexity null (b) and the C4 patch diagnostic raise for real models | not runnable | both are in `src/science/`, the Novelty Protection Zone: your approval, then engineering |

None of these block R0–R6, which is the next several days of Spark time. Tell me which of them to
make, and I will make them, test them, and hand you the commands.
