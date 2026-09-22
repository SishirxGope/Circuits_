# Plan A — local PC (RTX 4060, 8 GB)

**Scope:** Pythia-160M and Pythia-410M, both tasks, all 11 compression cells, Stage A → D.
Gemma-2-2B and Llama-3.2-1B do not fit in 8 GB and belong to Plan B.

Full reasoning: [`../../docs/PLAN_A_LOCAL_RTX4060.md`](../../docs/PLAN_A_LOCAL_RTX4060.md)

All scripts are PowerShell, run from the **repo root**, and use `.venv\Scripts\python.exe`.

---

## One command

```powershell
.\deploy\plan_a_local_pc\RUN_ALL.ps1
```

Runs everything below in order, unattended, and is **resumable** — re-run it after any crash,
OOM or reboot and it skips what already finished. It pauses at exactly one place: the freeze,
where it asks you to type `FREEZE`. Everything is transcribed to `logs/run_all/`.

Useful flags: `-SkipFetch` (don't re-download assets), `-StopBefore freeze` (stop early).

The steps below are the same things it runs; call them individually when you want control.

## The sequence

| Step | Script | What it does | Time |
|---|---|---|---|
| 0 | `00_repair_and_verify.ps1` | Repairs the git index, checks GPU/torch, runs the test suite, reports blockers | ~2 min |
| 1 | `01_fetch_assets.ps1` | Pinned Pythia weights, the corpora, the two upstream forks | ~20 min |
| 2 | `02_stage_a.ps1` | Stage A dense reference, 4 model-task pairs | ~5 min |
| 3 | `03_stage_b.ps1` | Stage B null draws — **44 cells, the expensive stage** | **~16 h** |
| 4 | `04_freeze.ps1` | **THE FREEZE** — irreversible, asks you to type `FREEZE` | ~1 min |
| 5 | `05_stage_c.ps1` | Stage C real compression against the frozen null | ~2 h |
| 6 | `06_stage_d_analysis.ps1` | Stage D CSI table, threshold sweep, cross-audit | ~10 min |

Add Pythia-410M and the total is **4–7 days of near-continuous GPU time** (410M is an
*estimated* 3–4× slower; measure it before trusting that).

---

## Before step 2

Steps 2–6 **will refuse to run** until `deploy/shared/preflight_blockers.py` passes. As of
2026-09-21 it reports 8 of 9 blocked. See [`../BLOCKERS.md`](../BLOCKERS.md).

This is not a configuration problem you can override. The code that compresses a real model
does not exist yet.

---

## Things worth knowing

**Step 3 is resumable.** Cells whose log records completion are skipped, so re-running after a
crash, an OOM or a reboot costs nothing. Expect to re-run it.

**One cell at a time.** An IOI pass peaks at 4.26 GiB, so two Pythia cells will not fit
side by side on 8 GB even though they are logically independent. Close browsers and anything
else using the GPU.

**On OOM:** halve the batch size and re-run. **Never switch precision** — float32 is
pre-registered for Pythia (Q5/Q3 notes in `configs/model/pythia160m.yaml`), and changing it
mid-grid makes cells incomparable.

**Step 4 cannot be undone.** It asks you to type `FREEZE` in capitals for a reason. Read the
dry-run output first: confirm every cell is present and points at the right run directory.
After it succeeds, `frozen/` is append-only forever.

**Step 6's cross-audit** must correlate against the feature-level paper's **frozen CSV
tables**, never its arXiv PDF. The authors corrected ρ = −1.0 to a range of −0.540 to +0.062.
The script prints this reminder; heed it.

---

## If you hand Pythia over to the Spark

Plan B covers Pythia too. If the Spark arrives mid-grid and you want it to take over,
**read Plan B's freeze-ownership rule first**. Two machines freezing the same cell produces
two nulls for one cell key and no principled way to choose between them — which is exactly the
re-tuning this whole design forbids.

Decide ownership, write it in `docs/HUMAN_DECISIONS.md`, and let the non-owning machine only
*read* the freeze.
