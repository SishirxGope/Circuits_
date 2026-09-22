# deploy/ — everything needed to run the project on either machine

Two self-contained folders, one per plan, plus shared tooling. Written so the Spark folder
can be followed **without an AI agent**: every step is a script you run, and every script
says what it expects and what it changes.

```
deploy/
├── README.md              ← you are here
├── BLOCKERS.md            ← READ FIRST. What is not implemented yet, and why nothing runs without it
├── shared/                ← tooling both machines use
│   ├── preflight_blockers.py    refuses to start science while the stubs are in place
│   ├── gen_cells.py             the ONE grid table → configs + run queues
│   ├── fetch_assets.py          pinned weights + corpora (reads hf_revision, never `main`)
│   ├── fetch_upstream_forks.ps1/.sh   the two missing upstream repos, at their pins
│   ├── make_freeze_config.py    builds the freeze cell list from completed Stage B runs
│   └── make_transfer_bundle.ps1 packs repo + caches + frozen/ for the trip to the office
├── plan_a_local_pc/       ← RTX 4060, Windows/PowerShell, Pythia only
└── plan_b_dgx_spark/      ← DGX Spark, Linux/bash, all four models
```

**Doing it step by step? Start at [`../EXECUTION_GUIDE.md`](../EXECUTION_GUIDE.md)** — the
single operational document, with every command for both machines in order.

The narrative plans live in [`../docs/PLAN_A_LOCAL_RTX4060.md`](../docs/PLAN_A_LOCAL_RTX4060.md)
and [`../docs/PLAN_B_DGX_SPARK.md`](../docs/PLAN_B_DGX_SPARK.md). **These folders are the
executable half; those documents are the reasoning half.** Read the plan once, then live here.

---

## Start here

```powershell
# On the PC:
.\deploy\plan_a_local_pc\00_repair_and_verify.ps1
```

```bash
# On the Spark:
bash deploy/plan_b_dgx_spark/00_inspect_hardware.sh
```

Both are safe, read-only-ish, and tell you exactly where you stand.

---

## The honest status, as of 2026-09-21

**None of the science can run yet.** Not because of configuration, but because the real-model
code paths are stubs: the five compressors and the matched-magnitude null all raise
`NotImplementedError` on anything that is not a mock model, and the chance-floor universe is
still the too-lenient `U*(U-1)`.

Run this any time for the current truth:

```bash
python deploy/shared/preflight_blockers.py
```

Every stage script calls it first and **refuses to run** if it fails. That is deliberate: the
alternative is discovering the problem four hours into an overnight queue.

See [`BLOCKERS.md`](BLOCKERS.md) for the specification of each missing piece.

---

## What is ready now

- **The compression grid** — 11 cells in `configs/compression/`, generated and verified to
  compose through Hydra.
- **The run queues** — 44 cells (Plan A) and 88 cells (Plan B) per stage, generated from the
  same table so the cell keys cannot drift apart. A mismatch between a frozen null's cell key
  and Stage C's cell key is a hash error that sends you looking in the wrong place.
- **Every orchestration script** — stage runners, resumable queues, the freeze with its typed
  confirmation, aggregation and analysis.
- **Transfer tooling** — git bundle plus caches, because nothing here is ever pushed to a
  remote.

## What is not

- The six blockers in `BLOCKERS.md`.
- The two upstream forks, which are referenced everywhere but absent from disk.
- Decisions C4, C5, attribution precision, and freeze ownership.

---

## Regenerating the grid

The grid table lives in exactly one place: `shared/gen_cells.py`.

```bash
python deploy/shared/gen_cells.py all
```

It refuses to overwrite a config whose content would change, because configs are **immutable
once a run references them** (`AI_RULES.md` 1.2). If you genuinely mean to change a cell,
delete the file deliberately — and understand that if a freeze already references it, you have
invalidated that frozen null.

---

## The one rule that outranks convenience

`frozen/` is **append-only**. After a freeze, it is never edited, never regenerated, never
tuned to match a result you prefer. If a run fails afterwards, you re-run the run — not the
null.

If the answer comes out as CSI ≈ 1 (compression damage indistinguishable from random weight
noise), that is a finding, and it is publishable. The frozen null exists precisely so that
answer is credible when it arrives.
