# Plan A checklist — print this

## Before any science
- [ ] `00_repair_and_verify.ps1` clean: git index rebuilt, CUDA True, 511 passed / 1 skipped
- [ ] Chat export and other >1 MB strays moved out of the repo root
- [ ] `01_fetch_assets.ps1`: Pythia weights at their PINNED revisions, corpora cached
- [ ] Upstream forks present beside the repo (circuit-tracer, sae-pruning-paper)
- [ ] `preflight_blockers.py` exits 0  ← **currently 8 of 9 BLOCKED**
- [ ] C4 and C5 decided and written into `docs/HUMAN_DECISIONS.md`
- [ ] Freeze ownership agreed if the Spark is also running Pythia

## Stage A
- [ ] All 4 model-task pairs completed
- [ ] CIRCUS grid diagnostics inspected: mean pairwise Jaccard NOT near 1.0
- [ ] Match rate (consensus == a single view) NOT ~100%  — that is the nesting artifact
- [ ] Dense-node nesting reported (82–116 of 120 view pairs nested, pre-registered outcome)

## Stage B  → the expensive one, ~16 h
- [ ] All 44 cells completed (re-run the script; it skips finished cells)
- [ ] Logs checked for silent failures, not just exit codes

## THE FREEZE — irreversible
- [ ] Dry run read line by line: every cell present, every run dir correct
- [ ] Typed `FREEZE`
- [ ] `frozen/FREEZE_MANIFEST.json` hash recorded in `docs/HUMAN_DECISIONS.md`
- [ ] `frozen/` committed locally (never pushed)

## Stage C
- [ ] All 44 cells ran against the frozen null, no hash mismatches

## Stage D and analysis
- [ ] CSI table exists, bootstrap CIs over B, S, r
- [ ] Benjamini-Hochberg applied at q = 0.05
- [ ] Every conclusion stated at BOTH levels (exact-edge and layer)
- [ ] Threshold sweep run
- [ ] Chance floor attached to every overlap statistic (B3 implemented — not U*(U-1))
- [ ] Cross-audit used the FROZEN CSV tables, not the arXiv PDF

## Done
- [ ] GPT-2 IOI exit gate passes (no longer skipped)
- [ ] Every number traces to a run directory in `runs/`
- [ ] No model weights anywhere in the repo
