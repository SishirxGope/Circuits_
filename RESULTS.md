# RESULTS — Do Circuits Survive Compression?

Every result the project produces, dated, from 2026-09-30 onward. Rules:

- **Append-only.** Rows are added, never edited or deleted. A result that turns out to be
  wrong gets a new row that says so and points back to the old one.
- **Numbers are copied from the run's own output files**, never retyped from memory or
  rounded into a different claim (AI_RULES.md 2.2).
- **Evidence vs non-evidence.** Timing pilots, smoke tests and Stage B *draft* nulls are
  not scientific results. They are recorded because the plan depends on them.
- **The run log at the bottom fills itself.** `deploy/plan_b_dgx_spark/_run_one.sh` calls
  `deploy/shared/record_result.py` after every Stage A/B/C cell, success or failure, and
  appends one row with the local date and time. The timing pilot
  (`experiments/time_attribution.py`) and the behaviour check
  (`deploy/shared/check_task_behaviour.py`) append their own rows too. Record anything else
  by hand in the table below, or with `python deploy/shared/record_result.py --run-dir runs/<run_name>`.
- **Either machine may append.** Run `git pull` before, then commit and push this file
  straight after. `.gitattributes` gives it the `union` merge driver, so rows appended on
  both machines merge without a conflict.
- This is a **project-wide rule** (CLAUDE.md §8): every result, evidence or not, gets a
  dated row here in the same session it is produced.

Decisions behind these results live in `docs/HUMAN_DECISIONS.md`; what is still to be run
lives in `docs/RUNPLAN.md`.

---

## Recorded results (entered by hand)

Times were not recorded for the entries before this file existed; they carry the date only.

| When | What | Result | Where / source | Status |
|---|---|---|---|---|
| 2026-09-14 | Attribution-pass timing pilot, Pythia-160M, RTX 4060 | IOI 42–48 s/pass, 4.26 GiB peak; greater-than 7.5–7.9 s/pass, 2.64 GiB | `docs/PROGRESS.md` | non-evidence (planning) |
| 2026-09-14 | Dense-node threshold grid, view diagnostics (Pythia-160M) | Pre-registered grid largely **nested**: 82–116 of 120 view pairs; the pre-registered fix passed for no candidate. Grid kept as pre-registered; both findings to be reported | `docs/preregistration/2026-09-14_dense_node_grid_outcome.md` | finding (methodological) |
| 2026-09-14 | Task sanity, Pythia-160M dense | IOI logit diff +4.56 → −0.27 under ABC corruption; greater-than prob diff +0.763 → −0.640 under year corruption | `docs/PROGRESS.md` | check passed |
| 2026-09-30 | GPTQ/AWQ full-size smoke, DGX Spark (bf16, full Q7 cache 1171×256) | Pythia-160M GPTQ 151 s / 3.3 GiB, AWQ 1017 s / 6.4 GiB; Llama-3.2-1B GPTQ 1328 s / 20.2 GiB, AWQ 10155 s / 29.5 GiB; Gemma-2-2B GPTQ 2015 s / 32.6 GiB, AWQ 18338 s / 42.9 GiB. One Pythia-160M AWQ tensor at ‖ΔW‖/‖W‖ = 0.60 (rest ~0.10), not yet identified | `deploy/shared/smoke_gptq_awq.py` output | non-evidence (engineering) |
| 2026-09-30 | **B6 exit gate, first criterion (Jaccard ≥ 0.2), seeds 0–4** | **FAILED**: Jaccard 0.028. Core band 37 edges, 27 in the 963-edge ACDC/Wang et al. reference (precision 0.73, recall 0.028) | Spark `runs/20260930_stageA_gpt2-small_ioi_dense_B16xS5_seed0` | gate result (reported in the paper) |
| 2026-09-30 | **B6 exit gate, replacement criterion (precision ≥ 0.3 and hypergeometric p ≤ 0.001), fresh seeds 5–9** — criterion changed after seeing the failure; disclosed | **PASSED**: precision 0.725, p = 1.27e-45; core band 51 edges, 37 shared, reference 963, universe 32,491; Jaccard 0.038 (reported only). config_hash `a324db7b…` | Spark `runs/20260930_stageA_gpt2-small_ioi_dense_B16xS5_seed5`; `data/reference/ioi_gpt2_small_gate_pass.json` | gate result (reported in the paper) |
| 2026-09-30 | Preflight on the Spark | **10 of 10 OK**; "All 10 blockers implemented. Clear to run science." | `deploy/shared/preflight_blockers.py` | engineering gate |
| 2026-10-01 (time not recorded) | Attribution-pass timing, Pythia-160M, DGX Spark, 2 seeds | IOI 19.97 s/pass, 35.1 min per cell at S5×R20; greater-than 10.20 s/pass, 18.0 min per cell. Q3 rule → S=5, R=20 | Spark `runs/pilot_timing/*_timing_pythia160m.json` | non-evidence (planning) |
| 2026-10-01 (time not recorded) | Attribution-pass timing, Pythia-410M, DGX Spark, 2 seeds | IOI 101.40 s/pass, 178.5 min per cell at S5×R20; greater-than 61.50 s/pass, 108.6 min per cell. Q3 rule → S=5, R=20 | Spark `runs/pilot_timing/*_timing_pythia410m.json` | non-evidence (planning) |
| 2026-10-01T01:19:20+05:30 | Attribution-pass timing, Gemma-2-2B (float32), IOI, DGX Spark, 2 seeds | 290.3 s/pass, peak 37.5 GiB. Q3 rule → **S=5, R=10**. Clean IOI metric −4.33 / −4.69 (negative: prefers the wrong name, see the behaviour check below) | Spark `runs/pilot_timing/20261001-011920_timing_gemma2_2b.json` | non-evidence (planning) |
| 2026-10-01T01:29:17+05:30 | Attribution-pass timing, Llama-3.2-1B (bf16), IOI, DGX Spark, 2 seeds | 52.2 s/pass, peak 22.5 GiB. Q3 rule → S=5, R=20. Clean IOI metric +4.97 / +5.04 | Spark `runs/pilot_timing/20261001-012917_timing_llama32_1b.json` | non-evidence (planning) |
| 2026-10-01T01:38:52+05:30 | Behaviour check, Pythia-160M, seed 0, n=300 (clean / corrupt metric) | IOI: no BOS (pre-registered) 4.560 / −0.273; BOS 5.418 / −0.250. greater-than: no BOS (pre-registered) 0.763 / −0.640; BOS 0.630 / −0.670. Does both tasks either way | Spark `runs/pilot_behaviour/20261001-014312_behaviour.json` (commit cfe622a) | non-evidence (check) |
| 2026-10-01T01:38:52+05:30 | Behaviour check, Pythia-410M, seed 0, n=300 | IOI: no BOS (pre-registered) 3.769 / −0.109; BOS 4.231 / −0.157. greater-than: no BOS (pre-registered) 0.910 / −0.707; BOS 0.934 / −0.673. Does both tasks either way | same file | non-evidence (check) |
| 2026-10-01T01:38:52+05:30 | Behaviour check, Gemma-2-2B, IOI, seed 0, n=300 | **no BOS (pre-registered): −4.326 / −0.164, DOES NOT do the task**; BOS: 5.311 / −0.170, does the task | same file | non-evidence (check); blocks Gemma until the PI decides BOS |
| 2026-10-01T01:38:52+05:30 | Behaviour check, Llama-3.2-1B, IOI, seed 0, n=300 | no BOS (pre-registered) 4.972 / −0.010; BOS 5.499 / −0.009. Does the task either way | same file | non-evidence (check) |
| 2026-10-01 (time not recorded) | First Stage A launch on the Spark (`05_run_queue.sh stageA 1 "pythia160m task=greater_than"`) | **Did not start**: `xargs: _run_one.sh: Permission denied`. The .sh files are committed without the execute bit. No cell ran and no run directory was created. Fixed: `05_run_queue.sh` now runs `bash _run_one.sh`; covered by `tests/test_record_result.py::test_no_script_is_executed_without_bash` | Spark terminal output | engineering failure (no result) |

---

## Run log

Appended automatically, one row per finished or failed cell. Local time with UTC offset.

| When (local) | Stage | Model | Task | Cell | Result | Run dir | Commit |
|---|---|---|---|---|---|---|---|
