#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Plan B step 7 - Stage D aggregation and the analyses
source "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/_common.sh"
LOG="$(log_dir stageD)"

mapfile -t DIRS < <(find runs -maxdepth 1 -type d -name '*_stageC_*' | sort)
if [ "${#DIRS[@]}" -eq 0 ]; then echo "No Stage C runs found."; exit 1; fi
echo "Aggregating ${#DIRS[@]} Stage C runs."

JOINED="$(IFS=,; echo "${DIRS[*]}")"
"$PY" experiments/run_stage_d.py mode=scientific_run stage=stageD \
  "stage_d.stage_c_dirs=[$JOINED]" 2>&1 | tee "$LOG/stage_d.log"

echo ""
echo "=== Threshold sweep ==="
"$PY" analysis/threshold_sweep.py 2>&1 | tee "$LOG/threshold_sweep.log"

echo ""
echo "=== Cross-audit (C5) ==="
echo "REMINDER: correlate against the frozen CSV tables of the feature-level paper,"
echo "NEVER the arXiv PDF. The authors corrected rho = -1.0 to a range of -0.540..0.062,"
echo "and Gemma-2's dense WikiText-2 PPL from 410 to 8.21. See docs/Run_Plan.md 0.2."
"$PY" analysis/cross_audit.py 2>&1 | tee "$LOG/cross_audit.log"
