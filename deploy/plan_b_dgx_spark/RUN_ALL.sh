#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Plan B one-command orchestrator (stops only where a human must decide)
#
#   ./deploy/plan_b_dgx_spark/RUN_ALL.sh [parallelism] [model-filter]
#
# Example:
#   ./RUN_ALL.sh 1                 # careful first pass, one cell at a time
#   ./RUN_ALL.sh 12 pythia160m     # then pack the small model densely
#   ./RUN_ALL.sh 4  gemma2_2b      # and the large model sparsely
#
# Runs every step in order and is RESUMABLE - re-run after any interruption and finished
# cells are skipped. It pauses at exactly ONE place, the freeze, because that is the
# pre-registration event and a machine must not perform it unattended.
#
# Run it under tmux. Stage B is the long pole.
set -uo pipefail
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO="$( cd "$HERE/../.." && pwd )"
PY="${PY:-python}"
cd "$REPO"

PAR="${1:-1}"
FILTER="${2:-}"
STARTED=$(date +%s)
mkdir -p logs/run_all
TRANSCRIPT="logs/run_all/run_all_$(date +%Y%m%d_%H%M%S).log"

# Everything below is tee'd to the transcript as well as the terminal.
exec > >(tee -a "$TRANSCRIPT") 2>&1

step () {
  local name="$1"; shift
  echo ""
  echo "======================================================================"
  echo "STEP: $name   ($(date +%H:%M:%S))"
  echo "======================================================================"
  if ! "$@"; then
    echo ""
    echo "STOPPED at '$name'. Fix it and re-run this script -"
    echo "completed work is skipped, so you lose nothing."
    exit 1
  fi
}

cat <<BANNER

  Plan B - full grid on the DGX Spark
  Parallelism: $PAR${FILTER:+   Filter: $FILTER}
  Log: $TRANSCRIPT

  This takes many GPU-hours. It pauses once, at the freeze.
  Run it under tmux so a dropped session does not kill it.

BANNER

step "hardware"    bash "$HERE/00_inspect_hardware.sh"
step "environment" bash "$HERE/01_setup_env.sh"
step "verify"      bash "$HERE/03_verify_gate.sh"

# The preflight gate.
if ! "$PY" deploy/shared/preflight_blockers.py; then
  cat <<'MSG'

  STOPPING: the real-model code paths are still stubs, so there is nothing to run.
  This is an engineering gap, not a configuration problem. See deploy/BLOCKERS.md.

  Hardware, environment and verification all passed - that part is done.

MSG
  exit 1
fi

step "timing"  bash "$HERE/04_timing.sh"

cat <<'MSG'

  PAUSE FOR A DECISION YOU MUST MAKE NOW, NOT LATER:

  Re-apply the Q3 rule to the timings just measured. If it changes B, S or R
  for this hardware, edit the configs and RECORD it in docs/HUMAN_DECISIONS.md
  BEFORE the freeze. Changing R after seeing results is the re-tuning the
  frozen null exists to prevent.

MSG
read -r -p "  Q3 re-applied and recorded? [y/N] " a
if [ "$a" != "y" ] && [ "$a" != "Y" ]; then
  echo "  Stopping so you can do that. Re-run this script afterwards."
  exit 0
fi

step "stageB" bash "$HERE/05_run_queue.sh" stageB "$PAR" "$FILTER"

cat <<'MSG'

======================================================================
  THE FREEZE - this is the pre-registration event.

  Confirm FIRST that the freeze-ownership table in docs/HUMAN_DECISIONS.md
  says THIS machine owns these cells. If the PC already froze them, copy that
  freeze across and skip - two freezes of one cell cannot be reconciled.

  06_freeze.sh will show the plan and ask you to type FREEZE.
======================================================================

MSG

step "freeze" bash "$HERE/06_freeze.sh"

if [ ! -f frozen/FREEZE_MANIFEST.json ]; then
  echo "No freeze manifest - you aborted the freeze. Stopping here."
  exit 0
fi

step "stageC" bash "$HERE/05_run_queue.sh" stageC "$PAR" "$FILTER"
step "stageD" bash "$HERE/07_stage_d_analysis.sh"

ELAPSED=$(( $(date +%s) - STARTED ))
echo ""
echo "======================================================================"
printf "COMPLETE in %dh %dm\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60))
echo "Now work through deploy/plan_b_dgx_spark/CHECKLIST.md, then run"
echo "08_bundle_results.sh to carry the results home."
echo "======================================================================"
