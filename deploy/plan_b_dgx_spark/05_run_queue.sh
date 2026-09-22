#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Plan B step 5 - parallel cell runner (the Spark's real advantage)
#
# Usage:
#   ./05_run_queue.sh stageB 1                 # ALWAYS start with 1 and check one cell
#   ./05_run_queue.sh stageB 8 pythia160m      # then scale, one model at a time
#   ./05_run_queue.sh stageC 4 gemma2_2b
#
# CONCURRENCY IS PER-MODEL, because peak memory is per-model. Pythia-160M peaked at
# 4.26 GiB (IOI) on the 4060; Gemma-2-2B is estimated near 25 GiB. Reusing one -P value
# across models will either OOM or leave most of the machine idle. Filter by model.
#
# Resumable: a cell whose log already records completion is skipped, so re-running after
# a failure or a dropped session costs nothing. Use tmux/screen for long queues.
set -euo pipefail
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO="$( cd "$HERE/../.." && pwd )"
PY="${PY:-python}"
cd "$REPO"

STAGE="${1:-stageB}"
PAR="${2:-1}"
FILTER="${3:-}"

case "$STAGE" in
  stageB) RUNNER="experiments/run_stage_b.py"; QUEUE="$HERE/cells_stageb.txt" ;;
  stageC) RUNNER="experiments/run_stage_c.py"; QUEUE="$HERE/cells_stagec.txt" ;;
  *) echo "usage: $0 [stageB|stageC] [parallelism] [model-filter]"; exit 2 ;;
esac

if ! "$PY" deploy/shared/preflight_blockers.py --quiet; then
  echo ""
  echo "STOPPED: real-model blockers are not implemented (deploy/BLOCKERS.md)."
  "$PY" deploy/shared/preflight_blockers.py || true
  exit 1
fi

if [ "$STAGE" = "stageC" ] && [ ! -f frozen/FREEZE_MANIFEST.json ]; then
  echo "No freeze manifest. Stage C has no denominator - run 06_freeze.sh first."
  exit 1
fi

LOG="$REPO/logs/$STAGE"; mkdir -p "$LOG"

if [ -n "$FILTER" ]; then
  mapfile -t CELLS < <(grep -v '^#' "$QUEUE" | grep -v '^[[:space:]]*$' | grep -- "$FILTER")
else
  mapfile -t CELLS < <(grep -v '^#' "$QUEUE" | grep -v '^[[:space:]]*$')
fi

echo "$STAGE: ${#CELLS[@]} cells, parallelism $PAR${FILTER:+, filter '$FILTER'}"
echo "Logs: $LOG"
if [ "$PAR" -gt 1 ]; then
  echo ""
  echo "Watch memory in another terminal:"
  echo "  nvidia-smi --query-gpu=memory.used --format=csv -l 5"
fi
echo ""

printf '%s\n' "${CELLS[@]}" \
  | xargs -P "$PAR" -I {} "$HERE/_run_one.sh" "$RUNNER" "$LOG" "{}" \
  || { echo ""; echo "One or more cells failed. Re-run: completed cells are skipped."; exit 1; }

echo ""
echo "$STAGE queue complete."
if [ "$STAGE" = "stageB" ]; then echo "NOTHING IS FROZEN YET - run 06_freeze.sh."; fi
