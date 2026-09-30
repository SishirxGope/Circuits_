#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Run exactly one grid cell (invoked by 05_run_queue.sh via xargs)
# modified: [AI-GEN] agent=Claude date=2026-09-30 task=explicit completion marker (a traceback no longer counts as done) + RESULTS.md row per cell
# Args: <runner.py> <log-dir> <hydra override string for one cell>
set -uo pipefail
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO="$( cd "$HERE/../.." && pwd )"
PY="${PY:-python}"
cd "$REPO"

RUNNER="$1"; LOG="$2"; CELL="$3"
TAG="$(echo "$CELL" | tr ' =' '__')"
LOGFILE="$LOG/$TAG.log"
DONE_MARK="CELL_COMPLETE"

# Done means THIS script wrote the marker after the runner exited 0. It used to grep for
# "run_dir", which every Python traceback from a runner contains (`run_dir = allocate_run_dir(`),
# so a cell that crashed was skipped as finished on every later pass.
if [ -f "$LOGFILE" ] && grep -q "^$DONE_MARK" "$LOGFILE" 2>/dev/null; then
  echo "  skip (done): $CELL"
  exit 0
fi

echo "  start: $CELL"
# shellcheck disable=SC2086
if "$PY" "$RUNNER" \
     mode=scientific_run pipeline=dense-node ensemble=default ensemble/decompose=final \
     nulls=default comparison=final comparison_level=both \
     $CELL seed=0 > "$LOGFILE" 2>&1; then
  echo "$DONE_MARK $(date -Iseconds)" >> "$LOGFILE"
  "$PY" deploy/shared/record_result.py --log "$LOGFILE" >/dev/null 2>&1 \
    || echo "  (could not add the RESULTS.md row for $CELL; the run itself is fine)"
  echo "  done : $CELL"
  exit 0
else
  "$PY" deploy/shared/record_result.py --failed --cell "$CELL" --log "$LOGFILE" >/dev/null 2>&1 || true
  echo "  FAIL : $CELL  -> $LOGFILE"
  exit 1
fi
