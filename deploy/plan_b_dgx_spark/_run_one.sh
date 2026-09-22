#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Run exactly one grid cell (invoked by 05_run_queue.sh via xargs)
# Args: <runner.py> <log-dir> <hydra override string for one cell>
set -uo pipefail
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO="$( cd "$HERE/../.." && pwd )"
PY="${PY:-python}"
cd "$REPO"

RUNNER="$1"; LOG="$2"; CELL="$3"
TAG="$(echo "$CELL" | tr ' =' '__')"
LOGFILE="$LOG/$TAG.log"

if [ -f "$LOGFILE" ] && grep -qE "run_dir|COMPLETE" "$LOGFILE" 2>/dev/null; then
  echo "  skip (done): $CELL"
  exit 0
fi

echo "  start: $CELL"
# shellcheck disable=SC2086
if "$PY" "$RUNNER" \
     mode=scientific_run pipeline=dense-node ensemble=default ensemble/decompose=final \
     nulls=default comparison=final comparison_level=both \
     $CELL seed=0 > "$LOGFILE" 2>&1; then
  echo "  done : $CELL"
  exit 0
else
  echo "  FAIL : $CELL  -> $LOGFILE"
  exit 1
fi
