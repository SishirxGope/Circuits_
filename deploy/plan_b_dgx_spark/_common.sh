#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Plan B shared helpers (sourced by every stage script)
set -euo pipefail
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO="$( cd "$HERE/../.." && pwd )"
PY="${PY:-python}"

cd "$REPO"

COMMON=(
  mode=scientific_run
  pipeline=dense-node
  ensemble=default
  ensemble/decompose=final
  nulls=default
  comparison=final
  comparison_level=both
)

assert_preflight () {
  if ! "$PY" "$REPO/deploy/shared/preflight_blockers.py" --quiet; then
    echo ""
    echo "STOPPED: real-model blockers are not implemented."
    echo "Run: python deploy/shared/preflight_blockers.py"
    echo "See: deploy/BLOCKERS.md"
    exit 1
  fi
}

log_dir () { local d="$REPO/logs/$1"; mkdir -p "$d"; echo "$d"; }

read_cells () { grep -v '^#' "$1" | grep -v '^[[:space:]]*$'; }
