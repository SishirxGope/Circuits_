#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Clone the two upstream forks at their Q9-pinned commits (Linux/Spark)
# URLs and pins quoted from docs/HUMAN_DECISIONS.md section 3.3 (verified 2026-09-12).
set -euo pipefail
WORKSPACE="$(dirname "$(pwd)")"

clone_pinned () {
  local name="$1" url="$2" pin="$3" note="$4"
  local dest="$WORKSPACE/$name"
  echo ""
  echo "  $name"
  echo "    note: $note"
  if [ -d "$dest" ]; then echo "    already present at $dest"; return 0; fi
  git clone "$url" "$dest"
  ( cd "$dest" && git checkout "$pin" )
  local head; head="$( cd "$dest" && git rev-parse HEAD )"
  if [ "$head" != "$pin" ]; then echo "    PIN MISMATCH: got $head"; else echo "    pinned at ${pin:0:12}"; fi
}

clone_pinned "circuit-tracer-0.5.2" \
  "https://github.com/decoderesearch/circuit-tracer.git" \
  "8f1e2438df612464e229e44c4a00ff637bf9379b" \
  "canonical URL; safety-research/* 301-redirects here"

clone_pinned "sae-pruning-paper-main" \
  "https://github.com/hecboar/sae-pruning-paper.git" \
  "261191804675e2d39d0a265320dbc0bc85afd30a" \
  "PIN IS INFERRED (HUMAN_DECISIONS 3.3) - verify before trusting it"

echo ""
echo "  Verify the cross-audit table exists:"
echo "    $WORKSPACE/sae-pruning-paper-main/results/E6/stat_tests.csv"
