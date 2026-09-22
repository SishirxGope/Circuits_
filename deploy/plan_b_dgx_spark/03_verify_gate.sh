#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Plan B step 3 - the verification gate
# Do not run science until all of this passes. This gate is what lets you claim results
# from two different machines are comparable.
source "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/_common.sh"

echo "=== 1. test suite (must match the PC baseline: 511 passed, 1 skipped) ==="
"$PY" -m pytest -q

echo ""
echo "=== 2. pinned loader imports ==="
"$PY" -c "from src.extraction.real_model import load_pinned_model; print('  loader ok')"

echo ""
echo "=== 3. reproduce the Pythia timing pilot ==="
echo "  Attribution SCORES must match the PC to floating-point tolerance."
echo "  TIMINGS will differ - that is expected and is what step 04 measures."
"$PY" -m experiments.time_attribution --model pythia160m --tasks ioi greater_than --seed 0 \
  | tee "$(log_dir verify)/pythia160m_repro.txt"

echo ""
echo "=== 4. frozen manifest still verifies after transfer ==="
if [ -f frozen/FREEZE_MANIFEST.json ]; then
  "$PY" -c "import json,pathlib; print(json.dumps(json.loads(pathlib.Path('frozen/FREEZE_MANIFEST.json').read_text()), indent=2)[:600])"
else
  echo "  no manifest (nothing frozen yet)"
fi

echo ""
echo "=== 5. real-model blockers ==="
"$PY" deploy/shared/preflight_blockers.py || true

echo ""
echo "Compare section 3 against docs/baseline_pc.txt from the PC."
echo "If the SCORES differ materially, stop: the two machines are measuring different"
echo "things and their cells cannot be pooled."
