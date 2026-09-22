#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Plan B step 4 - re-measure timing, then re-apply the Q3 rule
# Every compute figure in this project was measured on an RTX 4060. None of them transfer.
# This output is NON-EVIDENCE by construction: a planning measurement, not a result.
source "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/_common.sh"
LOG="$(log_dir timing)"

for model in pythia160m pythia410m gemma2_2b llama32_1b; do
  echo ""
  echo "=== $model ==="
  "$PY" -m experiments.time_attribution --model "$model" --tasks ioi greater_than --seeds 2 \
    2>&1 | tee "$LOG/$model.txt" || echo "  FAILED (licence not accepted? model too large? see log)"
done

cat <<'NOTE'

--------------------------------------------------------------------------
NEXT, AND DO NOT SKIP IT:

Re-apply the Q3 rule (docs/Run_Plan.md) to these measured numbers, per model.
Q3 gave B=16, S=5, R=20 from the 4060's Pythia timings. The RULE carries over,
not the answer. If the Spark is slower per pass, the rule may legitimately yield
a smaller R.

Whatever it yields must be decided and recorded in docs/HUMAN_DECISIONS.md
BEFORE the freeze. Changing R after seeing results is exactly the re-tuning the
frozen null exists to prevent.

Budget: Stage B passes per (model, task) = 11 cells x R x S.
        At R=20, S=5 that is 1100 passes per pair, 8800 for the full grid.
--------------------------------------------------------------------------
NOTE
