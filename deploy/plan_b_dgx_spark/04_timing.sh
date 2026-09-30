#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Plan B step 4 - re-measure timing, then re-apply the Q3 rule
# modified: [AI-GEN] agent=Claude date=2026-10-01 task=time only the tasks each model can run (gen_cells.viable_pairs); optional model list
# Every compute figure in this project was measured on an RTX 4060. None of them transfer.
# This output is NON-EVIDENCE by construction: a planning measurement, not a result.
#
#   bash deploy/plan_b_dgx_spark/04_timing.sh                       # all four models
#   bash deploy/plan_b_dgx_spark/04_timing.sh gemma2_2b llama32_1b  # just these
source "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/_common.sh"
LOG="$(log_dir timing)"

MODELS=("$@")
if [ "${#MODELS[@]}" -eq 0 ]; then MODELS=(pythia160m pythia410m gemma2_2b llama32_1b); fi

for model in "${MODELS[@]}"; do
  # Only the tasks this model can run. greater_than is structurally impossible under the
  # Gemma-2 and Llama-3.2 tokenizers (gen_cells.IMPOSSIBLE_CELLS). Asking for it anyway
  # crashed the timer AFTER IOI had been timed, and the report is written only at the end,
  # so the IOI number was lost too (2026-10-01).
  mapfile -t TASKS < <("$PY" -c "import sys; sys.path.insert(0, 'deploy/shared'); from gen_cells import viable_pairs; print('\n'.join(t for _, t in viable_pairs(['$model'])))")
  echo ""
  echo "=== $model: ${TASKS[*]} ==="
  "$PY" -m experiments.time_attribution --model "$model" --tasks "${TASKS[@]}" --seeds 2 \
    2>&1 | tee "$LOG/$model.txt" || echo "  FAILED - see the traceback above and $LOG/$model.txt"
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

Budget: Stage B attribution passes per cell = S x (R + 1) (the dense ensemble plus
        R null draws). At S=5, R=20 that is 105 per cell, 1155 per (model, task)
        pair of 11 cells, 6930 for the 66-cell grid.
--------------------------------------------------------------------------
NOTE
