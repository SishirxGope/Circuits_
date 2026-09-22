#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Plan B step 6 - THE FREEZE (irreversible)
#
# BEFORE RUNNING: confirm the freeze-ownership table in docs/HUMAN_DECISIONS.md says
# THIS machine owns these cells. If the PC already froze them, copy that freeze across
# and skip this script entirely. Two freezes of one cell cannot be reconciled afterwards.
source "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/_common.sh"
CFG="$REPO/freeze_config_plan_b.json"

echo "=== 1. Build the freeze config from completed Stage B runs ==="
"$PY" deploy/shared/make_freeze_config.py --plan b --out "$CFG"

echo ""
echo "=== 2. DRY RUN - what would be frozen ==="
"$PY" experiments/freeze_stage_b.py --config "$CFG" --dry-run

echo ""
echo "=== 3. Confirm ==="
echo "This is the pre-registration event. It cannot be undone."
echo "Check above that every cell is present and points at the right run directory."
read -r -p "Type FREEZE (all capitals) to proceed: " answer
if [ "$answer" != "FREEZE" ]; then echo "Aborted. Nothing written."; exit 0; fi

echo ""
echo "=== 4. Freezing ==="
"$PY" deploy/shared/make_freeze_config.py --plan b --out "$CFG" --approve
"$PY" experiments/freeze_stage_b.py --config "$CFG"

echo ""
echo "Frozen. Record the manifest hash in docs/HUMAN_DECISIONS.md now:"
cat frozen/FREEZE_MANIFEST.json
echo ""
echo "Commit frozen/ immediately (locally - never push)."
