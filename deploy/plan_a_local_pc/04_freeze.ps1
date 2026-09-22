# [AI-GEN] agent=Claude date=2026-09-21 task=Plan A step 4 - THE FREEZE (irreversible pre-registration event)
#
# Read docs/PLAN_A_LOCAL_RTX4060.md Stage B section before running this.
# After this succeeds, frozen/ is append-only. You may never edit it, re-tune it, or
# regenerate it to match a result you like better. That property IS the contribution.
. "$PSScriptRoot\_common.ps1"
Enter-Repo

$cfg = Join-Path $REPO "freeze_config_plan_a.json"

Write-Host "`n=== 1. Build the freeze config from completed Stage B runs ===" -ForegroundColor Cyan
& $PY deploy\shared\make_freeze_config.py --plan a --out $cfg
if ($LASTEXITCODE -ne 0) { Write-Host "Cannot build a complete freeze config. Stopping." -ForegroundColor Red; exit 1 }

Write-Host "`n=== 2. DRY RUN - what would be frozen ===" -ForegroundColor Cyan
& $PY experiments\freeze_stage_b.py --config $cfg --dry-run
if ($LASTEXITCODE -ne 0) { Write-Host "Dry run failed. Stopping." -ForegroundColor Red; exit 1 }

Write-Host "`n=== 3. Confirm ===" -ForegroundColor Yellow
Write-Host "This is the pre-registration event. It cannot be undone."
Write-Host "Check above that every cell is present and points at the right run directory."
$answer = Read-Host "`nType FREEZE (all capitals) to proceed, anything else to abort"
if ($answer -cne "FREEZE") { Write-Host "Aborted. Nothing was written." -ForegroundColor Green; exit 0 }

Write-Host "`n=== 4. Freezing ===" -ForegroundColor Cyan
& $PY deploy\shared\make_freeze_config.py --plan a --out $cfg --approve
& $PY experiments\freeze_stage_b.py --config $cfg
if ($LASTEXITCODE -ne 0) { Write-Host "FREEZE FAILED - read the precondition it names." -ForegroundColor Red; exit 1 }

Write-Host "`nFrozen. Record the manifest hash in docs/HUMAN_DECISIONS.md now:" -ForegroundColor Green
Get-Content (Join-Path $REPO "frozen\FREEZE_MANIFEST.json")
Write-Host "`nCommit frozen/ immediately - it is the pre-registration record." -ForegroundColor Green
