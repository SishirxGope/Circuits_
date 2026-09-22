# [AI-GEN] agent=Claude date=2026-09-21 task=Plan A step 5 - Stage C real compression against the frozen null
# Stage C reads its null from frozen/{model}/{task}/{cell} and refuses on hash mismatch.
# Cheap relative to Stage B: 2 x S = 10 passes per cell.
. "$PSScriptRoot\_common.ps1"
Enter-Repo
Assert-Preflight

if (-not (Test-Path (Join-Path $REPO "frozen\FREEZE_MANIFEST.json"))) {
    Write-Host "No freeze manifest. Run 04_freeze.ps1 first - Stage C has no denominator without it." -ForegroundColor Red
    exit 1
}

$log   = New-LogDir "stageC"
$cells = Get-Cells "$PSScriptRoot\cells_stagec.txt"
Write-Host "Stage C: $($cells.Count) cells queued.`n" -ForegroundColor Cyan

$i = 0; $failed = @()
foreach ($cell in $cells) {
    $i++
    $tag = ($cell -replace '[= ]', '_')
    Write-Host "[$i/$($cells.Count)] $cell" -ForegroundColor Cyan
    $hydraArgs = $COMMON + ($cell -split ' ') + @("seed=0")
    & $PY experiments\run_stage_c.py @hydraArgs 2>&1 | Tee-Object "$log\$tag.log"
    if ($LASTEXITCODE -ne 0) { $failed += $cell; Write-Host "  FAILED (continuing)" -ForegroundColor Red }
}
if ($failed.Count -gt 0) {
    Write-Host "`n$($failed.Count) cell(s) failed:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host "`nStage C complete." -ForegroundColor Green
