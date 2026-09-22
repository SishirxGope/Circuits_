# [AI-GEN] agent=Claude date=2026-09-21 task=Plan A step 3 - Stage B null draws (the expensive stage)
# 44 cells x R=20 draws x S=5 seeds. This is ~16 GPU-hours for Pythia-160M alone.
# Resumable: cells whose run directory already exists are skipped.
. "$PSScriptRoot\_common.ps1"
Enter-Repo
Assert-Preflight
$log   = New-LogDir "stageB"
$cells = Get-Cells "$PSScriptRoot\cells_stageb.txt"

Write-Host "Stage B: $($cells.Count) cells queued." -ForegroundColor Cyan
Write-Host "One GPU runs one cell at a time on 8 GB - this is a multi-day run. Use a" -ForegroundColor Yellow
Write-Host "terminal you can leave open, and expect to resume after interruptions.`n" -ForegroundColor Yellow

$i = 0
foreach ($cell in $cells) {
    $i++
    $tag = ($cell -replace '[= ]', '_')
    $logfile = "$log\$tag.log"
    if (Test-Path $logfile) {
        if (Select-String -Path $logfile -Pattern "STAGE B COMPLETE|run_dir" -Quiet) {
            Write-Host "[$i/$($cells.Count)] skip (done): $cell"; continue
        }
    }
    Write-Host "[$i/$($cells.Count)] $cell" -ForegroundColor Cyan
    $hydraArgs = $COMMON + ($cell -split ' ') + @("seed=0")
    & $PY experiments\run_stage_b.py @hydraArgs 2>&1 | Tee-Object $logfile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED at cell $i. Fix, then re-run this script - completed cells are skipped." -ForegroundColor Red
        exit 1
    }
}
Write-Host "`nStage B drafts complete. NOTHING IS FROZEN YET - run 04_freeze.ps1." -ForegroundColor Green
