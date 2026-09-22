# [AI-GEN] agent=Claude date=2026-09-21 task=Plan A step 6 - Stage D aggregation, CSI table, and the analyses
. "$PSScriptRoot\_common.ps1"
Enter-Repo

$stageC = Get-ChildItem (Join-Path $REPO "runs") -Directory | Where-Object { $_.Name -like "*_stageC_*" }
if ($stageC.Count -eq 0) { Write-Host "No Stage C runs found. Run 05_stage_c.ps1 first." -ForegroundColor Red; exit 1 }
Write-Host "Aggregating $($stageC.Count) Stage C runs.`n" -ForegroundColor Cyan

$dirs = ($stageC | ForEach-Object { "runs/$($_.Name)" }) -join ","
$log  = New-LogDir "stageD"

& $PY experiments\run_stage_d.py mode=scientific_run stage=stageD "stage_d.stage_c_dirs=[$dirs]" 2>&1 | Tee-Object "$log\stage_d.log"
if ($LASTEXITCODE -ne 0) { Write-Host "Stage D failed." -ForegroundColor Red; exit 1 }

Write-Host "`n=== Threshold sweep (does each conclusion survive the full range?) ===" -ForegroundColor Cyan
& $PY analysis\threshold_sweep.py 2>&1 | Tee-Object "$log\threshold_sweep.log"

Write-Host "`n=== Cross-audit (C5) ===" -ForegroundColor Cyan
Write-Host "REMINDER: correlate against the frozen CSV tables of the feature-level paper," -ForegroundColor Yellow
Write-Host "NEVER the arXiv PDF. The authors corrected rho = -1.0 to a range of -0.540..0.062." -ForegroundColor Yellow
& $PY analysis\cross_audit.py 2>&1 | Tee-Object "$log\cross_audit.log"

Write-Host "`nDone. Every number in the paper must trace to a run directory in runs/." -ForegroundColor Green
