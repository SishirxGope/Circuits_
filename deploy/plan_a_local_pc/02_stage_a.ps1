# [AI-GEN] agent=Claude date=2026-09-21 task=Plan A step 2 - Stage A dense reference (Pythia)
# Stage A establishes the pre-compression inclusion-frequency vector that every later
# distance is measured against. Cheap: S=5 passes per model-task.
. "$PSScriptRoot\_common.ps1"
Enter-Repo
Assert-Preflight
$log = New-LogDir "stageA"

foreach ($model in @("pythia160m", "pythia410m")) {
  foreach ($task in @("ioi", "greater_than")) {
    Write-Host "`n=== Stage A: $model / $task ===" -ForegroundColor Cyan
    $hydraArgs = $COMMON + @("stage=stageA", "model=$model", "task=$task",
                        "setting=dense", "compression_family=dense", "seed=0")
    & $PY experiments\run_stage_a.py @hydraArgs 2>&1 | Tee-Object "$log\${model}_${task}.log"
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $model/$task - see $log" -ForegroundColor Red; exit 1 }
  }
}
Write-Host "`nStage A complete. Check the CIRCUS grid diagnostics before freezing:" -ForegroundColor Green
Write-Host "  mean pairwise Jaccard across the B views (near 1.0 = configs are near-duplicates)"
Write-Host "  Match rate: consensus == a single view (~100% = the nesting artifact)"
