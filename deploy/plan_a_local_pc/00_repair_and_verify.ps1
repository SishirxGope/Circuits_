# [AI-GEN] agent=Claude date=2026-09-21 task=Plan A step 0 - repair the repo and verify the environment
# Run this FIRST, every time you sit down to work. It changes nothing scientific.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$PY = ".venv\Scripts\python.exe"

Write-Host "`n=== 1. Git index ===" -ForegroundColor Cyan
$tracked = (git ls-files | Measure-Object -Line).Lines
if ($tracked -eq 0) {
    Write-Host "  index is empty - rebuilding from HEAD (working tree untouched)" -ForegroundColor Yellow
    git reset
    $tracked = (git ls-files | Measure-Object -Line).Lines
}
Write-Host "  tracked files: $tracked  (expect ~156+)"

Write-Host "`n=== 2. Stray large files in the repo root ===" -ForegroundColor Cyan
Get-ChildItem -File | Where-Object { $_.Length -gt 1MB } | ForEach-Object {
    Write-Host ("  LARGE: {0}  {1:N1} MB - move it out before committing" -f $_.Name, ($_.Length/1MB)) -ForegroundColor Yellow
}

Write-Host "`n=== 3. GPU and torch ===" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
& $PY -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

Write-Host "`n=== 4. Test suite (regression gate) ===" -ForegroundColor Cyan
& $PY -m pytest -q
if ($LASTEXITCODE -ne 0) { Write-Host "  TESTS FAILED - fix before running anything scientific" -ForegroundColor Red; exit 1 }

Write-Host "`n=== 5. Real-model blockers ===" -ForegroundColor Cyan
& $PY deploy\shared\preflight_blockers.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n  Environment is healthy, but the science cannot run yet." -ForegroundColor Yellow
    Write-Host "  See deploy\BLOCKERS.md. Steps 02-06 will refuse until these are implemented." -ForegroundColor Yellow
}
Write-Host "`nDone.`n"

# This script reports on blockers; it does not fail because of them. The orchestrator
# (RUN_ALL.ps1) runs its own preflight gate and decides whether to continue.
exit 0
