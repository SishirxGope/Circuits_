# [AI-GEN] agent=Claude date=2026-09-21 task=Plan A step 1 - fetch pinned models, corpora and the upstream forks
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$PY = ".venv\Scripts\python.exe"

& $PY deploy\shared\fetch_assets.py --plan a --datasets

Write-Host "`n=== Upstream forks (pinned, Q9) ===" -ForegroundColor Cyan
Write-Host "  Needed for: magnitude/wanda real-model pruning (they wrap saediag.pruning)"
Write-Host "              and the C5 cross-audit frozen tables (results/E6/stat_tests.csv)"
& powershell -ExecutionPolicy Bypass -File "deploy\shared\fetch_upstream_forks.ps1"
Write-Host "`nDone.`n"
