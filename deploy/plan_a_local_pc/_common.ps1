# [AI-GEN] agent=Claude date=2026-09-21 task=Plan A shared helpers (dot-sourced by every stage script)
$ErrorActionPreference = "Stop"
$script:REPO = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$script:PY   = Join-Path $REPO ".venv\Scripts\python.exe"

function Enter-Repo { Set-Location $script:REPO }

function Assert-Preflight {
    # Refuse to run science against the stubs. See deploy/BLOCKERS.md.
    & $script:PY (Join-Path $script:REPO "deploy\shared\preflight_blockers.py") --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nSTOPPED: real-model blockers are not implemented." -ForegroundColor Red
        Write-Host "Run: .venv\Scripts\python.exe deploy\shared\preflight_blockers.py" -ForegroundColor Red
        exit 1
    }
}

function New-LogDir {
    param([string]$Stage)
    $dir = Join-Path $script:REPO "logs\$Stage"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    return $dir
}

function Get-Cells {
    param([string]$File)
    Get-Content $File | Where-Object { $_.Trim() -ne "" -and -not $_.StartsWith("#") }
}

# The overrides every scientific run shares. mode=scientific_run is what makes the
# guards strict; pipeline=dense-node is the proven Pythia path.
$script:COMMON = @(
    "mode=scientific_run",
    "pipeline=dense-node",
    "ensemble=default",
    "ensemble/decompose=final",
    "nulls=default",
    "comparison=final",
    "comparison_level=both"
)
