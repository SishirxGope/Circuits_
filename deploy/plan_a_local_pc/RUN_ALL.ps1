# [AI-GEN] agent=Claude date=2026-09-21 task=Plan A one-command orchestrator (stops only where a human must decide)
#
#   .\deploy\plan_a_local_pc\RUN_ALL.ps1
#
# Runs every step in order. Resumable: re-run it after any interruption and it picks up
# where it stopped. It pauses at exactly ONE place - the freeze - because that is the
# pre-registration event and a machine must not perform it unattended.
#
#   -SkipFetch    don't re-download assets
#   -StopBefore   stop before a named step, e.g. -StopBefore freeze
param([switch]$SkipFetch, [string]$StopBefore = "")
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\_common.ps1"
Enter-Repo
$started = Get-Date
$runlog  = New-LogDir "run_all"
$transcript = Join-Path $runlog ("run_all_{0:yyyyMMdd_HHmmss}.log" -f $started)
Start-Transcript -Path $transcript | Out-Null

function Step {
    param([string]$Name, [scriptblock]$Body)
    if ($StopBefore -eq $Name) {
        Write-Host "`n=== STOPPING BEFORE '$Name' as requested ===" -ForegroundColor Yellow
        Stop-Transcript | Out-Null
        exit 0
    }
    Write-Host "`n$('='*70)" -ForegroundColor Cyan
    Write-Host "STEP: $Name   ($(Get-Date -Format 'HH:mm:ss'))" -ForegroundColor Cyan
    Write-Host "$('='*70)" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nSTOPPED at '$Name'. Fix the problem and re-run this script -" -ForegroundColor Red
        Write-Host "completed work is skipped, so you lose nothing." -ForegroundColor Red
        Stop-Transcript | Out-Null
        exit 1
    }
}

Write-Host @"

  Plan A - full pipeline, Pythia-160M and Pythia-410M
  Log: $transcript

  This will take DAYS of GPU time. Stage B alone is ~16 hours for Pythia-160M.
  It pauses once, at the freeze. Everything else runs unattended.

"@ -ForegroundColor Green

Step "verify"   { & "$PSScriptRoot\00_repair_and_verify.ps1" }

# The preflight gate. Everything past here needs real-model code that may not exist yet.
& $PY deploy\shared\preflight_blockers.py
if ($LASTEXITCODE -ne 0) {
    Write-Host @"

  STOPPING: the real-model code paths are still stubs, so there is nothing to run.
  This is an engineering gap, not a configuration problem. See deploy\BLOCKERS.md.

  Environment and repository are healthy - that part passed.

"@ -ForegroundColor Yellow
    Stop-Transcript | Out-Null
    exit 1
}

if (-not $SkipFetch) { Step "fetch" { & "$PSScriptRoot\01_fetch_assets.ps1" } }
Step "stageA"   { & "$PSScriptRoot\02_stage_a.ps1" }
Step "stageB"   { & "$PSScriptRoot\03_stage_b.ps1" }

# ---- the one human gate -------------------------------------------------------------
Write-Host @"

$('='*70)
  THE FREEZE - this is the pre-registration event.

  Stage B is complete. Freezing writes the null distribution that every CSI
  number will be divided by, and frozen/ is append-only forever afterwards.

  A script must not do this unattended. 04_freeze.ps1 will show you the plan
  and ask you to type FREEZE.
$('='*70)

"@ -ForegroundColor Yellow

Step "freeze"   { & "$PSScriptRoot\04_freeze.ps1" }

if (-not (Test-Path (Join-Path $REPO "frozen\FREEZE_MANIFEST.json"))) {
    Write-Host "No freeze manifest - you aborted the freeze. Stopping here." -ForegroundColor Yellow
    Stop-Transcript | Out-Null
    exit 0
}

Step "stageC"   { & "$PSScriptRoot\05_stage_c.ps1" }
Step "stageD"   { & "$PSScriptRoot\06_stage_d_analysis.ps1" }

$elapsed = (Get-Date) - $started
Write-Host "`n$('='*70)" -ForegroundColor Green
Write-Host ("COMPLETE in {0:hh\:mm\:ss}" -f $elapsed) -ForegroundColor Green
Write-Host "Now work through deploy\plan_a_local_pc\CHECKLIST.md - the science is not" -ForegroundColor Green
Write-Host "done until the diagnostics and both comparison levels have been inspected." -ForegroundColor Green
Stop-Transcript | Out-Null
