# [AI-GEN] agent=Claude date=2026-09-21 task=Clone the two upstream forks at their Q9-pinned commits (Windows)
# URLs and pins are quoted from docs/HUMAN_DECISIONS.md section 3.3 (verified 2026-09-12).
# CLAUDE.md places these BESIDE the repo, at the workspace root - not inside it.
$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent (Get-Location).Path

$forks = @(
  @{ name = "circuit-tracer-0.5.2";   url = "https://github.com/decoderesearch/circuit-tracer.git"; pin = "8f1e2438df612464e229e44c4a00ff637bf9379b"; note = "canonical URL; safety-research/* 301-redirects here" },
  @{ name = "sae-pruning-paper-main"; url = "https://github.com/hecboar/sae-pruning-paper.git";     pin = "261191804675e2d39d0a265320dbc0bc85afd30a"; note = "PIN IS INFERRED (HUMAN_DECISIONS 3.3) - verify before trusting it" }
)

foreach ($f in $forks) {
    $dest = Join-Path $workspace $f.name
    Write-Host "`n  $($f.name)" -ForegroundColor Cyan
    Write-Host "    note: $($f.note)"
    if (Test-Path $dest) { Write-Host "    already present at $dest" -ForegroundColor Green; continue }
    git clone $f.url $dest
    Push-Location $dest
    git checkout $f.pin
    $head = (git rev-parse HEAD).Trim()
    Pop-Location
    if ($head -ne $f.pin) { Write-Host "    PIN MISMATCH: got $head" -ForegroundColor Red }
    else { Write-Host "    pinned at $($f.pin.Substring(0,12))" -ForegroundColor Green }
}
Write-Host "`n  Verify the cross-audit table exists:"
Write-Host "    $workspace\sae-pruning-paper-main\results\E6\stat_tests.csv"
