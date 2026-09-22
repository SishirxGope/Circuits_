# [AI-GEN] agent=Claude date=2026-09-21 task=Assemble everything to carry to the office/DGX Spark
# Standing project rule: nothing is pushed to a remote, so the project travels as files.
# Run from the repo root on the PC.
$ErrorActionPreference = "Stop"
$REPO = (Get-Location).Path
$OUT  = if ($args.Count -gt 0) { $args[0] } else { Join-Path $env:USERPROFILE "transfer" }
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

Write-Host "`n=== 1. repository as a git bundle (full history) ===" -ForegroundColor Cyan
git bundle create (Join-Path $OUT "cuc-repo.bundle") --all
git bundle verify (Join-Path $OUT "cuc-repo.bundle")

Write-Host "`n=== 2. HuggingFace cache (the big one) ===" -ForegroundColor Cyan
$cache = if ($env:HF_HOME) { $env:HF_HOME } else { Join-Path $env:USERPROFILE ".cache\huggingface" }
if (Test-Path $cache) {
    $size = (Get-ChildItem $cache -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB
    Write-Host ("  copying {0} ({1:N1} GB) - this takes a while" -f $cache, $size)
    Copy-Item $cache (Join-Path $OUT "hf_cache") -Recurse -Force
} else {
    Write-Host "  no cache found at $cache - run 01_fetch_assets.ps1 first" -ForegroundColor Yellow
}

Write-Host "`n=== 3. frozen store (the pre-registration record) ===" -ForegroundColor Cyan
if (Test-Path (Join-Path $REPO "frozen")) {
    Copy-Item (Join-Path $REPO "frozen") (Join-Path $OUT "frozen") -Recurse -Force
    Write-Host "  copied. On the Spark: READ these, do not regenerate them." -ForegroundColor Yellow
}

Write-Host "`n=== 4. upstream forks (if present beside the repo) ===" -ForegroundColor Cyan
$workspace = Split-Path -Parent $REPO
foreach ($f in @("circuit-tracer-0.5.2", "sae-pruning-paper-main")) {
    $src = Join-Path $workspace $f
    if (Test-Path $src) { Copy-Item $src (Join-Path $OUT $f) -Recurse -Force; Write-Host "  copied $f" }
    else { Write-Host "  MISSING: $f - clone it (deploy\shared\fetch_upstream_forks.ps1)" -ForegroundColor Yellow }
}

Write-Host "`n=== 5. baselines to verify against on the Spark ===" -ForegroundColor Cyan
foreach ($b in @("docs\baseline_pc.txt", "docs\baseline_tests.txt")) {
    if (Test-Path (Join-Path $REPO $b)) { Copy-Item (Join-Path $REPO $b) $OUT -Force; Write-Host "  copied $b" }
    else { Write-Host "  MISSING: $b - see PLAN_B section 1.3" -ForegroundColor Yellow }
}

Write-Host "`nBundle ready at: $OUT" -ForegroundColor Green
Get-ChildItem $OUT | Select-Object Name, @{n='Size';e={if($_.PSIsContainer){'<dir>'}else{"{0:N1} MB" -f ($_.Length/1MB)}}} | Format-Table -AutoSize
Write-Host "Copy the whole folder to the Spark, then run deploy/plan_b_dgx_spark/00_inspect_hardware.sh" -ForegroundColor Green
