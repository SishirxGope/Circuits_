#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Plan B step 8 - commit locally and bundle results for the trip home
# Standing project rule: NOTHING is ever pushed to a remote. Results travel as a file.
source "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/_common.sh"
OUT="${1:-$HOME/cuc-results.bundle}"

echo "=== staging results ==="
git add runs/ frozen/ docs/ logs/ 2>/dev/null || true
git status --short | head -30

echo ""
read -r -p "Commit these? [y/N] " a
if [ "$a" = "y" ] || [ "$a" = "Y" ]; then
  git commit -m "results: DGX Spark grid, stages A-D

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" || echo "  nothing to commit"
fi

echo ""
echo "=== bundling ==="
git bundle create "$OUT" --all
git bundle verify "$OUT"
echo ""
echo "Carry back to the PC:"
echo "  $OUT"
echo "  $REPO/runs/     (run directories)"
echo "  $REPO/frozen/   (the pre-registration record - keep cells SEPARATE, never merge)"
