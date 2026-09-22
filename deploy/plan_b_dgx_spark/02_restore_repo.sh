#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Plan B step 2 - restore the repo, caches and frozen store from the transfer bundle
# You carried these from the PC (see deploy/shared/make_transfer_bundle.ps1). Nothing is
# pushed or pulled from a remote - that is a standing project rule.
set -euo pipefail
BUNDLE="${1:-$HOME/transfer/cuc-repo.bundle}"
CACHE="${2:-$HOME/transfer/hf_cache}"
DEST="${3:-$HOME/Circuits_Under_Compression}"

echo "=== 1. repository ==="
if [ -d "$DEST/.git" ]; then
  echo "  $DEST already exists - fetching from the bundle instead of cloning"
  ( cd "$DEST" && git fetch "$BUNDLE" '+refs/heads/*:refs/remotes/bundle/*' && git log --oneline -3 )
else
  git clone "$BUNDLE" "$DEST"
  ( cd "$DEST" && git log --oneline -5 )
fi

echo ""
echo "=== 2. HuggingFace cache ==="
if [ -d "$CACHE" ]; then
  mkdir -p "$HOME/hf_cache"
  cp -rn "$CACHE/." "$HOME/hf_cache/"
  echo "  copied to $HOME/hf_cache"
  echo "  add this to your shell profile:  export HF_HOME=$HOME/hf_cache"
else
  echo "  no cache at $CACHE - models will download on first use (needs licences accepted)"
fi

echo ""
echo "=== 3. frozen store (the pre-registration record) ==="
if [ -d "$HOME/transfer/frozen" ]; then
  mkdir -p "$DEST/frozen"
  cp -rn "$HOME/transfer/frozen/." "$DEST/frozen/"
  echo "  copied. NEVER regenerate these to make them match - frozen/ is append-only."
  ls "$DEST/frozen"
else
  echo "  none carried over - this machine will create its own freeze."
  echo "  CHECK the freeze-ownership table in docs/HUMAN_DECISIONS.md first:"
  echo "  two machines freezing the same cell cannot be undone."
fi

echo ""
echo "=== 4. upstream forks ==="
( cd "$DEST" && bash deploy/shared/fetch_upstream_forks.sh ) || \
  echo "  fetch failed - carry them from the PC or clone manually (see PLAN_B)"
