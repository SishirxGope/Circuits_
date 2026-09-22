#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Plan B step 0 - record what this machine actually is
# Install nothing yet. Every later estimate depends on these numbers.
# Output goes to docs/spark_environment.md - keep it, it is part of the artifact record.
set -euo pipefail
OUT="docs/spark_environment.md"
mkdir -p docs

{
  echo "# Spark environment (recorded $(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo ""
  echo '## Architecture'
  echo '```'
  echo "uname -m      : $(uname -m)"
  echo "uname -a      : $(uname -a)"
  echo '```'
  echo ""
  echo '## OS'
  echo '```'
  cat /etc/os-release 2>/dev/null || echo "no /etc/os-release"
  echo '```'
  echo ""
  echo '## GPU'
  echo '```'
  nvidia-smi 2>&1 || echo "nvidia-smi not available"
  echo '```'
  echo ""
  echo '## Memory / CPU / disk'
  echo '```'
  free -g 2>/dev/null || echo "free unavailable"
  echo "cores: $(nproc 2>/dev/null || echo '?')"
  df -h "$HOME" 2>/dev/null
  echo '```'
  echo ""
  echo '## Toolchain'
  echo '```'
  echo "python3 : $(python3 --version 2>&1)"
  echo "docker  : $(docker --version 2>&1 || echo 'not installed')"
  echo "git     : $(git --version 2>&1)"
  echo '```'
} | tee "$OUT"

echo ""
echo "Written to $OUT"
echo ""
if [ "$(uname -m)" = "aarch64" ]; then
  echo "ARM64 confirmed -> follow Plan B sections 3 and 4 in full (container route recommended,"
  echo "and expect GPTQ/AWQ build trouble)."
else
  echo "Not ARM64 -> the environment build is easier; Plan A's setup mostly applies."
fi
