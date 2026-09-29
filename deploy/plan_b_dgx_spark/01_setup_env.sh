#!/usr/bin/env bash
# [AI-GEN] agent=Claude date=2026-09-21 task=Plan B step 1 - build the environment (native venv route)
# For the CONTAINER route (recommended on ARM), see docs/PLAN_B_DGX_SPARK.md section 3 Route A;
# run this script INSIDE the container to install the project itself.
set -euo pipefail

echo "=== 1. torch first, and verified before anything else ==="
if ! python -c "import torch" 2>/dev/null; then
  echo "  torch not present."
  echo "  Install the build that matches this machine, then re-run. On aarch64 prefer the"
  echo "  NGC container (nvcr.io/nvidia/pytorch) over pip - see PLAN_B section 3."
  exit 1
fi
python - <<'PYEOF'
import torch, sys
print("  torch", torch.__version__)
print("  cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device:", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print(f"  compute capability: {cap[0]}.{cap[1]}   (TORCH_CUDA_ARCH_LIST={cap[0]}.{cap[1]})")
else:
    print("  STOP: CUDA is not available. Fix this before installing anything else -")
    print("  every later failure will be a confusing symptom of this one.")
    sys.exit(1)
PYEOF

echo ""
echo "=== 2. project dependencies ==="
pip install -e .

echo ""
echo "=== 3. quantization libraries: none needed ==="
# GPTQ and AWQ are torch ports in src/compression/{gptq,awq}.py (2026-09-29), so
# auto-gptq/autoawq are deliberately NOT installed: they do not build on aarch64, and a
# pip install that half-succeeds can replace the CUDA torch this script just checked.
if python -c "import auto_gptq" 2>/dev/null || python -c "import awq" 2>/dev/null; then
  echo "  NOTE: auto-gptq/autoawq are installed but unused; the pipeline never imports them."
else
  echo "  OK - in-repo GPTQ/AWQ, nothing to install."
fi

echo ""
echo "=== 4. project test suite ==="
python -m pytest -q
