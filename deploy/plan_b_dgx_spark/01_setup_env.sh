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
echo "=== 3. optional quantization libraries (expected to be the hard part on ARM) ==="
if pip install auto-gptq autoawq 2>/dev/null && python -c "import auto_gptq, awq" 2>/dev/null; then
  echo "  auto-gptq + autoawq OK"
else
  echo "  auto-gptq / autoawq NOT available on this platform."
  echo "  This is the documented ARM64 failure. Options, in order (PLAN_B section 4):"
  echo "    1. use the in-repo torch implementations of GPTQ/AWQ (no build needed)"
  echo "    2. build from source with TORCH_CUDA_ARCH_LIST set to the value printed above"
  echo "    3. fall back to the libraries' CPU kernels (slow but scientifically equivalent)"
  echo "    4. cut the GPTQ/AWQ cells and RECORD the cut in docs/HUMAN_DECISIONS.md"
fi

echo ""
echo "=== 4. project test suite ==="
python -m pytest -q
