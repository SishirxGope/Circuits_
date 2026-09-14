# [AI-GEN] agent=OpenCode date=2026-08-07 task=Test path bootstrap
# reviewed-by: PENDING

"""Ensure the project root is importable (src.*, experiments.*) when pytest runs."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
