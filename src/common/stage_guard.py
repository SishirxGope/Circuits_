# [AI-GEN] agent=OpenCode date=2026-08-07 task=Stage guard: Stage C refuses to run without frozen Stage B artifacts
# reviewed-by: PENDING

"""Stage-ordering and frozen-store integrity guards.

Enforces, in code (AI_RULES.md 1.4, 1.5; ARCHITECTURE.md §1 dotted edge):

1. **Null before effect** — ``assert_stage_b_frozen``: a Stage C/D cell may only run
   once its Stage B (null) artifacts exist, are registered in the frozen-store
   manifest, and carry a ``null_frozen_hash``.
2. **Frozen-store integrity** — ``assert_null_frozen_hash``: the exact hash used in a
   CSI division must match the frozen cell's recorded hash, or the computation fails
   hard.

Frozen-store layout (ARCHITECTURE.md §2)::

    frozen/{model}/{task}/{compression_cell}/
        meta.json          # FrozenMeta schema (src/common/schema.py)
        dnull.parquet      # the frozen D_null distribution
    frozen/FREEZE_MANIFEST.json
        {"cells": {"{model}/{task}/{cell}": {"null_frozen_hash": "<sha256>", ...}},
         "freeze_commit": "<git ref>", "frozen_at": "<iso>"}

The manifest is written at the null-freeze commit; after that the directory is
append-only (new cells for new grid cells only), never edited (AI_RULES.md 1.4).
"""

from __future__ import annotations

import json
from pathlib import Path

FREEZE_MANIFEST_NAME = "FREEZE_MANIFEST.json"
FROZEN_META_NAME = "meta.json"
DNULL_FILE_NAME = "dnull.parquet"


def _load_json(path: Path, what: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {what} at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{what} at {path} must be a JSON object")
    return data


def assert_stage_b_frozen(cell_path: str | Path) -> None:
    """Raise RuntimeError unless the Stage B null artifacts for ``cell_path`` are frozen.

    ``cell_path`` = ``frozen/{model}/{task}/{compression_cell}`` (ARCHITECTURE.md §2).
    Checks, in order:
    1. ``meta.json`` and ``dnull.parquet`` exist.
    2. ``meta.json`` parses and carries a non-empty ``null_frozen_hash``.
    3. ``frozen/FREEZE_MANIFEST.json`` exists (a store without a manifest is not
       frozen) and registers this cell with a matching ``null_frozen_hash``.
    """
    cell = Path(cell_path)
    meta = cell / FROZEN_META_NAME
    dnull = cell / DNULL_FILE_NAME

    missing = [str(p) for p in (meta, dnull) if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Stage B not frozen for cell {cell}: missing {missing}. "
            f"Null before effect (AI_RULES.md 1.5); run Stage B and freeze before any Stage C cell."
        )

    meta_data = _load_json(meta, "frozen meta.json")
    null_hash = meta_data.get("null_frozen_hash")
    if not null_hash:
        raise RuntimeError(f"frozen meta.json at {meta} has no null_frozen_hash; cell is not frozen")

    store_root = cell.parents[2]  # frozen/<model>/<task>/<cell> -> frozen
    manifest_path = store_root / FREEZE_MANIFEST_NAME
    if not manifest_path.exists():
        raise RuntimeError(
            f"frozen store {store_root} has no {FREEZE_MANIFEST_NAME}; store is not frozen "
            f"(manifest is created at the freeze commit, AI_RULES.md 1.4)"
        )
    manifest = _load_json(manifest_path, "FREEZE_MANIFEST.json")
    cells = manifest.get("cells", {})
    rel = cell.relative_to(store_root).as_posix()
    entry = cells.get(rel) if isinstance(cells, dict) else None
    if entry is None:
        raise RuntimeError(f"cell {rel} is not registered in {manifest_path}; cell is not frozen")
    if entry.get("null_frozen_hash") != null_hash:
        raise RuntimeError(
            f"hash mismatch for cell {rel}: meta.json={null_hash} vs FREEZE_MANIFEST.json="
            f"{entry.get('null_frozen_hash')}"
        )


def assert_null_frozen_hash(meta_path: str | Path, expected_hash: str) -> None:
    """Raise RuntimeError unless the frozen cell's ``null_frozen_hash`` matches.

    Every CSI computation must verify the hash of the null it divides by, and fail
    hard on mismatch (AI_RULES.md 1.4).
    """
    meta = Path(meta_path)
    data = _load_json(meta, "frozen meta.json")
    actual = data.get("null_frozen_hash")
    if not actual:
        raise RuntimeError(f"frozen meta.json at {meta} has no null_frozen_hash")
    if actual != expected_hash:
        raise RuntimeError(
            f"null_frozen_hash mismatch: expected {expected_hash}, got {actual} "
            f"(frozen store integrity violated; AI_RULES.md 1.4)"
        )
