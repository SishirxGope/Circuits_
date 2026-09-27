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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


# [AI-GEN] agent=Claude date=2026-09-27 task=Guard 3 - the null must be matched to the cell it divides
def assert_null_matched_to_cell(meta_path: str | Path, expected: Mapping[str, Any]) -> None:
    """Raise RuntimeError unless the frozen null was matched to THIS compression cell.

    ``assert_null_frozen_hash`` proves the null file is intact. It cannot prove the null
    was matched to the compression now being divided by it: a null drawn against
    magnitude-0.3 magnitudes hashes perfectly well while being the wrong denominator for
    an rtn_int8 cell. That is not hypothetical - Stage B hardcoded a single stand-in
    compressor for every cell until 2026-09-27, so this guard is what keeps the fix
    enforced rather than merely applied (AI_RULES.md 1.4, 1.5; ARCHITECTURE.md §4).

    ``expected`` is ``src.compression.registry.compression_provenance(resolved)``.
    """
    meta = Path(meta_path)
    data = _load_json(meta, "frozen meta.json")

    if "compression_family" not in data:
        raise RuntimeError(
            f"frozen meta.json at {meta} records no compression provenance, so it cannot "
            "be shown to match this cell. Nulls written before 2026-09-27 were all matched "
            "to a hardcoded magnitude-prune stand-in regardless of their cell; re-run "
            "Stage B for this cell and re-freeze."
        )

    mismatched = {
        key: (data.get(key), expected[key])
        for key in ("compression_family", "compression_level", "compressor_kwargs")
        if _comparable(data.get(key)) != _comparable(expected.get(key))
    }
    if mismatched:
        detail = "; ".join(
            f"{k}: frozen={frozen!r} but this run declares {want!r}"
            for k, (frozen, want) in sorted(mismatched.items())
        )
        raise RuntimeError(
            f"the frozen null at {meta} was matched to a different compression than this "
            f"run measures ({detail}). CSI would divide D(c) by the median of the wrong "
            "D_null (ARCHITECTURE.md §4)."
        )


def _comparable(value: Any) -> Any:
    """Normalize for comparison across the JSON round-trip and Hydra's string coercion.

    Hydra hands ``+stage_c.compressor_kwargs.bits=8`` through as the string ``"8"`` in
    some paths and the int ``8`` in others, and levels appear as both. Comparing raw
    values would raise a spurious mismatch on a null that is in fact correctly matched.
    """
    if isinstance(value, Mapping):
        return {str(k): _comparable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        return text.lower()
