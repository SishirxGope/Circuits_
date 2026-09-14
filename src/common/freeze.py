# [AI-GEN] agent=Claude date=2026-08-08 task=Implement the Stage B freeze WRITER (the guard existed; nothing wrote frozen/)
# reviewed-by: PENDING

"""The null freeze — the pre-registration event (AI_RULES.md 1.4; proposal §4 rule 1).

`src/common/stage_guard.py` READS the frozen store and refuses Stage C when a cell is
not frozen. Until now nothing WROTE it: the single most load-bearing methodological
commitment in the project had no implementation. This module is that writer.

What "frozen" means here, enforced mechanically:

1. **Append-only.** A cell directory that already exists is never overwritten, never
   regenerated, never "corrected". `freeze_cell` raises on collision. New cells may be
   appended for grid cells whose real-compression counterpart has not yet run; nothing
   else may change (AI_RULES.md 1.4).
2. **Hash-verified on the way in.** The dnull file is copied, then re-hashed at the
   destination, and the recomputed hash must equal the hash recorded in the source
   meta. A corrupted copy fails loudly instead of silently becoming the floor that
   every CSI divides by.
3. **The manifest is the registry.** `frozen/FREEZE_MANIFEST.json` lists every frozen
   cell and its `null_frozen_hash`. `assert_stage_b_frozen` refuses any cell that is
   not registered, so writing the cell directory alone is not enough.
4. **The freeze is a commit.** `freeze_commit` records the git HEAD at freeze time.
   Freezing without git is possible but loses the pre-registration anchor, so it warns.

This module deliberately does NOT decide when to freeze. `experiments/freeze_stage_b.py`
owns the policy (scientific mode + explicit approval + no open decisions); this module
owns the mechanics, so the mechanics can be tested without a full scientific config.
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path
from typing import Any

from .hashing import hash_file
from .schema import validate_frozen_meta
from .stage_guard import DNULL_FILE_NAME, FREEZE_MANIFEST_NAME, FROZEN_META_NAME


class FreezeViolation(RuntimeError):
    """Raised when an operation would violate the append-only freeze (AI_RULES.md 1.4)."""


def cell_key(model: str, task: str, cell: str) -> str:
    """The manifest key for a cell: ``"{model}/{task}/{cell}"`` (posix, stable)."""
    return f"{model}/{task}/{cell}"


def read_manifest(frozen_root: str | Path) -> dict[str, Any]:
    """Read FREEZE_MANIFEST.json, or return an empty unfrozen manifest."""
    path = Path(frozen_root) / FREEZE_MANIFEST_NAME
    if not path.exists():
        return {"frozen": False, "freeze_commit": None, "frozen_at": None, "cells": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise FreezeViolation(f"{path} must be a JSON object")
    data.setdefault("cells", {})
    return data


def freeze_cell(
    frozen_root: str | Path,
    model: str,
    task: str,
    cell: str,
    source_run_dir: str | Path,
    *,
    freeze_commit: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze ONE Stage-B cell from a draft run directory. Append-only.

    Copies ``dnull.parquet`` + ``meta.json`` from ``source_run_dir`` into
    ``frozen/{model}/{task}/{cell}/`` and registers the cell in the manifest.

    Raises FreezeViolation if the cell already exists (never overwrite), if the source
    is incomplete, or if the copied dnull file's hash does not reproduce the hash the
    source meta recorded.

    Returns the frozen cell's meta dict.
    """
    src = Path(source_run_dir)
    src_dnull, src_meta_path = src / DNULL_FILE_NAME, src / FROZEN_META_NAME
    missing = [str(p) for p in (src_dnull, src_meta_path) if not p.exists()]
    if missing:
        raise FreezeViolation(
            f"cannot freeze from {src}: missing {missing}. Run Stage B for this cell first."
        )

    with open(src_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    validate_frozen_meta(meta)

    recorded_hash = meta["null_frozen_hash"]
    if hash_file(src_dnull) != recorded_hash:
        raise FreezeViolation(
            f"source dnull hash does not match the hash in {src_meta_path}: the draft "
            f"was modified after it was written. Re-run Stage B rather than freezing it."
        )

    dest = Path(frozen_root) / model / task / cell
    if dest.exists():
        raise FreezeViolation(
            f"cell {cell_key(model, task, cell)} is ALREADY frozen at {dest}. frozen/ is "
            f"append-only: a frozen null is never regenerated or corrected (AI_RULES.md "
            f"1.4). If the null is wrong, the affected cells are re-run from Stage B "
            f"into a NEW cell name and the change is disclosed in the paper."
        )

    manifest = read_manifest(frozen_root)
    key = cell_key(model, task, cell)
    if key in manifest.get("cells", {}):
        raise FreezeViolation(
            f"cell {key} is already registered in {FREEZE_MANIFEST_NAME} but its directory "
            f"is missing — the frozen store is inconsistent. Investigate before freezing."
        )

    dest.mkdir(parents=True)
    shutil.copy2(src_dnull, dest / DNULL_FILE_NAME)

    # Re-hash at the destination: a copy that did not survive intact must never become
    # the denominator of a CSI.
    copied_hash = hash_file(dest / DNULL_FILE_NAME)
    if copied_hash != recorded_hash:
        shutil.rmtree(dest)
        raise FreezeViolation(
            f"copied dnull hash {copied_hash} != source hash {recorded_hash}; the copy is "
            f"corrupt and the partial cell was removed. Nothing was frozen."
        )

    frozen_meta = {
        **meta,
        **(extra_meta or {}),
        "frozen": True,
        "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "freeze_commit": freeze_commit,
        "source_run_dir": str(src),
        "cell": key,
    }
    with open(dest / FROZEN_META_NAME, "w", encoding="utf-8") as f:
        json.dump(frozen_meta, f, indent=2, sort_keys=True)

    _append_to_manifest(frozen_root, key, recorded_hash, freeze_commit)
    return frozen_meta


def _append_to_manifest(
    frozen_root: str | Path, key: str, null_frozen_hash: str, freeze_commit: str | None
) -> None:
    """Append one cell to the manifest. Existing entries are never rewritten."""
    manifest = read_manifest(frozen_root)
    cells = manifest.setdefault("cells", {})
    if key in cells:  # pragma: no cover - guarded by freeze_cell before we get here
        raise FreezeViolation(f"cell {key} already registered; the manifest is append-only")
    cells[key] = {
        "null_frozen_hash": null_frozen_hash,
        "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    manifest["frozen"] = True
    manifest["frozen_at"] = manifest.get("frozen_at") or cells[key]["frozen_at"]
    if freeze_commit and not manifest.get("freeze_commit"):
        manifest["freeze_commit"] = freeze_commit

    path = Path(frozen_root) / FREEZE_MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def verify_frozen_store(frozen_root: str | Path) -> dict[str, Any]:
    """Re-verify the whole frozen store against its manifest.

    Recomputes every cell's dnull hash and compares it to both the cell meta and the
    manifest. Returns {"ok": bool, "n_cells": int, "problems": [...]}. Run this before
    the freeze commit, before any Stage C batch, and at artifact build time — it is the
    cheapest possible check that the floor has not moved under the results.
    """
    root = Path(frozen_root)
    manifest = read_manifest(root)
    problems: list[str] = []
    cells = manifest.get("cells", {})

    for key, entry in sorted(cells.items()):
        cell_dir = root / Path(key)
        meta_path, dnull_path = cell_dir / FROZEN_META_NAME, cell_dir / DNULL_FILE_NAME
        if not meta_path.exists() or not dnull_path.exists():
            problems.append(f"{key}: registered in the manifest but files are missing")
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        actual = hash_file(dnull_path)
        if actual != meta.get("null_frozen_hash"):
            problems.append(f"{key}: dnull.parquet hash {actual} != meta {meta.get('null_frozen_hash')}")
        if actual != entry.get("null_frozen_hash"):
            problems.append(f"{key}: dnull.parquet hash {actual} != manifest {entry.get('null_frozen_hash')}")

    # cells on disk that nobody registered are as dangerous as missing ones
    if root.exists():
        for meta_path in sorted(root.rglob(FROZEN_META_NAME)):
            key = meta_path.parent.relative_to(root).as_posix()
            if key not in cells:
                problems.append(f"{key}: present on disk but NOT registered in {FREEZE_MANIFEST_NAME}")

    return {"ok": not problems, "n_cells": len(cells), "problems": problems}


__all__ = [
    "FreezeViolation",
    "cell_key",
    "read_manifest",
    "freeze_cell",
    "verify_frozen_store",
]
