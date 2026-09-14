# [AI-GEN] agent=OpenCode date=2026-08-07 task=Content hashing utilities (config hashes, frozen-store verification)
# reviewed-by: PENDING

"""Content hashing utilities.

- ``hash_dict``: canonical SHA-256 of a nested dict/JSON object (keys sorted; floats
  and ints kept as-is) — used for config hashes (ARCHITECTURE.md §3: the
  resolved-config hash is a run's identity).
- ``hash_file``: SHA-256 of a file's bytes — used for artifact manifests.
- ``hash_config``: accepts a plain Mapping or an OmegaConf DictConfig (resolved).
- ``generate_manifest``: hash-manifest entries (ARCHITECTURE.md §2) for the frozen
  store and the artifact MANIFEST.json.

Hashes are the load-bearing integrity mechanism of the null freeze (AI_RULES.md
1.4): a CSI computation must verify ``null_frozen_hash`` and fail hard on mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_JSON_SEP = (",", ":")


def _normalize(obj: Any) -> Any:
    """Sort dicts by key recursively; convert Mapping/Iterable leaves to lists."""
    if isinstance(obj, Mapping):
        return {str(k): _normalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    return obj


def _canonical_json(obj: Any) -> str:
    return json.dumps(
        _normalize(obj), sort_keys=True, separators=_JSON_SEP, ensure_ascii=False, default=str
    )


def hash_dict(obj: Any) -> str:
    """Deterministic SHA-256 hex digest of a nested dict-like object."""
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def hash_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file's contents (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_config(cfg: Any) -> str:
    """Hash a resolved config: plain Mapping, or OmegaConf DictConfig (resolved).

    Raises TypeError for anything else. Config hashes are the run identity
    (ARCHITECTURE.md §3) and the key recorded in every run's tags and in frozen
    ``meta.json`` (ARCHITECTURE.md §2).
    """
    if isinstance(cfg, Mapping):
        return hash_dict(cfg)
    if hasattr(cfg, "to_container"):  # OmegaConf DictConfig / ListConfig
        try:
            return hash_dict(cfg.to_container(resolve=True))
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(f"could not resolve config for hashing: {exc}") from exc
    raise TypeError(
        f"hash_config expects a Mapping or OmegaConf config, got {type(cfg).__name__}"
    )


def manifest_entry(path: str | os.PathLike[str]) -> dict[str, Any]:
    """One hash-manifest entry: path, sha256, size_bytes (ARCHITECTURE.md §2)."""
    p = Path(path)
    return {"path": p.as_posix(), "sha256": hash_file(p), "size_bytes": p.stat().st_size}


def generate_manifest(paths: Iterable[str | os.PathLike[str]]) -> dict[str, dict[str, Any]]:
    """Generate a manifest dict: {relative-path-as-posix: entry} for all paths."""
    return {entry["path"]: entry for entry in (manifest_entry(p) for p in paths)}
