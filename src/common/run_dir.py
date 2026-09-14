# [AI-GEN] agent=Claude date=2026-08-09 task=Resolve run-directory collisions between configs that share grid coordinates but differ scientifically
# reviewed-by: PENDING

"""Allocating a run directory (CLAUDE.md §4 naming + ARCHITECTURE.md §3 identity).

Two governing rules meet here and they do not say the same thing:

- **CLAUDE.md §4** names a run by its GRID COORDINATES:
  ``{date}_{stage}_{model}_{task}_{setting}_{configset}_seed{S}``.
- **ARCHITECTURE.md §3** says *"the resolved-config hash is the run's identity"*.

Those disagree whenever two configs share grid coordinates but differ in something the
name does not encode — which is most of the scientific choices. Swapping the band
cutoffs (``ensemble/decompose=final`` vs ``provisional``), the null draw count R, the
distance function, or the comparison-level scheme all leave the run name identical.
The result was a hard ``FileExistsError`` on the second config the PI tried, whose only
suggested escapes were "use a fresh seed" (which changes the science) or "a new date"
(which means waiting until tomorrow).

Resolution, which keeps both rules intact:

- the **run name** is unchanged and stays exactly per CLAUDE.md §4 — it is what gets
  recorded in ``run_meta.json`` and used in the paper;
- the **directory** is the run name when that is free;
- on a collision the config hashes decide:
  - **same hash** -> a true re-run of an identical config. Refused: runs are immutable
    and are never overwritten (AI_RULES.md 1.2).
  - **different hash** -> two genuinely different experiments that happen to share grid
    coordinates. The directory is disambiguated as ``{run_name}__cfg{hash8}`` so both
    survive on disk, neither is overwritten, and the config hash — the actual identity —
    is visible in the path.

Nothing is ever overwritten in either branch.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_DUMP_NAME = "resolved_config.json"
RUN_META_NAME = "run_meta.json"
_HASH_PREFIX_LEN = 8


class RunDirExists(FileExistsError):
    """A run with this exact resolved config already exists (AI_RULES.md 1.2)."""


def _existing_config_hash(run_dir: Path) -> str | None:
    """The config hash recorded by a previous run, or None if unreadable."""
    meta = run_dir / RUN_META_NAME
    if meta.exists():
        try:
            with open(meta, "r", encoding="utf-8") as f:
                recorded = json.load(f).get("config_hash")
            if recorded:
                return str(recorded)
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    # Fall back to re-hashing the dumped config: a run that was interrupted before
    # run_meta.json was written still has resolved_config.json.
    dump = run_dir / CONFIG_DUMP_NAME
    if dump.exists():
        try:
            from .hashing import hash_dict

            with open(dump, "r", encoding="utf-8") as f:
                return hash_dict(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass
    return None


def allocate_run_dir(run_root: str | Path, run_name: str, config_hash: str) -> Path:
    """Return the directory this run should own, creating it. Never overwrites.

    Raises ``RunDirExists`` when a run with the SAME resolved config already exists.
    Disambiguates with a ``__cfg{hash8}`` suffix when the collision is between
    different configs.
    """
    root = Path(run_root)
    preferred = root / run_name

    if not preferred.exists():
        preferred.mkdir(parents=True)
        return preferred

    previous = _existing_config_hash(preferred)
    if previous == config_hash:
        raise RunDirExists(
            f"a run with this exact config already exists at {preferred} "
            f"(config_hash={config_hash[:12]}...). Runs are immutable and are never "
            f"overwritten (AI_RULES.md 1.2). If you meant to re-run, change the seed or "
            f"delete nothing — start a new run with a different config."
        )

    disambiguated = root / f"{run_name}__cfg{config_hash[:_HASH_PREFIX_LEN]}"
    if disambiguated.exists():
        if _existing_config_hash(disambiguated) == config_hash:
            raise RunDirExists(
                f"a run with this exact config already exists at {disambiguated} "
                f"(config_hash={config_hash[:12]}...). Runs are immutable (AI_RULES.md 1.2)."
            )
        raise RunDirExists(  # pragma: no cover - an 8-hex-char collision
            f"{disambiguated} exists but records a different config hash; refusing to "
            f"guess. Move it aside manually."
        )

    disambiguated.mkdir(parents=True)
    print(
        f"[run-dir] {run_name} is taken by a DIFFERENT config; this run gets "
        f"{disambiguated.name}. The run name is unchanged (CLAUDE.md §4); the suffix is "
        f"the resolved-config hash, which is the run's real identity (ARCHITECTURE.md §3)."
    )
    return disambiguated


__all__ = ["allocate_run_dir", "RunDirExists", "CONFIG_DUMP_NAME", "RUN_META_NAME"]
