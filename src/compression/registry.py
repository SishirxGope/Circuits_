# [AI-GEN] agent=Claude date=2026-09-27 task=Single compressor registry shared by Stage B and Stage C
# reviewed-by: PENDING

"""One place that maps ``compression_family`` -> compressor class.

**Why this module exists.** Stage B draws the matched-magnitude null from
``Compressor.weight_delta``; Stage C measures D(c) with ``Compressor.apply``. CSI(c) =
D(c) / median(D_null(c)) is only meaningful if both stages instantiated *the same
compressor with the same kwargs* (ARCHITECTURE.md §4: ``weight_delta`` is the single
source of truth "so the null can never drift from the compression it is matched to").
Before this module, Stage C read a registry it owned privately and Stage B did not
consult the configured family at all - the two could not be checked against each other.

``compressor_for`` is therefore the only sanctioned way for either stage to obtain a
compressor, and ``compression_provenance`` records what was built so the frozen null
carries the identity of the compression it was matched to.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Families that name no compression at all. A null matched to "no compression" has
# nothing to match: every draw would be a zero perturbation, D_null would collapse to a
# spike at 0, and CSI would divide by zero or by noise. Refused explicitly rather than
# quietly producing a degenerate denominator.
NON_COMPRESSION_FAMILIES: frozenset[str] = frozenset({"dense", "null", "none", ""})

_REGISTRY: dict[str, Any] = {}


def compressor_registry() -> dict[str, Any]:
    """Lazily built so importing this module never pulls in every compressor."""
    global _REGISTRY
    if not _REGISTRY:
        from src.compression.awq import AwqCompressor
        from src.compression.gptq import GptqCompressor
        from src.compression.magnitude_prune import MagnitudePruner
        from src.compression.rtn import RtnQuantizer
        from src.compression.wanda import WandaPruner

        _REGISTRY = {
            "rtn": RtnQuantizer,
            "gptq": GptqCompressor,
            "awq": AwqCompressor,
            "magnitude": MagnitudePruner,
            "wanda": WandaPruner,
        }
    return _REGISTRY


def compression_family_of(resolved: Mapping[str, Any]) -> str:
    return str(resolved.get("compression_family", "") or "").strip().lower()


def compressor_kwargs_of(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """The cell's compressor kwargs.

    Read from ``stage_c.compressor_kwargs`` because that is what
    ``deploy/shared/gen_cells.py`` emits on the command line
    (``+stage_c.compressor_kwargs.bits=8``) for BOTH stages. The identically-named block
    inside ``configs/compression/*.yaml`` populates ``compression.*`` and is documentary
    only - reading that instead would silently fall back to each compressor's default
    (e.g. every ``rtn_int8`` cell quantizing at 4 bits).
    """
    return dict((resolved.get("stage_c") or {}).get("compressor_kwargs") or {})


def compressor_for(resolved: Mapping[str, Any], *, stage: str) -> Any:
    """Build the compressor this cell names, or raise.

    ``stage`` only sharpens the error message; both stages must get the same object for
    the same config, which is the entire point of this function.
    """
    family = compression_family_of(resolved)
    if family in NON_COMPRESSION_FAMILIES:
        raise ValueError(
            f"{stage}: compression_family={family!r} names no compression, so there is "
            "nothing for a null to be matched to. Pass a real cell "
            "(see deploy/plan_b_dgx_spark/cells_stageb.txt)."
        )
    registry = compressor_registry()
    if family not in registry:
        raise ValueError(
            f"{stage}: unknown compression_family {family!r}; expected one of {sorted(registry)}"
        )
    kwargs = compressor_kwargs_of(resolved)
    try:
        return registry[family](**kwargs)
    except TypeError as exc:
        raise ValueError(
            f"{stage}: compression_family={family!r} does not accept "
            f"stage_c.compressor_kwargs={kwargs!r}: {exc}"
        ) from exc


def compression_provenance(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """The identity of the compression a frozen null was matched to.

    Written into the Stage B ``meta.json`` and re-checked by Stage C. Without it, the
    existing ``null_frozen_hash`` check proves only that the null file is *intact*, not
    that it was matched to the cell now being divided by it.
    """
    return {
        "compression_family": compression_family_of(resolved),
        "compression_level": resolved.get("compression_level"),
        "compressor_kwargs": compressor_kwargs_of(resolved),
    }


__all__ = [
    "NON_COMPRESSION_FAMILIES",
    "compression_family_of",
    "compression_provenance",
    "compressor_for",
    "compressor_kwargs_of",
    "compressor_registry",
]
