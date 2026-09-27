# [AI-GEN] agent=Claude date=2026-09-21 task=Functional preflight: refuse to start science while the real-model blockers are stubs
# reviewed-by: PENDING
#
# WHY THIS EXISTS
# ---------------
# Every stage runner works today - on the MOCK model. Hand any of them a real model and
# the compressors and the null perturber raise NotImplementedError. Without this check
# the failure arrives hours into a queue, in a log file, at night.
#
# Worse, one of the blockers fails SILENTLY rather than loudly: the chance-floor universe
# is still U*(U-1), roughly 10x too lenient, so results would come out looking better than
# they are and nothing would crash. A preflight that only catches crashes would miss it.
#
# Exit codes:  0 = clear to run science   1 = at least one blocker is unimplemented
#
# Usage:  python deploy/shared/preflight_blockers.py [--quiet]

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


class _NotAMockModel:
    """A stand-in real model: not a MockModel, so the real code path must handle it."""


def _probe_model():
    """The object a real-model code path is handed by this preflight.

    A bare sentinel is NOT good enough for the compressor probe. A correctly implemented
    compressor refuses an object that is neither a MockModel nor a torch module - with
    NotImplementedError, which this file reads as "still a stub". That made RTN report
    BLOCKED after its real path landed, and since ``assert_preflight`` gates the whole
    science queue, the gate would never have opened.

    So we hand the probe a genuine torch module carrying one projection-shaped parameter.
    An implemented compressor quantizes/prunes it and returns; a stub still raises.
    Falls back to the sentinel where torch is absent, which keeps every family BLOCKED -
    the safe direction.
    """
    try:
        import torch
        from torch import nn
    except Exception:  # noqa: BLE001 - no torch: stay conservative
        return _NotAMockModel()

    class _TinyTransformerLens(nn.Module):
        """Parameter names mirroring TransformerLens, including the two the pruners
        special-case: ``embed.W_E`` joins the threshold pool, ``unembed.W_U`` is excluded.

        A probe carrying only bare projections made magnitude pruning raise
        ValueError("embedding_names not present in the model"), which ``_probe`` scores as
        implemented because it is not a NotImplementedError - a false OK on a pruner that
        never pruned anything.
        """

        def __init__(self) -> None:
            super().__init__()
            g = torch.Generator().manual_seed(0)

            self.embed = nn.Module()
            self.embed.W_E = nn.Parameter(torch.randn(40, 8, generator=g))
            self.unembed = nn.Module()
            self.unembed.W_U = nn.Parameter(torch.randn(8, 40, generator=g))

            attn = nn.Module()
            attn.W_Q = nn.Parameter(torch.randn(2, 8, 4, generator=g))
            attn.W_K = nn.Parameter(torch.randn(2, 8, 4, generator=g))
            attn.W_V = nn.Parameter(torch.randn(2, 8, 4, generator=g))
            attn.W_O = nn.Parameter(torch.randn(2, 4, 8, generator=g))
            mlp = nn.Module()
            mlp.W_in = nn.Parameter(torch.randn(8, 16, generator=g))
            mlp.W_out = nn.Parameter(torch.randn(16, 8, generator=g))
            block = nn.Module()
            block.attn = attn
            block.mlp = mlp
            self.blocks = nn.ModuleList([block])

    return _TinyTransformerLens()


def _probe(fn, *args) -> tuple[bool, str]:
    """Call fn; return (implemented, detail).

    NotImplementedError means the stub is still in place. Any OTHER exception means the
    real path was entered and then failed on our dummy input, which is what we want -
    it is evidence the code is real, not that it works.
    """
    try:
        fn(*args)
        return True, "returned without error"
    except NotImplementedError as exc:
        return False, f"NotImplementedError: {str(exc)[:90]}"
    except Exception as exc:  # noqa: BLE001 - any other error means real code ran
        return True, f"real path entered ({type(exc).__name__})"


def check_null_perturber() -> tuple[str, bool, str]:
    """B1 - the null model itself. Nothing downstream means anything without it."""
    try:
        from src.science.matched_magnitude import MatchedMagnitudePerturber
    except Exception as exc:  # noqa: BLE001
        return "B1 matched-magnitude null", False, f"import failed: {exc}"
    p = MatchedMagnitudePerturber()
    method = getattr(p, "perturb", None) or getattr(p, "apply", None)
    if method is None:
        return "B1 matched-magnitude null", False, "no perturb/apply method found"
    # A real module and magnitudes naming its own parameters, for the same reason the
    # compressor probe uses one: handed a bare sentinel, a correctly implemented perturber
    # raises NotImplementedError, which this file reads as "still a stub".
    model = _probe_model()
    magnitudes = {
        name: 1.0 for name, _ in getattr(model, "named_parameters", list)()
    } or {"w": 1.0}
    rng = _probe_rng()
    ok, detail = _probe(method, model, magnitudes, rng)
    return "B1 matched-magnitude null", ok, detail


def _probe_rng():
    """A seeded numpy Generator, which is what the Perturber protocol is handed."""
    import numpy as np

    return np.random.default_rng(0)


def check_compressors() -> list[tuple[str, bool, str]]:
    """B2 - the five compression families on real models."""
    out: list[tuple[str, bool, str]] = []
    families = [
        ("rtn", "RtnQuantizer", "src.compression.rtn", {"bits": 4}),
        ("gptq", "GptqCompressor", "src.compression.gptq", {}),
        ("awq", "AwqCompressor", "src.compression.awq", {}),
        ("magnitude", "MagnitudePruner", "src.compression.magnitude_prune", {}),
        ("wanda", "WandaPruner", "src.compression.wanda", {}),
    ]
    for name, cls_name, module, kwargs in families:
        label = f"B2 compressor: {name}"
        try:
            mod = __import__(module, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            inst = cls(**kwargs) if kwargs else cls()
        except Exception as exc:  # noqa: BLE001
            out.append((label, False, f"could not instantiate: {exc}"))
            continue
        # Both halves of the Compressor protocol: Stage C needs apply(), Stage B needs
        # weight_delta() for the null magnitudes. One without the other is not usable.
        apply_ok, apply_detail = _probe(inst.apply, _probe_model(), None)
        delta_ok, delta_detail = _probe(inst.weight_delta, _probe_model(), None)
        if apply_ok and not delta_ok:
            out.append((label, False, f"apply() ok but weight_delta() {delta_detail}"))
        elif delta_ok and not apply_ok:
            out.append((label, False, f"weight_delta() ok but apply() {apply_detail}"))
        else:
            out.append((label, apply_ok, apply_detail))
    return out


def check_chance_floor() -> tuple[str, bool, str]:
    """B3 - the SILENT one. A flag in config.yaml records whether it is real yet."""
    cfg = (REPO / "configs" / "config.yaml").read_text(encoding="utf-8")
    for line in cfg.splitlines():
        if "chance_floor_universe_implemented" in line and not line.strip().startswith("#"):
            implemented = "true" in line.split(":", 1)[1].lower()
            return (
                "B3 chance-floor universe",
                implemented,
                "flag is true" if implemented
                else "still U*(U-1) - roughly 10x too lenient, and it fails SILENTLY",
            )
    return "B3 chance-floor universe", False, "flag not found in configs/config.yaml"


def check_normalized_l1() -> tuple[str, bool, str]:
    """B4 - pre-registered as a reported quantity (Q2), so shipping without it deviates."""
    src = (REPO / "src" / "science" / "distances.py").read_text(encoding="utf-8")
    found = "normalized_l1" in src or "normalised_l1" in src
    return (
        "B4 normalised L1",
        found,
        "present in distances.py" if found else "not in distances.py (pre-registered in Q2)",
    )


def check_exit_gate() -> tuple[str, bool, str]:
    """B6 - the GPT-2 IOI exit gate. Until it passes, no result counts."""
    hits = list((REPO / "tests").glob("*exit_gate*")) + list((REPO / "tests").glob("*gpt2*"))
    if not hits:
        return "B6 GPT-2 IOI exit gate", False, "no exit-gate test file found in tests/"
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in hits)
    skipped = "skip" in text.lower()
    return (
        "B6 GPT-2 IOI exit gate",
        not skipped,
        f"found {[p.name for p in hits]}" + (" but still skipped" if skipped else ""),
    )


def main() -> int:
    quiet = "--quiet" in sys.argv
    results = [check_null_perturber()]
    results += check_compressors()
    results += [check_chance_floor(), check_normalized_l1(), check_exit_gate()]

    blocked = [r for r in results if not r[1]]

    if not quiet:
        print("=" * 74)
        print("PREFLIGHT - real-model blockers (see deploy/BLOCKERS.md)")
        print("=" * 74)
        for label, ok, detail in results:
            print(f"  [{'OK ' if ok else 'BLOCKED'}] {label:32s} {detail}")
        print("-" * 74)

    if blocked:
        print(f"REFUSING TO RUN SCIENCE: {len(blocked)} of {len(results)} blockers unimplemented.")
        print("These are engineering gaps, not configuration problems. Implement them")
        print("first - deploy/BLOCKERS.md has the specification for each.")
        return 1

    print(f"All {len(results)} blockers implemented. Clear to run science.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
