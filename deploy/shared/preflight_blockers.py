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
    ok, detail = _probe(method, _NotAMockModel(), {"w": 1.0}, 0)
    return "B1 matched-magnitude null", ok, detail


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
        ok, detail = _probe(inst.apply, _NotAMockModel(), None)
        out.append((label, ok, detail))
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
