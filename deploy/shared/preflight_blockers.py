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


def _probe_tl_model():
    """A one-layer random HookedTransformer for the calibrated quantizers (GPTQ, AWQ).

    They walk ``model.blocks`` and read ``model.cfg``, so the bare-parameter probe above
    cannot exercise them: they would refuse it with TypeError, which ``_probe`` would score
    as "real path entered" - a false OK. Llama-shaped (the architecture AWQ has rules for)
    with 128 input channels, AWQ's published group size. Falls back to the sentinel where
    TransformerLens is absent, which keeps both families BLOCKED - the safe direction.
    """
    try:
        import warnings

        import torch
        from transformer_lens import HookedTransformer, HookedTransformerConfig

        from src.extraction.real_model import enable_extraction_hooks
    except Exception:  # noqa: BLE001 - no TransformerLens: stay conservative
        return _NotAMockModel()
    torch.manual_seed(0)
    cfg = HookedTransformerConfig(
        n_layers=1, d_model=128, n_ctx=8, d_head=32, n_heads=4, d_mlp=256, d_vocab=64,
        act_fn="silu", normalization_type="RMS", gated_mlp=True, n_key_value_heads=2,
        positional_embedding_type="rotary", rotary_dim=32,
        original_architecture="LlamaForCausalLM", seed=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return enable_extraction_hooks(HookedTransformer(cfg).eval())


def _probe_tokens():
    """Calibration tokens injected into the GPTQ/AWQ probe (the real cache is Q7's check)."""
    import numpy as np

    return np.random.default_rng(0).integers(0, 64, size=(4, 8))


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


def _probe_second_moments() -> dict:
    """Positive, non-uniform E[x^2] for every projection of the probe model.

    Non-uniform so Wanda's score is not a rescaled |W|: with a constant statistic the
    probe could not tell Wanda from per-matrix magnitude pruning.
    """
    try:
        import numpy as np

        from src.calibration.second_moment import broadcast_shape, input_source_for
        from src.compression.torch_weights import projection_parameters
    except Exception:  # noqa: BLE001 - no torch: the compressor probe stays BLOCKED anyway
        return {}
    model = _probe_model()
    params = dict(model.named_parameters()) if hasattr(model, "named_parameters") else {}
    rng = np.random.default_rng(0)
    out = {}
    for name in projection_parameters(model) if params else []:
        shape = broadcast_shape(tuple(params[name].shape), input_source_for(name)[1])
        out[name] = rng.uniform(0.1, 10.0, size=shape)
    return out


CALIBRATED_FAMILIES = ("wanda", "gptq", "awq")


def _calibrated_models() -> list[str]:
    """Model config names with a Wanda, GPTQ or AWQ cell in any run queue."""
    models: set[str] = set()
    for queue in sorted((REPO / "deploy").glob("plan_*/cells_stage*.txt")):
        for line in queue.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not any(
                f"compression_family={family}" in line.split() for family in CALIBRATED_FAMILIES
            ):
                continue
            for token in line.split():
                if token.startswith("model="):
                    models.add(token.split("=", 1)[1])
    return sorted(models)


FINGERPRINTS = Path(__file__).with_name("calibration_fingerprints.json")


def check_calibration_data(
    root: Path | None = None, registry: dict | None = None
) -> tuple[str, bool, str]:
    """Q7 - the calibration token caches Wanda, GPTQ and AWQ read.

    Data, not code: the fix is running the builder, which needs
    mode.allow_external_dataset_download. Each cache must exist, match its own sidecar
    (not altered since it was built), AND match the tracked record in
    calibration_fingerprints.json (the same tokens as the original build). The sidecar
    alone cannot catch a rebuild that came out different - e.g. from a newer dataset
    commit - because the rebuild writes a fresh sidecar that matches itself.
    """
    label = "Q7 calibration caches"
    try:
        import yaml

        from src.calibration.token_cache import cache_path, load_token_cache
    except Exception as exc:  # noqa: BLE001
        return label, False, f"import failed: {exc}"
    if registry is None:
        import json

        registry = (
            json.loads(FINGERPRINTS.read_text(encoding="utf-8")) if FINGERPRINTS.exists() else {}
        )
    recorded = dict(registry.get("caches") or {})
    models = _calibrated_models()
    if not models:
        return label, False, "no calibrated cell found in any queue; cannot tell which caches are needed"
    calibration = yaml.safe_load((REPO / "configs" / "calibration" / "final.yaml").read_text(encoding="utf-8"))
    missing, bad, unrecorded, differs = [], [], [], []
    for model in models:
        model_cfg = yaml.safe_load((REPO / "configs" / "model" / f"{model}.yaml").read_text(encoding="utf-8"))
        resolved = {"calibration": calibration, "model": model_cfg}
        if root is not None:
            resolved["calibration_root"] = str(root)
        path = cache_path(resolved)
        path = path if path.is_absolute() else REPO / path
        if not path.exists():
            missing.append(model)
            continue
        try:
            _, meta = load_token_cache(path)
        except Exception as exc:  # noqa: BLE001 - any failure means the cache is unusable
            bad.append(f"{model} ({type(exc).__name__})")
            continue
        if path.name not in recorded:
            unrecorded.append(model)
        elif meta.get("fingerprint") != recorded[path.name].get("fingerprint"):
            differs.append(model)
    if missing or bad:
        parts = []
        if missing:
            parts.append(f"not built for {missing}")
        if bad:
            parts.append(f"failed verification: {bad}")
        return label, False, "; ".join(parts) + " - needs allow_external_dataset_download"
    if unrecorded or differs:
        parts = []
        if differs:
            parts.append(f"differs from the recorded build for {differs}")
        if unrecorded:
            parts.append(f"no recorded fingerprint for {unrecorded}")
        return label, False, "; ".join(parts) + f" ({FINGERPRINTS.name})"
    return label, True, f"present and matching the recorded fingerprints for {len(models)} models"


def check_compressors() -> list[tuple[str, bool, str]]:
    """B2 - the five compression families on real models."""
    out: list[tuple[str, bool, str]] = []
    # GPTQ and AWQ calibrate on injected tokens and are handed a real HookedTransformer;
    # like Wanda, whether the calibration DATA exists is check_calibration_data's job.
    families = [
        ("rtn", "RtnQuantizer", "src.compression.rtn", {"bits": 4}),
        ("gptq", "GptqCompressor", "src.compression.gptq", {"bits": 4, "calibration": _probe_tokens()}),
        ("awq", "AwqCompressor", "src.compression.awq", {"bits": 4, "calibration": _probe_tokens()}),
        ("magnitude", "MagnitudePruner", "src.compression.magnitude_prune", {}),
        # Wanda's CODE is probed with injected second moments; whether the calibration
        # DATA exists is a separate check (check_calibration_data), so a missing cache
        # reads as missing data rather than as a stub.
        ("wanda", "WandaPruner", "src.compression.wanda",
         {"sparsity": 0.3, "calibration": _probe_second_moments()}),
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
        make_model = _probe_tl_model if name in ("gptq", "awq") else _probe_model
        apply_ok, apply_detail = _probe(inst.apply, make_model(), None)
        delta_ok, delta_detail = _probe(inst.weight_delta, make_model(), None)
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


GATE_REFERENCE = REPO / "data" / "reference" / "ioi_gpt2_small_edges.json"
GATE_PASS = REPO / "data" / "reference" / "ioi_gpt2_small_gate_pass.json"


def check_exit_gate(
    reference: Path | None = None, record: Path | None = None
) -> tuple[str, bool, str]:
    """B6 - the GPT-2 IOI exit gate. Until it passes, no result counts.

    Reads the record the gate writes when it passes
    (tests/test_regression_ioi_gpt2_small.py) and checks it was earned against the PI's
    reference file as it is NOW (sha256), at or above the PI's tolerance. The previous
    check searched the test file for the word "skip", which the file always contains
    (its skip condition is permanent code), so B6 could never have read OK.
    """
    import hashlib
    import json

    label = "B6 GPT-2 IOI exit gate"
    reference = GATE_REFERENCE if reference is None else reference
    record = GATE_PASS if record is None else record
    if not reference.exists():
        return label, False, (
            "no PI reference file data/reference/ioi_gpt2_small_edges.json "
            "(edges + tolerance, PI-owned; docs/HUMAN_DECISIONS.md Step 7)"
        )
    if not record.exists():
        return label, False, "PI reference present, but the gate has not passed on it yet"
    try:
        result = json.loads(record.read_text(encoding="utf-8"))
        tolerance = float(json.loads(reference.read_text(encoding="utf-8"))["tolerance"])
        jaccard = float(result["jaccard"])
    except Exception as exc:  # noqa: BLE001 - an unreadable record opens nothing
        return label, False, f"cannot read the gate record: {type(exc).__name__}: {exc}"
    if result.get("reference_sha256") != hashlib.sha256(reference.read_bytes()).hexdigest():
        return label, False, "the gate passed on a different reference file; re-run it"
    if result.get("passed") is not True or not jaccard >= tolerance:
        return label, False, f"recorded Jaccard {jaccard:.3f} is below the tolerance {tolerance}"
    return label, True, f"passed {result.get('date')}: Jaccard {jaccard:.3f} >= {tolerance}"


def main() -> int:
    quiet = "--quiet" in sys.argv
    results = [check_null_perturber()]
    results += check_compressors()
    results += [check_calibration_data()]
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
        print("Each is an engineering gap, missing calibration data, or a PI-owned input")
        print("(B6's reference edges and tolerance) - not a configuration problem.")
        print("deploy/BLOCKERS.md has the specification for each.")
        return 1

    print(f"All {len(results)} blockers implemented. Clear to run science.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
