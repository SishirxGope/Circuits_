# [AI-GEN] agent=Claude date=2026-09-29 task=Real-model smoke test of the GPTQ/AWQ ports, runnable while B6 keeps the queue shut
# reviewed-by: PENDING
#
# WHAT THIS IS
# ------------
# The unit tests prove the ports equal the reference code on tiny models. This runs them
# once on a REAL model at the REAL calibration size (the Q7 cache: 1171 x 256 tokens, the
# bf16 calibration forward), so time and peak GPU memory are known before a queue of 12
# parallel cells depends on them.
#
# WHAT IT DOES NOT DO
# -------------------
# It writes nothing: the tokens are handed in directly, and injected tokens are never
# cached (src/compression/gptq.py::calibrated_tensors), so no Stage B or Stage C run
# will ever read a result computed here. It produces no number that enters the paper and
# checks no threshold - it only fails on what cannot be right (non-finite weights, a
# projection left unchanged, a parameter outside the projections changed).
#
# Usage (repo root):
#   python deploy/shared/smoke_gptq_awq.py pythia160m               # both methods, full size
#   python deploy/shared/smoke_gptq_awq.py llama32_1b --method awq
#   python deploy/shared/smoke_gptq_awq.py pythia160m --n-seq 8      # a quick look
#   python deploy/shared/smoke_gptq_awq.py pythia160m --synthetic --n-seq 8   # no token cache
#
# Exit codes: 0 = every method ran and its result is well-formed   1 = otherwise

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="a configs/model/<name>.yaml, e.g. pythia160m")
    parser.add_argument("--method", choices=["gptq", "awq", "both"], default="both")
    parser.add_argument("--n-seq", type=int, default=None, help="first N sequences (default: all)")
    parser.add_argument("--synthetic", action="store_true", help="random tokens instead of the Q7 cache")
    args = parser.parse_args()

    import numpy as np
    import torch

    from src.calibration.token_cache import cache_path, load_token_cache, synthetic_token_cache
    from src.compression.awq import AwqCompressor
    from src.compression.gptq import GptqCompressor
    from src.compression.layerwise import tensors_delta
    from src.compression.torch_weights import projection_parameters
    from src.extraction.real_model import load_pinned_model

    model_file = REPO / "configs" / "model" / f"{args.model}.yaml"
    if not model_file.exists():
        print(f"no such model config: {model_file}")
        return 1
    # The approved mode for real models (its allow_model_download is what the loader checks),
    # and the pre-registered calibration block, exactly as a cell composes them.
    cfg = {
        "model": _yaml(model_file),
        "mode": _yaml(REPO / "configs" / "mode" / "scientific_run.yaml"),
        "calibration": _yaml(REPO / "configs" / "calibration" / "final.yaml"),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {cfg['model']['hf_id']} @ {cfg['model']['hf_revision'][:8]} on {device} ...")
    model = load_pinned_model(cfg, device=device)
    vocab = int(model.cfg.d_vocab)

    if args.synthetic:
        tokens = synthetic_token_cache(args.n_seq or 8, 256, vocab, seed=0)
        source = "synthetic tokens"
    else:
        tokens, _meta = load_token_cache(cache_path(cfg), vocab_size=vocab)
        tokens = tokens[: args.n_seq] if args.n_seq else tokens
        source = f"Q7 cache {cache_path(cfg).name}"
    tokens = np.ascontiguousarray(tokens)
    print(f"calibrating on {tokens.shape[0]} x {tokens.shape[1]} {source}, forward in "
          f"{cfg['calibration']['calibration']['dtype']}")

    wanted = set(projection_parameters(model))
    dense = {name: p.detach().to(torch.float32) for name, p in model.named_parameters() if name in wanted}
    methods = ["gptq", "awq"] if args.method == "both" else [args.method]
    ok = True
    for method in methods:
        compressor = (GptqCompressor if method == "gptq" else AwqCompressor)(calibration=tokens)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        tensors = compressor._quantized(model, cfg)
        elapsed = time.perf_counter() - start
        peak = f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if device == "cuda" else "n/a (cpu)"

        problems = []
        if set(tensors) != wanted:
            problems.append(f"changed {sorted(set(tensors) ^ wanted)[:5]} - not exactly the projections")
        delta = tensors_delta(model, {k: v for k, v in tensors.items() if k in wanted})
        rel = []
        for name in sorted(wanted & set(tensors)):
            if not torch.isfinite(tensors[name]).all():
                problems.append(f"{name}: non-finite values")
            if delta[name] == 0.0:
                problems.append(f"{name}: unchanged")
            rel.append(delta[name] / max(float(torch.linalg.vector_norm(dense[name])), 1e-30))
        rel_arr = np.array(rel) if rel else np.array([np.nan])
        status = "OK" if not problems else "FAILED"
        print(f"{status:6s} {method}: {len(tensors)} tensors in {elapsed:.1f} s, peak GPU {peak}; "
              f"||dW||/||W|| min {rel_arr.min():.4f} median {np.median(rel_arr):.4f} max {rel_arr.max():.4f}")
        for problem in problems[:10]:
            print(f"         {problem}")
        ok = ok and not problems
        del tensors
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
