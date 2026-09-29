# [AI-GEN] agent=Claude date=2026-09-29 task=Q7 calibration token cache (Wanda/GPTQ/AWQ input)
# reviewed-by: PENDING
#
# Adapted from: https://github.com/hecboar/sae-pruning-paper @ 261191804675e2d39d0a265320dbc0bc85afd30a, MIT
#   - notebooks/SAE_Pruning_Reproducibility.ipynb cell 5 (build_token_cache) and cell 3
#     (SAE_DATASET, CONTEXT_LEN): the procedure behind the caches the revision loads.
#   - license: THIRD_PARTY_LICENSES/sae-pruning-paper-LICENSE.txt

"""The calibration token cache: what Wanda (and later GPTQ/AWQ) see, and nothing else.

**What is pre-registered and what is read from upstream.** ``configs/calibration/final.yaml``
fixes the corpus (fineweb-edu), the token count (300,000) and the seed (7), and leaves
``context_length: null`` rather than invent it. The rest of the procedure is taken from
the notebook that built the caches the upstream revision consumes:

- the revision never builds a cache; it loads ``{tokenizer}_calib_{n}_{seed}.npy`` by that
  exact name (``saediag/models.py:68``), and the notebook's ``token_cache_path`` produces
  that name, with the seeds (7/42/123) and sizes of the revision's run matrix. So the
  notebook's builder is the procedure behind the revision's caches. This is **inferred**
  from matching names and parameters, like the Q9 pin; upstream does not state it.
- upstream did not ship the cache, so byte-identity with *their* tokens cannot be
  checked. What is checked is that *our* cache is identical wherever it is used: each
  build writes a fingerprint beside the array and each load verifies it.

**The packing rule** (:func:`pack_documents`) is pure so it can be tested without the
corpus: skip documents shorter than 50 characters after stripping; tokenize with no
special tokens (no BOS); cut each document into non-overlapping ``context_len`` windows
and drop its tail; stop once ``n_tokens`` are reached; keep ``n_tokens // context_len``
windows. At the pre-registered 300,000 tokens and 256 that is 1,171 windows = 299,776
tokens. The floor is upstream's, not a bug.

**The download gate.** Building fetches fineweb-edu, so it requires
``mode.allow_external_dataset_download`` exactly as model loading requires
``mode.allow_model_download`` (src/extraction/real_model.py) - checked in code, first.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Read from the pinned fork's notebook, not chosen here. Each value is what upstream's
# build_token_cache did; changing one produces a different cache than the revision used.
UPSTREAM_BUILD: dict[str, Any] = {
    "dataset_config": "sample-10BT",  # cell 3: SAE_DATASET
    "split": "train",                 # cell 3: SAE_DATASET
    "text_field": "text",             # cell 3: SAE_DATASET
    "shuffle_buffer": 10_000,         # cell 5: ds.shuffle(seed=spec.seed, buffer_size=10000)
    "min_chars": 50,                  # cell 5: if len(text.strip()) < 50: continue
    "context_len": 256,               # cell 3: CONTEXT_LEN
    "add_special_tokens": False,      # cell 5: tok.encode(text, add_special_tokens=False)
}
PROCEDURE_SOURCE = (
    "hecboar/sae-pruning-paper @ 261191804675e2d39d0a265320dbc0bc85afd30a "
    "notebooks/SAE_Pruning_Reproducibility.ipynb cells 3 and 5"
)
DEFAULT_ROOT = Path("calibration_cache")  # gitignored; override with top-level calibration_root


class CalibrationUnavailable(RuntimeError):
    """The calibration data a step needs has not been built on this machine.

    Distinct from ``NotImplementedError``: the code exists, the data does not. Preflight
    reports the two differently, because the fix for one is engineering and the fix for
    the other is running the builder.
    """


def calibration_spec(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """The pre-registered calibration block, with ``context_len`` filled from upstream."""
    block = dict(((resolved.get("calibration") or {}).get("calibration")) or {})
    missing = [key for key in ("hf_id", "n_tokens", "seed") if block.get(key) in (None, "")]
    if missing:
        raise ValueError(
            f"calibration config is missing {missing}; expected configs/calibration/final.yaml "
            "composed as the `calibration` group"
        )
    ctx = block.get("context_length")
    return {
        "corpus": block.get("corpus"),
        "hf_id": str(block["hf_id"]),
        "license": block.get("license"),
        "n_tokens": int(block["n_tokens"]),
        "seed": int(block["seed"]),
        "dtype": str(block.get("dtype") or "float32"),
        "context_len": int(ctx) if ctx is not None else int(UPSTREAM_BUILD["context_len"]),
        "context_len_source": (
            "configs/calibration" if ctx is not None
            else "upstream notebook CONTEXT_LEN (config context_length is null)"
        ),
    }


def calibration_root(resolved: Mapping[str, Any]) -> Path:
    return Path(str(resolved.get("calibration_root") or DEFAULT_ROOT))


def cache_filename(tokenizer_id: str, n_tokens: int, seed: int) -> str:
    """Upstream's name, so a cache is recognisable next to theirs (models.py:68)."""
    return f"{tokenizer_id.replace('/', '_')}_calib_{int(n_tokens)}_{int(seed)}.npy"


def cache_path(resolved: Mapping[str, Any], tokenizer_id: str | None = None) -> Path:
    """Where this cell's calibration cache lives. The tokenizer defaults to the model's."""
    spec = calibration_spec(resolved)
    tok = tokenizer_id or str((resolved.get("model") or {}).get("hf_id") or "")
    if not tok:
        raise ValueError("no tokenizer id: pass tokenizer_id or set model.hf_id")
    return (
        calibration_root(resolved) / "token_caches"
        / cache_filename(tok, spec["n_tokens"], spec["seed"])
    )


def pack_documents(
    texts: Iterable[str | None],
    encode: Callable[[str], Sequence[int]],
    *,
    n_tokens: int,
    context_len: int,
    min_chars: int = UPSTREAM_BUILD["min_chars"],
) -> np.ndarray:
    """Upstream's packing loop, same behaviour. Returns int64 ``[n_seq, context_len]``.

    Raises rather than return a short cache when the stream runs out: upstream would
    silently save fewer windows, and a short cache is a different calibration set.
    """
    if context_len < 1:
        raise ValueError(f"context_len must be >= 1, got {context_len}")
    if n_tokens < context_len:
        raise ValueError(
            f"n_tokens={n_tokens} is less than one window of {context_len}; the cache "
            "would be empty"
        )
    windows: list[Sequence[int]] = []
    total = 0
    for text in texts:
        text = text or ""
        if len(text.strip()) < min_chars:
            continue
        ids = list(encode(text))
        for start in range(0, len(ids) - context_len + 1, context_len):
            windows.append(ids[start:start + context_len])
            total += context_len
            if total >= n_tokens:
                break
        if total >= n_tokens:
            break
    keep = n_tokens // context_len
    if len(windows) < keep:
        raise ValueError(
            f"the corpus ran out after {len(windows)} windows of {context_len}; "
            f"{keep} are needed for {n_tokens} tokens"
        )
    return np.asarray(windows[:keep], dtype=np.int64).reshape(keep, context_len)


def fingerprint(tokens: np.ndarray) -> str:
    """sha256 over dtype, shape and bytes: equal fingerprints mean identical caches."""
    arr = np.ascontiguousarray(tokens)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(repr(tuple(arr.shape)).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def validate_token_cache(
    tokens: Any, *, vocab_size: int | None = None, context_len: int | None = None
) -> np.ndarray:
    """A 2-D, non-empty array of in-vocabulary token ids, as int64."""
    arr = np.asarray(tokens)
    if arr.ndim != 2:
        raise ValueError(f"token cache must be 2-D [n_seq, context_len], got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"token cache must hold integer ids, got dtype {arr.dtype}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"token cache is empty: shape {arr.shape}")
    if context_len is not None and arr.shape[1] != int(context_len):
        raise ValueError(f"token cache windows are {arr.shape[1]} long, expected {context_len}")
    if int(arr.min()) < 0:
        raise ValueError("token cache contains negative ids")
    if vocab_size is not None and int(arr.max()) >= int(vocab_size):
        raise ValueError(
            f"token id {int(arr.max())} is outside the model's vocabulary ({vocab_size}); "
            "the cache was built with a different tokenizer"
        )
    return arr.astype(np.int64, copy=False)


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def save_token_cache(path: Path, tokens: np.ndarray, provenance: Mapping[str, Any]) -> Path:
    """Write the array and a provenance sidecar carrying its fingerprint."""
    arr = validate_token_cache(tokens)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr, allow_pickle=False)
    meta = {
        **dict(provenance),
        "fingerprint": fingerprint(arr),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "n_tokens_kept": int(arr.size),
    }
    _sidecar(path).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_token_cache(path: Path, *, vocab_size: int | None = None) -> tuple[np.ndarray, dict]:
    """Load and verify a cache. Raises :class:`CalibrationUnavailable` if it is absent."""
    path = Path(path)
    if not path.exists():
        raise CalibrationUnavailable(
            f"no calibration token cache at {path}. Build it with `python "
            "experiments/build_calibration_cache.py mode=scientific_run model=<name>` "
            "(requires mode.allow_external_dataset_download)."
        )
    sidecar = _sidecar(path)
    if not sidecar.exists():
        raise ValueError(
            f"{path} has no provenance sidecar ({sidecar.name}); refusing a cache whose "
            "origin and fingerprint cannot be checked"
        )
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    arr = validate_token_cache(
        np.load(path, allow_pickle=False),
        vocab_size=vocab_size,
        context_len=(meta.get("shape") or [None, None])[1],
    )
    got = fingerprint(arr)
    if got != meta.get("fingerprint"):
        raise ValueError(
            f"{path} does not match the fingerprint recorded when it was built "
            f"({str(meta.get('fingerprint'))[:12]}... vs {got[:12]}...); the calibration "
            "set changed"
        )
    return arr, meta


def assert_download_allowed(resolved: Mapping[str, Any]) -> None:
    mode = resolved.get("mode") or {}
    if mode.get("allow_external_dataset_download") is not True:
        raise PermissionError(
            "building the calibration cache downloads fineweb-edu, and this run does not "
            "permit it (mode.allow_external_dataset_download is not true). Q7: "
            "configs/calibration/final.yaml."
        )


def build_token_cache(resolved: Mapping[str, Any], *, tokenizer: Any = None) -> Path:
    """Build (or verify and reuse) the cache for ``resolved['model']``. Network-gated.

    The dataset is pinned to the commit it resolves to at build time and that sha is
    recorded, so a later rebuild can request the same snapshot.
    """
    assert_download_allowed(resolved)  # before anything that could touch the network
    spec = calibration_spec(resolved)
    model_cfg = dict(resolved.get("model") or {})
    tokenizer_id = str(model_cfg.get("hf_id") or "")
    tokenizer_revision = str(model_cfg.get("hf_revision") or "")
    path = cache_path(resolved, tokenizer_id)
    if path.exists():
        load_token_cache(path)  # verify; never rebuild silently over an existing cache
        return path

    import datasets
    from huggingface_hub import HfApi

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, revision=tokenizer_revision or None
        )
    dataset_sha = HfApi().dataset_info(spec["hf_id"]).sha
    stream = datasets.load_dataset(
        spec["hf_id"], UPSTREAM_BUILD["dataset_config"], split=UPSTREAM_BUILD["split"],
        streaming=True, revision=dataset_sha,
    ).shuffle(seed=spec["seed"], buffer_size=UPSTREAM_BUILD["shuffle_buffer"])
    field = UPSTREAM_BUILD["text_field"]
    tokens = pack_documents(
        (example.get(field) for example in stream),
        lambda text: tokenizer.encode(
            text, add_special_tokens=UPSTREAM_BUILD["add_special_tokens"]
        ),
        n_tokens=spec["n_tokens"],
        context_len=spec["context_len"],
    )
    return save_token_cache(path, tokens, {
        **UPSTREAM_BUILD,
        **spec,
        "dataset_revision": dataset_sha,
        "datasets_version": datasets.__version__,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "procedure_source": PROCEDURE_SOURCE,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })


def synthetic_token_cache(
    n_seq: int, context_len: int, vocab_size: int, seed: int = 0
) -> np.ndarray:
    """Uniform random ids for engineering tests. NOT calibration data."""
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(vocab_size), size=(int(n_seq), int(context_len)), dtype=np.int64)


__all__ = [
    "DEFAULT_ROOT",
    "UPSTREAM_BUILD",
    "CalibrationUnavailable",
    "assert_download_allowed",
    "build_token_cache",
    "cache_filename",
    "cache_path",
    "calibration_root",
    "calibration_spec",
    "fingerprint",
    "load_token_cache",
    "pack_documents",
    "save_token_cache",
    "synthetic_token_cache",
    "validate_token_cache",
]
