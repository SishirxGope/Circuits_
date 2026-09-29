# [AI-GEN] agent=Claude date=2026-09-29 task=Q7 entrypoint: build the calibration token cache for one model
# reviewed-by: PENDING

"""Build (or verify) the Q7 calibration token cache for the configured model.

    python experiments/build_calibration_cache.py mode=scientific_run model=gemma2_2b

Downloads fineweb-edu, so it refuses unless ``mode.allow_external_dataset_download`` is
true (scientific_run sets it; engineering_dry_run does not). Run once per model: the
cache is per tokenizer. Re-running on an existing cache verifies its fingerprint and
does not rebuild. The E[x^2] statistics are NOT collected here - the first Wanda cell
collects them from this cache and every later cell, in either stage, reuses them.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:  # hydra-core is an optional dependency until real runs begin
    from hydra import main as hydra_main  # type: ignore[import-not-found]
    from omegaconf import DictConfig  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - CI/test environment
    hydra_main = None  # type: ignore[assignment]

from src.calibration.token_cache import build_token_cache


def _resolve(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, "to_container"):  # OmegaConf DictConfig
        return cfg.to_container(resolve=True)  # type: ignore[attr-defined]
    raise TypeError(f"expected Mapping or OmegaConf config, got {type(cfg).__name__}")


def build_calibration_cache(cfg: Any) -> Path:
    path = build_token_cache(_resolve(cfg))
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    print(f"calibration cache: {path}")
    print(f"  shape {meta['shape']}  fingerprint {meta['fingerprint']}")
    print(f"  fineweb-edu @ {meta.get('dataset_revision')}  tokenizer @ {meta.get('tokenizer_revision')}")
    return path


def main() -> None:
    """Hydra entrypoint: ``python experiments/build_calibration_cache.py`` (from project root)."""
    if hydra_main is None:  # pragma: no cover
        print("hydra-core + omegaconf are required to run the entrypoint (optional dep for tests).", file=sys.stderr)
        sys.exit(1)

    @hydra_main(version_base=None, config_path="../configs", config_name="config")
    def _run(cfg: DictConfig) -> None:
        build_calibration_cache(cfg)

    _run()


if __name__ == "__main__":
    main()
