# [AI-GEN] agent=Claude date=2026-09-21 task=Download pinned model weights and corpora, reading the pins from configs/model/*.yaml
# reviewed-by: PENDING
#
# The pins are the reproducibility story (Q5). This script never accepts `main` - it reads
# hf_revision out of the model config and downloads exactly that, so a silently-moved
# upstream HEAD cannot change what you measured.
#
# Usage:
#   python deploy/shared/fetch_assets.py --plan a     # pythia only
#   python deploy/shared/fetch_assets.py --plan b     # all four models
#   python deploy/shared/fetch_assets.py --plan b --datasets

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PLAN_A = ["pythia160m", "pythia410m"]
PLAN_B = ["pythia160m", "pythia410m", "gemma2_2b", "llama32_1b"]


def _read_pin(cfg_name: str) -> tuple[str, str]:
    import yaml

    path = REPO / "configs" / "model" / f"{cfg_name}.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    hf_id, rev = cfg.get("hf_id"), cfg.get("hf_revision")
    if not hf_id or not rev:
        raise SystemExit(f"{path.name}: missing hf_id or hf_revision - refusing to guess")
    return str(hf_id), str(rev)


def fetch_models(names: list[str]) -> int:
    from huggingface_hub import snapshot_download

    failed = 0
    for name in names:
        hf_id, rev = _read_pin(name)
        print(f"  {name:12s} {hf_id} @ {rev[:12]}")
        try:
            snapshot_download(hf_id, revision=rev)
            print("               cached")
        except Exception as exc:  # noqa: BLE001
            kind = type(exc).__name__
            print(f"               FAILED ({kind}): {str(exc)[:120]}")
            if "Gated" in kind or "gated" in str(exc) or "401" in str(exc):
                print("               -> licence not accepted yet. Accept it on the model page")
                print("                  and wait for approval; nothing works around this.")
            failed += 1
    return failed


def fetch_datasets() -> int:
    """Calibration + perplexity corpora, per configs/calibration/final.yaml (Q7)."""
    from datasets import load_dataset

    jobs = [
        ("FineWeb-Edu (calibration, 300k tokens, seed 7)",
         lambda: load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                              split="train", streaming=True)),
        # Canonical id, not the bare "wikitext": datasets>=5 parses the repo id as an
        # hf:// URI and rejects a name with no namespace before the Hub can redirect it.
        # Verified 2026-09-22: HfApi().dataset_info("wikitext").id == "Salesforce/wikitext".
        ("WikiText-2 test (perplexity)",
         lambda: load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")),
    ]
    failed = 0
    for label, fn in jobs:
        print(f"  {label}")
        try:
            fn()
            print("               ok")
        except Exception as exc:  # noqa: BLE001
            print(f"               FAILED: {str(exc)[:120]}")
            failed += 1
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["a", "b"], default="a")
    ap.add_argument("--datasets", action="store_true", help="also fetch the corpora")
    args = ap.parse_args()

    names = PLAN_A if args.plan == "a" else PLAN_B
    print(f"=== Pinned model weights (plan {args.plan.upper()}: {len(names)} models) ===")
    failed = fetch_models(names)

    if args.datasets:
        print("\n=== Corpora (Q7) ===")
        failed += fetch_datasets()

    print()
    if failed:
        print(f"{failed} item(s) failed. Resolve before running science.")
        return 1
    print("All assets cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
