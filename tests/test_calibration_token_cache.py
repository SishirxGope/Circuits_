# [AI-GEN] agent=Claude date=2026-09-29 task=Tests for the Q7 calibration token cache
# reviewed-by: PENDING

"""The calibration token cache (src/calibration/token_cache.py).

Everything a calibrated compressor learns about activations comes from this array, so
the tests pin three things:

1. **The packing rule is upstream's.** Checked against the notebook's loop itself,
   reproduced below, on documents that exercise every branch (short documents, tails,
   the stop condition, the final floor).
2. **The download is gated in code**, before anything can touch the network.
3. **A cache cannot change silently.** Every load re-verifies the fingerprint written
   when it was built, so Plan A and Plan B cannot calibrate on different tokens.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.calibration.token_cache import (
    UPSTREAM_BUILD,
    CalibrationUnavailable,
    assert_download_allowed,
    build_token_cache,
    cache_filename,
    cache_path,
    calibration_spec,
    fingerprint,
    load_token_cache,
    pack_documents,
    save_token_cache,
    synthetic_token_cache,
    validate_token_cache,
)

REPO = Path(__file__).resolve().parents[1]


def _encode(text: str) -> list[int]:
    """A character-level stand-in tokenizer: deterministic and length-preserving."""
    return [ord(ch) % 97 for ch in text]


def _upstream_notebook_pack(texts, encode, n_tokens, context_len):
    """notebooks/SAE_Pruning_Reproducibility.ipynb cell 5, minus dataset/tokenizer plumbing."""
    all_ids = []
    total = 0
    for text in texts:
        if len(text.strip()) < 50:
            continue
        ids = encode(text)
        for i in range(0, len(ids) - context_len + 1, context_len):
            all_ids.append(ids[i:i + context_len])
            total += context_len
            if total >= n_tokens:
                break
        if total >= n_tokens:
            break
    return np.array(all_ids[:n_tokens // context_len], dtype=np.int64)


def _documents(n: int, seed: int = 0) -> list[str]:
    """Random lengths from 0 to 600 characters: some below the 50-character floor, most
    with a tail that does not fill a window."""
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("abcdefghij klmnop"))
    return ["".join(rng.choice(alphabet, size=int(rng.integers(0, 600)))) for _ in range(n)]


def _real_calibration_block() -> dict:
    return yaml.safe_load((REPO / "configs" / "calibration" / "final.yaml").read_text(encoding="utf-8"))


class TestThePackingRuleIsUpstreams:
    @pytest.mark.parametrize(
        ("n_tokens", "context_len"), [(1000, 64), (1024, 64), (2000, 128), (777, 32), (4096, 256)]
    )
    def test_matches_the_notebook_loop(self, n_tokens, context_len):
        docs = _documents(400)
        ours = pack_documents(docs, _encode, n_tokens=n_tokens, context_len=context_len)
        theirs = _upstream_notebook_pack(docs, _encode, n_tokens, context_len)
        np.testing.assert_array_equal(ours, theirs)

    def test_the_pre_registered_budget_gives_1171_windows_of_256(self):
        """300,000 // 256 = 1,171 windows = 299,776 tokens: the documented floor."""
        docs = _documents(3000, seed=1)
        out = pack_documents(docs, _encode, n_tokens=300_000, context_len=256)
        assert out.shape == (1171, 256)
        assert out.size == 299_776

    def test_documents_under_the_character_floor_contribute_nothing(self):
        short = "x" * 49                      # 49 chars: skipped
        padded = "   " + "y" * 49 + "   "      # strips to 49: skipped
        long = "z" * 64                       # 64 chars: exactly one window of 64
        out = pack_documents([short, padded, None, long], _encode, n_tokens=64, context_len=64)
        np.testing.assert_array_equal(out, [_encode(long)])

    def test_windows_do_not_overlap_and_the_tail_is_dropped(self):
        doc = "".join(chr(97 + i % 26) for i in range(100))  # 100 tokens -> 3 windows of 32
        out = pack_documents([doc], _encode, n_tokens=96, context_len=32)
        ids = _encode(doc)
        np.testing.assert_array_equal(out, [ids[0:32], ids[32:64], ids[64:96]])

    def test_no_bos_or_other_token_is_inserted(self):
        doc = "q" * 128
        out = pack_documents([doc], _encode, n_tokens=128, context_len=64)
        assert set(out.ravel().tolist()) == {_encode("q")[0]}

    def test_a_stream_that_runs_out_raises_instead_of_returning_a_short_cache(self):
        with pytest.raises(ValueError, match="ran out"):
            pack_documents(["w" * 100], _encode, n_tokens=1000, context_len=32)

    def test_a_budget_under_one_window_raises(self):
        with pytest.raises(ValueError, match="less than one window"):
            pack_documents(["w" * 100], _encode, n_tokens=10, context_len=32)

    def test_output_is_int64_and_two_dimensional(self):
        out = pack_documents(_documents(50), _encode, n_tokens=256, context_len=64)
        assert out.dtype == np.int64 and out.shape == (4, 64)


class TestTheConfigBlock:
    def test_the_pre_registered_values_come_through(self):
        spec = calibration_spec({"calibration": _real_calibration_block()})
        assert spec["hf_id"] == "HuggingFaceFW/fineweb-edu"
        assert spec["n_tokens"] == 300_000
        assert spec["seed"] == 7
        assert spec["dtype"] == "bfloat16"

    def test_a_null_context_length_takes_upstreams_value_and_says_so(self):
        """final.yaml leaves it null rather than invent it; the provenance must record
        where the 256 came from, or it reads as a pre-registered value."""
        spec = calibration_spec({"calibration": _real_calibration_block()})
        assert spec["context_len"] == UPSTREAM_BUILD["context_len"] == 256
        assert "upstream" in spec["context_len_source"]

    def test_an_explicit_context_length_wins(self):
        block = _real_calibration_block()
        block["calibration"]["context_length"] = 128
        spec = calibration_spec({"calibration": block})
        assert spec["context_len"] == 128
        assert spec["context_len_source"] == "configs/calibration"

    def test_a_missing_block_is_an_error_not_a_default(self):
        with pytest.raises(ValueError, match="missing"):
            calibration_spec({})

    def test_the_filename_is_upstreams(self):
        """saediag/models.py:68: f"{model_id.replace('/', '_')}_calib_{n_tokens}_{seed}.npy"."""
        assert cache_filename("google/gemma-2-2b", 300_000, 7) == "google_gemma-2-2b_calib_300000_7.npy"

    def test_the_cache_is_per_tokenizer(self):
        block = _real_calibration_block()
        a = cache_path({"calibration": block, "model": {"hf_id": "EleutherAI/pythia-160m"}})
        b = cache_path({"calibration": block, "model": {"hf_id": "google/gemma-2-2b"}})
        assert a != b and a.parent == b.parent


class TestFingerprintAndValidation:
    def test_identical_arrays_share_a_fingerprint(self):
        a = synthetic_token_cache(4, 8, 50, seed=3)
        assert fingerprint(a) == fingerprint(a.copy())

    def test_any_change_moves_the_fingerprint(self):
        a = synthetic_token_cache(4, 8, 50, seed=3)
        b = a.copy()
        b[2, 5] = (b[2, 5] + 1) % 50
        assert fingerprint(a) != fingerprint(b)

    def test_shape_is_part_of_the_fingerprint(self):
        a = synthetic_token_cache(4, 8, 50, seed=3)
        assert fingerprint(a) != fingerprint(a.reshape(8, 4))

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            (np.zeros(8, dtype=np.int64), "2-D"),
            (np.zeros((2, 4), dtype=np.float32), "integer"),
            (np.zeros((0, 4), dtype=np.int64), "empty"),
            (np.full((2, 4), -1, dtype=np.int64), "negative"),
        ],
    )
    def test_malformed_caches_are_refused(self, bad, match):
        with pytest.raises(ValueError, match=match):
            validate_token_cache(bad)

    def test_an_id_outside_the_vocabulary_names_the_tokenizer_mismatch(self):
        with pytest.raises(ValueError, match="different tokenizer"):
            validate_token_cache(np.full((2, 4), 60, dtype=np.int64), vocab_size=50)


class TestSaveAndLoad:
    def test_round_trip_preserves_tokens_and_provenance(self, tmp_path):
        tokens = synthetic_token_cache(5, 16, 50, seed=0)
        path = save_token_cache(tmp_path / "c.npy", tokens, {"seed": 7})
        loaded, meta = load_token_cache(path, vocab_size=50)
        np.testing.assert_array_equal(loaded, tokens)
        assert meta["seed"] == 7
        assert meta["fingerprint"] == fingerprint(tokens)
        assert meta["shape"] == [5, 16]

    def test_an_altered_cache_is_refused(self, tmp_path):
        tokens = synthetic_token_cache(5, 16, 50, seed=0)
        path = save_token_cache(tmp_path / "c.npy", tokens, {})
        tampered = tokens.copy()
        tampered[0, 0] = (tampered[0, 0] + 1) % 50
        np.save(path, tampered)
        with pytest.raises(ValueError, match="fingerprint"):
            load_token_cache(path)

    def test_a_cache_without_its_sidecar_is_refused(self, tmp_path):
        path = tmp_path / "c.npy"
        np.save(path, synthetic_token_cache(2, 4, 50))
        with pytest.raises(ValueError, match="sidecar"):
            load_token_cache(path)

    def test_an_absent_cache_is_reported_as_missing_data_not_missing_code(self, tmp_path):
        """Preflight distinguishes the two; NotImplementedError would read as a stub."""
        with pytest.raises(CalibrationUnavailable) as info:
            load_token_cache(tmp_path / "absent.npy")
        assert not isinstance(info.value, NotImplementedError)
        assert "allow_external_dataset_download" in str(info.value)


# ------------------------------------------------------------------------- the builder

class _FakeStream:
    def __init__(self, rows):
        self.rows = rows
        self.shuffle_args = None

    def shuffle(self, seed, buffer_size):
        self.shuffle_args = {"seed": seed, "buffer_size": buffer_size}
        return self

    def __iter__(self):
        return iter(self.rows)


class _FakeTokenizer:
    def __init__(self):
        self.special_flags = set()

    def encode(self, text, add_special_tokens):
        self.special_flags.add(add_special_tokens)
        return _encode(text)


@pytest.fixture
def fake_hub(monkeypatch):
    """Stand-ins for `datasets` and `huggingface_hub` that record how they were called."""
    calls: dict = {"load": []}
    rows = [{"text": doc} for doc in _documents(200, seed=5)]

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.__version__ = "fake-9.9"

    def load_dataset(*args, **kwargs):
        stream = _FakeStream(rows)
        calls["load"].append((args, kwargs, stream))
        return stream

    fake_datasets.load_dataset = load_dataset
    fake_hf = types.ModuleType("huggingface_hub")

    class HfApi:
        def dataset_info(self, repo_id):
            calls["info"] = repo_id
            return types.SimpleNamespace(sha="f" * 40)

    fake_hf.HfApi = HfApi
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    return calls


def _build_config(tmp_path, *, allowed=True, n_tokens=768):
    block = _real_calibration_block()
    block["calibration"]["n_tokens"] = n_tokens
    return {
        "mode": {"allow_external_dataset_download": allowed},
        "calibration": block,
        "model": {"hf_id": "EleutherAI/pythia-160m", "hf_revision": "5" * 40},
        "calibration_root": str(tmp_path),
    }


class TestTheDownloadGate:
    @pytest.mark.parametrize("mode", [{}, {"allow_external_dataset_download": False},
                                      {"allow_external_dataset_download": "true"}])
    def test_anything_but_true_is_refused(self, mode):
        with pytest.raises(PermissionError, match="allow_external_dataset_download"):
            assert_download_allowed({"mode": mode})

    def test_the_real_engineering_mode_forbids_it(self):
        mode = yaml.safe_load((REPO / "configs" / "mode" / "engineering_dry_run.yaml").read_text(encoding="utf-8"))
        with pytest.raises(PermissionError):
            assert_download_allowed({"mode": mode})

    def test_the_gate_is_checked_before_the_network_is_touched(self, tmp_path, fake_hub):
        with pytest.raises(PermissionError):
            build_token_cache(_build_config(tmp_path, allowed=False), tokenizer=_FakeTokenizer())
        assert fake_hub["load"] == [] and "info" not in fake_hub


class TestTheBuilder:
    def test_it_streams_the_pre_registered_corpus_the_way_upstream_did(self, tmp_path, fake_hub):
        tok = _FakeTokenizer()
        build_token_cache(_build_config(tmp_path), tokenizer=tok)
        (args, kwargs, stream), = fake_hub["load"]
        assert args == ("HuggingFaceFW/fineweb-edu", "sample-10BT")
        assert kwargs == {"split": "train", "streaming": True, "revision": "f" * 40}
        assert stream.shuffle_args == {"seed": 7, "buffer_size": 10_000}
        assert tok.special_flags == {False}

    def test_the_dataset_is_pinned_to_the_commit_it_resolved_to(self, tmp_path, fake_hub):
        path = build_token_cache(_build_config(tmp_path), tokenizer=_FakeTokenizer())
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["dataset_revision"] == "f" * 40
        assert fake_hub["load"][0][1]["revision"] == meta["dataset_revision"]

    def test_the_cache_matches_packing_the_same_stream_directly(self, tmp_path, fake_hub):
        path = build_token_cache(_build_config(tmp_path, n_tokens=768), tokenizer=_FakeTokenizer())
        tokens, _ = load_token_cache(path)
        rows = fake_hub["load"][0][2].rows
        expected = _upstream_notebook_pack([r["text"] for r in rows], _encode, 768, 256)
        np.testing.assert_array_equal(tokens, expected)
        assert path.name == "EleutherAI_pythia-160m_calib_768_7.npy"

    def test_provenance_records_everything_needed_to_rebuild_it(self, tmp_path, fake_hub):
        path = build_token_cache(_build_config(tmp_path), tokenizer=_FakeTokenizer())
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        for key in ("hf_id", "dataset_config", "split", "seed", "shuffle_buffer", "min_chars",
                    "context_len", "context_len_source", "add_special_tokens", "tokenizer_id",
                    "tokenizer_revision", "datasets_version", "procedure_source", "fingerprint"):
            assert key in meta, key
        assert meta["tokenizer_revision"] == "5" * 40
        assert meta["context_len"] == 256 and "upstream" in meta["context_len_source"]

    def test_an_existing_cache_is_verified_and_reused_never_rebuilt(self, tmp_path, fake_hub):
        first = build_token_cache(_build_config(tmp_path), tokenizer=_FakeTokenizer())
        second = build_token_cache(_build_config(tmp_path), tokenizer=_FakeTokenizer())
        assert first == second
        assert len(fake_hub["load"]) == 1


class TestTheEntryPoint:
    """experiments/build_calibration_cache.py on the real composed config."""

    def _compose(self, overrides):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
            return compose("config", overrides=overrides)

    def test_the_default_mode_refuses(self, tmp_path, fake_hub):
        from experiments.build_calibration_cache import build_calibration_cache

        cfg = self._compose(["model=pythia160m", f"+calibration_root={tmp_path}"])
        with pytest.raises(PermissionError):
            build_calibration_cache(cfg)
        assert fake_hub["load"] == []

    def test_scientific_mode_builds_for_the_configured_model(self, tmp_path, fake_hub, monkeypatch, capsys):
        from experiments.build_calibration_cache import build_calibration_cache

        requested = {}
        fake_transformers = types.ModuleType("transformers")

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(name, revision=None):
                requested.update(name=name, revision=revision)
                return _FakeTokenizer()

        fake_transformers.AutoTokenizer = AutoTokenizer
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        cfg = self._compose([
            "mode=scientific_run", "model=pythia160m", f"+calibration_root={tmp_path}",
            "calibration.calibration.n_tokens=512",
        ])
        path = build_calibration_cache(cfg)
        assert path.name == "EleutherAI_pythia-160m_calib_512_7.npy"
        assert "fingerprint" in capsys.readouterr().out
        tokens, meta = load_token_cache(path)
        assert tokens.shape == (2, 256)
        assert meta["tokenizer_revision"] == "50f5173d932e8e61f858120bcb800b97af589f46"
        assert requested == {  # the tokenizer is pinned to the model's own revision
            "name": "EleutherAI/pythia-160m",
            "revision": "50f5173d932e8e61f858120bcb800b97af589f46",
        }
