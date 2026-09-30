# [AI-GEN] agent=Claude date=2026-09-27 task=Tests for the preflight gate that decides whether science may run
# reviewed-by: PENDING

"""The preflight gate (deploy/shared/preflight_blockers.py, deploy/BLOCKERS.md).

``_common.sh::assert_preflight`` refuses to start the science queue while this script
reports any blocker unimplemented, so both failure directions are expensive:

- a false **BLOCKED** keeps the gate shut after the work is genuinely done (this happened:
  the compressor probe passed a bare sentinel object, which a correct compressor refuses
  with ``NotImplementedError`` — the same signal the probe uses to mean "still a stub", so
  RTN read as BLOCKED after its real path landed);
- a false **OK** lets a queue start against a stub and waste hours of Spark time.

The script lives under ``deploy/`` rather than in a package, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import ClassVar

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "deploy" / "shared" / "preflight_blockers.py"


@pytest.fixture(scope="module")
def preflight():
    spec = importlib.util.spec_from_file_location("_preflight_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_compressor_probe_is_handed_a_real_torch_module(preflight):
    """A bare sentinel cannot distinguish "refuses junk" from "not implemented"."""
    torch = pytest.importorskip("torch")
    probe = preflight._probe_model()
    assert isinstance(probe, torch.nn.Module), (
        "the probe must be a real torch module, or an implemented compressor will "
        "correctly refuse it and be misreported as a stub"
    )

    from src.compression.rtn import quantizable_parameters

    assert quantizable_parameters(probe), (
        "the probe carries no projection-shaped parameter, so a compressor would have "
        "nothing to act on and could pass without doing anything"
    )


def test_b2_status_matches_which_families_are_actually_implemented(preflight):
    """RTN and magnitude landed 2026-09-27; Wanda, GPTQ and AWQ 2026-09-29."""
    pytest.importorskip("torch")
    status = {
        label.replace("B2 compressor: ", ""): ok
        for label, ok, _ in preflight.check_compressors()
    }
    for family in ("rtn", "gptq", "awq", "magnitude", "wanda"):
        assert status[family] is True, f"{family} is implemented; a BLOCKED keeps the gate shut"


def test_the_calibrated_quantizers_are_probed_on_a_real_transformerlens_model(preflight):
    """GPTQ and AWQ walk model.blocks; handed the bare-parameter probe they raise
    TypeError, which _probe would score as "real path entered" - a false OK."""
    pytest.importorskip("transformer_lens")
    model = preflight._probe_tl_model()
    assert hasattr(model, "blocks") and hasattr(model, "cfg")
    assert model.cfg.original_architecture == "LlamaForCausalLM"
    assert model.cfg.d_model % 128 == 0 and model.cfg.d_mlp % 128 == 0, "AWQ's group size is 128"

    from src.compression.gptq import GptqCompressor

    with pytest.raises(TypeError):
        GptqCompressor(calibration=preflight._probe_tokens()).apply(preflight._probe_model(), None)


def test_the_wanda_probe_statistics_can_tell_wanda_from_magnitude(preflight):
    """With a constant E[x^2], Wanda's score is a rescaled |W| and the probe would pass a
    Wanda that ignored its statistics."""
    pytest.importorskip("torch")
    moments = preflight._probe_second_moments()
    assert moments, "the probe supplies no statistics, so Wanda cannot be probed"
    for name, stat in moments.items():
        assert (stat > 0).all() and stat.std() > 0, name


class TestCalibrationData:
    """Q7 is data, not code: reported separately so a missing cache does not read as a
    stub, and so building the cache - not engineering - is what opens it."""

    def _write_caches(self, preflight, root, *, tamper=None):
        """Synthetic caches under the real filenames; returns a record of them."""
        import numpy as np
        import yaml

        from src.calibration.token_cache import cache_path, fingerprint, save_token_cache

        registry = {"caches": {}}
        calibration = yaml.safe_load((REPO / "configs" / "calibration" / "final.yaml").read_text(encoding="utf-8"))
        for model in preflight._calibrated_models():
            model_cfg = yaml.safe_load((REPO / "configs" / "model" / f"{model}.yaml").read_text(encoding="utf-8"))
            path = cache_path({"calibration": calibration, "model": model_cfg, "calibration_root": str(root)})
            tokens = np.arange(512, dtype=np.int64).reshape(2, 256)
            save_token_cache(path, tokens, {})
            registry["caches"][path.name] = {"fingerprint": fingerprint(tokens)}
            if model == tamper:
                np.save(path, tokens[::-1].copy())
        return registry

    def test_every_model_with_a_calibrated_cell_needs_a_cache(self, preflight):
        assert preflight._calibrated_models() == ["gemma2_2b", "llama32_1b", "pythia160m", "pythia410m"]

    def test_absent_caches_block_and_name_the_models(self, preflight, tmp_path):
        label, ok, detail = preflight.check_calibration_data(tmp_path)
        assert label == "Q7 calibration caches" and ok is False
        assert "not built for" in detail and "gemma2_2b" in detail
        assert "allow_external_dataset_download" in detail

    def test_recorded_caches_open_it(self, preflight, tmp_path):
        registry = self._write_caches(preflight, tmp_path)
        _, ok, detail = preflight.check_calibration_data(tmp_path, registry)
        assert ok is True, detail
        assert "matching the recorded fingerprints for 4 models" in detail

    def test_an_altered_cache_blocks(self, preflight, tmp_path):
        registry = self._write_caches(preflight, tmp_path, tamper="llama32_1b")
        _, ok, detail = preflight.check_calibration_data(tmp_path, registry)
        assert ok is False and "failed verification" in detail and "llama32_1b" in detail

    def test_a_self_consistent_rebuild_that_differs_from_the_record_blocks(
        self, preflight, tmp_path
    ):
        """The case the sidecar cannot catch: a rebuild writes a sidecar matching itself."""
        registry = self._write_caches(preflight, tmp_path)
        name = next(n for n in registry["caches"] if n.startswith("google_gemma"))
        registry["caches"][name] = {"fingerprint": "0" * 64}
        _, ok, detail = preflight.check_calibration_data(tmp_path, registry)
        assert ok is False
        assert "differs from the recorded build" in detail and "gemma2_2b" in detail

    def test_a_cache_missing_from_the_record_blocks(self, preflight, tmp_path):
        registry = self._write_caches(preflight, tmp_path)
        registry["caches"] = {n: v for n, v in registry["caches"].items() if "Llama" not in n}
        _, ok, detail = preflight.check_calibration_data(tmp_path, registry)
        assert ok is False and "no recorded fingerprint" in detail and "llama32_1b" in detail

    def test_the_tracked_record_covers_every_model_that_needs_a_cache(self, preflight):
        import json

        import yaml

        from src.calibration.token_cache import cache_path

        record = json.loads(preflight.FINGERPRINTS.read_text(encoding="utf-8"))
        calibration = yaml.safe_load(
            (REPO / "configs" / "calibration" / "final.yaml").read_text(encoding="utf-8")
        )
        for model in preflight._calibrated_models():
            model_cfg = yaml.safe_load(
                (REPO / "configs" / "model" / f"{model}.yaml").read_text(encoding="utf-8")
            )
            name = cache_path({"calibration": calibration, "model": model_cfg}).name
            entry = record["caches"][name]
            assert len(entry["fingerprint"]) == 64, model
            assert entry["tokenizer_revision"] == model_cfg["hf_revision"], model
            assert entry["shape"] == [300_000 // 256, 256], model


def test_an_implemented_family_reads_ok_by_succeeding_not_by_erroring(preflight):
    """Guards the false-OK class of bug directly.

    ``_probe`` scores any non-NotImplementedError as "real path entered", so a compressor
    that blew up on the probe's inputs would read as implemented. Magnitude pruning did
    exactly that when the probe carried no ``embed.W_E``: it raised ValueError and scored
    OK while pruning nothing.
    """
    pytest.importorskip("torch")
    details = {
        label.replace("B2 compressor: ", ""): detail
        for label, ok, detail in preflight.check_compressors()
        if ok
    }
    for family, detail in details.items():
        assert detail == "returned without error", (
            f"{family} reads OK via {detail!r} rather than by completing; the probe inputs "
            "are wrong for it and the OK is not evidence the path works"
        )


def test_the_probe_model_carries_the_names_the_pruners_special_case(preflight):
    """embed.W_E joins the threshold pool and unembed.W_U is excluded, so a probe missing
    them cannot exercise the pruning rule at all."""
    pytest.importorskip("torch")
    names = {n for n, _ in preflight._probe_model().named_parameters()}
    assert "embed.W_E" in names
    assert "unembed.W_U" in names


def test_a_family_needs_both_apply_and_weight_delta(preflight):
    """Stage C needs apply(); Stage B needs weight_delta() for the null magnitudes.

    Either one alone is unusable, so the probe must require both. Verified through the
    real reporting path with a compressor that implements only half the protocol.
    """
    pytest.importorskip("torch")

    class _HalfDone:
        def apply(self, model, cfg):
            return model

        def weight_delta(self, model, cfg):
            raise NotImplementedError("stub")

    inst = _HalfDone()
    apply_ok, _ = preflight._probe(inst.apply, preflight._probe_model(), None)
    delta_ok, detail = preflight._probe(inst.weight_delta, preflight._probe_model(), None)
    assert apply_ok and not delta_ok
    assert "NotImplementedError" in detail


def test_the_null_perturber_reads_implemented(preflight):
    """B1 closed 2026-09-27 (PI-approved): the Perturber now handles torch modules.

    The probe must hand it a real module - with a bare sentinel a correct perturber
    raises NotImplementedError and would be misreported as a stub, exactly as RTN was.
    """
    pytest.importorskip("torch")
    _, ok, detail = preflight.check_null_perturber()
    assert ok is True, f"B1 reads blocked: {detail}"


def test_the_b1_probe_actually_perturbs_something(preflight):
    """Guards against a false OK: a perturber that returned the model untouched would
    also "return without error" and read as implemented."""
    torch = pytest.importorskip("torch")
    from src.common.seeding import create_seed_generator
    from src.science.matched_magnitude import MatchedMagnitudePerturber

    model = preflight._probe_model()
    magnitudes = {name: 1.0 for name, _ in model.named_parameters()}
    rng, _ = create_seed_generator(0)
    perturbed = MatchedMagnitudePerturber().apply(model, magnitudes, rng)

    before = dict(model.named_parameters())
    moved = [n for n, p in perturbed.named_parameters() if not torch.equal(before[n], p)]
    assert set(moved) == set(magnitudes), "the probe model was not actually perturbed"


def test_the_gate_is_still_shut_overall(preflight):
    """Whole-script verdict: science must not be clear to run yet."""
    pytest.importorskip("torch")
    results = [preflight.check_null_perturber(), *preflight.check_compressors(),
               preflight.check_calibration_data(), preflight.check_chance_floor(),
               preflight.check_exit_gate()]
    assert not all(ok for _, ok, _ in results), (
        "the preflight gate reads clear; if that is intended, the remaining blockers in "
        "deploy/BLOCKERS.md should have been closed first"
    )


class TestExitGate:
    """B6 opens only on a pass record earned against the PI's reference file as it is now.

    The old check grepped the test file for "skip", which it always contains, so B6 could
    never have opened - a permanent false BLOCKED."""

    REFERENCE: ClassVar[dict] = {
        "edges": ["a->b", "b->c"], "source": "x", "decided_on": "2026-10-01",
        "criterion": {"min_precision": 0.3, "max_p_value": 0.001, "seed": 5},
    }
    PASS: ClassVar[dict] = {"passed": True, "precision": 0.73, "p_value": 1e-30, "date": "2026-10-02"}

    def _files(self, tmp_path, *, record=None, reference=None):
        import hashlib
        import json

        ref = tmp_path / "ref.json"
        ref.write_text(json.dumps(reference or self.REFERENCE), encoding="utf-8")
        rec = tmp_path / "pass.json"
        if record is not None:
            record = {"reference_sha256": hashlib.sha256(ref.read_bytes()).hexdigest(), **record}
            rec.write_text(json.dumps(record), encoding="utf-8")
        return ref, rec

    def test_no_reference_file_blocks_and_says_it_is_the_pis(self, preflight, tmp_path):
        _, ok, detail = preflight.check_exit_gate(tmp_path / "absent.json", tmp_path / "pass.json")
        assert ok is False and "PI" in detail and "criterion" in detail

    def test_a_reference_without_a_pass_blocks(self, preflight, tmp_path):
        ref, rec = self._files(tmp_path)
        _, ok, detail = preflight.check_exit_gate(ref, rec)
        assert ok is False and "has not passed" in detail

    def test_a_pass_on_the_current_reference_opens_it(self, preflight, tmp_path):
        ref, rec = self._files(tmp_path, record=self.PASS)
        _, ok, detail = preflight.check_exit_gate(ref, rec)
        assert ok is True and "precision 0.730 >= 0.3" in detail and "<= 0.001" in detail

    def test_a_pass_on_an_edited_reference_blocks(self, preflight, tmp_path):
        ref, rec = self._files(tmp_path, record=self.PASS)
        ref.write_text(ref.read_text(encoding="utf-8").replace("0.3", "0.2"), encoding="utf-8")
        _, ok, detail = preflight.check_exit_gate(ref, rec)
        assert ok is False and "different reference" in detail

    @pytest.mark.parametrize("change", [{"precision": 0.2}, {"p_value": 0.01}, {"passed": False}])
    def test_a_record_that_misses_the_criterion_blocks(self, preflight, tmp_path, change):
        ref, rec = self._files(tmp_path, record={**self.PASS, **change})
        _, ok, detail = preflight.check_exit_gate(ref, rec)
        assert ok is False and "does not meet the criterion" in detail

    def test_a_record_from_the_superseded_jaccard_gate_blocks(self, preflight, tmp_path):
        ref, rec = self._files(tmp_path, record={"passed": True, "jaccard": 0.85})
        _, ok, _ = preflight.check_exit_gate(ref, rec)
        assert ok is False

    def test_a_malformed_record_blocks(self, preflight, tmp_path):
        ref, rec = self._files(tmp_path, record={"passed": True})
        _, ok, _ = preflight.check_exit_gate(ref, rec)
        assert ok is False

    def test_the_gate_and_the_preflight_agree_on_the_file_paths(self, preflight):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_gate", REPO / "tests" / "test_regression_ioi_gpt2_small.py"
        )
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        assert gate.REFERENCE_FILE == preflight.GATE_REFERENCE
        assert gate.PASS_RECORD == preflight.GATE_PASS
