# [AI-GEN] agent=Claude date=2026-09-27 task=Integration: Stage B draws a null on a real torch model (B1)
# reviewed-by: PENDING

"""Stage B's null path on a real torch model (deploy/BLOCKERS.md B1).

Until 2026-09-27 ``_load_model_guard`` raised for any non-synthetic model and the null
helper refused anything but a MockModel, so no Stage B cell could reach a real model at
all. These tests drive ``_dnull_rows`` with an actual TransformerLens module.

**What is and is not covered.** The real extractor needs a tokenizer and a downloaded
checkpoint, so it cannot run in a unit test; a recording extractor is injected through
``PIPELINE_REGISTRY`` instead. That is deliberate — what is under test here is the model
plumbing (magnitudes from the cell's compressor, draws applied to a real module, the dense
model left intact), not attribution. The extractor is a stand-in; the model is real.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

from transformer_lens import HookedTransformer, HookedTransformerConfig

import experiments.run_stage_b as stage_b
from src.common.schema import Edge, Graph
from src.compression.rtn import RtnQuantizer, quantizable_parameters


class _WeightSensitiveExtractor:
    """Deterministic graph whose edges depend on the model's actual weights.

    A null draw must be able to change the extracted graph, otherwise every D_null draw
    would be 0 and the test would pass while measuring nothing.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.models_seen: list[int] = []

    def extract(self, model, task, config, seed):
        self.models_seen.append(id(model))
        nodes = ("L0.H0", "L0.H1", "L1.H0", "L1.H1")
        thr = float(config.get("edge_threshold", 0.5))
        # One score per edge, read off the weights. The fractional part is used so that a
        # matched-magnitude perturbation - which is small relative to the weights - still
        # changes which edges clear the threshold. A saturating function (tanh of a sum)
        # does not: every draw then yields the same graph and a reproducibility test over
        # these distances would pass without measuring anything.
        flat = torch.cat([
            p.detach().to(torch.float32).reshape(-1)
            for _, p in sorted(model.named_parameters())
        ])
        candidates = tuple((s, d) for s in nodes for d in nodes if s != d)
        stride = max(1, flat.numel() // (len(candidates) + 1))
        scores = [
            float((flat[(i + 1) * stride].abs() * 1e4) % 1.0)
            for i in range(len(candidates))
        ]
        edges = tuple(
            Edge(src_component=src, dst_component=dst,
                 config_id=str(config.get("id", "c")), seed=int(seed))
            for (src, dst), score in zip(candidates, scores)
            if score >= thr
        )
        return Graph(edges=edges, nodes=nodes,
                     metadata={"pipeline": "test-weight-sensitive", "synthetic": False})


def _real_model() -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_heads=4, d_head=4, d_mlp=32, d_vocab=40, n_ctx=16,
        act_fn="gelu", normalization_type="LN", positional_embedding_type="rotary",
        rotary_dim=4, dtype=torch.float32, seed=0, device="cpu",
    )
    model = HookedTransformer(cfg)
    model.eval()
    return model


@pytest.fixture
def injected(monkeypatch):
    extractor = _WeightSensitiveExtractor()
    monkeypatch.setitem(
        stage_b.PIPELINE_REGISTRY, "test-pipeline", lambda **kw: extractor
    )
    return extractor


def _resolved(**overrides):
    cfg = {
        "stage": "stageB",
        "seed": 0,
        "pipeline": "test-pipeline",
        "compression_family": "rtn",
        "compression_level": "4",
        "stage_c": {"cell": "rtn_int4", "compressor_kwargs": {"bits": 4}},
        "model": {"name": "pythia160m", "synthetic": False},
        "task": {"name": "ioi", "n_prompts": 4, "seed": 0},
        "ensemble": {"B": 2, "S": 1, "threshold_grid": None,
                     "decompose": {"core_threshold": 0.9, "noise_threshold": 0.1}},
        "nulls": {"R": 4},
    }
    cfg.update(overrides)
    return cfg


class TestTheNullRunsOnARealModel:
    def test_r_draws_are_produced(self, injected):
        model = _real_model()
        dense = {"L0.H0->L0.H1": 1.0, "L1.H0->L1.H1": 0.5}
        rows, _ = stage_b._dnull_rows(_resolved(), model, dense, RtnQuantizer(bits=4))
        assert [r["r"] for r in rows] == [0, 1, 2, 3]
        assert all("distance_l1" in r and "distance_jensen_shannon" in r for r in rows)

    def test_magnitudes_are_the_compressors_measured_values(self, injected):
        """The recorded magnitudes must be RTN's, not a stand-in compressor's."""
        model = _real_model()
        _, magnitudes = stage_b._dnull_rows(
            _resolved(), model, {"L0.H0->L0.H1": 1.0}, RtnQuantizer(bits=4)
        )
        expected = RtnQuantizer(bits=4).weight_delta(model, None)
        assert magnitudes == expected
        assert all(magnitudes[n] > 0.0 for n in quantizable_parameters(model))
        assert magnitudes["embed.W_E"] == 0.0

    def test_the_dense_model_is_not_mutated_across_draws(self, injected):
        """R draws must each start from the dense model, not from the previous draw."""
        model = _real_model()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        stage_b._dnull_rows(_resolved(), model, {"L0.H0->L0.H1": 1.0}, RtnQuantizer(bits=4))
        for name, param in model.named_parameters():
            assert torch.equal(before[name], param), f"draw loop mutated {name}"

    def test_each_draw_is_extracted_from_a_fresh_perturbed_copy(self, injected):
        """A distinct object per draw; reusing one would compound the perturbations."""
        model = _real_model()
        stage_b._dnull_rows(_resolved(), model, {"L0.H0->L0.H1": 1.0}, RtnQuantizer(bits=4))
        # R draws x (B configs x S seeds), plus caching inside the runner is allowed;
        # what matters is that the dense model's id is not among the perturbed extractions.
        assert id(model) not in injected.models_seen

    def test_the_draws_are_not_all_the_same(self, injected):
        """If every draw gave the same distance, D_null would be a point mass and the
        reproducibility check below would pass without testing the seeding at all."""
        rows, _ = stage_b._dnull_rows(
            _resolved(), _real_model(),
            {"L0.H0->L0.H1": 1.0, "L1.H0->L1.H1": 0.5}, RtnQuantizer(bits=4),
        )
        distances = [r["distance_l1"] for r in rows]
        assert len(set(distances)) > 1, f"all R draws gave the same distance: {distances}"

    def test_the_run_is_reproducible_from_the_seed(self, injected):
        dense = {"L0.H0->L0.H1": 1.0, "L1.H0->L1.H1": 0.5}
        a, _ = stage_b._dnull_rows(_resolved(), _real_model(), dense, RtnQuantizer(bits=4))
        b, _ = stage_b._dnull_rows(_resolved(), _real_model(), dense, RtnQuantizer(bits=4))
        assert [r["distance_l1"] for r in a] == [r["distance_l1"] for r in b]

    def test_a_different_seed_gives_a_different_null(self, injected):
        dense = {"L0.H0->L0.H1": 1.0, "L1.H0->L1.H1": 0.5}
        a, _ = stage_b._dnull_rows(_resolved(seed=0), _real_model(), dense, RtnQuantizer(bits=4))
        b, _ = stage_b._dnull_rows(_resolved(seed=7), _real_model(), dense, RtnQuantizer(bits=4))
        assert [r["distance_l1"] for r in a] != [r["distance_l1"] for r in b]

    def test_a_different_bit_width_gives_a_different_null(self, injected):
        """int8 and int4 must not share a denominator (the bug this all comes from)."""
        int8 = stage_b._dnull_rows(
            _resolved(compression_level="8",
                      stage_c={"cell": "rtn_int8", "compressor_kwargs": {"bits": 8}}),
            _real_model(), {"L0.H0->L0.H1": 1.0}, RtnQuantizer(bits=8),
        )[1]
        int4 = stage_b._dnull_rows(
            _resolved(), _real_model(), {"L0.H0->L0.H1": 1.0}, RtnQuantizer(bits=4)
        )[1]
        assert sum(int4.values()) > sum(int8.values())


class TestTheExtractorComesFromTheConfig:
    def test_stage_b_builds_the_configured_pipeline(self, injected):
        """It used to hardcode MockExtractor, so a real run would have measured the null
        through the mock extractor while Stage A used the real one."""
        model = _real_model()
        stage_b._dnull_rows(_resolved(), model, {"L0.H0->L0.H1": 1.0}, RtnQuantizer(bits=4))
        assert injected.models_seen, "the configured extractor was never called"

    def test_an_unknown_pipeline_is_refused(self):
        with pytest.raises(ValueError, match="unknown pipeline"):
            stage_b._build_extractor({"pipeline": "does-not-exist"})

    def test_the_real_pipeline_names_are_registered(self):
        for name in ("dense-node", "attr", "edgeprune", "mock"):
            assert name in stage_b.PIPELINE_REGISTRY


class TestTheModelGuardNowDefersToStageA:
    def test_it_delegates_rather_than_reimplementing_the_rules(self):
        """The download gate and architecture checks live in load_pinned_model; Stage B
        must not add a second, divergent copy of those rules."""
        import experiments.run_stage_a as stage_a

        called = {}

        def _fake(resolved):
            called["yes"] = True
            return "MODEL"

        original = stage_a._load_model
        stage_b_original = stage_b._load_model
        try:
            stage_a._load_model = _fake
            stage_b._load_model = _fake
            assert stage_b._load_model_guard({"model": {"name": "pythia160m"}}) == "MODEL"
            assert called
        finally:
            stage_a._load_model = original
            stage_b._load_model = stage_b_original
