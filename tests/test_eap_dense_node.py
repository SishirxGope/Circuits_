# [AI-GEN] agent=Claude date=2026-09-14 task=Correctness tests for EAP, dense-graph pruning and the dense-node extractor
# reviewed-by: PENDING

"""Edge attribution patching, threshold pruning and the dense-node extractor, offline.

Everything runs on tiny randomly-initialised TransformerLens models in float64 - no
download, no GPU. The central test is not a smoke test: it checks every EAP edge score
against a finite difference of an EXACT additive edge patch, which is the quantity EAP
claims to compute to first order. That check is independent of how nonlinear the model
is, so it can only pass if the hooks, the deltas, the gradients and the aggregation are
all right.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

from transformer_lens import HookedTransformer, HookedTransformerConfig  # noqa: E402

from src.extraction.dense_prune import find_threshold, owner, prune_dense_graph  # noqa: E402
from src.extraction.eap import candidate_edges, compute_eap  # noqa: E402
from src.tasks.base import PromptBatch, TaskPrompts  # noqa: E402

VOCAB = 40


def _model(parallel: bool, seed: int = 0) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_heads=4, d_head=4, d_mlp=32, d_vocab=VOCAB, n_ctx=16,
        act_fn="gelu", normalization_type="LN",
        positional_embedding_type="rotary" if parallel else "standard", rotary_dim=4,
        parallel_attn_mlp=parallel, use_attn_result=True, use_split_qkv_input=True,
        use_hook_mlp_in=True, dtype=torch.float64, seed=seed, device="cpu",
    )
    model = HookedTransformer(cfg)
    model.eval()
    return model


def _metric_sum(logits):
    return (logits[:, -1, 3] - logits[:, -1, 5]).sum()


def _prompts(n: int = 6, seq: int = 7, n_batches: int = 1, seed: int = 0) -> TaskPrompts:
    g = torch.Generator().manual_seed(seed)
    clean = torch.randint(0, VOCAB, (n, seq), generator=g)
    corrupt = clean.clone()
    corrupt[:, 2] = torch.randint(0, VOCAB, (n,), generator=g)
    corrupt[:, 4] = torch.randint(0, VOCAB, (n,), generator=g)
    size = n // n_batches
    batches = tuple(
        PromptBatch(clean=clean[i:i + size], corrupt=corrupt[i:i + size],
                    metric_sum=_metric_sum, units=tuple(f"u{j}" for j in range(i, i + size)))
        for i in range(0, n, size)
    )
    return TaskPrompts(task="synthetic-test", seed=seed, batches=batches)


class TestEdgeUniverse:
    def test_parallel_residual_has_no_same_layer_attention_to_mlp_edge(self):
        edges = set(candidate_edges(_model(parallel=True).cfg))
        phantom = {e for e in edges if e.split("->")[0].startswith("L0.H") and e.endswith("->L0.MLP")}
        assert not phantom, "a parallel block's attention output never reaches its own MLP input"

    def test_sequential_blocks_do_have_it(self):
        edges = set(candidate_edges(_model(parallel=False).cfg))
        assert "L0.H1->L0.MLP" in edges and "L1.H3->L1.MLP" in edges

    def test_candidate_counts_match_a_hand_count(self):
        # 2 layers, 4 heads. Parallel: EMB->27, each L0 head->14 (x4), L0.MLP->14,
        # each L1 head->LOGIT (x4), L1.MLP->LOGIT  = 27 + 56 + 14 + 4 + 1 = 102.
        # Sequential adds the same-layer head->MLP edges: 4 per layer x 2 = 8 -> 110.
        assert len(candidate_edges(_model(parallel=True).cfg)) == 102
        assert len(candidate_edges(_model(parallel=False).cfg)) == 110

    def test_compute_eap_scores_exactly_the_candidate_edges(self):
        model = _model(parallel=True)
        scores = compute_eap(model, _prompts())
        assert set(scores.edge_scores) == set(candidate_edges(model.cfg))


@pytest.mark.parametrize("parallel", [True, False])
def test_every_eap_edge_score_is_the_first_order_effect_of_an_exact_edge_patch(parallel):
    """|d metric / dh| when h * (a_u^clean - a_u^corrupt) is added at v's input, at h = 0.

    Adding at the pre-LayerNorm input hook patches exactly the one edge u->v and nothing
    else. A central finite difference in float64 recovers the derivative to ~1e-8, whatever
    the model's nonlinearity.
    """
    model = _model(parallel)
    prompts = _prompts()
    scores = compute_eap(model, prompts)
    batch = prompts.batches[0]
    n = prompts.n_prompts
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(batch.clean)
        _, corrupt_cache = model.run_with_cache(batch.corrupt)

    up_hook = {"EMB": ("blocks.0.hook_resid_pre", None)}
    down_hook = {"LOGIT": ("blocks.1.hook_resid_post", None)}
    for layer in range(2):
        up_hook[f"L{layer}.MLP"] = (f"blocks.{layer}.hook_mlp_out", None)
        down_hook[f"L{layer}.MLP"] = (f"blocks.{layer}.hook_mlp_in", None)
        for h in range(4):
            up_hook[f"L{layer}.H{h}"] = (f"blocks.{layer}.attn.hook_result", h)
            for x in "QKV":
                down_hook[f"L{layer}.H{h}.{x}"] = (f"blocks.{layer}.hook_{x.lower()}_input", h)

    cases = [
        "EMB->L1.H2.K", "L0.H1->L1.H3.V", "L0.MLP->L1.H0.Q", "L0.H2->L1.MLP",
        "L1.MLP->LOGIT", "L1.H0->LOGIT", "EMB->L0.H1.Q", "EMB->LOGIT",
    ]
    if not parallel:
        cases.append("L0.H2->L0.MLP")

    for edge in cases:
        src, dst = edge.split("->")
        u_name, u_head = up_hook[src]
        d_name, d_head = down_hook[dst]
        delta = clean_cache[u_name] - corrupt_cache[u_name]
        if u_head is not None:
            delta = delta[:, :, u_head, :]

        def metric_at(step, d_name=d_name, d_head=d_head, delta=delta):
            def patch(act, hook):
                act = act.clone()
                if d_head is None:
                    act = act + step * delta
                else:
                    act[:, :, d_head, :] = act[:, :, d_head, :] + step * delta
                return act

            with torch.no_grad():
                logits = model.run_with_hooks(batch.clean, fwd_hooks=[(d_name, patch)])
            return float(batch.metric_sum(logits)) / n

        h = 1e-4
        finite_difference = (metric_at(h) - metric_at(-h)) / (2 * h)
        assert scores.edge_scores[edge] == pytest.approx(abs(finite_difference), rel=1e-5, abs=1e-10), edge


class TestAggregation:
    def test_absolute_value_is_taken_once_after_all_batches(self):
        """|sum over pairs| is invariant to how prompts are batched; a per-batch |.| is not."""
        model = _model(parallel=True)
        whole = compute_eap(model, _prompts(n=8, n_batches=1))
        split = compute_eap(model, _prompts(n=8, n_batches=2))
        for edge, value in whole.edge_scores.items():
            assert split.edge_scores[edge] == pytest.approx(value, rel=1e-9, abs=1e-12), edge
        for node, value in whole.node_scores.items():
            assert split.node_scores[node] == pytest.approx(value, rel=1e-9, abs=1e-12), node

    def test_the_invariance_test_has_teeth(self):
        """At least one edge's halves have opposite signs, so a per-batch |.| WOULD differ."""
        model = _model(parallel=True)
        prompts = _prompts(n=8, n_batches=2)
        whole = compute_eap(model, prompts)
        halves = [
            compute_eap(model, TaskPrompts(task="t", seed=0, batches=(b,)))
            for b in prompts.batches
        ]
        # half scores are normalised by 4, the whole by 8: per-batch |.| would give their mean
        per_batch_abs = {e: 0.5 * (halves[0].edge_scores[e] + halves[1].edge_scores[e]) for e in whole.edge_scores}
        assert any(per_batch_abs[e] > whole.edge_scores[e] + 1e-9 for e in whole.edge_scores)

    def test_node_score_is_attribution_patching_at_the_output(self):
        """A node's score equals its first-order effect when its OUTPUT delta is added."""
        model = _model(parallel=True)
        prompts = _prompts()
        scores = compute_eap(model, prompts)
        batch = prompts.batches[0]
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(batch.clean)
            _, corrupt_cache = model.run_with_cache(batch.corrupt)
        name = "blocks.0.hook_mlp_out"
        delta = clean_cache[name] - corrupt_cache[name]

        def metric_at(step):
            with torch.no_grad():
                logits = model.run_with_hooks(batch.clean, fwd_hooks=[(name, lambda a, hook: a + step * delta)])
            return float(batch.metric_sum(logits)) / prompts.n_prompts

        fd = (metric_at(1e-4) - metric_at(-1e-4)) / 2e-4
        assert scores.node_scores["L0.MLP"] == pytest.approx(abs(fd), rel=1e-5, abs=1e-10)

    def test_parameters_are_left_as_they_were(self):
        model = _model(parallel=True)
        before = [p.requires_grad for p in model.parameters()]
        compute_eap(model, _prompts())
        assert [p.requires_grad for p in model.parameters()] == before
        assert all(p.grad is None for p in model.parameters())

    def test_deterministic(self):
        model = _model(parallel=True)
        a, b = compute_eap(model, _prompts()), compute_eap(model, _prompts())
        assert a.edge_scores == b.edge_scores and a.node_scores == b.node_scores


class TestThresholdPort:
    def test_find_threshold_matches_circuit_tracer_semantics(self):
        s = torch.tensor([2.0, 5.0, 3.0], dtype=torch.float64)   # sorted: 5, 3, 2 -> cumulative 0.5, 0.8, 1.0
        assert float(find_threshold(s, 0.5)) == 5.0    # left searchsorted lands ON 0.5
        assert float(find_threshold(s, 0.6)) == 3.0
        assert float(find_threshold(s, 0.8)) == 3.0
        assert float(find_threshold(s, 0.81)) == 2.0
        assert float(find_threshold(s, 1.0)) == 2.0    # clamped to the last index

    def test_owner(self):
        assert owner("L3.H0.Q") == "L3.H0" and owner("L3.MLP") == "L3.MLP" and owner("LOGIT") == "LOGIT"


class TestPruning:
    NODES = {"EMB": 1.0, "L0.H0": 5.0, "L0.H1": 0.1, "L0.MLP": 3.0, "L1.H0": 2.0}

    def test_pruned_nodes_take_their_edges_with_them(self):
        edges = {"EMB->L0.H0.Q": 4.0, "EMB->L0.H1.V": 4.0, "L0.H0->LOGIT": 3.0, "L0.H1->LOGIT": 3.0,
                 "EMB->L0.MLP": 2.0, "L0.MLP->LOGIT": 2.0}
        nodes, kept = prune_dense_graph(edges, self.NODES, node_threshold=0.8, edge_threshold=1.0)
        assert "L0.H1" not in nodes
        assert not any("L0.H1" in e for e in kept)

    def test_embedding_and_logit_are_always_kept(self):
        nodes, _ = prune_dense_graph({"EMB->L0.H0.Q": 1.0, "L0.H0->LOGIT": 1.0}, {"EMB": 0.0, "L0.H0": 1.0}, 0.5, 1.0)
        assert {"EMB", "LOGIT"} <= nodes

    def test_components_left_dangling_are_removed_iteratively(self):
        # L1.H0 keeps an incoming edge but its only outgoing edge is thresholded away, which
        # then strands L0.MLP (whose only outgoing edge fed L1.H0).
        nodes_scores = {"EMB": 1.0, "L0.MLP": 1.0, "L1.H0": 1.0, "L0.H0": 1.0}
        edges = {"EMB->L0.MLP": 50.0, "L0.MLP->L1.H0.K": 40.0, "L1.H0->LOGIT": 0.001,
                 "EMB->L0.H0.Q": 30.0, "L0.H0->LOGIT": 30.0}
        nodes, kept = prune_dense_graph(edges, nodes_scores, node_threshold=1.0, edge_threshold=0.9)
        assert "L1.H0" not in nodes and "L0.MLP" not in nodes
        assert kept == {"EMB->L0.H0.Q", "L0.H0->LOGIT"}

    def test_all_zero_scores_are_refused(self):
        with pytest.raises(ValueError, match="sum to zero"):
            prune_dense_graph({"EMB->LOGIT": 0.0}, {"EMB": 0.0}, 0.8, 0.98)

    def test_invalid_threshold_is_refused(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            prune_dense_graph({"EMB->LOGIT": 1.0}, {"EMB": 1.0}, 1.2, 0.98)

    def test_the_anti_diagonal_views_disagree_on_real_scores(self):
        """Non-nested views must not all return the same edge set on a real score profile."""
        from src.science.threshold_grid import generate_anti_diagonal_grid

        model = _model(parallel=True)
        scores = compute_eap(model, _prompts())
        views = {
            prune_dense_graph(scores.edge_scores, scores.node_scores, c["node_threshold"], c["edge_threshold"])[1]
            for c in generate_anti_diagonal_grid(4)
        }
        assert len(views) > 1


class TestDenseNodeExtractorCaching:
    def test_attribution_runs_once_per_seed_across_all_configs(self):
        from src.extraction.dense_node_variant import DenseNodeExtractor
        from src.science.threshold_grid import generate_anti_diagonal_grid

        class Offline(DenseNodeExtractor):
            def _build_prompts(self, model, task, seed):
                return _prompts(seed=seed)

        model = _model(parallel=True)
        extractor = Offline()
        grid = generate_anti_diagonal_grid(4)
        task = {"name": "synthetic-test", "n_prompts": 6, "prepend_bos": False}
        for seed in (0, 1, 2):
            for config in grid:
                graph = extractor.extract(model, task, config, seed)
                assert graph.metadata["pipeline"] == "dense-node"
                assert all(e.config_id == config["id"] and e.seed == seed for e in graph.edges)
        assert extractor.n_attribution_runs == 3, "EAP must run once per (model, task, seed), not once per config"

    def test_a_different_model_object_is_not_served_from_cache(self):
        from src.extraction.dense_node_variant import DenseNodeExtractor

        class Offline(DenseNodeExtractor):
            def _build_prompts(self, model, task, seed):
                return _prompts(seed=seed)

        extractor = Offline()
        task = {"name": "synthetic-test", "n_prompts": 6, "prepend_bos": False}
        config = {"id": "c0", "node_threshold": 0.8, "edge_threshold": 0.98}
        extractor.extract(_model(parallel=True, seed=0), task, config, 0)
        extractor.extract(_model(parallel=True, seed=1), task, config, 0)
        assert extractor.n_attribution_runs == 2, "a perturbed or compressed model must be re-attributed"
