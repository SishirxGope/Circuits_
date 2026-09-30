# [AI-GEN] agent=Claude date=2026-09-30 task=B6 - the exit gate's reference edges equal ACDC's own ground truth
# reviewed-by: PENDING

"""The IOI reference edges (src/tasks/ioi_reference.py) and the file that carries them.

The claim is that the port computes exactly what ACDC's ``get_ioi_true_edges`` computes.
So ACDC's code runs here verbatim (tests/reference_acdc_ioi.py, pinned commit, licence
shipped) on a stand-in for GPT-2 small's config, and the two edge sets must be equal.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import reference_acdc_ioi as acdc

from src.tasks.ioi_reference import IOI_CIRCUIT, SOURCE, _gpt2_small_edges, ioi_reference_edges

_SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "shared" / "make_ioi_reference.py"
_spec = importlib.util.spec_from_file_location("_make_ioi_reference_under_test", _SCRIPT)
make_ioi_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_ioi_reference)

GPT2_SMALL = SimpleNamespace(cfg=SimpleNamespace(n_layers=12, n_heads=12, attn_only=False))


def _our_id(name: str, index: tuple) -> str:
    if name == "blocks.0.hook_resid_pre":
        return "EMB"
    if name == "blocks.11.hook_resid_post":
        return "LOGIT"
    layer = int(name.split(".")[1])
    if name.endswith("attn.hook_result"):
        return f"L{layer}.H{index[2]}"
    if name.endswith(("hook_mlp_out", "hook_mlp_in")):
        return f"L{layer}.MLP"
    for letter in "qkv":
        if name.endswith(f"hook_{letter}_input"):
            return f"L{layer}.H{index[2]}.{letter.upper()}"
    raise ValueError(f"not a component input or output: {name}")


def _component_edges(keys) -> set[str]:
    """ACDC edges into component INPUTS, in our ids; its edges inside a component are dropped."""
    return {
        f"{_our_id(parent, p_idx)}->{_our_id(child, c_idx)}"
        for child, c_idx, parent, p_idx in keys
        if child.endswith(("_input", "hook_mlp_in", "hook_resid_post"))
    }


class TestAgainstAcdc:
    def test_the_edge_universe_is_acdcs_graph(self):
        corr = acdc.TLACDCCorrespondence.setup_from_model(GPT2_SMALL)
        keys = [(c, i.hashable_tuple, p, j.hashable_tuple) for c, i, p, j in corr.all_edges()]
        assert _component_edges(keys) == _gpt2_small_edges()

    def test_identical_to_get_ioi_true_edges(self):
        truth = _component_edges(acdc.get_ioi_true_edges(GPT2_SMALL).keys())
        assert ioi_reference_edges() == truth

    def test_the_comparison_is_not_vacuous(self):
        truth = ioi_reference_edges()
        assert 0 < len(truth) < len(_gpt2_small_edges())
        assert len(truth) == 963  # ACDC's 1131 present edges minus its 168 inside components

    def test_the_circuit_is_upstreams(self):
        assert IOI_CIRCUIT == acdc.IOI_CIRCUIT


class TestWhatTheEdgesSay:
    """Spot checks against the published circuit, readable without ACDC."""

    def test_name_movers_write_to_the_logits(self):
        edges = ioi_reference_edges()
        for layer, head in IOI_CIRCUIT["name mover"] + IOI_CIRCUIT["negative"]:
            assert f"L{layer}.H{head}->LOGIT" in edges

    def test_s_inhibition_reaches_name_movers_through_the_query_only(self):
        edges = ioi_reference_edges()
        assert "L7.H3->L9.H9.Q" in edges
        assert "L7.H3->L9.H9.K" not in edges and "L7.H3->L9.H9.V" not in edges

    def test_no_edge_touches_a_head_outside_the_circuit(self):
        circuit = {f"L{l}.H{h}" for heads in IOI_CIRCUIT.values() for l, h in heads}
        for edge in ioi_reference_edges():
            for node in edge.split("->"):
                if ".H" in node:
                    assert ".".join(node.split(".")[:2]) in circuit, edge

    def test_s_inhibition_heads_do_not_write_to_the_logits(self):
        edges = ioi_reference_edges()
        for layer, head in IOI_CIRCUIT["s2 inhibition"]:
            assert f"L{layer}.H{head}->LOGIT" not in edges


class TestTheReferenceFile:
    def _write(self, tmp_path, *args):
        out = tmp_path / "ref.json"
        return make_ioi_reference.main(list(args), out=out), out

    def test_it_writes_a_file_the_gate_accepts(self, tmp_path):
        from test_regression_ioi_gpt2_small import load_reference

        code, out = self._write(tmp_path, *self.ARGS)
        assert code == 0
        edges, crit = load_reference(out)
        assert edges == ioi_reference_edges()
        assert crit == {"min_precision": 0.3, "max_p_value": 0.001, "seed": 5}
        assert json.loads(out.read_text(encoding="utf-8"))["source"] == SOURCE

    ARGS = ("--min-precision", "0.3", "--max-p-value", "0.001", "--seed", "5", "--decided-on", "2026-10-01")

    @pytest.mark.parametrize("flag", ["--min-precision", "--max-p-value", "--seed"])
    def test_there_are_no_defaults(self, tmp_path, flag):
        args = list(self.ARGS)
        i = args.index(flag)
        del args[i:i + 2]
        with pytest.raises(SystemExit):
            self._write(tmp_path, *args)

    @pytest.mark.parametrize("flag,value", [
        ("--min-precision", "0"), ("--min-precision", "1.5"),
        ("--max-p-value", "0"), ("--max-p-value", "1"), ("--seed", "-1"),
    ])
    def test_the_criterion_is_range_checked(self, tmp_path, flag, value):
        args = list(self.ARGS)
        args[args.index(flag) + 1] = value
        code, out = self._write(tmp_path, *args)
        assert code == 1 and not out.exists()

    def test_the_date_must_be_iso(self, tmp_path):
        args = [*self.ARGS[:-1], "1 Oct"]
        code, out = self._write(tmp_path, *args)
        assert code == 1 and not out.exists()

    def test_an_existing_file_is_never_overwritten(self, tmp_path):
        out = tmp_path / "ref.json"
        out.write_text("{}", encoding="utf-8")
        assert make_ioi_reference.main(list(self.ARGS), out=out) == 1
        assert out.read_text(encoding="utf-8") == "{}"
