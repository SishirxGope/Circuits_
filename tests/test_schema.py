# [AI-GEN] agent=OpenCode date=2026-08-07 task=Unit tests: schema layer (ARCHITECTURE.md §2)
# reviewed-by: PENDING

import pytest

from src.common import schema


def test_edge_id_format():
    assert schema.edge_id("L11.H3", "L7.MLP") == "L11.H3->L7.MLP"
    assert schema.edge_id("L11.H3", "L7.MLP") != schema.edge_id("L7.MLP", "L11.H3")


def test_edges_record_validation():
    good = {"src_component": "L11.H3", "dst_component": "L7.MLP", "config_id": "c0", "seed": 0, "included": True}
    schema.validate_edges_record(good)
    with pytest.raises(ValueError):
        schema.validate_edges_record({"src_component": "L11.H3", "dst_component": "L7.MLP", "seed": 0})
    with pytest.raises(ValueError):
        schema.validate_edges_record({"src_component": "L11.H3", "dst_component": "L7.MLP", "config_id": "c0", "seed": "0"})


def test_freq_record_validation():
    good = {"edge_id": "L11.H3->L7.MLP", "s_e": 0.5, "band": "contingent"}
    schema.validate_freq_record(good)
    with pytest.raises(ValueError):
        schema.validate_freq_record({"edge_id": "L11.H3->L7.MLP"})  # missing s_e
    with pytest.raises(ValueError):
        schema.validate_freq_record({"edge_id": "L11.H3->L7.MLP", "s_e": 1.5})  # out of range
    with pytest.raises(ValueError):
        schema.validate_freq_record({"edge_id": "L11.H3->L7.MLP", "s_e": 0.5, "band": "bogus"})


def test_run_tags_stage_dependent():
    base = {
        "stage": "stageA", "model": "pythia160m", "task": "ioi",
        "compression_family": "dense", "comparison_level": "both", "pipeline": "mock",
        "B": 3, "S": 2, "seed": 0,
    }
    schema.validate_run_tags(base, stage="stageA")  # passes
    bad = dict(base)
    del bad["S"]
    with pytest.raises(ValueError, match="S"):
        schema.validate_run_tags(bad, stage="stageA")

    stage_c = dict(base)
    stage_c["stage"] = "stageC"
    stage_c["R"] = 20
    with pytest.raises(ValueError, match="null_frozen_hash"):
        schema.validate_run_tags(stage_c, stage="stageC")
    stage_c["null_frozen_hash"] = "abc"
    schema.validate_run_tags(stage_c, stage="stageC")  # passes

    stage_b = dict(base)
    stage_b["stage"] = "stageB"
    stage_b["R"] = 0
    with pytest.raises(ValueError, match="R"):
        schema.validate_run_tags(stage_b, stage="stageB")


def test_parquet_roundtrip_edges(tmp_path):
    pa = pytest.importorskip("pyarrow")
    records = [
        {"src_component": "L11.H3", "dst_component": "L7.MLP", "config_id": "c0", "seed": 0, "included": True},
        {"src_component": "L0.H0", "dst_component": "L2.MLP", "config_id": "c1", "seed": 1, "included": True},
    ]
    path = tmp_path / "edges.parquet"
    schema.write_edges_parquet(str(path), records)
    table = pa.parquet.read_table(str(path))
    assert table.column_names == list(schema.EDGES_COLUMNS)
    back = schema.read_edges_parquet(str(path))
    assert len(back) == 2
    assert back[0]["src_component"] == "L11.H3"


def test_parquet_roundtrip_freq(tmp_path):
    pa = pytest.importorskip("pyarrow")
    rows = [
        {"edge_id": "L11.H3->L7.MLP", "s_e": 1.0, "band": "core"},
        {"edge_id": "L0.H0->L2.MLP", "s_e": 1 / 3, "band": None},
    ]
    path = tmp_path / "freq.parquet"
    schema.write_freq_parquet(str(path), rows)
    table = pa.parquet.read_table(str(path))
    assert table.column_names == list(schema.FREQ_COLUMNS)
    back = schema.read_freq_parquet(str(path))
    assert back[0]["band"] == "core"
    assert back[1]["band"] is None


def test_csi_placeholder_columns(tmp_path):
    path = tmp_path / "csi_placeholder.csv"
    schema.write_csi_placeholder(str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert "PLACEHOLDER" in lines[0]  # AI_RULES.md 2.3 visual marking
    assert lines[1] == ",".join(schema.CSI_TABLE_COLUMNS)


def test_frozen_meta_validation():
    schema.validate_frozen_meta({"config_hash": "a", "seeds": [0], "R": 20, "null_frozen_hash": "b"})
    with pytest.raises(ValueError):
        schema.validate_frozen_meta({"config_hash": "a", "seeds": [0]})


def test_graph_and_freqvector_helpers():
    e1 = schema.Edge("L11.H3", "L7.MLP", "c0", 0)
    g = schema.Graph(edges=(e1,), nodes=("L11.H3", "L7.MLP"), metadata={"mock": True})
    assert g.included_edge_ids() == ("L11.H3->L7.MLP",)
    fv = schema.FreqVector(frequencies={"b": 0.5, "a": 1.0})
    assert fv.items_sorted() == [("a", 1.0), ("b", 0.5)]
