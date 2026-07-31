"""Graph IR: serialization round-trip and validator coverage."""

import json
import os

import numpy as np
import pytest

from kea_frontend.ir import KGRAPH_VERSION, KGraph, KGraphError, Node, QuantParams, Tensor


def _tiny_graph() -> KGraph:
    g = KGraph(name="tiny")
    g.add_tensor(Tensor("x", "int8", [1, 4, 4, 2], "NHWC",
                        QuantParams.per_tensor(0.05, -7), "input"))
    g.inputs.append("x")
    w = np.arange(-8, 8, dtype=np.int8).reshape(2, 2, 2, 2)   # OHWI
    g.add_const("w", w, "OHWI", QuantParams.per_channel([0.01, 0.02], 0))
    g.add_const("b", np.array([3, -4], dtype=np.int32), "C",
                QuantParams.per_channel([0.0005, 0.001], 0))
    g.add_tensor(Tensor("acc", "int32", [1, 3, 3, 2], "NHWC",
                        QuantParams.per_channel([0.0005, 0.001], 3)))
    g.add_node(Node("conv2d", "c0", ["x", "w", "b"], ["acc"], {
        "pad": [0, 0, 0, 0], "stride": [1, 1], "dilation": [1, 1],
        "input_zp": -7, "weight_zp": 0,
    }))
    g.add_tensor(Tensor("y", "int8", [1, 3, 3, 2], "NHWC",
                        QuantParams.per_tensor(0.02, 5)))
    g.add_node(Node("rescale", "r0", ["acc"], ["y"], {
        "input_zp": 0, "output_zp": 5,
        "multiplier": [1073741824, 1610612736], "shift": [36, 35],
        "per_channel": True, "channel_axis": 3, "out_dtype": "int8",
        "double_round": True,
    }))
    g.outputs.append("y")
    g.validate()
    return g


def test_roundtrip_preserves_everything(tmp_path):
    g = _tiny_graph()
    p = str(tmp_path / "t.kgraph.json")
    npz = g.save(p)
    assert os.path.exists(p) and os.path.exists(npz)

    g2 = KGraph.load(p)
    assert g2.version == KGRAPH_VERSION
    assert g2.name == g.name
    assert g2.inputs == g.inputs and g2.outputs == g.outputs
    assert len(g2.nodes) == len(g.nodes)
    assert set(g2.tensors) == set(g.tensors)
    for name, t in g.tensors.items():
        u = g2.tensors[name]
        assert (u.dtype, u.shape, u.layout, u.kind) == (t.dtype, t.shape, t.layout, t.kind)
        assert u.quant.kind == t.quant.kind and u.quant.axis == t.quant.axis
        assert np.allclose(u.quant.scale, t.quant.scale)
        assert u.quant.zero_point == t.quant.zero_point
    for k, v in g.consts.items():
        assert np.array_equal(g2.consts[k], v)
        assert g2.consts[k].dtype == v.dtype
    for a, b in zip(g.nodes, g2.nodes):
        assert (a.op, a.inputs, a.outputs) == (b.op, b.inputs, b.outputs)
        assert a.attrs == b.attrs


def test_roundtrip_is_stable(tmp_path):
    """save -> load -> save must be byte-identical."""
    g = _tiny_graph()
    p1 = str(tmp_path / "a.kgraph.json")
    g.save(p1)
    p2 = str(tmp_path / "b.kgraph.json")
    g2 = KGraph.load(p1)
    g2.save(p2)
    a = json.load(open(p1))
    b = json.load(open(p2))
    a.pop("weights_file"), b.pop("weights_file")
    assert a == b


def test_json_holds_no_bulk_data(tmp_path):
    """Weights live in the npz; the JSON must stay small and readable."""
    g = _tiny_graph()
    p = str(tmp_path / "t.kgraph.json")
    g.save(p)
    doc = json.load(open(p))
    assert doc["weights_file"] == "t.npz"
    for t in doc["tensors"]:
        assert "data" not in t and "values" not in t
    assert os.path.getsize(p) < 20000


def test_load_rejects_wrong_version(tmp_path):
    g = _tiny_graph()
    p = str(tmp_path / "t.kgraph.json")
    g.save(p)
    doc = json.load(open(p))
    doc["kgraph_version"] = "0.1"
    json.dump(doc, open(p, "w"))
    with pytest.raises(KGraphError, match="unsupported kgraph_version"):
        KGraph.load(p)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_rejects_non_topological_order():
    g = _tiny_graph()
    g.nodes = [g.nodes[1], g.nodes[0]]
    with pytest.raises(KGraphError, match="topological"):
        g.validate()


def test_rejects_asymmetric_per_channel_quant():
    g = _tiny_graph()
    g.tensors["w"].quant.zero_point = [1, 0]
    with pytest.raises(KGraphError, match="symmetric"):
        g.validate()


def test_rejects_per_channel_scale_length_mismatch():
    g = _tiny_graph()
    g.tensors["w"].quant.scale = [0.01]
    with pytest.raises(KGraphError, match="per_channel expects"):
        g.validate()


def test_rejects_const_shape_mismatch():
    g = _tiny_graph()
    g.consts["b"] = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(KGraphError, match="npz shape"):
        g.validate()


def test_rejects_shift_outside_tosa_require():
    g = _tiny_graph()
    g.nodes[1].attrs["shift"] = [63, 35]
    with pytest.raises(KGraphError, match="REQUIRE"):
        g.validate()


def test_rejects_negative_multiplier():
    g = _tiny_graph()
    g.nodes[1].attrs["multiplier"] = [-1, 1610612736]
    with pytest.raises(KGraphError, match="REQUIRE"):
        g.validate()


def test_rejects_per_channel_rescale_on_non_last_axis():
    """tosa.rescale per_channel=true is defined against the LAST dimension."""
    g = _tiny_graph()
    g.nodes[1].attrs["channel_axis"] = 1
    with pytest.raises(KGraphError, match="last"):
        g.validate()


def test_rejects_duplicate_definition():
    g = _tiny_graph()
    g.add_node(Node("rescale", "r1", ["acc"], ["y"], dict(g.nodes[1].attrs)))
    with pytest.raises(KGraphError, match="produced twice"):
        g.validate()


def test_rejects_unknown_op():
    g = _tiny_graph()
    g.nodes[0].op = "conv3d"
    with pytest.raises(KGraphError, match="unknown op"):
        g.validate()


def test_rejects_padded_avg_pool():
    g = KGraph(name="p")
    g.add_tensor(Tensor("x", "int8", [1, 4, 4, 2], "NHWC",
                        QuantParams.per_tensor(0.1, 0), "input"))
    g.inputs.append("x")
    g.add_tensor(Tensor("y", "int8", [1, 3, 3, 2], "NHWC", QuantParams.per_tensor(0.1, 0)))
    g.add_node(Node("avg_pool2d", "p0", ["x"], ["y"], {
        "kernel": [2, 2], "stride": [1, 1], "pad": [1, 0, 0, 0],
        "input_zp": 0, "output_zp": 0, "multiplier": 1073741824, "shift": 33,
        "out_dtype": "int8",
    }))
    g.outputs.append("y")
    with pytest.raises(KGraphError, match="padding is not supported"):
        g.validate()


def test_rejects_non_rank3_matmul():
    g = KGraph(name="m")
    for n, sh in (("a", [4, 8]), ("b", [8, 4])):
        g.add_tensor(Tensor(n, "int8", sh, "*", QuantParams.per_tensor(0.1, 0), "input"))
        g.inputs.append(n)
    g.add_tensor(Tensor("o", "int32", [4, 4], "*", QuantParams.per_tensor(0.01, 0)))
    g.add_node(Node("matmul", "m0", ["a", "b"], ["o"], {"a_zp": 0, "b_zp": 0}))
    g.outputs.append("o")
    with pytest.raises(KGraphError, match="rank 3"):
        g.validate()


def test_exported_models_validate():
    """Any artifact checked into models/ must load and validate."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    md = os.path.join(repo, "models")
    found = [f for f in sorted(os.listdir(md)) if f.endswith(".kgraph.json")] \
        if os.path.isdir(md) else []
    if not found:
        pytest.skip("no exported models present; run frontend/export_*.py")
    for f in found:
        g = KGraph.load(os.path.join(md, f))   # load() validates
        assert g.nodes and g.inputs and g.outputs
