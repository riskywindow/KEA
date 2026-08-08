"""`.kgraph.json` -> TOSA MLIR.

**Every test that emits anything runs the emitted text through `mlir-opt`.**
That is the only gate that means something: the emitter's job is to produce
TOSA that MLIR 20.1.6 accepts, and 20.1.6's TOSA differs from every published
description of the dialect (docs/TOSA_NOTES.md section 0).  A test that only
compared strings would pass while emitting unparseable IR.

Where the compiler is built, the emitted TOSA is also pushed through the real
pipeline (`--tosa-to-kea --kea-fuse --kea-tile --kea-schedule --kea-alloc`,
then `kea-translate`), because "mlir-opt parses it" and "the KEA backend can
lower it" are different claims.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from kea_frontend.ir import KGraph, KGraphError, Node, QuantParams, Tensor
from kea_frontend.tosa_emit import TosaEmitError, emit_module

from conftest import MODELS, REPO

MLIR_OPT = os.environ.get("MLIR_OPT", "/usr/local/opt/llvm/bin/mlir-opt")
KEA_OPT = os.path.join(REPO, "build", "compiler", "bin", "kea-opt")
KEA_TRANSLATE = os.path.join(REPO, "build", "compiler", "bin", "kea-translate")

needs_mlir = pytest.mark.skipif(
    not os.path.exists(MLIR_OPT), reason="mlir-opt not installed")
needs_compiler = pytest.mark.skipif(
    not (os.path.exists(KEA_OPT) and os.path.exists(KEA_TRANSLATE)),
    reason="run scripts/build_compiler.sh")


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def mlir_roundtrip(text, tmp_path, name="m.mlir"):
    """Parse + print + re-parse.  Returns mlir-opt's canonical output."""
    src = tmp_path / name
    src.write_text(text)
    r = subprocess.run([MLIR_OPT, str(src)], capture_output=True, text=True)
    assert r.returncode == 0, "mlir-opt rejected the emitter's output:\n%s\n%s" % (
        r.stderr, text[:4000])
    out = tmp_path / ("rt." + name)
    out.write_text(r.stdout)
    r2 = subprocess.run([MLIR_OPT, str(out)], capture_output=True, text=True)
    assert r2.returncode == 0, "output does not re-parse:\n%s" % r2.stderr
    return r.stdout


def kea_pipeline(text, tmp_path, function, name="m.mlir"):
    """The real backend pipeline, all the way to `.kasm`."""
    src = tmp_path / name
    src.write_text(text)
    l2 = tmp_path / "l2.mlir"
    r = subprocess.run(
        [KEA_OPT, str(src), "--tosa-to-kea", "--kea-fuse", "--kea-tile",
         "--kea-schedule", "--kea-alloc", "-o", str(l2)],
        capture_output=True, text=True)
    assert r.returncode == 0, "kea-opt rejected the emitter's output:\n%s" % r.stderr
    r = subprocess.run(
        [KEA_TRANSLATE, str(l2), "-o", str(tmp_path / "out"),
         "--function=" + function], capture_output=True, text=True)
    assert r.returncode == 0, "kea-translate failed:\n%s" % r.stderr
    return (tmp_path / "out.kasm").read_text()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _conv_graph(depthwise=False, with_clamp=True):
    g = KGraph(name="convnet")
    g.add_tensor(Tensor("x", "int8", [1, 8, 8, 4], "NHWC",
                        QuantParams.per_tensor(0.05, -7), "input"))
    g.inputs.append("x")
    if depthwise:
        w = np.arange(-18, 18, dtype=np.int8).reshape(3, 3, 4, 1)  # HWCM
        g.add_const("w", w, "HWCM", QuantParams.per_channel([0.01] * 4, 2))
        op, oc = "depthwise_conv2d", 4
    else:
        w = np.arange(-72, 72, dtype=np.int8).reshape(4, 3, 3, 4)  # OHWI
        g.add_const("w", w, "OHWI", QuantParams.per_channel([0.01] * 4, 0))
        op, oc = "conv2d", 4
    g.add_const("b", np.array([3, -4, 5, -6], dtype=np.int32), "C",
                QuantParams.per_channel([5e-4] * oc, 0))
    g.add_tensor(Tensor("acc", "int32", [1, 8, 8, oc], "NHWC",
                        QuantParams.per_channel([5e-4] * oc, 3)))
    g.add_node(Node(op, "c0", ["x", "w", "b"], ["acc"], {
        "pad": [1, 1, 1, 1], "stride": [1, 1], "dilation": [1, 1],
        "input_zp": -7, "weight_zp": 0}))
    g.add_tensor(Tensor("q", "int8", [1, 8, 8, oc], "NHWC",
                        QuantParams.per_tensor(0.02, -128)))
    g.add_node(Node("rescale", "r0", ["acc"], ["q"], {
        "input_zp": 0, "output_zp": -128,
        "multiplier": [1073741824, 1610612736, 1288490189, 1932735283],
        "shift": [36, 35, 37, 38],
        "per_channel": True, "channel_axis": 3, "out_dtype": "int8",
        "double_round": True}))
    last = "q"
    if with_clamp:
        g.add_tensor(Tensor("y", "int8", [1, 8, 8, oc], "NHWC",
                            QuantParams.per_tensor(0.02, -128)))
        g.add_node(Node("clamp", "k0", ["q"], ["y"],
                        {"min_val": -128, "max_val": 127}))
        last = "y"
    g.outputs.append(last)
    g.validate()
    return g


def _residual_graph():
    """The rescale/rescale/add/rescale sandwich `-kea-fuse` folds."""
    g = KGraph(name="resnet")
    for n in ("a", "b"):
        g.add_tensor(Tensor(n, "int8", [1, 4, 4, 8], "NHWC",
                            QuantParams.per_tensor(0.05, -5 if n == "a" else 3),
                            "input"))
        g.inputs.append(n)
    # these are MobileNetV2's own residual parameters; the KEA VADD cannot
    # express every (a, b, out) triple (a left shift in the output rescale has
    # no KEA_VADD form), so the fixture uses a triple a real PTQ export emits.
    for n, izp, m, s in (("a32", -5, 1073741824, 22), ("b32", 3, 1145562617, 22)):
        g.add_tensor(Tensor(n, "int32", [1, 4, 4, 8], "NHWC",
                            QuantParams.per_tensor(2e-4, 0)))
        g.add_node(Node("rescale", "r_" + n, [n[0]], [n], {
            "input_zp": izp, "output_zp": 0, "multiplier": [m], "shift": [s],
            "per_channel": False, "channel_axis": 0, "out_dtype": "int32",
            "double_round": True}))
    g.add_tensor(Tensor("s", "int32", [1, 4, 4, 8], "NHWC",
                        QuantParams.per_tensor(2e-4, 0)))
    g.add_node(Node("add", "add0", ["a32", "b32"], ["s"], {}))
    g.add_tensor(Tensor("y", "int8", [1, 4, 4, 8], "NHWC",
                        QuantParams.per_tensor(0.06, -7)))
    g.add_node(Node("rescale", "r_out", ["s"], ["y"], {
        "input_zp": 0, "output_zp": -7, "multiplier": [1440945660],
        "shift": [39], "per_channel": False, "channel_axis": 0,
        "out_dtype": "int8", "double_round": True}))
    g.outputs.append("y")
    g.validate()
    return g


def _pool_fc_graph():
    g = KGraph(name="headnet")
    g.add_tensor(Tensor("x", "int8", [1, 4, 4, 8], "NHWC",
                        QuantParams.per_tensor(0.05, -128), "input"))
    g.inputs.append("x")
    g.add_tensor(Tensor("p", "int8", [1, 2, 2, 8], "NHWC",
                        QuantParams.per_tensor(0.05, -128)))
    # multiplier/shift == exactly 1/(2*2), so this maps 1:1 onto avg_pool2d
    g.add_node(Node("avg_pool2d", "p0", ["x"], ["p"], {
        "kernel": [2, 2], "stride": [2, 2], "pad": [0, 0, 0, 0],
        "input_zp": -128, "output_zp": -128,
        "multiplier": 1 << 30, "shift": 32, "out_dtype": "int8",
        "double_round": True}))
    g.add_tensor(Tensor("g", "int8", [1, 1, 1, 8], "NHWC",
                        QuantParams.per_tensor(0.05, -128)))
    g.add_node(Node("global_avg_pool", "g0", ["p"], ["g"], {
        "input_zp": -128, "output_zp": -128,
        "multiplier": 1 << 30, "shift": 32, "out_dtype": "int8",
        "double_round": True}))
    g.add_tensor(Tensor("f", "int8", [1, 8], "*", QuantParams.per_tensor(0.05, -128)))
    g.add_node(Node("reshape", "s0", ["g"], ["f"], {"new_shape": [1, 8]}))
    g.add_const("w", np.arange(-24, 24, dtype=np.int8).reshape(6, 8), "OI",
                QuantParams.per_channel([0.01] * 6, 0))
    g.add_const("fb", np.arange(6, dtype=np.int32), "C",
                QuantParams.per_channel([5e-4] * 6, 0))
    g.add_tensor(Tensor("acc", "int32", [1, 6], "*",
                        QuantParams.per_channel([5e-4] * 6, 1)))
    g.add_node(Node("fully_connected", "fc0", ["f", "w", "fb"], ["acc"],
                    {"input_zp": -128, "weight_zp": 0}))
    g.add_tensor(Tensor("y", "int8", [1, 6], "*", QuantParams.per_tensor(0.1, -3)))
    g.add_node(Node("rescale", "r0", ["acc"], ["y"], {
        "input_zp": 0, "output_zp": -3,
        "multiplier": [1073741824] * 6, "shift": [34] * 6,
        "per_channel": True, "channel_axis": 1, "out_dtype": "int8",
        "double_round": True}))
    g.outputs.append("y")
    g.validate()
    return g


def _matmul_graph():
    g = KGraph(name="mmnet")
    g.add_tensor(Tensor("a", "int8", [1, 4, 8], "*",
                        QuantParams.per_tensor(0.05, -2), "input"))
    g.inputs.append("a")
    g.add_const("b", np.arange(-64, 64, dtype=np.int8).reshape(1, 8, 16), "*",
                QuantParams.per_tensor(0.01, 0))
    g.add_tensor(Tensor("acc", "int32", [1, 4, 16], "*",
                        QuantParams.per_tensor(5e-4, 0)))
    g.add_node(Node("matmul", "m0", ["a", "b"], ["acc"],
                    {"a_zp": -2, "b_zp": 0}))
    g.add_tensor(Tensor("y", "int8", [1, 4, 16], "*",
                        QuantParams.per_tensor(0.02, -3)))
    g.add_node(Node("rescale", "r0", ["acc"], ["y"], {
        "input_zp": 0, "output_zp": -3, "multiplier": [1073741824] * 16,
        "shift": [34] * 16, "per_channel": True, "channel_axis": 2,
        "out_dtype": "int8", "double_round": True}))
    g.outputs.append("y")
    g.validate()
    return g


def _shape_graph():
    """transpose / concat / pad / table -- the ops MobileNetV2 never touches."""
    g = KGraph(name="shapes")
    q = QuantParams.per_tensor(0.05, -5)
    g.add_tensor(Tensor("x", "int8", [1, 4, 4, 8], "NHWC", q, "input"))
    g.inputs.append("x")
    g.add_tensor(Tensor("t", "int8", [1, 8, 4, 4], "*", q))
    g.add_node(Node("transpose", "t0", ["x"], ["t"], {"perms": [0, 3, 1, 2]}))
    g.add_tensor(Tensor("c", "int8", [2, 8, 4, 4], "*", q))
    g.add_node(Node("concat", "c0", ["t", "t"], ["c"], {"axis": 0}))
    g.add_tensor(Tensor("p", "int8", [2, 8, 6, 4], "*", q))
    g.add_node(Node("pad", "p0", ["c"], ["p"],
                    {"padding": [0, 0, 0, 0, 1, 1, 0, 0], "pad_const": -5}))
    g.add_const("tab", np.arange(-128, 128, dtype=np.int8), "*",
                QuantParams.per_tensor(0.05, 0))
    g.add_tensor(Tensor("y", "int8", [2, 8, 6, 4], "*", q))
    g.add_node(Node("table", "l0", ["p", "tab"], ["y"], {}))
    g.outputs.append("y")
    g.validate()
    return g


ALL_FIXTURES = {
    "conv2d": lambda: _conv_graph(False),
    "depthwise": lambda: _conv_graph(True),
    "residual_add": _residual_graph,
    "pool_fc": _pool_fc_graph,
    "matmul": _matmul_graph,
    "shape_ops": _shape_graph,
}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@needs_mlir
@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_every_op_emits_parseable_tosa(name, tmp_path):
    res = emit_module(ALL_FIXTURES[name]())
    mlir_roundtrip(res.text, tmp_path, name + ".mlir")


@needs_mlir
def test_op_coverage_spans_what_mobilenetv2_needs(tmp_path):
    """conv2d, depthwise, add, clamp, avg_pool2d, fully_connected, reshape,
    rescale, matmul -- all present in at least one emitted fixture."""
    seen = set()
    for make in ALL_FIXTURES.values():
        text = emit_module(make()).text
        for op in ("tosa.conv2d", "tosa.depthwise_conv2d", "tosa.add",
                   "tosa.clamp", "tosa.avg_pool2d", "tosa.fully_connected",
                   "tosa.reshape", "tosa.rescale", "tosa.matmul",
                   "tosa.transpose", "tosa.concat", "tosa.pad", "tosa.table"):
            if op + " " in text:
                seen.add(op)
    assert seen == {
        "tosa.conv2d", "tosa.depthwise_conv2d", "tosa.add", "tosa.clamp",
        "tosa.avg_pool2d", "tosa.fully_connected", "tosa.reshape",
        "tosa.rescale", "tosa.matmul", "tosa.transpose", "tosa.concat",
        "tosa.pad", "tosa.table"}


@needs_mlir
def test_constants_survive_the_hex_encoding(tmp_path):
    """Weights go out as MLIR's hex byte form; check the values come back.

    MLIR prints dense attrs above ~100 elements back in hex, so the check has
    to use a constant small enough that it prints decimal -- otherwise the
    assertion would only be comparing our hex string to itself.
    """
    g = KGraph(name="hexcheck")
    g.add_tensor(Tensor("x", "int8", [1, 2, 2, 4], "NHWC",
                        QuantParams.per_tensor(0.05, 0), "input"))
    g.inputs.append("x")
    w = np.array([[-128, -1, 0, 1], [2, 3, 126, 127]], dtype=np.int8)
    g.add_const("w", w.reshape(2, 1, 1, 4), "OHWI",
                QuantParams.per_channel([0.01, 0.02], 0))
    g.add_const("b", np.array([-2147483648, 2147483647], dtype=np.int32), "C",
                QuantParams.per_channel([5e-4, 5e-4], 0))
    g.add_tensor(Tensor("acc", "int32", [1, 2, 2, 2], "NHWC",
                        QuantParams.per_channel([5e-4, 5e-4], 3)))
    g.add_node(Node("conv2d", "c0", ["x", "w", "b"], ["acc"], {
        "pad": [0, 0, 0, 0], "stride": [1, 1], "dilation": [1, 1],
        "input_zp": 0, "weight_zp": 0}))
    g.outputs.append("acc")
    g.validate()
    out = mlir_roundtrip(emit_module(g).text, tmp_path)
    assert "[[[[-128, -1, 0, 1]]], [[[2, 3, 126, 127]]]]" in out
    assert "[-2147483648, 2147483647]" in out           # i32 little-endian


@needs_mlir
def test_20_1_6_spellings_are_used(tmp_path):
    """The traps from docs/TOSA_NOTES.md, asserted on emitted text."""
    text = emit_module(_conv_graph(False)).text
    assert "acc_type = i32" in text                       # mandatory
    assert "#tosa.conv_quant<input_zp = -7, weight_zp = 0>" in text
    assert "shift = array<i8:" in text                    # NOT array<i32:>
    assert "scale32 = true" in text and "per_channel = true" in text
    assert "min_fp" in text and "max_fp" in text          # required on clamp
    dw = emit_module(_conv_graph(True)).text
    assert "tosa.depthwise_conv2d" in dw
    assert "tensor<3x3x4x1xi8>" in dw                     # HWCM, not OHWI
    sh = emit_module(_shape_graph()).text
    assert "new_shape = array<i64:" not in sh             # no reshape here
    assert "tosa.const_shape" in sh and "xindex>" in sh   # pad migrated
    assert "tosa.transpose %" in sh and ", %" in sh       # perms is an operand


def test_fused_kea_ops_are_refused_by_name():
    g = KGraph(name="sm")
    g.add_tensor(Tensor("x", "int8", [1, 4], "*",
                        QuantParams.per_tensor(0.05, 0), "input"))
    g.inputs.append("x")
    g.add_const("e", np.ones(256, dtype=np.int32), "*",
                QuantParams.per_tensor(1.0, 0))
    g.add_tensor(Tensor("y", "int32", [1, 4], "*",
                        QuantParams.per_tensor(2 ** -15, 0)))
    g.add_node(Node("softmax", "s0", ["x", "e"], ["y"],
                    {"axis": 1, "frac_bits": 15}))
    g.outputs.append("y")
    with pytest.raises(TosaEmitError) as e:
        emit_module(g)
    assert "softmax" in str(e.value)


@needs_mlir
def test_subgraph_slices_are_closed(tmp_path):
    """A node slice becomes a function whose arguments are what it reads."""
    g = _conv_graph(False)
    res = emit_module(g, function="tail", nodes=g.nodes[1:])
    assert res.inputs == ["acc"]
    assert "func.func @tail(%acc: tensor<1x8x8x4xi32>)" in res.text
    mlir_roundtrip(res.text, tmp_path)


@needs_mlir
def test_inexact_pool_scale_is_split_and_reported(tmp_path):
    """A pool whose folded scale is not 1/(kh*kw) cannot be one TOSA op."""
    g = _pool_fc_graph()
    pool = [n for n in g.nodes if n.op == "global_avg_pool"][0]
    pool.attrs["multiplier"] = 1947307623      # MobileNetV2's head value
    pool.attrs["shift"] = 36
    res = emit_module(g)
    assert res.notes and "NOT bit-exact" in res.notes[0]
    assert res.text.count("= tosa.rescale ") == 2   # the pool gained one
    mlir_roundtrip(res.text, tmp_path)


@needs_compiler
@needs_mlir
@pytest.mark.parametrize("name", ["conv2d", "depthwise", "residual_add",
                                  "matmul"])
def test_emitted_tosa_survives_the_kea_pipeline(name, tmp_path):
    g = ALL_FIXTURES[name]()
    res = emit_module(g)
    kasm = kea_pipeline(res.text, tmp_path, res.function)
    assert "HALT" in kasm


@needs_mlir
@pytest.mark.slow
def test_mobilenetv2_whole_graph_round_trips(tmp_path):
    path = os.path.join(MODELS, "mobilenetv2_int8.kgraph.json")
    if not os.path.exists(path):
        pytest.skip("run frontend/export_mobilenetv2.py")
    g = KGraph.load(path)
    res = emit_module(g, function="mobilenetv2")
    assert len(res.inputs) == 1 and res.outputs == ["fc"]
    mlir_roundtrip(res.text, tmp_path, "mnv2.mlir")


@needs_mlir
@pytest.mark.slow
def test_mobilenetv2_feature_extractor_round_trips(tmp_path):
    """The slice the demo actually compiles: nodes 0..178."""
    path = os.path.join(MODELS, "mobilenetv2_int8.kgraph.json")
    if not os.path.exists(path):
        pytest.skip("run frontend/export_mobilenetv2.py")
    g = KGraph.load(path)
    res = emit_module(g, function="mnv2_features", nodes=g.nodes[:179])
    assert res.outputs == ["head"]
    assert not res.notes           # no approximation anywhere in this slice
    mlir_roundtrip(res.text, tmp_path, "feat.mlir")
