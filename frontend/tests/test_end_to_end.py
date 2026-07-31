"""End-to-end: whole networks through the real pipeline.

These are the slowest tests.  They are also the ones that would catch a
systematic error the per-op tests miss -- a wrong layout convention, a
mis-threaded zero point, a scale attached to the wrong tensor -- because such a
bug degrades gracefully at one op and catastrophically over 50.
"""

import os

import numpy as np
import pytest
import torch

from kea_frontend import data as kdata
from kea_frontend.builder import TorchTracer
from kea_frontend.ir import KGraph
from kea_frontend.nets import (
    TinyViTConfig,
    build_mobilenetv2,
    build_tiny_vit,
    extract_mobilenetv2,
    make_tiny_vit_params,
)
from kea_frontend.pipeline import build_graph, calibrate
from kea_frontend.reference import execute, quantize

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(REPO, "models")


def _cal_batches(n, size=224, seed=0, bs=4):
    """Real images if fetched, otherwise deterministic synthetic ones."""
    items = kdata.list_images("val")
    rng = np.random.default_rng(seed)
    if items:
        idx = sorted(rng.permutation(len(items))[:n])
        sel = [items[i] for i in idx]
        return [kdata.load_batch(sel[i : i + bs], size)[0] for i in range(0, len(sel), bs)]
    return [rng.standard_normal((bs, size, size, 3)).astype(np.float32)
            for _ in range(max(1, n // bs))]


@pytest.fixture(scope="module")
def mobilenet():
    import torchvision

    try:
        w = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        m = torchvision.models.mobilenet_v2(weights=w).eval()
    except Exception as exc:
        pytest.skip(f"pretrained MobileNetV2 unavailable: {exc}")
    return m


# ---------------------------------------------------------------------------
# MobileNetV2
# ---------------------------------------------------------------------------


def test_bn_folding_is_exact(mobilenet):
    """Folding BN into the conv must not change the float model's output."""
    desc = extract_mobilenetv2(mobilenet)
    x = np.random.default_rng(0).standard_normal((1, 224, 224, 3)).astype(np.float32)
    with torch.no_grad():
        ref = mobilenet(torch.as_tensor(x).permute(0, 3, 1, 2).contiguous()).numpy()
        got = build_mobilenetv2(TorchTracer({}, record=False), torch.as_tensor(x),
                                desc).numpy()
    # float32 arithmetic over 52 folded convs; this is reassociation error only
    assert np.abs(got - ref).max() < 2e-3, np.abs(got - ref).max()
    assert got.argmax() == ref.argmax()


@pytest.mark.slow
def test_mobilenetv2_int8_tracks_float(mobilenet):
    """The full int8 graph must agree with the float model on real images."""
    desc = extract_mobilenetv2(mobilenet)
    build_fn = lambda net, x: build_mobilenetv2(net, x, desc)
    cal = _cal_batches(16, seed=1)
    calib = calibrate(build_fn, cal, observer="percentile", verbose=False)
    g = build_graph(build_fn, [1, 224, 224, 3], calib, "mnv2_test")

    xs = _cal_batches(6, seed=2, bs=1)
    agree = 0
    cs = []
    t = g.tensor(g.outputs[0])
    for x in xs:
        qi = execute(g, {"input": quantize(x, g.tensor("input"))})[g.outputs[0]]
        with torch.no_grad():
            ref = mobilenet(torch.as_tensor(x).permute(0, 3, 1, 2).contiguous()).numpy()
        # argmax on the raw integer logits: requantization is monotonic
        agree += int(np.asarray(qi).argmax() == ref.argmax())
        deq = (np.asarray(qi, dtype=np.float64) - t.quant.zero_point[0]) * t.quant.scale[0]
        cs.append(float(np.corrcoef(deq.ravel(), ref.ravel().astype(np.float64))[0, 1]))
    # Bounds are grounded in measurement: the fully-calibrated artifact scores
    # mean 0.987 / min 0.954 over real images. This graph is calibrated on only
    # 16 images, so the bounds are looser -- they are a smoke test for a
    # systematically wrong layout or zero point, not an accuracy target.
    assert np.mean(cs) > 0.95, f"mean int8/float logit correlation {np.mean(cs):.4f}"
    assert min(cs) > 0.90, f"worst int8/float logit correlation {min(cs):.4f}"
    assert agree >= len(xs) - 2, f"argmax agreement {agree}/{len(xs)}"


@pytest.mark.slow
def test_exported_mobilenetv2_artifact_runs():
    """The checked-in artifact must load and produce a sane prediction."""
    p = os.path.join(MODELS, "mobilenetv2_int8.kgraph.json")
    if not os.path.exists(p):
        pytest.skip("run frontend/export_mobilenetv2.py first")
    g = KGraph.load(p)
    assert g.tensor("input").shape == [1, 224, 224, 3]
    assert g.tensor("input").layout == "NHWC"

    items = kdata.list_images("val")
    if not items:
        pytest.skip("no evaluation images; run scripts/fetch_calibration_data.sh")
    import torchvision

    try:
        mw = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        fm = torchvision.models.mobilenet_v2(weights=mw).eval()
    except Exception as exc:
        pytest.skip(f"pretrained MobileNetV2 unavailable: {exc}")

    rng = np.random.default_rng(5)
    sel = [items[i] for i in sorted(rng.permutation(len(items))[:12])]
    t = g.tensor(g.outputs[0])
    ok = 0
    cs = []
    for path, label in sel:
        x = kdata.preprocess(path)[None]
        out = np.asarray(execute(g, {"input": quantize(x, g.tensor("input"))})[g.outputs[0]])
        ok += int(out.argmax() == label)
        deq = (out.astype(np.float64) - t.quant.zero_point[0]) * t.quant.scale[0]
        with torch.no_grad():
            ref = fm(torch.as_tensor(x).permute(0, 3, 1, 2).contiguous()).numpy()
        cs.append(float(np.corrcoef(deq.ravel(), ref.ravel().astype(np.float64))[0, 1]))
    # Measured on this artifact: mean 0.987, min 0.954.
    assert np.mean(cs) > 0.97, f"mean logit correlation {np.mean(cs):.4f}"
    assert min(cs) > 0.90, f"worst logit correlation {min(cs):.4f}"
    # 12 images from a 10-class subset; assert it is far better than chance
    assert ok >= 7, f"only {ok}/12 correct -- the artifact looks broken"


def test_every_intermediate_stays_integral(mobilenet):
    """No float can appear anywhere in the inference path."""
    desc = extract_mobilenetv2(mobilenet)
    build_fn = lambda net, x: build_mobilenetv2(net, x, desc)
    calib = calibrate(build_fn, _cal_batches(4, seed=3), verbose=False)
    g = build_graph(build_fn, [1, 224, 224, 3], calib, "mnv2_int")
    x = _cal_batches(1, seed=4, bs=1)[0]
    keep = [n for n, t in g.tensors.items() if t.kind == "activation"]
    env = execute(g, {"input": quantize(x, g.tensor("input"))}, keep=keep)
    for n, v in env.items():
        assert v.dtype.kind == "i", f"{n} has non-integer dtype {v.dtype}"
        assert v.dtype.itemsize <= 4, f"{n} is wider than int32"


# ---------------------------------------------------------------------------
# Tiny ViT
# ---------------------------------------------------------------------------


def test_tiny_vit_int8_tracks_float():
    """Exercises matmul / softmax / layernorm / table, which MobileNet does not."""
    cfg = TinyViTConfig(image_size=64, patch=16, dim=96, depth=2, heads=3)
    p = make_tiny_vit_params(cfg)
    build_fn = lambda net, x: build_tiny_vit(net, x, cfg, p)
    rng = np.random.default_rng(0)
    cal = [rng.standard_normal((2, 64, 64, 3)).astype(np.float32) for _ in range(4)]
    calib = calibrate(build_fn, cal, observer="percentile", verbose=False)
    g = build_graph(build_fn, [1, 64, 64, 3], calib, "vit_test")

    ops = {n.op for n in g.nodes}
    for required in ("matmul", "softmax", "layernorm", "table", "transpose"):
        assert required in ops, f"tiny ViT graph is missing {required}"

    t = g.tensor(g.outputs[0])
    cs = []
    with torch.no_grad():
        for _ in range(3):
            x = rng.standard_normal((1, 64, 64, 3)).astype(np.float32)
            qi = execute(g, {"input": quantize(x, g.tensor("input"))})[g.outputs[0]]
            deq = (np.asarray(qi, np.float64) - t.quant.zero_point[0]) * t.quant.scale[0]
            ref = build_fn(TorchTracer({}, record=False),
                           torch.as_tensor(x)).numpy().astype(np.float64)
            cs.append(float(deq.ravel() @ ref.ravel() /
                            (np.linalg.norm(deq) * np.linalg.norm(ref))))
    assert min(cs) > 0.98, f"logit cosine similarity {cs}"


def test_exported_tiny_vit_artifact():
    p = os.path.join(MODELS, "tiny_vit_int8.kgraph.json")
    if not os.path.exists(p):
        pytest.skip("run frontend/export_tiny_vit.py first")
    g = KGraph.load(p)
    assert g.metadata.get("trained") is False, \
        "the tiny ViT must advertise that it is not trained"
    ops = {n.op for n in g.nodes}
    assert {"matmul", "softmax", "layernorm", "table"} <= ops


# ---------------------------------------------------------------------------
# ONNX ingest
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_onnx_and_torch_paths_produce_identical_graphs(mobilenet, tmp_path):
    """The same network via ONNX and via torch must quantize to the same graph.

    Same calibration data, same observer => the same op sequence, the same
    tensor shapes, and quantized constants that agree to within 1 LSB.  Only
    the node *names* differ.  This is the strongest available check that the
    ONNX importer does not quietly reorder or reinterpret anything.

    ``do_constant_folding=False`` keeps BatchNormalization as real ONNX nodes,
    so *our* ``fold_conv_bn`` runs on both paths.

    Why 1 LSB rather than exact: ONNX stores the BN ``epsilon`` as a float32
    attribute, so the importer sees 9.99999974737875e-06 where the torch path
    sees Python's exact 1e-5.  That perturbs ``gamma/sqrt(var+eps)`` in the last
    bits, which occasionally tips a ``rint`` across a .5 boundary.  Measured:
    12 of 106 constants differ, always bias tensors, always by exactly 1 on 1-2
    elements out of hundreds.  Tightening this to exact equality would require
    the importer to lie about the epsilon it was given.
    """
    onnx = pytest.importorskip("onnx")
    from kea_frontend.onnx_ingest import make_build_fn, parse_onnx

    path = str(tmp_path / "m.onnx")
    torch.onnx.export(mobilenet, torch.randn(1, 3, 224, 224), path,
                      opset_version=13, do_constant_folding=False,
                      input_names=["input"], output_names=["logits"])
    steps, in_name, out_name = parse_onnx(onnx.load(path))
    onnx_build = make_build_fn(steps, in_name, out_name)

    desc = extract_mobilenetv2(mobilenet)
    torch_build = lambda net, x: build_mobilenetv2(net, x, desc)

    cal = _cal_batches(8, seed=7)
    g_onnx = build_graph(onnx_build, [1, 224, 224, 3],
                         calibrate(onnx_build, cal, verbose=False), "o")
    g_torch = build_graph(torch_build, [1, 224, 224, 3],
                          calibrate(torch_build, cal, verbose=False), "t")

    assert len(g_onnx.nodes) == len(g_torch.nodes)
    assert [n.op for n in g_onnx.nodes] == [n.op for n in g_torch.nodes]

    # Constants, compared in emission order (names differ between the paths).
    co = [g_onnx.consts[t.const_key] for t in g_onnx.tensors.values() if t.kind == "const"]
    ct = [g_torch.consts[t.const_key] for t in g_torch.tensors.values() if t.kind == "const"]
    assert len(co) == len(ct)
    n_diff_tensors = 0
    for a, b in zip(co, ct):
        assert a.shape == b.shape and a.dtype == b.dtype
        d = np.abs(a.astype(np.int64) - b.astype(np.int64))
        assert d.max() <= 1, f"constants differ by {d.max()} LSB, expected <= 1"
        frac = float((d != 0).mean())
        assert frac < 0.05, f"{frac:.1%} of elements differ; expected a handful"
        n_diff_tensors += int(d.any())
    # int8 weight tensors must be bit-identical; only int32 biases may wobble.
    for a, b in zip(co, ct):
        if a.dtype == np.int8:
            assert np.array_equal(a, b), "int8 weights differ between ingest paths"
    assert n_diff_tensors <= len(co) // 4, (
        f"{n_diff_tensors}/{len(co)} constants differ -- too many to be epsilon noise"
    )

    # Every structural attribute must match exactly; requant integers may shift
    # by one ulp of the derived scale for the same reason as the biases.
    for a, b in zip(g_onnx.nodes, g_torch.nodes):
        ka = {k: v for k, v in a.attrs.items() if k not in ("multiplier", "shift")}
        kb = {k: v for k, v in b.attrs.items() if k not in ("multiplier", "shift")}
        assert ka == kb, f"{a.op}: attrs differ\n{ka}\n{kb}"
        if "shift" in a.attrs:
            assert a.attrs["shift"] == b.attrs["shift"], f"{a.op}: shifts differ"
            ma = np.asarray(a.attrs["multiplier"], dtype=np.int64)
            mb = np.asarray(b.attrs["multiplier"], dtype=np.int64)
            rel = np.abs(ma - mb) / np.maximum(mb, 1)
            assert rel.max() < 1e-6, f"{a.op}: multipliers differ by {rel.max():.2g}"


def test_onnx_rejects_unsupported_ops_loudly():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from kea_frontend.onnx_ingest import UnsupportedOnnxOp, parse_onnx

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])
    node = helper.make_node("Sigmoid", ["x"], ["y"], name="sig")
    m = helper.make_model(helper.make_graph([node], "g", [x], [y]))
    with pytest.raises(UnsupportedOnnxOp, match="Sigmoid"):
        parse_onnx(m)


def test_onnx_rejects_non_relu6_clip():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from kea_frontend.onnx_ingest import UnsupportedOnnxOp, parse_onnx

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])
    node = helper.make_node("Clip", ["x"], ["y"], name="c", min=0.0, max=1.0)
    m = helper.make_model(helper.make_graph([node], "g", [x], [y]))
    with pytest.raises(UnsupportedOnnxOp, match="ReLU6"):
        parse_onnx(m)


# ---------------------------------------------------------------------------
# Golden I/O vectors -- the C++ simulator's cross-validation target
# ---------------------------------------------------------------------------


def test_golden_io_vectors_reproduce():
    """frontend/testdata/golden_io_*.npz must still match the interpreter.

    These files are what the simulator team asserts against.  If this test
    fails, either the reference semantics changed (in which case the vectors
    and docs/QUANTIZATION.md must be regenerated and the change announced) or
    the interpreter regressed.
    """
    import json

    td = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata")
    mp = os.path.join(td, "golden_io_manifest.json")
    if not os.path.exists(mp):
        pytest.skip("run frontend/gen_golden_io.py first")
    with open(mp) as f:
        manifest = json.load(f)

    for model, info in manifest["models"].items():
        gp = os.path.join(REPO, info["graph"])
        npz = os.path.join(td, f"golden_io_{model}.npz")
        if not (os.path.exists(gp) and os.path.exists(npz)):
            pytest.skip(f"{model} artifacts missing")
        g = KGraph.load(gp)
        assert g.outputs[0] == info["output_tensor"]
        with np.load(npz) as z:
            for i in range(info["cases"]):
                x = z[f"input_{i}"]
                want = z[f"output_{i}"]
                got = np.asarray(execute(g, {info["input_tensor"]: x})[g.outputs[0]])
                assert got.dtype == want.dtype
                assert np.array_equal(got, want), (
                    f"{model} case {i}: golden vector mismatch "
                    f"({int((got != want).sum())} of {want.size} elements differ)"
                )
