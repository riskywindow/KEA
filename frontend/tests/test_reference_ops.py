"""Reference interpreter: integer kernels vs a float model of the same op.

Each test builds a one-op graph through the real calibrate/build pipeline, runs
it through the integer interpreter, dequantizes, and compares against the float
result.  Tolerances are stated in *quantization steps of the output tensor*,
which is the only unit in which a quantization error bound is meaningful.
"""

import numpy as np
import pytest
import torch

from kea_frontend.builder import TorchTracer
from kea_frontend.ir import KGraph
from kea_frontend.pipeline import build_graph, calibrate
from kea_frontend.reference import (
    dequantize,
    execute,
    execute_float,
    idiv_trunc,
    isqrt,
    quantize,
)


def run_pair(build_fn, shape, x, observer="minmax", tol_max=3.0, tol_rms=1.5):
    """Build + run one graph; return (int8_dequantized, float_reference, step).

    Two bounds are checked, both in units of the output tensor's quantization
    step:

    ``tol_rms``
        the RMS error -- the tight, informative bound.  Quantization noise is
        roughly uniform, so RMS well under one step means the integer kernel is
        doing the right arithmetic.
    ``tol_max``
        the worst single element.  This necessarily grows with fan-in (errors
        from ``K`` input elements accumulate before the output is requantized),
        so it is looser and per-op.

    Both limits are set from measured values with roughly 2x headroom.  They
    bound *quantization* error only -- a wrong kernel fails by orders of
    magnitude, not by a factor of two.
    """
    rng = np.random.default_rng(0)
    cal = [rng.standard_normal((4,) + tuple(shape[1:])).astype(np.float32) for _ in range(3)]
    cal.append(np.ascontiguousarray(np.repeat(x, 4, axis=0)))
    calib = calibrate(build_fn, cal, observer=observer, verbose=False)
    g = build_graph(build_fn, shape, calib, "unit")
    out_name = g.outputs[0]
    t = g.tensor(out_name)
    xq = quantize(x, g.tensor("input"))
    qi = execute(g, {"input": xq})[out_name]
    got = dequantize(qi, t)
    ref = build_fn(TorchTracer({}, record=False), torch.as_tensor(x))
    ref = ref.detach().numpy().astype(np.float64)
    step = t.quant.scale[0]
    err = np.abs(got - ref) / step
    rms = float(np.sqrt((err ** 2).mean()))
    mx = float(err.max())
    detail = (f"step={step:.3g}, float range=[{ref.min():.3g}, {ref.max():.3g}], "
              f"max={mx:.2f} rms={rms:.2f} steps")
    assert rms <= tol_rms, f"RMS error {rms:.2f} > {tol_rms} steps; {detail}"
    assert mx <= tol_max, f"max error {mx:.2f} > {tol_max} steps; {detail}"
    return got, ref, step


# ---------------------------------------------------------------------------
# Integer primitives
# ---------------------------------------------------------------------------


def test_idiv_trunc_truncates_toward_zero():
    assert list(idiv_trunc([7, -7, 7, -7], [2, 2, -2, -2])) == [3, -3, -3, 3]
    assert int(idiv_trunc(-1, 2)) == 0          # NOT -1 (that would be floor)
    with pytest.raises(ZeroDivisionError):
        idiv_trunc(1, 0)


def test_isqrt_is_exact():
    v = np.array([0, 1, 2, 3, 4, 8, 9, 15, 16, 10**12, 10**12 + 1,
                  (1 << 40) - 1, 1 << 40, (1 << 52) + 7], dtype=np.int64)
    r = isqrt(v)
    assert np.all(r * r <= v)
    assert np.all((r + 1) * (r + 1) > v)
    rng = np.random.default_rng(3)
    v = rng.integers(0, 1 << 50, size=20000, dtype=np.int64)
    r = isqrt(v)
    assert np.all(r * r <= v) and np.all((r + 1) * (r + 1) > v)


# ---------------------------------------------------------------------------
# Per-op: integer vs float
# ---------------------------------------------------------------------------


def test_conv2d():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((6, 3, 3, 3)) * 0.3
    b = rng.standard_normal(6) * 0.1
    x = rng.standard_normal((1, 12, 12, 3)).astype(np.float32)
    run_pair(lambda n, t: n.conv2d("c", t, w, b, (2, 2), (1, 1, 1, 1), (1, 1), 1, None),
             [1, 12, 12, 3], x, tol_max=3.0, tol_rms=1.0)


def test_conv2d_relu6():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((6, 3, 3, 3)) * 0.5
    b = rng.standard_normal(6) * 0.1
    x = rng.standard_normal((1, 10, 10, 3)).astype(np.float32)
    got, ref, _ = run_pair(
        lambda n, t: n.conv2d("c", t, w, b, (1, 1), (1, 1, 1, 1), (1, 1), 1, "relu6"),
        [1, 10, 10, 3], x, tol_max=6.0, tol_rms=1.5)
    assert got.min() >= -1e-9 and got.max() <= 6.0 + 1e-9


def test_depthwise_conv2d():
    rng = np.random.default_rng(3)
    w = rng.standard_normal((4, 1, 3, 3)) * 0.4
    b = rng.standard_normal(4) * 0.1
    x = rng.standard_normal((1, 9, 9, 4)).astype(np.float32)
    run_pair(lambda n, t: n.conv2d("d", t, w, b, (1, 1), (1, 1, 1, 1), (1, 1), 4, None),
             [1, 9, 9, 4], x, tol_max=4.0, tol_rms=1.5)


def test_fully_connected():
    rng = np.random.default_rng(4)
    w = rng.standard_normal((7, 12)) * 0.3
    b = rng.standard_normal(7) * 0.1
    x = rng.standard_normal((1, 12)).astype(np.float32)
    run_pair(lambda n, t: n.linear("f", t, w, b), [1, 12], x, tol_max=4.0, tol_rms=1.5)


def test_residual_add():
    rng = np.random.default_rng(5)
    w = rng.standard_normal((3, 3, 1, 1)) * 0.4
    b = rng.standard_normal(3) * 0.1

    def build(n, t):
        y = n.conv2d("c", t, w, b, (1, 1), (0, 0, 0, 0), (1, 1), 1, None)
        return n.add("r", t, y)

    x = rng.standard_normal((1, 6, 6, 3)).astype(np.float32)
    run_pair(build, [1, 6, 6, 3], x, tol_max=5.0, tol_rms=1.5)


def test_global_avg_pool():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((1, 8, 8, 5)).astype(np.float32)
    run_pair(lambda n, t: n.global_avg_pool("g", t), [1, 8, 8, 5], x, tol_max=2.0, tol_rms=1.0)


def test_avg_pool2d():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((1, 8, 8, 4)).astype(np.float32)
    run_pair(lambda n, t: n.avg_pool2d("a", t, (2, 2), (2, 2)), [1, 8, 8, 4], x,
             tol_max=3.0, tol_rms=1.0)


def test_matmul_two_activations():
    rng = np.random.default_rng(8)
    wq = rng.standard_normal((6, 6)) * 0.4
    wk = rng.standard_normal((6, 6)) * 0.4
    bz = np.zeros(6)

    def build(n, t):
        f = n.reshape("f", t, [int(t.shape[0]), 5, 6])
        q = n.linear("q", f, wq, bz)
        k = n.linear("k", f, wk, bz)
        kt = n.transpose("kt", k, [0, 2, 1])
        return n.matmul("m", q, kt)

    x = rng.standard_normal((1, 5, 6)).astype(np.float32)
    run_pair(build, [1, 5, 6], x, tol_max=4.0, tol_rms=1.5)


def test_softmax():
    rng = np.random.default_rng(9)
    w = rng.standard_normal((8, 8)) * 0.8

    def build(n, t):
        f = n.reshape("f", t, [int(t.shape[0]), 4, 8])
        y = n.linear("l", f, w, np.zeros(8))
        return n.softmax("s", y, axis=-1)

    x = rng.standard_normal((1, 4, 8)).astype(np.float32)
    # A LUT-based integer softmax on a 1/256 output grid: ~2% of full scale.
    got, ref, _ = run_pair(build, [1, 4, 8], x, tol_max=11.0, tol_rms=3.5)
    # softmax output is on the canonical 1/256 grid and must still be a
    # distribution to within its own resolution
    assert got.min() >= -1e-9
    assert np.allclose(got.sum(axis=-1), 1.0, atol=0.05)


def test_layernorm():
    rng = np.random.default_rng(10)
    gm = rng.uniform(0.5, 1.5, 16)
    bt = rng.standard_normal(16) * 0.2

    def build(n, t):
        f = n.reshape("f", t, [int(t.shape[0]), 4, 16])
        return n.layernorm("ln", f, gm, bt)

    x = (rng.standard_normal((1, 4, 16)) * 2.0).astype(np.float32)
    run_pair(build, [1, 4, 16], x, tol_max=3.0, tol_rms=1.0)


def test_gelu_table():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((1, 4, 16)).astype(np.float32) * 2.0
    run_pair(lambda n, t: n.gelu("g", t), [1, 4, 16], x, tol_max=3.0, tol_rms=1.5)


def test_transpose_and_reshape_are_exact():
    """Layout ops must not perturb values at all: same scale in and out."""
    rng = np.random.default_rng(12)

    def build(n, t):
        y = n.transpose("t", t, [0, 3, 1, 2])
        return n.reshape("r", y, [int(t.shape[0]), 3 * 4 * 5])

    x = rng.standard_normal((1, 4, 5, 3)).astype(np.float32)
    cal = [rng.standard_normal((2, 4, 5, 3)).astype(np.float32) for _ in range(2)]
    calib = calibrate(build, cal, verbose=False)
    g = build_graph(build, [1, 4, 5, 3], calib, "layout")
    xq = quantize(x, g.tensor("input"))
    out = execute(g, {"input": xq})[g.outputs[0]]
    expect = np.transpose(xq, (0, 3, 1, 2)).reshape(1, -1)
    assert np.array_equal(out, expect)
    assert g.tensor(g.outputs[0]).quant.scale == g.tensor("input").quant.scale


# ---------------------------------------------------------------------------
# Determinism / purity of the integer path
# ---------------------------------------------------------------------------


def test_execute_is_deterministic_and_integral():
    rng = np.random.default_rng(13)
    w = rng.standard_normal((5, 3, 3, 3)) * 0.3

    def build(n, t):
        return n.conv2d("c", t, w, np.zeros(5), (1, 1), (1, 1, 1, 1), (1, 1), 1, "relu6")

    cal = [rng.standard_normal((3, 8, 8, 3)).astype(np.float32) for _ in range(2)]
    calib = calibrate(build, cal, verbose=False)
    g = build_graph(build, [1, 8, 8, 3], calib, "det")
    x = quantize(rng.standard_normal((1, 8, 8, 3)).astype(np.float32), g.tensor("input"))
    keep = [n for n in g.tensors if g.tensors[n].kind == "activation"]
    a = execute(g, {"input": x}, keep=keep)
    b = execute(g, {"input": x}, keep=keep)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"{k} is not deterministic"
        assert a[k].dtype.kind == "i", f"{k} escaped to {a[k].dtype}"


def test_batch_invariance():
    """Per-image results must not depend on how images are batched."""
    rng = np.random.default_rng(14)
    w = rng.standard_normal((4, 3, 3, 3)) * 0.3
    build = lambda n, t: n.conv2d("c", t, w, np.zeros(4), (2, 2), (1, 1, 1, 1),
                                  (1, 1), 1, "relu6")
    cal = [rng.standard_normal((3, 8, 8, 3)).astype(np.float32) for _ in range(2)]
    calib = calibrate(build, cal, verbose=False)
    g1 = build_graph(build, [1, 8, 8, 3], calib, "b1")
    g3 = build_graph(build, [3, 8, 8, 3], calib, "b3")
    x = rng.standard_normal((3, 8, 8, 3)).astype(np.float32)
    xq = quantize(x, g3.tensor("input"))
    big = execute(g3, {"input": xq})[g3.outputs[0]]
    for i in range(3):
        one = execute(g1, {"input": xq[i : i + 1]})[g1.outputs[0]]
        assert np.array_equal(one[0], big[i])
