"""Bit-exact integer reference interpreter for ``.kgraph.json``.

This is the **golden model**.  The C++ simulator and the hardware model are
validated against the output of :func:`execute`.  There is no floating point
anywhere in the inference path: every intermediate is an integer numpy array,
every rounding decision is spelled out.

Accumulator widths are stated per-op in ``docs/FRONTEND.md`` and enforced here:
where the spec says "int32 accumulator", this module computes in int64 and
raises :class:`OverflowError` if a value escapes the int32 range, rather than
silently wrapping.  A conforming implementation may therefore use plain int32
arithmetic.

A companion :func:`execute_float` runs the *same graph* in float64 using the
quantization metadata.  It exists only for testing -- it is not part of the
compiler contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .apply_scale import INT32_MAX, INT32_MIN, apply_scale_32
from .ir import DTYPES, KGraph, KGraphError, Node, Tensor

__all__ = ["execute", "execute_float", "idiv_trunc", "isqrt", "dequantize", "quantize"]

_DTYPE_RANGE = {"int8": (-128, 127), "int32": (INT32_MIN, INT32_MAX)}


# ---------------------------------------------------------------------------
# Integer primitives shared by several ops.  Reimplement these EXACTLY.
# ---------------------------------------------------------------------------


def idiv_trunc(a, b):
    """Integer division truncating **toward zero** (C/C++ ``/`` semantics).

    ``b`` must be non-zero.  numpy's ``//`` floors, which differs for negative
    operands, so this is spelled out explicitly.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if np.any(b == 0):
        raise ZeroDivisionError("idiv_trunc: divisor is zero")
    q = np.abs(a) // np.abs(b)
    neg = (a < 0) != (b < 0)
    return np.where(neg, -q, q).astype(np.int64)


def isqrt(v):
    """Exact integer square root: the unique ``r >= 0`` with ``r*r <= v < (r+1)**2``.

    Defined for ``v >= 0`` only.  Implementations may use any method (Newton,
    binary search, hardware sqrt with correction) as long as they produce this
    value; it is uniquely determined.
    """
    v = np.asarray(v, dtype=np.int64)
    if np.any(v < 0):
        raise ValueError("isqrt: negative input")
    r = np.floor(np.sqrt(v.astype(np.float64))).astype(np.int64)
    r = np.maximum(r, 0)
    # float64 sqrt is accurate to well under 1 ulp for v < 2**52; correct by a
    # couple of steps so the result is exact for the full int64 non-negative
    # range we actually use.
    for _ in range(4):
        r = np.where(r * r > v, r - 1, r)
        r = np.maximum(r, 0)
    for _ in range(4):
        r = np.where((r + 1) * (r + 1) <= v, r + 1, r)
    return r


#: float64 represents every integer up to 2**53 exactly, so a BLAS ``matmul``
#: over small integers is *bit-exact* provided every partial sum stays below
#: this bound.  Reductions are order-independent under that condition, so the
#: result does not depend on the BLAS implementation.
_EXACT_F64 = 2**53


def _exact_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Integer matmul, computed via float64 BLAS when provably exact.

    ``a`` and ``b`` are int64.  Falls back to integer numpy (slow, but correct)
    when the magnitude bound cannot be proven.
    """
    k = a.shape[-1]
    bound = int(np.abs(a).max(initial=0)) * int(np.abs(b).max(initial=0)) * int(k)
    if bound < _EXACT_F64:
        return np.matmul(a.astype(np.float64), b.astype(np.float64)).astype(np.int64)
    return np.matmul(a, b)


def _sat32(x, where: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    if np.any(x < INT32_MIN) or np.any(x > INT32_MAX):
        raise OverflowError(
            f"{where}: int32 accumulator overflow (range "
            f"[{int(x.min())}, {int(x.max())}])"
        )
    return x.astype(np.int32)


def _clamp_dtype(x, dtype: str) -> np.ndarray:
    lo, hi = _DTYPE_RANGE[dtype]
    return np.clip(np.asarray(x, dtype=np.int64), lo, hi).astype(DTYPES[dtype])


def _rescale_tail(scaled_i32: np.ndarray, output_zp: int, out_dtype: str) -> np.ndarray:
    """``clamp(int32_wrap(scaled + output_zp), MIN_T, MAX_T)``.

    The ``+ output_zp`` is done in **wrapping int32**, exactly as TOSA's
    lowering does (``arith.addi`` on i32), *before* the saturating clamp to the
    output element type.  Doing it in a wider type and then clamping would
    disagree whenever ``scaled`` is within ``|output_zp|`` of an int32 end --
    rare, but a real bit-level divergence.
    """
    s = np.asarray(scaled_i32, dtype=np.int64) + np.int64(output_zp)
    with np.errstate(over="ignore"):
        wrapped = (s & np.int64(0xFFFFFFFF)).astype(np.uint32).astype(np.int32)
    return _clamp_dtype(wrapped, out_dtype)


def dequantize(q: np.ndarray, t: Tensor) -> np.ndarray:
    """``real = scale * (q - zero_point)`` -- test/debug only."""
    qp = t.quant
    if qp.kind == "per_tensor":
        return (q.astype(np.float64) - qp.zero_point[0]) * qp.scale[0]
    shape = [1] * q.ndim
    shape[qp.axis] = len(qp.scale)
    s = qp.scale_array().reshape(shape)
    z = qp.zp_array().reshape(shape).astype(np.float64)
    return (q.astype(np.float64) - z) * s


def quantize(x: np.ndarray, t: Tensor) -> np.ndarray:
    """``q = clamp(round_half_even(real / scale) + zero_point)``.

    Uses numpy's round-half-to-even.  Only ever applied at the graph boundary
    (turning a float image into the int8 input tensor), never inside the graph.
    """
    qp = t.quant
    if qp.kind == "per_tensor":
        s, z = qp.scale[0], qp.zero_point[0]
    else:
        shape = [1] * x.ndim
        shape[qp.axis] = len(qp.scale)
        s = qp.scale_array().reshape(shape)
        z = qp.zp_array().reshape(shape)
    return _clamp_dtype(np.rint(x.astype(np.float64) / s) + z, t.dtype)


# ---------------------------------------------------------------------------
# Padding / windowing helpers
# ---------------------------------------------------------------------------


def _pad_nhwc(x: np.ndarray, pad, value: int) -> np.ndarray:
    """``pad`` is ``[top, bottom, left, right]`` (TOSA order)."""
    t, b, l, r = pad
    if t == b == l == r == 0:
        return x
    return np.pad(
        x, ((0, 0), (int(t), int(b)), (int(l), int(r)), (0, 0)),
        mode="constant", constant_values=int(value),
    )


def _im2col(x: np.ndarray, kh: int, kw: int, sh: int, sw: int, dh: int, dw: int):
    """NHWC -> (N, OH, OW, kh, kw, C) sliding window view."""
    n, h, w, c = x.shape
    oh = (h - (kh - 1) * dh - 1) // sh + 1
    ow = (w - (kw - 1) * dw - 1) // sw + 1
    if oh <= 0 or ow <= 0:
        raise KGraphError("convolution/pool window larger than padded input")
    sn, shh, sww, sc = x.strides
    view = np.lib.stride_tricks.as_strided(
        x,
        shape=(n, oh, ow, kh, kw, c),
        strides=(sn, shh * sh, sww * sw, shh * dh, sww * dw, sc),
        writeable=False,
    )
    return view, oh, ow


# ---------------------------------------------------------------------------
# Per-op integer kernels
# ---------------------------------------------------------------------------


def _op_conv2d(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    x, w, bias = ins
    a = node.attrs
    izp, wzp = int(a["input_zp"]), int(a["weight_zp"])
    sh, sw = a["stride"]
    dh, dw = a.get("dilation", [1, 1])
    xp = _pad_nhwc(x, a["pad"], izp)
    kh, kw = w.shape[1], w.shape[2]
    view, oh, ow = _im2col(xp, kh, kw, int(sh), int(sw), int(dh), int(dw))
    xc = view.astype(np.int64) - np.int64(izp)          # (N,OH,OW,kh,kw,C)
    wc = w.astype(np.int64) - np.int64(wzp)             # (OC,kh,kw,C)
    n = xc.shape[0]
    acc = _exact_matmul(
        np.ascontiguousarray(xc.reshape(n * oh * ow, -1)),
        np.ascontiguousarray(wc.reshape(wc.shape[0], -1).T),
    ).reshape(n, oh, ow, wc.shape[0])
    acc = acc + bias.astype(np.int64)
    return [_sat32(acc, f"conv2d {node.name}")]


def _op_depthwise_conv2d(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    x, w, bias = ins                                     # w is HWCM
    a = node.attrs
    izp, wzp = int(a["input_zp"]), int(a["weight_zp"])
    sh, sw = a["stride"]
    dh, dw = a.get("dilation", [1, 1])
    kh, kw, c, m = w.shape
    xp = _pad_nhwc(x, a["pad"], izp)
    view, oh, ow = _im2col(xp, kh, kw, int(sh), int(sw), int(dh), int(dw))
    xc = view.astype(np.int64) - np.int64(izp)           # (N,OH,OW,kh,kw,C)
    wc = w.astype(np.int64) - np.int64(wzp)              # (kh,kw,C,M)
    # (N,OH,OW,kh,kw,C,1) * (kh,kw,C,M) summed over kh,kw
    acc = np.einsum("nopijc,ijcm->nopcm", xc, wc, optimize=True)
    acc = acc.reshape(acc.shape[0], oh, ow, c * m)
    acc = acc + bias.astype(np.int64)
    return [_sat32(acc, f"depthwise_conv2d {node.name}")]


def _op_fully_connected(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    x, w, bias = ins
    a = node.attrs
    izp, wzp = int(a["input_zp"]), int(a["weight_zp"])
    xc = x.astype(np.int64) - np.int64(izp)
    wc = np.ascontiguousarray((w.astype(np.int64) - np.int64(wzp)).T)
    acc = _exact_matmul(xc, wc) + bias.astype(np.int64)
    return [_sat32(acc, f"fully_connected {node.name}")]


def _op_matmul(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    a_t, b_t = ins
    a = node.attrs
    azp, bzp = int(a["a_zp"]), int(a["b_zp"])
    ac = a_t.astype(np.int64) - np.int64(azp)
    bc = b_t.astype(np.int64) - np.int64(bzp)
    return [_sat32(_exact_matmul(ac, bc), f"matmul {node.name}")]


def _op_add(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    acc = ins[0].astype(np.int64)
    for t in ins[1:]:
        acc = acc + t.astype(np.int64)
    # saturating int32 add (see docs/FRONTEND.md: `add` saturates, it does not wrap)
    return [np.clip(acc, INT32_MIN, INT32_MAX).astype(np.int32)]


def _op_clamp(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    lo, hi = int(node.attrs["min_val"]), int(node.attrs["max_val"])
    x = ins[0]
    return [np.clip(x.astype(np.int64), lo, hi).astype(x.dtype)]


def _op_rescale(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    a = node.attrs
    x = ins[0]
    izp = np.int64(a["input_zp"])
    ozp = np.int64(a["output_zp"])
    dr = bool(a.get("double_round", g.double_round))
    out_dtype = a["out_dtype"]

    acc = _sat32(x.astype(np.int64) - izp, f"rescale {node.name}")

    mult = np.asarray(a["multiplier"], dtype=np.int64)
    shift = np.asarray(a["shift"], dtype=np.int64)
    if a.get("per_channel", False):
        axis = int(a["channel_axis"])
        shape = [1] * acc.ndim
        shape[axis] = mult.size
        mult = mult.reshape(shape)
        shift = shift.reshape(shape)
    else:
        mult = mult.reshape(())
        shift = shift.reshape(())

    scaled = apply_scale_32(acc, mult, shift, dr)
    return [_rescale_tail(scaled, int(ozp), out_dtype)]


def _op_avg_pool2d(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    a = node.attrs
    x = ins[0]
    izp, ozp = int(a["input_zp"]), int(a["output_zp"])
    kh, kw = a["kernel"]
    sh, sw = a["stride"]
    if any(int(p) != 0 for p in a["pad"]):
        # TOSA divides each output position by its ACTUAL window count, so a
        # padded avg_pool2d has a position-dependent divisor.  KEA uses one
        # constant multiplier, so padded avg_pool2d is simply not expressible.
        raise KGraphError(
            f"avg_pool2d {node.name}: padding is not supported (see docs/FRONTEND.md); "
            "emit an explicit `pad` node only if you also accept TOSA's "
            "per-position divisor, which KEA does not model"
        )
    view, oh, ow = _im2col(x, int(kh), int(kw), int(sh), int(sw), 1, 1)
    acc = (view.astype(np.int64) - np.int64(izp)).sum(axis=(3, 4))
    acc32 = _sat32(acc, f"avg_pool2d {node.name}")
    dr = bool(a.get("double_round", g.double_round))
    scaled = apply_scale_32(acc32, int(a["multiplier"]), int(a["shift"]), dr)
    return [_rescale_tail(scaled, int(ozp), a["out_dtype"])]


def _op_global_avg_pool(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    a = node.attrs
    x = ins[0]
    izp, ozp = int(a["input_zp"]), int(a["output_zp"])
    acc = (x.astype(np.int64) - np.int64(izp)).sum(axis=(1, 2), keepdims=True)
    acc32 = _sat32(acc, f"global_avg_pool {node.name}")
    dr = bool(a.get("double_round", g.double_round))
    scaled = apply_scale_32(acc32, int(a["multiplier"]), int(a["shift"]), dr)
    return [_rescale_tail(scaled, int(ozp), a["out_dtype"])]


def _op_reshape(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    return [ins[0].reshape(tuple(int(s) for s in node.attrs["new_shape"]))]


def _op_transpose(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    return [np.ascontiguousarray(np.transpose(ins[0], tuple(int(p) for p in node.attrs["perms"])))]


def _op_concat(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    return [np.concatenate(ins, axis=int(node.attrs["axis"]))]


def _op_pad(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    padding = [(int(b), int(a_)) for b, a_ in node.attrs["padding"]]
    return [
        np.pad(ins[0], padding, mode="constant", constant_values=int(node.attrs["pad_const"]))
    ]


def _op_table(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    x, table = ins
    idx = x.astype(np.int32) + 128
    return [table[idx]]


def _op_softmax(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    x, exp_table = ins
    axis = int(node.attrs["axis"])
    xi = x.astype(np.int32)
    m = xi.max(axis=axis, keepdims=True)
    d = xi - m                                    # in [-255, 0]
    e = exp_table[(d + 255).astype(np.int32)]     # int32, in [0, 32768]
    s = _sat32(e.astype(np.int64).sum(axis=axis, keepdims=True), f"softmax {node.name}")
    num = e.astype(np.int64) << np.int64(15)      # <= 2**30
    out = idiv_trunc(num, s.astype(np.int64))
    return [_sat32(out, f"softmax {node.name}")]


def _op_layernorm(g: KGraph, node: Node, ins: List[np.ndarray]) -> List[np.ndarray]:
    x, gamma, beta = ins
    a = node.attrs
    axis = int(a["axis"])
    if axis != x.ndim - 1:
        raise KGraphError("layernorm: axis must be the last dimension")
    n = np.int64(x.shape[axis])
    k = np.int64(a["frac_bits"])
    eps_term = np.int64(a["eps_term"])

    xc = x.astype(np.int64) - np.int64(a["input_zp"])
    acc_a = xc.sum(axis=axis, keepdims=True)                 # int64
    acc_b = (xc * xc).sum(axis=axis, keepdims=True)          # int64
    d = n * acc_b - acc_a * acc_a                            # == N^2 * var, >= 0
    d = np.maximum(d, 0)
    den = isqrt(d + eps_term)
    den = np.maximum(den, np.int64(1))

    num = n * xc - acc_a
    t = idiv_trunc(num << k, den)
    t = np.clip(t, INT32_MIN, INT32_MAX)

    acc = t * gamma.astype(np.int64) + beta.astype(np.int64)
    return [np.clip(acc, INT32_MIN, INT32_MAX).astype(np.int32)]


_KERNELS = {
    "conv2d": _op_conv2d,
    "depthwise_conv2d": _op_depthwise_conv2d,
    "fully_connected": _op_fully_connected,
    "matmul": _op_matmul,
    "add": _op_add,
    "clamp": _op_clamp,
    "rescale": _op_rescale,
    "avg_pool2d": _op_avg_pool2d,
    "global_avg_pool": _op_global_avg_pool,
    "reshape": _op_reshape,
    "transpose": _op_transpose,
    "concat": _op_concat,
    "pad": _op_pad,
    "table": _op_table,
    "softmax": _op_softmax,
    "layernorm": _op_layernorm,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def execute(
    g: KGraph,
    feeds: Dict[str, np.ndarray],
    keep: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Run ``g`` on integer ``feeds``.  Returns the graph outputs (plus ``keep``).

    ``feeds`` values must already be quantized integer arrays of the declared
    dtype and shape.  Use :func:`quantize` at the boundary.
    """
    env: Dict[str, np.ndarray] = {}
    for name in g.inputs:
        t = g.tensor(name)
        if name not in feeds:
            raise KGraphError(f"missing feed for graph input {name!r}")
        v = np.asarray(feeds[name])
        if v.dtype != DTYPES[t.dtype]:
            raise KGraphError(f"feed {name!r}: dtype {v.dtype} != declared {t.dtype}")
        if list(v.shape) != t.shape:
            raise KGraphError(f"feed {name!r}: shape {list(v.shape)} != declared {t.shape}")
        env[name] = v
    for name, t in g.tensors.items():
        if t.kind == "const":
            env[name] = g.consts[t.const_key]

    for node in g.nodes:
        ins = [env[i] for i in node.inputs]
        try:
            outs = _KERNELS[node.op](g, node, ins)
        except KeyError as exc:
            if node.op not in _KERNELS:
                raise KGraphError(f"no reference kernel for op {node.op!r}") from None
            raise
        if len(outs) != len(node.outputs):
            raise KGraphError(f"node {node.name!r}: arity mismatch")
        for name, val in zip(node.outputs, outs):
            t = g.tensor(name)
            if val.dtype != DTYPES[t.dtype]:
                raise KGraphError(
                    f"node {node.name!r} produced {name!r} with dtype {val.dtype}, "
                    f"declared {t.dtype}"
                )
            if list(val.shape) != t.shape:
                raise KGraphError(
                    f"node {node.name!r} produced {name!r} with shape "
                    f"{list(val.shape)}, declared {t.shape}"
                )
            env[name] = val

    wanted = list(g.outputs) + list(keep or [])
    return {n: env[n] for n in wanted}


# ---------------------------------------------------------------------------
# Float shadow interpreter -- TESTING ONLY, not part of the contract
# ---------------------------------------------------------------------------


def execute_float(
    g: KGraph, feeds: Dict[str, np.ndarray], keep: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """Execute the same graph in float64 by dequantizing every tensor.

    Used by the test suite to bound the quantization error of :func:`execute`
    op by op.  Deliberately *not* used by the compiler.
    """
    env: Dict[str, np.ndarray] = {}
    for name in g.inputs:
        env[name] = np.asarray(feeds[name], dtype=np.float64)
    for name, t in g.tensors.items():
        if t.kind == "const":
            env[name] = dequantize(g.consts[t.const_key], t)

    for node in g.nodes:
        ins = [env[i] for i in node.inputs]
        outs = _float_kernel(g, node, ins)
        for name, val in zip(node.outputs, outs):
            env[name] = val
    wanted = list(g.outputs) + list(keep or [])
    return {n: env[n] for n in wanted}


def _float_kernel(g: KGraph, node: Node, ins) -> List[np.ndarray]:
    a = node.attrs
    op = node.op
    if op == "conv2d":
        x, w, b = ins
        xp = np.pad(
            x, ((0, 0), (a["pad"][0], a["pad"][1]), (a["pad"][2], a["pad"][3]), (0, 0))
        )
        view, oh, ow = _im2col(
            np.ascontiguousarray(xp), w.shape[1], w.shape[2],
            int(a["stride"][0]), int(a["stride"][1]),
            int(a.get("dilation", [1, 1])[0]), int(a.get("dilation", [1, 1])[1]),
        )
        n = view.shape[0]
        out = view.reshape(n * oh * ow, -1) @ w.reshape(w.shape[0], -1).T
        return [out.reshape(n, oh, ow, w.shape[0]) + b]
    if op == "depthwise_conv2d":
        x, w, b = ins
        kh, kw, c, m = w.shape
        xp = np.pad(
            x, ((0, 0), (a["pad"][0], a["pad"][1]), (a["pad"][2], a["pad"][3]), (0, 0))
        )
        view, oh, ow = _im2col(
            np.ascontiguousarray(xp), kh, kw, int(a["stride"][0]), int(a["stride"][1]),
            int(a.get("dilation", [1, 1])[0]), int(a.get("dilation", [1, 1])[1]),
        )
        out = np.einsum("nopijc,ijcm->nopcm", view, w, optimize=True)
        return [out.reshape(out.shape[0], oh, ow, c * m) + b]
    if op == "fully_connected":
        x, w, b = ins
        return [x @ w.T + b]
    if op == "matmul":
        return [np.matmul(ins[0], ins[1])]
    if op == "add":
        out = ins[0]
        for t in ins[1:]:
            out = out + t
        return [out]
    if op == "clamp":
        t = g.tensor(node.inputs[0])
        q = t.quant
        lo = (float(a["min_val"]) - q.zero_point[0]) * q.scale[0]
        hi = (float(a["max_val"]) - q.zero_point[0]) * q.scale[0]
        return [np.clip(ins[0], lo, hi)]
    if op == "rescale":
        return [ins[0]]  # a rescale is the identity in the real-number domain
    if op == "avg_pool2d":
        x = ins[0]
        xp = np.pad(
            x, ((0, 0), (a["pad"][0], a["pad"][1]), (a["pad"][2], a["pad"][3]), (0, 0))
        )
        view, oh, ow = _im2col(
            np.ascontiguousarray(xp), int(a["kernel"][0]), int(a["kernel"][1]),
            int(a["stride"][0]), int(a["stride"][1]), 1, 1,
        )
        return [view.sum(axis=(3, 4)) / (a["kernel"][0] * a["kernel"][1])]
    if op == "global_avg_pool":
        return [ins[0].mean(axis=(1, 2), keepdims=True)]
    if op == "reshape":
        return [ins[0].reshape(tuple(int(s) for s in a["new_shape"]))]
    if op == "transpose":
        return [np.transpose(ins[0], tuple(int(p) for p in a["perms"]))]
    if op == "concat":
        return [np.concatenate(ins, axis=int(a["axis"]))]
    if op == "pad":
        return [np.pad(ins[0], [(int(b), int(c)) for b, c in a["padding"]])]
    if op == "table":
        # the table is an arbitrary map; evaluate it via its own quantization
        xt = g.tensor(node.inputs[0])
        ot = g.tensor(node.outputs[0])
        tq = g.consts[g.tensor(node.inputs[1]).const_key]
        q = np.clip(np.rint(ins[0] / xt.quant.scale[0]) + xt.quant.zero_point[0], -128, 127)
        return [dequantize(tq[(q + 128).astype(np.int32)], ot)]
    if op == "softmax":
        x = ins[0]
        e = np.exp(x - x.max(axis=int(a["axis"]), keepdims=True))
        return [e / e.sum(axis=int(a["axis"]), keepdims=True)]
    if op == "layernorm":
        x, gamma, beta = ins
        axis = int(a["axis"])
        mu = x.mean(axis=axis, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=axis, keepdims=True)
        return [(x - mu) / np.sqrt(var + float(a["eps"])) * gamma + beta]
    raise KGraphError(f"no float kernel for op {op!r}")
