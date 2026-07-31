"""Two backends behind one op API, so calibration and emission cannot drift.

A network is written *once* as a function ``build(net, x, params)`` against the
:class:`NetOps` interface.  Running it with a :class:`TorchTracer` executes the
float model and records the dynamic range of every tensor that will become a
graph tensor.  Running the *same function* with a :class:`KGraphBuilder` emits
the quantized ``.kgraph.json``.  Because both walks name tensors identically,
a calibration point can never end up attached to the wrong tensor.

All activations are **NHWC** and all conv weights are converted to the TOSA
layouts (``OHWI`` for conv2d, ``HWCM`` for depthwise) inside the builder; the
network description passes weights in torch's native ``OIHW``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .apply_scale import SHIFT_MAX
from .ir import KGraph, KGraphError, Node, QuantParams, Tensor
from .quant import (
    QMAX,
    QMIN,
    Observer,
    act_qparams_from_range,
    make_observer,
    quantize_bias,
    quantize_weights_per_channel,
    requant_params,
    safe_quantize_multiplier,
    weight_qparams,
)

__all__ = ["NetOps", "TorchTracer", "KGraphBuilder", "Calibration", "SOFTMAX_FRAC_BITS",
           "LAYERNORM_FRAC_BITS"]

#: Softmax emits int32 probabilities in Q15 (values 0..32768).
SOFTMAX_FRAC_BITS = 15
#: Layernorm's normalized value is computed in Q15 before gamma/beta.
LAYERNORM_FRAC_BITS = 15


# ---------------------------------------------------------------------------
# Calibration record
# ---------------------------------------------------------------------------


@dataclass
class Calibration:
    """Observed ``[lo, hi]`` per named tensor, plus the observer used."""

    observer: str = "minmax"
    percentile: float = 99.99
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def get(self, name: str) -> Tuple[float, float]:
        try:
            return self.ranges[name]
        except KeyError:
            raise KGraphError(
                f"no calibration range for tensor {name!r}; the calibrate and "
                "build passes disagree about tensor names"
            ) from None

    def qparams(self, name: str) -> QuantParams:
        lo, hi = self.get(name)
        return act_qparams_from_range(lo, hi)


# ---------------------------------------------------------------------------
# Op interface
# ---------------------------------------------------------------------------


class NetOps:
    """The op set a network description may use.  Both backends implement it."""

    def conv2d(self, name, x, w, b, stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1),
               groups=1, act=None): ...
    def linear(self, name, x, w, b, act=None): ...
    def matmul(self, name, a, b, extra_scale=1.0): ...
    def add(self, name, a, b, act=None): ...
    def global_avg_pool(self, name, x): ...
    def avg_pool2d(self, name, x, kernel, stride, pad=(0, 0, 0, 0)): ...
    def reshape(self, name, x, shape): ...
    def transpose(self, name, x, perms): ...
    def softmax(self, name, x, axis=-1): ...
    def layernorm(self, name, x, gamma, beta, eps=1e-5): ...
    def gelu(self, name, x): ...
    def clamp_relu6(self, name, x): ...


# ---------------------------------------------------------------------------
# Float tracer (torch)
# ---------------------------------------------------------------------------


class TorchTracer(NetOps):
    """Executes the network in float32 torch and records activation ranges.

    Handles are ``torch.Tensor`` in NHWC.
    """

    def __init__(self, observers: Dict[str, Observer], factory=None, record: bool = True) -> None:
        import torch  # local import: the IR/reference layer must not need torch

        self._torch = torch
        self.observers = observers
        self.record = record
        #: ``(name) -> Observer``, called the first time a tensor is seen.
        self.factory = factory or (lambda name: make_observer("minmax"))

    # -- bookkeeping --------------------------------------------------------

    def _obs(self, name: str, t) -> Any:
        if self.record:
            o = self.observers.get(name)
            if o is None:
                o = self.observers[name] = self.factory(name)
            o.observe(t.detach().cpu().numpy())
        return t

    # -- ops ----------------------------------------------------------------

    def conv2d(self, name, x, w, b, stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1),
               groups=1, act=None):
        F = self._torch.nn.functional
        wt = self._torch.as_tensor(np.ascontiguousarray(w), dtype=self._torch.float32)
        bt = self._torch.as_tensor(np.ascontiguousarray(b), dtype=self._torch.float32)
        xt = x.permute(0, 3, 1, 2)
        t, bo, l, r = pad
        xt = F.pad(xt, (int(l), int(r), int(t), int(bo)))
        y = F.conv2d(xt, wt, bt, stride=tuple(int(s) for s in stride),
                     dilation=tuple(int(d) for d in dilation), groups=int(groups))
        y = y.permute(0, 2, 3, 1).contiguous()
        y = _apply_act_torch(self._torch, y, act)
        return self._obs(name, y)

    def linear(self, name, x, w, b, act=None):
        wt = self._torch.as_tensor(np.ascontiguousarray(w), dtype=self._torch.float32)
        bt = self._torch.as_tensor(np.ascontiguousarray(b), dtype=self._torch.float32)
        y = self._torch.nn.functional.linear(x, wt, bt)
        y = _apply_act_torch(self._torch, y, act)
        return self._obs(name, y)

    def matmul(self, name, a, b, extra_scale=1.0):
        y = self._torch.matmul(a, b) * float(extra_scale)
        return self._obs(name, y)

    def add(self, name, a, b, act=None):
        y = _apply_act_torch(self._torch, a + b, act)
        return self._obs(name, y)

    def global_avg_pool(self, name, x):
        return self._obs(name, x.mean(dim=(1, 2), keepdim=True))

    def avg_pool2d(self, name, x, kernel, stride, pad=(0, 0, 0, 0)):
        F = self._torch.nn.functional
        t, bo, l, r = pad
        xt = F.pad(x.permute(0, 3, 1, 2), (int(l), int(r), int(t), int(bo)))
        y = F.avg_pool2d(xt, tuple(int(k) for k in kernel), tuple(int(s) for s in stride),
                         count_include_pad=True)
        return self._obs(name, y.permute(0, 2, 3, 1).contiguous())

    def reshape(self, name, x, shape):
        return x.reshape(tuple(int(s) for s in shape))

    def transpose(self, name, x, perms):
        return x.permute(tuple(int(p) for p in perms)).contiguous()

    def softmax(self, name, x, axis=-1):
        return self._torch.softmax(x, dim=int(axis))

    def layernorm(self, name, x, gamma, beta, eps=1e-5):
        g = self._torch.as_tensor(np.ascontiguousarray(gamma), dtype=self._torch.float32)
        bt = self._torch.as_tensor(np.ascontiguousarray(beta), dtype=self._torch.float32)
        y = self._torch.nn.functional.layer_norm(x, (x.shape[-1],), g, bt, float(eps))
        return self._obs(name, y)

    def gelu(self, name, x):
        return self._obs(name, self._torch.nn.functional.gelu(x))

    def clamp_relu6(self, name, x):
        return self._obs(name, self._torch.clamp(x, 0.0, 6.0))

    def constant(self, name, array):
        # A constant's range is known exactly; it is never calibrated.
        return self._torch.as_tensor(np.ascontiguousarray(array), dtype=self._torch.float32)


def _apply_act_torch(torch, y, act):
    if act is None:
        return y
    if act == "relu":
        return torch.relu(y)
    if act == "relu6":
        return torch.clamp(y, 0.0, 6.0)
    raise ValueError(f"unknown activation {act!r}")


# ---------------------------------------------------------------------------
# Quantized graph emitter
# ---------------------------------------------------------------------------


@dataclass
class H:
    """Builder handle: a graph tensor name plus its (cached) shape."""

    name: str
    shape: Tuple[int, ...]


class KGraphBuilder(NetOps):
    """Emits a :class:`~kea_frontend.ir.KGraph` from a network description."""

    def __init__(self, graph_name: str, calib: Calibration, double_round: bool = True) -> None:
        self.g = KGraph(name=graph_name, double_round=double_round)
        self.calib = calib
        self._uid = 0

    # -- helpers ------------------------------------------------------------

    def _u(self, stem: str) -> str:
        self._uid += 1
        return f"{stem}#{self._uid}"

    def _act(self, name: str, shape, qp: Optional[QuantParams] = None,
             dtype: str = "int8", layout: str = "*") -> H:
        shape = [int(s) for s in shape]
        if qp is None:
            qp = self.calib.qparams(name)
        self.g.add_tensor(Tensor(name, dtype, shape, layout, qp, "activation"))
        return H(name, tuple(shape))

    def _qp(self, h: H) -> QuantParams:
        return self.g.tensor(h.name).quant

    def input(self, name: str, shape: Sequence[int], layout: str = "NHWC",
              qp: Optional[QuantParams] = None) -> H:
        if qp is None:
            qp = self.calib.qparams(name)
        self.g.add_tensor(Tensor(name, "int8", [int(s) for s in shape], layout, qp, "input"))
        self.g.inputs.append(name)
        return H(name, tuple(int(s) for s in shape))

    def mark_output(self, h: H) -> None:
        self.g.outputs.append(h.name)

    def constant(self, name: str, array) -> H:
        """An int8 constant activation-like tensor (e.g. a positional embedding).

        Its range is exact, so it is quantized from its own min/max rather than
        calibrated.
        """
        a = np.asarray(array, dtype=np.float64)
        qp = act_qparams_from_range(float(a.min()), float(a.max()))
        q = np.clip(np.rint(a / qp.scale[0]) + qp.zero_point[0], QMIN, QMAX).astype(np.int8)
        self.g.add_const(name, q, "*", qp)
        return H(name, tuple(int(s) for s in a.shape))

    # -- primitive: rescale --------------------------------------------------

    def _rescale(self, out_name: str, src: H, out_qp: QuantParams, out_dtype: str,
                 layout: str = "*", channel_axis: Optional[int] = None) -> H:
        """int32/int8 -> int8/int32 requantization derived from tensor metadata."""
        sq = self._qp(src)
        if sq.kind == "per_channel":
            axis = sq.axis if channel_axis is None else channel_axis
            real = sq.scale_array() / out_qp.scale[0]
            mult, shift = safe_quantize_multiplier(real)
            attrs = {
                "input_zp": int(sq.zero_point[0]),
                "output_zp": int(out_qp.zero_point[0]),
                "multiplier": mult.tolist(),
                "shift": shift.tolist(),
                "per_channel": True,
                "channel_axis": int(axis),
                "out_dtype": out_dtype,
                "double_round": bool(self.g.double_round),
            }
        else:
            mult, shift = requant_params(sq.scale[0], None, out_qp.scale[0])
            attrs = {
                "input_zp": int(sq.zero_point[0]),
                "output_zp": int(out_qp.zero_point[0]),
                "multiplier": [int(mult[0])],
                "shift": [int(shift[0])],
                "per_channel": False,
                "channel_axis": 0,
                "out_dtype": out_dtype,
                "double_round": bool(self.g.double_round),
            }
        out = self._act(out_name, src.shape, out_qp, out_dtype, layout)
        self.g.add_node(Node("rescale", self._u("rescale"), [src.name], [out_name], attrs))
        return out

    # -- ops ----------------------------------------------------------------

    def conv2d(self, name, x: H, w, b, stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1),
               groups=1, act=None) -> H:
        w = np.asarray(w, dtype=np.float64)          # OIHW
        b = np.asarray(b, dtype=np.float64)
        oc, icg, kh, kw = w.shape
        in_c = x.shape[3]
        xq = self._qp(x)
        depthwise = groups != 1

        if depthwise:
            if groups != in_c or oc % in_c != 0:
                raise KGraphError(
                    f"{name}: only depthwise (groups == in_channels) grouped conv is supported"
                )
            m = oc // in_c
            # torch [C*M, 1, KH, KW] -> HWCM
            # per-output-channel scales live on the flattened C*M axis
            wq_flat = weight_qparams(w, out_axis=0)
            wscales = wq_flat.scale_array()
            wq_i8_oihw = quantize_weights_per_channel(w, wq_flat)
            w_i8 = wq_i8_oihw.reshape(in_c, m, kh, kw).transpose(2, 3, 0, 1)
            w_i8 = np.ascontiguousarray(w_i8)
            # declare the const's per-channel axis in HWCM terms: axis 2 (C) only
            # works when M == 1, which is the only case MobileNetV2 uses.
            if m != 1:
                raise KGraphError(f"{name}: channel_multiplier > 1 not supported")
            wqp = QuantParams.per_channel(wscales.tolist(), 2)
            self.g.add_const(f"{name}.w", w_i8, "HWCM", wqp)
            op = "depthwise_conv2d"
        else:
            wq = weight_qparams(w, out_axis=0)
            wscales = wq.scale_array()
            w_i8 = np.ascontiguousarray(
                quantize_weights_per_channel(w, wq).transpose(0, 2, 3, 1)
            )  # OIHW -> OHWI
            wqp = QuantParams.per_channel(wscales.tolist(), 0)
            wt = self.g.add_const(f"{name}.w", w_i8, "OHWI", wqp)
            op = "conv2d"

        bias_q = quantize_bias(b, xq.scale[0], wscales)
        acc_scales = xq.scale[0] * wscales
        self.g.add_const(
            f"{name}.b", bias_q, "C", QuantParams.per_channel(acc_scales.tolist(), 0)
        )

        n, h, wdt = x.shape[0], x.shape[1], x.shape[2]
        t, bo, l, r = pad
        oh = (h + t + bo - (kh - 1) * dilation[0] - 1) // stride[0] + 1
        ow = (wdt + l + r - (kw - 1) * dilation[1] - 1) // stride[1] + 1
        acc_name = f"{name}.acc"
        acc = self._act(
            acc_name, [n, oh, ow, oc],
            QuantParams.per_channel(acc_scales.tolist(), 3), "int32", "NHWC",
        )
        self.g.add_node(Node(op, self._u(op), [x.name, f"{name}.w", f"{name}.b"], [acc_name], {
            "pad": [int(p) for p in pad],
            "stride": [int(s) for s in stride],
            "dilation": [int(d) for d in dilation],
            "input_zp": int(xq.zero_point[0]),
            "weight_zp": 0,
        }))
        return self._finish(name, acc, act, "NHWC")

    def linear(self, name, x: H, w, b, act=None) -> H:
        w = np.asarray(w, dtype=np.float64)          # [OC, IC]
        b = np.asarray(b, dtype=np.float64)
        oc = w.shape[0]
        xq = self._qp(x)
        wq = weight_qparams(w, out_axis=0)
        wscales = wq.scale_array()
        w_i8 = quantize_weights_per_channel(w, wq)
        self.g.add_const(f"{name}.w", w_i8, "OI", QuantParams.per_channel(wscales.tolist(), 0))
        bias_q = quantize_bias(b, xq.scale[0], wscales)
        acc_scales = xq.scale[0] * wscales
        self.g.add_const(f"{name}.b", bias_q, "C",
                         QuantParams.per_channel(acc_scales.tolist(), 0))

        out_shape = list(x.shape[:-1]) + [oc]
        acc_name = f"{name}.acc"
        acc = self._act(acc_name, out_shape,
                        QuantParams.per_channel(acc_scales.tolist(), len(out_shape) - 1),
                        "int32", "*")
        self.g.add_node(Node("fully_connected", self._u("fc"),
                             [x.name, f"{name}.w", f"{name}.b"], [acc_name],
                             {"input_zp": int(xq.zero_point[0]), "weight_zp": 0}))
        return self._finish(name, acc, act, "*")

    def matmul(self, name, a: H, b: H, extra_scale=1.0) -> H:
        aq, bq = self._qp(a), self._qp(b)
        out_shape = list(a.shape[:-1]) + [b.shape[-1]]
        acc_name = f"{name}.acc"
        # `extra_scale` (e.g. the 1/sqrt(head_dim) of attention) is folded into
        # the accumulator's scale, so it costs nothing at inference.
        acc_scale = aq.scale[0] * bq.scale[0] * float(extra_scale)
        acc = self._act(acc_name, out_shape, QuantParams.per_tensor(acc_scale, 0), "int32", "*")
        self.g.add_node(Node("matmul", self._u("matmul"), [a.name, b.name], [acc_name],
                             {"a_zp": int(aq.zero_point[0]), "b_zp": int(bq.zero_point[0])}))
        return self._finish(name, acc, None, "*")

    def add(self, name, a: H, b: H, act=None) -> H:
        aq, bq = self._qp(a), self._qp(b)
        # Common accumulator scale, 8 fractional bits below the finer input.
        common = min(aq.scale[0], bq.scale[0]) / 256.0
        cqp = QuantParams.per_tensor(common, 0)
        ra = self._rescale(f"{name}.a32", a, cqp, "int32")
        rb = self._rescale(f"{name}.b32", b, cqp, "int32")
        acc_name = f"{name}.acc"
        acc = self._act(acc_name, a.shape, cqp, "int32", "*")
        self.g.add_node(Node("add", self._u("add"), [ra.name, rb.name], [acc_name], {}))
        return self._finish(name, acc, act, "*")

    def global_avg_pool(self, name, x: H) -> H:
        xq = self._qp(x)
        n, h, w, c = x.shape
        out_qp = self.calib.qparams(name)
        mult, shift = safe_quantize_multiplier(
            np.array([xq.scale[0] / (out_qp.scale[0] * h * w)])
        )
        out = self._act(name, [n, 1, 1, c], out_qp, "int8", "NHWC")
        self.g.add_node(Node("global_avg_pool", self._u("gap"), [x.name], [name], {
            "input_zp": int(xq.zero_point[0]),
            "output_zp": int(out_qp.zero_point[0]),
            "multiplier": int(mult[0]),
            "shift": int(shift[0]),
            "out_dtype": "int8",
            "double_round": bool(self.g.double_round),
        }))
        return out

    def avg_pool2d(self, name, x: H, kernel, stride, pad=(0, 0, 0, 0)) -> H:
        xq = self._qp(x)
        kh, kw = int(kernel[0]), int(kernel[1])
        n, h, w, c = x.shape
        t, bo, l, r = pad
        oh = (h + t + bo - kh) // int(stride[0]) + 1
        ow = (w + l + r - kw) // int(stride[1]) + 1
        out_qp = self.calib.qparams(name)
        mult, shift = safe_quantize_multiplier(
            np.array([xq.scale[0] / (out_qp.scale[0] * kh * kw)])
        )
        out = self._act(name, [n, oh, ow, c], out_qp, "int8", "NHWC")
        self.g.add_node(Node("avg_pool2d", self._u("avgpool"), [x.name], [name], {
            "kernel": [kh, kw],
            "stride": [int(s) for s in stride],
            "pad": [int(p) for p in pad],
            "input_zp": int(xq.zero_point[0]),
            "output_zp": int(out_qp.zero_point[0]),
            "multiplier": int(mult[0]),
            "shift": int(shift[0]),
            "out_dtype": "int8",
            "double_round": bool(self.g.double_round),
        }))
        return out

    def reshape(self, name, x: H, shape) -> H:
        shape = [int(s) for s in shape]
        out = self._act(name, shape, self._qp(x), self.g.tensor(x.name).dtype, "*")
        self.g.add_node(Node("reshape", self._u("reshape"), [x.name], [name],
                             {"new_shape": shape}))
        return out

    def transpose(self, name, x: H, perms) -> H:
        perms = [int(p) for p in perms]
        shape = [x.shape[p] for p in perms]
        out = self._act(name, shape, self._qp(x), self.g.tensor(x.name).dtype, "*")
        self.g.add_node(Node("transpose", self._u("transpose"), [x.name], [name],
                             {"perms": perms}))
        return out

    def softmax(self, name, x: H, axis=-1) -> H:
        xq = self._qp(x)
        axis = int(axis) % len(x.shape)
        # exp_table[i] = round(exp((i - 255) * s_in) * 2**15), i in [0, 255]
        idx = np.arange(256, dtype=np.float64) - 255.0
        table = np.rint(np.exp(idx * xq.scale[0]) * float(1 << SOFTMAX_FRAC_BITS))
        table = np.clip(table, 0, 1 << SOFTMAX_FRAC_BITS).astype(np.int32)
        self.g.add_const(f"{name}.exp", table, "C", QuantParams.per_tensor(2.0 ** -15, 0))
        acc_name = f"{name}.acc"
        acc_qp = QuantParams.per_tensor(2.0 ** -SOFTMAX_FRAC_BITS, 0)
        acc = self._act(acc_name, x.shape, acc_qp, "int32", "*")
        self.g.add_node(Node("softmax", self._u("softmax"), [x.name, f"{name}.exp"],
                             [acc_name], {"axis": axis, "frac_bits": SOFTMAX_FRAC_BITS}))
        # probabilities always land on the canonical [0,1] int8 grid
        out_qp = QuantParams.per_tensor(1.0 / 256.0, -128)
        return self._rescale(name, acc, out_qp, "int8")

    def layernorm(self, name, x: H, gamma, beta, eps=1e-5) -> H:
        gamma = np.asarray(gamma, dtype=np.float64)
        beta = np.asarray(beta, dtype=np.float64)
        xq = self._qp(x)
        n = x.shape[-1]
        k = LAYERNORM_FRAC_BITS
        s_g = max(float(np.abs(gamma).max()) / 127.0, 1e-9)
        g_i8 = np.clip(np.rint(gamma / s_g), -127, 127).astype(np.int8)
        acc_scale = (2.0 ** -k) * s_g
        beta_q = np.clip(np.rint(beta / acc_scale), -(2**31), 2**31 - 1).astype(np.int32)
        # den = isqrt(D + eps_term) with D = N^2 * var measured in units of s_in^2
        eps_term = int(round(float(n) ** 2 * float(eps) / (xq.scale[0] ** 2)))

        self.g.add_const(f"{name}.gamma", g_i8, "C", QuantParams.per_tensor(s_g, 0))
        self.g.add_const(f"{name}.beta", beta_q, "C", QuantParams.per_tensor(acc_scale, 0))
        acc_name = f"{name}.acc"
        acc = self._act(acc_name, x.shape, QuantParams.per_tensor(acc_scale, 0), "int32", "*")
        self.g.add_node(Node("layernorm", self._u("layernorm"),
                             [x.name, f"{name}.gamma", f"{name}.beta"], [acc_name], {
                                 "axis": len(x.shape) - 1,
                                 "input_zp": int(xq.zero_point[0]),
                                 "frac_bits": k,
                                 "eps_term": eps_term,
                                 "eps": float(eps),
                             }))
        return self._finish(name, acc, None, "*")

    def gelu(self, name, x: H) -> H:
        return self._table(name, x, lambda v: 0.5 * v * (1.0 + _erf(v / math.sqrt(2.0))))

    def clamp_relu6(self, name, x: H) -> H:
        # A standalone relu6 on an already-int8 tensor: same scale in and out,
        # implemented as a pure clamp.
        xq = self._qp(x)
        lo = int(xq.zero_point[0])
        hi = int(np.clip(round(6.0 / xq.scale[0]) + xq.zero_point[0], QMIN, QMAX))
        out = self._act(name, x.shape, xq, "int8", "*")
        self.g.add_node(Node("clamp", self._u("clamp"), [x.name], [name],
                             {"min_val": lo, "max_val": hi}))
        return out

    def _table(self, name, x: H, fn) -> H:
        xq = self._qp(x)
        out_qp = self.calib.qparams(name)
        q = np.arange(-128, 128, dtype=np.float64)
        real_in = (q - xq.zero_point[0]) * xq.scale[0]
        real_out = fn(real_in)
        tab = np.clip(
            np.rint(real_out / out_qp.scale[0]) + out_qp.zero_point[0], QMIN, QMAX
        ).astype(np.int8)
        self.g.add_const(f"{name}.tab", tab, "C", out_qp)
        out = self._act(name, x.shape, out_qp, "int8", "*")
        self.g.add_node(Node("table", self._u("table"), [x.name, f"{name}.tab"], [name], {}))
        return out

    # -- rescale + activation tail ------------------------------------------

    def _finish(self, name: str, acc: H, act: Optional[str], layout: str) -> H:
        """int32 accumulator -> int8, then the fused activation clamp."""
        out_qp = self.calib.qparams(name)
        if act is None:
            return self._rescale(name, acc, out_qp, "int8", layout)
        pre = self._rescale(f"{name}.rq", acc, out_qp, "int8", layout)
        lo = int(out_qp.zero_point[0])
        if act == "relu":
            hi = QMAX
        elif act == "relu6":
            hi = int(np.clip(round(6.0 / out_qp.scale[0]) + out_qp.zero_point[0], QMIN, QMAX))
        else:
            raise ValueError(f"unknown activation {act!r}")
        out = self._act(name, acc.shape, out_qp, "int8", layout)
        self.g.add_node(Node("clamp", self._u("clamp"), [pre.name], [name],
                             {"min_val": lo, "max_val": hi}))
        return out


def _erf(x: np.ndarray) -> np.ndarray:
    from math import erf

    return np.vectorize(erf)(x)
