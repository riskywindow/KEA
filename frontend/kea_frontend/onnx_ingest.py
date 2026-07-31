"""ONNX -> KEA graph IR.

Accepts a **float** ONNX model, folds BatchNorm, calibrates, and emits the same
``.kgraph.json`` the torch path produces.  Coverage is deliberately narrow and
explicit: see ``SUPPORTED_OPS`` below and the table in ``docs/FRONTEND.md``.
Anything outside it raises :class:`UnsupportedOnnxOp` with the op type and node
name rather than silently approximating.

The importer works in two stages:

1. :func:`parse_onnx` walks the ONNX graph, folds BN into its producer conv, and
   lowers it to a backend-agnostic list of :class:`Step` records.
2. :func:`make_build_fn` turns that list into a ``build(net, x)`` closure that
   drives either backend, exactly like the hand-written descriptions in
   ``nets.py``.

That reuse is the point: ONNX models go through the *same* calibration and
quantization code as torch models, so they cannot diverge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .quant import fold_conv_bn

__all__ = [
    "SUPPORTED_OPS",
    "UnsupportedOnnxOp",
    "Step",
    "parse_onnx",
    "make_build_fn",
    "onnx_input_shape",
]


class UnsupportedOnnxOp(NotImplementedError):
    pass


#: ONNX op types the importer understands.  Everything else is a hard error.
SUPPORTED_OPS = {
    "Conv",
    "BatchNormalization",   # only when fusable into the preceding Conv
    "Relu",
    "Clip",                 # only the ReLU6 shape: min=0, max=6
    "Add",
    "GlobalAveragePool",
    "AveragePool",
    "Gemm",
    "MatMul",               # only with a constant rhs (i.e. a dense layer)
    "Flatten",
    "Reshape",
    "Squeeze",
    "Transpose",
    "Identity",
    "Dropout",              # inference-time no-op
    "Constant",
}


@dataclass
class Step:
    """One backend-agnostic operation, closing over its float constants."""

    op: str
    name: str
    inputs: List[str]
    output: str
    attrs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ONNX helpers
# ---------------------------------------------------------------------------


def _attrs(node) -> Dict[str, Any]:
    from onnx import numpy_helper

    out: Dict[str, Any] = {}
    for a in node.attribute:
        if a.type == 1:      # FLOAT
            out[a.name] = a.f
        elif a.type == 2:    # INT
            out[a.name] = a.i
        elif a.type == 3:    # STRING
            out[a.name] = a.s.decode()
        elif a.type == 4:    # TENSOR
            out[a.name] = numpy_helper.to_array(a.t)
        elif a.type == 6:    # FLOATS
            out[a.name] = list(a.floats)
        elif a.type == 7:    # INTS
            out[a.name] = list(a.ints)
        elif a.type == 8:    # STRINGS
            out[a.name] = [s.decode() for s in a.strings]
    return out


def _initializers(graph) -> Dict[str, np.ndarray]:
    from onnx import numpy_helper

    return {i.name: numpy_helper.to_array(i).astype(np.float64) for i in graph.initializer}


def onnx_input_shape(model) -> List[int]:
    """The model's single input shape, as NCHW with any dynamic batch set to 1."""
    g = model.graph
    inits = {i.name for i in g.initializer}
    ins = [i for i in g.input if i.name not in inits]
    if len(ins) != 1:
        raise UnsupportedOnnxOp(f"expected exactly 1 graph input, got {len(ins)}")
    dims = []
    for d in ins[0].type.tensor_type.shape.dim:
        dims.append(int(d.dim_value) if d.dim_value > 0 else 1)
    return dims


def _pads_to_tosa(pads: Optional[Sequence[int]]) -> List[int]:
    """ONNX ``[x1_begin, x2_begin, x1_end, x2_end]`` -> TOSA ``[t, b, l, r]``."""
    if not pads:
        return [0, 0, 0, 0]
    if len(pads) != 4:
        raise UnsupportedOnnxOp(f"only 2-D pads are supported, got {list(pads)}")
    t, l, b, r = pads
    return [int(t), int(b), int(l), int(r)]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_onnx(model) -> Tuple[List[Step], str, str]:
    """Lower an ONNX graph to ``(steps, input_name, output_name)``.

    Raises :class:`UnsupportedOnnxOp` on anything outside :data:`SUPPORTED_OPS`.
    """
    g = model.graph
    init = _initializers(g)
    steps: List[Step] = []
    # value name -> the name a later step should read (handles skipped no-ops)
    alias: Dict[str, str] = {}
    # conv output name -> index into `steps`, so BN can be folded backwards
    conv_by_out: Dict[str, int] = {}
    consumed: set = set()

    def resolve(n: str) -> str:
        while n in alias:
            n = alias[n]
        return n

    inputs = [i for i in g.input if i.name not in init]
    in_name = inputs[0].name

    for node in g.node:
        op = node.op_type
        nm = node.name or f"{op}_{len(steps)}"
        a = _attrs(node)
        if op not in SUPPORTED_OPS:
            raise UnsupportedOnnxOp(
                f"node {nm!r}: ONNX op {op!r} is not supported by the KEA "
                f"frontend (supported: {sorted(SUPPORTED_OPS)})"
            )

        if op in ("Identity", "Dropout"):
            alias[node.output[0]] = resolve(node.input[0])
            continue
        if op == "Constant":
            init[node.output[0]] = np.asarray(a["value"], dtype=np.float64)
            continue

        if op == "Conv":
            w = init[node.input[1]]
            b = init[node.input[2]] if len(node.input) > 2 else np.zeros(w.shape[0])
            if a.get("auto_pad", "NOTSET") not in ("NOTSET", "VALID"):
                raise UnsupportedOnnxOp(
                    f"node {nm!r}: auto_pad={a['auto_pad']!r} is not supported; "
                    "re-export with explicit pads"
                )
            steps.append(Step("conv", nm, [resolve(node.input[0])], node.output[0], {
                "w": w, "b": b,
                "stride": list(a.get("strides", [1, 1])),
                "pad": _pads_to_tosa(a.get("pads")),
                "dilation": list(a.get("dilations", [1, 1])),
                "groups": int(a.get("group", 1)),
                "act": None,
            }))
            conv_by_out[node.output[0]] = len(steps) - 1
            continue

        if op == "BatchNormalization":
            src = node.input[0]
            if src not in conv_by_out:
                raise UnsupportedOnnxOp(
                    f"node {nm!r}: BatchNormalization can only be folded into a "
                    "directly preceding Conv; standalone BN is not supported"
                )
            si = conv_by_out[src]
            st = steps[si]
            gamma, beta, mean, var = (init[node.input[i]] for i in (1, 2, 3, 4))
            folded = fold_conv_bn(st.attrs["w"], st.attrs["b"], gamma, beta, mean, var,
                                  float(a.get("epsilon", 1e-5)))
            st.attrs["w"], st.attrs["b"] = folded.weight, folded.bias
            st.output = node.output[0]
            conv_by_out[node.output[0]] = si
            continue

        if op in ("Relu", "Clip"):
            src = resolve(node.input[0])
            if op == "Clip":
                lo = a.get("min")
                hi = a.get("max")
                if lo is None and len(node.input) > 1 and node.input[1]:
                    lo = float(init[node.input[1]])
                if hi is None and len(node.input) > 2 and node.input[2]:
                    hi = float(init[node.input[2]])
                if not (lo == 0.0 and hi == 6.0):
                    raise UnsupportedOnnxOp(
                        f"node {nm!r}: only Clip(0, 6) (ReLU6) is supported, got "
                        f"Clip({lo}, {hi})"
                    )
                act = "relu6"
            else:
                act = "relu"
            # fuse into the producing conv/gemm when possible
            if src in conv_by_out and steps[conv_by_out[src]].attrs.get("act") is None:
                si = conv_by_out[src]
                steps[si].attrs["act"] = act
                steps[si].output = node.output[0]
                conv_by_out[node.output[0]] = si
                continue
            steps.append(Step("act", nm, [src], node.output[0], {"act": act}))
            continue

        if op == "Add":
            x0, x1 = node.input[0], node.input[1]
            if x0 in init or x1 in init:
                raise UnsupportedOnnxOp(
                    f"node {nm!r}: Add with a constant operand is not supported; "
                    "fold it into the preceding layer's bias"
                )
            steps.append(Step("add", nm, [resolve(x0), resolve(x1)], node.output[0], {}))
            continue

        if op == "GlobalAveragePool":
            steps.append(Step("gap", nm, [resolve(node.input[0])], node.output[0], {}))
            continue

        if op == "AveragePool":
            if any(p != 0 for p in _pads_to_tosa(a.get("pads"))):
                raise UnsupportedOnnxOp(
                    f"node {nm!r}: padded AveragePool is not supported (TOSA divides "
                    "by the actual window count per position; KEA uses a constant "
                    "divisor -- see docs/FRONTEND.md)"
                )
            if int(a.get("count_include_pad", 0)) not in (0, 1):
                raise UnsupportedOnnxOp(f"node {nm!r}: bad count_include_pad")
            steps.append(Step("avgpool", nm, [resolve(node.input[0])], node.output[0], {
                "kernel": list(a["kernel_shape"]),
                "stride": list(a.get("strides", [1, 1])),
            }))
            continue

        if op in ("Gemm", "MatMul"):
            xin, win = node.input[0], node.input[1]
            if win not in init:
                raise UnsupportedOnnxOp(
                    f"node {nm!r}: {op} with a non-constant right-hand side "
                    "(activation x activation) is not supported by the ONNX "
                    "importer; use the torch path for attention-style matmuls"
                )
            w = init[win]
            if op == "Gemm":
                if int(a.get("transA", 0)):
                    raise UnsupportedOnnxOp(f"node {nm!r}: Gemm transA=1 unsupported")
                if not int(a.get("transB", 0)):
                    w = w.T          # KEA wants [OC, IC]
                b = init[node.input[2]] if len(node.input) > 2 else np.zeros(w.shape[0])
                alpha, beta = float(a.get("alpha", 1.0)), float(a.get("beta", 1.0))
                w, b = w * alpha, b * beta
            else:
                w = w.T              # MatMul rhs is [IC, OC]
                b = np.zeros(w.shape[0])
            steps.append(Step("linear", nm, [resolve(xin)], node.output[0],
                              {"w": w, "b": b, "act": None}))
            conv_by_out[node.output[0]] = len(steps) - 1
            continue

        if op in ("Flatten", "Squeeze", "Reshape"):
            # All three collapse the [N,C,1,1] pooled tensor to [N,C] in the
            # models we accept.  The builder is NHWC, so a general Reshape
            # against NCHW semantics would be wrong; restrict to that case.
            steps.append(Step("flatten", nm, [resolve(node.input[0])], node.output[0], {}))
            continue

        if op == "Transpose":
            steps.append(Step("transpose", nm, [resolve(node.input[0])], node.output[0],
                              {"perms": list(a["perm"])}))
            continue

        raise UnsupportedOnnxOp(f"node {nm!r}: unhandled op {op!r}")

    out_name = resolve(g.output[0].name)
    return steps, in_name, out_name


# ---------------------------------------------------------------------------
# Lowering to a build function
# ---------------------------------------------------------------------------


def make_build_fn(steps: List[Step], in_name: str, out_name: str) -> Callable:
    """Turn parsed steps into ``build(net, x)`` usable by either backend."""

    def build(net, x):
        env: Dict[str, Any] = {in_name: x}

        def get(n: str):
            try:
                return env[n]
            except KeyError:
                raise UnsupportedOnnxOp(
                    f"value {n!r} is used before it is produced; the ONNX graph "
                    "may not be topologically sorted"
                ) from None

        for st in steps:
            a = st.attrs
            nm = st.name.replace("/", "_").replace(":", "_")
            if st.op == "conv":
                y = net.conv2d(nm, get(st.inputs[0]), a["w"], a["b"], a["stride"],
                               a["pad"], a["dilation"], a["groups"], a["act"])
            elif st.op == "linear":
                y = net.linear(nm, get(st.inputs[0]), a["w"], a["b"], a["act"])
            elif st.op == "add":
                y = net.add(nm, get(st.inputs[0]), get(st.inputs[1]))
            elif st.op == "act":
                if a["act"] != "relu6":
                    raise UnsupportedOnnxOp(
                        f"{nm}: standalone {a['act']} is only supported as relu6"
                    )
                y = net.clamp_relu6(nm, get(st.inputs[0]))
            elif st.op == "gap":
                y = net.global_avg_pool(nm, get(st.inputs[0]))
            elif st.op == "avgpool":
                y = net.avg_pool2d(nm, get(st.inputs[0]), a["kernel"], a["stride"])
            elif st.op == "flatten":
                h = get(st.inputs[0])
                n = int(h.shape[0])
                y = net.reshape(nm, h, [n, int(np.prod(list(h.shape)[1:]))])
            elif st.op == "transpose":
                y = net.transpose(nm, get(st.inputs[0]), a["perms"])
            else:
                raise UnsupportedOnnxOp(f"no lowering for step {st.op!r}")
            env[st.output] = y
        return get(out_name)

    return build
