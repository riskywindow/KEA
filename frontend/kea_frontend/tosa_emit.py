# SPDX-License-Identifier: Apache-2.0
"""``.kgraph.json`` -> TOSA MLIR, for **MLIR 20.1.6 only**.

This is the missing link between the frontend and the compiler: the compiler
ingests TOSA (``-tosa-to-kea``), the frontend emits ``.kgraph.json``, and
``docs/FRONTEND.md`` section 5 specifies the node-to-op mapping.  This module
implements it.

Everything here follows **``docs/TOSA_NOTES.md``**, which was machine-verified
against this machine's ``mlir-opt``.  20.1.6 is pre-TOSA-1.0, so:

* zero points are ``#tosa.conv_quant`` / ``#tosa.matmul_quant`` /
  ``#tosa.unary_quant`` / ``#tosa.pad_quant`` **attributes**, not operands;
* ``tosa.fully_connected`` still exists;
* ``tosa.reshape`` takes ``new_shape`` as a ``DenseI64ArrayAttr``;
* ``tosa.transpose``'s permutation is an **operand**;
* ``tosa.rescale``'s ``shift`` is ``array<i8: ...>`` and ``scale32`` /
  ``double_round`` / ``per_channel`` are all mandatory;
* ``tosa.clamp`` needs ``min_fp`` / ``max_fp`` even on an integer tensor;
* ``conv2d`` weights are OHWI while ``depthwise_conv2d`` weights are HWCM;
* ``acc_type = i32`` is mandatory on conv / pool.

Usage::

    .venv/bin/python -m kea_frontend.tosa_emit models/mobilenetv2_int8.kgraph.json \\
        -o build/mobilenetv2.tosa.mlir

    from kea_frontend.tosa_emit import emit_module
    text = emit_module(kgraph)

Constants are emitted in MLIR's **hex** ``dense<"0x...">`` form (raw
little-endian bytes), because MobileNetV2's 3.5 MB of weights would otherwise
be ~25 MB of decimal text.  Splat constants collapse to ``dense<v>``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .ir import KGraph, Node, Tensor

__all__ = [
    "TosaEmitError",
    "emit_module",
    "subgraph_nodes",
    "EmitResult",
]

#: TOSA ops this module knows how to produce.  ``softmax`` and ``layernorm``
#: are deliberately absent: they are fused KEA ops with no TOSA counterpart
#: (docs/FRONTEND.md section 5).
SUPPORTED_OPS = {
    "conv2d",
    "depthwise_conv2d",
    "fully_connected",
    "matmul",
    "add",
    "clamp",
    "rescale",
    "avg_pool2d",
    "global_avg_pool",
    "reshape",
    "transpose",
    "concat",
    "pad",
    "table",
}

MLIR_DTYPE = {"int8": "i8", "int32": "i32"}


class TosaEmitError(RuntimeError):
    """Raised when a node cannot be expressed in TOSA 20.1.6."""


# ---------------------------------------------------------------------------
# small textual helpers
# ---------------------------------------------------------------------------


def _ty(shape: Sequence[int], dtype: str) -> str:
    return "tensor<%sx%s>" % ("x".join(str(int(s)) for s in shape), MLIR_DTYPE[dtype])


def _i64_array(vals: Sequence[int]) -> str:
    return "array<i64: %s>" % ", ".join(str(int(v)) for v in vals)


def _i32_array(vals: Sequence[int]) -> str:
    return "array<i32: %s>" % ", ".join(str(int(v)) for v in vals)


def _i8_array(vals: Sequence[int]) -> str:
    return "array<i8: %s>" % ", ".join(str(int(v)) for v in vals)


def _fp(x: float) -> str:
    """A float literal MLIR's parser accepts unambiguously."""
    return "%.6e" % float(x)


def _dense(arr: np.ndarray) -> str:
    """A ``dense<...>`` literal for ``arr``.

    Splats collapse; everything else goes out as MLIR's hex byte form, which
    is the only sane encoding for megabytes of weights.  ``numpy.tobytes()`` is
    C-contiguous little-endian, which is exactly what MLIR's hex parser wants.
    """
    if arr.size == 0:
        raise TosaEmitError("empty constant")
    flat = arr.reshape(-1)
    if bool((flat == flat[0]).all()):
        return "dense<%d>" % int(flat[0])
    return 'dense<"0x%s">' % np.ascontiguousarray(arr).tobytes().hex().upper()


def _sanitize(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch in "_$.-") else "_")
    s = "".join(out)
    if not s or s[0].isdigit():
        s = "v" + s
    return s


# ---------------------------------------------------------------------------
# subgraph selection
# ---------------------------------------------------------------------------


def subgraph_nodes(
    g: KGraph,
    first: Optional[str] = None,
    last: Optional[str] = None,
) -> List[Node]:
    """The contiguous node slice ``[first, last]`` by node name (inclusive).

    ``nodes`` is topologically ordered by construction (the validator enforces
    it), so a contiguous slice is a legal subgraph.
    """
    names = [n.name for n in g.nodes]
    lo = 0 if first is None else names.index(first)
    hi = len(names) - 1 if last is None else names.index(last)
    if lo > hi:
        raise TosaEmitError("empty node range %r..%r" % (first, last))
    return list(g.nodes[lo : hi + 1])


class EmitResult:
    """The emitted text plus what the emitter had to decide along the way."""

    def __init__(self, text: str, function: str, inputs: List[str],
                 outputs: List[str], notes: List[str]) -> None:
        self.text = text
        self.function = function
        self.inputs = inputs
        self.outputs = outputs
        #: human-readable notes about anything that is not a 1:1 mapping
        self.notes = notes

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.text


# ---------------------------------------------------------------------------
# the emitter
# ---------------------------------------------------------------------------


class _Emitter:
    def __init__(self, g: KGraph, nodes: List[Node], function: str) -> None:
        self.g = g
        self.nodes = nodes
        self.function = function
        self.lines: List[str] = []
        self.notes: List[str] = []
        self.ssa: Dict[str, str] = {}
        self.used: Set[str] = set()
        self.tmp = 0

    # -- naming ------------------------------------------------------------
    def _fresh(self, hint: str) -> str:
        base = _sanitize(hint)
        cand = base
        while cand in self.used:
            self.tmp += 1
            cand = "%s_%d" % (base, self.tmp)
        self.used.add(cand)
        return cand

    def bind(self, tensor_name: str, hint: Optional[str] = None) -> str:
        v = self._fresh(hint or tensor_name)
        self.ssa[tensor_name] = v
        return v

    def val(self, tensor_name: str) -> str:
        if tensor_name not in self.ssa:
            raise TosaEmitError(
                "tensor %r is read before it is produced; the node slice is "
                "not closed" % tensor_name
            )
        return "%" + self.ssa[tensor_name]

    def t(self, name: str) -> Tensor:
        return self.g.tensor(name)

    def ty(self, name: str) -> str:
        t = self.t(name)
        return _ty(t.shape, t.dtype)

    def emit(self, line: str) -> None:
        self.lines.append("  " + line)

    # -- constants ---------------------------------------------------------
    def const(self, tensor_name: str) -> str:
        """Materialize a graph constant as a ``tosa.const``, once."""
        if tensor_name in self.ssa:
            return self.val(tensor_name)
        t = self.t(tensor_name)
        if t.kind != "const":
            raise TosaEmitError("tensor %r is not a constant" % tensor_name)
        arr = self.g.const(tensor_name)
        v = self.bind(tensor_name)
        ty = _ty(t.shape, t.dtype)
        self.emit('%%%s = "tosa.const"() {value = %s : %s} : () -> %s'
                  % (v, _dense(arr), ty, ty))
        return "%" + v

    def literal_const(self, hint: str, arr: np.ndarray) -> Tuple[str, str]:
        """A constant that is not a graph tensor (transpose perms, pad const)."""
        v = self._fresh(hint)
        dtype = {np.dtype("int8"): "int8", np.dtype("int32"): "int32"}[arr.dtype]
        ty = _ty(arr.shape, dtype)
        self.emit('%%%s = "tosa.const"() {value = %s : %s} : () -> %s'
                  % (v, _dense(arr), ty, ty))
        return "%" + v, ty

    # -- module ------------------------------------------------------------
    def run(self, inputs: List[str], outputs: List[str]) -> EmitResult:
        arg_decls = []
        for i, name in enumerate(inputs):
            v = self.bind(name, name)
            arg_decls.append("%%%s: %s" % (v, self.ty(name)))
        ret_tys = ", ".join(self.ty(o) for o in outputs)
        if len(outputs) != 1:
            ret_tys = "(%s)" % ret_tys

        header = "func.func @%s(%s) -> %s {" % (
            self.function, ", ".join(arg_decls), ret_tys)

        for node in self.nodes:
            self.node(node)

        body = list(self.lines)
        rets = ", ".join(self.val(o) for o in outputs)
        rtys = ", ".join(self.ty(o) for o in outputs)
        body.append("  return %s : %s" % (rets, rtys))

        text = "\n".join([header] + body + ["}", ""])
        return EmitResult(text, self.function, inputs, outputs, self.notes)

    # -- per-node dispatch -------------------------------------------------
    def node(self, n: Node) -> None:
        fn = getattr(self, "_op_" + n.op, None)
        if fn is None:
            raise TosaEmitError(
                "node %r: op %r has no TOSA 20.1.6 counterpart. "
                "docs/FRONTEND.md section 5: softmax and layernorm are fused "
                "KEA ops and need a kea.* custom op, not TOSA."
                % (n.name, n.op)
            )
        self.emit("// %s %s" % (n.op, n.name))
        fn(n)

    # -- convolution -------------------------------------------------------
    def _conv_like(self, n: Node, mnemonic: str) -> None:
        a = n.attrs
        x, w, b = n.inputs[0], n.inputs[1], n.inputs[2]
        xv, wv, bv = self.val(x), self.const(w), self.const(b)
        out = n.outputs[0]
        ov = self.bind(out)
        self.emit(
            "%%%s = %s %s, %s, %s {\n"
            "    pad = %s, stride = %s, dilation = %s, acc_type = i32,\n"
            "    quantization_info = #tosa.conv_quant<input_zp = %d, weight_zp = %d>\n"
            "  } : (%s, %s, %s) -> %s"
            % (ov, mnemonic, xv, wv, bv,
               _i64_array(a["pad"]), _i64_array(a["stride"]), _i64_array(a["dilation"]),
               int(a["input_zp"]), int(a["weight_zp"]),
               self.ty(x), self.ty(w), self.ty(b), self.ty(out))
        )

    def _op_conv2d(self, n: Node) -> None:
        # weights are already OHWI in the .kgraph, which is tosa.conv2d's layout
        self._conv_like(n, "tosa.conv2d")

    def _op_depthwise_conv2d(self, n: Node) -> None:
        # weights are already HWCM, which is tosa.depthwise_conv2d's layout
        # (TOSA_NOTES section 3: this differs from conv2d and is the classic trap)
        w = self.t(n.inputs[1])
        if w.shape[3] != 1:
            raise TosaEmitError(
                "depthwise %r: channel multiplier %d != 1 is not emitted"
                % (n.name, w.shape[3]))
        self._conv_like(n, "tosa.depthwise_conv2d")

    # -- matmul / fully connected -----------------------------------------
    def _op_fully_connected(self, n: Node) -> None:
        a = n.attrs
        x, w, b = n.inputs
        out = n.outputs[0]
        xt, ot = self.t(x), self.t(out)
        xv = self.val(x)
        wv, bv = self.const(w), self.const(b)

        # tosa.fully_connected is strictly rank 2 in, rank 2 out.  Rank > 2
        # needs a reshape to [prod(leading), IC] and back (FRONTEND.md 5).
        xty, oty = self.ty(x), self.ty(out)
        flat_in = list(xt.shape)
        flat_out = list(ot.shape)
        if len(xt.shape) != 2:
            ic = xt.shape[-1]
            rows = int(np.prod(xt.shape[:-1]))
            flat_in = [rows, ic]
            flat_out = [rows, ot.shape[-1]]
            v = self._fresh(x + "_2d")
            nty = _ty(flat_in, xt.dtype)
            self.emit("%%%s = tosa.reshape %s {new_shape = %s} : (%s) -> %s"
                      % (v, xv, _i64_array(flat_in), xty, nty))
            xv, xty = "%" + v, nty

        accty = _ty(flat_out, ot.dtype)
        need_reshape = flat_out != list(ot.shape)
        ov = self._fresh(out + "_2d") if need_reshape else self.bind(out)
        self.emit(
            "%%%s = tosa.fully_connected %s, %s, %s {\n"
            "    quantization_info = #tosa.conv_quant<input_zp = %d, weight_zp = %d>\n"
            "  } : (%s, %s, %s) -> %s"
            % (ov, xv, wv, bv, int(a["input_zp"]), int(a["weight_zp"]),
               xty, self.ty(w), self.ty(b), accty))
        if need_reshape:
            fv = self.bind(out)
            self.emit("%%%s = tosa.reshape %%%s {new_shape = %s} : (%s) -> %s"
                      % (fv, ov, _i64_array(ot.shape), accty, self.ty(out)))

    def _op_matmul(self, n: Node) -> None:
        a = n.attrs
        x, y = n.inputs
        out = n.outputs[0]
        xv = self.val(x) if self.t(x).kind != "const" else self.const(x)
        yv = self.val(y) if self.t(y).kind != "const" else self.const(y)
        ov = self.bind(out)
        self.emit(
            "%%%s = tosa.matmul %s, %s {\n"
            "    quantization_info = #tosa.matmul_quant<a_zp = %d, b_zp = %d>\n"
            "  } : (%s, %s) -> %s"
            % (ov, xv, yv, int(a["a_zp"]), int(a["b_zp"]),
               self.ty(x), self.ty(y), self.ty(out)))

    # -- elementwise -------------------------------------------------------
    def _op_add(self, n: Node) -> None:
        out = n.outputs[0]
        oty = self.ty(out)
        acc = self.val(n.inputs[0])
        accty = self.ty(n.inputs[0])
        for i, rhs in enumerate(n.inputs[1:]):
            last = i == len(n.inputs) - 2
            v = self.bind(out) if last else self._fresh(out + "_p%d" % i)
            self.emit("%%%s = tosa.add %s, %s : (%s, %s) -> %s"
                      % (v, acc, self.val(rhs), accty, self.ty(rhs),
                         oty if last else accty))
            acc, accty = "%" + v, oty if last else accty

    def _op_clamp(self, n: Node) -> None:
        a = n.attrs
        x, out = n.inputs[0], n.outputs[0]
        ov = self.bind(out)
        lo, hi = int(a["min_val"]), int(a["max_val"])
        # min_fp / max_fp are ignored by the integer lowering but are a PARSE
        # requirement on 20.1.6 (TOSA_NOTES section 5).
        self.emit(
            "%%%s = tosa.clamp %s {\n"
            "    min_int = %d : i64, max_int = %d : i64,\n"
            "    min_fp = %s : f32, max_fp = %s : f32\n"
            "  } : (%s) -> %s"
            % (ov, self.val(x), lo, hi, _fp(lo), _fp(hi),
               self.ty(x), self.ty(out)))

    def _rescale(self, dst_hint: str, src: str, src_ty: str, dst_ty: str,
                 input_zp: int, output_zp: int, mult: Sequence[int],
                 shift: Sequence[int], per_channel: bool,
                 double_round: bool, bind_to: Optional[str] = None) -> str:
        v = self.bind(bind_to) if bind_to is not None else self._fresh(dst_hint)
        self.emit(
            "%%%s = tosa.rescale %s {\n"
            "    input_zp = %d : i32, output_zp = %d : i32,\n"
            "    multiplier = %s,\n"
            "    shift = %s,\n"
            "    scale32 = true, double_round = %s, per_channel = %s\n"
            "  } : (%s) -> %s"
            % (v, src, input_zp, output_zp,
               _i32_array(mult), _i8_array(shift),
               "true" if double_round else "false",
               "true" if per_channel else "false",
               src_ty, dst_ty))
        return "%" + v

    def _op_rescale(self, n: Node) -> None:
        a = n.attrs
        x, out = n.inputs[0], n.outputs[0]
        self._rescale(out, self.val(x), self.ty(x), self.ty(out),
                      int(a["input_zp"]), int(a["output_zp"]),
                      a["multiplier"], a["shift"],
                      bool(a.get("per_channel", False)),
                      bool(a.get("double_round", True)),
                      bind_to=out)

    # -- pooling -----------------------------------------------------------
    def _pool(self, n: Node, kernel: Sequence[int], stride: Sequence[int],
              pad: Sequence[int]) -> None:
        """``avg_pool2d`` / ``global_avg_pool`` -> ``tosa.avg_pool2d``.

        KEA's pooling nodes fold an arbitrary requantization into
        ``multiplier``/``shift``; ``tosa.avg_pool2d`` can only divide by the
        window count and rebase the zero point.  When the folded scale is
        exactly ``1/(kh*kw)`` the mapping is 1:1.  When it is not -- which is
        the normal case for MobileNetV2's head, whose pool also converts
        between two different activation scales -- the pool is emitted at the
        *input* scale and a ``tosa.rescale`` carries the remaining factor.
        That is a second rounding, so it is **not bit-exact** against the
        frontend reference; it is recorded in ``notes``.
        """
        a = n.attrs
        x, out = n.inputs[0], n.outputs[0]
        izp, ozp = int(a["input_zp"]), int(a["output_zp"])
        mult, shift = int(a["multiplier"]), int(a["shift"])
        count = int(kernel[0]) * int(kernel[1])

        # what apply_scale(acc, mult, shift) really multiplies by
        folded = mult / float(1 << shift)
        exact = abs(folded * count - 1.0) < 1e-9

        if exact:
            ov = self.bind(out)
            self.emit(
                "%%%s = tosa.avg_pool2d %s {\n"
                "    kernel = %s, stride = %s, pad = %s, acc_type = i32,\n"
                "    quantization_info = #tosa.unary_quant<input_zp = %d, output_zp = %d>\n"
                "  } : (%s) -> %s"
                % (ov, self.val(x), _i64_array(kernel), _i64_array(stride),
                   _i64_array(pad), izp, ozp, self.ty(x), self.ty(out)))
            return

        # inexact: pool at the input scale, then rescale.
        residual = folded * count
        rmult, rshift = _quantize_multiplier(residual)
        mid = self._fresh(out + "_pooled")
        midty = _ty(self.t(out).shape, self.t(x).dtype)
        self.emit(
            "%%%s = tosa.avg_pool2d %s {\n"
            "    kernel = %s, stride = %s, pad = %s, acc_type = i32,\n"
            "    quantization_info = #tosa.unary_quant<input_zp = %d, output_zp = %d>\n"
            "  } : (%s) -> %s"
            % (mid, self.val(x), _i64_array(kernel), _i64_array(stride),
               _i64_array(pad), izp, izp, self.ty(x), midty))
        self._rescale(out, "%" + mid, midty, self.ty(out), izp, ozp,
                      [rmult], [rshift], False,
                      bool(a.get("double_round", True)), bind_to=out)
        self.notes.append(
            "%s (%s): the node folds a %.6f scale change into its "
            "multiplier/shift; tosa.avg_pool2d can only divide by the window "
            "count, so the pool is emitted at the input scale followed by a "
            "tosa.rescale of %d >> %d. This rounds twice where the frontend "
            "reference rounds once and is therefore NOT bit-exact."
            % (n.name, n.op, residual, rmult, rshift))

    def _op_avg_pool2d(self, n: Node) -> None:
        a = n.attrs
        self._pool(n, a["kernel"], a["stride"], a["pad"])

    def _op_global_avg_pool(self, n: Node) -> None:
        xs = self.t(n.inputs[0]).shape
        self._pool(n, [xs[1], xs[2]], [1, 1], [0, 0, 0, 0])

    # -- data movement -----------------------------------------------------
    def _op_reshape(self, n: Node) -> None:
        x, out = n.inputs[0], n.outputs[0]
        ov = self.bind(out)
        self.emit("%%%s = tosa.reshape %s {new_shape = %s} : (%s) -> %s"
                  % (ov, self.val(x), _i64_array(n.attrs["new_shape"]),
                     self.ty(x), self.ty(out)))

    def _op_transpose(self, n: Node) -> None:
        x, out = n.inputs[0], n.outputs[0]
        perms = np.asarray([int(p) for p in n.attrs["perms"]], dtype=np.int32)
        pv, pty = self.literal_const(n.name + "_perms", perms)
        ov = self.bind(out)
        # 20.1.6: the permutation is an OPERAND, not an attribute.
        self.emit("%%%s = tosa.transpose %s, %s : (%s, %s) -> %s"
                  % (ov, self.val(x), pv, self.ty(x), pty, self.ty(out)))

    def _op_concat(self, n: Node) -> None:
        out = n.outputs[0]
        ov = self.bind(out)
        vals = ", ".join(self.val(i) for i in n.inputs)
        tys = ", ".join(self.ty(i) for i in n.inputs)
        self.emit("%%%s = tosa.concat %s {axis = %d : i32} : (%s) -> %s"
                  % (ov, vals, int(n.attrs["axis"]), tys, self.ty(out)))

    def _op_pad(self, n: Node) -> None:
        x, out = n.inputs[0], n.outputs[0]
        padding = [int(p) for p in n.attrs["padding"]]
        sv = self._fresh(n.name + "_pad")
        # tosa.pad HAS migrated to !tosa.shape on 20.1.6, and const_shape's
        # value must be tensor<Nxindex> (TOSA_NOTES section 9).
        self.emit("%%%s = tosa.const_shape {value = dense<[%s]> : tensor<%dxindex>}"
                  " : () -> !tosa.shape<%d>"
                  % (sv, ", ".join(str(p) for p in padding), len(padding),
                     len(padding)))
        ov = self.bind(out)
        self.emit("%%%s = tosa.pad %s, %%%s {quantization_info = "
                  "#tosa.pad_quant<input_zp = %d>} : (%s, !tosa.shape<%d>) -> %s"
                  % (ov, self.val(x), sv, int(n.attrs.get("pad_const", 0)),
                     self.ty(x), len(padding), self.ty(out)))

    def _op_table(self, n: Node) -> None:
        x, tab = n.inputs
        out = n.outputs[0]
        tv = self.const(tab)
        ov = self.bind(out)
        self.emit("%%%s = tosa.table %s, %s : (%s, %s) -> %s"
                  % (ov, self.val(x), tv, self.ty(x), self.ty(tab),
                     self.ty(out)))


def _quantize_multiplier(real: float) -> Tuple[int, int]:
    """``real`` -> a TOSA ``(multiplier, shift)`` pair with ``shift`` in [2,62].

    Thin wrapper over the frontend's normative routine so this module cannot
    drift from ``docs/QUANTIZATION.md``.
    """
    from .apply_scale import quantize_multiplier

    m, s = quantize_multiplier(real)
    return int(m), int(s)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def emit_module(
    g: KGraph,
    function: Optional[str] = None,
    nodes: Optional[List[Node]] = None,
    header: bool = True,
) -> EmitResult:
    """Emit one ``func.func`` containing ``nodes`` (default: the whole graph).

    The function's arguments are every tensor the slice reads without
    producing (constants excluded, which become ``tosa.const``); its results
    are the graph outputs that the slice produces, or the slice's final
    node output when no graph output is in range.

    Exactly one function is emitted per module on purpose: ``kea-opt`` runs its
    pipeline over the whole module, so an unrelated second function that fails
    to legalize would take the whole compile down with it.
    """
    nodes = list(g.nodes) if nodes is None else list(nodes)
    function = function or _sanitize(g.name)

    produced: Set[str] = set()
    reads: List[str] = []
    for n in nodes:
        for i in n.inputs:
            if i not in produced and g.tensor(i).kind != "const" and i not in reads:
                reads.append(i)
        produced.update(n.outputs)

    outputs = [o for o in g.outputs if o in produced]
    if not outputs:
        outputs = [nodes[-1].outputs[0]]

    for n in nodes:
        if n.op not in SUPPORTED_OPS:
            raise TosaEmitError(
                "node %r: op %r has no TOSA 20.1.6 counterpart "
                "(docs/FRONTEND.md section 5)" % (n.name, n.op))

    em = _Emitter(g, nodes, function)
    res = em.run(reads, outputs)

    if header:
        lines = [
            "// Generated by kea_frontend.tosa_emit from %r (kgraph v%s)."
            % (g.name, g.version),
            "// TOSA for MLIR 20.1.6 ONLY -- see docs/TOSA_NOTES.md section 16.",
            "// %d nodes, inputs %s, outputs %s"
            % (len(nodes), res.inputs, res.outputs),
        ]
        for note in res.notes:
            for i, chunk in enumerate(_wrap(note, 74)):
                lines.append("// %s%s" % ("NOTE: " if i == 0 else "      ", chunk))
        res.text = "\n".join(lines) + "\n" + res.text
    return res


def _wrap(s: str, width: int) -> List[str]:
    out, line = [], ""
    for word in s.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word) if line else word
    if line:
        out.append(line)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m kea_frontend.tosa_emit",
        description="Emit TOSA MLIR (MLIR 20.1.6) from a .kgraph.json")
    ap.add_argument("kgraph")
    ap.add_argument("-o", "--output", help="output .mlir (default: stdout)")
    ap.add_argument("--function", help="function name (default: the graph name)")
    ap.add_argument("--first-node", help="first node of the slice, by node name")
    ap.add_argument("--last-node", help="last node of the slice, by node name")
    ap.add_argument("--first-index", type=int, help="first node of the slice, by index")
    ap.add_argument("--last-index", type=int, help="last node of the slice, by index")
    ap.add_argument("--notes-json", help="write the emitter's notes here")
    args = ap.parse_args(argv)

    g = KGraph.load(args.kgraph)
    nodes = list(g.nodes)
    if args.first_index is not None or args.last_index is not None:
        lo = args.first_index or 0
        hi = len(nodes) - 1 if args.last_index is None else args.last_index
        nodes = nodes[lo : hi + 1]
    if args.first_node is not None or args.last_node is not None:
        names = [n.name for n in nodes]
        lo = 0 if args.first_node is None else names.index(args.first_node)
        hi = len(names) - 1 if args.last_node is None else names.index(args.last_node)
        nodes = nodes[lo : hi + 1]

    res = emit_module(g, function=args.function, nodes=nodes)
    if args.output:
        with open(args.output, "w") as f:
            f.write(res.text)
        sys.stderr.write("wrote %s (%d nodes, %.1f MB)\n"
                         % (args.output, len(nodes),
                            os.path.getsize(args.output) / 1e6))
    else:
        sys.stdout.write(res.text)
    if args.notes_json:
        with open(args.notes_json, "w") as f:
            json.dump({"function": res.function, "inputs": res.inputs,
                       "outputs": res.outputs, "notes": res.notes}, f, indent=1)
    for note in res.notes:
        sys.stderr.write("note: %s\n" % note)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
