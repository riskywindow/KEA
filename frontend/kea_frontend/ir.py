"""The KEA quantized graph IR: ``.kgraph.json`` + companion ``.npz``.

The format is deliberately boring: a flat, topologically-ordered list of nodes
over a flat table of named tensors.  Every tensor states its dtype, shape,
layout and quantization parameters explicitly; every node that performs a
requantization carries the *already-derived integer* multiplier and shift, so a
consumer never has to touch a float to execute the graph.

See ``docs/FRONTEND.md`` for the normative spec.  This module is the reference
(de)serializer and validator.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

KGRAPH_VERSION = "1.0"

#: Element types a KEA tensor may have.  ``int32`` appears only on accumulator
#: tensors and bias constants; ``int8`` is the only storage type for
#: activations and weights.
DTYPES = {
    "int8": np.int8,
    "int32": np.int32,
}

#: Node opcodes.  See ``docs/FRONTEND.md`` for per-op semantics.
OPS = {
    "conv2d",
    "depthwise_conv2d",
    "fully_connected",
    "matmul",
    "add",
    "clamp",
    "avg_pool2d",
    "global_avg_pool",
    "reshape",
    "transpose",
    "concat",
    "pad",
    "rescale",
    "table",
    "softmax",
    "layernorm",
}

#: Tensor roles.
KINDS = {"input", "output", "activation", "const"}


class KGraphError(ValueError):
    """Raised for any structural or semantic violation of the KEA graph IR."""


# ---------------------------------------------------------------------------
# Quantization descriptor
# ---------------------------------------------------------------------------


@dataclass
class QuantParams:
    """Affine quantization: ``real = scale * (q - zero_point)``.

    ``kind == "per_tensor"``   -> ``axis is None``, ``scale``/``zero_point`` length 1.
    ``kind == "per_channel"``  -> ``axis`` is an index into the tensor's shape,
    and ``scale``/``zero_point`` have length ``shape[axis]``.

    KEA restricts (and the validator enforces) the combinations actually used:

    * weights  -> per-channel **symmetric** (every zero point 0)
    * activations -> per-tensor **asymmetric** (a single, possibly non-zero zp)

    The floats here are *metadata*.  They document provenance and let tools
    reason about the graph, but the executable semantics of every node depend
    only on integers (zero points, multipliers, shifts, clamp bounds).
    """

    kind: str = "per_tensor"
    axis: Optional[int] = None
    scale: List[float] = field(default_factory=lambda: [1.0])
    zero_point: List[int] = field(default_factory=lambda: [0])

    def __post_init__(self) -> None:
        self.scale = [float(s) for s in np.atleast_1d(np.asarray(self.scale)).tolist()]
        self.zero_point = [
            int(z) for z in np.atleast_1d(np.asarray(self.zero_point)).tolist()
        ]

    @property
    def is_symmetric(self) -> bool:
        return all(z == 0 for z in self.zero_point)

    def scale_array(self) -> np.ndarray:
        return np.asarray(self.scale, dtype=np.float64)

    def zp_array(self) -> np.ndarray:
        return np.asarray(self.zero_point, dtype=np.int32)

    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "axis": self.axis,
            "scale": self.scale,
            "zero_point": self.zero_point,
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "QuantParams":
        return QuantParams(
            kind=d["kind"],
            axis=d.get("axis"),
            scale=d["scale"],
            zero_point=d["zero_point"],
        )

    @staticmethod
    def per_tensor(scale: float, zero_point: int) -> "QuantParams":
        return QuantParams("per_tensor", None, [float(scale)], [int(zero_point)])

    @staticmethod
    def per_channel(scales: Sequence[float], axis: int) -> "QuantParams":
        scales = [float(s) for s in scales]
        return QuantParams("per_channel", int(axis), scales, [0] * len(scales))

    @staticmethod
    def identity() -> "QuantParams":
        """Scale 1.0, zp 0 -- used for tensors whose integer value *is* the value."""
        return QuantParams.per_tensor(1.0, 0)


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------


@dataclass
class Tensor:
    name: str
    dtype: str
    shape: List[int]
    layout: str
    quant: QuantParams
    kind: str = "activation"
    #: key into the companion ``.npz``; required iff ``kind == "const"``
    const_key: Optional[str] = None

    def __post_init__(self) -> None:
        self.shape = [int(s) for s in self.shape]

    def to_json(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "dtype": self.dtype,
            "shape": self.shape,
            "layout": self.layout,
            "kind": self.kind,
            "quant": self.quant.to_json(),
        }
        if self.const_key is not None:
            d["const_key"] = self.const_key
        return d

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Tensor":
        return Tensor(
            name=d["name"],
            dtype=d["dtype"],
            shape=list(d["shape"]),
            layout=d["layout"],
            quant=QuantParams.from_json(d["quant"]),
            kind=d.get("kind", "activation"),
            const_key=d.get("const_key"),
        )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@dataclass
class Node:
    op: str
    name: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "name": self.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": _jsonify(self.attrs),
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Node":
        return Node(
            op=d["op"],
            name=d["name"],
            inputs=list(d["inputs"]),
            outputs=list(d["outputs"]),
            attrs=dict(d.get("attrs", {})),
        )


def _jsonify(x: Any) -> Any:
    """Convert numpy scalars/arrays inside attrs into plain JSON types."""
    if isinstance(x, dict):
        return {k: _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonify(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@dataclass
class KGraph:
    name: str
    tensors: Dict[str, Tensor] = field(default_factory=dict)
    nodes: List[Node] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    consts: Dict[str, np.ndarray] = field(default_factory=dict)
    producer: str = "kea-frontend"
    #: Global requantization policy.  Recorded so nothing is implied.
    double_round: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = KGRAPH_VERSION

    # -- construction helpers ------------------------------------------------

    def add_tensor(self, t: Tensor) -> Tensor:
        if t.name in self.tensors:
            raise KGraphError(f"duplicate tensor name {t.name!r}")
        self.tensors[t.name] = t
        return t

    def add_const(
        self,
        name: str,
        array: np.ndarray,
        layout: str,
        quant: QuantParams,
    ) -> Tensor:
        dtype = {np.dtype(np.int8): "int8", np.dtype(np.int32): "int32"}.get(array.dtype)
        if dtype is None:
            raise KGraphError(f"const {name!r} has unsupported dtype {array.dtype}")
        key = name
        self.consts[key] = np.ascontiguousarray(array)
        return self.add_tensor(
            Tensor(name, dtype, list(array.shape), layout, quant, "const", key)
        )

    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def tensor(self, name: str) -> Tensor:
        try:
            return self.tensors[name]
        except KeyError:
            raise KGraphError(f"unknown tensor {name!r}") from None

    def const(self, name: str) -> np.ndarray:
        t = self.tensor(name)
        if t.kind != "const" or t.const_key is None:
            raise KGraphError(f"tensor {name!r} is not a const")
        return self.consts[t.const_key]

    # -- serialization -------------------------------------------------------

    def to_json(self, weights_file: str) -> Dict[str, Any]:
        return {
            "kgraph_version": self.version,
            "name": self.name,
            "producer": self.producer,
            "weights_file": weights_file,
            "requant": {
                "algorithm": "tosa_apply_scale_32",
                "double_round": bool(self.double_round),
            },
            "metadata": _jsonify(self.metadata),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            # A list, not an object: ordering is part of the file and diffs
            # stay readable.
            "tensors": [self.tensors[n].to_json() for n in self._tensor_order()],
            "nodes": [n.to_json() for n in self.nodes],
        }

    def _tensor_order(self) -> List[str]:
        # graph inputs, then consts, then everything in first-definition order
        seen: List[str] = []

        def push(n: str) -> None:
            if n not in seen:
                seen.append(n)

        for n in self.inputs:
            push(n)
        for n, t in self.tensors.items():
            if t.kind == "const":
                push(n)
        for node in self.nodes:
            for n in node.inputs:
                push(n)
            for n in node.outputs:
                push(n)
        for n in self.tensors:
            push(n)
        return seen

    def save(self, json_path: str) -> str:
        """Write ``json_path`` and the sibling ``.npz``.  Returns the npz path."""
        json_path = os.path.abspath(json_path)
        base = json_path
        for suffix in (".kgraph.json", ".json"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        npz_path = base + ".npz"
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        doc = self.to_json(os.path.basename(npz_path))
        with open(json_path, "w") as f:
            json.dump(doc, f, indent=1, sort_keys=False)
            f.write("\n")
        np.savez(npz_path, **self.consts)
        return npz_path

    @staticmethod
    def load(json_path: str) -> "KGraph":
        json_path = os.path.abspath(json_path)
        with open(json_path) as f:
            doc = json.load(f)
        if doc.get("kgraph_version") != KGRAPH_VERSION:
            raise KGraphError(
                f"unsupported kgraph_version {doc.get('kgraph_version')!r}; "
                f"this build understands {KGRAPH_VERSION!r}"
            )
        npz_path = os.path.join(os.path.dirname(json_path), doc["weights_file"])
        consts: Dict[str, np.ndarray] = {}
        if os.path.exists(npz_path):
            with np.load(npz_path) as z:
                consts = {k: z[k] for k in z.files}
        g = KGraph(
            name=doc["name"],
            producer=doc.get("producer", ""),
            double_round=bool(doc.get("requant", {}).get("double_round", True)),
            metadata=doc.get("metadata", {}),
            version=doc["kgraph_version"],
        )
        for td in doc["tensors"]:
            g.add_tensor(Tensor.from_json(td))
        g.nodes = [Node.from_json(nd) for nd in doc["nodes"]]
        g.inputs = list(doc["inputs"])
        g.outputs = list(doc["outputs"])
        g.consts = consts
        g.validate()
        return g

    # -- validation ----------------------------------------------------------

    def validate(self) -> None:
        for name, t in self.tensors.items():
            if t.name != name:
                raise KGraphError(f"tensor key {name!r} != tensor.name {t.name!r}")
            if t.dtype not in DTYPES:
                raise KGraphError(f"tensor {name!r}: bad dtype {t.dtype!r}")
            if t.kind not in KINDS:
                raise KGraphError(f"tensor {name!r}: bad kind {t.kind!r}")
            q = t.quant
            if q.kind == "per_tensor":
                if q.axis is not None:
                    raise KGraphError(f"tensor {name!r}: per_tensor quant must have axis=null")
                if len(q.scale) != 1 or len(q.zero_point) != 1:
                    raise KGraphError(f"tensor {name!r}: per_tensor quant needs exactly 1 scale/zp")
            elif q.kind == "per_channel":
                if q.axis is None or not (0 <= q.axis < len(t.shape)):
                    raise KGraphError(f"tensor {name!r}: per_channel axis {q.axis} out of range")
                n = t.shape[q.axis]
                if len(q.scale) != n or len(q.zero_point) != n:
                    raise KGraphError(
                        f"tensor {name!r}: per_channel expects {n} scales, got {len(q.scale)}"
                    )
                if not q.is_symmetric:
                    raise KGraphError(
                        f"tensor {name!r}: per_channel quantization must be symmetric (zp == 0)"
                    )
            else:
                raise KGraphError(f"tensor {name!r}: bad quant kind {q.kind!r}")
            if any(s <= 0 for s in q.scale):
                raise KGraphError(f"tensor {name!r}: scales must be > 0")
            lo, hi = (-128, 127) if t.dtype == "int8" else (-(2**31), 2**31 - 1)
            if any(not (lo <= z <= hi) for z in q.zero_point):
                raise KGraphError(f"tensor {name!r}: zero_point outside {t.dtype} range")
            if t.kind == "const":
                if t.const_key is None:
                    raise KGraphError(f"const tensor {name!r} has no const_key")
                if self.consts and t.const_key not in self.consts:
                    raise KGraphError(f"const tensor {name!r}: key {t.const_key!r} missing from npz")
                if self.consts:
                    a = self.consts[t.const_key]
                    if list(a.shape) != t.shape:
                        raise KGraphError(
                            f"const {name!r}: npz shape {list(a.shape)} != declared {t.shape}"
                        )
                    if a.dtype != DTYPES[t.dtype]:
                        raise KGraphError(
                            f"const {name!r}: npz dtype {a.dtype} != declared {t.dtype}"
                        )

        for n in self.inputs + self.outputs:
            self.tensor(n)
        for n in self.inputs:
            if self.tensor(n).kind != "input":
                raise KGraphError(f"graph input {n!r} must have kind='input'")

        # topological order + SSA
        defined = {n for n in self.inputs}
        defined |= {n for n, t in self.tensors.items() if t.kind == "const"}
        produced: Dict[str, str] = {}
        names = set()
        for node in self.nodes:
            if node.op not in OPS:
                raise KGraphError(f"node {node.name!r}: unknown op {node.op!r}")
            if node.name in names:
                raise KGraphError(f"duplicate node name {node.name!r}")
            names.add(node.name)
            for i in node.inputs:
                self.tensor(i)
                if i not in defined:
                    raise KGraphError(
                        f"node {node.name!r} reads {i!r} before it is defined "
                        "(nodes must be in topological order)"
                    )
            for o in node.outputs:
                self.tensor(o)
                if o in produced:
                    raise KGraphError(f"tensor {o!r} produced twice ({produced[o]}, {node.name})")
                if o in self.inputs:
                    raise KGraphError(f"node {node.name!r} overwrites graph input {o!r}")
                produced[o] = node.name
                defined.add(o)
        for o in self.outputs:
            if o not in produced and o not in self.inputs:
                raise KGraphError(f"graph output {o!r} is never produced")

        self._validate_node_attrs()

    def _validate_node_attrs(self) -> None:
        """Per-op constraints that keep the graph directly TOSA-lowerable.

        These exist because of specific verified behaviours of MLIR 20.1.6's
        TOSA dialect (see ``docs/TOSA_NOTES.md``); violating them would produce
        a graph the emitter cannot express without changing semantics.
        """
        for node in self.nodes:
            a = node.attrs
            if node.op == "rescale":
                n_out = len(self.tensor(node.outputs[0]).shape)
                mult, shift = a["multiplier"], a["shift"]
                if len(mult) != len(shift):
                    raise KGraphError(f"rescale {node.name!r}: multiplier/shift length mismatch")
                if a.get("per_channel", False):
                    # tosa.rescale with per_channel=true requires the
                    # multiplier/shift arrays to match the LAST dimension.
                    if int(a["channel_axis"]) != n_out - 1:
                        raise KGraphError(
                            f"rescale {node.name!r}: per-channel axis must be the last "
                            f"dimension ({n_out - 1}), got {a['channel_axis']}"
                        )
                    if len(mult) != self.tensor(node.outputs[0]).shape[-1]:
                        raise KGraphError(
                            f"rescale {node.name!r}: per-channel length != last dim"
                        )
                elif len(mult) != 1:
                    raise KGraphError(
                        f"rescale {node.name!r}: per_channel=false requires exactly 1 "
                        "multiplier/shift"
                    )
                for s in shift:
                    if not (2 <= int(s) <= 62):
                        raise KGraphError(
                            f"rescale {node.name!r}: shift {s} outside TOSA's "
                            "REQUIRE(2 <= shift <= 62)"
                        )
                for m in mult:
                    if not (0 <= int(m) <= 2**31 - 1):
                        raise KGraphError(
                            f"rescale {node.name!r}: multiplier {m} outside "
                            "TOSA's REQUIRE(0 <= multiplier < 2**31)"
                        )
            elif node.op == "avg_pool2d":
                if any(int(p) != 0 for p in a["pad"]):
                    raise KGraphError(
                        f"avg_pool2d {node.name!r}: padding is not supported. TOSA "
                        "divides each output position by its ACTUAL window count, "
                        "which a single constant multiplier cannot express."
                    )
            elif node.op == "matmul":
                for t in list(node.inputs) + list(node.outputs):
                    if len(self.tensor(t).shape) != 3:
                        raise KGraphError(
                            f"matmul {node.name!r}: tosa.matmul operands are strictly "
                            f"rank 3; {t!r} has rank {len(self.tensor(t).shape)}"
                        )
            elif node.op == "layernorm":
                if int(a["axis"]) != len(self.tensor(node.inputs[0]).shape) - 1:
                    raise KGraphError(
                        f"layernorm {node.name!r}: axis must be the last dimension"
                    )

    # -- misc ----------------------------------------------------------------

    def summary(self) -> str:
        from collections import Counter

        c = Counter(n.op for n in self.nodes)
        nbytes = sum(a.nbytes for a in self.consts.values())
        lines = [
            f"kgraph {self.name!r} v{self.version}",
            f"  {len(self.nodes)} nodes, {len(self.tensors)} tensors, "
            f"{len(self.consts)} consts ({nbytes/1e6:.2f} MB)",
            f"  inputs:  {self.inputs}",
            f"  outputs: {self.outputs}",
            "  ops: " + ", ".join(f"{k}x{v}" for k, v in sorted(c.items())),
        ]
        return "\n".join(lines)


def dataclass_asdict(x) -> Dict[str, Any]:
    return dataclasses.asdict(x)
