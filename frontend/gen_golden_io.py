#!/usr/bin/env python3
"""Emit golden input/output vectors for the exported models.

    .venv/bin/python frontend/gen_golden_io.py

Writes ``frontend/testdata/golden_io_<model>.npz`` containing, for a handful of
fixed inputs, the **already-quantized int8 input tensor** and the **exact
integer output** of the reference interpreter.

This is the cross-validation target for the C++ simulator: load the npz, feed
``input_<i>`` to your implementation of the matching ``.kgraph.json``, and
assert bit equality with ``output_<i>``.  No tolerance -- the reference
interpreter is deterministic and reproducible across processes and machines
(every op is exactly reduction-order-independent within its stated accumulator
width).

The inputs are stored post-quantization on purpose: it removes image decoding,
resizing and the float->int8 boundary from the comparison, so a mismatch can
only mean a disagreement about integer graph semantics.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kea_frontend import data as kdata
from kea_frontend.ir import KGraph
from kea_frontend.reference import execute, quantize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")
N_CASES = 4


def inputs_for(g: KGraph, n: int):
    """Deterministic inputs: real images when available, else seeded noise."""
    tin = g.tensor("input")
    _, h, w, _ = tin.shape
    items = kdata.list_images("val")
    rng = np.random.default_rng(20240601)
    if items and h == 224:
        sel = [items[i] for i in sorted(rng.permutation(len(items))[:n])]
        return [quantize(kdata.preprocess(p, h)[None], tin) for p, _ in sel], "imagenette2-160/val"
    return ([quantize(rng.standard_normal((1, h, w, 3)).astype(np.float32), tin)
             for _ in range(n)], "seeded gaussian noise")


def main() -> int:
    os.makedirs(TESTDATA, exist_ok=True)
    manifest = {}
    for model in ("mobilenetv2_int8", "tiny_vit_int8"):
        p = os.path.join(REPO, "models", f"{model}.kgraph.json")
        if not os.path.exists(p):
            print(f"skip {model}: not exported yet")
            continue
        g = KGraph.load(p)
        xs, src = inputs_for(g, N_CASES)
        out_name = g.outputs[0]
        arrays = {}
        h = hashlib.sha256()
        for i, x in enumerate(xs):
            y = np.ascontiguousarray(execute(g, {"input": x})[out_name])
            arrays[f"input_{i}"] = np.ascontiguousarray(x)
            arrays[f"output_{i}"] = y
            h.update(y.tobytes())
        path = os.path.join(TESTDATA, f"golden_io_{model}.npz")
        np.savez_compressed(path, **arrays)
        manifest[model] = {
            "graph": f"models/{model}.kgraph.json",
            "cases": len(xs),
            "input_tensor": "input",
            "input_dtype": g.tensor("input").dtype,
            "input_shape": g.tensor("input").shape,
            "output_tensor": out_name,
            "output_dtype": g.tensor(out_name).dtype,
            "output_shape": g.tensor(out_name).shape,
            "input_source": src,
            "sha256_of_concatenated_outputs": h.hexdigest(),
        }
        print(f"wrote {path}  ({len(xs)} cases, {os.path.getsize(path)/1024:.0f} KiB)")
        print(f"       output sha256 {h.hexdigest()[:32]}...")

    mp = os.path.join(TESTDATA, "golden_io_manifest.json")
    with open(mp, "w") as f:
        json.dump({
            "format": "kea.golden_io",
            "version": 1,
            "note": "Feed input_<i> to your implementation of the named graph and "
                    "assert exact integer equality with output_<i>. No tolerance.",
            "models": manifest,
        }, f, indent=2)
        f.write("\n")
    print(f"wrote {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
