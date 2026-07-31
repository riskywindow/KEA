#!/usr/bin/env python3
"""Quantize a float ONNX model into the KEA graph IR.

    .venv/bin/python frontend/export_onnx.py model.onnx -o models/model_int8.kgraph.json

Uses the same calibration/quantization/emission code as the torch path, so an
ONNX-sourced graph and a torch-sourced graph of the same network are identical
apart from node names.  Coverage is listed in ``docs/FRONTEND.md``; unsupported
ops raise with the offending node name rather than being approximated.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kea_frontend import data as kdata
from kea_frontend.onnx_ingest import make_build_fn, onnx_input_shape, parse_onnx
from kea_frontend.pipeline import build_graph, calibrate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="path to a float .onnx model")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--observer", choices=["minmax", "percentile"], default="percentile")
    ap.add_argument("--percentile", type=float, default=99.99)
    ap.add_argument("--calib-images", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import onnx

    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    steps, in_name, out_name = parse_onnx(model)
    print(f"parsed {args.model}: {len(steps)} steps, input {in_name!r}, output {out_name!r}")

    nchw = onnx_input_shape(model)
    if len(nchw) != 4:
        raise SystemExit(f"expected a 4-D NCHW input, got shape {nchw}")
    n, c, h, w = nchw
    nhwc = [1, h, w, c]
    print(f"  input NCHW {nchw} -> KEA NHWC {nhwc}")

    build_fn = make_build_fn(steps, in_name, out_name)

    items = kdata.list_images("train")
    if items and h == 224 and c == 3:
        rng = np.random.default_rng(args.seed)
        idx = sorted(rng.permutation(len(items))[: args.calib_images])
        sel = [items[i] for i in idx]
        cal = [b for b, _ in (kdata.load_batch(sel[i : i + args.batch])
                              for i in range(0, len(sel), args.batch))]
        print(f"  calibrating on {len(sel)} imagenette images")
    else:
        print("  WARNING: no usable image calibration set for this input shape;")
        print("           using SYNTHETIC gaussian calibration data. Scales will")
        print("           not be representative of real inputs.")
        rng = np.random.default_rng(args.seed)
        cal = [rng.standard_normal([args.batch] + nhwc[1:]).astype(np.float32)
               for _ in range(4)]

    calib = calibrate(build_fn, cal, observer=args.observer, percentile=args.percentile)
    name = os.path.splitext(os.path.basename(args.model))[0] + "_int8"
    g = build_graph(build_fn, nhwc, calib, name, metadata={
        "source": f"onnx:{os.path.basename(args.model)}",
        "onnx_ir_version": int(model.ir_version),
        "onnx_producer": model.producer_name,
    })
    print(g.summary())

    out = args.out or os.path.join(REPO, "models", name + ".kgraph.json")
    npz = g.save(out)
    print(f"wrote {out}\n      {npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
