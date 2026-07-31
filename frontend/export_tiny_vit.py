#!/usr/bin/env python3
"""Export the KEA tiny vision transformer -> ``models/tiny_vit_int8.kgraph.json``.

    .venv/bin/python frontend/export_tiny_vit.py

**This model has RANDOM-INITIALISED weights and is not trained.**  It exists to
stress the parts of the ISA that MobileNetV2 never touches: rank-3 ``matmul``
with two activation operands, integer ``softmax``, integer ``layernorm``,
``table`` (GELU), ``transpose`` and ``reshape``.  Any classification accuracy
computed from it would be chance, so none is reported.  What *is* reported is
agreement between the integer reference interpreter and the float model, which
is the property the compiler and simulator actually need.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kea_frontend import data as kdata
from kea_frontend.builder import TorchTracer
from kea_frontend.nets import TinyViTConfig, build_tiny_vit, make_tiny_vit_params
from kea_frontend.pipeline import build_graph, calibrate
from kea_frontend.reference import execute, quantize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--observer", choices=["minmax", "percentile"], default="percentile")
    ap.add_argument("--percentile", type=float, default=99.99)
    ap.add_argument("--calib-images", type=int, default=32)
    ap.add_argument("--check-images", type=int, default=16)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "models"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    cfg = TinyViTConfig(image_size=args.image_size, patch=args.patch, dim=args.dim,
                        depth=args.depth, heads=args.heads)
    p = make_tiny_vit_params(cfg)
    build_fn = lambda net, x: build_tiny_vit(net, x, cfg, p)

    print("== KEA tiny ViT int8 export ==")
    print(f"  RANDOM-INIT (seed {cfg.seed}) -- compiler/ISA stress test, not a "
          "trained model")
    print(f"  {cfg.depth} layers, dim {cfg.dim}, {cfg.heads} heads, patch {cfg.patch}, "
          f"{cfg.tokens} tokens")

    items = kdata.list_images("val")
    rng = np.random.default_rng(args.seed)
    if items:
        idx = sorted(rng.permutation(len(items))[: args.calib_images + args.check_images])
        sel = [items[i] for i in idx]
        cal_items, chk_items = sel[: args.calib_images], sel[args.calib_images :]
        cal = [kdata.load_batch(cal_items[i : i + 4], cfg.image_size)[0]
               for i in range(0, len(cal_items), 4)]
        chk = [kdata.load_batch([it], cfg.image_size)[0] for it in chk_items]
        print(f"  calibrating on {len(cal_items)} real images")
    else:
        print("  WARNING: no images under models/data/; using synthetic gaussian")
        print("           calibration. Run scripts/fetch_calibration_data.sh.")
        cal = [rng.standard_normal((4, cfg.image_size, cfg.image_size, 3)).astype(np.float32)
               for _ in range(args.calib_images // 4)]
        chk = [rng.standard_normal((1, cfg.image_size, cfg.image_size, 3)).astype(np.float32)
               for _ in range(args.check_images)]

    calib = calibrate(build_fn, cal, observer=args.observer, percentile=args.percentile)
    g = build_graph(build_fn, [1, cfg.image_size, cfg.image_size, 3], calib,
                    "tiny_vit_int8", metadata={
                        "trained": False,
                        "purpose": "compiler/ISA stress test for matmul, softmax, "
                                   "layernorm and table paths",
                        "init_seed": cfg.seed,
                        "config": {"depth": cfg.depth, "dim": cfg.dim,
                                   "heads": cfg.heads, "patch": cfg.patch,
                                   "tokens": cfg.tokens,
                                   "image_size": cfg.image_size},
                    })
    print("  " + g.summary().replace("\n", "\n  "))

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "tiny_vit_int8.kgraph.json")
    npz = g.save(path)
    print(f"  wrote {path}\n        {npz}")

    # Integer-vs-float agreement: the meaningful quality metric for this model.
    print(f"\n  checking int8 reference vs float on {len(chk)} inputs...")
    tin = g.tensor("input")
    tout = g.tensor(g.outputs[0])
    tracer = TorchTracer({}, record=False)
    cos, r1, r5, sd = [], 0, 0, []
    with torch.no_grad():
        for x in chk:
            xq = quantize(x, tin)
            oi = np.asarray(execute(g, {"input": xq})[g.outputs[0]], dtype=np.float64)
            oi = (oi - tout.quant.zero_point[0]) * tout.quant.scale[0]
            of = build_fn(tracer, torch.as_tensor(x)).numpy().astype(np.float64)
            cos.append(float((oi.ravel() @ of.ravel()) /
                             (np.linalg.norm(oi) * np.linalg.norm(of))))
            r1 += int(oi.argmax() == of.argmax())
            r5 += int(oi.argmax() in np.argpartition(-of.ravel(), 5)[:5])
            sd.append(float(np.abs(oi - of).max()))
    stats = {
        "inputs": len(chk),
        "logit_cosine_similarity_mean": float(np.mean(cos)),
        "logit_cosine_similarity_min": float(np.min(cos)),
        "argmax_match": r1,
        "int8_argmax_in_float_top5": r5,
        "max_abs_logit_error_mean": float(np.mean(sd)),
    }
    print(f"    logit cosine similarity: mean {stats['logit_cosine_similarity_mean']:.5f}, "
          f"min {stats['logit_cosine_similarity_min']:.5f}")
    print(f"    argmax agreement: {r1}/{len(chk)}   "
          f"int8 argmax within float top-5: {r5}/{len(chk)}")
    print(f"    mean max-abs logit error: {stats['max_abs_logit_error_mean']:.4f}")

    rp = os.path.join(args.out_dir, "tiny_vit_int8_check.json")
    with open(rp, "w") as f:
        json.dump({
            "model": "tiny_vit",
            "trained": False,
            "note": "Random-init weights. Classification accuracy is meaningless "
                    "and is deliberately not reported; this measures int8-vs-float "
                    "numerical agreement only.",
            "observer": calib.observer,
            "calib_images": sum(b.shape[0] for b in cal),
            **stats,
        }, f, indent=2)
        f.write("\n")
    print(f"  wrote {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
