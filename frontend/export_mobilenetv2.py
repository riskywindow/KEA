#!/usr/bin/env python3
"""End-to-end MobileNetV2 -> ``models/mobilenetv2_int8.kgraph.json`` + ``.npz``.

    source scripts/env.sh
    .venv/bin/python frontend/export_mobilenetv2.py

Steps: load the pretrained float model, fold every BatchNorm into its
convolution, calibrate activation ranges on real images, quantize
(per-output-channel symmetric int8 weights, per-tensor asymmetric int8
activations), derive every requantization multiplier/shift, emit the graph, and
measure float vs int8 top-1 with the bit-exact integer reference interpreter.

Both calibration observers (min/max and percentile) are built and evaluated so
the numbers can be compared; ``--observer`` selects which one becomes the
canonical artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kea_frontend import data as kdata
from kea_frontend.builder import TorchTracer
from kea_frontend.ir import KGraph
from kea_frontend.nets import build_mobilenetv2, extract_mobilenetv2
from kea_frontend.pipeline import build_graph, calibrate
from kea_frontend.reference import execute, quantize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_float_model():
    import torchvision

    try:
        w = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        m = torchvision.models.mobilenet_v2(weights=w)
    except Exception as exc:  # pragma: no cover - network/cert failure path
        print(f"ERROR: could not load pretrained MobileNetV2 weights: {exc}")
        print("Did you `source scripts/env.sh` (it sets SSL_CERT_FILE)?")
        raise
    m.eval()
    return m


def get_images(n: int, split: str, seed: int):
    items = kdata.list_images(split)
    if not items:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))[:n]
    return [items[i] for i in sorted(idx)]


def batches(items, bs, size=224):
    for i in range(0, len(items), bs):
        yield kdata.load_batch(items[i : i + bs], size)


def _topk(logits: np.ndarray, k: int = 5) -> np.ndarray:
    """Indices of the top ``k`` logits per row, unordered."""
    return np.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]


def eval_float(model, items, bs):
    """Top-1/top-5 of the *original torchvision model* -- the float baseline."""
    import torch

    c1 = c5 = total = 0
    preds = []
    with torch.no_grad():
        for x, y in batches(items, bs):
            xt = torch.as_tensor(x).permute(0, 3, 1, 2).contiguous()
            lg = model(xt).numpy()
            p = lg.argmax(axis=1)
            preds.append(p)
            c1 += int((p == y).sum())
            c5 += int((_topk(lg) == y[:, None]).any(axis=1).sum())
            total += len(y)
    return c1, c5, total, np.concatenate(preds) if preds else np.array([])


def eval_int8(g: KGraph, items, bs, log_every=10):
    """Top-1/top-5 from the bit-exact integer reference interpreter.

    Argmax is taken on the raw int32/int8 logits: requantization is monotonic,
    so ranking them is identical to ranking their dequantized values, and this
    keeps the evaluation path free of float entirely.
    """
    c1 = c5 = total = 0
    preds = []
    t0 = time.time()
    tin = g.tensor("input")
    for bi, (x, y) in enumerate(batches(items, bs)):
        xq = quantize(x, tin)
        out = np.asarray(execute(g, {"input": xq})[g.outputs[0]])
        p = out.argmax(axis=-1)
        preds.append(p)
        c1 += int((p == y).sum())
        c5 += int((_topk(out.astype(np.int32)) == y[:, None]).any(axis=1).sum())
        total += len(y)
        if log_every and bi % log_every == 0:
            el = time.time() - t0
            print(f"    int8 eval {total}/{len(items)}  "
                  f"({el:.0f}s, {el/max(total,1):.2f}s/img)", flush=True)
    return c1, c5, total, np.concatenate(preds) if preds else np.array([])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib-images", type=int, default=256)
    ap.add_argument("--eval-images", type=int, default=400)
    ap.add_argument("--percentile", type=float, default=99.99)
    ap.add_argument("--observer", choices=["minmax", "percentile"], default="percentile",
                    help="which calibration becomes models/mobilenetv2_int8.kgraph.json")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "models"))
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("== KEA MobileNetV2 int8 export ==")
    model = load_float_model()
    desc = extract_mobilenetv2(model)
    n_conv = 1 + sum(len(b["convs"]) for b in desc.blocks) + 1
    print(f"  folded BatchNorm into {n_conv} convolutions "
          f"({len(desc.blocks)} inverted-residual blocks)")

    calib_items = get_images(args.calib_images, "train", args.seed)
    synthetic = calib_items is None
    if synthetic:
        print("  WARNING: no calibration images found under models/data/.")
        print("           Run scripts/fetch_calibration_data.sh.")
        print("           Falling back to SYNTHETIC calibration data -- the")
        print("           resulting scales are NOT representative and accuracy")
        print("           will NOT be evaluated.")
        rng = np.random.default_rng(args.seed)
        calib_batches = [rng.standard_normal((args.batch, 224, 224, 3)).astype(np.float32)
                         for _ in range(4)]
    else:
        print(f"  calibration set: {len(calib_items)} images from imagenette2-160/train")
        calib_batches = [b for b, _ in batches(calib_items, args.batch)]

    build_fn = lambda net, x: build_mobilenetv2(net, x, desc)

    eval_items = None if (synthetic or args.no_eval) else get_images(
        args.eval_images, "val", args.seed + 1
    )

    float_line = None
    if eval_items:
        print(f"\n  float baseline on {len(eval_items)} imagenette2-160/val images...")
        c1, c5, t, fpred = eval_float(model, eval_items, args.batch)
        float_line = (c1, c5, t)
        print(f"  float top-1: {c1}/{t} = {100.0*c1/t:.2f}%   "
              f"top-5: {c5}/{t} = {100.0*c5/t:.2f}%")
    else:
        fpred = None

    report = {
        "model": "mobilenetv2",
        "weights": "torchvision IMAGENET1K_V1",
        "eval_set": "imagenette2-160/val (10-class ImageNet-1k subset)",
        "eval_note":
            "The classifier is 1000-way; only the 10 Imagenette classes appear as "
            "labels. Top-1 is depressed by genuine ImageNet class ambiguity "
            "(e.g. 'church' vs monastery/dome/mosque), which is why top-5 is also "
            "reported. This is NOT ImageNet-1k top-1.",
        "eval_images": len(eval_items) if eval_items else 0,
        "calib_images": 0 if synthetic else len(calib_items),
        "calib_source": "synthetic gaussian" if synthetic else "imagenette2-160/train",
        "float_top1": None if not float_line else float_line[0] / float_line[2],
        "float_top5": None if not float_line else float_line[1] / float_line[2],
        "observers": {},
    }

    os.makedirs(args.out_dir, exist_ok=True)
    for obs in ("minmax", "percentile"):
        print(f"\n-- observer: {obs} --")
        calib = calibrate(build_fn, calib_batches, observer=obs, percentile=args.percentile)
        meta = {
            "source": "torchvision.models.mobilenet_v2 (IMAGENET1K_V1)",
            "preprocess": "resize 256 shortest side, center crop 224, "
                          "(x/255 - imagenet_mean) / imagenet_std, NHWC",
            "calib_images": 0 if synthetic else len(calib_items),
            "calib_source": report["calib_source"],
        }
        g = build_graph(build_fn, [1, 224, 224, 3], calib, f"mobilenetv2_int8_{obs}",
                        metadata=meta)
        print("  " + g.summary().replace("\n", "\n  "))

        path = os.path.join(args.out_dir, f"mobilenetv2_int8_{obs}.kgraph.json")
        npz = g.save(path)
        print(f"  wrote {path}\n        {npz}")

        entry = {"observer": obs}
        if obs == "percentile":
            entry["percentile"] = args.percentile
        if eval_items:
            geval = build_graph(build_fn, [args.eval_batch, 224, 224, 3], calib,
                                f"mobilenetv2_int8_{obs}_eval")
            c1, c5, t, ipred = eval_int8(geval, eval_items, args.eval_batch)
            entry.update(
                int8_top1=c1 / t, int8_top5=c5 / t,
                int8_top1_correct=c1, int8_top5_correct=c5, int8_total=t,
                top1_agreement_with_float=float((ipred == fpred).mean()),
            )
            print(f"  int8 top-1 ({obs}): {c1}/{t} = {100.0*c1/t:.2f}%   "
                  f"top-5: {c5}/{t} = {100.0*c5/t:.2f}%   "
                  f"argmax agreement with float: {100.0*(ipred==fpred).mean():.2f}%")
        report["observers"][obs] = entry

        if obs == args.observer:
            canon = os.path.join(args.out_dir, "mobilenetv2_int8.kgraph.json")
            g2 = KGraph.load(path)
            g2.name = "mobilenetv2_int8"
            g2.metadata["canonical_observer"] = obs
            g2.save(canon)
            print(f"  canonical artifact -> {canon}")

    rp = os.path.join(args.out_dir, "mobilenetv2_int8_accuracy.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"\nwrote accuracy report {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
