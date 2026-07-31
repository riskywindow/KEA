"""Calibration / evaluation image loading.

The dataset is Imagenette (a 10-class subset of ImageNet-1k), fetched by
``scripts/fetch_calibration_data.sh`` into ``models/data/``.  The 10 Imagenette
WordNet ids map onto real ImageNet-1k class indices, so top-1 against the
1000-way MobileNetV2 classifier is meaningful -- but it is a **10-class subset**
and accuracy on it is not ImageNet-1k top-1.  Every reported number says so.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "IMAGENETTE_TO_IMAGENET",
    "DATA_ROOT",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "find_imagenette",
    "list_images",
    "load_batch",
    "preprocess",
]

#: Imagenette WordNet id -> ImageNet-1k class index.
IMAGENETTE_TO_IMAGENET = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "data"
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_imagenette(root: Optional[str] = None) -> Optional[str]:
    """Locate the extracted Imagenette tree, or ``None`` if it is not present."""
    root = root or DATA_ROOT
    for cand in ("imagenette2-160", "imagenette2-320", "imagenette2"):
        p = os.path.join(root, cand)
        if os.path.isdir(p):
            return p
    return None


def list_images(split: str = "val", root: Optional[str] = None) -> List[Tuple[str, int]]:
    """``[(path, imagenet_class_index)]``, deterministically ordered."""
    base = find_imagenette(root)
    if base is None:
        return []
    d = os.path.join(base, split)
    if not os.path.isdir(d):
        return []
    out: List[Tuple[str, int]] = []
    for wnid in sorted(os.listdir(d)):
        if wnid not in IMAGENETTE_TO_IMAGENET:
            continue
        cls = IMAGENETTE_TO_IMAGENET[wnid]
        sub = os.path.join(d, wnid)
        for fn in sorted(os.listdir(sub)):
            if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                out.append((os.path.join(sub, fn), cls))
    return out


def preprocess(path: str, size: int = 224, resize: int = 256) -> np.ndarray:
    """Resize-shortest-side -> center crop -> normalize.  Returns HWC float32."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w < h:
            nw, nh = resize, max(1, round(h * resize / w))
        else:
            nh, nw = resize, max(1, round(w * resize / h))
        im = im.resize((nw, nh), Image.BILINEAR)
        left = (nw - size) // 2
        top = (nh - size) // 2
        im = im.crop((left, top, left + size, top + size))
        a = np.asarray(im, dtype=np.float32) / 255.0
    return (a - IMAGENET_MEAN) / IMAGENET_STD


def load_batch(items: Sequence[Tuple[str, int]], size: int = 224) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``items`` into one NHWC float32 batch plus its label vector."""
    xs = [preprocess(p, size) for p, _ in items]
    return np.stack(xs).astype(np.float32), np.array([c for _, c in items], dtype=np.int64)
