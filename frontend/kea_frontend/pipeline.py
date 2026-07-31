"""Calibrate-then-emit driver shared by the export scripts."""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .builder import Calibration, KGraphBuilder, TorchTracer
from .ir import KGraph
from .quant import HistogramObserver, MinMaxObserver, Observer

__all__ = ["calibrate", "build_graph", "INPUT_NAME"]

INPUT_NAME = "input"


def _run_pass(build_fn, images: Sequence[np.ndarray], factory) -> Dict[str, Observer]:
    import torch

    observers: Dict[str, Observer] = {}
    tracer = TorchTracer(observers, factory=factory)
    with torch.no_grad():
        for batch in images:
            x = torch.as_tensor(np.ascontiguousarray(batch), dtype=torch.float32)
            tracer._obs(INPUT_NAME, x)
            build_fn(tracer, x)
    return observers


def calibrate(
    build_fn: Callable,
    images: Sequence[np.ndarray],
    observer: str = "minmax",
    percentile: float = 99.99,
    verbose: bool = True,
) -> Calibration:
    """Collect per-tensor activation ranges by running the float model.

    ``images`` is a sequence of NHWC float32 batches (already preprocessed).

    ``observer='minmax'`` needs one pass.  ``observer='percentile'`` needs two:
    the first fits absolute bounds, the second builds an exact histogram inside
    those bounds.  Two passes avoid the range-expansion/rebinning heuristics
    that make single-pass histogram observers implementation-defined.
    """
    t0 = time.time()
    mm = _run_pass(build_fn, images, lambda name: MinMaxObserver())
    if verbose:
        print(f"  [calib] min/max pass over {len(images)} batches "
              f"({sum(b.shape[0] for b in images)} images) in {time.time()-t0:.1f}s, "
              f"{len(mm)} tensors")

    if observer == "minmax":
        ranges = {k: o.range() for k, o in mm.items()}
        return Calibration("minmax", percentile, ranges)

    if observer != "percentile":
        raise ValueError(f"unknown observer {observer!r}")

    bounds = {k: o.range() for k, o in mm.items()}
    t1 = time.time()
    hist = _run_pass(
        build_fn, images,
        lambda name: HistogramObserver(*bounds[name], percentile=percentile)
        if name in bounds else MinMaxObserver(),
    )
    if verbose:
        print(f"  [calib] percentile({percentile}) pass in {time.time()-t1:.1f}s")
    ranges = {k: o.range() for k, o in hist.items()}
    return Calibration("percentile", percentile, ranges)


def build_graph(
    build_fn: Callable,
    input_shape: Sequence[int],
    calib: Calibration,
    name: str,
    double_round: bool = True,
    metadata: Optional[Dict] = None,
) -> KGraph:
    """Emit the quantized graph using ``calib``.  ``input_shape`` is NHWC."""
    b = KGraphBuilder(name, calib, double_round=double_round)
    x = b.input(INPUT_NAME, list(input_shape), "NHWC")
    out = build_fn(b, x)
    b.mark_output(out)
    b.g.metadata.update(metadata or {})
    b.g.metadata.setdefault("observer", calib.observer)
    if calib.observer == "percentile":
        b.g.metadata.setdefault("percentile", calib.percentile)
    b.g.validate()
    return b.g
