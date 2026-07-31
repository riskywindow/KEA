"""Post-training static quantization: observers, scale derivation, BN folding.

Policy (fixed, and enforced by ``ir.KGraph.validate``):

* **weights** -- per-output-channel **symmetric** int8, zero point 0,
  ``scale[c] = max|W[c]| / 127``
* **activations** -- per-tensor **asymmetric** int8 with a zero point,
  derived from a calibrated ``[lo, hi]`` that is always widened to contain 0
* **bias** -- int32, ``scale = act_scale * weight_scale[c]``, zero point 0
  (never observed; always derived, so the bias is exact in the accumulator's
  own units)

Everything float in this module stops at the graph boundary: what lands in the
``.kgraph.json`` is the derived ``(multiplier, shift)`` integers plus the
scales/zero-points as provenance metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .apply_scale import SHIFT_MAX, SHIFT_MIN, quantize_multiplier
from .ir import QuantParams

__all__ = [
    "QMIN",
    "QMAX",
    "Observer",
    "MinMaxObserver",
    "HistogramObserver",
    "make_observer",
    "act_qparams_from_range",
    "weight_qparams",
    "quantize_weights_per_channel",
    "quantize_bias",
    "requant_params",
    "safe_quantize_multiplier",
    "fold_conv_bn",
]

QMIN, QMAX = -128, 127

#: Scales are floored here so a degenerate all-zero weight channel cannot
#: produce a zero or denormal scale.
SCALE_FLOOR = 1e-9


# ---------------------------------------------------------------------------
# Observers
# ---------------------------------------------------------------------------


class Observer:
    """Collects the dynamic range of one activation tensor over calibration."""

    name = "observer"

    def observe(self, x: np.ndarray) -> None:
        raise NotImplementedError

    def range(self) -> Tuple[float, float]:
        raise NotImplementedError


class MinMaxObserver(Observer):
    """Absolute running min/max.  No clipping, so no outlier rejection."""

    name = "minmax"

    def __init__(self) -> None:
        self.lo = np.inf
        self.hi = -np.inf
        self.count = 0

    def observe(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        self.lo = min(self.lo, float(x.min()))
        self.hi = max(self.hi, float(x.max()))
        self.count += x.size

    def range(self) -> Tuple[float, float]:
        if self.count == 0:
            return (-1.0, 1.0)
        return (self.lo, self.hi)


class HistogramObserver(Observer):
    """Percentile observer backed by a fixed-bin histogram.

    Requires the tensor's absolute ``[lo, hi]`` up front -- calibration is run
    in two passes: pass 1 fits :class:`MinMaxObserver`s, pass 2 fits these with
    the pass-1 bounds.  That keeps the histogram exact (no rebinning, no
    range-expansion heuristics) at the cost of one extra forward sweep, which
    is cheap for a few hundred images.

    ``percentile = 99.99`` clips the outer 0.01% of the mass, split evenly
    between the two tails.
    """

    name = "percentile"

    def __init__(self, lo: float, hi: float, percentile: float = 99.99, bins: int = 4096) -> None:
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = max(hi, lo + 1e-6)
        self.abs_lo = float(lo)
        self.abs_hi = float(hi)
        self.percentile = float(percentile)
        self.bins = int(bins)
        self.edges = np.linspace(self.abs_lo, self.abs_hi, self.bins + 1)
        self.counts = np.zeros(self.bins, dtype=np.int64)
        self.count = 0

    def observe(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        c, _ = np.histogram(np.clip(x, self.abs_lo, self.abs_hi), bins=self.edges)
        self.counts += c
        self.count += x.size

    def range(self) -> Tuple[float, float]:
        if self.count == 0:
            return (self.abs_lo, self.abs_hi)
        alpha = (100.0 - self.percentile) / 100.0
        total = self.counts.sum()
        if total == 0:
            return (self.abs_lo, self.abs_hi)
        cdf = np.cumsum(self.counts) / total
        # lower edge: last bin whose cdf is still below alpha/2
        i_lo = int(np.searchsorted(cdf, alpha / 2.0, side="left"))
        i_hi = int(np.searchsorted(cdf, 1.0 - alpha / 2.0, side="left"))
        i_lo = min(max(i_lo, 0), self.bins - 1)
        i_hi = min(max(i_hi, 0), self.bins - 1)
        lo = float(self.edges[i_lo])
        hi = float(self.edges[i_hi + 1])
        if hi <= lo:
            hi = lo + 1e-6
        return (lo, hi)


def make_observer(kind: str, lo: float = 0.0, hi: float = 0.0, percentile: float = 99.99) -> Observer:
    if kind == "minmax":
        return MinMaxObserver()
    if kind == "percentile":
        return HistogramObserver(lo, hi, percentile)
    raise ValueError(f"unknown observer kind {kind!r} (expected 'minmax' or 'percentile')")


# ---------------------------------------------------------------------------
# Scale / zero-point derivation
# ---------------------------------------------------------------------------


def act_qparams_from_range(lo: float, hi: float) -> QuantParams:
    """Per-tensor asymmetric int8 params for a calibrated ``[lo, hi]``.

    The range is always widened to include 0 so that the real value 0 is
    exactly representable -- convolution zero-padding depends on this.

        scale = (hi - lo) / 255
        zp    = clamp(round_half_even(-128 - lo/scale), -128, 127)
    """
    lo = float(min(lo, 0.0))
    hi = float(max(hi, 0.0))
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    scale = max((hi - lo) / float(QMAX - QMIN), SCALE_FLOOR)
    zp = int(np.clip(np.rint(QMIN - lo / scale), QMIN, QMAX))
    return QuantParams.per_tensor(scale, zp)


def weight_qparams(w: np.ndarray, out_axis: int = 0) -> QuantParams:
    """Per-output-channel symmetric int8 params.  ``scale[c] = max|W[c]|/127``."""
    axes = tuple(a for a in range(w.ndim) if a != out_axis)
    amax = np.abs(w.astype(np.float64)).max(axis=axes)
    scales = np.maximum(amax / float(QMAX), SCALE_FLOOR)
    return QuantParams.per_channel(scales.tolist(), out_axis)


def quantize_weights_per_channel(w: np.ndarray, qp: QuantParams) -> np.ndarray:
    """Symmetric int8 quantization with round-half-to-even, clamped to [-127, 127].

    The low end is clamped to -127 rather than -128 so the representation is
    exactly symmetric; this is what makes ``weight_zp = 0`` safe and lets a
    hardware MAC assume a symmetric operand range.
    """
    shape = [1] * w.ndim
    shape[qp.axis] = len(qp.scale)
    s = qp.scale_array().reshape(shape)
    q = np.rint(w.astype(np.float64) / s)
    return np.clip(q, -127, 127).astype(np.int8)


def quantize_bias(b: np.ndarray, in_scale: float, w_scales: np.ndarray) -> np.ndarray:
    """int32 bias at ``scale = in_scale * w_scale[c]``, zero point 0."""
    s = np.asarray(in_scale, dtype=np.float64) * np.asarray(w_scales, dtype=np.float64)
    q = np.rint(np.asarray(b, dtype=np.float64) / s)
    return np.clip(q, -(2**31), 2**31 - 1).astype(np.int32)


def safe_quantize_multiplier(real_multiplier) -> Tuple[np.ndarray, np.ndarray]:
    """:func:`quantize_multiplier`, but degenerate-safe and always array-valued.

    * ``real == 0`` (or so small it needs ``shift > 62``) -> ``(0, SHIFT_MAX)``,
      i.e. the rescale produces 0, which is the right answer.
    * ``real`` needing ``shift < 2`` is a genuine modelling error and raises --
      it means an op wants to amplify its accumulator by >= 2**29.
    """
    arr = np.atleast_1d(np.asarray(real_multiplier, dtype=np.float64))
    mult = np.zeros(arr.shape, dtype=np.int32)
    shift = np.full(arr.shape, SHIFT_MAX, dtype=np.int32)
    for i, v in enumerate(arr):
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"requant multiplier must be finite and >= 0, got {v}")
        if v == 0:
            continue
        frac, exp = np.frexp(v)
        need_shift = 31 - int(exp)
        if need_shift > SHIFT_MAX:
            continue  # underflows to zero
        if need_shift < SHIFT_MIN:
            raise ValueError(
                f"requant multiplier {v} is too large: needs shift {need_shift} "
                f"< {SHIFT_MIN}.  Check the calibrated output scale."
            )
        m, s = quantize_multiplier(float(v))
        mult[i] = m
        shift[i] = s
    return mult, shift


def requant_params(
    in_scale: float, weight_scales: Optional[np.ndarray], out_scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """``(multiplier, shift)`` for ``real = in_scale * weight_scale / out_scale``.

    Pass ``weight_scales=None`` for a plain per-tensor rescale.
    """
    if weight_scales is None:
        real = np.array([float(in_scale) / float(out_scale)])
    else:
        real = np.asarray(weight_scales, dtype=np.float64) * float(in_scale) / float(out_scale)
    return safe_quantize_multiplier(real)


# ---------------------------------------------------------------------------
# Conv + BatchNorm folding
# ---------------------------------------------------------------------------


@dataclass
class FoldedConv:
    weight: np.ndarray  # float64, same layout as the input weight
    bias: np.ndarray    # float64, shape [out_channels]


def fold_conv_bn(
    w: np.ndarray,
    b: Optional[np.ndarray],
    bn_gamma: np.ndarray,
    bn_beta: np.ndarray,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    bn_eps: float,
    out_axis: int = 0,
) -> FoldedConv:
    """Fold a BatchNorm into the preceding convolution's weights and bias.

    ``y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta`` becomes a plain
    convolution with::

        f  = gamma / sqrt(var + eps)
        W' = W * f            (broadcast along the output-channel axis)
        b' = (b - mean) * f + beta

    Done directly on the float arrays -- deliberately not via
    ``torch.ao.quantization.fuse_modules``, whose graph-mode variants are
    unreliable in torch 2.2.
    """
    w = np.asarray(w, dtype=np.float64)
    oc = w.shape[out_axis]
    if b is None:
        b = np.zeros(oc, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    f = np.asarray(bn_gamma, dtype=np.float64) / np.sqrt(
        np.asarray(bn_var, dtype=np.float64) + float(bn_eps)
    )
    shape = [1] * w.ndim
    shape[out_axis] = oc
    w2 = w * f.reshape(shape)
    b2 = (b - np.asarray(bn_mean, dtype=np.float64)) * f + np.asarray(bn_beta, dtype=np.float64)
    return FoldedConv(w2, b2)
