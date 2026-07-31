"""Integer requantization primitives -- the bit-exact ground truth for KEA.

Everything downstream (the numpy reference interpreter, the C++ simulator, the
RTL/hardware model) must agree bit-for-bit with this module.  If you are
reimplementing this in C++, read ``docs/QUANTIZATION.md`` -- it is written to be
sufficient on its own -- and validate against
``frontend/testdata/apply_scale_vectors.json``.

Two algorithms live here:

``apply_scale_32``
    The TOSA ``apply_scale_32`` reference pseudocode.  This is the KEA normative
    semantics: the ``rescale`` node in a ``.kgraph.json`` means exactly this.

``multiply_by_quantized_multiplier``
    The gemmlowp / TFLite ``MultiplyByQuantizedMultiplier`` algorithm
    (saturating-rounding-doubling-high-multiply followed by a rounding right
    shift).  Provided for cross-checking and because published TFLite accuracy
    numbers are produced with it.  It is *not* what KEA executes.  See
    ``docs/QUANTIZATION.md`` for the (small, well-characterised) set of inputs on
    which the two disagree.

All functions are pure numpy, vectorised, and operate on int64 intermediates so
that no Python bignum arithmetic leaks in.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "INT32_MIN",
    "INT32_MAX",
    "SHIFT_MIN",
    "SHIFT_MAX",
    "apply_scale_32",
    "multiply_by_quantized_multiplier",
    "quantize_multiplier",
    "quantize_multiplier_tflite",
    "saturating_rounding_doubling_high_mul",
    "rounding_divide_by_pot",
    "clamp_to_int8",
    "clamp_to_uint8",
    "clamp_to_int32",
]

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1

#: Inclusive range of the ``shift`` operand the KEA frontend will ever *emit*.
#: Matches the TOSA spec ``REQUIRE(2 <= shift && shift <= 62)``.
SHIFT_MIN = 2
SHIFT_MAX = 62

#: Inclusive range over which :func:`apply_scale_32` is *defined* (and over
#: which the published test vectors range).  MLIR 20.1.6 takes ``shift`` as an
#: i8 that it zero-extends, so 0..63 are all well-defined in the 64-bit
#: lowering; 64..255 produce poison (out-of-range shift amounts) and are
#: excluded here.
SHIFT_DEFINED_MIN = 0
SHIFT_DEFINED_MAX = 63


def _as_i64(x) -> np.ndarray:
    """Coerce to an int64 ndarray without silently truncating Python ints."""
    a = np.asarray(x)
    if a.dtype.kind not in "iu":
        # Accept float only if it is exactly integral (helps callers that build
        # arrays from Python lists via np.array on mixed input).
        if not np.all(a == np.floor(a)):
            raise TypeError(f"apply_scale operands must be integral, got dtype {a.dtype}")
    return a.astype(np.int64)


# ---------------------------------------------------------------------------
# TOSA apply_scale_32  -- KEA normative semantics
# ---------------------------------------------------------------------------


def apply_scale_32(value, multiplier, shift, double_round: bool = True, strict: bool = True):
    """TOSA ``apply_scale_32``.  Bit-exact reference.

    Computes, in exact 64-bit integer arithmetic::

        round = int64( uint64(1) << shift ) >>u 1        # logical shift right
        if double_round and shift > 31:
            round += (1 << 30) if value >= 0 else -(1 << 30)
        prod   = int64(value) * int64(multiplier) + round
        result = int32_wrap( prod >>s shift )            # arithmetic shift, then
                                                         # truncate to 32 bits

    ``>>s`` is an **arithmetic** (sign-propagating, floor) shift right, so
    rounding is **half-up, toward +infinity** -- *not* half-away-from-zero.
    ``-1.5`` requantizes to ``-1``, not ``-2``.  This is the single most common
    place independent implementations disagree.

    The final narrowing is a **wrapping truncation** to 32 bits, not a
    saturation.  This matches MLIR 20.1.6's ``--tosa-to-arith`` lowering, which
    ends in ``arith.trunci``.  KEA graphs are constructed so this never
    overflows, and ``strict=True`` (the default) raises if it would.

    Parameters
    ----------
    value :
        int32 accumulator value, in ``[-2**31, 2**31-1]``.
    multiplier :
        int32 interpreted as Q0.31: the real scale being applied is
        ``multiplier * 2**-shift``.  TOSA REQUIREs ``multiplier >= 0`` and the
        KEA frontend never emits a negative one.  Beware: with a negative
        multiplier the ``double_round`` bias keys on ``sign(value)``, so it is
        applied in the *wrong direction* relative to the result's sign.  That
        is faithful to the spec and to MLIR, and is exactly why the REQUIRE
        exists.
    shift :
        Integer.  KEA emits ``[2, 62]``; the function is defined over
        ``[0, 63]`` (matching MLIR's zero-extended i8 operand in its 64-bit
        lowering).  ``shift == 0`` gives ``round == 0``.  ``shift >= 64`` is
        undefined (poison in MLIR) and rejected unconditionally.
    double_round :
        Enables the TOSA double-rounding correction term.  It has **no effect
        unless ``shift > 31``**.  KEA records the value per graph, never
        implies it.
    strict :
        When true, enforce the TOSA ``REQUIRE``s (``multiplier >= 0``,
        ``2 <= shift <= 62``, ``value`` and the result fit int32) and raise on
        violation.  When false, reproduce MLIR's raw lowering over the whole
        defined domain, wrapping included.  ``strict`` never changes a returned
        value -- it only turns some inputs into exceptions.

    Notes
    -----
    Broadcasting follows numpy rules, so per-output-channel multipliers can be
    applied to a whole tensor in one call.
    """
    v = _as_i64(value)
    m = _as_i64(multiplier)
    s = _as_i64(shift)

    if np.any(s < SHIFT_DEFINED_MIN) or np.any(s > SHIFT_DEFINED_MAX):
        raise ValueError(
            f"apply_scale_32: shift must be in "
            f"[{SHIFT_DEFINED_MIN}, {SHIFT_DEFINED_MAX}]; outside that range "
            "MLIR's lowering produces poison"
        )
    if np.any(m < INT32_MIN) or np.any(m > INT32_MAX):
        raise ValueError("apply_scale_32: multiplier must fit in int32")
    if np.any(v < INT32_MIN) or np.any(v > INT32_MAX):
        raise ValueError("apply_scale_32: value must fit in int32")
    if strict:
        if np.any(m < 0):
            raise ValueError("apply_scale_32: multiplier must be >= 0 (TOSA REQUIRE)")
        if np.any(s < SHIFT_MIN) or np.any(s > SHIFT_MAX):
            raise ValueError(
                f"apply_scale_32: shift must be in [{SHIFT_MIN}, {SHIFT_MAX}] (TOSA REQUIRE)"
            )

    # round = (uint64(1) << shift) >>u 1.  Equal to 1 << (shift-1) for shift>=1
    # but also well-defined at shift==0 (-> 0) and shift==63 (-> 2**62).
    rnd = np.right_shift(
        np.left_shift(np.uint64(1), s.astype(np.uint64)), np.uint64(1)
    ).astype(np.int64)

    if double_round:
        # Applied only when shift > 31; the sign follows `value`, NOT the
        # sign of value*multiplier.
        corr = np.where(v >= 0, np.int64(1) << 30, -(np.int64(1) << 30))
        rnd = rnd + np.where(s > 31, corr, np.int64(0))

    # value * multiplier cannot overflow int64: |v| <= 2**31, |m| <= 2**31 =>
    # |v*m| <= 2**62.  Adding round (<= 2**62 + 2**30) can reach ~2**63 only
    # for shift == 63, where the product side is correspondingly small in
    # practice; we do the add in int64 exactly as the lowering does.
    with np.errstate(over="ignore"):
        prod = v * m + rnd
        # Arithmetic shift right.  numpy's >> on signed int64 is arithmetic.
        result = np.right_shift(prod, s)

    if strict and (np.any(result < INT32_MIN) or np.any(result > INT32_MAX)):
        raise OverflowError(
            "apply_scale_32: result does not fit in int32; MLIR would wrap. "
            "Pass strict=False to reproduce the wrap."
        )

    # Wrapping truncation to 32 bits (arith.trunci), NOT saturation.
    out = _wrap_i32(result)
    return out if out.ndim else np.int32(out)


def _wrap_i32(x: np.ndarray) -> np.ndarray:
    """Truncate int64 -> int32 with two's-complement wraparound."""
    with np.errstate(over="ignore"):
        u = (np.asarray(x, dtype=np.int64) & np.int64(0xFFFFFFFF)).astype(np.uint32)
    return u.astype(np.int32)


# ---------------------------------------------------------------------------
# gemmlowp / TFLite  -- cross-check only
# ---------------------------------------------------------------------------


def saturating_rounding_doubling_high_mul(a, b):
    """gemmlowp ``SaturatingRoundingDoublingHighMul`` for int32.

    Returns ``sat_i32(round(a * b / 2**31))`` with ties rounded away from zero,
    and the single overflow case ``a == b == INT32_MIN`` saturating to
    ``INT32_MAX``.
    """
    a64 = _as_i64(a)
    b64 = _as_i64(b)
    overflow = (a64 == INT32_MIN) & (b64 == INT32_MIN)
    ab = a64 * b64
    # gemmlowp: nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30))
    nudge = np.where(ab >= 0, np.int64(1) << 30, np.int64(1) - (np.int64(1) << 30))
    num = ab + nudge
    # C++ integer division truncates toward zero; numpy's >> floors.
    quo = np.where(num >= 0, num >> 31, -((-num) >> 31))
    res = np.where(overflow, np.int64(INT32_MAX), quo)
    return np.clip(res, INT32_MIN, INT32_MAX).astype(np.int32)


def rounding_divide_by_pot(x, exponent):
    """gemmlowp ``RoundingDivideByPOT``: divide by ``2**exponent``, round half away from zero."""
    x64 = _as_i64(x)
    e = _as_i64(exponent)
    if np.any(e < 0) or np.any(e > 31):
        raise ValueError("rounding_divide_by_pot: exponent must be in [0, 31]")
    mask = (np.int64(1) << e) - 1
    remainder = x64 & mask
    threshold = (mask >> 1) + np.where(x64 < 0, np.int64(1), np.int64(0))
    return ((x64 >> e) + (remainder > threshold).astype(np.int64)).astype(np.int32)


def multiply_by_quantized_multiplier(value, multiplier, shift):
    """TFLite ``MultiplyByQuantizedMultiplier``.

    Here ``shift`` uses the **TFLite sign convention**: positive means left
    shift the input first, negative means right shift after the high-mul.  That
    is the opposite of the TOSA ``shift`` convention.  The relationship is::

        tflite_shift = 31 - tosa_shift
    """
    v = _as_i64(value)
    s_arr = _as_i64(shift)
    left = np.maximum(s_arr, 0)
    right = np.maximum(-s_arr, 0)
    shifted = v * (np.int64(1) << left)
    hi = saturating_rounding_doubling_high_mul(shifted, _as_i64(multiplier))
    return rounding_divide_by_pot(hi, right)


# ---------------------------------------------------------------------------
# Deriving (multiplier, shift) from a real scale
# ---------------------------------------------------------------------------


def quantize_multiplier(real_multiplier):
    """Decompose a positive real scale into TOSA ``(multiplier, shift)``.

    Returns ``(multiplier, shift)`` such that ``multiplier * 2**-shift``
    approximates ``real_multiplier`` with ``multiplier`` an int32 in
    ``[2**30, 2**31-1]`` (i.e. the Q0.31 mantissa is normalised) and
    ``shift`` in ``[SHIFT_MIN, SHIFT_MAX]``.

    ``real_multiplier == 0`` maps to ``(0, SHIFT_MIN)``.

    This is deterministic and must be reproduced exactly by any other producer
    of KEA graphs -- but note that the *graph file records the resulting
    integers*, so consumers never re-derive them.  This function only runs in
    the frontend.
    """
    arr = np.asarray(real_multiplier, dtype=np.float64)
    scalar = arr.ndim == 0
    arr = np.atleast_1d(arr)

    if np.any(arr < 0):
        raise ValueError("quantize_multiplier: real_multiplier must be >= 0")
    if np.any(~np.isfinite(arr)):
        raise ValueError("quantize_multiplier: real_multiplier must be finite")

    mult = np.zeros(arr.shape, dtype=np.int64)
    shift = np.full(arr.shape, SHIFT_MIN, dtype=np.int64)

    nz = arr > 0
    if np.any(nz):
        x = arr[nz]
        # frexp: x = frac * 2**exp with frac in [0.5, 1)
        frac, exp = np.frexp(x)
        # We want x = m * 2**-s with m in [2**30, 2**31).
        # x = frac * 2**exp = (frac * 2**31) * 2**(exp-31)  => s = 31 - exp
        m = np.round(frac * np.float64(1 << 31)).astype(np.int64)
        s = np.int64(31) - exp.astype(np.int64)
        # frac can round up to exactly 2**31; renormalise.
        bump = m >= (np.int64(1) << 31)
        m = np.where(bump, m >> 1, m)
        s = np.where(bump, s - 1, s)

        if np.any(s > SHIFT_MAX):
            raise ValueError(
                "quantize_multiplier: real_multiplier too small to represent "
                f"(needs shift {int(s.max())} > {SHIFT_MAX})"
            )
        if np.any(s < SHIFT_MIN):
            raise ValueError(
                "quantize_multiplier: real_multiplier too large to represent "
                f"(needs shift {int(s.min())} < {SHIFT_MIN}); the graph must "
                "not require a rescale with real scale >= 2**29"
            )
        mult[nz] = m
        shift[nz] = s

    if scalar:
        return int(mult[0]), int(shift[0])
    return mult.astype(np.int32), shift.astype(np.int32)


def quantize_multiplier_tflite(real_multiplier):
    """TFLite-convention ``(multiplier, shift)`` for the same real scale.

    ``shift`` here is ``31 - tosa_shift`` (positive = pre-left-shift).
    """
    m, s = quantize_multiplier(real_multiplier)
    if isinstance(m, int):
        return m, 31 - s
    return m, (np.int32(31) - s).astype(np.int32)


# ---------------------------------------------------------------------------
# Saturation helpers
# ---------------------------------------------------------------------------


def clamp_to_int8(x):
    return np.clip(_as_i64(x), -128, 127).astype(np.int8)


def clamp_to_uint8(x):
    return np.clip(_as_i64(x), 0, 255).astype(np.uint8)


def clamp_to_int32(x):
    return np.clip(_as_i64(x), INT32_MIN, INT32_MAX).astype(np.int32)
