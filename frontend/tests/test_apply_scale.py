"""apply_scale_32: the normative requantization primitive.

If anything in this file fails, the simulator, the compiler and the hardware
model are all wrong together, because they are all validated against it.
"""

import json
import os

import numpy as np
import pytest

from kea_frontend.apply_scale import (
    INT32_MAX,
    INT32_MIN,
    SHIFT_MAX,
    SHIFT_MIN,
    apply_scale_32,
    multiply_by_quantized_multiplier,
    quantize_multiplier,
    rounding_divide_by_pot,
    saturating_rounding_doubling_high_mul,
)

VECTORS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata",
                       "apply_scale_vectors.json")


@pytest.fixture(scope="module")
def vectors():
    with open(VECTORS) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# The published vector file
# ---------------------------------------------------------------------------


def test_vector_file_matches_implementation(vectors):
    """Every published case must reproduce.  This is the C++ team's contract."""
    cases = np.array(vectors["cases"], dtype=np.int64)
    assert len(cases) > 3000, "vector file suspiciously small"
    for dr in (0, 1):
        sel = cases[cases[:, 3] == dr]
        got = apply_scale_32(sel[:, 0], sel[:, 1], sel[:, 2], bool(dr), strict=False)
        bad = np.flatnonzero(got.astype(np.int64) != sel[:, 4])
        assert bad.size == 0, (
            f"{bad.size} mismatches, first: value={sel[bad[0], 0]} "
            f"multiplier={sel[bad[0], 1]} shift={sel[bad[0], 2]} "
            f"double_round={dr} expected={sel[bad[0], 4]} got={got[bad[0]]}"
        )


def test_vector_file_coverage(vectors):
    """The vectors must actually straddle the shift==31 boundary in both signs."""
    cov = vectors["coverage"]
    assert cov["shift_le_31"] > 1000
    assert cov["shift_gt_31"] > 1000
    assert cov["shift_gt_31_value_negative"] > 500
    assert cov["shift_gt_31_value_nonnegative"] > 500
    # cases where double_round actually changes the answer
    assert cov["double_round_changes_result"] > 50
    # cases exercising the wrapping (non-saturating) truncate
    assert cov["result_would_overflow_int32_before_wrap"] > 100


# ---------------------------------------------------------------------------
# Values measured from MLIR 20.1.6 itself
# ---------------------------------------------------------------------------

#: (value, multiplier, shift, double_round_result, single_round_result),
#: obtained by lowering `tosa.apply_scale` with
#: `--tosa-to-arith='include-apply-rescale=true'` and executing it with
#: `mlir-runner` on this machine.
MLIR_MEASURED = [
    (1, 1073741824, 32, 1, 0),
    (-1, 2147483647, 32, -1, 0),
    (5, 2147483647, 32, 3, 2),
    (-5, 2147483647, 32, -3, -2),
    (12345, 1073741824, 32, 3087, 3086),
    (-12345, 2147483647, 32, -6173, -6172),
    (2147483647, 1073741824, 62, 1, 0),
    (-2147483648, 2147483647, 32, -1073741824, -1073741823),
    (-2147483648, 2147483647, 31, -2147483647, -2147483647),
    (1073741824, 2147483647, 2, -268435456, -268435456),
    (2147483647, 2147483647, 30, -4, -4),
    (-1073741824, 2147483647, 15, 32768, 32768),
    (-3, 1, 1, -1, -1),
    (-5, 3, 1, -7, -7),
    (7, 1, 0, 7, 7),
    (-7, 3, 0, -21, -21),
]


@pytest.mark.parametrize("value,mult,shift,dr,sr", MLIR_MEASURED)
def test_matches_mlir_20_1_6(value, mult, shift, dr, sr):
    assert int(apply_scale_32(value, mult, shift, True, strict=False)) == dr
    assert int(apply_scale_32(value, mult, shift, False, strict=False)) == sr


# ---------------------------------------------------------------------------
# Semantic properties that are easy to get wrong
# ---------------------------------------------------------------------------


def test_rounding_is_half_up_not_half_away_from_zero():
    """-1.5 must requantize to -1, not -2.  The classic divergence."""
    # multiplier=1, shift=1 => result = floor((v + 1) / 2)
    assert int(apply_scale_32(-3, 1, 1, False, strict=False)) == -1   # -1.5 -> -1
    assert int(apply_scale_32(3, 1, 1, False, strict=False)) == 2     # +1.5 -> +2
    assert int(apply_scale_32(-1, 1, 1, False, strict=False)) == 0    # -0.5 -> 0
    assert int(apply_scale_32(1, 1, 1, False, strict=False)) == 1     # +0.5 -> +1


def test_double_round_is_a_noop_at_or_below_shift_31():
    rng = np.random.default_rng(11)
    v = rng.integers(INT32_MIN, INT32_MAX, size=4000, dtype=np.int64)
    m = rng.integers(0, INT32_MAX, size=4000, dtype=np.int64)
    for s in range(0, 32):
        a = apply_scale_32(v, m, s, True, strict=False)
        b = apply_scale_32(v, m, s, False, strict=False)
        assert np.array_equal(a, b), f"double_round changed the result at shift={s}"


def test_double_round_does_something_above_shift_31():
    diff = 0
    rng = np.random.default_rng(12)
    v = rng.integers(INT32_MIN, INT32_MAX, size=4000, dtype=np.int64)
    m = rng.integers(1 << 30, INT32_MAX, size=4000, dtype=np.int64)
    for s in (32, 40, 62):
        a = apply_scale_32(v, m, s, True, strict=False)
        b = apply_scale_32(v, m, s, False, strict=False)
        diff += int((a != b).sum())
    assert diff > 0, "double_round must matter for shift > 31"


def test_double_round_bias_keys_on_sign_of_value_not_product():
    """The correction's sign follows `value`, never `value * multiplier`.

    With ``multiplier >= 0`` (all KEA ever emits) the two rules coincide, so the
    only way to pin the distinction is a negative multiplier -- legal for the
    primitive, forbidden by TOSA's REQUIRE, hence ``strict=False``.

    Value measured from MLIR 20.1.6:
      apply_scale(-2**30, -2, shift=32, double_round=true) == 0
    Keying the bias off the (positive) product would give 1.
    """
    got = int(apply_scale_32(-(1 << 30), -2, 32, True, strict=False))
    assert got == 0, got

    # Show the alternative rule really would differ, so this is a live check.
    prod = (-(1 << 30)) * (-2)
    bias_by_product = ((1 << 32) >> 1) + (1 << 30)   # product >= 0
    assert (prod + bias_by_product) >> 32 == 1

    # And the primitive rejects a negative multiplier in strict mode.
    with pytest.raises(ValueError):
        apply_scale_32(-(1 << 30), -2, 32, True, strict=True)


def test_final_narrowing_wraps_and_does_not_saturate():
    # 2**30 * (2**31-1) >> 2 is far outside int32 and must wrap, matching
    # MLIR's arith.trunci.
    got = int(apply_scale_32(1073741824, INT32_MAX, 2, False, strict=False))
    assert got == -268435456
    with pytest.raises(OverflowError):
        apply_scale_32(1073741824, INT32_MAX, 2, False, strict=True)


def test_strict_mode_enforces_tosa_requires():
    with pytest.raises(ValueError):
        apply_scale_32(1, -1, 31, True)                      # multiplier < 0
    with pytest.raises(ValueError):
        apply_scale_32(1, 1 << 30, SHIFT_MIN - 1, True)      # shift too small
    with pytest.raises(ValueError):
        apply_scale_32(1, 1 << 30, SHIFT_MAX + 1, True)      # shift too large
    with pytest.raises(ValueError):
        apply_scale_32(1, 1 << 30, 64, True, strict=False)   # outside defined domain


def test_strict_never_changes_a_returned_value():
    rng = np.random.default_rng(13)
    for _ in range(2000):
        v = int(rng.integers(INT32_MIN, INT32_MAX, dtype=np.int64))
        m = int(rng.integers(0, INT32_MAX, dtype=np.int64))
        s = int(rng.integers(SHIFT_MIN, SHIFT_MAX + 1))
        dr = bool(rng.integers(0, 2))
        loose = int(apply_scale_32(v, m, s, dr, strict=False))
        try:
            strict = int(apply_scale_32(v, m, s, dr, strict=True))
        except OverflowError:
            continue
        assert strict == loose


def test_broadcasting_matches_elementwise():
    rng = np.random.default_rng(14)
    x = rng.integers(-(1 << 20), 1 << 20, size=(2, 3, 8), dtype=np.int64)
    m = rng.integers(1 << 30, INT32_MAX, size=8, dtype=np.int64)
    s = rng.integers(31, 40, size=8, dtype=np.int64)
    got = apply_scale_32(x, m, s, True, strict=False)
    for c in range(8):
        exp = apply_scale_32(x[..., c], int(m[c]), int(s[c]), True, strict=False)
        assert np.array_equal(got[..., c], exp)


# ---------------------------------------------------------------------------
# quantize_multiplier
# ---------------------------------------------------------------------------


def test_quantize_multiplier_normalised_and_accurate():
    rng = np.random.default_rng(15)
    for _ in range(3000):
        real = float(np.exp(rng.uniform(np.log(1e-9), np.log(0.99))))
        m, s = quantize_multiplier(real)
        assert (1 << 30) <= m < (1 << 31), (real, m, s)
        assert SHIFT_MIN <= s <= SHIFT_MAX
        approx = m * 2.0 ** -s
        assert abs(approx - real) <= real * 2 ** -30


def test_quantize_multiplier_zero():
    assert quantize_multiplier(0.0) == (0, SHIFT_MIN)


def test_apply_scale_approximates_real_multiplication():
    """The whole point: apply_scale(v, *quantize_multiplier(r)) ~= v * r."""
    rng = np.random.default_rng(16)
    for _ in range(2000):
        real = float(np.exp(rng.uniform(np.log(1e-6), np.log(0.5))))
        m, s = quantize_multiplier(real)
        v = int(rng.integers(-(1 << 28), 1 << 28))
        got = int(apply_scale_32(v, m, s, True, strict=False))
        assert abs(got - v * real) <= 1.0 + abs(v) * real * 2 ** -29


# ---------------------------------------------------------------------------
# Relationship to gemmlowp / TFLite  (documentation, not a requirement)
# ---------------------------------------------------------------------------


def test_gemmlowp_equivalence_on_normalised_domain():
    """double_round=True tracks gemmlowp on the domain KEA actually emits.

    This is an empirical property, recorded so that a future change which
    breaks it is noticed.  KEA is defined by apply_scale_32 regardless.
    """
    rng = np.random.default_rng(17)
    total = differ = 0
    for s in range(31, 63):
        v = rng.integers(INT32_MIN, INT32_MAX, size=20000, dtype=np.int64)
        m = rng.integers(1 << 30, 1 << 31, size=20000, dtype=np.int64)
        a = apply_scale_32(v, m, s, True, strict=False).astype(np.int64)
        g = multiply_by_quantized_multiplier(v, m, 31 - s).astype(np.int64)
        total += v.size
        differ += int((a != g).sum())
    assert total > 500000
    assert differ == 0, f"{differ}/{total} divergences from gemmlowp appeared"


def test_single_round_does_diverge_from_gemmlowp():
    """The converse: double_round=False is NOT gemmlowp.  Guards against
    someone 'simplifying' the flag away."""
    rng = np.random.default_rng(18)
    v = rng.integers(INT32_MIN, INT32_MAX, size=20000, dtype=np.int64)
    m = rng.integers(1 << 30, 1 << 31, size=20000, dtype=np.int64)
    b = apply_scale_32(v, m, 40, False, strict=False).astype(np.int64)
    g = multiply_by_quantized_multiplier(v, m, 31 - 40).astype(np.int64)
    assert int((b != g).sum()) > 0


def test_gemmlowp_helpers_self_consistent():
    assert int(saturating_rounding_doubling_high_mul(INT32_MIN, INT32_MIN)) == INT32_MAX
    assert int(saturating_rounding_doubling_high_mul(1 << 30, 1 << 30)) == 1 << 29
    assert int(rounding_divide_by_pot(5, 1)) == 3       # half away from zero
    assert int(rounding_divide_by_pot(-5, 1)) == -3
    assert int(rounding_divide_by_pot(4, 1)) == 2
