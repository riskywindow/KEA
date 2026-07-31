# KEA quantization & requantization

**This document is normative.** It is written so that `apply_scale_32` and the
whole int8 pipeline can be reimplemented in C++ from this text alone, without
reading the Python. Where the Python and this document disagree, that is a bug —
report it.

| | |
|---|---|
| Reference implementation | `frontend/kea_frontend/apply_scale.py` |
| Machine-checkable vectors | `frontend/testdata/apply_scale_vectors.json` (18,506 cases) |
| Tests | `frontend/tests/test_apply_scale.py` |
| Verified against | MLIR/LLVM **20.1.6**, `--tosa-to-arith='include-apply-rescale=true'`, executed with `mlir-runner` |

---

## 0. The one-paragraph version

KEA is **TOSA-semantics, not TFLite/gemmlowp-semantics.** Requantization is a
single 64-bit multiply-accumulate with a pre-added rounding term and one
arithmetic right shift. It rounds **half-up (toward +infinity)**, not
half-away-from-zero. The `double_round` correction is a **no-op unless
`shift > 31`**, and when it applies, its sign follows the sign of `value`, not
the sign of `value * multiplier`. The final narrowing to int32 **wraps**; the
subsequent clamp to the output element type **saturates**. If you are porting
kernels from a TFLite background, every one of those sentences contradicts what
you are used to.

---

## 1. `apply_scale_32` — the primitive

```c
// value      : int32, the accumulator (already zero-point corrected)
// multiplier : int32, Q0.31.  TOSA REQUIREs multiplier >= 0.
// shift      : the exponent.  TOSA REQUIREs 2 <= shift <= 62.
//              Defined here over [0, 63]; >= 64 is poison.
// double_round: enables the correction term; no effect when shift <= 31.
int32_t apply_scale_32(int32_t value, int32_t multiplier,
                       int32_t shift, bool double_round)
{
    // (1) rounding term.  NOTE: (1 << shift) >> 1 with a LOGICAL right shift,
    //     which equals 1 << (shift-1) for shift >= 1 but is also well-defined
    //     at shift == 0 (giving 0) and at shift == 63 (giving 2^62).
    int64_t round = (int64_t)(((uint64_t)1 << shift) >> 1);

    // (2) the double-rounding correction.  ONLY when shift > 31.
    //     The sign follows `value`, NOT the sign of value*multiplier.
    if (double_round && shift > 31)
        round += (value >= 0) ? (int64_t)(1 << 30) : -(int64_t)(1 << 30);

    // (3) one 64-bit multiply-accumulate
    int64_t prod = (int64_t)value * (int64_t)multiplier + round;

    // (4) ARITHMETIC (sign-propagating) right shift, then a WRAPPING
    //     truncation to 32 bits.  This does NOT saturate.
    return (int32_t)(prod >> shift);
}
```

That is the whole algorithm. Four steps, no loops, no second rounding.

### 1.1 Why the rounding is half-up

Step (4) is an arithmetic shift, which **floors**. Adding `2^(shift-1)` first
turns floor into round-half-up. So a value exactly halfway between two outputs
rounds toward **positive infinity**:

| exact quotient | `apply_scale_32` | half-away-from-zero (gemmlowp) |
|---|---|---|
| `+1.5` | `+2` | `+2` |
| `-1.5` | **`-1`** | `-2` |
| `-0.5` | **`0`** | `-1` |

Concretely, with `multiplier = 1, shift = 1`:
`apply_scale_32(-3, 1, 1, *) == -1`, `apply_scale_32(-1, 1, 1, *) == 0`.

**This is the single most common source of silent mismatch.** A C++ port that
uses `std::round`, or that special-cases negatives, will be wrong on roughly
half of all exact ties.

### 1.2 Why `double_round` only matters above shift 31

MLIR's lowering emits `arith.cmpi sgt` against 31 on the (zero-extended) shift
and selects between the corrected and uncorrected accumulator. For
`shift <= 31` the branch is not taken, so the flag has **no observable effect**.
Verified exhaustively over shifts 0–31 in
`test_double_round_is_a_noop_at_or_below_shift_31`.

Since KEA emits Q0.31 multipliers, `shift` lands in `[2, 62]` and both branches
occur in real graphs — a rescale with a real scale below `2^-1` uses
`shift > 31`. Both paths matter.

### 1.3 Why the correction keys on `sign(value)`

With `multiplier >= 0` (which TOSA requires and KEA always satisfies),
`sign(value)` and `sign(value * multiplier)` agree, so the distinction is
invisible. It becomes visible only with a negative multiplier, where the
correction is applied in the *wrong* direction relative to the result's sign —
which is precisely why TOSA forbids negative multipliers. Measured from MLIR:

```
apply_scale_32(-2^30, -2, shift=32, double_round=true) == 0
```

Keying the bias off the (positive) product would give `1`.

### 1.4 Overflow

The final `(int32_t)` cast is a **wrapping** truncation (`arith.trunci`), not a
saturation. Example, measured from MLIR:

```
apply_scale_32(1073741824, 2147483647, shift=2, *) == -268435456
```

KEA graphs are constructed so this never happens, and the reference
implementation's `strict=True` mode raises if it would. The vectors nonetheless
include 3,051 wrapping cases so that independent implementations agree even
outside the intended domain.

`shift >= 64` is **poison** — MLIR emits an out-of-range `arith.shli`/`shrsi`
and nothing clamps it. Do not rely on any behaviour there.

### 1.5 Divergence from MLIR's 32-bit lowering

`--tosa-to-arith` has a `use-32-bit=true` mode built on `arith.mulsi_extended`.
It is numerically identical to the default 64-bit lowering for every shift in
1–63, and **diverges only at `shift == 0`**, where it computes `1 << -1` and
produces garbage. KEA specifies the **64-bit lowering**. Do not implement the
32-bit variant.

---

## 2. Relationship to TFLite / gemmlowp

TFLite's `MultiplyByQuantizedMultiplier` is a *different algorithm*: it does
`SaturatingRoundingDoublingHighMul` (round half away from zero, with
saturation) followed by `RoundingDivideByPOT` (round half away from zero again).
It rounds **twice**; TOSA rounds once. TFLite's `shift` also uses the opposite
sign convention:

```
tflite_shift = 31 - tosa_shift
```

The TOSA `double_round` flag exists specifically to approximate gemmlowp's
second rounding. How well it does so was measured, not assumed:

| Domain | Cases | `double_round=true` differs | `double_round=false` differs |
|---|---|---|---|
| multiplier ∈ [2³⁰, 2³¹), tosa shift ∈ [31, 62], value uniform over int32 | 6,400,000 | **0** | **100,302 (1.57%)** |

So on the normalised domain KEA actually emits, `double_round=true` is
empirically indistinguishable from gemmlowp — which is **why KEA defaults to
`double_round=true`**, and why published TFLite-derived accuracy expectations
carry over. But:

- This is an empirical result on a large sample, **not a proof**.
- It says nothing about `shift < 31`, where TFLite needs a pre-left-shift.
- gemmlowp saturates where TOSA wraps.

**Implement `apply_scale_32`. Do not implement gemmlowp and assume it matches.**
`frontend/testdata/apply_scale_vectors.json` carries these numbers in its
`gemmlowp_bruteforce` field, and `test_single_round_does_diverge_from_gemmlowp`
guards against anyone deleting the flag.

---

## 3. The `rescale` operation

`apply_scale_32` is the kernel; `rescale` is the graph node that uses it.

```c
// per element; `c` is the index along the LAST dimension when per_channel
int32_t v      = (int32_t)input - input_zp;            // int32 arithmetic
int32_t scaled = apply_scale_32(v, multiplier[c], shift[c], double_round);
int32_t biased = scaled + output_zp;                   // WRAPPING int32 add
out            = clamp(biased, MIN_T, MAX_T);          // SATURATING to out dtype
```

Three details that are easy to get wrong:

1. **`+ output_zp` happens in wrapping int32, before the clamp.** Doing it in a
   wider type and then clamping disagrees whenever `scaled` is within
   `|output_zp|` of an int32 end. Rare, but a genuine bit-level divergence.
2. **The clamp saturates** to the output element type: int8 → `[-128, 127]`,
   int32 → the full range. This is the opposite of step (4) of `apply_scale_32`,
   which wraps. Both behaviours are correct in their own place.
3. **`per_channel` indexes the last dimension.** TOSA defines it that way and
   the KEA validator enforces `channel_axis == rank - 1`.

---

## 4. Deriving `(multiplier, shift)` from a real scale

Every requantization in a KEA graph starts life as a real number — a ratio of
tensor scales. The frontend converts it once, at export time, and **writes the
integers into the graph file**. A consumer never re-derives them and never needs
floating point.

```
Given real r > 0:
    (frac, exp) = frexp(r)            # r == frac * 2^exp, frac in [0.5, 1)
    multiplier  = round(frac * 2^31)  # lands in [2^30, 2^31]
    shift       = 31 - exp
    if multiplier == 2^31:            # frac rounded up to exactly 1.0
        multiplier >>= 1
        shift      -= 1
```

Guarantees: `multiplier ∈ [2^30, 2^31)` (normalised) and, for scales KEA
accepts, `shift ∈ [2, 62]`. Special cases:

- `r == 0`, or `r` so small it needs `shift > 62` → `(0, 62)`, i.e. the rescale
  produces 0. That is the correct answer.
- `r` needing `shift < 2` (i.e. `r >= 2^29`) is a **modelling error** and raises.
  It means an op wants to amplify its accumulator by half a billion.

Relative accuracy is better than `2^-30`, asserted in
`test_quantize_multiplier_normalised_and_accurate`.

---

## 5. Tensor quantization scheme

`real = scale * (q - zero_point)`.

| Tensor class | Scheme | Zero point | Scale |
|---|---|---|---|
| **Weights** (conv, depthwise, fc, matmul rhs) | per-**output-channel** symmetric int8 | **always 0** | `max|W[c]| / 127` |
| **Activations** (everything else) | per-**tensor** asymmetric int8 | calibrated, in `[-128, 127]` | `(hi - lo) / 255` |
| **Bias** | per-output-channel int32 | **always 0** | `act_scale * weight_scale[c]` — *derived, never observed* |

Enforced by `KGraph.validate()`; a per-channel tensor with a non-zero zero point
is rejected.

### 5.1 Weights

```
scale[c] = max(max|W[c]|, 1e-9) / 127
q        = clamp(round_half_even(W / scale[c]), -127, +127)
```

Note the clamp is to **±127**, not `[-128, 127]`. The representable range is then
exactly symmetric, which is what makes `weight_zp = 0` sound and lets a hardware
MAC assume a symmetric operand. Giving up the single value −128 costs nothing
measurable.

### 5.2 Activations

```
lo = min(observed_lo, 0)        # the range ALWAYS contains 0
hi = max(observed_hi, 0)
scale = (hi - lo) / 255
zp    = clamp(round_half_even(-128 - lo/scale), -128, 127)
```

Widening the range to include 0 is not cosmetic: convolution zero-padding
inserts the value `input_zp`, which must dequantize to exactly real 0, or every
padded border picks up a bias.

### 5.3 Bias

The bias is never calibrated. It lives in the accumulator's own units:

```
bias_scale[c] = input_scale * weight_scale[c]
bias_q[c]     = round_half_even(bias_float[c] / bias_scale[c])   # int32
```

So `conv_accumulator + bias_q` is exact — a single int32 add, no rescale.

### 5.4 The resulting requantization

For a convolution:

```
real_multiplier[c] = (input_scale * weight_scale[c]) / output_scale
(multiplier[c], shift[c]) = quantize_multiplier(real_multiplier[c])
```

which is exactly the per-channel `rescale` that follows every conv in the graph.

---

## 6. Calibration

Two observers, selectable per export; both are reported in the accuracy table.

### `minmax`
Running absolute min/max over the calibration set. One forward pass. No outlier
rejection, so a single extreme activation stretches the scale for every image.

### `percentile` (default, 99.99)
Clips the outer `100 - p` percent of the mass, split evenly between the tails.
Implemented as **two passes**:

1. fit absolute `[lo, hi]` with min/max observers;
2. build an exact 4096-bin histogram inside those bounds and read off the
   quantile.

Two passes deliberately avoid the range-expansion and re-binning heuristics that
make single-pass histogram observers implementation-defined. Calibration is a
few hundred images; the extra sweep is cheap.

Measured effect on MobileNetV2 (1000 Imagenette val images, 384 calibration
images) — see `docs/FRONTEND.md` §8 for the full table:

| Observer | int8 top-1 | argmax agreement with float |
|---|---|---|
| minmax | 67.00% | 87.80% |
| percentile 99.99 | **67.90%** | **91.30%** |

Percentile wins on both, which is why it is the default.

---

## 7. Conv + BatchNorm folding

Done on the float arrays before anything is quantized — deliberately **not** via
`torch.ao.quantization`, whose graph-mode fusion is unreliable in torch 2.2.

```
f  = gamma / sqrt(var + eps)
W' = W * f              # broadcast along the output-channel axis
b' = (b - mean) * f + beta
```

`test_bn_folding_is_exact` asserts the folded float model matches the original
torchvision model to < 2e-3 on the logits (float32 reassociation error only).

Folding must happen **before** weight-scale derivation: `f` varies by more than
an order of magnitude across channels in MobileNetV2, so folding afterwards
would leave the per-channel scales badly mismatched.

---

## 8. Integer helpers used by `softmax` and `layernorm`

Two primitives, specified once and shared, because inconsistent division
conventions are a classic source of divergence.

### `idiv_trunc(a, b)` — truncate toward zero

C/C++ `/` semantics. **Not** `floor`. numpy's `//` floors, so the reference
spells it out explicitly.

```
idiv_trunc( 7,  2) ==  3      idiv_trunc(-7,  2) == -3
idiv_trunc( 7, -2) == -3      idiv_trunc(-7, -2) ==  3
idiv_trunc(-1,  2) ==  0      # floor would give -1
```

### `isqrt(v)` — exact integer square root

The unique `r >= 0` with `r*r <= v < (r+1)*(r+1)`, for `v >= 0`. Uniquely
determined, so any method (Newton, binary search, hardware sqrt plus
correction) is conforming as long as it produces this value. A float64 `sqrt`
followed by correction steps is exact for the full non-negative int64 range KEA
uses; `test_isqrt_is_exact` checks 20,000 random values plus the boundaries.

---

## 9. Test vectors

`frontend/testdata/apply_scale_vectors.json`, regenerate with
`.venv/bin/python frontend/gen_apply_scale_vectors.py`.

```json
{
  "fields": ["value", "multiplier", "shift", "double_round", "result"],
  "cases": [[value, multiplier, shift, 0|1, result], ...]
}
```

18,506 cases. Coverage, all machine-counted and asserted by
`test_vector_file_coverage`:

| Property | Cases |
|---|---|
| `shift <= 31` | 10,070 |
| `shift > 31` | 8,436 |
| `shift > 31` **and** `value < 0` | 4,097 |
| `shift > 31` **and** `value >= 0` | 4,339 |
| where `double_round` changes the result | 152 |
| where the result wraps int32 | 3,051 |

The `shift <= 31` / `shift > 31` boundary is straddled densely in both signs of
`value` (shifts 28–35 × 19 values × 9 multipliers × both flags), because that
boundary is exactly where a C++ reimplementation diverges silently.

The file also carries the exhaustive tie grid, the realistic-scale grid, and the
gemmlowp comparison described in §2.

**To validate a C++ implementation:** load `cases`, call your `apply_scale_32`
on the first four fields, assert equality with the fifth. There is no tolerance;
it is exact integer equality or it is a bug.

---

## 10. What every implementation must agree on, bit for bit

1. `apply_scale_32` in full, including half-up rounding, the `shift > 31` gate
   on `double_round`, the `sign(value)` keying, and the wrapping truncate.
2. `rescale`: zero-point subtract in int32, `+ output_zp` in wrapping int32,
   saturating clamp to the output element type.
3. Accumulator widths: int32 for conv/depthwise/fc/matmul/pool; int64 for the
   `layernorm` sum-of-squares (see `docs/FRONTEND.md` §5).
4. `idiv_trunc` (toward zero) and `isqrt` (exact floor).
5. `add` saturates to int32 rather than wrapping.
6. Weight quantization clamped to ±127, not −128.

Everything else — layout, tiling, loop order, vectorisation — is free, because
all KEA integer ops are exactly reduction-order-independent within their stated
accumulator width.

### How to check you got it right

1. **Unit level:** `frontend/testdata/apply_scale_vectors.json` — 18,506 cases,
   exact integer equality.
2. **Whole-graph level:** `frontend/testdata/golden_io_*.npz` — fixed int8
   inputs and the exact int8 outputs of the reference interpreter for the two
   shipped models. Exact equality, no tolerance. See `docs/FRONTEND.md` §4.1.

Passing (1) and failing (2) means the primitive is right but something in an op
kernel, an accumulator width, or the rescale tail is not.
