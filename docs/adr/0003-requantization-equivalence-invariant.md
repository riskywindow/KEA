# ADR-0003: The normalised-multiplier invariant

**Status:** accepted
**Date:** 2026-08-01

## Context

There are two different requantization algorithms in this project, and both are
correct:

- **`keaRequantize()` in `include/kea/isa.h`** is what the `VQUANT` instruction
  does. It is the gemmlowp/TFLite formulation: a saturating rounding doubling
  high multiply (`keaSrdhm`, which absorbs a `>>31`) followed by a rounding
  power-of-two divide (`keaRdpot`). It rounds half-away-from-zero and saturates.
- **TOSA `apply_scale_32`**, specified in `docs/QUANTIZATION.md`, is what the
  frontend's golden numpy reference does. It is a single 64-bit
  multiply-accumulate with a pre-added rounding term. It rounds **half-up
  toward +∞** and the final narrow **wraps**.

These are not the same function. On unconstrained inputs they disagree
constantly — `-1.5` requantizes to `-2` under gemmlowp and `-1` under TOSA.

This matters because the two meet at the end-to-end demo: the compiler derives
`KeaQuantParam.mult`/`.shift` from the frontend's scales, the simulator executes
`VQUANT` via `keaRequantize`, and the result is compared against the frontend's
TOSA-semantics golden output. If they disagree, the demo produces wrong
classifications for reasons that look like a kernel bug and are not.

## Decision

**The compiler must only ever emit normalised quantization parameters, and on
that domain the two algorithms are provably identical.**

The shift conventions differ by exactly 31, because `keaSrdhm` already performs
a `>>31` that TOSA's formulation carries explicitly:

```
kea_shift = tosa_shift - 31
```

**The invariant:**

> For every requantization the compiler emits, either `tosa_shift >= 31`, or the
> accumulator satisfies `|acc + bias| < 2^tosa_shift`.

Measured evidence:

| domain | divergences |
| --- | --- |
| normalised: `mult ∈ [2^30,2^31)`, `tosa_shift ∈ [31,62]` | **0 / 400,000** |
| un-normalised `mult < 2^30`, `tosa_shift >= 31` | **0 / 200,000** |
| `tosa_shift = 22` (`kea_shift = -9`), `|v| < 2^20` | **0 / 200,000** |
| `tosa_shift = 22` (`kea_shift = -9`), `|v| < 2^24` | 79,048 / 200,000 (39.5%) |
| `tosa_shift ∈ {25,28,30}`, `|v| < 2^24` | **0 / 200,000** each |

Normalising the multiplier is hygiene; it is not what carries the equivalence.

### Why `shift < 31` is usually fine

A negative `KeaQuantParam.shift` selects `keaRequantize`'s pre-multiply left
shift, `v << (31 - tosa_shift)`, which has no TOSA counterpart. But it is
*numerically exact* until that shift overflows int32 — which is precisely when
`|v| >= 2^tosa_shift`. That is why `tosa_shift = 22` is exact at `|v| < 2^20`
and 39.5% wrong at `|v| < 2^24`, while `tosa_shift = 25` is exact at both.

This matters because real models need it. MobileNetV2's rescale shifts, as
emitted by our frontend, range over **22 to 48** — a blanket `tosa_shift >= 31`
rule would reject the network we are built to run. A rescale with
`tosa_shift < 31` is one whose scale factor exceeds 1, which is entirely normal
where an operand's scale is much finer than its consumer's.

## Consequences

- **The compiler backend must carry an accumulator range bound** for every
  requantization it emits, and check it against the invariant. It already needs
  such a bound for errata E5 (int32 ACC overflow), so this is one bound serving
  two purposes. When `tosa_shift < 31` and the bound cannot establish
  `|acc| < 2^tosa_shift`, the backend must fail loudly rather than emit a stream
  the golden model cannot reproduce.
- `tests/invariants/test_requant_equivalence.cpp` re-proves this on every build,
  including the exactness of the negative-shift path within its range and its
  failure outside — so the test cannot quietly become vacuous if someone makes
  the two algorithms trivially equal.
- A conservative fallback exists if a layer ever violates the bound: split the
  rescale into a `tosa_shift >= 31` requantization followed by an explicit
  scale-up, at the cost of an extra VPU pass. Not currently needed.

## Why not just make them the same function?

Changing `VQUANT` to TOSA semantics would make the hardware's rounding depend on
a software framework's choice, and half-up-toward-+∞ is a worse rounding rule
for a datapath (it biases positive). Changing the frontend to gemmlowp would
break bit-exactness against TOSA, which is the interchange format we ingest.
Constraining the domain keeps both correct in their own terms and costs nothing,
because every parameter a real quantized network produces already lands inside
it.
