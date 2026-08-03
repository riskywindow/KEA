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

The invariant, in the two algorithms' own units:

| Parameter | Constraint |
| --- | --- |
| `mult` | `[2^30, 2^31)` — normalised Q31, top bit set |
| TOSA `shift` | `>= 31` |
| `KeaQuantParam.shift` | `>= 0` |

The shift conventions differ by exactly 31, because `keaSrdhm` already performs
a `>>31` that TOSA's formulation carries explicitly:

```
kea_shift = tosa_shift - 31
```

Measured over 400,000 random cases on the normalised domain: **0 divergences.**
Measured with an un-normalised multiplier (`mult < 2^30`) but `shift >= 31`:
**0 divergences** — normalising the multiplier is good hygiene but is not what
carries the equivalence. Measured with `tosa_shift < 31`, i.e. a *negative*
`KeaQuantParam.shift`: **~48% divergence.**

So the load-bearing half of the invariant is `KeaQuantParam.shift >= 0`. The
negative-shift path in `keaRequantize` — the pre-multiply left shift — has no
TOSA counterpart and is where the two designs part company.

## Consequences

- `kea-as` rejects a `KeaQuantParam` with `shift < 0`, and the compiler backend
  must normalise before emitting. A quantization scale that would require a
  negative shift means the output scale is larger than the accumulator scale,
  which for a real quantized network means something upstream is wrong; failing
  loudly beats silently producing a stream the golden model can't reproduce.
- `tests/invariants/test_requant_equivalence.cpp` re-proves the equivalence on
  every build, and asserts divergence outside the domain so the test cannot
  quietly become vacuous.
- `keaRequantize`'s negative-shift path stays in the ISA — it is legal hardware
  behaviour, and a future non-TOSA frontend could use it. It is simply outside
  the domain this compiler targets.

## Why not just make them the same function?

Changing `VQUANT` to TOSA semantics would make the hardware's rounding depend on
a software framework's choice, and half-up-toward-+∞ is a worse rounding rule
for a datapath (it biases positive). Changing the frontend to gemmlowp would
break bit-exactness against TOSA, which is the interchange format we ingest.
Constraining the domain keeps both correct in their own terms and costs nothing,
because every parameter a real quantized network produces already lands inside
it.
