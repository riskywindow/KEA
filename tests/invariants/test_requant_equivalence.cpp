// Proves ADR-0003: on the normalised domain, the VQUANT instruction's
// requantization (gemmlowp, via isa.h's keaRequantize) and the frontend golden
// model's requantization (TOSA apply_scale_32) are the same function.
//
// This is the seam where the hardware model and the reference model meet. They
// are genuinely different algorithms -- they only coincide because the compiler
// is constrained to emit normalised parameters. If that constraint ever slips,
// the end-to-end demo produces wrong classifications that look like a kernel
// bug. This test is the tripwire.

#include <kea/isa.h>

#include <cstdint>
#include <cstdio>
#include <random>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAILED: %s\n", what);
    ++g_failures;
  }
}

/// TOSA apply_scale_32, per docs/QUANTIZATION.md. Deliberately written out in
/// full here rather than shared with the simulator: this test is only
/// meaningful if it is an *independent* statement of the algorithm.
std::int32_t applyScale32(std::int32_t value, std::int32_t multiplier,
                          std::int32_t shift, bool double_round) {
  std::int64_t acc = static_cast<std::int64_t>(value) *
                         static_cast<std::int64_t>(multiplier) +
                     static_cast<std::int64_t>((1ull << shift) >> 1);
  if (double_round && shift > 31)
    acc += (value >= 0) ? (1ll << 30) : -(1ll << 30);
  // Narrowing to int32 wraps in TOSA; it does not saturate.
  return static_cast<std::int32_t>(static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(acc >> shift)));
}

std::int32_t tosaVquant(std::int32_t v, std::int32_t mult, std::int32_t tosa_shift,
                        std::int32_t out_zp, std::int32_t lo, std::int32_t hi) {
  return kea::keaClamp(applyScale32(v, mult, tosa_shift, true) + out_zp, lo, hi);
}

/// Sweep a parameter box and count divergences between the two algorithms.
/// `tosa_shift` is the TOSA convention; the KEA convention is 31 less, because
/// keaSrdhm already performs the >>31 that TOSA carries explicitly (ADR-0003).
long sweepBounded(std::uint64_t seed, std::int64_t mult_lo, std::int64_t mult_hi,
                  int shift_lo, int shift_hi, std::int64_t value_bound, int n) {
  std::mt19937_64 rng(seed);
  long divergences = 0;
  for (int i = 0; i < n; ++i) {
    auto mult = static_cast<std::int32_t>(
        mult_lo + static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(mult_hi - mult_lo)));
    auto tosa_shift = static_cast<std::int32_t>(
        shift_lo + static_cast<int>(rng() % static_cast<std::uint32_t>(shift_hi - shift_lo + 1)));
    auto v = static_cast<std::int32_t>(
        static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(2 * value_bound)) -
        value_bound);

    const std::int32_t from_hw =
        kea::keaRequantize(v, mult, tosa_shift - 31, 0, -128, 127);
    const std::int32_t from_ref = tosaVquant(v, mult, tosa_shift, 0, -128, 127);
    if (from_hw != from_ref) ++divergences;
  }
  return divergences;
}

/// Full int32-ish value range, the default for the in-domain sweeps.
long sweep(std::uint64_t seed, std::int64_t mult_lo, std::int64_t mult_hi,
           int shift_lo, int shift_hi, int n) {
  return sweepBounded(seed, mult_lo, mult_hi, shift_lo, shift_hi, 1000000000, n);
}

}  // namespace

int main() {
  constexpr std::int64_t kNormLo = 1ll << 30;   // normalised Q31 multiplier
  constexpr std::int64_t kNormHi = 1ll << 31;

  // 1. The invariant itself: normalised multiplier, TOSA shift >= 31.
  long d = sweep(0xC0FFEE, kNormLo, kNormHi, 31, 62, 400000);
  std::printf("normalised domain (mult in [2^30,2^31), tosa_shift 31..62): "
              "%ld / 400000 divergences\n", d);
  check(d == 0, "keaRequantize must equal TOSA apply_scale_32 on the normalised domain");

  // 2. Normalising the multiplier is hygiene; the shift is what carries the
  //    equivalence. Assert that explicitly so the docs can't drift from reality.
  d = sweep(0xBEEF, 1, kNormLo, 31, 62, 200000);
  std::printf("un-normalised multiplier, tosa_shift >= 31:                  "
              "%ld / 200000 divergences\n", d);
  check(d == 0, "equivalence should survive an un-normalised multiplier when shift >= 31");

  // 3. tosa_shift < 31 selects keaRequantize's pre-multiply LEFT shift, which
  //    has no TOSA counterpart. It is nonetheless numerically exact until that
  //    shift overflows int32, i.e. while |v| < 2^tosa_shift. Real models need
  //    this: MobileNetV2's rescale shifts run from 22 to 48.
  d = sweepBounded(0xD00D, kNormLo, kNormHi, 22, 30, 1 << 20, 200000);
  std::printf("tosa_shift 22..30, |v| < 2^20 (within range):                 "
              "%ld / 200000 divergences\n", d);
  check(d == 0, "the negative-shift path must be exact while the left shift fits");

  // 4. And it MUST break once the left shift overflows -- both because that is
  //    the real hazard the backend has to bound, and because without a case
  //    that diverges this test could silently become vacuous if someone made
  //    the two algorithms trivially equal.
  d = sweepBounded(0xD00D, kNormLo, kNormHi, 22, 22, 1 << 24, 200000);
  std::printf("tosa_shift 22, |v| < 2^24 (overflows the left shift):         "
              "%ld / 200000 divergences\n", d);
  check(d > 20000, "the two algorithms must genuinely differ once the left shift "
                   "overflows, otherwise this test proves nothing");

  // 5. The tie-breaking rules of the two primitives genuinely differ, which is
  //    why the domain constraint is doing real work. Compare at -1.5:
  //      - TOSA's single shift rounds half-up toward +inf   -> -1
  //      - gemmlowp's RoundingDivideByPOT rounds half away from zero -> -2
  //    Note the divergence lives in keaRdpot, NOT in keaSrdhm: keaSrdhm's nudge
  //    also rounds negative ties toward +inf, so the *pipeline* only diverges
  //    once a nonzero right shift reaches keaRdpot. That is exactly why the
  //    equivalence above survives -- on the normalised domain the multiply
  //    stage agrees and the shift stage never sees a tie the other resolves
  //    differently.
  const std::int32_t tosa_half = applyScale32(-3, 1 << 30, 31, false);
  check(tosa_half == -1, "TOSA rounds -1.5 half-up toward +inf, giving -1");
  const std::int32_t rdpot_half = kea::keaRdpot(-3, 1);
  check(rdpot_half == -2, "keaRdpot rounds -1.5 half away from zero, giving -2");
  const std::int32_t srdhm_half = kea::keaSrdhm(-3, 1 << 30);
  check(srdhm_half == -1, "keaSrdhm rounds -1.5 toward +inf, agreeing with TOSA");
  std::printf("tie at -1.5: TOSA %d, keaRdpot %d, keaSrdhm %d\n",
              tosa_half, rdpot_half, srdhm_half);

  if (g_failures == 0) {
    std::printf("ADR-0003 invariant holds.\n");
    return 0;
  }
  std::printf("test_requant_equivalence: %d FAILURE(S)\n", g_failures);
  return 1;
}
