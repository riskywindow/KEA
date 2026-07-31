// SPDX-License-Identifier: Apache-2.0
//
// kea/sim/quant.h --- TOSA `apply_scale_32`, the frontend's ground truth.
//
// WHY THIS EXISTS, AND WHY IT IS NOT `keaRequantize`
// --------------------------------------------------
// There are two requantization algorithms in this repository and they are NOT
// the same function:
//
//   * `kea::keaRequantize` (include/kea/isa.h, FROZEN) is gemmlowp/TFLite:
//     SaturatingRoundingDoublingHighMul followed by RoundingDivideByPOT.  It
//     rounds twice and rounds half-away-from-zero.  ISA.md §10.1 defines
//     VQUANT by reference to it and says "Do not reimplement this. Call the
//     header."  The simulator obeys that: VQUANT calls keaRequantize.
//
//   * TOSA `apply_scale_32` (below) is one 64-bit multiply with a pre-added
//     rounding term and a single arithmetic shift.  It rounds once, half-UP
//     (toward +infinity), and the final narrowing to int32 WRAPS rather than
//     saturating.  This is what `frontend/testdata/apply_scale_vectors.json`
//     specifies and what the compiler frontend emits scales for.
//
// The vector file's own `gemmlowp_comparison` section records that over
// 6.4M random cases on the domain KEA actually uses (multiplier in
// [2^30, 2^31), TOSA shift in [31, 62]) `apply_scale_32(.., double_round=true)`
// and gemmlowp agree on *every* case -- but that is an empirical finding, not
// a proof, and it says nothing outside that domain.
//
// So the simulator implements apply_scale_32 here, conformance-tests it
// against all 18506 published vectors, AND cross-checks that VQUANT's
// keaRequantize agrees with it across the KEA domain.  See docs/SIMULATOR.md
// §"Requantization: two algorithms, one machine".

#pragma once

#include <cstdint>

namespace kea {
namespace sim {

/// TOSA `apply_scale_32`, bit exact with
/// frontend/testdata/apply_scale_vectors.json.
///
/// `shift` is defined over [0, 63].  `round` is built with a LOGICAL shift
/// right by one of `1 << shift`, the double-round correction is applied only
/// when `shift > 31` and takes the sign of `value` (not of the product), and
/// the narrowing back to int32 wraps.
inline std::int32_t applyScale32(std::int32_t value, std::int32_t multiplier,
                                 std::int32_t shift, bool double_round) {
  // (int64_t)((uint64_t)1 << shift) >> 1 -- logical shift right by 1.
  std::int64_t round = static_cast<std::int64_t>(
      (static_cast<std::uint64_t>(1) << (shift & 63)) >> 1);
  if (double_round && shift > 31)
    round += (value >= 0) ? (1 << 30) : -(1 << 30);
  const std::int64_t prod =
      static_cast<std::int64_t>(value) * static_cast<std::int64_t>(multiplier) +
      round;
  // Arithmetic shift right, then a WRAPPING truncation to 32 bits.
  const std::int64_t shifted = prod >> (shift & 63);
  return static_cast<std::int32_t>(static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(shifted) & 0xFFFFFFFFull));
}

/// The KEA `KeaQuantParam::shift` convention (>0 right shift after a Q31
/// multiply) expressed as a TOSA shift: TOSA folds the Q31 normalisation into
/// its single shift, so tosa_shift = 31 + kea_shift.
inline std::int32_t keaShiftToTosaShift(std::int32_t kea_shift) {
  return 31 + kea_shift;
}

}  // namespace sim
}  // namespace kea
