//===- Passes.h - Conversions into KEA Level 1 ------------------*- C++ -*-===//
//
// The pass declarations themselves come from `kea/Transforms/Passes.h`,
// because `include/kea/Conversion/Passes.td` is included into
// `include/kea/Transforms/Passes.td` so that there is a single generated
// `registerKeaPasses()`. See the comment at the top of Conversion/Passes.td.
//
// What lives here is the *pattern-level* API, so the conversions can be reused
// from a larger pipeline without going through the pass registry.
//
//===----------------------------------------------------------------------===//
#ifndef KEA_CONVERSION_PASSES_H
#define KEA_CONVERSION_PASSES_H

#include "kea/Transforms/Passes.h"

namespace mlir {
class RewritePatternSet;
class ConversionTarget;
class OpBuilder;
class Location;
class Value;

namespace kea {

/// Materializes `hwcm` -- depthwise weights in the HWCM `[KH, KW, C, M]`
/// layout that both TOSA and linalg use -- in the KEA canonical depthwise
/// layout `[C*M, KH, KW, 1]`. Constant-folds when `hwcm` is constant,
/// otherwise emits `kea.transpose` + `kea.reshape`. See
/// lib/Conversion/WeightLayout.cpp for the index algebra.
Value materializeCanonicalDepthwiseWeights(OpBuilder &builder, Location loc,
                                           Value hwcm);

/// Populates `patterns` with the TOSA -> KEA Level 1 conversion patterns and
/// marks `tosa` illegal / `kea` + `arith` legal on `target`.
void populateTosaToKeaPatterns(RewritePatternSet &patterns);
void configureTosaToKeaTarget(ConversionTarget &target);

/// Populates `patterns` with the linalg named op -> KEA Level 1 patterns.
/// Unlike the TOSA path this is a *partial* conversion: the caller is expected
/// to run it greedily and then check for leftover contraction-like linalg ops.
void populateLinalgToKeaPatterns(RewritePatternSet &patterns);

} // namespace kea
} // namespace mlir

#endif // KEA_CONVERSION_PASSES_H
