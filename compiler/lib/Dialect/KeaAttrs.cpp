//===- KeaAttrs.cpp - KEA attributes and enums ------------------*- C++ -*-===//

#include "kea/Dialect/KeaAttrs.h"
#include "kea/Dialect/KeaDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::kea;

// stringifyAddressSpace / symbolizeAddressSpace.
#include "kea/Dialect/KeaEnums.cpp.inc"

// llvm::TypeSwitch is required by the generated printer; include it above.
#define GET_ATTRDEF_CLASSES
#include "kea/Dialect/KeaAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// TileConfigAttr
//===----------------------------------------------------------------------===//

LogicalResult
TileConfigAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       int64_t rows, int64_t cols, int64_t depth) {
  if (rows <= 0 || cols <= 0 || depth <= 0)
    return emitError() << "tile_config dimensions must be positive, got "
                       << rows << "x" << cols << "x" << depth;
  return success();
}

//===----------------------------------------------------------------------===//
// QuantAttr
//===----------------------------------------------------------------------===//

bool QuantAttr::isIdentityScale() const {
  ArrayRef<int32_t> mult = getMultiplier().asArrayRef();
  ArrayRef<int8_t> shift = getShift().asArrayRef();
  if (mult.size() != shift.size())
    return false;
  for (auto [m, s] : llvm::zip(mult, shift)) {
    // `1 << s` must be representable as a positive i32, so s <= 30.
    if (s < 0 || s > 30)
      return false;
    if (m != (int32_t(1) << s))
      return false;
  }
  return true;
}

LogicalResult QuantAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                DenseI32ArrayAttr multiplier,
                                DenseI8ArrayAttr shift, int32_t /*inputZp*/,
                                int32_t /*outputZp*/, int64_t axis,
                                RoundingMode /*rounding*/) {
  if (!multiplier || !shift)
    return emitError() << "kea.quant requires both multiplier and shift";

  ArrayRef<int32_t> mults = multiplier.asArrayRef();
  ArrayRef<int8_t> shifts = shift.asArrayRef();

  if (mults.size() != shifts.size())
    return emitError() << "kea.quant multiplier and shift must have the same "
                          "length, got "
                       << mults.size() << " and " << shifts.size();
  if (mults.empty())
    return emitError() << "kea.quant needs at least one multiplier/shift pair";

  if (axis < -1)
    return emitError() << "kea.quant axis must be -1 (per tensor) or a "
                          "non-negative dimension index, got "
                       << axis;
  if (axis < 0 && mults.size() != 1)
    return emitError() << "kea.quant with axis = -1 is per tensor and needs "
                          "exactly one multiplier/shift pair, got "
                       << mults.size();

  for (int32_t m : mults)
    if (m <= 0)
      return emitError() << "kea.quant multipliers must be positive, got " << m;

  // TOSA requires 2 <= shift <= 62 (docs/QUANTIZATION.md §1); we additionally
  // permit 0 and 1 so that the exact identity `multiplier = 1 << shift` can be
  // spelled at small shifts. Negative (pre-)shifts are rejected outright: the
  // KEA ISA calls them rare and says the compiler should normalise them away
  // (docs/ISA.md §10.1).
  for (int8_t s : shifts)
    if (s < 0 || s > 62)
      return emitError() << "kea.quant shift must be in [0, 62], got "
                         << static_cast<int>(s);

  return success();
}

//===----------------------------------------------------------------------===//
// EpilogueAttr
//===----------------------------------------------------------------------===//

LogicalResult EpilogueAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   QuantAttr requant, DenseI64ArrayAttr clamp,
                                   QuantAttr accum, QuantAttr residual,
                                   QuantAttr output) {
  if (clamp) {
    if (clamp.size() != 2)
      return emitError() << "kea.epilogue clamp must be [lo, hi], got "
                         << clamp.size() << " elements";
    if (clamp[0] > clamp[1])
      return emitError() << "kea.epilogue clamp lo " << clamp[0]
                         << " exceeds hi " << clamp[1];
  }

  unsigned residualStages = !!accum + !!residual + !!output;
  if (residualStages != 0 && residualStages != 3)
    return emitError() << "kea.epilogue accum/residual/output are "
                          "all-or-nothing, got "
                       << residualStages << " of 3";

  if (!requant && (clamp || residualStages))
    return emitError() << "kea.epilogue without a requant stage cannot carry "
                          "any later stage";

  return success();
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void KeaDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "kea/Dialect/KeaAttrs.cpp.inc"
      >();
}
