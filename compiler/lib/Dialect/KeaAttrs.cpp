//===- KeaAttrs.cpp - KEA attributes and enums ------------------*- C++ -*-===//

#include "kea/Dialect/KeaAttrs.h"
#include "kea/Dialect/KeaDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
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
// Registration
//===----------------------------------------------------------------------===//

void KeaDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "kea/Dialect/KeaAttrs.cpp.inc"
      >();
}
