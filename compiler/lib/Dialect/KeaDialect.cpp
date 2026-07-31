//===- KeaDialect.cpp - KEA dialect registration ----------------*- C++ -*-===//

#include "kea/Dialect/KeaDialect.h"
#include "kea/Dialect/KeaAttrs.h"
#include "kea/Dialect/KeaOps.h"
#include "kea/Dialect/KeaTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::kea;

#include "kea/Dialect/KeaOpsDialect.cpp.inc"

void KeaDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "kea/Dialect/KeaOps.cpp.inc"
      >();
}
