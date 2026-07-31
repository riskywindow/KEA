//===- KeaTypes.h - KEA types -----------------------------------*- C++ -*-===//
#ifndef KEA_DIALECT_KEATYPES_H
#define KEA_DIALECT_KEATYPES_H

#include "kea/Dialect/KeaAttrs.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include <optional>

#define GET_TYPEDEF_CLASSES
#include "kea/Dialect/KeaOpsTypes.h.inc"

#endif // KEA_DIALECT_KEATYPES_H
