//===- KeaAttrs.h - KEA attributes and enums --------------------*- C++ -*-===//
#ifndef KEA_DIALECT_KEAATTRS_H
#define KEA_DIALECT_KEAATTRS_H

#include "kea/Dialect/KeaDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"

// The plain C++ enum `mlir::kea::AddressSpace` plus stringifyAddressSpace() /
// symbolizeAddressSpace(). Must come before the AttrDef and TypeDef includes,
// which reference the enum by name.
#include "kea/Dialect/KeaEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "kea/Dialect/KeaAttrs.h.inc"

#endif // KEA_DIALECT_KEAATTRS_H
