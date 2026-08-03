//===- KeaMachineOps.h - KEA Level 2 (buffer level) ops ---------*- C++ -*-===//
//
// Declarations for the ops defined in KeaMachineOps.td. Level 1 ops are in
// KeaOps.h; the two headers are independent so the two halves of the compiler
// do not collide. Including this header also pulls in KeaOps.h because both
// levels share the dialect, types and attributes.
//
//===----------------------------------------------------------------------===//
#ifndef KEA_DIALECT_KEAMACHINEOPS_H
#define KEA_DIALECT_KEAMACHINEOPS_H

#include "kea/Dialect/KeaAttrs.h"
#include "kea/Dialect/KeaDialect.h"
#include "kea/Dialect/KeaTypes.h"

// MLIR >= 17 gives every ODS op a BytecodeOpInterface::Trait, so this include
// is mandatory even though nothing here mentions bytecode. Leaving it out
// produces a wall of "no member named 'BytecodeOpInterface' in namespace
// 'mlir'" errors from the generated header.
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "kea/Dialect/KeaMachineOps.h.inc"

#endif // KEA_DIALECT_KEAMACHINEOPS_H
