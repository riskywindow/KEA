//===- KeaOps.h - KEA operations --------------------------------*- C++ -*-===//
#ifndef KEA_DIALECT_KEAOPS_H
#define KEA_DIALECT_KEAOPS_H

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
// ODS silently adds InferTypeOpInterface::Trait to any op carrying
// SameOperandsAndResultType (kea.clamp), so this include is mandatory even
// though no op mentions the interface. Symptom of omitting it: "no member
// named 'InferTypeOpInterface' in namespace 'mlir'" followed by a wall of
// bogus "use of undeclared identifier 'getOperation'".
#include "mlir/Interfaces/InferTypeOpInterface.h"
// ConditionallySpeculatable / MemoryEffectOpInterface, pulled in by `Pure` and
// by the Arg<..., [MemRead]> effect annotations.
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "kea/Dialect/KeaOps.h.inc"

#endif // KEA_DIALECT_KEAOPS_H
