//===- Passes.h - KEA transform passes --------------------------*- C++ -*-===//
#ifndef KEA_TRANSFORMS_PASSES_H
#define KEA_TRANSFORMS_PASSES_H

#include "kea/Dialect/KeaDialect.h"
// The generated pass bases name ::mlir::func::FuncOp and ::mlir::ModuleOp as
// the anchor op types, and ::mlir::arith::ArithDialect as a dependent dialect,
// so those declarations must be complete here.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class ModuleOp;

namespace kea {

// Declares the Options structs and the createXxx() factories.
#define GEN_PASS_DECL
#include "kea/Transforms/Passes.h.inc"

/// Registers every KEA pass with the global pass registry. Called from
/// tools/kea-opt.
#define GEN_PASS_REGISTRATION
#include "kea/Transforms/Passes.h.inc"

} // namespace kea
} // namespace mlir

#endif // KEA_TRANSFORMS_PASSES_H
