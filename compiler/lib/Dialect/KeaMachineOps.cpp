//===- KeaMachineOps.cpp - KEA Level 2 op implementations -------*- C++ -*-===//
//
// Verifiers for the buffer-level ops declared in KeaMachineOps.td. Level 1 op
// implementations are in KeaOps.cpp.
//
//===----------------------------------------------------------------------===//

#include "kea/Dialect/KeaMachineOps.h"
#include "kea/Dialect/KeaDialect.h"
#include "kea/Dialect/KeaTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::kea;

#define GET_OP_CLASSES
#include "kea/Dialect/KeaMachineOps.cpp.inc"

//===----------------------------------------------------------------------===//
// DmaLoadOp
//===----------------------------------------------------------------------===//

LogicalResult DmaLoadOp::verify() {
  auto srcTy = llvm::cast<MemRefType>(getSource().getType());
  auto dstTy = llvm::cast<BufferType>(getDest().getType());

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("element type mismatch: ")
           << srcTy.getElementType() << " vs " << dstTy.getElementType();

  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("shape mismatch between source memref and destination "
                       "buffer");

  if (!dstTy.isOnChip())
    return emitOpError("destination must be an on-chip buffer (A, W or ACC), "
                       "not DRAM");

  return success();
}

//===----------------------------------------------------------------------===//
// MmOp
//===----------------------------------------------------------------------===//

LogicalResult MmOp::verify() {
  auto lhsTy = llvm::cast<BufferType>(getLhs().getType());
  auto rhsTy = llvm::cast<BufferType>(getRhs().getType());
  auto accTy = llvm::cast<BufferType>(getAcc().getType());

  if (lhsTy.getShape().size() != 2 || rhsTy.getShape().size() != 2 ||
      accTy.getShape().size() != 2)
    return emitOpError("all operands must be rank-2 buffers");

  int64_t m = lhsTy.getShape()[0];
  int64_t k = lhsTy.getShape()[1];
  int64_t rhsK = getTransposeRhs() ? rhsTy.getShape()[1] : rhsTy.getShape()[0];
  int64_t n = getTransposeRhs() ? rhsTy.getShape()[0] : rhsTy.getShape()[1];

  if (k != rhsK)
    return emitOpError("contraction dimension mismatch: lhs K=")
           << k << " but rhs K=" << rhsK;

  if (accTy.getShape()[0] != m || accTy.getShape()[1] != n)
    return emitOpError("accumulator shape must be ")
           << m << "x" << n << ", got " << accTy.getShape()[0] << "x"
           << accTy.getShape()[1];

  return success();
}
