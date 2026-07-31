//===- KeaOps.cpp - KEA op implementations ----------------------*- C++ -*-===//

#include "kea/Dialect/KeaOps.h"
#include "kea/Dialect/KeaDialect.h"
#include "kea/Dialect/KeaTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::kea;

#define GET_OP_CLASSES
#include "kea/Dialect/KeaOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

LogicalResult Conv2DOp::verify() {
  auto inputTy = llvm::cast<RankedTensorType>(getInput().getType());
  auto weightsTy = llvm::cast<RankedTensorType>(getWeights().getType());
  auto outputTy = llvm::cast<RankedTensorType>(getOutput().getType());

  if (inputTy.getRank() != 4)
    return emitOpError("expects a rank-4 NHWC input, got ") << inputTy;
  if (weightsTy.getRank() != 4)
    return emitOpError("expects rank-4 OHWI weights, got ") << weightsTy;
  if (outputTy.getRank() != 4)
    return emitOpError("expects a rank-4 NHWC output, got ") << outputTy;

  if (getStrides().size() != 2)
    return emitOpError("expects exactly 2 strides (h, w), got ")
           << getStrides().size();
  if (getPads().size() != 4)
    return emitOpError("expects exactly 4 pads (t, b, l, r), got ")
           << getPads().size();

  for (int64_t s : getStrides())
    if (s <= 0)
      return emitOpError("strides must be positive");
  for (int64_t p : getPads())
    if (p < 0)
      return emitOpError("pads must be non-negative");

  // Input channels (NHWC dim 3) must match weight channels (OHWI dim 3).
  if (inputTy.getDimSize(3) != weightsTy.getDimSize(3))
    return emitOpError("input channel count ")
           << inputTy.getDimSize(3) << " does not match weight channel count "
           << weightsTy.getDimSize(3);

  // Output channels (NHWC dim 3) must match the number of filters (OHWI dim 0).
  if (outputTy.getDimSize(3) != weightsTy.getDimSize(0))
    return emitOpError("output channel count ")
           << outputTy.getDimSize(3) << " does not match filter count "
           << weightsTy.getDimSize(0);

  if (Value bias = getBias()) {
    auto biasTy = llvm::cast<RankedTensorType>(bias.getType());
    if (biasTy.getRank() != 1 ||
        biasTy.getDimSize(0) != weightsTy.getDimSize(0))
      return emitOpError("bias must be a rank-1 tensor of ")
             << weightsTy.getDimSize(0) << " elements, got " << biasTy;
  }

  // NOTE: an F64Attr getter returns llvm::APFloat, not double, and APFloat has
  // neither comparison against double nor a Diagnostic streaming operator.
  double scale = getScale().convertToDouble();
  if (scale <= 0.0)
    return emitOpError("scale must be positive, got ") << scale;

  return success();
}

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
// MatmulOp
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
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
