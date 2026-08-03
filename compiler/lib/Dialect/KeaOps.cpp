//===- KeaOps.cpp - KEA Level 1 op implementations --------------*- C++ -*-===//
//
// Verifiers for the tensor-level ops declared in KeaOps.td. Level 2 op
// implementations are in KeaMachineOps.cpp.
//
// The verifiers here deliberately check *shapes*, not just types: an op that
// only checks "rank 4 in, rank 4 out" would happily accept a convolution whose
// output spatial extent disagrees with its pad/stride/dilation, which is
// exactly the class of bug the conversion passes can introduce.
//
//===----------------------------------------------------------------------===//

#include "kea/Dialect/KeaOps.h"
#include "kea/Dialect/KeaDialect.h"
#include "kea/Dialect/KeaTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::kea;

#define GET_OP_CLASSES
#include "kea/Dialect/KeaOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Shared verification helpers
//===----------------------------------------------------------------------===//

namespace {

RankedTensorType tensorOf(Value v) {
  return llvm::cast<RankedTensorType>(v.getType());
}

/// `[t, b, l, r]` / `[h, w]` attribute sanity, shared by conv/dwconv/pool.
LogicalResult verifyWindowAttrs(Operation *op, ArrayRef<int64_t> strides,
                                ArrayRef<int64_t> pads,
                                ArrayRef<int64_t> dilations) {
  if (strides.size() != 2)
    return op->emitOpError("expects exactly 2 strides (h, w), got ")
           << strides.size();
  if (pads.size() != 4)
    return op->emitOpError("expects exactly 4 pads (t, b, l, r), got ")
           << pads.size();
  if (dilations.size() != 2)
    return op->emitOpError("expects exactly 2 dilations (h, w), got ")
           << dilations.size();
  for (int64_t s : strides)
    if (s <= 0)
      return op->emitOpError("strides must be positive");
  for (int64_t d : dilations)
    if (d <= 0)
      return op->emitOpError("dilations must be positive");
  for (int64_t p : pads)
    if (p < 0)
      return op->emitOpError("pads must be non-negative");
  return success();
}

/// TOSA's output-extent formula (docs/TOSA_NOTES.md §2), applied to one axis.
int64_t windowedExtent(int64_t in, int64_t padLo, int64_t padHi, int64_t kernel,
                       int64_t stride, int64_t dilation) {
  return (in + padLo + padHi - dilation * (kernel - 1) - 1) / stride + 1;
}

LogicalResult verifySpatial(Operation *op, RankedTensorType in,
                            RankedTensorType out, int64_t kh, int64_t kw,
                            ArrayRef<int64_t> strides, ArrayRef<int64_t> pads,
                            ArrayRef<int64_t> dilations) {
  int64_t expectedH = windowedExtent(in.getDimSize(1), pads[0], pads[1], kh,
                                     strides[0], dilations[0]);
  int64_t expectedW = windowedExtent(in.getDimSize(2), pads[2], pads[3], kw,
                                     strides[1], dilations[1]);
  if (expectedH <= 0 || expectedW <= 0)
    return op->emitOpError("pad/stride/dilation give a degenerate output "
                           "spatial size ")
           << expectedH << "x" << expectedW;
  if (out.getDimSize(1) != expectedH || out.getDimSize(2) != expectedW)
    return op->emitOpError("output spatial size ")
           << out.getDimSize(1) << "x" << out.getDimSize(2)
           << " disagrees with pad/stride/dilation, which give " << expectedH
           << "x" << expectedW;
  if (out.getDimSize(0) != in.getDimSize(0))
    return op->emitOpError("batch size ")
           << out.getDimSize(0) << " does not match the input's "
           << in.getDimSize(0);
  return success();
}

/// A rank-1 i32 bias of exactly `outChannels` elements.
LogicalResult verifyBias(Operation *op, Value bias, int64_t outChannels) {
  if (!bias)
    return success();
  auto biasTy = tensorOf(bias);
  if (biasTy.getRank() != 1)
    return op->emitOpError("bias must be rank 1, got ") << biasTy;
  if (biasTy.getDimSize(0) != outChannels)
    return op->emitOpError("bias must have ")
           << outChannels << " elements, got " << biasTy.getDimSize(0);
  if (!biasTy.getElementType().isSignlessInteger(32))
    return op->emitOpError("bias element type must be i32, got ")
           << biasTy.getElementType();
  return success();
}

/// Per-channel scale arrays must be indexed by a real dimension of `ty` and
/// have exactly that many entries.
LogicalResult verifyQuantAgainst(Operation *op, QuantAttr q,
                                 RankedTensorType ty, StringRef what) {
  if (!q || !q.isPerChannel())
    return success();
  int64_t axis = q.getAxis();
  if (axis >= ty.getRank())
    return op->emitOpError(what)
           << " is per-channel on axis " << axis
           << " but the tensor has rank " << ty.getRank();
  if (q.getNumScales() != ty.getDimSize(axis))
    return op->emitOpError(what)
           << " has " << q.getNumScales() << " scales but dimension " << axis
           << " of the tensor is " << ty.getDimSize(axis);
  return success();
}

/// Shared epilogue / bias / residual rules for the four contraction ops.
/// `outChannels` is the length a bias must have; `resultTy` is the op result.
LogicalResult verifyEpilogue(Operation *op, EpilogueAttr epilogue,
                             Value residual, RankedTensorType resultTy,
                             int64_t outChannels) {
  QuantAttr requant = epilogue ? epilogue.getRequant() : QuantAttr();

  // With nothing fused the op yields its raw accumulator.
  if (!requant) {
    if (!resultTy.getElementType().isSignlessInteger(32))
      return op->emitOpError("without an epilogue requant stage the result is "
                             "the raw accumulator and must be i32, got ")
             << resultTy.getElementType();
  }

  if (failed(verifyQuantAgainst(op, requant, resultTy, "epilogue requant")))
    return failure();

  if (requant && requant.isPerChannel() &&
      requant.getNumScales() != outChannels)
    return op->emitOpError("epilogue requant has ")
           << requant.getNumScales() << " per-channel scales but the op has "
           << outChannels << " output channels";

  bool hasResidualQuants = epilogue && epilogue.hasResidual();
  if (residual && !hasResidualQuants)
    return op->emitOpError("a residual operand requires the epilogue's "
                           "accum/residual/output requantizations");
  if (hasResidualQuants && !residual)
    return op->emitOpError("the epilogue's residual requantizations require a "
                           "residual operand");

  if (residual) {
    auto resTy = tensorOf(residual);
    if (resTy.getShape() != resultTy.getShape())
      return op->emitOpError("residual shape ")
             << resTy << " does not match the result shape " << resultTy;
  }

  if (epilogue) {
    if (failed(verifyQuantAgainst(op, epilogue.getAccum(), resultTy,
                                  "epilogue accum")) ||
        failed(verifyQuantAgainst(op, epilogue.getResidual(), resultTy,
                                  "epilogue residual")) ||
        failed(verifyQuantAgainst(op, epilogue.getOutput(), resultTy,
                                  "epilogue output")))
      return failure();

    if (auto clamp = epilogue.getClamp()) {
      unsigned width = resultTy.getElementType().getIntOrFloatBitWidth();
      int64_t lo = llvm::APInt::getSignedMinValue(width).getSExtValue();
      int64_t hi = llvm::APInt::getSignedMaxValue(width).getSExtValue();
      if (clamp[0] < lo || clamp[1] > hi)
        return op->emitOpError("epilogue clamp [")
               << clamp[0] << ", " << clamp[1]
               << "] does not fit the result element type range [" << lo << ", "
               << hi << "]";
    }
  }

  return success();
}

/// Integer element types only: Level 1 is the quantized graph.
LogicalResult verifyIntegerElements(Operation *op, RankedTensorType ty,
                                    StringRef what) {
  if (!ty.getElementType().isSignlessInteger())
    return op->emitOpError(what)
           << " must have a signless integer element type, got "
           << ty.getElementType();
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

LogicalResult Conv2DOp::verify() {
  auto inputTy = tensorOf(getInput());
  auto weightsTy = tensorOf(getWeights());
  auto outputTy = tensorOf(getOutput());

  if (inputTy.getRank() != 4)
    return emitOpError("expects a rank-4 NHWC input, got ") << inputTy;
  if (weightsTy.getRank() != 4)
    return emitOpError("expects rank-4 OHWI weights, got ") << weightsTy;
  if (outputTy.getRank() != 4)
    return emitOpError("expects a rank-4 NHWC output, got ") << outputTy;

  if (failed(verifyIntegerElements(getOperation(), inputTy, "input")) ||
      failed(verifyIntegerElements(getOperation(), weightsTy, "weights")) ||
      failed(verifyIntegerElements(getOperation(), outputTy, "output")))
    return failure();

  if (failed(verifyWindowAttrs(getOperation(), getStrides(), getPads(),
                               getDilations())))
    return failure();

  // Input channels (NHWC dim 3) vs weight channels (OHWI dim 3).
  if (inputTy.getDimSize(3) != weightsTy.getDimSize(3))
    return emitOpError("input channel count ")
           << inputTy.getDimSize(3) << " does not match weight channel count "
           << weightsTy.getDimSize(3);

  // Output channels (NHWC dim 3) vs filter count (OHWI dim 0).
  int64_t outChannels = weightsTy.getDimSize(0);
  if (outputTy.getDimSize(3) != outChannels)
    return emitOpError("output channel count ")
           << outputTy.getDimSize(3) << " does not match filter count "
           << outChannels;

  if (failed(verifySpatial(getOperation(), inputTy, outputTy,
                           weightsTy.getDimSize(1), weightsTy.getDimSize(2),
                           getStrides(), getPads(), getDilations())))
    return failure();

  if (failed(verifyBias(getOperation(), getBias(), outChannels)))
    return failure();

  return verifyEpilogue(getOperation(), getEpilogueAttr(), getResidual(),
                        outputTy, outChannels);
}

//===----------------------------------------------------------------------===//
// DWConv2DOp
//===----------------------------------------------------------------------===//

LogicalResult DWConv2DOp::verify() {
  auto inputTy = tensorOf(getInput());
  auto weightsTy = tensorOf(getWeights());
  auto outputTy = tensorOf(getOutput());

  if (inputTy.getRank() != 4)
    return emitOpError("expects a rank-4 NHWC input, got ") << inputTy;
  if (weightsTy.getRank() != 4)
    return emitOpError("expects rank-4 canonical [OC, KH, KW, 1] weights, got ")
           << weightsTy;
  if (outputTy.getRank() != 4)
    return emitOpError("expects a rank-4 NHWC output, got ") << outputTy;

  if (failed(verifyIntegerElements(getOperation(), inputTy, "input")) ||
      failed(verifyIntegerElements(getOperation(), weightsTy, "weights")) ||
      failed(verifyIntegerElements(getOperation(), outputTy, "output")))
    return failure();

  // The canonical depthwise weight layout is OHWI with IC == 1. TOSA's HWCM
  // layout is normalised into this by -tosa-to-kea; rejecting HWCM here is the
  // whole point of having one canonical form.
  if (weightsTy.getDimSize(3) != 1)
    return emitOpError("canonical depthwise weights are [OC, KH, KW, 1]; "
                       "the trailing dimension must be 1, got ")
           << weightsTy;

  if (failed(verifyWindowAttrs(getOperation(), getStrides(), getPads(),
                               getDilations())))
    return failure();

  int64_t inChannels = inputTy.getDimSize(3);
  int64_t outChannels = weightsTy.getDimSize(0);
  if (outputTy.getDimSize(3) != outChannels)
    return emitOpError("output channel count ")
           << outputTy.getDimSize(3) << " does not match weight count "
           << outChannels;
  if (inChannels == 0 || outChannels % inChannels != 0)
    return emitOpError("output channel count ")
           << outChannels << " must be a multiple of the input channel count "
           << inChannels << " (the channel multiplier)";

  if (failed(verifySpatial(getOperation(), inputTy, outputTy,
                           weightsTy.getDimSize(1), weightsTy.getDimSize(2),
                           getStrides(), getPads(), getDilations())))
    return failure();

  if (failed(verifyBias(getOperation(), getBias(), outChannels)))
    return failure();

  return verifyEpilogue(getOperation(), getEpilogueAttr(), getResidual(),
                        outputTy, outChannels);
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  auto aTy = tensorOf(getA());
  auto bTy = tensorOf(getB());
  auto outTy = tensorOf(getOutput());

  if (aTy.getRank() != 3 || bTy.getRank() != 3 || outTy.getRank() != 3)
    return emitOpError("expects strictly rank-3 [B, M, K] x [B, K, N] "
                       "operands and a rank-3 result");

  if (failed(verifyIntegerElements(getOperation(), aTy, "a")) ||
      failed(verifyIntegerElements(getOperation(), bTy, "b")) ||
      failed(verifyIntegerElements(getOperation(), outTy, "output")))
    return failure();

  if (aTy.getDimSize(0) != bTy.getDimSize(0) ||
      outTy.getDimSize(0) != aTy.getDimSize(0))
    return emitOpError("batch dimensions disagree: ")
           << aTy.getDimSize(0) << ", " << bTy.getDimSize(0) << ", "
           << outTy.getDimSize(0);

  if (aTy.getDimSize(2) != bTy.getDimSize(1))
    return emitOpError("contraction dimension mismatch: a K=")
           << aTy.getDimSize(2) << " but b K=" << bTy.getDimSize(1);

  if (outTy.getDimSize(1) != aTy.getDimSize(1))
    return emitOpError("result M ")
           << outTy.getDimSize(1) << " does not match a's M "
           << aTy.getDimSize(1);

  int64_t n = bTy.getDimSize(2);
  if (outTy.getDimSize(2) != n)
    return emitOpError("result N ")
           << outTy.getDimSize(2) << " does not match b's N " << n;

  if (failed(verifyBias(getOperation(), getBias(), n)))
    return failure();

  return verifyEpilogue(getOperation(), getEpilogueAttr(), getResidual(), outTy,
                        n);
}

//===----------------------------------------------------------------------===//
// FullyConnectedOp
//===----------------------------------------------------------------------===//

LogicalResult FullyConnectedOp::verify() {
  auto inTy = tensorOf(getInput());
  auto wTy = tensorOf(getWeights());
  auto outTy = tensorOf(getOutput());

  if (inTy.getRank() != 2 || wTy.getRank() != 2 || outTy.getRank() != 2)
    return emitOpError("expects rank-2 [N, IC] input, [OC, IC] weights and "
                       "[N, OC] result");

  if (failed(verifyIntegerElements(getOperation(), inTy, "input")) ||
      failed(verifyIntegerElements(getOperation(), wTy, "weights")) ||
      failed(verifyIntegerElements(getOperation(), outTy, "output")))
    return failure();

  if (inTy.getDimSize(1) != wTy.getDimSize(1))
    return emitOpError("input feature count ")
           << inTy.getDimSize(1) << " does not match weight feature count "
           << wTy.getDimSize(1) << " (weights are [OC, IC], already transposed)";

  if (outTy.getDimSize(0) != inTy.getDimSize(0))
    return emitOpError("result rows ")
           << outTy.getDimSize(0) << " do not match input rows "
           << inTy.getDimSize(0);

  int64_t outChannels = wTy.getDimSize(0);
  if (outTy.getDimSize(1) != outChannels)
    return emitOpError("result columns ")
           << outTy.getDimSize(1) << " do not match the weight's OC "
           << outChannels;

  if (failed(verifyBias(getOperation(), getBias(), outChannels)))
    return failure();

  return verifyEpilogue(getOperation(), getEpilogueAttr(), getResidual(), outTy,
                        outChannels);
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() {
  auto lhsTy = tensorOf(getLhs());
  auto rhsTy = tensorOf(getRhs());
  auto outTy = tensorOf(getOutput());

  if (failed(verifyIntegerElements(getOperation(), outTy, "output")))
    return failure();

  if (lhsTy.getRank() != outTy.getRank() || rhsTy.getRank() != outTy.getRank())
    return emitOpError("operands and result must have equal rank (TOSA-style "
                       "broadcasting, no rank promotion)");

  for (int64_t d = 0; d < outTy.getRank(); ++d) {
    int64_t l = lhsTy.getDimSize(d), r = rhsTy.getDimSize(d),
            o = outTy.getDimSize(d);
    if ((l != o && l != 1) || (r != o && r != 1))
      return emitOpError("dimension ")
             << d << " (" << l << ", " << r << ") is not broadcastable to " << o;
  }

  unsigned nQuants = !!getLhsQuantAttr() + !!getRhsQuantAttr() +
                     !!getOutQuantAttr();
  if (nQuants != 0 && nQuants != 3)
    return emitOpError("lhs_quant/rhs_quant/out_quant are all-or-nothing, got ")
           << nQuants << " of 3";

  if (nQuants == 0 && lhsTy.getElementType() != rhsTy.getElementType())
    return emitOpError("an unquantized add needs matching operand element "
                       "types, got ")
           << lhsTy.getElementType() << " and " << rhsTy.getElementType();

  if (failed(verifyQuantAgainst(getOperation(), getLhsQuantAttr(), outTy,
                                "lhs_quant")) ||
      failed(verifyQuantAgainst(getOperation(), getRhsQuantAttr(), outTy,
                                "rhs_quant")) ||
      failed(verifyQuantAgainst(getOperation(), getOutQuantAttr(), outTy,
                                "out_quant")))
    return failure();

  if (auto clamp = getClamp()) {
    if (clamp->size() != 2)
      return emitOpError("clamp must be [lo, hi], got ") << clamp->size()
                                                         << " elements";
    if ((*clamp)[0] > (*clamp)[1])
      return emitOpError("clamp lo ")
             << (*clamp)[0] << " exceeds hi " << (*clamp)[1];
    if (nQuants == 0)
      return emitOpError("a clamp on kea.add requires the quantized form");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ClampOp
//===----------------------------------------------------------------------===//

LogicalResult ClampOp::verify() {
  auto ty = tensorOf(getInput());
  if (failed(verifyIntegerElements(getOperation(), ty, "input")))
    return failure();

  int64_t lo = getMin(), hi = getMax();
  if (lo > hi)
    return emitOpError("min ") << lo << " exceeds max " << hi;

  unsigned width = ty.getElementType().getIntOrFloatBitWidth();
  int64_t tyLo = llvm::APInt::getSignedMinValue(width).getSExtValue();
  int64_t tyHi = llvm::APInt::getSignedMaxValue(width).getSExtValue();
  if (lo < tyLo || hi > tyHi)
    return emitOpError("clamp bounds [")
           << lo << ", " << hi << "] do not fit the element type range [" << tyLo
           << ", " << tyHi << "]";

  return success();
}

//===----------------------------------------------------------------------===//
// RescaleOp
//===----------------------------------------------------------------------===//

LogicalResult RescaleOp::verify() {
  auto inTy = tensorOf(getInput());
  auto outTy = tensorOf(getOutput());

  if (failed(verifyIntegerElements(getOperation(), inTy, "input")) ||
      failed(verifyIntegerElements(getOperation(), outTy, "output")))
    return failure();

  if (inTy.getShape() != outTy.getShape())
    return emitOpError("rescale preserves shape, got ")
           << inTy << " -> " << outTy;

  return verifyQuantAgainst(getOperation(), getQuantAttr(), outTy, "quant");
}

//===----------------------------------------------------------------------===//
// PoolOp
//===----------------------------------------------------------------------===//

LogicalResult PoolOp::verify() {
  auto inTy = tensorOf(getInput());
  auto outTy = tensorOf(getOutput());

  if (inTy.getRank() != 4 || outTy.getRank() != 4)
    return emitOpError("expects rank-4 NHWC input and output");
  if (failed(verifyIntegerElements(getOperation(), inTy, "input")))
    return failure();
  if (inTy.getElementType() != outTy.getElementType())
    return emitOpError("pooling preserves the element type, got ")
           << inTy.getElementType() << " -> " << outTy.getElementType();

  ArrayRef<int64_t> kernel = getKernel();
  if (kernel.size() != 2)
    return emitOpError("expects exactly 2 kernel extents (h, w), got ")
           << kernel.size();
  for (int64_t k : kernel)
    if (k <= 0)
      return emitOpError("kernel extents must be positive");

  // Pooling has no dilation; pass 1s so the shared helpers stay honest.
  SmallVector<int64_t, 2> ones{1, 1};
  if (failed(verifyWindowAttrs(getOperation(), getStrides(), getPads(), ones)))
    return failure();

  if (inTy.getDimSize(3) != outTy.getDimSize(3))
    return emitOpError("pooling preserves channels, got ")
           << inTy.getDimSize(3) << " -> " << outTy.getDimSize(3);

  if (failed(verifySpatial(getOperation(), inTy, outTy, kernel[0], kernel[1],
                           getStrides(), getPads(), ones)))
    return failure();

  QuantAttr quant = getQuantAttr();
  if (getKind() == PoolKind::MAX && quant)
    return emitOpError("max pooling is monotonic and carries no "
                       "quantization; drop the quant attribute");
  if (quant) {
    if (!quant.isIdentityScale())
      return emitOpError("average pooling can only rebase the zero point; its "
                         "quant must have an identity multiplier/shift, use a "
                         "separate kea.rescale for a real scale change");
    if (failed(verifyQuantAgainst(getOperation(), quant, outTy, "quant")))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  auto inTy = tensorOf(getInput());
  auto outTy = tensorOf(getOutput());

  if (inTy.getElementType() != outTy.getElementType())
    return emitOpError("reshape preserves the element type, got ")
           << inTy.getElementType() << " -> " << outTy.getElementType();

  ArrayRef<int64_t> newShape = getNewShape();
  for (int64_t d : newShape)
    if (d < 0)
      return emitOpError("new_shape must be fully resolved at Level 1; a "
                         "-1 placeholder is not allowed");

  if (newShape != outTy.getShape())
    return emitOpError("new_shape does not match the result type ") << outTy;

  int64_t inElems = 1, outElems = 1;
  for (int64_t d : inTy.getShape())
    inElems *= d;
  for (int64_t d : newShape)
    outElems *= d;
  if (inElems != outElems)
    return emitOpError("reshape changes the element count, ")
           << inElems << " -> " << outElems;

  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  auto inTy = tensorOf(getInput());
  auto outTy = tensorOf(getOutput());
  ArrayRef<int64_t> perms = getPerms();

  if (inTy.getElementType() != outTy.getElementType())
    return emitOpError("transpose preserves the element type, got ")
           << inTy.getElementType() << " -> " << outTy.getElementType();

  if (static_cast<int64_t>(perms.size()) != inTy.getRank() ||
      outTy.getRank() != inTy.getRank())
    return emitOpError("perms must have exactly one entry per input dimension, "
                       "got ")
           << perms.size() << " for rank " << inTy.getRank();

  SmallVector<bool> seen(inTy.getRank(), false);
  for (int64_t p : perms) {
    if (p < 0 || p >= inTy.getRank())
      return emitOpError("perms entry ") << p << " is out of range";
    if (seen[p])
      return emitOpError("perms entry ") << p << " appears more than once";
    seen[p] = true;
  }

  for (int64_t d = 0; d < outTy.getRank(); ++d)
    if (outTy.getDimSize(d) != inTy.getDimSize(perms[d]))
      return emitOpError("result dimension ")
             << d << " is " << outTy.getDimSize(d) << " but input dimension "
             << perms[d] << " is " << inTy.getDimSize(perms[d]);

  return success();
}
