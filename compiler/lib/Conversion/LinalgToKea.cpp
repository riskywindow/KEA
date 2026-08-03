//===- LinalgToKea.cpp - -linalg-to-kea ------------------------*- C++ -*-===//
//
// The secondary graph-ingest path: the four quantized linalg named ops that
// `--tosa-to-linalg-named` produces (docs/TOSA_NOTES.md §11.3, §12) into KEA
// Level 1.
//
// Two structural facts drive everything here:
//
//   1. linalg convolutions have NO padding. Padding is a separate
//      `tensor.pad`. We therefore always emit `pads = [0, 0, 0, 0]` and leave
//      any `tensor.pad` in place; recovering it into the conv would need a
//      proof that the pad value equals the input zero point, which is the TOSA
//      path's job.
//   2. `outs` is the accumulator *initializer*, not a shape hint. When it is
//      provably zero we drop it; otherwise we emit an explicit `kea.add`,
//      which is exactly linalg's `outs + A*B` semantics.
//
// Zero points arrive as trailing scalar `i32` operands inside `ins(...)` and
// must be constants to become a `#kea.zp` attribute.
//
//===----------------------------------------------------------------------===//

#include "kea/Conversion/Passes.h"
#include "kea/Dialect/KeaOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace kea {
#define GEN_PASS_DEF_LINALGTOKEA
#include "kea/Transforms/Passes.h.inc"
} // namespace kea
} // namespace mlir

using namespace mlir;
using namespace mlir::kea;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Extracts a constant scalar i32 zero point operand.
FailureOr<int32_t> constantZeroPoint(Value v) {
  IntegerAttr attr;
  if (!matchPattern(v, m_Constant(&attr)))
    return failure();
  return static_cast<int32_t>(attr.getValue().getSExtValue());
}

/// linalg spells strides/dilations as `dense<...> : tensor<2xi64>`.
SmallVector<int64_t> denseToVector(DenseIntElementsAttr attr) {
  SmallVector<int64_t> out;
  for (const APInt &v : attr.getValues<APInt>())
    out.push_back(v.getSExtValue());
  if (out.size() == 1)
    out.push_back(out[0]);
  return out;
}

/// True when `init` is provably an all-zero accumulator, so it can be dropped.
bool isZeroInit(Value init) {
  Operation *def = init.getDefiningOp();
  if (!def)
    return false;
  // tensor.empty is uninitialized; every producer of quantized linalg in this
  // project means "zero" by it, and linalg itself only reads it through a
  // fill. Treating it as zero is documented in DIALECT_L1.md §5.2.
  if (isa<tensor::EmptyOp>(def))
    return true;
  if (auto fill = dyn_cast<linalg::FillOp>(def)) {
    IntegerAttr attr;
    return matchPattern(fill.getInputs()[0], m_Constant(&attr)) &&
           attr.getValue().isZero();
  }
  return false;
}

/// Emits `kea.add %value, %init` when the linalg accumulator initializer is
/// not provably zero.
Value applyInit(PatternRewriter &rewriter, Location loc, Value value,
                Value init) {
  if (isZeroInit(init))
    return value;
  return rewriter.create<AddOp>(loc, value.getType(), value, init,
                                /*lhs_quant=*/QuantAttr(),
                                /*rhs_quant=*/QuantAttr(),
                                /*out_quant=*/QuantAttr(),
                                /*clamp=*/DenseI64ArrayAttr());
}

//===----------------------------------------------------------------------===//
// linalg.conv_2d_nhwc_fhwc_q
//===----------------------------------------------------------------------===//

struct ConvertConv2DNhwcFhwcQ
    : public OpRewritePattern<linalg::Conv2DNhwcFhwcQOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcFhwcQOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "KEA Level 1 is value semantic; the memref form has no result");

    ValueRange ins = op.getInputs();
    if (ins.size() != 4)
      return rewriter.notifyMatchFailure(op, "expects 4 ins operands");

    FailureOr<int32_t> izp = constantZeroPoint(ins[2]);
    FailureOr<int32_t> wzp = constantZeroPoint(ins[3]);
    if (failed(izp) || failed(wzp))
      return rewriter.notifyMatchFailure(
          op, "zero point operands must be constants");

    SmallVector<int64_t> strides = denseToVector(op.getStrides());
    SmallVector<int64_t> dilations = denseToVector(op.getDilations());

    Value conv = rewriter.create<Conv2DOp>(
        op.getLoc(), op.getResultTensors()[0].getType(), ins[0], ins[1],
        /*bias=*/Value(), /*residual=*/Value(),
        ZeroPointAttr::get(op.getContext(), *izp, *wzp),
        rewriter.getDenseI64ArrayAttr(strides),
        rewriter.getDenseI64ArrayAttr({0, 0, 0, 0}),
        rewriter.getDenseI64ArrayAttr(dilations), /*epilogue=*/EpilogueAttr());

    rewriter.replaceOp(op,
                       applyInit(rewriter, op.getLoc(), conv, op.getOutputs()[0]));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// linalg.depthwise_conv_2d_nhwc_hwcm_q
//===----------------------------------------------------------------------===//

struct ConvertDepthwiseHwcmQ
    : public OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcmQOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::DepthwiseConv2DNhwcHwcmQOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "KEA Level 1 is value semantic; the memref form has no result");

    ValueRange ins = op.getInputs();
    if (ins.size() != 4)
      return rewriter.notifyMatchFailure(op, "expects 4 ins operands");

    FailureOr<int32_t> izp = constantZeroPoint(ins[2]);
    FailureOr<int32_t> wzp = constantZeroPoint(ins[3]);
    if (failed(izp) || failed(wzp))
      return rewriter.notifyMatchFailure(
          op, "zero point operands must be constants");

    auto rank5Ty =
        llvm::dyn_cast<RankedTensorType>(op.getResultTensors()[0].getType());
    if (!rank5Ty || rank5Ty.getRank() != 5)
      return rewriter.notifyMatchFailure(
          op, "expects the rank-5 [N, H, W, C, M] result");

    // KEA's depthwise result is rank 4 with the channel multiplier folded into
    // the channel dimension; linalg keeps it separate, so reshape back.
    ArrayRef<int64_t> s5 = rank5Ty.getShape();
    SmallVector<int64_t> shape4{s5[0], s5[1], s5[2], s5[3] * s5[4]};
    auto rank4Ty = RankedTensorType::get(shape4, rank5Ty.getElementType());

    Value weights =
        materializeCanonicalDepthwiseWeights(rewriter, op.getLoc(), ins[1]);

    SmallVector<int64_t> strides = denseToVector(op.getStrides());
    SmallVector<int64_t> dilations = denseToVector(op.getDilations());

    Value conv = rewriter.create<DWConv2DOp>(
        op.getLoc(), rank4Ty, ins[0], weights, /*bias=*/Value(),
        /*residual=*/Value(), ZeroPointAttr::get(op.getContext(), *izp, *wzp),
        rewriter.getDenseI64ArrayAttr(strides),
        rewriter.getDenseI64ArrayAttr({0, 0, 0, 0}),
        rewriter.getDenseI64ArrayAttr(dilations), /*epilogue=*/EpilogueAttr());

    Value reshaped = rewriter.create<ReshapeOp>(
        op.getLoc(), rank5Ty, conv, rewriter.getDenseI64ArrayAttr(s5));

    rewriter.replaceOp(
        op, applyInit(rewriter, op.getLoc(), reshaped, op.getOutputs()[0]));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// linalg.quantized_matmul / linalg.matmul
//===----------------------------------------------------------------------===//

/// `[M, K] x [K, N] -> [M, N]` via the rank-3 `kea.matmul`.
LogicalResult rewriteMatmulLike(Operation *op, Value a, Value b, int32_t aZp,
                                int32_t bZp, Value init, Type resultTy,
                                PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  auto aTy = llvm::dyn_cast<RankedTensorType>(a.getType());
  auto bTy = llvm::dyn_cast<RankedTensorType>(b.getType());
  auto outTy = llvm::dyn_cast<RankedTensorType>(resultTy);
  if (!aTy || !bTy || !outTy || aTy.getRank() != 2 || bTy.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "expects rank-2 tensor operands");

  auto batch = [&](RankedTensorType t) {
    SmallVector<int64_t> s{1, t.getDimSize(0), t.getDimSize(1)};
    return std::make_pair(
        RankedTensorType::get(s, t.getElementType()),
        SmallVector<int64_t>(s));
  };

  auto [a3Ty, a3Shape] = batch(aTy);
  auto [b3Ty, b3Shape] = batch(bTy);
  auto [o3Ty, o3Shape] = batch(outTy);

  Value a3 = rewriter.create<ReshapeOp>(
      loc, a3Ty, a, rewriter.getDenseI64ArrayAttr(a3Shape));
  Value b3 = rewriter.create<ReshapeOp>(
      loc, b3Ty, b, rewriter.getDenseI64ArrayAttr(b3Shape));
  Value mm = rewriter.create<MatmulOp>(
      loc, o3Ty, a3, b3, /*bias=*/Value(), /*residual=*/Value(),
      ZeroPointAttr::get(op->getContext(), aZp, bZp),
      /*epilogue=*/EpilogueAttr());
  Value flat = rewriter.create<ReshapeOp>(
      loc, outTy, mm, rewriter.getDenseI64ArrayAttr(outTy.getShape()));

  rewriter.replaceOp(op, applyInit(rewriter, loc, flat, init));
  return success();
}

struct ConvertQuantizedMatmul
    : public OpRewritePattern<linalg::QuantizedMatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::QuantizedMatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "KEA Level 1 is value semantic; the memref form has no result");
    ValueRange ins = op.getInputs();
    if (ins.size() != 4)
      return rewriter.notifyMatchFailure(op, "expects 4 ins operands");
    FailureOr<int32_t> aZp = constantZeroPoint(ins[2]);
    FailureOr<int32_t> bZp = constantZeroPoint(ins[3]);
    if (failed(aZp) || failed(bZp))
      return rewriter.notifyMatchFailure(
          op, "zero point operands must be constants");
    return rewriteMatmulLike(op, ins[0], ins[1], *aZp, *bZp, op.getOutputs()[0],
                             op.getResultTensors()[0].getType(), rewriter);
  }
};

struct ConvertMatmul : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "KEA Level 1 is value semantic; the memref form has no result");
    // `linalg.matmul indexing_maps = [...]` generalizes the op to transposed /
    // broadcast operands. It also does not round-trip in 20.1.6 (TOSA_NOTES
    // §13.10), so we deliberately do not try to support it.
    if (op.hasUserDefinedMaps())
      return rewriter.notifyMatchFailure(
          op, "linalg.matmul with user-defined indexing_maps is not supported");
    ValueRange ins = op.getInputs();
    if (ins.size() != 2)
      return rewriter.notifyMatchFailure(op, "expects 2 ins operands");
    return rewriteMatmulLike(op, ins[0], ins[1], /*aZp=*/0, /*bZp=*/0,
                             op.getOutputs()[0],
                             op.getResultTensors()[0].getType(), rewriter);
  }
};

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

/// linalg ops that MUST be converted for the result to be a legal Level 1
/// graph. `linalg.generic`, `linalg.fill`, `linalg.transpose` and friends are
/// left alone on purpose -- they are plumbing, not compute we own.
bool mustBeConverted(Operation *op) {
  if (!isa<linalg::LinalgOp>(op))
    return false;
  StringRef name = op->getName().stripDialect();
  return name.contains("conv") || name.contains("matmul") ||
         name.contains("pooling");
}

struct LinalgToKeaPass
    : public mlir::kea::impl::LinalgToKeaBase<LinalgToKeaPass> {
  using mlir::kea::impl::LinalgToKeaBase<LinalgToKeaPass>::LinalgToKeaBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgToKeaPatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Anything contraction-shaped that survived is something we do not model;
    // say so precisely rather than silently emitting a half-lowered graph.
    bool failed = false;
    getOperation().walk([&](Operation *op) {
      if (!mustBeConverted(op))
        return;
      op->emitError()
          << "-linalg-to-kea does not handle '" << op->getName()
          << "'; the supported named ops are linalg.conv_2d_nhwc_fhwc_q, "
             "linalg.depthwise_conv_2d_nhwc_hwcm_q, linalg.quantized_matmul "
             "and linalg.matmul, all in tensor form with constant zero points";
      failed = true;
    });
    if (failed)
      signalPassFailure();
  }
};

} // namespace

void mlir::kea::populateLinalgToKeaPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertConv2DNhwcFhwcQ, ConvertDepthwiseHwcmQ,
               ConvertQuantizedMatmul, ConvertMatmul>(patterns.getContext());
}
