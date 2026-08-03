//===- TosaToKea.cpp - -tosa-to-kea ----------------------------*- C++ -*-===//
//
// Quantized TOSA (LLVM/MLIR 20.1.6 flavour -- see docs/TOSA_NOTES.md) into KEA
// Level 1. This pass NORMALISES; it deliberately does not fuse. `-kea-fuse`
// is what turns `kea.conv2d` + `kea.rescale` + `kea.clamp` into one op.
//
// The 20.1.6 specifics this file depends on, all verified in TOSA_NOTES:
//   * zero points are ATTRIBUTES (`#tosa.conv_quant`, `#tosa.matmul_quant`,
//     `#tosa.unary_quant`), not operands;
//   * `tosa.rescale`'s `shift` is `array<i8>` and `scale32`/`double_round`/
//     `per_channel` are mandatory;
//   * `tosa.reshape` takes `new_shape` as an ATTRIBUTE (which may contain -1);
//   * `tosa.transpose` takes its permutation as an OPERAND;
//   * `tosa.depthwise_conv2d` weights are HWCM while `tosa.conv2d` weights are
//     OHWI;
//   * unknown attributes are silently accepted on TOSA ops, so this file reads
//     only the real ODS operands/attributes and never trusts a stray one --
//     notably `tosa.max_pool2d` has no quantization at all and any
//     `quantization_info` written on it is ignored here, exactly as every
//     upstream pass ignores it.
//
//===----------------------------------------------------------------------===//

#include "kea/Conversion/Passes.h"
#include "kea/Dialect/KeaOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace kea {
#define GEN_PASS_DEF_TOSATOKEA
#include "kea/Transforms/Passes.h.inc"
} // namespace kea
} // namespace mlir

using namespace mlir;
using namespace mlir::kea;

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

namespace {

RankedTensorType rankedOf(Value v) {
  return llvm::dyn_cast<RankedTensorType>(v.getType());
}

/// Builds `#kea.zp` from an optional `#tosa.conv_quant`. Absent means 0/0.
ZeroPointAttr convZeroPoints(MLIRContext *ctx,
                             std::optional<tosa::ConvOpQuantizationAttr> q) {
  if (!q)
    return ZeroPointAttr::get(ctx, 0, 0);
  return ZeroPointAttr::get(ctx, static_cast<int32_t>(q->getInputZp()),
                            static_cast<int32_t>(q->getWeightZp()));
}

/// Builds `#kea.quant` from a `tosa.rescale`, or fails with a diagnostic on
/// the parts of TOSA rescale the KEA VPU cannot express.
FailureOr<QuantAttr> quantFromRescale(tosa::RescaleOp op,
                                      ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = op.getContext();

  // These are hard rejections rather than silent match failures: the user
  // needs to know *why* an otherwise ordinary rescale did not convert.
  if (!op.getScale32())
    return op.emitError(
        "scale32 = false (the 16-bit multiplier path) has no KEA Level 1 "
        "equivalent; KEA_VQUANT's multiplier is a normalized Q31 value "
        "(docs/ISA.md 10.1)");
  if (op.getInputUnsigned() || op.getOutputUnsigned())
    return op.emitError("unsigned rescale operands are not modelled at KEA "
                        "Level 1; all activations are signed int8");

  auto outTy = rankedOf(op.getOutput());
  if (!outTy)
    return rewriter.notifyMatchFailure(op, "expects a ranked result");

  ArrayRef<int32_t> mult = op.getMultiplier();
  ArrayRef<int8_t> shift = op.getShift();

  // TOSA per-channel always indexes the LAST dimension (TOSA_NOTES §4).
  int64_t axis = op.getPerChannel() ? outTy.getRank() - 1 : -1;
  if (op.getPerChannel() && static_cast<int64_t>(mult.size()) !=
                                outTy.getDimSize(outTy.getRank() - 1))
    return rewriter.notifyMatchFailure(
        op, "per-channel rescale length does not match the last dimension");

  return QuantAttr::get(
      ctx, DenseI32ArrayAttr::get(ctx, mult), DenseI8ArrayAttr::get(ctx, shift),
      static_cast<int32_t>(op.getInputZp()),
      static_cast<int32_t>(op.getOutputZp()), axis,
      op.getDoubleRound() ? RoundingMode::DOUBLE : RoundingMode::SINGLE);
}

/// The identity requantization: `out = in - inputZp + outputZp`, no scaling.
/// Spelled as the normalised Q31 identity `multiplier = 2^30, shift = 30`
/// rather than `1 / 0`, so that it stays inside TOSA's required shift range
/// (docs/QUANTIZATION.md §1) and inside KEA_VQUANT's normalised-multiplier
/// window. `(v << 30 + 2^29) >> 30 == v` exactly for every i32 `v`.
QuantAttr identityQuant(MLIRContext *ctx, int32_t inputZp, int32_t outputZp) {
  return QuantAttr::get(ctx, DenseI32ArrayAttr::get(ctx, {1 << 30}),
                        DenseI8ArrayAttr::get(ctx, {30}), inputZp, outputZp,
                        /*axis=*/-1, RoundingMode::SINGLE);
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

struct ConvertConst : public OpConversionPattern<tosa::ConstOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::ConstOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = llvm::dyn_cast<TypedAttr>(op.getValue());
    if (!value)
      return rewriter.notifyMatchFailure(op, "non-typed constant value");
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getType(), value);
    return success();
  }
};

struct ConvertConv2D : public OpConversionPattern<tosa::Conv2DOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<Conv2DOp>(
        op, op.getType(), adaptor.getInput(), adaptor.getWeight(),
        adaptor.getBias(), /*residual=*/Value(),
        convZeroPoints(op.getContext(), op.getQuantizationInfo()),
        rewriter.getDenseI64ArrayAttr(op.getStride()),
        rewriter.getDenseI64ArrayAttr(op.getPad()),
        rewriter.getDenseI64ArrayAttr(op.getDilation()),
        /*epilogue=*/EpilogueAttr());
    return success();
  }
};

struct ConvertDepthwiseConv2D
    : public OpConversionPattern<tosa::DepthwiseConv2DOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::DepthwiseConv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto wTy = rankedOf(adaptor.getWeight());
    if (!wTy || wTy.getRank() != 4)
      return rewriter.notifyMatchFailure(op, "expects rank-4 HWCM weights");

    // HWCM [KH, KW, C, M] -> canonical [C*M, KH, KW, 1]; see
    // lib/Conversion/WeightLayout.cpp.
    Value weights = materializeCanonicalDepthwiseWeights(
        rewriter, op.getLoc(), adaptor.getWeight());

    rewriter.replaceOpWithNewOp<DWConv2DOp>(
        op, op.getType(), adaptor.getInput(), weights, adaptor.getBias(),
        /*residual=*/Value(),
        convZeroPoints(op.getContext(), op.getQuantizationInfo()),
        rewriter.getDenseI64ArrayAttr(op.getStride()),
        rewriter.getDenseI64ArrayAttr(op.getPad()),
        rewriter.getDenseI64ArrayAttr(op.getDilation()),
        /*epilogue=*/EpilogueAttr());
    return success();
  }
};

struct ConvertMatMul : public OpConversionPattern<tosa::MatMulOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto q = op.getQuantizationInfo();
    auto zp = q ? ZeroPointAttr::get(op.getContext(),
                                     static_cast<int32_t>(q->getAZp()),
                                     static_cast<int32_t>(q->getBZp()))
                : ZeroPointAttr::get(op.getContext(), 0, 0);
    rewriter.replaceOpWithNewOp<MatmulOp>(
        op, op.getType(), adaptor.getA(), adaptor.getB(), /*bias=*/Value(),
        /*residual=*/Value(), zp, /*epilogue=*/EpilogueAttr());
    return success();
  }
};

struct ConvertFullyConnected
    : public OpConversionPattern<tosa::FullyConnectedOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::FullyConnectedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<FullyConnectedOp>(
        op, op.getType(), adaptor.getInput(), adaptor.getWeight(),
        adaptor.getBias(), /*residual=*/Value(),
        convZeroPoints(op.getContext(), op.getQuantizationInfo()),
        /*epilogue=*/EpilogueAttr());
    return success();
  }
};

struct ConvertAdd : public OpConversionPattern<tosa::AddOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // tosa.add carries no quantization by construction (TOSA_NOTES §6): the
    // quantization lives in the surrounding rescales, and -kea-fuse is what
    // pulls it in.
    rewriter.replaceOpWithNewOp<AddOp>(
        op, op.getType(), adaptor.getInput1(), adaptor.getInput2(),
        /*lhs_quant=*/QuantAttr(), /*rhs_quant=*/QuantAttr(),
        /*out_quant=*/QuantAttr(), /*clamp=*/DenseI64ArrayAttr());
    return success();
  }
};

struct ConvertRescale : public OpConversionPattern<tosa::RescaleOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::RescaleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<QuantAttr> quant = quantFromRescale(op, rewriter);
    if (failed(quant))
      return failure();
    rewriter.replaceOpWithNewOp<RescaleOp>(op, op.getType(), adaptor.getInput(),
                                           *quant);
    return success();
  }
};

struct ConvertClamp : public OpConversionPattern<tosa::ClampOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::ClampOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ty = rankedOf(op.getOutput());
    if (!ty || !ty.getElementType().isSignlessInteger())
      return rewriter.notifyMatchFailure(
          op, "only integer clamps exist at KEA Level 1; min_fp/max_fp are "
              "ignored, as they are by TOSA's own integer lowering");
    rewriter.replaceOpWithNewOp<ClampOp>(op, ty, adaptor.getInput(),
                                         op.getMinIntAttr(),
                                         op.getMaxIntAttr());
    return success();
  }
};

struct ConvertAvgPool : public OpConversionPattern<tosa::AvgPool2dOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::AvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    QuantAttr quant;
    if (auto q = op.getQuantizationInfo()) {
      int32_t izp = static_cast<int32_t>(q->getInputZp());
      int32_t ozp = static_cast<int32_t>(q->getOutputZp());
      // Averaging is affine, so a zero-point rebase commutes exactly with it:
      // avg(x_i - izp) + ozp == avg(x_i) - izp + ozp, and the division is by
      // the same (padding-aware) count on both sides. Anything beyond a
      // rebase would need a real multiplier, which VPOOL does not have.
      if (izp != 0 || ozp != 0)
        quant = identityQuant(ctx, izp, ozp);
    }
    rewriter.replaceOpWithNewOp<PoolOp>(
        op, op.getType(), adaptor.getInput(),
        PoolKindAttr::get(ctx, PoolKind::AVG),
        rewriter.getDenseI64ArrayAttr(op.getKernel()),
        rewriter.getDenseI64ArrayAttr(op.getStride()),
        rewriter.getDenseI64ArrayAttr(op.getPad()), quant);
    return success();
  }
};

struct ConvertMaxPool : public OpConversionPattern<tosa::MaxPool2dOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::MaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // NB: tosa.max_pool2d has NO acc_type and NO quantization_info in its ODS
    // definition. A `quantization_info` written on it parses and round-trips
    // as a discardable attribute (TOSA_NOTES §13.17) and is meaningless; we
    // read the real operand list only, so it is correctly ignored here.
    rewriter.replaceOpWithNewOp<PoolOp>(
        op, op.getType(), adaptor.getInput(),
        PoolKindAttr::get(op.getContext(), PoolKind::MAX),
        rewriter.getDenseI64ArrayAttr(op.getKernel()),
        rewriter.getDenseI64ArrayAttr(op.getStride()),
        rewriter.getDenseI64ArrayAttr(op.getPad()), /*quant=*/QuantAttr());
    return success();
  }
};

struct ConvertReshape : public OpConversionPattern<tosa::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outTy = rankedOf(op.getOutput());
    if (!outTy || !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expects a static result shape");
    // `new_shape` may contain a single -1; Level 1 requires it resolved, and
    // the (static) result type already carries the answer.
    rewriter.replaceOpWithNewOp<ReshapeOp>(
        op, outTy, adaptor.getInput1(),
        rewriter.getDenseI64ArrayAttr(outTy.getShape()));
    return success();
  }
};

struct ConvertTranspose : public OpConversionPattern<tosa::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // In 20.1.6 the permutation is an OPERAND, not an attribute. Fold it.
    DenseIntElementsAttr perms;
    if (!matchPattern(op.getPerms(), m_Constant(&perms)))
      return op.emitError(
          "tosa.transpose's permutation operand must be constant to become "
          "the kea.transpose 'perms' attribute");
    SmallVector<int64_t> permValues;
    for (const APInt &p : perms.getValues<APInt>())
      permValues.push_back(p.getSExtValue());

    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, op.getType(), adaptor.getInput1(),
        rewriter.getDenseI64ArrayAttr(permValues));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct TosaToKeaPass : public mlir::kea::impl::TosaToKeaBase<TosaToKeaPass> {
  using mlir::kea::impl::TosaToKeaBase<TosaToKeaPass>::TosaToKeaBase;

  void runOnOperation() override {
    ConversionTarget target(getContext());
    configureTosaToKeaTarget(target);

    RewritePatternSet patterns(&getContext());
    populateTosaToKeaPatterns(patterns);

    if (failed(applyFullConversion(getOperation(), target,
                                   std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // The permutation constants of tosa.transpose, and the original HWCM
    // depthwise weights, become dead `arith.constant`s. Clean them up so the
    // output is the normalised graph and nothing else.
    SmallVector<Operation *> ops;
    getOperation().walk([&](Operation *op) { ops.push_back(op); });
    for (Operation *op : llvm::reverse(ops))
      if (isOpTriviallyDead(op))
        op->erase();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void mlir::kea::populateTosaToKeaPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertConst, ConvertConv2D, ConvertDepthwiseConv2D,
               ConvertMatMul, ConvertFullyConnected, ConvertAdd,
               ConvertRescale, ConvertClamp, ConvertAvgPool, ConvertMaxPool,
               ConvertReshape, ConvertTranspose>(patterns.getContext());
}

void mlir::kea::configureTosaToKeaTarget(ConversionTarget &target) {
  target.addIllegalDialect<tosa::TosaDialect>();
  target.addLegalDialect<KeaDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.markUnknownOpDynamicallyLegal([](Operation *op) {
    return op->getDialect() &&
           op->getDialect()->getNamespace() != tosa::TosaDialect::getDialectNamespace();
  });
}
