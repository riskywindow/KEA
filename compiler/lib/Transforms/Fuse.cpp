//===- Fuse.cpp - -kea-fuse -------------------------------------*- C++ -*-===//
//
// Level 1 -> Level 1 epilogue fusion.
//
// EVERY REWRITE IN THIS FILE IS BIT-EXACT. The pass moves work between ops but
// never changes the arithmetic, because the whole value of ingesting TOSA
// rather than linalg (ADR-0002) is that the quantization is preserved
// exactly. Where a rewrite would only be *approximately* right it is refused
// and counted in `num-rescale-refused`; see docs/DIALECT_L1.md §6.3 for the
// precise conditions.
//
//===----------------------------------------------------------------------===//

#include "kea/Dialect/KeaOps.h"
#include "kea/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace kea {
#define GEN_PASS_DEF_KEAFUSE
#include "kea/Transforms/Passes.h.inc"
} // namespace kea
} // namespace mlir

using namespace mlir;
using namespace mlir::kea;

namespace {

//===----------------------------------------------------------------------===//
// Counters
//===----------------------------------------------------------------------===//
//
// The patterns bump these plain counters and the pass copies them into the
// ODS-declared `Pass::Statistic`s at the end.
//
// WHY NOT INCREMENT THE Statistic DIRECTLY: the Homebrew LLVM 20.1.6 bottle is
// a Release build with NDEBUG and without LLVM_FORCE_ENABLE_STATS, so
// `llvm::Statistic` is `NoopStatistic` (llvm/ADT/Statistic.h line 38) and
// `-mlir-pass-statistics` prints an empty report. This is not specific to us:
// upstream `mlir-opt --symbol-dce -mlir-pass-statistics` prints nothing on
// this machine either, and it cannot be fixed from out of tree because the
// *printer* lives in the prebuilt libMLIRPass. Keeping a real counter means
// `-kea-fuse=report-stats=true` works here, while the `Statistic`s light up
// unchanged on a stats-enabled LLVM.

struct FusionCounters {
  unsigned requant = 0;
  unsigned clamp = 0;
  unsigned bias = 0;
  unsigned quantAdd = 0;
  unsigned residual = 0;
  unsigned composed = 0;
  unsigned removed = 0;
  unsigned refused = 0;
  unsigned shape = 0;
};

//===----------------------------------------------------------------------===//
// Contraction helpers
//===----------------------------------------------------------------------===//

bool isContraction(Operation *op) {
  return isa_and_nonnull<Conv2DOp, DWConv2DOp, MatmulOp, FullyConnectedOp>(op);
}

EpilogueAttr epilogueOf(Operation *op) {
  return op->getAttrOfType<EpilogueAttr>("epilogue");
}

Value residualOf(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<Conv2DOp, DWConv2DOp, MatmulOp, FullyConnectedOp>(
          [](auto typed) { return typed.getResidual(); })
      .Default([](Operation *) { return Value(); });
}

Value biasOf(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<Conv2DOp, DWConv2DOp, MatmulOp, FullyConnectedOp>(
          [](auto typed) { return typed.getBias(); })
      .Default([](Operation *) { return Value(); });
}

/// Rebuilds a contraction with a new result type / epilogue / bias / residual,
/// keeping everything else. Written as an explicit TypeSwitch over the four op
/// kinds rather than a generic OperationState clone, because the ops store
/// their attributes as ODS *properties* and a generic clone-with-new-types is
/// far easier to get subtly wrong.
Value rebuildContraction(PatternRewriter &rewriter, Operation *op,
                         Type resultTy, EpilogueAttr epilogue, Value bias,
                         Value residual) {
  Location loc = op->getLoc();
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<Conv2DOp>([&](Conv2DOp c) -> Value {
        return rewriter.create<Conv2DOp>(
            loc, resultTy, c.getInput(), c.getWeights(), bias, residual,
            c.getZeroPointsAttr(), c.getStridesAttr(), c.getPadsAttr(),
            c.getDilationsAttr(), epilogue);
      })
      .Case<DWConv2DOp>([&](DWConv2DOp c) -> Value {
        return rewriter.create<DWConv2DOp>(
            loc, resultTy, c.getInput(), c.getWeights(), bias, residual,
            c.getZeroPointsAttr(), c.getStridesAttr(), c.getPadsAttr(),
            c.getDilationsAttr(), epilogue);
      })
      .Case<MatmulOp>([&](MatmulOp c) -> Value {
        return rewriter.create<MatmulOp>(loc, resultTy, c.getA(), c.getB(),
                                         bias, residual, c.getZeroPointsAttr(),
                                         epilogue);
      })
      .Case<FullyConnectedOp>([&](FullyConnectedOp c) -> Value {
        return rewriter.create<FullyConnectedOp>(
            loc, resultTy, c.getInput(), c.getWeights(), bias, residual,
            c.getZeroPointsAttr(), epilogue);
      })
      .Default([](Operation *) { return Value(); });
}

/// True when `v` is available at `op`, i.e. it can legally become an operand.
bool dominatesOp(Value v, Operation *op) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return true; // block argument
  return def->getBlock() == op->getBlock() && def->isBeforeInBlock(op);
}

//===----------------------------------------------------------------------===//
// 1. contraction + kea.rescale -> epilogue.requant
//===----------------------------------------------------------------------===//

struct FuseRequant : public OpRewritePattern<RescaleOp> {
  FuseRequant(MLIRContext *ctx, unsigned &stat)
      : OpRewritePattern(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(RescaleOp op,
                                PatternRewriter &rewriter) const override {
    Operation *producer = op.getInput().getDefiningOp();
    if (!isContraction(producer) || !producer->hasOneUse())
      return failure();
    if (epilogueOf(producer))
      return failure(); // already fused; a second requant is a real op

    auto epilogue = EpilogueAttr::get(op.getContext(), op.getQuantAttr(),
                                      /*clamp=*/DenseI64ArrayAttr(),
                                      /*accum=*/QuantAttr(),
                                      /*residual=*/QuantAttr(),
                                      /*output=*/QuantAttr());

    rewriter.setInsertionPoint(producer);
    Value fused = rebuildContraction(rewriter, producer, op.getType(), epilogue,
                                     biasOf(producer), residualOf(producer));
    if (!fused)
      return failure();
    rewriter.replaceOp(op, fused);
    rewriter.eraseOp(producer);
    ++stat;
    return success();
  }

  unsigned &stat;
};

//===----------------------------------------------------------------------===//
// 2. contraction + kea.clamp -> epilogue.clamp
//===----------------------------------------------------------------------===//

struct FuseClamp : public OpRewritePattern<ClampOp> {
  FuseClamp(MLIRContext *ctx, unsigned &stat)
      : OpRewritePattern(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(ClampOp op,
                                PatternRewriter &rewriter) const override {
    Operation *producer = op.getInput().getDefiningOp();
    if (!isContraction(producer) || !producer->hasOneUse())
      return failure();

    EpilogueAttr epilogue = epilogueOf(producer);
    if (!epilogue || !epilogue.getRequant())
      return failure(); // clamping a raw i32 accumulator is a separate op
    if (epilogue.getClamp())
      return failure(); // one activation clamp per epilogue
    // The epilogue's clamp is stage B, immediately after `requant`. If the
    // residual stages are already populated, this clamp runs *after* `output`
    // instead and folding it here would reorder the arithmetic.
    if (epilogue.hasResidual())
      return failure();

    auto fusedEpilogue = EpilogueAttr::get(
        op.getContext(), epilogue.getRequant(),
        // NB: an I64Attr getter returns uint64_t, so cast before building the
        // signed array or the initializer list narrows.
        rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(op.getMin()),
                                       static_cast<int64_t>(op.getMax())}),
        epilogue.getAccum(), epilogue.getResidual(), epilogue.getOutput());

    rewriter.setInsertionPoint(producer);
    Value fused =
        rebuildContraction(rewriter, producer, op.getType(), fusedEpilogue,
                           biasOf(producer), residualOf(producer));
    if (!fused)
      return failure();
    rewriter.replaceOp(op, fused);
    rewriter.eraseOp(producer);
    ++stat;
    return success();
  }

  unsigned &stat;
};

//===----------------------------------------------------------------------===//
// 3. contraction + unquantized kea.add of a constant -> the bias operand
//===----------------------------------------------------------------------===//

/// Extracts a rank-1 `[OC]` i32 bias from a constant broadcast against the
/// contraction result. Only leading unit dimensions are allowed, so the
/// constant really is per output channel.
DenseElementsAttr extractBiasConstant(Value v, int64_t outChannels) {
  DenseElementsAttr cst;
  if (!matchPattern(v, m_Constant(&cst)))
    return {};
  auto ty = llvm::dyn_cast<RankedTensorType>(cst.getType());
  if (!ty || !ty.getElementType().isSignlessInteger(32))
    return {};
  for (int64_t d = 0, e = ty.getRank() - 1; d < e; ++d)
    if (ty.getDimSize(d) != 1)
      return {};
  if (ty.getRank() == 0 || ty.getDimSize(ty.getRank() - 1) != outChannels)
    return {};

  auto flatTy = RankedTensorType::get({outChannels}, ty.getElementType());
  auto range = cst.getValues<APInt>();
  SmallVector<APInt> values(range.begin(), range.end());
  return DenseElementsAttr::get(flatTy, values);
}

struct FuseBias : public OpRewritePattern<AddOp> {
  FuseBias(MLIRContext *ctx, unsigned &stat)
      : OpRewritePattern(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLhsQuantAttr() || op.getClamp())
      return failure(); // a quantized add is the residual pattern, not a bias

    for (unsigned i = 0; i < 2; ++i) {
      Value contractionSide = op->getOperand(i);
      Value constantSide = op->getOperand(1 - i);
      Operation *producer = contractionSide.getDefiningOp();
      if (!isContraction(producer) || !producer->hasOneUse())
        continue;
      if (epilogueOf(producer) || biasOf(producer))
        continue;
      if (contractionSide.getType() != op.getType())
        continue;

      auto resTy = llvm::cast<RankedTensorType>(op.getType());
      DenseElementsAttr bias = extractBiasConstant(
          constantSide, resTy.getDimSize(resTy.getRank() - 1));
      if (!bias)
        continue;

      rewriter.setInsertionPoint(producer);
      Value biasValue = rewriter.create<arith::ConstantOp>(op.getLoc(), bias);
      Value fused =
          rebuildContraction(rewriter, producer, op.getType(), EpilogueAttr(),
                             biasValue, residualOf(producer));
      if (!fused)
        return failure();
      rewriter.replaceOp(op, fused);
      rewriter.eraseOp(producer);
      ++stat;
      return success();
    }
    return failure();
  }

  unsigned &stat;
};

//===----------------------------------------------------------------------===//
// 4a. rescale/rescale/add/rescale -> one quantized kea.add
//===----------------------------------------------------------------------===//

struct FormQuantizedAdd : public OpRewritePattern<RescaleOp> {
  FormQuantizedAdd(MLIRContext *ctx, unsigned &stat)
      : OpRewritePattern(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(RescaleOp op,
                                PatternRewriter &rewriter) const override {
    auto add = op.getInput().getDefiningOp<AddOp>();
    if (!add || !add->hasOneUse() || add.getLhsQuantAttr() || add.getClamp())
      return failure();

    auto lhs = add.getLhs().getDefiningOp<RescaleOp>();
    auto rhs = add.getRhs().getDefiningOp<RescaleOp>();
    if (!lhs || !rhs || !lhs->hasOneUse() || !rhs->hasOneUse())
      return failure();

    // The shared accumulate domain must be i32: that is what makes the plain
    // add exact and is what KEA_VADD does internally.
    auto addTy = llvm::cast<RankedTensorType>(add.getType());
    if (!addTy.getElementType().isSignlessInteger(32))
      return failure();
    if (llvm::cast<RankedTensorType>(lhs.getType()) != addTy ||
        llvm::cast<RankedTensorType>(rhs.getType()) != addTy)
      return failure();

    // KEA_VADD is per tensor (docs/ISA.md §10.2); a per-channel add would need
    // a different instruction, so leave those alone.
    if (lhs.getQuantAttr().isPerChannel() ||
        rhs.getQuantAttr().isPerChannel() || op.getQuantAttr().isPerChannel())
      return failure();

    if (lhs.getInput().getType() != rhs.getInput().getType())
      return failure();

    rewriter.replaceOpWithNewOp<AddOp>(
        op, op.getType(), lhs.getInput(), rhs.getInput(), lhs.getQuantAttr(),
        rhs.getQuantAttr(), op.getQuantAttr(), /*clamp=*/DenseI64ArrayAttr());
    ++stat;
    return success();
  }

  unsigned &stat;
};

//===----------------------------------------------------------------------===//
// 4b. contraction + quantized kea.add -> epilogue residual triple
//===----------------------------------------------------------------------===//

struct FuseResidual : public OpRewritePattern<AddOp> {
  FuseResidual(MLIRContext *ctx, unsigned &stat)
      : OpRewritePattern(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getLhsQuantAttr() || !op.getRhsQuantAttr() || !op.getOutQuantAttr())
      return failure();
    // The epilogue has a single clamp slot and it sits before the residual
    // stages, so a clamp on the add itself cannot be represented.
    if (op.getClamp())
      return failure();

    for (unsigned i = 0; i < 2; ++i) {
      Value contractionSide = op->getOperand(i);
      Value residual = op->getOperand(1 - i);
      Operation *producer = contractionSide.getDefiningOp();
      if (!isContraction(producer) || !producer->hasOneUse())
        continue;

      EpilogueAttr epilogue = epilogueOf(producer);
      if (!epilogue || !epilogue.getRequant() || epilogue.hasResidual())
        continue;
      if (residualOf(producer))
        continue;
      if (residual.getType() != op.getType())
        continue;
      // The residual must already be computable where the contraction is.
      if (!dominatesOp(residual, producer))
        continue;

      QuantAttr accum = i == 0 ? op.getLhsQuantAttr() : op.getRhsQuantAttr();
      QuantAttr resQuant = i == 0 ? op.getRhsQuantAttr() : op.getLhsQuantAttr();

      auto fusedEpilogue = EpilogueAttr::get(
          op.getContext(), epilogue.getRequant(), epilogue.getClamp(), accum,
          resQuant, op.getOutQuantAttr());

      rewriter.setInsertionPoint(producer);
      Value fused = rebuildContraction(rewriter, producer, op.getType(),
                                       fusedEpilogue, biasOf(producer),
                                       residual);
      if (!fused)
        return failure();
      rewriter.replaceOp(op, fused);
      rewriter.eraseOp(producer);
      ++stat;
      return success();
    }
    return failure();
  }

  unsigned &stat;
};

//===----------------------------------------------------------------------===//
// 5. rescale algebra
//===----------------------------------------------------------------------===//

/// A rescale is a bit-exact no-op when it neither scales nor moves the zero
/// point and cannot clamp (same in/out element type).
struct DropNoOpRescale : public OpRewritePattern<RescaleOp> {
  DropNoOpRescale(MLIRContext *ctx, unsigned &stat)
      : OpRewritePattern(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(RescaleOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInput().getType() != op.getType())
      return failure();
    if (!op.getQuantAttr().isNoOp())
      return failure();
    rewriter.replaceOp(op, op.getInput());
    ++stat;
    return success();
  }

  unsigned &stat;
};

/// Collapses `b(a(x))` in the two cases where composition is EXACT.
///
/// Both require the intermediate tensor to be `i32`, so that `a` cannot clamp
/// (an i32 result clamped to the i32 range is the identity). Then:
///
///  (a) `a` only rebases the zero point (`isIdentityScale`), so
///      `a(x) = x - a.izp + a.ozp` exactly, and the pair equals `b` with
///      `izp' = a.izp - a.ozp + b.izp`.
///
///  (b) `b` only rebases the zero point, so the pair equals `a` with
///      `ozp' = a.ozp - b.izp + b.ozp` and `b`'s result type (whose clamp is
///      then the only clamp, exactly as in the two-op form).
///
/// Anything else -- in particular the ubiquitous `i32 -> i8 -> i32` pair -- is
/// REFUSED, because the intermediate `i8` saturation and the intermediate
/// rounding are both observable and cannot be recovered from a single
/// multiply-shift.
struct ComposeRescales : public OpRewritePattern<RescaleOp> {
  ComposeRescales(MLIRContext *ctx, unsigned &stat)
      : OpRewritePattern(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(RescaleOp b,
                                PatternRewriter &rewriter) const override {
    auto a = b.getInput().getDefiningOp<RescaleOp>();
    if (!a || !a->hasOneUse())
      return failure();

    auto midTy = llvm::cast<RankedTensorType>(a.getType());
    if (!midTy.getElementType().isSignlessInteger(32))
      return failure(); // the intermediate saturates: not composable

    QuantAttr qa = a.getQuantAttr(), qb = b.getQuantAttr();
    MLIRContext *ctx = b.getContext();

    if (qa.isIdentityScale()) {
      auto composed = QuantAttr::get(
          ctx, qb.getMultiplier(), qb.getShift(),
          qa.getInputZp() - qa.getOutputZp() + qb.getInputZp(),
          qb.getOutputZp(), qb.getAxis(), qb.getRounding());
      rewriter.replaceOpWithNewOp<RescaleOp>(b, b.getType(), a.getInput(),
                                             composed);
      ++stat;
      return success();
    }

    if (qb.isIdentityScale()) {
      auto composed = QuantAttr::get(
          ctx, qa.getMultiplier(), qa.getShift(), qa.getInputZp(),
          qa.getOutputZp() - qb.getInputZp() + qb.getOutputZp(), qa.getAxis(),
          qa.getRounding());
      rewriter.replaceOpWithNewOp<RescaleOp>(b, b.getType(), a.getInput(),
                                             composed);
      ++stat;
      return success();
    }

    return failure();
  }

  unsigned &stat;
};

//===----------------------------------------------------------------------===//
// 6. layout no-ops
//===----------------------------------------------------------------------===//

/// A permutation is a *layout* no-op when the dimensions of extent != 1 keep
/// their relative order: only unit dimensions move, so no element changes its
/// row-major position.
bool isLayoutNoOpPermutation(ArrayRef<int64_t> shape, ArrayRef<int64_t> perms) {
  int64_t last = -1;
  for (int64_t p : perms) {
    if (shape[p] == 1)
      continue;
    if (p < last)
      return false;
    last = p;
  }
  return true;
}

struct FoldTranspose : public OpRewritePattern<TransposeOp> {
  FoldTranspose(MLIRContext *ctx, unsigned &stat)
      : OpRewritePattern(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto inTy = llvm::cast<RankedTensorType>(op.getInput().getType());
    ArrayRef<int64_t> perms = op.getPerms();

    bool identity = true;
    for (auto [d, p] : llvm::enumerate(perms))
      identity &= (static_cast<int64_t>(d) == p);
    if (identity) {
      rewriter.replaceOp(op, op.getInput());
      ++stat;
      return success();
    }

    if (!isLayoutNoOpPermutation(inTy.getShape(), perms))
      return failure();

    auto outTy = llvm::cast<RankedTensorType>(op.getType());
    rewriter.replaceOpWithNewOp<ReshapeOp>(
        op, outTy, op.getInput(),
        rewriter.getDenseI64ArrayAttr(outTy.getShape()));
    ++stat;
    return success();
  }

  unsigned &stat;
};

struct FoldReshape : public OpRewritePattern<ReshapeOp> {
  FoldReshape(MLIRContext *ctx, unsigned &stat)
      : OpRewritePattern(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInput().getType() == op.getType()) {
      rewriter.replaceOp(op, op.getInput());
      ++stat;
      return success();
    }
    // reshape(reshape(x)) is one reshape: both are row-major reinterpretations.
    if (auto inner = op.getInput().getDefiningOp<ReshapeOp>()) {
      if (!inner->hasOneUse())
        return failure();
      rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), inner.getInput(),
                                             op.getNewShapeAttr());
      ++stat;
      return success();
    }
    return failure();
  }

  unsigned &stat;
};

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct KeaFusePass : public mlir::kea::impl::KeaFuseBase<KeaFusePass> {
  using mlir::kea::impl::KeaFuseBase<KeaFusePass>::KeaFuseBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    FusionCounters counts;

    RewritePatternSet patterns(ctx);
    patterns.add<FuseRequant>(ctx, counts.requant);
    patterns.add<FuseClamp>(ctx, counts.clamp);
    patterns.add<FuseBias>(ctx, counts.bias);
    patterns.add<FormQuantizedAdd>(ctx, counts.quantAdd);
    patterns.add<FuseResidual>(ctx, counts.residual);
    patterns.add<DropNoOpRescale>(ctx, counts.removed);
    patterns.add<ComposeRescales>(ctx, counts.composed);
    patterns.add<FoldTranspose>(ctx, counts.shape);
    patterns.add<FoldReshape>(ctx, counts.shape);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Count the rescale chains we deliberately did NOT collapse. Doing it here
    // rather than inside ComposeRescales keeps the number deterministic: the
    // greedy driver may re-visit a failing pattern any number of times.
    getOperation().walk([&](RescaleOp op) {
      if (op.getInput().getDefiningOp<RescaleOp>())
        ++counts.refused;
    });

    numRequantFused = counts.requant;
    numClampFused = counts.clamp;
    numBiasFused = counts.bias;
    numQuantAddFormed = counts.quantAdd;
    numResidualFused = counts.residual;
    numRescaleComposed = counts.composed;
    numRescaleRemoved = counts.removed;
    numRescaleRefused = counts.refused;
    numShapeOpsFolded = counts.shape;

    if (reportStats) {
      OpBuilder builder(ctx);
      auto u = [&](unsigned v) { return builder.getI64IntegerAttr(v); };
      getOperation()->setAttr(
          "kea.fusion_stats",
          builder.getDictionaryAttr({
              builder.getNamedAttr("bias", u(counts.bias)),
              builder.getNamedAttr("clamp", u(counts.clamp)),
              builder.getNamedAttr("quant_add", u(counts.quantAdd)),
              builder.getNamedAttr("requant", u(counts.requant)),
              builder.getNamedAttr("rescale_composed", u(counts.composed)),
              builder.getNamedAttr("rescale_refused", u(counts.refused)),
              builder.getNamedAttr("rescale_removed", u(counts.removed)),
              builder.getNamedAttr("residual", u(counts.residual)),
              builder.getNamedAttr("shape_folded", u(counts.shape)),
          }));
    }
  }
};

} // namespace
