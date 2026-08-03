//===- WeightLayout.cpp - depthwise weight normalisation --------*- C++ -*-===//
//
// KEA Level 1 has exactly ONE depthwise weight layout: canonical OHWI with
// IC == 1, i.e. `[OC, KH, KW, 1]` with `OC = C * M`. Both ingest paths spell
// depthwise weights HWCM `[KH, KW, C, M]`:
//
//   TOSA   tosa.depthwise_conv2d weight          [KH, KW, C, M]
//   linalg linalg.depthwise_conv_2d_nhwc_hwcm_q  [KH, KW, C, M]
//
// The index algebra of the normalisation is
//
//   w'[c * M + m, kh, kw, 0]  ==  w[kh, kw, c, m]
//
// which is a permutation `[2, 3, 0, 1]` (HWCM -> CMHW) followed by an
// element-order-preserving reshape, because the row-major linear index of
// `[C, M, KH, KW]` at `(c, m, kh, kw)` is `((c*M + m)*KH + kh)*KW + kw`, and
// the row-major linear index of `[C*M, KH, KW, 1]` at `(c*M + m, kh, kw, 0)`
// is the same expression.
//
//===----------------------------------------------------------------------===//

#include "kea/Conversion/Passes.h"
#include "kea/Dialect/KeaOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::kea;

namespace {

/// Constant-folds the HWCM -> canonical relayout.
DenseElementsAttr relayoutHWCMConstant(DenseElementsAttr src) {
  auto srcTy = llvm::cast<ShapedType>(src.getType());
  ArrayRef<int64_t> s = srcTy.getShape();
  int64_t kh = s[0], kw = s[1], c = s[2], m = s[3];

  auto dstTy = RankedTensorType::get({c * m, kh, kw, int64_t(1)},
                                     srcTy.getElementType());

  auto range = src.getValues<APInt>();
  SmallVector<APInt> in(range.begin(), range.end());
  SmallVector<APInt> out(in.size(), APInt(srcTy.getElementTypeBitWidth(), 0));
  for (int64_t i = 0; i < kh; ++i)
    for (int64_t j = 0; j < kw; ++j)
      for (int64_t ci = 0; ci < c; ++ci)
        for (int64_t mi = 0; mi < m; ++mi)
          out[((ci * m + mi) * kh + i) * kw + j] =
              in[((i * kw + j) * c + ci) * m + mi];

  return DenseElementsAttr::get(dstTy, out);
}

} // namespace

Value mlir::kea::materializeCanonicalDepthwiseWeights(OpBuilder &builder,
                                                      Location loc,
                                                      Value hwcm) {
  auto srcTy = llvm::cast<RankedTensorType>(hwcm.getType());
  ArrayRef<int64_t> s = srcTy.getShape();
  int64_t kh = s[0], kw = s[1], c = s[2], m = s[3];
  MLIRContext *ctx = builder.getContext();

  // The common case: an exporter emits the weights as a constant, so do the
  // relayout at compile time and leave no ops behind at all.
  DenseElementsAttr cst;
  if (matchPattern(hwcm, m_Constant(&cst)))
    return builder.create<arith::ConstantOp>(loc, relayoutHWCMConstant(cst));

  auto permTy = RankedTensorType::get({c, m, kh, kw}, srcTy.getElementType());
  Value permuted = builder.create<TransposeOp>(
      loc, permTy, hwcm, DenseI64ArrayAttr::get(ctx, {2, 3, 0, 1}));

  SmallVector<int64_t> canonShape{c * m, kh, kw, 1};
  auto canonTy = RankedTensorType::get(canonShape, srcTy.getElementType());
  return builder.create<ReshapeOp>(loc, canonTy, permuted,
                                   DenseI64ArrayAttr::get(ctx, canonShape));
}
