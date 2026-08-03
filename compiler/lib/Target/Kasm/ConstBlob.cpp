//===- ConstBlob.cpp - materialize the CONST blob ---------------*- C++ -*-===//
//
// docs/DIALECT_L2.md §4.5 lists seven `layout` names a DRAM `kea.alloc` can
// declare. This file turns each of them into bytes. Every formula here is a
// transcription of ISA.md §8.1 / §8.6 / §9.1 or of DIALECT_L2.md §4.3 / §4.4 --
// nothing is invented, and `-kea-tile` already sized every buffer with the same
// arithmetic, so a mismatch between the declared extent and what this file
// produces is a hard error rather than a silent truncation.
//
//===----------------------------------------------------------------------===//

#include "kea/Target/Kasm/EmitKasm.h"

#include "kea/Dialect/KeaAttrs.h"
#include "kea/Dialect/KeaMachineOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "kea/hw_config.h"
#include "kea/isa.h"

#include <algorithm>
#include <climits>

using namespace mlir;
using namespace mlir::kea;

namespace {

// Every constant here comes from the frozen machine contract; nothing is a
// number this file chose.
constexpr int64_t kMxuK = ::kea::KEA_MXU_K;
constexpr int64_t kMxuN = ::kea::KEA_MXU_N;
constexpr int64_t kTileBytes = kMxuK * kMxuN; // one dense 16x16 int8 tile
constexpr int64_t kVpuLanes = ::kea::KEA_VPU_LANES;
constexpr int64_t kQuantParamBytes = sizeof(::kea::KeaQuantParam);
constexpr int64_t kAddParamBytes = sizeof(::kea::KeaAddParam);

int64_t roundUp(int64_t v, int64_t m) { return ((v + m - 1) / m) * m; }

void putI32(std::vector<uint8_t> &out, size_t off, int32_t v) {
  auto u = static_cast<uint32_t>(v);
  out[off + 0] = static_cast<uint8_t>(u);
  out[off + 1] = static_cast<uint8_t>(u >> 8);
  out[off + 2] = static_cast<uint8_t>(u >> 16);
  out[off + 3] = static_cast<uint8_t>(u >> 24);
}

/// Every element of a constant tensor, row major, sign extended.
bool readConstant(Value v, SmallVectorImpl<int64_t> &flat,
                  SmallVectorImpl<int64_t> &shape) {
  DenseElementsAttr cst;
  if (!v || !matchPattern(v, m_Constant(&cst)))
    return false;
  auto ty = llvm::dyn_cast<ShapedType>(cst.getType());
  if (!ty || !ty.getElementType().isIntOrIndex())
    return false;
  shape.assign(ty.getShape().begin(), ty.getShape().end());
  flat.clear();
  flat.reserve(ty.getNumElements());
  for (const APInt &e : cst.getValues<APInt>())
    flat.push_back(e.getSExtValue());
  return true;
}

/// `[OC, KH, KW, IC]`, normalising the rank-2 `[OC, IC]` fully-connected case
/// and the rank-3 `[B, K, N]` batched-matmul case. `ocLast` is set for the
/// latter, which is the one shape whose output channel is the trailing
/// dimension (docs/DIALECT_L2.md §4.5, `mxu_tiles_16x16_kn`).
struct WeightShape {
  int64_t OC = 0, KH = 1, KW = 1, IC = 0, batch = 1;
  bool ocLast = false;

  /// Index into the row-major element array.
  int64_t index(int64_t b, int64_t oc, int64_t kh, int64_t kw,
                int64_t ic) const {
    if (ocLast)
      return ((b * IC + ic) * OC) + oc; // [B][K][N]
    return (((oc * KH + kh) * KW + kw) * IC) + ic;
  }
};

LogicalResult weightShapeOf(Operation *op, ArrayRef<int64_t> shape,
                            WeightShape &ws) {
  switch (shape.size()) {
  case 2: // fully connected: [OC, IC]
    ws.OC = shape[0];
    ws.IC = shape[1];
    return success();
  case 3: // batched matmul rhs: [B, K, N]
    ws.batch = shape[0];
    ws.IC = shape[1];
    ws.OC = shape[2];
    ws.ocLast = true;
    return success();
  case 4: // conv / depthwise: [OC, KH, KW, IC]
    ws.OC = shape[0];
    ws.KH = shape[1];
    ws.KW = shape[2];
    ws.IC = shape[3];
    return success();
  default:
    return op->emitOpError("weight constant must be rank 2 ([OC, IC]), rank 3 "
                           "([B, K, N]) or rank 4 ([OC, KH, KW, IC]), got rank ")
           << shape.size();
  }
}

//===----------------------------------------------------------------------===//
// mxu_tiles_16x16 / _packed / _kn -- ISA.md §8.1 and §8.6
//===----------------------------------------------------------------------===//

/// Dense 16x16 int8 tiles of 256 bytes ordered `[batch][oc0][ic0][kh][kw]`,
/// with `W[k][n]` at byte `k*16 + n` inside a tile and zero everywhere the
/// real weight array does not reach -- which is exactly how `LOAD_W`'s
/// `k_rows` / `n_cols` tail tiles work without any masking (ISA.md §7.2).
LogicalResult layoutMxuTiles(Operation *op, ArrayRef<int64_t> w,
                             const WeightShape &ws, bool packed,
                             std::vector<uint8_t> &out) {
  const int64_t icTiles = packed ? 1 : (ws.IC + kMxuK - 1) / kMxuK;
  const int64_t ocGroups = (ws.OC + kMxuN - 1) / kMxuN;
  const int64_t taps = packed ? ws.KH : (ws.KH * ws.KW);
  const int64_t kRows = packed ? (ws.KW * ws.IC) : std::min(ws.IC, kMxuK);

  if (packed && ws.KW * ws.IC > kMxuK)
    return op->emitOpError("layout \"mxu_tiles_16x16_packed\" needs KW*IC <= ")
           << kMxuK << " (ISA.md §8.6), got " << ws.KW << "*" << ws.IC;

  out.assign(ws.batch * ocGroups * icTiles * taps * kTileBytes, 0);

  for (int64_t b = 0; b < ws.batch; ++b)
    for (int64_t oc0 = 0; oc0 < ocGroups; ++oc0)
      for (int64_t ic0 = 0; ic0 < icTiles; ++ic0)
        for (int64_t kh = 0; kh < ws.KH; ++kh)
          for (int64_t kw = 0; kw < (packed ? 1 : ws.KW); ++kw) {
            const int64_t tile =
                ((((b * ocGroups + oc0) * icTiles + ic0) * ws.KH + kh) *
                     (packed ? 1 : ws.KW) +
                 kw) *
                kTileBytes;
            for (int64_t k = 0; k < kRows; ++k)
              for (int64_t n = 0; n < kMxuN; ++n) {
                const int64_t oc = oc0 * kMxuN + n;
                if (oc >= ws.OC)
                  continue;
                int64_t srcKw = kw, srcIc;
                if (packed) {
                  // The reduction row IS (kw, ic): a whole kernel row folded
                  // into one tile, `W[kw][ic][oc]` in that order.
                  srcKw = k / ws.IC;
                  srcIc = k % ws.IC;
                } else {
                  srcIc = ic0 * kMxuK + k;
                  if (srcIc >= ws.IC)
                    continue;
                }
                out[tile + k * kMxuN + n] = static_cast<uint8_t>(
                    static_cast<int8_t>(w[ws.index(b, oc, kh, srcKw, srcIc)]));
              }
          }
  return success();
}

//===----------------------------------------------------------------------===//
// dwu_planes -- ISA.md §9.1
//===----------------------------------------------------------------------===//

/// `[KH][KW][C_pad]` int8, contiguous, `C_pad = roundUp(C, 16)` bytes per tap
/// plane, zero in the padding lanes. The zeros matter: `DWCONV.channels` is
/// rounded up to a multiple of 16, so those lanes are read, and a zero weight
/// is what makes reading them harmless (DIALECT_L2.md §6.4).
LogicalResult layoutDwuPlanes(Operation *op, ArrayRef<int64_t> w,
                              const WeightShape &ws,
                              std::vector<uint8_t> &out) {
  if (ws.IC != 1)
    return op->emitOpError("layout \"dwu_planes\" expects canonical depthwise "
                           "weights [OC, KH, KW, 1], got IC = ")
           << ws.IC;
  const int64_t cPad = roundUp(ws.OC, kVpuLanes);
  out.assign(ws.KH * ws.KW * cPad, 0);
  for (int64_t kh = 0; kh < ws.KH; ++kh)
    for (int64_t kw = 0; kw < ws.KW; ++kw)
      for (int64_t c = 0; c < ws.OC; ++c)
        out[(kh * ws.KW + kw) * cPad + c] =
            static_cast<uint8_t>(static_cast<int8_t>(w[ws.index(0, c, kh, kw, 0)]));
  return success();
}

//===----------------------------------------------------------------------===//
// quant_params -- docs/DIALECT_L2.md §4.3
//===----------------------------------------------------------------------===//

LogicalResult layoutQuantParams(AllocOp alloc, std::vector<uint8_t> &out) {
  Operation *op = alloc.getOperation();
  auto quant = alloc.getQuant();
  if (!quant)
    return op->emitOpError("layout \"quant_params\" needs a `quant` attribute");
  if (alloc.getSource().empty())
    return op->emitOpError("layout \"quant_params\" needs the weights as a "
                           "`source` operand: the zero-point correction "
                           "folded into the bias is computed from them");

  // `source` is `[bias?, weights]`.
  Value weightsVal = alloc.getSource().back();
  Value biasVal =
      alloc.getSource().size() >= 2 ? alloc.getSource().front() : Value();

  SmallVector<int64_t> w, wShape;
  if (!readConstant(weightsVal, w, wShape))
    return op->emitOpError("weights feeding a \"quant_params\" block are not a "
                           "compile-time constant, so the zero-point fold "
                           "`bias[c] - input_zp * sum_k w[c][k]` cannot be "
                           "computed");
  WeightShape ws;
  if (failed(weightShapeOf(op, wShape, ws)))
    return failure();

  SmallVector<int64_t> bias, biasShape;
  if (biasVal && !readConstant(biasVal, bias, biasShape))
    return op->emitOpError("bias feeding a \"quant_params\" block is not a "
                           "compile-time constant");
  if (!bias.empty() && static_cast<int64_t>(bias.size()) == 1)
    bias.assign(ws.OC, bias[0]); // a splat bias
  if (!bias.empty() && static_cast<int64_t>(bias.size()) != ws.OC)
    return op->emitOpError("bias has ")
           << bias.size() << " elements but the weights have " << ws.OC
           << " output channels";

  const int64_t inputZp = alloc.getInputZp().value_or(0);

  // sum_k w[c][k], per batch. The MXU computes the raw `sum a*w`, so the
  // `-a_zp * sum w` correction has to live in the bias -- DIALECT_L2.md §4.3.
  SmallVector<int64_t> sumW(ws.batch * ws.OC, 0);
  for (int64_t b = 0; b < ws.batch; ++b)
    for (int64_t oc = 0; oc < ws.OC; ++oc) {
      int64_t s = 0;
      for (int64_t kh = 0; kh < ws.KH; ++kh)
        for (int64_t kw = 0; kw < ws.KW; ++kw)
          for (int64_t ic = 0; ic < ws.IC; ++ic)
            s += w[ws.index(b, oc, kh, kw, ic)];
      sumW[b * ws.OC + oc] = s;
    }
  if (ws.batch > 1 && inputZp != 0)
    for (int64_t b = 1; b < ws.batch; ++b)
      for (int64_t oc = 0; oc < ws.OC; ++oc)
        if (sumW[b * ws.OC + oc] != sumW[oc])
          return op->emitOpError()
                 << "a batched matmul with input_zp = " << inputZp
                 << " needs a per-batch zero-point correction, but one "
                    "KeaQuantParam block serves every batch and batch "
                 << b << " channel " << oc << " has sum_k w = "
                 << sumW[b * ws.OC + oc] << " against batch 0's " << sumW[oc];

  ArrayRef<int32_t> mult = quant->getMultiplier().asArrayRef();
  ArrayRef<int8_t> shift = quant->getShift().asArrayRef();
  const bool perChannel = quant->isPerChannel();
  if (perChannel && static_cast<int64_t>(mult.size()) != ws.OC)
    return op->emitOpError("per-channel requantization has ")
           << mult.size() << " multipliers but the weights have " << ws.OC
           << " output channels";

  const int64_t records = alloc.getExtent() / kQuantParamBytes;
  if (alloc.getExtent() % kQuantParamBytes != 0)
    return op->emitOpError("a \"quant_params\" block is a whole number of "
                           "12-byte KeaQuantParam records; ")
           << alloc.getExtent() << " is not";
  if (records < ws.OC)
    return op->emitOpError("a \"quant_params\" block of ")
           << records << " records cannot hold " << ws.OC << " channels";

  out.assign(records * kQuantParamBytes, 0);
  for (int64_t c = 0; c < ws.OC; ++c) {
    const int64_t i = perChannel ? c : 0;
    const int64_t b = bias.empty() ? 0 : bias[c];
    const int64_t folded = b - inputZp * sumW[c];
    if (folded < INT32_MIN || folded > INT32_MAX)
      return op->emitOpError("channel ")
             << c << "'s zero-point-folded bias " << folded
             << " does not fit in the int32 KeaQuantParam.bias";
    // ADR-0003: `kea_shift = tosa_shift - 31`, because keaSrdhm absorbs a >>31
    // that TOSA's formulation carries explicitly.
    putI32(out, c * kQuantParamBytes + 0, static_cast<int32_t>(folded));
    putI32(out, c * kQuantParamBytes + 4, mult[i]);
    putI32(out, c * kQuantParamBytes + 8, static_cast<int32_t>(shift[i]) - 31);
  }
  // Records [OC, records) are the padding channels VQUANT reads because
  // `channels` must be a multiple of 16. Their accumulators come from zero
  // weight lanes, and a zero multiplier keeps them at the output zero point.
  // They are never stored for a real output. DIALECT_L2.md §4.3: write zeros.
  return success();
}

//===----------------------------------------------------------------------===//
// add_params -- docs/DIALECT_L2.md §4.4
//===----------------------------------------------------------------------===//

LogicalResult layoutAddParams(AllocOp alloc, std::vector<uint8_t> &out) {
  Operation *op = alloc.getOperation();
  auto ap = alloc.getAddParam();
  if (!ap)
    return op->emitOpError("layout \"add_params\" needs the derived "
                           "`add_param` array (docs/DIALECT_L2.md §4.4)");
  if (ap->size() != 9)
    return op->emitOpError("`add_param` is [a_mult, b_mult, o_mult, a_shift, "
                           "b_shift, o_shift, a_zp, b_zp, o_zp]; got ")
           << ap->size() << " elements";

  ArrayRef<int64_t> v = *ap;
  for (int i = 0; i < 3; ++i)
    if (v[i] < INT32_MIN || v[i] > INT32_MAX)
      return op->emitOpError("add_param multiplier ") << v[i] << " is not int32";
  for (int i = 3; i < 9; ++i)
    if (v[i] < -128 || v[i] > 127)
      return op->emitOpError("add_param shift/zero-point ")
             << v[i] << " is not int8";

  out.assign(kAddParamBytes, 0);
  putI32(out, 0, static_cast<int32_t>(v[0]));  // a_mult
  putI32(out, 4, static_cast<int32_t>(v[1]));  // b_mult
  putI32(out, 8, static_cast<int32_t>(v[2]));  // o_mult
  out[12] = static_cast<uint8_t>(static_cast<int8_t>(v[3]));  // a_shift
  out[13] = static_cast<uint8_t>(static_cast<int8_t>(v[4]));  // b_shift
  out[14] = static_cast<uint8_t>(static_cast<int8_t>(v[5]));  // o_shift
  out[15] = 0;                                                // reserved0
  out[16] = static_cast<uint8_t>(static_cast<int8_t>(v[6]));  // a_zp
  out[17] = static_cast<uint8_t>(static_cast<int8_t>(v[7]));  // b_zp
  out[18] = static_cast<uint8_t>(static_cast<int8_t>(v[8]));  // o_zp
  out[19] = 0;                                                // reserved1
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//

LogicalResult mlir::kea::materializeConstant(Operation *allocOp,
                                             std::vector<uint8_t> &out) {
  auto alloc = llvm::dyn_cast<AllocOp>(allocOp);
  if (!alloc)
    return allocOp->emitError("expected a kea.alloc");

  StringRef layout = alloc.getLayout().value_or("");
  if (layout.empty())
    return allocOp->emitOpError("role \"")
           << alloc.getRole()
           << "\" is a DRAM constant, so it needs a `layout` naming the byte "
              "layout to materialize (docs/DIALECT_L2.md §4.5)";

  if (layout == "quant_params") {
    if (failed(layoutQuantParams(alloc, out)))
      return failure();
  } else if (layout == "add_params") {
    if (failed(layoutAddParams(alloc, out)))
      return failure();
  } else {
    if (alloc.getSource().empty())
      return allocOp->emitOpError("layout \"")
             << layout << "\" needs the constant tensor as a `source` operand";
    SmallVector<int64_t> w, shape;
    if (!readConstant(alloc.getSource().back(), w, shape))
      return allocOp->emitOpError("the tensor feeding layout \"")
             << layout
             << "\" is not a compile-time constant, so there is nothing to "
                "write into the weight blob";
    WeightShape ws;
    if (layout == "nhwc") {
      // A dense activation constant: the bytes as they already are.
      out.assign(w.size(), 0);
      for (size_t i = 0; i < w.size(); ++i)
        out[i] = static_cast<uint8_t>(static_cast<int8_t>(w[i]));
    } else if (layout == "mxu_tiles_16x16" || layout == "mxu_tiles_16x16_kn") {
      if (failed(weightShapeOf(allocOp, shape, ws)) ||
          failed(layoutMxuTiles(allocOp, w, ws, /*packed=*/false, out)))
        return failure();
      if ((layout == "mxu_tiles_16x16_kn") != ws.ocLast)
        return allocOp->emitOpError("layout \"")
               << layout << "\" does not match a rank-" << shape.size()
               << " weight tensor: only the rank-3 [B, K, N] batched-matmul rhs "
                  "has its output channel last";
    } else if (layout == "mxu_tiles_16x16_packed") {
      if (failed(weightShapeOf(allocOp, shape, ws)) ||
          failed(layoutMxuTiles(allocOp, w, ws, /*packed=*/true, out)))
        return failure();
    } else if (layout == "dwu_planes") {
      if (failed(weightShapeOf(allocOp, shape, ws)) ||
          failed(layoutDwuPlanes(allocOp, w, ws, out)))
        return failure();
    } else {
      return allocOp->emitOpError("unknown layout \"")
             << layout
             << "\"; docs/DIALECT_L2.md §4.5 defines mxu_tiles_16x16, "
                "mxu_tiles_16x16_packed, mxu_tiles_16x16_kn, dwu_planes, "
                "quant_params, add_params and nhwc";
    }
  }

  if (static_cast<int64_t>(out.size()) != alloc.getExtent())
    return allocOp->emitOpError("layout \"")
           << layout << "\" materializes " << out.size()
           << " bytes but the buffer declares " << alloc.getExtent()
           << "; -kea-tile and -kea-emit disagree about the constant's size";
  return success();
}
