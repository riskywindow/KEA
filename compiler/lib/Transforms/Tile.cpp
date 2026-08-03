//===- Tile.cpp - -kea-tile: KEA Level 1 -> Level 2 -------------*- C++ -*-===//
//
// THE PASS THAT CROSSES THE LEVEL BOUNDARY (ADR-0002).
//
// In:  fused Level 1 ops on tensors (`kea.conv2d`, `kea.dwconv2d`, ...).
// Out: a straight-line Level 2 program of KEA-1 instructions on symbolic
//      buffers -- no addresses (that is `-kea-alloc`), no queues and no
//      semaphores (that is `-kea-schedule`).
//
// Read docs/DIALECT_L2.md alongside this file:
//   §4  the symbolic-buffer / live-range interface this pass produces
//   §5  the tiling cost model and how tile sizes get chosen
//   §6  the ISA.md §8.5 convolution lowering, with a worked example
//   §7  how errata E5/E6/E7 and ADR-0003 are discharged
//
// EVERY machine constant comes from include/kea/hw_config.h and every timing
// formula from its `constexpr` occupancy functions. Nothing is hardcoded --
// that is the whole point of those headers being the frozen contract.
//
//===----------------------------------------------------------------------===//

#include "kea/Transforms/Passes.h"

#include "kea/Dialect/KeaAttrs.h"
#include "kea/Dialect/KeaMachineOps.h"
#include "kea/Dialect/KeaOps.h"
#include "kea/Dialect/KeaTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "kea/hw_config.h"
#include "kea/isa.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace kea {
#define GEN_PASS_DEF_KEATILE
#include "kea/Transforms/Passes.h.inc"
} // namespace kea
} // namespace mlir

using namespace mlir;
using namespace mlir::kea;

//===----------------------------------------------------------------------===//
// Small numeric helpers
//===----------------------------------------------------------------------===//

namespace {

int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
int64_t roundUp(int64_t a, int64_t b) { return ceilDiv(a, b) * b; }

/// Divisors of `n`, ascending, always including 1 and `n`.
SmallVector<int64_t> divisorsOf(int64_t n) {
  SmallVector<int64_t> out;
  for (int64_t d = 1; d <= n; ++d)
    if (n % d == 0)
      out.push_back(d);
  return out;
}

/// `2^e` as an int64, saturating well below the int64 range.
int64_t pow2(int64_t e) {
  if (e <= 0)
    return 1;
  if (e >= 62)
    return int64_t(1) << 62;
  return int64_t(1) << e;
}

//===----------------------------------------------------------------------===//
// Machine constants, all from hw_config.h
//===----------------------------------------------------------------------===//

constexpr int64_t kMxuK = ::kea::KEA_MXU_K;                   // 16
constexpr int64_t kMxuN = ::kea::KEA_MXU_N;                   // 16
constexpr int64_t kMxuTileBytes = kMxuK * kMxuN;              // 256
constexpr int64_t kVpuLanes = ::kea::KEA_VPU_LANES;           // 16
constexpr int64_t kDwuLanes = ::kea::KEA_DWU_LANES;           // 16
constexpr int64_t kAccWords = ::kea::KEA_ACC_WORDS;           // 32768
constexpr int64_t kSpmABytes = ::kea::KEA_SPM_A_BYTES;        // 262144
constexpr int64_t kSpmWBytes = ::kea::KEA_SPM_W_BYTES;        // 262144
constexpr int64_t kMaxRows = ::kea::KEA_MXU_MAX_ROWS;         // 2048
constexpr int64_t kDmaMaxLen0 = ::kea::KEA_DMA_MAX_LEN0;      // 65535
constexpr int64_t kDmaMaxN1 = ::kea::KEA_DMA_MAX_N1;          // 65535
constexpr int64_t kDmaMaxN2 = ::kea::KEA_DMA_MAX_N2;          // 255
constexpr int64_t kQuantParamBytes = sizeof(::kea::KeaQuantParam);  // 12
constexpr int64_t kAddParamBytes = sizeof(::kea::KeaAddParam);      // 20

/// `kea.mm` always reads 16 activation bytes and writes 16 ACC words per row
/// regardless of the resident tile's `k_rows`/`n_cols` (ISA.md §7.3), so every
/// activation tile is over-allocated by one array row.
constexpr int64_t kActTailPad = kMxuK;

/// ADR-0003's shift convention: `kea_shift = tosa_shift - 31`.
constexpr int64_t kQ31 = 31;

/// Errata E5: int32 ACC wraps. For int8 x int8 the worst case per accumulated
/// tap is 127*127, so a reduction chain of K taps is safe iff
/// `K * 127 * 127 < 2^31`.
constexpr int64_t kInt8ProductMax = 127 * 127;
constexpr int64_t kInt31 = int64_t(1) << 31;

//===----------------------------------------------------------------------===//
// Level 1 shape / attribute readers
//===----------------------------------------------------------------------===//

ArrayRef<int64_t> shapeOf(Value v) {
  return llvm::cast<RankedTensorType>(v.getType()).getShape();
}

int64_t numElems(Value v) {
  return llvm::cast<RankedTensorType>(v.getType()).getNumElements();
}

int64_t elemBytes(Value v) {
  auto t = llvm::cast<RankedTensorType>(v.getType());
  return t.getElementTypeBitWidth() / 8;
}

/// The `#kea.epilogue` of a Level 1 contraction, unpacked.
struct Epilogue {
  QuantAttr requant;   // null => raw i32 result, which Level 2 cannot store
  int64_t clampLo = -128;
  int64_t clampHi = 127;
  QuantAttr accum, residual, output; // the KeaAddParam triple
  Value residualTensor;              // the Level 1 `residual` operand
  bool hasResidual() const { return accum && residual && output; }
};

Epilogue readEpilogue(std::optional<EpilogueAttr> eo, Value residual) {
  Epilogue out;
  if (!eo)
    return out;
  EpilogueAttr e = *eo;
  out.requant = e.getRequant();
  if (auto c = e.getClamp()) {
    if (c.size() == 2) {
      out.clampLo = c[0];
      out.clampHi = c[1];
    }
  }
  out.accum = e.getAccum();
  out.residual = e.getResidual();
  out.output = e.getOutput();
  out.residualTensor = residual;
  return out;
}

/// Per-output-channel `sum_k |w[oc][k]|`. This is what makes ADR-0003's bound
/// tight enough to be usable: the worst-case `K * 127 * 127` bound is true but
/// so loose that it rejects real MobileNetV2 rescales with `tosa_shift = 22`,
/// while the actual trained weights are nowhere near saturated. Returns
/// `nullopt` when the weights are not a compile-time constant, in which case
/// the caller must fall back to the worst case.
std::optional<SmallVector<int64_t>>
sumAbsWeightsPerOutChannel(Value weights, int64_t outChannels,
                           bool outChannelIsLastDim) {
  DenseElementsAttr cst;
  if (!matchPattern(weights, m_Constant(&cst)))
    return std::nullopt;
  if (!cst.getElementType().isInteger())
    return std::nullopt;

  SmallVector<int64_t> sums(outChannels, 0);
  int64_t n = cst.getNumElements();
  if (outChannels <= 0 || n % outChannels != 0)
    return std::nullopt;
  int64_t perChannel = n / outChannels;

  int64_t i = 0;
  for (const APInt &v : cst.getValues<APInt>()) {
    int64_t idx = outChannelIsLastDim ? (i % outChannels) : (i / perChannel);
    sums[idx] += std::abs(v.getSExtValue());
    ++i;
  }
  return sums;
}

/// Per-output-channel `|bias[oc]|`, or nullopt if the bias is not constant.
/// A missing bias operand is an all-zero bias, which is constant.
std::optional<SmallVector<int64_t>> absBiasPerChannel(Value bias,
                                                      int64_t outChannels) {
  if (!bias)
    return SmallVector<int64_t>(outChannels, 0);
  DenseElementsAttr cst;
  if (!matchPattern(bias, m_Constant(&cst)))
    return std::nullopt;
  SmallVector<int64_t> out;
  for (const APInt &v : cst.getValues<APInt>())
    out.push_back(std::abs(v.getSExtValue()));
  if (out.size() == 1)
    out.assign(outChannels, out[0]);
  if (static_cast<int64_t>(out.size()) != outChannels)
    return std::nullopt;
  return out;
}

//===----------------------------------------------------------------------===//
// Tiling cost model (docs/DIALECT_L2.md §5)
//===----------------------------------------------------------------------===//

/// The shape of one convolution-like layer, after normalisation. `kea.conv2d`,
/// `kea.fully_connected` and `kea.matmul` all reduce to this: an FC layer is
/// `KH = KW = 1, OH = 1, OW = batch`, a batched matmul is the same repeated per
/// batch. That is not a trick -- ISA.md §8.2 says a 1x1 convolution is the
/// degenerate one-tap case of the same addressing identity.
struct ConvShape {
  int64_t IH = 0, IW = 0, IC = 0;
  int64_t OH = 0, OW = 0, OC = 0;
  int64_t KH = 1, KW = 1;
  int64_t strideH = 1, strideW = 1;
  int64_t dilH = 1, dilW = 1;
  int64_t padTop = 0, padLeft = 0;
  int64_t batch = 1;
  int64_t inputZp = 0;
  /// ISA.md §8.6: when `KW * IC <= 16` and dilation is 1, a whole kernel *row*
  /// is one reduction tile, so there are `KH` taps instead of `KH * KW` and
  /// `k_rows = KW * IC` instead of `IC`.
  bool channelPacked = false;

  int64_t icTiles() const { return channelPacked ? 1 : ceilDiv(IC, kMxuK); }
  int64_t ocGroups() const { return ceilDiv(OC, kMxuN); }
  int64_t tapsPerGroup() const { return channelPacked ? KH : (KH * KW); }
  int64_t kRows() const { return channelPacked ? (KW * IC) : std::min(IC, kMxuK); }
  /// Reduction chain length accumulated into one ACC word, padded up to the
  /// array's K -- this is the number errata E5 bounds.
  int64_t reductionTaps() const {
    return channelPacked ? (KH * KW * IC) : (icTiles() * kMxuK * KH * KW);
  }
};

/// One candidate tiling, plus its costed footprint. Everything here is in the
/// units the allocator and the ISA use: bytes for SPM, int32 words for ACC.
struct ConvTiling {
  int64_t ohT = 0, owT = 0;   // output tile extent
  int64_t ocgT = 0;           // output-channel groups resident in ACC at once
  int64_t ihP = 0, iwP = 0;   // padded input tile extent
  int64_t inBytes = 0, outBytes = 0, wBytes = 0, qpBytes = 0, accWords = 0;
  uint64_t cycles = 0;
};

/// Occupancy of one spatial tile on each of the three concurrent resources,
/// straight out of hw_config.h. `-kea-schedule` overlaps them, so the tile's
/// cost is the max, not the sum; the pass that does the overlapping is not this
/// one, but the tile size must be chosen as though it will.
struct TileCost {
  uint64_t mxu = 0, vpu = 0, dma = 0;
  uint64_t bound() const { return std::max(mxu, std::max(vpu, dma)); }
};

TileCost costConvTile(const ConvShape &s, const ConvTiling &t) {
  TileCost c;

  // MXU: one LOAD_W + MATMUL pair per tap per reduction tile per oc group.
  // Consecutive pairs alternate weight banks, so every LOAD_W but the first
  // hides under the previous MATMUL (ISA.md §8.3) -- the model charges one.
  uint64_t mm = ::kea::keaMatmulOccupancy(static_cast<uint32_t>(t.owT),
                                          static_cast<uint32_t>(t.ohT),
                                          /*int4=*/false);
  uint64_t pairs = static_cast<uint64_t>(t.ocgT) * s.icTiles() * s.tapsPerGroup();
  c.mxu = pairs * mm +
          ::kea::keaLoadWOccupancy(static_cast<uint32_t>(s.kRows()), false);

  // VPU: one VQUANT per output-channel group, plus the halo fill when the tile
  // straddles the image border.
  c.vpu = static_cast<uint64_t>(t.ocgT) *
          ::kea::keaVquantOccupancy(static_cast<uint32_t>(t.ohT * t.owT),
                                    static_cast<uint32_t>(kMxuN));
  if (s.padTop || s.padLeft || t.ihP > s.IH || t.iwP > s.IW)
    c.vpu += ::kea::keaVcopyOccupancy(
        static_cast<uint32_t>(t.iwP * s.IC), static_cast<uint32_t>(t.ihP));

  // DMA: the activation tile in, the output tile out. Both are one descriptor.
  c.dma = ::kea::keaDmaOccupancy(static_cast<uint32_t>(t.iwP * s.IC),
                                 static_cast<uint32_t>(t.ihP), 1) +
          ::kea::keaDmaOccupancy(static_cast<uint32_t>(t.ocgT * kMxuN),
                                 static_cast<uint32_t>(t.owT),
                                 static_cast<uint32_t>(t.ohT));
  return c;
}

/// Enumerate and score. Divisors only, so no tile is ragged: a ragged tail tile
/// would need its own instruction sequence and buys nothing on the shapes
/// MobileNetV2 actually has (every feature map is a power of two times a small
/// odd number).
std::optional<ConvTiling> chooseConvTiling(const ConvShape &s,
                                           int64_t spmReserve) {
  const int64_t spmABudget = kSpmABytes / spmReserve;
  const int64_t spmWBudget = kSpmWBytes / spmReserve;

  std::optional<ConvTiling> best;
  for (int64_t ohT : divisorsOf(s.OH)) {
    if (ohT > kDmaMaxN2) // output DMA uses n2 = ohT
      break;
    for (int64_t owT : divisorsOf(s.OW)) {
      if (ohT * owT > kMaxRows) // MATMUL m_inner * m_outer bound
        continue;
      if (owT > kDmaMaxN1)
        continue;

      const int64_t ihP = (ohT - 1) * s.strideH + (s.KH - 1) * s.dilH + 1;
      const int64_t iwP = (owT - 1) * s.strideW + (s.KW - 1) * s.dilW + 1;
      if (ihP > kDmaMaxN1 || iwP * s.IC > kDmaMaxLen0)
        continue;
      const int64_t inBytes = ihP * iwP * s.IC + kActTailPad;
      if (inBytes >= spmABudget)
        continue;

      for (int64_t ocgT = s.ocGroups(); ocgT >= 1; --ocgT) {
        const int64_t accWords = ocgT * ohT * owT * kMxuN;
        if (accWords > kAccWords)
          continue;
        const int64_t outBytes = ohT * owT * ocgT * kMxuN + kActTailPad;
        if (inBytes + outBytes > spmABudget)
          continue;
        const int64_t wPerGroup =
            s.icTiles() * s.tapsPerGroup() * kMxuTileBytes;
        const int64_t wBytes = ocgT * wPerGroup;
        const int64_t qpBytes = ocgT * kMxuN * kQuantParamBytes;
        if (wBytes + qpBytes > spmWBudget)
          continue;
        // The weight DMA is one descriptor: either `ocgT` runs of `wPerGroup`
        // bytes, or -- when one group is itself too long for `len0` -- a 3D
        // walk of `tapsPerGroup` tiles x `icTiles` x `ocgT`, whose outer count
        // is the instruction's `aux` byte and so caps at 255.
        if (wPerGroup > kDmaMaxLen0 && ocgT > kDmaMaxN2)
          continue;
        if (qpBytes > kDmaMaxLen0)
          continue;
        // Output store: len0 is this tile's real channel byte count.
        if (ocgT * kMxuN > kDmaMaxLen0)
          continue;

        ConvTiling t;
        t.ohT = ohT;
        t.owT = owT;
        t.ocgT = ocgT;
        t.ihP = ihP;
        t.iwP = iwP;
        t.inBytes = inBytes;
        t.outBytes = outBytes;
        t.wBytes = wBytes;
        t.qpBytes = qpBytes;
        t.accWords = accWords;

        const int64_t nSpatial = (s.OH / ohT) * (s.OW / owT);
        const int64_t nOcTiles = ceilDiv(s.ocGroups(), ocgT);
        const uint64_t weightDma =
            ::kea::keaDmaOccupancy(static_cast<uint32_t>(wBytes), 1, 1) +
            ::kea::keaDmaOccupancy(static_cast<uint32_t>(qpBytes), 1, 1);
        t.cycles = static_cast<uint64_t>(nOcTiles) *
                   (weightDma + static_cast<uint64_t>(nSpatial) * s.batch *
                                    costConvTile(s, t).bound());

        if (!best || t.cycles < best->cycles ||
            (t.cycles == best->cycles &&
             ohT * owT * ocgT > best->ohT * best->owT * best->ocgT))
          best = t;
      }
    }
  }
  return best;
}

//===----------------------------------------------------------------------===//
// KeaAddParam derivation (errata E6)
//===----------------------------------------------------------------------===//

/// The machine-form `KeaAddParam`, in isa.h field order.
struct AddParam {
  int64_t aMult = 0, bMult = 0, oMult = 0;
  int64_t aShift = 0, bShift = 0, oShift = 0;
  int64_t aZp = 0, bZp = 0, oZp = 0;

  SmallVector<int64_t> asArray() const {
    return {aMult, bMult, oMult, aShift, bShift, oShift, aZp, bZp, oZp};
  }
};

/// Convert a Level 1 (TOSA-form) `#kea.quant` triple into `KeaAddParam`.
///
/// The two formulations differ by two fixed amounts:
///   * ADR-0003's `kea_shift = tosa_shift - 31`, because `keaSrdhm` absorbs a
///     `>>31` that TOSA carries explicitly;
///   * `KEA_VADD_LEFT_SHIFT` (20), which `keaQuantizedAdd` applies to both
///     *inputs* and TOSA has no counterpart for.
/// So the exact input shifts are `tosa_shift - 31 + 20 = tosa_shift - 11`.
///
/// `keaRdpot` ignores a negative exponent (it is `return x` for `exp <= 0`), so
/// a negative input shift would be silently dropped -- and it cannot be folded
/// into the multiplier either, since a normalized multiplier is already in
/// `[2^30, 2^31)` and doubling it overflows int32. The fix is to shift *both*
/// inputs down by the same `d`, which is exact, and to take the same `d` back
/// out of the output shift.
std::optional<AddParam> deriveAddParam(QuantAttr a, QuantAttr b, QuantAttr o,
                                       StringRef &why) {
  auto onlyScale = [&](QuantAttr q, int64_t &mult, int64_t &shift) {
    if (q.getMultiplier().size() != 1 || q.getShift().size() != 1) {
      why = "KEA_VADD parameters are per tensor, not per channel";
      return false;
    }
    mult = q.getMultiplier()[0];
    shift = q.getShift()[0];
    return true;
  };

  AddParam p;
  int64_t aT = 0, bT = 0, oT = 0;
  if (!onlyScale(a, p.aMult, aT) || !onlyScale(b, p.bMult, bT) ||
      !onlyScale(o, p.oMult, oT))
    return std::nullopt;

  const int64_t kIn = kQ31 - ::kea::KEA_VADD_LEFT_SHIFT; // 11
  const int64_t d =
      std::max<int64_t>(0, std::max(kIn - aT, kIn - bT));
  p.aShift = aT - kIn + d;
  p.bShift = bT - kIn + d;
  p.oShift = oT - kQ31 - d;
  if (p.oShift < 0) {
    why = "the output rescale would need a left shift KEA_VADD cannot express "
          "(keaRdpot ignores a negative exponent)";
    return std::nullopt;
  }

  p.aZp = a.getInputZp();
  p.bZp = b.getInputZp();
  p.oZp = o.getOutputZp();
  return p;
}

} // namespace

//===----------------------------------------------------------------------===//
// The tiler
//===----------------------------------------------------------------------===//

namespace {

class Tiler {
public:
  Tiler(func::FuncOp func, int64_t spmReserve)
      : func(func), b(func.getContext()), spmReserve(spmReserve) {}

  LogicalResult run();

  int64_t numLayers = 0, numTiles = 0, numMatmuls = 0;
  SmallVector<Attribute> tileReport;

private:
  func::FuncOp func;
  OpBuilder b;
  int64_t spmReserve;
  int64_t layerId = 0;
  DenseMap<Value, Value> dramFor; // Level 1 tensor -> DRAM !kea.buffer

  /// Count of LOAD_W/MATMUL pairs emitted so far, monotonic over the whole
  /// function. `bank = mxuPairs & 1` is ISA.md §8.3's `t & 1`, but with `t`
  /// counted over the entire MXU stream rather than restarted per
  /// output-channel group as §8.3's pseudocode does.
  ///
  /// The literal reading costs a real overlap. LOAD_W occupies the bank it
  /// targets, and the MXU tracks the two banks as separate resources precisely
  /// so a load into the idle one runs under a MATMUL on the other (ISA.md
  /// §5.3, "what does NOT need synchronization"). Restarting `t` per group
  /// makes the last pair of group `g` and the first pair of group `g+1` share
  /// a bank, so the second LOAD_W stalls until the first MATMUL releases it --
  /// and for a 1x1 convolution with a single reduction tile, where each group
  /// is exactly ONE pair, every single load serialises against the matmul
  /// before it. That is the common case in MobileNetV2: every pointwise layer.
  ///
  /// A monotonic counter is still `t & 1` and still gives every ACC region the
  /// same instructions; it only changes which physical bank each pair uses. It
  /// strictly dominates, it is what the cost model in chooseConvTiling()
  /// already assumes (one LOAD_W charged per tile, not one per tap), and it
  /// keeps errata E7 trivially satisfied because every MATMUL still reads the
  /// bank the LOAD_W immediately before it wrote.
  int64_t mxuPairs = 0;

  //--- naming -------------------------------------------------------------
  std::string layerName(StringRef suffix) {
    return (func.getName() + "." + Twine(layerId) + "." + suffix).str();
  }

  //--- buffer construction ------------------------------------------------
  Value makeBuffer(Location loc, int64_t extent, AddressSpace as,
                   StringRef name, StringRef role, ValueRange source = {}) {
    Type elem = (as == AddressSpace::ACC) ? b.getIntegerType(32)
                                          : b.getIntegerType(8);
    auto ty = BufferType::get(b.getContext(), {extent}, elem, as);
    auto op = b.create<AllocOp>(loc, ty, source);
    op->setAttr("name", b.getStringAttr(name));
    op->setAttr("role", b.getStringAttr(role));
    return op.getResult();
  }

  Value scratch(Location loc, int64_t extent, AddressSpace as, StringRef name) {
    return makeBuffer(loc, extent, as, name, "scratch");
  }

  /// The DRAM buffer backing a Level 1 tensor value, created on demand.
  ///
  /// A block argument that is actually read as an activation is a model
  /// *input*; the caller's `role` applies to everything else. Creating these
  /// lazily rather than up front matters: a block argument that turns out to be
  /// a weight tensor is reached through `makeBuffer(role = "weights")` instead
  /// and must not also acquire a dead `input` buffer.
  Value dram(Value tensor, StringRef role, StringRef name) {
    auto it = dramFor.find(tensor);
    if (it != dramFor.end())
      return it->second;
    std::string argName;
    if (auto arg = dyn_cast<BlockArgument>(tensor)) {
      role = "input";
      argName = (func.getName() + ".input" + Twine(arg.getArgNumber())).str();
      name = argName; // argName outlives every use of `name` below
    }
    OpBuilder::InsertionGuard g(b);
    // Anchor DRAM buffers at the top of the block so they dominate every use
    // and so the resulting Level 2 function reads as "here is the memory map,
    // here is the program".
    b.setInsertionPointToStart(&func.front());
    if (auto *def = tensor.getDefiningOp())
      b.setInsertionPointAfter(def);
    int64_t bytes = numElems(tensor) * elemBytes(tensor);
    Value v = makeBuffer(tensor.getLoc(), bytes, AddressSpace::DRAM, name, role);
    dramFor[tensor] = v;
    return v;
  }

  //--- lowering entry points ----------------------------------------------
  LogicalResult lower(Operation *op);
  LogicalResult lowerContraction(Operation *op, ConvShape s, Value inTensor,
                                 Value wTensor, Value biasTensor,
                                 Epilogue epi, Value resultTensor,
                                 StringRef weightLayout);
  LogicalResult lowerDwconv(DWConv2DOp op);
  LogicalResult lowerPool(PoolOp op);
  LogicalResult lowerAdd(AddOp op);

  //--- constraint discharge ------------------------------------------------
  LogicalResult checkAccumulatorBound(Operation *op, int64_t reductionTaps,
                                      Value weights, Value bias,
                                      bool outChannelIsLastDim, int64_t outCh,
                                      int64_t inputZp, QuantAttr requant);
  LogicalResult buildAddParam(Operation *op, QuantAttr a, QuantAttr bq,
                              QuantAttr o, AddParam &out);

  //--- shared emission helpers ---------------------------------------------
  Value emitQParamBlock(Location loc, Value biasTensor, Value wTensor,
                        QuantAttr requant, int64_t inputZp, int64_t channels,
                        StringRef name);
  void emitFill(Location loc, Value tile, int64_t bytes, int64_t zp);
};

//===----------------------------------------------------------------------===//
// Constraint discharge
//===----------------------------------------------------------------------===//

LogicalResult Tiler::checkAccumulatorBound(Operation *op, int64_t taps,
                                           Value weights, Value bias,
                                           bool ocLast, int64_t outCh,
                                           int64_t inputZp, QuantAttr requant) {
  //--- Errata E5: int32 ACC wraps, it does not saturate --------------------
  if (taps > 0 && taps > (kInt31 - 1) / kInt8ProductMax)
    return op->emitOpError()
           << "reduction chain of " << taps
           << " taps can overflow the int32 accumulator (errata E5: ACC wraps, "
              "it does not saturate; the bound is K * 127 * 127 < 2^31, i.e. K "
              "< "
           << ((kInt31 - 1) / kInt8ProductMax) << ")";

  if (!requant)
    return success();

  //--- ADR-0003: tosa_shift >= 31, or |acc + bias| < 2^tosa_shift ----------
  ArrayRef<int8_t> shifts = requant.getShift().asArrayRef();
  bool anySmallShift = false;
  for (int8_t s : shifts)
    if (s < kQ31)
      anySmallShift = true;
  if (!anySmallShift)
    return success(); // the blanket rule discharges every channel

  // Only now is the tight bound needed. `acc + bias` is what VQUANT
  // requantizes; with weight_zp = 0 the Level 1 semantics is
  //   sum_k (a - a_zp) * w  +  bias
  // and |a - a_zp| <= 255 for int8 with an int8 zero point.
  SmallVector<int64_t> bound;
  auto sumAbsW = sumAbsWeightsPerOutChannel(weights, outCh, ocLast);
  auto absBias = absBiasPerChannel(bias, outCh);
  const int64_t kMaxCentred = 255;
  if (sumAbsW && absBias) {
    bound.reserve(outCh);
    for (int64_t c = 0; c < outCh; ++c)
      bound.push_back(kMaxCentred * (*sumAbsW)[c] + (*absBias)[c]);
  } else {
    // No constant weights: fall back to the worst case E5 already bounded.
    int64_t worst = taps * kInt8ProductMax;
    if (absBias)
      worst += *std::max_element(absBias->begin(), absBias->end());
    else
      worst = kInt31; // unknown bias: nothing can discharge a small shift
    bound.assign(outCh, worst);
  }
  (void)inputZp;

  for (int64_t c = 0; c < outCh; ++c) {
    int64_t s = shifts.size() == 1 ? shifts[0]
                                   : shifts[std::min<int64_t>(
                                         c, (int64_t)shifts.size() - 1)];
    if (s >= kQ31)
      continue;
    if (bound[c] < pow2(s))
      continue;
    return op->emitOpError()
           << "channel " << c << " requantizes with tosa_shift = " << s
           << " (< 31) but the accumulator range bound is " << bound[c]
           << ", which is not < 2^" << s << " = " << pow2(s)
           << "; ADR-0003's normalised-multiplier invariant cannot be "
              "discharged, so VQUANT and the frontend's TOSA golden model "
              "would disagree. Split the rescale, or requantize with "
              "tosa_shift >= 31.";
  }
  return success();
}

LogicalResult Tiler::buildAddParam(Operation *op, QuantAttr a, QuantAttr bq,
                                   QuantAttr o, AddParam &out) {
  StringRef why;
  auto p = deriveAddParam(a, bq, o, why);
  if (!p)
    return op->emitOpError("cannot express this quantized add as KEA_VADD: ")
           << why;

  //--- Errata E6: |xa| + |xb| must stay inside int32 -----------------------
  int64_t worst = addParamWorstCaseSum(p->aMult, p->aShift, p->bMult, p->bShift);
  if (worst >= kInt31)
    return op->emitOpError()
           << "KeaAddParam would let |xa| + |xb| reach " << worst
           << " >= 2^31; keaQuantizedAdd sums them in plain int32 (errata E6)";
  out = *p;
  return success();
}

//===----------------------------------------------------------------------===//
// Shared emission helpers
//===----------------------------------------------------------------------===//

/// A DRAM `KeaQuantParam[channels]` block. `-kea-emit` materializes it from the
/// attributes and the two `source` operands; see KeaMachineOps.td.
Value Tiler::emitQParamBlock(Location loc, Value biasTensor, Value wTensor,
                             QuantAttr requant, int64_t inputZp,
                             int64_t channels, StringRef name) {
  SmallVector<Value> src;
  if (biasTensor)
    src.push_back(biasTensor);
  src.push_back(wTensor);
  Value v = makeBuffer(loc, channels * kQuantParamBytes, AddressSpace::DRAM,
                       name, "qparam", src);
  auto op = v.getDefiningOp<AllocOp>();
  op->setAttr("layout", b.getStringAttr("quant_params"));
  op->setAttr("quant", requant);
  op->setAttr("input_zp", b.getI64IntegerAttr(inputZp));
  return v;
}

/// Pre-fill a whole SPM_A tile with an int8 value, as one flat `VCOPY` run.
///
/// This is ISA.md §8.4(a)'s zero-point halo, and it is emitted for EVERY
/// activation tile, not only for tiles that straddle the image border. Two
/// reasons, both correctness rather than convenience:
///
///  * `kea.mm` always reads 16 activation bytes per row whatever the resident
///    tile's `k_rows` (ISA.md §7.3), so the last row of every tile reads up to
///    15 bytes past its last real pixel. Those bytes multiply against the zeros
///    `LOAD_W` installed and contribute nothing -- but they are *read*, and
///    ISA.md §2.3 says a simulator should poison unwritten scratchpad and trap.
///    An untraceable "read of never-written scratchpad" report is exactly the
///    compiler bug MICROARCH.md §8.2 calls the most likely one.
///  * `DWCONV` reads `channels` bytes per pixel, and `channels` is rounded up
///    to a multiple of 16, so a 24-channel depthwise reads 8 lanes that no DMA
///    wrote. Their weight planes are zero, so again the arithmetic is right and
///    the read still has to be defined.
///
/// The cost is `KEA_VPU_ISSUE_OVERHEAD + ceil(bytes / 32)` cycles on a unit
/// that is idle at that point anyway, against a DMA of the same tile at 16
/// B/cycle -- i.e. at most half the DMA it precedes, and `-kea-schedule` hides
/// it entirely.
void Tiler::emitFill(Location loc, Value tile, int64_t bytes, int64_t zp) {
  b.create<VcopyOp>(loc, /*src=*/Value(), tile,
                    /*src_addr=*/0, /*dst_addr=*/0,
                    /*row_bytes=*/bytes, /*rows=*/1,
                    /*src_row_stride=*/0, /*dst_row_stride=*/bytes,
                    /*fill_value=*/zp, /*fill=*/true);
}

//===----------------------------------------------------------------------===//
// The convolution lowering -- ISA.md §8.5, normative
//===----------------------------------------------------------------------===//

LogicalResult Tiler::lowerContraction(Operation *op, ConvShape s,
                                      Value inTensor, Value wTensor,
                                      Value biasTensor, Epilogue epi,
                                      Value resultTensor,
                                      StringRef weightLayout) {
  Location loc = op->getLoc();
  if (!epi.requant)
    return op->emitOpError(
        "needs a requantized epilogue to reach Level 2 -- a raw i32 "
        "accumulator has nowhere to live in SPM_A. Run -kea-fuse first.");

  if (failed(checkAccumulatorBound(op, s.reductionTaps(), wTensor, biasTensor,
                                   /*ocLast=*/weightLayout ==
                                       "mxu_tiles_16x16_kn",
                                   s.OC, s.inputZp, epi.requant)))
    return failure();

  AddParam addParam;
  if (epi.hasResidual() &&
      failed(buildAddParam(op, epi.accum, epi.residual, epi.output, addParam)))
    return failure();

  auto tOpt = chooseConvTiling(s, spmReserve);
  if (!tOpt)
    return op->emitOpError()
           << "no output tile fits the scratchpad: even a 1x1 output tile of "
           << s.OC << " channels needs more than SPM_A/" << spmReserve << " ("
           << (kSpmABytes / spmReserve) << " B) or SPM_W/" << spmReserve
           << " or ACC (" << kAccWords << " words)";
  const ConvTiling t = *tOpt;

  // Level 2 buffers for the layer's DRAM-resident operands.
  Value inBuf = dram(inTensor, "activation", layerName("in"));
  Value outBuf = dram(resultTensor, "activation", layerName("out"));
  Value wBuf = makeBuffer(loc, s.batch * s.ocGroups() * s.icTiles() *
                                   s.tapsPerGroup() * kMxuTileBytes,
                          AddressSpace::DRAM, layerName("weights"), "weights",
                          ValueRange{wTensor});
  wBuf.getDefiningOp<AllocOp>()->setAttr("layout",
                                         b.getStringAttr(weightLayout));
  Value qpBuf = emitQParamBlock(loc, biasTensor, wTensor, epi.requant,
                                s.inputZp, roundUp(s.OC, kVpuLanes),
                                layerName("qparams"));

  Value residualBuf, addParamBuf;
  if (epi.hasResidual()) {
    residualBuf = dram(epi.residualTensor, "activation", layerName("residual"));
    addParamBuf = makeBuffer(loc, kAddParamBytes, AddressSpace::DRAM,
                             layerName("addparams"), "addparam");
    auto ap = addParamBuf.getDefiningOp<AllocOp>();
    ap->setAttr("layout", b.getStringAttr("add_params"));
    ap->setAttr("quant", epi.accum);
    ap->setAttr("residual_quant", epi.residual);
    ap->setAttr("output_quant", epi.output);
    ap->setAttr("add_param", b.getDenseI64ArrayAttr(addParam.asArray()));
  }

  b.create<TraceOp>(loc, b.getStringAttr("begin"), layerId, /*payload=*/0,
                    /*unit=*/StringAttr());

  const int64_t ocTiles = ceilDiv(s.ocGroups(), t.ocgT);
  const int64_t tileChanBytes = t.ocgT * kMxuN;  // padded channels in this tile
  const int64_t sp = s.IC;                       // §8.1 pixel stride
  const int64_t sr = t.iwP * sp;                 // §8.1 row stride

  for (int64_t bat = 0; bat < s.batch; ++bat) {
    for (int64_t oct = 0; oct < ocTiles; ++oct) {
      const int64_t ocgBase = oct * t.ocgT;
      const int64_t ocgCount = std::min(t.ocgT, s.ocGroups() - ocgBase);
      const int64_t ocBase = ocgBase * kMxuN;
      const int64_t ocCount = std::min(ocgCount * kMxuN, s.OC - ocBase);

      // Weights and quantization parameters for this output-channel tile, in
      // SPM_W. Hoisted out of the spatial loop: they are loop invariant, and
      // the tiling search already guaranteed they fit.
      const int64_t wPerGroup = s.icTiles() * s.tapsPerGroup() * kMxuTileBytes;
      const int64_t wTileBytes = ocgCount * wPerGroup;
      Value wTile =
          scratch(loc, wTileBytes, AddressSpace::W, layerName("wtile"));
      const int64_t wDramBase = (bat * s.ocGroups() + ocgBase) * wPerGroup;
      if (wPerGroup <= kDmaMaxLen0) {
        // One contiguous run per output-channel group.
        b.create<DmaLoadOp>(loc, wBuf, wTile, /*dram_addr=*/wDramBase,
                            /*spm_addr=*/0, /*len0=*/wPerGroup,
                            /*n1=*/ocgCount, /*n2=*/1, /*dram_s1=*/wPerGroup,
                            /*dram_s2=*/0, /*spm_s1=*/wPerGroup, /*spm_s2=*/0,
                            /*unit=*/StringAttr());
      } else {
        // A single group's weights exceed `len0` (16 bits): split the run at
        // the reduction-tile boundary and use all three descriptor levels.
        const int64_t perIc = s.tapsPerGroup() * kMxuTileBytes;
        b.create<DmaLoadOp>(loc, wBuf, wTile, /*dram_addr=*/wDramBase,
                            /*spm_addr=*/0, /*len0=*/perIc,
                            /*n1=*/s.icTiles(), /*n2=*/ocgCount,
                            /*dram_s1=*/perIc, /*dram_s2=*/wPerGroup,
                            /*spm_s1=*/perIc, /*spm_s2=*/wPerGroup,
                            /*unit=*/StringAttr());
      }

      const int64_t qpTileBytes = ocgCount * kMxuN * kQuantParamBytes;
      Value qpTile = scratch(loc, qpTileBytes, AddressSpace::W,
                             layerName("qptile"));
      b.create<DmaLoadOp>(loc, qpBuf, qpTile,
                          /*dram_addr=*/ocBase * kQuantParamBytes,
                          /*spm_addr=*/0, /*len0=*/qpTileBytes, /*n1=*/1,
                          /*n2=*/1, /*dram_s1=*/qpTileBytes,
                          /*dram_s2=*/qpTileBytes, /*spm_s1=*/qpTileBytes,
                          /*spm_s2=*/qpTileBytes, /*unit=*/StringAttr());

      Value apTile;
      if (epi.hasResidual()) {
        apTile = scratch(loc, kAddParamBytes, AddressSpace::W,
                         layerName("aptile"));
        b.create<DmaLoadOp>(loc, addParamBuf, apTile, 0, 0, kAddParamBytes, 1,
                            1, kAddParamBytes, kAddParamBytes, kAddParamBytes,
                            kAddParamBytes, StringAttr());
      }

      for (int64_t oh0 = 0; oh0 < s.OH; oh0 += t.ohT) {
        for (int64_t ow0 = 0; ow0 < s.OW; ow0 += t.owT) {
          ++numTiles;

          //--- SPM_A activation tile, padded per ISA.md §8.4(a) -------------
          Value aTile = scratch(loc, t.inBytes, AddressSpace::A,
                                layerName("atile"));

          // Window of the *input* this output tile reads, in input coordinates.
          const int64_t ih0 = oh0 * s.strideH - s.padTop;
          const int64_t iw0 = ow0 * s.strideW - s.padLeft;
          const int64_t ihEnd = ih0 + t.ihP;
          const int64_t iwEnd = iw0 + t.iwP;
          emitFill(loc, aTile, t.inBytes, s.inputZp);

          // The in-bounds sub-rectangle, DMA'd into the middle of the tile.
          const int64_t srcH0 = std::max<int64_t>(ih0, 0);
          const int64_t srcW0 = std::max<int64_t>(iw0, 0);
          const int64_t srcH1 = std::min(ihEnd, s.IH);
          const int64_t srcW1 = std::min(iwEnd, s.IW);
          if (srcH1 > srcH0 && srcW1 > srcW0) {
            const int64_t rows = srcH1 - srcH0;
            const int64_t cols = srcW1 - srcW0;
            const int64_t dstOff = (srcH0 - ih0) * sr + (srcW0 - iw0) * sp;
            const int64_t srcOff =
                ((bat * s.IH + srcH0) * s.IW + srcW0) * s.IC;
            b.create<DmaLoadOp>(loc, inBuf, aTile,
                                /*dram_addr=*/srcOff, /*spm_addr=*/dstOff,
                                /*len0=*/cols * s.IC, /*n1=*/rows, /*n2=*/1,
                                /*dram_s1=*/s.IW * s.IC, /*dram_s2=*/0,
                                /*spm_s1=*/sr, /*spm_s2=*/0,
                                /*unit=*/StringAttr());
          }

          Value accTile =
              scratch(loc, ocgCount * t.ohT * t.owT * kMxuN, AddressSpace::ACC,
                      layerName("acc"));
          Value oTile = scratch(loc, t.ohT * t.owT * tileChanBytes + kActTailPad,
                                AddressSpace::A, layerName("otile"));

          //--- ISA.md §8.5: LOAD_W + MATMUL per tap ------------------------
          for (int64_t g = 0; g < ocgCount; ++g) {
            const int64_t qBase = g * t.ohT * t.owT * kMxuN;
            const int64_t nCols =
                std::min<int64_t>(kMxuN, s.OC - (ocBase + g * kMxuN));
            int64_t tap = 0;
            for (int64_t ic0 = 0; ic0 < s.icTiles(); ++ic0) {
              for (int64_t kh = 0; kh < s.KH; ++kh) {
                for (int64_t kw = 0; kw < (s.channelPacked ? 1 : s.KW); ++kw) {
                  // Alternate the weight bank on EVERY consecutive pair, so the
                  // LOAD_W lands in the idle bank while the previous MATMUL is
                  // still streaming the other one. `mxuPairs` is monotonic over
                  // the whole function, NOT reset per output-channel group --
                  // see the note on the member for why that matters.
                  const int64_t bank = mxuPairs++ & 1;
                  const int64_t kRows =
                      s.channelPacked ? s.kRows()
                                      : std::min<int64_t>(kMxuK,
                                                          s.IC - ic0 * kMxuK);
                  // §8.1: tiles are ordered [oc0][ic0][kh][kw], 256 B each.
                  const int64_t wOff =
                      (((g * s.icTiles() + ic0) * s.KH + kh) *
                           (s.channelPacked ? 1 : s.KW) +
                       kw) *
                      kMxuTileBytes;
                  b.create<LoadWOp>(loc, wTile, /*w_addr=*/wOff,
                                    /*w_row_stride=*/kMxuN, /*k_rows=*/kRows,
                                    /*n_cols=*/nCols, /*bank=*/bank,
                                    /*int4=*/false);

                  // §8.2's addressing identity, verbatim.
                  const int64_t aAddr = ic0 * kMxuK + kh * s.dilH * sr +
                                        (s.channelPacked ? 0 : kw * s.dilW * sp);
                  b.create<MmOp>(loc, aTile, accTile,
                                 /*a_addr=*/aAddr,
                                 /*a_inner_stride=*/s.strideW * sp,
                                 /*a_outer_stride=*/s.strideH * sr,
                                 /*m_inner=*/t.owT, /*m_outer=*/t.ohT,
                                 /*acc_addr=*/qBase,
                                 /*acc_inner_stride=*/kMxuN,
                                 /*acc_outer_stride=*/t.owT * kMxuN,
                                 /*bank=*/bank,
                                 // `accumulate` is false on exactly ONE
                                 // instruction per ACC region: the first tap of
                                 // the first reduction tile. That counter is
                                 // per group and has nothing to do with banks.
                                 /*accumulate=*/tap != 0,
                                 /*int4=*/false);
                  ++numMatmuls;
                  ++tap;
                }
              }
            }

            //--- the fused epilogue ---------------------------------------
            b.create<VquantOp>(loc, accTile, oTile, qpTile,
                               /*acc_addr=*/qBase,
                               /*out_addr=*/g * kMxuN,
                               /*qparam_addr=*/g * kMxuN * kQuantParamBytes,
                               /*num_pixels=*/t.ohT * t.owT,
                               /*channels=*/kMxuN,
                               /*acc_pix_stride=*/kMxuN,
                               /*out_pix_stride=*/tileChanBytes,
                               /*out_zp=*/epi.requant.getOutputZp(),
                               /*clamp_lo=*/epi.clampLo,
                               /*clamp_hi=*/epi.clampHi, /*int4=*/false);
          }

          //--- the residual add, if this layer has one ---------------------
          Value storeTile = oTile;
          if (epi.hasResidual()) {
            Value rTile =
                scratch(loc, t.ohT * t.owT * tileChanBytes + kActTailPad,
                        AddressSpace::A, layerName("rtile"));
            // VADD is a flat pass over `ohT*owT*tileChanBytes` lanes, but the
            // DMA below only fills `ocCount` of every `tileChanBytes`. Define
            // the rest so VADD never reads poison; the extra lanes are dropped
            // again by the DMA_ST.
            if (ocCount != tileChanBytes)
              emitFill(loc, rTile, t.ohT * t.owT * tileChanBytes,
                       epi.residual.getInputZp());
            // The residual has the layer's *output* shape, so it is read with
            // the same tiling as the store below.
            b.create<DmaLoadOp>(
                loc, residualBuf, rTile,
                /*dram_addr=*/((bat * s.OH + oh0) * s.OW + ow0) * s.OC + ocBase,
                /*spm_addr=*/0, /*len0=*/ocCount, /*n1=*/t.owT, /*n2=*/t.ohT,
                /*dram_s1=*/s.OC, /*dram_s2=*/s.OW * s.OC,
                /*spm_s1=*/tileChanBytes,
                /*spm_s2=*/t.owT * tileChanBytes, /*unit=*/StringAttr());
            b.create<VaddOp>(loc, oTile, rTile, oTile, apTile,
                             /*a_addr=*/0, /*b_addr=*/0, /*out_addr=*/0,
                             /*param_addr=*/0,
                             /*num_elems=*/t.ohT * t.owT * tileChanBytes,
                             /*clamp_lo=*/-128, /*clamp_hi=*/127);
          }

          b.create<DmaStoreOp>(
              loc, storeTile, outBuf,
              /*dram_addr=*/((bat * s.OH + oh0) * s.OW + ow0) * s.OC + ocBase,
              /*spm_addr=*/0, /*len0=*/ocCount, /*n1=*/t.owT, /*n2=*/t.ohT,
              /*dram_s1=*/s.OC, /*dram_s2=*/s.OW * s.OC,
              /*spm_s1=*/tileChanBytes, /*spm_s2=*/t.owT * tileChanBytes,
              /*unit=*/StringAttr());
        }
      }
    }
  }

  b.create<TraceOp>(loc, b.getStringAttr("end"), layerId, 0, StringAttr());

  tileReport.push_back(b.getDictionaryAttr({
      b.getNamedAttr("layer", b.getI64IntegerAttr(layerId)),
      b.getNamedAttr("op", b.getStringAttr(op->getName().getStringRef())),
      b.getNamedAttr("oh", b.getI64IntegerAttr(t.ohT)),
      b.getNamedAttr("ow", b.getI64IntegerAttr(t.owT)),
      b.getNamedAttr("oc_groups", b.getI64IntegerAttr(t.ocgT)),
      b.getNamedAttr("spm_a", b.getI64IntegerAttr(t.inBytes + t.outBytes)),
      b.getNamedAttr("spm_w", b.getI64IntegerAttr(t.wBytes + t.qpBytes)),
      b.getNamedAttr("acc", b.getI64IntegerAttr(t.accWords)),
      b.getNamedAttr("taps", b.getI64IntegerAttr(s.reductionTaps())),
      b.getNamedAttr("cycles", b.getI64IntegerAttr((int64_t)t.cycles)),
  }));
  return success();
}

//===----------------------------------------------------------------------===//
// Depthwise -- the DWU, ISA.md §9
//===----------------------------------------------------------------------===//

LogicalResult Tiler::lowerDwconv(DWConv2DOp op) {
  Location loc = op.getLoc();
  auto in = shapeOf(op.getInput());   // [N, IH, IW, C]
  auto w = shapeOf(op.getWeights());  // [OC, KH, KW, 1]
  auto out = shapeOf(op.getOutput()); // [N, OH, OW, OC]
  if (in.size() != 4 || out.size() != 4)
    return op.emitOpError("expects rank-4 NHWC operands");
  if (in[0] != 1)
    return op.emitOpError("only batch 1 is lowered; batch ")
           << in[0] << " would need an outer loop the ISA has no room for";

  const int64_t C = in[3], OC = out[3];
  if (OC != C)
    return op.emitOpError("channel multiplier != 1 is not lowered: DWCONV has "
                          "one lane per channel and no output fan-out");
  const int64_t KH = w[1], KW = w[2];
  if (KH != KW || (KH != ::kea::KEA_DWU_KERNEL_3 && KH != ::kea::KEA_DWU_KERNEL_5))
    return op.emitOpError("DWCONV supports square 3x3 or 5x5 kernels only, got ")
           << KH << "x" << KW;
  auto strides = op.getStrides();
  auto dil = op.getDilations();
  auto pads = op.getPads();
  if (strides[0] != strides[1] || strides[0] > ::kea::KEA_DWU_MAX_STRIDE)
    return op.emitOpError("DWCONV supports square stride 1 or 2 only");
  if (dil[0] != 1 || dil[1] != 1)
    return op.emitOpError("DWCONV has no dilation field; dilation > 1 would "
                          "need a strided re-slice that is not implemented");

  Epilogue epi = readEpilogue(op.getEpilogue(), op.getResidual());
  if (!epi.requant)
    return op.emitOpError("needs a requantized epilogue to reach Level 2");
  if (epi.hasResidual())
    return op.emitOpError("a residual add on a depthwise layer is not lowered "
                          "(MobileNetV2 never has one)");

  const int64_t S = strides[0];
  const int64_t IH = in[1], IW = in[2], OH = out[1], OW = out[2];
  const int64_t cPad = roundUp(C, kDwuLanes);

  if (failed(checkAccumulatorBound(op, KH * KW, op.getWeights(), op.getBias(),
                                   /*ocLast=*/false, OC,
                                   op.getZeroPoints().getInput(), epi.requant)))
    return failure();

  // Tiling: rows and columns of the output, subject to ACC and SPM_A. There is
  // no channel tiling -- DWCONV iterates channel groups internally for the same
  // cycle count in one instruction (ISA.md §9.1).
  const int64_t spmABudget = kSpmABytes / spmReserve;
  std::optional<std::pair<int64_t, int64_t>> best;
  int64_t bestIn = 0, bestOut = 0, bestAcc = 0;
  uint64_t bestCycles = ~uint64_t(0);
  for (int64_t ohT : divisorsOf(OH)) {
    if (ohT > kDmaMaxN2)
      break;
    for (int64_t owT : divisorsOf(OW)) {
      const int64_t ihP = (ohT - 1) * S + KH;
      const int64_t iwP = (owT - 1) * S + KW;
      const int64_t inBytes = ihP * iwP * cPad + kDwuLanes;
      const int64_t outBytes = ohT * owT * cPad + kDwuLanes;
      const int64_t accWords = ohT * owT * cPad;
      if (accWords > kAccWords || inBytes + outBytes > spmABudget)
        continue;
      if (iwP * cPad > kDmaMaxLen0 || ihP > kDmaMaxN1)
        continue;
      uint64_t cyc = std::max(
          ::kea::keaDwconvOccupancy(ohT, owT, cPad, KH, KW),
          std::max(::kea::keaVquantOccupancy(ohT * owT, cPad),
                   ::kea::keaDmaOccupancy(iwP * C, ihP, 1) +
                       ::kea::keaDmaOccupancy(C, owT, ohT)));
      cyc *= static_cast<uint64_t>((OH / ohT) * (OW / owT));
      if (cyc < bestCycles ||
          (cyc == bestCycles && ohT * owT > best->first * best->second)) {
        bestCycles = cyc;
        best = {ohT, owT};
        bestIn = inBytes;
        bestOut = outBytes;
        bestAcc = accWords;
      }
    }
  }
  if (!best)
    return op.emitOpError()
           << "no output tile fits: a 1x1 depthwise tile of " << cPad
           << " channels already needs " << (KH * KW * cPad)
           << " B of SPM_A budget " << spmABudget;
  const int64_t ohT = best->first, owT = best->second;
  const int64_t ihP = (ohT - 1) * S + KH, iwP = (owT - 1) * S + KW;
  const int64_t sp = cPad, sr = iwP * cPad;

  Value inBuf = dram(op.getInput(), "activation", layerName("in"));
  Value outBuf = dram(op.getOutput(), "activation", layerName("out"));
  Value wBuf = makeBuffer(loc, KH * KW * cPad, AddressSpace::DRAM,
                          layerName("weights"), "weights",
                          ValueRange{op.getWeights()});
  wBuf.getDefiningOp<AllocOp>()->setAttr("layout", b.getStringAttr("dwu_planes"));
  Value qpBuf = emitQParamBlock(loc, op.getBias(), op.getWeights(), epi.requant,
                                op.getZeroPoints().getInput(), cPad,
                                layerName("qparams"));

  b.create<TraceOp>(loc, b.getStringAttr("begin"), layerId, 0, StringAttr());

  Value wTile = scratch(loc, KH * KW * cPad, AddressSpace::W,
                        layerName("wtile"));
  // One contiguous run per [kh][kw] tap plane of `cPad` bytes, so `len0` stays
  // well inside its 16 bits whatever the channel count.
  b.create<DmaLoadOp>(loc, wBuf, wTile, /*dram_addr=*/0, /*spm_addr=*/0,
                      /*len0=*/cPad, /*n1=*/KH * KW, /*n2=*/1,
                      /*dram_s1=*/cPad, /*dram_s2=*/0, /*spm_s1=*/cPad,
                      /*spm_s2=*/0, StringAttr());
  Value qpTile = scratch(loc, cPad * kQuantParamBytes, AddressSpace::W,
                         layerName("qptile"));
  b.create<DmaLoadOp>(loc, qpBuf, qpTile, 0, 0, cPad * kQuantParamBytes, 1, 1,
                      cPad * kQuantParamBytes, cPad * kQuantParamBytes,
                      cPad * kQuantParamBytes, cPad * kQuantParamBytes,
                      StringAttr());

  for (int64_t oh0 = 0; oh0 < OH; oh0 += ohT) {
    for (int64_t ow0 = 0; ow0 < OW; ow0 += owT) {
      ++numTiles;
      Value aTile = scratch(loc, bestIn, AddressSpace::A, layerName("atile"));
      const int64_t ih0 = oh0 * S - pads[0];
      const int64_t iw0 = ow0 * S - pads[2];
      // The fill also defines the padded channel lanes [C, cPad); their
      // weight planes are zero, so they contribute exactly nothing.
      emitFill(loc, aTile, bestIn, op.getZeroPoints().getInput());

      const int64_t sh0 = std::max<int64_t>(ih0, 0);
      const int64_t sw0 = std::max<int64_t>(iw0, 0);
      const int64_t sh1 = std::min(ih0 + ihP, IH);
      const int64_t sw1 = std::min(iw0 + iwP, IW);
      if (sh1 > sh0 && sw1 > sw0) {
        const int64_t dstOff = (sh0 - ih0) * sr + (sw0 - iw0) * sp;
        const int64_t srcOff = (sh0 * IW + sw0) * C;
        if (cPad == C) {
          // Dense: one contiguous run per row.
          b.create<DmaLoadOp>(loc, inBuf, aTile, srcOff, dstOff,
                              (sw1 - sw0) * C, sh1 - sh0, 1, IW * C, 0, sr, 0,
                              StringAttr());
        } else {
          // Channel padded: one run per pixel, gathering C of every cPad.
          b.create<DmaLoadOp>(loc, inBuf, aTile, srcOff, dstOff, C, sw1 - sw0,
                              sh1 - sh0, C, IW * C, sp, sr, StringAttr());
        }
      }

      Value accTile = scratch(loc, bestAcc, AddressSpace::ACC, layerName("acc"));
      Value oTile = scratch(loc, bestOut, AddressSpace::A, layerName("otile"));

      b.create<DwconvOp>(loc, aTile, wTile, accTile, /*a_addr=*/0,
                         /*w_addr=*/0, /*acc_addr=*/0, /*out_h=*/ohT,
                         /*out_w=*/owT, /*channels=*/cPad,
                         /*a_row_stride=*/sr, /*a_pix_stride=*/sp,
                         /*kernel=*/KH, /*stride=*/S, /*accumulate=*/false);

      b.create<VquantOp>(loc, accTile, oTile, qpTile, /*acc_addr=*/0,
                         /*out_addr=*/0, /*qparam_addr=*/0,
                         /*num_pixels=*/ohT * owT, /*channels=*/cPad,
                         /*acc_pix_stride=*/cPad, /*out_pix_stride=*/cPad,
                         /*out_zp=*/epi.requant.getOutputZp(),
                         /*clamp_lo=*/epi.clampLo, /*clamp_hi=*/epi.clampHi,
                         /*int4=*/false);

      b.create<DmaStoreOp>(loc, oTile, outBuf,
                           /*dram_addr=*/(oh0 * OW + ow0) * OC,
                           /*spm_addr=*/0, /*len0=*/C, /*n1=*/owT, /*n2=*/ohT,
                           /*dram_s1=*/OC, /*dram_s2=*/OW * OC,
                           /*spm_s1=*/cPad, /*spm_s2=*/owT * cPad,
                           /*unit=*/StringAttr());
    }
  }

  b.create<TraceOp>(loc, b.getStringAttr("end"), layerId, 0, StringAttr());
  tileReport.push_back(b.getDictionaryAttr({
      b.getNamedAttr("layer", b.getI64IntegerAttr(layerId)),
      b.getNamedAttr("op", b.getStringAttr("kea.dwconv2d")),
      b.getNamedAttr("oh", b.getI64IntegerAttr(ohT)),
      b.getNamedAttr("ow", b.getI64IntegerAttr(owT)),
      b.getNamedAttr("channels", b.getI64IntegerAttr(cPad)),
      b.getNamedAttr("spm_a", b.getI64IntegerAttr(bestIn + bestOut)),
      b.getNamedAttr("spm_w",
                     b.getI64IntegerAttr(KH * KW * cPad +
                                         cPad * kQuantParamBytes)),
      b.getNamedAttr("acc", b.getI64IntegerAttr(bestAcc)),
      b.getNamedAttr("taps", b.getI64IntegerAttr(KH * KW)),
  }));
  return success();
}

//===----------------------------------------------------------------------===//
// Pooling -- VPOOL, ISA.md §10.3
//===----------------------------------------------------------------------===//

LogicalResult Tiler::lowerPool(PoolOp op) {
  Location loc = op.getLoc();
  auto in = shapeOf(op.getInput());
  auto out = shapeOf(op.getOutput());
  if (in.size() != 4 || in[0] != 1)
    return op.emitOpError("only rank-4 NHWC batch-1 pooling is lowered");
  auto pads = op.getPads();
  if (llvm::any_of(pads, [](int64_t p) { return p != 0; }))
    return op.emitOpError(
        "VPOOL is VALID padding only; a padded average pool would also need "
        "TOSA's padding-aware divisor, which the unit does not have");
  if (auto q = op.getQuant())
    if (!q->isNoOp())
      return op.emitOpError(
          "VPOOL requires identical input and output quantization (ISA.md "
          "§10.3); emit a separate kea.rescale instead");

  const int64_t IH = in[1], IW = in[2], C = in[3];
  const int64_t OH = out[1], OW = out[2];
  auto kern = op.getKernel();
  auto strides = op.getStrides();
  const bool avg = op.getKind() == PoolKind::AVG;

  const int64_t spmABudget = kSpmABytes / spmReserve;
  std::optional<int64_t> ohT;
  for (int64_t cand : llvm::reverse(divisorsOf(OH))) {
    const int64_t ihP = (cand - 1) * strides[0] + kern[0];
    if (ihP > IH)
      continue;
    if (ihP * IW * C + cand * OW * C > spmABudget)
      continue;
    if (IW * C > kDmaMaxLen0 || ihP > kDmaMaxN1)
      continue;
    ohT = cand;
    break;
  }
  if (!ohT)
    return op.emitOpError() << "no row band of the pooling input fits SPM_A/"
                            << spmReserve;
  const int64_t ihP = (*ohT - 1) * strides[0] + kern[0];

  Value inBuf = dram(op.getInput(), "activation", layerName("in"));
  Value outBuf = dram(op.getOutput(), "activation", layerName("out"));
  b.create<TraceOp>(loc, b.getStringAttr("begin"), layerId, 0, StringAttr());

  for (int64_t oh0 = 0; oh0 < OH; oh0 += *ohT) {
    ++numTiles;
    Value aTile = scratch(loc, ihP * IW * C, AddressSpace::A, layerName("in"));
    Value oTile =
        scratch(loc, *ohT * OW * C, AddressSpace::A, layerName("out"));
    b.create<DmaLoadOp>(loc, inBuf, aTile,
                        /*dram_addr=*/oh0 * strides[0] * IW * C,
                        /*spm_addr=*/0, /*len0=*/IW * C, /*n1=*/ihP, /*n2=*/1,
                        /*dram_s1=*/IW * C, /*dram_s2=*/0, /*spm_s1=*/IW * C,
                        /*spm_s2=*/0, StringAttr());
    b.create<VpoolOp>(loc, aTile, oTile, /*in_addr=*/0, /*out_addr=*/0,
                      /*out_h=*/*ohT, /*out_w=*/OW, /*channels=*/C,
                      /*kh=*/kern[0], /*kw=*/kern[1], /*stride_h=*/strides[0],
                      /*stride_w=*/strides[1], /*in_row_stride=*/IW * C,
                      /*out_row_stride=*/OW * C, /*avg=*/avg);
    b.create<DmaStoreOp>(loc, oTile, outBuf, /*dram_addr=*/oh0 * OW * C,
                         /*spm_addr=*/0, /*len0=*/OW * C, /*n1=*/*ohT,
                         /*n2=*/1, /*dram_s1=*/OW * C, /*dram_s2=*/OW * C,
                         /*spm_s1=*/OW * C, /*spm_s2=*/0, StringAttr());
  }

  b.create<TraceOp>(loc, b.getStringAttr("end"), layerId, 0, StringAttr());
  tileReport.push_back(b.getDictionaryAttr({
      b.getNamedAttr("layer", b.getI64IntegerAttr(layerId)),
      b.getNamedAttr("op", b.getStringAttr("kea.pool")),
      b.getNamedAttr("oh", b.getI64IntegerAttr(*ohT)),
      b.getNamedAttr("ow", b.getI64IntegerAttr(OW)),
      b.getNamedAttr("channels", b.getI64IntegerAttr(C)),
      b.getNamedAttr("spm_a",
                     b.getI64IntegerAttr(ihP * IW * C + *ohT * OW * C)),
  }));
  return success();
}

//===----------------------------------------------------------------------===//
// Standalone quantized add -- VADD, ISA.md §10.2
//===----------------------------------------------------------------------===//

LogicalResult Tiler::lowerAdd(AddOp op) {
  Location loc = op.getLoc();
  if (!op.getLhsQuant() || !op.getRhsQuant() || !op.getOutQuant())
    return op.emitOpError("only a fully quantized kea.add maps onto KEA_VADD; "
                          "an unquantized add has no instruction");
  if (shapeOf(op.getLhs()) != shapeOf(op.getRhs()))
    return op.emitOpError("broadcasting adds are not lowered: VADD is a flat "
                          "elementwise pass");

  AddParam ap;
  if (failed(buildAddParam(op, *op.getLhsQuant(), *op.getRhsQuant(),
                           *op.getOutQuant(), ap)))
    return failure();

  int64_t lo = -128, hi = 127;
  if (auto c = op.getClamp())
    if (c->size() == 2) {
      lo = (*c)[0];
      hi = (*c)[1];
    }

  const int64_t n = numElems(op.getOutput());
  // Three live tiles (lhs, rhs, out) share the SPM_A budget.
  const int64_t chunkMax = (kSpmABytes / spmReserve) / 3;
  int64_t chunk = std::min(n, chunkMax);
  if (chunk < 1)
    return op.emitOpError("SPM_A budget leaves no room for a VADD tile");

  Value aBuf = dram(op.getLhs(), "activation", layerName("lhs"));
  Value bBuf = dram(op.getRhs(), "activation", layerName("rhs"));
  Value oBuf = dram(op.getOutput(), "activation", layerName("out"));
  Value pBuf = makeBuffer(loc, kAddParamBytes, AddressSpace::DRAM,
                          layerName("addparams"), "addparam");
  auto pAlloc = pBuf.getDefiningOp<AllocOp>();
  pAlloc->setAttr("layout", b.getStringAttr("add_params"));
  pAlloc->setAttr("quant", *op.getLhsQuant());
  pAlloc->setAttr("residual_quant", *op.getRhsQuant());
  pAlloc->setAttr("output_quant", *op.getOutQuant());
  pAlloc->setAttr("add_param", b.getDenseI64ArrayAttr(ap.asArray()));

  b.create<TraceOp>(loc, b.getStringAttr("begin"), layerId, 0, StringAttr());
  Value pTile = scratch(loc, kAddParamBytes, AddressSpace::W,
                        layerName("aptile"));
  b.create<DmaLoadOp>(loc, pBuf, pTile, 0, 0, kAddParamBytes, 1, 1,
                      kAddParamBytes, kAddParamBytes, kAddParamBytes,
                      kAddParamBytes, StringAttr());

  for (int64_t off = 0; off < n; off += chunk) {
    ++numTiles;
    const int64_t len = std::min(chunk, n - off);
    Value aT = scratch(loc, len, AddressSpace::A, layerName("lhstile"));
    Value bT = scratch(loc, len, AddressSpace::A, layerName("rhstile"));
    Value oT = scratch(loc, len, AddressSpace::A, layerName("outtile"));
    b.create<DmaLoadOp>(loc, aBuf, aT, off, 0, len, 1, 1, len, len, len, len,
                        StringAttr());
    b.create<DmaLoadOp>(loc, bBuf, bT, off, 0, len, 1, 1, len, len, len, len,
                        StringAttr());
    b.create<VaddOp>(loc, aT, bT, oT, pTile, 0, 0, 0, 0, len, lo, hi);
    b.create<DmaStoreOp>(loc, oT, oBuf, off, 0, len, 1, 1, len, len, len, len,
                         StringAttr());
  }
  b.create<TraceOp>(loc, b.getStringAttr("end"), layerId, 0, StringAttr());
  tileReport.push_back(b.getDictionaryAttr({
      b.getNamedAttr("layer", b.getI64IntegerAttr(layerId)),
      b.getNamedAttr("op", b.getStringAttr("kea.add")),
      b.getNamedAttr("elems", b.getI64IntegerAttr(chunk)),
      b.getNamedAttr("spm_a", b.getI64IntegerAttr(3 * chunk)),
  }));
  return success();
}

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

LogicalResult Tiler::lower(Operation *op) {
  if (auto conv = dyn_cast<Conv2DOp>(op)) {
    auto in = shapeOf(conv.getInput());
    auto w = shapeOf(conv.getWeights()); // [OC, KH, KW, IC]
    auto out = shapeOf(conv.getOutput());
    if (in.size() != 4 || out.size() != 4 || w.size() != 4)
      return conv.emitOpError("expects rank-4 NHWC/OHWI operands");
    if (in[0] != 1)
      return conv.emitOpError("only batch 1 is lowered");
    ConvShape s;
    s.IH = in[1];
    s.IW = in[2];
    s.IC = in[3];
    s.OH = out[1];
    s.OW = out[2];
    s.OC = out[3];
    s.KH = w[1];
    s.KW = w[2];
    auto st = conv.getStrides(), dl = conv.getDilations(), pd = conv.getPads();
    s.strideH = st[0];
    s.strideW = st[1];
    s.dilH = dl[0];
    s.dilW = dl[1];
    s.padTop = pd[0];
    s.padLeft = pd[2];
    s.inputZp = conv.getZeroPoints().getInput();
    // ISA.md §8.6: fold a whole kernel row into one reduction tile when it
    // fits. This is the IC = 3 first layer of MobileNetV2 at 56% MXU
    // utilization instead of 19%, and it needs no new instruction -- only a
    // weight layout and a different k_rows.
    // Only worth it when it actually folds something: for KW == 1 the packed
    // and unpacked tilings are byte-identical and the layout name would only
    // confuse -kea-emit.
    s.channelPacked =
        s.KW > 1 && s.dilW == 1 && s.IC < kMxuK && s.KW * s.IC <= kMxuK;
    return lowerContraction(op, s, conv.getInput(), conv.getWeights(),
                            conv.getBias(),
                            readEpilogue(conv.getEpilogue(), conv.getResidual()),
                            conv.getOutput(),
                            s.channelPacked ? "mxu_tiles_16x16_packed"
                                            : "mxu_tiles_16x16");
  }

  if (auto fc = dyn_cast<FullyConnectedOp>(op)) {
    auto in = shapeOf(fc.getInput());   // [N, IC]
    auto w = shapeOf(fc.getWeights());  // [OC, IC]
    if (in.size() != 2 || w.size() != 2)
      return fc.emitOpError("expects rank-2 operands");
    ConvShape s;
    s.IH = 1;
    s.IW = in[0];
    s.IC = in[1];
    s.OH = 1;
    s.OW = in[0];
    s.OC = w[0];
    return lowerContraction(op, s, fc.getInput(), fc.getWeights(), fc.getBias(),
                            readEpilogue(fc.getEpilogue(), fc.getResidual()),
                            fc.getOutput(), "mxu_tiles_16x16");
  }

  if (auto mm = dyn_cast<MatmulOp>(op)) {
    auto a = shapeOf(mm.getA()); // [B, M, K]
    auto w = shapeOf(mm.getB()); // [B, K, N]
    if (a.size() != 3 || w.size() != 3)
      return mm.emitOpError("expects rank-3 operands");
    ConvShape s;
    s.batch = a[0];
    s.IH = 1;
    s.IW = a[1];
    s.IC = a[2];
    s.OH = 1;
    s.OW = a[1];
    s.OC = w[2];
    return lowerContraction(op, s, mm.getA(), mm.getB(), mm.getBias(),
                            readEpilogue(mm.getEpilogue(), mm.getResidual()),
                            mm.getOutput(), "mxu_tiles_16x16_kn");
  }

  if (auto dw = dyn_cast<DWConv2DOp>(op))
    return lowerDwconv(dw);
  if (auto p = dyn_cast<PoolOp>(op))
    return lowerPool(p);
  if (auto ad = dyn_cast<AddOp>(op))
    return lowerAdd(ad);

  if (auto rs = dyn_cast<ReshapeOp>(op)) {
    // A Level 1 reshape is a row-major reinterpretation of the same bytes, so
    // at Level 2 it is the same DRAM buffer under a new name. No instruction.
    dramFor[rs.getOutput()] =
        dram(rs.getInput(), "activation", layerName("reshape"));
    return success();
  }

  return op->emitOpError(
      "has no Level 2 lowering. -kea-tile lowers kea.conv2d, kea.dwconv2d, "
      "kea.matmul, kea.fully_connected, kea.pool, quantized kea.add and "
      "kea.reshape; kea.rescale, kea.clamp and kea.transpose must be "
      "eliminated by -kea-fuse (KEA-1 has no transpose unit and no way to "
      "rescale a tensor that never entered ACC).");
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

bool isLevel1(Operation *op) {
  return isa<Conv2DOp, DWConv2DOp, MatmulOp, FullyConnectedOp, AddOp, ClampOp,
             RescaleOp, PoolOp, ReshapeOp, TransposeOp>(op);
}

LogicalResult Tiler::run() {
  if (func.isExternal())
    return success();
  if (!func.getBody().hasOneBlock())
    return func.emitOpError(
        "-kea-tile needs a single-block function: KEA-1 has no branches");

  Block &body = func.front();
  SmallVector<Operation *> l1;
  for (Operation &op : body)
    if (isLevel1(&op))
      l1.push_back(&op);
  if (l1.empty()) {
    // Nothing to lower. Still refresh the live ranges and re-check errata E7 --
    // running -kea-tile over an already-Level-2 function is the cheapest way to
    // validate one, and E7 is not a property any single op verifier can see.
    refreshLiveRanges(func);
    return verifyWeightBanks(func);
  }

  b.setInsertionPoint(body.getTerminator());
  for (Operation *op : l1) {
    if (failed(lower(op)))
      return failure();
    ++layerId;
    ++numLayers;
  }

  // Whatever the function returned is a model output.
  Operation *term = body.getTerminator();
  for (Value v : term->getOperands()) {
    auto it = dramFor.find(v);
    if (it == dramFor.end())
      continue;
    if (auto alloc = it->second.getDefiningOp<AllocOp>())
      alloc->setAttr("role", b.getStringAttr("output"));
  }

  // Level 2 is a machine program, not a value-semantic function: it returns
  // nothing and ends with HALT (ISA.md §5.2).
  b.setInsertionPointToEnd(&body);
  Location loc = term->getLoc();
  term->erase();
  b.create<HaltOp>(loc, /*exit_code=*/0);
  b.create<func::ReturnOp>(loc);
  func.setType(b.getFunctionType(body.getArgumentTypes(), {}));

  // Erase the Level 1 ops, then anything that became dead.
  for (Operation *op : llvm::reverse(l1)) {
    op->dropAllUses();
    op->erase();
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : llvm::make_early_inc_range(body)) {
      if (op.getNumResults() == 0 || !op.use_empty())
        continue;
      if (!isMemoryEffectFree(&op))
        continue;
      op.erase();
      changed = true;
    }
  }

  // ISA.md §11.2: a program is at most KEA_MAX_INSTRUCTIONS long, and IMEM is
  // not paged. -kea-schedule will add SIGNAL/WAIT on top of this, so hitting
  // the limit here means the layer needs bigger tiles or the model needs to be
  // split across invocations.
  int64_t instrs = 0;
  for (Operation &op : body)
    if (op.getDialect() == func->getDialect() && !isa<AllocOp>(op))
      ++instrs;
  if (instrs > static_cast<int64_t>(::kea::KEA_MAX_INSTRUCTIONS))
    return func.emitOpError()
           << "lowers to " << instrs
           << " KEA-1 instructions, more than IMEM holds ("
           << ::kea::KEA_MAX_INSTRUCTIONS
           << "), and -kea-schedule still has to add its SIGNAL/WAIT pairs";

  refreshLiveRanges(func);
  return verifyWeightBanks(func); // errata E7
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct KeaTilePass : public mlir::kea::impl::KeaTileBase<KeaTilePass> {
  using mlir::kea::impl::KeaTileBase<KeaTilePass>::KeaTileBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (spmReserveFactor < 1) {
      func.emitError("spm-reserve-factor must be >= 1");
      return signalPassFailure();
    }
    Tiler tiler(func, spmReserveFactor);
    if (failed(tiler.run()))
      return signalPassFailure();

    numLayers = tiler.numLayers;
    numTiles = tiler.numTiles;
    numMatmuls = tiler.numMatmuls;

    if (reportTiles && !tiler.tileReport.empty()) {
      OpBuilder b(func.getContext());
      func->setAttr("kea.tiling", b.getArrayAttr(tiler.tileReport));
    }
  }
};

} // namespace
