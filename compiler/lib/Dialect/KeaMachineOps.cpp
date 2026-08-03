//===- KeaMachineOps.cpp - KEA Level 2 op implementations -------*- C++ -*-===//
//
// Verifiers for the machine-level ops declared in KeaMachineOps.td, plus the
// whole-function checks declared in KeaMachineOps.h. Level 1 op implementations
// are in KeaOps.cpp.
//
// EVERY CONSTANT HERE COMES FROM include/kea/hw_config.h. There are no magic
// numbers: `KEA_ALIGN_ACC_WORDS`, `KEA_MXU_MAX_ROWS`, `KEA_SPM_A_BYTES` and
// friends are the frozen contract (docs/ISA.md, header wins on disagreement),
// and duplicating any of them here is how the compiler and the assembler drift
// apart.
//
// The verifiers are deliberately eager. Every rule below is one the assembler
// would reject later anyway (`kea::keaValidate()` in isa.h checks the same
// list) -- catching it on a `kea.mm` in the middle of a readable Level 2
// function, with a source location, is worth far more than catching it on line
// 4,712 of a `.kasm` file.
//
//===----------------------------------------------------------------------===//

#include "kea/Dialect/KeaMachineOps.h"
#include "kea/Dialect/KeaDialect.h"
#include "kea/Dialect/KeaTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

// The frozen machine contract. Both headers are header-only C++17 with no
// dependencies, so including them links nothing (ADR-0001 keeps the MLIR half
// from linking native code; constexpr in a header is not linking).
//
// `::kea::` (leading `::`) is mandatory throughout: inside `namespace mlir::kea`
// a bare `kea::` resolves to `mlir::kea` itself and then fails to find anything.
#include "kea/hw_config.h"
#include "kea/isa.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::kea;

#define GET_OP_CLASSES
#include "kea/Dialect/KeaMachineOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

namespace {

/// Capacity of an address space, in the elements that space is addressed in.
int64_t spaceCapacity(AddressSpace as) {
  switch (as) {
  case AddressSpace::A:
    return static_cast<int64_t>(::kea::KEA_SPM_A_BYTES);
  case AddressSpace::W:
    return static_cast<int64_t>(::kea::KEA_SPM_W_BYTES);
  case AddressSpace::ACC:
    return static_cast<int64_t>(::kea::KEA_ACC_WORDS);
  case AddressSpace::DRAM:
    // 4 GiB; clamp to int64 so callers can compare without overflow.
    return static_cast<int64_t>(::kea::KEA_DRAM_BYTES);
  }
  return 0;
}

const char *spaceName(AddressSpace as) {
  switch (as) {
  case AddressSpace::A:
    return "SPM_A";
  case AddressSpace::W:
    return "SPM_W";
  case AddressSpace::ACC:
    return "ACC";
  case AddressSpace::DRAM:
    return "DRAM";
  }
  return "?";
}

BufferType bufTy(Value v) { return llvm::cast<BufferType>(v.getType()); }

/// Level 2 requires the flat element convention of KeaMachineOps.td: SPM and
/// DRAM buffers are `i8` (so an offset is a byte), ACC buffers are `i32` (so an
/// offset is a word). Without this the byte/word distinction is a comment
/// rather than a type.
LogicalResult checkElementType(Operation *op, Value v, StringRef what) {
  BufferType t = bufTy(v);
  unsigned bits = t.getElementType().getIntOrFloatBitWidth();
  if (t.getAddressSpace() == AddressSpace::ACC) {
    if (bits != 32)
      return op->emitOpError(what)
             << " must be an ACC buffer of i32 (ACC is addressed in int32 "
                "words), got "
             << t.getElementType();
  } else if (bits != 8) {
    return op->emitOpError(what)
           << " must be a buffer of i8 -- Level 2 SPM/DRAM buffers are flat "
              "byte extents, and int4 is a packing, not an element type; got "
           << t.getElementType();
  }
  return success();
}

/// `[lo, hi)` must be inside `[0, extent(v))`.
LogicalResult checkInBounds(Operation *op, Value v, Span s, StringRef what) {
  int64_t n = bufferExtent(v);
  AddressSpace as = bufTy(v).getAddressSpace();
  const char *unit = (as == AddressSpace::ACC) ? "words" : "bytes";
  if (s.lo < 0 || s.hi > n)
    return op->emitOpError(what)
           << " accesses [" << s.lo << ", " << s.hi << ") of a "
           << spaceName(as) << " buffer of " << n << ' ' << unit;
  return success();
}

LogicalResult checkAligned(Operation *op, int64_t v, int64_t align,
                           StringRef field, StringRef unit) {
  if (align != 0 && (v % align) != 0)
    return op->emitOpError() << field << " (" << v << ") must be a multiple of "
                             << align << ' ' << unit;
  return success();
}

LogicalResult checkRange(Operation *op, int64_t v, int64_t lo, int64_t hi,
                         StringRef field) {
  if (v < lo || v > hi)
    return op->emitOpError()
           << field << " must be in [" << lo << ", " << hi << "], got " << v;
  return success();
}

/// Every ACC displacement and stride in KEA-1 is a multiple of 16 words
/// (`KEA_ALIGN_ACC_WORDS`). Factored out because it applies to MATMUL, DWCONV
/// and VQUANT alike and getting it wrong on any one of them is the same bug.
LogicalResult checkAccAligned(Operation *op, int64_t v, StringRef field) {
  return checkAligned(op, v, ::kea::KEA_ALIGN_ACC_WORDS, field, "words");
}

/// A KEA-1 DMA descriptor, shared by dma_load and dma_store.
template <typename OpT>
LogicalResult verifyDmaDescriptor(OpT op, bool isLoad) {
  Value dram = op.getDram();
  Value spm = op.getSpm();
  if (failed(checkElementType(op, dram, "DRAM operand")) ||
      failed(checkElementType(op, spm, "SPM operand")))
    return failure();

  if (failed(checkRange(op, op.getLen0(), 1, ::kea::KEA_DMA_MAX_LEN0, "len0")) ||
      failed(checkRange(op, op.getN1(), 1, ::kea::KEA_DMA_MAX_N1, "n1")) ||
      failed(checkRange(op, op.getN2(), 1, ::kea::KEA_DMA_MAX_N2, "n2")))
    return failure();

  // ISA.md §6.1: a zero stride on the *destination* side is last-writer-wins,
  // which is never what a compiler means. A zero source stride is a legal
  // broadcast.
  int64_t n1 = op.getN1(), n2 = op.getN2();
  int64_t dst1 = isLoad ? op.getSpmS1() : op.getDramS1();
  int64_t dst2 = isLoad ? op.getSpmS2() : op.getDramS2();
  const char *dstSide = isLoad ? "SPM" : "DRAM";
  if ((n1 > 1 && dst1 == 0) || (n2 > 1 && dst2 == 0))
    return op.emitOpError()
           << "a zero " << dstSide
           << " stride on the destination side of a DMA is last-writer-wins "
              "and is rejected by the assembler";

  Span dspan = stridedSpan(op.getDramAddr(), n1, op.getDramS1(), n2,
                           op.getDramS2(), op.getLen0());
  Span sspan = stridedSpan(op.getSpmAddr(), n1, op.getSpmS1(), n2,
                           op.getSpmS2(), op.getLen0());
  if (failed(checkInBounds(op, dram, dspan, "DRAM side")) ||
      failed(checkInBounds(op, spm, sspan, "SPM side")))
    return failure();

  if (auto u = op.getUnit())
    if (*u != "DMA0" && *u != "DMA1")
      return op.emitOpError("unit must be \"DMA0\" or \"DMA1\", got \"")
             << *u << '"';
  return success();
}

LogicalResult verifyQueueAgnosticUnit(Operation *op, std::optional<StringRef> u) {
  if (!u)
    return success();
  static constexpr StringRef kUnits[] = {"MXU", "DWU", "VPU", "DMA0", "DMA1",
                                         "CTRL"};
  if (!llvm::is_contained(kUnits, *u))
    return op->emitOpError("unit must be one of MXU, DWU, VPU, DMA0, DMA1, "
                           "CTRL; got \"")
           << *u << '"';
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Exported helpers
//===----------------------------------------------------------------------===//

int64_t mlir::kea::bufferExtent(Value v) {
  return llvm::cast<BufferType>(v.getType()).getNumElements();
}

mlir::kea::Span mlir::kea::stridedSpan(int64_t base, int64_t count0,
                                       int64_t stride0, int64_t count1,
                                       int64_t stride1, int64_t extent) {
  int64_t lo = base, hi = base;
  auto apply = [&](int64_t count, int64_t stride) {
    if (count <= 1)
      return;
    int64_t d = (count - 1) * stride;
    if (d < 0)
      lo += d;
    else
      hi += d;
  };
  apply(count0, stride0);
  apply(count1, stride1);
  return {lo, hi + extent};
}

int64_t mlir::kea::addParamWorstCaseSum(int64_t aMult, int64_t aShift,
                                        int64_t bMult, int64_t bShift) {
  // |a - a_zp| <= 255 for int8 operands with an int8 zero point.
  const int64_t kMaxShifted = int64_t(255) << ::kea::KEA_VADD_LEFT_SHIFT;
  auto term = [&](int64_t mult, int64_t shift) -> int64_t {
    if (mult < 0)
      mult = -mult;
    // keaSrdhm(x, m) == round(x*m / 2^31), then keaRdpot(.., shift) for
    // shift > 0 (a non-positive shift is the identity).
    int64_t v = (kMaxShifted * mult) >> 31;
    if (shift > 0)
      v >>= std::min<int64_t>(shift, 62);
    return v + 1; // +1 covers the rounding nudge in both stages.
  };
  return term(aMult, aShift) + term(bMult, bShift);
}

LogicalResult mlir::kea::verifyWeightBanks(Operation *root) {
  LogicalResult result = success();
  root->walk([&](Block *block) {
    bool written[::kea::KEA_MXU_WEIGHT_BANKS] = {false, false};
    for (Operation &op : *block) {
      if (auto ldw = dyn_cast<LoadWOp>(op)) {
        int64_t b = ldw.getBank();
        if (b >= 0 && b < ::kea::KEA_MXU_WEIGHT_BANKS)
          written[b] = true;
      } else if (auto mm = dyn_cast<MmOp>(op)) {
        int64_t b = mm.getBank();
        if (b < 0 || b >= ::kea::KEA_MXU_WEIGHT_BANKS)
          continue; // the op verifier already reported this
        if (!written[b]) {
          mm.emitOpError("reads MXU weight bank ")
              << b
              << " but no kea.load_w has written it (errata E7: banks are "
                 "undefined at reset and the zero-fill must not be relied on)";
          result = failure();
        }
      }
    }
  });
  return result;
}

void mlir::kea::refreshLiveRanges(Operation *root) {
  root->walk([&](Block *block) {
    // Position of every op in the block.
    DenseMap<Operation *, int64_t> index;
    int64_t n = 0;
    for (Operation &op : *block)
      index[&op] = n++;

    for (Operation &op : *block) {
      auto alloc = dyn_cast<AllocOp>(op);
      if (!alloc)
        continue;
      BufferType t = llvm::cast<BufferType>(alloc.getResult().getType());
      if (!t.isOnChip()) {
        // DRAM buffers are not allocated by -kea-alloc; they are symbols
        // resolved by the assembler against model.map.json (ADR-0001 rule 3).
        alloc->removeAttr("live");
        continue;
      }
      int64_t first = index[&op];
      int64_t last = first;
      for (Operation *user : alloc.getResult().getUsers()) {
        auto it = index.find(user);
        if (it != index.end())
          last = std::max(last, it->second);
      }
      alloc->setAttr("live", DenseI64ArrayAttr::get(alloc.getContext(),
                                                    {first, last}));
    }
  });
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

LogicalResult AllocOp::verify() {
  BufferType t = llvm::cast<BufferType>(getResult().getType());

  if (t.getShape().size() != 1)
    return emitOpError("Level 2 buffers are flat: expected a rank-1 result "
                       "type, got rank ")
           << t.getShape().size();
  if (failed(checkElementType(*this, getResult(), "result")))
    return failure();

  static constexpr StringRef kRoles[] = {"input",   "output",   "activation",
                                         "weights", "qparam",   "addparam",
                                         "scratch"};
  if (!llvm::is_contained(kRoles, getRole()))
    return emitOpError("unknown role \"")
           << getRole()
           << "\"; expected one of input, output, activation, weights, "
              "qparam, addparam, scratch";

  bool onChip = t.isOnChip();
  if (onChip && getRole() != "scratch")
    return emitOpError("an on-chip buffer must have role \"scratch\", got \"")
           << getRole() << '"';
  if (!onChip && getRole() == "scratch")
    return emitOpError("role \"scratch\" is for on-chip buffers; this one is "
                       "in DRAM");

  if (getAlignment() <= 0)
    return emitOpError("alignment must be positive, got ") << getAlignment();

  int64_t cap = spaceCapacity(t.getAddressSpace());
  if (getExtent() > cap)
    return emitOpError("needs ")
           << getExtent() << " elements of " << spaceName(t.getAddressSpace())
           << ", which holds " << cap;

  if (auto live = getLive()) {
    if (live->size() != 2)
      return emitOpError("live must be [first, last], got ")
             << live->size() << " elements";
    if ((*live)[0] < 0 || (*live)[1] < (*live)[0])
      return emitOpError("live range [")
             << (*live)[0] << ", " << (*live)[1] << "] is not ordered";
    if (!onChip)
      return emitOpError("a DRAM buffer has no live range: it is a symbol "
                         "resolved by the assembler, not an allocation");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DmaLoadOp / DmaStoreOp
//===----------------------------------------------------------------------===//

LogicalResult DmaLoadOp::verify() {
  return verifyDmaDescriptor(*this, /*isLoad=*/true);
}

LogicalResult DmaStoreOp::verify() {
  return verifyDmaDescriptor(*this, /*isLoad=*/false);
}

//===----------------------------------------------------------------------===//
// LoadWOp
//===----------------------------------------------------------------------===//

LogicalResult LoadWOp::verify() {
  if (failed(checkElementType(*this, getW(), "weight operand")))
    return failure();
  if (failed(checkRange(*this, getKRows(), 1, ::kea::KEA_MXU_K, "k_rows")) ||
      failed(checkRange(*this, getNCols(), 1, ::kea::KEA_MXU_N, "n_cols")) ||
      failed(checkRange(*this, getBank(), 0, ::kea::KEA_MXU_WEIGHT_BANKS - 1,
                        "bank")))
    return failure();

  // ISA.md §7.2 / §11.1.
  const int64_t align = getInt4() ? ::kea::KEA_ALIGN_LOADW_INT4
                                  : ::kea::KEA_ALIGN_LOADW_INT8;
  if (failed(checkAligned(*this, getWAddr(), align, "w_addr", "bytes")) ||
      failed(checkAligned(*this, getWRowStride(), align, "w_row_stride",
                          "bytes")))
    return failure();
  if (getWRowStride() < 0)
    return emitOpError("w_row_stride must be >= 0, got ") << getWRowStride();

  // The bank always latches 16 rows; rows >= k_rows are zeros installed by the
  // hardware, so only k_rows of them are actually fetched.
  int64_t rowBytes = getInt4() ? (::kea::KEA_MXU_N / 2) : ::kea::KEA_MXU_N;
  Span s = stridedSpan(getWAddr(), getKRows(), getWRowStride(), 1, 0, rowBytes);
  return checkInBounds(*this, getW(), s, "weight tile");
}

//===----------------------------------------------------------------------===//
// MmOp
//===----------------------------------------------------------------------===//

LogicalResult MmOp::verify() {
  if (failed(checkElementType(*this, getA(), "activation operand")) ||
      failed(checkElementType(*this, getAcc(), "accumulator operand")))
    return failure();

  if (getMInner() < 1 || getMOuter() < 1)
    return emitOpError("m_inner and m_outer must be >= 1, got ")
           << getMInner() << " and " << getMOuter();

  // ISA.md §7.3: M*16 int32 words must fit in ACC.
  int64_t rows = getMInner() * getMOuter();
  if (rows > static_cast<int64_t>(::kea::KEA_MXU_MAX_ROWS))
    return emitOpError("m_inner * m_outer = ")
           << rows << " exceeds KEA_MXU_MAX_ROWS (" << ::kea::KEA_MXU_MAX_ROWS
           << "), the ACC capacity bound";

  if (failed(checkRange(*this, getBank(), 0, ::kea::KEA_MXU_WEIGHT_BANKS - 1,
                        "bank")))
    return failure();

  if (failed(checkAccAligned(*this, getAccAddr(), "acc_addr")) ||
      failed(checkAccAligned(*this, getAccInnerStride(), "acc_inner_stride")) ||
      failed(checkAccAligned(*this, getAccOuterStride(), "acc_outer_stride")))
    return failure();

  // The array always reads 16 activation elements and always writes 16 ACC
  // words per row, whatever the resident tile's k_rows/n_cols. Both must be in
  // bounds -- that is exactly why conv activation tiles are over-allocated.
  int64_t aElems = getInt4() ? (::kea::KEA_MXU_K / 2) : ::kea::KEA_MXU_K;
  Span as = stridedSpan(getAAddr(), getMInner(), getAInnerStride(), getMOuter(),
                        getAOuterStride(), aElems);
  Span qs = stridedSpan(getAccAddr(), getMInner(), getAccInnerStride(),
                        getMOuter(), getAccOuterStride(), ::kea::KEA_MXU_N);
  if (failed(checkInBounds(*this, getA(), as, "activation walk")) ||
      failed(checkInBounds(*this, getAcc(), qs, "accumulator walk")))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// DwconvOp
//===----------------------------------------------------------------------===//

LogicalResult DwconvOp::verify() {
  if (failed(checkElementType(*this, getA(), "activation operand")) ||
      failed(checkElementType(*this, getW(), "kernel operand")) ||
      failed(checkElementType(*this, getAcc(), "accumulator operand")))
    return failure();

  if (getOutH() < 1 || getOutW() < 1)
    return emitOpError("out_h and out_w must be >= 1");
  if (getKernel() != ::kea::KEA_DWU_KERNEL_3 &&
      getKernel() != ::kea::KEA_DWU_KERNEL_5)
    return emitOpError("kernel must be 3 or 5, got ") << getKernel();
  if (getStride() < 1 || getStride() > ::kea::KEA_DWU_MAX_STRIDE)
    return emitOpError("stride must be 1 or 2, got ") << getStride();

  if (getChannels() < ::kea::KEA_DWU_LANES)
    return emitOpError("channels must be >= ")
           << ::kea::KEA_DWU_LANES << ", got " << getChannels();
  if (failed(checkAligned(*this, getChannels(), ::kea::KEA_DWU_LANES,
                          "channels", "channels")))
    return failure();
  if (failed(checkAccAligned(*this, getAccAddr(), "acc_addr")))
    return failure();

  int64_t k = getKernel(), s = getStride();
  // Input window: (out - 1)*S + K rows / pixels.
  int64_t rows = (getOutH() - 1) * s + k;
  int64_t pix = (getOutW() - 1) * s + k;
  Span as = stridedSpan(getAAddr(), pix, getAPixStride(), rows,
                        getARowStride(), getChannels());
  if (failed(checkInBounds(*this, getA(), as, "activation window")))
    return failure();

  // Weights are [KH][KW][C] int8, contiguous.
  Span ws = stridedSpan(getWAddr(), 1, 0, 1, 0, k * k * getChannels());
  if (failed(checkInBounds(*this, getW(), ws, "kernel planes")))
    return failure();

  // The ACC region is dense [out_h][out_w][channels].
  Span qs = stridedSpan(getAccAddr(), 1, 0, 1, 0,
                        getOutH() * getOutW() * getChannels());
  return checkInBounds(*this, getAcc(), qs, "accumulator tile");
}

//===----------------------------------------------------------------------===//
// VquantOp
//===----------------------------------------------------------------------===//

LogicalResult VquantOp::verify() {
  if (failed(checkElementType(*this, getAcc(), "accumulator operand")) ||
      failed(checkElementType(*this, getOut(), "output operand")) ||
      failed(checkElementType(*this, getQparam(), "qparam operand")))
    return failure();

  if (getNumPixels() < 1)
    return emitOpError("num_pixels must be >= 1");
  if (getChannels() < ::kea::KEA_VPU_LANES)
    return emitOpError("channels must be >= ")
           << ::kea::KEA_VPU_LANES << ", got " << getChannels();
  if (failed(checkAligned(*this, getChannels(), ::kea::KEA_VPU_LANES,
                          "channels", "channels")))
    return failure();

  if (failed(checkAccAligned(*this, getAccAddr(), "acc_addr")) ||
      failed(checkAccAligned(*this, getAccPixStride(), "acc_pix_stride")))
    return failure();
  if (failed(checkAligned(*this, getQparamAddr(), ::kea::KEA_ALIGN_QPARAM,
                          "qparam_addr", "bytes")))
    return failure();

  if (failed(checkRange(*this, getOutZp(), -128, 127, "out_zp")) ||
      failed(checkRange(*this, getClampLo(), -128, 127, "clamp_lo")) ||
      failed(checkRange(*this, getClampHi(), -128, 127, "clamp_hi")))
    return failure();
  if (getClampLo() > getClampHi())
    return emitOpError("clamp_lo (")
           << getClampLo() << ") > clamp_hi (" << getClampHi() << ')';
  if (getInt4() && (getClampLo() < ::kea::KEA_INT4_MIN ||
                    getClampHi() > ::kea::KEA_INT4_MAX))
    return emitOpError("int4 output requires the clamps to lie within [-8, 7]");

  Span qs = stridedSpan(getAccAddr(), getNumPixels(), getAccPixStride(), 1, 0,
                        getChannels());
  if (failed(checkInBounds(*this, getAcc(), qs, "accumulator walk")))
    return failure();

  int64_t outElems = getInt4() ? (getChannels() + 1) / 2 : getChannels();
  Span os = stridedSpan(getOutAddr(), getNumPixels(), getOutPixStride(), 1, 0,
                        outElems);
  if (failed(checkInBounds(*this, getOut(), os, "output walk")))
    return failure();

  // `channels` KeaQuantParam records, back to back.
  Span ps = stridedSpan(getQparamAddr(), 1, 0, 1, 0,
                        getChannels() *
                            static_cast<int64_t>(sizeof(::kea::KeaQuantParam)));
  return checkInBounds(*this, getQparam(), ps, "KeaQuantParam array");
}

//===----------------------------------------------------------------------===//
// VaddOp
//===----------------------------------------------------------------------===//

LogicalResult VaddOp::verify() {
  for (auto [v, what] : {std::make_pair(getA(), StringRef("lhs operand")),
                         std::make_pair(getB(), StringRef("rhs operand")),
                         std::make_pair(getOut(), StringRef("result operand")),
                         std::make_pair(getParam(), StringRef("param operand"))})
    if (failed(checkElementType(*this, v, what)))
      return failure();

  if (getNumElems() < 1)
    return emitOpError("num_elems must be >= 1");
  if (failed(checkAligned(*this, getParamAddr(), ::kea::KEA_ALIGN_QPARAM,
                          "param_addr", "bytes")))
    return failure();
  if (failed(checkRange(*this, getClampLo(), -128, 127, "clamp_lo")) ||
      failed(checkRange(*this, getClampHi(), -128, 127, "clamp_hi")))
    return failure();
  if (getClampLo() > getClampHi())
    return emitOpError("clamp_lo > clamp_hi");

  // VADD is flat: `num_elems` contiguous int8 lanes on each side.
  const int64_t n = getNumElems();
  if (failed(checkInBounds(*this, getA(), Span{getAAddr(), getAAddr() + n},
                           "lhs")) ||
      failed(checkInBounds(*this, getB(), Span{getBAddr(), getBAddr() + n},
                           "rhs")) ||
      failed(checkInBounds(*this, getOut(),
                           Span{getOutAddr(), getOutAddr() + n}, "result")))
    return failure();

  // One KeaAddParam record.
  return checkInBounds(
      *this, getParam(),
      Span{getParamAddr(),
           getParamAddr() + static_cast<int64_t>(sizeof(::kea::KeaAddParam))},
      "KeaAddParam");
}

//===----------------------------------------------------------------------===//
// VpoolOp
//===----------------------------------------------------------------------===//

LogicalResult VpoolOp::verify() {
  if (failed(checkElementType(*this, getIn(), "input operand")) ||
      failed(checkElementType(*this, getOut(), "output operand")))
    return failure();

  if (getOutH() < 1 || getOutW() < 1 || getChannels() < 1)
    return emitOpError("out_h, out_w and channels must be >= 1");
  if (failed(checkRange(*this, getKh(), 1, 32, "kh")) ||
      failed(checkRange(*this, getKw(), 1, 32, "kw")) ||
      failed(checkRange(*this, getStrideH(), 1, 8, "stride_h")) ||
      failed(checkRange(*this, getStrideW(), 1, 8, "stride_w")))
    return failure();

  // Input/output pixel strides are implicitly `channels` bytes (ISA.md §10.3).
  int64_t inRows = (getOutH() - 1) * getStrideH() + getKh();
  int64_t inCols = (getOutW() - 1) * getStrideW() + getKw();
  Span is = stridedSpan(getInAddr(), inCols, getChannels(), inRows,
                        getInRowStride(), getChannels());
  Span os = stridedSpan(getOutAddr(), getOutW(), getChannels(), getOutH(),
                        getOutRowStride(), getChannels());
  if (failed(checkInBounds(*this, getIn(), is, "pooling window")))
    return failure();
  return checkInBounds(*this, getOut(), os, "pooling output");
}

//===----------------------------------------------------------------------===//
// VcopyOp
//===----------------------------------------------------------------------===//

LogicalResult VcopyOp::verify() {
  if (failed(checkElementType(*this, getDst(), "destination operand")))
    return failure();
  if (getRowBytes() < 1 || getRows() < 1)
    return emitOpError("row_bytes and rows must be >= 1");

  if (getFill()) {
    if (getSrc())
      return emitOpError("fill mode ignores the source; do not pass one");
    if (getSrcAddr() != 0 || getSrcRowStride() != 0)
      return emitOpError("fill mode ignores src_addr / src_row_stride; they "
                         "must be 0 so the encoded instruction is canonical");
    if (failed(checkRange(*this, getFillValue(), -128, 127, "fill_value")))
      return failure();
  } else {
    if (!getSrc())
      return emitOpError("copy mode needs a source operand (add `fill` for a "
                         "constant fill)");
    if (failed(checkElementType(*this, getSrc(), "source operand")))
      return failure();
    if (getFillValue() != 0)
      return emitOpError("fill_value is meaningful only in fill mode");
    Span ss = stridedSpan(getSrcAddr(), getRows(), getSrcRowStride(), 1, 0,
                          getRowBytes());
    if (failed(checkInBounds(*this, getSrc(), ss, "copy source")))
      return failure();
  }

  Span ds = stridedSpan(getDstAddr(), getRows(), getDstRowStride(), 1, 0,
                        getRowBytes());
  return checkInBounds(*this, getDst(), ds, "copy destination");
}

//===----------------------------------------------------------------------===//
// SignalOp / WaitOp / TraceOp / HaltOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyEventOp(Operation *op, int64_t event, int64_t value,
                                   std::optional<StringRef> unit) {
  if (event >= ::kea::KEA_NUM_EVENTS)
    return op->emitOpError("event id must be 0..")
           << (::kea::KEA_NUM_EVENTS - 1) << ", got " << event;
  if (value < 0 || value > static_cast<int64_t>(::kea::KEA_EVENT_MAX))
    return op->emitOpError("value must be 0..")
           << ::kea::KEA_EVENT_MAX << ", got " << value;
  return verifyQueueAgnosticUnit(op, unit);
}

LogicalResult SignalOp::verify() {
  return verifyEventOp(*this, getEvent(), getValue(), getUnit());
}

LogicalResult WaitOp::verify() {
  return verifyEventOp(*this, getEvent(), getValue(), getUnit());
}

LogicalResult TraceOp::verify() {
  StringRef k = getKind();
  if (k != "marker" && k != "begin" && k != "end")
    return emitOpError("kind must be \"marker\", \"begin\" or \"end\", got \"")
           << k << '"';
  return verifyQueueAgnosticUnit(*this, getUnit());
}

LogicalResult HaltOp::verify() {
  // ISA.md §5.2: HALT is the last instruction; anything after it is
  // unreachable and the assembler rejects it.
  Operation *next = getOperation()->getNextNode();
  if (next && !next->hasTrait<OpTrait::IsTerminator>())
    return emitOpError("must be the last instruction in the block, but is "
                       "followed by ")
           << next->getName();
  return success();
}
