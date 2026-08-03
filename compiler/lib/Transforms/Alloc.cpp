//===- Alloc.cpp - -kea-alloc: the exact static memory planner --*- C++ -*-===//
//
// THE PASS THAT MAKES "NO RUNTIME ALLOCATOR" TRUE.
//
// In:  Level 2 with symbolic `kea.alloc` buffers (docs/DIALECT_L2.md §4).
// Out: the same program with every buffer's base address decided at compile
//      time, plus the DRAM arena geometry and the SPM allocation map.
//
// Read docs/MEMORY_PLANNING.md alongside this file. In brief:
//
//   * On-chip buffers are placed by **live-range packing**: two buffers may
//     share storage iff their `live = [first, last]` block-index intervals are
//     disjoint. The packer is greedy-by-size / first-fit-by-offset.
//   * SPM_A and SPM_W are byte addressed; **ACC is addressed in int32 words**.
//     Sizes, offsets and capacities are always in the space's own unit, which
//     is also the buffer's element type (`i8` vs `i32`).
//   * Alignment is *derived from the ops that use the buffer* (ISA.md §11.1),
//     then re-checked on the absolute addresses. Nothing is assumed.
//   * DRAM is laid out as the three-region `KeafDramLayout` split
//     (include/kea/keaf.h): constants, host I/O, and a live-range packed
//     intermediate-activation scratch region.
//
// Addresses are stamped as `addr` on the `kea.alloc`, not folded into the
// per-op displacements. That is docs/DIALECT_L2.md §1.1(a)'s contract --
// `-kea-emit` writes `instr.X_addr = base(X) + X_addr` -- and it is also forced:
// the Level 2 verifiers check that every strided walk stays inside its buffer,
// so a displacement carrying an absolute address would read as out of bounds.
//
// EVERY capacity and alignment constant comes from include/kea/hw_config.h.
//
//===----------------------------------------------------------------------===//

#include "kea/Transforms/Passes.h"

#include "kea/Dialect/KeaAttrs.h"
#include "kea/Dialect/KeaMachineOps.h"
#include "kea/Dialect/KeaOps.h"
#include "kea/Dialect/KeaTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "kea/hw_config.h"

#include <algorithm>
#include <cstdint>
#include <string>

namespace mlir {
namespace kea {
#define GEN_PASS_DEF_KEAALLOC
#include "kea/Transforms/Passes.h.inc"
} // namespace kea
} // namespace mlir

using namespace mlir;
using namespace mlir::kea;

namespace {

//===----------------------------------------------------------------------===//
// Attribute names this pass writes
//===----------------------------------------------------------------------===//

/// The base address `-kea-emit` adds to every displacement into this buffer.
/// Bytes in SPM_A / SPM_W / DRAM, int32 words in ACC.
constexpr StringLiteral kAddr = "addr";
/// Mirrors the KEAF `SPM_MAP` section / model.map.json `spm_map` (debug).
constexpr StringLiteral kSpmMap = "kea.spm_map";
/// Mirrors `KeafDramLayout` / model.map.json `dram` (load bearing, not debug).
constexpr StringLiteral kDramLayout = "kea.dram_layout";
/// What the packing achieved, per space.
constexpr StringLiteral kOccupancy = "kea.occupancy";

//===----------------------------------------------------------------------===//
// Small helpers
//===----------------------------------------------------------------------===//

int64_t alignUp(int64_t v, int64_t a) {
  return a <= 1 ? v : ((v + a - 1) / a) * a;
}

/// Capacity of a space, in the unit that space is *addressed* in. Never
/// hardcoded: SPM_A/SPM_W are bytes, ACC is int32 words (ISA.md §2.2).
int64_t spaceCapacity(AddressSpace as) {
  switch (as) {
  case AddressSpace::A:
    return static_cast<int64_t>(::kea::KEA_SPM_A_BYTES);
  case AddressSpace::W:
    return static_cast<int64_t>(::kea::KEA_SPM_W_BYTES);
  case AddressSpace::ACC:
    return static_cast<int64_t>(::kea::KEA_ACC_WORDS);
  case AddressSpace::DRAM:
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

/// The unit an offset in this space is counted in. ACC is the one that trips
/// people, so it is spelled out in every diagnostic.
const char *spaceUnit(AddressSpace as) {
  return as == AddressSpace::ACC ? "int32 words" : "bytes";
}

/// The three on-chip spaces, in report order.
constexpr AddressSpace kOnChipSpaces[] = {AddressSpace::A, AddressSpace::W,
                                          AddressSpace::ACC};

/// True for the Level 2 ops that become a KEA-1 instruction. `kea.alloc` is the
/// only op in the dialect that does not, so PCs are "kea ops that are not an
/// alloc" rather than a list that would rot when an op is added.
bool isInstruction(Operation *op) {
  return op->getDialect() &&
         op->getDialect()->getNamespace() == KeaDialect::getDialectNamespace() &&
         !isa<AllocOp>(op);
}

//===----------------------------------------------------------------------===//
// One planned buffer
//===----------------------------------------------------------------------===//

struct Buffer {
  AllocOp alloc;
  StringRef name;
  StringRef role;
  AddressSpace space;
  /// Size in the space's addressing unit (bytes, or int32 words for ACC).
  int64_t size = 0;
  /// Required base alignment, same unit as `size`.
  int64_t align = 1;
  /// Live range, as block operation indices, closed at both ends.
  int64_t first = 0;
  int64_t last = 0;
  /// Instruction indices of the first and last instruction that touches the
  /// buffer, for `KeafSpmEntry`. -1 when nothing does.
  int64_t firstPc = -1;
  int64_t lastPc = -1;
  /// The answer.
  int64_t offset = -1;
  /// Whether this buffer competes for space with others (i.e. is packed) or
  /// simply gets its own region.
  bool packed = true;

  /// `AllocOp` is a value-semantic handle, so a copy of it is a perfectly good
  /// mutable one. This exists only so the diagnostics can be emitted from the
  /// `const Buffer &` the analysis passes around.
  AllocOp op() const { return alloc; }

  bool livesWith(const Buffer &o) const {
    return first <= o.last && o.first <= last;
  }
  int64_t end() const { return offset + size; }
};

/// Sum of live sizes at each block position, and where it peaks. This is
/// `maxlive`: a lower bound on what *any* allocator needs, because those
/// buffers all hold live data simultaneously.
struct MaxLive {
  int64_t bytes = 0;
  int64_t position = 0;
};

MaxLive computeMaxLive(ArrayRef<const Buffer *> bufs, int64_t numPositions) {
  // Difference array over [first, last].
  SmallVector<int64_t> delta(numPositions + 2, 0);
  for (const Buffer *b : bufs) {
    delta[b->first] += b->size;
    delta[b->last + 1] -= b->size;
  }
  MaxLive peak;
  int64_t running = 0;
  for (int64_t i = 0; i < numPositions; ++i) {
    running += delta[i];
    if (running > peak.bytes) {
      peak.bytes = running;
      peak.position = i;
    }
  }
  return peak;
}

/// `name (size unit)` for at most `limit` of the buffers live at `position`,
/// largest first, with a tail count. One line, because a multi-line diagnostic
/// is unreadable in a `-verify-diagnostics` expectation.
std::string describeLiveAt(ArrayRef<const Buffer *> bufs, int64_t position,
                           AddressSpace space, unsigned limit = 8) {
  SmallVector<const Buffer *> live;
  for (const Buffer *b : bufs)
    if (b->first <= position && position <= b->last)
      live.push_back(b);
  llvm::sort(live, [](const Buffer *a, const Buffer *b) {
    if (a->size != b->size)
      return a->size > b->size;
    return a->name < b->name;
  });

  std::string out;
  llvm::raw_string_ostream os(out);
  for (auto [i, b] : llvm::enumerate(live)) {
    if (i == limit) {
      os << ", and " << (live.size() - limit) << " more";
      break;
    }
    if (i)
      os << ", ";
    os << '\'' << b->name << "' " << b->size;
  }
  os << ' ' << spaceUnit(space);
  return out;
}

//===----------------------------------------------------------------------===//
// The packer
//===----------------------------------------------------------------------===//

/// Greedy-by-size, first-fit-by-offset -- TensorFlow Lite Micro's
/// `GreedyMemoryPlanner`, and what every production NPU compiler ships some
/// variant of.
///
/// Offline dynamic storage allocation (place fixed-size, fixed-lifetime blocks
/// in the smallest arena) is NP-hard, so this is an approximation. The order
/// matters: placing the largest buffers first means the awkward blocks get the
/// clean, unfragmented arena, and the small ones fill the gaps they leave. See
/// docs/MEMORY_PLANNING.md for what it costs against optimal.
///
/// Returns the failing buffer, or nullptr on success. `placed` accumulates
/// everything successfully placed, which is what the diagnostic needs.
Buffer *packGreedy(MutableArrayRef<Buffer *> order, int64_t capacity,
                   int64_t &numShared) {
  // Largest first; longest-lived first among equals; then by name so the
  // output is deterministic no matter what order the IR walk produced.
  llvm::sort(order, [](const Buffer *a, const Buffer *b) {
    if (a->size != b->size)
      return a->size > b->size;
    int64_t la = a->last - a->first, lb = b->last - b->first;
    if (la != lb)
      return la > lb;
    return a->name < b->name;
  });

  SmallVector<Buffer *> placed;
  for (Buffer *b : order) {
    // Everything already placed whose live range overlaps this one. Those are
    // the only blocks whose storage this buffer must avoid; everything else is
    // free real estate.
    SmallVector<Buffer *> conflicts;
    for (Buffer *p : placed)
      if (p->livesWith(*b))
        conflicts.push_back(p);
    llvm::sort(conflicts, [](const Buffer *x, const Buffer *y) {
      return x->offset < y->offset;
    });

    // Walk the conflicting blocks bottom up, taking the first aligned gap that
    // is big enough.
    int64_t candidate = 0;
    for (Buffer *p : conflicts) {
      if (p->offset >= candidate + b->size)
        break; // the gap below `p` fits.
      candidate = std::max(candidate, alignUp(p->end(), b->align));
    }

    if (candidate + b->size > capacity)
      return b;

    b->offset = candidate;
    // "Shared" means this buffer sits where some non-conflicting buffer also
    // sits -- the whole point of packing, and the number worth reporting.
    for (Buffer *p : placed) {
      if (!p->livesWith(*b) && p->offset < b->end() && b->offset < p->end()) {
        ++numShared;
        break;
      }
    }
    placed.push_back(b);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct KeaAllocPass : public mlir::kea::impl::KeaAllocBase<KeaAllocPass> {
  using mlir::kea::impl::KeaAllocBase<KeaAllocPass>::KeaAllocBase;

  void runOnOperation() override;

private:
  /// Gather every `kea.alloc` in `block` into `buffers`, deriving each one's
  /// live range, PC range and required alignment from the SSA def-use chain and
  /// the ops that use it.
  void collect(Block &block, SmallVectorImpl<Buffer> &buffers,
               int64_t &numPositions);

  /// Base alignment `b` needs, derived from the ops that use it (ISA.md §11.1).
  int64_t requiredAlignment(const Buffer &b);

  LogicalResult placeOnChip(MutableArrayRef<Buffer> buffers,
                            int64_t numPositions);
  LogicalResult placeDram(MutableArrayRef<Buffer> buffers,
                          int64_t numPositions);

  /// The correctness property: no two buffers whose live ranges overlap may
  /// have been given overlapping storage. Also re-checks alignment against the
  /// absolute addresses and capacity against the high-water mark.
  LogicalResult verifyPlacement(func::FuncOp func, ArrayRef<Buffer> buffers);
  /// ISA.md §11.1 on the addresses that will actually be encoded.
  LogicalResult verifyAbsoluteAlignment(Block &block);

  void attachReports(func::FuncOp func, ArrayRef<Buffer> buffers,
                     int64_t numPositions);

  /// Emits the whole out-of-capacity story: which buffer failed, what the
  /// theoretical floor is, and who is holding the space.
  void reportOutOfCapacity(const Buffer &failed, ArrayRef<const Buffer *> pool,
                           int64_t capacity, int64_t numPositions);

  /// Filled in by placeDram, published as `kea.dram_layout`.
  struct DramLayout {
    int64_t constOffset = 0, constBytes = 0;
    int64_t ioOffset = 0, ioBytes = 0;
    int64_t scratchOffset = 0, scratchBytes = 0;
    int64_t totalBytes = 0;
  } dram;
};

//===----------------------------------------------------------------------===//
// Collection
//===----------------------------------------------------------------------===//

void KeaAllocPass::collect(Block &block, SmallVectorImpl<Buffer> &buffers,
                           int64_t &numPositions) {
  // Block position of every op, and the instruction index (PC) of every op that
  // becomes an instruction. The two differ because `kea.alloc`, `arith.constant`
  // and `func.return` occupy a block position but emit nothing.
  DenseMap<Operation *, int64_t> position;
  DenseMap<Operation *, int64_t> pc;
  int64_t n = 0, instr = 0;
  for (Operation &op : block) {
    position[&op] = n++;
    if (isInstruction(&op))
      pc[&op] = instr++;
  }
  numPositions = n;

  for (Operation &op : block) {
    auto alloc = dyn_cast<AllocOp>(op);
    if (!alloc)
      continue;

    Buffer b;
    b.alloc = alloc;
    b.name = alloc.getNameAttr().getValue();
    b.role = alloc.getRoleAttr().getValue();
    b.space = llvm::cast<BufferType>(alloc.getResult().getType())
                  .getAddressSpace();
    b.size = alloc.getExtent();
    b.first = position[&op];
    b.last = b.first;

    int64_t firstUse = -1, lastUse = -1;
    for (Operation *user : alloc.getResult().getUsers()) {
      // A user nested inside a region belonging to an op in this block is
      // charged to that op's position. Level 2 has no region-carrying ops
      // today, but under-approximating a live range is the one error here that
      // produces silent aliasing, so it is not left to chance.
      Operation *inBlock = block.findAncestorOpInBlock(*user);
      if (!inBlock)
        continue; // not reachable from this block at all.
      auto it = position.find(inBlock);
      if (it == position.end())
        continue;
      b.last = std::max(b.last, it->second);
      if (firstUse < 0 || it->second < firstUse)
        firstUse = it->second;
      lastUse = std::max(lastUse, it->second);
      auto pcIt = pc.find(inBlock);
      if (pcIt != pc.end()) {
        if (b.firstPc < 0 || pcIt->second < b.firstPc)
          b.firstPc = pcIt->second;
        b.lastPc = std::max(b.lastPc, pcIt->second);
      }
    }

    if (b.space == AddressSpace::DRAM) {
      // A DRAM object holds nothing between its declaration and the first
      // instruction that touches it -- the `kea.alloc` is a symbol, not a
      // store (docs/DIALECT_L2.md §4.1). Starting the range at the first *use*
      // is what lets a deep network reuse one activation buffer across many
      // layers; starting it at the declaration, where -kea-tile hoists every
      // symbol to the top of the function, would make every range overlap
      // every other and defeat the packing entirely.
      b.first = firstUse < 0 ? b.first : firstUse;
      b.last = lastUse < 0 ? b.first : lastUse;
      // Only inter-layer activations are packed. Constants are live for the
      // whole program by definition, and the host binds I/O tensors by name at
      // an address it is told once.
      b.packed = b.role == "activation";
    }
    // On-chip buffers keep docs/DIALECT_L2.md §4.2's range: from the defining
    // `kea.alloc` to its last user in block order. The stamped `live` attribute
    // is deliberately *not* trusted -- ADR-0002's amendment puts -kea-schedule
    // ahead of this pass, and moving an op shifts every block index a live
    // range is expressed in. `refreshLiveRanges()` has already re-derived and
    // re-stamped it from the SSA def-use chain, which is the authority.

    b.align = requiredAlignment(b);
    buffers.push_back(b);
  }
}

int64_t KeaAllocPass::requiredAlignment(const Buffer &b) {
  // The declared alignment is a floor, not the answer: the ops decide.
  int64_t align = std::max<int64_t>(1, b.op().getAlignment());

  auto bump = [&](int64_t a) { align = std::max(align, a); };

  switch (b.space) {
  case AddressSpace::ACC:
    // ISA.md §11.1: MATMUL/DWCONV/VQUANT acc_addr, in words. Mandatory, and
    // the displacements are already multiples of 16, so a 16-word aligned base
    // keeps every absolute address legal.
    bump(::kea::KEA_ALIGN_ACC_WORDS);
    break;
  case AddressSpace::DRAM:
    bump(::kea::KEA_ALIGN_DMA_RECOMMENDED);
    break;
  case AddressSpace::A:
  case AddressSpace::W:
    break;
  }

  // Now the per-user rules. Derived, not assumed: a W buffer used only as a
  // qparam block needs 4-byte alignment, one fed to LOAD_W needs 16 (8 in
  // int4), and a buffer used as both needs the stronger of the two.
  for (Operation *user : b.op().getResult().getUsers()) {
    if (auto ldw = dyn_cast<LoadWOp>(user)) {
      if (ldw.getW() == b.op().getResult())
        bump(ldw.getInt4() ? ::kea::KEA_ALIGN_LOADW_INT4
                           : ::kea::KEA_ALIGN_LOADW_INT8);
    } else if (auto vq = dyn_cast<VquantOp>(user)) {
      if (vq.getQparam() == b.op().getResult())
        bump(::kea::KEA_ALIGN_QPARAM);
    } else if (auto va = dyn_cast<VaddOp>(user)) {
      if (va.getParam() == b.op().getResult())
        bump(::kea::KEA_ALIGN_QPARAM);
    } else if (auto dw = dyn_cast<DwconvOp>(user)) {
      // ISA.md §11.1 marks DWCONV.a_addr advisory. Honour it -- it costs
      // nothing here and real silicon would care.
      if (dw.getA() == b.op().getResult())
        bump(::kea::KEA_ALIGN_DWU);
    }
  }
  return align;
}

//===----------------------------------------------------------------------===//
// Placement
//===----------------------------------------------------------------------===//

void KeaAllocPass::reportOutOfCapacity(const Buffer &failed,
                                       ArrayRef<const Buffer *> pool,
                                       int64_t capacity,
                                       int64_t numPositions) {
  MaxLive peak = computeMaxLive(pool, numPositions);
  const char *unit = spaceUnit(failed.space);

  InFlightDiagnostic diag =
      failed.op().emitOpError()
      << "out of " << spaceName(failed.space) << ": cannot place '"
      << failed.name << "' (" << failed.size << ' ' << unit
      << ", live [" << failed.first << ", " << failed.last << "]) -- "
      << spaceName(failed.space) << " holds " << capacity << ' ' << unit;

  if (peak.bytes > capacity) {
    diag.attachNote(failed.op().getLoc())
        << "peak demand is " << peak.bytes << ' ' << unit
        << " at block position " << peak.position << ", which is "
        << (peak.bytes - capacity) << ' ' << unit << " over the " << capacity
        << ' ' << unit << " capacity. No allocator can fit this: those buffers "
           "hold live data at the same instant. Re-run -kea-tile with smaller "
           "tiles";
  } else {
    diag.attachNote(failed.op().getLoc())
        << "peak demand is " << peak.bytes << ' ' << unit
        << " at block position " << peak.position << ", which does fit in "
        << capacity << ' ' << unit
        << ": the shortfall is fragmentation this packer introduced, not a "
           "property of the program";
  }
  diag.attachNote(failed.op().getLoc())
      << "live at block position " << peak.position << ": "
      << describeLiveAt(pool, peak.position, failed.space);
}

LogicalResult KeaAllocPass::placeOnChip(MutableArrayRef<Buffer> buffers,
                                        int64_t numPositions) {
  for (AddressSpace space : kOnChipSpaces) {
    SmallVector<Buffer *> pool;
    for (Buffer &b : buffers)
      if (b.space == space)
        pool.push_back(&b);
    if (pool.empty())
      continue;

    int64_t capacity = spaceCapacity(space);
    int64_t shared = 0;
    if (Buffer *failed = packGreedy(pool, capacity, shared)) {
      SmallVector<const Buffer *> constPool(pool.begin(), pool.end());
      reportOutOfCapacity(*failed, constPool, capacity, numPositions);
      return failure();
    }
    numShared += shared;
    numBuffers += pool.size();
  }
  return success();
}

LogicalResult KeaAllocPass::placeDram(MutableArrayRef<Buffer> buffers,
                                      int64_t numPositions) {
  // ---- Region 1: constants. Weights, quantization parameter blocks and add
  // parameter records. These are staged into DRAM once before START and are
  // live for the whole program, so there is nothing to pack: they are laid out
  // in program order, which is also the order -kea-emit must write the CONST
  // blob in.
  SmallVector<Buffer *> constants, io, activations;
  for (Buffer &b : buffers) {
    if (b.space != AddressSpace::DRAM)
      continue;
    if (b.role == "weights" || b.role == "qparam" || b.role == "addparam")
      constants.push_back(&b);
    else if (b.role == "input" || b.role == "output")
      io.push_back(&b);
    else if (b.role == "activation")
      activations.push_back(&b);
  }

  int64_t cursor = 0;
  dram.constOffset = 0;
  for (Buffer *b : constants) {
    cursor = alignUp(cursor, b->align);
    b->offset = cursor;
    cursor += b->size;
  }
  dram.constBytes = cursor - dram.constOffset;

  // ---- Region 2: host-visible I/O. The runtime binds these by name and may
  // write an input or read an output at any time relative to the program, so
  // they are never packed and never overlap anything.
  cursor = alignUp(cursor, ::kea::KEA_DRAM_BASE_ALIGN);
  dram.ioOffset = cursor;
  for (Buffer *b : io) {
    cursor = alignUp(cursor, b->align);
    b->offset = cursor;
    cursor += b->size;
  }
  dram.ioBytes = cursor - dram.ioOffset;

  // ---- Region 3: intermediate activations. Same live-range packing as the
  // scratchpads -- this is the region where a 50-layer network reuses a handful
  // of buffers instead of allocating one per layer.
  cursor = alignUp(cursor, ::kea::KEA_DRAM_BASE_ALIGN);
  dram.scratchOffset = cursor;
  if (!activations.empty()) {
    int64_t shared = 0;
    // Pack relative to 0, then rebase onto the region.
    int64_t capacity = static_cast<int64_t>(::kea::KEA_DRAM_BYTES) - cursor;
    if (Buffer *failed = packGreedy(activations, capacity, shared)) {
      SmallVector<const Buffer *> pool(activations.begin(), activations.end());
      reportOutOfCapacity(*failed, pool, capacity, numPositions);
      return failure();
    }
    numShared += shared;
    int64_t high = 0;
    for (Buffer *b : activations) {
      b->offset += dram.scratchOffset;
      high = std::max(high, b->end());
    }
    cursor = high;
  }
  dram.scratchBytes = cursor - dram.scratchOffset;

  dram.totalBytes = alignUp(cursor, ::kea::KEA_DRAM_BASE_ALIGN);
  numBuffers += constants.size() + io.size() + activations.size();
  return success();
}

//===----------------------------------------------------------------------===//
// Verification -- the property that is cheap to check and catastrophic to miss
//===----------------------------------------------------------------------===//

LogicalResult KeaAllocPass::verifyPlacement(func::FuncOp func,
                                            ArrayRef<Buffer> buffers) {
  bool ok = true;

  for (const Buffer &b : buffers) {
    if (b.offset < 0) {
      b.op().emitOpError("was not assigned an address");
      ok = false;
      continue;
    }
    if (b.align > 1 && (b.offset % b.align) != 0) {
      b.op().emitOpError()
          << "base address " << b.offset << " is not a multiple of "
          << b.align << ' ' << spaceUnit(b.space) << ", which "
          << spaceName(b.space) << " requires for this buffer (ISA.md §11.1)";
      ok = false;
    }
    int64_t capacity = spaceCapacity(b.space);
    if (b.end() > capacity) {
      b.op().emitOpError()
          << "occupies [" << b.offset << ", " << b.end() << ") of "
          << spaceName(b.space) << ", which holds " << capacity << ' '
          << spaceUnit(b.space);
      ok = false;
    }
  }

  // The aliasing check, per space. Sort by offset and compare each buffer only
  // against the ones whose storage it actually reaches, so the quadratic term
  // is the number of *storage overlaps*, not the number of buffer pairs.
  for (AddressSpace space :
       {AddressSpace::A, AddressSpace::W, AddressSpace::ACC,
        AddressSpace::DRAM}) {
    SmallVector<const Buffer *> pool;
    for (const Buffer &b : buffers)
      if (b.space == space && b.offset >= 0)
        pool.push_back(&b);
    llvm::sort(pool, [](const Buffer *a, const Buffer *b) {
      if (a->offset != b->offset)
        return a->offset < b->offset;
      return a->name < b->name;
    });

    for (size_t i = 0; i < pool.size(); ++i) {
      for (size_t j = i + 1; j < pool.size(); ++j) {
        const Buffer *a = pool[i], *b = pool[j];
        if (b->offset >= a->end())
          break; // sorted by offset: nothing further can overlap `a`.
        if (!a->livesWith(*b))
          continue; // disjoint lifetimes -- sharing storage is the point.
        a->op().emitOpError()
            << "aliases '" << b->name << "': '" << a->name << "' occupies ["
            << a->offset << ", " << a->end() << ") of " << spaceName(space)
            << " over live range [" << a->first << ", " << a->last << "] and '"
            << b->name << "' occupies [" << b->offset << ", " << b->end()
            << ") over [" << b->first << ", " << b->last
            << "]; the ranges overlap, so the storage must not";
        ok = false;
      }
    }
  }

  if (failed(verifyAbsoluteAlignment(func.getBody().front())))
    ok = false;

  return success(ok);
}

LogicalResult KeaAllocPass::verifyAbsoluteAlignment(Block &block) {
  bool ok = true;

  // The base a displacement into `v` is measured from. Absent means the
  // producing op is not a kea.alloc, which the Level 2 verifiers already
  // reject; nothing to add here.
  auto base = [&](Value v) -> std::optional<int64_t> {
    auto alloc = v.getDefiningOp<AllocOp>();
    if (!alloc)
      return std::nullopt;
    auto attr = alloc->getAttrOfType<IntegerAttr>(kAddr);
    if (!attr)
      return std::nullopt;
    return attr.getInt();
  };

  auto check = [&](Operation *op, Value buf, int64_t disp, int64_t align,
                   StringRef field, const char *unit) {
    auto b = base(buf);
    if (!b)
      return;
    int64_t abs = *b + disp;
    if (align > 1 && (abs % align) != 0) {
      op->emitOpError()
          << "absolute " << field << " = " << *b << " + " << disp << " = "
          << abs << " is not a multiple of " << align << ' ' << unit
          << " (ISA.md §11.1)";
      ok = false;
    }
    if (abs < 0) {
      op->emitOpError() << "absolute " << field << " is negative: " << abs;
      ok = false;
    }
  };

  // ISA.md §11.1, applied to what will actually be encoded. The Level 2 op
  // verifiers checked the displacements; only the sum can be wrong, and only
  // if a base was misaligned.
  for (Operation &op : block) {
    if (auto ldw = dyn_cast<LoadWOp>(op)) {
      check(&op, ldw.getW(), ldw.getWAddr(),
            ldw.getInt4() ? ::kea::KEA_ALIGN_LOADW_INT4
                          : ::kea::KEA_ALIGN_LOADW_INT8,
            "w_addr", "bytes");
    } else if (auto mm = dyn_cast<MmOp>(op)) {
      check(&op, mm.getAcc(), mm.getAccAddr(), ::kea::KEA_ALIGN_ACC_WORDS,
            "acc_addr", "int32 words");
    } else if (auto dw = dyn_cast<DwconvOp>(op)) {
      check(&op, dw.getAcc(), dw.getAccAddr(), ::kea::KEA_ALIGN_ACC_WORDS,
            "acc_addr", "int32 words");
    } else if (auto vq = dyn_cast<VquantOp>(op)) {
      check(&op, vq.getAcc(), vq.getAccAddr(), ::kea::KEA_ALIGN_ACC_WORDS,
            "acc_addr", "int32 words");
      check(&op, vq.getQparam(), vq.getQparamAddr(), ::kea::KEA_ALIGN_QPARAM,
            "qparam_addr", "bytes");
    } else if (auto va = dyn_cast<VaddOp>(op)) {
      check(&op, va.getParam(), va.getParamAddr(), ::kea::KEA_ALIGN_QPARAM,
            "param_addr", "bytes");
    }
  }
  return success(ok);
}

//===----------------------------------------------------------------------===//
// Reporting
//===----------------------------------------------------------------------===//

void KeaAllocPass::attachReports(func::FuncOp func, ArrayRef<Buffer> buffers,
                                 int64_t numPositions) {
  OpBuilder b(func.getContext());

  // ---- kea.spm_map: the KEAF SPM_MAP section / model.map.json `spm_map`,
  // field for field (ARTIFACT_FORMAT.md §5, ASSEMBLY.md §7.5). Nothing at run
  // time consults it -- every address in the stream is already absolute -- it
  // exists so `kea-dis --annotate` can put names to addresses.
  SmallVector<Attribute> entries;
  for (const Buffer &buf : buffers) {
    if (buf.space == AddressSpace::DRAM)
      continue;
    entries.push_back(b.getDictionaryAttr({
        b.getNamedAttr("name", b.getStringAttr(buf.name)),
        b.getNamedAttr("space", b.getStringAttr(spaceName(buf.space))),
        b.getNamedAttr("offset", b.getI64IntegerAttr(buf.offset)),
        b.getNamedAttr("size", b.getI64IntegerAttr(buf.size)),
        // `first_pc` / `last_pc` are instruction indices, not the block
        // positions `live` uses: KeafSpmEntry documents them as "instruction
        // index that touches it", and kea.alloc is not an instruction.
        // -kea-schedule inserts SIGNAL/WAIT and must refresh them.
        b.getNamedAttr("first_pc", b.getI64IntegerAttr(std::max<int64_t>(0, buf.firstPc))),
        b.getNamedAttr("last_pc", b.getI64IntegerAttr(std::max<int64_t>(0, buf.lastPc))),
    }));
  }
  func->setAttr(kSpmMap, b.getArrayAttr(entries));

  // ---- kea.dram_layout: KeafDramLayout, field for field.
  func->setAttr(
      kDramLayout,
      b.getDictionaryAttr({
          b.getNamedAttr("total_bytes", b.getI64IntegerAttr(dram.totalBytes)),
          b.getNamedAttr("const_offset", b.getI64IntegerAttr(dram.constOffset)),
          b.getNamedAttr("const_bytes", b.getI64IntegerAttr(dram.constBytes)),
          b.getNamedAttr("io_offset", b.getI64IntegerAttr(dram.ioOffset)),
          b.getNamedAttr("io_bytes", b.getI64IntegerAttr(dram.ioBytes)),
          b.getNamedAttr("scratch_offset",
                         b.getI64IntegerAttr(dram.scratchOffset)),
          b.getNamedAttr("scratch_bytes",
                         b.getI64IntegerAttr(dram.scratchBytes)),
          b.getNamedAttr("alignment",
                         b.getI64IntegerAttr(::kea::KEA_DRAM_BASE_ALIGN)),
      }));

  // ---- kea.occupancy: what the packing achieved, per space.
  //
  // `peak` is the high-water mark -- the number that has to fit. `maxlive` is
  // the largest total of simultaneously-live data, which is a lower bound on
  // what any allocator could achieve. The difference is fragmentation, i.e.
  // exactly what this packer leaves on the table.
  SmallVector<NamedAttribute> occ;
  // `regionBase` is where the space's packed arena starts. It is 0 for the
  // scratchpads, which are base-0 address spaces, and the scratch region's
  // offset for DRAM -- occupancy is a property of the region, not of how far
  // into the arena the region happens to sit.
  auto report = [&](StringRef key, AddressSpace space, int64_t capacity,
                    int64_t regionBase = 0) {
    SmallVector<const Buffer *> pool;
    int64_t peak = 0, total = 0;
    for (const Buffer &buf : buffers) {
      if (buf.space != space || !buf.packed)
        continue;
      pool.push_back(&buf);
      peak = std::max(peak, buf.end() - regionBase);
      total += buf.size;
    }
    if (pool.empty())
      return;
    MaxLive ml = computeMaxLive(pool, numPositions);
    occ.push_back(b.getNamedAttr(
        key, b.getDictionaryAttr({
                 b.getNamedAttr("buffers", b.getI64IntegerAttr(pool.size())),
                 b.getNamedAttr("peak", b.getI64IntegerAttr(peak)),
                 b.getNamedAttr("capacity", b.getI64IntegerAttr(capacity)),
                 b.getNamedAttr("maxlive", b.getI64IntegerAttr(ml.bytes)),
                 b.getNamedAttr("fragmentation",
                                b.getI64IntegerAttr(peak - ml.bytes)),
                 // What packing bought: the arena a no-sharing allocator needs
                 // divided by the one we produced.
                 b.getNamedAttr("unpacked", b.getI64IntegerAttr(total)),
             })));
  };
  report("spm_a", AddressSpace::A, spaceCapacity(AddressSpace::A));
  report("spm_w", AddressSpace::W, spaceCapacity(AddressSpace::W));
  report("acc", AddressSpace::ACC, spaceCapacity(AddressSpace::ACC));
  report("dram_scratch", AddressSpace::DRAM, dram.scratchBytes,
         dram.scratchOffset);
  func->setAttr(kOccupancy, b.getDictionaryAttr(occ));
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

void KeaAllocPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (func.isExternal() || func.getBody().empty())
    return;

  // Level 2 is a straight-line machine program: one block, and live ranges are
  // positions in it. A second block would make two buffers' intervals
  // incomparable, and silently packing them against each other is exactly the
  // aliasing bug this pass exists to make impossible.
  if (!llvm::hasSingleElement(func.getBody())) {
    bool hasAlloc = false;
    func.walk([&](AllocOp) { hasAlloc = true; });
    if (hasAlloc) {
      func.emitOpError("-kea-alloc needs a straight-line Level 2 program: this "
                       "function has ")
          << func.getBody().getBlocks().size()
          << " blocks, and kea.alloc live ranges are positions in one block";
      return signalPassFailure();
    }
    return;
  }
  Block &block = func.getBody().front();

  // ADR-0002's amendment puts -kea-schedule *before* this pass, precisely so
  // that double buffering shows up here as an ordinary live-range overlap
  // rather than as a concept this pass has to know about. The price is that the
  // `live` attribute -kea-tile stamped is stale by the time it arrives: the
  // scheduler inserts kea.signal / kea.wait and hoists DMA, and every index in
  // a live range is a position in the block those edits renumber. So the SSA
  // def-use chain is re-read and re-stamped before anything is placed.
  refreshLiveRanges(func);

  SmallVector<Buffer> buffers;
  int64_t numPositions = 0;
  collect(block, buffers, numPositions);
  if (buffers.empty())
    return;

  if (verifyOnly) {
    // Read back the placement already in the IR and check it. This is how a
    // hand-corrupted allocation is caught, and it makes the pass a checker for
    // IR this pass did not produce.
    for (Buffer &b : buffers) {
      auto attr = b.alloc->getAttrOfType<IntegerAttr>(kAddr);
      if (!attr) {
        b.alloc.emitOpError("has no ")
            << kAddr << ": -kea-alloc=verify-only=true verifies an existing "
                        "placement, it does not create one";
        return signalPassFailure();
      }
      b.offset = attr.getInt();
    }
    if (failed(verifyPlacement(func, buffers)))
      return signalPassFailure();
    return;
  }

  if (failed(placeOnChip(buffers, numPositions)) ||
      failed(placeDram(buffers, numPositions)))
    return signalPassFailure();

  // Stamp the answer. `addr` is a declared optional attribute on `kea.alloc`
  // (KeaMachineOps.td), so the op verifier checks the base against the
  // buffer's alignment and capacity as soon as it is set.
  OpBuilder b(func.getContext());
  for (Buffer &buf : buffers)
    buf.alloc->setAttr(kAddr, b.getI64IntegerAttr(buf.offset));

  if (verifyPacking && failed(verifyPlacement(func, buffers)))
    return signalPassFailure();

  if (reportMap)
    attachReports(func, buffers, numPositions);
}

} // namespace
