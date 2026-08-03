//===- EmitKasm.cpp - Level 2 MLIR -> .kasm + .bin + .map.json --*- C++ -*-===//
//
// The `-kea-emit` backend. ADR-0001 rule 1: docs/ASSEMBLY.md is owned by the
// assembler and this file conforms to it. Every operand name and order below
// is `kea::keaOpInfo()`'s, which is also `runtime/src/op_fields.cpp`'s -- so
// the mapping from a Level 2 op to a line of assembly is mechanical and can be
// diffed against isa.h side by side.
//
// Three things happen here that are not pure transcription, and each is called
// out where it happens:
//
//   1. `instr.X_addr = addr(X) + X_addr`  -- the (buffer, displacement) pairs
//      of DIALECT_L2.md §1.1(a) become the absolute integers of ASSEMBLY.md
//      §1.1. DRAM stays symbolic (ADR-0001 rule 3).
//   2. Cross-unit `SIGNAL`/`WAIT` insertion, when the IR has none. See the
//      block comment above `insertSync()`: this is NOT scheduling.
//   3. The model.map.json I/O descriptors, which need shape and quantization
//      facts the Level 2 IR does not carry. Everything derived rather than
//      given is derived from the instruction stream and documented.
//
// See docs/CODEGEN.md.
//
//===----------------------------------------------------------------------===//

#include "kea/Target/Kasm/EmitKasm.h"

#include "kea/Dialect/KeaAttrs.h"
#include "kea/Dialect/KeaMachineOps.h"
#include "kea/Dialect/KeaTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "kea/hw_config.h"
#include "kea/isa.h"
#include "kea/keaf.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <memory>
#include <set>

using namespace mlir;
using namespace mlir::kea;

namespace {

//===----------------------------------------------------------------------===//
// Small formatting helpers
//===----------------------------------------------------------------------===//

std::string num(int64_t v) { return std::to_string(v); }

/// ASSEMBLY.md §1.2: an ACC stride MUST carry a `w`, and a byte stride must
/// not. The suffix is added exactly where the value came out of an ACC buffer.
std::string accWords(int64_t v) { return std::to_string(v) + "w"; }

std::string jsonEscape(StringRef s) {
  std::string out;
  for (char c : s) {
    switch (c) {
    case '"': out += "\\\""; break;
    case '\\': out += "\\\\"; break;
    case '\n': out += "\\n"; break;
    case '\t': out += "\\t"; break;
    case '\r': out += "\\r"; break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char buf[8];
        snprintf(buf, sizeof(buf), "\\u%04x", static_cast<int>(c));
        out += buf;
      } else {
        out += c;
      }
    }
  }
  return out;
}

/// The map's `scale` is the only float in the whole backend; print it so it
/// reads back bit-identically.
std::string jsonNumber(double v) {
  char buf[40];
  snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

//===----------------------------------------------------------------------===//
// Units
//===----------------------------------------------------------------------===//

enum Unit { MXU = 0, DWU = 1, VPU = 2, DMA0 = 3, DMA1 = 4, CTRL = 5, NUNITS = 6 };

const char *unitName(int u) {
  static const char *kNames[NUNITS] = {"MXU",  "DWU",  "VPU",
                                       "DMA0", "DMA1", "CTRL"};
  return kNames[u];
}

std::optional<int> parseUnit(StringRef s) {
  for (int u = 0; u < NUNITS; ++u)
    if (s == unitName(u))
      return u;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Storage intervals
//===----------------------------------------------------------------------===//
//
// Dependencies are tracked over STORAGE, not over SSA values, and that is not
// a detail. `-kea-alloc` gives two buffers the same address whenever their
// block-order live ranges are disjoint (MEMORY_PLANNING.md §2.2) -- layer 0's
// ACC region and layer 1's ACC region really are the same 2048 words. Keying
// on the `kea.alloc` result would miss that aliasing and emit a program whose
// second layer starts overwriting the accumulator the first layer's VQUANT is
// still reading.
//
struct Interval {
  int space = 0; // AddressSpace as an int, DRAM included
  int64_t lo = 0, hi = 0;
  bool overlaps(const Interval &o) const {
    return space == o.space && lo < o.hi && o.lo < hi;
  }
};

//===----------------------------------------------------------------------===//
// One emitted instruction
//===----------------------------------------------------------------------===//

struct Line {
  int unit = CTRL;
  std::string mnem;
  std::string fields;
  std::string label;   ///< printed on its own line just above
  std::string comment; ///< `--annotate` only
  Operation *op = nullptr;
};

/// Field-list builder. Keeps the `keaOpInfo()` order honest by construction.
struct Fields {
  std::string s;
  void add(StringRef name, const std::string &value) {
    if (!s.empty())
      s += ", ";
    s += name.str();
    s += '=';
    s += value;
  }
};

//===----------------------------------------------------------------------===//
// Buffers
//===----------------------------------------------------------------------===//

struct BufInfo {
  AllocOp alloc;
  StringRef name;
  StringRef role;
  AddressSpace space = AddressSpace::DRAM;
  int64_t base = 0;
  int64_t extent = 0;
  /// Instruction indices, filled after emission (for `spm_map`).
  int64_t firstPc = -1, lastPc = -1;
};

bool isConstRole(StringRef role) {
  return role == "weights" || role == "qparam" || role == "addparam";
}

//===----------------------------------------------------------------------===//
// The emitter
//===----------------------------------------------------------------------===//

class Emitter {
public:
  Emitter(func::FuncOp fn, const EmitOptions &opts) : fn(fn), opts(opts) {}
  LogicalResult run(EmitResult &result);

private:
  func::FuncOp fn;
  const EmitOptions &opts;

  llvm::DenseMap<Value, BufInfo *> bufOf;
  std::vector<std::unique_ptr<BufInfo>> bufs;
  SmallVector<Line> lines;

  struct Deps {
    SmallVector<Interval, 4> reads, writes;
  };
  SmallVector<Deps> deps;

  std::vector<std::pair<int, std::string>> events;
  std::map<int64_t, std::string> regions;

  LogicalResult collectBuffers();
  LogicalResult emitInstructions();
  LogicalResult emitOne(Operation *op, Line &line, Deps &d);
  LogicalResult insertSync(int64_t &numInserted);
  void assignRegionUnits();

  std::string renderKasm() const;
  LogicalResult renderMap(std::string &out);
  LogicalResult buildConstBlob(std::vector<uint8_t> &out,
                               std::string &listing);

  BufInfo *info(Value v) const {
    auto it = bufOf.find(v);
    return it == bufOf.end() ? nullptr : it->second;
  }

  Interval span(Value v) const {
    BufInfo *b = info(v);
    if (!b)
      return {};
    return {static_cast<int>(b->space), b->base, b->base + b->extent};
  }

  /// `instr.X_addr = addr(X) + X_addr` (DIALECT_L2.md §1.1(a)), with the space
  /// prefix ASSEMBLY.md §1.1/§1.2 requires. The offset is already in the
  /// space's own addressing unit, so there is no conversion here and there
  /// must never be one.
  std::string spmAddr(Value buf, int64_t disp) const {
    BufInfo *b = info(buf);
    int64_t abs = (b ? b->base : 0) + disp;
    switch (b ? b->space : AddressSpace::A) {
    case AddressSpace::A: return "a:" + num(abs);
    case AddressSpace::W: return "w:" + num(abs);
    case AddressSpace::ACC: return "acc:" + num(abs);
    default: return num(abs);
    }
  }

  /// ADR-0001 rule 3: DRAM stays symbolic. The assembler adds the
  /// displacement to the symbol's `offset` in model.map.json.
  std::string dramAddr(Value buf, int64_t disp) const {
    BufInfo *b = info(buf);
    std::string s = "@" + (b ? b->name.str() : std::string("unknown"));
    if (disp > 0)
      s += "+" + num(disp);
    else if (disp < 0)
      s += "-" + num(-disp);
    return s;
  }

  static const char *spaceFlag(AddressSpace as) {
    return as == AddressSpace::W ? "SPM_W" : "SPM_A";
  }
  AddressSpace spaceOf(Value v) const {
    return llvm::cast<BufferType>(v.getType()).getAddressSpace();
  }

  std::optional<int64_t> inputZeroPoint(BufInfo &b) const;
  std::optional<int64_t> outputZeroPoint(BufInfo &b) const;
  bool inferShape(BufInfo &b, SmallVectorImpl<int64_t> &shape) const;
};

//===----------------------------------------------------------------------===//

LogicalResult Emitter::collectBuffers() {
  Block &block = fn.getBody().front();
  llvm::StringSet<> seen;
  for (auto alloc : block.getOps<AllocOp>()) {
    auto b = std::make_unique<BufInfo>();
    b->alloc = alloc;
    b->name = alloc.getName();
    b->role = alloc.getRole();
    b->space = alloc.getSpace();
    b->extent = alloc.getExtent();
    auto a = alloc.getAddr();
    if (!a)
      return alloc.emitOpError(
          "has no `addr`: -kea-emit turns (buffer, displacement) pairs into "
          "absolute addresses and needs the base -kea-alloc assigns");
    b->base = *a;
    if (!seen.insert(b->name).second)
      return alloc.emitOpError("duplicate buffer name \"")
             << b->name
             << "\"; a DRAM name is also the .kasm symbol and must be unique "
                "(docs/DIALECT_L2.md §4.1)";
    bufOf[alloc.getResult()] = b.get();
    bufs.push_back(std::move(b));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// One op -> one line. docs/ASSEMBLY.md §5, in keaOpInfo() field order.
//===----------------------------------------------------------------------===//

LogicalResult Emitter::emitOne(Operation *op, Line &line, Deps &d) {
  Fields f;
  line.op = op;

  auto unitAttr = [&](int fallback) {
    if (auto u = op->getAttrOfType<StringAttr>("unit"))
      if (auto parsed = parseUnit(u.getValue()))
        return *parsed;
    return fallback;
  };
  auto dmaFields = [&](auto dma) {
    f.add("spm_space", spaceFlag(spaceOf(dma.getSpm())));
    f.add("dram_addr", dramAddr(dma.getDram(), dma.getDramAddr()));
    f.add("spm_addr", spmAddr(dma.getSpm(), dma.getSpmAddr()));
    f.add("len0", num(dma.getLen0()));
    f.add("n1", num(dma.getN1()));
    f.add("n2", num(dma.getN2()));
    f.add("dram_s1", num(dma.getDramS1()));
    f.add("dram_s2", num(dma.getDramS2()));
    f.add("spm_s1", num(dma.getSpmS1()));
    f.add("spm_s2", num(dma.getSpmS2()));
  };

  if (auto dma = llvm::dyn_cast<DmaLoadOp>(op)) {
    line.unit = unitAttr(DMA0);
    line.mnem = "DMA_LD";
    dmaFields(dma);
    d.reads.push_back(span(dma.getDram()));
    d.writes.push_back(span(dma.getSpm()));
  } else if (auto dma = llvm::dyn_cast<DmaStoreOp>(op)) {
    line.unit = unitAttr(DMA0);
    line.mnem = "DMA_ST";
    dmaFields(dma);
    d.reads.push_back(span(dma.getSpm()));
    d.writes.push_back(span(dma.getDram()));
  } else if (auto lw = llvm::dyn_cast<LoadWOp>(op)) {
    line.unit = MXU;
    line.mnem = "LOAD_W";
    f.add("w_addr", spmAddr(lw.getW(), lw.getWAddr()));
    f.add("w_row_stride", num(lw.getWRowStride()));
    f.add("k_rows", num(lw.getKRows()));
    f.add("n_cols", num(lw.getNCols()));
    f.add("bank", num(lw.getBank()));
    f.add("dtype", lw.getInt4() ? "int4" : "int8");
    d.reads.push_back(span(lw.getW()));
  } else if (auto mm = llvm::dyn_cast<MmOp>(op)) {
    line.unit = MXU;
    line.mnem = "MATMUL";
    f.add("a_addr", spmAddr(mm.getA(), mm.getAAddr()));
    f.add("a_inner_stride", num(mm.getAInnerStride()));
    f.add("a_outer_stride", num(mm.getAOuterStride()));
    f.add("m_inner", num(mm.getMInner()));
    f.add("m_outer", num(mm.getMOuter()));
    f.add("acc_addr", spmAddr(mm.getAcc(), mm.getAccAddr()));
    f.add("acc_inner_stride", accWords(mm.getAccInnerStride()));
    f.add("acc_outer_stride", accWords(mm.getAccOuterStride()));
    f.add("bank", num(mm.getBank()));
    f.add("acc_mode", mm.getAccumulate() ? "accumulate" : "overwrite");
    f.add("dtype", mm.getInt4() ? "int4" : "int8");
    d.reads.push_back(span(mm.getA()));
    d.writes.push_back(span(mm.getAcc()));
  } else if (auto dw = llvm::dyn_cast<DwconvOp>(op)) {
    line.unit = DWU;
    line.mnem = "DWCONV";
    f.add("a_addr", spmAddr(dw.getA(), dw.getAAddr()));
    f.add("w_addr", spmAddr(dw.getW(), dw.getWAddr()));
    f.add("acc_addr", spmAddr(dw.getAcc(), dw.getAccAddr()));
    f.add("out_h", num(dw.getOutH()));
    f.add("out_w", num(dw.getOutW()));
    f.add("channels", num(dw.getChannels()));
    f.add("a_row_stride", num(dw.getARowStride()));
    f.add("a_pix_stride", num(dw.getAPixStride()));
    f.add("kernel", num(dw.getKernel()));
    f.add("stride", num(dw.getStride()));
    f.add("acc_mode", dw.getAccumulate() ? "accumulate" : "overwrite");
    d.reads.push_back(span(dw.getA()));
    d.reads.push_back(span(dw.getW()));
    d.writes.push_back(span(dw.getAcc()));
  } else if (auto vq = llvm::dyn_cast<VquantOp>(op)) {
    line.unit = VPU;
    line.mnem = "VQUANT";
    f.add("acc_addr", spmAddr(vq.getAcc(), vq.getAccAddr()));
    f.add("out_addr", spmAddr(vq.getOut(), vq.getOutAddr()));
    f.add("qparam_addr", spmAddr(vq.getQparam(), vq.getQparamAddr()));
    f.add("num_pixels", num(vq.getNumPixels()));
    f.add("channels", num(vq.getChannels()));
    f.add("acc_pix_stride", accWords(vq.getAccPixStride()));
    f.add("out_pix_stride", num(vq.getOutPixStride()));
    f.add("out_zp", num(vq.getOutZp()));
    f.add("clamp_lo", num(vq.getClampLo()));
    f.add("clamp_hi", num(vq.getClampHi()));
    f.add("dtype", vq.getInt4() ? "int4" : "int8");
    d.reads.push_back(span(vq.getAcc()));
    d.reads.push_back(span(vq.getQparam()));
    d.writes.push_back(span(vq.getOut()));
  } else if (auto va = llvm::dyn_cast<VaddOp>(op)) {
    line.unit = VPU;
    line.mnem = "VADD";
    f.add("a_addr", spmAddr(va.getA(), va.getAAddr()));
    f.add("b_addr", spmAddr(va.getB(), va.getBAddr()));
    f.add("out_addr", spmAddr(va.getOut(), va.getOutAddr()));
    f.add("param_addr", spmAddr(va.getParam(), va.getParamAddr()));
    f.add("num_elems", num(va.getNumElems()));
    f.add("clamp_lo", num(va.getClampLo()));
    f.add("clamp_hi", num(va.getClampHi()));
    d.reads.push_back(span(va.getA()));
    d.reads.push_back(span(va.getB()));
    d.reads.push_back(span(va.getParam()));
    d.writes.push_back(span(va.getOut()));
  } else if (auto vp = llvm::dyn_cast<VpoolOp>(op)) {
    line.unit = VPU;
    line.mnem = "VPOOL";
    f.add("mode", vp.getAvg() ? "avg" : "max");
    f.add("in_addr", spmAddr(vp.getIn(), vp.getInAddr()));
    f.add("out_addr", spmAddr(vp.getOut(), vp.getOutAddr()));
    f.add("out_h", num(vp.getOutH()));
    f.add("out_w", num(vp.getOutW()));
    f.add("channels", num(vp.getChannels()));
    f.add("kh", num(vp.getKh()));
    f.add("kw", num(vp.getKw()));
    f.add("stride_h", num(vp.getStrideH()));
    f.add("stride_w", num(vp.getStrideW()));
    f.add("in_row_stride", num(vp.getInRowStride()));
    f.add("out_row_stride", num(vp.getOutRowStride()));
    d.reads.push_back(span(vp.getIn()));
    d.writes.push_back(span(vp.getOut()));
  } else if (auto vc = llvm::dyn_cast<VcopyOp>(op)) {
    line.unit = VPU;
    line.mnem = "VCOPY";
    f.add("mode", vc.getFill() ? "fill" : "copy");
    if (vc.getSrc()) {
      f.add("src_space", spaceFlag(spaceOf(vc.getSrc())));
      f.add("dst_space", spaceFlag(spaceOf(vc.getDst())));
      f.add("src_addr", spmAddr(vc.getSrc(), vc.getSrcAddr()));
      f.add("dst_addr", spmAddr(vc.getDst(), vc.getDstAddr()));
      f.add("row_bytes", num(vc.getRowBytes()));
      f.add("rows", num(vc.getRows()));
      f.add("src_row_stride", num(vc.getSrcRowStride()));
      d.reads.push_back(span(vc.getSrc()));
    } else {
      // ASSEMBLY.md §5.5: in fill mode the source fields are ignored but still
      // occupy encoding bits that have to be deterministic. Emit exactly the
      // triple the spec names.
      f.add("src_space", "SPM_A");
      f.add("dst_space", spaceFlag(spaceOf(vc.getDst())));
      f.add("src_addr", "a:0");
      f.add("dst_addr", spmAddr(vc.getDst(), vc.getDstAddr()));
      f.add("row_bytes", num(vc.getRowBytes()));
      f.add("rows", num(vc.getRows()));
      f.add("src_row_stride", "0");
    }
    f.add("dst_row_stride", num(vc.getDstRowStride()));
    f.add("fill_value", num(vc.getFillValue()));
    d.writes.push_back(span(vc.getDst()));
  } else if (auto sg = llvm::dyn_cast<SignalOp>(op)) {
    line.unit = unitAttr(CTRL);
    line.mnem = "SIGNAL";
    f.add("event", num(sg.getEvent()));
    f.add("inc", num(sg.getValue()));
  } else if (auto wt = llvm::dyn_cast<WaitOp>(op)) {
    line.unit = unitAttr(CTRL);
    line.mnem = "WAIT";
    f.add("event", num(wt.getEvent()));
    f.add("threshold", num(wt.getValue()));
  } else if (auto tr = llvm::dyn_cast<TraceOp>(op)) {
    line.unit = unitAttr(CTRL);
    line.mnem = "TRACE";
    f.add("kind", tr.getKind().str());
    f.add("tag", num(tr.getTag()));
    f.add("payload", num(tr.getPayload()));
  } else if (auto ht = llvm::dyn_cast<HaltOp>(op)) {
    line.unit = CTRL;
    line.mnem = "HALT";
    f.add("exit_code", num(ht.getExitCode()));
  } else {
    return op->emitOpError("has no KEA-1 instruction; -kea-emit expects a "
                           "pure Level 2 function (docs/DIALECT_L2.md §1)");
  }

  line.fields = f.s;
  return success();
}

LogicalResult Emitter::emitInstructions() {
  Block &block = fn.getBody().front();
  bool sawHalt = false;
  for (Operation &op : block) {
    if (llvm::isa<AllocOp>(op) || llvm::isa<func::ReturnOp>(op))
      continue;
    // `-kea-tile` leaves the weight/bias `arith.constant`s behind as the
    // `source` operands of the DRAM allocs. They are data, not instructions.
    if (op.hasTrait<OpTrait::ConstantLike>())
      continue;
    if (op.getName().getDialectNamespace() != "kea")
      return op.emitOpError("is not a Level 2 kea op; -kea-emit needs a "
                            "function that is already a machine program");
    if (sawHalt)
      return op.emitOpError("follows kea.halt; ISA.md §5.2 makes HALT the last "
                            "instruction and anything after it unreachable");
    Line line;
    Deps d;
    if (failed(emitOne(&op, line, d)))
      return failure();
    if (line.mnem == "HALT")
      sawHalt = true;
    lines.push_back(std::move(line));
    deps.push_back(std::move(d));
  }
  if (!sawHalt)
    return fn.emitOpError("has no kea.halt; ISA.md §5.2 requires exactly one "
                          "HALT and it must be last");
  return success();
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//
//
// `-kea-tile` emits a correct SEQUENTIAL program: right if executed one
// instruction at a time (DIALECT_L2.md §4.6). KEA-1 does not execute one
// instruction at a time. Five queues run concurrently and the only ordering
// between them is the 32 counting semaphores (ISA.md §5.3), so a program with
// no `SIGNAL`/`WAIT` is a set of data races and running it proves nothing.
//
// When the IR carries no semaphores, this inserts exactly the cross-unit edges
// the sequential program's STORAGE dependencies imply -- RAW, WAW and WAR,
// over absolute address ranges rather than SSA values, because the allocator
// aliases buffers whose live ranges are disjoint.
//
// It is NOT a scheduler. Nothing is reordered, hoisted, software pipelined or
// double buffered; the instruction order is exactly the IR's. When
// `-kea-schedule` runs, it owns this and the emitter passes its semaphores
// through untouched (`--sync=none`, or `auto` on IR that has any).
//
// The counting scheme is one event per ordered unit pair `(producer,
// consumer)`. Signals on a pair are emitted in producer order; a unit's queue
// is in-order and `SIGNAL` is a local drain barrier, so the event's value is
// (completed producers on that pair) - (counts already consumed). A consumer
// that needs producer number `r` therefore waits for `r - consumed` counts,
// which is 0 -- no instruction at all -- when an earlier `WAIT` on the same
// queue already covered it. Rule D (ISA.md §5.5) holds by construction: every
// supplying `SIGNAL` sits at a smaller stream position than the `WAIT`,
// because every producer does.
//
//===----------------------------------------------------------------------===//

LogicalResult Emitter::insertSync(int64_t &numInserted) {
  const size_t n = lines.size();

  struct Access {
    Interval iv;
    size_t idx;
    bool isWrite;
  };
  SmallVector<Access> history;
  SmallVector<SmallVector<int64_t, 4>> depsOf(n);

  for (size_t i = 0; i < n; ++i) {
    // The latest conflicting access per other unit is enough: that unit's
    // queue is in-order, so waiting for its last conflicting instruction
    // implies every earlier one retired too.
    int64_t need[NUNITS];
    for (int u = 0; u < NUNITS; ++u)
      need[u] = -1;

    auto conflict = [&](const Interval &iv, bool write) {
      for (const Access &a : history) {
        if (!write && !a.isWrite)
          continue; // read-read is not a dependency
        if (!a.iv.overlaps(iv))
          continue;
        int u = lines[a.idx].unit;
        if (u == lines[i].unit)
          continue; // same in-order queue (ISA.md §5.3)
        need[u] = std::max(need[u], static_cast<int64_t>(a.idx));
      }
    };
    for (const Interval &iv : deps[i].reads)
      conflict(iv, /*write=*/false);
    for (const Interval &iv : deps[i].writes)
      conflict(iv, /*write=*/true);

    for (int u = 0; u < NUNITS; ++u)
      if (need[u] >= 0)
        depsOf[i].push_back(need[u]);

    // A later access to the same interval from the same unit shadows an
    // earlier one completely -- that unit's queue is in-order -- so drop the
    // earlier record. Without this, `history` grows with every access and the
    // sweep is quadratic in instructions; with it, it is quadratic in the
    // number of distinct live intervals, which is the buffer count.
    auto push = [&](const Interval &iv, bool isWrite) {
      llvm::erase_if(history, [&](const Access &a) {
        return a.iv.lo == iv.lo && a.iv.hi == iv.hi && a.iv.space == iv.space &&
               a.isWrite == isWrite && lines[a.idx].unit == lines[i].unit;
      });
      history.push_back({iv, i, isWrite});
    };
    for (const Interval &iv : deps[i].reads)
      push(iv, false);
    for (const Interval &iv : deps[i].writes)
      push(iv, true);
  }

  // Rank the producers of each (producer unit, consumer unit) pair, and give
  // each pair an event.
  //
  // Allocation starts above every event id the IR already uses, so
  // `--sync=insert` over a partially scheduled program cannot silently share a
  // counter with the scheduler's own -- which would corrupt both.
  int firstEvent = 0;
  for (const Line &l : lines)
    if (auto sg = llvm::dyn_cast_or_null<SignalOp>(l.op))
      firstEvent = std::max<int>(firstEvent, sg.getEvent() + 1);
    else if (auto wt = llvm::dyn_cast_or_null<WaitOp>(l.op))
      firstEvent = std::max<int>(firstEvent, wt.getEvent() + 1);

  using Pair = std::pair<int, int>;
  std::map<Pair, std::vector<int64_t>> pairProducers;
  std::map<Pair, int> eventOf;
  for (size_t i = 0; i < n; ++i)
    for (int64_t j : depsOf[i]) {
      Pair p{lines[j].unit, lines[i].unit};
      auto &v = pairProducers[p];
      if (!llvm::is_contained(v, j))
        v.push_back(j);
      if (!eventOf.count(p)) {
        int id = firstEvent + static_cast<int>(eventOf.size());
        if (id >= ::kea::KEA_NUM_EVENTS)
          return fn.emitOpError("needs more than ")
                 << ::kea::KEA_NUM_EVENTS
                 << " semaphores to order this program";
        eventOf[p] = id;
        events.emplace_back(id, std::string(unitName(p.first)) + "_to_" +
                                    unitName(p.second));
      }
    }
  if (pairProducers.empty())
    return success();
  for (auto &kv : pairProducers)
    llvm::sort(kv.second);

  std::map<Pair, std::map<int64_t, int64_t>> rank;
  for (auto &kv : pairProducers)
    for (size_t k = 0; k < kv.second.size(); ++k)
      rank[kv.first][kv.second[k]] = static_cast<int64_t>(k) + 1;

  // Which consuming units each producer must be visible to.
  std::map<int64_t, std::set<int>> consumersOf;
  for (size_t i = 0; i < n; ++i)
    for (int64_t j : depsOf[i])
      consumersOf[j].insert(lines[i].unit);

  std::map<Pair, int64_t> consumed;
  SmallVector<Line> out;
  out.reserve(n * 2);
  for (size_t i = 0; i < n; ++i) {
    std::map<int, int64_t> needByUnit;
    for (int64_t j : depsOf[i]) {
      int pu = lines[j].unit;
      needByUnit[pu] = std::max(needByUnit[pu], rank[{pu, lines[i].unit}][j]);
    }
    for (auto &kv : needByUnit) {
      Pair p{kv.first, lines[i].unit};
      int64_t thr = kv.second - consumed[p];
      if (thr <= 0)
        continue;
      consumed[p] = kv.second;
      Line w;
      w.unit = lines[i].unit;
      w.mnem = "WAIT";
      w.fields = "event=" + num(eventOf[p]) + ", threshold=" + num(thr);
      out.push_back(std::move(w));
      ++numInserted;
    }
    out.push_back(lines[i]);
    for (int cu : consumersOf[static_cast<int64_t>(i)]) {
      Pair p{lines[i].unit, cu};
      Line s;
      s.unit = lines[i].unit;
      s.mnem = "SIGNAL";
      s.fields = "event=" + num(eventOf[p]) + ", inc=1";
      out.push_back(std::move(s));
      ++numInserted;
    }
  }
  lines = std::move(out);
  return success();
}

//===----------------------------------------------------------------------===//
// Regions and labels
//===----------------------------------------------------------------------===//

void Emitter::assignRegionUnits() {
  // A `TRACE begin`/`end` pair must sit on the SAME queue or the region never
  // closes: regions nest per (unit, tag) (ISA.md §5.4). Put both on the unit
  // that does the most interesting work inside the region, which is what makes
  // `kea-sim`'s per-region roofline mean something.
  static const int kPriority[] = {MXU, DWU, VPU, DMA0, DMA1};

  std::map<int64_t, std::set<int>> unitsInTag;
  SmallVector<int64_t> openTags;
  for (Line &l : lines) {
    auto tr = llvm::dyn_cast_or_null<TraceOp>(l.op);
    if (tr && tr.getKind() == "begin") {
      openTags.push_back(tr.getTag());
      continue;
    }
    if (tr && tr.getKind() == "end") {
      if (!openTags.empty())
        openTags.pop_back();
      continue;
    }
    if (l.unit == CTRL)
      continue;
    for (int64_t t : openTags)
      unitsInTag[t].insert(l.unit);
  }

  std::map<int64_t, int> unitForTag;
  for (Line &l : lines) {
    auto tr = llvm::dyn_cast_or_null<TraceOp>(l.op);
    if (!tr || tr->hasAttr("unit"))
      continue;
    int64_t tag = tr.getTag();
    auto it = unitForTag.find(tag);
    if (it == unitForTag.end()) {
      int chosen = CTRL;
      for (int u : kPriority)
        if (unitsInTag[tag].count(u)) {
          chosen = u;
          break;
        }
      it = unitForTag.emplace(tag, chosen).first;
    }
    l.unit = it->second;
  }

  // `.region` names and a label per region. The name is the layer prefix
  // `-kea-tile` gave that layer's buffers, so a `kea-sim` per-region line is
  // traceable back to the graph.
  std::string prefix = (fn.getName() + ".").str();
  for (Line &l : lines) {
    auto tr = llvm::dyn_cast_or_null<TraceOp>(l.op);
    if (!tr || tr.getKind() != "begin")
      continue;
    int64_t tag = tr.getTag();
    if (regions.count(tag))
      continue;
    std::string name = "region" + num(tag);
    std::string layerPrefix = prefix + num(tag) + ".";
    for (auto &b : bufs)
      if (b->role != "scratch" && StringRef(b->name).starts_with(layerPrefix)) {
        name = (fn.getName() + "." + num(tag)).str();
        break;
      }
    regions[tag] = name;
    if (opts.labels)
      l.label = "layer" + num(tag);
  }
}

//===----------------------------------------------------------------------===//
// Rendering
//===----------------------------------------------------------------------===//

std::string Emitter::renderKasm() const {
  std::string s;
  llvm::raw_string_ostream os(s);

  os << ".arch \"" << ::kea::KEA_ARCH_NAME << "\"\n";
  os << ".isa_revision " << ::kea::KEA_ISA_REVISION << "\n";
  os << ".entry 0\n\n";

  if (!events.empty() || !regions.empty()) {
    for (auto &e : events)
      os << ".event " << e.first << ", \"" << jsonEscape(e.second) << "\"\n";
    for (auto &r : regions)
      os << ".region " << r.first << ", \"" << jsonEscape(r.second) << "\"\n";
    os << "\n";
  }

  int64_t pc = 0;
  for (const Line &l : lines) {
    if (!l.label.empty())
      os << l.label << ":\n";
    // Canonical form (ASSEMBLY.md §4): two spaces, unit padded to 4, two
    // spaces, mnemonic padded to 6, two spaces, then the fields.
    os << "  " << llvm::left_justify(unitName(l.unit), 4) << "  "
       << llvm::left_justify(l.mnem, 6) << "  " << l.fields;
    if (opts.annotate) {
      os << "  ; pc=" << pc;
      if (!l.comment.empty())
        os << ' ' << l.comment;
    }
    os << "\n";
    ++pc;
  }
  return s;
}

//===----------------------------------------------------------------------===//
// model.map.json
//===----------------------------------------------------------------------===//

std::optional<int64_t> Emitter::inputZeroPoint(BufInfo &b) const {
  // ISA.md §8.4(a): the SPM_A tile an activation is loaded into is VCOPY-filled
  // with the *input zero point* first, so the halo dequantizes to real 0. That
  // fill value is this tensor's zero point, read off the instruction stream
  // rather than assumed. `--io-quant` overrides it.
  std::optional<int64_t> zp;
  for (const Line &l : lines) {
    auto ld = llvm::dyn_cast_or_null<DmaLoadOp>(l.op);
    if (!ld || info(ld.getDram()) != &b)
      continue;
    Value tile = ld.getSpm();
    for (Operation *user : tile.getUsers()) {
      auto vc = llvm::dyn_cast<VcopyOp>(user);
      if (!vc || !vc.getFill() || vc.getDst() != tile)
        continue;
      if (zp && *zp != vc.getFillValue())
        return std::nullopt; // disagreement: say nothing rather than guess
      zp = vc.getFillValue();
    }
  }
  return zp;
}

std::optional<int64_t> Emitter::outputZeroPoint(BufInfo &b) const {
  std::optional<int64_t> zp;
  auto note = [&](int64_t v, bool &bad) {
    if (zp && *zp != v)
      bad = true;
    zp = v;
  };
  bool bad = false;
  for (const Line &l : lines) {
    auto st = llvm::dyn_cast_or_null<DmaStoreOp>(l.op);
    if (!st || info(st.getDram()) != &b)
      continue;
    Value tile = st.getSpm();
    for (Operation *user : tile.getUsers()) {
      if (auto vq = llvm::dyn_cast<VquantOp>(user)) {
        if (vq.getOut() == tile)
          note(vq.getOutZp(), bad);
      } else if (auto va = llvm::dyn_cast<VaddOp>(user)) {
        // VADD's output zero point is `o_zp`, the last field of the
        // KeaAddParam record it reads (docs/DIALECT_L2.md §4.4).
        if (va.getOut() != tile)
          continue;
        for (Operation *pu : va.getParam().getUsers())
          if (auto ld = llvm::dyn_cast<DmaLoadOp>(pu))
            if (ld.getSpm() == va.getParam())
              if (BufInfo *src = info(ld.getDram()))
                if (auto ap = src->alloc.getAddParam())
                  if (ap->size() == 9)
                    note((*ap)[8], bad);
      }
    }
  }
  return bad ? std::nullopt : zp;
}

bool Emitter::inferShape(BufInfo &b, SmallVectorImpl<int64_t> &shape) const {
  // A whole-tensor store is `len0` contiguous channel bytes, `n1` pixels of a
  // row and `n2` rows -- exactly the descriptor `-kea-tile` emits when one tile
  // covers the tensor. Anything else (a tiled output, several stores) leaves
  // the shape unknown, and the map then says FLAT rather than inventing one.
  DmaStoreOp only;
  int count = 0;
  for (const Line &l : lines)
    if (auto st = llvm::dyn_cast_or_null<DmaStoreOp>(l.op))
      if (info(st.getDram()) == &b) {
        only = st;
        ++count;
      }
  if (count != 1)
    return false;
  if (only.getDramAddr() != 0 || only.getDramS1() != only.getLen0() ||
      only.getDramS2() != only.getLen0() * only.getN1())
    return false;
  if (only.getLen0() * only.getN1() * only.getN2() != b.extent)
    return false;
  shape.assign({int64_t(1), only.getN2(), only.getN1(), only.getLen0()});
  return true;
}

LogicalResult Emitter::renderMap(std::string &out) {
  auto layout = fn->getAttrOfType<DictionaryAttr>("kea.dram_layout");
  if (!layout)
    return fn.emitOpError("has no `kea.dram_layout`; -kea-emit needs the DRAM "
                          "arena geometry -kea-alloc publishes "
                          "(docs/MEMORY_PLANNING.md §5)");
  auto field = [&](StringRef k, int64_t dflt) -> int64_t {
    if (auto a = layout.getAs<IntegerAttr>(k))
      return a.getInt();
    return dflt;
  };

  std::string s;
  llvm::raw_string_ostream os(s);
  os << "{\n";
  os << "  \"arch\": \"" << ::kea::KEA_ARCH_NAME << "\",\n";
  os << "  \"isa_revision\": " << ::kea::KEA_ISA_REVISION << ",\n";
  os << "  \"entry_pc\": 0,\n\n";

  os << "  \"dram\": {\n";
  os << "    \"total_bytes\": " << field("total_bytes", 0) << ",\n";
  os << "    \"const_offset\": " << field("const_offset", 0) << ",\n";
  os << "    \"const_bytes\": " << field("const_bytes", 0) << ",\n";
  os << "    \"io_offset\": " << field("io_offset", 0) << ",\n";
  os << "    \"io_bytes\": " << field("io_bytes", 0) << ",\n";
  os << "    \"scratch_offset\": " << field("scratch_offset", 0) << ",\n";
  os << "    \"scratch_bytes\": " << field("scratch_bytes", 0) << ",\n";
  os << "    \"alignment\": " << field("alignment", 64) << "\n";
  os << "  },\n\n";

  //--- symbols: every DRAM object that is not host visible ------------------
  os << "  \"symbols\": [\n";
  bool first = true;
  for (auto &b : bufs) {
    if (b->space != AddressSpace::DRAM || b->role == "input" ||
        b->role == "output")
      continue;
    if (!first)
      os << ",\n";
    first = false;
    os << "    { \"name\": \"" << jsonEscape(b->name)
       << "\", \"offset\": " << b->base << ", \"size\": " << b->extent << " }";
  }
  os << (first ? "" : "\n") << "  ],\n\n";

  //--- tensors: the host-visible I/O ----------------------------------------
  os << "  \"tensors\": [\n";
  first = true;
  int64_t inIdx = 0, outIdx = 0;
  for (auto &b : bufs) {
    if (b->space != AddressSpace::DRAM ||
        (b->role != "input" && b->role != "output"))
      continue;
    if (b->name.size() >= ::kea::KEAF_TENSOR_NAME_BYTES)
      return b->alloc.emitOpError("a model.map.json tensor name is limited to ")
             << (::kea::KEAF_TENSOR_NAME_BYTES - 1) << " characters; \""
             << b->name << "\" is " << b->name.size()
             << ". Shorten the function name";
    const bool isIn = b->role == "input";

    SmallVector<int64_t> shape;
    StringRef layoutName = "NHWC";
    if (isIn) {
      // `-kea-tile` names a model input `<func>.input<argno>`, so its shape is
      // exactly that block argument's (docs/DIALECT_L2.md §3).
      StringRef nm = b->name;
      size_t pos = nm.rfind(".input");
      if (pos != StringRef::npos) {
        int64_t argNo = 0;
        if (!nm.drop_front(pos + 6).getAsInteger(10, argNo) && argNo >= 0 &&
            argNo < static_cast<int64_t>(fn.getNumArguments()))
          if (auto t = llvm::dyn_cast<RankedTensorType>(
                  fn.getArgument(argNo).getType()))
            shape.assign(t.getShape().begin(), t.getShape().end());
      }
    } else {
      inferShape(*b, shape);
    }
    if (shape.empty()) {
      shape.assign({b->extent});
      layoutName = "FLAT";
    }

    IoQuant q;
    auto ov = opts.ioQuant.find(b->name.str());
    if (ov != opts.ioQuant.end())
      q = ov->second;
    if (!q.hasZeroPoint)
      if (auto zp = isIn ? inputZeroPoint(*b) : outputZeroPoint(*b))
        q.zeroPoint = *zp;

    if (!first)
      os << ",\n";
    first = false;
    os << "    { \"name\": \"" << jsonEscape(b->name) << "\", \"kind\": \""
       << (isIn ? "input" : "output")
       << "\", \"index\": " << (isIn ? inIdx++ : outIdx++) << ",\n"
       << "      \"offset\": " << b->base << ", \"size_bytes\": " << b->extent
       << ", \"dtype\": \"int8\", \"layout\": \"" << layoutName << "\",\n"
       << "      \"shape\": [";
    for (size_t i = 0; i < shape.size(); ++i)
      os << (i ? ", " : "") << shape[i];
    os << "], \"scale\": " << jsonNumber(q.scale)
       << ", \"zero_point\": " << q.zeroPoint << " }";
  }
  os << (first ? "" : "\n") << "  ],\n\n";

  //--- spm_map: names for addresses, debug only -----------------------------
  os << "  \"spm_map\": [\n";
  first = true;
  std::string fnPrefix = (fn.getName() + ".").str();
  for (auto &b : bufs) {
    if (b->space == AddressSpace::DRAM || b->firstPc < 0)
      continue;
    // KeafSpmEntry's name field is 40 bytes. These are debug names inside one
    // map, so the function prefix every one of them shares is redundant.
    std::string nm = b->name.str();
    if (StringRef(nm).starts_with(fnPrefix))
      nm = nm.substr(fnPrefix.size());
    if (nm.size() >= ::kea::KEAF_SPM_NAME_BYTES)
      nm = nm.substr(nm.size() - (::kea::KEAF_SPM_NAME_BYTES - 1));
    const char *space = b->space == AddressSpace::A   ? "SPM_A"
                        : b->space == AddressSpace::W ? "SPM_W"
                                                      : "ACC";
    if (!first)
      os << ",\n";
    first = false;
    os << "    { \"name\": \"" << jsonEscape(nm) << "\", \"space\": \"" << space
       << "\", \"offset\": " << b->base << ", \"size\": " << b->extent
       << ", \"first_pc\": " << b->firstPc << ", \"last_pc\": " << b->lastPc
       << " }";
  }
  os << (first ? "" : "\n") << "  ],\n\n";

  os << "  \"metadata\": {\n";
  os << "    \"producer\": \"" << jsonEscape(opts.producer) << "\",\n";
  os << "    \"function\": \"" << jsonEscape(fn.getName()) << "\"";
  if (!opts.sourceName.empty())
    os << ",\n    \"source\": \"" << jsonEscape(opts.sourceName) << "\"";
  os << "\n  }\n}\n";

  out = std::move(s);
  return success();
}


/// A readable rendering of one constant. Records are decoded rather than
/// hex-dumped where isa.h gives them a shape, because "bias=168 mult=... 
/// shift=5" is a thing a human can check against the graph and 16 hex bytes
/// are not.
void appendListing(std::string &out, BufInfo &b,
                   const std::vector<uint8_t> &bytes) {
  llvm::raw_string_ostream os(out);
  StringRef layout = b.alloc.getLayout().value_or("");
  os << "; " << b.name << "  role=" << b.role << " layout=" << layout
     << " offset=" << b.base << " size=" << bytes.size() << "\n";

  auto i32At = [&](size_t off) {
    return static_cast<int32_t>(uint32_t(bytes[off]) |
                                (uint32_t(bytes[off + 1]) << 8) |
                                (uint32_t(bytes[off + 2]) << 16) |
                                (uint32_t(bytes[off + 3]) << 24));
  };
  if (layout == "quant_params") {
    for (size_t i = 0; i + 12 <= bytes.size(); i += 12)
      os << "  [" << (i / 12) << "] bias=" << i32At(i)
         << " mult=" << i32At(i + 4) << " shift=" << i32At(i + 8) << "\n";
    return;
  }
  if (layout == "add_params" && bytes.size() >= 20) {
    os << "  a_mult=" << i32At(0) << " b_mult=" << i32At(4)
       << " o_mult=" << i32At(8) << "\n";
    os << "  a_shift=" << int(int8_t(bytes[12]))
       << " b_shift=" << int(int8_t(bytes[13]))
       << " o_shift=" << int(int8_t(bytes[14])) << "\n";
    os << "  a_zp=" << int(int8_t(bytes[16])) << " b_zp="
       << int(int8_t(bytes[17])) << " o_zp=" << int(int8_t(bytes[18])) << "\n";
    return;
  }
  // Weight layouts: 16 signed bytes per line, which for the MXU layouts is
  // exactly one `k` row of one 16x16 tile.
  for (size_t i = 0; i < bytes.size(); i += 16) {
    os << "  " << llvm::format_decimal(int64_t(i), 6) << ":";
    for (size_t j = i; j < i + 16 && j < bytes.size(); ++j)
      os << " " << llvm::format_decimal(int(int8_t(bytes[j])), 4);
    os << "\n";
  }
}

//===----------------------------------------------------------------------===//
// The CONST blob
//===----------------------------------------------------------------------===//

LogicalResult Emitter::buildConstBlob(std::vector<uint8_t> &out,
                                      std::string &listing) {
  auto layout = fn->getAttrOfType<DictionaryAttr>("kea.dram_layout");
  int64_t constOffset = 0, constBytes = 0;
  if (auto a = layout.getAs<IntegerAttr>("const_offset"))
    constOffset = a.getInt();
  if (auto a = layout.getAs<IntegerAttr>("const_bytes"))
    constBytes = a.getInt();

  // ASSEMBLY.md §7.2: `const_bytes` must equal the size of the `--const` file
  // exactly, and the region's internal alignment padding is part of it -- so
  // the blob is sized from the layout and written into, never appended to.
  out.assign(constBytes, 0);
  for (auto &b : bufs) {
    if (b->space != AddressSpace::DRAM || !isConstRole(b->role))
      continue;
    std::vector<uint8_t> bytes;
    if (failed(materializeConstant(b->alloc.getOperation(), bytes)))
      return failure();
    const int64_t at = b->base - constOffset;
    if (at < 0 || at + static_cast<int64_t>(bytes.size()) > constBytes)
      return b->alloc.emitOpError("is placed at ")
             << b->base << " with " << bytes.size()
             << " bytes, which does not fit the CONST region [" << constOffset
             << ", " << (constOffset + constBytes) << ")";
    std::copy(bytes.begin(), bytes.end(), out.begin() + at);
    appendListing(listing, *b, bytes);
  }
  return success();
}

//===----------------------------------------------------------------------===//

LogicalResult Emitter::run(EmitResult &result) {
  if (fn.getBody().getBlocks().size() != 1)
    return fn.emitOpError("must be a single straight-line block: KEA-1 has no "
                          "control flow but HALT (ISA.md §1)");
  if (failed(collectBuffers()) || failed(emitInstructions()))
    return failure();

  const bool hasSemaphores = llvm::any_of(lines, [](const Line &l) {
    return l.mnem == "SIGNAL" || l.mnem == "WAIT";
  });
  const bool doSync = opts.sync == SyncMode::Insert ||
                      (opts.sync == SyncMode::Auto && !hasSemaphores);

  assignRegionUnits();
  if (doSync && failed(insertSync(result.numInsertedSync)))
    return failure();

  // PCs are only final once synchronization is woven in, so the `spm_map` live
  // ranges are computed here rather than reused from `-kea-alloc`'s attribute,
  // which counts block positions in the pre-sync program.
  for (size_t pc = 0; pc < lines.size(); ++pc) {
    Operation *op = lines[pc].op;
    if (!op)
      continue;
    for (Value v : op->getOperands())
      if (BufInfo *b = info(v)) {
        if (b->firstPc < 0)
          b->firstPc = static_cast<int64_t>(pc);
        b->lastPc = static_cast<int64_t>(pc);
      }
  }

  result.kasm = renderKasm();
  result.numInstructions = static_cast<int64_t>(lines.size());
  if (failed(renderMap(result.mapJson)) ||
      failed(buildConstBlob(result.constBlob, result.constListing)))
    return failure();
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//

LogicalResult mlir::kea::emitKasm(ModuleOp module, const EmitOptions &options,
                                  EmitResult &result) {
  SmallVector<func::FuncOp> candidates;
  module.walk([&](func::FuncOp fn) {
    bool isL2 = false;
    fn.walk([&](HaltOp) { isL2 = true; });
    if (isL2)
      candidates.push_back(fn);
  });

  if (candidates.empty())
    return module.emitError(
        "no Level 2 function to emit: no function in this module ends with "
        "kea.halt. Run -kea-tile and -kea-alloc first");

  auto names = [&] {
    std::string s;
    for (func::FuncOp fn : candidates)
      s += (s.empty() ? "" : ", ") + fn.getName().str();
    return s;
  };

  func::FuncOp chosen;
  if (!options.functionName.empty()) {
    for (func::FuncOp fn : candidates)
      if (fn.getName() == options.functionName)
        chosen = fn;
    if (!chosen) {
      InFlightDiagnostic diag = module.emitError("no Level 2 function named '")
                                << options.functionName << "'";
      diag.attachNote() << "candidates: " << names();
      return failure();
    }
  } else if (candidates.size() == 1) {
    chosen = candidates.front();
  } else {
    InFlightDiagnostic diag =
        module.emitError("this module has ")
        << candidates.size()
        << " Level 2 functions and a KEAF artifact holds exactly one program; "
           "pass --function to choose";
    diag.attachNote() << "candidates: " << names();
    return failure();
  }

  Emitter emitter(chosen, options);
  return emitter.run(result);
}
