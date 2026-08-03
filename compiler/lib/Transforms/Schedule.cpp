//===- Schedule.cpp - -kea-schedule: sequential L2 -> concurrent L2 -*- C++ -*-===//
//
// THE PASS THAT MAKES THE HARDWARE'S CONCURRENCY REAL.
//
// In:  the correct *sequential* Level 2 program `-kea-tile` emits -- no queue
//      on any instruction, no `kea.signal`, no `kea.wait`, no overlap.
// Out: the same instructions in a *concurrent* order: a queue on every one,
//      DMA spread over both engines, each tile's `DMA_LD` hoisted above the
//      previous tile's compute and its `DMA_ST` sunk below the next tile's,
//      and exactly the semaphores the real dependences need.
//
// Read docs/SCHEDULING.md alongside this file. The four properties that are
// not negotiable, and where each is established:
//
//   Rule D (ISA.md §5.5)   `assignSync()`. Guaranteed by construction, because
//                          events are per-(producer queue, consumer queue)
//                          token channels and a SIGNAL sits adjacent to a
//                          producer that a topological order has already
//                          placed before the consumer. `checkRuleD()` re-proves
//                          it over the finished block.
//   Weight banks (L2 §6.1) MXU instructions are never reordered against each
//                          other (the same-queue chain in `buildDeps()`) and
//                          nothing is slipped between a `kea.load_w` and the
//                          `kea.mm` reading its bank (`protectWeightPairs()`
//                          plus the fix-up at the end of `assignSync()`).
//   ADR-0002 soundness     `hoistAllocs()`. If two ops can execute
//                          concurrently, every buffer they touch is given
//                          overlapping block-order live ranges, so -kea-alloc
//                          cannot alias them into a data race.
//   Queue depth / capacity `listSchedule()` models the depth-16 queues and the
//                          scratchpad high-water mark while it emits, so a
//                          prefetch is never hoisted far enough to wedge the
//                          in-order dispatcher or to overflow a scratchpad.
//
// EVERY machine constant and timing formula comes from include/kea/hw_config.h.
//
//===----------------------------------------------------------------------===//

#include "kea/Transforms/Passes.h"

#include "kea/Dialect/KeaMachineOps.h"
#include "kea/Dialect/KeaTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "kea/hw_config.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>

namespace mlir {
namespace kea {
#define GEN_PASS_DEF_KEASCHEDULE
#include "kea/Transforms/Passes.h.inc"
} // namespace kea
} // namespace mlir

using namespace mlir;
using namespace mlir::kea;

namespace {

//===----------------------------------------------------------------------===//
// Queues
//===----------------------------------------------------------------------===//
//
// The five queued units of hw_config.h §2, in its own id order -- which is
// also the order WAITs are served in when several unblock in one cycle
// (ISA.md §5.3), so the numbering is not arbitrary.

enum : int {
  QMXU = ::kea::KEA_UNIT_MXU,
  QDWU = ::kea::KEA_UNIT_DWU,
  QVPU = ::kea::KEA_UNIT_VPU,
  QDMA0 = ::kea::KEA_UNIT_DMA0,
  QDMA1 = ::kea::KEA_UNIT_DMA1,
  QCOUNT = ::kea::KEA_NUM_QUEUES,
};

static StringRef queueName(int q) {
  switch (q) {
  case QMXU:
    return "MXU";
  case QDWU:
    return "DWU";
  case QVPU:
    return "VPU";
  case QDMA0:
    return "DMA0";
  default:
    return "DMA1";
  }
}

/// Cycles between a unit's last read and its last architectural write becoming
/// visible -- what a `SIGNAL` on that unit drains before it retires (ISA.md
/// §5.3 step 1). DMA engines have none.
static int64_t pipelineDepth(int q) {
  switch (q) {
  case QMXU:
    return ::kea::KEA_MXU_PIPELINE_DEPTH;
  case QDWU:
    return ::kea::KEA_DWU_PIPELINE_DEPTH;
  case QVPU:
    return ::kea::KEA_VPU_PIPELINE_DEPTH;
  default:
    return 0;
  }
}

/// The queue an instruction runs on. DMA is the only real choice; every other
/// opcode names its unit, which is why `unit` is not even an attribute on
/// those ops (DIALECT_L2.md §1.1). -1 for anything that is not a queued
/// Level 2 instruction.
static int fixedQueueOf(Operation *op) {
  if (isa<LoadWOp, MmOp>(op))
    return QMXU;
  if (isa<DwconvOp>(op))
    return QDWU;
  if (isa<VquantOp, VaddOp, VpoolOp, VcopyOp>(op))
    return QVPU;
  return -1;
}

static bool isDmaOp(Operation *op) { return isa<DmaLoadOp, DmaStoreOp>(op); }

/// Cycles the instruction holds its functional resource. Every formula is
/// hw_config.h's; none is reproduced here.
static int64_t occupancyOf(Operation *op) {
  using namespace ::kea;
  if (auto o = dyn_cast<LoadWOp>(op))
    return (int64_t)keaLoadWOccupancy(o.getKRows(), o.getInt4());
  if (auto o = dyn_cast<MmOp>(op))
    return (int64_t)keaMatmulOccupancy(o.getMInner(), o.getMOuter(),
                                       o.getInt4());
  if (auto o = dyn_cast<DwconvOp>(op))
    return (int64_t)keaDwconvOccupancy(o.getOutH(), o.getOutW(),
                                       o.getChannels(), o.getKernel(),
                                       o.getKernel());
  if (auto o = dyn_cast<VquantOp>(op))
    return (int64_t)keaVquantOccupancy(o.getNumPixels(), o.getChannels());
  if (auto o = dyn_cast<VaddOp>(op))
    return (int64_t)keaVaddOccupancy(o.getNumElems());
  if (auto o = dyn_cast<VpoolOp>(op))
    return (int64_t)keaVpoolOccupancy(o.getOutH(), o.getOutW(), o.getChannels(),
                                      o.getKh(), o.getKw());
  if (auto o = dyn_cast<VcopyOp>(op))
    return (int64_t)keaVcopyOccupancy(o.getRowBytes(), o.getRows());
  if (auto o = dyn_cast<DmaLoadOp>(op))
    return (int64_t)keaDmaOccupancy(o.getLen0(), o.getN1(), o.getN2());
  if (auto o = dyn_cast<DmaStoreOp>(op))
    return (int64_t)keaDmaOccupancy(o.getLen0(), o.getN1(), o.getN2());
  return 0;
}

/// Bytes a DMA descriptor moves; 0 for anything else. Feeds the DRAM port
/// contention term -- two live engines get 8 B/cycle each (MICROARCH.md §6.3).
static int64_t dmaBytesOf(Operation *op) {
  if (auto o = dyn_cast<DmaLoadOp>(op))
    return (int64_t)::kea::keaDmaBytes(o.getLen0(), o.getN1(), o.getN2());
  if (auto o = dyn_cast<DmaStoreOp>(op))
    return (int64_t)::kea::keaDmaBytes(o.getLen0(), o.getN1(), o.getN2());
  return 0;
}

static int64_t spaceCapacity(AddressSpace s) {
  switch (s) {
  case AddressSpace::A:
    return (int64_t)::kea::KEA_SPM_A_BYTES;
  case AddressSpace::W:
    return (int64_t)::kea::KEA_SPM_W_BYTES;
  case AddressSpace::ACC:
    return (int64_t)::kea::KEA_ACC_WORDS;
  default:
    return 0;
  }
}

//===----------------------------------------------------------------------===//
// The scheduling graph
//===----------------------------------------------------------------------===//

struct Node {
  Operation *op = nullptr;
  int origIndex = 0; ///< position in -kea-tile's sequential program
  int queue = -1;    ///< QMXU..QDMA1; chosen during scheduling for DMA
  bool dma = false;  ///< queue is not fixed by the opcode
  int64_t occ = 0;   ///< resource occupancy, cycles
  int64_t bytes = 0; ///< DRAM bytes (DMA only)
  SmallVector<int, 4> preds;
  SmallVector<int, 4> succs;
  int predsLeft = 0;
  int64_t height = 0; ///< longest remaining dependence path, in cycles

  int64_t start = 0;
  int64_t finish = 0;  ///< resource released
  int64_t visible = 0; ///< finish + pipeline drain: when another queue sees it
  int streamPos = -1;
};

/// A `kea.trace` REGION_BEGIN/REGION_END pair and the instructions it brackets.
struct Region {
  TraceOp begin;
  TraceOp end;
  int firstOrig = 0; ///< first original index inside the region
  int lastOrig = -1; ///< last original index inside the region
  int queue = QMXU;
  int lo = -1, hi = -1; ///< stream positions the markers attach to
};

/// An on-chip `kea.alloc` and the instructions that touch it.
struct BufferInfo {
  AllocOp alloc;
  AddressSpace space = AddressSpace::A;
  int64_t extent = 0;
  SmallVector<int, 8> users; ///< node ids, original order
  int liveCount = 0;
  bool live = false;
  int hoistTo = -1; ///< stream position the alloc is moved in front of
};

//===----------------------------------------------------------------------===//

struct Sched {
  Sched(func::FuncOp f, StringRef mode, int64_t depth, bool annotate)
      : func(f), serial(mode == "serial"), annotateUnits(annotate) {
    queueDepth = depth > 0 ? depth : (int64_t)::kea::KEA_QUEUE_DEPTH;
  }

  func::FuncOp func;
  Block *block = nullptr;
  bool serial = false;
  bool annotateUnits = true;
  int64_t queueDepth = ::kea::KEA_QUEUE_DEPTH;

  SmallVector<Node> nodes;
  SmallVector<Region> regions;
  SmallVector<BufferInfo> buffers;
  DenseMap<Value, int> bufferOf;
  HaltOp halt;

  SmallVector<int> order; ///< node ids in stream order

  // The cycle-approximate model's output.
  int64_t modelledCycles = 0;
  std::array<int64_t, QCOUNT> busy = {};
  std::array<int64_t, QCOUNT> maxQueue = {};
  std::array<int64_t, QCOUNT> dmaBytes = {};
  std::array<int64_t, QCOUNT> instrs = {};
  int64_t dispatchStall = 0;

  // assignSync()'s output.
  SmallVector<SmallVector<std::pair<int, int>, 2>> waitsBefore;
  SmallVector<SmallVector<int, 2>> signalsAfter;
  SmallVector<std::array<int, QCOUNT>> hbFront;
  std::array<std::array<int, QCOUNT>, QCOUNT> eventId = {};
  int numEvents = 0;

  int64_t nSignals = 0, nWaits = 0, nHoisted = 0;

  std::array<int, 3> window = {}; ///< buffers of a space that may be live
  int capacityIters = 0;

  LogicalResult run(bool reportSchedule);

  void resetSchedule();
  std::array<int64_t, 3> spmPeak() const;
  void collect();
  void buildDeps();
  void protectWeightPairs();
  void listSchedule();
  void serialSchedule();
  void assignSync();
  void locateRegions();
  void materialize();
  void hoistAllocs();
  LogicalResult checkRuleD();
  void report();
};

//===----------------------------------------------------------------------===//
// 1. Collect the sequential program
//===----------------------------------------------------------------------===//

void Sched::collect() {
  // Re-running the scheduler must be a fixed point, so sync already in the
  // block is dropped and recomputed rather than reasoned about.
  SmallVector<Operation *> stale;
  for (Operation &op : *block)
    if (isa<SignalOp, WaitOp>(op))
      stale.push_back(&op);
  for (Operation *op : stale)
    op->erase();

  SmallVector<int> regionStack;
  for (Operation &op : *block) {
    if (auto alloc = dyn_cast<AllocOp>(op)) {
      auto t = llvm::cast<BufferType>(alloc.getResult().getType());
      if (t.isOnChip()) {
        bufferOf[alloc.getResult()] = (int)buffers.size();
        BufferInfo bi;
        bi.alloc = alloc;
        bi.space = t.getAddressSpace();
        bi.extent = t.getNumElements();
        buffers.push_back(bi);
      }
      continue;
    }
    if (auto h = dyn_cast<HaltOp>(op)) {
      halt = h;
      continue;
    }
    if (auto tr = dyn_cast<TraceOp>(op)) {
      if (tr.getKind() == "begin") {
        Region r;
        r.begin = tr;
        r.firstOrig = (int)nodes.size();
        regionStack.push_back((int)regions.size());
        regions.push_back(r);
      } else if (tr.getKind() == "end" && !regionStack.empty()) {
        Region &r = regions[regionStack.pop_back_val()];
        r.end = tr;
        r.lastOrig = (int)nodes.size() - 1;
      }
      continue;
    }
    if (fixedQueueOf(&op) < 0 && !isDmaOp(&op))
      continue; // arith.constant, func.return, ... -- not an instruction

    Node n;
    n.op = &op;
    n.origIndex = (int)nodes.size();
    n.queue = fixedQueueOf(&op);
    n.dma = n.queue < 0;
    n.occ = occupancyOf(&op);
    n.bytes = dmaBytesOf(&op);
    nodes.push_back(n);
  }

  for (auto [i, n] : llvm::enumerate(nodes))
    for (Value v : n.op->getOperands()) {
      auto it = bufferOf.find(v);
      if (it != bufferOf.end() &&
          !llvm::is_contained(buffers[it->second].users, (int)i))
        buffers[it->second].users.push_back((int)i);
    }
}

//===----------------------------------------------------------------------===//
// 2. Dependences
//===----------------------------------------------------------------------===//
//
// Level 2 attaches its memory effects to the *operands* (DIALECT_L2.md §1.2),
// so MLIR's effect analysis already reports per-buffer reads and writes and
// the dependence graph is the ordinary RAW / WAR / WAW closure over each
// buffer's use list. Every edge points forwards in -kea-tile's order, so the
// graph is acyclic and any topological order of it executes the same program.
//
// DRAM buffers get exactly the same treatment as scratchpads: a `DMA_ST` into
// an inter-layer activation and the next layer's `DMA_LD` of it are a real
// dependence even though nothing on chip connects them.
//
// One extra family of edge, about queues rather than data: a chain between
// consecutive instructions on the same *fixed* queue. Each queue is in-order
// so this costs nothing, and it buys two things -- "no synchronization is
// needed within a unit" (ISA.md §7.3) becomes true of the output by
// construction, and MXU instructions can never be reordered against each
// other, which is what DIALECT_L2.md §6.1's weight-bank invariant requires.

void Sched::buildDeps() {
  auto addEdge = [&](int from, int to) {
    if (from == to || llvm::is_contained(nodes[from].succs, to))
      return;
    nodes[from].succs.push_back(to);
    nodes[to].preds.push_back(from);
  };

  struct Access {
    int node;
    bool read;
    bool write;
  };
  DenseMap<Value, SmallVector<Access, 8>> accesses;
  for (auto [i, n] : llvm::enumerate(nodes)) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(n.op);
    if (!iface)
      continue;
    SmallVector<MemoryEffects::EffectInstance> effects;
    iface.getEffects(effects);
    llvm::MapVector<Value, std::pair<bool, bool>> perValue;
    for (auto &e : effects) {
      Value v = e.getValue();
      if (!v || !llvm::isa<BufferType>(v.getType()))
        continue;
      auto &rw = perValue[v];
      rw.first |= isa<MemoryEffects::Read>(e.getEffect());
      rw.second |= isa<MemoryEffects::Write>(e.getEffect());
    }
    for (auto &kv : perValue)
      accesses[kv.first].push_back({(int)i, kv.second.first, kv.second.second});
  }

  for (auto &kv : accesses) {
    SmallVector<Access, 8> &as = kv.second;
    llvm::sort(as, [](const Access &a, const Access &b) {
      return a.node < b.node;
    });
    int lastWrite = -1;
    SmallVector<int, 8> readsSinceWrite;
    for (const Access &a : as) {
      if (a.read && lastWrite >= 0)
        addEdge(lastWrite, a.node); // RAW
      if (a.write) {
        if (lastWrite >= 0)
          addEdge(lastWrite, a.node); // WAW
        for (int r : readsSinceWrite)
          addEdge(r, a.node); // WAR
        lastWrite = a.node;
        readsSinceWrite.clear();
      } else if (a.read) {
        readsSinceWrite.push_back(a.node);
      }
    }
  }

  // THE ONE QUEUE WHOSE ORDER IS FROZEN. `bank = t & 1` (DIALECT_L2.md §6.1)
  // makes every `kea.mm` read the bank the immediately preceding `kea.load_w`
  // wrote, and that is only an invariant if MXU instructions keep their
  // relative order. It costs nothing: the MXU queue is in-order and a chain of
  // MATMULs accumulating into one ACC region needs no synchronization anyway
  // (ISA.md §7.3). Every *other* queue is free to be reordered, and that
  // freedom matters -- keeping the VPU in -kea-tile's order would pin tile
  // N+1's halo fill behind tile N's requantize, which serializes the whole
  // pipeline through the VPU for no reason. Reordering within a queue needs no
  // synchronization either, because the emitted order *is* the execution order.
  int lastMxu = -1;
  for (auto [i, n] : llvm::enumerate(nodes)) {
    if (n.queue != QMXU)
      continue;
    if (lastMxu >= 0)
      addEdge(lastMxu, (int)i);
    lastMxu = (int)i;
  }

  protectWeightPairs();

  for (int i = (int)nodes.size() - 1; i >= 0; --i) {
    int64_t h = 0;
    for (int s : nodes[i].succs)
      h = std::max(h, nodes[s].height);
    nodes[i].height = h + nodes[i].occ;
  }
  for (Node &n : nodes)
    n.predsLeft = (int)n.preds.size();
}

/// DIALECT_L2.md §6.1: for every `kea.mm`, the `kea.load_w` that most recently
/// targeted the same bank must be the *immediately preceding MXU instruction*.
/// Nothing may be slipped between them -- and a `kea.signal` is an MXU
/// instruction. A `kea.load_w` only ever *reads* SPM_W, so the only way it can
/// be a producer at all is a WAR edge from a later DMA refilling that weight
/// tile. Redirect such edges to the paired `kea.mm`, one instruction later, so
/// the SIGNAL lands after the pair rather than inside it.
void Sched::protectWeightPairs() {
  for (auto [i, n] : llvm::enumerate(nodes)) {
    if (!isa<LoadWOp>(n.op))
      continue;
    int mm = -1;
    for (int j = (int)i + 1; j < (int)nodes.size(); ++j) {
      if (nodes[j].queue != QMXU)
        continue;
      if (isa<MmOp>(nodes[j].op))
        mm = j;
      break;
    }
    if (mm < 0)
      continue;
    SmallVector<int, 4> succs(nodes[i].succs.begin(), nodes[i].succs.end());
    for (int s : succs) {
      if (s <= mm || llvm::is_contained(nodes[mm].succs, s))
        continue;
      nodes[mm].succs.push_back(s);
      nodes[s].preds.push_back(mm);
    }
  }
}

//===----------------------------------------------------------------------===//
// 3. List scheduling
//===----------------------------------------------------------------------===//
//
// A cycle-approximate model of the machine drives a greedy topological order.
// At each step the ready set is priced with hw_config.h's occupancies and the
// candidate that can *start* soonest wins; ties go to the longest remaining
// dependence path, then to -kea-tile's order, so the result is deterministic.
//
// Prefetching is not a special case in here. Tile N+1's `DMA_LD` writes a
// fresh `kea.alloc`, so it depends on nothing tile N does; it is ready from
// the start, an idle DMA engine can run it immediately, and the greedy choice
// therefore places it above tile N's compute. That *is* the double buffering.
//
// Two guards keep the greedy choice honest, and they are why this is a model
// rather than "hoist everything to the top":
//
//   * QUEUE DEPTH. The dispatcher is in-order and its only stall condition is
//     a full target queue (ISA.md §5.5). Pushing a prefetch into a queue that
//     already holds `queueDepth` unstarted instructions stalls *every* unit,
//     so the delay is priced into the candidate's key twice -- once as a later
//     start, once as an explicit penalty -- and a candidate that does not
//     stall wins.
//   * SCRATCHPAD HIGH-WATER MARK. Hoisting a load makes its tile live earlier,
//     which is the entire point (ADR-0002) -- but three tile sets live at once
//     do not fit in the half scratchpad `-kea-tile` reserved. A candidate that
//     would push a space past its capacity loses to any candidate that fits.

void Sched::listSchedule() {
  const int n = (int)nodes.size();
  order.clear();
  order.reserve(n);

  std::array<int64_t, QCOUNT> queueFree = {};
  std::array<SmallVector<int64_t, 32>, QCOUNT> pending;
  std::array<int64_t, 4> spaceUse = {};

  int64_t dispatch = 0;
  SmallVector<int> ready;
  for (int i = 0; i < n; ++i)
    if (nodes[i].predsLeft == 0)
      ready.push_back(i);

  // Slots of queue `q` still occupied at cycle `at`: everything dispatched
  // into it whose execution has not begun. Errata E3 -- a slot freed this
  // cycle is usable this cycle -- so an instruction starting at `at` no longer
  // holds one.
  auto liveInQueue = [&](int q, int64_t at) {
    int64_t c = 0;
    for (int64_t s : pending[q])
      if (s > at)
        ++c;
    return c;
  };
  // When the target queue frees a slot: the cycle the oldest still-unstarted
  // instruction starts. Errata E3 -- a slot freed this cycle is usable this
  // cycle -- so no +1.
  auto slotFreeAt = [&](int q, int64_t at) {
    SmallVector<int64_t, 32> future;
    for (int64_t s : pending[q])
      if (s > at)
        future.push_back(s);
    if ((int64_t)future.size() < queueDepth)
      return at;
    llvm::sort(future);
    return future[future.size() - queueDepth];
  };

  while (!ready.empty()) {
    int best = -1, bestQueue = -1;
    int64_t bestKey = 0, bestStart = 0, bestOcc = 0, bestStall = 0;
    bool bestFits = false;

    for (int cand : ready) {
      Node &nd = nodes[cand];

      // Would emitting this bring new buffers to life, and does the result
      // still fit on chip?
      std::array<int64_t, 4> delta = {};
      for (Value v : nd.op->getOperands()) {
        auto it = bufferOf.find(v);
        if (it == bufferOf.end())
          continue;
        BufferInfo &bi = buffers[it->second];
        if (!bi.live)
          delta[(int)bi.space] += bi.extent;
      }
      bool fits = true;
      for (int s = 0; s < 3; ++s)
        if (delta[s] &&
            spaceUse[s] + delta[s] > spaceCapacity((AddressSpace)s))
          fits = false;

      SmallVector<int, 2> qs;
      if (nd.queue >= 0)
        qs.push_back(nd.queue);
      else {
        qs.push_back(QDMA0);
        qs.push_back(QDMA1);
      }

      // ENGINE SELECTION. Two DMA engines exist so a prefetch can run while a
      // store runs (ISA.md §12), which only happens if the work is actually
      // spread. Pick the engine that starts this descriptor soonest; break the
      // tie -- which is the common case, because both engines are usually idle
      // when a descriptor becomes ready -- on the shorter contended occupancy,
      // then on the less loaded engine. That last term is what stops every
      // descriptor from piling onto DMA0 and turning the second engine into
      // decoration.
      int chosenQ = -1;
      int64_t chosenStart = 0, chosenOcc = 0, chosenStall = 0;
      for (int q : qs) {
        const int64_t d = std::max(dispatch, slotFreeAt(q, dispatch));
        const int64_t stall = d - dispatch;
        int64_t start = std::max(d, queueFree[q]);
        for (int p : nd.preds)
          start = std::max(start, nodes[p].queue == q ? nodes[p].finish
                                                      : nodes[p].visible);
        int64_t occ = nd.occ;
        if (nd.bytes > 0) {
          // Both engines share one 16 B/cycle DRAM port, so while the other
          // engine is also in its data phase this one gets 8 B/cycle and the
          // byte term stretches (MICROARCH.md §6.3).
          const int other = (q == QDMA0) ? QDMA1 : QDMA0;
          const int64_t data =
              (nd.bytes + ::kea::KEA_DMA_ENGINE_BYTES_PER_CYCLE - 1) /
              ::kea::KEA_DMA_ENGINE_BYTES_PER_CYCLE;
          occ += std::min(data, std::max<int64_t>(0, queueFree[other] - start));
        }
        bool take = chosenQ < 0;
        if (!take && start != chosenStart)
          take = start < chosenStart;
        else if (!take && occ != chosenOcc)
          take = occ < chosenOcc;
        else if (!take && start == chosenStart && occ == chosenOcc)
          take = busy[q] < busy[chosenQ];
        if (take) {
          chosenQ = q;
          chosenStart = start;
          chosenOcc = occ;
          chosenStall = stall;
        }
      }

      // Earliest start wins; the critical path breaks ties, scaled down so it
      // never outweighs a real cycle of delay by much.
      const int64_t key = chosenStart + chosenStall - nd.height / 64;
      bool better;
      if (best < 0)
        better = true;
      else if (fits != bestFits)
        better = fits;
      else if (key != bestKey)
        better = key < bestKey;
      else
        better = cand < best;
      if (better) {
        best = cand;
        bestQueue = chosenQ;
        bestKey = key;
        bestStart = chosenStart;
        bestOcc = chosenOcc;
        bestStall = chosenStall;
        bestFits = fits;
      }
    }

    Node &nd = nodes[best];
    nd.queue = bestQueue;
    nd.start = bestStart;
    nd.finish = bestStart + bestOcc;
    nd.visible = nd.finish + pipelineDepth(bestQueue);
    nd.streamPos = (int)order.size();
    order.push_back(best);
    if (nd.origIndex > nd.streamPos)
      ++nHoisted;

    const int64_t dispatchedAt = dispatch + bestStall;
    // Occupancy at the moment of dispatch, counting this instruction. The
    // stall above is exactly what keeps this at or below `queueDepth`, so the
    // reported high-water mark is the proof that the schedule never wedges the
    // in-order dispatcher on a full queue.
    maxQueue[bestQueue] = std::max(
        maxQueue[bestQueue], liveInQueue(bestQueue, dispatchedAt) + 1);
    dispatch = dispatchedAt + ::kea::KEA_DISPATCH_PER_CYCLE;
    dispatchStall += bestStall;
    queueFree[bestQueue] = nd.finish;
    busy[bestQueue] += bestOcc;
    dmaBytes[bestQueue] += nd.bytes;
    instrs[bestQueue]++;
    pending[bestQueue].push_back(nd.start);
    modelledCycles = std::max(modelledCycles, nd.visible);

    for (Value v : nd.op->getOperands()) {
      auto it = bufferOf.find(v);
      if (it == bufferOf.end())
        continue;
      BufferInfo &bi = buffers[it->second];
      if (!bi.live) {
        bi.live = true;
        bi.liveCount = (int)bi.users.size();
        spaceUse[(int)bi.space] += bi.extent;
      }
      if (--bi.liveCount == 0)
        spaceUse[(int)bi.space] -= bi.extent;
    }

    ready.erase(llvm::find(ready, best));
    for (int s : nd.succs)
      if (--nodes[s].predsLeft == 0)
        ready.push_back(s);
  }
}

/// The A/B control for docs/SCHEDULING.md's measurement: -kea-tile's order,
/// one DMA engine, and a full handshake at every cross-queue adjacency.
/// Nothing overlaps -- which is what the sequential program costs when it is
/// actually run on five queues.
void Sched::serialSchedule() {
  order.clear();
  // Every cross-queue adjacency becomes a dependence, so §4's minimal-sync
  // machinery -- and with it Rule D, and the same token-channel events --
  // applies unchanged and the baseline is synchronized the same way the real
  // schedule is. The only difference is that it has nothing left to overlap.
  for (size_t i = 1; i < nodes.size(); ++i) {
    int prevQ = nodes[i - 1].queue < 0 ? QDMA0 : nodes[i - 1].queue;
    int curQ = nodes[i].queue < 0 ? QDMA0 : nodes[i].queue;
    if (prevQ != curQ && !llvm::is_contained(nodes[i].preds, (int)i - 1))
      nodes[i].preds.push_back((int)i - 1);
  }
  int64_t t = 0;
  for (auto [i, n] : llvm::enumerate(nodes)) {
    Node &nd = nodes[i];
    nd.queue = nd.queue < 0 ? QDMA0 : nd.queue;
    nd.streamPos = (int)i;
    nd.start = t;
    nd.finish = t + nd.occ;
    nd.visible = nd.finish + pipelineDepth(nd.queue);
    t = nd.visible;
    busy[nd.queue] += nd.occ;
    dmaBytes[nd.queue] += nd.bytes;
    instrs[nd.queue]++;
    maxQueue[nd.queue] = std::max<int64_t>(maxQueue[nd.queue], 1);
    order.push_back((int)i);
  }
  modelledCycles = t;
}

//===----------------------------------------------------------------------===//
// 4. Semaphores
//===----------------------------------------------------------------------===//
//
// EVENTS ARE TOKEN CHANNELS. One event per ordered pair of queues (producer,
// consumer): at most 5*4 = 20 of the 32, and in practice fewer. Every SIGNAL
// increments by 1 and every WAIT consumes 1, so the channel's counter is
// simply "signals issued by P minus waits taken by C". Both queues are
// in-order, so C's i-th wait is released by P's i-th signal -- a counting
// semaphore used as a FIFO of tokens between two totally ordered streams.
//
// WHY THAT MAKES RULE D STRUCTURAL. `need[c][u]` is the latest position on
// queue u that c transitively depends on, and `front[C][u]` -- what C has
// already waited for, directly or transitively -- only ever moves forwards.
// So the producer positions C waits for on channel (P,C) are strictly
// increasing, and the signals on that channel are emitted adjacent to those
// same positions in the same increasing order. Each producer precedes its
// consumer in the stream because the stream is a topological order of the
// dependence graph. Hence for every i and every channel the i-th signal is at
// a lower stream position than the i-th wait: Rule D, by construction, with
// nothing to repair afterwards.
//
// WHY IT IS MINIMAL. A wait is emitted only when the producer position is
// beyond what this consumer queue is *already* ordered after. Two instructions
// on the same queue are never synchronized (ISA.md §7.3), and a dependence
// that an earlier wait on the same queue already covers costs nothing.
//
// HAPPENS-BEFORE IN FIVE INTEGERS. Because each queue is in-order, "everything
// on queue u that c is ordered after" is a *prefix* of u's stream. So the
// whole happens-before relation is exactly `hbFront[c][u]`, five numbers per
// instruction: HB(q, c) iff pos(q) <= hbFront[c][queue(q)]. §6 needs that.

void Sched::assignSync() {
  const int n = (int)order.size();
  waitsBefore.assign(n, {});
  signalsAfter.assign(n, {});
  hbFront.assign(n, {});

  SmallVector<std::array<int, QCOUNT>> need(n);
  for (auto &a : need)
    a.fill(-1);
  for (int p = 0; p < n; ++p) {
    Node &c = nodes[order[p]];
    for (int pid : c.preds) {
      const int pp = nodes[pid].streamPos;
      for (int u = 0; u < QCOUNT; ++u)
        need[p][u] = std::max(need[p][u], need[pp][u]);
      const int pq = nodes[pid].queue;
      need[p][pq] = std::max(need[p][pq], pp);
    }
  }

  std::array<std::array<int, QCOUNT>, QCOUNT> front;
  for (auto &a : front)
    a.fill(-1);

  for (int p = 0; p < n; ++p) {
    const int C = nodes[order[p]].queue;
    for (int u = 0; u < QCOUNT; ++u) {
      if (u == C || need[p][u] <= front[C][u])
        continue;
      const int q = need[p][u];
      waitsBefore[p].push_back({q, u});
      if (!llvm::is_contained(signalsAfter[q], C))
        signalsAfter[q].push_back(C);
      // Waiting for q orders C after everything q was ordered after, too.
      for (int w = 0; w < QCOUNT; ++w)
        front[C][w] = std::max(front[C][w], hbFront[q][w]);
      front[C][u] = std::max(front[C][u], q);
    }
    hbFront[p] = front[C];
    front[C][C] = p;
  }

  // Nothing may be slipped between a `kea.load_w` and the `kea.mm` reading its
  // bank (DIALECT_L2.md §6.1): pull that mm's waits above the load, and push
  // the load's signals below the mm. Both moves keep Rule D. The waits move
  // earlier, and the signals feeding them are earlier still -- a same-queue
  // producer never signals, so the producer is at most position p-2. The
  // signals move later, but their consumer waits at least one position past
  // the mm, again because a same-queue consumer never waits.
  for (int p = 1; p < n; ++p) {
    if (!isa<MmOp>(nodes[order[p]].op) || !isa<LoadWOp>(nodes[order[p - 1]].op))
      continue;
    for (auto &w : waitsBefore[p])
      waitsBefore[p - 1].push_back(w);
    waitsBefore[p].clear();
    for (int c : signalsAfter[p - 1])
      if (!llvm::is_contained(signalsAfter[p], c))
        signalsAfter[p].push_back(c);
    signalsAfter[p - 1].clear();
  }

  for (auto &a : eventId)
    a.fill(-1);
  for (int p = 0; p < n; ++p)
    for (auto &w : waitsBefore[p]) {
      int &id = eventId[w.second][nodes[order[p]].queue];
      if (id < 0)
        id = numEvents++;
    }
}

//===----------------------------------------------------------------------===//
// 5. Regions and materialization
//===----------------------------------------------------------------------===//

void Sched::locateRegions() {
  // A region's queue is the one that does its arithmetic, so errata E4's "the
  // region keeps accumulating until the issuing unit's pipeline drains"
  // attributes the tail of a layer to that layer rather than to its successor.
  // Both markers go on that one queue.
  for (Region &r : regions) {
    if (!r.begin || !r.end || r.lastOrig < r.firstOrig)
      continue;
    int q = -1;
    for (int i = r.firstOrig; i <= r.lastOrig && i < (int)nodes.size(); ++i) {
      const int nq = nodes[i].queue;
      if (nq == QMXU) {
        q = QMXU;
        break;
      }
      if (nq == QDWU)
        q = QDWU;
      else if (nq == QVPU && q < 0)
        q = QVPU;
    }
    if (q < 0)
      q = nodes[r.firstOrig].queue;
    r.queue = q;
    // The markers bracket every instruction of the region, on whichever queue
    // it ended up: a region's counters cover every unit's activity in its
    // cycle window (SIMULATOR.md §5), so a layer must own the DMA that feeds
    // it. Software pipelining means consecutive regions can overlap, which
    // errata E4 already anticipates and which is the honest attribution: the
    // tail of layer n really is executing while layer n+1 starts.
    for (int i = r.firstOrig; i <= r.lastOrig && i < (int)nodes.size(); ++i) {
      const int sp = nodes[i].streamPos;
      if (r.lo < 0 || sp < r.lo)
        r.lo = sp;
      if (sp > r.hi)
        r.hi = sp;
    }
    auto name = StringAttr::get(func.getContext(), queueName(q));
    r.begin->setAttr("unit", name);
    r.end->setAttr("unit", name);
  }
}

void Sched::materialize() {
  MLIRContext *ctx = func.getContext();
  OpBuilder b(ctx);
  Operation *term = block->getTerminator();
  b.setInsertionPoint(term);

  auto unitAttr = [&](int q) { return StringAttr::get(ctx, queueName(q)); };

  SmallVector<Operation *> stream;
  for (int p = 0; p < (int)order.size(); ++p) {
    Node &nd = nodes[order[p]];
    for (Region &r : regions)
      if (r.lo == p && r.begin)
        stream.push_back(r.begin);
    for (auto &w : waitsBefore[p]) {
      auto wait = b.create<WaitOp>(nd.op->getLoc(),
                                   b.getI64IntegerAttr(eventId[w.second][nd.queue]),
                                   b.getI64IntegerAttr(1), unitAttr(nd.queue));
      stream.push_back(wait);
      ++nWaits;
    }
    if (isDmaOp(nd.op))
      nd.op->setAttr("unit", unitAttr(nd.queue));
    else if (annotateUnits)
      // `kea.load_w` / `kea.mm` / `kea.dwconv` / `kea.v*` have no `unit` field
      // in KeaMachineOps.td: their queue follows from the opcode, which is
      // stronger than an attribute. Stamp it anyway, discardably, so the
      // assignment is readable and testable. -kea-emit must not need it.
      nd.op->setAttr("kea.unit", unitAttr(nd.queue));
    stream.push_back(nd.op);
    for (int c : signalsAfter[p]) {
      auto sig = b.create<SignalOp>(nd.op->getLoc(),
                                    b.getI64IntegerAttr(eventId[nd.queue][c]),
                                    b.getI64IntegerAttr(1), unitAttr(nd.queue));
      stream.push_back(sig);
      ++nSignals;
    }
    for (Region &r : llvm::reverse(regions))
      if (r.hi == p && r.end)
        stream.push_back(r.end);
  }

  for (Operation *op : stream)
    op->moveBefore(term);
  if (halt)
    halt->moveBefore(term);

  // The on-chip allocs go back in front of the instruction each has to be live
  // from; see §6. Everything else (DRAM symbols, `arith.constant`) stays where
  // -kea-tile put it, at the top of the block.
  for (BufferInfo &bi : buffers) {
    if (bi.hoistTo < 0 || bi.hoistTo >= (int)order.size())
      continue;
    bi.alloc->moveBefore(nodes[order[bi.hoistTo]].op);
  }
}

//===----------------------------------------------------------------------===//
// 6. ADR-0002's soundness obligation (normative)
//===----------------------------------------------------------------------===//
//
//   "After -kea-schedule, block order must be a sound over-approximation of
//    temporal liveness. If two operations can execute concurrently -- different
//    queues, with no semaphore ordering them -- then every buffer they touch
//    must have overlapping block-order live ranges."
//
// -kea-alloc does not trust the stamped `live` attribute; it re-derives the
// range from the SSA def-use chain, as `[position of the kea.alloc, position of
// its last user]` (MEMORY_PLANNING.md §2.1). So the obligation cannot be
// discharged by writing a wider attribute. It is discharged by *moving the
// alloc*, which is free -- `kea.alloc` is not an instruction.
//
// THE RULE. For every on-chip buffer b, move its `kea.alloc` to
//
//     lo(b) = min over users o of b, of the earliest stream position q <= pos(o)
//             such that q is NOT ordered before o and the instruction at q also
//             touches an on-chip buffer in b's address space.
//
// "not ordered before" is exact and cheap: HB(q, o) iff q <= hbFront[o][queue(q)]
// (§4). Restricting to b's own space is exact too, because -kea-alloc only ever
// aliases buffers within one space.
//
// WHY IT IS SUFFICIENT. Take any two instructions X at position x and Y at
// position y > x that may run concurrently, with X touching b1 and Y touching
// b2, both on chip in the same space S. Then
//
//   * x is in b1's range: first(b1) <= x because X is a user of b1 and the
//     alloc precedes every user, and last(b1) >= x for the same reason;
//   * x is in b2's range: first(b2) = lo(b2) <= x, because X sits at x <= y,
//     is unordered with the b2-user Y, and touches an S buffer -- which is
//     exactly the condition lo(b2) minimizes over; and last(b2) >= y > x.
//
// Both ranges contain position x, so they overlap, so -kea-alloc must separate
// them. That is the obligation, discharged for every concurrent pair at once,
// and it needs no extension of the *end* of any range.
//
// It is also *tight enough to be useful*: happens-before is dense in a real
// program (a tile's DMA is ordered before the compute that consumes it, which
// is ordered before the requantize, ...), so lo(b) typically reaches back
// exactly one tile -- which is precisely the window `-kea-tile`'s
// `spm-reserve-factor = 2` was sized for.

void Sched::hoistAllocs() {
  const int n = (int)order.size();
  // Per space, per queue: the stream positions of instructions touching an
  // on-chip buffer in that space, ascending.
  std::array<std::array<SmallVector<int>, QCOUNT>, 3> touch;
  for (int p = 0; p < n; ++p) {
    Node &nd = nodes[order[p]];
    std::array<bool, 3> seen = {};
    for (Value v : nd.op->getOperands()) {
      auto it = bufferOf.find(v);
      if (it != bufferOf.end())
        seen[(int)buffers[it->second].space] = true;
    }
    for (int s = 0; s < 3; ++s)
      if (seen[s])
        touch[s][nd.queue].push_back(p);
  }

  for (BufferInfo &bi : buffers) {
    int lo = std::numeric_limits<int>::max();
    const int s = (int)bi.space;
    for (int uid : bi.users) {
      const int p = nodes[uid].streamPos;
      lo = std::min(lo, p);
      for (int u = 0; u < QCOUNT; ++u) {
        const SmallVector<int> &v = touch[s][u];
        // First S-touching position on queue u that o is *not* ordered after.
        auto it = std::upper_bound(v.begin(), v.end(), hbFront[p][u]);
        if (it != v.end() && *it < p)
          lo = std::min(lo, *it);
      }
    }
    if (lo != std::numeric_limits<int>::max())
      bi.hoistTo = lo;
  }
}

//===----------------------------------------------------------------------===//
// 7. Rule D, re-proved over the finished block
//===----------------------------------------------------------------------===//
//
// §4 establishes Rule D by construction. This walks the emitted stream anyway
// and simulates the counters, because the assembler rejects a violating
// program and `kea-sim` reports it, and a compiler that can only find out at
// that distance from the mistake is not much use. It has never fired; if it
// ever does, the bug is in this file and the diagnostic points at the WAIT.

LogicalResult Sched::checkRuleD() {
  std::array<int64_t, ::kea::KEA_NUM_EVENTS> avail = {};
  for (Operation &op : *block) {
    if (auto s = dyn_cast<SignalOp>(op)) {
      if (s.getEvent() < 0 || s.getEvent() >= ::kea::KEA_NUM_EVENTS)
        return s.emitOpError("event id out of range");
      avail[s.getEvent()] += s.getValue();
    } else if (auto w = dyn_cast<WaitOp>(op)) {
      if (w.getEvent() < 0 || w.getEvent() >= ::kea::KEA_NUM_EVENTS)
        return w.emitOpError("event id out of range");
      if (avail[w.getEvent()] < w.getValue())
        return w.emitOpError("violates Rule D (ISA.md §5.5): only ")
               << avail[w.getEvent()] << " of the " << w.getValue()
               << " counts this WAIT consumes are signalled earlier in the "
                  "stream, so a full queue behind it would wedge the in-order "
                  "dispatcher";
      avail[w.getEvent()] -= w.getValue();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// 8. The report
//===----------------------------------------------------------------------===//

void Sched::report() {
  OpBuilder b(func.getContext());
  SmallVector<NamedAttribute> queues;
  for (int q = 0; q < QCOUNT; ++q) {
    SmallVector<NamedAttribute> e{
        b.getNamedAttr("instrs", b.getI64IntegerAttr(instrs[q])),
        b.getNamedAttr("busy", b.getI64IntegerAttr(busy[q])),
        b.getNamedAttr("max_queue", b.getI64IntegerAttr(maxQueue[q])),
    };
    if (q == QDMA0 || q == QDMA1)
      e.push_back(b.getNamedAttr("dram_bytes", b.getI64IntegerAttr(dmaBytes[q])));
    queues.push_back(b.getNamedAttr(queueName(q), b.getDictionaryAttr(e)));
  }
  SmallVector<NamedAttribute> top{
      b.getNamedAttr("mode", b.getStringAttr(serial ? "serial" : "overlap")),
      b.getNamedAttr("modelled_cycles", b.getI64IntegerAttr(modelledCycles)),
      b.getNamedAttr("dispatch_stall", b.getI64IntegerAttr(dispatchStall)),
      b.getNamedAttr("queue_depth", b.getI64IntegerAttr(queueDepth)),
      b.getNamedAttr("events", b.getI64IntegerAttr(numEvents)),
      b.getNamedAttr("signals", b.getI64IntegerAttr(nSignals)),
      b.getNamedAttr("waits", b.getI64IntegerAttr(nWaits)),
      b.getNamedAttr("hoisted", b.getI64IntegerAttr(nHoisted)),
      b.getNamedAttr("capacity_iters", b.getI64IntegerAttr(capacityIters + 1)),
      b.getNamedAttr("buffers_in_flight",
                     b.getI64ArrayAttr({window[0], window[1], window[2]})),
      b.getNamedAttr("spm_a_peak", b.getI64IntegerAttr(spmPeak()[0])),
      b.getNamedAttr("spm_w_peak", b.getI64IntegerAttr(spmPeak()[1])),
      b.getNamedAttr("acc_peak", b.getI64IntegerAttr(spmPeak()[2])),
      b.getNamedAttr("queues", b.getDictionaryAttr(queues)),
  };
  func->setAttr("kea.schedule", b.getDictionaryAttr(top));
}

//===----------------------------------------------------------------------===//

void Sched::resetSchedule() {
  for (Node &n : nodes) {
    if (n.dma)
      n.queue = -1;
    n.predsLeft = (int)n.preds.size();
    n.streamPos = -1;
    n.start = n.finish = n.visible = 0;
  }
  for (BufferInfo &b : buffers) {
    b.live = false;
    b.liveCount = 0;
    b.hoistTo = -1;
  }
  for (Region &r : regions)
    r.lo = r.hi = -1;
  order.clear();
  busy.fill(0);
  maxQueue.fill(0);
  dmaBytes.fill(0);
  instrs.fill(0);
  dispatchStall = 0;
  modelledCycles = 0;
  nHoisted = 0;
  numEvents = 0;
}

/// The high-water mark of simultaneously live on-chip data implied by the
/// live ranges this schedule will hand `-kea-alloc`, per space. This is that
/// pass's `maxlive`: the lower bound no allocator can beat.
std::array<int64_t, 3> Sched::spmPeak() const {
  std::array<int64_t, 3> peak = {};
  SmallVector<std::array<int64_t, 3>> delta(order.size() + 1);
  for (auto &d : delta)
    d.fill(0);
  for (const BufferInfo &b : buffers) {
    if (b.hoistTo < 0 || b.users.empty())
      continue;
    int last = 0;
    for (int uid : b.users)
      last = std::max(last, nodes[uid].streamPos);
    delta[b.hoistTo][(int)b.space] += b.extent;
    delta[last + 1][(int)b.space] -= b.extent;
  }
  std::array<int64_t, 3> cur = {};
  for (auto &d : delta)
    for (int s = 0; s < 3; ++s) {
      cur[s] += d[s];
      peak[s] = std::max(peak[s], cur[s]);
    }
  return peak;
}

LogicalResult Sched::run(bool reportSchedule) {
  block = &func.getBody().front();
  collect();
  if (nodes.empty())
    return success();

  buildDeps();

  // THE CAPACITY FIXPOINT -- how many tile buffers may be in flight.
  //
  // -kea-tile gives every tile a fresh `kea.alloc`, so nothing in the IR says
  // "there are only two activation buffers and they rotate". Left alone, the
  // list scheduler will happily run four tiles ahead, and §6 will then -- quite
  // correctly -- give all four overlapping live ranges, at which point
  // `-kea-alloc` needs four tiles' worth of SPM and refuses. Shaving the ranges
  // instead is not an option: it would trade a loud allocator failure for a
  // silent data race.
  //
  // So the rotation is made explicit, exactly as ISA.md §12 writes it by hand.
  // In each address space, buffer *i*'s first use is ordered after buffer
  // *i-K*'s last use -- an ordinary dependence, which §4 turns into an ordinary
  // WAIT, and which is precisely the "buffer 0 free" handshake of the
  // hand-written double-buffered layer. `K` is how many of that space's buffers
  // fit at once, computed from the real extents; if the *extended* ranges still
  // do not fit, K comes down and the schedule is recomputed. K = 1 is the floor
  // and always fits: no two buffers of a space are ever live together.
  const SmallVector<Node> baseGraph = nodes;

  std::array<SmallVector<int>, 3> perSpace; ///< buffer indices, original order
  for (auto [i, b] : llvm::enumerate(buffers))
    if (!b.users.empty())
      perSpace[(int)b.space].push_back((int)i);

  for (int s = 0; s < 3; ++s) {
    const int64_t cap = spaceCapacity((AddressSpace)s);
    const SmallVector<int> &v = perSpace[s];
    window[s] = 1;
    for (int k = 2; k <= (int)v.size(); ++k) {
      int64_t worst = 0;
      for (int i = 0; i + k <= (int)v.size(); ++i) {
        int64_t sum = 0;
        for (int j = i; j < i + k; ++j)
          sum += buffers[v[j]].extent;
        worst = std::max(worst, sum);
      }
      if (worst > cap)
        break;
      window[s] = k;
    }
  }

  bool fits = false;
  for (capacityIters = 0;; ++capacityIters) {
    nodes = baseGraph;
    for (int s = 0; s < 3; ++s) {
      const SmallVector<int> &v = perSpace[s];
      for (int i = window[s]; i < (int)v.size(); ++i) {
        const BufferInfo &prev = buffers[v[i - window[s]]];
        const BufferInfo &cur = buffers[v[i]];
        const int from = *std::max_element(prev.users.begin(), prev.users.end());
        const int to = *std::min_element(cur.users.begin(), cur.users.end());
        if (from >= to || llvm::is_contained(nodes[from].succs, to))
          continue;
        nodes[from].succs.push_back(to);
        nodes[to].preds.push_back(from);
      }
    }
    resetSchedule();
    if (serial)
      serialSchedule();
    else
      listSchedule();
    assignSync();
    hoistAllocs();
    const std::array<int64_t, 3> peak = spmPeak();
    fits = true;
    bool reduced = false;
    for (int s = 0; s < 3; ++s)
      if (peak[s] > spaceCapacity((AddressSpace)s)) {
        fits = false;
        if (window[s] > 1) {
          window[s]--;
          reduced = true;
        }
      }
    // Out of room to give back: -kea-tile's own tiles do not fit, which is its
    // diagnostic to give (and -kea-alloc's to repeat), not this pass's.
    if (fits || serial || !reduced)
      break;
  }

  if (numEvents > ::kea::KEA_NUM_EVENTS)
    return func.emitOpError("needs ")
           << numEvents << " events, but KEA-1 has only "
           << ::kea::KEA_NUM_EVENTS;

  locateRegions();
  materialize();

  // Any pass that inserts, erases or moves a Level 2 op must re-stamp the
  // live ranges (DIALECT_L2.md §4.2). This pass does all three.
  refreshLiveRanges(func);
  if (failed(verifyWeightBanks(func)) || failed(checkRuleD()))
    return failure();
  if (reportSchedule)
    report();
  return success();
}

//===----------------------------------------------------------------------===//

struct KeaSchedulePass
    : public mlir::kea::impl::KeaScheduleBase<KeaSchedulePass> {
  using mlir::kea::impl::KeaScheduleBase<KeaSchedulePass>::KeaScheduleBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal() || func.getBody().empty())
      return;

    if (mode != "overlap" && mode != "serial") {
      func.emitOpError("-kea-schedule=mode= must be \"overlap\" or "
                       "\"serial\", got \"")
          << mode << "\"";
      return signalPassFailure();
    }
    // Level 2 is a straight-line machine program in one block; stream position
    // is what Rule D and every live range are expressed in, and two blocks
    // would make neither comparable.
    if (!llvm::hasSingleElement(func.getBody())) {
      bool isL2 = false;
      func.walk([&](Operation *op) {
        if (fixedQueueOf(op) >= 0 || isDmaOp(op))
          isL2 = true;
      });
      if (isL2) {
        func.emitOpError("-kea-schedule needs a straight-line Level 2 program: "
                         "this function has ")
            << func.getBody().getBlocks().size() << " blocks";
        return signalPassFailure();
      }
      return;
    }

    Sched s(func, mode, queueDepth, annotateUnits);
    if (failed(s.run(reportSchedule)))
      return signalPassFailure();
    numSignals += s.nSignals;
    numWaits += s.nWaits;
    numHoisted += s.nHoisted;
  }
};

} // namespace
