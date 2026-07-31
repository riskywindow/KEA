// SPDX-License-Identifier: Apache-2.0
//
// simulator.cpp --- the concurrent, cycle-approximate timing model.
//
// This is MICROARCH.md §8's normative cycle-stepped algorithm, with two
// refinements that the document itself asks for elsewhere and that are spelled
// out in docs/SIMULATOR.md §"Ambiguities":
//
//   1. Within a cycle, SIGNAL increments are applied to *every* eligible unit
//      before *any* waiter is evaluated (ISA.md §5.3: "the SIGNAL's increment
//      is applied first, then waiters are evaluated"), and waiters are then
//      served in ascending unit id.  MICROARCH.md §8's single ascending pass
//      would let a low-id waiter miss a same-cycle signal from a high-id unit.
//
//   2. DMA engines are advanced at the END of the cycle, so the cycle in which
//      a descriptor starts is also the first cycle of its overhead phase.
//      That is the only placement that reproduces `keaDmaOccupancy()` exactly,
//      and that function is static_assert'd in hw_config.h.

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "kea/sim/simulator.h"

namespace kea {
namespace sim {

const char* resourceName(int r) {
  switch (r) {
    case RES_MXU_ARRAY:   return "MXU.ARRAY";
    case RES_MXU_WPORT:   return "MXU.WPORT";
    case RES_MXU_BANK0:   return "MXU.BANK0";
    case RES_MXU_BANK1:   return "MXU.BANK1";
    case RES_DWU_PIPE:    return "DWU.DWPIPE";
    case RES_VPU_PIPE:    return "VPU.VPIPE";
    case RES_DMA0_ENGINE: return "DMA0.ENGINE";
    case RES_DMA1_ENGINE: return "DMA1.ENGINE";
    default:              return "<invalid-resource>";
  }
}

const char* statusName(SimStatus s) {
  switch (s) {
    case SimStatus::Ok:             return "ok";
    case SimStatus::LoadError:      return "load-error";
    case SimStatus::RuleDViolation: return "rule-d-violation";
    case SimStatus::Deadlock:       return "deadlock";
    case SimStatus::EventOverflow:  return "event-overflow";
    case SimStatus::Fault:          return "fault";
    case SimStatus::MaxCycles:      return "max-cycles";
  }
  return "<unknown>";
}

// ---------------------------------------------------------------------------
// Rule D (ISA.md §5.5)
// ---------------------------------------------------------------------------
//
// "For every WAIT e,thr at stream position p, the SIGNALs that supply those
//  thr counts must appear at stream positions < p."
//
// Dispatch is in order and the stream is straight-line, so walking it once and
// checking that cumulative supply never falls behind cumulative demand is an
// exact test, not a conservative one.

std::string checkRuleD(const KeaProgram& prog) {
  std::uint64_t supply[KEA_NUM_EVENTS] = {0};
  std::uint64_t demand[KEA_NUM_EVENTS] = {0};
  std::string out;
  for (std::size_t pc = 0; pc < prog.code.size(); ++pc) {
    const KeaInstr& in = prog.code[pc];
    const Opcode op = keaOpcode(in);
    if (op != Opcode::SIGNAL && op != Opcode::WAIT) continue;
    const KeaEventOp e = keaDecode<KeaEventOp>(in);
    const unsigned ev = e.h.aux;
    if (ev >= KEA_NUM_EVENTS) continue;  // keaValidate rejects this separately
    if (op == Opcode::SIGNAL) {
      supply[ev] += e.value;
    } else {
      demand[ev] += e.value;
      if (demand[ev] > supply[ev]) {
        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      "Rule D violation at pc %zu: %s WAIT event=%u "
                      "threshold=%u needs %llu cumulative counts but only %llu "
                      "are SIGNALled earlier in stream order. The dispatcher "
                      "is in order, so this WAIT can wedge the whole machine "
                      "(ISA.md 5.5).\n",
                      pc, unitName(keaUnit(in)), ev, e.value,
                      static_cast<unsigned long long>(demand[ev]),
                      static_cast<unsigned long long>(supply[ev]));
        out += buf;
        // Keep the accounting monotone so later WAITs are still meaningful.
        supply[ev] = demand[ev];
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Simulator::Simulator(const KeaProgram& program, const SimConfig& config)
    : prog_(program), cfg_(config) {
  inflight_.reserve(32);
}

void Simulator::stageConstants() {
  if (prog_.const_data.empty()) return;
  mem_.dram.write(static_cast<std::int64_t>(prog_.dram.const_offset),
                  static_cast<std::int64_t>(prog_.const_data.size()),
                  prog_.const_data.data());
}

// ---------------------------------------------------------------------------
// Timing model -- every formula comes from hw_config.h
// ---------------------------------------------------------------------------

std::uint64_t Simulator::occupancyOf(const KeaInstr& in) const {
  switch (keaOpcode(in)) {
    case Opcode::NOP:
      return keaDecode<KeaNop>(in).cycles;
    case Opcode::HALT:
    case Opcode::TRACE:
      return 0;
    case Opcode::SIGNAL:
    case Opcode::WAIT:
      return 1;
    case Opcode::DMA_LD:
    case Opcode::DMA_ST: {
      const KeaDma d = keaDecode<KeaDma>(in);
      return keaDmaOccupancy(d.len0, d.n1, d.h.aux);
    }
    case Opcode::LOAD_W: {
      const KeaLoadW w = keaDecode<KeaLoadW>(in);
      return keaLoadWOccupancy(w.k_rows, (w.h.flags & KEA_LDW_F_INT4) != 0);
    }
    case Opcode::MATMUL: {
      const KeaMatmul m = keaDecode<KeaMatmul>(in);
      return keaMatmulOccupancy(m.m_inner, m.m_outer,
                                (m.h.flags & KEA_MM_F_INT4) != 0);
    }
    case Opcode::DWCONV: {
      const KeaDwconv d = keaDecode<KeaDwconv>(in);
      const std::uint32_t k = (d.h.flags & KEA_DW_F_KERNEL5) ? KEA_DWU_KERNEL_5
                                                             : KEA_DWU_KERNEL_3;
      return keaDwconvOccupancy(d.out_h, d.out_w, d.channels, k, k);
    }
    case Opcode::VQUANT: {
      const KeaVquant v = keaDecode<KeaVquant>(in);
      return keaVquantOccupancy(v.num_pixels, v.channels);
    }
    case Opcode::VADD:
      return keaVaddOccupancy(keaDecode<KeaVadd>(in).num_elems);
    case Opcode::VPOOL: {
      const KeaVpool v = keaDecode<KeaVpool>(in);
      return keaVpoolOccupancy(v.out_h, v.out_w, v.channels, v.kh, v.kw);
    }
    case Opcode::VCOPY: {
      const KeaVcopy v = keaDecode<KeaVcopy>(in);
      return keaVcopyOccupancy(v.row_bytes, v.rows);
    }
    default:
      return 0;
  }
}

std::uint64_t Simulator::latencyOf(const KeaInstr& in) const {
  const std::uint64_t occ = occupancyOf(in);
  switch (keaOpcode(in)) {
    case Opcode::MATMUL: return occ + KEA_MXU_PIPELINE_DEPTH;
    case Opcode::DWCONV: return occ + KEA_DWU_PIPELINE_DEPTH;
    case Opcode::VQUANT:
    case Opcode::VADD:
    case Opcode::VPOOL:
    case Opcode::VCOPY:  return occ + KEA_VPU_PIPELINE_DEPTH;
    default:             return occ;  // LOAD_W, DMA, NOP: latency == occupancy
  }
}

void Simulator::resourcesOf(const KeaInstr& in, int* out, int* n) const {
  *n = 0;
  const Unit u = keaUnit(in);
  switch (keaOpcode(in)) {
    case Opcode::LOAD_W: {
      out[(*n)++] = RES_MXU_WPORT;
      out[(*n)++] = (keaFlags(in) & KEA_LDW_F_BANK1) ? RES_MXU_BANK1
                                                     : RES_MXU_BANK0;
      return;
    }
    case Opcode::MATMUL: {
      out[(*n)++] = RES_MXU_ARRAY;
      out[(*n)++] = (keaFlags(in) & KEA_MM_F_BANK1) ? RES_MXU_BANK1
                                                    : RES_MXU_BANK0;
      return;
    }
    case Opcode::DWCONV: out[(*n)++] = RES_DWU_PIPE; return;
    case Opcode::VQUANT:
    case Opcode::VADD:
    case Opcode::VPOOL:
    case Opcode::VCOPY:  out[(*n)++] = RES_VPU_PIPE; return;
    case Opcode::DMA_LD:
    case Opcode::DMA_ST:
      out[(*n)++] = (u == Unit::DMA1) ? RES_DMA1_ENGINE : RES_DMA0_ENGINE;
      return;
    case Opcode::NOP:
      // NOP occupies "the unit pipe" (ISA.md §5.1).  For the MXU that is the
      // array; the weight port stays free, matching LOAD_W/MATMUL splitting.
      switch (u) {
        case Unit::MXU:  out[(*n)++] = RES_MXU_ARRAY; return;
        case Unit::DWU:  out[(*n)++] = RES_DWU_PIPE; return;
        case Unit::VPU:  out[(*n)++] = RES_VPU_PIPE; return;
        case Unit::DMA0: out[(*n)++] = RES_DMA0_ENGINE; return;
        case Unit::DMA1: out[(*n)++] = RES_DMA1_ENGINE; return;
        default: return;
      }
    default:
      return;  // HALT / SIGNAL / WAIT / TRACE hold no functional resource
  }
}

bool Simulator::unitDrained(int unit, std::uint64_t t) const {
  return t >= unit_retire_all_[unit];
}

// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------

void Simulator::accountStart(const KeaInstr& in, int unit, std::uint64_t occ) {
  const Opcode op = keaOpcode(in);
  forEachCounter([&](Counters& c) {
    c.unit[unit].instrs += 1;
    c.unit[unit].occupancy += occ;
  });

  switch (op) {
    case Opcode::MATMUL: {
      const KeaMatmul m = keaDecode<KeaMatmul>(in);
      const int bank = (m.h.flags & KEA_MM_F_BANK1) ? 1 : 0;
      const bool int4 = (m.h.flags & KEA_MM_F_INT4) != 0;
      const std::uint64_t M =
          static_cast<std::uint64_t>(m.m_inner) * m.m_outer;
      const std::uint64_t useful = M * mem_.wbank_k_rows[bank] *
                                   mem_.wbank_n_cols[bank];
      const std::uint64_t issued = M * KEA_MXU_PES;
      forEachCounter([&](Counters& c) {
        c.mxu_macs_useful += useful;
        c.mxu_macs_issued += issued;
        if (int4) c.mxu_macs_useful_int4 += useful;
        c.mxu_matmuls += 1;
      });
      break;
    }
    case Opcode::LOAD_W:
      forEachCounter([](Counters& c) { c.mxu_loadws += 1; });
      break;
    case Opcode::DWCONV: {
      const KeaDwconv d = keaDecode<KeaDwconv>(in);
      const std::uint64_t k = (d.h.flags & KEA_DW_F_KERNEL5) ? KEA_DWU_KERNEL_5
                                                             : KEA_DWU_KERNEL_3;
      const std::uint64_t pix =
          static_cast<std::uint64_t>(d.out_h) * d.out_w * k * k;
      const std::uint64_t useful = pix * d.channels;
      const std::uint64_t issued =
          pix * keaRoundUp(d.channels, KEA_DWU_LANES);
      forEachCounter([&](Counters& c) {
        c.dwu_macs_useful += useful;
        c.dwu_macs_issued += issued;
        c.dwu_convs += 1;
      });
      break;
    }
    case Opcode::VQUANT: {
      const KeaVquant v = keaDecode<KeaVquant>(in);
      const std::uint64_t e =
          static_cast<std::uint64_t>(v.num_pixels) * v.channels;
      forEachCounter([&](Counters& c) {
        c.vpu_elems += e;
        c.vpu_ops[0] += 1;
      });
      break;
    }
    case Opcode::VADD: {
      const std::uint64_t e = keaDecode<KeaVadd>(in).num_elems;
      forEachCounter([&](Counters& c) {
        c.vpu_elems += e;
        c.vpu_ops[1] += 1;
      });
      break;
    }
    case Opcode::VPOOL: {
      const KeaVpool v = keaDecode<KeaVpool>(in);
      const std::uint64_t e = static_cast<std::uint64_t>(v.out_h) * v.out_w *
                              v.channels * v.kh * v.kw;
      forEachCounter([&](Counters& c) {
        c.vpu_elems += e;
        c.vpu_ops[2] += 1;
      });
      break;
    }
    case Opcode::VCOPY: {
      const KeaVcopy v = keaDecode<KeaVcopy>(in);
      const std::uint64_t e = static_cast<std::uint64_t>(v.rows) * v.row_bytes;
      forEachCounter([&](Counters& c) {
        c.vpu_elems += e;
        c.vpu_ops[3] += 1;
      });
      break;
    }
    case Opcode::DMA_LD:
    case Opcode::DMA_ST: {
      const KeaDma d = keaDecode<KeaDma>(in);
      const std::uint64_t bytes = keaDmaBytes(d.len0, d.n1, d.h.aux);
      const int e = unit - KEA_UNIT_DMA0;
      const bool is_load = op == Opcode::DMA_LD;
      forEachCounter([&](Counters& c) {
        c.dma_bytes[e] += bytes;
        c.dma_descs[e] += 1;
        if (is_load)
          c.dram_bytes_load += bytes;
        else
          c.dram_bytes_store += bytes;
      });
      break;
    }
    default:
      break;
  }
}

void Simulator::handleTrace(const KeaInstr& in, int unit) {
  const KeaTrace t = keaDecode<KeaTrace>(in);
  const TraceKind kind = static_cast<TraceKind>(t.h.flags);
  switch (kind) {
    case TraceKind::MARKER:
      stats_.markers.push_back(Marker{cycle_, unit, t.tag, t.payload});
      break;
    case TraceKind::REGION_BEGIN: {
      Region r;
      r.tag = t.tag;
      r.payload = t.payload;
      r.unit = unit;
      r.depth = static_cast<int>(open_regions_.size());
      r.begin = cycle_;
      stats_.regions.push_back(r);
      open_regions_.push_back(stats_.regions.size() - 1);
      break;
    }
    case TraceKind::REGION_END: {
      // Regions nest per (unit, tag): close the innermost match.  The region
      // is *marked* closing here but keeps accumulating until the issuing
      // unit's pipeline has drained, so that a region bracketing a MATMUL
      // chain covers the array's 32-cycle drain instead of stopping the
      // instant the last MATMUL merely started.  TRACE itself still costs zero
      // cycles and perturbs nothing.
      bool found = false;
      for (std::size_t i = open_regions_.size(); i-- > 0;) {
        Region& r = stats_.regions[open_regions_[i]];
        if (r.unit == unit && r.tag == t.tag && !r.closing) {
          r.closing = true;
          found = true;
          break;
        }
      }
      if (!found)
        warn(0, "TRACE REGION_END on " +
                    std::string(unitName(static_cast<Unit>(unit))) + " tag " +
                    std::to_string(t.tag) + " has no matching REGION_BEGIN");
      break;
    }
    default:
      warn(0, "TRACE with unknown kind " + std::to_string(t.h.flags));
      break;
  }
}

// ---------------------------------------------------------------------------
// DRAM port arbitration (MICROARCH.md §6.3)
// ---------------------------------------------------------------------------

void Simulator::advanceDma() {
  // Overhead phase consumes a cycle and requests no bandwidth.
  bool in_data[KEA_NUM_DMA_ENGINES] = {false, false};
  for (int e = 0; e < KEA_NUM_DMA_ENGINES; ++e) {
    DmaEngine& d = dma_[e];
    if (!d.active) continue;
    if (d.overhead_remaining > 0) {
      --d.overhead_remaining;
      continue;  // an engine never does overhead and data in the same cycle
    }
    in_data[e] = d.bytes_remaining > 0;
  }

  int active = 0;
  for (int e = 0; e < KEA_NUM_DMA_ENGINES; ++e) active += in_data[e] ? 1 : 0;

  std::uint64_t grant[KEA_NUM_DMA_ENGINES] = {0, 0};
  if (active > 0) {
    const std::uint64_t share = KEA_DRAM_BYTES_PER_CYCLE / active;
    std::uint64_t remainder = KEA_DRAM_BYTES_PER_CYCLE % active;
    for (int e = 0; e < KEA_NUM_DMA_ENGINES; ++e) {  // ascending id: DMA0 first
      if (!in_data[e]) continue;
      grant[e] = share + remainder;  // odd remainder to the lower unit id
      remainder = 0;
      if (grant[e] > static_cast<std::uint64_t>(KEA_DMA_ENGINE_BYTES_PER_CYCLE))
        grant[e] = KEA_DMA_ENGINE_BYTES_PER_CYCLE;
    }
  }

  const bool contended = active > 1;
  for (int e = 0; e < KEA_NUM_DMA_ENGINES; ++e) {
    if (!in_data[e]) continue;
    DmaEngine& d = dma_[e];
    d.bytes_remaining -= std::min(grant[e], d.bytes_remaining);
    const double lost =
        1.0 - static_cast<double>(grant[e]) / KEA_DMA_ENGINE_BYTES_PER_CYCLE;
    forEachCounter([&](Counters& c) {
      c.dram_data_cycles[e] += 1;
      if (contended) c.dram_contended_cycles[e] += 1;
      c.dram_cycles_lost_to_contention += lost;
    });
  }
  if (active > 0) forEachCounter([](Counters& c) { c.dram_busy_cycles += 1; });

  // Retire completed descriptors.
  for (int e = 0; e < KEA_NUM_DMA_ENGINES; ++e) {
    DmaEngine& d = dma_[e];
    if (!d.active) continue;
    if (d.overhead_remaining != 0 || d.bytes_remaining != 0) continue;
    const int unit = KEA_UNIT_DMA0 + e;
    d.active = false;
    resource_free_[RES_DMA0_ENGINE + e] = cycle_ + 1;
    unit_retire_all_[unit] = std::max(d.prev_retire_all, cycle_ + 1);
    if (d.hazard_idx != static_cast<std::size_t>(-1)) {
      inflight_[d.hazard_idx].retire = cycle_ + 1;
      d.hazard_idx = static_cast<std::size_t>(-1);
    }
    if (d.trace_idx != static_cast<std::size_t>(-1)) {
      trace_[d.trace_idx].retire = cycle_ + 1;
      d.trace_idx = static_cast<std::size_t>(-1);
    }
  }
}

// ---------------------------------------------------------------------------
// Per-cycle accounting
// ---------------------------------------------------------------------------

void Simulator::perCycleAccounting() {
  static const int kUnitRes[KEA_NUM_QUEUES][4] = {
      {RES_MXU_ARRAY, RES_MXU_WPORT, RES_MXU_BANK0, RES_MXU_BANK1},
      {RES_DWU_PIPE, -1, -1, -1},
      {RES_VPU_PIPE, -1, -1, -1},
      {RES_DMA0_ENGINE, -1, -1, -1},
      {RES_DMA1_ENGINE, -1, -1, -1},
  };

  bool busy[KEA_NUM_QUEUES];
  for (int u = 0; u < KEA_NUM_QUEUES; ++u) {
    busy[u] = false;
    for (int i = 0; i < 4; ++i) {
      const int r = kUnitRes[u][i];
      if (r >= 0 && resource_free_[r] > cycle_) busy[u] = true;
    }
  }

  forEachCounter([&](Counters& c) {
    c.cycles += 1;
    for (int u = 0; u < KEA_NUM_QUEUES; ++u) {
      UnitStats& s = c.unit[u];
      const std::uint64_t depth = queue_[u].size();
      s.queue_occupancy += depth;
      if (depth > s.max_queue_depth) s.max_queue_depth = depth;
      if (started_[u]) s.start_cycles += 1;
      if (busy[u])
        s.busy_cycles += 1;
      else if (head_stall_[u] == StallReason::Semaphore ||
               head_stall_[u] == StallReason::Drain)
        s.stall_sem_cycles += 1;
      else if (head_stall_[u] == StallReason::Resource || !queue_[u].empty())
        s.stall_res_cycles += 1;
      else
        s.idle_cycles += 1;
    }
  });
}

// ---------------------------------------------------------------------------
// The main loop
// ---------------------------------------------------------------------------

SimResult Simulator::run() {
  SimResult res;

  // --- load-time checks --------------------------------------------------
  if (const char* e = prog_.validate()) {
    res.status = SimStatus::LoadError;
    res.message = e;
    return res;
  }
  {
    bool have_halt = false;
    for (std::size_t i = 0; i < prog_.code.size(); ++i) {
      if (keaOpcode(prog_.code[i]) == Opcode::HALT) {
        have_halt = true;
        if (i + 1 != prog_.code.size()) {
          res.status = SimStatus::LoadError;
          res.message = "HALT at pc " + std::to_string(i) +
                        " is not the last instruction";
          return res;
        }
      }
    }
    if (!have_halt) {
      res.status = SimStatus::LoadError;
      res.message = "program has no HALT";
      return res;
    }
  }
  if (cfg_.check_rule_d) {
    const std::string rd = checkRuleD(prog_);
    if (!rd.empty()) {
      res.status = SimStatus::RuleDViolation;
      res.message = rd;
      res.diagnostics.push_back(rd);
      return res;
    }
  }

  pc_ = prog_.entry_pc;
  std::uint64_t total_cycles = 0;
  std::uint64_t ctrl_busy_until = 0;

  for (;;) {
    for (int u = 0; u < KEA_NUM_QUEUES; ++u) {
      started_[u] = false;
      head_stall_[u] = StallReason::None;
    }
    bool dispatched = false;
    pruneInflight();

    // --- Phase A: SIGNALs.  Increments land before any waiter is evaluated.
    for (int u = 0; u < KEA_NUM_QUEUES && status_ == SimStatus::Ok; ++u) {
      if (queue_[u].empty()) continue;
      QueueEntry& q = queue_[u].front();
      const KeaInstr& in = prog_.code[q.pc];
      if (keaOpcode(in) != Opcode::SIGNAL) continue;
      if (cycle_ < unit_next_issue_[u]) {
        head_stall_[u] = StallReason::Resource;
        continue;
      }
      // A SIGNAL is a local drain barrier: it may not retire until every
      // instruction this unit issued before it has fully retired.
      if (!unitDrained(u, cycle_)) {
        head_stall_[u] = StallReason::Drain;
        ++q.stall_sem;
        continue;
      }
      const KeaEventOp e = keaDecode<KeaEventOp>(in);
      const std::uint64_t next =
          static_cast<std::uint64_t>(event_[e.h.aux]) + e.value;
      if (next > KEA_EVENT_MAX) {
        status_ = SimStatus::EventOverflow;
        message_ = "pc " + std::to_string(q.pc) + ": SIGNAL drives event " +
                   std::to_string(e.h.aux) + " to " + std::to_string(next) +
                   " (> KEA_EVENT_MAX). A producer is running unbounded ahead "
                   "of its consumer.";
        break;
      }
      event_[e.h.aux] = static_cast<std::uint32_t>(next);
      if (q.trace_idx != static_cast<std::size_t>(-1)) {
        trace_[q.trace_idx].start = cycle_;
        trace_[q.trace_idx].retire = cycle_ + 1;
        trace_[q.trace_idx].stall_sem = q.stall_sem;
        trace_[q.trace_idx].stall_res = q.stall_res;
      }
      forEachCounter([&](Counters& c) { c.unit[u].instrs += 1; });
      queue_[u].pop_front();
      started_[u] = true;
      unit_next_issue_[u] = cycle_ + 1;
    }

    // --- Phase B: waiters, ascending unit id (ISA.md §5.3).
    for (int u = 0; u < KEA_NUM_QUEUES && status_ == SimStatus::Ok; ++u) {
      if (started_[u] || queue_[u].empty()) continue;
      QueueEntry& q = queue_[u].front();
      const KeaInstr& in = prog_.code[q.pc];
      if (keaOpcode(in) != Opcode::WAIT) continue;
      if (cycle_ < unit_next_issue_[u]) {
        head_stall_[u] = StallReason::Resource;
        continue;
      }
      const KeaEventOp e = keaDecode<KeaEventOp>(in);
      if (event_[e.h.aux] < e.value) {
        head_stall_[u] = StallReason::Semaphore;
        ++q.stall_sem;
        continue;
      }
      event_[e.h.aux] -= e.value;  // counting acquire: consumes exactly `thr`
      if (q.trace_idx != static_cast<std::size_t>(-1)) {
        trace_[q.trace_idx].start = cycle_;
        trace_[q.trace_idx].retire = cycle_ + 1;
        trace_[q.trace_idx].stall_sem = q.stall_sem;
        trace_[q.trace_idx].stall_res = q.stall_res;
      }
      forEachCounter([&](Counters& c) { c.unit[u].instrs += 1; });
      queue_[u].pop_front();
      started_[u] = true;
      unit_next_issue_[u] = cycle_ + 1;
    }

    // --- Phase C: start everything else.
    for (int u = 0; u < KEA_NUM_QUEUES && status_ == SimStatus::Ok; ++u) {
      if (started_[u] || queue_[u].empty()) continue;
      QueueEntry& q = queue_[u].front();
      const KeaInstr& in = prog_.code[q.pc];
      const Opcode op = keaOpcode(in);
      if (op == Opcode::SIGNAL || op == Opcode::WAIT) continue;
      if (cycle_ < unit_next_issue_[u]) {
        head_stall_[u] = StallReason::Resource;
        ++q.stall_res;
        continue;
      }

      if (op == Opcode::TRACE) {
        handleTrace(in, u);
        if (q.trace_idx != static_cast<std::size_t>(-1)) {
          trace_[q.trace_idx].start = cycle_;
          trace_[q.trace_idx].retire = cycle_;
        }
        queue_[u].pop_front();
        started_[u] = true;
        unit_next_issue_[u] = cycle_ + 1;
        continue;
      }

      int rs[4];
      int nres = 0;
      resourcesOf(in, rs, &nres);
      bool blocked = false;
      for (int i = 0; i < nres; ++i)
        if (resource_free_[rs[i]] > cycle_) blocked = true;
      if (blocked) {
        head_stall_[u] = StallReason::Resource;
        ++q.stall_res;
        continue;
      }

      const std::uint64_t occ = occupancyOf(in);
      const std::uint64_t lat = latencyOf(in);
      exec_retire_ = cycle_ + lat;

      if (op == Opcode::DMA_LD || op == Opcode::DMA_ST) {
        const KeaDma d = keaDecode<KeaDma>(in);
        const int e = u - KEA_UNIT_DMA0;
        DmaEngine& eng = dma_[e];
        eng.active = true;
        eng.pc = q.pc;
        eng.start_cycle = cycle_;
        eng.overhead_remaining =
            static_cast<std::uint64_t>(KEA_DMA_DESC_OVERHEAD_CYCLES) +
            static_cast<std::uint64_t>(d.n1) * d.h.aux *
                KEA_DMA_ROW_OVERHEAD_CYCLES;
        eng.bytes_remaining = keaDmaBytes(d.len0, d.n1, d.h.aux);
        eng.prev_retire_all = unit_retire_all_[u];
        eng.trace_idx = q.trace_idx;
        // The engine is held for an amount of time that depends on DRAM port
        // contention, so it is released by advanceDma(), not by a precomputed
        // free cycle.
        resource_free_[RES_DMA0_ENGINE + e] = UINT64_MAX;
        unit_retire_all_[u] = UINT64_MAX;
        execute(q.pc, in, u);
      } else {
        for (int i = 0; i < nres; ++i) resource_free_[rs[i]] = cycle_ + occ;
        unit_retire_all_[u] = std::max(unit_retire_all_[u], cycle_ + lat);
        execute(q.pc, in, u);
      }

      if (q.trace_idx != static_cast<std::size_t>(-1)) {
        trace_[q.trace_idx].start = cycle_;
        trace_[q.trace_idx].retire = cycle_ + lat;
        trace_[q.trace_idx].stall_sem = q.stall_sem;
        trace_[q.trace_idx].stall_res = q.stall_res;
      }
      accountStart(in, u, occ);
      queue_[u].pop_front();
      started_[u] = true;
      unit_next_issue_[u] = cycle_ + KEA_ISSUE_COST_CYCLES;
    }

    if (status_ != SimStatus::Ok) break;

    // --- Phase D: dispatch (1 instruction/cycle, strictly in program order).
    if (!halted_ && cycle_ >= ctrl_busy_until && pc_ < prog_.code.size()) {
      const KeaInstr& in = prog_.code[pc_];
      const Unit tu = keaUnit(in);
      const Opcode op = keaOpcode(in);
      if (tu == Unit::CTRL) {
        // CTRL instructions retire at dispatch and never enter a queue.
        bool advance = true;
        switch (op) {
          case Opcode::HALT:
            halted_ = true;
            exit_code_ = keaDecode<KeaHalt>(in).exit_code;
            break;
          case Opcode::TRACE:
            handleTrace(in, KEA_UNIT_CTRL);
            break;
          case Opcode::NOP:
            ctrl_busy_until = cycle_ + keaDecode<KeaNop>(in).cycles;
            break;
          case Opcode::SIGNAL: {
            const KeaEventOp e = keaDecode<KeaEventOp>(in);
            const std::uint64_t next =
                static_cast<std::uint64_t>(event_[e.h.aux]) + e.value;
            if (next > KEA_EVENT_MAX) {
              status_ = SimStatus::EventOverflow;
              message_ = "pc " + std::to_string(pc_) +
                         ": CTRL SIGNAL drives event " +
                         std::to_string(e.h.aux) + " past KEA_EVENT_MAX";
              advance = false;
            } else {
              event_[e.h.aux] = static_cast<std::uint32_t>(next);
            }
            break;
          }
          case Opcode::WAIT: {
            const KeaEventOp e = keaDecode<KeaEventOp>(in);
            if (event_[e.h.aux] >= e.value)
              event_[e.h.aux] -= e.value;
            else
              advance = false;  // wedges the dispatcher; that is the semantics
            break;
          }
          default:
            break;
        }
        if (advance) {
          ++pc_;
          dispatched = true;
          forEachCounter([](Counters& c) { c.dispatched += 1; });
        }
      } else {
        const int qi = static_cast<int>(tu);
        if (queue_[qi].size() < static_cast<std::size_t>(KEA_QUEUE_DEPTH)) {
          QueueEntry q;
          q.pc = pc_;
          q.issue_cycle = cycle_;
          if (cfg_.trace) {
            TraceRecord t;
            t.pc = pc_;
            t.unit = static_cast<std::uint8_t>(qi);
            t.opcode = static_cast<std::uint8_t>(op);
            t.issue = cycle_;
            trace_.push_back(t);
            q.trace_idx = trace_.size() - 1;
          }
          queue_[qi].push_back(q);
          ++pc_;
          dispatched = true;
          forEachCounter([](Counters& c) { c.dispatched += 1; });
        } else {
          // The only implicit backpressure in the machine (MICROARCH.md §1.1).
          forEachCounter([&](Counters& c) {
            c.dispatch_stall[qi] += 1;
            c.dispatch_stall_total += 1;
          });
        }
      }
    }

    if (status_ != SimStatus::Ok) break;

    // --- Phase E: advance the DMA engines and arbitrate the DRAM port.
    advanceDma();

    // --- Phase F: cycle accounting.
    perCycleAccounting();

    // Close regions whose REGION_END has retired and whose issuing unit has
    // now drained.
    for (std::size_t i = open_regions_.size(); i-- > 0;) {
      Region& r = stats_.regions[open_regions_[i]];
      if (!r.closing) continue;
      if (r.unit < KEA_NUM_QUEUES && unit_retire_all_[r.unit] > cycle_ + 1)
        continue;
      r.end = cycle_ + 1;
      r.closed = true;
      r.c.cycles = r.end - r.begin;
      open_regions_.erase(open_regions_.begin() +
                          static_cast<std::ptrdiff_t>(i));
    }

    // --- Phase G: termination and error checks.
    bool queues_empty = true;
    for (int u = 0; u < KEA_NUM_QUEUES; ++u)
      if (!queue_[u].empty()) queues_empty = false;
    bool units_idle = true;
    for (int r = 0; r < RES_COUNT; ++r)
      if (resource_free_[r] > cycle_ + 1) units_idle = false;
    for (int u = 0; u < KEA_NUM_QUEUES; ++u)
      if (unit_retire_all_[u] > cycle_ + 1) units_idle = false;

    if (halted_ && queues_empty && units_idle) {
      total_cycles = cycle_ + 1;
      break;
    }

    // "In flight" must include pipeline drain, not just resource occupancy: a
    // MATMUL whose array occupancy has ended is still 32 cycles away from
    // letting a SIGNAL retire, and that is progress, not deadlock.
    bool any_busy = false;
    for (int r = 0; r < RES_COUNT; ++r)
      if (resource_free_[r] > cycle_) any_busy = true;
    for (int u = 0; u < KEA_NUM_QUEUES; ++u)
      if (unit_retire_all_[u] > cycle_) any_busy = true;
    bool any_started = false;
    for (int u = 0; u < KEA_NUM_QUEUES; ++u) any_started |= started_[u];

    if (!any_started && !dispatched && !any_busy && cycle_ >= ctrl_busy_until) {
      // Nothing ran, nothing was dispatched, nothing is in flight.  If any
      // queue head is a blocked WAIT the machine can never make progress.
      std::string blocked;
      if (!halted_ && pc_ < prog_.code.size() &&
          keaUnit(prog_.code[pc_]) == Unit::CTRL &&
          keaOpcode(prog_.code[pc_]) == Opcode::WAIT) {
        const KeaEventOp e = keaDecode<KeaEventOp>(prog_.code[pc_]);
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "  CTRL pc=%u WAIT event=%u threshold=%u (event=%u)\n",
                      pc_, e.h.aux, e.value, event_[e.h.aux]);
        blocked += buf;
      }
      for (int u = 0; u < KEA_NUM_QUEUES; ++u) {
        if (queue_[u].empty()) continue;
        const KeaInstr& in = prog_.code[queue_[u].front().pc];
        if (keaOpcode(in) != Opcode::WAIT) continue;
        const KeaEventOp e = keaDecode<KeaEventOp>(in);
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "  %-4s pc=%u WAIT event=%u threshold=%u (event=%u)\n",
                      unitName(static_cast<Unit>(u)), queue_[u].front().pc,
                      e.h.aux, e.value, event_[e.h.aux]);
        blocked += buf;
      }
      if (!blocked.empty()) {
        status_ = SimStatus::Deadlock;
        message_ = "DEADLOCK at cycle " + std::to_string(cycle_) +
                   ". Blocked units:\n" + blocked + "Event counters:";
        for (int e = 0; e < KEA_NUM_EVENTS; ++e) {
          if (event_[e] == 0) continue;
          message_ += " ev" + std::to_string(e) + "=" +
                      std::to_string(event_[e]);
        }
        message_ += "\n";
        total_cycles = cycle_ + 1;
        break;
      }
    }

    ++cycle_;
    if (cfg_.max_cycles && cycle_ >= cfg_.max_cycles) {
      status_ = SimStatus::MaxCycles;
      message_ = "hit --max-cycles " + std::to_string(cfg_.max_cycles);
      total_cycles = cycle_;
      break;
    }
  }

  if (total_cycles == 0) total_cycles = cycle_ + 1;

  // Close any region still open when the machine stopped.
  for (std::size_t idx : open_regions_) {
    Region& r = stats_.regions[idx];
    r.end = total_cycles;
    r.c.cycles = r.end > r.begin ? r.end - r.begin : 0;
    r.closed = r.closing;
    if (!r.closing)
      warn(0, "TRACE region tag " + std::to_string(r.tag) + " on " +
                  std::string(unitName(static_cast<Unit>(r.unit))) +
                  " was never closed");
  }
  open_regions_.clear();
  stats_.global.cycles = total_cycles;

  if (poison_reads_ > 0 && !cfg_.strict_poison)
    diagnostics_.push_back(std::to_string(poison_reads_) +
                           " read(s) of never-written scratchpad (poison). "
                           "Re-run with --strict-poison to make this fatal.");
  if (hazards_ > 0 && !cfg_.strict_hazards)
    diagnostics_.push_back(
        std::to_string(hazards_) +
        " cross-unit read(s) of data whose producing instruction had not "
        "retired. Re-run with --strict-hazards to make this fatal.");

  res.status = status_;
  res.message = message_;
  res.exit_code = exit_code_;
  res.cycles = total_cycles;
  res.stats = stats_;
  res.trace = trace_;
  for (const std::string& d : diagnostics_) res.diagnostics.push_back(d);
  return res;
}

}  // namespace sim
}  // namespace kea
