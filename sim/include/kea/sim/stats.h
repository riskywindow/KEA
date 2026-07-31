// SPDX-License-Identifier: Apache-2.0
//
// kea/sim/stats.h --- the simulator's instrumentation.
//
// Every counter here is a deliverable: the end-to-end write-up quotes these
// numbers, so each one has an exact definition and none of them are estimates.
// See docs/SIMULATOR.md §"Reading the stats" for what each field means and
// which of them are affected by the modelling approximations in
// MICROARCH.md §10.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kea/hw_config.h"

namespace kea {
namespace sim {

/// Per-unit cycle accounting.  `busy`, `stall_sem`, `stall_res` and `idle`
/// PARTITION every cycle of the run in that priority order, so they always sum
/// to the total cycle count:
///
///   busy       at least one of the unit's resources was occupied
///   stall_sem  otherwise, the queue head was a blocked WAIT or a SIGNAL
///              still waiting for the unit's pipeline to drain
///   stall_res  otherwise, the queue head was ready but could not start
///   idle       otherwise (queue empty)
///
/// A unit that is draining a MATMUL while its next instruction sits on a WAIT
/// therefore counts as *busy*, not stalled -- it is still doing useful work.
struct UnitStats {
  std::uint64_t instrs = 0;            ///< instructions started (or retired, for WAIT/SIGNAL/TRACE)
  std::uint64_t occupancy = 0;         ///< sum of per-instruction occupancy
  std::uint64_t busy_cycles = 0;
  std::uint64_t stall_sem_cycles = 0;
  std::uint64_t stall_res_cycles = 0;
  std::uint64_t idle_cycles = 0;
  std::uint64_t start_cycles = 0;      ///< cycles in which an instruction started
  std::uint64_t queue_occupancy = 0;   ///< sum of queue depth over cycles (for avg)
  std::uint64_t max_queue_depth = 0;
};

/// Everything the simulator accumulates, either globally or per TRACE region.
struct Counters {
  std::uint64_t cycles = 0;

  UnitStats unit[KEA_NUM_QUEUES];

  /// Dispatcher stall cycles, attributed to the unit whose queue was full.
  std::uint64_t dispatch_stall[KEA_NUM_QUEUES] = {0, 0, 0, 0, 0};
  std::uint64_t dispatch_stall_total = 0;
  std::uint64_t dispatched = 0;

  // --- MXU ---------------------------------------------------------------
  std::uint64_t mxu_macs_useful = 0;   ///< M * k_rows * n_cols, resident tile
  std::uint64_t mxu_macs_issued = 0;   ///< M * 256, i.e. what the array ran
  std::uint64_t mxu_macs_useful_int4 = 0;  ///< subset of the above run in int4
  std::uint64_t mxu_matmuls = 0;
  std::uint64_t mxu_loadws = 0;

  // --- DWU ---------------------------------------------------------------
  std::uint64_t dwu_macs_useful = 0;
  std::uint64_t dwu_macs_issued = 0;
  std::uint64_t dwu_convs = 0;

  // --- VPU ---------------------------------------------------------------
  std::uint64_t vpu_elems = 0;         ///< lane-elements of real work
  std::uint64_t vpu_ops[4] = {0, 0, 0, 0};  ///< VQUANT, VADD, VPOOL, VCOPY counts

  // --- DMA / DRAM --------------------------------------------------------
  std::uint64_t dma_bytes[KEA_NUM_DMA_ENGINES] = {0, 0};
  std::uint64_t dma_descs[KEA_NUM_DMA_ENGINES] = {0, 0};
  std::uint64_t dram_bytes_load = 0;
  std::uint64_t dram_bytes_store = 0;
  /// Cycles each engine spent in its data phase (i.e. actually requesting the
  /// DRAM port) and, of those, cycles in which it had to share the port.
  std::uint64_t dram_data_cycles[KEA_NUM_DMA_ENGINES] = {0, 0};
  std::uint64_t dram_contended_cycles[KEA_NUM_DMA_ENGINES] = {0, 0};
  /// Engine-cycles of DRAM bandwidth lost purely to sharing the port, i.e.
  /// sum over cycles of (1 - granted/KEA_DMA_ENGINE_BYTES_PER_CYCLE).
  double dram_cycles_lost_to_contention = 0.0;
  /// Cycles in which at least one engine moved data.
  std::uint64_t dram_busy_cycles = 0;

  std::uint64_t dramBytes() const { return dram_bytes_load + dram_bytes_store; }
  std::uint64_t opsUseful() const {
    return 2 * (mxu_macs_useful + dwu_macs_useful);
  }
  std::uint64_t opsIssued() const {
    return 2 * (mxu_macs_issued + dwu_macs_issued);
  }
};

/// One TRACE REGION_BEGIN/REGION_END pair.  Regions nest per (unit, tag);
/// a region's counters cover every unit's activity inside its cycle window,
/// which is what per-layer attribution wants (a layer opened on the MXU still
/// owns the DMA traffic issued for it).
struct Region {
  std::uint32_t tag = 0;
  std::uint32_t payload = 0;
  int unit = 0;
  int depth = 0;             ///< nesting depth at REGION_BEGIN
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
  bool closed = false;
  /// REGION_END has retired but the issuing unit's pipeline has not drained
  /// yet, so the region is still accumulating.  See docs/SIMULATOR.md
  /// "Ambiguities": TRACE costs zero cycles, but a REGION_END that closed the
  /// instant it reached the queue head would stop counting up to a whole
  /// pipeline depth before the work it brackets actually finished.
  bool closing = false;
  Counters c;
};

/// A point TRACE marker.
struct Marker {
  std::uint64_t cycle = 0;
  int unit = 0;
  std::uint32_t tag = 0;
  std::uint32_t payload = 0;
};

struct Stats {
  Counters global;
  std::vector<Region> regions;
  std::vector<Marker> markers;
};

/// Roofline position of a Counters block, computed exactly as MICROARCH.md
/// §9.2 specifies.
struct Roofline {
  double intensity = 0.0;      ///< useful ops per DRAM byte
  double achieved = 0.0;       ///< useful ops/s
  double attainable = 0.0;     ///< min(peak, intensity * peak DRAM BW)
  double efficiency = 0.0;     ///< achieved / attainable
  double padding_loss = 0.0;   ///< ops_useful / ops_issued
  double peak_ops = 0.0;       ///< the compute roof used
  bool memory_bound = false;
};

Roofline computeRoofline(const Counters& c);

/// Human-readable report.
std::string formatReport(const Stats& s, std::uint64_t total_cycles);

/// Machine-readable report (`--stats-json`).  Hand-rolled so `sim/` has no
/// dependency on `runtime/`.
std::string formatJson(const Stats& s, std::uint64_t total_cycles,
                       const std::string& status, std::uint32_t exit_code,
                       const std::vector<std::string>& diagnostics);

}  // namespace sim
}  // namespace kea
