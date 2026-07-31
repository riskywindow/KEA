// SPDX-License-Identifier: Apache-2.0
//
// kea/sim/simulator.h --- the KEA-1 cycle-approximate simulator.
//
// Structure:
//
//   * a bit-exact functional model  (sim/src/functional.cpp)
//   * a concurrent timing model     (sim/src/simulator.cpp)
//   * instrumentation               (sim/src/stats.cpp)
//
// The timing model follows MICROARCH.md §8's normative cycle-stepped
// algorithm.  Every latency, occupancy and bandwidth comes from
// include/kea/hw_config.h; nothing is hardcoded here.
//
// Determinism: units are always serviced in ascending unit id, event
// increments are applied before waiters are evaluated, DRAM remainders go to
// the lower unit id, and no unordered container appears anywhere in the
// scheduling path.  Two runs of the same program produce identical cycle
// counts, identical traces and identical memory images.

#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "kea/isa.h"
#include "kea/program.h"
#include "kea/sim/memory.h"
#include "kea/sim/stats.h"

namespace kea {
namespace sim {

// ---------------------------------------------------------------------------
// Resources.  MICROARCH.md §2: a unit may start its next instruction while a
// previous one still occupies a *different* resource.  Splitting the MXU into
// ARRAY / WPORT / BANK0 / BANK1 is what makes weight double buffering free.
// ---------------------------------------------------------------------------
enum Resource {
  RES_MXU_ARRAY = 0,
  RES_MXU_WPORT,
  RES_MXU_BANK0,
  RES_MXU_BANK1,
  RES_DWU_PIPE,
  RES_VPU_PIPE,
  RES_DMA0_ENGINE,
  RES_DMA1_ENGINE,
  RES_COUNT
};

const char* resourceName(int r);

struct SimConfig {
  /// Hard stop, so a broken program cannot hang a CI run.  0 means unlimited.
  std::uint64_t max_cycles = 0;
  /// Record an issue/start/retire row per instruction.
  bool trace = false;
  /// Statically verify Rule D (ISA.md §5.5) at load and fail if it is broken.
  bool check_rule_d = true;
  /// Promote reads of never-written scratchpad from a warning to a hard fault.
  bool strict_poison = false;
  /// Promote cross-unit read-during-in-flight-write hazards to a hard fault.
  bool strict_hazards = false;
  /// Cap on how many distinct diagnostic messages are retained.
  std::size_t max_diagnostics = 64;
};

enum class SimStatus {
  Ok,
  LoadError,
  RuleDViolation,
  Deadlock,
  EventOverflow,
  Fault,
  MaxCycles,
};

const char* statusName(SimStatus s);

/// Why an instruction was not able to start on a given cycle.
enum class StallReason : std::uint8_t { None = 0, Semaphore, Resource, Drain };

struct TraceRecord {
  std::uint32_t pc = 0;
  std::uint8_t unit = 0;
  std::uint8_t opcode = 0;
  std::uint64_t issue = 0;    ///< cycle the dispatcher pushed it into the queue
  std::uint64_t start = 0;    ///< cycle it began executing (or retired, for WAIT)
  std::uint64_t retire = 0;   ///< cycle its last architectural write is visible
  std::uint64_t stall_sem = 0;
  std::uint64_t stall_res = 0;
};

struct SimResult {
  SimStatus status = SimStatus::Ok;
  std::string message;
  std::uint32_t exit_code = 0;
  std::uint64_t cycles = 0;
  Stats stats;
  std::vector<TraceRecord> trace;
  std::vector<std::string> diagnostics;

  bool ok() const { return status == SimStatus::Ok; }
};

/// Static Rule D check (ISA.md §5.5).  Walks the stream in dispatch order and
/// verifies that every WAIT's counts have already been supplied by SIGNALs at
/// strictly earlier stream positions.  Straight-line code makes this exact.
/// Returns an empty string when the program is clean.
std::string checkRuleD(const KeaProgram& prog);

class Simulator {
 public:
  Simulator(const KeaProgram& program, const SimConfig& config);

  /// Stage the KEAF CONST image into the DRAM arena.  Call before run().
  void stageConstants();

  Machine& machine() { return mem_; }
  const Machine& machine() const { return mem_; }

  SimResult run();

 private:
  // --- queue -------------------------------------------------------------
  struct QueueEntry {
    std::uint32_t pc = 0;
    std::uint64_t issue_cycle = 0;
    std::uint64_t stall_sem = 0;
    std::uint64_t stall_res = 0;
    std::size_t trace_idx = static_cast<std::size_t>(-1);
  };

  struct DmaEngine {
    bool active = false;
    std::uint32_t pc = 0;
    std::uint64_t start_cycle = 0;
    std::uint64_t overhead_remaining = 0;
    std::uint64_t bytes_remaining = 0;
    std::uint64_t prev_retire_all = 0;
    std::size_t trace_idx = static_cast<std::size_t>(-1);
    std::size_t hazard_idx = static_cast<std::size_t>(-1);
  };

  /// A write that has started but whose last architectural write is not yet
  /// visible.  A read of an overlapping range by a *different* unit before
  /// `retire` means the program is missing a SIGNAL/WAIT pair.
  struct InflightWrite {
    int space = 0;  ///< Space enum value
    std::int64_t lo = 0;
    std::int64_t hi = 0;
    std::uint64_t retire = 0;
    std::uint32_t pc = 0;
    int unit = 0;
    bool live = false;
  };

  // --- timing ------------------------------------------------------------
  std::uint64_t occupancyOf(const KeaInstr& in) const;
  std::uint64_t latencyOf(const KeaInstr& in) const;
  void resourcesOf(const KeaInstr& in, int* out, int* n) const;
  bool unitDrained(int unit, std::uint64_t t) const;

  // --- functional (sim/src/functional.cpp) --------------------------------
  void execute(std::uint32_t pc, const KeaInstr& in, int unit);
  void execLoadW(std::uint32_t pc, const KeaInstr& in);
  void execMatmul(std::uint32_t pc, const KeaInstr& in);
  void execDwconv(std::uint32_t pc, const KeaInstr& in);
  void execVquant(std::uint32_t pc, const KeaInstr& in);
  void execVadd(std::uint32_t pc, const KeaInstr& in);
  void execVpool(std::uint32_t pc, const KeaInstr& in);
  void execVcopy(std::uint32_t pc, const KeaInstr& in);
  void execDma(std::uint32_t pc, const KeaInstr& in, int unit);

  // --- diagnostics -------------------------------------------------------
  void fault(std::uint32_t pc, const std::string& msg);
  void warn(std::uint32_t pc, const std::string& msg);
  void notePoison(std::uint32_t pc, const char* space, std::int64_t addr,
                  std::int64_t len);

  bool checkSpm(std::uint32_t pc, const Scratchpad& s, const char* name,
                std::int64_t addr, std::int64_t len);
  bool checkAcc(std::uint32_t pc, std::int64_t word, std::int64_t count);
  bool checkDram(std::uint32_t pc, std::int64_t addr, std::int64_t len);
  bool readableSpm(std::uint32_t pc, const Scratchpad& s, const char* name,
                   std::int64_t addr, std::int64_t len);
  bool readableAcc(std::uint32_t pc, std::int64_t word, std::int64_t count);

  // --- hazards -----------------------------------------------------------
  void addInflightWrite(int space, std::int64_t lo, std::int64_t hi,
                        std::uint64_t retire, std::uint32_t pc, int unit,
                        std::size_t* slot_out);
  void checkHazard(int space, std::int64_t lo, std::int64_t hi,
                   std::uint32_t pc, int unit);
  void pruneInflight();

  // --- counters ----------------------------------------------------------
  /// Apply `fn` to the global counters and to every currently open region.
  template <typename Fn>
  void forEachCounter(Fn fn) {
    fn(stats_.global);
    for (std::size_t idx : open_regions_) fn(stats_.regions[idx].c);
  }

  void accountStart(const KeaInstr& in, int unit, std::uint64_t occ);
  void handleTrace(const KeaInstr& in, int unit);
  void perCycleAccounting();
  void advanceDma();

  // --- state -------------------------------------------------------------
  const KeaProgram& prog_;
  SimConfig cfg_;
  Machine mem_;
  Stats stats_;

  std::uint64_t cycle_ = 0;
  /// Cycle at which the currently-executing instruction's writes become
  /// architecturally visible.  Set by the timing engine immediately before it
  /// calls execute(), and consumed by the hazard tracker.
  std::uint64_t exec_retire_ = 0;
  std::uint32_t pc_ = 0;
  bool halted_ = false;
  std::uint32_t exit_code_ = 0;

  std::deque<QueueEntry> queue_[KEA_NUM_QUEUES];
  std::uint64_t resource_free_[RES_COUNT] = {0};
  std::uint64_t unit_next_issue_[KEA_NUM_QUEUES] = {0};
  std::uint64_t unit_retire_all_[KEA_NUM_QUEUES] = {0};
  std::uint32_t event_[KEA_NUM_EVENTS] = {0};
  DmaEngine dma_[KEA_NUM_DMA_ENGINES];

  std::vector<std::size_t> open_regions_;
  std::vector<InflightWrite> inflight_;

  std::vector<TraceRecord> trace_;
  std::vector<std::string> diagnostics_;
  std::uint64_t poison_reads_ = 0;
  std::uint64_t hazards_ = 0;

  SimStatus status_ = SimStatus::Ok;
  std::string message_;

  // Per-cycle scratch, reset at the top of every cycle.
  bool started_[KEA_NUM_QUEUES] = {false};
  StallReason head_stall_[KEA_NUM_QUEUES] = {StallReason::None};
};

}  // namespace sim
}  // namespace kea
