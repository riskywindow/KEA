// SPDX-License-Identifier: Apache-2.0
//
// Semaphore semantics, Rule D, deadlock detection and determinism.

#include <cstdio>
#include <string>
#include <vector>

#include "kea/sim/program_builder.h"
#include "kea/sim/simulator.h"
#include "test_util.h"

using namespace kea;
using namespace kea::sim;

namespace {

const TraceRecord* findPc(const SimResult& r, std::uint32_t pc) {
  for (const TraceRecord& t : r.trace)
    if (t.pc == pc) return &t;
  return nullptr;
}

// ---------------------------------------------------------------------------

void testRuleDWaitBeforeSignal() {
  ProgramBuilder b;
  b.wait(Unit::MXU, 0, 1);
  b.signal(Unit::DMA0, 0, 1);
  b.finish();
  Simulator sim(b.program(), SimConfig{});
  SimResult r = sim.run();
  CHECK(r.status == SimStatus::RuleDViolation);
  CHECK_MSG(r.message.find("Rule D") != std::string::npos, r.message);
  CHECK_MSG(r.message.find("pc 0") != std::string::npos, r.message);
}

void testRuleDCountsSplitBetweenWaiters() {
  // Two units waiting on one event SPLIT its counts (ISA.md §5.3), so two
  // WAIT e,1 need two SIGNAL e,1 -- one is not enough, and Rule D says so
  // statically.
  ProgramBuilder b;
  b.signal(Unit::DMA0, 3, 1);
  b.wait(Unit::MXU, 3, 1);
  b.wait(Unit::DWU, 3, 1);
  b.finish();
  Simulator sim(b.program(), SimConfig{});
  SimResult r = sim.run();
  CHECK(r.status == SimStatus::RuleDViolation);
  CHECK_MSG(r.message.find("event=3") != std::string::npos, r.message);
}

void testRuleDAcceptsCorrectSchedule() {
  ProgramBuilder b;
  b.signal(Unit::DMA0, 3, 3);
  b.wait(Unit::MXU, 3, 2);
  b.trace(Unit::MXU, TraceKind::MARKER, 1);
  b.wait(Unit::DWU, 3, 1);
  b.trace(Unit::DWU, TraceKind::MARKER, 2);
  b.finish();
  Simulator sim(b.program(), SimConfig{});
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);
  CHECK_EQ(r.stats.markers.size(), std::size_t{2});
  CHECK_EQ(r.stats.markers[0].tag, 1u);
  CHECK_EQ(r.stats.markers[1].tag, 2u);
}

// ---------------------------------------------------------------------------

void testDeadlockInsufficientCounts() {
  ProgramBuilder b;
  b.wait(Unit::MXU, 5, 5);
  b.signal(Unit::DMA0, 5, 1);
  b.finish();
  SimConfig cfg;
  cfg.check_rule_d = false;  // exercise the runtime detector, not the static one
  cfg.max_cycles = 10000;
  Simulator sim(b.program(), cfg);
  SimResult r = sim.run();
  CHECK(r.status == SimStatus::Deadlock);
  CHECK_MSG(r.message.find("MXU  pc=0") != std::string::npos, r.message);
  CHECK_MSG(r.message.find("event=5") != std::string::npos, r.message);
  CHECK_MSG(r.message.find("threshold=5") != std::string::npos, r.message);
}

void testDeadlockQueueWedge() {
  // The Rule D hazard in its pure form: the MXU queue fills behind a blocked
  // WAIT, the dispatcher then stalls on the next MXU instruction, and the
  // SIGNAL that would have released the MXU is never dispatched at all.
  ProgramBuilder b;
  b.wait(Unit::MXU, 0, 1);
  for (int i = 0; i < KEA_QUEUE_DEPTH + 1; ++i) b.nop(Unit::MXU, 1);
  b.signal(Unit::DMA0, 0, 1);
  b.finish();
  SimConfig cfg;
  cfg.check_rule_d = false;
  cfg.max_cycles = 10000;
  Simulator sim(b.program(), cfg);
  SimResult r = sim.run();
  CHECK(r.status == SimStatus::Deadlock);
  CHECK_MSG(r.message.find("MXU  pc=0") != std::string::npos, r.message);
  CHECK(r.stats.global.dispatch_stall[KEA_UNIT_MXU] > 0);
  std::printf("queue-wedge deadlock detected after %llu cycles, %llu "
              "dispatcher stall cycles on MXU\n",
              (unsigned long long)r.cycles,
              (unsigned long long)r.stats.global.dispatch_stall[KEA_UNIT_MXU]);
}

// ---------------------------------------------------------------------------

void testSignalDrainsTheUnitPipeline() {
  // ISA.md §5.3 step 1: a SIGNAL may not retire until every instruction the
  // unit issued before it has RETIRED, which for a MATMUL is occupancy + the
  // 32-cycle array drain.  And ISA.md §5.3's same-cycle rule says a waiter
  // released by that increment retires in the SAME cycle.
  ProgramBuilder b;
  b.add(keaMakeMatmul(0, 16, 0, /*m_inner=*/64, /*m_outer=*/1, 0, 16, 0, 0,
                      false, false));
  b.signal(Unit::MXU, 0);
  b.wait(Unit::VPU, 0);
  b.nop(Unit::VPU, 0);
  b.finish();

  SimConfig cfg;
  cfg.trace = true;
  Simulator sim(b.program(), cfg);
  for (int i = 0; i < 2048; ++i)
    sim.machine().spm_a.write(i, static_cast<std::uint8_t>(i));
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);

  const TraceRecord* mm = findPc(r, 0);
  const TraceRecord* sig = findPc(r, 1);
  const TraceRecord* wt = findPc(r, 2);
  CHECK(mm && sig && wt);
  if (!mm || !sig || !wt) return;

  const std::uint64_t lat = keaMatmulLatency(64, 1, false);  // 68 + 32 = 100
  CHECK_EQ(mm->start, std::uint64_t{1});
  CHECK_EQ(mm->retire, 1 + lat);
  CHECK_MSG(sig->start == mm->retire,
            "SIGNAL retired at " + std::to_string(sig->start) +
                " but the MXU pipeline drains at " +
                std::to_string(mm->retire));
  CHECK_MSG(wt->start == sig->start,
            "the same-cycle increment did not release the waiter: WAIT at " +
                std::to_string(wt->start) + ", SIGNAL at " +
                std::to_string(sig->start));
  CHECK(sig->stall_sem >= KEA_MXU_PIPELINE_DEPTH);
  std::printf("SIGNAL drain: MATMUL 1..%llu, SIGNAL and the released WAIT "
              "both at %llu\n",
              (unsigned long long)mm->retire, (unsigned long long)sig->start);
}

void testWaitZeroThresholdIsFree() {
  ProgramBuilder b;
  b.wait(Unit::MXU, 0, 0);
  b.trace(Unit::MXU, TraceKind::MARKER, 99);
  b.finish();
  Simulator sim(b.program(), SimConfig{});
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);
  CHECK_EQ(r.stats.markers.size(), std::size_t{1});
}

void testEventOverflow() {
  ProgramBuilder b;
  b.signal(Unit::DMA0, 0, KEA_EVENT_MAX);
  b.signal(Unit::DMA0, 0, 1);
  b.finish();
  Simulator sim(b.program(), SimConfig{});
  SimResult r = sim.run();
  CHECK(r.status == SimStatus::EventOverflow);
  CHECK_MSG(r.message.find("KEA_EVENT_MAX") != std::string::npos, r.message);
}

void testCtrlWaitBlocksTheDispatcher() {
  // A CTRL-targeted WAIT retires at dispatch, so it stalls the dispatcher
  // itself rather than a queue.  Legal, and occasionally what you want.
  ProgramBuilder b;
  b.add(keaMakeDma(true, Unit::DMA0, false, 0, 0, 64, 1, 1, 0, 0, 0, 0));
  b.signal(Unit::DMA0, 1);
  b.wait(Unit::CTRL, 1);
  b.trace(Unit::MXU, TraceKind::MARKER, 5);
  b.finish();
  Simulator sim(b.program(), SimConfig{});
  sim.stageConstants();
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);
  CHECK_EQ(r.stats.markers.size(), std::size_t{1});
  // The marker cannot precede the DMA's completion.
  CHECK(r.stats.markers[0].cycle >= keaDmaOccupancy(64, 1, 1));
}

// ---------------------------------------------------------------------------

std::uint64_t hashState(const Machine& m) {
  std::uint64_t h = 1469598103934665603ull;
  for (std::size_t i = 0; i < 65536; ++i) {
    h ^= m.spm_a.readU8(static_cast<std::int64_t>(i));
    h *= 1099511628211ull;
  }
  for (std::size_t i = 0; i < 4096; ++i) {
    h ^= static_cast<std::uint64_t>(
        static_cast<std::uint32_t>(m.acc.read(static_cast<std::int64_t>(i))));
    h *= 1099511628211ull;
  }
  return h;
}

KeaProgram buildMixedProgram() {
  ProgramBuilder b(1 << 20);
  std::vector<std::uint8_t> data(16384);
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<std::uint8_t>((i * 29 + 7) & 0xFF);
  b.putConst(0, data);

  b.add(keaMakeDma(true, Unit::DMA0, false, 0, 0, 4096, 2, 1, 4096, 0, 4096, 0));
  b.add(keaMakeDma(true, Unit::DMA1, true, 8192, 0, 4096, 2, 1, 4096, 0, 4096,
                   0));
  b.signal(Unit::DMA0, 0);
  b.signal(Unit::DMA1, 1);

  b.wait(Unit::MXU, 0);
  b.wait(Unit::MXU, 1);
  b.trace(Unit::MXU, TraceKind::REGION_BEGIN, 1);
  for (int i = 0; i < 6; ++i) {
    const std::uint8_t bank = static_cast<std::uint8_t>(i & 1);
    b.add(keaMakeLoadW(static_cast<std::uint32_t>(i * 256), 16, 16, 16, bank,
                       false));
    b.add(keaMakeMatmul(static_cast<std::uint32_t>(i * 16), 16, 0, 64, 1, 0, 16,
                        0, bank, i != 0, false));
  }
  b.trace(Unit::MXU, TraceKind::REGION_END, 1);
  b.signal(Unit::MXU, 2);

  b.add(keaMakeDwconv(0, 0, 2048, 8, 8, 32, 32 * 10, 32, 3, 1, false));
  b.signal(Unit::DWU, 3);

  b.wait(Unit::VPU, 2);
  b.wait(Unit::VPU, 3);
  b.add(keaMakeVfill(32768, 4096, 1, 0, 5));
  b.add(keaMakeVadd(0, 32768, 40960, 4096, 1024, -128, 127));
  b.finish();
  return b.program();
}

void testDeterminism() {
  KeaProgram p = buildMixedProgram();
  SimConfig cfg;
  cfg.trace = true;

  Simulator s1(p, cfg);
  s1.stageConstants();
  SimResult r1 = s1.run();
  const std::uint64_t h1 = hashState(s1.machine());

  Simulator s2(p, cfg);
  s2.stageConstants();
  SimResult r2 = s2.run();
  const std::uint64_t h2 = hashState(s2.machine());

  CHECK_MSG(r1.ok(), r1.message);
  CHECK_MSG(r2.ok(), r2.message);
  CHECK_EQ(r1.cycles, r2.cycles);
  CHECK_EQ(h1, h2);
  CHECK_EQ(r1.trace.size(), r2.trace.size());
  for (std::size_t i = 0; i < r1.trace.size() && i < r2.trace.size(); ++i) {
    CHECK_EQ(r1.trace[i].pc, r2.trace[i].pc);
    CHECK_EQ(r1.trace[i].start, r2.trace[i].start);
    CHECK_EQ(r1.trace[i].retire, r2.trace[i].retire);
  }
  CHECK_EQ(r1.stats.global.mxu_macs_useful, r2.stats.global.mxu_macs_useful);
  CHECK_EQ(r1.stats.global.dram_data_cycles[0],
           r2.stats.global.dram_data_cycles[0]);
  CHECK_EQ(r1.stats.regions.size(), r2.stats.regions.size());
  for (std::size_t i = 0; i < r1.stats.regions.size(); ++i)
    CHECK_EQ(r1.stats.regions[i].c.cycles, r2.stats.regions[i].c.cycles);
  std::printf("determinism: two runs both took %llu cycles, state hash %llx\n",
              (unsigned long long)r1.cycles, (unsigned long long)h1);
}

void testUnsynchronizedReadIsReported() {
  // The most likely compiler bug: consume a DMA target without waiting for it.
  ProgramBuilder b(1 << 20);
  b.putConst(0, std::vector<std::uint8_t>(8192, 3));
  b.add(keaMakeDma(true, Unit::DMA0, false, 0, 0, 4096, 1, 1, 0, 0, 0, 0));
  b.add(keaMakeLoadW(0, 16, 16, 16, 0, false));
  b.add(keaMakeMatmul(0, 16, 0, 64, 1, 0, 16, 0, 0, false, false));
  b.finish();
  SimConfig cfg;
  cfg.strict_hazards = true;
  Simulator sim(b.program(), cfg);
  for (int i = 0; i < 512; ++i) sim.machine().spm_w.write(i, 1);
  sim.stageConstants();
  SimResult r = sim.run();
  CHECK(r.status == SimStatus::Fault);
  CHECK_MSG(r.message.find("unsynchronized") != std::string::npos, r.message);
}

}  // namespace

int main() {
  testRuleDWaitBeforeSignal();
  testRuleDCountsSplitBetweenWaiters();
  testRuleDAcceptsCorrectSchedule();
  testDeadlockInsufficientCounts();
  testDeadlockQueueWedge();
  testSignalDrainsTheUnitPipeline();
  testWaitZeroThresholdIsFree();
  testEventOverflow();
  testCtrlWaitBlocksTheDispatcher();
  testDeterminism();
  testUnsynchronizedReadIsReported();
  TEST_MAIN_END();
}
