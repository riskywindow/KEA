// SPDX-License-Identifier: Apache-2.0
//
// Timing model tests.  These check the properties that make the project's
// roofline story real:
//
//   * LOAD_W into the idle weight bank genuinely overlaps a MATMUL on the
//     other bank, so weight double buffering is free;
//   * a double-buffered DMA + compute schedule genuinely overlaps, while the
//     naive single-buffered version genuinely does not;
//   * DMA0 and DMA1 genuinely share one 16 B/cycle DRAM port, so per-engine
//     bandwidth halves when both run;
//   * a full queue genuinely stalls the dispatcher.
//
// Each asserts both a direction and an approximate magnitude, because a model
// that gets the direction right and the magnitude wrong is still fiction.

#include <algorithm>
#include <cstdio>
#include <vector>

#include "kea/sim/program_builder.h"
#include "kea/sim/simulator.h"
#include "test_util.h"

using namespace kea;
using namespace kea::sim;

namespace {

/// Give the MXU defined activations and weights so the programs below run
/// without poison diagnostics.
void primeMxu(Machine& m, std::int64_t a_bytes = 16384) {
  for (std::int64_t i = 0; i < a_bytes; ++i)
    m.spm_a.write(i, static_cast<std::uint8_t>(i));
  for (std::int64_t i = 0; i < 1024; ++i)
    m.spm_w.write(i, static_cast<std::uint8_t>(i));
}

// ---------------------------------------------------------------------------
// 1. LOAD_W / MATMUL weight-bank overlap
// ---------------------------------------------------------------------------

std::uint64_t runBankProgram(bool alternate, int pairs) {
  ProgramBuilder b;
  for (int i = 0; i < pairs; ++i) {
    const std::uint8_t bank =
        alternate ? static_cast<std::uint8_t>(i & 1) : std::uint8_t{0};
    b.add(keaMakeLoadW(static_cast<std::uint32_t>((i % 2) * 256), 16, 16, 16,
                       bank, false));
    b.add(keaMakeMatmul(/*a_addr=*/0, /*a_inner_stride=*/0,
                        /*a_outer_stride=*/0, /*m_inner=*/64, /*m_outer=*/1,
                        /*acc_addr=*/0, 16, 0, bank, /*accumulate=*/i != 0,
                        false));
  }
  b.finish();
  Simulator sim(b.program(), SimConfig{});
  primeMxu(sim.machine());
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);
  return r.cycles;
}

void testWeightBankOverlap() {
  const int pairs = 20;
  const std::uint64_t alt = runBankProgram(true, pairs);
  const std::uint64_t same = runBankProgram(false, pairs);

  const std::uint64_t mm = keaMatmulOccupancy(64, 1, false);  // 4 + 64 = 68
  const std::uint64_t lw = keaLoadWOccupancy(16, false);      // 2 + 8  = 10
  CHECK_EQ(mm, std::uint64_t{68});
  CHECK_EQ(lw, std::uint64_t{10});

  CHECK_MSG(alt < same, "alternating banks (" + std::to_string(alt) +
                            ") should beat reusing one bank (" +
                            std::to_string(same) + ")");
  CHECK_MSG(alt <= static_cast<std::uint64_t>(pairs) * mm +
                       KEA_MXU_PIPELINE_DEPTH + 20,
            "alternating banks took " + std::to_string(alt) +
                " cycles, expected about " + std::to_string(pairs * mm));
  CHECK_MSG(same >= static_cast<std::uint64_t>(pairs) * (mm + lw) - 20,
            "same bank took " + std::to_string(same) +
                " cycles, expected about " + std::to_string(pairs * (mm + lw)));
  const double ratio = static_cast<double>(same) / static_cast<double>(alt);
  const double want = static_cast<double>(mm + lw) / static_cast<double>(mm);
  CHECK_MSG(ratio > want - 0.05 && ratio < want + 0.05,
            "LOAD_W/MATMUL bank-overlap speedup was " + std::to_string(ratio) +
                "x, expected about " + std::to_string(want) + "x");
  std::printf("bank overlap: alternating %llu vs same-bank %llu cycles "
              "(%.3fx, model %.3fx)\n",
              (unsigned long long)alt, (unsigned long long)same, ratio, want);
}

// ---------------------------------------------------------------------------
// 2. Double-buffered DMA + compute
// ---------------------------------------------------------------------------

constexpr std::uint16_t kTileBytes = 4096;
constexpr int kIters = 8;
constexpr std::uint32_t kBufA = 0;
constexpr std::uint32_t kBufB = 8192;

void addCompute(ProgramBuilder& b, std::uint32_t buf, int i) {
  const std::uint8_t bank = static_cast<std::uint8_t>(i & 1);
  b.add(keaMakeLoadW(static_cast<std::uint32_t>((i % 2) * 256), 16, 16, 16,
                     bank, false));
  b.add(keaMakeMatmul(buf, /*a_inner_stride=*/16, /*a_outer_stride=*/0,
                      /*m_inner=*/256, /*m_outer=*/1, /*acc_addr=*/0, 16, 0,
                      bank, /*accumulate=*/i != 0, false));
}

/// Single buffer: load, wait, compute, wait, load ...  Nothing overlaps.
std::uint64_t runNaive() {
  ProgramBuilder b(1 << 20);
  b.putConst(0, std::vector<std::uint8_t>(kTileBytes, 3));
  for (int i = 0; i < kIters; ++i) {
    b.add(keaMakeDma(true, Unit::DMA0, false, 0, kBufA, kTileBytes, 1, 1, 0, 0,
                     0, 0));
    b.signal(Unit::DMA0, 0);
    b.wait(Unit::MXU, 0);
    addCompute(b, kBufA, i);
    b.signal(Unit::MXU, 1);
    b.wait(Unit::DMA0, 1);
  }
  b.finish();
  Simulator sim(b.program(), SimConfig{});
  primeMxu(sim.machine());
  sim.stageConstants();
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);
  return r.cycles;
}

/// Two buffers: the DMA engine stays two tiles ahead of the MXU.
std::uint64_t runDoubleBuffered() {
  const std::uint32_t bufs[2] = {kBufA, kBufB};
  ProgramBuilder b(1 << 20);
  b.putConst(0, std::vector<std::uint8_t>(kTileBytes, 3));

  for (int i = 0; i < 2; ++i) {  // prologue: prime both buffers
    b.add(keaMakeDma(true, Unit::DMA0, false, 0, bufs[i], kTileBytes, 1, 1, 0,
                     0, 0, 0));
    b.signal(Unit::DMA0, static_cast<std::uint8_t>(i));  // ev0/ev1 = buffer full
  }
  for (int i = 0; i < kIters; ++i) {
    const int s = i & 1;
    b.wait(Unit::MXU, static_cast<std::uint8_t>(s));
    addCompute(b, bufs[s], i);
    b.signal(Unit::MXU, static_cast<std::uint8_t>(2 + s));  // ev2/ev3 = free
    if (i + 2 < kIters) {
      b.wait(Unit::DMA0, static_cast<std::uint8_t>(2 + s));
      b.add(keaMakeDma(true, Unit::DMA0, false, 0, bufs[s], kTileBytes, 1, 1, 0,
                       0, 0, 0));
      b.signal(Unit::DMA0, static_cast<std::uint8_t>(s));
    }
  }
  b.finish();
  Simulator sim(b.program(), SimConfig{});
  primeMxu(sim.machine());
  sim.stageConstants();
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);
  return r.cycles;
}

void testDoubleBufferingOverlap() {
  const std::uint64_t dma = keaDmaOccupancy(kTileBytes, 1, 1);  // 274
  const std::uint64_t mm = keaMatmulOccupancy(256, 1, false);   // 260
  const std::uint64_t naive = runNaive();
  const std::uint64_t db = runDoubleBuffered();

  // Serialised: every iteration pays the DMA, then the MATMUL, then the MXU's
  // 32-cycle SIGNAL drain, with nothing in flight in between.
  const double serial_model =
      static_cast<double>(kIters) *
      static_cast<double>(dma + mm + KEA_MXU_PIPELINE_DEPTH);
  // Overlapped: bounded by whichever pipeline is slower.
  const double overlap_model =
      static_cast<double>(kIters) *
      static_cast<double>(std::max(dma, mm + KEA_MXU_PIPELINE_DEPTH));

  CHECK_MSG(db < naive, "double buffering did not help at all");
  CHECK_MSG(static_cast<double>(naive) > 0.80 * serial_model &&
                static_cast<double>(naive) < 1.30 * serial_model,
            "naive schedule took " + std::to_string(naive) +
                " cycles, serial model predicts " +
                std::to_string(serial_model));
  CHECK_MSG(static_cast<double>(db) < 1.30 * overlap_model,
            "double-buffered schedule took " + std::to_string(db) +
                " cycles, near-full overlap predicts " +
                std::to_string(overlap_model));
  const double speedup = static_cast<double>(naive) / static_cast<double>(db);
  CHECK_MSG(speedup > 1.5,
            "double buffering only gave " + std::to_string(speedup) + "x");
  std::printf("double buffering: naive %llu vs overlapped %llu cycles "
              "(%.3fx; DMA %llu, MATMUL %llu per tile)\n",
              (unsigned long long)naive, (unsigned long long)db, speedup,
              (unsigned long long)dma, (unsigned long long)mm);
}

// ---------------------------------------------------------------------------
// 3. DRAM port contention
// ---------------------------------------------------------------------------

struct DmaRun {
  std::uint64_t cycles = 0;
  std::uint64_t data_cycles[2] = {0, 0};
  std::uint64_t contended[2] = {0, 0};
  std::uint64_t bytes[2] = {0, 0};
};

DmaRun runDma(bool both_engines, std::uint32_t total_bytes) {
  const std::uint16_t len0 = 4096;
  const std::uint16_t n1 = static_cast<std::uint16_t>(total_bytes / len0);

  ProgramBuilder b(1 << 22);
  b.putConst(0, std::vector<std::uint8_t>(total_bytes, 7));
  b.add(keaMakeDma(true, Unit::DMA0, /*spm_is_w=*/false, 0, 0, len0, n1, 1,
                   len0, 0, len0, 0));
  if (both_engines)
    b.add(keaMakeDma(true, Unit::DMA1, /*spm_is_w=*/true, 0, 0, len0, n1, 1,
                     len0, 0, len0, 0));
  b.finish();

  Simulator sim(b.program(), SimConfig{});
  sim.stageConstants();
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);
  DmaRun out;
  out.cycles = r.cycles;
  for (int e = 0; e < 2; ++e) {
    out.data_cycles[e] = r.stats.global.dram_data_cycles[e];
    out.contended[e] = r.stats.global.dram_contended_cycles[e];
    out.bytes[e] = r.stats.global.dma_bytes[e];
  }
  return out;
}

void testDramPortContention() {
  const std::uint32_t bytes = 65536;
  const DmaRun solo = runDma(false, bytes);
  const DmaRun both = runDma(true, bytes);

  const std::uint64_t ideal_solo = bytes / KEA_DRAM_BYTES_PER_CYCLE;  // 4096
  CHECK_EQ(solo.data_cycles[0], ideal_solo);
  CHECK_EQ(solo.contended[0], std::uint64_t{0});
  CHECK_EQ(both.bytes[0], static_cast<std::uint64_t>(bytes));
  CHECK_EQ(both.bytes[1], static_cast<std::uint64_t>(bytes));

  // Both engines active: each is granted 8 B/cycle, so each engine's data
  // phase takes twice as long as it would alone.
  for (int e = 0; e < 2; ++e) {
    CHECK_MSG(static_cast<double>(both.data_cycles[e]) >
                      1.9 * static_cast<double>(ideal_solo) &&
                  static_cast<double>(both.data_cycles[e]) <
                      2.1 * static_cast<double>(ideal_solo),
              "engine " + std::to_string(e) + " spent " +
                  std::to_string(both.data_cycles[e]) +
                  " data cycles, expected about " +
                  std::to_string(2 * ideal_solo));
    CHECK(both.contended[e] > ideal_solo);
  }

  // Aggregate bandwidth is unchanged: two engines move 2x the bytes in ~2x the
  // cycles.  MICROARCH.md §6.1's point exactly -- the second engine exists for
  // scheduling, not for bandwidth.
  const double solo_bw =
      static_cast<double>(bytes) / static_cast<double>(solo.cycles);
  const double both_bw =
      static_cast<double>(2 * bytes) / static_cast<double>(both.cycles);
  CHECK_MSG(both_bw > 0.95 * solo_bw && both_bw < 1.05 * solo_bw,
            "aggregate B/cycle changed: solo " + std::to_string(solo_bw) +
                ", both " + std::to_string(both_bw));
  std::printf("DRAM contention: solo %llu data cycles, concurrent %llu/%llu; "
              "aggregate %.3f vs %.3f B/cycle of %d peak\n",
              (unsigned long long)solo.data_cycles[0],
              (unsigned long long)both.data_cycles[0],
              (unsigned long long)both.data_cycles[1], solo_bw, both_bw,
              KEA_DRAM_BYTES_PER_CYCLE);
}

// ---------------------------------------------------------------------------
// 4. Queue backpressure and occupancy bookkeeping
// ---------------------------------------------------------------------------

void testQueueBackpressure() {
  ProgramBuilder b;
  for (int i = 0; i < KEA_QUEUE_DEPTH + 8; ++i)
    b.add(keaMakeVfill(0, 32, /*rows=*/64, 32, 0));  // 4 + 64 = 68 cycles each
  b.finish();
  Simulator sim(b.program(), SimConfig{});
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);
  CHECK(r.stats.global.dispatch_stall_total > 0);
  CHECK_EQ(r.stats.global.dispatch_stall[KEA_UNIT_VPU],
           r.stats.global.dispatch_stall_total);
  CHECK_EQ(r.stats.global.unit[KEA_UNIT_VPU].max_queue_depth,
           static_cast<std::uint64_t>(KEA_QUEUE_DEPTH));
  std::printf("queue backpressure: %llu dispatcher stall cycles, all "
              "attributed to VPU\n",
              (unsigned long long)r.stats.global.dispatch_stall_total);
}

void testOccupancyMatchesHwConfig() {
  ProgramBuilder b;
  b.add(keaMakeVfill(0, 1024, 4, 1024, 0));
  b.add(keaMakeVadd(0, 0, 2048, 0, 777, -128, 127));
  b.add(keaMakeVpool(false, 0, 8192, 3, 3, 16, 2, 2, 1, 1, 16 * 8, 16 * 3));
  b.finish();
  Simulator sim(b.program(), SimConfig{});
  for (int i = 0; i < 32; ++i) sim.machine().spm_w.write(i, 0);
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);
  const std::uint64_t want = keaVcopyOccupancy(1024, 4) +
                             keaVaddOccupancy(777) +
                             keaVpoolOccupancy(3, 3, 16, 2, 2);
  CHECK_EQ(r.stats.global.unit[KEA_UNIT_VPU].occupancy, want);
}

/// The DMA state machine must reproduce keaDmaOccupancy() exactly when it has
/// the port to itself -- that function is what the compiler costs against.
void testDmaOccupancyExact() {
  const std::uint16_t len0 = 64;
  const std::uint16_t n1 = 8;
  const std::uint8_t n2 = 1;
  ProgramBuilder b(1 << 16);
  b.putConst(0, std::vector<std::uint8_t>(4096, 1));
  b.add(keaMakeDma(true, Unit::DMA0, false, 0, 0, len0, n1, n2, len0, 0, len0,
                   0));
  b.add(keaMakeDma(true, Unit::DMA0, false, 0, 1024, len0, n1, n2, len0, 0,
                   len0, 0));
  b.finish();
  Simulator sim(b.program(), SimConfig{});
  sim.stageConstants();
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);
  const std::uint64_t occ = keaDmaOccupancy(len0, n1, n2);
  CHECK_EQ(occ, std::uint64_t{64});  // hw_config.h static_asserts this
  // Descriptor 1 starts at cycle 1 and runs back to back with descriptor 2.
  CHECK_EQ(r.stats.global.unit[KEA_UNIT_DMA0].busy_cycles, 2 * occ);
  CHECK_EQ(r.stats.global.dram_data_cycles[0],
           2 * keaCeilDiv(static_cast<std::uint64_t>(len0) * n1 * n2,
                          KEA_DMA_ENGINE_BYTES_PER_CYCLE));
}

}  // namespace

int main() {
  testWeightBankOverlap();
  testDoubleBufferingOverlap();
  testDramPortContention();
  testQueueBackpressure();
  testOccupancyMatchesHwConfig();
  testDmaOccupancyExact();
  TEST_MAIN_END();
}
