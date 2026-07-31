// SPDX-License-Identifier: Apache-2.0
//
// A MobileNetV2-scale synthetic workload, used to (a) prove the simulator
// handles programs of the size the project actually targets and (b) measure
// simulated cycles per wallclock second.
//
// The shape is MICROARCH.md §9.3(c)'s mid-network pointwise block repeated:
// 24 reduction tiles x 6 output tiles of M = 196 rows, i.e. 144 MATMULs and
// ~28,800 cycles per layer, plus a depthwise layer, a VQUANT and the DMA
// traffic to feed them.  At 45 layers that is ~1.3M cycles and ~14k
// instructions -- the same order as a real MobileNetV2 artifact.

#include <chrono>
#include <cstdio>
#include <vector>

#include "kea/sim/program_builder.h"
#include "kea/sim/simulator.h"
#include "test_util.h"

using namespace kea;
using namespace kea::sim;

namespace {

constexpr int kLayers = 45;
constexpr int kIcTiles = 24;
constexpr int kOcTiles = 6;
constexpr int kM = 196;

constexpr std::uint32_t kWeightBytes = kIcTiles * kOcTiles * 256;  // 36864
constexpr std::uint32_t kQpAddr = kWeightBytes;                    // in SPM_W
constexpr std::uint32_t kQpBytes = 16 * sizeof(KeaQuantParam);     // 192
constexpr std::uint32_t kActBytes = 8192;
constexpr std::uint32_t kOutAddr = 65536;
constexpr std::uint32_t kDwWAddr = 200000;  // SPM_W, past the weight tiles

}  // namespace

int main() {
  // --- DRAM image ---------------------------------------------------------
  std::vector<std::uint8_t> arena(kWeightBytes + kQpBytes + kActBytes + 65536);
  for (std::size_t i = 0; i < kWeightBytes; ++i)
    arena[i] = static_cast<std::uint8_t>((i * 31 + 3) & 0xFF);
  {
    std::vector<KeaQuantParam> qp(16);
    for (int c = 0; c < 16; ++c) {
      qp[c].bias = c * 101;
      qp[c].mult = static_cast<std::int32_t>(1u << 30) + c * 777;
      qp[c].shift = 14;
    }
    std::memcpy(arena.data() + kQpAddr, qp.data(), kQpBytes);
  }
  for (std::size_t i = 0; i < kActBytes; ++i)
    arena[kWeightBytes + kQpBytes + i] =
        static_cast<std::uint8_t>((i * 13 + 5) & 0xFF);
  const std::uint32_t kDrAct = kWeightBytes + kQpBytes;

  // --- the program --------------------------------------------------------
  ProgramBuilder b(1u << 22);
  b.putConst(0, arena);

  // Depthwise kernel taps, loaded once.
  b.add(keaMakeDma(true, Unit::DMA1, /*spm_is_w=*/true, 0, kDwWAddr,
                   /*len0=*/3 * 3 * 64, 1, 1, 0, 0, 0, 0));
  b.signal(Unit::DMA1, 7);
  b.wait(Unit::DWU, 7);

  for (int layer = 0; layer < kLayers; ++layer) {
    b.trace(Unit::MXU, TraceKind::REGION_BEGIN, static_cast<std::uint32_t>(layer));

    b.add(keaMakeDma(true, Unit::DMA0, /*spm_is_w=*/true, 0, 0,
                     /*len0=*/static_cast<std::uint16_t>(kWeightBytes + kQpBytes),
                     1, 1, 0, 0, 0, 0));
    b.add(keaMakeDma(true, Unit::DMA0, /*spm_is_w=*/false, kDrAct, 0,
                     /*len0=*/static_cast<std::uint16_t>(kActBytes), 1, 1, 0, 0,
                     0, 0));
    b.signal(Unit::DMA0, 0);

    b.wait(Unit::MXU, 0);
    for (int oc0 = 0; oc0 < kOcTiles; ++oc0)
      for (int ic0 = 0; ic0 < kIcTiles; ++ic0) {
        const int t = oc0 * kIcTiles + ic0;
        const std::uint8_t bank = static_cast<std::uint8_t>(t & 1);
        b.add(keaMakeLoadW(static_cast<std::uint32_t>(t * 256), 16, 16, 16,
                           bank, false));
        b.add(keaMakeMatmul(static_cast<std::uint32_t>(ic0 * 16), 16, 0, kM, 1,
                            static_cast<std::uint32_t>(oc0 * kM * 16), 16, 0,
                            bank, /*accumulate=*/ic0 != 0, false));
      }
    b.signal(Unit::MXU, 1);

    // A depthwise layer running concurrently on the DWU.
    b.add(keaMakeDwconv(/*a_addr=*/0, /*w_addr=*/kDwWAddr, /*acc_addr=*/20000,
                        /*out_h=*/14, /*out_w=*/14, /*channels=*/64,
                        /*a_row_stride=*/64 * 16, /*a_pix_stride=*/64, 3, 1,
                        false));

    b.wait(Unit::VPU, 1);
    for (int oc0 = 0; oc0 < kOcTiles; ++oc0)
      b.add(keaMakeVquant(static_cast<std::uint32_t>(oc0 * kM * 16),
                          static_cast<std::uint32_t>(kOutAddr + oc0 * 16),
                          kQpAddr, kM, 16, 16, kOcTiles * 16, 0, -128, 127,
                          false));
    b.signal(Unit::VPU, 2);

    b.wait(Unit::DMA1, 2);
    b.add(keaMakeDma(false, Unit::DMA1, false,
                     /*dram=*/kWeightBytes + kQpBytes + kActBytes, kOutAddr,
                     static_cast<std::uint16_t>(kM * kOcTiles * 16 / 16), 16, 1,
                     0, 0, 0, 0));
    b.signal(Unit::DMA1, 3);
    b.wait(Unit::MXU, 3);

    b.trace(Unit::MXU, TraceKind::REGION_END, static_cast<std::uint32_t>(layer));
  }
  b.finish();

  std::printf("benchmark program: %zu instructions\n", b.program().code.size());

  Simulator sim(b.program(), SimConfig{});
  sim.stageConstants();
  const auto t0 = std::chrono::steady_clock::now();
  SimResult r = sim.run();
  const auto t1 = std::chrono::steady_clock::now();
  CHECK_MSG(r.ok(), r.message);

  const double secs =
      std::chrono::duration<double>(t1 - t0).count();
  const double rate = static_cast<double>(r.cycles) / (secs > 0 ? secs : 1e-9);
  std::printf(
      "benchmark: %llu simulated cycles in %.3f s wallclock = %.2f M "
      "simulated-cycles/s\n",
      (unsigned long long)r.cycles, secs, rate / 1e6);
  std::printf("  %llu useful MAC, MXU %.1f%% of peak, DRAM %llu B, intensity "
              "%.1f ops/byte\n",
              (unsigned long long)r.stats.global.mxu_macs_useful,
              100.0 * static_cast<double>(r.stats.global.mxu_macs_useful) /
                  (static_cast<double>(r.cycles) * KEA_MXU_INT8_MACS_PER_CYCLE),
              (unsigned long long)r.stats.global.dramBytes(),
              computeRoofline(r.stats.global).intensity);
  std::printf("  %zu TRACE regions\n", r.stats.regions.size());

  CHECK(r.cycles > 1000000);
  CHECK_EQ(r.stats.regions.size(), static_cast<std::size_t>(kLayers));
  // Guard against a pathological slowdown; the real number is printed above.
  CHECK_MSG(secs < 120.0,
            "simulation took " + std::to_string(secs) +
                " s, which is minutes-not-seconds territory");
  TEST_MAIN_END();
}
