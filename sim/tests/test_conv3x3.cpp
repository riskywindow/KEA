// SPDX-License-Identifier: Apache-2.0
//
// The single most important test in the simulator.
//
// It builds, by hand, the complete 3x3 stride-1 convolution worked through in
// docs/ISA.md §8.5 -- IC=16, OC=32, 8x8 input with pad=1, int8, a 10x10x16
// zero-haloed SPM_A tile with sp=16 and sr=160 -- as a real KEA-1 program:
// VCOPY halo fill, DMA_LD of the interior and the weights, eighteen
// LOAD_W/MATMUL pairs accumulating into ACC, two VQUANTs, a DMA_ST, and the
// SIGNAL/WAIT protocol that ties the four units together.
//
// It then checks the result against a plain-as-possible reference convolution
// written from the mathematical definition, with no reference to the ISA's
// addressing identity at all.  If the "convolution is just LOAD_W + MATMUL
// with a shifted activation base" claim -- the central design claim of KEA-1 --
// were wrong, this test would fail.

#include <cstdio>
#include <cstring>
#include <vector>

#include "kea/sim/program_builder.h"
#include "kea/sim/simulator.h"
#include "test_util.h"

using namespace kea;
using namespace kea::sim;

namespace {

constexpr int IC = 16, OC = 32;
constexpr int IH = 8, IW = 8, KH = 3, KW = 3, PAD = 1, S = 1;
constexpr int OH = 8, OW = 8;

// SPM_A geometry from ISA.md §8.5.
constexpr int SP = IC;                 // pixel stride, bytes = 16
constexpr int SR = (IW + 2 * PAD) * SP;  // row stride, bytes = 160
constexpr int A_BASE = 0;
constexpr int A_BYTES = (IH + 2 * PAD) * SR;  // 1600

constexpr std::int8_t IN_ZP = -3;

// DRAM arena.
constexpr std::uint32_t DR_IN = 0;
constexpr std::uint32_t DR_W = 1024;
constexpr std::uint32_t DR_QP = DR_W + 2 * 9 * 256;
constexpr std::uint32_t DR_OUT = 65536;

constexpr std::uint32_t SPMW_W = 0;
constexpr std::uint32_t SPMW_QP = 8192;
constexpr std::uint32_t SPMA_OUT = 65536;

int idx4(int a, int b, int c, int d, int nb, int nc, int nd) {
  return ((a * nb + b) * nc + c) * nd + d;
}

}  // namespace

int main() {
  // --- data ---------------------------------------------------------------
  std::vector<std::int8_t> input(IH * IW * IC);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<std::int8_t>((i * 61 + 17) % 251 - 125);

  std::vector<std::int8_t> weights(OC * KH * KW * IC);  // [oc][kh][kw][ic]
  for (std::size_t i = 0; i < weights.size(); ++i)
    weights[i] = static_cast<std::int8_t>((i * 97 + 5) % 197 - 98);

  std::vector<KeaQuantParam> qp(OC);
  for (int c = 0; c < OC; ++c) {
    qp[c].bias = (c - 16) * 4099;
    qp[c].mult = static_cast<std::int32_t>(1u << 30) + c * 3141592;
    qp[c].shift = 12 + (c % 4);
  }
  const std::int8_t OUT_ZP = -11;
  const std::int8_t CLAMP_LO = OUT_ZP;  // ReLU
  const std::int8_t CLAMP_HI = 127;

  // --- reference convolution (independent of the ISA lowering) -------------
  std::vector<std::int32_t> ref_acc(OH * OW * OC, 0);
  std::vector<std::int8_t> ref_out(OH * OW * OC, 0);
  for (int oh = 0; oh < OH; ++oh)
    for (int ow = 0; ow < OW; ++ow)
      for (int oc = 0; oc < OC; ++oc) {
        std::int32_t sum = 0;
        for (int kh = 0; kh < KH; ++kh)
          for (int kw = 0; kw < KW; ++kw) {
            const int ih = oh * S + kh - PAD;
            const int iw = ow * S + kw - PAD;
            for (int ic = 0; ic < IC; ++ic) {
              const std::int32_t a =
                  (ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                      ? IN_ZP  // the halo carries the input zero point
                      : input[idx4(0, ih, iw, ic, 1, IW, IC)];
              sum += a * weights[idx4(oc, kh, kw, ic, KH, KW, IC)];
            }
          }
        ref_acc[idx4(0, oh, ow, oc, 1, OW, OC)] = sum;
        const std::int32_t v = sum + qp[oc].bias;
        ref_out[idx4(0, oh, ow, oc, 1, OW, OC)] =
            static_cast<std::int8_t>(keaRequantize(v, qp[oc].mult, qp[oc].shift,
                                                   OUT_ZP, CLAMP_LO, CLAMP_HI));
      }

  // --- DRAM image ---------------------------------------------------------
  std::vector<std::uint8_t> arena(DR_QP + OC * sizeof(KeaQuantParam));
  for (std::size_t i = 0; i < input.size(); ++i)
    arena[DR_IN + i] = static_cast<std::uint8_t>(input[i]);
  // Weight tiles, pre-tiled by the compiler exactly as ISA.md §8.1 specifies:
  // dense 16x16 int8 tiles of 256 bytes ordered [oc0][ic0][kh][kw], so
  // w_row_stride is always 16.
  for (int og = 0; og < OC / 16; ++og)
    for (int kh = 0; kh < KH; ++kh)
      for (int kw = 0; kw < KW; ++kw) {
        const int tile = og * KH * KW + kh * KW + kw;
        for (int k = 0; k < 16; ++k)      // k = reduction (input channel)
          for (int n = 0; n < 16; ++n) {  // n = output column
            arena[DR_W + tile * 256 + k * 16 + n] = static_cast<std::uint8_t>(
                weights[idx4(og * 16 + n, kh, kw, k, KH, KW, IC)]);
          }
      }
  std::memcpy(arena.data() + DR_QP, qp.data(), OC * sizeof(KeaQuantParam));

  // --- the program --------------------------------------------------------
  //
  // Events: 0 = halo filled, 1 = tile+weights resident, 2 = ACC ready,
  //         3 = output ready.
  ProgramBuilder b(1 << 20);
  b.putConst(0, arena);

  b.add(keaMakeVfill(A_BASE, A_BYTES, 1, 0, IN_ZP));
  b.signal(Unit::VPU, 0);

  b.wait(Unit::DMA0, 0);
  // Interior of the padded buffer: 8 rows of 8 pixels x 16 channels, landing
  // at A_BASE + 1*sr + 1*sp = 176.
  b.add(keaMakeDma(true, Unit::DMA0, false, DR_IN, A_BASE + SR + SP,
                   /*len0=*/IW * IC, /*n1=*/IH, /*n2=*/1,
                   /*dram_s1=*/IW * IC, 0, /*spm_s1=*/SR, 0));
  b.add(keaMakeDma(true, Unit::DMA1, true, DR_W, SPMW_W, /*len0=*/2 * 9 * 256,
                   1, 1, 0, 0, 0, 0));
  b.add(keaMakeDma(true, Unit::DMA1, true, DR_QP, SPMW_QP,
                   OC * sizeof(KeaQuantParam), 1, 1, 0, 0, 0, 0));
  b.signal(Unit::DMA0, 1);
  b.signal(Unit::DMA1, 1);

  b.wait(Unit::MXU, 1, 2);  // counting acquire: consumes both DMA signals
  b.trace(Unit::MXU, TraceKind::REGION_BEGIN, /*tag=*/7);

  // The per-tap a_addr values ISA.md §8.5 tabulates.
  const int expect_a_addr[9] = {0, 16, 32, 160, 176, 192, 320, 336, 352};

  for (int og = 0; og < OC / 16; ++og) {
    const std::uint32_t q_base = static_cast<std::uint32_t>(og * OH * OW * 16);
    int t = 0;
    for (int kh = 0; kh < KH; ++kh)
      for (int kw = 0; kw < KW; ++kw, ++t) {
        const std::uint8_t bank = static_cast<std::uint8_t>(t & 1);
        const std::uint32_t w_addr =
            SPMW_W + static_cast<std::uint32_t>((og * KH * KW + kh * KW + kw) * 256);
        const std::uint32_t a_addr =
            static_cast<std::uint32_t>(A_BASE + kh * SR + kw * SP);
        CHECK_EQ(static_cast<int>(a_addr), expect_a_addr[t]);
        b.add(keaMakeLoadW(w_addr, /*w_row_stride=*/16, /*k_rows=*/16,
                           /*n_cols=*/16, bank, false));
        b.add(keaMakeMatmul(a_addr, /*a_inner_stride=*/S * SP,
                            /*a_outer_stride=*/S * SR, /*m_inner=*/OW,
                            /*m_outer=*/OH, q_base, /*acc_inner_stride=*/16,
                            /*acc_outer_stride=*/OW * 16, bank,
                            /*accumulate=*/t != 0, false));
      }
  }
  b.trace(Unit::MXU, TraceKind::REGION_END, 7);
  b.signal(Unit::MXU, 2);

  b.wait(Unit::VPU, 2);
  for (int og = 0; og < OC / 16; ++og)
    b.add(keaMakeVquant(
        /*acc_addr=*/static_cast<std::uint32_t>(og * OH * OW * 16),
        /*out_addr=*/static_cast<std::uint32_t>(SPMA_OUT + og * 16),
        /*qparam_addr=*/static_cast<std::uint32_t>(SPMW_QP + og * 16 * sizeof(KeaQuantParam)),
        /*num_pixels=*/OH * OW, /*channels=*/16,
        /*acc_pix_stride=*/16, /*out_pix_stride=*/OC, OUT_ZP, CLAMP_LO,
        CLAMP_HI, false));
  b.signal(Unit::VPU, 3);

  b.wait(Unit::DMA0, 3);
  b.add(keaMakeDma(false, Unit::DMA0, false, DR_OUT, SPMA_OUT,
                   /*len0=*/OH * OW * OC, 1, 1, 0, 0, 0, 0));
  b.finish();

  // --- run ----------------------------------------------------------------
  SimConfig cfg;
  cfg.strict_poison = true;   // the halo fill must cover every byte the array reads
  cfg.strict_hazards = true;  // and the SIGNAL/WAIT protocol must be complete
  Simulator sim(b.program(), cfg);
  sim.stageConstants();
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);
  if (!r.ok()) {
    for (const std::string& d : r.diagnostics) std::fprintf(stderr, "  %s\n", d.c_str());
    TEST_MAIN_END();
  }

  // --- ACC must match the reference exactly -------------------------------
  std::size_t acc_bad = 0;
  for (int og = 0; og < OC / 16; ++og)
    for (int oh = 0; oh < OH; ++oh)
      for (int ow = 0; ow < OW; ++ow)
        for (int n = 0; n < 16; ++n) {
          const std::int64_t q = og * OH * OW * 16 + oh * OW * 16 + ow * 16 + n;
          const std::int32_t want =
              ref_acc[idx4(0, oh, ow, og * 16 + n, 1, OW, OC)];
          if (sim.machine().acc.read(q) != want) {
            if (acc_bad < 5)
              std::fprintf(stderr,
                           "  ACC[%lld] (oh=%d ow=%d oc=%d) = %d, want %d\n",
                           static_cast<long long>(q), oh, ow, og * 16 + n,
                           sim.machine().acc.read(q), want);
            ++acc_bad;
          }
        }
  CHECK_EQ(acc_bad, std::size_t{0});

  // --- and so must the requantized int8 output, back in DRAM --------------
  std::vector<std::uint8_t> got(OH * OW * OC);
  sim.machine().dram.read(DR_OUT, static_cast<std::int64_t>(got.size()),
                          got.data());
  std::size_t out_bad = 0;
  for (std::size_t i = 0; i < got.size(); ++i)
    if (static_cast<std::int8_t>(got[i]) != ref_out[i]) ++out_bad;
  CHECK_EQ(out_bad, std::size_t{0});

  // --- and the cost model must agree with ISA.md §8.5 ---------------------
  //
  // "Useful MACs = 8*8*9*16*32 = 294,912"; "each MATMUL occupies the array for
  // 4 + 8*8 = 68 cycles"; "each LOAD_W occupies the weight port for 10";
  // "Per output-channel group: 9 * 68 = 612 cycles. Two groups: 1224."
  CHECK_EQ(r.stats.global.mxu_macs_useful, std::uint64_t{294912});
  CHECK_EQ(r.stats.global.mxu_matmuls, std::uint64_t{18});
  CHECK_EQ(r.stats.global.mxu_loadws, std::uint64_t{18});
  CHECK_EQ(r.stats.global.unit[KEA_UNIT_MXU].occupancy,
           std::uint64_t{18 * 68 + 18 * 10});
  CHECK_EQ(r.stats.regions.size(), std::size_t{1});
  const std::uint64_t region_cycles = r.stats.regions[0].c.cycles;
  CHECK_MSG(region_cycles >= 1224 && region_cycles <= 1400,
            "MXU region took " + std::to_string(region_cycles) +
                " cycles; ISA.md §8.5 predicts 1224 plus the un-hidden first "
                "LOAD_W and the bank collision between the two oc groups");
  CHECK_EQ(r.stats.regions[0].c.mxu_macs_useful, std::uint64_t{294912});

  std::printf(
      "conv 3x3: %llu total cycles, MXU region %llu cycles for 294912 useful "
      "MACs (%.1f%% of the 1224-cycle ideal)\n",
      static_cast<unsigned long long>(r.cycles),
      static_cast<unsigned long long>(region_cycles),
      100.0 * 1224.0 / static_cast<double>(region_cycles));

  TEST_MAIN_END();
}
