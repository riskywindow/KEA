// SPDX-License-Identifier: Apache-2.0
//
// Per-opcode functional tests.  Every expected value is either hand computed
// or produced by a straightforward reference loop written independently of the
// simulator's implementation.

#include <cstring>
#include <vector>

#include "kea/sim/program_builder.h"
#include "kea/sim/simulator.h"
#include "test_util.h"

using namespace kea;
using namespace kea::sim;

namespace {

void pokeA(Machine& m, std::int64_t addr, const std::vector<std::int8_t>& v) {
  for (std::size_t i = 0; i < v.size(); ++i)
    m.spm_a.write(addr + static_cast<std::int64_t>(i),
                  static_cast<std::uint8_t>(v[i]));
}

void pokeW(Machine& m, std::int64_t addr, const void* p, std::size_t n) {
  const auto* b = static_cast<const std::uint8_t*>(p);
  for (std::size_t i = 0; i < n; ++i)
    m.spm_w.write(addr + static_cast<std::int64_t>(i), b[i]);
}

void pokeWi8(Machine& m, std::int64_t addr, const std::vector<std::int8_t>& v) {
  for (std::size_t i = 0; i < v.size(); ++i)
    m.spm_w.write(addr + static_cast<std::int64_t>(i),
                  static_cast<std::uint8_t>(v[i]));
}

// ---------------------------------------------------------------------------

void testLoadWMatmulInt8() {
  // W[k][n] = k - n, laid out as 16-byte rows at SPM_W 0.
  std::vector<std::int8_t> w(16 * 16);
  for (int k = 0; k < 16; ++k)
    for (int n = 0; n < 16; ++n) w[k * 16 + n] = static_cast<std::int8_t>(k - n);
  std::vector<std::int8_t> a(16);
  for (int k = 0; k < 16; ++k) a[k] = static_cast<std::int8_t>(k * 3 - 20);

  ProgramBuilder b;
  b.add(keaMakeLoadW(0, 16, 16, 16, 0, false));
  b.add(keaMakeMatmul(/*a_addr=*/0, /*a_inner=*/16, /*a_outer=*/0,
                      /*m_inner=*/1, /*m_outer=*/1, /*acc_addr=*/0,
                      /*acc_inner=*/16, /*acc_outer=*/0, /*bank=*/0,
                      /*accumulate=*/false, /*int4=*/false));
  b.signal(Unit::MXU, 0);
  b.wait(Unit::VPU, 0);
  b.finish();

  Simulator sim(b.program(), SimConfig{});
  pokeWi8(sim.machine(), 0, w);
  pokeA(sim.machine(), 0, a);
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);

  for (int n = 0; n < 16; ++n) {
    std::int32_t want = 0;
    for (int k = 0; k < 16; ++k) want += a[k] * w[k * 16 + n];
    CHECK_EQ(sim.machine().acc.read(n), want);
  }
}

void testLoadWTailTileZeroing() {
  // k_rows=3, n_cols=5: rows >= 3 and columns >= 5 must load as ZERO, which is
  // how channel counts that are not multiples of 16 avoid needing masking.
  std::vector<std::int8_t> w(16 * 16, 7);
  std::vector<std::int8_t> a(16, 2);

  ProgramBuilder b;
  b.add(keaMakeLoadW(0, 16, 3, 5, 1, false));
  b.add(keaMakeMatmul(0, 16, 0, 1, 1, 0, 16, 0, /*bank=*/1, false, false));
  b.finish();

  Simulator sim(b.program(), SimConfig{});
  pokeWi8(sim.machine(), 0, w);
  pokeA(sim.machine(), 0, a);
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);

  for (int n = 0; n < 16; ++n) {
    const std::int32_t want = (n < 5) ? (3 * 2 * 7) : 0;
    CHECK_EQ(sim.machine().acc.read(n), want);
  }
}

void testMatmul2dTilingAndAccumulate() {
  // Two MATMULs into the same ACC tile with independent inner/outer strides on
  // both the activation and the accumulator side.
  const int m_inner = 3, m_outer = 4;
  const int a_inner = 32, a_outer = 256;  // deliberately not dense
  const int q_inner = 16, q_outer = 64;

  std::vector<std::int8_t> w0(16 * 16), w1(16 * 16);
  for (int k = 0; k < 16; ++k)
    for (int n = 0; n < 16; ++n) {
      w0[k * 16 + n] = static_cast<std::int8_t>((k * 5 + n) % 11 - 5);
      w1[k * 16 + n] = static_cast<std::int8_t>((k * 3 + n * 7) % 9 - 4);
    }

  std::vector<std::int8_t> spm(4096, 0);
  for (int mo = 0; mo < m_outer; ++mo)
    for (int mi = 0; mi < m_inner; ++mi)
      for (int k = 0; k < 16; ++k)
        spm[mo * a_outer + mi * a_inner + k] =
            static_cast<std::int8_t>((mo * 7 + mi * 13 + k * 3) % 61 - 30);

  ProgramBuilder b;
  b.add(keaMakeLoadW(0, 16, 16, 16, 0, false));
  b.add(keaMakeMatmul(0, a_inner, a_outer, m_inner, m_outer, 0, q_inner,
                      q_outer, 0, /*accumulate=*/false, false));
  b.add(keaMakeLoadW(256, 16, 16, 16, 1, false));
  b.add(keaMakeMatmul(0, a_inner, a_outer, m_inner, m_outer, 0, q_inner,
                      q_outer, 1, /*accumulate=*/true, false));
  b.finish();

  Simulator sim(b.program(), SimConfig{});
  pokeWi8(sim.machine(), 0, w0);
  pokeWi8(sim.machine(), 256, w1);
  pokeA(sim.machine(), 0, spm);
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);

  for (int mo = 0; mo < m_outer; ++mo)
    for (int mi = 0; mi < m_inner; ++mi)
      for (int n = 0; n < 16; ++n) {
        std::int32_t want = 0;
        for (int k = 0; k < 16; ++k) {
          const std::int8_t av = spm[mo * a_outer + mi * a_inner + k];
          want += av * w0[k * 16 + n];
          want += av * w1[k * 16 + n];
        }
        CHECK_EQ(sim.machine().acc.read(mo * q_outer + mi * q_inner + n), want);
      }
}

void testMatmulInt4() {
  // int4 activations are little-nibble-first: element k is nibble k of the 8
  // bytes at a_addr.
  std::vector<std::int8_t> avals(16);
  for (int k = 0; k < 16; ++k) avals[k] = static_cast<std::int8_t>((k % 15) - 7);
  std::vector<std::int8_t> packed(8);
  for (int i = 0; i < 8; ++i)
    packed[i] = static_cast<std::int8_t>(
        keaPackInt4(avals[2 * i], avals[2 * i + 1]));

  std::vector<std::int8_t> wvals(16 * 16);
  for (int k = 0; k < 16; ++k)
    for (int n = 0; n < 16; ++n)
      wvals[k * 16 + n] = static_cast<std::int8_t>(((k + n) % 15) - 7);
  // int4 weight tile: row stride 8 bytes, two columns per byte.
  std::vector<std::int8_t> wpacked(16 * 8);
  for (int k = 0; k < 16; ++k)
    for (int i = 0; i < 8; ++i)
      wpacked[k * 8 + i] = static_cast<std::int8_t>(
          keaPackInt4(wvals[k * 16 + 2 * i], wvals[k * 16 + 2 * i + 1]));

  ProgramBuilder b;
  b.add(keaMakeLoadW(0, 8, 16, 16, 0, /*int4=*/true));
  b.add(keaMakeMatmul(0, 8, 0, 1, 1, 0, 16, 0, 0, false, /*int4=*/true));
  b.finish();

  Simulator sim(b.program(), SimConfig{});
  pokeWi8(sim.machine(), 0, wpacked);
  pokeA(sim.machine(), 0, packed);
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);

  for (int n = 0; n < 16; ++n) {
    std::int32_t want = 0;
    for (int k = 0; k < 16; ++k) want += avals[k] * wvals[k * 16 + n];
    CHECK_EQ(sim.machine().acc.read(n), want);
  }
}

void testDwconv(int K, int S) {
  const int C = 32;
  const int out_h = 5, out_w = 4;
  const int in_w = (out_w - 1) * S + K;
  const int in_h = (out_h - 1) * S + K;
  const int pix_stride = C;
  const int row_stride = in_w * C;

  std::vector<std::int8_t> a(static_cast<std::size_t>(in_h) * row_stride);
  for (std::size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<std::int8_t>((i * 37 + 11) % 251 - 125);
  std::vector<std::int8_t> w(static_cast<std::size_t>(K) * K * C);
  for (std::size_t i = 0; i < w.size(); ++i)
    w[i] = static_cast<std::int8_t>((i * 53 + 7) % 199 - 99);

  ProgramBuilder b;
  b.add(keaMakeDwconv(/*a_addr=*/0, /*w_addr=*/0, /*acc_addr=*/0, out_h, out_w,
                      C, row_stride, pix_stride, static_cast<std::uint8_t>(K),
                      static_cast<std::uint8_t>(S), /*accumulate=*/false));
  b.finish();

  Simulator sim(b.program(), SimConfig{});
  pokeA(sim.machine(), 0, a);
  pokeWi8(sim.machine(), 0, w);
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);

  // Independent reference.
  for (int oh = 0; oh < out_h; ++oh)
    for (int ow = 0; ow < out_w; ++ow)
      for (int c = 0; c < C; ++c) {
        std::int32_t want = 0;
        for (int kh = 0; kh < K; ++kh)
          for (int kw = 0; kw < K; ++kw)
            want += a[(oh * S + kh) * row_stride + (ow * S + kw) * pix_stride + c] *
                    w[(kh * K + kw) * C + c];
        CHECK_EQ(sim.machine().acc.read((oh * out_w + ow) * C + c), want);
      }
}

void testVquant(bool int4_out) {
  const int C = 16;
  const int P = 5;
  std::vector<KeaQuantParam> qp(C);
  for (int c = 0; c < C; ++c) {
    qp[c].bias = (c - 8) * 1237;
    qp[c].mult = static_cast<std::int32_t>(1u << 30) + c * 1234567;
    qp[c].shift = 4 + (c % 5);
  }
  std::vector<std::int32_t> accv(static_cast<std::size_t>(P) * C);
  for (std::size_t i = 0; i < accv.size(); ++i)
    accv[i] = static_cast<std::int32_t>((i * 977) % 100003) - 50000;

  const std::int8_t out_zp = int4_out ? 0 : -7;
  const std::int8_t lo = int4_out ? -8 : -128;
  const std::int8_t hi = int4_out ? 7 : 120;

  ProgramBuilder b;
  b.add(keaMakeVquant(/*acc_addr=*/0, /*out_addr=*/64, /*qparam_addr=*/0, P, C,
                      /*acc_pix_stride=*/C, /*out_pix_stride=*/int4_out ? C / 2 : C,
                      out_zp, lo, hi, int4_out));
  b.finish();

  Simulator sim(b.program(), SimConfig{});
  pokeW(sim.machine(), 0, qp.data(), qp.size() * sizeof(KeaQuantParam));
  for (std::size_t i = 0; i < accv.size(); ++i)
    sim.machine().acc.write(static_cast<std::int64_t>(i), accv[i]);
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);

  for (int p = 0; p < P; ++p)
    for (int c = 0; c < C; ++c) {
      const std::int32_t v = accv[p * C + c] + qp[c].bias;
      const std::int32_t want =
          keaRequantize(v, qp[c].mult, qp[c].shift, out_zp, lo, hi);
      std::int32_t got;
      if (int4_out) {
        const std::uint8_t byte =
            sim.machine().spm_a.readU8(64 + p * (C / 2) + c / 2);
        got = keaUnpackInt4(byte, static_cast<unsigned>(c & 1));
      } else {
        got = sim.machine().spm_a.readI8(64 + p * C + c);
      }
      CHECK_EQ(got, want);
    }
}

void testVadd() {
  const int N = 100;
  KeaAddParam p{};
  p.a_mult = static_cast<std::int32_t>(1u << 30) + 111;
  p.b_mult = static_cast<std::int32_t>(1u << 30) + 999999;
  p.o_mult = static_cast<std::int32_t>(1u << 30) + 5;
  p.a_shift = 20;
  p.b_shift = 20;
  p.o_shift = 19;
  p.a_zp = -3;
  p.b_zp = 4;
  p.o_zp = 1;

  std::vector<std::int8_t> av(N), bv(N);
  for (int i = 0; i < N; ++i) {
    av[i] = static_cast<std::int8_t>((i * 7) % 255 - 127);
    bv[i] = static_cast<std::int8_t>((i * 13) % 255 - 127);
  }

  ProgramBuilder b;
  b.add(keaMakeVadd(/*a=*/0, /*b=*/256, /*out=*/512, /*param=*/0, N, -100, 100));
  // In place: out == a.
  b.add(keaMakeVadd(/*a=*/0, /*b=*/256, /*out=*/0, /*param=*/0, N, -100, 100));
  b.finish();

  Simulator sim(b.program(), SimConfig{});
  pokeW(sim.machine(), 0, &p, sizeof(p));
  pokeA(sim.machine(), 0, av);
  pokeA(sim.machine(), 256, bv);
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);

  for (int i = 0; i < N; ++i) {
    const std::int32_t want = keaQuantizedAdd(av[i], bv[i], p, -100, 100);
    CHECK_EQ(sim.machine().spm_a.readI8(512 + i), want);
    CHECK_EQ(sim.machine().spm_a.readI8(i), want);
  }
}

void testVpool() {
  const int C = 8, out_h = 3, out_w = 3, kh = 7, kw = 7, sh = 2, sw = 2;
  const int in_w = (out_w - 1) * sw + kw;
  const int in_h = (out_h - 1) * sh + kh;
  const int in_row = in_w * C;
  const int out_row = out_w * C;

  std::vector<std::int8_t> in(static_cast<std::size_t>(in_h) * in_row);
  for (std::size_t i = 0; i < in.size(); ++i)
    in[i] = static_cast<std::int8_t>((i * 41 + 3) % 255 - 127);

  ProgramBuilder b;
  b.add(keaMakeVpool(false, 0, 8192, out_h, out_w, C, kh, kw, sh, sw, in_row,
                     out_row));
  b.add(keaMakeVpool(true, 0, 16384, out_h, out_w, C, kh, kw, sh, sw, in_row,
                     out_row));
  b.finish();

  Simulator sim(b.program(), SimConfig{});
  pokeA(sim.machine(), 0, in);
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);

  for (int oh = 0; oh < out_h; ++oh)
    for (int ow = 0; ow < out_w; ++ow)
      for (int c = 0; c < C; ++c) {
        std::int32_t mx = -128, sum = 0;
        for (int i = 0; i < kh; ++i)
          for (int j = 0; j < kw; ++j) {
            const std::int32_t x =
                in[(oh * sh + i) * in_row + (ow * sw + j) * C + c];
            if (x > mx) mx = x;
            sum += x;
          }
        CHECK_EQ(sim.machine().spm_a.readI8(8192 + oh * out_row + ow * C + c),
                 mx);
        CHECK_EQ(sim.machine().spm_a.readI8(16384 + oh * out_row + ow * C + c),
                 keaPoolAverage(sum, kh * kw));
      }
}

void testVcopy() {
  std::vector<std::int8_t> src(256);
  for (std::size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<std::int8_t>(i - 128);

  ProgramBuilder b;
  // 2D strided copy SPM_A -> SPM_A.
  b.add(keaMakeVcopy(/*src=*/0, /*dst=*/1024, /*row_bytes=*/16, /*rows=*/8,
                     /*src_row_stride=*/32, /*dst_row_stride=*/48));
  // Fill SPM_A with the input zero point (the conv halo pattern).
  b.add(keaMakeVfill(/*dst=*/4096, /*row_bytes=*/100, /*rows=*/3,
                     /*dst_row_stride=*/128, /*value=*/-5));
  // Copy into SPM_W.
  b.add(keaMakeVcopy(0, 0, 64, 1, 0, 0, /*src_is_w=*/false, /*dst_is_w=*/true));
  b.finish();

  Simulator sim(b.program(), SimConfig{});
  pokeA(sim.machine(), 0, src);
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);

  for (int row = 0; row < 8; ++row)
    for (int i = 0; i < 16; ++i)
      CHECK_EQ(sim.machine().spm_a.readI8(1024 + row * 48 + i),
               src[row * 32 + i]);
  for (int row = 0; row < 3; ++row)
    for (int i = 0; i < 100; ++i)
      CHECK_EQ(sim.machine().spm_a.readI8(4096 + row * 128 + i), -5);
  for (int i = 0; i < 64; ++i)
    CHECK_EQ(sim.machine().spm_w.readI8(i), src[i]);
}

void testDmaRoundTrip() {
  // Pull a [4 rows][6 cols][8 ch] tile out of a 10x10x8 DRAM feature map into
  // a dense SPM_A buffer, then store it back to a different DRAM location.
  const int IH = 10, IW = 10, C = 8;
  std::vector<std::uint8_t> fm(static_cast<std::size_t>(IH) * IW * C);
  for (std::size_t i = 0; i < fm.size(); ++i)
    fm[i] = static_cast<std::uint8_t>((i * 31 + 5) & 0xFF);

  ProgramBuilder b(1 << 20);
  b.putConst(0, fm);
  const std::uint32_t dst_dram = 65536;
  b.add(keaMakeDma(/*is_load=*/true, Unit::DMA0, /*spm_is_w=*/false,
                   /*dram_addr=*/(2 * IW + 3) * C, /*spm_addr=*/0,
                   /*len0=*/C, /*n1=*/6, /*n2=*/4,
                   /*dram_s1=*/C, /*dram_s2=*/IW * C,
                   /*spm_s1=*/C, /*spm_s2=*/6 * C));
  b.signal(Unit::DMA0, 0);
  b.wait(Unit::DMA1, 0);
  b.add(keaMakeDma(/*is_load=*/false, Unit::DMA1, false, dst_dram, 0,
                   /*len0=*/6 * C * 4, /*n1=*/1, /*n2=*/1, 0, 0, 0, 0));
  b.finish();

  Simulator sim(b.program(), SimConfig{});
  sim.stageConstants();
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);

  for (int i2 = 0; i2 < 4; ++i2)
    for (int i1 = 0; i1 < 6; ++i1)
      for (int c = 0; c < C; ++c)
        CHECK_EQ(sim.machine().spm_a.readU8(i2 * 6 * C + i1 * C + c),
                 fm[(2 + i2) * IW * C + (3 + i1) * C + c]);

  std::vector<std::uint8_t> back(6 * C * 4);
  sim.machine().dram.read(dst_dram, static_cast<std::int64_t>(back.size()),
                          back.data());
  for (std::size_t i = 0; i < back.size(); ++i)
    CHECK_EQ(back[i], sim.machine().spm_a.readU8(static_cast<std::int64_t>(i)));

  CHECK_EQ(r.stats.global.dma_bytes[0], static_cast<std::uint64_t>(6 * C * 4));
  CHECK_EQ(r.stats.global.dma_bytes[1], static_cast<std::uint64_t>(6 * C * 4));
  CHECK_EQ(r.stats.global.dram_bytes_load, static_cast<std::uint64_t>(6 * C * 4));
}

void testDmaNegativeStrideAndBroadcast() {
  std::vector<std::uint8_t> data(64);
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<std::uint8_t>(i);

  ProgramBuilder b(1 << 16);
  b.putConst(0, data);
  // Reverse the row order with a negative DRAM stride.
  b.add(keaMakeDma(true, Unit::DMA0, false, /*dram=*/48, /*spm=*/0, /*len0=*/16,
                   /*n1=*/4, /*n2=*/1, /*dram_s1=*/-16, 0, /*spm_s1=*/16, 0));
  // Broadcast one DRAM run into three SPM rows (zero DRAM stride on a load).
  b.add(keaMakeDma(true, Unit::DMA0, false, /*dram=*/0, /*spm=*/1024, 16, 3, 1,
                   /*dram_s1=*/0, 0, /*spm_s1=*/16, 0));
  b.finish();

  Simulator sim(b.program(), SimConfig{});
  sim.stageConstants();
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);

  for (int row = 0; row < 4; ++row)
    for (int i = 0; i < 16; ++i)
      CHECK_EQ(sim.machine().spm_a.readU8(row * 16 + i),
               data[(3 - row) * 16 + i]);
  for (int row = 0; row < 3; ++row)
    for (int i = 0; i < 16; ++i)
      CHECK_EQ(sim.machine().spm_a.readU8(1024 + row * 16 + i), data[i]);
}

void testNopTraceHalt() {
  ProgramBuilder b;
  b.trace(Unit::MXU, TraceKind::REGION_BEGIN, 42, 7);
  b.nop(Unit::MXU, 100);
  b.trace(Unit::MXU, TraceKind::MARKER, 43);
  b.trace(Unit::MXU, TraceKind::REGION_END, 42);
  b.halt(9);

  Simulator sim(b.program(), SimConfig{});
  SimResult r = sim.run();
  CHECK_MSG(r.ok(), r.message);
  CHECK_EQ(r.exit_code, 9u);
  CHECK_EQ(r.stats.regions.size(), std::size_t{1});
  CHECK_EQ(r.stats.regions[0].tag, 42u);
  CHECK_EQ(r.stats.regions[0].payload, 7u);
  CHECK(r.stats.regions[0].closed);
  // The region must cover at least the NOP's 100 cycles of occupancy.
  CHECK(r.stats.regions[0].c.cycles >= 100);
  CHECK_EQ(r.stats.markers.size(), std::size_t{1});
  CHECK_EQ(r.stats.markers[0].tag, 43u);
}

void testAddressFaults() {
  {  // ACC is WORD addressed: an acc_addr near the end must fault, not wrap.
    ProgramBuilder b;
    b.add(keaMakeLoadW(0, 16, 16, 16, 0, false));
    b.add(keaMakeMatmul(0, 16, 0, /*m_inner=*/64, /*m_outer=*/1,
                        /*acc_addr=*/KEA_ACC_WORDS - 16, 16, 0, 0, false,
                        false));
    b.finish();
    Simulator sim(b.program(), SimConfig{});
    SimResult r = sim.run();
    CHECK(r.status == SimStatus::Fault);
    CHECK(r.message.find("ACC") != std::string::npos);
  }
  {  // Poison: MATMUL over a never-written SPM_A region.
    ProgramBuilder b;
    b.add(keaMakeLoadW(0, 16, 16, 16, 0, false));
    b.add(keaMakeMatmul(0, 16, 0, 1, 1, 0, 16, 0, 0, false, false));
    b.finish();
    SimConfig cfg;
    cfg.strict_poison = true;
    Simulator sim(b.program(), cfg);
    SimResult r = sim.run();
    CHECK(r.status == SimStatus::Fault);
    CHECK(r.message.find("never-written") != std::string::npos);
  }
}

}  // namespace

int main() {
  testLoadWMatmulInt8();
  testLoadWTailTileZeroing();
  testMatmul2dTilingAndAccumulate();
  testMatmulInt4();
  testDwconv(3, 1);
  testDwconv(3, 2);
  testDwconv(5, 1);
  testDwconv(5, 2);
  testVquant(false);
  testVquant(true);
  testVadd();
  testVpool();
  testVcopy();
  testDmaRoundTrip();
  testDmaNegativeStrideAndBroadcast();
  testNopTraceHalt();
  testAddressFaults();
  TEST_MAIN_END();
}
