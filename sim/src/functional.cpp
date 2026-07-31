// SPDX-License-Identifier: Apache-2.0
//
// functional.cpp --- the bit-exact functional model.
//
// Every opcode's semantics come straight out of docs/ISA.md, and every piece
// of arithmetic that has a reference implementation in include/kea/isa.h calls
// that implementation rather than restating it.
//
// Two things bite everybody once and are called out explicitly below:
//   * ACC is addressed in int32 WORDS, not bytes.
//   * int4 activations are little-nibble-first, two elements per byte.

#include <string>
#include <vector>

#include "kea/sim/simulator.h"

namespace kea {
namespace sim {
namespace {

/// Bounding interval of a 2D strided walk, so a whole instruction's footprint
/// can be range/hazard checked once instead of per row.
struct Extent {
  std::int64_t lo;
  std::int64_t hi;  // exclusive
};

Extent walkExtent(std::int64_t base, std::int64_t inner_stride,
                  std::int64_t inner_count, std::int64_t outer_stride,
                  std::int64_t outer_count, std::int64_t elem) {
  const std::int64_t di = (inner_count - 1) * inner_stride;
  const std::int64_t doo = (outer_count - 1) * outer_stride;
  Extent e;
  e.lo = base + (di < 0 ? di : 0) + (doo < 0 ? doo : 0);
  e.hi = base + (di > 0 ? di : 0) + (doo > 0 ? doo : 0) + elem;
  return e;
}

std::string hex(std::int64_t v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v));
  return buf;
}

}  // namespace

// ---------------------------------------------------------------------------
// Diagnostics helpers
// ---------------------------------------------------------------------------

void Simulator::fault(std::uint32_t pc, const std::string& msg) {
  if (status_ == SimStatus::Ok) {
    status_ = SimStatus::Fault;
    message_ = "pc " + std::to_string(pc) + ": " + msg;
  }
  warn(pc, msg);
}

void Simulator::warn(std::uint32_t pc, const std::string& msg) {
  if (diagnostics_.size() < cfg_.max_diagnostics)
    diagnostics_.push_back("pc " + std::to_string(pc) + " cycle " +
                           std::to_string(cycle_) + ": " + msg);
}

void Simulator::notePoison(std::uint32_t pc, const char* space,
                           std::int64_t addr, std::int64_t len) {
  ++poison_reads_;
  const std::string m = std::string("read of never-written ") + space + " [" +
                        hex(addr) + ", " + hex(addr + len) +
                        ") -- missing DMA/VCOPY or missing SIGNAL/WAIT";
  if (cfg_.strict_poison)
    fault(pc, m);
  else if (poison_reads_ <= 8)
    warn(pc, m);
}

bool Simulator::checkSpm(std::uint32_t pc, const Scratchpad& s, const char* name,
                         std::int64_t addr, std::int64_t len) {
  if (!s.inRange(addr, len)) {
    fault(pc, std::string(name) + " access [" + hex(addr) + ", " +
                  hex(addr + len) + ") escapes the " + hex(0) + ".." +
                  hex(static_cast<std::int64_t>(s.size())) + " space");
    return false;
  }
  return true;
}

bool Simulator::checkAcc(std::uint32_t pc, std::int64_t word,
                         std::int64_t count) {
  if (!mem_.acc.inRange(word, count)) {
    fault(pc, "ACC access [" + hex(word) + ", " + hex(word + count) +
                  ") words escapes the 0.." +
                  hex(static_cast<std::int64_t>(mem_.acc.words())) +
                  " word space (remember: ACC is word addressed)");
    return false;
  }
  return true;
}

bool Simulator::checkDram(std::uint32_t pc, std::int64_t addr,
                          std::int64_t len) {
  if (!Dram::inRange(addr, len)) {
    fault(pc, "DRAM access [" + hex(addr) + ", " + hex(addr + len) +
                  ") escapes the 4 GiB address space");
    return false;
  }
  return true;
}

bool Simulator::readableSpm(std::uint32_t pc, const Scratchpad& s,
                            const char* name, std::int64_t addr,
                            std::int64_t len) {
  if (!checkSpm(pc, s, name, addr, len)) return false;
  if (!s.defined(addr, len)) notePoison(pc, name, addr, len);
  return status_ == SimStatus::Ok;
}

bool Simulator::readableAcc(std::uint32_t pc, std::int64_t word,
                            std::int64_t count) {
  if (!checkAcc(pc, word, count)) return false;
  if (!mem_.acc.defined(word, count)) notePoison(pc, "ACC", word, count);
  return status_ == SimStatus::Ok;
}

// ---------------------------------------------------------------------------
// Cross-unit hazard tracking
// ---------------------------------------------------------------------------

void Simulator::addInflightWrite(int space, std::int64_t lo, std::int64_t hi,
                                 std::uint64_t retire, std::uint32_t pc,
                                 int unit, std::size_t* slot_out) {
  std::size_t slot = inflight_.size();
  for (std::size_t i = 0; i < inflight_.size(); ++i) {
    if (!inflight_[i].live) {
      slot = i;
      break;
    }
  }
  if (slot == inflight_.size()) inflight_.push_back(InflightWrite{});
  InflightWrite& w = inflight_[slot];
  w.space = space;
  w.lo = lo;
  w.hi = hi;
  w.retire = retire;
  w.pc = pc;
  w.unit = unit;
  w.live = true;
  if (slot_out) *slot_out = slot;
}

void Simulator::checkHazard(int space, std::int64_t lo, std::int64_t hi,
                            std::uint32_t pc, int unit) {
  for (const InflightWrite& w : inflight_) {
    if (!w.live || w.space != space || w.unit == unit) continue;
    if (w.retire <= cycle_) continue;
    if (hi <= w.lo || w.hi <= lo) continue;
    ++hazards_;
    const std::string m =
        std::string("unsynchronized read of ") +
        spaceName(static_cast<Space>(space)) + " [" + hex(lo) + ", " +
        hex(hi) + ") while " + unitName(static_cast<Unit>(w.unit)) +
        " pc " + std::to_string(w.pc) + " is still writing [" + hex(w.lo) +
        ", " + hex(w.hi) + ") -- a SIGNAL/WAIT pair is missing";
    if (cfg_.strict_hazards)
      fault(pc, m);
    else if (hazards_ <= 8)
      warn(pc, m);
    return;
  }
}

void Simulator::pruneInflight() {
  for (InflightWrite& w : inflight_)
    if (w.live && w.retire <= cycle_) w.live = false;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

void Simulator::execute(std::uint32_t pc, const KeaInstr& in, int unit) {
  switch (keaOpcode(in)) {
    case Opcode::LOAD_W: execLoadW(pc, in); break;
    case Opcode::MATMUL: execMatmul(pc, in); break;
    case Opcode::DWCONV: execDwconv(pc, in); break;
    case Opcode::VQUANT: execVquant(pc, in); break;
    case Opcode::VADD:   execVadd(pc, in); break;
    case Opcode::VPOOL:  execVpool(pc, in); break;
    case Opcode::VCOPY:  execVcopy(pc, in); break;
    case Opcode::DMA_LD:
    case Opcode::DMA_ST: execDma(pc, in, unit); break;
    default: break;  // NOP / HALT / SIGNAL / WAIT / TRACE have no data effect
  }
}

// ---------------------------------------------------------------------------
// MXU
// ---------------------------------------------------------------------------

void Simulator::execLoadW(std::uint32_t pc, const KeaInstr& in) {
  const KeaLoadW w = keaDecode<KeaLoadW>(in);
  const bool int4 = (w.h.flags & KEA_LDW_F_INT4) != 0;
  const int bank = (w.h.flags & KEA_LDW_F_BANK1) ? 1 : 0;
  const std::int64_t row_bytes = int4 ? 8 : 16;

  const std::int64_t base = w.w_addr;
  const std::int64_t stride = w.w_row_stride;
  const std::int64_t lo = base;
  const std::int64_t hi = base + (w.k_rows - 1) * stride + row_bytes;
  if (!readableSpm(pc, mem_.spm_w, "SPM_W", lo, hi - lo)) return;
  checkHazard(static_cast<int>(Space::SPM_W), lo, hi, pc, KEA_UNIT_MXU);

  // Rows >= k_rows and columns >= n_cols load as ZERO.  That is how tail
  // tiles work for channel counts that are not multiples of 16 (ISA.md §7.2),
  // and it is why MATMUL never needs masking.
  for (int k = 0; k < KEA_MXU_K; ++k) {
    for (int n = 0; n < KEA_MXU_N; ++n) {
      std::int8_t v = 0;
      if (k < w.k_rows && n < w.n_cols) {
        const std::int64_t row = base + static_cast<std::int64_t>(k) * stride;
        v = int4 ? keaUnpackInt4(mem_.spm_w.readU8(row + n / 2),
                                 static_cast<unsigned>(n & 1))
                 : mem_.spm_w.readI8(row + n);
      }
      mem_.wbank[bank][k][n] = v;
    }
  }
  mem_.wbank_k_rows[bank] = w.k_rows;
  mem_.wbank_n_cols[bank] = w.n_cols;
  mem_.wbank_loaded[bank] = true;
}

void Simulator::execMatmul(std::uint32_t pc, const KeaInstr& in) {
  const KeaMatmul m = keaDecode<KeaMatmul>(in);
  const bool accumulate = (m.h.flags & KEA_MM_F_ACCUMULATE) != 0;
  const bool int4 = (m.h.flags & KEA_MM_F_INT4) != 0;
  const int bank = (m.h.flags & KEA_MM_F_BANK1) ? 1 : 0;
  const std::int64_t row_bytes = int4 ? 8 : 16;

  if (!mem_.wbank_loaded[bank]) {
    warn(pc, "MATMUL reads weight bank " + std::to_string(bank) +
                 " before any LOAD_W has targeted it (reset state is "
                 "architecturally undefined; simulated as zero)");
    mem_.wbank_loaded[bank] = true;  // report once
  }

  const Extent ae =
      walkExtent(m.a_addr, m.a_inner_stride, m.m_inner, m.a_outer_stride,
                 m.m_outer, row_bytes);
  const Extent qe =
      walkExtent(m.acc_addr, m.acc_inner_stride, m.m_inner, m.acc_outer_stride,
                 m.m_outer, KEA_MXU_N);
  if (!checkSpm(pc, mem_.spm_a, "SPM_A", ae.lo, ae.hi - ae.lo)) return;
  if (!checkAcc(pc, qe.lo, qe.hi - qe.lo)) return;
  checkHazard(static_cast<int>(Space::SPM_A), ae.lo, ae.hi, pc, KEA_UNIT_MXU);

  const std::int8_t(*W)[KEA_MXU_N] = mem_.wbank[bank];

  for (std::int64_t mo = 0; mo < m.m_outer; ++mo) {
    const std::int64_t a_row = static_cast<std::int64_t>(m.a_addr) +
                               mo * m.a_outer_stride;
    const std::int64_t q_row = static_cast<std::int64_t>(m.acc_addr) +
                               mo * m.acc_outer_stride;
    for (std::int64_t mi = 0; mi < m.m_inner; ++mi) {
      const std::int64_t a = a_row + mi * m.a_inner_stride;
      const std::int64_t q = q_row + mi * m.acc_inner_stride;
      if (!mem_.spm_a.defined(a, row_bytes))
        notePoison(pc, "SPM_A", a, row_bytes);
      if (accumulate && !mem_.acc.defined(q, KEA_MXU_N))
        notePoison(pc, "ACC", q, KEA_MXU_N);
      if (status_ != SimStatus::Ok) return;

      // The array always reads 16 activation elements per row regardless of
      // the resident tile's k_rows; unused lanes multiply the zeros LOAD_W
      // installed.
      std::int32_t av[KEA_MXU_K];
      if (int4) {
        for (int k = 0; k < KEA_MXU_K; ++k)
          av[k] = keaUnpackInt4(mem_.spm_a.readU8(a + k / 2),
                                static_cast<unsigned>(k & 1));
      } else {
        for (int k = 0; k < KEA_MXU_K; ++k) av[k] = mem_.spm_a.readI8(a + k);
      }

      for (int n = 0; n < KEA_MXU_N; ++n) {
        std::int32_t sum = 0;
        for (int k = 0; k < KEA_MXU_K; ++k) sum += av[k] * W[k][n];
        // ISA.md does not define int32 accumulator overflow.  KEA-1 wraps
        // (two's complement); see docs/SIMULATOR.md "Ambiguities".
        const std::uint32_t base =
            accumulate ? static_cast<std::uint32_t>(mem_.acc.read(q + n)) : 0u;
        mem_.acc.write(q + n, static_cast<std::int32_t>(
                                  base + static_cast<std::uint32_t>(sum)));
      }
    }
  }
  addInflightWrite(static_cast<int>(Space::ACC), qe.lo, qe.hi, exec_retire_, pc,
                   KEA_UNIT_MXU, nullptr);
}

// ---------------------------------------------------------------------------
// DWU
// ---------------------------------------------------------------------------

void Simulator::execDwconv(std::uint32_t pc, const KeaInstr& in) {
  const KeaDwconv d = keaDecode<KeaDwconv>(in);
  const int K = (d.h.flags & KEA_DW_F_KERNEL5) ? KEA_DWU_KERNEL_5
                                               : KEA_DWU_KERNEL_3;
  const int S = (d.h.flags & KEA_DW_F_STRIDE2) ? 2 : 1;
  const bool accumulate = (d.h.flags & KEA_DW_F_ACCUMULATE) != 0;
  const std::int64_t C = d.channels;

  const Extent ae = walkExtent(
      d.a_addr, d.a_pix_stride, static_cast<std::int64_t>(d.out_w - 1) * S + K,
      d.a_row_stride, static_cast<std::int64_t>(d.out_h - 1) * S + K, C);
  const std::int64_t w_bytes = static_cast<std::int64_t>(K) * K * C;
  const std::int64_t q_words =
      static_cast<std::int64_t>(d.out_h) * d.out_w * C;
  if (!checkSpm(pc, mem_.spm_a, "SPM_A", ae.lo, ae.hi - ae.lo)) return;
  if (!readableSpm(pc, mem_.spm_w, "SPM_W", d.w_addr, w_bytes)) return;
  if (!checkAcc(pc, d.acc_addr, q_words)) return;
  checkHazard(static_cast<int>(Space::SPM_A), ae.lo, ae.hi, pc, KEA_UNIT_DWU);
  checkHazard(static_cast<int>(Space::SPM_W), d.w_addr, d.w_addr + w_bytes, pc,
              KEA_UNIT_DWU);

  const std::uint8_t* wp = mem_.spm_w.data() + d.w_addr;

  for (std::int64_t oh = 0; oh < d.out_h; ++oh) {
    for (std::int64_t ow = 0; ow < d.out_w; ++ow) {
      const std::int64_t q =
          static_cast<std::int64_t>(d.acc_addr) + (oh * d.out_w + ow) * C;
      if (!accumulate) {
        for (std::int64_t c = 0; c < C; ++c) mem_.acc.write(q + c, 0);
      } else if (!mem_.acc.defined(q, C)) {
        notePoison(pc, "ACC", q, C);
        if (status_ != SimStatus::Ok) return;
      }
      for (int kh = 0; kh < K; ++kh) {
        const std::int64_t arow =
            static_cast<std::int64_t>(d.a_addr) +
            (oh * S + kh) * static_cast<std::int64_t>(d.a_row_stride);
        for (int kw = 0; kw < K; ++kw) {
          const std::int64_t a =
              arow + (ow * S + kw) * static_cast<std::int64_t>(d.a_pix_stride);
          if (!mem_.spm_a.defined(a, C)) {
            notePoison(pc, "SPM_A", a, C);
            if (status_ != SimStatus::Ok) return;
          }
          const std::uint8_t* ap = mem_.spm_a.data() + a;
          const std::uint8_t* tap = wp + static_cast<std::int64_t>(kh * K + kw) * C;
          for (std::int64_t c = 0; c < C; ++c) {
            const std::int32_t prod = static_cast<std::int32_t>(
                static_cast<std::int8_t>(ap[c])) *
                static_cast<std::int32_t>(static_cast<std::int8_t>(tap[c]));
            mem_.acc.write(q + c, static_cast<std::int32_t>(
                                      static_cast<std::uint32_t>(
                                          mem_.acc.read(q + c)) +
                                      static_cast<std::uint32_t>(prod)));
          }
        }
      }
    }
  }
  addInflightWrite(static_cast<int>(Space::ACC), d.acc_addr,
                   d.acc_addr + q_words, exec_retire_, pc, KEA_UNIT_DWU,
                   nullptr);
}

// ---------------------------------------------------------------------------
// VPU
// ---------------------------------------------------------------------------

void Simulator::execVquant(std::uint32_t pc, const KeaInstr& in) {
  const KeaVquant v = keaDecode<KeaVquant>(in);
  const bool int4 = (v.h.flags & KEA_VQ_F_OUT_INT4) != 0;
  const std::int64_t C = v.channels;
  const std::int32_t out_zp = static_cast<std::int8_t>(v.h.aux);

  const std::int64_t qp_bytes = C * static_cast<std::int64_t>(sizeof(KeaQuantParam));
  if (!readableSpm(pc, mem_.spm_w, "SPM_W", v.qparam_addr, qp_bytes)) return;
  checkHazard(static_cast<int>(Space::SPM_W), v.qparam_addr,
              v.qparam_addr + qp_bytes, pc, KEA_UNIT_VPU);

  std::vector<KeaQuantParam> qp(static_cast<std::size_t>(C));
  for (std::int64_t c = 0; c < C; ++c) {
    const std::uint8_t* p =
        mem_.spm_w.data() + v.qparam_addr + c * static_cast<std::int64_t>(sizeof(KeaQuantParam));
    KeaQuantParam r{};
    std::copy_n(p, sizeof(KeaQuantParam), reinterpret_cast<std::uint8_t*>(&r));
    qp[static_cast<std::size_t>(c)] = r;
  }

  const std::int64_t out_elem = int4 ? (C + 1) / 2 : C;
  const Extent ae = walkExtent(v.acc_addr, v.acc_pix_stride, v.num_pixels, 0, 1, C);
  const Extent oe = walkExtent(v.out_addr, v.out_pix_stride, v.num_pixels, 0, 1, out_elem);
  if (!checkAcc(pc, ae.lo, ae.hi - ae.lo)) return;
  if (!checkSpm(pc, mem_.spm_a, "SPM_A", oe.lo, oe.hi - oe.lo)) return;
  checkHazard(static_cast<int>(Space::ACC), ae.lo, ae.hi, pc, KEA_UNIT_VPU);

  for (std::int64_t p = 0; p < v.num_pixels; ++p) {
    const std::int64_t q =
        static_cast<std::int64_t>(v.acc_addr) + p * v.acc_pix_stride;
    const std::int64_t o =
        static_cast<std::int64_t>(v.out_addr) + p * v.out_pix_stride;
    if (!mem_.acc.defined(q, C)) {
      notePoison(pc, "ACC", q, C);
      if (status_ != SimStatus::Ok) return;
    }
    for (std::int64_t c = 0; c < C; ++c) {
      const KeaQuantParam& r = qp[static_cast<std::size_t>(c)];
      const std::int32_t acc_v = static_cast<std::int32_t>(
          static_cast<std::uint32_t>(mem_.acc.read(q + c)) +
          static_cast<std::uint32_t>(r.bias));
      const std::int32_t out = keaRequantize(acc_v, r.mult, r.shift, out_zp,
                                             v.clamp_lo, v.clamp_hi);
      if (int4) {
        const std::int64_t byte = o + c / 2;
        // Read-modify-write of a nibble pair.  Deliberately reads the raw
        // byte without a poison check: a half-written byte is normal here.
        std::uint8_t cur = mem_.spm_a.readU8(byte);
        if (c & 1)
          cur = static_cast<std::uint8_t>((cur & 0x0Fu) |
                                          ((static_cast<std::uint8_t>(out) & 0x0Fu) << 4));
        else
          cur = static_cast<std::uint8_t>((cur & 0xF0u) |
                                          (static_cast<std::uint8_t>(out) & 0x0Fu));
        mem_.spm_a.write(byte, cur);
      } else {
        mem_.spm_a.write(o + c, static_cast<std::uint8_t>(
                                    static_cast<std::int8_t>(out)));
      }
    }
  }
  addInflightWrite(static_cast<int>(Space::SPM_A), oe.lo, oe.hi, exec_retire_,
                   pc, KEA_UNIT_VPU, nullptr);
}

void Simulator::execVadd(std::uint32_t pc, const KeaInstr& in) {
  const KeaVadd v = keaDecode<KeaVadd>(in);
  const std::int64_t n = v.num_elems;

  if (!readableSpm(pc, mem_.spm_w, "SPM_W", v.param_addr, sizeof(KeaAddParam)))
    return;
  KeaAddParam p{};
  std::copy_n(mem_.spm_w.data() + v.param_addr, sizeof(KeaAddParam),
              reinterpret_cast<std::uint8_t*>(&p));

  if (!readableSpm(pc, mem_.spm_a, "SPM_A", v.a_addr, n)) return;
  if (!readableSpm(pc, mem_.spm_a, "SPM_A", v.b_addr, n)) return;
  if (!checkSpm(pc, mem_.spm_a, "SPM_A", v.out_addr, n)) return;
  checkHazard(static_cast<int>(Space::SPM_A), v.a_addr, v.a_addr + n, pc,
              KEA_UNIT_VPU);
  checkHazard(static_cast<int>(Space::SPM_A), v.b_addr, v.b_addr + n, pc,
              KEA_UNIT_VPU);

  // In-place is legal (ISA.md §10.2): the VPU reads before it writes within a
  // lane, so a straight forward loop is correct even when out == a.
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int32_t a = mem_.spm_a.readI8(v.a_addr + i);
    const std::int32_t b = mem_.spm_a.readI8(v.b_addr + i);
    const std::int32_t o = keaQuantizedAdd(a, b, p, v.clamp_lo, v.clamp_hi);
    mem_.spm_a.write(v.out_addr + i,
                     static_cast<std::uint8_t>(static_cast<std::int8_t>(o)));
  }
  addInflightWrite(static_cast<int>(Space::SPM_A), v.out_addr, v.out_addr + n,
                   exec_retire_, pc, KEA_UNIT_VPU, nullptr);
}

void Simulator::execVpool(std::uint32_t pc, const KeaInstr& in) {
  const KeaVpool v = keaDecode<KeaVpool>(in);
  const bool avg = (v.h.flags & KEA_VP_F_AVG) != 0;
  const std::int64_t C = v.channels;

  const Extent ie = walkExtent(
      v.in_addr, C, static_cast<std::int64_t>(v.out_w - 1) * v.stride_w + v.kw,
      v.in_row_stride, static_cast<std::int64_t>(v.out_h - 1) * v.stride_h + v.kh,
      C);
  const Extent oe =
      walkExtent(v.out_addr, C, v.out_w, v.out_row_stride, v.out_h, C);
  if (!checkSpm(pc, mem_.spm_a, "SPM_A", ie.lo, ie.hi - ie.lo)) return;
  if (!checkSpm(pc, mem_.spm_a, "SPM_A", oe.lo, oe.hi - oe.lo)) return;
  checkHazard(static_cast<int>(Space::SPM_A), ie.lo, ie.hi, pc, KEA_UNIT_VPU);

  const std::int32_t count = static_cast<std::int32_t>(v.kh) * v.kw;
  for (std::int64_t oh = 0; oh < v.out_h; ++oh) {
    for (std::int64_t ow = 0; ow < v.out_w; ++ow) {
      const std::int64_t o = static_cast<std::int64_t>(v.out_addr) +
                             oh * v.out_row_stride + ow * C;
      for (std::int64_t c = 0; c < C; ++c) {
        std::int32_t acc = avg ? 0 : -128;
        for (int i = 0; i < v.kh; ++i) {
          const std::int64_t row = static_cast<std::int64_t>(v.in_addr) +
                                   (oh * v.stride_h + i) * v.in_row_stride;
          for (int j = 0; j < v.kw; ++j) {
            const std::int64_t a = row + (ow * v.stride_w + j) * C + c;
            if (!mem_.spm_a.defined(a, 1)) {
              notePoison(pc, "SPM_A", a, 1);
              if (status_ != SimStatus::Ok) return;
            }
            const std::int32_t x = mem_.spm_a.readI8(a);
            if (avg)
              acc += x;
            else if (x > acc)
              acc = x;
          }
        }
        const std::int32_t out = avg ? keaPoolAverage(acc, count) : acc;
        mem_.spm_a.write(o + c,
                         static_cast<std::uint8_t>(static_cast<std::int8_t>(out)));
      }
    }
  }
  addInflightWrite(static_cast<int>(Space::SPM_A), oe.lo, oe.hi, exec_retire_,
                   pc, KEA_UNIT_VPU, nullptr);
}

void Simulator::execVcopy(std::uint32_t pc, const KeaInstr& in) {
  const KeaVcopy v = keaDecode<KeaVcopy>(in);
  const bool fill = (v.h.flags & KEA_VC_F_FILL) != 0;
  Scratchpad& src = (v.h.flags & KEA_VC_F_SRC_W) ? mem_.spm_w : mem_.spm_a;
  Scratchpad& dst = (v.h.flags & KEA_VC_F_DST_W) ? mem_.spm_w : mem_.spm_a;
  const char* src_name = (v.h.flags & KEA_VC_F_SRC_W) ? "SPM_W" : "SPM_A";
  const char* dst_name = (v.h.flags & KEA_VC_F_DST_W) ? "SPM_W" : "SPM_A";
  const int dst_space = (v.h.flags & KEA_VC_F_DST_W) ? static_cast<int>(Space::SPM_W)
                                                     : static_cast<int>(Space::SPM_A);
  const std::uint8_t value = v.h.aux;
  const std::int64_t rb = v.row_bytes;

  const Extent se = walkExtent(v.src_addr, v.src_row_stride, v.rows, 0, 1, rb);
  const Extent de = walkExtent(v.dst_addr, v.dst_row_stride, v.rows, 0, 1, rb);
  if (!fill && !checkSpm(pc, src, src_name, se.lo, se.hi - se.lo)) return;
  if (!checkSpm(pc, dst, dst_name, de.lo, de.hi - de.lo)) return;
  if (!fill) {
    const int src_space = (v.h.flags & KEA_VC_F_SRC_W) ? static_cast<int>(Space::SPM_W)
                                                       : static_cast<int>(Space::SPM_A);
    checkHazard(src_space, se.lo, se.hi, pc, KEA_UNIT_VPU);
  }

  for (std::int64_t r = 0; r < v.rows; ++r) {
    const std::int64_t d = static_cast<std::int64_t>(v.dst_addr) + r * v.dst_row_stride;
    if (fill) {
      std::fill_n(dst.data() + d, static_cast<std::size_t>(rb), value);
      dst.markDefined(d, rb);
    } else {
      const std::int64_t s = static_cast<std::int64_t>(v.src_addr) + r * v.src_row_stride;
      if (!src.defined(s, rb)) {
        notePoison(pc, src_name, s, rb);
        if (status_ != SimStatus::Ok) return;
      }
      std::copy_n(src.data() + s, static_cast<std::size_t>(rb), dst.data() + d);
      dst.markDefined(d, rb);
    }
  }
  addInflightWrite(dst_space, de.lo, de.hi, exec_retire_, pc, KEA_UNIT_VPU,
                   nullptr);
}

// ---------------------------------------------------------------------------
// DMA
// ---------------------------------------------------------------------------

void Simulator::execDma(std::uint32_t pc, const KeaInstr& in, int unit) {
  const KeaDma d = keaDecode<KeaDma>(in);
  const bool is_load = keaOpcode(in) == Opcode::DMA_LD;
  Scratchpad& spm = (d.h.flags & KEA_DMA_F_SPM_W) ? mem_.spm_w : mem_.spm_a;
  const char* spm_name = (d.h.flags & KEA_DMA_F_SPM_W) ? "SPM_W" : "SPM_A";
  const int spm_space = (d.h.flags & KEA_DMA_F_SPM_W)
                            ? static_cast<int>(Space::SPM_W)
                            : static_cast<int>(Space::SPM_A);
  const std::int64_t len0 = d.len0;
  const std::int64_t n1 = d.n1;
  const std::int64_t n2 = d.h.aux;

  const Extent se = walkExtent(d.spm_addr, d.spm_s1, n1, d.spm_s2, n2, len0);
  const Extent de = walkExtent(d.dram_addr, d.dram_s1, n1, d.dram_s2, n2, len0);
  if (!checkSpm(pc, spm, spm_name, se.lo, se.hi - se.lo)) return;
  if (!checkDram(pc, de.lo, de.hi - de.lo)) return;

  std::vector<std::uint8_t> buf(static_cast<std::size_t>(len0));
  for (std::int64_t i2 = 0; i2 < n2; ++i2) {
    for (std::int64_t i1 = 0; i1 < n1; ++i1) {
      const std::int64_t da = static_cast<std::int64_t>(d.dram_addr) +
                              i2 * d.dram_s2 + i1 * d.dram_s1;
      const std::int64_t sa = static_cast<std::int64_t>(d.spm_addr) +
                              i2 * d.spm_s2 + i1 * d.spm_s1;
      if (is_load) {
        mem_.dram.read(da, len0, buf.data());
        std::copy_n(buf.data(), static_cast<std::size_t>(len0), spm.data() + sa);
        spm.markDefined(sa, len0);
      } else {
        if (!spm.defined(sa, len0)) {
          notePoison(pc, spm_name, sa, len0);
          if (status_ != SimStatus::Ok) return;
        }
        std::copy_n(spm.data() + sa, static_cast<std::size_t>(len0), buf.data());
        mem_.dram.write(da, len0, buf.data());
      }
    }
  }

  if (is_load) {
    checkHazard(static_cast<int>(Space::DRAM), de.lo, de.hi, pc, unit);
    std::size_t slot = 0;
    addInflightWrite(spm_space, se.lo, se.hi, /*retire=*/UINT64_MAX, pc, unit,
                     &slot);
    dma_[unit - KEA_UNIT_DMA0].hazard_idx = slot;
  } else {
    checkHazard(spm_space, se.lo, se.hi, pc, unit);
    std::size_t slot = 0;
    addInflightWrite(static_cast<int>(Space::DRAM), de.lo, de.hi,
                     /*retire=*/UINT64_MAX, pc, unit, &slot);
    dma_[unit - KEA_UNIT_DMA0].hazard_idx = slot;
  }
}

}  // namespace sim
}  // namespace kea
