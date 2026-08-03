// SPDX-License-Identifier: Apache-2.0
#include "kea/rt/op_fields.h"

#include "kea/hw_config.h"

namespace kea {
namespace rt {
namespace {

constexpr std::int64_t kU16Max = 65535;
constexpr std::int64_t kU32Max = 4294967295LL;
constexpr std::int64_t kI32Min = -2147483648LL;
constexpr std::int64_t kI32Max = 2147483647LL;

// Symbolic value spellings. These are exactly the strings produced by
// kea::spaceName / kea::dtypeName / kea::traceKindName so the disassembler and
// the ISA helpers can never disagree.
const char* const kSpmSpaces[2] = {"SPM_A", "SPM_W"};
const char* const kDTypes[2] = {"int8", "int4"};
const char* const kBanks[2] = {"0", "1"};
const char* const kAccModes[2] = {"overwrite", "accumulate"};
const char* const kKernels[2] = {"3", "5"};
const char* const kDwStrides[2] = {"1", "2"};
const char* const kPoolModes[2] = {"max", "avg"};
const char* const kCopyModes[2] = {"copy", "fill"};
const char* const kTraceKinds[3] = {"marker", "begin", "end"};

// name, repr, offset, mask, choices, nchoices, units, min, max, align, space_field
#define FD_NUM(nm, rp, off, un, lo, hi, al) \
  { nm, FieldRepr::rp, off, 0, nullptr, 0, FieldUnits::un, lo, hi, al, nullptr }
#define FD_ADDR(nm, off, un, hi, al) \
  { nm, FieldRepr::U32, off, 0, nullptr, 0, FieldUnits::un, 0, hi, al, nullptr }
#define FD_ADDR_BY(nm, off, sp) \
  { nm, FieldRepr::U32, off, 0, nullptr, 0, FieldUnits::AddrSpmByFlag, 0, kU32Max, 0, sp }
#define FD_FLAG(nm, msk, ch) \
  { nm, FieldRepr::Flag, 2, msk, ch, 2, FieldUnits::Symbolic, 0, 1, 0, nullptr }
#define FD_ENUM8(nm, ch, n) \
  { nm, FieldRepr::Enum8, 2, 0, ch, n, FieldUnits::Symbolic, 0, (n)-1, 0, nullptr }

const FieldDef kNop[] = {
    FD_NUM("cycles", U32, 4, Count, 0, kU32Max, 0),
};

const FieldDef kHalt[] = {
    FD_NUM("exit_code", U32, 4, Count, 0, kU32Max, 0),
};

const FieldDef kSignal[] = {
    FD_NUM("event", U8, 3, Count, 0, KEA_NUM_EVENTS - 1, 0),
    FD_NUM("inc", U32, 4, Count, 0, KEA_EVENT_MAX, 0),
};

const FieldDef kWait[] = {
    FD_NUM("event", U8, 3, Count, 0, KEA_NUM_EVENTS - 1, 0),
    FD_NUM("threshold", U32, 4, Count, 0, KEA_EVENT_MAX, 0),
};

const FieldDef kTrace[] = {
    FD_ENUM8("kind", kTraceKinds, 3),
    FD_NUM("tag", U32, 4, Count, 0, kU32Max, 0),
    FD_NUM("payload", U32, 8, Count, 0, kU32Max, 0),
};

const FieldDef kDma[] = {
    FD_FLAG("spm_space", KEA_DMA_F_SPM_W, kSpmSpaces),
    FD_ADDR("dram_addr", 4, AddrDram, kU32Max, 0),
    FD_ADDR_BY("spm_addr", 8, "spm_space"),
    FD_NUM("len0", U16, 12, Bytes, 1, KEA_DMA_MAX_LEN0, 0),
    FD_NUM("n1", U16, 14, Count, 1, KEA_DMA_MAX_N1, 0),
    FD_NUM("n2", U8, 3, Count, 1, KEA_DMA_MAX_N2, 0),
    FD_NUM("dram_s1", I32, 16, Bytes, kI32Min, kI32Max, 0),
    FD_NUM("dram_s2", I32, 20, Bytes, kI32Min, kI32Max, 0),
    FD_NUM("spm_s1", I32, 24, Bytes, kI32Min, kI32Max, 0),
    FD_NUM("spm_s2", I32, 28, Bytes, kI32Min, kI32Max, 0),
};

const FieldDef kLoadW[] = {
    // Alignment of w_addr / w_row_stride depends on `dtype` (16 for int8, 8
    // for int4), so it is checked by the assembler's per-opcode hook rather
    // than by the table's static `align`.
    FD_ADDR("w_addr", 4, AddrSpmW, KEA_SPM_W_BYTES - 1, 0),
    FD_NUM("w_row_stride", I32, 8, Bytes, 0, kI32Max, 0),
    FD_NUM("k_rows", U8, 12, Count, 1, KEA_MXU_K, 0),
    FD_NUM("n_cols", U8, 13, Count, 1, KEA_MXU_N, 0),
    FD_FLAG("bank", KEA_LDW_F_BANK1, kBanks),
    FD_FLAG("dtype", KEA_LDW_F_INT4, kDTypes),
};

const FieldDef kMatmul[] = {
    FD_ADDR("a_addr", 4, AddrSpmA, KEA_SPM_A_BYTES - 1, 0),
    FD_NUM("a_inner_stride", I32, 16, Bytes, kI32Min, kI32Max, 0),
    FD_NUM("a_outer_stride", I32, 20, Bytes, kI32Min, kI32Max, 0),
    FD_NUM("m_inner", U16, 12, Count, 1, kU16Max, 0),
    FD_NUM("m_outer", U16, 14, Count, 1, kU16Max, 0),
    FD_ADDR("acc_addr", 8, AddrAcc, KEA_ACC_WORDS - 1, KEA_ALIGN_ACC_WORDS),
    FD_NUM("acc_inner_stride", I32, 24, AccWords, kI32Min, kI32Max, KEA_ALIGN_ACC_WORDS),
    FD_NUM("acc_outer_stride", I32, 28, AccWords, kI32Min, kI32Max, KEA_ALIGN_ACC_WORDS),
    FD_FLAG("bank", KEA_MM_F_BANK1, kBanks),
    FD_FLAG("acc_mode", KEA_MM_F_ACCUMULATE, kAccModes),
    FD_FLAG("dtype", KEA_MM_F_INT4, kDTypes),
};

const FieldDef kDwconv[] = {
    FD_ADDR("a_addr", 4, AddrSpmA, KEA_SPM_A_BYTES - 1, 0),
    FD_ADDR("w_addr", 8, AddrSpmW, KEA_SPM_W_BYTES - 1, 0),
    FD_ADDR("acc_addr", 12, AddrAcc, KEA_ACC_WORDS - 1, KEA_ALIGN_ACC_WORDS),
    FD_NUM("out_h", U16, 16, Count, 1, kU16Max, 0),
    FD_NUM("out_w", U16, 18, Count, 1, kU16Max, 0),
    FD_NUM("channels", U16, 20, Count, KEA_DWU_LANES, 4096, KEA_DWU_LANES),
    FD_NUM("a_row_stride", I32, 24, Bytes, kI32Min, kI32Max, 0),
    FD_NUM("a_pix_stride", I32, 28, Bytes, kI32Min, kI32Max, 0),
    FD_FLAG("kernel", KEA_DW_F_KERNEL5, kKernels),
    FD_FLAG("stride", KEA_DW_F_STRIDE2, kDwStrides),
    FD_FLAG("acc_mode", KEA_DW_F_ACCUMULATE, kAccModes),
};

const FieldDef kVquant[] = {
    FD_ADDR("acc_addr", 4, AddrAcc, KEA_ACC_WORDS - 1, KEA_ALIGN_ACC_WORDS),
    FD_ADDR("out_addr", 8, AddrSpmA, KEA_SPM_A_BYTES - 1, 0),
    FD_ADDR("qparam_addr", 12, AddrSpmW, KEA_SPM_W_BYTES - 1, KEA_ALIGN_QPARAM),
    FD_NUM("num_pixels", U32, 16, Count, 1, kU32Max, 0),
    FD_NUM("channels", U16, 20, Count, KEA_VPU_LANES, kU16Max, KEA_VPU_LANES),
    FD_NUM("acc_pix_stride", I32, 24, AccWords, kI32Min, kI32Max, KEA_ALIGN_ACC_WORDS),
    FD_NUM("out_pix_stride", I32, 28, Bytes, kI32Min, kI32Max, 0),
    FD_NUM("out_zp", I8, 3, Count, -128, 127, 0),
    FD_NUM("clamp_lo", I8, 22, Count, -128, 127, 0),
    FD_NUM("clamp_hi", I8, 23, Count, -128, 127, 0),
    FD_FLAG("dtype", KEA_VQ_F_OUT_INT4, kDTypes),
};

const FieldDef kVadd[] = {
    FD_ADDR("a_addr", 4, AddrSpmA, KEA_SPM_A_BYTES - 1, 0),
    FD_ADDR("b_addr", 8, AddrSpmA, KEA_SPM_A_BYTES - 1, 0),
    FD_ADDR("out_addr", 12, AddrSpmA, KEA_SPM_A_BYTES - 1, 0),
    FD_ADDR("param_addr", 16, AddrSpmW, KEA_SPM_W_BYTES - 1, KEA_ALIGN_QPARAM),
    FD_NUM("num_elems", U32, 20, Count, 1, kU32Max, 0),
    FD_NUM("clamp_lo", I8, 24, Count, -128, 127, 0),
    FD_NUM("clamp_hi", I8, 25, Count, -128, 127, 0),
};

const FieldDef kVpool[] = {
    FD_FLAG("mode", KEA_VP_F_AVG, kPoolModes),
    FD_ADDR("in_addr", 4, AddrSpmA, KEA_SPM_A_BYTES - 1, 0),
    FD_ADDR("out_addr", 8, AddrSpmA, KEA_SPM_A_BYTES - 1, 0),
    FD_NUM("out_h", U16, 12, Count, 1, kU16Max, 0),
    FD_NUM("out_w", U16, 14, Count, 1, kU16Max, 0),
    FD_NUM("channels", U16, 16, Count, 1, kU16Max, 0),
    FD_NUM("kh", U8, 18, Count, 1, 32, 0),
    FD_NUM("kw", U8, 19, Count, 1, 32, 0),
    FD_NUM("stride_h", U8, 20, Count, 1, 8, 0),
    FD_NUM("stride_w", U8, 21, Count, 1, 8, 0),
    FD_NUM("in_row_stride", I32, 24, Bytes, kI32Min, kI32Max, 0),
    FD_NUM("out_row_stride", I32, 28, Bytes, kI32Min, kI32Max, 0),
};

const FieldDef kVcopy[] = {
    FD_FLAG("mode", KEA_VC_F_FILL, kCopyModes),
    FD_FLAG("src_space", KEA_VC_F_SRC_W, kSpmSpaces),
    FD_FLAG("dst_space", KEA_VC_F_DST_W, kSpmSpaces),
    FD_ADDR_BY("src_addr", 4, "src_space"),
    FD_ADDR_BY("dst_addr", 8, "dst_space"),
    FD_NUM("row_bytes", U32, 12, Bytes, 1, kU32Max, 0),
    FD_NUM("rows", U32, 16, Count, 1, kU32Max, 0),
    FD_NUM("src_row_stride", I32, 20, Bytes, kI32Min, kI32Max, 0),
    FD_NUM("dst_row_stride", I32, 24, Bytes, kI32Min, kI32Max, 0),
    FD_NUM("fill_value", I8, 3, Count, -128, 127, 0),
};

#undef FD_NUM
#undef FD_ADDR
#undef FD_ADDR_BY
#undef FD_FLAG
#undef FD_ENUM8

template <std::size_t N>
constexpr std::uint8_t countOf(const FieldDef (&)[N]) {
  return static_cast<std::uint8_t>(N);
}

const OpDef kOps[] = {
    {Opcode::NOP,    "NOP",    true,  Unit::CTRL, kNop,    countOf(kNop)},
    {Opcode::HALT,   "HALT",   false, Unit::CTRL, kHalt,   countOf(kHalt)},
    {Opcode::SIGNAL, "SIGNAL", true,  Unit::CTRL, kSignal, countOf(kSignal)},
    {Opcode::WAIT,   "WAIT",   true,  Unit::CTRL, kWait,   countOf(kWait)},
    {Opcode::TRACE,  "TRACE",  true,  Unit::CTRL, kTrace,  countOf(kTrace)},
    {Opcode::DMA_LD, "DMA_LD", true,  Unit::DMA0, kDma,    countOf(kDma)},
    {Opcode::DMA_ST, "DMA_ST", true,  Unit::DMA0, kDma,    countOf(kDma)},
    {Opcode::LOAD_W, "LOAD_W", false, Unit::MXU,  kLoadW,  countOf(kLoadW)},
    {Opcode::MATMUL, "MATMUL", false, Unit::MXU,  kMatmul, countOf(kMatmul)},
    {Opcode::DWCONV, "DWCONV", false, Unit::DWU,  kDwconv, countOf(kDwconv)},
    {Opcode::VQUANT, "VQUANT", false, Unit::VPU,  kVquant, countOf(kVquant)},
    {Opcode::VADD,   "VADD",   false, Unit::VPU,  kVadd,   countOf(kVadd)},
    {Opcode::VPOOL,  "VPOOL",  false, Unit::VPU,  kVpool,  countOf(kVpool)},
    {Opcode::VCOPY,  "VCOPY",  false, Unit::VPU,  kVcopy,  countOf(kVcopy)},
};

}  // namespace

const OpDef* opDefTable(std::size_t& count) {
  count = sizeof(kOps) / sizeof(kOps[0]);
  return kOps;
}

const OpDef* opDefByMnemonic(const std::string& mnemonic) {
  for (const OpDef& d : kOps)
    if (mnemonic == d.mnemonic) return &d;
  return nullptr;
}

const OpDef* opDefFor(Opcode op) {
  for (const OpDef& d : kOps)
    if (d.opcode == op) return &d;
  return nullptr;
}

const FieldDef* fieldByName(const OpDef& op, const std::string& name) {
  for (std::uint8_t i = 0; i < op.nfields; ++i)
    if (name == op.fields[i].name) return &op.fields[i];
  return nullptr;
}

std::string operandListOf(const OpDef& op) {
  std::string s;
  if (op.unit_is_operand) s = "unit";
  for (std::uint8_t i = 0; i < op.nfields; ++i) {
    if (!s.empty()) s += ',';
    s += op.fields[i].name;
  }
  return s;
}

std::int64_t fieldGet(const KeaInstr& in, const FieldDef& f) {
  switch (f.repr) {
    case FieldRepr::Flag:
      return (in.b[2] & f.mask) ? 1 : 0;
    case FieldRepr::Enum8:
      return in.b[2];
    case FieldRepr::U8:
      return in.b[f.offset];
    case FieldRepr::I8:
      return static_cast<std::int8_t>(in.b[f.offset]);
    case FieldRepr::U16:
      return static_cast<std::int64_t>(static_cast<std::uint16_t>(
          in.b[f.offset] | (static_cast<std::uint16_t>(in.b[f.offset + 1]) << 8)));
    case FieldRepr::U32:
    case FieldRepr::I32: {
      std::uint32_t v = 0;
      for (int i = 3; i >= 0; --i)
        v = (v << 8) | in.b[f.offset + static_cast<std::uint8_t>(i)];
      if (f.repr == FieldRepr::I32) return static_cast<std::int32_t>(v);
      return static_cast<std::int64_t>(v);
    }
  }
  return 0;
}

void fieldSet(KeaInstr& in, const FieldDef& f, std::int64_t value) {
  switch (f.repr) {
    case FieldRepr::Flag:
      if (value)
        in.b[2] = static_cast<std::uint8_t>(in.b[2] | f.mask);
      else
        in.b[2] = static_cast<std::uint8_t>(in.b[2] & static_cast<std::uint8_t>(~f.mask));
      return;
    case FieldRepr::Enum8:
      in.b[2] = static_cast<std::uint8_t>(value);
      return;
    case FieldRepr::U8:
    case FieldRepr::I8:
      in.b[f.offset] = static_cast<std::uint8_t>(value & 0xFF);
      return;
    case FieldRepr::U16: {
      const std::uint16_t v = static_cast<std::uint16_t>(value);
      in.b[f.offset] = static_cast<std::uint8_t>(v & 0xFF);
      in.b[f.offset + 1] = static_cast<std::uint8_t>((v >> 8) & 0xFF);
      return;
    }
    case FieldRepr::U32:
    case FieldRepr::I32: {
      const std::uint32_t v = static_cast<std::uint32_t>(value);
      for (std::uint8_t i = 0; i < 4; ++i)
        in.b[f.offset + i] = static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF);
      return;
    }
  }
}

Space fieldSpace(const OpDef& op, const FieldDef& f, const KeaInstr& in) {
  switch (f.units) {
    case FieldUnits::AddrSpmA: return Space::SPM_A;
    case FieldUnits::AddrSpmW: return Space::SPM_W;
    case FieldUnits::AddrAcc:  return Space::ACC;
    case FieldUnits::AddrDram: return Space::DRAM;
    case FieldUnits::AddrSpmByFlag: {
      const FieldDef* sf = fieldByName(op, f.space_field);
      return (sf && fieldGet(in, *sf) == 1) ? Space::SPM_W : Space::SPM_A;
    }
    default: return Space::SPM_A;
  }
}

std::uint64_t spaceCapacity(Space s) {
  switch (s) {
    case Space::SPM_A: return KEA_SPM_A_BYTES;
    case Space::SPM_W: return KEA_SPM_W_BYTES;
    case Space::ACC:   return KEA_ACC_WORDS;
    case Space::DRAM:  return KEA_DRAM_BYTES;
  }
  return 0;
}

const char* spaceUnitName(Space s) {
  return (s == Space::ACC) ? "int32 words" : "bytes";
}

}  // namespace rt
}  // namespace kea
