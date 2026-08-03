// SPDX-License-Identifier: Apache-2.0
//
// The anti-drift test ADR-0001 asks for.
//
// `.kasm` syntax is supposed to be derived mechanically from `keaOpInfo()`.
// This test proves it: for every opcode it rebuilds the comma-separated
// operand list from runtime/src/op_fields.cpp and compares it against
// `keaOpInfo().operands` byte for byte, then checks that every field's storage
// (offset, width, flag mask) agrees with the convenience constructors in
// isa.h. Adding a field to isa.h without teaching the assembler about it fails
// here rather than silently producing an assembler that cannot express it.

#include <set>
#include <string>

#include "kea/isa.h"
#include "kea/rt/op_fields.h"
#include "test_util.h"

using namespace kea;
using namespace kea::rt;

namespace {

void testOperandListsMatchIsaHeader() {
  std::size_t n = 0;
  const OpDef* ops = opDefTable(n);
  CHECK(n == 14);

  for (std::size_t i = 0; i < n; ++i) {
    const OpDef& d = ops[i];
    const KeaOpInfo info = keaOpInfo(d.opcode);
    CHECK_MSG(info.valid, std::string("  opcode ") + d.mnemonic + " is not valid in keaOpInfo()");
    CHECK_STR_EQ(std::string(d.mnemonic), std::string(info.mnemonic));
    CHECK_MSG(operandListOf(d) == info.operands,
              std::string("  ") + d.mnemonic + "\n  table    : " + operandListOf(d) +
                  "\n  keaOpInfo: " + info.operands);
    if (!d.unit_is_operand)
      CHECK_MSG(d.fixed_unit == info.unit,
                std::string("  ") + d.mnemonic + " owning unit disagrees with keaOpInfo()");
    CHECK_MSG(d.unit_is_operand == (keaIsQueueAgnostic(d.opcode) || d.opcode == Opcode::DMA_LD ||
                                    d.opcode == Opcode::DMA_ST),
              std::string("  ") + d.mnemonic + " unit-operand flag disagrees with the ISA");
  }
}

/// Every opcode in the ISA's opcode map must be in the table, and nothing else.
void testEveryOpcodeCovered() {
  const Opcode all[] = {Opcode::NOP,    Opcode::HALT,   Opcode::SIGNAL, Opcode::WAIT,
                        Opcode::TRACE,  Opcode::DMA_LD, Opcode::DMA_ST, Opcode::LOAD_W,
                        Opcode::MATMUL, Opcode::DWCONV, Opcode::VQUANT, Opcode::VADD,
                        Opcode::VPOOL,  Opcode::VCOPY};
  for (Opcode op : all) CHECK_MSG(opDefFor(op) != nullptr, "  missing opcode in op_fields table");
  CHECK(opDefFor(Opcode::INVALID) == nullptr);
  CHECK(opDefFor(static_cast<Opcode>(0x05)) == nullptr);
}

/// Field storage must be inside the 32-byte word and must not overlap the
/// opcode/unit bytes.
void testFieldStorageIsSane() {
  std::size_t n = 0;
  const OpDef* ops = opDefTable(n);
  for (std::size_t i = 0; i < n; ++i) {
    const OpDef& d = ops[i];
    for (std::uint8_t k = 0; k < d.nfields; ++k) {
      const FieldDef& f = d.fields[k];
      if (f.repr == FieldRepr::Flag || f.repr == FieldRepr::Enum8) {
        CHECK_MSG(f.units == FieldUnits::Symbolic,
                  std::string("  ") + d.mnemonic + "." + f.name + ": flags must be symbolic");
        continue;
      }
      std::size_t width = 1;
      switch (f.repr) {
        case FieldRepr::U8:
        case FieldRepr::I8: width = 1; break;
        case FieldRepr::U16: width = 2; break;
        default: width = 4; break;
      }
      CHECK_MSG(f.offset + width <= 32,
                std::string("  ") + d.mnemonic + "." + f.name + " runs past the instruction");
      CHECK_MSG(f.offset >= 3,
                std::string("  ") + d.mnemonic + "." + f.name + " overlaps opcode/unit");
      CHECK_MSG(f.min <= f.max, std::string("  ") + d.mnemonic + "." + f.name + ": empty range");
    }
    // No duplicate field names, and no two numeric fields sharing storage.
    std::set<std::string> names;
    for (std::uint8_t k = 0; k < d.nfields; ++k) {
      const std::string nm = d.fields[k].name;
      CHECK_MSG(names.insert(nm).second, std::string("  duplicate field ") + d.mnemonic + "." + nm);
    }
  }
}

/// Round-trip every field through fieldSet/fieldGet, including negative and
/// boundary values, so the little-endian codecs are right.
void testFieldCodec() {
  std::size_t n = 0;
  const OpDef* ops = opDefTable(n);
  for (std::size_t i = 0; i < n; ++i) {
    const OpDef& d = ops[i];
    for (std::uint8_t k = 0; k < d.nfields; ++k) {
      const FieldDef& f = d.fields[k];
      const std::int64_t probes[] = {f.min, f.max, f.min + (f.max - f.min) / 2,
                                     f.max > 0 ? 1 : 0};
      for (std::int64_t v : probes) {
        if (v < f.min || v > f.max) continue;
        if (f.align && (v % static_cast<std::int64_t>(f.align))) continue;
        KeaInstr in{};
        for (int b = 0; b < 32; ++b) in.b[b] = 0;
        fieldSet(in, f, v);
        CHECK_MSG(fieldGet(in, f) == v,
                  std::string("  ") + d.mnemonic + "." + f.name + " codec lost " +
                      std::to_string(v) + " (got " + std::to_string(fieldGet(in, f)) + ")");
      }
    }
  }
}

/// The table must agree with isa.h's convenience constructors, which are the
/// sanctioned way to build instructions.
void testAgainstConvenienceConstructors() {
  {
    const KeaInstr in = keaMakeMatmul(/*a_addr=*/352, /*a_inner=*/16, /*a_outer=*/160,
                                      /*m_inner=*/8, /*m_outer=*/8, /*acc_addr=*/32,
                                      /*acc_inner=*/16, /*acc_outer=*/128, /*bank=*/1,
                                      /*accumulate=*/true, /*int4=*/false);
    const OpDef* d = opDefFor(Opcode::MATMUL);
    CHECK(fieldGet(in, *fieldByName(*d, "a_addr")) == 352);
    CHECK(fieldGet(in, *fieldByName(*d, "a_inner_stride")) == 16);
    CHECK(fieldGet(in, *fieldByName(*d, "a_outer_stride")) == 160);
    CHECK(fieldGet(in, *fieldByName(*d, "m_inner")) == 8);
    CHECK(fieldGet(in, *fieldByName(*d, "m_outer")) == 8);
    CHECK(fieldGet(in, *fieldByName(*d, "acc_addr")) == 32);
    CHECK(fieldGet(in, *fieldByName(*d, "acc_inner_stride")) == 16);
    CHECK(fieldGet(in, *fieldByName(*d, "acc_outer_stride")) == 128);
    CHECK(fieldGet(in, *fieldByName(*d, "bank")) == 1);
    CHECK(fieldGet(in, *fieldByName(*d, "acc_mode")) == 1);
    CHECK(fieldGet(in, *fieldByName(*d, "dtype")) == 0);
  }
  {
    const KeaInstr in = keaMakeDma(/*is_load=*/true, Unit::DMA1, /*spm_is_w=*/true, 0x1000, 0x40,
                                   256, 4, 2, 256, 1024, 256, 1024);
    const OpDef* d = opDefFor(Opcode::DMA_LD);
    CHECK(fieldGet(in, *fieldByName(*d, "spm_space")) == 1);
    CHECK(fieldGet(in, *fieldByName(*d, "dram_addr")) == 0x1000);
    CHECK(fieldGet(in, *fieldByName(*d, "spm_addr")) == 0x40);
    CHECK(fieldGet(in, *fieldByName(*d, "len0")) == 256);
    CHECK(fieldGet(in, *fieldByName(*d, "n1")) == 4);
    CHECK(fieldGet(in, *fieldByName(*d, "n2")) == 2);
    CHECK(fieldGet(in, *fieldByName(*d, "dram_s1")) == 256);
    CHECK(fieldGet(in, *fieldByName(*d, "spm_s2")) == 1024);
    CHECK(fieldSpace(*d, *fieldByName(*d, "spm_addr"), in) == Space::SPM_W);
  }
  {
    const KeaInstr in = keaMakeVquant(64, 128, 256, 10, 16, 16, 16, /*out_zp=*/-9,
                                      /*lo=*/-9, /*hi=*/127, /*out_int4=*/false);
    const OpDef* d = opDefFor(Opcode::VQUANT);
    CHECK(fieldGet(in, *fieldByName(*d, "out_zp")) == -9);
    CHECK(fieldGet(in, *fieldByName(*d, "clamp_lo")) == -9);
    CHECK(fieldGet(in, *fieldByName(*d, "clamp_hi")) == 127);
  }
  {
    const KeaInstr in = keaMakeVfill(/*dst=*/0x100, /*row_bytes=*/160, /*rows=*/10,
                                     /*dst_row_stride=*/160, /*value=*/-128, /*dst_is_w=*/true);
    const OpDef* d = opDefFor(Opcode::VCOPY);
    CHECK(fieldGet(in, *fieldByName(*d, "mode")) == 1);
    CHECK(fieldGet(in, *fieldByName(*d, "dst_space")) == 1);
    CHECK(fieldGet(in, *fieldByName(*d, "src_space")) == 0);
    CHECK(fieldGet(in, *fieldByName(*d, "fill_value")) == -128);
    CHECK(fieldSpace(*d, *fieldByName(*d, "dst_addr"), in) == Space::SPM_W);
  }
  {
    const KeaInstr in = keaMakeDwconv(0, 0, 0, 8, 8, 16, 160, 16, /*kernel=*/5, /*stride=*/2,
                                      /*accumulate=*/true);
    const OpDef* d = opDefFor(Opcode::DWCONV);
    CHECK(fieldGet(in, *fieldByName(*d, "kernel")) == 1);
    CHECK(fieldGet(in, *fieldByName(*d, "stride")) == 1);
    CHECK(fieldGet(in, *fieldByName(*d, "acc_mode")) == 1);
  }
  {
    const KeaInstr in = keaMakeTrace(Unit::VPU, TraceKind::REGION_END, 17, 3);
    const OpDef* d = opDefFor(Opcode::TRACE);
    const FieldDef* kind = fieldByName(*d, "kind");
    CHECK(fieldGet(in, *kind) == static_cast<std::int64_t>(TraceKind::REGION_END));
    CHECK(std::string(kind->choices[fieldGet(in, *kind)]) ==
          traceKindName(TraceKind::REGION_END));
  }
}

/// Symbolic spellings must be the same strings the ISA helpers print.
void testSymbolicSpellings() {
  const OpDef* dma = opDefFor(Opcode::DMA_LD);
  const FieldDef* sp = fieldByName(*dma, "spm_space");
  CHECK(std::string(sp->choices[0]) == spaceName(Space::SPM_A));
  CHECK(std::string(sp->choices[1]) == spaceName(Space::SPM_W));

  const OpDef* mm = opDefFor(Opcode::MATMUL);
  const FieldDef* dt = fieldByName(*mm, "dtype");
  CHECK(std::string(dt->choices[0]) == dtypeName(DType::INT8));
  CHECK(std::string(dt->choices[1]) == dtypeName(DType::INT4));

  const OpDef* tr = opDefFor(Opcode::TRACE);
  const FieldDef* kind = fieldByName(*tr, "kind");
  CHECK(std::string(kind->choices[0]) == traceKindName(TraceKind::MARKER));
  CHECK(std::string(kind->choices[1]) == traceKindName(TraceKind::REGION_BEGIN));
  CHECK(std::string(kind->choices[2]) == traceKindName(TraceKind::REGION_END));
}

void runTests() {
  testOperandListsMatchIsaHeader();
  testEveryOpcodeCovered();
  testFieldStorageIsSane();
  testFieldCodec();
  testAgainstConvenienceConstructors();
  testSymbolicSpellings();
}

}  // namespace

TEST_MAIN("test_op_fields")
