// SPDX-License-Identifier: Apache-2.0
//
// Golden binary encodings.
//
// A round-trip test is self-consistent: if the assembler and the disassembler
// agree on a wrong bit position, `disassemble(assemble(x)) == x` still passes.
// This test closes that hole by comparing assembled instructions against byte
// arrays computed by hand from the field tables in docs/ISA.md §5-§10, one
// nibble at a time.
//
// If one of these fails, do NOT "fix" the expectation without re-deriving it
// from ISA.md --- the whole point is that it was written independently of the
// code under test.

#include <cstring>
#include <string>
#include <vector>

#include "kea/rt/assembler.h"
#include "kea/rt/model_map.h"
#include "test_util.h"

using namespace kea;
using namespace kea::rt;

namespace {

const char* kKasmDir = KEA_KASM_DIR;

// The program whose encoding is pinned below. Field values are deliberately
// asymmetric (distinct values in every slot) so a swapped pair of fields
// cannot go unnoticed.
const char* kSource = R"(.arch "KEA-1"
.isa_revision 1
.entry 0

  MXU   LOAD_W  w_addr=w:256, w_row_stride=8, k_rows=9, n_cols=12, bank=1, dtype=int4
  MXU   MATMUL  a_addr=a:352, a_inner_stride=8, a_outer_stride=80, m_inner=8, m_outer=8, acc_addr=acc:1024, acc_inner_stride=16w, acc_outer_stride=128w, bank=1, acc_mode=accumulate, dtype=int4
  DWU   DWCONV  a_addr=a:2048, w_addr=w:4096, acc_addr=acc:4096, out_h=4, out_w=4, channels=32, a_row_stride=320, a_pix_stride=32, kernel=5, stride=2, acc_mode=accumulate
  VPU   VQUANT  acc_addr=acc:1024, out_addr=a:9216, qparam_addr=w:8196, num_pixels=64, channels=32, acc_pix_stride=32w, out_pix_stride=16, out_zp=3, clamp_lo=-8, clamp_hi=7, dtype=int4
  VPU   VADD    a_addr=a:8192, b_addr=a:9216, out_addr=a:10240, param_addr=w:12288, num_elems=1024, clamp_lo=-128, clamp_hi=127
  VPU   VPOOL   mode=avg, in_addr=a:8192, out_addr=a:16448, out_h=1, out_w=1, channels=16, kh=7, kw=7, stride_h=1, stride_w=1, in_row_stride=112, out_row_stride=16
  VPU   VCOPY   mode=fill, src_space=SPM_W, dst_space=SPM_A, src_addr=w:0, dst_addr=a:20480, row_bytes=160, rows=10, src_row_stride=0, dst_row_stride=160, fill_value=-128
  DMA0  DMA_LD  spm_space=SPM_A, dram_addr=@input, spm_addr=a:176, len0=16, n1=8, n2=8, dram_s1=16, dram_s2=128, spm_s1=16, spm_s2=160
  MXU   TRACE   kind=begin, tag=17, payload=3
  DMA0  SIGNAL  event=0, inc=1
  CTRL  HALT    exit_code=0
)";

struct Golden {
  const char* what;
  std::uint8_t bytes[32];
};

// Derived by hand from ISA.md. Byte 0 = opcode, 1 = unit, 2 = flags,
// 3 = aux; everything after is little-endian at the documented offset.
const Golden kGolden[] = {
    // LOAD_W (0x20) on MXU (unit 0). flags = INT4(bit0) | BANK1(bit1) = 0x03.
    // w_addr=256 @4, w_row_stride=8 @8, k_rows=9 @12, n_cols=12 @13.
    {"LOAD_W int4 bank1",
     {0x20, 0x00, 0x03, 0x00,
      0x00, 0x01, 0x00, 0x00,
      0x08, 0x00, 0x00, 0x00,
      0x09, 0x0C, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00}},

    // MATMUL (0x21) on MXU. flags = ACCUMULATE(bit0)|INT4(bit1)|BANK1(bit2) = 0x07.
    // a_addr=352=0x160 @4, acc_addr=1024=0x400 @8, m_inner=8 @12, m_outer=8 @14,
    // a_inner_stride=8 @16, a_outer_stride=80=0x50 @20,
    // acc_inner_stride=16 @24, acc_outer_stride=128=0x80 @28.
    {"MATMUL int4 accumulate bank1",
     {0x21, 0x00, 0x07, 0x00,
      0x60, 0x01, 0x00, 0x00,
      0x00, 0x04, 0x00, 0x00,
      0x08, 0x00, 0x08, 0x00,
      0x08, 0x00, 0x00, 0x00,
      0x50, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00,
      0x80, 0x00, 0x00, 0x00}},

    // DWCONV (0x30) on DWU (unit 1). flags = KERNEL5|STRIDE2|ACCUMULATE = 0x07.
    // a_addr=2048=0x800 @4, w_addr=4096=0x1000 @8, acc_addr=4096 @12,
    // out_h=4 @16, out_w=4 @18, channels=32 @20, reserved @22,
    // a_row_stride=320=0x140 @24, a_pix_stride=32 @28.
    {"DWCONV 5x5 stride2 accumulate",
     {0x30, 0x01, 0x07, 0x00,
      0x00, 0x08, 0x00, 0x00,
      0x00, 0x10, 0x00, 0x00,
      0x00, 0x10, 0x00, 0x00,
      0x04, 0x00, 0x04, 0x00,
      0x20, 0x00, 0x00, 0x00,
      0x40, 0x01, 0x00, 0x00,
      0x20, 0x00, 0x00, 0x00}},

    // VQUANT (0x40) on VPU (unit 2). flags = OUT_INT4 = 0x01, aux = out_zp = 3.
    // acc_addr=1024 @4, out_addr=9216=0x2400 @8, qparam_addr=8196=0x2004 @12,
    // num_pixels=64 @16, channels=32 @20, clamp_lo=-8=0xF8 @22, clamp_hi=7 @23,
    // acc_pix_stride=32 @24, out_pix_stride=16 @28.
    {"VQUANT int4 out",
     {0x40, 0x02, 0x01, 0x03,
      0x00, 0x04, 0x00, 0x00,
      0x00, 0x24, 0x00, 0x00,
      0x04, 0x20, 0x00, 0x00,
      0x40, 0x00, 0x00, 0x00,
      0x20, 0x00, 0xF8, 0x07,
      0x20, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00}},

    // VADD (0x41) on VPU. flags/aux reserved = 0.
    // a=8192=0x2000 @4, b=9216=0x2400 @8, out=10240=0x2800 @12,
    // param=12288=0x3000 @16, num_elems=1024=0x400 @20,
    // clamp_lo=-128=0x80 @24, clamp_hi=127=0x7F @25.
    {"VADD",
     {0x41, 0x02, 0x00, 0x00,
      0x00, 0x20, 0x00, 0x00,
      0x00, 0x24, 0x00, 0x00,
      0x00, 0x28, 0x00, 0x00,
      0x00, 0x30, 0x00, 0x00,
      0x00, 0x04, 0x00, 0x00,
      0x80, 0x7F, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00}},

    // VPOOL (0x42) on VPU. flags = AVG = 0x01.
    // in=8192 @4, out=16448=0x4040 @8, out_h=1 @12, out_w=1 @14, channels=16 @16,
    // kh=7 @18, kw=7 @19, stride_h=1 @20, stride_w=1 @21, reserved @22,
    // in_row_stride=112=0x70 @24, out_row_stride=16 @28.
    {"VPOOL avg 7x7",
     {0x42, 0x02, 0x01, 0x00,
      0x00, 0x20, 0x00, 0x00,
      0x40, 0x40, 0x00, 0x00,
      0x01, 0x00, 0x01, 0x00,
      0x10, 0x00, 0x07, 0x07,
      0x01, 0x01, 0x00, 0x00,
      0x70, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00}},

    // VCOPY (0x43) on VPU. flags = FILL(bit0)|SRC_W(bit1) = 0x03 (dst stays
    // SPM_A), aux = fill_value = -128 = 0x80.
    // src=0 @4, dst=20480=0x5000 @8, row_bytes=160=0xA0 @12, rows=10 @16,
    // src_row_stride=0 @20, dst_row_stride=160 @24, reserved @28.
    {"VCOPY fill",
     {0x43, 0x02, 0x03, 0x80,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x50, 0x00, 0x00,
      0xA0, 0x00, 0x00, 0x00,
      0x0A, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0xA0, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00}},

    // DMA_LD (0x10) on DMA0 (unit 3). flags = 0 (SPM_A), aux = n2 = 8.
    // dram_addr = @input = 65536 = 0x00010000 @4, spm_addr=176=0xB0 @8,
    // len0=16 @12, n1=8 @14, dram_s1=16 @16, dram_s2=128=0x80 @20,
    // spm_s1=16 @24, spm_s2=160=0xA0 @28.
    {"DMA_LD strided",
     {0x10, 0x03, 0x00, 0x08,
      0x00, 0x00, 0x01, 0x00,
      0xB0, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x08, 0x00,
      0x10, 0x00, 0x00, 0x00,
      0x80, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00,
      0xA0, 0x00, 0x00, 0x00}},

    // TRACE (0x04) on MXU. flags = TraceKind::REGION_BEGIN = 1.
    // tag=17=0x11 @4, payload=3 @8.
    {"TRACE begin",
     {0x04, 0x00, 0x01, 0x00,
      0x11, 0x00, 0x00, 0x00,
      0x03, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00}},

    // SIGNAL (0x02) on DMA0. aux = event id = 0, value = 1 @4.
    {"SIGNAL",
     {0x02, 0x03, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00}},

    // HALT (0x01) on CTRL (unit 5). exit_code = 0 @4.
    {"HALT",
     {0x01, 0x05, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00}},
};

std::string hexdump(const std::uint8_t* b) {
  static const char* hex = "0123456789abcdef";
  std::string s;
  for (int i = 0; i < 32; ++i) {
    if (i && i % 4 == 0) s += ' ';
    s += hex[b[i] >> 4];
    s += hex[b[i] & 0xF];
  }
  return s;
}

void testGoldenEncodings() {
  Diagnostics diags;
  ModelMap map;
  const bool map_ok = loadModelMapFile(std::string(kKasmDir) + "/all_opcodes.map.json", map, diags);
  CHECK_MSG(map_ok, "  " + diags.render());

  AsmOptions opts;
  opts.map = &map;
  std::vector<std::uint8_t> consts(static_cast<std::size_t>(map.dram.const_bytes), 0);
  opts.const_data = &consts;

  KeaProgram p;
  const bool ok = assemble(kSource, "<golden>", opts, p, diags);
  CHECK_MSG(ok, "  " + diags.render());
  if (!ok) return;

  const std::size_t n = sizeof(kGolden) / sizeof(kGolden[0]);
  CHECK_MSG(p.code.size() == n, "  the golden table and the source have drifted apart");
  if (p.code.size() != n) return;

  for (std::size_t i = 0; i < n; ++i) {
    if (std::memcmp(p.code[i].b, kGolden[i].bytes, 32) != 0) {
      keatest::report(__FILE__, __LINE__, std::string("encoding of ") + kGolden[i].what,
                      "  got : " + hexdump(p.code[i].b) + "\n  want: " + hexdump(kGolden[i].bytes));
    }
  }
}

/// The assembler must produce exactly what isa.h's own constructors produce.
void testAgreesWithIsaConstructors() {
  const KeaInstr a = keaMakeMatmul(352, 8, 80, 8, 8, 1024, 16, 128, /*bank=*/1,
                                   /*accumulate=*/true, /*int4=*/true);
  CHECK(std::memcmp(a.b, kGolden[1].bytes, 32) == 0);

  const KeaInstr b = keaMakeDma(/*is_load=*/true, Unit::DMA0, /*spm_is_w=*/false, 65536, 176, 16, 8,
                                8, 16, 128, 16, 160);
  CHECK(std::memcmp(b.b, kGolden[7].bytes, 32) == 0);

  const KeaInstr c = keaMakeVfill(20480, 160, 10, 160, -128, /*dst_is_w=*/false);
  // keaMakeVfill leaves src_space = SPM_A; the golden line sets SPM_W, which
  // is ignored in fill mode, so compare everything except that flag bit.
  KeaInstr expect;
  std::memcpy(expect.b, kGolden[6].bytes, 32);
  expect.b[2] = static_cast<std::uint8_t>(expect.b[2] & ~KEA_VC_F_SRC_W);
  CHECK(std::memcmp(c.b, expect.b, 32) == 0);

  const KeaInstr d = keaMakeHalt(0);
  CHECK(std::memcmp(d.b, kGolden[10].bytes, 32) == 0);
}

/// Every golden instruction must survive the frozen validator.
void testGoldenAreValid() {
  for (const Golden& g : kGolden) {
    KeaInstr in;
    std::memcpy(in.b, g.bytes, 32);
    const char* e = keaValidate(in);
    CHECK_MSG(e == nullptr, std::string("  ") + g.what + ": " + (e ? e : ""));
  }
}

void runTests() {
  testGoldenEncodings();
  testAgreesWithIsaConstructors();
  testGoldenAreValid();
}

}  // namespace

TEST_MAIN("test_golden_encoding")
