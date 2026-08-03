// SPDX-License-Identifier: Apache-2.0
//
// Assembler diagnostics.
//
// The compiler backend is going to be debugged through these messages, so each
// one is a feature with a test. Every case below asserts a *specific* message,
// not merely that assembly failed.

#include <string>
#include <vector>

#include "kea/rt/assembler.h"
#include "kea/rt/model_map.h"
#include "test_util.h"

using namespace kea;
using namespace kea::rt;

namespace {

const char* kKasmDir = KEA_KASM_DIR;

ModelMap& theMap() {
  static ModelMap map;
  static bool loaded = false;
  if (!loaded) {
    Diagnostics d;
    loaded = loadModelMapFile(std::string(kKasmDir) + "/all_opcodes.map.json", map, d);
    CHECK_MSG(loaded, "  " + d.render());
  }
  return map;
}

/// Wrap a fragment in a minimal but complete program.
std::string prog(const std::string& body) {
  return std::string(".arch \"KEA-1\"\n.isa_revision 1\n") + body + "\n  CTRL  HALT    exit_code=0\n";
}

struct Result {
  bool ok = false;
  Diagnostics diags;
};

Result run(const std::string& source, bool with_map = true, bool rule_d = true) {
  Result r;
  AsmOptions opts;
  opts.map = with_map ? &theMap() : nullptr;
  opts.rule_d_is_error = rule_d;
  KeaProgram p;
  r.ok = assemble(source, "t.kasm", opts, p, r.diags);
  return r;
}

/// Assert that `source` fails with a diagnostic containing `needle`.
void expectError(const std::string& source, const std::string& needle, bool with_map = true) {
  Result r = run(source, with_map);
  if (r.ok) {
    keatest::report(__FILE__, __LINE__, "expected an error: " + needle,
                    "  source assembled cleanly:\n" + source);
    return;
  }
  if (!r.diags.contains(needle))
    keatest::report(__FILE__, __LINE__, "wrong diagnostic",
                    "  wanted a message containing: " + needle + "\n  got:\n" + r.diags.render());
}

void expectOk(const std::string& source, bool with_map = true) {
  Result r = run(source, with_map);
  CHECK_MSG(r.ok, "  expected clean assembly, got:\n" + r.diags.render());
}

// ---------------------------------------------------------------------------

const char* kGoodMatmul =
    "  MXU   MATMUL  a_addr=a:0, a_inner_stride=16, a_outer_stride=160, m_inner=8, m_outer=8, "
    "acc_addr=acc:0, acc_inner_stride=16w, acc_outer_stride=128w, bank=0, acc_mode=overwrite, "
    "dtype=int8";

const char* kGoodLoadW =
    "  MXU   LOAD_W  w_addr=w:0, w_row_stride=16, k_rows=16, n_cols=16, bank=0, dtype=int8";

const char* kGoodDma =
    "  DMA0  DMA_LD  spm_space=SPM_A, dram_addr=@input, spm_addr=a:0, len0=16, n1=8, n2=1, "
    "dram_s1=16, dram_s2=128, spm_s1=16, spm_s2=128";

std::string sub(std::string s, const std::string& from, const std::string& to) {
  const std::size_t at = s.find(from);
  CHECK_MSG(at != std::string::npos, "  substitution target '" + from + "' not found");
  if (at == std::string::npos) return s;
  return s.replace(at, from.size(), to);
}

void testSanity() {
  expectOk(prog(kGoodMatmul));
  expectOk(prog(kGoodLoadW));
  expectOk(prog(kGoodDma));
}

void testBadMnemonic() {
  expectError(prog("  MXU   MATMULL  a_addr=a:0"), "unknown mnemonic 'MATMULL'");
  expectError(prog("  MXU   MATMULL  a_addr=a:0"), "known mnemonics: NOP, HALT, SIGNAL");
  // Mnemonics are upper case, deliberately: it keeps the unit column and the
  // opcode column visually distinct from the lower-case field names.
  expectError(prog("  MXU   matmul  a_addr=a:0"), "unknown mnemonic 'matmul'");
}

void testBadUnit() {
  expectError(prog(sub(kGoodMatmul, "  MXU   MATMUL", "  VPU   MATMUL")),
              "MATMUL must be issued on MXU, not VPU");
  expectError(prog(sub(kGoodLoadW, "  MXU   LOAD_W", "  DWU   LOAD_W")),
              "LOAD_W must be issued on MXU, not DWU");
  expectError(prog(sub(kGoodDma, "  DMA0  DMA_LD", "  VPU   DMA_LD")),
              "DMA_LD must be issued on DMA0 or DMA1, not VPU");
  expectError(prog("  MXU   HALT    exit_code=0"), "HALT must be issued on CTRL, not MXU");
  expectError(prog("  MXU2  NOP     cycles=0"), "'MXU2' is not a unit name");
  // Queue-agnostic opcodes may go anywhere, including CTRL.
  expectOk(prog("  DMA1  NOP     cycles=3"));
  expectOk(prog("  CTRL  NOP     cycles=3"));
  expectOk(prog("  DWU   TRACE   kind=marker, tag=1, payload=0"));
}

void testFieldRanges() {
  expectError(prog(sub(kGoodLoadW, "k_rows=16", "k_rows=17")), "'k_rows' must be in 1..16, got 17");
  expectError(prog(sub(kGoodLoadW, "k_rows=16", "k_rows=0")), "'k_rows' must be in 1..16, got 0");
  expectError(prog(sub(kGoodDma, "n2=1", "n2=256")), "'n2' must be in 1..255, got 256");
  expectError(prog(sub(kGoodDma, "len0=16", "len0=0")), "'len0' must be in 1..65535, got 0");
  expectError(prog("  MXU   SIGNAL  event=32, inc=1"), "'event' must be in 0..31, got 32");
  expectError(prog("  MXU   WAIT    event=0, threshold=1"), "Rule D violation");
  expectError(prog(sub(std::string(kGoodMatmul), "m_outer=8", "m_outer=512")),
              "m_inner * m_outer = 4096 exceeds KEA_MXU_MAX_ROWS (2048)");
}

void testAddressSpaces() {
  // ACC is word-addressed. This is the mistake the syntax exists to prevent.
  expectError(prog(sub(kGoodMatmul, "acc_addr=acc:0", "acc_addr=acc:8")),
              "'acc_addr' must be a multiple of 16 int32 words");
  expectError(prog(sub(kGoodMatmul, "acc_addr=acc:0", "acc_addr=acc:8")),
              "byte offset 32");
  expectError(prog(sub(kGoodMatmul, "acc_inner_stride=16w", "acc_inner_stride=24w")),
              "'acc_inner_stride' must be a multiple of 16");
  expectError(prog(sub(kGoodMatmul, "acc_inner_stride=16w", "acc_inner_stride=16")),
              "is measured in ACC int32 words and must carry the 'w' suffix");
  expectError(prog(sub(kGoodMatmul, "a_inner_stride=16", "a_inner_stride=16w")),
              "is measured in bytes; the 'w' (int32 ACC words) suffix is not allowed here");
  expectError(prog(sub(kGoodMatmul, "a_addr=a:0", "a_addr=w:0")),
              "'a_addr' on MATMUL addresses SPM_A, but 'w:' was written");
  expectError(prog(sub(kGoodMatmul, "acc_addr=acc:0", "acc_addr=a:0")),
              "'acc_addr' on MATMUL addresses ACC, but 'a:' was written");
  expectError(prog(sub(kGoodMatmul, "a_addr=a:0", "a_addr=a:999999")),
              "is outside SPM_A, which holds 262144 bytes");
  expectError(prog(sub(kGoodMatmul, "acc_addr=acc:0", "acc_addr=acc:40000")),
              "is outside ACC, which holds 32768 int32 words");
  expectError(prog(sub(kGoodMatmul, "a_addr=a:0", "a_addr=0")),
              "expected a scratchpad address for 'a_addr'");
  expectError(prog(sub(kGoodMatmul, "a_addr=a:0", "a_addr=spm:0")),
              "unknown address space prefix 'spm:'");
  // The DMA space flag and the address prefix are redundant on purpose.
  expectError(prog(sub(kGoodDma, "spm_space=SPM_A", "spm_space=SPM_W")),
              "'spm_addr' is written as an SPM_A address but 'spm_space' says SPM_W");
}

void testDramSymbols() {
  expectError(prog(sub(kGoodDma, "dram_addr=@input", "dram_addr=@conv9.weights")),
              "unknown DRAM symbol '@conv9.weights'");
  expectError(prog(sub(kGoodDma, "dram_addr=@input", "dram_addr=@conv9.weights")),
              "declare it in the 'symbols' or 'tensors' array of model.map.json");
  expectError(prog(kGoodDma), "cannot resolve '@input': no model map was supplied",
              /*with_map=*/false);
  expectError(prog(sub(kGoodDma, "dram_addr=@input", "dram_addr=@conv1.weights-64")),
              "resolves to a negative DRAM address");
  expectError(prog(sub(kGoodDma, "dram_addr=@input", "dram_addr=dram:2000000")),
              "is outside the 1048576-byte arena");
  expectError(prog(sub(kGoodDma, "dram_addr=@input", "dram_addr=0")),
              "expected a DRAM address for 'dram_addr'");
  // Offsets and the dram: escape are both fine.
  expectOk(prog(sub(kGoodDma, "dram_addr=@input", "dram_addr=@input+1024")));
  expectOk(prog(sub(kGoodDma, "dram_addr=@input", "dram_addr=dram:65536")));
}

void testFieldSetErrors() {
  expectError(prog(sub(kGoodMatmul, "acc_addr=acc:0", "acc_address=acc:0")),
              "unknown field 'acc_address' for MATMUL");
  expectError(prog(sub(kGoodMatmul, "acc_addr=acc:0", "acc_address=acc:0")),
              "MATMUL fields are: a_addr, a_inner_stride");
  expectError(prog(sub(kGoodMatmul, ", dtype=int8", "")),
              "MATMUL is missing required field(s): dtype");
  expectError(prog(sub(kGoodMatmul, "m_inner=8", "a_addr=a:0")), "duplicate field 'a_addr'");
  expectError(prog(sub(kGoodMatmul, "dtype=int8", "dtype=int16")),
              "'int16' is not a valid value for 'dtype' on MATMUL");
  expectError(prog(sub(kGoodMatmul, "dtype=int8", "dtype=int16")), "must be one of {int8, int4}");
  expectError(prog(sub(kGoodMatmul, "acc_mode=overwrite", "acc_mode=acc")),
              "must be one of {overwrite, accumulate}");
  expectError(prog(sub(kGoodMatmul, "a_addr=a:0", "a_addr")), "expected '=' after 'a_addr'");
  expectError(prog(sub(kGoodLoadW, "k_rows=16", "k_rows=16kb")),
              "the only recognised suffixes are 'b' (bytes) and 'w' (int32 ACC words)");
}

void testOpcodeSpecificRules() {
  // LOAD_W alignment depends on the datatype.
  expectError(prog(sub(kGoodLoadW, "w_addr=w:0", "w_addr=w:8")),
              "'w_addr' must be a multiple of 16 bytes for dtype=int8, got 8");
  expectOk(prog(sub(sub(kGoodLoadW, "w_addr=w:0", "w_addr=w:8"), "dtype=int8", "dtype=int4")));
  expectError(prog(sub(sub(kGoodLoadW, "w_row_stride=16", "w_row_stride=4"), "dtype=int8",
                       "dtype=int4")),
              "'w_row_stride' must be a multiple of 8 bytes for dtype=int4, got 4");

  // A zero destination stride collapses every run onto one address.
  expectError(prog(sub(kGoodDma, "spm_s1=16", "spm_s1=0")),
              "'spm_s1' is 0 with n1=8, so every run would be written to the same SPM address");
  // The same descriptor is legal in the other direction, where the SPM side is
  // the source and a zero DRAM stride is what would be undefined.
  expectError(prog(sub(sub(kGoodDma, "DMA_LD", "DMA_ST"), "dram_s1=16", "dram_s1=0")),
              "'dram_s1' is 0 with n1=8, so every run would be written to the same DRAM address");
  // Broadcasting on a load is legal: zero stride on the *source* side.
  expectOk(prog(sub(sub(kGoodDma, "dram_s1=16", "dram_s1=0"), "n2=1", "n2=1")));

  // DWCONV channel granularity and ACC capacity.
  const std::string dw =
      "  DWU   DWCONV  a_addr=a:0, w_addr=w:0, acc_addr=acc:0, out_h=8, out_w=8, channels=16, "
      "a_row_stride=160, a_pix_stride=16, kernel=3, stride=1, acc_mode=overwrite";
  expectOk(prog(dw));
  expectError(prog(sub(dw, "channels=16", "channels=24")),
              "'channels' must be a multiple of 16, got 24");
  expectError(prog(sub(sub(dw, "out_h=8", "out_h=256"), "out_w=8", "out_w=256")),
              "runs past the end of ACC");

  // int4 VQUANT output has to clamp inside the 4-bit range.
  const std::string vq =
      "  VPU   VQUANT  acc_addr=acc:0, out_addr=a:0, qparam_addr=w:0, num_pixels=4, channels=16, "
      "acc_pix_stride=16w, out_pix_stride=16, out_zp=0, clamp_lo=-128, clamp_hi=127, dtype=int8";
  expectOk(prog(vq));
  expectError(prog(sub(vq, "dtype=int8", "dtype=int4")),
              "with dtype=int4 the clamps must lie inside [-8, 7]");
  expectError(prog(sub(sub(vq, "clamp_lo=-128", "clamp_lo=100"), "clamp_hi=127", "clamp_hi=10")),
              "clamp_lo (100) is greater than clamp_hi (10)");
  expectError(prog(sub(vq, "qparam_addr=w:0", "qparam_addr=w:2")),
              "'qparam_addr' must be a multiple of 4 bytes");
}

void testHaltDiscipline() {
  expectError(".arch \"KEA-1\"\n  MXU   NOP     cycles=0\n", "does not end with HALT");
  expectError(".arch \"KEA-1\"\n  MXU   NOP     cycles=0\n", "append '  CTRL  HALT");
  expectError(".arch \"KEA-1\"\n  CTRL  HALT    exit_code=0\n  MXU   NOP     cycles=0\n",
              "HALT must be the last instruction; 1 instruction(s) follow it");
  expectError(".arch \"KEA-1\"\n", "the program contains no instructions");
}

void testRuleD() {
  const std::string bad =
      "  VPU   WAIT    event=3, threshold=1\n"
      "  MXU   SIGNAL  event=3, inc=1";
  expectError(prog(bad), "Rule D violation: WAIT event=3, threshold=1 at pc 0 is only supplied "
                         "with 0 count(s)");
  expectError(prog(bad), "dispatch is in-order");

  // Reversed, it is the canonical prefetch pattern and must assemble.
  expectOk(prog("  MXU   SIGNAL  event=3, inc=1\n  VPU   WAIT    event=3, threshold=1"));

  // A counting acquire needs the full threshold to have been supplied.
  expectError(prog("  MXU   SIGNAL  event=3, inc=1\n  VPU   WAIT    event=3, threshold=2"),
              "is only supplied with 1 count(s)");
  expectOk(prog("  MXU   SIGNAL  event=3, inc=2\n  VPU   WAIT    event=3, threshold=2"));

  // Two waiters split the counts, so the second one is short.
  expectError(prog("  MXU   SIGNAL  event=3, inc=1\n"
                   "  VPU   WAIT    event=3, threshold=1\n"
                   "  DWU   WAIT    event=3, threshold=1"),
              "at pc 2 is only supplied with 0 count(s)");

  // ... and the check can be downgraded, because the simulator needs a
  // deliberately deadlocking program to test its deadlock detector.
  {
    Result r = run(prog(bad), /*with_map=*/true, /*rule_d=*/false);
    CHECK_MSG(r.ok, "  --no-rule-d should still assemble:\n" + r.diags.render());
    CHECK(r.diags.contains("Rule D violation"));
  }
}

void testDirectives() {
  expectError(".arch \"KEA-2\"\n  CTRL  HALT    exit_code=0\n",
              "this assembler implements KEA-1, not 'KEA-2'");
  expectError(".isa_revision 2\n  CTRL  HALT    exit_code=0\n",
              "source targets ISA revision 2, this assembler implements 1");
  expectError(".fancy 1\n  CTRL  HALT    exit_code=0\n", "unknown directive '.fancy'");
  expectError(".fancy 1\n  CTRL  HALT    exit_code=0\n",
              "known directives: .arch, .isa_revision, .entry, .event, .region");
  expectError(".entry nowhere\n  CTRL  HALT    exit_code=0\n", "unknown label 'nowhere' in .entry");
  expectError(".entry 9\n  CTRL  HALT    exit_code=0\n",
              "entry point 9 is outside the 1-instruction program");
  expectError(".event 40, \"x\"\n  CTRL  HALT    exit_code=0\n", "event id must be in 0..31");
  expectError("start:\nstart:\n  CTRL  HALT    exit_code=0\n", "duplicate label 'start'");
  expectOk(".entry start\nstart:\n  CTRL  HALT    exit_code=0\n");
}

void testConstBlob() {
  Diagnostics diags;
  ModelMap map;
  CHECK(loadModelMapFile(std::string(kKasmDir) + "/double_buffered.map.json", map, diags));

  const std::string src = prog("  MXU   NOP     cycles=0");
  {
    AsmOptions opts;
    opts.map = &map;
    KeaProgram p;
    Diagnostics d;
    CHECK(!assemble(src, "t.kasm", opts, p, d));
    CHECK_MSG(d.contains("the map declares dram.const_bytes = 704 but no constant blob was supplied"),
              "  got:\n" + d.render());
    CHECK(d.contains("pass --const model.weights.bin"));
  }
  {
    std::vector<std::uint8_t> wrong(100, 0);
    AsmOptions opts;
    opts.map = &map;
    opts.const_data = &wrong;
    KeaProgram p;
    Diagnostics d;
    CHECK(!assemble(src, "t.kasm", opts, p, d));
    CHECK_MSG(d.contains("the constant blob is 100 bytes but the map declares dram.const_bytes = 704"),
              "  got:\n" + d.render());
  }
}

/// Diagnostics must carry a usable location and render a caret under the token.
void testDiagnosticRendering() {
  const std::string src =
      ".arch \"KEA-1\"\n"
      "  MXU   MATMUL  a_addr=a:0, a_inner_stride=16, a_outer_stride=160, m_inner=8, m_outer=8, "
      "acc_addr=acc:8, acc_inner_stride=16w, acc_outer_stride=128w, bank=0, acc_mode=overwrite, "
      "dtype=int8\n"
      "  CTRL  HALT    exit_code=0\n";
  Result r = run(src);
  CHECK(!r.ok);
  CHECK(r.diags.all().size() >= 1);
  const Diagnostic& d = r.diags.all()[0];
  CHECK(d.file == "t.kasm");
  CHECK_EQ(d.line, 2);
  CHECK(d.is_error);
  CHECK(!d.notes.empty());

  const std::string rendered = r.diags.render();
  CHECK_MSG(rendered.find("t.kasm:2:") == 0, "  got:\n" + rendered);
  CHECK_MSG(rendered.find("^") != std::string::npos, "  no caret in:\n" + rendered);
  CHECK_MSG(rendered.find("note: ") != std::string::npos, "  no note in:\n" + rendered);

  // The caret must sit under the offending value, not at the start of the line.
  const std::size_t caret_line_start = rendered.rfind('\n', rendered.find('^')) + 1;
  const std::size_t caret_col = rendered.find('^') - caret_line_start;
  CHECK_MSG(caret_col + 1 == static_cast<std::size_t>(d.col) + 2,
            "  caret column " + std::to_string(caret_col) + " does not match d.col " +
                std::to_string(d.col));
}

void testLexerErrors() {
  expectError(prog("  MXU   NOP     cycles=$"), "unexpected character '$'");
  expectError(prog("  MXU   NOP     cycles=0x"), "expected at least one hexadecimal digit");
  expectError(prog("  DMA0  DMA_LD  dram_addr=@"), "expected a DRAM symbol name after '@'");
  expectError(".arch \"KEA-1\n  CTRL  HALT    exit_code=0\n", "unterminated string literal");
}

void runTests() {
  testSanity();
  testBadMnemonic();
  testBadUnit();
  testFieldRanges();
  testAddressSpaces();
  testDramSymbols();
  testFieldSetErrors();
  testOpcodeSpecificRules();
  testHaltDiscipline();
  testRuleD();
  testDirectives();
  testConstBlob();
  testDiagnosticRendering();
  testLexerErrors();
}

}  // namespace

TEST_MAIN("test_asm_errors")
