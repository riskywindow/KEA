// SPDX-License-Identifier: Apache-2.0
//
// The ADR-0001 round trip: disassemble(assemble(x)) == x, character for
// character, over a corpus of hand-written `.kasm` covering all 12 opcodes and
// every field.
//
// The corpus lives in runtime/tests/kasm/ and is deliberately checked in as
// human-written text: it is also the vehicle the simulator is tested with
// before the compiler exists.

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "kea/rt/assembler.h"
#include "kea/rt/disassembler.h"
#include "kea/rt/model_map.h"
#include "kea/rt/op_fields.h"
#include "test_program_eq.h"
#include "test_util.h"

using namespace kea;
using namespace kea::rt;

namespace {

const char* kKasmDir = KEA_KASM_DIR;

std::string readFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  CHECK_MSG(static_cast<bool>(in), "  cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool fileExists(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return static_cast<bool>(in);
}

struct Loaded {
  ModelMap map;
  bool have_map = false;
  std::vector<std::uint8_t> const_data;
};

Loaded loadSidecar(const std::string& stem, Diagnostics& diags) {
  Loaded l;
  const std::string map_path = std::string(kKasmDir) + "/" + stem + ".map.json";
  if (fileExists(map_path)) {
    l.have_map = loadModelMapFile(map_path, l.map, diags);
    CHECK_MSG(l.have_map, "  map failed to load:\n" + diags.render());
    l.const_data.assign(static_cast<std::size_t>(l.map.dram.const_bytes), 0);
  }
  return l;
}

/// Assemble `text`; on failure print the diagnostics, which is what the
/// compiler backend author will see.
bool assembleOrReport(const std::string& text, const std::string& name, const AsmOptions& opts,
                      KeaProgram& program) {
  Diagnostics diags;
  const bool ok = assemble(text, name, opts, program, diags);
  CHECK_MSG(ok, "  " + name + " failed to assemble:\n" + diags.render());
  return ok;
}

void testCanonicalRoundTrip(const std::string& stem) {
  const std::string path = std::string(kKasmDir) + "/" + stem + ".kasm";
  const std::string text = readFile(path);

  Diagnostics diags;
  Loaded side = loadSidecar(stem, diags);

  AsmOptions opts;
  opts.map = side.have_map ? &side.map : nullptr;
  opts.const_data = side.have_map ? &side.const_data : nullptr;

  KeaProgram program;
  if (!assembleOrReport(text, path, opts, program)) return;

  DisasmOptions dopts;
  dopts.map = side.have_map ? &side.map : nullptr;
  const std::string printed = disassemble(program, dopts);

  CHECK_MSG(printed == text, "  " + stem + ".kasm is not a fixed point of assemble/disassemble\n" +
                                 keatest::diffText(printed, text));

  // ... and re-assembling the printed form must reproduce the same program.
  KeaProgram again;
  if (!assembleOrReport(printed, path + "(printed)", opts, again)) return;
  std::string why;
  CHECK_MSG(keatest::programsEqual(program, again, why), "  " + stem + ":\n" + why);
}

/// A non-canonical source (comments, labels, hex, `b` suffixes, continuation
/// lines) must normalize to a fixed point after one pass.
void testMessyNormalizes() {
  const std::string path = std::string(kKasmDir) + "/messy.kasm";
  const std::string text = readFile(path);

  AsmOptions opts;
  KeaProgram p1;
  if (!assembleOrReport(text, path, opts, p1)) return;

  DisasmOptions dopts;
  const std::string c1 = disassemble(p1, dopts);

  KeaProgram p2;
  if (!assembleOrReport(c1, path + "(canonical)", opts, p2)) return;
  const std::string c2 = disassemble(p2, dopts);
  CHECK_STR_EQ(c2, c1);

  std::string why;
  CHECK_MSG(keatest::programsEqual(p1, p2, why), "  messy.kasm:\n" + why);

  // `.entry setup` must have resolved to the label's pc, which is 0.
  CHECK(p1.entry_pc == 0);
  // Labels survive into the canonical listing.
  CHECK(c1.find("setup:\n") != std::string::npos);
  CHECK(c1.find("loop_body:\n") != std::string::npos);
  CHECK(c1.find("epilogue:\n") != std::string::npos);
  // Hex and `b` suffixes are normalized to plain decimal.
  CHECK(c1.find("0x") == std::string::npos);
  CHECK(c1.find("a_inner_stride=16,") != std::string::npos);
  CHECK(c1.find("dram_addr=dram:4096,") != std::string::npos);
  // Comments do not survive; that is what "canonical" means.
  CHECK(c1.find(';') == std::string::npos);
}

/// Every opcode and every field name must actually appear in the corpus,
/// otherwise the round trip is proving less than it claims.
void testCorpusCoversEverything() {
  const std::string all = readFile(std::string(kKasmDir) + "/all_opcodes.kasm");
  std::size_t n = 0;
  const OpDef* ops = opDefTable(n);
  for (std::size_t i = 0; i < n; ++i) {
    const OpDef& d = ops[i];
    CHECK_MSG(all.find(std::string(d.mnemonic) + "  ") != std::string::npos ||
                  all.find(std::string(d.mnemonic) + " ") != std::string::npos,
              std::string("  corpus never uses ") + d.mnemonic);
    for (std::uint8_t k = 0; k < d.nfields; ++k) {
      const std::string needle = std::string(d.fields[k].name) + "=";
      CHECK_MSG(all.find(needle) != std::string::npos,
                std::string("  corpus never sets ") + d.mnemonic + "." + d.fields[k].name);
    }
    // Both spellings of every two-valued symbolic field must appear.
    for (std::uint8_t k = 0; k < d.nfields; ++k) {
      const FieldDef& f = d.fields[k];
      if (f.units != FieldUnits::Symbolic) continue;
      for (std::uint8_t c = 0; c < f.nchoices; ++c) {
        const std::string needle = std::string(f.name) + "=" + f.choices[c];
        CHECK_MSG(all.find(needle) != std::string::npos,
                  std::string("  corpus never uses ") + d.mnemonic + " " + needle);
      }
    }
  }
  // All six unit ids must appear as an issuing queue.
  for (int u = 0; u < KEA_NUM_UNITS; ++u)
    CHECK_MSG(all.find(std::string("  ") + unitName(static_cast<Unit>(u)) + " ") != std::string::npos,
              std::string("  corpus never issues on ") + unitName(static_cast<Unit>(u)));
}

/// --annotate must resolve SPM addresses through KeaProgram::spm_map.
void testAnnotate() {
  const std::string stem = "double_buffered";
  Diagnostics diags;
  Loaded side = loadSidecar(stem, diags);
  CHECK(side.have_map);

  AsmOptions opts;
  opts.map = &side.map;
  opts.const_data = &side.const_data;
  KeaProgram program;
  if (!assembleOrReport(readFile(std::string(kKasmDir) + "/" + stem + ".kasm"),
                        stem, opts, program))
    return;

  DisasmOptions dopts;
  dopts.map = &side.map;
  dopts.annotate = true;
  const std::string text = disassemble(program, dopts);
  CHECK(text.find("; pc=0 ") != std::string::npos);
  CHECK(text.find("spm_addr=conv1/weights") != std::string::npos);
  CHECK(text.find("acc_addr=conv1/acc[0]") != std::string::npos);
  CHECK(text.find("a_addr=conv1/act[buf0]+16") != std::string::npos);
}

void runTests() {
  testCanonicalRoundTrip("all_opcodes");
  testCanonicalRoundTrip("double_buffered");
  testCanonicalRoundTrip("echo");
  testMessyNormalizes();
  testCorpusCoversEverything();
  testAnnotate();
}

}  // namespace

TEST_MAIN("test_roundtrip_kasm")
