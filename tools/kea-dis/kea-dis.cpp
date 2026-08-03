// SPDX-License-Identifier: Apache-2.0
//
// kea-dis --- the KEA-1 disassembler.
//
//   kea-dis model.keaf [--map model.map.json] [--annotate]
//
// Writes canonical `.kasm` (docs/ASSEMBLY.md §3) to stdout. Without --map, DRAM
// addresses are symbolized against the artifact's own TENSORS section; with
// --map they are symbolized against the full compiler symbol table.

#include <cstdio>
#include <string>
#include <vector>

#include "kea/rt/disassembler.h"
#include "kea/rt/keaf_io.h"
#include "kea/rt/load.h"
#include "kea/rt/model_map.h"

namespace {

void usage(FILE* f) {
  std::fprintf(f,
               "usage: kea-dis <input.keaf|input.kasm> [options]\n"
               "\n"
               "      --map <file>      model.map.json, for full DRAM symbolization\n"
               "      --annotate        append '; pc=N' and SPM_MAP buffer names\n"
               "      --no-symbols      print raw dram:<address> literals\n"
               "      --no-directives   omit the .arch/.isa_revision/.entry prologue\n"
               "      --emit-map        print the implied model.map.json instead of .kasm\n"
               "      --no-crc          skip CRC verification while loading\n"
               "  -o, --output <file>   write here instead of stdout\n"
               "  -h, --help\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string input, map_path, output;
  bool annotate = false, symbols = true, directives = true, emit_map = false, check_crc = true;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "kea-dis: %s expects an argument\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      usage(stdout);
      return 0;
    } else if (a == "--map") {
      map_path = next("--map");
    } else if (a == "--annotate") {
      annotate = true;
    } else if (a == "--no-symbols") {
      symbols = false;
    } else if (a == "--no-directives") {
      directives = false;
    } else if (a == "--emit-map") {
      emit_map = true;
    } else if (a == "--no-crc") {
      check_crc = false;
    } else if (a == "-o" || a == "--output") {
      output = next("-o");
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "kea-dis: unknown option '%s'\n", a.c_str());
      usage(stderr);
      return 2;
    } else if (input.empty()) {
      input = a;
    } else {
      std::fprintf(stderr, "kea-dis: unexpected extra input '%s'\n", a.c_str());
      return 2;
    }
  }
  if (input.empty()) {
    usage(stderr);
    return 2;
  }

  kea::rt::Diagnostics diags;
  kea::rt::ModelMap map;
  bool have_map = false;
  if (!map_path.empty()) {
    if (!kea::rt::loadModelMapFile(map_path, map, diags)) {
      std::fputs(diags.render().c_str(), stderr);
      return 1;
    }
    have_map = true;
  }

  kea::rt::LoadOptions lopts;
  lopts.map = have_map ? &map : nullptr;
  lopts.check_crc = check_crc;

  kea::KeaProgram program;
  if (!kea::rt::loadProgramFile(input, lopts, program, diags)) {
    std::fputs(diags.render().c_str(), stderr);
    return 1;
  }

  if (!have_map) map = kea::rt::modelMapFromProgram(program);

  std::string text;
  if (emit_map) {
    text = kea::rt::dumpModelMap(map);
  } else {
    kea::rt::DisasmOptions dopts;
    dopts.map = symbols ? &map : nullptr;
    dopts.annotate = annotate;
    dopts.emit_directives = directives;
    text = kea::rt::disassemble(program, dopts);
  }

  if (output.empty()) {
    std::fwrite(text.data(), 1, text.size(), stdout);
  } else {
    std::string err;
    if (!kea::rt::writeWholeFile(output, text.data(), text.size(), err)) {
      std::fprintf(stderr, "kea-dis: %s\n", err.c_str());
      return 1;
    }
  }
  return 0;
}
