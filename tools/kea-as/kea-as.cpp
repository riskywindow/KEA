// SPDX-License-Identifier: Apache-2.0
//
// kea-as --- the KEA-1 assembler.
//
//   kea-as model.kasm --map model.map.json --const model.weights.bin -o model.keaf
//
// See docs/ASSEMBLY.md for the `.kasm` syntax and the `model.map.json` schema.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "kea/rt/assembler.h"
#include "kea/rt/keaf_io.h"
#include "kea/rt/model_map.h"

namespace {

void usage(FILE* f) {
  std::fprintf(f,
               "usage: kea-as <input.kasm> [-o <output.keaf>] [options]\n"
               "\n"
               "  -o, --output <file>   artifact to write (default: input with .keaf)\n"
               "      --map <file>      model.map.json: DRAM symbols, tensors, SPM map, metadata\n"
               "      --const <file>    constant/weight blob staged at dram.const_offset\n"
               "      --strip           omit the SPM_MAP and METADATA debug sections\n"
               "      --no-crc          store 0 instead of the whole-file CRC-32\n"
               "      --no-rule-d       downgrade Rule D violations to warnings\n"
               "      --check           parse and validate only; write nothing\n"
               "  -h, --help\n");
}

std::string replaceExtension(const std::string& path, const char* ext) {
  const std::size_t slash = path.find_last_of('/');
  const std::size_t dot = path.find_last_of('.');
  if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
    return path.substr(0, dot) + ext;
  return path + ext;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input, output, map_path, const_path;
  bool strip = false, crc = true, rule_d = true, check_only = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "kea-as: %s expects an argument\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      usage(stdout);
      return 0;
    } else if (a == "-o" || a == "--output") {
      output = next("-o");
    } else if (a == "--map") {
      map_path = next("--map");
    } else if (a == "--const") {
      const_path = next("--const");
    } else if (a == "--strip") {
      strip = true;
    } else if (a == "--no-crc") {
      crc = false;
    } else if (a == "--no-rule-d") {
      rule_d = false;
    } else if (a == "--check") {
      check_only = true;
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "kea-as: unknown option '%s'\n", a.c_str());
      usage(stderr);
      return 2;
    } else if (input.empty()) {
      input = a;
    } else {
      std::fprintf(stderr, "kea-as: unexpected extra input '%s'\n", a.c_str());
      return 2;
    }
  }

  if (input.empty()) {
    usage(stderr);
    return 2;
  }
  if (output.empty()) output = replaceExtension(input, ".keaf");

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

  std::vector<std::uint8_t> const_data;
  bool have_const = false;
  if (!const_path.empty()) {
    std::string err;
    if (!kea::rt::readWholeFile(const_path, const_data, err)) {
      std::fprintf(stderr, "kea-as: %s\n", err.c_str());
      return 1;
    }
    have_const = true;
  }

  kea::rt::AsmOptions opts;
  opts.map = have_map ? &map : nullptr;
  opts.const_data = have_const ? &const_data : nullptr;
  opts.rule_d_is_error = rule_d;

  kea::KeaProgram program;
  const bool ok = kea::rt::assembleFile(input, opts, program, diags);
  std::fputs(diags.render().c_str(), stderr);
  if (!ok) return 1;

  if (check_only) {
    std::fprintf(stderr, "kea-as: %s: %zu instructions, ok\n", input.c_str(), program.code.size());
    return 0;
  }

  kea::rt::KeafWriteOptions wopts;
  wopts.include_spm_map = !strip;
  wopts.include_metadata = !strip;
  wopts.compute_crc = crc;

  std::string err;
  if (!kea::rt::writeKeafFile(program, output, err, wopts)) {
    std::fprintf(stderr, "kea-as: %s\n", err.c_str());
    return 1;
  }
  return 0;
}
