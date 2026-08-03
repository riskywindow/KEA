// SPDX-License-Identifier: Apache-2.0
//
// kea-rt --- load a KEA-1 artifact and run it.
//
//   kea-rt model.keaf --input input=in.bin --output output=out.bin
//
// This is the deployment-shaped driver, as opposed to `kea-sim`, which is the
// analysis-shaped one. It follows ARTIFACT_FORMAT.md §11 literally:
//
//   1. load and validate the artifact                  (kea::rt::loadProgramFile)
//   2. allocate ONE DRAM arena, stage CONST into it    (kea::rt::DramArena)
//   3. write the host's input tensors into the arena   (arena.writeTensor)
//   4. hand the arena to the device and ring the bell  (kea::sim::Simulator)
//   5. read the output tensors back out of the arena   (arena.readTensor)
//
// Step 2 is the only allocation in the whole sequence, which is the property
// runtime/tests/test_no_alloc.cpp pins down. The simulator stands in for the
// device: the arena is pushed into its DRAM once before the run and the
// output ranges are pulled back once after it, exactly as a real driver would
// DMA them across a bus.
//
// The process exit status is the program's own HALT exit_code when the run
// succeeded, so a `.kasm` self-test can report pass/fail to a shell.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "kea/program.h"
#include "kea/rt/diagnostics.h"
#include "kea/rt/dram_arena.h"
#include "kea/rt/keaf_io.h"
#include "kea/rt/load.h"
#include "kea/rt/model_map.h"
#include "kea/sim/simulator.h"

using namespace kea;

namespace {

struct Binding {
  std::string name;
  std::string path;
};

void usage(std::FILE* f) {
  std::fprintf(f,
               "usage: kea-rt <program.keaf|program.kasm> [options]\n"
               "\n"
               "      --input NAME=FILE   stage FILE into input tensor NAME\n"
               "      --output NAME=FILE  write output tensor NAME to FILE after the run\n"
               "      --map FILE          model.map.json (required for .kasm input)\n"
               "      --const FILE        constant blob (required for .kasm with const_bytes)\n"
               "      --list-tensors      print the tensor table and exit\n"
               "      --check             load and stage only; do not run\n"
               "      --max-cycles N      abort the run after N simulated cycles\n"
               "      --no-crc            skip KEAF CRC verification\n"
               "  -q, --quiet             suppress the completion summary\n"
               "  -h, --help\n");
}

bool splitBinding(const std::string& arg, Binding& out) {
  const std::size_t eq = arg.find('=');
  if (eq == std::string::npos || eq == 0 || eq + 1 == arg.size()) return false;
  out.name = arg.substr(0, eq);
  out.path = arg.substr(eq + 1);
  return true;
}

const char* dtypeName(KeafDType d) { return keafDTypeName(d); }

void printTensors(const KeaProgram& p) {
  std::printf("%-8s %-24s %10s %10s  %-6s %s\n", "kind", "name", "dram_off", "bytes", "dtype",
              "shape");
  for (const TensorBinding& t : p.tensors) {
    std::string shape = "[";
    for (std::size_t i = 0; i < t.shape.size(); ++i) {
      if (i) shape += ",";
      shape += std::to_string(t.shape[i]);
    }
    shape += "]";
    std::printf("%-8s %-24s %10llu %10llu  %-6s %s  scale=%g zp=%d\n",
                keafTensorKindName(t.kind), t.name.c_str(),
                static_cast<unsigned long long>(t.dram_offset),
                static_cast<unsigned long long>(t.size_bytes), dtypeName(t.dtype), shape.c_str(),
                static_cast<double>(t.scale), t.zero_point);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string input_path, map_path, const_path;
  std::vector<Binding> inputs, outputs;
  bool list_tensors = false, check_only = false, quiet = false, check_crc = true;
  std::uint64_t max_cycles = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "kea-rt: %s expects an argument\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      usage(stdout);
      return 0;
    } else if (a == "--input" || a == "--output") {
      Binding b;
      const std::string spec = next(a.c_str());
      if (!splitBinding(spec, b)) {
        std::fprintf(stderr, "kea-rt: %s expects NAME=FILE, got '%s'\n", a.c_str(), spec.c_str());
        return 2;
      }
      (a == "--input" ? inputs : outputs).push_back(b);
    } else if (a == "--map") {
      map_path = next("--map");
    } else if (a == "--const") {
      const_path = next("--const");
    } else if (a == "--list-tensors") {
      list_tensors = true;
    } else if (a == "--check") {
      check_only = true;
    } else if (a == "--no-crc") {
      check_crc = false;
    } else if (a == "--max-cycles") {
      max_cycles = std::strtoull(next("--max-cycles").c_str(), nullptr, 10);
    } else if (a == "-q" || a == "--quiet") {
      quiet = true;
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "kea-rt: unknown option '%s'\n", a.c_str());
      usage(stderr);
      return 2;
    } else if (input_path.empty()) {
      input_path = a;
    } else {
      std::fprintf(stderr, "kea-rt: unexpected extra input '%s'\n", a.c_str());
      return 2;
    }
  }
  if (input_path.empty()) {
    usage(stderr);
    return 2;
  }

  // --- 1. load -------------------------------------------------------------
  rt::Diagnostics diags;
  rt::ModelMap map;
  bool have_map = false;
  if (!map_path.empty()) {
    if (!rt::loadModelMapFile(map_path, map, diags)) {
      std::fputs(diags.render().c_str(), stderr);
      return 1;
    }
    have_map = true;
  }
  std::vector<std::uint8_t> const_data;
  bool have_const = false;
  if (!const_path.empty()) {
    std::string err;
    if (!rt::readWholeFile(const_path, const_data, err)) {
      std::fprintf(stderr, "kea-rt: %s\n", err.c_str());
      return 1;
    }
    have_const = true;
  }

  rt::LoadOptions lopts;
  lopts.map = have_map ? &map : nullptr;
  lopts.const_data = have_const ? &const_data : nullptr;
  lopts.check_crc = check_crc;

  KeaProgram program;
  if (!rt::loadProgramFile(input_path, lopts, program, diags)) {
    std::fputs(diags.render().c_str(), stderr);
    return 1;
  }
  std::fputs(diags.render().c_str(), stderr);

  if (list_tensors) {
    printTensors(program);
    return 0;
  }

  // --- 2. allocate the arena and stage the constants ------------------------
  rt::DramArena arena;
  std::string err;
  if (!arena.reset(program, err)) {
    std::fprintf(stderr, "kea-rt: %s\n", err.c_str());
    return 1;
  }

  // --- 3. stage the host's inputs ------------------------------------------
  for (const Binding& b : inputs) {
    const TensorBinding* t = program.findTensor(b.name);
    if (!t) {
      std::fprintf(stderr, "kea-rt: no tensor named '%s' in the artifact\n", b.name.c_str());
      return 1;
    }
    std::vector<std::uint8_t> bytes;
    if (!rt::readWholeFile(b.path, bytes, err)) {
      std::fprintf(stderr, "kea-rt: %s\n", err.c_str());
      return 1;
    }
    if (bytes.size() != t->size_bytes) {
      std::fprintf(stderr, "kea-rt: '%s' is %zu bytes but tensor '%s' expects %llu\n",
                   b.path.c_str(), bytes.size(), b.name.c_str(),
                   static_cast<unsigned long long>(t->size_bytes));
      return 1;
    }
    if (!arena.writeTensor(*t, bytes.data(), bytes.size())) {
      std::fprintf(stderr, "kea-rt: tensor '%s' does not fit in the DRAM arena\n", b.name.c_str());
      return 1;
    }
  }

  if (check_only) {
    if (!quiet)
      std::fprintf(stderr, "kea-rt: %s loaded and staged: %zu instructions, %llu-byte arena\n",
                   input_path.c_str(), program.code.size(),
                   static_cast<unsigned long long>(arena.size()));
    return 0;
  }

  // --- 4. hand the arena to the device and start it -------------------------
  sim::SimConfig cfg;
  cfg.max_cycles = max_cycles;
  sim::Simulator machine(program, cfg);
  if (arena.size())
    machine.machine().dram.write(0, static_cast<std::int64_t>(arena.size()), arena.data());

  const sim::SimResult result = machine.run();
  for (const std::string& d : result.diagnostics) std::fprintf(stderr, "kea-rt: %s\n", d.c_str());

  if (!result.ok()) {
    std::fprintf(stderr, "kea-rt: run failed (%s): %s\n", sim::statusName(result.status),
                 result.message.c_str());
    return 1;
  }

  // --- 5. read the outputs back out of the arena ----------------------------
  for (const Binding& b : outputs) {
    const TensorBinding* t = program.findTensor(b.name);
    if (!t) {
      std::fprintf(stderr, "kea-rt: no tensor named '%s' in the artifact\n", b.name.c_str());
      return 1;
    }
    std::uint8_t* dst = arena.tensorData(*t);
    if (!dst) {
      std::fprintf(stderr, "kea-rt: tensor '%s' does not fit in the DRAM arena\n", b.name.c_str());
      return 1;
    }
    machine.machine().dram.read(static_cast<std::int64_t>(t->dram_offset),
                                static_cast<std::int64_t>(t->size_bytes), dst);
    if (!rt::writeWholeFile(b.path, dst, static_cast<std::size_t>(t->size_bytes), err)) {
      std::fprintf(stderr, "kea-rt: %s\n", err.c_str());
      return 1;
    }
  }

  if (!quiet)
    std::fprintf(stderr, "kea-rt: %s completed in %llu cycles, exit_code=%u\n", input_path.c_str(),
                 static_cast<unsigned long long>(result.cycles), result.exit_code);
  return static_cast<int>(result.exit_code);
}
