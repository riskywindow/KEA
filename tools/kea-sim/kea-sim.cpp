// SPDX-License-Identifier: Apache-2.0
//
// kea-sim --- run a KEA-1 program on the cycle-approximate simulator.
//
//   kea-sim program.keaf [--map model.map.json]
//           [--input name=file.bin ...] [--output name=file.bin ...]
//           [--stats-json out.json] [--trace[=file]] [--max-cycles N]
//
// Loading `.keaf` / `.kasm` is `runtime/`'s job (kea::rt::loadProgramFile);
// this tool only ever sees a `kea::KeaProgram`.  When the build is configured
// without `runtime/`, everything except artifact loading still builds and the
// loader reports why.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kea/program.h"
#include "kea/sim/simulator.h"
#include "kea/sim/stats.h"

#if KEA_SIM_HAVE_RUNTIME
#include "kea/rt/diagnostics.h"
#include "kea/rt/load.h"
#include "kea/rt/model_map.h"
#endif

using namespace kea;
using namespace kea::sim;

namespace {

struct Binding {
  std::string name;
  std::string path;
};

void usage() {
  std::fprintf(
      stderr,
      "usage: kea-sim <program.keaf|program.kasm> [options]\n"
      "\n"
      "  --map FILE             model map JSON (required for .kasm input)\n"
      "  --const FILE           raw constant blob for .kasm input\n"
      "  --input NAME=FILE      stage FILE into the DRAM arena at tensor NAME\n"
      "  --output NAME=FILE     write tensor NAME back out after the run\n"
      "  --stats-json FILE      machine-readable statistics\n"
      "  --trace[=FILE]         per-instruction issue/start/retire trace\n"
      "  --max-cycles N         abort after N simulated cycles\n"
      "  --quiet                suppress the human-readable report\n"
      "  --strict-poison        reads of never-written scratchpad are fatal\n"
      "  --strict-hazards       cross-unit unsynchronized reads are fatal\n"
      "  --no-rule-d            do not statically enforce Rule D (ISA.md 5.5)\n"
      "  --list-tensors         print the tensor table and exit\n");
}

bool splitBinding(const char* arg, Binding& out) {
  const char* eq = std::strchr(arg, '=');
  if (!eq) return false;
  out.name.assign(arg, static_cast<std::size_t>(eq - arg));
  out.path = eq + 1;
  return !out.name.empty() && !out.path.empty();
}

bool readFile(const std::string& path, std::vector<std::uint8_t>& out) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::uint8_t buf[65536];
  std::size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.insert(out.end(), buf, buf + n);
  std::fclose(f);
  return true;
}

bool writeFile(const std::string& path, const void* data, std::size_t n) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const bool ok = std::fwrite(data, 1, n, f) == n;
  std::fclose(f);
  return ok;
}

bool loadProgram(const std::string& path, const std::string& map_path,
                 const std::string& const_path, bool rule_d, KeaProgram& out) {
#if KEA_SIM_HAVE_RUNTIME
  rt::Diagnostics diags;
  rt::ModelMap map;
  rt::LoadOptions opts;
  opts.rule_d_is_error = rule_d;
  if (!map_path.empty()) {
    if (!rt::loadModelMapFile(map_path, map, diags)) {
      std::fputs(diags.render().c_str(), stderr);
      return false;
    }
    opts.map = &map;
  }
  std::vector<std::uint8_t> const_data;
  if (!const_path.empty()) {
    if (!readFile(const_path, const_data)) {
      std::fprintf(stderr, "kea-sim: cannot read %s\n", const_path.c_str());
      return false;
    }
    opts.const_data = &const_data;
  }
  const bool ok = rt::loadProgramFile(path, opts, out, diags);
  if (!diags.all().empty()) std::fputs(diags.render().c_str(), stderr);
  return ok;
#else
  (void)path;
  (void)map_path;
  (void)const_path;
  (void)rule_d;
  (void)out;
  std::fprintf(stderr,
               "kea-sim: this build has no artifact loader.\n"
               "  Loading .keaf / .kasm lives in runtime/ (kea_runtime); this "
               "binary was configured without it.\n"
               "  Re-run cmake with runtime/ present.\n");
  return false;
#endif
}

void printTensors(const KeaProgram& p) {
  std::printf("%-32s %-8s %-6s %12s %12s  shape\n", "name", "kind", "dtype",
              "dram_offset", "size_bytes");
  for (const TensorBinding& t : p.tensors) {
    std::printf("%-32s %-8s %-6s %12llu %12llu  [", t.name.c_str(),
                keafTensorKindName(t.kind), keafDTypeName(t.dtype),
                static_cast<unsigned long long>(t.dram_offset),
                static_cast<unsigned long long>(t.size_bytes));
    for (std::size_t i = 0; i < t.shape.size(); ++i)
      std::printf("%s%d", i ? ", " : "", t.shape[i]);
    std::printf("]\n");
  }
}

void writeTrace(std::FILE* f, const SimResult& r) {
  std::fprintf(f, "%6s %-4s %-8s %10s %10s %10s %8s %8s\n", "pc", "unit",
               "opcode", "issue", "start", "retire", "sem", "res");
  for (const TraceRecord& t : r.trace) {
    std::fprintf(f, "%6u %-4s %-8s %10llu %10llu %10llu %8llu %8llu\n", t.pc,
                 unitName(static_cast<Unit>(t.unit)),
                 opcodeName(static_cast<Opcode>(t.opcode)),
                 static_cast<unsigned long long>(t.issue),
                 static_cast<unsigned long long>(t.start),
                 static_cast<unsigned long long>(t.retire),
                 static_cast<unsigned long long>(t.stall_sem),
                 static_cast<unsigned long long>(t.stall_res));
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string program_path, map_path, const_path, stats_json, trace_path;
  std::vector<Binding> inputs, outputs;
  SimConfig cfg;
  cfg.max_cycles = 2000000000ull;
  bool quiet = false, list_tensors = false, rule_d = true;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "kea-sim: %s needs an argument\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else if (a == "--map") {
      map_path = need("--map");
    } else if (a == "--const") {
      const_path = need("--const");
    } else if (a == "--input" || a == "--output") {
      Binding b;
      const char* v = need(a.c_str());
      if (!splitBinding(v, b)) {
        std::fprintf(stderr, "kea-sim: %s expects NAME=FILE, got '%s'\n",
                     a.c_str(), v);
        return 2;
      }
      (a == "--input" ? inputs : outputs).push_back(b);
    } else if (a == "--stats-json") {
      stats_json = need("--stats-json");
    } else if (a == "--trace") {
      cfg.trace = true;
    } else if (a.rfind("--trace=", 0) == 0) {
      cfg.trace = true;
      trace_path = a.substr(8);
    } else if (a == "--max-cycles") {
      cfg.max_cycles = std::strtoull(need("--max-cycles"), nullptr, 0);
    } else if (a == "--quiet") {
      quiet = true;
    } else if (a == "--strict-poison") {
      cfg.strict_poison = true;
    } else if (a == "--strict-hazards") {
      cfg.strict_hazards = true;
    } else if (a == "--no-rule-d") {
      cfg.check_rule_d = false;
      rule_d = false;
    } else if (a == "--list-tensors") {
      list_tensors = true;
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "kea-sim: unknown option '%s'\n", a.c_str());
      usage();
      return 2;
    } else if (program_path.empty()) {
      program_path = a;
    } else {
      std::fprintf(stderr, "kea-sim: unexpected argument '%s'\n", a.c_str());
      return 2;
    }
  }

  if (program_path.empty()) {
    usage();
    return 2;
  }

  KeaProgram program;
  if (!loadProgram(program_path, map_path, const_path, rule_d, program))
    return 1;

  if (list_tensors) {
    printTensors(program);
    return 0;
  }

  Simulator sim(program, cfg);
  sim.stageConstants();

  // Stage the named inputs into the DRAM arena.
  for (const Binding& b : inputs) {
    const TensorBinding* t = program.findTensor(b.name);
    if (!t) {
      std::fprintf(stderr, "kea-sim: no tensor named '%s'\n", b.name.c_str());
      return 1;
    }
    std::vector<std::uint8_t> bytes;
    if (!readFile(b.path, bytes)) {
      std::fprintf(stderr, "kea-sim: cannot read %s\n", b.path.c_str());
      return 1;
    }
    if (bytes.size() != t->size_bytes)
      std::fprintf(stderr,
                   "kea-sim: warning: %s is %zu bytes but tensor '%s' is %llu\n",
                   b.path.c_str(), bytes.size(), b.name.c_str(),
                   static_cast<unsigned long long>(t->size_bytes));
    const std::size_t n = bytes.size() < t->size_bytes
                              ? bytes.size()
                              : static_cast<std::size_t>(t->size_bytes);
    sim.machine().dram.write(static_cast<std::int64_t>(t->dram_offset),
                             static_cast<std::int64_t>(n), bytes.data());
  }

  SimResult r = sim.run();

  if (!quiet)
    std::fputs(formatReport(r.stats, r.cycles).c_str(), stdout);

  for (const std::string& d : r.diagnostics)
    std::fprintf(stderr, "kea-sim: %s\n", d.c_str());
  if (!r.ok())
    std::fprintf(stderr, "kea-sim: %s: %s\n", statusName(r.status),
                 r.message.c_str());

  if (!stats_json.empty()) {
    const std::string j = formatJson(r.stats, r.cycles, statusName(r.status),
                                     r.exit_code, r.diagnostics);
    if (!writeFile(stats_json, j.data(), j.size())) {
      std::fprintf(stderr, "kea-sim: cannot write %s\n", stats_json.c_str());
      return 1;
    }
  }

  if (cfg.trace) {
    if (trace_path.empty()) {
      writeTrace(stdout, r);
    } else {
      std::FILE* f = std::fopen(trace_path.c_str(), "w");
      if (!f) {
        std::fprintf(stderr, "kea-sim: cannot write %s\n", trace_path.c_str());
        return 1;
      }
      writeTrace(f, r);
      std::fclose(f);
    }
  }

  // Outputs are written even on a failed run, so a partial result can be
  // inspected.
  for (const Binding& b : outputs) {
    const TensorBinding* t = program.findTensor(b.name);
    if (!t) {
      std::fprintf(stderr, "kea-sim: no tensor named '%s'\n", b.name.c_str());
      return 1;
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(t->size_bytes));
    sim.machine().dram.read(static_cast<std::int64_t>(t->dram_offset),
                            static_cast<std::int64_t>(bytes.size()),
                            bytes.data());
    if (!writeFile(b.path, bytes.data(), bytes.size())) {
      std::fprintf(stderr, "kea-sim: cannot write %s\n", b.path.c_str());
      return 1;
    }
  }

  if (!r.ok()) return 1;
  return r.exit_code == 0 ? 0 : static_cast<int>(r.exit_code & 0x7F);
}
