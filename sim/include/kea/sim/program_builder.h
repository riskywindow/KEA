// SPDX-License-Identifier: Apache-2.0
//
// kea/sim/program_builder.h --- assemble a KeaProgram in memory.
//
// The simulator's input is always `kea::KeaProgram`, whether it came from a
// `.keaf` binary, from `.kasm` text, or from code.  This tiny builder is the
// third path: it wraps the keaMake* constructors from isa.h so tests (and
// small experiments) can produce a runnable program without a file on disk.
//
// It deliberately does NOT parse anything -- loading artifacts is `runtime/`'s
// job.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kea/isa.h"
#include "kea/program.h"

namespace kea {
namespace sim {

class ProgramBuilder {
 public:
  explicit ProgramBuilder(std::uint64_t dram_bytes = 1u << 20) {
    prog_.dram.total_bytes = dram_bytes;
    prog_.dram.alignment = KEA_DRAM_BASE_ALIGN;
    prog_.dram.const_offset = 0;
    prog_.dram.const_bytes = 0;
    prog_.dram.io_offset = 0;
    prog_.dram.io_bytes = dram_bytes;
    prog_.dram.scratch_offset = dram_bytes;
    prog_.dram.scratch_bytes = 0;
  }

  /// Append one instruction; returns its pc.
  std::uint32_t add(const KeaInstr& in) {
    prog_.code.push_back(in);
    return static_cast<std::uint32_t>(prog_.code.size() - 1);
  }

  ProgramBuilder& nop(Unit u, std::uint32_t cycles = 0) {
    add(keaMakeNop(u, cycles));
    return *this;
  }
  ProgramBuilder& signal(Unit u, std::uint8_t ev, std::uint32_t inc = 1) {
    add(keaMakeSignal(u, ev, inc));
    return *this;
  }
  ProgramBuilder& wait(Unit u, std::uint8_t ev, std::uint32_t thr = 1) {
    add(keaMakeWait(u, ev, thr));
    return *this;
  }
  ProgramBuilder& trace(Unit u, TraceKind k, std::uint32_t tag,
                        std::uint32_t payload = 0) {
    add(keaMakeTrace(u, k, tag, payload));
    return *this;
  }
  ProgramBuilder& halt(std::uint32_t exit_code = 0) {
    add(keaMakeHalt(exit_code));
    return *this;
  }

  /// Contents of the DRAM arena staged before execution (the KEAF CONST
  /// section's role).  `data` is placed at `offset`.
  void putConst(std::uint64_t offset, const std::vector<std::uint8_t>& data) {
    if (prog_.const_data.size() < offset + data.size())
      prog_.const_data.resize(offset + data.size(), 0);
    for (std::size_t i = 0; i < data.size(); ++i)
      prog_.const_data[offset + i] = data[i];
    prog_.dram.const_offset = 0;
    prog_.dram.const_bytes = prog_.const_data.size();
    if (prog_.dram.total_bytes < prog_.dram.const_bytes)
      prog_.dram.total_bytes = prog_.dram.const_bytes;
    prog_.dram.io_offset = prog_.dram.const_bytes;
    prog_.dram.io_bytes = prog_.dram.total_bytes - prog_.dram.const_bytes;
    prog_.dram.scratch_offset = prog_.dram.total_bytes;
    prog_.dram.scratch_bytes = 0;
  }

  void addTensor(const TensorBinding& t) { prog_.tensors.push_back(t); }

  KeaProgram& program() { return prog_; }
  const KeaProgram& program() const { return prog_; }

  /// Finish the program with a HALT (if it does not already end in one).
  KeaProgram& finish(std::uint32_t exit_code = 0) {
    if (prog_.code.empty() ||
        keaOpcode(prog_.code.back()) != Opcode::HALT)
      halt(exit_code);
    return prog_;
  }

 private:
  KeaProgram prog_;
};

}  // namespace sim
}  // namespace kea
