// SPDX-License-Identifier: Apache-2.0
//
// kea/rt/disassembler.h --- KeaProgram -> canonical `.kasm` text.
//
// The output is the canonical form defined in docs/ASSEMBLY.md §3: one
// instruction per line, fields in `keaOpInfo()` order, decimal literals, no
// comments. `assemble(disassemble(p)) == p` always holds, and for a source
// file already written in canonical form `disassemble(assemble(x)) == x` holds
// character for character. That round trip is the ADR-0001 anti-drift test.

#pragma once

#include <string>

#include "kea/program.h"
#include "kea/rt/model_map.h"

namespace kea {
namespace rt {

struct DisasmOptions {
  /// DRAM symbol table. When present, DRAM addresses print as `@sym` /
  /// `@sym+off`; otherwise they print as the `dram:<byte-address>` escape.
  const ModelMap* map = nullptr;

  /// Append `; pc=N` and, where `KeaProgram::spm_map` covers an operand, the
  /// planner's buffer name. Annotated output is NOT canonical and does not
  /// re-assemble to a byte-identical listing (the comments are dropped).
  bool annotate = false;

  /// Emit the `.arch` / `.isa_revision` / `.entry` prologue and the
  /// `.event` / `.region` / label declarations recovered from metadata.
  bool emit_directives = true;
};

std::string disassemble(const KeaProgram& program, const DisasmOptions& opts);

/// Single instruction, in the canonical `UNIT MNEMONIC field=value, ...` form
/// (without the leading indent). Handy for simulator traces.
std::string disassembleInstruction(const KeaInstr& in, const ModelMap* map = nullptr);

}  // namespace rt
}  // namespace kea
