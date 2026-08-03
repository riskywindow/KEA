// SPDX-License-Identifier: Apache-2.0
//
// kea/rt/load.h --- the one entry point the simulator needs.
//
// ADR-0001 rule 4: `.kasm` is the simulator's *other* front door. `kea-sim`
// accepts either a `.keaf` or a `.kasm` directly, and this function is how it
// does that: it sniffs the KEAF magic and either decodes the binary or runs
// the assembler, producing the same `KeaProgram` either way.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kea/program.h"
#include "kea/rt/assembler.h"
#include "kea/rt/diagnostics.h"
#include "kea/rt/model_map.h"

namespace kea {
namespace rt {

struct LoadOptions {
  /// Used only when the input is `.kasm`: resolves `@symbol` operands and
  /// supplies the DRAM layout, tensors, SPM map and metadata.
  const ModelMap* map = nullptr;
  /// Used only when the input is `.kasm`: the constant blob.
  const std::vector<std::uint8_t>* const_data = nullptr;
  /// Verify KEAF CRCs. Cheap for the artifacts we produce; skip it in a hot
  /// load path once the file has been verified at install time.
  bool check_crc = true;
  /// See AsmOptions::rule_d_is_error.
  bool rule_d_is_error = true;
};

/// True when `data` starts with the KEAF magic.
bool looksLikeKeaf(const void* data, std::size_t size);

/// Load a `.keaf` or a `.kasm` from disk.
bool loadProgramFile(const std::string& path, const LoadOptions& opts, KeaProgram& program,
                     Diagnostics& diags);

/// Load from memory. `filename` only appears in diagnostics.
bool loadProgram(const std::vector<std::uint8_t>& bytes, const std::string& filename,
                 const LoadOptions& opts, KeaProgram& program, Diagnostics& diags);

}  // namespace rt
}  // namespace kea
