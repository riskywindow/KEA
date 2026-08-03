// SPDX-License-Identifier: Apache-2.0
//
// kea/rt/assembler.h --- `.kasm` text -> KeaProgram.
//
// The syntax is specified in docs/ASSEMBLY.md and derived mechanically from
// kea/rt/op_fields.h (which is itself checked against kea::keaOpInfo()).
//
// Two rules from ADR-0001 are enforced here and are the reason the assembler
// exists at all:
//
//   * SPM_A / SPM_W / ACC addresses are ABSOLUTE and are never relocated. They
//     are written with a mandatory space prefix (`a:`, `w:`, `acc:`) and land
//     in the encoding bit for bit.
//   * DRAM addresses are SYMBOLIC (`@sym`, `@sym+off`) and are resolved against
//     model.map.json. A `dram:<literal>` escape exists for hand-written tests.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kea/program.h"
#include "kea/rt/diagnostics.h"
#include "kea/rt/model_map.h"

namespace kea {
namespace rt {

struct AsmOptions {
  /// DRAM symbol table + tensor/SPM/metadata descriptors. May be null, in
  /// which case `@symbol` operands are an error and the resulting program has
  /// an empty DRAM arena.
  const ModelMap* map = nullptr;

  /// The `.bin` constant/weight blob that the map's `dram.const_bytes`
  /// describes. Required whenever `const_bytes != 0`; the resulting program
  /// owns a copy.
  const std::vector<std::uint8_t>* const_data = nullptr;

  /// Rule D (ISA.md §5.5): every `WAIT e, thr` must be preceded in stream
  /// order by `SIGNAL`s supplying `thr` counts. Violating it deadlocks the
  /// machine, so it is an error by default; simulator deadlock-detection tests
  /// legitimately need to turn it into a warning.
  bool rule_d_is_error = true;
};

/// Assemble `source`. Returns false and fills `diags` on any error; `program`
/// is then unspecified. `filename` only appears in diagnostics.
bool assemble(const std::string& source, const std::string& filename, const AsmOptions& opts,
              KeaProgram& program, Diagnostics& diags);

/// Read `path` and assemble it.
bool assembleFile(const std::string& path, const AsmOptions& opts, KeaProgram& program,
                  Diagnostics& diags);

}  // namespace rt
}  // namespace kea
