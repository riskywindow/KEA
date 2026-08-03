// SPDX-License-Identifier: Apache-2.0
//
// kea/rt/model_map.h --- `model.map.json`: the DRAM symbol table and tensor
// descriptor file that accompanies a `.kasm` listing.
//
// ADR-0001 rule 3: DRAM addresses in `.kasm` are symbolic and are resolved by
// the assembler against this file. SPM/ACC addresses are absolute integers and
// appear nowhere here except as debug names in `spm_map`.
//
// The schema is specified in docs/ASSEMBLY.md §7.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kea/program.h"
#include "kea/rt/diagnostics.h"
#include "kea/rt/json.h"

namespace kea {
namespace rt {

/// A named region of the DRAM arena. `@name` in `.kasm` resolves to `offset`.
struct MapSymbol {
  std::string name;
  std::uint64_t offset = 0;
  std::uint64_t size = 0;
  /// True when the symbol was synthesised from a `tensors[]` entry rather than
  /// declared in `symbols[]`. Only affects diagnostics.
  bool from_tensor = false;
};

struct ModelMap {
  KeafDramLayout dram{};
  std::vector<MapSymbol> symbols;   ///< `symbols[]` followed by `tensors[]`
  std::vector<TensorBinding> tensors;
  std::vector<SpmBinding> spm_map;
  std::uint32_t entry_pc = 0;
  /// The `metadata` object, re-serialized. Becomes the KEAF METADATA section.
  std::string metadata_json;

  const MapSymbol* find(const std::string& name) const;
  /// The symbol whose [offset, offset+size) contains `addr`, preferring the
  /// tightest match. Used by the disassembler to re-symbolize DRAM addresses.
  const MapSymbol* covering(std::uint64_t addr) const;
};

/// Parse a `model.map.json`. Returns false and fills `diags` on any error.
bool parseModelMap(const std::string& text, const std::string& filename, ModelMap& out,
                   Diagnostics& diags);
bool loadModelMapFile(const std::string& path, ModelMap& out, Diagnostics& diags);

/// Serialize a ModelMap back to JSON text (used by tests and `kea-dis --emit-map`).
std::string dumpModelMap(const ModelMap& map);

/// Build the map implied by an already-loaded program, so `kea-dis` can
/// symbolize a `.keaf` without a side-car file.
ModelMap modelMapFromProgram(const KeaProgram& program);

// Enum <-> string helpers shared with the JSON schema.
bool parseDType(const std::string& s, KeafDType& out);
bool parseTensorKind(const std::string& s, KeafTensorKind& out);
bool parseLayout(const std::string& s, KeafLayout& out);
bool parseSpmSpace(const std::string& s, KeafSpace& out);
const char* layoutName(KeafLayout l);
const char* spmSpaceName(KeafSpace s);

}  // namespace rt
}  // namespace kea
