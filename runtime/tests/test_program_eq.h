// SPDX-License-Identifier: Apache-2.0
//
// Deep equality for KeaProgram, with a human-readable explanation of the first
// difference. Used by the KEAF and `.kasm` round-trip tests.

#pragma once

#include <cstring>
#include <string>

#include "kea/program.h"

namespace keatest {

inline bool programsEqual(const kea::KeaProgram& a, const kea::KeaProgram& b, std::string& why) {
  auto fail = [&](const std::string& m) {
    why = "  " + m;
    return false;
  };
  if (a.entry_pc != b.entry_pc)
    return fail("entry_pc " + std::to_string(a.entry_pc) + " != " + std::to_string(b.entry_pc));
  if (a.code.size() != b.code.size())
    return fail("code size " + std::to_string(a.code.size()) + " != " + std::to_string(b.code.size()));
  for (std::size_t i = 0; i < a.code.size(); ++i) {
    if (std::memcmp(a.code[i].b, b.code[i].b, 32) != 0) {
      std::string ha, hb;
      const char* hex = "0123456789abcdef";
      for (int k = 0; k < 32; ++k) {
        ha += hex[a.code[i].b[k] >> 4];
        ha += hex[a.code[i].b[k] & 0xF];
        hb += hex[b.code[i].b[k] >> 4];
        hb += hex[b.code[i].b[k] & 0xF];
      }
      return fail("instruction " + std::to_string(i) + " differs\n    a: " + ha + "\n    b: " + hb);
    }
  }
  if (std::memcmp(&a.dram, &b.dram, sizeof(kea::KeafDramLayout)) != 0)
    return fail("DRAM_LAYOUT differs");
  if (a.const_data != b.const_data) return fail("const_data differs");
  if (a.tensors.size() != b.tensors.size())
    return fail("tensor count " + std::to_string(a.tensors.size()) + " != " +
                std::to_string(b.tensors.size()));
  for (std::size_t i = 0; i < a.tensors.size(); ++i) {
    const kea::TensorBinding& x = a.tensors[i];
    const kea::TensorBinding& y = b.tensors[i];
    if (x.name != y.name || x.dram_offset != y.dram_offset || x.size_bytes != y.size_bytes ||
        x.shape != y.shape || x.dtype != y.dtype || x.kind != y.kind || x.layout != y.layout ||
        x.scale != y.scale || x.zero_point != y.zero_point || x.index != y.index)
      return fail("tensor '" + x.name + "' differs from '" + y.name + "'");
  }
  if (a.spm_map.size() != b.spm_map.size()) return fail("spm_map size differs");
  for (std::size_t i = 0; i < a.spm_map.size(); ++i) {
    const kea::SpmBinding& x = a.spm_map[i];
    const kea::SpmBinding& y = b.spm_map[i];
    if (x.name != y.name || x.space != y.space || x.offset != y.offset || x.size != y.size ||
        x.first_pc != y.first_pc || x.last_pc != y.last_pc)
      return fail("spm_map entry '" + x.name + "' differs");
  }
  if (a.metadata_json != b.metadata_json)
    return fail("metadata differs\n    a: " + a.metadata_json + "\n    b: " + b.metadata_json);
  return true;
}

}  // namespace keatest
