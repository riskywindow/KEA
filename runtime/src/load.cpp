// SPDX-License-Identifier: Apache-2.0
#include "kea/rt/load.h"

#include "kea/rt/keaf_io.h"

namespace kea {
namespace rt {

bool looksLikeKeaf(const void* data, std::size_t size) {
  if (size < 4) return false;
  const auto* p = static_cast<const std::uint8_t*>(data);
  return p[0] == KEAF_MAGIC0 && p[1] == KEAF_MAGIC1 && p[2] == KEAF_MAGIC2 && p[3] == KEAF_MAGIC3;
}

bool loadProgram(const std::vector<std::uint8_t>& bytes, const std::string& filename,
                 const LoadOptions& opts, KeaProgram& program, Diagnostics& diags) {
  if (looksLikeKeaf(bytes.data(), bytes.size())) {
    std::string err;
    if (!readKeaf(bytes.data(), bytes.size(), program, err, opts.check_crc)) {
      diags.error(filename, 0, 0, 1, err);
      return false;
    }
    return true;
  }
  AsmOptions aopts;
  aopts.map = opts.map;
  aopts.const_data = opts.const_data;
  aopts.rule_d_is_error = opts.rule_d_is_error;
  const std::string text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  return assemble(text, filename, aopts, program, diags);
}

bool loadProgramFile(const std::string& path, const LoadOptions& opts, KeaProgram& program,
                     Diagnostics& diags) {
  std::vector<std::uint8_t> bytes;
  std::string err;
  if (!readWholeFile(path, bytes, err)) {
    diags.error(err);
    return false;
  }
  return loadProgram(bytes, path, opts, program, diags);
}

}  // namespace rt
}  // namespace kea
