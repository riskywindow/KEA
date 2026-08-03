// SPDX-License-Identifier: Apache-2.0
//
// kea/rt/keaf_io.h --- KEAF artifact reader and writer.
//
// The writer follows docs/ARTIFACT_FORMAT.md §10 exactly: 64-byte header,
// 64-byte-aligned payloads, section table last, per-section CRC-32 and a
// whole-file CRC-32 computed with the header's own CRC field treated as zero.
//
// The reader follows §11 and, critically, §4: **unknown section types are
// skipped, not rejected**. That is what makes the format forward compatible,
// and `runtime/tests/test_keaf_io.cpp` asserts it.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kea/program.h"

namespace kea {
namespace rt {

struct KeafWriteOptions {
  /// Emit the SPM_MAP section (KEAF_SECF_DEBUG_ONLY). Dropping it sets
  /// KEAF_FILEF_STRIPPED.
  bool include_spm_map = true;
  /// Emit the METADATA section.
  bool include_metadata = true;
  /// Compute the whole-file CRC. `false` stores 0, which the format defines as
  /// "not computed".
  bool compute_crc = true;
};

bool writeKeaf(const KeaProgram& program, std::vector<std::uint8_t>& out, std::string& error,
               const KeafWriteOptions& opts = KeafWriteOptions{});
bool writeKeafFile(const KeaProgram& program, const std::string& path, std::string& error,
                   const KeafWriteOptions& opts = KeafWriteOptions{});

bool readKeaf(const void* data, std::size_t size, KeaProgram& program, std::string& error,
              bool check_crc = true);
bool readKeafFile(const std::string& path, KeaProgram& program, std::string& error,
                  bool check_crc = true);

/// Read a whole file into memory. Returns false and sets `error` on failure.
bool readWholeFile(const std::string& path, std::vector<std::uint8_t>& out, std::string& error);
bool writeWholeFile(const std::string& path, const void* data, std::size_t size, std::string& error);

}  // namespace rt
}  // namespace kea
