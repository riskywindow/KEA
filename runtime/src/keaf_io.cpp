// SPDX-License-Identifier: Apache-2.0
#include "kea/rt/keaf_io.h"

#include <cstring>
#include <fstream>

namespace kea {
namespace rt {
namespace {

void padTo(std::vector<std::uint8_t>& buf, std::size_t align) {
  while (buf.size() % align) buf.push_back(0);
}

void append(std::vector<std::uint8_t>& buf, const void* data, std::size_t n) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  buf.insert(buf.end(), p, p + n);
}

/// Add one section: pad, append the payload, record the table entry.
void addSection(std::vector<std::uint8_t>& buf, std::vector<KeafSection>& table,
                KeafSectionType type, std::uint32_t flags, const void* data, std::size_t size,
                std::uint32_t entry_count, std::uint32_t entry_size) {
  padTo(buf, KEAF_SECTION_ALIGN);
  KeafSection s{};
  s.type = static_cast<std::uint32_t>(type);
  s.flags = flags;
  s.offset = buf.size();
  s.size = size;
  s.entry_count = entry_count;
  s.entry_size = entry_size;
  s.crc32 = size ? keafCrc32(data, size) : keafCrc32("", 0);
  s.reserved = 0;
  if (size) append(buf, data, size);
  table.push_back(s);
}

void setName(char* field, std::size_t field_bytes, const std::string& name) {
  std::memset(field, 0, field_bytes);
  const std::size_t n = name.size() < field_bytes ? name.size() : field_bytes;
  std::memcpy(field, name.data(), n);
}

std::string getName(const char* field, std::size_t field_bytes) {
  std::size_t n = 0;
  while (n < field_bytes && field[n] != '\0') ++n;
  return std::string(field, n);
}

}  // namespace

bool writeKeaf(const KeaProgram& program, std::vector<std::uint8_t>& out, std::string& error,
               const KeafWriteOptions& opts) {
  error.clear();
  if (const char* e = program.validate()) {
    error = std::string("refusing to write an invalid program: ") + e;
    return false;
  }
  if (keaOpcode(program.code.back()) != Opcode::HALT) {
    error = "refusing to write a CODE section whose last instruction is not HALT";
    return false;
  }

  out.clear();
  out.resize(sizeof(KeafHeader), 0);
  std::vector<KeafSection> table;

  addSection(out, table, KeafSectionType::CODE, 0, program.code.data(),
             program.code.size() * sizeof(KeaInstr), static_cast<std::uint32_t>(program.code.size()),
             static_cast<std::uint32_t>(KEA_INSTR_BYTES));

  if (!program.const_data.empty()) {
    addSection(out, table, KeafSectionType::CONST, KEAF_SECF_STAGE_TO_DRAM,
               program.const_data.data(), program.const_data.size(), 0, 0);
  }

  addSection(out, table, KeafSectionType::DRAM_LAYOUT, 0, &program.dram, sizeof(KeafDramLayout), 0, 0);

  std::vector<KeafTensorEntry> tensors;
  tensors.reserve(program.tensors.size());
  for (const TensorBinding& t : program.tensors) {
    KeafTensorEntry e{};
    setName(e.name, KEAF_TENSOR_NAME_BYTES, t.name);
    e.dram_offset = t.dram_offset;
    e.size_bytes = t.size_bytes;
    if (t.shape.size() > KEAF_MAX_RANK) {
      error = "tensor '" + t.name + "' has rank " + std::to_string(t.shape.size()) +
              ", above KEAF_MAX_RANK";
      return false;
    }
    for (std::size_t i = 0; i < t.shape.size(); ++i) e.shape[i] = t.shape[i];
    e.rank = static_cast<std::uint8_t>(t.shape.size());
    e.dtype = static_cast<std::uint8_t>(t.dtype);
    e.kind = static_cast<std::uint8_t>(t.kind);
    e.layout = static_cast<std::uint8_t>(t.layout);
    e.scale = t.scale;
    e.zero_point = t.zero_point;
    e.index = t.index;
    tensors.push_back(e);
  }
  addSection(out, table, KeafSectionType::TENSORS, 0, tensors.data(),
             tensors.size() * sizeof(KeafTensorEntry), static_cast<std::uint32_t>(tensors.size()),
             static_cast<std::uint32_t>(sizeof(KeafTensorEntry)));

  const bool emit_spm = opts.include_spm_map && !program.spm_map.empty();
  if (emit_spm) {
    std::vector<KeafSpmEntry> entries;
    entries.reserve(program.spm_map.size());
    for (const SpmBinding& b : program.spm_map) {
      KeafSpmEntry e{};
      setName(e.name, KEAF_SPM_NAME_BYTES, b.name);
      e.space = static_cast<std::uint32_t>(b.space);
      e.offset = b.offset;
      e.size = b.size;
      e.first_pc = b.first_pc;
      e.last_pc = b.last_pc;
      entries.push_back(e);
    }
    addSection(out, table, KeafSectionType::SPM_MAP, KEAF_SECF_DEBUG_ONLY, entries.data(),
               entries.size() * sizeof(KeafSpmEntry), static_cast<std::uint32_t>(entries.size()),
               static_cast<std::uint32_t>(sizeof(KeafSpmEntry)));
  }

  const bool emit_md = opts.include_metadata && !program.metadata_json.empty();
  if (emit_md) {
    addSection(out, table, KeafSectionType::METADATA, KEAF_SECF_DEBUG_ONLY,
               program.metadata_json.data(), program.metadata_json.size(), 0, 0);
  }

  padTo(out, KEAF_SECTION_ALIGN);
  const std::uint64_t table_offset = out.size();
  append(out, table.data(), table.size() * sizeof(KeafSection));

  KeafHeader h{};
  h.magic[0] = KEAF_MAGIC0;
  h.magic[1] = KEAF_MAGIC1;
  h.magic[2] = KEAF_MAGIC2;
  h.magic[3] = KEAF_MAGIC3;
  h.version_major = KEAF_VERSION_MAJOR;
  h.version_minor = KEAF_VERSION_MINOR;
  h.header_bytes = sizeof(KeafHeader);
  h.file_bytes = out.size();
  h.section_count = static_cast<std::uint32_t>(table.size());
  h.section_table_offset = table_offset;
  h.isa_revision = KEA_ISA_REVISION;
  h.flags = (!emit_spm && !program.spm_map.empty()) ? KEAF_FILEF_STRIPPED : 0u;
  h.crc32 = 0;
  h.entry_pc = program.entry_pc;
  std::memcpy(out.data(), &h, sizeof(h));

  if (opts.compute_crc) {
    const std::uint32_t crc = keafFileCrc32(out.data(), out.size());
    std::memcpy(out.data() + offsetof(KeafHeader, crc32), &crc, sizeof(crc));
  }

  if (const char* e = keafValidate(out.data(), out.size(), opts.compute_crc)) {
    error = std::string("the writer produced an invalid artifact: ") + e;
    return false;
  }
  return true;
}

bool readKeaf(const void* data, std::size_t size, KeaProgram& program, std::string& error,
              bool check_crc) {
  error.clear();
  if (const char* e = keafValidate(data, size, check_crc)) {
    error = e;
    return false;
  }

  const auto* base = static_cast<const std::uint8_t*>(data);
  const KeafHeader* h = keafHeader(data);
  const KeafSection* sections = keafSections(data);

  program = KeaProgram{};
  program.entry_pc = h->entry_pc;

  bool const_seen = false;
  for (std::uint32_t i = 0; i < h->section_count; ++i) {
    const KeafSection& s = sections[i];
    const std::uint8_t* payload = base + s.offset;
    const bool zero_fill = (s.flags & KEAF_SECF_ZERO_FILL) != 0;

    // Per-section CRC. A zero CRC means "not computed" (ARTIFACT_FORMAT §9).
    if (check_crc && !zero_fill && s.crc32 != 0) {
      if (keafCrc32(payload, static_cast<std::size_t>(s.size)) != s.crc32) {
        error = std::string("CRC mismatch in the ") +
                keafSectionTypeName(static_cast<KeafSectionType>(s.type)) + " section";
        return false;
      }
    }

    switch (static_cast<KeafSectionType>(s.type)) {
      case KeafSectionType::CODE: {
        program.code.resize(s.entry_count);
        if (s.entry_count) std::memcpy(program.code.data(), payload, static_cast<std::size_t>(s.size));
        break;
      }
      case KeafSectionType::CONST: {
        const_seen = true;
        if (zero_fill) {
          program.const_data.assign(static_cast<std::size_t>(s.size), 0);
        } else {
          program.const_data.assign(payload, payload + s.size);
        }
        break;
      }
      case KeafSectionType::DRAM_LAYOUT:
        std::memcpy(&program.dram, payload, sizeof(KeafDramLayout));
        break;
      case KeafSectionType::TENSORS: {
        program.tensors.reserve(s.entry_count);
        for (std::uint32_t k = 0; k < s.entry_count; ++k) {
          KeafTensorEntry e{};
          std::memcpy(&e, payload + static_cast<std::size_t>(k) * sizeof(KeafTensorEntry),
                      sizeof(KeafTensorEntry));
          TensorBinding t;
          t.name = getName(e.name, KEAF_TENSOR_NAME_BYTES);
          t.dram_offset = e.dram_offset;
          t.size_bytes = e.size_bytes;
          if (e.rank > KEAF_MAX_RANK) {
            error = "tensor '" + t.name + "' declares a rank above KEAF_MAX_RANK";
            return false;
          }
          for (std::uint8_t d = 0; d < e.rank; ++d) t.shape.push_back(e.shape[d]);
          t.dtype = static_cast<KeafDType>(e.dtype);
          t.kind = static_cast<KeafTensorKind>(e.kind);
          t.layout = static_cast<KeafLayout>(e.layout);
          t.scale = e.scale;
          t.zero_point = e.zero_point;
          t.index = e.index;
          program.tensors.push_back(std::move(t));
        }
        break;
      }
      case KeafSectionType::SPM_MAP: {
        program.spm_map.reserve(s.entry_count);
        for (std::uint32_t k = 0; k < s.entry_count; ++k) {
          KeafSpmEntry e{};
          std::memcpy(&e, payload + static_cast<std::size_t>(k) * sizeof(KeafSpmEntry),
                      sizeof(KeafSpmEntry));
          SpmBinding b;
          b.name = getName(e.name, KEAF_SPM_NAME_BYTES);
          b.space = static_cast<KeafSpace>(e.space);
          b.offset = e.offset;
          b.size = e.size;
          b.first_pc = e.first_pc;
          b.last_pc = e.last_pc;
          program.spm_map.push_back(std::move(b));
        }
        break;
      }
      case KeafSectionType::METADATA:
        program.metadata_json.assign(reinterpret_cast<const char*>(payload),
                                     static_cast<std::size_t>(s.size));
        break;
      case KeafSectionType::NONE:
      default:
        // ARTIFACT_FORMAT.md §4: unknown section types must be skipped
        // silently, otherwise this loader cannot read artifacts from a newer
        // compiler. This `default` is that rule.
        break;
    }
  }

  if (!const_seen && program.dram.const_bytes != 0) {
    // No CONST payload but the layout reserves room for one: stage zeros so
    // KeaProgram::validate() and the DRAM arena agree.
    program.const_data.assign(static_cast<std::size_t>(program.dram.const_bytes), 0);
  }

  if (const char* e = program.validate()) {
    error = std::string("artifact loaded but failed KeaProgram::validate(): ") + e;
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------

bool readWholeFile(const std::string& path, std::vector<std::uint8_t>& out, std::string& error) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    error = "cannot open '" + path + "'";
    return false;
  }
  const std::streamsize n = in.tellg();
  in.seekg(0);
  out.resize(n > 0 ? static_cast<std::size_t>(n) : 0);
  if (n > 0 && !in.read(reinterpret_cast<char*>(out.data()), n)) {
    error = "short read on '" + path + "'";
    return false;
  }
  return true;
}

bool writeWholeFile(const std::string& path, const void* data, std::size_t size,
                    std::string& error) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    error = "cannot create '" + path + "'";
    return false;
  }
  if (size) out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  if (!out) {
    error = "short write on '" + path + "'";
    return false;
  }
  return true;
}

bool writeKeafFile(const KeaProgram& program, const std::string& path, std::string& error,
                   const KeafWriteOptions& opts) {
  std::vector<std::uint8_t> buf;
  if (!writeKeaf(program, buf, error, opts)) return false;
  return writeWholeFile(path, buf.data(), buf.size(), error);
}

bool readKeafFile(const std::string& path, KeaProgram& program, std::string& error,
                  bool check_crc) {
  std::vector<std::uint8_t> buf;
  if (!readWholeFile(path, buf, error)) return false;
  return readKeaf(buf.data(), buf.size(), program, error, check_crc);
}

}  // namespace rt
}  // namespace kea
