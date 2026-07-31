// SPDX-License-Identifier: Apache-2.0
//
// kea/keaf.h --- KEAF: the KEA Executable Artifact Format (FROZEN CONTRACT)
//
// A KEAF file is what `keac` emits and `kea-rt` / `kea-sim` consume.  It is
// offset-based, mmap-friendly, little-endian, and entirely POD: a loader may
// mmap the file and cast pointers into it without any parsing pass.
//
// Header-only, C++17.  Depends only on <cstdint>, <cstddef>, and
// "kea/hw_config.h".
//
// See docs/ARTIFACT_FORMAT.md for the byte-level layout tables, the load
// procedure, and the METADATA JSON schema.

#pragma once

#include <cstddef>
#include <cstdint>

#include "kea/hw_config.h"

namespace kea {

// ===========================================================================
// 1. Constants
// ===========================================================================

/// File magic: the four ASCII bytes 'K','E','A','1'.
inline constexpr std::uint8_t KEAF_MAGIC0 = 'K';
inline constexpr std::uint8_t KEAF_MAGIC1 = 'E';
inline constexpr std::uint8_t KEAF_MAGIC2 = 'A';
inline constexpr std::uint8_t KEAF_MAGIC3 = '1';

/// Format version.  Major changes break loaders; minor changes only add
/// sections or trailing fields in reserved space.
inline constexpr std::uint16_t KEAF_VERSION_MAJOR = 1;
inline constexpr std::uint16_t KEAF_VERSION_MINOR = 0;

/// All section payloads begin at a file offset that is a multiple of this.
inline constexpr std::uint32_t KEAF_SECTION_ALIGN = 64;

/// Fixed-size name fields.
inline constexpr std::size_t KEAF_TENSOR_NAME_BYTES = 48;
inline constexpr std::size_t KEAF_SPM_NAME_BYTES = 40;

/// Maximum tensor rank recorded in a KeafTensorEntry.
inline constexpr std::size_t KEAF_MAX_RANK = 6;

/// Section type ids.
enum class KeafSectionType : std::uint32_t {
  NONE        = 0,
  CODE        = 1,  ///< instruction stream, entry_size == 32
  CONST       = 2,  ///< weight/constant blob, staged into DRAM at load
  DRAM_LAYOUT = 3,  ///< exactly one KeafDramLayout
  TENSORS     = 4,  ///< array of KeafTensorEntry (inputs, outputs, consts)
  SPM_MAP     = 5,  ///< array of KeafSpmEntry (debug only, may be absent)
  METADATA    = 6,  ///< raw UTF-8 JSON blob, not NUL-terminated
};

/// Section flag bits.
inline constexpr std::uint32_t KEAF_SECF_STAGE_TO_DRAM = 1u << 0;  ///< copy to DRAM at load
inline constexpr std::uint32_t KEAF_SECF_DEBUG_ONLY    = 1u << 1;  ///< safe to strip
inline constexpr std::uint32_t KEAF_SECF_ZERO_FILL     = 1u << 2;  ///< size bytes of zeros, no payload

/// File-level flag bits.
inline constexpr std::uint32_t KEAF_FILEF_STRIPPED = 1u << 0;  ///< debug sections removed

/// Tensor roles.
enum class KeafTensorKind : std::uint8_t {
  INPUT  = 0,
  OUTPUT = 1,
  CONST  = 2,
  SCRATCH = 3,
};

/// Datatypes recorded in the artifact.  Superset of the ISA datatypes because
/// I/O tensors may be presented to the host as uint8 or fp32.
enum class KeafDType : std::uint8_t {
  INT8  = 0,
  UINT8 = 1,
  INT4  = 2,
  INT32 = 3,
  FP32  = 4,
  INT16 = 5,
};

/// Logical tensor layout, for host-side interpretation only.
enum class KeafLayout : std::uint8_t {
  NHWC = 0,
  NCHW = 1,
  FLAT = 2,
};

/// Scratchpad space ids used by KeafSpmEntry.  Match kea::Space.
enum class KeafSpace : std::uint32_t {
  SPM_A = 0,
  SPM_W = 1,
  ACC   = 2,
};

// ===========================================================================
// 2. Structures
// ===========================================================================

#pragma pack(push, 1)

/// File header.  Always at offset 0.  Exactly 64 bytes.
struct KeafHeader {
  std::uint8_t magic[4];               ///< 'K','E','A','1'
  std::uint16_t version_major;         ///< KEAF_VERSION_MAJOR
  std::uint16_t version_minor;         ///< KEAF_VERSION_MINOR
  std::uint32_t header_bytes;          ///< sizeof(KeafHeader) == 64
  std::uint64_t file_bytes;            ///< total file size
  std::uint32_t section_count;
  std::uint64_t section_table_offset;  ///< byte offset of KeafSection[section_count]
  std::uint32_t isa_revision;          ///< KEA_ISA_REVISION the code targets
  std::uint32_t flags;                 ///< KEAF_FILEF_*
  std::uint32_t crc32;                 ///< CRC-32 of the whole file, this field zeroed
  std::uint32_t entry_pc;              ///< first instruction index, normally 0
  std::uint32_t reserved1[4];
};

/// Section table entry.  Exactly 40 bytes.
struct KeafSection {
  std::uint32_t type;         ///< KeafSectionType
  std::uint32_t flags;        ///< KEAF_SECF_*
  std::uint64_t offset;       ///< byte offset from file start, KEAF_SECTION_ALIGN aligned
  std::uint64_t size;         ///< payload size in bytes
  std::uint32_t entry_count;  ///< number of records (0 for opaque blobs)
  std::uint32_t entry_size;   ///< bytes per record (0 for opaque blobs)
  std::uint32_t crc32;        ///< CRC-32 of the payload
  std::uint32_t reserved;
};

/// The DRAM_LAYOUT section payload.  Exactly 64 bytes.
///
/// The runtime allocates one contiguous DRAM arena of `total_bytes` bytes,
/// aligned to `alignment`, and treats every DRAM address in the instruction
/// stream as an offset into it.  The three regions below are disjoint and
/// together cover [0, total_bytes).
struct KeafDramLayout {
  std::uint64_t total_bytes;     ///< arena size the runtime must provide
  std::uint64_t const_offset;    ///< where the CONST section is staged
  std::uint64_t const_bytes;
  std::uint64_t io_offset;       ///< input/output tensor arena
  std::uint64_t io_bytes;
  std::uint64_t scratch_offset;  ///< intermediate activation arena
  std::uint64_t scratch_bytes;
  std::uint32_t alignment;       ///< >= KEA_DRAM_BASE_ALIGN
  std::uint32_t reserved;
};

/// One entry of the TENSORS section.  Exactly 112 bytes.
struct KeafTensorEntry {
  char name[KEAF_TENSOR_NAME_BYTES];  ///< NUL-padded, not necessarily NUL-terminated
  std::uint64_t dram_offset;          ///< offset into the DRAM arena
  std::uint64_t size_bytes;           ///< bytes occupied
  std::int32_t shape[KEAF_MAX_RANK];  ///< dims[0..rank), remainder zero
  std::uint8_t rank;                  ///< 0..KEAF_MAX_RANK
  std::uint8_t dtype;                 ///< KeafDType
  std::uint8_t kind;                  ///< KeafTensorKind
  std::uint8_t layout;                ///< KeafLayout
  float scale;                        ///< quantization scale (per-tensor)
  std::int32_t zero_point;            ///< quantization zero point (per-tensor)
  std::uint32_t index;                ///< ordinal within its kind (input 0, 1, ...)
  std::uint32_t reserved[2];
};

/// One entry of the SPM_MAP section (debug).  Exactly 64 bytes.
struct KeafSpmEntry {
  char name[KEAF_SPM_NAME_BYTES];  ///< buffer name from the memory planner
  std::uint32_t space;             ///< KeafSpace
  std::uint32_t offset;            ///< bytes for SPM_A/SPM_W, int32 words for ACC
  std::uint32_t size;              ///< same units as offset
  std::uint32_t first_pc;          ///< first instruction index that touches it
  std::uint32_t last_pc;           ///< last instruction index that touches it
  std::uint32_t reserved;
};

#pragma pack(pop)

// ---------------------------------------------------------------------------
// Size guarantees
// ---------------------------------------------------------------------------

static_assert(sizeof(KeafHeader) == 64, "KeafHeader must be 64 bytes");
static_assert(sizeof(KeafSection) == 40, "KeafSection must be 40 bytes");
static_assert(sizeof(KeafDramLayout) == 64, "KeafDramLayout must be 64 bytes");
static_assert(sizeof(KeafTensorEntry) == 112, "KeafTensorEntry must be 112 bytes");
static_assert(sizeof(KeafSpmEntry) == 64, "KeafSpmEntry must be 64 bytes");
static_assert(sizeof(KeafSection) % 8 == 0, "section table must stay 8-byte pitched");
static_assert(sizeof(KeafTensorEntry) % 8 == 0, "tensor table must stay 8-byte pitched");
static_assert(offsetof(KeafHeader, section_table_offset) == 24, "KeafHeader layout drift");
static_assert(offsetof(KeafHeader, crc32) == 40, "KeafHeader layout drift");
static_assert(offsetof(KeafHeader, entry_pc) == 44, "KeafHeader layout drift");
static_assert(offsetof(KeafSection, offset) == 8, "KeafSection layout drift");
static_assert(offsetof(KeafTensorEntry, dram_offset) == 48, "KeafTensorEntry layout drift");
static_assert(offsetof(KeafTensorEntry, shape) == 64, "KeafTensorEntry layout drift");
static_assert(offsetof(KeafTensorEntry, scale) == 92, "KeafTensorEntry layout drift");
static_assert(sizeof(float) == 4, "KEAF assumes IEEE-754 binary32 float");

// ===========================================================================
// 3. Checksum
// ===========================================================================
//
// CRC-32 (IEEE 802.3), reflected, polynomial 0xEDB88320, init 0xFFFFFFFF,
// final XOR 0xFFFFFFFF.  Bitwise so the header stays dependency-free; the
// runtime may substitute a table-driven implementation with identical output.

inline std::uint32_t keafCrc32Update(std::uint32_t crc, const void* data, std::size_t n) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < n; ++i) {
    crc ^= p[i];
    for (int b = 0; b < 8; ++b) {
      crc = (crc >> 1) ^ (0xEDB88320u & (~(crc & 1u) + 1u));
    }
  }
  return crc;
}

inline std::uint32_t keafCrc32(const void* data, std::size_t n) {
  return keafCrc32Update(0xFFFFFFFFu, data, n) ^ 0xFFFFFFFFu;
}

/// CRC-32 of an entire KEAF image with KeafHeader::crc32 treated as zero.
/// `file` must point at `n` bytes starting with the header.
inline std::uint32_t keafFileCrc32(const void* file, std::size_t n) {
  if (n < sizeof(KeafHeader)) return 0;
  const auto* p = static_cast<const std::uint8_t*>(file);
  const std::size_t off = offsetof(KeafHeader, crc32);
  const std::uint32_t zero = 0;
  std::uint32_t crc = 0xFFFFFFFFu;
  crc = keafCrc32Update(crc, p, off);
  crc = keafCrc32Update(crc, &zero, sizeof(zero));
  crc = keafCrc32Update(crc, p + off + sizeof(std::uint32_t), n - off - sizeof(std::uint32_t));
  return crc ^ 0xFFFFFFFFu;
}

// ===========================================================================
// 4. Accessors
// ===========================================================================

inline bool keafHostIsLittleEndian() {
  const std::uint32_t x = 1;
  return *reinterpret_cast<const std::uint8_t*>(&x) == 1;
}

inline const KeafHeader* keafHeader(const void* file) {
  return static_cast<const KeafHeader*>(file);
}

inline const KeafSection* keafSections(const void* file) {
  const auto* p = static_cast<const std::uint8_t*>(file);
  const KeafHeader* h = keafHeader(file);
  return reinterpret_cast<const KeafSection*>(p + h->section_table_offset);
}

/// First section of the given type, or nullptr.
inline const KeafSection* keafFindSection(const void* file, KeafSectionType type) {
  const KeafHeader* h = keafHeader(file);
  const KeafSection* s = keafSections(file);
  for (std::uint32_t i = 0; i < h->section_count; ++i) {
    if (s[i].type == static_cast<std::uint32_t>(type)) return &s[i];
  }
  return nullptr;
}

/// Pointer to a section payload inside the mapped image.
inline const void* keafSectionData(const void* file, const KeafSection& s) {
  return static_cast<const std::uint8_t*>(file) + s.offset;
}

inline const char* keafSectionTypeName(KeafSectionType t) {
  switch (t) {
    case KeafSectionType::NONE:        return "NONE";
    case KeafSectionType::CODE:        return "CODE";
    case KeafSectionType::CONST:       return "CONST";
    case KeafSectionType::DRAM_LAYOUT: return "DRAM_LAYOUT";
    case KeafSectionType::TENSORS:     return "TENSORS";
    case KeafSectionType::SPM_MAP:     return "SPM_MAP";
    case KeafSectionType::METADATA:    return "METADATA";
  }
  return "<unknown>";
}

inline const char* keafDTypeName(KeafDType t) {
  switch (t) {
    case KeafDType::INT8:  return "int8";
    case KeafDType::UINT8: return "uint8";
    case KeafDType::INT4:  return "int4";
    case KeafDType::INT32: return "int32";
    case KeafDType::FP32:  return "fp32";
    case KeafDType::INT16: return "int16";
  }
  return "<unknown>";
}

/// Size in bits of one element of `t`.  int4 is 4 bits; everything else is a
/// whole number of bytes.
inline std::uint32_t keafDTypeBits(KeafDType t) {
  switch (t) {
    case KeafDType::INT4:  return 4;
    case KeafDType::INT8:
    case KeafDType::UINT8: return 8;
    case KeafDType::INT16: return 16;
    case KeafDType::INT32:
    case KeafDType::FP32:  return 32;
  }
  return 0;
}

inline const char* keafTensorKindName(KeafTensorKind k) {
  switch (k) {
    case KeafTensorKind::INPUT:   return "input";
    case KeafTensorKind::OUTPUT:  return "output";
    case KeafTensorKind::CONST:   return "const";
    case KeafTensorKind::SCRATCH: return "scratch";
  }
  return "<unknown>";
}

/// Compare a fixed-width NUL-padded name field against a C string.
inline bool keafNameEquals(const char* field, std::size_t field_bytes, const char* str) {
  std::size_t i = 0;
  for (; i < field_bytes; ++i) {
    const char c = str[i];
    if (c == '\0') break;
    if (field[i] != c) return false;
  }
  if (i == field_bytes) return str[i] == '\0';
  return field[i] == '\0';
}

// ===========================================================================
// 5. Structural validation
// ===========================================================================

/// Validate a mapped KEAF image of `n` bytes.  Returns nullptr when the file
/// is structurally sound, otherwise a static string naming the first problem.
///
/// `check_crc` is expensive on large artifacts; loaders should do it once at
/// install time and skip it on subsequent loads.
inline const char* keafValidate(const void* file, std::size_t n, bool check_crc = true) {
  if (!keafHostIsLittleEndian()) return "KEAF requires a little-endian host";
  if (n < sizeof(KeafHeader)) return "file smaller than header";

  const KeafHeader* h = keafHeader(file);
  if (h->magic[0] != KEAF_MAGIC0 || h->magic[1] != KEAF_MAGIC1 ||
      h->magic[2] != KEAF_MAGIC2 || h->magic[3] != KEAF_MAGIC3)
    return "bad magic";
  if (h->version_major != KEAF_VERSION_MAJOR) return "unsupported KEAF major version";
  if (h->header_bytes != sizeof(KeafHeader)) return "unexpected header size";
  if (h->file_bytes != n) return "file_bytes does not match mapping size";
  if (h->isa_revision != KEA_ISA_REVISION) return "artifact targets a different ISA revision";
  if (h->section_count == 0) return "no sections";

  const std::uint64_t tbl_end =
      h->section_table_offset + static_cast<std::uint64_t>(h->section_count) * sizeof(KeafSection);
  if (h->section_table_offset < sizeof(KeafHeader) || tbl_end > n)
    return "section table out of bounds";

  const KeafSection* s = keafSections(file);
  bool have_code = false, have_layout = false, have_tensors = false;
  for (std::uint32_t i = 0; i < h->section_count; ++i) {
    if (s[i].flags & KEAF_SECF_ZERO_FILL) {
      if (s[i].offset != 0) return "zero-fill section must have offset 0";
    } else {
      if (s[i].offset + s[i].size > n) return "section payload out of bounds";
      if (s[i].offset % KEAF_SECTION_ALIGN) return "section payload misaligned";
    }
    if (s[i].entry_size != 0 && s[i].size != static_cast<std::uint64_t>(s[i].entry_size) * s[i].entry_count)
      return "section size does not match entry_count * entry_size";

    switch (static_cast<KeafSectionType>(s[i].type)) {
      case KeafSectionType::CODE:
        if (s[i].entry_size != KEA_INSTR_BYTES) return "CODE entry_size must be 32";
        if (s[i].entry_count == 0) return "CODE section is empty";
        if (s[i].entry_count > KEA_MAX_INSTRUCTIONS) return "program exceeds IMEM";
        have_code = true;
        break;
      case KeafSectionType::DRAM_LAYOUT:
        if (s[i].size != sizeof(KeafDramLayout)) return "DRAM_LAYOUT has wrong size";
        have_layout = true;
        break;
      case KeafSectionType::TENSORS:
        if (s[i].entry_size != sizeof(KeafTensorEntry)) return "TENSORS entry_size mismatch";
        have_tensors = true;
        break;
      case KeafSectionType::SPM_MAP:
        if (s[i].entry_size != sizeof(KeafSpmEntry)) return "SPM_MAP entry_size mismatch";
        break;
      default:
        break;
    }
  }
  if (!have_code) return "missing CODE section";
  if (!have_layout) return "missing DRAM_LAYOUT section";
  if (!have_tensors) return "missing TENSORS section";

  const KeafSection* ls = keafFindSection(file, KeafSectionType::DRAM_LAYOUT);
  const auto* layout = static_cast<const KeafDramLayout*>(keafSectionData(file, *ls));
  if (layout->alignment < KEA_DRAM_BASE_ALIGN) return "DRAM alignment below architectural minimum";
  if (layout->const_offset + layout->const_bytes > layout->total_bytes)
    return "CONST region escapes the DRAM arena";
  if (layout->io_offset + layout->io_bytes > layout->total_bytes)
    return "IO region escapes the DRAM arena";
  if (layout->scratch_offset + layout->scratch_bytes > layout->total_bytes)
    return "SCRATCH region escapes the DRAM arena";
  if (layout->total_bytes > KEA_DRAM_BYTES) return "DRAM arena exceeds the address space";

  if (h->entry_pc >= keafFindSection(file, KeafSectionType::CODE)->entry_count)
    return "entry_pc outside the CODE section";

  if (check_crc && h->crc32 != 0 && keafFileCrc32(file, n) != h->crc32)
    return "file CRC mismatch";

  return nullptr;
}

}  // namespace kea
