// SPDX-License-Identifier: Apache-2.0
//
// KEAF: read(write(p)) == p, CRC corruption detection, and the forward
// compatibility rule from ARTIFACT_FORMAT.md §4 --- an artifact containing a
// section type this loader has never heard of must still load.

#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "kea/rt/assembler.h"
#include "kea/rt/keaf_io.h"
#include "kea/rt/model_map.h"
#include "test_program_eq.h"
#include "test_util.h"

using namespace kea;
using namespace kea::rt;

namespace {

const char* kKasmDir = KEA_KASM_DIR;

std::string readFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  CHECK_MSG(static_cast<bool>(in), "  cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

/// A fully-populated program: code, constants, tensors, an SPM map and
/// metadata, so every section type gets exercised.
KeaProgram buildProgram() {
  Diagnostics diags;
  ModelMap map;
  const std::string map_path = std::string(kKasmDir) + "/double_buffered.map.json";
  const bool map_ok = loadModelMapFile(map_path, map, diags);
  CHECK_MSG(map_ok, "  " + diags.render());

  std::vector<std::uint8_t> consts(static_cast<std::size_t>(map.dram.const_bytes));
  for (std::size_t i = 0; i < consts.size(); ++i)
    consts[i] = static_cast<std::uint8_t>((i * 31u + 7u) & 0xFFu);

  AsmOptions opts;
  opts.map = &map;
  opts.const_data = &consts;

  KeaProgram p;
  const bool ok = assemble(readFile(std::string(kKasmDir) + "/double_buffered.kasm"),
                           "double_buffered.kasm", opts, p, diags);
  CHECK_MSG(ok, "  " + diags.render());
  return p;
}

void testRoundTrip() {
  const KeaProgram p = buildProgram();
  CHECK(!p.const_data.empty());
  CHECK(!p.tensors.empty());
  CHECK(!p.spm_map.empty());
  CHECK(!p.metadata_json.empty());

  std::vector<std::uint8_t> bytes;
  std::string err;
  CHECK_MSG(writeKeaf(p, bytes, err), "  " + err);

  // Structural expectations from ARTIFACT_FORMAT.md §2.
  CHECK(bytes.size() % 1 == 0);
  const KeafHeader* h = keafHeader(bytes.data());
  CHECK(h->file_bytes == bytes.size());
  CHECK(h->header_bytes == sizeof(KeafHeader));
  CHECK(h->isa_revision == KEA_ISA_REVISION);
  CHECK(h->entry_pc == p.entry_pc);
  CHECK(h->crc32 != 0);
  const KeafSection* sections = keafSections(bytes.data());
  for (std::uint32_t i = 0; i < h->section_count; ++i) {
    CHECK_MSG(sections[i].offset % KEAF_SECTION_ALIGN == 0,
              "  section payload is not 64-byte aligned");
    CHECK(sections[i].offset + sections[i].size <= bytes.size());
    if (sections[i].size)
      CHECK_MSG(keafCrc32(bytes.data() + sections[i].offset,
                          static_cast<std::size_t>(sections[i].size)) == sections[i].crc32,
                "  per-section CRC is wrong");
  }
  CHECK(keafFindSection(bytes.data(), KeafSectionType::CODE) != nullptr);
  CHECK(keafFindSection(bytes.data(), KeafSectionType::CONST) != nullptr);
  CHECK(keafFindSection(bytes.data(), KeafSectionType::DRAM_LAYOUT) != nullptr);
  CHECK(keafFindSection(bytes.data(), KeafSectionType::TENSORS) != nullptr);
  CHECK(keafFindSection(bytes.data(), KeafSectionType::SPM_MAP) != nullptr);
  CHECK(keafFindSection(bytes.data(), KeafSectionType::METADATA) != nullptr);
  CHECK((keafFindSection(bytes.data(), KeafSectionType::CONST)->flags & KEAF_SECF_STAGE_TO_DRAM) != 0);
  CHECK((keafFindSection(bytes.data(), KeafSectionType::SPM_MAP)->flags & KEAF_SECF_DEBUG_ONLY) != 0);

  KeaProgram back;
  CHECK_MSG(readKeaf(bytes.data(), bytes.size(), back, err), "  " + err);
  std::string why;
  CHECK_MSG(keatest::programsEqual(p, back, why), why);

  // Writing what we read must be byte-identical.
  std::vector<std::uint8_t> again;
  CHECK_MSG(writeKeaf(back, again, err), "  " + err);
  CHECK(again == bytes);
}

void testStripped() {
  const KeaProgram p = buildProgram();
  std::vector<std::uint8_t> bytes;
  std::string err;
  KeafWriteOptions opts;
  opts.include_spm_map = false;
  opts.include_metadata = false;
  CHECK_MSG(writeKeaf(p, bytes, err, opts), "  " + err);
  CHECK(keafFindSection(bytes.data(), KeafSectionType::SPM_MAP) == nullptr);
  CHECK(keafFindSection(bytes.data(), KeafSectionType::METADATA) == nullptr);
  CHECK((keafHeader(bytes.data())->flags & KEAF_FILEF_STRIPPED) != 0);

  KeaProgram back;
  CHECK_MSG(readKeaf(bytes.data(), bytes.size(), back, err), "  " + err);
  CHECK(back.spm_map.empty());
  CHECK(back.metadata_json.empty());
  // A stripped artifact executes identically: same code, same DRAM layout.
  CHECK(back.code.size() == p.code.size());
  CHECK(std::memcmp(&back.dram, &p.dram, sizeof(KeafDramLayout)) == 0);
}

void testNoCrc() {
  const KeaProgram p = buildProgram();
  std::vector<std::uint8_t> bytes;
  std::string err;
  KeafWriteOptions opts;
  opts.compute_crc = false;
  CHECK_MSG(writeKeaf(p, bytes, err, opts), "  " + err);
  CHECK(keafHeader(bytes.data())->crc32 == 0);
  // "not computed" is explicitly not an error (ARTIFACT_FORMAT.md §9).
  KeaProgram back;
  CHECK_MSG(readKeaf(bytes.data(), bytes.size(), back, err, true), "  " + err);
}

void testCrcCorruption() {
  const KeaProgram p = buildProgram();
  std::vector<std::uint8_t> bytes;
  std::string err;
  CHECK_MSG(writeKeaf(p, bytes, err), "  " + err);

  // Flip one bit inside the CODE payload.
  const KeafSection* code = keafFindSection(bytes.data(), KeafSectionType::CODE);
  const std::size_t victim = static_cast<std::size_t>(code->offset) + 8;
  std::vector<std::uint8_t> bad = bytes;
  bad[victim] ^= 0x01;

  KeaProgram back;
  CHECK_MSG(!readKeaf(bad.data(), bad.size(), back, err, /*check_crc=*/true),
            "  a corrupted artifact loaded without complaint");
  CHECK_MSG(err.find("CRC") != std::string::npos, "  got: " + err);

  // Without CRC checking the structural validation still passes, which is the
  // documented trade-off for skipping verification on a hot load path.
  CHECK_MSG(readKeaf(bad.data(), bad.size(), back, err, /*check_crc=*/false), "  " + err);

  // Corrupting the header is caught too.
  bad = bytes;
  bad[offsetof(KeafHeader, entry_pc)] ^= 0x80;
  CHECK(!readKeaf(bad.data(), bad.size(), back, err, true));

  // Truncation is caught by the file_bytes check even without CRCs.
  bad = bytes;
  bad.resize(bad.size() - 64);
  CHECK(!readKeaf(bad.data(), bad.size(), back, err, false));
  CHECK_MSG(err.find("file_bytes") != std::string::npos, "  got: " + err);

  // Bad magic.
  bad = bytes;
  bad[0] = 'X';
  CHECK(!readKeaf(bad.data(), bad.size(), back, err, false));
  CHECK_MSG(err.find("magic") != std::string::npos, "  got: " + err);
}

/// Rebuild `bytes` with one extra section of an unknown type appended.
std::vector<std::uint8_t> withUnknownSection(const std::vector<std::uint8_t>& bytes,
                                             std::uint32_t type) {
  const KeafHeader* h = keafHeader(bytes.data());
  const KeafSection* sections = keafSections(bytes.data());
  std::vector<KeafSection> table(sections, sections + h->section_count);

  // Everything before the section table is payload; keep it verbatim.
  std::vector<std::uint8_t> out(bytes.begin(),
                                bytes.begin() + static_cast<std::ptrdiff_t>(h->section_table_offset));

  const std::uint8_t payload[96] = {0xDE, 0xAD, 0xBE, 0xEF};
  while (out.size() % KEAF_SECTION_ALIGN) out.push_back(0);
  KeafSection extra{};
  extra.type = type;
  extra.flags = 0;
  extra.offset = out.size();
  extra.size = sizeof(payload);
  extra.entry_count = 0;
  extra.entry_size = 0;
  extra.crc32 = keafCrc32(payload, sizeof(payload));
  out.insert(out.end(), payload, payload + sizeof(payload));
  table.push_back(extra);

  while (out.size() % KEAF_SECTION_ALIGN) out.push_back(0);
  const std::uint64_t table_offset = out.size();
  const auto* tp = reinterpret_cast<const std::uint8_t*>(table.data());
  out.insert(out.end(), tp, tp + table.size() * sizeof(KeafSection));

  KeafHeader nh = *h;
  nh.section_count = static_cast<std::uint32_t>(table.size());
  nh.section_table_offset = table_offset;
  nh.file_bytes = out.size();
  nh.crc32 = 0;
  std::memcpy(out.data(), &nh, sizeof(nh));
  const std::uint32_t crc = keafFileCrc32(out.data(), out.size());
  std::memcpy(out.data() + offsetof(KeafHeader, crc32), &crc, sizeof(crc));
  return out;
}

void testForwardCompatibility() {
  const KeaProgram p = buildProgram();
  std::vector<std::uint8_t> bytes;
  std::string err;
  CHECK_MSG(writeKeaf(p, bytes, err), "  " + err);

  // A section type from a hypothetical future keac. §4: skip, do not reject.
  for (std::uint32_t type : {7u, 42u, 0xFFFFu}) {
    const std::vector<std::uint8_t> future = withUnknownSection(bytes, type);
    KeaProgram back;
    CHECK_MSG(readKeaf(future.data(), future.size(), back, err, true),
              "  an unknown section type " + std::to_string(type) +
                  " must be skipped, not rejected: " + err);
    std::string why;
    CHECK_MSG(keatest::programsEqual(p, back, why), why);
  }

  // A higher *minor* version must still load (§3).
  std::vector<std::uint8_t> newer = bytes;
  KeafHeader h = *keafHeader(newer.data());
  h.version_minor = KEAF_VERSION_MINOR + 7;
  h.crc32 = 0;
  std::memcpy(newer.data(), &h, sizeof(h));
  const std::uint32_t crc = keafFileCrc32(newer.data(), newer.size());
  std::memcpy(newer.data() + offsetof(KeafHeader, crc32), &crc, sizeof(crc));
  KeaProgram back;
  CHECK_MSG(readKeaf(newer.data(), newer.size(), back, err, true), "  " + err);

  // A different *major* version must be rejected.
  std::vector<std::uint8_t> incompatible = bytes;
  h = *keafHeader(incompatible.data());
  h.version_major = KEAF_VERSION_MAJOR + 1;
  h.crc32 = 0;
  std::memcpy(incompatible.data(), &h, sizeof(h));
  const std::uint32_t crc2 = keafFileCrc32(incompatible.data(), incompatible.size());
  std::memcpy(incompatible.data() + offsetof(KeafHeader, crc32), &crc2, sizeof(crc2));
  CHECK(!readKeaf(incompatible.data(), incompatible.size(), back, err, true));
  CHECK_MSG(err.find("major version") != std::string::npos, "  got: " + err);
}

void testCrcVector() {
  // ARTIFACT_FORMAT.md §9 pins the CRC-32 definition with a check value.
  CHECK(keafCrc32("123456789", 9) == 0xCBF43926u);
}

void testWriterRefusesBadPrograms() {
  KeaProgram p = buildProgram();
  std::vector<std::uint8_t> bytes;
  std::string err;

  // Last instruction not HALT.
  KeaProgram no_halt = p;
  no_halt.code.back() = keaMakeNop(Unit::VPU, 0);
  CHECK(!writeKeaf(no_halt, bytes, err));
  CHECK_MSG(err.find("HALT") != std::string::npos, "  got: " + err);

  // const_data disagreeing with DRAM_LAYOUT.
  KeaProgram bad_const = p;
  bad_const.const_data.pop_back();
  CHECK(!writeKeaf(bad_const, bytes, err));

  // Empty program.
  KeaProgram empty;
  empty.dram.alignment = KEA_DRAM_BASE_ALIGN;
  CHECK(!writeKeaf(empty, bytes, err));
}

void testModelMapRoundTrip() {
  Diagnostics diags;
  ModelMap map;
  CHECK(loadModelMapFile(std::string(kKasmDir) + "/all_opcodes.map.json", map, diags));
  const std::string text = dumpModelMap(map);
  ModelMap back;
  CHECK_MSG(parseModelMap(text, "<dumped>", back, diags), "  " + diags.render());
  CHECK(std::memcmp(&map.dram, &back.dram, sizeof(KeafDramLayout)) == 0);
  CHECK(map.symbols.size() == back.symbols.size());
  CHECK(map.tensors.size() == back.tensors.size());
  CHECK(map.spm_map.size() == back.spm_map.size());
  CHECK_STR_EQ(dumpModelMap(back), text);
  // Symbol lookup and coverage, which is what the assembler and disassembler use.
  CHECK(map.find("conv1.weights") != nullptr);
  CHECK(map.find("input") != nullptr);
  CHECK(map.find("nope") == nullptr);
  CHECK(map.covering(65536) != nullptr && map.covering(65536)->name == "input");
  CHECK(map.covering(300) != nullptr && map.covering(300)->name == "conv1.weights");
  CHECK(map.covering(4096) != nullptr && map.covering(4096)->name == "conv1.qparams");
  CHECK(map.covering(1000000) == nullptr);
}

void runTests() {
  testCrcVector();
  testRoundTrip();
  testStripped();
  testNoCrc();
  testCrcCorruption();
  testForwardCompatibility();
  testWriterRefusesBadPrograms();
  testModelMapRoundTrip();
}

}  // namespace

TEST_MAIN("test_keaf_io")
