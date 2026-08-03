// SPDX-License-Identifier: Apache-2.0
//
// "No allocation during execution" is the project's thesis, so it gets a test
// rather than a comment.
//
// This translation unit replaces the global operator new / operator delete
// with counting versions. `DramArena::reset()` is allowed to allocate exactly
// once; everything the runtime does afterwards --- staging inputs, restaging
// constants, handing the machine a pointer, extracting outputs --- must do
// zero heap operations.

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include "kea/rt/assembler.h"
#include "kea/rt/dram_arena.h"
#include "kea/rt/model_map.h"
#include "test_util.h"

// ---------------------------------------------------------------------------
// Counting allocator
// ---------------------------------------------------------------------------

namespace {
std::size_t g_new_count = 0;
std::size_t g_delete_count = 0;
bool g_counting = false;

void* alignedAlloc(std::size_t n, std::size_t align) {
  if (align < sizeof(void*)) align = sizeof(void*);
  void* p = nullptr;
  if (posix_memalign(&p, align, n ? n : 1) != 0) return nullptr;
  return p;
}
}  // namespace

void* operator new(std::size_t n) {
  if (g_counting) ++g_new_count;
  void* p = std::malloc(n ? n : 1);
  if (!p) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept {
  if (p && g_counting) ++g_delete_count;
  std::free(p);
}
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }

void* operator new(std::size_t n, std::align_val_t a) {
  if (g_counting) ++g_new_count;
  void* p = alignedAlloc(n, static_cast<std::size_t>(a));
  if (!p) throw std::bad_alloc();
  return p;
}
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  if (g_counting) ++g_new_count;
  return alignedAlloc(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void* p, std::align_val_t) noexcept {
  if (p && g_counting) ++g_delete_count;
  std::free(p);
}
void operator delete[](void* p, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete(void* p, std::size_t, std::align_val_t a) noexcept {
  ::operator delete(p, a);
}
void operator delete[](void* p, std::size_t, std::align_val_t a) noexcept {
  ::operator delete(p, a);
}

// ---------------------------------------------------------------------------

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

KeaProgram buildProgram(std::vector<std::uint8_t>& consts, ModelMap& map) {
  Diagnostics diags;
  const bool map_ok = loadModelMapFile(std::string(kKasmDir) + "/double_buffered.map.json", map, diags);
  CHECK_MSG(map_ok, "  " + diags.render());
  consts.assign(static_cast<std::size_t>(map.dram.const_bytes), 0);
  for (std::size_t i = 0; i < consts.size(); ++i)
    consts[i] = static_cast<std::uint8_t>((i * 7u + 3u) & 0xFFu);

  AsmOptions opts;
  opts.map = &map;
  opts.const_data = &consts;
  KeaProgram p;
  const bool ok = assemble(readFile(std::string(kKasmDir) + "/double_buffered.kasm"),
                           "double_buffered.kasm", opts, p, diags);
  CHECK_MSG(ok, "  " + diags.render());
  return p;
}

void testStaging() {
  std::vector<std::uint8_t> consts;
  ModelMap map;
  const KeaProgram p = buildProgram(consts, map);

  DramArena arena;
  std::string err;
  CHECK_MSG(arena.reset(p, err), "  " + err);
  CHECK(arena.size() == p.dram.total_bytes);
  CHECK(arena.data() != nullptr);
  CHECK_MSG(reinterpret_cast<std::uintptr_t>(arena.data()) % p.dram.alignment == 0,
            "  the arena is not aligned to KeafDramLayout::alignment");

  // CONST is staged at const_offset, and nothing outside it was touched.
  CHECK(std::memcmp(arena.data() + p.dram.const_offset, p.const_data.data(),
                    p.const_data.size()) == 0);
  for (std::uint64_t i = p.dram.const_offset + p.dram.const_bytes; i < p.dram.io_offset; ++i)
    if (arena.data()[i] != 0) {
      CHECK_MSG(false, "  the arena was not zeroed outside the CONST region");
      break;
    }
}

void testNoAllocationInTheExecutionPath() {
  std::vector<std::uint8_t> consts;
  ModelMap map;
  const KeaProgram p = buildProgram(consts, map);

  DramArena arena;
  std::string err;
  CHECK_MSG(arena.reset(p, err), "  " + err);

  const TensorBinding* in = p.findTensor("input");
  const TensorBinding* out = p.findTensor("output");
  CHECK(in != nullptr && out != nullptr);
  if (!in || !out) return;

  // Host-side staging buffers, allocated up front like a real embedded host.
  std::vector<std::uint8_t> host_in(static_cast<std::size_t>(in->size_bytes));
  std::vector<std::uint8_t> host_out(static_cast<std::size_t>(out->size_bytes));
  for (std::size_t i = 0; i < host_in.size(); ++i)
    host_in[i] = static_cast<std::uint8_t>(i & 0x7F);

  // ---- everything past this line must be allocation free -------------------
  g_new_count = 0;
  g_delete_count = 0;
  g_counting = true;

  const bool w = arena.writeTensor(*in, host_in.data(), host_in.size());
  std::uint8_t* raw = arena.tensorData(*in);
  const bool restaged = arena.restageConstants();
  const bool r = arena.readTensor(*out, host_out.data(), host_out.size());
  const std::size_t sz = arena.size();
  const std::uint8_t* base = arena.data();
  const bool bad_size = arena.writeTensor(*in, host_in.data(), host_in.size() - 1);

  g_counting = false;
  // --------------------------------------------------------------------------

  CHECK_MSG(g_new_count == 0,
            "  the execution path performed " + std::to_string(g_new_count) + " allocation(s)");
  CHECK_MSG(g_delete_count == 0,
            "  the execution path performed " + std::to_string(g_delete_count) + " free(s)");
  CHECK(w);
  CHECK(r);
  CHECK(restaged);
  CHECK(!bad_size);
  CHECK(raw == base + in->dram_offset);
  CHECK(sz == p.dram.total_bytes);

  // The bytes really landed where the instruction stream expects them.
  CHECK(std::memcmp(arena.data() + in->dram_offset, host_in.data(), host_in.size()) == 0);
}

void testRejections() {
  std::vector<std::uint8_t> consts;
  ModelMap map;
  KeaProgram p = buildProgram(consts, map);
  DramArena arena;
  std::string err;

  KeaProgram bad = p;
  bad.dram.alignment = 8;  // below KEA_DRAM_BASE_ALIGN
  CHECK(!arena.reset(bad, err));
  CHECK_MSG(err.find("below the architectural minimum") != std::string::npos, "  got: " + err);

  bad = p;
  bad.dram.alignment = 96;  // not a power of two
  CHECK(!arena.reset(bad, err));
  CHECK_MSG(err.find("power of two") != std::string::npos, "  got: " + err);

  bad = p;
  bad.const_data.pop_back();
  CHECK(!arena.reset(bad, err));
  CHECK_MSG(err.find("constant image") != std::string::npos, "  got: " + err);

  // reset() is idempotent: a second call frees and reallocates cleanly.
  CHECK_MSG(arena.reset(p, err), "  " + err);
  CHECK_MSG(arena.reset(p, err), "  " + err);
  arena.release();
  CHECK(arena.data() == nullptr);
}

void runTests() {
  testStaging();
  testNoAllocationInTheExecutionPath();
  testRejections();
}

}  // namespace

TEST_MAIN("test_no_alloc")
