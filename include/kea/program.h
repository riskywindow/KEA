#pragma once
// kea/program.h --- the in-memory form of a loaded KEA program.
//
// This is the interface contract between the three native components:
//
//   producers                             consumer
//   ---------                             --------
//   runtime/  KEAF reader   -----\
//   runtime/  .kasm assembler ----+------> sim/  the simulator
//   compiler backend (via .kasm) -/
//
// `KeaProgram` is deliberately a plain owning aggregate of std:: containers,
// *not* a view over an mmap'd file. A program is a few hundred KiB at most, it
// is loaded once per run, and having one owning type means the simulator has a
// single input shape regardless of whether the program came from a `.keaf`
// binary or from hand-written `.kasm` text. Zero-copy is not worth an extra
// lifetime rule here.
//
// The on-disk equivalents of these fields are specified in `keaf.h` and
// `docs/ARTIFACT_FORMAT.md`; the enums and semantics are shared, so a KEAF
// reader is a near-mechanical transcription into this struct.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kea/hw_config.h"
#include "kea/isa.h"
#include "kea/keaf.h"

namespace kea {

/// One input, output, or named constant tensor living in the DRAM arena.
/// Mirrors `KeafTensorEntry`, but with an owning name and shape.
struct TensorBinding {
  std::string name;
  std::uint64_t dram_offset = 0;  ///< offset into the DRAM arena
  std::uint64_t size_bytes = 0;
  std::vector<std::int32_t> shape;
  KeafDType dtype = KeafDType::INT8;
  KeafTensorKind kind = KeafTensorKind::INPUT;
  KeafLayout layout = KeafLayout::NHWC;
  float scale = 1.0f;            ///< per-tensor quantization scale
  std::int32_t zero_point = 0;   ///< per-tensor quantization zero point
  std::uint32_t index = 0;       ///< ordinal within its kind

  std::size_t elementCount() const {
    std::size_t n = 1;
    for (std::int32_t d : shape) n *= static_cast<std::size_t>(d < 0 ? 0 : d);
    return shape.empty() ? 0 : n;
  }
};

/// One statically planned scratchpad buffer. Debug information only: nothing at
/// run time consults this, because every address in the instruction stream is
/// already absolute. It exists so `kea-dis`, the simulator's trace, and the
/// memory-planner write-up can put names to addresses.
struct SpmBinding {
  std::string name;
  KeafSpace space = KeafSpace::SPM_A;
  std::uint32_t offset = 0;  ///< bytes for SPM_A/SPM_W, int32 words for ACC
  std::uint32_t size = 0;    ///< same units as offset
  std::uint32_t first_pc = 0;
  std::uint32_t last_pc = 0;
};

/// A fully loaded, ready-to-execute KEA program.
struct KeaProgram {
  /// Straight-line instruction stream. Index == PC. There are no branches, so
  /// this is also exactly the dispatch order (see ISA.md, Rule D).
  std::vector<KeaInstr> code;
  std::uint32_t entry_pc = 0;

  /// DRAM arena geometry. The runtime allocates `dram.total_bytes` once; every
  /// DRAM address appearing in `code` is an offset into that arena. There is no
  /// allocator at run time -- this layout was fixed by the compiler.
  KeafDramLayout dram{};

  /// Constant/weight image, staged into the arena at `dram.const_offset`
  /// before execution. Size must equal `dram.const_bytes`.
  std::vector<std::uint8_t> const_data;

  /// Inputs, outputs, and any named constants.
  std::vector<TensorBinding> tensors;

  /// Scratchpad allocation map (debug).
  std::vector<SpmBinding> spm_map;

  /// Free-form JSON from the artifact's METADATA section.
  std::string metadata_json;

  const TensorBinding* findTensor(const std::string& name) const {
    for (const auto& t : tensors)
      if (t.name == name) return &t;
    return nullptr;
  }

  std::vector<const TensorBinding*> byKind(KeafTensorKind k) const {
    std::vector<const TensorBinding*> out;
    for (const auto& t : tensors)
      if (t.kind == k) out.push_back(&t);
    return out;
  }

  std::vector<const TensorBinding*> inputs() const {
    return byKind(KeafTensorKind::INPUT);
  }
  std::vector<const TensorBinding*> outputs() const {
    return byKind(KeafTensorKind::OUTPUT);
  }

  /// Structural sanity check independent of `keafValidate`, so hand-written
  /// `.kasm` gets the same scrutiny a binary artifact does. Returns nullptr if
  /// the program is well formed, else a static description of the first fault.
  const char* validate() const {
    if (code.empty()) return "program has no instructions";
    if (code.size() > KEA_MAX_INSTRUCTIONS) return "program exceeds IMEM capacity";
    if (entry_pc >= code.size()) return "entry_pc is out of range";
    if (const_data.size() != dram.const_bytes) return "const_data size != dram.const_bytes";
    if (dram.const_offset + dram.const_bytes > dram.total_bytes)
      return "const region overflows the DRAM arena";
    if (dram.io_offset + dram.io_bytes > dram.total_bytes)
      return "io region overflows the DRAM arena";
    if (dram.scratch_offset + dram.scratch_bytes > dram.total_bytes)
      return "scratch region overflows the DRAM arena";
    for (const auto& t : tensors)
      if (t.dram_offset + t.size_bytes > dram.total_bytes)
        return "tensor overflows the DRAM arena";
    for (std::size_t i = 0; i < code.size(); ++i)
      if (const char* err = keaValidate(code[i])) return err;
    return nullptr;
  }
};

}  // namespace kea
