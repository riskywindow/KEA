// SPDX-License-Identifier: Apache-2.0
//
// kea/rt/dram_arena.h --- the load-time DRAM staging arena.
//
// KEA-1's thesis is that everything is decided at compile time, so the runtime
// is "a loader and a doorbell" (ARTIFACT_FORMAT.md §11). This class is the
// loader half:
//
//   reset()      allocates ONE aligned block of `dram.total_bytes`, copies the
//                CONST image to `dram.const_offset`, and zeroes the rest. This
//                is the only allocation the runtime ever performs.
//   everything   is `noexcept`, allocation-free, and operates on that block.
//   else
//
// `runtime/tests/test_no_alloc.cpp` replaces global operator new/delete with a
// counting version and asserts that the execution-phase API performs exactly
// zero heap operations. That test IS the guarantee; this comment is only the
// documentation of it.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "kea/program.h"

namespace kea {
namespace rt {

class DramArena {
 public:
  DramArena() = default;
  ~DramArena();
  DramArena(const DramArena&) = delete;
  DramArena& operator=(const DramArena&) = delete;

  /// Allocate the arena for `program` and stage its CONST image. `program`
  /// must outlive the arena. Idempotent: re-calling frees and reallocates.
  bool reset(const KeaProgram& program, std::string& error);

  /// Release the block. After this the arena is unusable until reset().
  void release() noexcept;

  // --- execution phase: no allocation past this line -----------------------

  bool ready() const noexcept { return data_ != nullptr || bytes_ == 0; }
  std::uint8_t* data() noexcept { return data_; }
  const std::uint8_t* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return bytes_; }
  std::size_t alignment() const noexcept { return align_; }
  const KeaProgram* program() const noexcept { return program_; }

  /// Pointer to a tensor's bytes inside the arena, or nullptr if the binding
  /// does not fit.
  std::uint8_t* tensorData(const TensorBinding& t) noexcept;
  const std::uint8_t* tensorData(const TensorBinding& t) const noexcept;

  /// Copy `n` bytes into / out of a tensor. `n` must equal `t.size_bytes`.
  bool writeTensor(const TensorBinding& t, const void* src, std::size_t n) noexcept;
  bool readTensor(const TensorBinding& t, void* dst, std::size_t n) const noexcept;

  /// Re-stage the CONST image without reallocating. Used between runs.
  bool restageConstants() noexcept;

 private:
  const KeaProgram* program_ = nullptr;
  std::uint8_t* data_ = nullptr;
  std::size_t bytes_ = 0;
  std::size_t align_ = KEA_DRAM_BASE_ALIGN;
};

}  // namespace rt
}  // namespace kea
