// SPDX-License-Identifier: Apache-2.0
#include "kea/rt/dram_arena.h"

#include <cstring>
#include <new>

namespace kea {
namespace rt {
namespace {

bool isPowerOfTwo(std::size_t v) { return v != 0 && (v & (v - 1)) == 0; }

}  // namespace

DramArena::~DramArena() { release(); }

void DramArena::release() noexcept {
  if (data_) {
    ::operator delete(static_cast<void*>(data_), std::align_val_t(align_));
    data_ = nullptr;
  }
  bytes_ = 0;
  program_ = nullptr;
}

bool DramArena::reset(const KeaProgram& program, std::string& error) {
  release();
  error.clear();

  const KeafDramLayout& l = program.dram;
  if (l.alignment < KEA_DRAM_BASE_ALIGN) {
    error = "DRAM alignment " + std::to_string(l.alignment) + " is below the architectural minimum " +
            std::to_string(KEA_DRAM_BASE_ALIGN);
    return false;
  }
  if (!isPowerOfTwo(l.alignment)) {
    error = "DRAM alignment " + std::to_string(l.alignment) + " is not a power of two";
    return false;
  }
  if (l.total_bytes > KEA_DRAM_BYTES) {
    error = "DRAM arena of " + std::to_string(l.total_bytes) + " bytes exceeds the 4 GiB address space";
    return false;
  }
  if (program.const_data.size() != l.const_bytes) {
    error = "the program's constant image is " + std::to_string(program.const_data.size()) +
            " bytes but DRAM_LAYOUT reserves " + std::to_string(l.const_bytes);
    return false;
  }
  if (l.const_offset + l.const_bytes > l.total_bytes) {
    error = "the CONST region escapes the DRAM arena";
    return false;
  }

  program_ = &program;
  align_ = l.alignment;
  bytes_ = static_cast<std::size_t>(l.total_bytes);
  if (bytes_ == 0) return true;

  void* p = ::operator new(bytes_, std::align_val_t(align_), std::nothrow);
  if (!p) {
    error = "failed to allocate a " + std::to_string(bytes_) + "-byte DRAM arena";
    bytes_ = 0;
    program_ = nullptr;
    return false;
  }
  data_ = static_cast<std::uint8_t*>(p);
  std::memset(data_, 0, bytes_);
  restageConstants();
  return true;
}

bool DramArena::restageConstants() noexcept {
  if (!program_) return false;
  const KeafDramLayout& l = program_->dram;
  if (l.const_bytes == 0) return true;
  if (!data_ || l.const_offset + l.const_bytes > bytes_) return false;
  std::memcpy(data_ + l.const_offset, program_->const_data.data(),
              static_cast<std::size_t>(l.const_bytes));
  return true;
}

std::uint8_t* DramArena::tensorData(const TensorBinding& t) noexcept {
  if (!data_) return nullptr;
  if (t.dram_offset + t.size_bytes > bytes_) return nullptr;
  return data_ + t.dram_offset;
}

const std::uint8_t* DramArena::tensorData(const TensorBinding& t) const noexcept {
  if (!data_) return nullptr;
  if (t.dram_offset + t.size_bytes > bytes_) return nullptr;
  return data_ + t.dram_offset;
}

bool DramArena::writeTensor(const TensorBinding& t, const void* src, std::size_t n) noexcept {
  if (n != t.size_bytes) return false;
  std::uint8_t* dst = tensorData(t);
  if (!dst) return false;
  std::memcpy(dst, src, n);
  return true;
}

bool DramArena::readTensor(const TensorBinding& t, void* dst, std::size_t n) const noexcept {
  if (n != t.size_bytes) return false;
  const std::uint8_t* src = tensorData(t);
  if (!src) return false;
  std::memcpy(dst, src, n);
  return true;
}

}  // namespace rt
}  // namespace kea
