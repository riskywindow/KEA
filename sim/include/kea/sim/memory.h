// SPDX-License-Identifier: Apache-2.0
//
// kea/sim/memory.h --- the four KEA-1 address spaces as the simulator sees
// them.
//
// SPM_A, SPM_W and ACC carry per-element "has this ever been written"
// (poison) tracking, because ISA.md §2.3 declares them undefined at reset and
// MICROARCH.md §8.2 asks the simulator to trap reads of poison: an
// unsynchronized read of a DMA target is the single most likely compiler bug
// and it otherwise produces plausible-looking wrong numbers.
//
// DRAM is a flat 4 GiB space, so it is paged: 64 KiB pages allocated on first
// touch and zero filled.  DRAM is *not* poison tracked -- its contents are
// staged by the runtime from the KEAF CONST section plus the input tensors,
// so "never written" is not a bug there.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "kea/hw_config.h"

namespace kea {
namespace sim {

/// A byte-addressed scratchpad (SPM_A or SPM_W) with per-byte poison bits.
class Scratchpad {
 public:
  explicit Scratchpad(std::size_t bytes) : data_(bytes, 0), defined_(bytes, 0) {}

  std::size_t size() const { return data_.size(); }

  bool inRange(std::int64_t addr, std::int64_t len) const {
    return addr >= 0 && len >= 0 &&
           addr + len <= static_cast<std::int64_t>(data_.size());
  }

  std::uint8_t* data() { return data_.data(); }
  const std::uint8_t* data() const { return data_.data(); }

  std::int8_t readI8(std::int64_t addr) const {
    return static_cast<std::int8_t>(data_[static_cast<std::size_t>(addr)]);
  }
  std::uint8_t readU8(std::int64_t addr) const {
    return data_[static_cast<std::size_t>(addr)];
  }
  void write(std::int64_t addr, std::uint8_t v) {
    data_[static_cast<std::size_t>(addr)] = v;
    defined_[static_cast<std::size_t>(addr)] = 1;
  }

  /// True if every byte of [addr, addr+len) has been written since reset.
  bool defined(std::int64_t addr, std::int64_t len) const {
    const std::uint8_t* p = defined_.data() + addr;
    for (std::int64_t i = 0; i < len; ++i)
      if (!p[i]) return false;
    return true;
  }

  void markDefined(std::int64_t addr, std::int64_t len) {
    std::fill_n(defined_.begin() + static_cast<std::ptrdiff_t>(addr),
                static_cast<std::size_t>(len), std::uint8_t{1});
  }

  void reset() {
    std::fill(data_.begin(), data_.end(), std::uint8_t{0});
    std::fill(defined_.begin(), defined_.end(), std::uint8_t{0});
  }

 private:
  std::vector<std::uint8_t> data_;
  std::vector<std::uint8_t> defined_;
};

/// The accumulator scratchpad.  Addressed in int32 WORDS, never bytes --
/// ISA.md §2.2 is emphatic about this and it is the single most common way to
/// get KEA-1 addressing wrong.
class AccMem {
 public:
  AccMem() : data_(KEA_ACC_WORDS, 0), defined_(KEA_ACC_WORDS, 0) {}

  std::size_t words() const { return data_.size(); }

  bool inRange(std::int64_t word, std::int64_t count) const {
    return word >= 0 && count >= 0 &&
           word + count <= static_cast<std::int64_t>(data_.size());
  }

  std::int32_t read(std::int64_t word) const {
    return data_[static_cast<std::size_t>(word)];
  }
  void write(std::int64_t word, std::int32_t v) {
    data_[static_cast<std::size_t>(word)] = v;
    defined_[static_cast<std::size_t>(word)] = 1;
  }

  bool defined(std::int64_t word, std::int64_t count) const {
    const std::uint8_t* p = defined_.data() + word;
    for (std::int64_t i = 0; i < count; ++i)
      if (!p[i]) return false;
    return true;
  }

  void reset() {
    std::fill(data_.begin(), data_.end(), 0);
    std::fill(defined_.begin(), defined_.end(), std::uint8_t{0});
  }

 private:
  std::vector<std::int32_t> data_;
  std::vector<std::uint8_t> defined_;
};

/// Flat 4 GiB DRAM, paged so an unused arena costs nothing.  Pages are zero
/// filled on first touch.
class Dram {
 public:
  static constexpr unsigned kPageBits = 16;                   // 64 KiB
  static constexpr std::uint64_t kPageSize = 1ull << kPageBits;
  static constexpr std::size_t kNumPages =
      static_cast<std::size_t>(KEA_DRAM_BYTES >> kPageBits);  // 65536

  Dram() : pages_(kNumPages) {}

  static bool inRange(std::int64_t addr, std::int64_t len) {
    return addr >= 0 && len >= 0 &&
           static_cast<std::uint64_t>(addr) + static_cast<std::uint64_t>(len) <=
               KEA_DRAM_BYTES;
  }

  void read(std::int64_t addr, std::int64_t len, std::uint8_t* dst) const {
    copy(addr, len, dst, /*to_dram=*/false);
  }
  void write(std::int64_t addr, std::int64_t len, const std::uint8_t* src) {
    copy(addr, len, const_cast<std::uint8_t*>(src), /*to_dram=*/true);
  }

  void reset() {
    for (auto& p : pages_) p.reset();
  }

 private:
  std::uint8_t* page(std::size_t idx) const {
    if (!pages_[idx]) {
      pages_[idx].reset(new std::uint8_t[kPageSize]);
      std::fill_n(pages_[idx].get(), kPageSize, std::uint8_t{0});
    }
    return pages_[idx].get();
  }

  void copy(std::int64_t addr, std::int64_t len, std::uint8_t* host,
            bool to_dram) const {
    std::int64_t done = 0;
    while (done < len) {
      const std::uint64_t a = static_cast<std::uint64_t>(addr + done);
      const std::size_t pi = static_cast<std::size_t>(a >> kPageBits);
      const std::uint64_t off = a & (kPageSize - 1);
      const std::int64_t n = std::min<std::int64_t>(
          len - done, static_cast<std::int64_t>(kPageSize - off));
      std::uint8_t* p = page(pi) + off;
      if (to_dram)
        std::copy_n(host + done, n, p);
      else
        std::copy_n(p, n, host + done);
      done += n;
    }
  }

  mutable std::vector<std::unique_ptr<std::uint8_t[]>> pages_;
};

/// All four spaces plus the MXU's two weight banks, i.e. every piece of
/// architectural state the functional model touches.
struct Machine {
  Scratchpad spm_a{KEA_SPM_A_BYTES};
  Scratchpad spm_w{KEA_SPM_W_BYTES};
  AccMem acc;
  Dram dram;

  /// W[bank][k][n], already zero extended past k_rows / n_cols by LOAD_W.
  std::int8_t wbank[KEA_MXU_WEIGHT_BANKS][KEA_MXU_K][KEA_MXU_N]{};
  /// Valid extents of the resident tile, kept for utilisation accounting.
  std::uint8_t wbank_k_rows[KEA_MXU_WEIGHT_BANKS]{};
  std::uint8_t wbank_n_cols[KEA_MXU_WEIGHT_BANKS]{};
  bool wbank_loaded[KEA_MXU_WEIGHT_BANKS]{};

  void reset() {
    spm_a.reset();
    spm_w.reset();
    acc.reset();
    dram.reset();
    for (int b = 0; b < KEA_MXU_WEIGHT_BANKS; ++b) {
      for (int k = 0; k < KEA_MXU_K; ++k)
        for (int n = 0; n < KEA_MXU_N; ++n) wbank[b][k][n] = 0;
      wbank_k_rows[b] = 0;
      wbank_n_cols[b] = 0;
      wbank_loaded[b] = false;
    }
  }
};

}  // namespace sim
}  // namespace kea
