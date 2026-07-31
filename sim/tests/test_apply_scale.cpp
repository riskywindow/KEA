// SPDX-License-Identifier: Apache-2.0
//
// Requantization conformance.
//
//   1. Every one of the vectors in frontend/testdata/apply_scale_vectors.json
//      must reproduce exactly, including the shift > 31 double-round branch,
//      its sign-of-`value` behaviour, and the wrapping narrow to int32.
//
//   2. VQUANT is defined by ISA.md §10.1 in terms of `kea::keaRequantize`
//      (gemmlowp), which is a *different* algorithm.  The vector file's own
//      `gemmlowp_bruteforce` block claims the two are indistinguishable on the
//      domain KEA actually uses.  We do not take that on faith: the second
//      test re-derives it here over a deterministic sweep, so if the claim
//      ever stops holding the simulator's own tests say so.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "kea/isa.h"
#include "kea/sim/quant.h"
#include "test_util.h"

using namespace kea;
using namespace kea::sim;

namespace {

std::string readFile(const char* path) {
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return std::string();
  std::string s;
  char buf[65536];
  std::size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
  std::fclose(f);
  return s;
}

/// Minimal scanner for the `"cases": [[a,b,c,d,e], ...]` array.  Deliberately
/// not a general JSON parser -- runtime/ owns that, and sim/ must not depend
/// on runtime/.
bool parseCases(const std::string& j,
                std::vector<std::array<std::int64_t, 5>>& out) {
  const std::string key = "\"cases\":";
  std::size_t p = j.find(key);
  if (p == std::string::npos) return false;
  p += key.size();
  while (p < j.size() && (j[p] == ' ' || j[p] == '\n')) ++p;
  if (p >= j.size() || j[p] != '[') return false;
  ++p;
  while (p < j.size()) {
    while (p < j.size() && (j[p] == ' ' || j[p] == ',' || j[p] == '\n')) ++p;
    if (p < j.size() && j[p] == ']') return true;  // end of "cases"
    if (p >= j.size() || j[p] != '[') return false;
    ++p;
    std::array<std::int64_t, 5> row{};
    for (int i = 0; i < 5; ++i) {
      while (p < j.size() && (j[p] == ' ' || j[p] == ',')) ++p;
      bool neg = false;
      if (p < j.size() && j[p] == '-') {
        neg = true;
        ++p;
      }
      std::int64_t v = 0;
      if (p >= j.size() || j[p] < '0' || j[p] > '9') return false;
      while (p < j.size() && j[p] >= '0' && j[p] <= '9')
        v = v * 10 + (j[p++] - '0');
      row[static_cast<std::size_t>(i)] = neg ? -v : v;
    }
    while (p < j.size() && j[p] != ']') ++p;
    if (p >= j.size()) return false;
    ++p;  // consume ']'
    out.push_back(row);
  }
  return false;
}

void testVectors(const char* path) {
  const std::string j = readFile(path);
  CHECK_MSG(!j.empty(), std::string("cannot read ") + path);
  if (j.empty()) return;

  std::vector<std::array<std::int64_t, 5>> cases;
  CHECK_MSG(parseCases(j, cases), "could not parse the \"cases\" array");

  // The file declares how many vectors it holds; insist that we ran every one
  // of them, so a parser bug can never silently shrink the sweep.
  {
    const std::string key = "\"count\":";
    const std::size_t p = j.find(key);
    CHECK_MSG(p != std::string::npos, "no \"count\" field in the vector file");
    if (p != std::string::npos) {
      std::size_t q = p + key.size();
      while (q < j.size() && (j[q] == ' ' || j[q] == '\n')) ++q;
      std::size_t declared = 0;
      while (q < j.size() && j[q] >= '0' && j[q] <= '9')
        declared = declared * 10 + static_cast<std::size_t>(j[q++] - '0');
      CHECK_EQ(cases.size(), declared);
    }
  }
  CHECK_MSG(cases.size() > 1000,
            "expected thousands of vectors, got " +
                std::to_string(cases.size()));

  std::size_t bad = 0;
  std::size_t shift_gt_31 = 0, dr_true = 0;
  for (const auto& c : cases) {
    const std::int32_t value = static_cast<std::int32_t>(c[0]);
    const std::int32_t mult = static_cast<std::int32_t>(c[1]);
    const std::int32_t shift = static_cast<std::int32_t>(c[2]);
    const bool dr = c[3] != 0;
    const std::int32_t want = static_cast<std::int32_t>(c[4]);
    if (shift > 31) ++shift_gt_31;
    if (dr) ++dr_true;
    const std::int32_t got = applyScale32(value, mult, shift, dr);
    if (got != want) {
      if (bad < 10)
        std::fprintf(stderr,
                     "  apply_scale_32(%d, %d, %d, %d) = %d, expected %d\n",
                     value, mult, shift, dr ? 1 : 0, got, want);
      ++bad;
    }
  }
  CHECK_EQ(bad, std::size_t{0});
  CHECK(shift_gt_31 > 0);  // the double-round branch really is exercised
  CHECK(dr_true > 0);
  std::printf("apply_scale_32: %zu vectors, %zu with shift > 31, %zu with "
              "double_round\n",
              cases.size(), shift_gt_31, dr_true);
}

/// Deterministic 64-bit LCG.  No <random>, so the sweep is reproducible on
/// every platform and standard library.
struct Lcg {
  std::uint64_t s;
  std::uint64_t next() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return s >> 16;
  }
};

void testGemmlowpEquivalenceOnKeaDomain() {
  // The KEA domain: `mult` is a normalised Q31 multiplier in [2^30, 2^31) and
  // `KeaQuantParam::shift` is a post-multiply right shift, which in TOSA terms
  // is 31 + shift.  That is exactly the [31, 62] band the vector file's
  // brute-force sweep covers.
  Lcg rng{0x5EED1234ABCDEF01ull};
  std::size_t diff = 0, tested = 0;
  for (int trial = 0; trial < 400000; ++trial) {
    const std::int32_t mult = static_cast<std::int32_t>(
        (1u << 30) + (rng.next() % (1u << 30)));
    const std::int32_t kea_shift = static_cast<std::int32_t>(rng.next() % 32);
    const std::int32_t value = static_cast<std::int32_t>(
        static_cast<std::uint32_t>(rng.next()));
    const std::int32_t a =
        keaRequantize(value, mult, kea_shift, 0, INT32_MIN, INT32_MAX);
    const std::int32_t b =
        applyScale32(value, mult, keaShiftToTosaShift(kea_shift), true);
    ++tested;
    if (a != b) {
      if (diff < 10)
        std::fprintf(stderr,
                     "  divergence: value=%d mult=%d kea_shift=%d "
                     "gemmlowp=%d tosa=%d\n",
                     value, mult, kea_shift, a, b);
      ++diff;
    }
  }
  CHECK_MSG(diff == 0,
            "keaRequantize (gemmlowp, what VQUANT runs) diverged from "
            "apply_scale_32 (TOSA, the frontend's ground truth) on " +
                std::to_string(diff) + " of " + std::to_string(tested) +
                " cases in the normalised KEA domain");

  // ... and confirm double_round=false really is a different function, so the
  // test above is not vacuous.
  Lcg rng2{0x5EED1234ABCDEF01ull};
  std::size_t diff_no_dr = 0;
  for (int trial = 0; trial < 400000; ++trial) {
    const std::int32_t mult = static_cast<std::int32_t>(
        (1u << 30) + (rng2.next() % (1u << 30)));
    const std::int32_t kea_shift = static_cast<std::int32_t>(rng2.next() % 32);
    const std::int32_t value = static_cast<std::int32_t>(
        static_cast<std::uint32_t>(rng2.next()));
    if (keaRequantize(value, mult, kea_shift, 0, INT32_MIN, INT32_MAX) !=
        applyScale32(value, mult, keaShiftToTosaShift(kea_shift), false))
      ++diff_no_dr;
  }
  CHECK(diff_no_dr > 0);
  std::printf("gemmlowp vs apply_scale_32 on the KEA domain: %zu/%zu differ "
              "with double_round, %zu without\n",
              diff, tested, diff_no_dr);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <apply_scale_vectors.json>\n", argv[0]);
    return 2;
  }
  testVectors(argv[1]);
  testGemmlowpEquivalenceOnKeaDomain();
  TEST_MAIN_END();
}
