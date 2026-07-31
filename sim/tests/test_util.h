// SPDX-License-Identifier: Apache-2.0
//
// A three-macro test harness.  The simulator's tests are all "build a program,
// run it, compare against an independently written reference", which does not
// need a framework.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

namespace keatest {

inline int& failures() {
  static int n = 0;
  return n;
}

inline void report(const char* file, int line, const std::string& what) {
  std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, what.c_str());
  ++failures();
}

}  // namespace keatest

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) keatest::report(__FILE__, __LINE__, "CHECK(" #cond ")"); \
  } while (0)

#define CHECK_EQ(a, b)                                                     \
  do {                                                                     \
    const auto _a = (a);                                                   \
    const auto _b = (b);                                                   \
    if (!(_a == _b)) {                                                     \
      keatest::report(__FILE__, __LINE__,                                  \
                      std::string(#a " == " #b " (") +                     \
                          std::to_string(static_cast<long long>(_a)) +     \
                          " vs " +                                         \
                          std::to_string(static_cast<long long>(_b)) + ")"); \
    }                                                                      \
  } while (0)

#define CHECK_MSG(cond, msg)                                        \
  do {                                                              \
    if (!(cond))                                                    \
      keatest::report(__FILE__, __LINE__,                           \
                      std::string("CHECK(" #cond "): ") + (msg));   \
  } while (0)

#define TEST_MAIN_END()                                                   \
  do {                                                                    \
    if (keatest::failures()) {                                            \
      std::fprintf(stderr, "%d check(s) failed\n", keatest::failures());   \
      return 1;                                                           \
    }                                                                     \
    std::printf("all checks passed\n");                                   \
    return 0;                                                             \
  } while (0)
