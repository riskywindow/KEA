// SPDX-License-Identifier: Apache-2.0
//
// A three-macro test harness. Each test file is its own executable and its own
// ctest entry, so there is no runner, no registration and no framework.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

namespace keatest {

inline int& failures() {
  static int n = 0;
  return n;
}

inline void report(const char* file, int line, const std::string& expr, const std::string& detail) {
  std::fprintf(stderr, "%s:%d: FAILED  %s\n", file, line, expr.c_str());
  if (!detail.empty()) std::fprintf(stderr, "%s\n", detail.c_str());
  ++failures();
}

/// Print two multi-line strings side by side up to the first difference.
inline std::string diffText(const std::string& got, const std::string& want) {
  std::size_t i = 0;
  while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
  std::size_t line_start = got.rfind('\n', i == 0 ? 0 : i - 1);
  line_start = (line_start == std::string::npos) ? 0 : line_start + 1;
  int line = 1;
  for (std::size_t k = 0; k < line_start; ++k)
    if (got[k] == '\n') ++line;
  auto lineAt = [](const std::string& s, std::size_t from) {
    const std::size_t end = s.find('\n', from);
    return s.substr(from, (end == std::string::npos ? s.size() : end) - from);
  };
  std::string out = "  first difference at line " + std::to_string(line) + ", byte " +
                    std::to_string(i) + "\n";
  out += "  got : " + (line_start < got.size() ? lineAt(got, line_start) : std::string("<eof>")) + "\n";
  out += "  want: " + (line_start < want.size() ? lineAt(want, line_start) : std::string("<eof>")) + "\n";
  return out;
}

inline int finish(const char* name) {
  if (failures() == 0) {
    std::fprintf(stderr, "%s: OK\n", name);
    return 0;
  }
  std::fprintf(stderr, "%s: %d FAILURE(S)\n", name, failures());
  return 1;
}

}  // namespace keatest

#define CHECK(cond)                                                             \
  do {                                                                          \
    if (!(cond)) keatest::report(__FILE__, __LINE__, #cond, std::string());      \
  } while (0)

#define CHECK_MSG(cond, detail)                                                 \
  do {                                                                          \
    if (!(cond)) keatest::report(__FILE__, __LINE__, #cond, (detail));           \
  } while (0)

#define CHECK_EQ(a, b)                                                          \
  do {                                                                          \
    const auto& _a = (a);                                                       \
    const auto& _b = (b);                                                       \
    if (!(_a == _b))                                                            \
      keatest::report(__FILE__, __LINE__, #a " == " #b,                          \
                      "  got : " + std::to_string(_a) + "\n  want: " + std::to_string(_b)); \
  } while (0)

#define CHECK_STR_EQ(a, b)                                                      \
  do {                                                                          \
    const std::string _a = (a);                                                 \
    const std::string _b = (b);                                                 \
    if (_a != _b)                                                               \
      keatest::report(__FILE__, __LINE__, #a " == " #b, keatest::diffText(_a, _b)); \
  } while (0)

#define TEST_MAIN(name)                                                         \
  int main() {                                                                  \
    runTests();                                                                 \
    return keatest::finish(name);                                               \
  }
