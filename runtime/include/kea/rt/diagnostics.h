// SPDX-License-Identifier: Apache-2.0
//
// kea/rt/diagnostics.h --- source-located error reporting for `.kasm` and
// `.map.json`.
//
// The compiler backend will be debugged through these messages, so they carry
// a file, a 1-based line and column, the width of the offending token, an
// explanation of what was expected, and optional notes. `Diagnostics::render()`
// produces the familiar caret form:
//
//   conv1.kasm:12:26: error: 'acc_addr' must be a multiple of 16 int32 words
//                            (ACC is word addressed), got 8
//     MXU   MATMUL  a_addr=a:0, acc_addr=acc:8, ...
//                            ^~~~~
//   note: ACC addresses count int32 words; byte address 32 is acc:8

#pragma once

#include <string>
#include <vector>

namespace kea {
namespace rt {

struct Diagnostic {
  std::string file;
  int line = 0;    ///< 1-based; 0 means "no source location"
  int col = 0;     ///< 1-based
  int length = 1;  ///< characters underlined by the caret run
  std::string message;
  std::vector<std::string> notes;
  bool is_error = true;
};

/// Collects diagnostics and renders them against the source text they refer to.
class Diagnostics {
 public:
  /// Register the text of a file so `render()` can quote the offending line.
  void addSource(const std::string& file, const std::string& text);

  Diagnostic& error(const std::string& file, int line, int col, int length,
                    const std::string& message);
  Diagnostic& warning(const std::string& file, int line, int col, int length,
                      const std::string& message);
  /// A diagnostic with no source location (bad command line, I/O failure...).
  Diagnostic& error(const std::string& message);

  bool hasErrors() const { return error_count_ > 0; }
  std::size_t errorCount() const { return error_count_; }
  const std::vector<Diagnostic>& all() const { return diags_; }
  void clear();

  /// Human-readable report, one block per diagnostic, newline terminated.
  std::string render() const;

  /// True if any diagnostic message contains `needle`. Used by the tests.
  bool contains(const std::string& needle) const;

 private:
  const std::string* sourceFor(const std::string& file) const;

  std::vector<Diagnostic> diags_;
  std::vector<std::pair<std::string, std::string>> sources_;
  std::size_t error_count_ = 0;
};

}  // namespace rt
}  // namespace kea
