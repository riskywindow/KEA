// SPDX-License-Identifier: Apache-2.0
#include "kea/rt/diagnostics.h"

namespace kea {
namespace rt {

void Diagnostics::addSource(const std::string& file, const std::string& text) {
  for (auto& kv : sources_) {
    if (kv.first == file) {
      kv.second = text;
      return;
    }
  }
  sources_.emplace_back(file, text);
}

const std::string* Diagnostics::sourceFor(const std::string& file) const {
  for (const auto& kv : sources_)
    if (kv.first == file) return &kv.second;
  return nullptr;
}

Diagnostic& Diagnostics::error(const std::string& file, int line, int col, int length,
                               const std::string& message) {
  Diagnostic d;
  d.file = file;
  d.line = line;
  d.col = col;
  d.length = length < 1 ? 1 : length;
  d.message = message;
  d.is_error = true;
  diags_.push_back(std::move(d));
  ++error_count_;
  return diags_.back();
}

Diagnostic& Diagnostics::warning(const std::string& file, int line, int col, int length,
                                 const std::string& message) {
  Diagnostic d;
  d.file = file;
  d.line = line;
  d.col = col;
  d.length = length < 1 ? 1 : length;
  d.message = message;
  d.is_error = false;
  diags_.push_back(std::move(d));
  return diags_.back();
}

Diagnostic& Diagnostics::error(const std::string& message) {
  return error(std::string(), 0, 0, 1, message);
}

void Diagnostics::clear() {
  diags_.clear();
  error_count_ = 0;
}

std::string Diagnostics::render() const {
  std::string out;
  for (const Diagnostic& d : diags_) {
    if (d.line > 0) {
      out += d.file;
      out += ':';
      out += std::to_string(d.line);
      out += ':';
      out += std::to_string(d.col);
      out += ": ";
    } else if (!d.file.empty()) {
      out += d.file;
      out += ": ";
    }
    out += d.is_error ? "error: " : "warning: ";
    out += d.message;
    out += '\n';

    const std::string* src = sourceFor(d.file);
    if (src && d.line > 0) {
      // Extract the 1-based line `d.line`.
      std::size_t begin = 0;
      int cur = 1;
      while (cur < d.line && begin < src->size()) {
        const std::size_t nl = src->find('\n', begin);
        if (nl == std::string::npos) {
          begin = src->size();
          break;
        }
        begin = nl + 1;
        ++cur;
      }
      if (cur == d.line && begin <= src->size()) {
        std::size_t end = src->find('\n', begin);
        if (end == std::string::npos) end = src->size();
        const std::string text = src->substr(begin, end - begin);
        out += "  ";
        out += text;
        out += '\n';
        out += "  ";
        for (int i = 1; i < d.col && i <= static_cast<int>(text.size()) + 1; ++i)
          out += (i - 1 < static_cast<int>(text.size()) && text[static_cast<std::size_t>(i - 1)] == '\t')
                     ? '\t'
                     : ' ';
        out += '^';
        for (int i = 1; i < d.length; ++i) out += '~';
        out += '\n';
      }
    }
    for (const std::string& n : d.notes) {
      out += "note: ";
      out += n;
      out += '\n';
    }
  }
  return out;
}

bool Diagnostics::contains(const std::string& needle) const {
  for (const Diagnostic& d : diags_) {
    if (d.message.find(needle) != std::string::npos) return true;
    for (const std::string& n : d.notes)
      if (n.find(needle) != std::string::npos) return true;
  }
  return false;
}

}  // namespace rt
}  // namespace kea
