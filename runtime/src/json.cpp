// SPDX-License-Identifier: Apache-2.0
#include "kea/rt/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace kea {
namespace rt {

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

std::int64_t Json::asInt(std::int64_t dflt) const {
  if (type_ == Type::Int) return i_;
  if (type_ == Type::Double) return static_cast<std::int64_t>(d_);
  return dflt;
}

double Json::asDouble(double dflt) const {
  if (type_ == Type::Double) return d_;
  if (type_ == Type::Int) return static_cast<double>(i_);
  return dflt;
}

std::size_t Json::size() const {
  if (type_ == Type::Array) return arr_.size();
  if (type_ == Type::Object) return obj_.size();
  return 0;
}

const Json* Json::at(std::size_t i) const {
  if (type_ != Type::Array || i >= arr_.size()) return nullptr;
  return &arr_[i];
}

const Json* Json::find(const std::string& key) const {
  if (type_ != Type::Object) return nullptr;
  for (const auto& kv : obj_)
    if (kv.first == key) return &kv.second;
  return nullptr;
}

void Json::push_back(Json v) {
  if (type_ != Type::Array) {
    type_ = Type::Array;
    arr_.clear();
  }
  arr_.push_back(std::move(v));
}

void Json::set(const std::string& key, Json v) {
  if (type_ != Type::Object) {
    type_ = Type::Object;
    obj_.clear();
  }
  for (auto& kv : obj_) {
    if (kv.first == key) {
      kv.second = std::move(v);
      return;
    }
  }
  obj_.emplace_back(key, std::move(v));
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

std::string jsonEscape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (unsigned char c : in) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

namespace {

/// Shortest representation that round-trips through strtod, with a decimal
/// point or exponent forced so the value stays a JSON double.
std::string formatDouble(double d) {
  if (std::isnan(d) || std::isinf(d)) return "null";
  char buf[40];
  for (int prec = 6; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof(buf), "%.*g", prec, d);
    if (std::strtod(buf, nullptr) == d) break;
  }
  std::string s(buf);
  if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
      s.find("inf") == std::string::npos && s.find("nan") == std::string::npos) {
    s += ".0";
  }
  return s;
}

void newlineIndent(std::string& out, int indent, int level) {
  if (indent < 0) return;
  out += '\n';
  out.append(static_cast<std::size_t>(indent * level), ' ');
}

}  // namespace

void Json::dumpTo(std::string& out, int indent, int level) const {
  switch (type_) {
    case Type::Null: out += "null"; return;
    case Type::Bool: out += b_ ? "true" : "false"; return;
    case Type::Int: out += std::to_string(i_); return;
    case Type::Double: out += formatDouble(d_); return;
    case Type::String:
      out += '"';
      out += jsonEscape(s_);
      out += '"';
      return;
    case Type::Array: {
      if (arr_.empty()) {
        out += "[]";
        return;
      }
      out += '[';
      for (std::size_t i = 0; i < arr_.size(); ++i) {
        if (i) out += ',';
        newlineIndent(out, indent, level + 1);
        arr_[i].dumpTo(out, indent, level + 1);
      }
      newlineIndent(out, indent, level);
      out += ']';
      return;
    }
    case Type::Object: {
      if (obj_.empty()) {
        out += "{}";
        return;
      }
      out += '{';
      for (std::size_t i = 0; i < obj_.size(); ++i) {
        if (i) out += ',';
        newlineIndent(out, indent, level + 1);
        out += '"';
        out += jsonEscape(obj_[i].first);
        out += '"';
        out += ':';
        if (indent >= 0) out += ' ';
        obj_[i].second.dumpTo(out, indent, level + 1);
      }
      newlineIndent(out, indent, level);
      out += '}';
      return;
    }
  }
}

std::string Json::dump(int indent) const {
  std::string out;
  dumpTo(out, indent, 0);
  return out;
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

namespace {

class Parser {
 public:
  Parser(const std::string& text) : s_(text) {}

  bool run(Json& out, std::string& error) {
    skipWs();
    if (!parseValue(out, 0)) {
      error = error_;
      return false;
    }
    skipWs();
    if (p_ != s_.size()) {
      fail("trailing content after the top-level JSON value");
      error = error_;
      return false;
    }
    return true;
  }

 private:
  const std::string& s_;
  std::size_t p_ = 0;
  std::string error_;

  bool eof() const { return p_ >= s_.size(); }
  char peek() const { return p_ < s_.size() ? s_[p_] : '\0'; }

  void skipWs() {
    while (p_ < s_.size()) {
      const char c = s_[p_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++p_;
      } else {
        break;
      }
    }
  }

  bool fail(const std::string& msg) {
    if (!error_.empty()) return false;
    int line = 1, col = 1;
    for (std::size_t i = 0; i < p_ && i < s_.size(); ++i) {
      if (s_[i] == '\n') {
        ++line;
        col = 1;
      } else {
        ++col;
      }
    }
    error_ = std::to_string(line) + ":" + std::to_string(col) + ": " + msg;
    return false;
  }

  bool literal(const char* lit) {
    const std::size_t n = std::strlen(lit);
    if (s_.compare(p_, n, lit) != 0) return false;
    p_ += n;
    return true;
  }

  bool parseValue(Json& out, int depth) {
    if (depth > KEA_JSON_MAX_DEPTH) return fail("maximum nesting depth exceeded");
    if (eof()) return fail("unexpected end of input, expected a value");
    switch (peek()) {
      case '{': return parseObject(out, depth);
      case '[': return parseArray(out, depth);
      case '"': {
        std::string str;
        if (!parseString(str)) return false;
        out = Json(std::move(str));
        return true;
      }
      case 't':
        if (!literal("true")) return fail("invalid literal, expected 'true'");
        out = Json(true);
        return true;
      case 'f':
        if (!literal("false")) return fail("invalid literal, expected 'false'");
        out = Json(false);
        return true;
      case 'n':
        if (!literal("null")) return fail("invalid literal, expected 'null'");
        out = Json();
        return true;
      default:
        if (peek() == '-' || (peek() >= '0' && peek() <= '9')) return parseNumber(out);
        return fail(std::string("unexpected character '") + peek() + "', expected a value");
    }
  }

  bool parseObject(Json& out, int depth) {
    ++p_;  // '{'
    out = Json::object();
    skipWs();
    if (peek() == '}') {
      ++p_;
      return true;
    }
    for (;;) {
      skipWs();
      if (peek() != '"') return fail("expected a '\"'-quoted object key");
      std::string key;
      if (!parseString(key)) return false;
      skipWs();
      if (peek() != ':') return fail("expected ':' after an object key");
      ++p_;
      skipWs();
      Json v;
      if (!parseValue(v, depth + 1)) return false;
      out.set(key, std::move(v));
      skipWs();
      if (peek() == ',') {
        ++p_;
        continue;
      }
      if (peek() == '}') {
        ++p_;
        return true;
      }
      return fail("expected ',' or '}' in an object");
    }
  }

  bool parseArray(Json& out, int depth) {
    ++p_;  // '['
    out = Json::array();
    skipWs();
    if (peek() == ']') {
      ++p_;
      return true;
    }
    for (;;) {
      skipWs();
      Json v;
      if (!parseValue(v, depth + 1)) return false;
      out.push_back(std::move(v));
      skipWs();
      if (peek() == ',') {
        ++p_;
        continue;
      }
      if (peek() == ']') {
        ++p_;
        return true;
      }
      return fail("expected ',' or ']' in an array");
    }
  }

  static void appendUtf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  bool parseHex4(std::uint32_t& v) {
    if (p_ + 4 > s_.size()) return fail("truncated \\u escape");
    v = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = s_[p_ + static_cast<std::size_t>(i)];
      std::uint32_t d;
      if (c >= '0' && c <= '9') d = static_cast<std::uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') d = static_cast<std::uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') d = static_cast<std::uint32_t>(c - 'A' + 10);
      else return fail("invalid hex digit in a \\u escape");
      v = (v << 4) | d;
    }
    p_ += 4;
    return true;
  }

  bool parseString(std::string& out) {
    ++p_;  // '"'
    out.clear();
    for (;;) {
      if (eof()) return fail("unterminated string");
      const unsigned char c = static_cast<unsigned char>(s_[p_]);
      if (c == '"') {
        ++p_;
        return true;
      }
      if (c < 0x20) return fail("unescaped control character in a string");
      if (c != '\\') {
        out += static_cast<char>(c);
        ++p_;
        continue;
      }
      ++p_;
      if (eof()) return fail("unterminated escape sequence");
      const char e = s_[p_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          std::uint32_t cp;
          if (!parseHex4(cp)) return false;
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (p_ + 1 < s_.size() && s_[p_] == '\\' && s_[p_ + 1] == 'u') {
              const std::size_t save = p_;
              p_ += 2;
              std::uint32_t lo;
              if (!parseHex4(lo)) return false;
              if (lo >= 0xDC00 && lo <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              } else {
                p_ = save;  // lone high surrogate; emit replacement below
                cp = 0xFFFD;
              }
            } else {
              cp = 0xFFFD;
            }
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            cp = 0xFFFD;  // lone low surrogate
          }
          appendUtf8(out, cp);
          break;
        }
        default: --p_; return fail("unknown escape sequence");
      }
    }
  }

  bool parseNumber(Json& out) {
    const std::size_t start = p_;
    if (peek() == '-') ++p_;
    if (eof() || peek() < '0' || peek() > '9') return fail("expected a number");
    if (peek() == '0') {
      ++p_;
      if (!eof() && peek() >= '0' && peek() <= '9')
        return fail("leading zeros are not allowed in a JSON number");
    } else {
      while (!eof() && peek() >= '0' && peek() <= '9') ++p_;
    }
    bool integral = true;
    if (peek() == '.') {
      integral = false;
      ++p_;
      if (eof() || peek() < '0' || peek() > '9')
        return fail("expected at least one digit after the decimal point");
      while (!eof() && peek() >= '0' && peek() <= '9') ++p_;
    }
    if (peek() == 'e' || peek() == 'E') {
      integral = false;
      ++p_;
      if (peek() == '+' || peek() == '-') ++p_;
      if (eof() || peek() < '0' || peek() > '9')
        return fail("expected at least one digit in the exponent");
      while (!eof() && peek() >= '0' && peek() <= '9') ++p_;
    }
    const std::string text = s_.substr(start, p_ - start);
    if (integral) {
      errno = 0;
      char* end = nullptr;
      const long long v = std::strtoll(text.c_str(), &end, 10);
      if (errno == 0 && end && *end == '\0') {
        out = Json(static_cast<std::int64_t>(v));
        return true;
      }
    }
    out = Json(std::strtod(text.c_str(), nullptr));
    return true;
  }
};

}  // namespace

bool Json::parse(const std::string& text, Json& out, std::string& error) {
  Parser p(text);
  out = Json();
  error.clear();
  return p.run(out, error);
}

}  // namespace rt
}  // namespace kea
