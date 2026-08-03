// SPDX-License-Identifier: Apache-2.0
//
// kea/rt/json.h --- a very small, dependency-free JSON reader/writer.
//
// Exists because `model.map.json` and the KEAF METADATA section are JSON and
// the native half of KEA is not allowed to pull in a third-party library. It
// implements RFC 8259 with two deliberate restrictions:
//
//   * nesting is capped (KEA_JSON_MAX_DEPTH) so a hostile file cannot blow the
//     C++ stack, and
//   * numbers are kept as int64 when they are integral and representable,
//     otherwise as double. `1` and `1.0` therefore round-trip differently,
//     which is what a memory map wants.
//
// Object member order is preserved on both read and write, so a metadata blob
// survives a read/modify/write cycle textually recognisable.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace kea {
namespace rt {

/// Maximum array/object nesting accepted by the parser.
inline constexpr int KEA_JSON_MAX_DEPTH = 200;

class Json {
 public:
  enum class Type { Null, Bool, Int, Double, String, Array, Object };

  Json() = default;
  explicit Json(bool b) : type_(Type::Bool), b_(b) {}
  explicit Json(std::int64_t i) : type_(Type::Int), i_(i) {}
  explicit Json(int i) : type_(Type::Int), i_(i) {}
  explicit Json(double d) : type_(Type::Double), d_(d) {}
  explicit Json(std::string s) : type_(Type::String), s_(std::move(s)) {}
  explicit Json(const char* s) : type_(Type::String), s_(s) {}

  static Json array() {
    Json j;
    j.type_ = Type::Array;
    return j;
  }
  static Json object() {
    Json j;
    j.type_ = Type::Object;
    return j;
  }

  Type type() const { return type_; }
  bool isNull() const { return type_ == Type::Null; }
  bool isBool() const { return type_ == Type::Bool; }
  bool isInt() const { return type_ == Type::Int; }
  bool isDouble() const { return type_ == Type::Double; }
  bool isNumber() const { return type_ == Type::Int || type_ == Type::Double; }
  bool isString() const { return type_ == Type::String; }
  bool isArray() const { return type_ == Type::Array; }
  bool isObject() const { return type_ == Type::Object; }

  bool asBool(bool dflt = false) const { return isBool() ? b_ : dflt; }
  std::int64_t asInt(std::int64_t dflt = 0) const;
  double asDouble(double dflt = 0.0) const;
  const std::string& asString() const { return s_; }

  /// Array/object element access. Both return nullptr when absent, so callers
  /// never have to check the type first.
  std::size_t size() const;
  const Json* at(std::size_t i) const;
  const Json* find(const std::string& key) const;

  const std::vector<Json>& items() const { return arr_; }
  const std::vector<std::pair<std::string, Json>>& members() const { return obj_; }

  /// Mutation. `set` replaces an existing member in place, preserving order.
  void push_back(Json v);
  void set(const std::string& key, Json v);

  /// Serialize. `indent < 0` produces the compact single-line form.
  std::string dump(int indent = 2) const;

  /// Parse `text`. On failure returns a Null value and fills `error` with a
  /// `line:col: message` description.
  static bool parse(const std::string& text, Json& out, std::string& error);

 private:
  void dumpTo(std::string& out, int indent, int level) const;

  Type type_ = Type::Null;
  bool b_ = false;
  std::int64_t i_ = 0;
  double d_ = 0.0;
  std::string s_;
  std::vector<Json> arr_;
  std::vector<std::pair<std::string, Json>> obj_;
};

/// Escape `in` as a JSON string body (without the surrounding quotes).
std::string jsonEscape(const std::string& in);

}  // namespace rt
}  // namespace kea
