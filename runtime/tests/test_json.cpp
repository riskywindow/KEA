// SPDX-License-Identifier: Apache-2.0
//
// The JSON reader/writer: malformed input, unicode escapes, deep nesting and
// numeric edge cases.

#include <string>

#include "kea/rt/json.h"
#include "test_util.h"

using kea::rt::Json;

namespace {

Json parseOk(const std::string& text) {
  Json j;
  std::string err;
  const bool ok = Json::parse(text, j, err);
  CHECK_MSG(ok, "  input: " + text + "\n  error: " + err);
  return j;
}

void expectError(const std::string& text, const std::string& needle) {
  Json j;
  std::string err;
  const bool ok = Json::parse(text, j, err);
  CHECK_MSG(!ok, "  expected a parse error for: " + text);
  if (!ok)
    CHECK_MSG(err.find(needle) != std::string::npos,
              "  input: " + text + "\n  wanted a message containing: " + needle +
                  "\n  got: " + err);
}

void testScalars() {
  CHECK(parseOk("null").isNull());
  CHECK(parseOk("true").asBool() == true);
  CHECK(parseOk("false").asBool() == false);
  CHECK(parseOk("  \n\t 42 ").isInt());
  CHECK(parseOk("42").asInt() == 42);
  CHECK(parseOk("-7").asInt() == -7);
  CHECK(parseOk("\"hi\"").asString() == "hi");
}

void testNumbers() {
  // Integral vs. fractional is a real distinction: a DRAM offset must not
  // become a double.
  CHECK(parseOk("1").isInt());
  CHECK(parseOk("1.0").isDouble());
  CHECK(parseOk("1e3").isDouble());
  CHECK(parseOk("-0").isInt());
  CHECK(parseOk("-0").asInt() == 0);
  CHECK(parseOk("9223372036854775807").asInt() == 9223372036854775807LL);
  // Overflowing int64 falls back to double rather than wrapping.
  CHECK(parseOk("9223372036854775808").isDouble());
  CHECK(parseOk("0.0078125").asDouble() == 0.0078125);
  CHECK(parseOk("1e-3").asDouble() == 1e-3);
  CHECK(parseOk("1E+3").asDouble() == 1000.0);

  expectError("01", "leading zeros");
  expectError("1.", "digit after the decimal point");
  expectError("1e", "digit in the exponent");
  expectError("+1", "unexpected character '+', expected a value");
  expectError(".5", "unexpected character '.', expected a value");
  expectError("-", "expected a number");

  // Doubles must round-trip through dump().
  const double values[] = {0.0078125, 0.023529412, 1e-30, 1.7976931348623157e308, -3.5};
  for (double v : values) {
    const std::string text = Json(v).dump(-1);
    Json back = parseOk(text);
    CHECK_MSG(back.asDouble() == v, "  round trip lost precision on " + text);
  }
  // An integral value keeps its integer spelling.
  CHECK(Json(static_cast<std::int64_t>(65536)).dump(-1) == "65536");
  // ... and a double keeps a decimal point so it stays a double.
  CHECK(Json(2.0).dump(-1) == "2.0");
}

void testStrings() {
  CHECK(parseOk("\"a\\nb\"").asString() == "a\nb");
  CHECK(parseOk("\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"").asString() == "\"\\/\b\f\n\r\t");
  // Unicode escapes -> UTF-8.
  CHECK(parseOk("\"\\u0041\"").asString() == "A");
  CHECK(parseOk("\"\\u00e9\"").asString() == "\xc3\xa9");
  CHECK(parseOk("\"\\u20ac\"").asString() == "\xe2\x82\xac");
  // Surrogate pair -> U+1F600.
  CHECK(parseOk("\"\\ud83d\\ude00\"").asString() == "\xf0\x9f\x98\x80");
  // Lone surrogates become U+FFFD rather than corrupting the output.
  CHECK(parseOk("\"\\ud83d\"").asString() == "\xef\xbf\xbd");
  CHECK(parseOk("\"\\udc00\"").asString() == "\xef\xbf\xbd");
  // Raw UTF-8 passes through untouched.
  CHECK(parseOk("\"caf\xc3\xa9\"").asString() == "caf\xc3\xa9");

  expectError("\"abc", "unterminated string");
  expectError("\"\\q\"", "unknown escape sequence");
  expectError("\"\\u00g0\"", "invalid hex digit");
  expectError("\"\\u00\"", "truncated \\u escape");
  // A raw newline inside a string is a control character, and saying so is
  // more useful than reporting the eventual "unterminated string".
  expectError("\"a\nb\"", "unescaped control character in a string");
  expectError("\"a\x01\"", "unescaped control character");

  // Control characters are escaped on output and survive a round trip.
  const std::string raw = std::string("tab\there\x01\x1f") + "end";
  Json back;
  std::string err;
  CHECK(Json::parse(Json(raw).dump(-1), back, err));
  CHECK(back.asString() == raw);
}

void testContainers() {
  Json a = parseOk("[1, 2, [3, {\"k\": null}]]");
  CHECK(a.isArray());
  CHECK(a.size() == 3);
  CHECK(a.at(0)->asInt() == 1);
  CHECK(a.at(2)->at(1)->find("k") != nullptr);
  CHECK(a.at(2)->at(1)->find("k")->isNull());
  CHECK(a.at(9) == nullptr);

  Json o = parseOk("{\"b\": 1, \"a\": 2}");
  CHECK(o.isObject());
  // Member order is preserved, which is what makes a re-dumped model map diff
  // cleanly against the original.
  CHECK(o.members()[0].first == "b");
  CHECK(o.members()[1].first == "a");
  CHECK(o.find("missing") == nullptr);

  CHECK(parseOk("[]").dump(-1) == "[]");
  CHECK(parseOk("{}").dump(-1) == "{}");

  expectError("{1: 2}", "expected a '\"'-quoted object key");
  expectError("{\"a\" 2}", "expected ':'");
  expectError("{\"a\": 2,}", "expected a '\"'-quoted object key");
  expectError("[1 2]", "expected ',' or ']'");
  expectError("[1,]", "unexpected character ']', expected a value");
  expectError("", "unexpected end of input");
  expectError("{} {}", "trailing content");
  expectError("tru", "expected 'true'");
}

void testNesting() {
  // 150 levels is fine; 300 must be refused rather than smashing the stack.
  auto nest = [](int depth) {
    return std::string(static_cast<std::size_t>(depth), '[') +
           std::string(static_cast<std::size_t>(depth), ']');
  };
  Json j;
  std::string err;
  CHECK(Json::parse(nest(150), j, err));
  expectError(nest(300), "maximum nesting depth exceeded");
  // Objects too.
  std::string deep;
  for (int i = 0; i < 300; ++i) deep += "{\"a\":";
  deep += "1";
  for (int i = 0; i < 300; ++i) deep += "}";
  expectError(deep, "maximum nesting depth exceeded");
}

void testErrorPositions() {
  Json j;
  std::string err;
  CHECK(!Json::parse("{\n  \"a\": 1,\n  \"b\": ?\n}", j, err));
  // Line 3 is `  "b": ?`, so the '?' is at 1-based column 8.
  CHECK_MSG(err.substr(0, 4) == "3:8:", "  got: " + err);
  CHECK_MSG(err.find("unexpected character '?'") != std::string::npos, "  got: " + err);
}

void testRoundTrip() {
  const std::string src =
      "{\"producer\":\"keac 0.1.0\",\"events\":[{\"id\":0,\"name\":\"a0\"}],"
      "\"peak\":{\"spm_a_bytes\":245760},\"scale\":0.0078125,\"neg\":-1,\"flag\":true,"
      "\"none\":null,\"empty_a\":[],\"empty_o\":{}}";
  Json a = parseOk(src);
  Json b = parseOk(a.dump(2));
  CHECK_STR_EQ(b.dump(2), a.dump(2));
  CHECK_STR_EQ(a.dump(-1), src);
}

void runTests() {
  testScalars();
  testNumbers();
  testStrings();
  testContainers();
  testNesting();
  testErrorPositions();
  testRoundTrip();
}

}  // namespace

TEST_MAIN("test_json")
