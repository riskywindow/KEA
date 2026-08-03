// SPDX-License-Identifier: Apache-2.0
#include "kea/rt/assembler.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

#include "kea/rt/json.h"
#include "kea/rt/op_fields.h"

namespace kea {
namespace rt {
namespace {

// ===========================================================================
// Lexer
// ===========================================================================

struct Token {
  enum class Kind { End, Newline, Ident, Number, String, Directive, Symbol, Punct };
  Kind kind = Kind::End;
  std::string text;      ///< identifier / directive name / symbol name / literal digits
  std::int64_t num = 0;  ///< Number only
  char suffix = 0;       ///< Number only: 0, 'b' (bytes) or 'w' (int32 words)
  int line = 1;
  int col = 1;
  int length = 1;
};

bool isIdentStart(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
bool isIdentChar(char c) {
  return isIdentStart(c) || (c >= '0' && c <= '9');
}
bool isSymbolChar(char c) {
  return isIdentChar(c) || c == '.' || c == '/';
}
bool isDigit(char c) { return c >= '0' && c <= '9'; }

class Lexer {
 public:
  Lexer(const std::string& src, const std::string& file, Diagnostics& d)
      : s_(src), file_(file), diags_(d) {}

  bool run(std::vector<Token>& out) {
    toks_ = &out;
    bool ok = true;
    for (;;) {
      skipSpace();
      if (p_ >= s_.size()) break;
      const char c = s_[p_];
      if (c == '\n') {
        push(Token::Kind::Newline, "\\n", 1);
        ++p_;
        ++line_;
        bol_ = p_;
        continue;
      }
      if (c == ';' || c == '#') {
        while (p_ < s_.size() && s_[p_] != '\n') ++p_;
        continue;
      }
      if (c == '.' && p_ + 1 < s_.size() && isIdentStart(s_[p_ + 1])) {
        const std::size_t b = p_;
        ++p_;
        while (p_ < s_.size() && isIdentChar(s_[p_])) ++p_;
        pushAt(Token::Kind::Directive, s_.substr(b + 1, p_ - b - 1), b, p_ - b);
        continue;
      }
      if (c == '@') {
        const std::size_t b = p_;
        ++p_;
        while (p_ < s_.size() && isSymbolChar(s_[p_])) ++p_;
        if (p_ == b + 1) {
          errAt(b, 1, "expected a DRAM symbol name after '@'");
          ok = false;
          ++p_;
          continue;
        }
        pushAt(Token::Kind::Symbol, s_.substr(b + 1, p_ - b - 1), b, p_ - b);
        continue;
      }
      if (c == '"') {
        const std::size_t b = p_;
        ++p_;
        std::string text;
        bool closed = false;
        while (p_ < s_.size()) {
          if (s_[p_] == '"') {
            ++p_;
            closed = true;
            break;
          }
          if (s_[p_] == '\n') break;
          if (s_[p_] == '\\' && p_ + 1 < s_.size()) {
            ++p_;
            const char e = s_[p_++];
            text += (e == 'n') ? '\n' : (e == 't') ? '\t' : e;
            continue;
          }
          text += s_[p_++];
        }
        if (!closed) {
          errAt(b, 1, "unterminated string literal");
          ok = false;
        }
        pushAt(Token::Kind::String, text, b, p_ - b);
        continue;
      }
      if (isIdentStart(c)) {
        const std::size_t b = p_;
        while (p_ < s_.size() && isIdentChar(s_[p_])) ++p_;
        pushAt(Token::Kind::Ident, s_.substr(b, p_ - b), b, p_ - b);
        continue;
      }
      if (isDigit(c)) {
        if (!lexNumber()) ok = false;
        continue;
      }
      if (c == ':' || c == '=' || c == ',' || c == '+' || c == '-') {
        pushAt(Token::Kind::Punct, std::string(1, c), p_, 1);
        ++p_;
        continue;
      }
      errAt(p_, 1, std::string("unexpected character '") + c + "'");
      ok = false;
      ++p_;
    }
    push(Token::Kind::Newline, "\\n", 1);
    push(Token::Kind::End, "<eof>", 1);
    return ok;
  }

 private:
  bool lexNumber() {
    const std::size_t b = p_;
    std::int64_t value = 0;
    if (s_[p_] == '0' && p_ + 1 < s_.size() && (s_[p_ + 1] == 'x' || s_[p_ + 1] == 'X')) {
      p_ += 2;
      const std::size_t digits = p_;
      while (p_ < s_.size() && (isDigit(s_[p_]) || (s_[p_] >= 'a' && s_[p_] <= 'f') ||
                                (s_[p_] >= 'A' && s_[p_] <= 'F')))
        ++p_;
      if (p_ == digits) {
        errAt(b, 2, "expected at least one hexadecimal digit after '0x'");
        return false;
      }
      value = static_cast<std::int64_t>(std::strtoull(s_.substr(digits, p_ - digits).c_str(), nullptr, 16));
    } else {
      const std::size_t digits = p_;
      while (p_ < s_.size() && isDigit(s_[p_])) ++p_;
      value = static_cast<std::int64_t>(std::strtoull(s_.substr(digits, p_ - digits).c_str(), nullptr, 10));
    }
    char suffix = 0;
    if (p_ < s_.size() && (s_[p_] == 'b' || s_[p_] == 'w')) {
      const char sfx = s_[p_];
      if (p_ + 1 >= s_.size() || !isIdentChar(s_[p_ + 1])) {
        suffix = sfx;
        ++p_;
      }
    }
    if (p_ < s_.size() && isIdentChar(s_[p_])) {
      const std::size_t bad = p_;
      while (p_ < s_.size() && isIdentChar(s_[p_])) ++p_;
      errAt(bad, static_cast<int>(p_ - bad),
            "unexpected '" + s_.substr(bad, p_ - bad) +
                "' after a number; the only recognised suffixes are 'b' (bytes) and 'w' (int32 ACC words)");
      return false;
    }
    Token t;
    t.kind = Token::Kind::Number;
    t.text = s_.substr(b, p_ - b);
    t.num = value;
    t.suffix = suffix;
    locate(b, static_cast<int>(p_ - b), t);
    toks_->push_back(t);
    return true;
  }

  void skipSpace() {
    while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\r')) ++p_;
  }

  void locate(std::size_t off, int len, Token& t) {
    t.line = line_;
    t.col = static_cast<int>(off - bol_) + 1;
    t.length = len;
  }

  void push(Token::Kind k, const std::string& text, int len) { pushAt(k, text, p_, static_cast<std::size_t>(len)); }

  void pushAt(Token::Kind k, const std::string& text, std::size_t off, std::size_t len) {
    Token t;
    t.kind = k;
    t.text = text;
    locate(off, static_cast<int>(len), t);
    toks_->push_back(t);
  }

  void errAt(std::size_t off, int len, const std::string& msg) {
    diags_.error(file_, line_, static_cast<int>(off - bol_) + 1, len, msg);
  }

  const std::string& s_;
  std::string file_;
  Diagnostics& diags_;
  std::vector<Token>* toks_ = nullptr;
  std::size_t p_ = 0;
  std::size_t bol_ = 0;
  int line_ = 1;
};

// ===========================================================================
// Parser
// ===========================================================================

/// The opcode's field names joined with ", " for a human-facing note.
/// `operandListOf()` stays comma-only because it has to reproduce
/// `keaOpInfo().operands` byte for byte.
std::string fieldListPretty(const OpDef& op) {
  std::string s;
  for (std::uint8_t i = 0; i < op.nfields; ++i) {
    if (i) s += ", ";
    s += op.fields[i].name;
  }
  return s;
}

struct EventName {
  std::uint32_t id;
  std::string name;
};
struct RegionName {
  std::uint32_t tag;
  std::string name;
};
struct LabelDef {
  std::uint32_t pc;
  std::string name;
};

class Assembler {
 public:
  Assembler(const std::string& file, const AsmOptions& opts, Diagnostics& diags)
      : file_(file), opts_(opts), diags_(diags) {}

  bool run(const std::vector<Token>& toks, KeaProgram& program);

 private:
  // -- token helpers --------------------------------------------------------
  const Token& cur() const { return toks_->at(p_); }
  const Token& peek(std::size_t n = 1) const {
    return toks_->at(p_ + n < toks_->size() ? p_ + n : toks_->size() - 1);
  }
  void advance() {
    if (p_ + 1 < toks_->size()) ++p_;
  }
  bool isPunct(const Token& t, char c) const {
    return t.kind == Token::Kind::Punct && t.text.size() == 1 && t.text[0] == c;
  }

  Diagnostic& err(const Token& t, const std::string& msg) {
    return diags_.error(file_, t.line, t.col, t.length, msg);
  }
  Diagnostic& warn(const Token& t, const std::string& msg) {
    return diags_.warning(file_, t.line, t.col, t.length, msg);
  }

  void skipToEol() {
    while (cur().kind != Token::Kind::Newline && cur().kind != Token::Kind::End) advance();
  }

  // -- productions ----------------------------------------------------------
  bool parseDirective();
  bool parseInstruction();
  bool parseFieldValue(const OpDef& op, const FieldDef& f, KeaInstr& in, Space& parsed_space,
                       bool& has_space);
  bool parseSignedNumber(std::int64_t& value, char& suffix, Token& first);
  bool semanticChecks(const OpDef& op, const KeaInstr& in, const Token& mnemonic_tok,
                      const std::vector<const Token*>& value_tok);
  bool checkRuleD();
  void finish(KeaProgram& program);

  std::string file_;
  const AsmOptions& opts_;
  Diagnostics& diags_;
  const std::vector<Token>* toks_ = nullptr;
  std::size_t p_ = 0;

  std::vector<KeaInstr> code_;
  std::vector<Token> instr_tok_;  // location of each instruction's mnemonic
  std::vector<EventName> events_;
  std::vector<RegionName> regions_;
  std::vector<LabelDef> labels_;
  bool have_entry_ = false;
  bool entry_is_label_ = false;
  std::uint32_t entry_pc_ = 0;
  std::string entry_label_;
  Token entry_tok_;
  bool ok_ = true;
};

bool Assembler::parseSignedNumber(std::int64_t& value, char& suffix, Token& first) {
  first = cur();
  std::int64_t sign = 1;
  if (isPunct(cur(), '-') || isPunct(cur(), '+')) {
    sign = isPunct(cur(), '-') ? -1 : 1;
    advance();
  }
  if (cur().kind != Token::Kind::Number) {
    err(cur(), "expected an integer, found '" + cur().text + "'");
    return false;
  }
  value = sign * cur().num;
  suffix = cur().suffix;
  first.length = (cur().col + cur().length) - first.col;
  advance();
  return true;
}

bool Assembler::parseFieldValue(const OpDef& op, const FieldDef& f, KeaInstr& in,
                                Space& parsed_space, bool& has_space) {
  has_space = false;
  const Token start = cur();

  // ---- symbolic (flag / enum) values --------------------------------------
  if (f.units == FieldUnits::Symbolic) {
    std::string text;
    if (cur().kind == Token::Kind::Ident) {
      text = cur().text;
    } else if (cur().kind == Token::Kind::Number && cur().suffix == 0) {
      text = std::to_string(cur().num);
    } else {
      std::string expected;
      for (std::uint8_t i = 0; i < f.nchoices; ++i) {
        if (i) expected += ", ";
        expected += f.choices[i];
      }
      err(cur(), std::string("expected one of {") + expected + "} for '" + f.name + "'");
      return false;
    }
    for (std::uint8_t i = 0; i < f.nchoices; ++i) {
      if (text == f.choices[i]) {
        fieldSet(in, f, i);
        advance();
        return true;
      }
    }
    std::string expected;
    for (std::uint8_t i = 0; i < f.nchoices; ++i) {
      if (i) expected += ", ";
      expected += f.choices[i];
    }
    Diagnostic& d = err(cur(), "'" + text + "' is not a valid value for '" + std::string(f.name) +
                                   "' on " + op.mnemonic);
    d.notes.push_back(std::string(f.name) + " must be one of {" + expected + "}");
    return false;
  }

  // ---- addresses ----------------------------------------------------------
  if (f.units == FieldUnits::AddrDram) {
    std::uint64_t addr = 0;
    if (cur().kind == Token::Kind::Symbol) {
      const Token sym = cur();
      advance();
      std::int64_t off = 0;
      if (isPunct(cur(), '+') || isPunct(cur(), '-')) {
        const std::int64_t sign = isPunct(cur(), '-') ? -1 : 1;
        advance();
        if (cur().kind != Token::Kind::Number) {
          err(cur(), "expected an integer offset after '@" + sym.text + "'");
          return false;
        }
        if (cur().suffix == 'w') {
          err(cur(), "DRAM offsets are measured in bytes; the 'w' (int32 words) suffix is only "
                     "legal on ACC quantities");
          return false;
        }
        off = sign * cur().num;
        advance();
      }
      if (!opts_.map) {
        Diagnostic& d = err(sym, "cannot resolve '@" + sym.text + "': no model map was supplied");
        d.notes.push_back("pass --map model.map.json, or use the dram:<byte-address> escape");
        return false;
      }
      const MapSymbol* s = opts_.map->find(sym.text);
      if (!s) {
        Diagnostic& d = err(sym, "unknown DRAM symbol '@" + sym.text + "'");
        d.notes.push_back("declare it in the 'symbols' or 'tensors' array of model.map.json");
        return false;
      }
      const std::int64_t resolved = static_cast<std::int64_t>(s->offset) + off;
      if (resolved < 0) {
        err(sym, "'@" + sym.text + "' resolves to a negative DRAM address");
        return false;
      }
      addr = static_cast<std::uint64_t>(resolved);
    } else if (cur().kind == Token::Kind::Ident && cur().text == "dram" && isPunct(peek(), ':')) {
      advance();
      advance();
      if (cur().kind != Token::Kind::Number || cur().suffix != 0) {
        err(cur(), "expected a byte address after 'dram:'");
        return false;
      }
      addr = static_cast<std::uint64_t>(cur().num);
      advance();
    } else {
      Diagnostic& d = err(cur(), "expected a DRAM address for '" + std::string(f.name) + "'");
      d.notes.push_back("DRAM addresses are symbolic: '@name', '@name+offset', or the "
                        "'dram:<byte-address>' escape for hand-written tests");
      return false;
    }
    if (addr > 0xFFFFFFFFull) {
      err(start, "DRAM address " + std::to_string(addr) + " does not fit in 32 bits");
      return false;
    }
    if (opts_.map && opts_.map->dram.total_bytes && addr >= opts_.map->dram.total_bytes) {
      err(start, "DRAM address " + std::to_string(addr) + " is outside the " +
                     std::to_string(opts_.map->dram.total_bytes) + "-byte arena");
      return false;
    }
    fieldSet(in, f, static_cast<std::int64_t>(addr));
    return true;
  }

  if (f.units == FieldUnits::AddrSpmA || f.units == FieldUnits::AddrSpmW ||
      f.units == FieldUnits::AddrAcc || f.units == FieldUnits::AddrSpmByFlag) {
    const char* want = nullptr;
    switch (f.units) {
      case FieldUnits::AddrSpmA: want = "a"; break;
      case FieldUnits::AddrSpmW: want = "w"; break;
      case FieldUnits::AddrAcc: want = "acc"; break;
      default: break;
    }
    if (cur().kind != Token::Kind::Ident || !isPunct(peek(), ':')) {
      Diagnostic& d = err(cur(), "expected a scratchpad address for '" + std::string(f.name) + "'");
      if (want)
        d.notes.push_back(std::string("write it as '") + want + ":<" +
                          (f.units == FieldUnits::AddrAcc ? "int32-word index" : "byte offset") +
                          ">, e.g. " + want + ":0");
      else
        d.notes.push_back("write it as 'a:<byte offset>' (SPM_A) or 'w:<byte offset>' (SPM_W)");
      return false;
    }
    const Token pfx = cur();
    const std::string prefix = pfx.text;
    Space space = Space::SPM_A;
    if (prefix == "a") space = Space::SPM_A;
    else if (prefix == "w") space = Space::SPM_W;
    else if (prefix == "acc") space = Space::ACC;
    else {
      Diagnostic& d = err(pfx, "unknown address space prefix '" + prefix + ":'");
      d.notes.push_back("the address-space prefixes are 'a:' (SPM_A bytes), 'w:' (SPM_W bytes), "
                        "'acc:' (ACC int32 words) and '@sym' (DRAM)");
      return false;
    }
    if (want && prefix != want) {
      Diagnostic& d = err(pfx, "'" + std::string(f.name) + "' on " + op.mnemonic + " addresses " +
                                   spaceName(f.units == FieldUnits::AddrSpmA   ? Space::SPM_A
                                             : f.units == FieldUnits::AddrSpmW ? Space::SPM_W
                                                                               : Space::ACC) +
                                   ", but '" + prefix + ":' was written");
      d.notes.push_back(std::string("use '") + want + ":'");
      return false;
    }
    if (!want && space == Space::ACC) {
      err(pfx, "'" + std::string(f.name) + "' cannot address ACC; use 'a:' or 'w:'");
      return false;
    }
    advance();  // prefix
    advance();  // ':'
    if (cur().kind != Token::Kind::Number) {
      err(cur(), "expected an address after '" + prefix + ":'");
      return false;
    }
    if (cur().suffix != 0) {
      err(cur(), "an address after '" + prefix + ":' takes no unit suffix; it is already " +
                     spaceUnitName(space) + " of " + spaceName(space));
      return false;
    }
    const std::int64_t value = cur().num;
    const std::uint64_t cap = spaceCapacity(space);
    if (value < 0 || static_cast<std::uint64_t>(value) >= cap) {
      err(cur(), "'" + std::string(f.name) + "' = " + std::to_string(value) + " is outside " +
                     spaceName(space) + ", which holds " + std::to_string(cap) + " " +
                     spaceUnitName(space));
      return false;
    }
    if (f.align && static_cast<std::uint64_t>(value) % f.align) {
      Diagnostic& d =
          err(cur(), "'" + std::string(f.name) + "' must be a multiple of " + std::to_string(f.align) +
                         " " + spaceUnitName(space) + ", got " + std::to_string(value));
      if (space == Space::ACC)
        d.notes.push_back("ACC is addressed in int32 WORDS, not bytes: acc:" +
                          std::to_string(value) + " is byte offset " + std::to_string(value * 4));
      return false;
    }
    advance();
    parsed_space = space;
    has_space = (f.units == FieldUnits::AddrSpmByFlag);
    fieldSet(in, f, value);
    return true;
  }

  // ---- plain numbers ------------------------------------------------------
  std::int64_t value = 0;
  char suffix = 0;
  Token first;
  if (!parseSignedNumber(value, suffix, first)) return false;

  if (f.units == FieldUnits::AccWords) {
    if (suffix != 'w') {
      Diagnostic& d = err(first, "'" + std::string(f.name) + "' is measured in ACC int32 words and "
                                 "must carry the 'w' suffix");
      d.notes.push_back("write " + std::to_string(value) + "w (that is " +
                        std::to_string(value * 4) + " bytes)");
      return false;
    }
  } else if (suffix == 'w') {
    Diagnostic& d = err(first, "'" + std::string(f.name) + "' is measured in bytes; the 'w' (int32 "
                               "ACC words) suffix is not allowed here");
    d.notes.push_back("only ACC strides (acc_inner_stride, acc_outer_stride, acc_pix_stride) take 'w'");
    return false;
  } else if (suffix == 'b' && f.units != FieldUnits::Bytes) {
    err(first, "'" + std::string(f.name) + "' is a plain count, not a byte quantity; drop the 'b' suffix");
    return false;
  }

  if (value < f.min || value > f.max) {
    err(first, "'" + std::string(f.name) + "' must be in " + std::to_string(f.min) + ".." +
                   std::to_string(f.max) + ", got " + std::to_string(value));
    return false;
  }
  if (f.align && (value % static_cast<std::int64_t>(f.align)) != 0) {
    Diagnostic& d = err(first, "'" + std::string(f.name) + "' must be a multiple of " +
                                   std::to_string(f.align) + ", got " + std::to_string(value));
    if (f.units == FieldUnits::AccWords)
      d.notes.push_back("ACC strides count int32 words and the accumulator tile is 16 lanes wide");
    return false;
  }
  fieldSet(in, f, value);
  return true;
}

bool Assembler::parseInstruction() {
  const Token unit_tok = cur();
  Unit unit = Unit::CTRL;
  bool unit_ok = false;
  for (int u = 0; u < KEA_NUM_UNITS; ++u) {
    if (unit_tok.text == unitName(static_cast<Unit>(u))) {
      unit = static_cast<Unit>(u);
      unit_ok = true;
      break;
    }
  }
  if (!unit_ok) {
    Diagnostic& d = err(unit_tok, "'" + unit_tok.text + "' is not a unit name");
    d.notes.push_back("every statement starts with the issuing queue: MXU, DWU, VPU, DMA0, DMA1 or CTRL");
    skipToEol();
    return false;
  }
  advance();

  const Token mn = cur();
  if (mn.kind != Token::Kind::Ident) {
    err(mn, "expected an instruction mnemonic after '" + unit_tok.text + "'");
    skipToEol();
    return false;
  }
  const OpDef* op = opDefByMnemonic(mn.text);
  if (!op) {
    Diagnostic& d = err(mn, "unknown mnemonic '" + mn.text + "'");
    std::size_t n = 0;
    const OpDef* all = opDefTable(n);
    std::string list;
    for (std::size_t i = 0; i < n; ++i) {
      if (i) list += ", ";
      list += all[i].mnemonic;
    }
    d.notes.push_back("known mnemonics: " + list);
    skipToEol();
    return false;
  }
  advance();

  // Unit legality, checked here so the message can name the opcode.
  if (!op->unit_is_operand && unit != op->fixed_unit) {
    Diagnostic& d = err(unit_tok, std::string(op->mnemonic) + " must be issued on " +
                                      unitName(op->fixed_unit) + ", not " + unitName(unit));
    d.notes.push_back(std::string(op->mnemonic) + " is owned by the " + unitName(op->fixed_unit) +
                      "; only NOP, SIGNAL, WAIT and TRACE may target any queue");
    skipToEol();
    return false;
  }
  if ((op->opcode == Opcode::DMA_LD || op->opcode == Opcode::DMA_ST) && unit != Unit::DMA0 &&
      unit != Unit::DMA1) {
    Diagnostic& d =
        err(unit_tok, std::string(op->mnemonic) + " must be issued on DMA0 or DMA1, not " + unitName(unit));
    d.notes.push_back("the two DMA engines exist so loads can be double-buffered against stores");
    skipToEol();
    return false;
  }

  KeaInstr in{};
  for (std::size_t i = 0; i < 32; ++i) in.b[i] = 0;
  in.b[0] = static_cast<std::uint8_t>(op->opcode);
  in.b[1] = static_cast<std::uint8_t>(unit);

  std::vector<bool> seen(op->nfields, false);
  std::vector<const Token*> value_tok(op->nfields, nullptr);
  std::vector<Space> addr_space(op->nfields, Space::SPM_A);
  std::vector<bool> addr_has_space(op->nfields, false);
  bool failed = false;

  while (cur().kind != Token::Kind::Newline && cur().kind != Token::Kind::End) {
    const Token name_tok = cur();
    if (name_tok.kind != Token::Kind::Ident) {
      err(name_tok, "expected a 'field=value' pair, found '" + name_tok.text + "'");
      failed = true;
      break;
    }
    const FieldDef* f = fieldByName(*op, name_tok.text);
    if (!f) {
      Diagnostic& d = err(name_tok, "unknown field '" + name_tok.text + "' for " + op->mnemonic);
      d.notes.push_back(std::string(op->mnemonic) + " fields are: " + fieldListPretty(*op));
      failed = true;
      break;
    }
    const std::size_t idx = static_cast<std::size_t>(f - op->fields);
    if (seen[idx]) {
      err(name_tok, "duplicate field '" + name_tok.text + "'");
      failed = true;
      break;
    }
    advance();
    if (!isPunct(cur(), '=')) {
      err(cur(), "expected '=' after '" + name_tok.text + "'");
      failed = true;
      break;
    }
    advance();
    const std::size_t vstart = p_;
    Space sp = Space::SPM_A;
    bool has_sp = false;
    if (!parseFieldValue(*op, *f, in, sp, has_sp)) {
      failed = true;
      break;
    }
    seen[idx] = true;
    value_tok[idx] = &(*toks_)[vstart];
    addr_space[idx] = sp;
    addr_has_space[idx] = has_sp;

    if (isPunct(cur(), ',')) {
      advance();
      // A trailing comma continues the statement onto the next line.
      while (cur().kind == Token::Kind::Newline) advance();
      continue;
    }
    break;
  }

  if (failed) {
    skipToEol();
    return false;
  }

  // Missing fields.
  std::string missing;
  for (std::uint8_t i = 0; i < op->nfields; ++i) {
    if (!seen[i]) {
      if (!missing.empty()) missing += ", ";
      missing += op->fields[i].name;
    }
  }
  if (!missing.empty()) {
    Diagnostic& d = err(mn, std::string(op->mnemonic) + " is missing required field(s): " + missing);
    d.notes.push_back("every operand must be written explicitly; " + std::string(op->mnemonic) +
                      " takes: " + fieldListPretty(*op));
    skipToEol();
    return false;
  }

  // Address-prefix / space-flag agreement for the two-space opcodes.
  for (std::uint8_t i = 0; i < op->nfields; ++i) {
    const FieldDef& f = op->fields[i];
    if (f.units != FieldUnits::AddrSpmByFlag || !addr_has_space[i]) continue;
    const Space encoded = fieldSpace(*op, f, in);
    if (encoded != addr_space[i]) {
      const Token& t = *value_tok[i];
      Diagnostic& d = diags_.error(file_, t.line, t.col, t.length,
                                   "'" + std::string(f.name) + "' is written as an " +
                                       spaceName(addr_space[i]) + " address but '" + f.space_field +
                                       "' says " + spaceName(encoded));
      d.notes.push_back("keep the prefix and the space flag in agreement; they are redundant on "
                        "purpose so a mistake is caught here rather than in silicon");
      skipToEol();
      return false;
    }
  }

  if (!semanticChecks(*op, in, mn, value_tok)) {
    skipToEol();
    return false;
  }

  if (const char* e = keaValidate(in)) {
    Diagnostic& d = err(mn, std::string("instruction rejected by keaValidate(): ") + e);
    d.notes.push_back("this is the frozen ISA validator in include/kea/isa.h");
    skipToEol();
    return false;
  }

  if (code_.size() >= KEA_MAX_INSTRUCTIONS) {
    err(mn, "program exceeds IMEM: more than " + std::to_string(KEA_MAX_INSTRUCTIONS) +
                " instructions");
    skipToEol();
    return false;
  }

  code_.push_back(in);
  instr_tok_.push_back(mn);
  return true;
}

bool Assembler::semanticChecks(const OpDef& op, const KeaInstr& in, const Token& mn,
                               const std::vector<const Token*>& value_tok) {
  auto tokFor = [&](const char* name) -> const Token& {
    const FieldDef* f = fieldByName(op, name);
    if (f) {
      const std::size_t idx = static_cast<std::size_t>(f - op.fields);
      if (idx < value_tok.size() && value_tok[idx]) return *value_tok[idx];
    }
    return mn;
  };
  auto get = [&](const char* name) -> std::int64_t {
    const FieldDef* f = fieldByName(op, name);
    return f ? fieldGet(in, *f) : 0;
  };

  switch (op.opcode) {
    case Opcode::LOAD_W: {
      const bool int4 = get("dtype") == 1;
      const std::uint32_t align = int4 ? KEA_ALIGN_LOADW_INT4 : KEA_ALIGN_LOADW_INT8;
      const std::int64_t addr = get("w_addr");
      const std::int64_t stride = get("w_row_stride");
      if (addr % align) {
        Diagnostic& d = err(tokFor("w_addr"), "'w_addr' must be a multiple of " +
                                                  std::to_string(align) + " bytes for dtype=" +
                                                  (int4 ? "int4" : "int8") + ", got " +
                                                  std::to_string(addr));
        d.notes.push_back("a 16x16 weight tile is 256 bytes in int8 and 128 bytes in int4");
        return false;
      }
      if (stride % align) {
        err(tokFor("w_row_stride"), "'w_row_stride' must be a multiple of " + std::to_string(align) +
                                        " bytes for dtype=" + (int4 ? "int4" : "int8") + ", got " +
                                        std::to_string(stride));
        return false;
      }
      return true;
    }
    case Opcode::MATMUL: {
      const std::int64_t rows = get("m_inner") * get("m_outer");
      if (rows > static_cast<std::int64_t>(KEA_MXU_MAX_ROWS)) {
        Diagnostic& d = err(tokFor("m_outer"), "m_inner * m_outer = " + std::to_string(rows) +
                                                   " exceeds KEA_MXU_MAX_ROWS (" +
                                                   std::to_string(KEA_MXU_MAX_ROWS) + ")");
        d.notes.push_back("each row writes 16 int32 ACC words, and ACC holds " +
                          std::to_string(KEA_ACC_WORDS) + " words");
        return false;
      }
      return true;
    }
    case Opcode::DWCONV: {
      const std::uint64_t need = static_cast<std::uint64_t>(get("out_h")) *
                                 static_cast<std::uint64_t>(get("out_w")) *
                                 static_cast<std::uint64_t>(get("channels"));
      if (static_cast<std::uint64_t>(get("acc_addr")) + need > KEA_ACC_WORDS) {
        err(tokFor("acc_addr"), "the dense [out_h][out_w][channels] output (" +
                                    std::to_string(need) + " words) starting at acc:" +
                                    std::to_string(get("acc_addr")) + " runs past the end of ACC");
        return false;
      }
      return true;
    }
    case Opcode::VQUANT: {
      const std::int64_t lo = get("clamp_lo"), hi = get("clamp_hi");
      if (lo > hi) {
        err(tokFor("clamp_lo"), "clamp_lo (" + std::to_string(lo) + ") is greater than clamp_hi (" +
                                    std::to_string(hi) + ")");
        return false;
      }
      if (get("dtype") == 1 && (lo < KEA_INT4_MIN || hi > KEA_INT4_MAX)) {
        Diagnostic& d = err(tokFor("clamp_lo"), "with dtype=int4 the clamps must lie inside [-8, 7], got [" +
                                                    std::to_string(lo) + ", " + std::to_string(hi) + "]");
        d.notes.push_back("int4 values are sign-extended from 4 bits (ISA.md §3)");
        return false;
      }
      return true;
    }
    case Opcode::VADD: {
      if (get("clamp_lo") > get("clamp_hi")) {
        err(tokFor("clamp_lo"), "clamp_lo (" + std::to_string(get("clamp_lo")) +
                                    ") is greater than clamp_hi (" + std::to_string(get("clamp_hi")) + ")");
        return false;
      }
      return true;
    }
    case Opcode::DMA_LD:
    case Opcode::DMA_ST: {
      // ISA.md §6.1: a zero stride on the *destination* side collapses runs on
      // top of each other and is last-writer-wins, i.e. undefined.
      const bool is_load = op.opcode == Opcode::DMA_LD;
      const char* s1 = is_load ? "spm_s1" : "dram_s1";
      const char* s2 = is_load ? "spm_s2" : "dram_s2";
      const char* side = is_load ? "SPM" : "DRAM";
      if (get("n1") > 1 && get(s1) == 0) {
        Diagnostic& d = err(tokFor(s1), std::string("'") + s1 + "' is 0 with n1=" +
                                            std::to_string(get("n1")) + ", so every run would be "
                                            "written to the same " + side + " address");
        d.notes.push_back("a zero stride on the destination side of a DMA is last-writer-wins and "
                          "therefore undefined");
        return false;
      }
      if (get("n2") > 1 && get(s2) == 0) {
        Diagnostic& d = err(tokFor(s2), std::string("'") + s2 + "' is 0 with n2=" +
                                            std::to_string(get("n2")) + ", so every plane would be "
                                            "written to the same " + side + " address");
        d.notes.push_back("a zero stride on the destination side of a DMA is last-writer-wins and "
                          "therefore undefined");
        return false;
      }
      return true;
    }
    default:
      return true;
  }
}

bool Assembler::parseDirective() {
  const Token dir = cur();
  advance();
  const std::string& name = dir.text;

  if (name == "arch") {
    if (cur().kind != Token::Kind::String) {
      err(cur(), ".arch takes a quoted architecture name, e.g. .arch \"KEA-1\"");
      skipToEol();
      return false;
    }
    if (cur().text != KEA_ARCH_NAME) {
      err(cur(), "this assembler implements " + std::string(KEA_ARCH_NAME) + ", not '" + cur().text + "'");
      skipToEol();
      return false;
    }
    advance();
    return true;
  }
  if (name == "isa_revision") {
    if (cur().kind != Token::Kind::Number) {
      err(cur(), ".isa_revision takes an integer");
      skipToEol();
      return false;
    }
    if (cur().num != static_cast<std::int64_t>(KEA_ISA_REVISION)) {
      err(cur(), "source targets ISA revision " + std::to_string(cur().num) +
                     ", this assembler implements " + std::to_string(KEA_ISA_REVISION));
      skipToEol();
      return false;
    }
    advance();
    return true;
  }
  if (name == "entry") {
    entry_tok_ = cur();
    if (cur().kind == Token::Kind::Number) {
      entry_pc_ = static_cast<std::uint32_t>(cur().num);
      entry_is_label_ = false;
    } else if (cur().kind == Token::Kind::Ident) {
      entry_label_ = cur().text;
      entry_is_label_ = true;
    } else {
      err(cur(), ".entry takes an instruction index or a label name");
      skipToEol();
      return false;
    }
    have_entry_ = true;
    advance();
    return true;
  }
  if (name == "event" || name == "region") {
    if (cur().kind != Token::Kind::Number) {
      err(cur(), "." + name + " takes an integer " + (name == "event" ? "event id" : "region tag") +
                     " and a quoted name");
      skipToEol();
      return false;
    }
    const std::int64_t id = cur().num;
    if (name == "event" && (id < 0 || id >= KEA_NUM_EVENTS)) {
      err(cur(), "event id must be in 0.." + std::to_string(KEA_NUM_EVENTS - 1) + ", got " +
                     std::to_string(id));
      skipToEol();
      return false;
    }
    advance();
    if (!isPunct(cur(), ',')) {
      err(cur(), "expected ',' between the " + std::string(name == "event" ? "event id" : "region tag") +
                     " and its name");
      skipToEol();
      return false;
    }
    advance();
    if (cur().kind != Token::Kind::String) {
      err(cur(), "expected a quoted name");
      skipToEol();
      return false;
    }
    if (name == "event")
      events_.push_back({static_cast<std::uint32_t>(id), cur().text});
    else
      regions_.push_back({static_cast<std::uint32_t>(id), cur().text});
    advance();
    return true;
  }

  Diagnostic& d = err(dir, "unknown directive '." + name + "'");
  d.notes.push_back("known directives: .arch, .isa_revision, .entry, .event, .region");
  skipToEol();
  return false;
}

bool Assembler::checkRuleD() {
  // Rule D (ISA.md §5.5): the SIGNALs supplying a WAIT's counts must appear
  // earlier in stream order, otherwise the in-order dispatcher can wedge.
  std::int64_t balance[KEA_NUM_EVENTS] = {0};
  bool ok = true;
  for (std::size_t pc = 0; pc < code_.size(); ++pc) {
    const Opcode op = keaOpcode(code_[pc]);
    if (op != Opcode::SIGNAL && op != Opcode::WAIT) continue;
    const KeaEventOp e = keaDecode<KeaEventOp>(code_[pc]);
    const int id = e.h.aux;
    if (op == Opcode::SIGNAL) {
      balance[id] += e.value;
      continue;
    }
    if (balance[id] < static_cast<std::int64_t>(e.value)) {
      const Token& t = instr_tok_[pc];
      const std::string msg =
          "Rule D violation: WAIT event=" + std::to_string(id) + ", threshold=" +
          std::to_string(e.value) + " at pc " + std::to_string(pc) + " is only supplied with " +
          std::to_string(balance[id]) + " count(s) by preceding SIGNALs";
      Diagnostic& d = opts_.rule_d_is_error ? diags_.error(file_, t.line, t.col, t.length, msg)
                                            : diags_.warning(file_, t.line, t.col, t.length, msg);
      d.notes.push_back("dispatch is in-order and its only stall is a full queue, so a WAIT whose "
                        "SIGNALs come later in the stream can deadlock the machine (ISA.md §5.5)");
      if (opts_.rule_d_is_error) ok = false;
    }
    balance[id] -= static_cast<std::int64_t>(e.value);
  }
  return ok;
}

void Assembler::finish(KeaProgram& program) {
  program.code = code_;
  program.entry_pc = entry_pc_;

  if (opts_.map) {
    program.dram = opts_.map->dram;
    program.tensors = opts_.map->tensors;
    program.spm_map = opts_.map->spm_map;
  } else {
    program.dram = KeafDramLayout{};
    program.dram.alignment = KEA_DRAM_BASE_ALIGN;
  }

  if (opts_.const_data) {
    program.const_data = *opts_.const_data;
    if (program.const_data.size() != program.dram.const_bytes) {
      diags_.error(file_, 0, 0, 1,
                   "the constant blob is " + std::to_string(program.const_data.size()) +
                       " bytes but the map declares dram.const_bytes = " +
                       std::to_string(program.dram.const_bytes));
      ok_ = false;
    }
  } else if (program.dram.const_bytes != 0) {
    Diagnostic& d = diags_.error(file_, 0, 0, 1,
                                 "the map declares dram.const_bytes = " +
                                     std::to_string(program.dram.const_bytes) +
                                     " but no constant blob was supplied");
    d.notes.push_back("pass --const model.weights.bin");
    ok_ = false;
  }

  Json md;
  std::string jerr;
  if (opts_.map && !opts_.map->metadata_json.empty()) {
    if (!Json::parse(opts_.map->metadata_json, md, jerr) || !md.isObject()) md = Json::object();
  } else {
    md = Json::object();
  }
  if (!events_.empty()) {
    Json arr = Json::array();
    for (const EventName& e : events_) {
      Json o = Json::object();
      o.set("id", Json(static_cast<std::int64_t>(e.id)));
      o.set("name", Json(e.name));
      arr.push_back(std::move(o));
    }
    md.set("events", std::move(arr));
  }
  if (!regions_.empty()) {
    Json arr = Json::array();
    for (const RegionName& r : regions_) {
      Json o = Json::object();
      o.set("tag", Json(static_cast<std::int64_t>(r.tag)));
      o.set("name", Json(r.name));
      arr.push_back(std::move(o));
    }
    md.set("regions", std::move(arr));
  }
  if (!labels_.empty()) {
    Json arr = Json::array();
    for (const LabelDef& l : labels_) {
      Json o = Json::object();
      o.set("pc", Json(static_cast<std::int64_t>(l.pc)));
      o.set("name", Json(l.name));
      arr.push_back(std::move(o));
    }
    md.set("labels", std::move(arr));
  }
  program.metadata_json = (md.size() == 0) ? std::string() : md.dump(2);
}

bool Assembler::run(const std::vector<Token>& toks, KeaProgram& program) {
  toks_ = &toks;
  p_ = 0;

  while (cur().kind != Token::Kind::End) {
    if (cur().kind == Token::Kind::Newline) {
      advance();
      continue;
    }
    if (cur().kind == Token::Kind::Directive) {
      if (!parseDirective()) ok_ = false;
    } else if (cur().kind == Token::Kind::Ident && isPunct(peek(), ':') &&
               (peek(2).kind == Token::Kind::Newline || peek(2).kind == Token::Kind::Ident ||
                peek(2).kind == Token::Kind::End)) {
      // A label: IDENT ':' at statement position. Address prefixes (`a:`,
      // `acc:`) are only ever reachable in operand position, so there is no
      // ambiguity.
      const Token lbl = cur();
      for (const LabelDef& l : labels_) {
        if (l.name == lbl.text) {
          err(lbl, "duplicate label '" + lbl.text + "'");
          ok_ = false;
        }
      }
      labels_.push_back({static_cast<std::uint32_t>(code_.size()), lbl.text});
      advance();
      advance();
      continue;
    } else if (cur().kind == Token::Kind::Ident) {
      if (!parseInstruction()) ok_ = false;
    } else {
      err(cur(), "expected a unit name, a label or a directive, found '" + cur().text + "'");
      ok_ = false;
      skipToEol();
    }
    if (cur().kind != Token::Kind::Newline && cur().kind != Token::Kind::End) {
      err(cur(), "unexpected '" + cur().text + "' at the end of the statement");
      ok_ = false;
      skipToEol();
    }
  }

  if (code_.empty()) {
    diags_.error(file_, 0, 0, 1, "the program contains no instructions");
    return false;
  }

  // HALT discipline (ISA.md §5.2, §11.3).
  for (std::size_t pc = 0; pc + 1 < code_.size(); ++pc) {
    if (keaOpcode(code_[pc]) == Opcode::HALT) {
      const Token& t = instr_tok_[pc];
      Diagnostic& d = diags_.error(file_, t.line, t.col, t.length,
                                   "HALT must be the last instruction; " +
                                       std::to_string(code_.size() - pc - 1) +
                                       " instruction(s) follow it and are unreachable");
      d.notes.push_back("there are no branches in KEA-1, so anything after HALT can never execute");
      ok_ = false;
      break;
    }
  }
  if (keaOpcode(code_.back()) != Opcode::HALT) {
    const Token& t = instr_tok_.back();
    Diagnostic& d = diags_.error(file_, t.line, t.col, t.length,
                                 "the program does not end with HALT");
    d.notes.push_back("append '  CTRL  HALT    exit_code=0'");
    ok_ = false;
  }

  if (entry_is_label_) {
    bool found = false;
    for (const LabelDef& l : labels_) {
      if (l.name == entry_label_) {
        entry_pc_ = l.pc;
        found = true;
        break;
      }
    }
    if (!found) {
      err(entry_tok_, "unknown label '" + entry_label_ + "' in .entry");
      ok_ = false;
    }
  } else if (!have_entry_ && opts_.map) {
    entry_pc_ = opts_.map->entry_pc;
  }
  if (entry_pc_ >= code_.size()) {
    const std::string msg = "entry point " + std::to_string(entry_pc_) + " is outside the " +
                            std::to_string(code_.size()) + "-instruction program";
    if (have_entry_)
      err(entry_tok_, msg);
    else
      diags_.error(file_, 0, 0, 1, msg);
    ok_ = false;
  }

  if (!checkRuleD()) ok_ = false;

  finish(program);
  return ok_ && !diags_.hasErrors();
}

}  // namespace

bool assemble(const std::string& source, const std::string& filename, const AsmOptions& opts,
              KeaProgram& program, Diagnostics& diags) {
  diags.addSource(filename, source);
  std::vector<Token> toks;
  Lexer lex(source, filename, diags);
  const bool lexed = lex.run(toks);

  Assembler as(filename, opts, diags);
  const bool parsed = as.run(toks, program);
  if (!lexed || !parsed) return false;

  if (const char* e = program.validate()) {
    diags.error(filename, 0, 0, 1, std::string("assembled program failed KeaProgram::validate(): ") + e);
    return false;
  }
  return true;
}

bool assembleFile(const std::string& path, const AsmOptions& opts, KeaProgram& program,
                  Diagnostics& diags) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    diags.error("cannot open '" + path + "'");
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return assemble(ss.str(), path, opts, program, diags);
}

}  // namespace rt
}  // namespace kea
