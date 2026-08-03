// SPDX-License-Identifier: Apache-2.0
//
// kea/rt/op_fields.h --- the one table that defines `.kasm` operand syntax.
//
// ADR-0001 requires the assembly syntax to be derived *mechanically* from
// `kea::keaOpInfo()` so the two cannot drift. This header is that derivation:
// for every opcode it lists, in `keaOpInfo().operands` order, one `FieldDef`
// per named operand, giving the field's location in the 32-byte instruction,
// its measurement unit (plain count / bytes / int32 ACC words / an address in
// a specific space), its legal range, and its alignment.
//
// The assembler, the disassembler, and the syntax documentation all read this
// table and nothing else. `runtime/tests/test_op_fields.cpp` asserts that
// reconstructing a comma-separated name list from the table reproduces
// `keaOpInfo().operands` byte for byte, for every opcode, so adding a field to
// `isa.h` without updating this table fails the build's test suite.

#pragma once

#include <cstdint>
#include <string>

#include "kea/isa.h"

namespace kea {
namespace rt {

/// How a field is stored inside the 32-byte instruction word.
enum class FieldRepr : std::uint8_t {
  U8,     ///< unsigned byte at `offset`
  U16,    ///< unsigned little-endian halfword at `offset`
  U32,    ///< unsigned little-endian word at `offset`
  I8,     ///< signed byte at `offset`
  I32,    ///< signed little-endian word at `offset`
  Flag,   ///< a single bit of the flags byte; two symbolic spellings
  Enum8,  ///< the whole flags byte; `nchoices` symbolic spellings
};

/// What the field's value *means*, which is what decides its `.kasm` spelling.
enum class FieldUnits : std::uint8_t {
  Count,          ///< a pure count/id; written bare, no suffix allowed
  Bytes,          ///< a byte count or byte stride; optional `b` suffix
  AccWords,       ///< an int32-ACC-word stride; **mandatory** `w` suffix
  AddrSpmA,       ///< SPM_A byte address; written `a:<bytes>`
  AddrSpmW,       ///< SPM_W byte address; written `w:<bytes>`
  AddrAcc,        ///< ACC word index; written `acc:<words>`
  AddrDram,       ///< DRAM address; written `@sym`, `@sym+off`, or `dram:<bytes>`
  AddrSpmByFlag,  ///< SPM address whose space comes from `space_field`
  Symbolic,       ///< Flag/Enum8 value, written by name
};

struct FieldDef {
  const char* name;
  FieldRepr repr;
  std::uint8_t offset;   ///< byte offset in KeaInstr (ignored for Flag/Enum8)
  std::uint8_t mask;     ///< Flag: the bit within the flags byte
  const char* const* choices;  ///< Flag: 2 names; Enum8: `nchoices` names
  std::uint8_t nchoices;
  FieldUnits units;
  std::int64_t min;      ///< inclusive lower bound (numeric fields)
  std::int64_t max;      ///< inclusive upper bound (numeric fields)
  std::uint32_t align;   ///< required multiple, or 0
  const char* space_field;  ///< AddrSpmByFlag: the field naming the space
};

struct OpDef {
  Opcode opcode;
  const char* mnemonic;
  /// True when `keaOpInfo().operands` begins with `unit`, i.e. the opcode is
  /// queue-agnostic or DMA and the unit column really is an operand.
  bool unit_is_operand;
  /// The unit an instruction must target when `unit_is_operand` is false.
  Unit fixed_unit;
  const FieldDef* fields;
  std::uint8_t nfields;
};

/// All 12 opcodes plus NOP/HALT, in opcode order.
const OpDef* opDefTable(std::size_t& count);

/// Lookup by mnemonic (case sensitive, upper case) or by opcode.
const OpDef* opDefByMnemonic(const std::string& mnemonic);
const OpDef* opDefFor(Opcode op);
const FieldDef* fieldByName(const OpDef& op, const std::string& name);

/// The comma-separated operand list this table implies, including the leading
/// `unit` when applicable. Must equal `keaOpInfo(op).operands`.
std::string operandListOf(const OpDef& op);

/// Extract / insert a field value. `Flag` and `Enum8` values are the choice
/// index; every other representation is the raw (sign-extended) field value.
std::int64_t fieldGet(const KeaInstr& in, const FieldDef& f);
void fieldSet(KeaInstr& in, const FieldDef& f, std::int64_t value);

/// The address space a `AddrSpmByFlag` / `AddrSpm*` field addresses, resolved
/// against the instruction's flag bits.
Space fieldSpace(const OpDef& op, const FieldDef& f, const KeaInstr& in);

/// Capacity of `space`, in that space's own addressing unit.
std::uint64_t spaceCapacity(Space s);

/// `"bytes"` or `"int32 words"`, for diagnostics.
const char* spaceUnitName(Space s);

}  // namespace rt
}  // namespace kea
