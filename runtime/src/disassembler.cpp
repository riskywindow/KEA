// SPDX-License-Identifier: Apache-2.0
#include "kea/rt/disassembler.h"

#include <algorithm>
#include <string>
#include <vector>

#include "kea/rt/json.h"
#include "kea/rt/op_fields.h"

namespace kea {
namespace rt {
namespace {

constexpr int kUnitColumn = 4;   // widest unit name: DMA0 / DMA1 / CTRL
constexpr int kMnemColumn = 6;   // widest mnemonic: DMA_LD / MATMUL / VQUANT

void pad(std::string& s, std::size_t width) {
  while (s.size() < width) s += ' ';
}

std::string formatField(const OpDef& op, const FieldDef& f, const KeaInstr& in,
                        const ModelMap* map) {
  const std::int64_t v = fieldGet(in, f);
  switch (f.units) {
    case FieldUnits::Symbolic: {
      const std::uint8_t idx = static_cast<std::uint8_t>(v);
      if (idx < f.nchoices) return f.choices[idx];
      return "<invalid:" + std::to_string(v) + ">";
    }
    case FieldUnits::AccWords:
      return std::to_string(v) + "w";
    case FieldUnits::AddrSpmA:
      return "a:" + std::to_string(v);
    case FieldUnits::AddrSpmW:
      return "w:" + std::to_string(v);
    case FieldUnits::AddrAcc:
      return "acc:" + std::to_string(v);
    case FieldUnits::AddrSpmByFlag:
      return (fieldSpace(op, f, in) == Space::SPM_W ? "w:" : "a:") + std::to_string(v);
    case FieldUnits::AddrDram: {
      const std::uint64_t addr = static_cast<std::uint64_t>(v);
      if (map) {
        if (const MapSymbol* s = map->covering(addr)) {
          if (addr == s->offset) return "@" + s->name;
          return "@" + s->name + "+" + std::to_string(addr - s->offset);
        }
      }
      return "dram:" + std::to_string(addr);
    }
    case FieldUnits::Count:
    case FieldUnits::Bytes:
      return std::to_string(v);
  }
  return std::to_string(v);
}

/// The SPM/ACC buffer covering `value` in `space`, for --annotate.
const SpmBinding* spmCovering(const KeaProgram& p, KeafSpace space, std::uint32_t value) {
  const SpmBinding* best = nullptr;
  for (const SpmBinding& b : p.spm_map) {
    if (b.space != space) continue;
    if (value < b.offset) continue;
    const std::uint32_t end = b.offset + (b.size ? b.size : 1);
    if (value >= end) continue;
    if (!best || b.size < best->size) best = &b;
  }
  return best;
}

KeafSpace toKeafSpace(Space s) {
  switch (s) {
    case Space::SPM_A: return KeafSpace::SPM_A;
    case Space::SPM_W: return KeafSpace::SPM_W;
    default: return KeafSpace::ACC;
  }
}

}  // namespace

std::string disassembleInstruction(const KeaInstr& in, const ModelMap* map) {
  const Opcode opc = keaOpcode(in);
  const OpDef* op = opDefFor(opc);
  std::string line;
  std::string u = unitName(keaUnit(in));
  pad(u, kUnitColumn);
  line += u;
  line += "  ";
  if (!op) {
    line += "<invalid opcode 0x";
    const char* hex = "0123456789ABCDEF";
    line += hex[(static_cast<unsigned>(opc) >> 4) & 0xF];
    line += hex[static_cast<unsigned>(opc) & 0xF];
    line += ">";
    return line;
  }
  std::string m = op->mnemonic;
  pad(m, kMnemColumn);
  line += m;
  line += "  ";
  for (std::uint8_t i = 0; i < op->nfields; ++i) {
    if (i) line += ", ";
    line += op->fields[i].name;
    line += '=';
    line += formatField(*op, op->fields[i], in, map);
  }
  return line;
}

std::string disassemble(const KeaProgram& program, const DisasmOptions& opts) {
  std::string out;

  // Recover names declared in the source from the metadata blob.
  std::vector<std::pair<std::uint32_t, std::string>> events, regions, labels;
  if (!program.metadata_json.empty()) {
    Json md;
    std::string err;
    if (Json::parse(program.metadata_json, md, err) && md.isObject()) {
      auto collect = [&](const char* key, const char* idkey,
                         std::vector<std::pair<std::uint32_t, std::string>>& into) {
        const Json* arr = md.find(key);
        if (!arr || !arr->isArray()) return;
        for (std::size_t i = 0; i < arr->size(); ++i) {
          const Json& e = *arr->at(i);
          const Json* id = e.find(idkey);
          const Json* nm = e.find("name");
          if (id && nm && id->isInt() && nm->isString())
            into.emplace_back(static_cast<std::uint32_t>(id->asInt()), nm->asString());
        }
      };
      collect("events", "id", events);
      collect("regions", "tag", regions);
      collect("labels", "pc", labels);
    }
  }

  if (opts.emit_directives) {
    out += ".arch \"";
    out += KEA_ARCH_NAME;
    out += "\"\n";
    out += ".isa_revision " + std::to_string(KEA_ISA_REVISION) + "\n";
    out += ".entry " + std::to_string(program.entry_pc) + "\n";
    if (!events.empty() || !regions.empty()) {
      out += "\n";
      for (const auto& e : events)
        out += ".event " + std::to_string(e.first) + ", \"" + e.second + "\"\n";
      for (const auto& r : regions)
        out += ".region " + std::to_string(r.first) + ", \"" + r.second + "\"\n";
    }
    out += "\n";
  }

  for (std::size_t pc = 0; pc < program.code.size(); ++pc) {
    for (const auto& l : labels)
      if (l.first == pc) out += l.second + ":\n";

    const KeaInstr& in = program.code[pc];
    std::string line = "  " + disassembleInstruction(in, opts.map);

    if (opts.annotate) {
      std::string note = "  ; pc=" + std::to_string(pc);
      const OpDef* op = opDefFor(keaOpcode(in));
      if (op) {
        for (std::uint8_t i = 0; i < op->nfields; ++i) {
          const FieldDef& f = op->fields[i];
          if (f.units != FieldUnits::AddrSpmA && f.units != FieldUnits::AddrSpmW &&
              f.units != FieldUnits::AddrAcc && f.units != FieldUnits::AddrSpmByFlag)
            continue;
          const std::uint32_t v = static_cast<std::uint32_t>(fieldGet(in, f));
          const SpmBinding* b = spmCovering(program, toKeafSpace(fieldSpace(*op, f, in)), v);
          if (!b) continue;
          note += "  ";
          note += f.name;
          note += "=";
          note += b->name;
          if (v != b->offset) note += "+" + std::to_string(v - b->offset);
        }
      }
      line += note;
    }
    out += line;
    out += '\n';
  }
  return out;
}

}  // namespace rt
}  // namespace kea
