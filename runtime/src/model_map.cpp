// SPDX-License-Identifier: Apache-2.0
#include "kea/rt/model_map.h"

#include <cstdio>
#include <fstream>
#include <sstream>

namespace kea {
namespace rt {

const MapSymbol* ModelMap::find(const std::string& name) const {
  for (const MapSymbol& s : symbols)
    if (s.name == name) return &s;
  return nullptr;
}

const MapSymbol* ModelMap::covering(std::uint64_t addr) const {
  const MapSymbol* best = nullptr;
  for (const MapSymbol& s : symbols) {
    if (addr < s.offset) continue;
    const std::uint64_t end = s.offset + (s.size ? s.size : 1);
    if (addr >= end) continue;
    if (!best || s.size < best->size) best = &s;
  }
  return best;
}

// ---------------------------------------------------------------------------
// Enum spellings
// ---------------------------------------------------------------------------

bool parseDType(const std::string& s, KeafDType& out) {
  if (s == "int8") { out = KeafDType::INT8; return true; }
  if (s == "uint8") { out = KeafDType::UINT8; return true; }
  if (s == "int4") { out = KeafDType::INT4; return true; }
  if (s == "int32") { out = KeafDType::INT32; return true; }
  if (s == "fp32") { out = KeafDType::FP32; return true; }
  if (s == "int16") { out = KeafDType::INT16; return true; }
  return false;
}

bool parseTensorKind(const std::string& s, KeafTensorKind& out) {
  if (s == "input") { out = KeafTensorKind::INPUT; return true; }
  if (s == "output") { out = KeafTensorKind::OUTPUT; return true; }
  if (s == "const") { out = KeafTensorKind::CONST; return true; }
  if (s == "scratch") { out = KeafTensorKind::SCRATCH; return true; }
  return false;
}

bool parseLayout(const std::string& s, KeafLayout& out) {
  if (s == "NHWC") { out = KeafLayout::NHWC; return true; }
  if (s == "NCHW") { out = KeafLayout::NCHW; return true; }
  if (s == "FLAT") { out = KeafLayout::FLAT; return true; }
  return false;
}

bool parseSpmSpace(const std::string& s, KeafSpace& out) {
  if (s == "SPM_A") { out = KeafSpace::SPM_A; return true; }
  if (s == "SPM_W") { out = KeafSpace::SPM_W; return true; }
  if (s == "ACC") { out = KeafSpace::ACC; return true; }
  return false;
}

const char* layoutName(KeafLayout l) {
  switch (l) {
    case KeafLayout::NHWC: return "NHWC";
    case KeafLayout::NCHW: return "NCHW";
    case KeafLayout::FLAT: return "FLAT";
  }
  return "NHWC";
}

const char* spmSpaceName(KeafSpace s) {
  switch (s) {
    case KeafSpace::SPM_A: return "SPM_A";
    case KeafSpace::SPM_W: return "SPM_W";
    case KeafSpace::ACC: return "ACC";
  }
  return "SPM_A";
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

namespace {

class MapReader {
 public:
  MapReader(const std::string& file, Diagnostics& diags) : file_(file), diags_(diags) {}

  bool err(const std::string& path, const std::string& msg) {
    diags_.error(file_, 0, 0, 1, path + ": " + msg);
    return false;
  }

  bool wantObject(const Json& v, const std::string& path) {
    if (!v.isObject()) return err(path, "expected a JSON object");
    return true;
  }

  bool getUInt(const Json& obj, const char* key, const std::string& path, bool required,
               std::uint64_t& out) {
    const Json* v = obj.find(key);
    if (!v) {
      if (!required) return true;
      return err(path + "." + key, "required key is missing");
    }
    if (!v->isInt() || v->asInt() < 0)
      return err(path + "." + key, "expected a non-negative integer");
    out = static_cast<std::uint64_t>(v->asInt());
    return true;
  }

  bool getString(const Json& obj, const char* key, const std::string& path, bool required,
                 std::string& out) {
    const Json* v = obj.find(key);
    if (!v) {
      if (!required) return true;
      return err(path + "." + key, "required key is missing");
    }
    if (!v->isString()) return err(path + "." + key, "expected a string");
    out = v->asString();
    return true;
  }

 private:
  std::string file_;
  Diagnostics& diags_;
};

}  // namespace

bool parseModelMap(const std::string& text, const std::string& filename, ModelMap& out,
                   Diagnostics& diags) {
  out = ModelMap{};
  Json root;
  std::string jerr;
  if (!Json::parse(text, root, jerr)) {
    // jerr already carries "line:col: message".
    const std::size_t c1 = jerr.find(':');
    const std::size_t c2 = (c1 == std::string::npos) ? std::string::npos : jerr.find(':', c1 + 1);
    int line = 0, col = 0;
    std::string msg = jerr;
    if (c2 != std::string::npos) {
      line = std::atoi(jerr.substr(0, c1).c_str());
      col = std::atoi(jerr.substr(c1 + 1, c2 - c1 - 1).c_str());
      msg = jerr.substr(c2 + 2);
    }
    diags.addSource(filename, text);
    diags.error(filename, line, col, 1, "invalid JSON: " + msg);
    return false;
  }

  MapReader r(filename, diags);
  if (!r.wantObject(root, "<root>")) return false;

  if (const Json* arch = root.find("arch")) {
    if (!arch->isString() || arch->asString() != KEA_ARCH_NAME)
      return r.err("arch", std::string("expected \"") + KEA_ARCH_NAME + "\"");
  }
  if (const Json* rev = root.find("isa_revision")) {
    if (!rev->isInt() || rev->asInt() != static_cast<std::int64_t>(KEA_ISA_REVISION))
      return r.err("isa_revision",
                   "artifact targets ISA revision " + std::to_string(rev->asInt()) +
                       ", this build implements " + std::to_string(KEA_ISA_REVISION));
  }
  if (const Json* pc = root.find("entry_pc")) {
    if (!pc->isInt() || pc->asInt() < 0) return r.err("entry_pc", "expected a non-negative integer");
    out.entry_pc = static_cast<std::uint32_t>(pc->asInt());
  }

  // --- dram -----------------------------------------------------------------
  out.dram.alignment = KEA_DRAM_BASE_ALIGN;
  if (const Json* d = root.find("dram")) {
    if (!r.wantObject(*d, "dram")) return false;
    std::uint64_t v = 0;
    if (!r.getUInt(*d, "total_bytes", "dram", true, v)) return false;
    out.dram.total_bytes = v;
    v = 0; if (!r.getUInt(*d, "const_offset", "dram", false, v)) return false; out.dram.const_offset = v;
    v = 0; if (!r.getUInt(*d, "const_bytes", "dram", false, v)) return false; out.dram.const_bytes = v;
    v = 0; if (!r.getUInt(*d, "io_offset", "dram", false, v)) return false; out.dram.io_offset = v;
    v = 0; if (!r.getUInt(*d, "io_bytes", "dram", false, v)) return false; out.dram.io_bytes = v;
    v = 0; if (!r.getUInt(*d, "scratch_offset", "dram", false, v)) return false; out.dram.scratch_offset = v;
    v = 0; if (!r.getUInt(*d, "scratch_bytes", "dram", false, v)) return false; out.dram.scratch_bytes = v;
    v = KEA_DRAM_BASE_ALIGN;
    if (!r.getUInt(*d, "alignment", "dram", false, v)) return false;
    if (v < KEA_DRAM_BASE_ALIGN)
      return r.err("dram.alignment", "must be at least KEA_DRAM_BASE_ALIGN (" +
                                         std::to_string(KEA_DRAM_BASE_ALIGN) + ")");
    out.dram.alignment = static_cast<std::uint32_t>(v);
  } else {
    return r.err("dram", "required key is missing");
  }

  if (out.dram.total_bytes > KEA_DRAM_BYTES)
    return r.err("dram.total_bytes", "exceeds the 4 GiB DRAM address space");
  if (out.dram.const_offset + out.dram.const_bytes > out.dram.total_bytes)
    return r.err("dram", "the const region escapes the arena");
  if (out.dram.io_offset + out.dram.io_bytes > out.dram.total_bytes)
    return r.err("dram", "the io region escapes the arena");
  if (out.dram.scratch_offset + out.dram.scratch_bytes > out.dram.total_bytes)
    return r.err("dram", "the scratch region escapes the arena");

  // --- symbols --------------------------------------------------------------
  if (const Json* syms = root.find("symbols")) {
    if (!syms->isArray()) return r.err("symbols", "expected an array");
    for (std::size_t i = 0; i < syms->size(); ++i) {
      const Json& s = *syms->at(i);
      const std::string path = "symbols[" + std::to_string(i) + "]";
      if (!r.wantObject(s, path)) return false;
      MapSymbol sym;
      if (!r.getString(s, "name", path, true, sym.name)) return false;
      if (!r.getUInt(s, "offset", path, true, sym.offset)) return false;
      if (!r.getUInt(s, "size", path, false, sym.size)) return false;
      if (sym.offset + sym.size > out.dram.total_bytes)
        return r.err(path, "symbol '" + sym.name + "' escapes the DRAM arena");
      if (out.find(sym.name)) return r.err(path, "duplicate symbol '" + sym.name + "'");
      out.symbols.push_back(sym);
    }
  }

  // --- tensors --------------------------------------------------------------
  if (const Json* ts = root.find("tensors")) {
    if (!ts->isArray()) return r.err("tensors", "expected an array");
    for (std::size_t i = 0; i < ts->size(); ++i) {
      const Json& t = *ts->at(i);
      const std::string path = "tensors[" + std::to_string(i) + "]";
      if (!r.wantObject(t, path)) return false;
      TensorBinding tb;
      if (!r.getString(t, "name", path, true, tb.name)) return false;
      if (tb.name.size() >= KEAF_TENSOR_NAME_BYTES)
        return r.err(path + ".name", "tensor names are limited to " +
                                         std::to_string(KEAF_TENSOR_NAME_BYTES - 1) +
                                         " characters");
      std::uint64_t v = 0;
      if (!r.getUInt(t, "offset", path, true, v)) return false;
      tb.dram_offset = v;
      v = 0;
      if (!r.getUInt(t, "size_bytes", path, true, v)) return false;
      tb.size_bytes = v;
      if (tb.dram_offset + tb.size_bytes > out.dram.total_bytes)
        return r.err(path, "tensor '" + tb.name + "' escapes the DRAM arena");

      std::string s;
      if (!r.getString(t, "dtype", path, true, s)) return false;
      if (!parseDType(s, tb.dtype))
        return r.err(path + ".dtype",
                     "unknown dtype '" + s + "'; expected one of int8, uint8, int4, int32, fp32, int16");
      s.clear();
      if (!r.getString(t, "kind", path, true, s)) return false;
      if (!parseTensorKind(s, tb.kind))
        return r.err(path + ".kind",
                     "unknown kind '" + s + "'; expected one of input, output, const, scratch");
      s = "NHWC";
      if (!r.getString(t, "layout", path, false, s)) return false;
      if (!parseLayout(s, tb.layout))
        return r.err(path + ".layout", "unknown layout '" + s + "'; expected NHWC, NCHW or FLAT");

      if (const Json* sh = t.find("shape")) {
        if (!sh->isArray()) return r.err(path + ".shape", "expected an array of integers");
        if (sh->size() > KEAF_MAX_RANK)
          return r.err(path + ".shape",
                       "rank " + std::to_string(sh->size()) + " exceeds KEAF_MAX_RANK (" +
                           std::to_string(KEAF_MAX_RANK) + ")");
        for (std::size_t k = 0; k < sh->size(); ++k) {
          const Json& e = *sh->at(k);
          if (!e.isInt()) return r.err(path + ".shape", "expected an array of integers");
          tb.shape.push_back(static_cast<std::int32_t>(e.asInt()));
        }
      }
      if (const Json* sc = t.find("scale")) {
        if (!sc->isNumber()) return r.err(path + ".scale", "expected a number");
        tb.scale = static_cast<float>(sc->asDouble());
      }
      if (const Json* zp = t.find("zero_point")) {
        if (!zp->isInt()) return r.err(path + ".zero_point", "expected an integer");
        tb.zero_point = static_cast<std::int32_t>(zp->asInt());
      }
      if (const Json* ix = t.find("index")) {
        if (!ix->isInt() || ix->asInt() < 0) return r.err(path + ".index", "expected a non-negative integer");
        tb.index = static_cast<std::uint32_t>(ix->asInt());
      } else {
        std::uint32_t n = 0;
        for (const TensorBinding& prev : out.tensors)
          if (prev.kind == tb.kind) ++n;
        tb.index = n;
      }
      if (out.find(tb.name))
        return r.err(path, "'" + tb.name + "' collides with an earlier symbol or tensor name");
      MapSymbol sym;
      sym.name = tb.name;
      sym.offset = tb.dram_offset;
      sym.size = tb.size_bytes;
      sym.from_tensor = true;
      out.symbols.push_back(sym);
      out.tensors.push_back(std::move(tb));
    }
  }

  // --- spm_map (debug) ------------------------------------------------------
  if (const Json* ss = root.find("spm_map")) {
    if (!ss->isArray()) return r.err("spm_map", "expected an array");
    for (std::size_t i = 0; i < ss->size(); ++i) {
      const Json& e = *ss->at(i);
      const std::string path = "spm_map[" + std::to_string(i) + "]";
      if (!r.wantObject(e, path)) return false;
      SpmBinding sb;
      if (!r.getString(e, "name", path, true, sb.name)) return false;
      if (sb.name.size() >= KEAF_SPM_NAME_BYTES)
        return r.err(path + ".name", "SPM buffer names are limited to " +
                                         std::to_string(KEAF_SPM_NAME_BYTES - 1) + " characters");
      std::string s;
      if (!r.getString(e, "space", path, true, s)) return false;
      if (!parseSpmSpace(s, sb.space))
        return r.err(path + ".space", "unknown space '" + s + "'; expected SPM_A, SPM_W or ACC");
      std::uint64_t v = 0;
      if (!r.getUInt(e, "offset", path, true, v)) return false;
      sb.offset = static_cast<std::uint32_t>(v);
      v = 0;
      if (!r.getUInt(e, "size", path, true, v)) return false;
      sb.size = static_cast<std::uint32_t>(v);
      v = 0;
      if (!r.getUInt(e, "first_pc", path, false, v)) return false;
      sb.first_pc = static_cast<std::uint32_t>(v);
      v = 0;
      if (!r.getUInt(e, "last_pc", path, false, v)) return false;
      sb.last_pc = static_cast<std::uint32_t>(v);
      out.spm_map.push_back(std::move(sb));
    }
  }

  // --- metadata -------------------------------------------------------------
  if (const Json* md = root.find("metadata")) {
    if (!md->isObject()) return r.err("metadata", "expected a JSON object");
    out.metadata_json = md->dump(2);
  }
  return true;
}

bool loadModelMapFile(const std::string& path, ModelMap& out, Diagnostics& diags) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    diags.error("cannot open map file '" + path + "'");
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return parseModelMap(ss.str(), path, out, diags);
}

std::string dumpModelMap(const ModelMap& map) {
  Json root = Json::object();
  root.set("arch", Json(KEA_ARCH_NAME));
  root.set("isa_revision", Json(static_cast<std::int64_t>(KEA_ISA_REVISION)));
  root.set("entry_pc", Json(static_cast<std::int64_t>(map.entry_pc)));

  Json d = Json::object();
  d.set("total_bytes", Json(static_cast<std::int64_t>(map.dram.total_bytes)));
  d.set("const_offset", Json(static_cast<std::int64_t>(map.dram.const_offset)));
  d.set("const_bytes", Json(static_cast<std::int64_t>(map.dram.const_bytes)));
  d.set("io_offset", Json(static_cast<std::int64_t>(map.dram.io_offset)));
  d.set("io_bytes", Json(static_cast<std::int64_t>(map.dram.io_bytes)));
  d.set("scratch_offset", Json(static_cast<std::int64_t>(map.dram.scratch_offset)));
  d.set("scratch_bytes", Json(static_cast<std::int64_t>(map.dram.scratch_bytes)));
  d.set("alignment", Json(static_cast<std::int64_t>(map.dram.alignment)));
  root.set("dram", std::move(d));

  Json syms = Json::array();
  for (const MapSymbol& s : map.symbols) {
    if (s.from_tensor) continue;
    Json o = Json::object();
    o.set("name", Json(s.name));
    o.set("offset", Json(static_cast<std::int64_t>(s.offset)));
    o.set("size", Json(static_cast<std::int64_t>(s.size)));
    syms.push_back(std::move(o));
  }
  root.set("symbols", std::move(syms));

  Json ts = Json::array();
  for (const TensorBinding& t : map.tensors) {
    Json o = Json::object();
    o.set("name", Json(t.name));
    o.set("kind", Json(keafTensorKindName(t.kind)));
    o.set("index", Json(static_cast<std::int64_t>(t.index)));
    o.set("offset", Json(static_cast<std::int64_t>(t.dram_offset)));
    o.set("size_bytes", Json(static_cast<std::int64_t>(t.size_bytes)));
    Json shape = Json::array();
    for (std::int32_t dim : t.shape) shape.push_back(Json(static_cast<std::int64_t>(dim)));
    o.set("shape", std::move(shape));
    o.set("dtype", Json(keafDTypeName(t.dtype)));
    o.set("layout", Json(layoutName(t.layout)));
    o.set("scale", Json(static_cast<double>(t.scale)));
    o.set("zero_point", Json(static_cast<std::int64_t>(t.zero_point)));
    ts.push_back(std::move(o));
  }
  root.set("tensors", std::move(ts));

  if (!map.spm_map.empty()) {
    Json ss = Json::array();
    for (const SpmBinding& s : map.spm_map) {
      Json o = Json::object();
      o.set("name", Json(s.name));
      o.set("space", Json(spmSpaceName(s.space)));
      o.set("offset", Json(static_cast<std::int64_t>(s.offset)));
      o.set("size", Json(static_cast<std::int64_t>(s.size)));
      o.set("first_pc", Json(static_cast<std::int64_t>(s.first_pc)));
      o.set("last_pc", Json(static_cast<std::int64_t>(s.last_pc)));
      ss.push_back(std::move(o));
    }
    root.set("spm_map", std::move(ss));
  }

  if (!map.metadata_json.empty()) {
    Json md;
    std::string err;
    if (Json::parse(map.metadata_json, md, err) && md.isObject()) root.set("metadata", std::move(md));
  }
  return root.dump(2) + "\n";
}

ModelMap modelMapFromProgram(const KeaProgram& program) {
  ModelMap m;
  m.dram = program.dram;
  m.tensors = program.tensors;
  m.spm_map = program.spm_map;
  m.entry_pc = program.entry_pc;
  m.metadata_json = program.metadata_json;
  for (const TensorBinding& t : program.tensors) {
    MapSymbol s;
    s.name = t.name;
    s.offset = t.dram_offset;
    s.size = t.size_bytes;
    s.from_tensor = true;
    m.symbols.push_back(std::move(s));
  }
  return m;
}

}  // namespace rt
}  // namespace kea
