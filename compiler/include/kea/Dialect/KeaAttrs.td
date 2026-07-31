//===- KeaAttrs.td - KEA attributes and enums --------------*- tablegen -*-===//
//
// Enums and AttrDefs. This file is a TableGen *entry point*: the CMake rules in
// this directory run `-gen-enum-{decls,defs}` and `-gen-attrdef-{decls,defs}`
// on it, so anything defined here must be self-contained (only include
// KeaDialect.td and upstream .td files).
//
//===----------------------------------------------------------------------===//

#ifndef KEA_ATTRS
#define KEA_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "kea/Dialect/KeaDialect.td"

//===----------------------------------------------------------------------===//
// AddressSpace enum
//===----------------------------------------------------------------------===//

// Which physical memory a `!kea.buffer` lives in.
def Kea_AS_A    : I32EnumAttrCase<"A",    0, "A">;     // activation scratchpad
def Kea_AS_W    : I32EnumAttrCase<"W",    1, "W">;     // weight scratchpad
def Kea_AS_ACC  : I32EnumAttrCase<"ACC",  2, "ACC">;   // accumulator bank
def Kea_AS_DRAM : I32EnumAttrCase<"DRAM", 3, "DRAM">;  // off-chip

def Kea_AddressSpace : I32EnumAttr<"AddressSpace",
    "KEA NPU memory space",
    [Kea_AS_A, Kea_AS_W, Kea_AS_ACC, Kea_AS_DRAM]> {
  let cppNamespace = "::mlir::kea";
  // We want the plain C++ enum plus a hand-rolled `EnumAttr` wrapper below,
  // not the legacy auto-generated `AddressSpaceAttr`. Leaving this at 1 would
  // generate a second, differently-spelled attribute class.
  let genSpecializedAttr = 0;
}

// The attribute form, spelled `#kea.address_space<ACC>`. Note that when this
// enum is used as a *type parameter* (see KeaTypes.td) it is spelled as a bare
// keyword instead, because `EnumParameter` prints/parses the enum directly.
def Kea_AddressSpaceAttr : EnumAttr<Kea_Dialect, Kea_AddressSpace, "address_space"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// TileConfig attribute
//===----------------------------------------------------------------------===//

// Example of a plain AttrDef with a declarative assembly format. The real
// compiler will grow scheduling/quantization attributes exactly like this.
def Kea_TileConfigAttr : Kea_Attr<"TileConfig", "tile_config"> {
  let summary = "Systolic-array tiling for a KEA op";
  let description = [{
    Describes how an op is decomposed onto the PE array. Spelled
    `#kea.tile_config<16, 16, 32>` for (rows, cols, depth).

    GOTCHA: do NOT use `x` as the separator here (`16x16x32`). The MLIR lexer
    tokenizes `16x16x32` as the integer `16` followed by the bare identifier
    `x16x32`, so a `` `x` `` literal in the format can never match. Only the
    dedicated `parseDimensionList` API (see custom<Shape> in KeaTypes.td)
    handles the `AxBxC` spelling.
  }];

  let parameters = (ins "int64_t":$rows, "int64_t":$cols, "int64_t":$depth);
  let assemblyFormat = "`<` $rows `,` $cols `,` $depth `>`";
  let genVerifyDecl = 1;
}

#endif // KEA_ATTRS
