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
// RoundingMode enum (Level 1 quantization)
//===----------------------------------------------------------------------===//

// Which rounding rule the requantization multiply-shift uses.
//
//   SINGLE  round = 2^(shift-1) added once, then an arithmetic shift right.
//   DOUBLE  as SINGLE, plus TOSA's extra sign-dependent +/- 2^30 correction,
//           which by TOSA's own definition only has an effect when shift > 31.
//
// This exists as an enum rather than a `bool` parameter because a bare `bool`
// AttrDef parameter round-trips as `0`/`1` (FieldParser<bool> goes through the
// generic integral path), which is unreadable in the IR.
def Kea_RM_Single : I32EnumAttrCase<"SINGLE", 0, "SINGLE">;
def Kea_RM_Double : I32EnumAttrCase<"DOUBLE", 1, "DOUBLE">;

def Kea_RoundingMode : I32EnumAttr<"RoundingMode",
    "Rounding rule used by a KEA requantization",
    [Kea_RM_Single, Kea_RM_Double]> {
  let cppNamespace = "::mlir::kea";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// PoolKind enum
//===----------------------------------------------------------------------===//

def Kea_PK_Max : I32EnumAttrCase<"MAX", 0, "MAX">;
def Kea_PK_Avg : I32EnumAttrCase<"AVG", 1, "AVG">;

def Kea_PoolKind : I32EnumAttr<"PoolKind", "Pooling reduction kind",
    [Kea_PK_Max, Kea_PK_Avg]> {
  let cppNamespace = "::mlir::kea";
  let genSpecializedAttr = 0;
}

// Spelled `#kea.pool_kind<AVG>`.
def Kea_PoolKindAttr : EnumAttr<Kea_Dialect, Kea_PoolKind, "pool_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// #kea.zp -- operand zero points
//===----------------------------------------------------------------------===//

def Kea_ZeroPointAttr : Kea_Attr<"ZeroPoint", "zp"> {
  let summary = "Zero points of the two operands of a KEA contraction";
  let description = [{
    The affine zero points of the *inputs* of a contraction-like op, spelled

    ```mlir
    #kea.zp<input = -5, weight = 0>
    ```

    Role of each field per op:

    | op | `input` | `weight` |
    | --- | --- | --- |
    | `kea.conv2d` / `kea.dwconv2d` | activation zero point | weight zero point |
    | `kea.fully_connected` | activation zero point | weight zero point |
    | `kea.matmul` | `a` zero point | `b` zero point |

    Zero points on the *input* side are per tensor by construction: TOSA models
    them as scalars (`#tosa.conv_quant` / `#tosa.matmul_quant`), linalg models
    them as trailing scalar `i32` operands, and no PTQ toolchain emits a
    per-channel *activation* zero point. Per-channel information lives in
    `#kea.quant` (multiplier/shift/axis) instead.

    Symmetric weight quantization (`weight = 0`) is what per-channel PTQ
    produces and what the KEA MXU assumes; a non-zero `weight` is accepted and
    carried, but the backend will have to materialize the correction term.
  }];

  let parameters = (ins "int32_t":$input, "int32_t":$weight);
  let assemblyFormat = "`<` `input` `=` $input `,` `weight` `=` $weight `>`";
}

//===----------------------------------------------------------------------===//
// #kea.quant -- a requantization
//===----------------------------------------------------------------------===//

def Kea_QuantAttr : Kea_Attr<"Quant", "quant"> {
  let summary = "A multiply-shift requantization with zero points";
  let description = [{
    The single structured description of "move a tensor from one affine
    quantization grid to another". It is used

      * as the payload of `kea.rescale`,
      * as any of the four requantization slots of `#kea.epilogue`,
      * as the zero-point rebase of an average `kea.pool`.

    ```mlir
    #kea.quant<multiplier = [1073741824], shift = [30],
               input_zp = 0, output_zp = -128, axis = -1, rounding = DOUBLE>
    ```

    Semantics, per element, exactly matching TOSA 20.1.6 `tosa.rescale` with
    `scale32 = true` (see docs/TOSA_NOTES.md §4.1). `i` is the index along
    `axis` when `axis >= 0`, else `0`:

    ```
    v     = i32(in) - input_zp
    prod  = sext_i64(v) * sext_i64(multiplier[i])
    acc   = prod + (i64(1) << shift[i]) >> 1
    if (rounding == DOUBLE && shift[i] > 31)
        acc += (v >= 0) ? (i64(1) << 30) : -(i64(1) << 30)
    out   = clamp(i32(acc >> shift[i]) + output_zp, MIN(out_ty), MAX(out_ty))
    ```

    `axis = -1` means per tensor and requires exactly one multiplier/shift
    pair; `axis >= 0` means per channel and requires one pair per element of
    dimension `axis` of the tensor the attribute is attached to.

    Only the 32-bit multiplier path is modelled. TOSA's `scale32 = false`
    (16-bit multiplier, i32 accumulate) has no representation here and is
    rejected by `-tosa-to-kea`, because `KEA_VQUANT`'s multiplier is a
    normalized Q31 value (docs/ISA.md §10.1).
  }];

  let parameters = (ins
    "::mlir::DenseI32ArrayAttr":$multiplier,
    "::mlir::DenseI8ArrayAttr":$shift,
    "int32_t":$inputZp,
    "int32_t":$outputZp,
    "int64_t":$axis,
    EnumParameter<Kea_RoundingMode>:$rounding
  );

  let assemblyFormat = [{
    `<` `multiplier` `=` $multiplier `,` `shift` `=` $shift `,`
        `input_zp` `=` $inputZp `,` `output_zp` `=` $outputZp `,`
        `axis` `=` $axis `,` `rounding` `=` $rounding `>`
  }];

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// True when the multipliers are indexed by a tensor dimension.
    bool isPerChannel() const { return getAxis() >= 0; }
    /// Number of (multiplier, shift) pairs. 1 when per tensor.
    int64_t getNumScales() const { return getMultiplier().size(); }
    /// True when the multiply-shift is exactly the identity on every channel,
    /// i.e. `multiplier[i] == 1 << shift[i]`, so the only effect is the zero
    /// point rebase `out = in - input_zp + output_zp`.
    bool isIdentityScale() const;
    /// True when this requantization is a bit-exact no-op on values of type
    /// `elementType`: identity scale AND `input_zp == output_zp`.
    bool isNoOp() const {
      return isIdentityScale() && getInputZp() == getOutputZp();
    }
  }];
}

//===----------------------------------------------------------------------===//
// #kea.epilogue -- the fused VPU pass
//===----------------------------------------------------------------------===//

def Kea_EpilogueAttr : Kea_Attr<"Epilogue", "epilogue"> {
  let summary = "The post-accumulator work a KEA op performs in one VPU pass";
  let description = [{
    KEA's VPU turns an int32 accumulator into a storable int8 activation in a
    single pass (`KEA_VQUANT`, docs/ISA.md §10.1) and can fold a quantized
    residual add on top (`KEA_VADD`, §10.2). `#kea.epilogue` is the structured
    description of that pass, so a whole MobileNetV2 bottleneck collapses into
    three `kea` ops instead of thirteen `tosa` ones.

    Every field is optional, and `-kea-fuse` fills them in progressively: the
    conversion passes emit ops with **no** epilogue at all, then fusion adds
    `requant`, then `clamp`, then the residual triple.

    Stages, in the order they are applied to the raw accumulator `acc`
    (`bias` is an *operand* of the op, not part of this attribute):

    | field | stage |
    | --- | --- |
    | `requant` | `t = requant(acc + bias)`, saturating into the primary domain |
    | `clamp` | `t = clamp(t, clamp[0], clamp[1])`, the fused activation |
    | `accum` | `u = accum(t)`, into the shared add domain (i32) |
    | `residual` | `v = residual(%residual)`, into the same shared add domain |
    | `output` | `out = output(u + v)`, back to the result element type |

    `accum`, `residual` and `output` are all-or-nothing and require the op's
    optional `residual` operand to be present; that triple is exactly
    `KeaAddParam`'s `{a_mult,a_shift,a_zp}` / `{b_...}` / `{o_...}`.

    Without `requant` the op produces its raw `i32` accumulator and no other
    field may be set.

    ```mlir
    #kea.epilogue<requant = <multiplier = [1073741824], shift = [30],
                             input_zp = 0, output_zp = -128, axis = -1,
                             rounding = DOUBLE>,
                  clamp = [-128, 127]>
    ```

    Note the nested `#kea.quant` prints without its `#kea.quant` prefix
    (`printStrippedAttrOrType`); both spellings parse.

    ```mlir
    ```
  }];

  let parameters = (ins
    OptionalParameter<"::mlir::kea::QuantAttr">:$requant,
    OptionalParameter<"::mlir::DenseI64ArrayAttr">:$clamp,
    OptionalParameter<"::mlir::kea::QuantAttr">:$accum,
    OptionalParameter<"::mlir::kea::QuantAttr">:$residual,
    OptionalParameter<"::mlir::kea::QuantAttr">:$output
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// True when the residual triple (accum/residual/output) is populated.
    bool hasResidual() const {
      return getAccum() && getResidual() && getOutput();
    }
    /// The element-type domain the op's result lives in is decided by the last
    /// populated stage; true when nothing at all is set.
    bool isEmpty() const {
      return !getRequant() && !getClamp() && !getAccum() && !getResidual() &&
             !getOutput();
    }
  }];
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
