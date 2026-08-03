//===- KeaOps.td - KEA Level 1 (tensor level) operations ---*- tablegen -*-===//
//
// This file is the TableGen entry point consumed by `add_mlir_dialect(KeaOps
// kea)`, which generates *all* of:
//   KeaOps.h.inc / KeaOps.cpp.inc               (-gen-op-{decls,defs})
//   KeaOpsTypes.h.inc / KeaOpsTypes.cpp.inc     (-gen-typedef-{decls,defs})
//   KeaOpsDialect.h.inc / KeaOpsDialect.cpp.inc (-gen-dialect-{decls,defs})
// which is why it must transitively include KeaTypes.td.
//
// SCOPE: this file holds **Level 1 only** -- the value-semantic tensor ops that
// `-tosa-to-kea` / `-linalg-to-kea` produce and `-kea-fuse` rewrites. The
// buffer-level ops that map 1:1 onto KEA instructions live in
// KeaMachineOps.td, which has its own TableGen target. See
// docs/adr/0002-two-level-kea-dialect.md.
//
// ASSEMBLY NOTE: the optional `bias` / `residual` operands are introduced by a
// bare keyword with NO leading comma (`%in, %w bias %b residual %r`). A
// comma-led optional group (`(`,` `bias` $bias^)?`) commits to the group as
// soon as it sees the comma, so `%in, %w, residual %r` -- bias absent, residual
// present, which -kea-fuse produces for a bias-less conv -- would fail to parse
// with "expected 'bias'".
//
// CANONICAL LAYOUTS (enforced by the verifiers, see docs/DIALECT_L1.md §3):
//   activations   NHWC          [N, H, W, C]
//   conv weights  OHWI          [OC, KH, KW, IC]
//   dwconv weights OHWI, IC=1   [OC = C*M, KH, KW, 1]   <-- normalised from
//                                                            TOSA's HWCM
//   bias          rank 1 i32    [OC]
//   pads          [top, bottom, left, right]
//   strides/dilations [h, w]
//
//===----------------------------------------------------------------------===//

#ifndef KEA_OPS
#define KEA_OPS

include "mlir/IR/OpBase.td"
// SameOperandsAndResultType lives here, not in OpBase.td, since MLIR 16.
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "kea/Dialect/KeaDialect.td"
include "kea/Dialect/KeaAttrs.td"
include "kea/Dialect/KeaTypes.td"

//===----------------------------------------------------------------------===//
// Shared fragments
//===----------------------------------------------------------------------===//

// Every contraction-like Level 1 op has the same optional bias / optional
// residual / epilogue shape, so factor it out.
class Kea_L1Op<string mnemonic, list<Trait> traits = []>
    : Kea_Op<mnemonic, !listconcat(traits, [Pure, AttrSizedOperandSegments])>;

//===----------------------------------------------------------------------===//
// Contractions
//===----------------------------------------------------------------------===//

def Kea_Conv2DOp : Kea_L1Op<"conv2d"> {
  let summary = "Quantized NHWC/OHWI 2-D convolution with a fused epilogue";
  let description = [{
    The canonical dense convolution of Level 1, the target of `tosa.conv2d` and
    `linalg.conv_2d_nhwc_fhwc_q`.

    * `input`  is NHWC, `weights` are OHWI `[OC, KH, KW, IC]`.
    * `bias` is optional, rank 1, `i32`, length `OC`. It is added into the
      accumulator *at accumulator scale*, before the epilogue.
    * `residual` is optional and, when present, is added in the epilogue's
      shared i32 domain -- this is the MobileNetV2 inverted-residual skip.
    * `zero_points` carries the activation and weight zero points.
    * `epilogue` carries requantize / activation clamp / residual quantization.
      With no `epilogue` the result is the raw `i32` accumulator.

    ```mlir
    %0 = kea.conv2d %in, %w bias %b {
           zero_points = #kea.zp<input = -5, weight = 0>,
           strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
           dilations = array<i64: 1, 1>}
       : (tensor<1x8x8x4xi8>, tensor<24x1x1x4xi8>, tensor<24xi32>)
         -> tensor<1x8x8x24xi32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weights,
    Optional<AnyRankedTensor>:$bias,
    Optional<AnyRankedTensor>:$residual,
    Kea_ZeroPointAttr:$zero_points,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$pads,
    DenseI64ArrayAttr:$dilations,
    OptionalAttr<Kea_EpilogueAttr>:$epilogue
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input `,` $weights (`bias` $bias^)? (`residual` $residual^)?
      attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def Kea_DWConv2DOp : Kea_L1Op<"dwconv2d"> {
  let summary = "Quantized NHWC depthwise convolution with a fused epilogue";
  let description = [{
    Depthwise convolution with **canonical** weights `[OC, KH, KW, 1]`, where
    `OC = C * M` and `M` is the channel multiplier. Output channel `oc` reads
    input channel `oc / M`.

    TOSA spells depthwise weights HWCM `[KH, KW, C, M]`; `-tosa-to-kea`
    normalises them here so every downstream pass has exactly one weight
    layout to handle. See docs/DIALECT_L1.md §3.2 for the index algebra.

    Apart from the weight shape rule this op is identical to `kea.conv2d`.

    ```mlir
    %0 = kea.dwconv2d %in, %w bias %b {
           zero_points = #kea.zp<input = -128, weight = 0>,
           strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
           dilations = array<i64: 1, 1>}
       : (tensor<1x8x8x24xi8>, tensor<24x3x3x1xi8>, tensor<24xi32>)
         -> tensor<1x8x8x24xi32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weights,
    Optional<AnyRankedTensor>:$bias,
    Optional<AnyRankedTensor>:$residual,
    Kea_ZeroPointAttr:$zero_points,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$pads,
    DenseI64ArrayAttr:$dilations,
    OptionalAttr<Kea_EpilogueAttr>:$epilogue
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input `,` $weights (`bias` $bias^)? (`residual` $residual^)?
      attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def Kea_MatmulOp : Kea_L1Op<"matmul"> {
  let summary = "Quantized batched matrix multiply with a fused epilogue";
  let description = [{
    `out[b, m, n] = sum_k (a[b, m, k] - a_zp) * (b[b, k, n] - b_zp)`, all
    operands strictly rank 3, matching `tosa.matmul`. `zero_points.input` is
    the `a` zero point and `zero_points.weight` is the `b` zero point.

    `bias`, when present, is rank 1 of length `N` and is broadcast over
    `[batch, M]`.

    ```mlir
    %0 = kea.matmul %a, %b {zero_points = #kea.zp<input = -2, weight = 1>}
       : (tensor<1x4x8xi8>, tensor<1x8x16xi8>) -> tensor<1x4x16xi32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$a,
    AnyRankedTensor:$b,
    Optional<AnyRankedTensor>:$bias,
    Optional<AnyRankedTensor>:$residual,
    Kea_ZeroPointAttr:$zero_points,
    OptionalAttr<Kea_EpilogueAttr>:$epilogue
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $a `,` $b (`bias` $bias^)? (`residual` $residual^)?
      attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def Kea_FullyConnectedOp : Kea_L1Op<"fully_connected"> {
  let summary = "Quantized 2-D fully-connected layer with a fused epilogue";
  let description = [{
    `out[n, oc] = sum_ic (a[n, ic] - a_zp) * (w[oc, ic] - w_zp)`.

    The weight is `[OC, IC]`, i.e. **already transposed** relative to a matmul
    right-hand side. That is TOSA's `tosa.fully_connected` layout and also the
    layout a PyTorch `nn.Linear` exports, so it is kept as-is rather than
    forcing a transpose at ingest.

    ```mlir
    %0 = kea.fully_connected %a, %w bias %b {
           zero_points = #kea.zp<input = -128, weight = 0>}
       : (tensor<1x64xi8>, tensor<10x64xi8>, tensor<10xi32>) -> tensor<1x10xi32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weights,
    Optional<AnyRankedTensor>:$bias,
    Optional<AnyRankedTensor>:$residual,
    Kea_ZeroPointAttr:$zero_points,
    OptionalAttr<Kea_EpilogueAttr>:$epilogue
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input `,` $weights (`bias` $bias^)? (`residual` $residual^)?
      attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Elementwise
//===----------------------------------------------------------------------===//

def Kea_AddOp : Kea_Op<"add", [Pure]> {
  let summary = "Elementwise add, optionally fully quantized";
  let description = [{
    With no quantization attributes this is a plain elementwise add on equal
    element types (TOSA-style rank-equal broadcasting of size-1 dimensions is
    allowed).

    With `lhs_quant` / `rhs_quant` / `out_quant` it is the normalised form of
    TOSA's rescale-sandwich idiom, and maps 1:1 onto `KEA_VADD`:

    ```
    u   = lhs_quant(lhs)          // into the shared i32 domain
    v   = rhs_quant(rhs)          // into the same shared i32 domain
    out = clamp(out_quant(u + v))
    ```

    The three quantizations are all-or-nothing.

    ```mlir
    %0 = kea.add %a, %b {
      lhs_quant = #kea.quant<multiplier = [1073741824], shift = [10],
                             input_zp = -5, output_zp = 0, axis = -1,
                             rounding = DOUBLE>,
      rhs_quant = #kea.quant<...>, out_quant = #kea.quant<...>
    } : (tensor<1x8x8x4xi8>, tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi8>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    OptionalAttr<Kea_QuantAttr>:$lhs_quant,
    OptionalAttr<Kea_QuantAttr>:$rhs_quant,
    OptionalAttr<Kea_QuantAttr>:$out_quant,
    OptionalAttr<DenseI64ArrayAttr>:$clamp
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def Kea_ClampOp : Kea_Op<"clamp", [Pure, SameOperandsAndResultType]> {
  let summary = "Saturating clamp to an integer interval";
  let description = [{
    `out = min(max(in, min), max)` in the tensor's own element type. As in
    TOSA, **no zero-point subtraction happens**: to express a ReLU on a tensor
    whose zero point is `zp`, pass `min = zp`.

    `-kea-fuse` folds this into the producing op's `#kea.epilogue` `clamp`
    field whenever the producer is a KEA contraction with a `requant` stage.

    ```mlir
    %0 = kea.clamp %a {min = -128 : i64, max = 127 : i64}
       : tensor<1x8x8x24xi8>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$min,
    I64Attr:$max
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($output)";

  let hasVerifier = 1;
}

def Kea_RescaleOp : Kea_Op<"rescale", [Pure]> {
  let summary = "Standalone requantization";
  let description = [{
    Applies a `#kea.quant` to a tensor. This is the un-fused form; the whole
    point of `-kea-fuse` is to make most of these disappear into the producing
    op's `#kea.epilogue`.

    ```mlir
    %0 = kea.rescale %acc {
           quant = #kea.quant<multiplier = [1073741824], shift = [30],
                              input_zp = 0, output_zp = -5, axis = -1,
                              rounding = DOUBLE>}
       : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$input, Kea_QuantAttr:$quant);
  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Pooling
//===----------------------------------------------------------------------===//

def Kea_PoolOp : Kea_Op<"pool", [Pure]> {
  let summary = "NHWC max or average pooling";
  let description = [{
    `kind` selects max or average. Element type is preserved.

    Average pooling divides by the **real** (padding-aware) window count, as
    TOSA does -- see docs/TOSA_NOTES.md §7. Its optional `quant` carries the
    zero-point rebase `out = in - input_zp + output_zp`; because averaging is
    affine, that is exact, and the verifier therefore requires the multiplier
    to be the identity (`multiplier[i] == 1 << shift[i]`). A general rescale
    after a pool must be written as a separate `kea.rescale`.

    Max pooling is monotonic and carries no quantization at all.

    ```mlir
    %0 = kea.pool %a {kind = #kea.pool_kind<AVG>, kernel = array<i64: 2, 2>,
                      strides = array<i64: 2, 2>, pads = array<i64: 0, 0, 0, 0>}
       : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    Kea_PoolKindAttr:$kind,
    DenseI64ArrayAttr:$kernel,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$pads,
    OptionalAttr<Kea_QuantAttr>:$quant
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Shape manipulation
//===----------------------------------------------------------------------===//

def Kea_ReshapeOp : Kea_Op<"reshape", [Pure]> {
  let summary = "Element-order-preserving reshape";
  let description = [{
    Row-major reinterpretation of the same elements. Unlike `tosa.reshape` the
    `new_shape` is fully resolved -- a `-1` placeholder is illegal at Level 1,
    `-tosa-to-kea` substitutes the inferred extent.

    ```mlir
    %0 = kea.reshape %a {new_shape = array<i64: 1, 1568>}
       : (tensor<1x7x7x32xi8>) -> tensor<1x1568xi8>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$input, DenseI64ArrayAttr:$new_shape);
  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def Kea_TransposeOp : Kea_Op<"transpose", [Pure]> {
  let summary = "Dimension permutation";
  let description = [{
    Same convention as `tosa.transpose`:
    `result_shape[d] = input_shape[perms[d]]`, and the element at input index
    `i` moves to the output index obtained by reading `i` through `perms`.

    TOSA passes the permutation as a rank-1 `i32` **operand** in 20.1.6;
    `-tosa-to-kea` folds that constant into this attribute, and fails cleanly
    if it is not constant.

    ```mlir
    %0 = kea.transpose %a {perms = array<i64: 0, 3, 1, 2>}
       : (tensor<1x8x8x16xi8>) -> tensor<1x16x8x8xi8>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$input, DenseI64ArrayAttr:$perms);
  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

#endif // KEA_OPS
