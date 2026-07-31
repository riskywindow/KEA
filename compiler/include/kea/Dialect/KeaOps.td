//===- KeaOps.td - KEA operations --------------------------*- tablegen -*-===//
//
// This file is the TableGen entry point consumed by `add_mlir_dialect(KeaOps
// kea)`, which generates *all* of:
//   KeaOps.h.inc / KeaOps.cpp.inc               (-gen-op-{decls,defs})
//   KeaOpsTypes.h.inc / KeaOpsTypes.cpp.inc     (-gen-typedef-{decls,defs})
//   KeaOpsDialect.h.inc / KeaOpsDialect.cpp.inc (-gen-dialect-{decls,defs})
// which is why it must transitively include KeaTypes.td.
//
//===----------------------------------------------------------------------===//

#ifndef KEA_OPS
#define KEA_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "kea/Dialect/KeaDialect.td"
include "kea/Dialect/KeaAttrs.td"
include "kea/Dialect/KeaTypes.td"

//===----------------------------------------------------------------------===//
// Tensor level
//===----------------------------------------------------------------------===//

def Kea_Conv2DOp : Kea_Op<"conv2d", [Pure]> {
  let summary = "Quantized NHWC/OHWI 2-D convolution";
  let description = [{
    Tensor-level convolution, the target of the `tosa.conv2d` lowering. Layout
    is NHWC for the input/output and OHWI for the weights; `pads` is
    `[top, bottom, left, right]`. `scale` and `zero_point` carry the
    requantization applied on the way out of the accumulator.

    ```mlir
    %0 = kea.conv2d %in, %w {strides = array<i64: 1, 1>,
                             pads = array<i64: 1, 1, 1, 1>,
                             scale = 1.250000e-01 : f64,
                             zero_point = 0 : i32}
       : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weights,
    Optional<AnyRankedTensor>:$bias,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$pads,
    DefaultValuedAttr<F64Attr, "1.0">:$scale,
    DefaultValuedAttr<I32Attr, "0">:$zero_point
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input `,` $weights (`,` $bias^)? attr-dict
      `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Buffer level
//===----------------------------------------------------------------------===//

def Kea_DmaLoadOp : Kea_Op<"dma_load"> {
  let summary = "Asynchronously copy a memref into a scratchpad buffer";
  let description = [{
    Issues a DMA from `source` into `dest`. The op has no results; completion is
    observed with `kea.wait` on the matching `event`.

    The memory effects come from the `Arg<..., [MemRead]>` / `[MemWrite]`
    annotations below -- ODS notices them and implicitly adds
    `MemoryEffectOpInterface` plus a generated `getEffects()`. Do NOT also list
    a `MemoryEffects<[...]>` trait or you get two conflicting definitions.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "DMA source", [MemRead]>:$source,
    Arg<Kea_BufferType, "scratchpad destination", [MemWrite]>:$dest,
    OptionalAttr<I64Attr>:$event
  );

  // GOTCHA: `qualified()` is required around type($dest). Without it MLIR
  // prints a dialect type belonging to the *same* dialect as the op in its
  // short form -- `<8x16xi8, A>` instead of `!kea.buffer<8x16xi8, A>`. It still
  // round-trips, but it is unreadable and breaks naive FileCheck patterns.
  let assemblyFormat = [{
    $source `to` $dest attr-dict `:` type($source) `to` qualified(type($dest))
  }];

  let hasVerifier = 1;
}

def Kea_MatmulOp : Kea_Op<"matmul"> {
  let summary = "Systolic-array matrix multiply accumulating into ACC";
  let description = [{
    `acc += lhs * rhs` on the PE array. Operands are constrained by *address
    space*: activations must live in A, weights in W and the accumulator in
    ACC, which is checked by the type constraints rather than the verifier.

    ```mlir
    kea.matmul %a, %w, %acc {tile = #kea.tile_config<16, 16, 32>}
      : !kea.buffer<8x16xi8, A>, !kea.buffer<16x8xi8, W>, !kea.buffer<8x8xi32, ACC>
    ```
  }];

  let arguments = (ins
    Arg<Kea_ABuffer, "activations", [MemRead]>:$lhs,
    Arg<Kea_WBuffer, "weights", [MemRead]>:$rhs,
    Arg<Kea_AccBuffer, "accumulator", [MemRead, MemWrite]>:$acc,
    UnitAttr:$transpose_rhs,
    OptionalAttr<Kea_TileConfigAttr>:$tile
  );

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc attr-dict
      `:` qualified(type($lhs)) `,` qualified(type($rhs)) `,` qualified(type($acc))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

// Trait-level (as opposed to Arg-level) memory effects: these ops have no
// operands, so the only way to keep them from being DCE'd is to declare the
// effect on the op itself.
def Kea_SignalOp : Kea_Op<"signal", [MemoryEffects<[MemWrite]>]> {
  let summary = "Raise the given event id";
  let arguments = (ins ConfinedAttr<I64Attr, [IntNonNegative]>:$event);
  let assemblyFormat = "$event attr-dict";
}

def Kea_WaitOp : Kea_Op<"wait", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Block until the given event id has been raised";
  let arguments = (ins ConfinedAttr<I64Attr, [IntNonNegative]>:$event);
  let assemblyFormat = "$event attr-dict";
}

#endif // KEA_OPS
