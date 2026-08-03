//===- KeaMachineOps.td - KEA Level 2 (buffer level) ops ---*- tablegen -*-===//
//
// LEVEL 2 OF THE `kea` DIALECT. See docs/adr/0002-two-level-kea-dialect.md.
//
// Ops here work on `!kea.buffer<shape, elemtype, addrspace>` with absolute
// integer scratchpad addresses, explicit DMA, explicit queue assignment and
// explicit semaphores. There is a 1:1 correspondence between a Level 2 op and
// a KEA instruction (docs/ISA.md) -- this level is the ISA in SSA clothing.
//
// -----------------------------------------------------------------------------
// OWNERSHIP NOTE
// -----------------------------------------------------------------------------
// The Level 1 (tensor level) ops live in KeaOps.td and are owned by the graph
// normalisation half of the compiler. This file is the Level 2 half. The two
// share one dialect but have SEPARATE TableGen targets:
//
//   KeaOps.td         -> add_mlir_dialect(KeaOps kea) -> KeaOps.{h,cpp}.inc
//   KeaMachineOps.td  -> -gen-op-{decls,defs}         -> KeaMachineOps.{h,cpp}.inc
//
// so that adding an op here never touches a Level 1 file. `KeaDialect.cpp`
// registers both op lists; `lib/Dialect/KeaMachineOps.cpp` holds the verifiers.
//
// ALREADY HERE (moved verbatim from the de-risking spike, minus the rename of
// `kea.matmul` -> `kea.mm`, which was forced because ADR-0002 gives the name
// `kea.matmul` to the Level 1 tensor op and gives this one the name `kea.mm`):
//
//   kea.dma_load, kea.mm, kea.signal, kea.wait
//
// STILL TO BE ADDED by the Level 2 work (ADR-0002 §"Level 2"):
//
//   kea.dma_store, kea.load_w, kea.dwconv, kea.vquant, kea.vadd, kea.vpool,
//   kea.vcopy
//
// plus the passes -kea-tile / -kea-alloc / -kea-schedule.
//
//===----------------------------------------------------------------------===//

#ifndef KEA_MACHINE_OPS
#define KEA_MACHINE_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "kea/Dialect/KeaDialect.td"
include "kea/Dialect/KeaAttrs.td"
include "kea/Dialect/KeaTypes.td"

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

def Kea_MmOp : Kea_Op<"mm"> {
  let summary = "Systolic-array matrix multiply accumulating into ACC";
  let description = [{
    `acc += lhs * rhs` on the PE array. Operands are constrained by *address
    space*: activations must live in A, weights in W and the accumulator in
    ACC, which is checked by the type constraints rather than the verifier.

    ```mlir
    kea.mm %a, %w, %acc {tile = #kea.tile_config<16, 16, 32>}
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

#endif // KEA_MACHINE_OPS
