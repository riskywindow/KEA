//===- KeaMachineOps.td - KEA Level 2 (machine level) ops --*- tablegen -*-===//
//
// LEVEL 2 OF THE `kea` DIALECT. See docs/adr/0002-two-level-kea-dialect.md and
// docs/DIALECT_L2.md.
//
// There is EXACTLY ONE op here per KEA-1 instruction (docs/ISA.md §4.1), plus
// one non-instruction op (`kea.alloc`) that gives buffers a symbolic identity
// until `-kea-alloc` turns them into absolute addresses. This level is the ISA
// in SSA clothing: every operand and attribute below is named after -- and maps
// 1:1 onto -- a field of the matching `struct Kea*` in `include/kea/isa.h`, so
// the `-kea-emit` backend is a mechanical transcription and a reader can diff
// the two side by side.
//
// -----------------------------------------------------------------------------
// THE TWO CONVENTIONS THAT MAKE THAT MAPPING WORK
// -----------------------------------------------------------------------------
//
// 1. ADDRESSES ARE (SYMBOLIC BUFFER, OFFSET) PAIRS.
//
//    Every `X_addr` field of an isa.h struct appears here as
//
//        an operand  `$X`       -- an SSA `!kea.buffer` value, the *identity*
//        an attribute `$X_addr` -- a displacement INSIDE that buffer
//
//    and `-kea-emit` writes `instr.X_addr = base(X) + X_addr`, where `base()`
//    is what `-kea-alloc` assigned. `-kea-tile` must NOT assign base addresses;
//    that is a different pass owned by a different agent.
//
// 2. OFFSETS AND STRIDES ARE IN ELEMENTS OF THE BUFFER THEY INDEX.
//
//    `ACC` is addressed in int32 WORDS, not bytes (ISA.md §2.2) -- the single
//    most common way to get KEA-1 wrong. Here that is unforgeable, because an
//    ACC buffer's element type is `i32` (enforced by `BufferType::verify`) and
//    an SPM/DRAM buffer's element type is `i8` (enforced by the verifiers
//    below), and *every* offset/stride is counted in the element type of the
//    buffer it applies to. `acc_addr = 16` is 16 words because ACC elements are
//    words; `a_addr = 16` is 16 bytes because SPM_A elements are bytes. There
//    is no unit to remember and no unit to convert.
//
//    int4 activations/weights are still stored in `i8` buffers -- int4 is a
//    packing (two elements per byte, ISA.md §3.1), not a scratchpad element
//    type, and KEA-1 addresses are byte granular. The `int4` unit attribute on
//    `kea.load_w` / `kea.mm` / `kea.vquant` is the isa.h flag bit.
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
// NOT MODELLED: `NOP` (0x00). ISA.md §5.1 says outright that "the compiler
// should not need it" -- it exists for scheduling experiments and simulator
// self-tests. Adding an op nothing emits would be a stub, so there is none.
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
// Kea_I64 -- a SIGNED 64-bit attribute
//===----------------------------------------------------------------------===//
//
// Upstream's `I64Attr` gives accessors of type `uint64_t`, which is exactly
// wrong for this dialect: `a_inner_stride`, `dram_s1`, `spm_s2` and friends are
// `int32_t` in isa.h and are routinely negative (reversed layouts, ISA.md
// §6.1). Casting at every use site is how sign bugs get in. This is the same
// constraint and the same `-16 : i64` spelling, with `int64_t` accessors.
def Kea_I64 : Attr<CPred<"::llvm::isa<::mlir::IntegerAttr>($_self) && "
                         "::llvm::cast<::mlir::IntegerAttr>($_self).getType()"
                         ".isSignlessInteger(64)">,
                   "64-bit signless integer attribute"> {
  let storageType = "::mlir::IntegerAttr";
  let returnType = "int64_t";
  let valueType = I64;
  let convertFromStorage = "$_self.getValue().getSExtValue()";
  let constBuilderCall = "$_builder.getI64IntegerAttr($0)";
}

//===----------------------------------------------------------------------===//
// Operand type constraints
//===----------------------------------------------------------------------===//

// Kea_ABuffer / Kea_WBuffer / Kea_AccBuffer come from KeaTypes.td. Level 2 also
// needs "DRAM" and "either scratchpad", so instantiate them here rather than
// editing a Level 1 file.
def Kea_DramBuffer : Kea_BufferInSpace<"DRAM", "buffer in DRAM">;

def Kea_SpmBuffer
    : Type<And<[Kea_BufferType.predicate,
                CPred<"::llvm::cast<::mlir::kea::BufferType>($_self)"
                      ".getAddressSpace() == ::mlir::kea::AddressSpace::A || "
                      "::llvm::cast<::mlir::kea::BufferType>($_self)"
                      ".getAddressSpace() == ::mlir::kea::AddressSpace::W">]>,
           "buffer in a scratchpad (A or W)", "::mlir::kea::BufferType">;

//===----------------------------------------------------------------------===//
// kea.alloc -- symbolic buffer identity (NOT an instruction)
//===----------------------------------------------------------------------===//

def Kea_AllocOp : Kea_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Introduce a symbolic buffer with a size, a role and a live range";
  let description = [{
    The one Level 2 op that is not a KEA-1 instruction. It exists because
    `-kea-tile` must name the memory it operates on before `-kea-alloc` has
    decided where that memory goes. See docs/DIALECT_L2.md §4 -- that section is
    the contract between `-kea-tile` and the three passes downstream of it.

    * The **result type** carries the address space and the size: an on-chip
      buffer is always rank 1, `!kea.buffer<Nxi8, A>` / `<Nxi8, W>` /
      `<Nxi32, ACC>`, so `N` is directly the number of bytes (SPM) or int32
      words (ACC) `-kea-alloc` has to find room for.
    * `name` is a stable identifier. For a DRAM buffer it is also the `.kasm`
      DRAM symbol (`@conv0.weights`, ADR-0001 rule 3) and must be unique in the
      module.
    * `role` says what the buffer is for. One of:

      | role | space | meaning |
      | --- | --- | --- |
      | `input` | DRAM | a model input; staged by the runtime |
      | `output` | DRAM | a model output |
      | `activation` | DRAM | an inter-layer feature map |
      | `weights` | DRAM | constant weights, laid out per `layout` |
      | `qparam` | DRAM | an array of `KeaQuantParam` for `kea.vquant` |
      | `addparam` | DRAM | one `KeaAddParam` for `kea.vadd` |
      | `scratch` | A / W / ACC | an on-chip tile |

    * `layout` names the byte layout `-kea-emit` must materialize for a DRAM
      constant: `mxu_tiles_16x16` (ISA.md §8.1 weight tiling),
      `mxu_tiles_16x16_packed` (§8.6 channel packing), `dwu_planes`
      (`[KH][KW][C]`), `nhwc`, `quant_params`, `add_params`.
    * `source` is the Level 1 SSA value the DRAM contents come from -- a
      constant weight or bias tensor. `-kea-emit` reads the constant through
      this operand, so nothing has to be smuggled through attributes.
    * `quant` / `residual_quant` / `output_quant` carry the requantizations a
      `qparam` / `addparam` block encodes, in Level 1 (TOSA) form.
    * `add_param` is the *derived*, machine-form `KeaAddParam`, in isa.h field
      order: `[a_mult, b_mult, o_mult, a_shift, b_shift, o_shift, a_zp, b_zp,
      o_zp]`. `-kea-tile` has to derive it anyway to discharge errata E6, so it
      publishes it rather than making `-kea-emit` redo the derivation.
    * `input_zp` is the *activation* zero point of the contraction a `qparam`
      block belongs to. `-kea-emit` needs it to fold the zero-point correction
      into `KeaQuantParam.bias`:
      `bias[c] = bias_l1[c] - input_zp * sum_k w[c][k]`. For `role = "qparam"`
      the `source` operands are `[bias?, weights]`, so every input to that
      formula is reachable from the op.
    * `live = [first, last]` is the advisory live range, in block operation
      indices. It is what `-kea-alloc` sizes its interference graph from. It is
      *advisory* because the authoritative range is the SSA def-use range;
      `mlir::kea::refreshLiveRanges()` recomputes it and any pass that moves ops
      must call it.
    * `addr` is the **base address** `-kea-alloc` assigns, in the same elements
      as every other offset in the buffer's space: bytes for SPM_A / SPM_W /
      DRAM, int32 **words** for ACC. It is absent out of `-kea-tile` and
      present after `-kea-alloc`; `-kea-emit` computes every instruction field
      as `instr.X_addr = addr(X) + X_addr` (docs/MEMORY_PLANNING.md §1.1).

      The verifier enforces what can be checked from the op alone: `addr >= 0`,
      `addr % alignment == 0` -- which is what makes an aligned *displacement*
      imply an aligned absolute address (docs/DIALECT_L2.md §4.1.1) -- and
      `addr + extent <= capacity`. `-kea-alloc` additionally strengthens
      `alignment` from the ops that use the buffer and re-checks the absolute
      addresses; this verifier is the floor, not the whole rule.

      This attribute was a *discardable* `kea.addr` until it was promoted here.
      The old spelling is now rejected outright rather than silently ignored,
      because a base address that quietly does not reach `-kea-emit` produces a
      program that addresses byte 0 of every scratchpad.

    ```mlir
    %w = kea.alloc from %cst : tensor<24x1x1x4xi8>
         {name = "conv0.weights", role = "weights",
          layout = "mxu_tiles_16x16", alignment = 16 : i64}
       : !kea.buffer<1536xi8, DRAM>

    %tile = kea.alloc {name = "conv0.act", role = "scratch",
                       alignment = 16 : i64, live = array<i64: 7, 19>}
          : !kea.buffer<1616xi8, A>
    ```
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$source,
    StrAttr:$name,
    StrAttr:$role,
    DefaultValuedAttr<Kea_I64, "16">:$alignment,
    OptionalAttr<StrAttr>:$layout,
    OptionalAttr<DenseI64ArrayAttr>:$live,
    OptionalAttr<Kea_QuantAttr>:$quant,
    OptionalAttr<Kea_QuantAttr>:$residual_quant,
    OptionalAttr<Kea_QuantAttr>:$output_quant,
    OptionalAttr<Kea_I64>:$input_zp,
    OptionalAttr<DenseI64ArrayAttr>:$add_param,
    // The base address -kea-alloc assigns. Absent until then.
    OptionalAttr<Kea_I64>:$addr
  );

  let results = (outs Kea_BufferType:$result);

  let assemblyFormat = [{
    (`from` $source^ `:` type($source))? attr-dict `:` qualified(type($result))
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Number of elements the allocator must reserve: bytes for SPM_A / SPM_W /
    /// DRAM, int32 words for ACC.
    int64_t getExtent() {
      return ::llvm::cast<::mlir::kea::BufferType>(getResult().getType())
          .getNumElements();
    }
    ::mlir::kea::AddressSpace getSpace() {
      return ::llvm::cast<::mlir::kea::BufferType>(getResult().getType())
          .getAddressSpace();
    }
  }];
}

//===----------------------------------------------------------------------===//
// DMA -- `KeaDma`, isa.h
//===----------------------------------------------------------------------===//

// Shared by DMA_LD and DMA_ST, which have one encoding (ISA.md §6.1). The
// `spm_space` flag bit (KEA_DMA_F_SPM_W) is NOT an attribute: it is implied by
// the address space of the `$spm` operand, which is stronger.
//
// NB this is a `defvar` holding a dag, not a base class with `let arguments`.
// Deriving both DMA ops from a common `Kea_Op` subclass and then referring to
// `That<"...">.arguments` would instantiate an *anonymous* record deriving from
// `Op`, and -gen-op-defs would emit it as a third, nameless operation.
defvar Kea_DmaDescriptorFields = (ins
  Kea_I64:$dram_addr,     // KeaDma::dram_addr displacement, bytes
  Kea_I64:$spm_addr,      // KeaDma::spm_addr  displacement, bytes
  Kea_I64:$len0,          // KeaDma::len0,  1..65535
  Kea_I64:$n1,            // KeaDma::n1,    1..65535
  Kea_I64:$n2,            // KeaDma::h.aux, 1..255
  Kea_I64:$dram_s1,       // KeaDma::dram_s1
  Kea_I64:$dram_s2,       // KeaDma::dram_s2
  Kea_I64:$spm_s1,        // KeaDma::spm_s1
  Kea_I64:$spm_s2,        // KeaDma::spm_s2
  // KeaHead::unit. Absent until -kea-schedule picks an engine; "DMA0"/"DMA1".
  OptionalAttr<StrAttr>:$unit
);

def Kea_DmaLoadOp : Kea_Op<"dma_load"> {
  let summary = "DMA_LD (0x10): 3D strided copy DRAM -> SPM_A/SPM_W";
  let description = [{
    `KeaDma` with `opcode = DMA_LD`. Semantics, verbatim from ISA.md §6.1:

    ```
    for i2 in [0, n2):
      for i1 in [0, n1):
        copy len0 bytes
          DRAM side: dram_addr + i2*dram_s2 + i1*dram_s1
          SPM  side: spm_addr  + i2*spm_s2  + i1*spm_s1
    ```

    `unit` is absent out of `-kea-tile`: choosing DMA0 vs DMA1 is
    `-kea-schedule`'s job, and so is the `kea.signal` that publishes completion.

    ```mlir
    kea.dma_load %dram -> %spm
        {dram_addr = 0, spm_addr = 176, len0 = 128, n1 = 8, n2 = 1,
         dram_s1 = 128, dram_s2 = 1024, spm_s1 = 160, spm_s2 = 1280}
      : !kea.buffer<1024xi8, DRAM> -> !kea.buffer<1616xi8, A>
    ```
  }];

  let arguments = !con(
    (ins Arg<Kea_DramBuffer, "DRAM source", [MemRead]>:$dram,
         Arg<Kea_SpmBuffer, "scratchpad destination", [MemWrite]>:$spm),
    Kea_DmaDescriptorFields);

  let assemblyFormat = [{
    $dram `->` $spm attr-dict
      `:` qualified(type($dram)) `->` qualified(type($spm))
  }];

  let hasVerifier = 1;
}

def Kea_DmaStoreOp : Kea_Op<"dma_store"> {
  let summary = "DMA_ST (0x11): 3D strided copy SPM_A/SPM_W -> DRAM";
  let description = [{
    `KeaDma` with `opcode = DMA_ST`. Identical encoding and identical loop nest
    to `kea.dma_load`, opposite direction. A zero `dram_s1`/`dram_s2` is a
    last-writer-wins aliasing bug on a store and is rejected by the verifier.
  }];

  let arguments = !con(
    (ins Arg<Kea_SpmBuffer, "scratchpad source", [MemRead]>:$spm,
         Arg<Kea_DramBuffer, "DRAM destination", [MemWrite]>:$dram),
    Kea_DmaDescriptorFields);

  let assemblyFormat = [{
    $spm `->` $dram attr-dict
      `:` qualified(type($spm)) `->` qualified(type($dram))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MXU -- `KeaLoadW` / `KeaMatmul`, isa.h
//===----------------------------------------------------------------------===//

def Kea_LoadWOp : Kea_Op<"load_w"> {
  let summary = "LOAD_W (0x20): SPM_W -> one 16x16 MXU weight bank";
  let description = [{
    `KeaLoadW`. Loads `W[k][n] = SPM_W[w_addr + k*w_row_stride + n]` for
    `k < k_rows`, `n < n_cols`; every other element of the 16x16 bank is loaded
    as **zero**, which is how tail tiles for channel counts that are not
    multiples of 16 work without any masking in `kea.mm` (ISA.md §7.2).

    `bank` alternates 0/1 across a conv's taps so that the load hides entirely
    under the previous `kea.mm` (ISA.md §7.1, §8.3).

    Alignment (ISA.md §11.1): `w_addr` and `w_row_stride` are multiples of
    `KEA_ALIGN_LOADW_INT8` = 16 bytes, or `KEA_ALIGN_LOADW_INT4` = 8 in int4.
    The verifier enforces it -- the assembler would reject it anyway, and a
    diagnostic on the op is far more useful than one on a line of `.kasm`.

    ```mlir
    kea.load_w %w {w_addr = 256, w_row_stride = 16, k_rows = 16, n_cols = 16,
                   bank = 1} : !kea.buffer<2304xi8, W>
    ```
  }];

  let arguments = (ins
    Arg<Kea_WBuffer, "weight tile", [MemRead]>:$w,
    Kea_I64:$w_addr,          // KeaLoadW::w_addr displacement, bytes
    Kea_I64:$w_row_stride,    // KeaLoadW::w_row_stride, bytes, >= 0
    Kea_I64:$k_rows,          // KeaLoadW::k_rows, 1..16
    Kea_I64:$n_cols,          // KeaLoadW::n_cols, 1..16
    Kea_I64:$bank,            // KEA_LDW_F_BANK1, 0 or 1
    UnitAttr:$int4            // KEA_LDW_F_INT4
  );

  let assemblyFormat = "$w attr-dict `:` qualified(type($w))";
  let hasVerifier = 1;
}

def Kea_MmOp : Kea_Op<"mm"> {
  let summary = "MATMUL (0x21): stream SPM_A rows through the MXU into ACC";
  let description = [{
    `KeaMatmul`. **There is no weight operand** -- the weights are whatever the
    last `kea.load_w` put in `bank`. That is the hardware, and modelling it any
    other way would hide errata E7 (a `MATMUL` against a bank no `LOAD_W` ever
    wrote reads undefined state; `mlir::kea::verifyWeightBanks()` checks it over
    a whole function, since it is not a property of one op).

    ```
    for mo in [0, m_outer):
      for mi in [0, m_inner):
        a = a_addr   + mo*a_outer_stride   + mi*a_inner_stride    // SPM_A bytes
        q = acc_addr + mo*acc_outer_stride + mi*acc_inner_stride  // ACC words
        for n in [0, 16):
          ACC[q+n] = (accumulate ? ACC[q+n] : 0)
                   + sum_{k<16} A[a+k] * W[k][n]
    ```

    The array always reads 16 activation bytes and always writes 16 ACC words
    per row regardless of the resident tile's `k_rows`/`n_cols`, so the verifier
    checks readability/writability with an extent of 16, not of `k_rows`.

    `accumulate` is false on exactly one `kea.mm` per ACC region -- the first
    tap of the first reduction tile. There is no ACC-clear instruction and none
    is needed (ISA.md §8.3).

    ```mlir
    kea.mm %a, %acc
        {a_addr = 352, a_inner_stride = 16, a_outer_stride = 160,
         m_inner = 8, m_outer = 8,
         acc_addr = 0, acc_inner_stride = 16, acc_outer_stride = 128,
         bank = 1, accumulate}
      : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
    ```
  }];

  let arguments = (ins
    Arg<Kea_ABuffer, "activations", [MemRead]>:$a,
    Arg<Kea_AccBuffer, "accumulator", [MemRead, MemWrite]>:$acc,
    Kea_I64:$a_addr,            // KeaMatmul::a_addr displacement, bytes
    Kea_I64:$a_inner_stride,    // KeaMatmul::a_inner_stride, bytes
    Kea_I64:$a_outer_stride,    // KeaMatmul::a_outer_stride, bytes
    Kea_I64:$m_inner,           // KeaMatmul::m_inner, >= 1
    Kea_I64:$m_outer,           // KeaMatmul::m_outer, >= 1
    Kea_I64:$acc_addr,          // KeaMatmul::acc_addr displacement, WORDS
    Kea_I64:$acc_inner_stride,  // KeaMatmul::acc_inner_stride, WORDS, %16 == 0
    Kea_I64:$acc_outer_stride,  // KeaMatmul::acc_outer_stride, WORDS, %16 == 0
    Kea_I64:$bank,              // KEA_MM_F_BANK1, 0 or 1
    UnitAttr:$accumulate,       // KEA_MM_F_ACCUMULATE
    UnitAttr:$int4              // KEA_MM_F_INT4
  );

  let assemblyFormat = [{
    $a `,` $acc attr-dict `:` qualified(type($a)) `,` qualified(type($acc))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DWU -- `KeaDwconv`, isa.h
//===----------------------------------------------------------------------===//

def Kea_DwconvOp : Kea_Op<"dwconv"> {
  let summary = "DWCONV (0x30): depthwise convolution, SPM_A x SPM_W -> ACC";
  let description = [{
    `KeaDwconv`. int8 only; VALID padding only, so the SPM_A tile must already
    contain its zero-point halo (put there by a `kea.vcopy` fill, ISA.md §8.4a).
    Weights are `[KH][KW][C]` int8, `channels` bytes per tap plane. The ACC
    region is written **densely** as `[out_h][out_w][channels]` int32 -- there
    is no ACC stride field.

    `kernel` is 3 or 5 (`KEA_DW_F_KERNEL5`), `stride` is 1 or 2
    (`KEA_DW_F_STRIDE2`). `channels` must be a multiple of `KEA_DWU_LANES` = 16;
    `-kea-tile` rounds a 24-channel depthwise up to 32 and zero-pads the weight
    planes, which contributes exactly nothing to the accumulator.

    ```mlir
    kea.dwconv %a, %w, %acc
        {a_addr = 0, w_addr = 0, acc_addr = 0,
         out_h = 8, out_w = 8, channels = 32,
         a_row_stride = 320, a_pix_stride = 32, kernel = 3, stride = 1}
      : !kea.buffer<3200xi8, A>, !kea.buffer<288xi8, W>,
        !kea.buffer<2048xi32, ACC>
    ```
  }];

  let arguments = (ins
    Arg<Kea_ABuffer, "activations", [MemRead]>:$a,
    Arg<Kea_WBuffer, "depthwise kernel", [MemRead]>:$w,
    Arg<Kea_AccBuffer, "accumulator", [MemRead, MemWrite]>:$acc,
    Kea_I64:$a_addr,          // KeaDwconv::a_addr displacement, bytes
    Kea_I64:$w_addr,          // KeaDwconv::w_addr displacement, bytes
    Kea_I64:$acc_addr,        // KeaDwconv::acc_addr displacement, WORDS, %16==0
    Kea_I64:$out_h,           // KeaDwconv::out_h, >= 1
    Kea_I64:$out_w,           // KeaDwconv::out_w, >= 1
    Kea_I64:$channels,        // KeaDwconv::channels, multiple of 16
    Kea_I64:$a_row_stride,    // KeaDwconv::a_row_stride, bytes
    Kea_I64:$a_pix_stride,    // KeaDwconv::a_pix_stride, bytes
    Kea_I64:$kernel,          // KEA_DW_F_KERNEL5: 3 or 5
    Kea_I64:$stride,          // KEA_DW_F_STRIDE2: 1 or 2
    UnitAttr:$accumulate      // KEA_DW_F_ACCUMULATE
  );

  let assemblyFormat = [{
    $a `,` $w `,` $acc attr-dict
      `:` qualified(type($a)) `,` qualified(type($w)) `,` qualified(type($acc))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VPU -- `KeaVquant` / `KeaVadd` / `KeaVpool` / `KeaVcopy`, isa.h
//===----------------------------------------------------------------------===//

def Kea_VquantOp : Kea_Op<"vquant"> {
  let summary = "VQUANT (0x40): requantize an int32 ACC tile into SPM_A int8/int4";
  let description = [{
    `KeaVquant`, the fused epilogue of every contraction.

    ```
    qp = (KeaQuantParam*) SPM_W[qparam_addr]        // `channels` records
    for p in [0, num_pixels):
      for c in [0, channels):
        v   = ACC[acc_addr + p*acc_pix_stride + c] + qp[c].bias
        out = keaRequantize(v, qp[c].mult, qp[c].shift, out_zp,
                            clamp_lo, clamp_hi)
        store out at out_addr + p*out_pix_stride + c
    ```

    The fused activation is expressed purely through `clamp_lo`/`clamp_hi`
    (ISA.md §10.1): ReLU is `clamp_lo = out_zp`, ReLU6 is
    `clamp_lo = out_zp, clamp_hi = out_zp + round(6/out_scale)`.

    `-kea-tile` must satisfy ADR-0003 for every one of these: for each channel,
    either `tosa_shift >= 31`, or the accumulator range bound establishes
    `|acc + bias| < 2^tosa_shift`. See docs/DIALECT_L2.md §7.
  }];

  let arguments = (ins
    Arg<Kea_AccBuffer, "accumulator", [MemRead]>:$acc,
    Arg<Kea_ABuffer, "requantized output", [MemWrite]>:$out,
    Arg<Kea_WBuffer, "KeaQuantParam array", [MemRead]>:$qparam,
    Kea_I64:$acc_addr,        // KeaVquant::acc_addr displacement, WORDS, %16==0
    Kea_I64:$out_addr,        // KeaVquant::out_addr displacement, bytes
    Kea_I64:$qparam_addr,     // KeaVquant::qparam_addr displacement, bytes, %4==0
    Kea_I64:$num_pixels,      // KeaVquant::num_pixels, >= 1
    Kea_I64:$channels,        // KeaVquant::channels, multiple of 16
    Kea_I64:$acc_pix_stride,  // KeaVquant::acc_pix_stride, WORDS, %16 == 0
    Kea_I64:$out_pix_stride,  // KeaVquant::out_pix_stride, bytes
    Kea_I64:$out_zp,          // KeaHead::aux, int8
    Kea_I64:$clamp_lo,        // KeaVquant::clamp_lo, int8
    Kea_I64:$clamp_hi,        // KeaVquant::clamp_hi, int8
    UnitAttr:$int4            // KEA_VQ_F_OUT_INT4
  );

  let assemblyFormat = [{
    $acc `,` $out `,` $qparam attr-dict
      `:` qualified(type($acc)) `,` qualified(type($out)) `,`
          qualified(type($qparam))
  }];

  let hasVerifier = 1;
}

def Kea_VaddOp : Kea_Op<"vadd"> {
  let summary = "VADD (0x41): per-tensor quantized elementwise add in SPM_A";
  let description = [{
    `KeaVadd`. Parameters come from a single 20-byte `KeaAddParam` in SPM_W:

    ```
    sa = (a - a_zp) << KEA_VADD_LEFT_SHIFT;   // 20
    sb = (b - b_zp) << KEA_VADD_LEFT_SHIFT;
    xa = keaRdpot(keaSrdhm(sa, a_mult), a_shift);
    xb = keaRdpot(keaSrdhm(sb, b_mult), b_shift);
    o  = keaRdpot(keaSrdhm(xa + xb, o_mult), o_shift) + o_zp;
    out = clamp(o, clamp_lo, clamp_hi);
    ```

    In-place is legal (`out` may be `a`); the VPU reads before it writes within
    a lane. Errata E6 bounds the parameters: `-kea-tile` must emit a
    `KeaAddParam` with `|xa| + |xb| < 2^31`, checked by
    `mlir::kea::checkAddParamBound()`.
  }];

  let arguments = (ins
    Arg<Kea_ABuffer, "lhs", [MemRead]>:$a,
    Arg<Kea_ABuffer, "rhs", [MemRead]>:$b,
    Arg<Kea_ABuffer, "result", [MemWrite]>:$out,
    Arg<Kea_WBuffer, "KeaAddParam record", [MemRead]>:$param,
    Kea_I64:$a_addr,      // KeaVadd::a_addr displacement, bytes
    Kea_I64:$b_addr,      // KeaVadd::b_addr displacement, bytes
    Kea_I64:$out_addr,    // KeaVadd::out_addr displacement, bytes
    Kea_I64:$param_addr,  // KeaVadd::param_addr displacement, bytes, %4 == 0
    Kea_I64:$num_elems,   // KeaVadd::num_elems, >= 1
    Kea_I64:$clamp_lo,    // KeaVadd::clamp_lo, int8
    Kea_I64:$clamp_hi     // KeaVadd::clamp_hi, int8
  );

  let assemblyFormat = [{
    $a `,` $b `,` $out `,` $param attr-dict
      `:` qualified(type($a)) `,` qualified(type($b)) `,`
          qualified(type($out)) `,` qualified(type($param))
  }];

  let hasVerifier = 1;
}

def Kea_VpoolOp : Kea_Op<"vpool"> {
  let summary = "VPOOL (0x42): max / average pooling, SPM_A -> SPM_A";
  let description = [{
    `KeaVpool`. int8 only, VALID padding only. Input and output pixel strides
    are implicitly `channels` bytes.

    ISA.md §10.3 constrains this hard: **input and output quantization
    parameters must be identical**. Averaging raw int8 is then exactly right
    because `avg(x_i - zp) + zp = avg(x_i)`. `-kea-tile` rejects a `kea.pool`
    whose `quant` is not a zero-point-preserving identity.
  }];

  let arguments = (ins
    Arg<Kea_ABuffer, "input", [MemRead]>:$in,
    Arg<Kea_ABuffer, "output", [MemWrite]>:$out,
    Kea_I64:$in_addr,          // KeaVpool::in_addr displacement, bytes
    Kea_I64:$out_addr,         // KeaVpool::out_addr displacement, bytes
    Kea_I64:$out_h,            // KeaVpool::out_h, >= 1
    Kea_I64:$out_w,            // KeaVpool::out_w, >= 1
    Kea_I64:$channels,         // KeaVpool::channels, >= 1
    Kea_I64:$kh,               // KeaVpool::kh, 1..32
    Kea_I64:$kw,               // KeaVpool::kw, 1..32
    Kea_I64:$stride_h,         // KeaVpool::stride_h, 1..8
    Kea_I64:$stride_w,         // KeaVpool::stride_w, 1..8
    Kea_I64:$in_row_stride,    // KeaVpool::in_row_stride, bytes
    Kea_I64:$out_row_stride,   // KeaVpool::out_row_stride, bytes
    UnitAttr:$avg              // KEA_VP_F_AVG (absent => max pool)
  );

  let assemblyFormat = [{
    $in `,` $out attr-dict `:` qualified(type($in)) `,` qualified(type($out))
  }];

  let hasVerifier = 1;
}

def Kea_VcopyOp : Kea_Op<"vcopy"> {
  let summary = "VCOPY (0x43): 2D strided copy or constant fill inside SPM";
  let description = [{
    `KeaVcopy`.

    ```
    for r in [0, rows):
      dst[dst_addr + r*dst_row_stride ... +row_bytes) =
          fill ? splat(fill_value) : src[src_addr + r*src_row_stride ... )
    ```

    Fill mode is how a convolution halo gets its zero point (ISA.md §8.4a) --
    note that the fill value is the input **zero point**, not 0, which is why
    `fill_value` is an arbitrary int8. In fill mode `$src` is absent and
    `src_addr`/`src_row_stride` are ignored (and must be 0 here).

    `ACC` is not reachable from `VCOPY`; accumulators are initialized by a
    `kea.mm` / `kea.dwconv` without `accumulate`.

    ```mlir
    kea.vcopy to %tile {fill, fill_value = -128, dst_addr = 0,
                        row_bytes = 160, rows = 10, dst_row_stride = 160,
                        src_addr = 0, src_row_stride = 0}
      : !kea.buffer<1616xi8, A>
    ```
  }];

  let arguments = (ins
    Arg<Optional<Kea_SpmBuffer>, "copy source", [MemRead]>:$src,
    Arg<Kea_SpmBuffer, "destination", [MemWrite]>:$dst,
    Kea_I64:$src_addr,        // KeaVcopy::src_addr displacement, bytes
    Kea_I64:$dst_addr,        // KeaVcopy::dst_addr displacement, bytes
    Kea_I64:$row_bytes,       // KeaVcopy::row_bytes, >= 1
    Kea_I64:$rows,            // KeaVcopy::rows, >= 1
    Kea_I64:$src_row_stride,  // KeaVcopy::src_row_stride, bytes
    Kea_I64:$dst_row_stride,  // KeaVcopy::dst_row_stride, bytes
    DefaultValuedAttr<Kea_I64, "0">:$fill_value,  // KeaHead::aux, int8
    UnitAttr:$fill            // KEA_VC_F_FILL
  );

  // KEA_VC_F_SRC_W / KEA_VC_F_DST_W are implied by the operands' address
  // spaces, so they are not attributes.
  let assemblyFormat = [{
    (`from` $src^ `:` qualified(type($src)) `,`)?
      `to` $dst attr-dict `:` qualified(type($dst))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CTRL -- `KeaEventOp` / `KeaTrace` / `KeaHalt`, isa.h
//===----------------------------------------------------------------------===//
//
// Trait-level (as opposed to Arg-level) memory effects: these ops have no
// operands, so the only way to keep them from being DCE'd and from being
// reordered against the instructions they order is to declare the effect on the
// op itself.

def Kea_SignalOp : Kea_Op<"signal", [MemoryEffects<[MemWrite]>]> {
  let summary = "SIGNAL (0x02): drain this unit, then event[e] += value";
  let description = [{
    `KeaEventOp` with `opcode = SIGNAL`. `event` is `KeaHead::aux` (0..31),
    `value` is `KeaEventOp::value` (the increment, default 1), `unit` is
    `KeaHead::unit` -- `SIGNAL` is queue-agnostic, so the unit is part of the
    meaning and is chosen by `-kea-schedule`. `-kea-tile` emits none of these:
    it emits a correct *sequential* program and scheduling makes it parallel.
  }];
  let arguments = (ins
    ConfinedAttr<Kea_I64, [IntNonNegative]>:$event,
    DefaultValuedAttr<Kea_I64, "1">:$value,
    OptionalAttr<StrAttr>:$unit
  );
  let assemblyFormat = "$event attr-dict";
  let hasVerifier = 1;
}

def Kea_WaitOp : Kea_Op<"wait", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "WAIT (0x03): block until event[e] >= value, then subtract it";
  let description = [{
    `KeaEventOp` with `opcode = WAIT`. A counting acquire: it consumes exactly
    `value` counts. Rule D (ISA.md §5.5) -- every supplying `kea.signal` must
    appear earlier in stream order -- is `-kea-schedule`'s obligation, not this
    op's, because it is a property of a whole program.
  }];
  let arguments = (ins
    ConfinedAttr<Kea_I64, [IntNonNegative]>:$event,
    DefaultValuedAttr<Kea_I64, "1">:$value,
    OptionalAttr<StrAttr>:$unit
  );
  let assemblyFormat = "$event attr-dict";
  let hasVerifier = 1;
}

def Kea_TraceOp : Kea_Op<"trace", [MemoryEffects<[MemWrite]>]> {
  let summary = "TRACE (0x04): profiling marker; architecturally a NOP";
  let description = [{
    `KeaTrace`. `kind` is `KeaHead::flags` as a `TraceKind`: `marker`, `begin`
    or `end`. `tag` indexes the KEAF metadata JSON. `-kea-tile` brackets every
    fused layer with a `begin`/`end` pair so `kea-sim` can report per-layer
    occupancy and roofline position (ISA.md §5.4).

    Errata E4: `end` retires immediately and costs nothing, but the region keeps
    accumulating until the issuing unit's pipeline drains -- which is why the
    unit a `kea.trace` is placed on matters and is left to `-kea-schedule`.
  }];
  let arguments = (ins
    StrAttr:$kind,                       // "marker" | "begin" | "end"
    Kea_I64:$tag,                        // KeaTrace::tag
    DefaultValuedAttr<Kea_I64, "0">:$payload,  // KeaTrace::payload
    OptionalAttr<StrAttr>:$unit
  );
  let assemblyFormat = "$kind $tag attr-dict";
  let hasVerifier = 1;
}

def Kea_HaltOp : Kea_Op<"halt", [MemoryEffects<[MemWrite]>]> {
  let summary = "HALT (0x01): stop instruction fetch; must be last";
  let description = [{
    `KeaHalt`. Always targets `CTRL`, so there is no `unit` attribute. Must be
    the last instruction in the stream; the verifier checks that nothing but the
    block terminator follows it.
  }];
  let arguments = (ins DefaultValuedAttr<Kea_I64, "0">:$exit_code);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // KEA_MACHINE_OPS
