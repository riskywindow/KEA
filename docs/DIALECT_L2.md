# `kea` Level 2 — the machine level

**Status: implemented.** Everything in this document is exercised by
`compiler/test/{l2-roundtrip,l2-invalid,kea-tile,kea-tile-invalid,mobilenet-e2e}.mlir`
and verified by `bash scripts/build_compiler.sh`.

Level 2 is **the ISA in SSA clothing**. There is exactly one op per KEA-1
instruction, every operand and attribute is named after the field of the
matching `struct Kea*` in [`include/kea/isa.h`](../include/kea/isa.h) it lowers
to, and `-kea-emit` is a mechanical transcription. Level 1 — the value-semantic
tensor level — is [DIALECT_L1.md](DIALECT_L1.md); the boundary between them is
[ADR-0002](adr/0002-two-level-kea-dialect.md).

| | |
|---|---|
| Ops | `compiler/include/kea/Dialect/KeaMachineOps.td` |
| Verifiers, whole-function checks | `compiler/lib/Dialect/KeaMachineOps.cpp` |
| The L1 → L2 pass | `compiler/lib/Transforms/Tile.cpp` (`-kea-tile`) |
| Machine contract | `include/kea/isa.h`, `include/kea/hw_config.h` (**frozen**) |
| Tests | `compiler/test/l2-*.mlir`, `compiler/test/kea-tile*.mlir` |

```
Level 1 ──-kea-tile──▶ Level 2 ──-kea-alloc──▶ Level 2 ──-kea-schedule──▶ Level 2 ──-kea-emit──▶ .kasm
          (this doc)   symbolic    addresses    absolute    queues +       runnable
                       buffers                  addresses   semaphores
```

---

## 1. The op set

Thirteen instruction ops, one per KEA-1 opcode, plus `kea.alloc`, which is not
an instruction.

| op | opcode | isa.h struct | operands | attributes |
|---|---|---|---|---|
| `kea.alloc` | — | — | `source...: tensor` | `name`, `role`, `alignment`, `layout?`, `live?`, `quant?`, `residual_quant?`, `output_quant?`, `input_zp?`, `add_param?` |
| `kea.dma_load` | `0x10` | `KeaDma` | `dram: DRAM`, `spm: A\|W` | `dram_addr`, `spm_addr`, `len0`, `n1`, `n2`, `dram_s1`, `dram_s2`, `spm_s1`, `spm_s2`, `unit?` |
| `kea.dma_store` | `0x11` | `KeaDma` | `spm: A\|W`, `dram: DRAM` | *(same)* |
| `kea.load_w` | `0x20` | `KeaLoadW` | `w: W` | `w_addr`, `w_row_stride`, `k_rows`, `n_cols`, `bank`, `int4` |
| `kea.mm` | `0x21` | `KeaMatmul` | `a: A`, `acc: ACC` | `a_addr`, `a_inner_stride`, `a_outer_stride`, `m_inner`, `m_outer`, `acc_addr`, `acc_inner_stride`, `acc_outer_stride`, `bank`, `accumulate`, `int4` |
| `kea.dwconv` | `0x30` | `KeaDwconv` | `a: A`, `w: W`, `acc: ACC` | `a_addr`, `w_addr`, `acc_addr`, `out_h`, `out_w`, `channels`, `a_row_stride`, `a_pix_stride`, `kernel`, `stride`, `accumulate` |
| `kea.vquant` | `0x40` | `KeaVquant` | `acc: ACC`, `out: A`, `qparam: W` | `acc_addr`, `out_addr`, `qparam_addr`, `num_pixels`, `channels`, `acc_pix_stride`, `out_pix_stride`, `out_zp`, `clamp_lo`, `clamp_hi`, `int4` |
| `kea.vadd` | `0x41` | `KeaVadd` | `a: A`, `b: A`, `out: A`, `param: W` | `a_addr`, `b_addr`, `out_addr`, `param_addr`, `num_elems`, `clamp_lo`, `clamp_hi` |
| `kea.vpool` | `0x42` | `KeaVpool` | `in: A`, `out: A` | `in_addr`, `out_addr`, `out_h`, `out_w`, `channels`, `kh`, `kw`, `stride_h`, `stride_w`, `in_row_stride`, `out_row_stride`, `avg` |
| `kea.vcopy` | `0x43` | `KeaVcopy` | `src?: A\|W`, `dst: A\|W` | `src_addr`, `dst_addr`, `row_bytes`, `rows`, `src_row_stride`, `dst_row_stride`, `fill_value`, `fill` |
| `kea.signal` | `0x02` | `KeaEventOp` | — | `event`, `value`, `unit?` |
| `kea.wait` | `0x03` | `KeaEventOp` | — | `event`, `value`, `unit?` |
| `kea.trace` | `0x04` | `KeaTrace` | — | `kind`, `tag`, `payload`, `unit?` |
| `kea.halt` | `0x01` | `KeaHalt` | — | `exit_code` |

**`NOP` (`0x00`) has no op.** ISA.md §5.1 says outright that "the compiler
should not need it" — it exists for scheduling experiments and simulator
self-tests. An op nothing emits would be a stub.

### 1.1 The two conventions that make the 1:1 mapping work

**(a) Addresses are `(symbolic buffer, offset)` pairs.** Every `X_addr` field of
an isa.h struct appears here as an operand `$X` — the buffer's *identity* — and
an attribute `$X_addr` — a displacement inside it. `-kea-emit` writes

```
instr.X_addr = base(X) + X_addr
```

where `base()` is what `-kea-alloc` assigned. `-kea-tile` never assigns a base.

**(b) Offsets and strides are in ELEMENTS of the buffer they index.** ACC is
addressed in int32 **words**, not bytes (ISA.md §2.2) — "this trips everyone
once". Here it cannot: an ACC buffer's element type is `i32` (enforced by
`BufferType::verify`) and an SPM/DRAM buffer's element type is `i8` (enforced by
every Level 2 verifier), and every offset and stride is counted in the element
type of the buffer it applies to. `acc_addr = 16` is 16 words because ACC
elements are words; `a_addr = 16` is 16 bytes because SPM_A elements are bytes.
There is no unit to remember and no unit to convert, and the mistake is not
expressible.

int4 data still lives in `i8` buffers: int4 is a *packing* (two elements per
byte, ISA.md §3.1), not a scratchpad element type, and KEA-1 addresses are byte
granular. The `int4` unit attribute is the isa.h flag bit.

**Flag bits** are `UnitAttr`s named after the `KEA_*_F_*` constant they set
(`accumulate`, `int4`, `fill`, `avg`), except where isa.h's own constructor takes
a value rather than a bit (`kernel` ∈ {3,5}, `stride` ∈ {1,2}). The
`spm_space` / `src_space` / `dst_space` bits are **not** attributes: they follow
from the address space of the operand, which is stronger than an attribute
because the type system checks it.

`unit` (the `KeaHead::unit` field) is an *optional* string attribute and is
absent out of `-kea-tile`. For DMA it names the engine; for the queue-agnostic
`SIGNAL`/`WAIT`/`TRACE` it names the queue. Both are `-kea-schedule`'s decisions.

### 1.2 Verifiers

Every rule the assembler's `kea::keaValidate()` checks, checked here first,
where the diagnostic can point at a readable op instead of at line 4,712 of a
`.kasm` file. All constants come from `hw_config.h`; nothing is duplicated.

| rule | ops |
|---|---|
| `w_addr`, `w_row_stride` multiple of 16 B (8 B in int4) | `load_w` |
| `k_rows`, `n_cols` in 1…16; `bank` in 0…1 | `load_w`, `mm` |
| **every ACC address and stride a multiple of 16 words** | `mm`, `dwconv`, `vquant` |
| `m_inner * m_outer <= 2048` (`KEA_MXU_MAX_ROWS`, the ACC capacity bound) | `mm` |
| `channels` a multiple of 16 | `dwconv`, `vquant` |
| `qparam_addr` / `param_addr` a multiple of 4 B | `vquant`, `vadd` |
| `len0`, `n1` in 1…65535; `n2` in 1…255; no zero destination stride | `dma_*` |
| `kernel` ∈ {3,5}, `stride` ∈ {1,2} | `dwconv` |
| `kh`,`kw` in 1…32; `stride_h`,`stride_w` in 1…8 | `vpool` |
| `clamp_lo <= clamp_hi`, both int8; int4 output clamps inside [-8,7] | `vquant`, `vadd` |
| event id 0…31, value ≤ `0x7FFFFFFF` | `signal`, `wait` |
| **every strided walk stays inside its buffer**, with an extent of 16 for `kea.mm` because the array always reads 16 bytes and writes 16 words | all |
| on-chip allocation ≤ its space's capacity | `alloc` |
| `HALT` is the last instruction in its block | `halt` |

Memory effects are attached to the **operands** (`Arg<..., [MemRead]>` /
`[MemWrite]`), not declared as op-level traits, so MLIR's effect analysis sees
per-buffer reads and writes rather than one global resource and cannot reorder
two hardware-visible ops that touch the same buffer. Ops with no buffer operand
(`signal`, `wait`, `trace`, `halt`) carry a trait-level `MemWrite` so they are
neither DCE'd nor sunk past the work they order.

---

## 2. Two whole-function checks

Some properties are not properties of a single op. These live in
`KeaMachineOps.h` and are callable from any pass:

```cpp
LogicalResult mlir::kea::verifyWeightBanks(Operation *root);   // errata E7
void          mlir::kea::refreshLiveRanges(Operation *root);   // §4
int64_t       mlir::kea::addParamWorstCaseSum(aMult, aShift,
                                              bMult, bShift);  // errata E6
int64_t       mlir::kea::bufferExtent(Value);
Span          mlir::kea::stridedSpan(base, n0, s0, n1, s1, extent);
```

`-kea-tile` runs `verifyWeightBanks` and `refreshLiveRanges` over its own output.
Running `-kea-tile` on a function that is **already** Level 2 does nothing except
re-run both, which makes it a cheap validator for hand-written or
`-kea-alloc`-rewritten IR.

---

## 3. What `-kea-tile` produces

```mlir
func.func @conv3x3_s1(%arg0: tensor<1x8x8x16xi8>, %arg1: tensor<32x3x3x16xi8>) {
  // 1. The memory map: every DRAM object the layer touches, as a symbol.
  %0 = kea.alloc {name = "conv3x3_s1.input0", role = "input"}
     : !kea.buffer<1024xi8, DRAM>
  %1 = kea.alloc {name = "conv3x3_s1.0.out", role = "output"}
     : !kea.buffer<2048xi8, DRAM>
  %2 = kea.alloc from %arg1 : tensor<32x3x3x16xi8>
       {layout = "mxu_tiles_16x16", name = "conv3x3_s1.0.weights",
        role = "weights"} : !kea.buffer<4608xi8, DRAM>
  %3 = kea.alloc from %arg1 : tensor<32x3x3x16xi8>
       {input_zp = -5 : i64, layout = "quant_params",
        name = "conv3x3_s1.0.qparams", quant = #kea.quant<…>, role = "qparam"}
     : !kea.buffer<384xi8, DRAM>

  // 2. The program: one TRACE-bracketed region per fused layer.
  kea.trace "begin" 0
  %4 = kea.alloc {live = array<i64: 6, 51>, …, role = "scratch"}
     : !kea.buffer<4608xi8, W>
  kea.dma_load %2 -> %4 {…}
  …
  kea.load_w %4 {bank = 0 : i64, k_rows = 16 : i64, n_cols = 16 : i64,
                 w_addr = 0 : i64, w_row_stride = 16 : i64}
  kea.mm %7, %8 {a_addr = 0 : i64, a_inner_stride = 16 : i64,
                 a_outer_stride = 160 : i64, acc_addr = 0 : i64,
                 acc_inner_stride = 16 : i64, acc_outer_stride = 128 : i64,
                 bank = 0 : i64, m_inner = 8 : i64, m_outer = 8 : i64}
  …
  kea.vquant %8, %9, %6 {…}
  kea.dma_store %9 -> %1 {…}
  kea.trace "end" 0
  kea.halt
  return
}
```

The function loses its results: Level 2 is a machine program, not a
value-semantic function. Model inputs and outputs are `kea.alloc` DRAM objects
with `role = "input"` / `"output"`, and the block arguments survive only as the
`source` operands that tell `-kea-emit` which tensor each one is.

**Every layer reads from DRAM and writes to DRAM.** Keeping an activation
resident in SPM_A across a layer boundary is a real optimization that this pass
does not do; see §9.

---

## 4. The symbolic-buffer and live-range interface

**This section is the contract for `-kea-alloc`, `-kea-schedule` and
`-kea-emit`.**

### 4.1 `kea.alloc`

```mlir
%b = kea.alloc [from %src... : type...] {
       name      = "<unique string>",     // required
       role      = "<role>",              // required
       alignment = <int>,                 // default 16
       layout    = "<layout>",            // DRAM constants only
       live      = array<i64: first, last>,   // on-chip only
       quant / residual_quant / output_quant = #kea.quant<…>,
       input_zp  = <int>,
       add_param = array<i64: …>
     } : !kea.buffer<Nx{i8|i32}, {A|W|ACC|DRAM}>
```

* **The result type carries the space and the size.** On-chip buffers are always
  rank 1: `!kea.buffer<Nxi8, A>`, `!kea.buffer<Nxi8, W>`,
  `!kea.buffer<Nxi32, ACC>`. `N` is directly the number of **bytes** (SPM) or
  **int32 words** (ACC) to find room for — `AllocOp::getExtent()` returns it.
* **`name` is stable and unique in the module.** For a DRAM buffer it is also
  the `.kasm` DRAM symbol (`@conv0.weights`, ADR-0001 rule 3). It is
  `"<func>.<layer>.<kind>"` by construction.
* **`role`:**

  | role | space | meaning for the consumer |
  |---|---|---|
  | `input` | DRAM | model input; the runtime stages it, `-kea-emit` records it in `model.map.json` |
  | `output` | DRAM | model output |
  | `activation` | DRAM | inter-layer feature map; `-kea-emit` reserves arena space |
  | `weights` | DRAM | constant weights; `-kea-emit` materializes `source[0]` in `layout` into `model.weights.bin` |
  | `qparam` | DRAM | `KeaQuantParam[channels]`; see §4.3 |
  | `addparam` | DRAM | one `KeaAddParam`; see §4.4 |
  | `scratch` | A / W / ACC | an on-chip tile — **the only role `-kea-alloc` places** |

* **DRAM buffers are never allocated by `-kea-alloc`.** They are symbols; the
  assembler resolves them against `model.map.json`. The verifier rejects a
  `live` range on one.

### 4.1.1 Alignment — the one rule `-kea-alloc` must not get wrong

`alignment` is **in the same elements as the buffer's offsets**: bytes for
SPM_A / SPM_W / DRAM, int32 **words** for ACC. `-kea-tile` leaves it at the
default 16 for every buffer it creates, and

> **`-kea-alloc` must place every on-chip buffer at a base that is a multiple of
> its `alignment`.**

That single rule is what makes the ISA's alignment contract hold end to end. The
Level 2 verifiers check the *offsets* — `w_addr % 16 == 0`, `acc_addr % 16 == 0`
in words, `qparam_addr % 4 == 0` — and an offset being aligned only implies
`base + offset` is aligned if the base is too. Concretely, with `alignment = 16`:

| space | base multiple of | keeps aligned |
|---|---|---|
| SPM_W | 16 bytes | `LOAD_W.w_addr` (16 B), `VQUANT.qparam_addr` / `VADD.param_addr` (4 B) |
| ACC | 16 words | `MATMUL.acc_addr`, `DWCONV.acc_addr`, `VQUANT.acc_addr` (16 words) |
| SPM_A | 16 bytes | nothing the ISA requires — `MATMUL.a_addr` is deliberately byte granular (§11.1), since the conv lowering shifts it by `kw*sp` and `sp` is 3 for an RGB first layer. 16 is the recommended DMA alignment (`KEA_ALIGN_DMA_RECOMMENDED`), not a requirement |

An int4 `LOAD_W` needs only 8-byte alignment, so 16 covers it too. Every ACC
buffer `-kea-tile` emits has an extent that is already a multiple of 16 words
(`OCG_t·OH_t·OW_t·16` for the MXU path, `OH_t·OW_t·C_pad` with `C_pad % 16 == 0`
for the DWU path), so packing ACC at 16-word granularity wastes nothing.

`-kea-emit` computes each instruction field as

```
instr.X_addr = base(X) + X_addr
```

with no unit conversion anywhere: the offset is already in the space's own
addressing unit (§1.1(b)).

### 4.2 Live ranges

```
live = array<i64: first, last>
```

`first` and `last` are **indices into the enclosing block's operation list**:
`first` is the position of the `kea.alloc` itself, `last` the position of its
last user. Two on-chip buffers may share storage iff their `[first, last]`
intervals are disjoint **and** they are in the same address space.

The attribute is a *materialization* of the SSA def-use range, not an
independent source of truth:

* The authoritative range is always "from the defining `kea.alloc` to the last
  operation in block order that uses the result". `-kea-alloc` may recompute it
  and must get the same answer.
* `mlir::kea::refreshLiveRanges(op)` recomputes and re-stamps every one of them
  in one walk. **Any pass that inserts, erases or moves a Level 2 op must call
  it** — in particular `-kea-schedule`, which will insert `kea.signal` /
  `kea.wait` and may software-pipeline, is required to call it before handing
  off.
* The op verifier only checks well-formedness (two elements, ordered,
  non-negative, on-chip).

Two consequences worth stating explicitly, because they are what make the
interface usable:

1. **`-kea-tile` allocates at the finest granularity it can.** A fresh
   `kea.alloc` per spatial tile, not one buffer reused across the loop. That
   gives `-kea-alloc` short, disjoint intervals and maximum freedom to pack —
   including the freedom to *not* share, which is what enables double buffering.
2. **`-kea-tile` spends at most half of each scratchpad** (the
   `spm-reserve-factor` option, default 2). The other half is headroom for
   `-kea-schedule` to keep two sets of tiles live at once. If a scheduler wants
   a different split, re-run `-kea-tile` with a different factor rather than
   trying to shrink tiles afterwards.

### 4.3 `role = "qparam"` — what `-kea-emit` must write

`source` is `[bias?, weights]` and the attributes are `quant` (the Level 1
`#kea.quant`) and `input_zp` (the *activation* zero point of the contraction).
For each output channel `c`, `-kea-emit` writes a 12-byte `KeaQuantParam`:

```
mult  = quant.multiplier[c]                          // or [0] when per tensor
shift = quant.shift[c] - 31                          // ADR-0003's convention
bias  = bias_l1[c] - input_zp * sum_k weights[c][k]   // the zero-point fold
```

The third line is why the weights are a `source` operand of a *quantization*
block: the MXU computes the raw `sum a*w`, so the `-a_zp * sum w` correction has
to be folded into the bias, and the bias record is the only place it fits. The
block is `roundUp(OC, 16) * 12` bytes long; the padding channels exist because
`VQUANT.channels` must be a multiple of 16, and their records are never read for
a real output (write zeros).

### 4.4 `role = "addparam"` — what `-kea-emit` must write

`add_param` is already the machine-form record, in isa.h field order:

```
[a_mult, b_mult, o_mult, a_shift, b_shift, o_shift, a_zp, b_zp, o_zp]
```

`-kea-emit` writes it into a 20-byte `KeaAddParam` verbatim. `quant`,
`residual_quant` and `output_quant` carry the Level 1 form it was derived from,
for provenance and debugging. The derivation, and why it is not a straight copy,
is §7.3. **`a` is the layer's own output and `b` is the residual operand**; for a
standalone `kea.add`, `a` is `lhs` and `b` is `rhs`.

### 4.5 `layout` — what `-kea-emit` must lay out

| layout | source | bytes |
|---|---|---|
| `mxu_tiles_16x16` | `[OC,KH,KW,IC]` conv or `[OC,IC]` FC weights | dense 16×16 `int8` tiles of 256 B, ordered `[oc0][ic0][kh][kw]`, exactly ISA.md §8.1. Tile for tap `(kh,kw)` of group `(oc0,ic0)` is at `((((oc0/16)*ceil(IC/16) + ic0/16)*KH + kh)*KW + kw) * 256`, and `w_row_stride` is always 16 |
| `mxu_tiles_16x16_packed` | `[OC,KH,KW,IC]` with `KW*IC <= 16` | ISA.md §8.6: `KH` tiles per oc group instead of `KH*KW`, each packing `W[kw][ic][oc]` in that order into `k_rows = KW*IC` rows |
| `mxu_tiles_16x16_kn` | `[B,K,N]` batched-matmul rhs | as `mxu_tiles_16x16` but the output channel is `N`, the last dimension, and the tile array repeats per batch |
| `dwu_planes` | `[OC,KH,KW,1]` depthwise weights | `[KH][KW][C_pad]` int8, contiguous, `C_pad = roundUp(C,16)` bytes per tap plane, zero in the padding lanes |
| `quant_params` | see §4.3 | |
| `add_params` | see §4.4 | |
| `nhwc` | — | dense NHWC, the layout every activation buffer uses |

### 4.6 What `-kea-tile` deliberately leaves undone

* **No absolute addresses.** Every address field is a displacement inside a
  `kea.alloc`.
* **No `kea.signal` / `kea.wait`.** The output is a correct *sequential*
  program: it is right if executed one instruction at a time.
* **No `unit` attribute anywhere.** No DMA engine chosen, no queue chosen for
  the queue-agnostic `kea.trace`.
* **No double buffering.** Tiles are sized so it is possible, not performed.

---

## 5. The tiling cost model

Everything in this section is `compiler/lib/Transforms/Tile.cpp`, and every
constant and formula is `constexpr` in `hw_config.h`. Nothing is a heuristic
pulled from the air, and nothing is duplicated from the header.

### 5.1 What is being chosen

For a convolution-like layer (`kea.conv2d`, `kea.fully_connected`,
`kea.matmul` — an FC is `KH = KW = 1, OH = 1, OW = batch`, which is exactly
ISA.md §8.2's degenerate case), the free variables are

* `OH_t`, `OW_t` — the output tile, restricted to **divisors** of `OH` and `OW`
  so no tile is ragged. A ragged tail tile needs its own instruction sequence
  and buys nothing on MobileNetV2's shapes, which are all a power of two times a
  small odd number.
* `OCG_t` — how many 16-channel output groups are resident in ACC at once,
  `1 … ceil(OC/16)`.

The loop nest that results is

```
for oc tile:                       # OCG_t groups at a time
  DMA weights + KeaQuantParams for this oc tile into SPM_W   (loop invariant)
  for (oh0, ow0) in output tiles:
    VCOPY-fill the padded activation tile with the input zero point
    DMA the in-bounds part of the input window into it
    for g in oc groups of this tile:
      for ic0, kh, kw:  LOAD_W ; MATMUL          # ISA.md §8.5
      VQUANT that group's ACC region into the output tile
    [DMA the residual in; VADD]
    DMA the output tile out
```

### 5.2 Feasibility — the constraints a candidate must satisfy

With `IH_p = (OH_t-1)*S + (KH-1)*D + 1` and `IW_p` likewise (the padded input
window), `sp = IC`, `sr = IW_p * sp`:

| constraint | from |
|---|---|
| `OCG_t * OH_t * OW_t * 16 <= KEA_ACC_WORDS` (32768) | ACC capacity |
| `OH_t * OW_t <= KEA_MXU_MAX_ROWS` (2048) | `MATMUL.m_inner * m_outer` |
| `IH_p*IW_p*IC + 16` + `OH_t*OW_t*OCG_t*16 + 16` `<= KEA_SPM_A_BYTES / R` | SPM_A, `R` = `spm-reserve-factor` |
| `OCG_t*ceil(IC/16)*taps*256 + OCG_t*16*12 <= KEA_SPM_W_BYTES / R` | SPM_W |
| `OH_t <= 255` | the output store's `n2` is the instruction's `aux` byte |
| `IW_p * IC <= 65535`, `IH_p <= 65535`, `OCG_t*16 <= 65535` | DMA `len0` / `n1` |

The two `+ 16` terms are the **activation tail pad**: `kea.mm` always reads 16
activation bytes per row whatever the resident tile's `k_rows` (ISA.md §7.3), so
the last row of every tile reads up to 15 bytes past its last real pixel.

### 5.3 Cost — the objective

Per spatial tile, on each of the three concurrent resources:

```
MXU = ocg * icTiles * taps * keaMatmulOccupancy(OW_t, OH_t, int4)
    + keaLoadWOccupancy(k_rows, int4)
VPU = ocg * keaVquantOccupancy(OH_t*OW_t, 16)
    + keaVcopyOccupancy(IW_p*IC, IH_p)                  [when there is a halo]
DMA = keaDmaOccupancy(IW_p*IC, IH_p, 1)                 [activations in]
    + keaDmaOccupancy(OCG_t*16, OW_t, OH_t)             [activations out]
```

Only **one** `LOAD_W` is charged per tile, not one per tap: consecutive pairs
alternate weight banks, so every load but the first hides entirely under the
previous `MATMUL` (ISA.md §7.1). That is not an optimistic assumption, it is
what the bank alternation rule in §6.1 is *for* — and that rule has to be the
monotonic one for the assumption to hold at output-channel group boundaries.

The three resources run concurrently and `-kea-schedule` overlaps them, so the
tile's cost is the **max**, not the sum:

```
total = nOcTiles * ( weightDMA + nSpatialTiles * batch * max(MXU, VPU, DMA) )
```

Minimize `total`; break ties toward the larger tile (fewer instructions, less
dispatch pressure). If no candidate is feasible, `-kea-tile` fails with a
diagnostic naming which capacity was exceeded — it never silently produces
something that will not fit.

The model is deliberately *scheduling-aware while emitting an unscheduled
program*. Costing tiles as `MXU + VPU + DMA` would systematically prefer tiles
that are too small, because it would price overlap that the next pass is
guaranteed to deliver.

**Depthwise** (`kea.dwconv2d`) uses the same shape of search over `(OH_t, OW_t)`
with `keaDwconvOccupancy` in place of the MXU term. There is no channel tiling:
`DWCONV` iterates channel groups internally for the same cycle count in one
instruction (ISA.md §9.1), so splitting channels would only add instructions.

**Pooling** tiles over output row bands only; **`kea.add`** splits the flat
element range into chunks of `SPM_A / (3R)`, three tiles being live at once.

### 5.4 What it picks for MobileNetV2

Run `-kea-tile=report-tiles=true` and read `kea.tiling` off the function. For
MobileNetV2 1.0/224 at the default `spm-reserve-factor=2` (so the budget is
131072 B of SPM_A, 131072 B of SPM_W, 32768 ACC words):

| layer | shape | `OH_t × OW_t` | `OCG_t` | SPM_A | SPM_W | ACC | est. cycles |
|---|---|---|---|---|---|---|---|
| conv 3×3 s2, 3→32, 112² out | packed (§8.6) | 8 × 112 | 2 | 40,179 | 1,920 | 28,672 | 75,854 |
| dw 3×3 s1, 112²×32 | | 16 × 56 | — | 62,112 | 672 | 28,672 | |
| pw 32→16, 112² | | 16 × 112 | 1 | 86,048 | 704 | 28,672 | 63,248 |
| pw 16→96, 112² | | 2 × 112 | 6 | 25,120 | 2,688 | 21,504 | 115,116 |
| dw 3×3 s2, 112²×96 | | 14 × 14 | — | 99,584 | 2,016 | 18,816 | |
| pw 96→24, 56² | | 14 × 56 | 2 | 100,384 | 3,456 | 25,088 | 38,116 |
| dw 3×3 s1, 56²×144 | | 14 × 14 | — | 65,120 | 3,024 | 28,224 | |
| pw 384→96, 14² | | 14 × 14 | 6 | 94,112 | 38,016 | 18,816 | 31,222 |
| pw 960→320, 7² | | 7 × 7 | 5 | 50,992 | 77,760 | 3,920 | 83,224 |
| pw 320→1280, 7² | | 7 × 7 | 20 | 31,392 | 106,240 | 15,680 | 111,544 |
| avgpool 7×7×1280 | | 1 × 1 | — | 64,000 | — | — | |
| FC 1280→1000 | | 1 × 1 | 3 | 1,360 | 62,016 | 48 | 107,562 |

Three things are worth reading off that table:

* **The 112²×32 layers are row bands**, exactly as MICROARCH.md §9.3(b)
  predicted: `112·112·32 = 401,408` bytes does not fit in a 256 KiB SPM_A, so
  the tiler picks 8- or 16-row bands.
* **The late 7×7 layers tile over output channels instead** (`OCG_t` 5 and 20 of
  20 and 80 groups): there the activations are tiny and the *weights* are what
  does not fit, so the loop that gets split is the channel loop.
* **The classifier's estimate, 107,562 cycles, is DMA bound and matches
  MICROARCH.md §9.3(d)** — 80,080 cycles of weight streaming plus ~25,200 cycles
  of `MATMUL` setup on a batch-1 `m_inner = 1` GEMM. The model reproduces the
  hand analysis without being told about it.

---

## 6. The convolution lowering (ISA.md §8.5, normative)

### 6.1 The rule, as implemented

```
t = 0                                  # monotonic over the whole MXU stream
for oc0 in output-channel groups of this ACC batch:
  for ic0 in reduction tiles:
    for kh, kw:
      bank = t & 1
      LOAD_W  w_addr        = ((((oc0)*icTiles + ic0)*KH + kh)*KW + kw) * 256
              w_row_stride  = 16
              k_rows        = min(16, IC - ic0*16)
              n_cols        = min(16, OC - oc0*16)
      MATMUL  a_addr           = ic0*16 + kh*D*sr + kw*D*sp
              a_inner_stride   = S * sp
              a_outer_stride   = S * sr
              m_inner          = OW_t
              m_outer          = OH_t
              acc_addr         = oc0_local * OH_t*OW_t*16
              acc_inner_stride = 16
              acc_outer_stride = OW_t * 16
              accumulate       = (t != 0)
      t += 1
```

`accumulate` is absent on exactly one instruction per ACC region — the first tap
of the first reduction tile. There is no ACC-clear instruction and none is
needed. That counter *is* per output-channel group, and has nothing to do with
banks.

#### The weight-bank rule

**`bank = t & 1` where `t` counts LOAD_W/MATMUL pairs over the entire MXU
stream, not restarted per output-channel group.** This is a deliberate
departure from the literal reading of §8.3's pseudocode, which writes
`t = tap counter across (ic0, kh, kw)` inside the `oc0` loop.

`LOAD_W` occupies the bank it targets, and the MXU tracks the two banks as
separate resources precisely so a load into the idle one runs concurrently with
a `MATMUL` streaming the other (ISA.md §5.3, "what does *not* need
synchronization"; §7.1, "this is what makes weight double-buffering free").
Restarting `t` per group makes the last pair of group *g* and the first pair of
group *g+1* share a bank, so that second `LOAD_W` stalls until the first
`MATMUL` releases it. For a 1×1 convolution with a single reduction tile — where
each output-channel group is *exactly one pair* — that is not an edge case at
the boundary, it is **every** load in the layer. MobileNetV2 is mostly 1×1
convolutions, and the simulator measures the bank overlap as a 1.135× effect in
isolation.

A monotonic `t` is still `t & 1`; it emits the identical instruction sequence
and the identical arithmetic, and only changes which physical bank each pair
uses. It strictly dominates, and it is what §5.3's cost model already assumes
(one `LOAD_W` charged per tile, not one per tap).

**Bank assignment is `-kea-tile`'s job and is final.** `-kea-schedule` may
assume the invariant below and is *not* expected to renumber banks:

> For every `kea.mm`, the `kea.load_w` that most recently targeted the same
> `bank` is the immediately preceding MXU instruction in the block, and no other
> `kea.load_w` intervenes between them.

That is what makes errata E7 (§7.4) trivially true rather than incidental, and
`mlir::kea::verifyWeightBanks()` re-checks it over the whole function on every
run. A scheduler that reorders MXU instructions relative to each other would
break it — so it must not, which is fine: the MXU queue is in-order and
consecutive `MATMUL`s accumulating into one ACC region need no synchronization
at all (ISA.md §7.3, "Ordering").

`compiler/test/kea-tile.mlir`'s `@banks_alternate_across_groups` pins the rule in
its sharpest form (four one-tap groups, banks 0/1/0/1), and
`compiler/test/mobilenet-e2e.mlir` pins it on the real expand convolution.

### 6.2 The worked example, reproduced

`IC = 16, OC = 32`, input `8×8×16` with `pad = 1`, output `8×8×32`, int8. The
SPM_A tile is padded to `10×10×16`, so `sp = 16`, `sr = 160`.
`compiler/test/kea-tile.mlir` asserts the whole sequence; the per-tap `a_addr`
table is exactly ISA.md's:

| tap | `a_addr` | tap | `a_addr` | tap | `a_addr` |
|---|---|---|---|---|---|
| (0,0) | **0** | (1,0) | **160** | (2,0) | **320** |
| (0,1) | **16** | (1,1) | **176** | (2,1) | **336** |
| (0,2) | **32** | (1,2) | **192** | (2,2) | **352** |

all nine sharing `a_inner_stride = 16`, `a_outer_stride = 160`, `m_inner = 8`,
`m_outer = 8`, `acc_inner_stride = 16`, `acc_outer_stride = 128`. Two ACC
regions of 1024 words at 0 and 1024. The interior lands at
`1*160 + 1*16 = 176`.

One difference from the ISA's prose: the buffer is **1616 bytes, not 1600**. The
ISA's own spot check shows the last 16-byte read ending at exactly 1600, which is
in bounds — but only because `IC` happens to be 16 there. Whenever
`IC % 16 != 0` the last row reads past the last pixel, so `-kea-tile`
over-allocates by one array row unconditionally. See §6.4.

### 6.3 Padding

`kea.mm` has no predication, so padding is entirely a memory-layout problem
(ISA.md §8.4(a)). For each output tile, `-kea-tile`:

1. computes the input window `[ih0, ih0+IH_p) × [iw0, iw0+IW_p)` where
   `ih0 = oh0*S - pad_top`;
2. emits a `kea.vcopy` fill of the whole activation tile with the **input zero
   point** — not 0; for asymmetric int8 that is a different number and getting
   it wrong is a silent accuracy bug;
3. emits a `kea.dma_load` of only the in-bounds sub-rectangle, into the
   corresponding offset inside the tile.

So the halo is correct for *interior* tiles too, where part of the window is a
real neighbouring pixel and part may still be off the image edge.

### 6.4 Why the fill is unconditional

It would be cheaper to fill only tiles that touch the image border. `-kea-tile`
fills every one, for two correctness reasons rather than convenience:

* **The tail pad is read.** `kea.mm` always reads 16 activation bytes per row.
  ISA.md §2.3 says a simulator should poison unwritten scratchpad and trap on
  reads of it, and MICROARCH.md §8.2 calls an unsynchronized read of a DMA
  target "the single most likely compiler bug". A tile whose last 15 bytes are
  poison would produce exactly that report, for a read whose arithmetic is
  provably harmless.
* **`DWCONV` reads padded channel lanes.** `channels` is rounded up to a
  multiple of 16, so a 24-channel depthwise reads 8 lanes no DMA wrote. Their
  weight planes are zero, so again the arithmetic is right and the read still
  has to be defined.

The cost is `KEA_VPU_ISSUE_OVERHEAD + ceil(bytes/32)` cycles on a unit that is
idle at that point, against a DMA of the same tile at 16 B/cycle — at most half
the DMA it precedes, and `-kea-schedule` hides it entirely. It is priced into
the cost model (§5.3).

### 6.5 Channel packing for the first layer (ISA.md §8.6)

**Implemented.** When `KW > 1`, dilation is 1, `IC < 16` and `KW * IC <= 16`, a
whole kernel *row* becomes one reduction tile:

* `k_rows = KW * IC` (9 for the `IC = 3` first layer),
* `KH` taps instead of `KH * KW`,
* `a_addr = A_base + kh*D*sr` — the `kw` term disappears entirely, because `kw`
  now lives *inside* the reduction tile,
* `a_inner_stride = S * sp = 2 * 3 = 6`,
* weight `layout = "mxu_tiles_16x16_packed"`.

MobileNetV2's first layer goes from 19% to 56% MXU utilization (MICROARCH.md
§9.3(a)) with no new instruction — it is purely a weight-layout choice. The
gating requires `KW > 1` because for `KW == 1` the packed and unpacked tilings
are byte-identical and the layout name would only confuse `-kea-emit`.

### 6.6 Depthwise

`DWCONV` with `channels = roundUp(C, 16)`, `a_pix_stride = C_pad`,
`a_row_stride = IW_p * C_pad`, a dense ACC region of `OH_t*OW_t*C_pad` words, and
a `VQUANT` with `acc_pix_stride = out_pix_stride = C_pad`. When `C_pad != C` the
activation DMA gathers `C` of every `C_pad` bytes (one run per pixel) and the
output DMA scatters them back; when `C_pad == C` — which is every real
MobileNetV2 depthwise layer, all of 32/96/144/192/384/576/960 being multiples of
16 — both are one contiguous run per row.

---

## 7. Discharging the errata and ADR-0003

### 7.1 Errata E5 — the int32 ACC wraps

`-kea-tile` computes the reduction chain length accumulated into one ACC word:

```
taps = ceil(IC/16) * 16 * KH * KW          (or KH*KW*IC when channel packed)
```

and asserts `taps * 127 * 127 < 2^31`, i.e. `taps <= 133,144`. A layer that
exceeds it is rejected with a diagnostic naming the bound, rather than being
allowed to wrap silently. MobileNetV2's largest is 960 (the `960→320` pointwise),
comfortably inside — but it is a real bound and it is checked, not assumed.

### 7.2 ADR-0003 — the normalised-multiplier invariant

> For every requantization the compiler emits, either `tosa_shift >= 31`, or the
> accumulator satisfies `|acc + bias| < 2^tosa_shift`.

This is live, not hypothetical: MobileNetV2's rescale shifts run from **22 to
48**, so a blanket `tosa_shift >= 31` rule would reject the network the machine
exists to run.

`-kea-tile` checks the blanket rule first and stops if it holds for every
channel. Otherwise it computes a **per-output-channel accumulator range bound**
from the actual constant weights:

```
bound[c] = 255 * sum_k |w[c][k]|  +  |bias[c]|
```

`255` is `max |a - a_zp|` for int8 activations with an int8 zero point; the
Level 1 semantics is `sum_k (a - a_zp) * w + bias` with `w_zp = 0`, so this is
a true upper bound on what `VQUANT` requantizes. Every channel with
`tosa_shift < 31` must satisfy `bound[c] < 2^tosa_shift`, and if one does not,
the pass **fails loudly** with the channel, the shift, the bound and the
threshold — never emitting a stream the frontend's golden model cannot
reproduce.

Using the real weights matters. The worst-case `K * 127 * 127` bound E5 uses is
true but so loose that it rejects working layers: a `tosa_shift = 22` rescale
after an `IC = 960` pointwise needs `bound < 4,194,304`, and the worst case is
`15.5 M` while real trained weights land two orders of magnitude below it. When
the weights are *not* a compile-time constant the pass falls back to the worst
case and, if that cannot discharge the invariant, refuses. Both directions are
tested (`kea-tile-invalid.mlir`: `@adr0003_bound_too_large` fails,
`@adr0003_bound_discharged` compiles, and they differ only in the weight values).

One bound, two obligations: E5 and ADR-0003 use the same computation, as the ADR
anticipated.

### 7.3 Errata E6 — `KeaAddParam` bounds

Converting a Level 1 `#kea.quant` triple into a `KeaAddParam` is not a copy. The
two formulations differ by two fixed amounts:

* ADR-0003's `kea_shift = tosa_shift - 31`, because `keaSrdhm` absorbs a `>>31`
  TOSA carries explicitly;
* `KEA_VADD_LEFT_SHIFT` (20), which `keaQuantizedAdd` applies to both *inputs*
  and TOSA has no counterpart for.

so the exact input shifts are `tosa_shift - 11`. That is frequently negative
(MobileNetV2's residual rescales come out at `tosa_shift` 10 and 11), and
`keaRdpot` **silently ignores a negative exponent** — it is `return x` for
`exp <= 0`. It cannot be folded into the multiplier either: a normalized
multiplier is already in `[2^30, 2^31)` and doubling it overflows int32.

The fix is to shift *both* inputs down by the same `d = max(0, 11-T_a, 11-T_b)`,
which is exact because both terms scale together, and to take the same `d` back
out of the output shift:

```
a_shift = T_a - 11 + d      b_shift = T_b - 11 + d      o_shift = T_o - 31 - d
```

For the MobileNetV2 inverted residual (`T_a = 11, T_b = 10, T_o = 40`) that is
`d = 1`, giving `a_shift = 1, b_shift = 0, o_shift = 8` — all non-negative, and
asserted in `compiler/test/mobilenet-e2e.mlir`. If `o_shift` comes out negative
the pass fails loudly rather than emitting a rescale that would silently vanish.

Then E6 itself. `mlir::kea::addParamWorstCaseSum()` evaluates

```
|sa| <= 255 << KEA_VADD_LEFT_SHIFT
|xa| <= ((|sa| * a_mult) >> 31) >> max(a_shift, 0)
```

for both terms and `-kea-tile` requires `|xa| + |xb| < 2^31`. For the residual
above that is ~2.3e8, an order of magnitude clear. Note the derivation of `d`
*helps* here: shifting both inputs down can only shrink the sum.

### 7.4 Errata E7 — no `MATMUL` against an unwritten weight bank

`mlir::kea::verifyWeightBanks()` walks each block in program order tracking
which of the two MXU banks a `kea.load_w` has written, and reports the first
`kea.mm` that reads an unwritten one. `-kea-tile` runs it over its own output
before returning, so the check is not merely available, it is *exercised on
every compile*. It is not an op verifier because it is not a property of one op.

Running `-kea-tile` on a function that is already Level 2 does nothing but this
check plus `refreshLiveRanges`, which makes it usable as a validator for
hand-written IR — that is how `@e7_unwritten_weight_bank` is tested.

---

## 8. Options and statistics

```bash
kea-opt in.mlir -kea-tile
kea-opt in.mlir -kea-tile=spm-reserve-factor=1     # spend the whole scratchpad
kea-opt in.mlir -kea-tile=report-tiles=true        # publish kea.tiling
```

`spm-reserve-factor` divides each scratchpad's capacity before tiling; the
default 2 leaves `-kea-schedule` room for a second set of tiles. `1` produces
the largest tiles and a program that cannot be double buffered.

`report-tiles` attaches a `kea.tiling` array to the function: one dictionary per
lowered layer with the chosen tile, the SPM_A / SPM_W / ACC footprint, the
reduction chain length and the estimated cycles. Pass `Statistic`s are compiled
out by this LLVM install (Release+NDEBUG ⇒ `NoopStatistic`; see
`compiler/README.md` gotcha 21a), so this attribute is how the tiling is
observed and FileCheck-tested.

---

## 9. What is not implemented

Stated plainly rather than stubbed:

* **Inter-layer SPM residency.** Every layer reads its input from DRAM and
  writes its output to DRAM, even when the tensor would fit on chip. This is
  correct but leaves DRAM traffic on the table for exactly the layers
  MICROARCH.md §9.3(c) says should never reach DRAM. It is a separate pass, not
  a change to this one: it needs a whole-function liveness and residency
  analysis that has nothing to do with tiling.
* **Ragged tiles.** `OH_t` and `OW_t` are divisors of `OH` and `OW`. A layer
  whose only divisors are 1 and itself gets a coarse choice.
* **Region splitting for padding** (ISA.md §8.4(b)). Only the recommended zero
  halo is implemented. Option (b) trades memory for instructions and is only
  worth it under SPM pressure the tiler has not yet seen.
* **int4.** The ops, verifiers and cost model all carry it (`int4` flags,
  8-byte `LOAD_W` alignment, `keaMatmulOccupancy(..., int4)`), but `-kea-tile`
  never emits it — nothing upstream produces an int4 Level 1 graph yet.
* **Batch > 1** for `kea.conv2d` and `kea.dwconv2d`. `kea.matmul` loops its
  batch dimension; the convolutions reject `N != 1` rather than pretending.
* **`kea.transpose`, standalone `kea.rescale`, standalone `kea.clamp`.** KEA-1
  has no transpose unit (ISA.md §13) and no way to requantize a tensor that
  never entered ACC. `-kea-fuse` is expected to eliminate all three; if one
  survives, `-kea-tile` refuses with a diagnostic saying so.
* **Dilated depthwise.** `DWCONV` has no dilation field. ISA.md §9.1 suggests
  emulating it by scaling the activation strides, but that also scales the
  *output* stride, so it needs a re-slice that is not implemented.
* **Padded pooling.** `VPOOL` is VALID only, and a padded average pool also
  needs TOSA's padding-aware divisor, which the unit does not have.
* **`NOP`.** No op, on purpose (§1).
