# The backend — `kea-translate`, `keac`, and the three files

**Status: implemented.** Everything in this document is exercised by
`compiler/test/kea-emit*.mlir`, `tools/keac/tests/` and
`bash scripts/build_compiler.sh` + `bash scripts/build.sh`.

[ADR-0001](adr/0001-text-assembly-as-the-compiler-backend-boundary.md) puts text
assembly between the MLIR compiler and the binary artifact. This is the half
above that line: scheduled, allocated Level 2 MLIR in, three files out, and
never a byte of KEAF.

| | |
|---|---|
| Tool | `kea-translate`, `compiler/tools/kea-translate/` |
| Library | `compiler/lib/Target/Kasm/{EmitKasm,ConstBlob}.cpp` |
| Driver | `keac`, `tools/keac/keac.cpp` |
| Input contract | [DIALECT_L2.md](DIALECT_L2.md) §1.1, §4 — `(buffer, displacement)` addresses, roles, layouts |
| Address source | [MEMORY_PLANNING.md](MEMORY_PLANNING.md) — `addr` on each `kea.alloc`, `kea.dram_layout` on the function |
| Output contract | [ASSEMBLY.md](ASSEMBLY.md) — **owned by the assembler**; this conforms to it |
| Tests | `compiler/test/kea-emit{,-const,-map,-sync,-invalid,-e2e}.mlir`, `tools/keac/tests/` |

```
model.tosa.mlir
   │  kea-opt --tosa-to-kea --kea-fuse --kea-tile [--kea-schedule] --kea-alloc
   ▼
Level 2 MLIR
   │  kea-translate
   ├──────────────┬───────────────────┐
   ▼              ▼                   ▼
model.kasm   model.weights.bin   model.map.json
   └──────────────┴───────────────────┘
                  │  kea-as
                  ▼
             model.keaf  ──▶ kea-rt / kea-sim
```

---

## 1. Why a binary and not a pass

`-kea-emit` is spelled `kea-translate` for three reasons, and the name in
DIALECT_L2.md is kept as the *stage* name.

* It produces **three** outputs, one of them binary, and consumes IR without
  producing any. That is what upstream calls a translation and gives its own
  driver (`mlir-translate`); a pass would have to write files as a side effect
  of running, which makes `kea-opt in.mlir -kea-emit` print nothing useful.
* `kea-opt` stays an IR-to-IR tool with no file-writing or CLI surface.
* `kea-translate` links only `MLIRKeaDialect`, the emission library, `arith`
  and `func`. It is a fraction of `kea-opt`'s 108 MB and is not on the critical
  path of an ordinary rebuild.

---

## 2. The emission model

One Level 2 op becomes one line. There is exactly one piece of arithmetic:

```
instr.X_addr = addr(X) + X_addr
```

`addr(X)` is what `-kea-alloc` stamped on the `kea.alloc`; `X_addr` is the
displacement `-kea-tile` left on the op. **There is no unit conversion here and
there must never be one** — DIALECT_L2.md §1.1(b) makes every offset and stride
already count in the element type of the buffer it indexes, so an ACC
displacement is words because ACC buffers are `i32`, and an SPM one is bytes
because SPM buffers are `i8`.

That fact is carried straight into the text, because ASSEMBLY.md §1.2 makes the
units visible in the syntax:

| space | address | stride |
|---|---|---|
| SPM_A | `a:<byte>` | plain integer |
| SPM_W | `w:<byte>` | plain integer |
| ACC | `acc:<word>` | `<n>w`, mandatory |
| DRAM | `@symbol[+off]` | plain integer |

DRAM stays symbolic (ADR-0001 rule 3): the displacement is emitted as
`@name+off` and the assembler adds it to the symbol's `offset` in
`model.map.json`. `dram:<n>` is never emitted.

Field names and their order are `kea::keaOpInfo()`'s, i.e.
`runtime/src/op_fields.cpp`'s. Every operand is written explicitly; ASSEMBLY.md
§3 has no defaults and no optional fields, deliberately, because a defaulted
`acc_mode` is exactly the kind of bug that produces plausible wrong numbers.

### 2.1 Units, regions and labels

`unit` on an op is honoured wherever `-kea-schedule` put one. Absent, the
opcode decides: DMA on `DMA0`, `LOAD_W`/`MATMUL` on `MXU`, `DWCONV` on `DWU`,
the four VPU ops on `VPU`, `HALT` on `CTRL`.

`kea.trace` is queue-agnostic and needs a choice. A `begin`/`end` pair **must
land on the same queue** or the region never closes — regions nest per
`(unit, tag)` (ISA.md §5.4) — so both go on the unit that does the most
interesting work inside the region, preferring `MXU > DWU > VPU > DMA0 > DMA1`.
That is also what makes `kea-sim`'s per-region roofline mean something.

Each region gets a `.region <tag>, "<name>"` directive and a `layer<tag>:`
label. The name is the layer prefix `-kea-tile` gave that layer's buffers
(`mobilenet_v2_inverted_residual.1`), so a line of the simulator's per-region
report is traceable back to the graph. `--no-labels` omits the labels.

---

## 3. Synchronization — what the emitter inserts, and why that is not scheduling

`-kea-tile` emits a correct **sequential** program (DIALECT_L2.md §4.6): right
if executed one instruction at a time. KEA-1 does not execute one instruction
at a time. Five queues run concurrently and the only ordering between them is
the 32 counting semaphores (ISA.md §5.3), so an unsynchronized stream is a set
of data races and running it proves nothing.

So when the IR carries no `kea.signal`/`kea.wait`, `kea-translate` inserts
exactly the cross-unit edges the sequential program's dependencies imply:

* **RAW** — a read whose last writer ran on another unit;
* **WAW / WAR** — a write whose last writer, or any reader since, ran on
  another unit.

**It is not a scheduler.** Nothing is reordered, hoisted, software pipelined or
double buffered; the instruction order is exactly the IR's. When
`-kea-schedule` runs it owns synchronization and the emitter passes its
semaphores through untouched. `--sync` selects: `auto` (default — insert iff
the IR has none), `insert`, `none`.

### 3.1 Dependencies are tracked over storage, not over SSA values

This is the part that is easy to get wrong. `-kea-alloc` gives two buffers the
same address whenever their block-order live ranges are disjoint
(MEMORY_PLANNING.md §2.2). Layer 0's ACC region and layer 1's ACC region really
*are* the same 2048 words. Keying the dependence analysis on the `kea.alloc`
result would miss that and emit a program whose second layer starts overwriting
the accumulator the first layer's `VQUANT` is still reading — a data race that
looks exactly like a kernel bug.

So each op contributes intervals `(space, [addr, addr+extent))` and two
accesses conflict when their intervals overlap and at least one is a write.
`compiler/test/kea-emit-sync.mlir`'s `@aliased` pins it.

### 3.2 The counting scheme

One event per ordered unit pair `(producer, consumer)`, named after it:
`.event 3, "MXU_to_VPU"`. At most 20 of the 32 semaphores can ever be used.

Signals on a pair are emitted in producer order; a unit's queue is in-order and
`SIGNAL` is a local drain barrier (ISA.md §5.3), so the event's value is
`(completed producers on that pair) - (counts already consumed)`. A consumer
that needs producer number *r* waits for `r - consumed` counts, which is **0 —
no instruction at all** — when an earlier `WAIT` on the same queue already
covered it. That is why the MXU below waits twice for one `DMA0_to_MXU` event
rather than once for a threshold of 2, and why the second `MATMUL` waits not at
all:

```
  DMA0  DMA_LD  ... @conv.weights ...
  DMA0  SIGNAL  event=0, inc=1
  DMA0  DMA_LD  ... @input ...
  DMA0  SIGNAL  event=0, inc=1
  MXU   WAIT    event=0, threshold=1      ; the weights are in
  MXU   LOAD_W  ...
  MXU   WAIT    event=0, threshold=1      ; ... and now the activations
  MXU   MATMUL  ...
  MXU   SIGNAL  event=1, inc=1
  VPU   WAIT    event=1, threshold=1
  VPU   VQUANT  ...
```

**Rule D (ISA.md §5.5) holds by construction**: every supplying `SIGNAL` sits
at a smaller stream position than the `WAIT`, because every producer does.
`kea-as` re-checks it and refuses the program otherwise.

---

## 4. The constant blob

`compiler/lib/Target/Kasm/ConstBlob.cpp`. Every DRAM `kea.alloc` whose `role`
is `weights`, `qparam` or `addparam` is materialized in the byte layout its
`layout` attribute names (DIALECT_L2.md §4.5) and written at
`addr - dram.const_offset` in a blob sized from `dram.const_bytes` — including
the internal alignment padding, because ASSEMBLY.md §7.2 requires
`const_bytes` to equal the size of the `--const` file **exactly**. A mismatch
between the declared extent and what the layout produces is a hard error, not a
truncation: that is the stale-weight-blob bug, which is otherwise silent.

`kea-translate --emit-const-listing=<file>` prints the same bytes decoded. Use
it whenever a layer's numbers are wrong; §8.

### 4.1 `mxu_tiles_16x16` — ISA.md §8.1

Source `[OC, KH, KW, IC]` (or `[OC, IC]` for a fully connected layer, which is
the degenerate `KH = KW = 1`). Dense 16×16 int8 tiles of 256 bytes, ordered
`[oc0][ic0][kh][kw]`:

```
tile(oc0, ic0, kh, kw) = ((((oc0 * icTiles + ic0) * KH + kh) * KW + kw)) * 256
byte k*16 + n of that tile = W[oc0*16 + n][kh][kw][ic0*16 + k]
```

with `icTiles = ceil(IC/16)`, and **zero wherever the weight array does not
reach** — `k >= IC - ic0*16` or `n >= OC - oc0*16`. Those zeros are what make
`LOAD_W`'s `k_rows` / `n_cols` tail tiles work with no masking in `MATMUL`
(ISA.md §7.2). `w_row_stride` is therefore always 16.

Note the transpose: the array is `[oc][…][ic]` and the tile is `[k][n]`, so
`ic` becomes the row and `oc` the column.

**Worked example** (`compiler/test/kea-emit-const.mlir`), `OC = 2`, `IC = 3`,
one tap, `W = [[[[1,2,3]]], [[[4,5,6]]]]` — one 256-byte tile:

```
     0:    1    4    0 …      k=0: W[0][0][0][0]=1, W[1][0][0][0]=4
    16:    2    5    0 …      k=1
    32:    3    6    0 …      k=2
    48:    0    0    0 …      k=3..15: past IC, loaded as zero
```

### 4.2 `mxu_tiles_16x16_packed` — ISA.md §8.6

For `KW * IC <= 16` with dilation 1 and `KW > 1`: a whole kernel *row* is one
reduction tile, so there are `KH` tiles per output-channel group instead of
`KH * KW`, and the reduction row index is `k = kw * IC + ic`:

```
tile(oc0, kh) = (oc0 * KH + kh) * 256
byte k*16 + n = W[oc0*16 + n][kh][k / IC][k % IC]
```

This is the MobileNetV2 first layer (`IC = 3`, `k_rows = 9`), 19% → 56% MXU
utilization for the price of a weight layout. The `kw` term disappears from
`MATMUL.a_addr` entirely, because `kw` now lives inside the reduction tile.

**Worked example**, `OC = 2`, `KH = KW = IC = 2`,
`w[oc][kh][kw][ic] = oc*8 + kh*4 + kw*2 + ic + 1`:

```
tile kh=0, at 0            tile kh=1, at 256
     0:  1  9   (kw0,ic0)     256:  5 13
    16:  2 10   (kw0,ic1)     272:  6 14
    32:  3 11   (kw1,ic0)     288:  7 15
    48:  4 12   (kw1,ic1)     304:  8 16
```

### 4.3 `mxu_tiles_16x16_kn`

The batched-matmul rhs, `[B, K, N]`. As `mxu_tiles_16x16` with `oc0 → n0`,
`ic0 → k0`, `KH = KW = 1`, and the whole tile array repeated per batch:

```
tile(b, n0, k0) = ((b * nTiles + n0) * kTiles + k0) * 256
byte k*16 + n   = W[b][k0*16 + k][n0*16 + n]
```

This is the one layout whose output channel is the **last** weight dimension,
so the tile is the array as written rather than its transpose.

### 4.4 `dwu_planes` — ISA.md §9.1

Canonical depthwise weights `[OC, KH, KW, 1]` become `[KH][KW][C_pad]` int8,
contiguous, `C_pad = roundUp(C, 16)` bytes per tap plane:

```
byte (kh * KW + kw) * C_pad + c = W[c][kh][kw][0],  zero for c >= C
```

The padding lanes must be zero and it matters: `DWCONV.channels` is rounded up
to a multiple of 16, so those lanes *are* read, and a zero weight is what makes
reading them arithmetically harmless (DIALECT_L2.md §6.4).

### 4.5 `quant_params` — the zero-point fold

`source` is `[bias?, weights]`, plus the `quant` attribute (the Level 1
`#kea.quant`) and `input_zp` (the *activation* zero point of the contraction).
For each output channel `c`, a 12-byte `KeaQuantParam`:

```
mult  = quant.multiplier[c]                          (or [0] when per tensor)
shift = quant.shift[c] - 31                          ADR-0003's convention
bias  = bias_l1[c] - input_zp * sum_k weights[c][k]   the zero-point fold
```

**The third line is the one to get right.** The MXU computes the raw
`Σ a·w`, not `Σ (a - a_zp)·w`, so the `-a_zp · Σ w` correction has to be folded
somewhere, and the bias record is the only place it fits. Drop it and every
output is biased by a per-channel constant — the network still runs, the
numbers are quietly wrong, and nothing downstream can tell.

`sum_k` runs over every reduction element of that output channel: `KH·KW·IC`
for a convolution, `IC` for a fully connected layer, `K` for a matmul. For
`mxu_tiles_16x16_kn` the output channel is the last dimension, so the sum is
over `w[b][k][c]` for fixed `c`. A batched matmul with `B > 1`, a non-zero
`input_zp` and batch-varying weights is **rejected**: one `KeaQuantParam` block
serves every batch and cannot carry a per-batch correction.

The block is `roundUp(OC, 16) * 12` bytes. The padding channels exist because
`VQUANT.channels` must be a multiple of 16; their records are **zero**, so a
zero multiplier keeps them at the output zero point, and they are never stored
for a real output.

**Worked example**, the MobileNetV2 expand layer: `bias_l1 = 128`,
`input_zp = -5`, weights all 2 with `IC = 4` so `sum_k w = 8`, `tosa_shift = 36`:

```
bias  = 128 - (-5)*8 = 168
shift = 36 - 31      = 5
  [0] bias=168 mult=1073741824 shift=5
```

### 4.6 `add_params` — DIALECT_L2.md §4.4

`add_param` is already the machine-form record, in isa.h field order
`[a_mult, b_mult, o_mult, a_shift, b_shift, o_shift, a_zp, b_zp, o_zp]`, so
this is a verbatim write into a 20-byte `KeaAddParam` (with the two reserved
bytes zeroed). `-kea-tile` derived it — including errata E6's shared shift `d`
— and publishes it rather than making the backend redo the derivation.

For the MobileNetV2 inverted residual:

```
  a_mult=1610612736 b_mult=1073741824 o_mult=1503238553
  a_shift=1 b_shift=0 o_shift=8
  a_zp=-5 b_zp=-5 o_zp=-5
```

### 4.7 `nhwc`

A dense activation constant, byte for byte. Nothing in the current pipeline
emits one, but the layout name exists in DIALECT_L2.md §4.5 and a constant
activation is a legal graph.

---

## 5. `model.map.json`

ASSEMBLY.md §7 is the schema. Most of it is a transcription:

| key | source |
|---|---|
| `dram` | `kea.dram_layout` on the function, field for field |
| `symbols` | every DRAM `kea.alloc` that is not `input`/`output`: `name`, `addr`, extent |
| `spm_map` | every on-chip `kea.alloc`, with `first_pc`/`last_pc` **recomputed** after synchronization is woven in, because `-kea-alloc`'s live ranges count block positions in the pre-sync program |
| `metadata` | producer, function, source path |

`tensors` is the part that needs facts the Level 2 IR does not carry, so each
one is either derived from the instruction stream — and says how — or defaulted
and overridable with `--io-quant <tensor>=<scale>[,<zp>]`:

* **shape**, input: `-kea-tile` names a model input `<func>.input<argno>`, so
  it is that block argument's shape, exactly.
* **shape**, output: `[1, n2, n1, len0]` when exactly one contiguous `DMA_ST`
  covers the whole buffer, which is the descriptor the tiler emits for a
  whole-tensor store. Anything else (a tiled output, several stores) leaves the
  shape unknown and the map says `FLAT` rather than inventing one.
* **zero point**, input: the value the activation tile is `VCOPY`-filled with
  before the input is DMA'd into it. ISA.md §8.4(a) requires that to be the
  input zero point, so it is read off the stream, not guessed.
* **zero point**, output: the `out_zp` of the VPU op that produced the stored
  tile — `VQUANT`'s attribute, or `KeaAddParam.o_zp` for a `VADD`.
* **scale**: **not derivable.** A TOSA `i8` tensor carries no scale; the
  frontend has it and the graph does not. The map says `1.0` unless
  `--io-quant` supplies it. Nothing on the device uses it — it exists so the
  host can dequantize with `real = scale * (q - zero_point)`.

Two DRAM activation symbols can share an offset: the scratch region is
live-range packed, so `layer0.out` and `layer1.out` are the same 1,536 bytes.
That is legal, and it is the reason the round-trip property is stated over the
artifact rather than over the text — see §7.

---

## 6. `keac`

```
keac <model.mlir> [-o <model.keaf>] [options]
```

A process driver: it runs `kea-opt`, `kea-translate` and `kea-as` in turn and
compiles nothing itself. It lives in the native half of the tree, next to
`kea-as`, because it links neither half — it only spawns them.

| flag | |
|---|---|
| `-o <file>` | artifact to write; default is the input with `.keaf` |
| `--function <name>` | which function; required when the module has more than one |
| `--keep-intermediates` | keep `<base>.l2.mlir`, `.kasm`, `.weights.bin`, `.map.json` |
| `--emit mlir\|kasm\|keaf` | stop after a stage |
| `--schedule` | insert `--kea-schedule` into the pipeline |
| `--spm-reserve <n>` | `-kea-tile=spm-reserve-factor=<n>` |
| `--sync auto\|insert\|none` | passed to `kea-translate` |
| `--io-quant <spec>` | `<tensor>=<scale>[,<zp>]`, repeatable |
| `--annotate` | annotate the `.kasm` (implies `--keep-intermediates`) |
| `--pipeline "<passes>"` | override the `kea-opt` pass list entirely |
| `--kea-opt/--kea-translate/--kea-as <path>` | tool overrides; also `$KEA_OPT`, `$KEA_TRANSLATE`, `$KEA_AS` |
| `-v` | echo every command |

The MLIR tools are found next to `keac`, then in the sibling
`build/compiler/bin`, then on `$PATH`, so the two independent builds find each
other with no environment set up.

The end-to-end command this document's numbers come from:

```bash
build/native/bin/keac tests/mlir/tosa/mobilenet_block.mlir \
    --function mobilenet_v2_inverted_residual \
    -o /tmp/mb.keaf --keep-intermediates -v

build/native/bin/kea-sim /tmp/mb.keaf --stats-json /tmp/mb.stats.json
```

`.kgraph.json` input is **not** supported: ingesting one needs the frontend's
TOSA emitter, which FRONTEND.md §5 specifies as "a mechanical walk" and which
nothing implements yet. `keac` says exactly that rather than failing obscurely.

---

## 7. What it measures

`keac` + `kea-sim` on `tests/mlir/tosa/mobilenet_block.mlir`, **without**
`-kea-schedule` — so this is the unscheduled baseline: correct, synchronized,
and with no overlap anyone asked for.

| | inverted residual | stride-2 variant |
|---|---|---|
| instructions | 105 (66 inserted `SIGNAL`/`WAIT`) | 93 (58 inserted) |
| constants | 2,292 B | 2,272 B |
| DRAM arena | 4,352 B | 4,224 B |
| **cycles** | **2,563** | **1,553** |
| MXU busy | 11.3% | 12.5% |
| DWU busy | 5.9% | 2.8% |
| VPU busy | 23.8% | 19.8% |
| DMA0 busy | **58.7%** | **62.6%** |
| dispatcher stalled | 51.1% | 48.5% |
| MAC utilisation | 1.87% of 256 int8 MAC/cycle | — |
| achieved | 23.97 GOPS of 106.8 attainable | — |
| bound | MEMORY (below the ridge point) | MEMORY |

Three things worth reading off that:

* **DMA0 is the busy unit and DMA1 is idle**, because with no scheduler every
  transfer goes on one engine. Splitting them is a scheduling decision, not an
  emission one.
* **Half the cycles are dispatcher stalls on a full DMA0 queue.** The queue is
  16 deep and the program hands it 37 descriptors with nothing to overlap them
  against.
* **MAC utilisation is 1.87%** — a padding efficiency of 18.75% (an `IC = 4`
  layer uses 4 of 16 reduction lanes) times an occupancy the lack of overlap
  keeps low. Both are known and neither is the backend's to fix: the first is
  the shape of the fixture, the second is `-kea-schedule`'s.

And with `keac --schedule`, i.e. `-kea-schedule` between tile and alloc, the
emitter inserts **nothing** and passes the scheduler's semaphores through
verbatim (`--sync=auto` sees they are there). Same program, same answer:

| inverted residual | no scheduler | `--schedule` |
|---|---|---|
| instructions | 105 (66 emitter-inserted) | 103 (**0** emitter-inserted) |
| cycles | 2,563 | **2,132** |
| DMA0 / DMA1 busy | 58.7% / 0.0% | 41.9% / 34.4% |
| dispatcher stalled | 51.1% | 15.1% |
| output vs the numpy reference | 256/256 exact | 256/256 exact |

That is the pass-through path working end to end against a pass written in
parallel with this one, and it is why `--sync=auto` defaults the way it does:
the scheduler hoists DMA to create overlap, and a second opinion from the
emitter would only serialize it again.

The round-trip property holds at the artifact level:

```
assemble(disassemble(p)) == p     byte for byte, on both functions
```

The *text* is not identical, and cannot be: `.0.out` and `.1.out` are packed at
the same DRAM offset with the same size, so an address inside them has two
equally tight covering symbols and the disassembler picks one. That renames a
symbol; it does not change a byte. `tools/keac/tests/e2e.sh` and
`compiler/test/native-tools.sh` both assert the byte-level property.

### 7.1 The numerical check

Compiling and running is not evidence that the numbers are right. Two checks,
both against a numpy reference that shares nothing with the compiler except the
frontend's normative `apply_scale_32` (QUANTIZATION.md §1):

**`tools/keac/tests/numeric_check.py`** — generates TOSA fixtures with
**distinct** weights along every axis, compiles them with `keac`, runs them on
`kea-sim` and requires bit-exact equality with the reference. The distinctness
is the point: a splat weight tensor passes under *every* wrong permutation of
the tile ordering, so it tests nothing about layout.

| case | layout under test | result |
|---|---|---|
| `conv3x3` 8→16, pad 1, `input_zp = -3` | `mxu_tiles_16x16` | 1024/1024 exact, 100% unclamped |
| `conv3x3_ic3_s2` 3→16, stride 2, `input_zp = -11` | `mxu_tiles_16x16_packed` | 256/256 exact |
| `depthwise3x3`, C = 24 (not a multiple of 16) | `dwu_planes` | 1536/1536 exact |
| `fc` 32→48 | `mxu_tiles_16x16`, rank 2 | 192/192 exact |
| `bmm` [1,6,32]×[1,32,16] | `mxu_tiles_16x16_kn` | 96/96 exact |
| `qadd`, two inputs, zero points −5 and 3 | `add_params` | 128/128 exact |
| `fc_input_zp`, the same FC with `input_zp = -9` | — | **XFAIL**, §9 |

Every scale is chosen so the outputs land in the interior of int8 rather than
against the clamps, so the comparison is against real values.

**`tools/keac/tests/block_check.py`** — the whole MobileNetV2 inverted residual
from `tests/mlir/tosa/mobilenet_block.mlir`, with a random int8 input:
**256/256 values exact**. That covers the three-layer chain, the DRAM scratch
packing, the depthwise halo and the residual `VADD` — but the fixture's weights
are splats, so it says nothing about tile ordering. The two checks are
complementary and neither is sufficient alone.

The `qadd` result is worth a note: `VADD` is gemmlowp `srdhm`/`rdpot` on inputs
pre-shifted by `KEA_VADD_LEFT_SHIFT`, which is a *different rounding* from
TOSA's rescale–add–rescale even though it is the same value. It came out
bit-identical on every seed tried (4 seeds × 128 values). That is an
observation, not a proof; ADR-0003 proves the `VQUANT` case and says nothing
about this one.

---

## 8. Debugging a miscompile by reading the `.kasm`

This is the reason ADR-0001 exists. The order below goes from cheapest to most
expensive.

**1. Compile with the intermediates kept.**

```bash
keac model.mlir -o /tmp/m.keaf --keep-intermediates --annotate
```

`--annotate` appends `; pc=N` to every line. It is **not** canonical form, so
do not feed annotated assembly back into a round-trip test.

**2. Does the assembler accept it?** If `kea-as` rejects the compiler's own
output, that is a backend bug and the diagnostic names the field, the line and
the column. `keac` says so explicitly rather than reporting a tool failure.

**3. Read the layer.** Find `layer<N>:` and read down to the next one. The
questions worth asking, in order:

* **Is `acc_mode=overwrite` on exactly one instruction per ACC region?** It
  should appear on the first tap of the first reduction tile and nowhere else
  (ISA.md §8.3). Two `overwrite`s means a tap's accumulation was lost; zero
  means the region starts with whatever the previous layer left.
* **Do the `bank`s alternate across consecutive `LOAD_W`/`MATMUL` pairs?**
  If not, every weight load stalls behind the previous `MATMUL`
  (DIALECT_L2.md §6.1).
* **Does every `WAIT` have `SIGNAL`s above it?** `kea-as` enforces Rule D, so a
  program that assembles cannot deadlock on dispatch order — but reading the
  `.event` names tells you *which* producer a consumer is waiting for.
* **Do the `a_addr`s of a conv's taps differ by `kh*D*sr + kw*D*sp`?** That is
  the addressing identity (ISA.md §8.2), and the per-tap table in
  DIALECT_L2.md §6.2 is the worked example to compare against.
* **Is the `qparam_addr` slice right?** Group *g* reads at
  `qptile + g*16*12`.

**4. Read the constants.**

```bash
kea-translate m.l2.mlir --emit-const-listing=- | less
```

`KeaQuantParam` and `KeaAddParam` records come out decoded, one per line. A
wrong `bias` here is the zero-point fold (§4.5); a wrong `shift` is the ±31
convention (ADR-0003); zeros where a real channel should be is a layout size
disagreement. The weight layouts print 16 int8 per line, which for the MXU
layouts is exactly one `k` row of one tile — so a transposed layout is visible
at a glance.

**5. Localize it in the run.**

```bash
kea-sim m.keaf --trace=/tmp/t.txt --strict-poison --strict-hazards
```

`--strict-poison` makes a read of never-written scratchpad fatal, which catches
a missing DMA or a tile that was not filled. `--strict-hazards` makes an
unsynchronized cross-unit read fatal, which catches a missing semaphore —
including one this emitter should have inserted. MICROARCH.md §8.2 calls that
"the single most likely compiler bug".

**6. Compare against the reference.** `tools/keac/tests/numeric_check.py` is
the template: put the layer in a one-function TOSA file, write the numpy
reference for it, and compare tensors rather than eyeballing.

---

## 9. What is not implemented

Stated plainly rather than stubbed.

* **`.kgraph.json` ingestion.** `keac` takes TOSA (or Level 1 `kea`) MLIR. The
  frontend's TOSA emitter — FRONTEND.md §5's "mechanical walk" — does not
  exist, so there is nothing to chain to. `keac` names the missing component.
* **Host-visible scales.** `model.map.json` says `scale: 1.0` unless
  `--io-quant` supplies one, because a TOSA `i8` tensor carries no scale. When
  the frontend emitter lands it should pass the `.kgraph.json` scales through;
  the flag is already there.
* **int4.** The emitter writes `dtype=int4` wherever the `int4` unit attribute
  is set and the layouts would need 8-byte tiles, but nothing upstream produces
  an int4 graph, so the packed weight layouts are untested and
  `mxu_tiles_16x16` assumes one byte per weight.
* **`--annotate` does not name buffers yet.** It emits `; pc=N`. The `spm_map`
  in the artifact already carries the names, and `kea-dis --annotate` prints
  them, so the information is available from the other side.
* **A known upstream defect, caught by the numerical check.** `-kea-tile` sets
  `ConvShape::inputZp` only on the convolution path
  (`s.inputZp = conv.getZeroPoints().getInput()`); the `fully_connected` and
  `matmul` paths leave it at 0. So the `input_zp` attribute on their qparam
  block is 0 and the zero-point fold silently drops its correction term. Every
  fully connected or matmul layer with a non-zero activation zero point — which
  is every one a real PTQ export produces — is wrong by a per-channel constant.
  The emitter materializes exactly what the attribute says, and `fc` / `bmm`
  with `input_zp = 0` are bit-exact, so the fix is one line in
  `compiler/lib/Transforms/Tile.cpp`, in a pass this backend does not own.
  `tools/keac/tests/numeric_check.py`'s `fc_input_zp` case is the XFAIL that
  will turn into an XPASS the day it is fixed.
