# KEA-1 Instruction Set Architecture

**Status: FROZEN.** Revision `KEA-1` (`KEA_ISA_REVISION == 1`).
Normative machine-readable form: [`include/kea/isa.h`](../include/kea/isa.h) and
[`include/kea/hw_config.h`](../include/kea/hw_config.h).

Where this document and the headers disagree, **the headers win** — but that is a bug, please
report it. Every field table below is backed by a `static_assert` on `offsetof`.

Companion documents: [MICROARCH.md](MICROARCH.md) (timing model, simulator reference),
[ARTIFACT_FORMAT.md](ARTIFACT_FORMAT.md) (KEAF file format).

---

## 1. Design philosophy

KEA-1 is an edge inference accelerator with exactly one strong opinion:

> **Everything that can be decided at compile time is decided at compile time.**

Concretely, that means:

- **No branches, no loops, no predication, no exceptions.** The instruction stream is
  straight-line. The only control-flow instruction is `HALT`.
- **No caches, no MMU, no coherence, no runtime allocator.** All four address spaces are flat,
  software-managed, and disjoint. Every byte of scratchpad is placed by the compiler's static
  memory planner.
- **No hardware dependency tracking between units.** Five units run concurrently and never look
  at each other's registers. Cross-unit ordering is expressed exclusively with 32 counting
  semaphores.
- **No convolution instruction.** Convolution is a compiler lowering onto `LOAD_W` + `MATMUL`
  (§8). The hardware knows only about GEMM tiles, depthwise sliding windows, and elementwise
  passes.

The payoff is that double-buffering, prefetch distance, compute/DMA overlap, and fusion all
become *scheduling* problems solvable in the compiler with a cost model, rather than emergent
properties of hardware you cannot see. That is the point of the project. The cost is that a
badly scheduled program runs badly and a malformed one deadlocks — the machine will not save
you.

---

## 2. Machine model at a glance

```
                 ┌──────────────────────────────────────────────┐
   IMEM (1 MiB)  │  fetch / dispatch  — 1 instr/cycle, in order │
   32768 instrs  └───┬──────┬──────┬───────┬────────┬───────────┘
                     │      │      │       │        │        (queue full ⇒ dispatcher stalls)
                  ┌──▼──┐┌──▼──┐┌──▼──┐┌───▼───┐┌───▼───┐
                  │ MXU ││ DWU ││ VPU ││ DMA0  ││ DMA1  │   depth-16 in-order queues
                  └──┬──┘└──┬──┘└──┬──┘└───┬───┘└───┬───┘
                     │      │      │       └────┬───┘
        ┌────────────┴──────┴──┐   │            │
        │  SPM_A 256 KiB       │◀──┴────────────┤  16 B/cycle aggregate
        │  SPM_W 256 KiB       │◀───────────────┤  ⇄  DRAM (4 GiB)
        │  ACC   32768 × int32 │                │
        └──────────────────────┘                │
                     ▲                          │
              32 event counters ────────────────┘
```

### 2.1 Execution units

| id | Unit   | Queue? | Opcodes                        | Peak                    |
|----|--------|--------|--------------------------------|-------------------------|
| 0  | `MXU`  | yes    | `LOAD_W`, `MATMUL`             | 256 int8 MAC/cyc, 512 int4 MAC/cyc |
| 1  | `DWU`  | yes    | `DWCONV`                       | 128 int8 MAC/cyc        |
| 2  | `VPU`  | yes    | `VQUANT`, `VADD`, `VPOOL`, `VCOPY` | 16 elem/cyc         |
| 3  | `DMA0` | yes    | `DMA_LD`, `DMA_ST`             | 16 B/cyc                |
| 4  | `DMA1` | yes    | `DMA_LD`, `DMA_ST`             | 16 B/cyc                |
| 5  | `CTRL` | **no** | `HALT` (and any CTRL-class op) | 1 instr/cyc             |

`NOP`, `SIGNAL`, `WAIT` and `TRACE` are *queue-agnostic*: their `unit` field names whichever
queue should execute them. `HALT` must target `CTRL`. Everything else must target its owning
unit (DMA ops may pick either engine).

Units 0–4 own a 16-entry in-order instruction queue. Unit 5 is the dispatcher itself: a CTRL-
targeted instruction retires the moment it is dispatched.

### 2.2 Address spaces

All four spaces are disjoint, flat, and base-0. There is no aliasing and no translation.

| Space   | Size            | Addressed in | Written by            | Read by             |
|---------|-----------------|--------------|-----------------------|---------------------|
| `SPM_A` | 256 KiB         | bytes        | DMA, VPU              | DMA, MXU, DWU, VPU  |
| `SPM_W` | 256 KiB         | bytes        | DMA, VPU (`VCOPY`)    | DMA, MXU, DWU, VPU  |
| `ACC`   | 32768 × int32 (128 KiB) | **int32 words** | MXU, DWU | VPU                 |
| `DRAM`  | 4 GiB           | bytes        | DMA                   | DMA                 |

> **`ACC` addresses are word indices, not byte offsets.** `acc_addr = 16` means the 17th int32,
> i.e. byte 64. This trips everyone once. It buys 2 bits of encoding and makes the natural unit
> of an accumulator tile (16 lanes) exactly 16.

DRAM is 32-bit (4 GiB). This is deliberate: an edge part never sees more, and it keeps the 3D
DMA descriptor inside the 32-byte instruction (§6). KEAF records DRAM offsets as `uint64` so a
future revision can widen it without a format break.

### 2.3 Reset state

At `START` the machine has:

- All 32 event counters = 0.
- All queues empty, all units idle, `PC = KeafHeader::entry_pc` (normally 0).
- `SPM_A`, `SPM_W`, `ACC`, and both MXU weight banks **undefined**. The program must write
  before it reads. Simulators should poison these regions and trap on reads of poison.
- DRAM contents as staged by the runtime from the KEAF `CONST` section plus the input tensors.

The machine is *done* when `HALT` has been dispatched **and** all five queues are empty **and**
all five units are idle. The runtime observes completion by polling for done, not by observing
`HALT` itself; instructions dispatched before `HALT` are always allowed to finish.

---

## 3. Datatypes

| Name    | Bits | Range        | Where                                              |
|---------|------|--------------|----------------------------------------------------|
| `int8`  | 8    | −128 … 127   | activations, weights, VPU outputs                   |
| `int4`  | 4    | −8 … 7       | MXU activations/weights, `VQUANT` output            |
| `int32` | 32   | full         | `ACC`, biases, requantization multipliers           |

**int4 is an MXU-only datatype.** `LOAD_W`, `MATMUL` and `VQUANT` (output side) support it.
`DWCONV`, `VADD` and `VPOOL` are int8-only in KEA-1; their int4 flag bits are reserved and must
be zero.

### 3.1 int4 packing (normative)

Two elements per byte, **little-nibble-first**:

```
byte i  bits 0..3  →  element 2i      (low nibble)
byte i  bits 4..7  →  element 2i + 1  (high nibble)
```

Values are sign-extended from 4 bits. An int4 vector always begins at a byte boundary, i.e. at
an even element index. Helpers: `kea::keaPackInt4`, `kea::keaUnpackInt4`.

---

## 4. Instruction format

**Fixed 32 bytes. Little-endian. No exceptions, no prefixes, no variable length.** There is no
instruction cache; decode is a fixed-offset field extract with no shifting across byte
boundaries except inside explicitly documented flag bytes.

Every instruction begins with the same 4-byte header:

| Byte | Field    | Meaning                                                        |
|------|----------|----------------------------------------------------------------|
| 0    | `opcode` | `kea::Opcode`                                                  |
| 1    | `unit`   | `kea::Unit` — the queue this instruction is pushed into        |
| 2    | `flags`  | opcode-specific bit flags; **reserved bits must be 0**         |
| 3    | `aux`    | opcode-specific byte (event id, `n2`, zero point, fill value…) |

Bytes 4…31 are opcode-specific. All unused trailing bytes are reserved and must be zero; a
decoder may reject non-zero reserved bytes.

### 4.1 Opcode map

The **high nibble names the unit class**, so a disassembler can bucket an opcode with a shift.

| Opcode   | Value  | Unit class | Summary                                             |
|----------|--------|------------|-----------------------------------------------------|
| `NOP`    | `0x00` | CTRL       | occupy a queue slot for N cycles                    |
| `HALT`   | `0x01` | CTRL       | stop instruction fetch                              |
| `SIGNAL` | `0x02` | CTRL       | `event[e] += inc`                                   |
| `WAIT`   | `0x03` | CTRL       | block until `event[e] >= thr`, then `-= thr`        |
| `TRACE`  | `0x04` | CTRL       | emit a profiling marker; architecturally a NOP      |
| `DMA_LD` | `0x10` | DMA        | DRAM → SPM_A/SPM_W, 3D strided                      |
| `DMA_ST` | `0x11` | DMA        | SPM_A/SPM_W → DRAM, 3D strided                      |
| `LOAD_W` | `0x20` | MXU        | SPM_W → 16×16 weight bank                           |
| `MATMUL` | `0x21` | MXU        | stream SPM_A rows through the array into ACC        |
| `DWCONV` | `0x30` | DWU        | depthwise conv, SPM_A × SPM_W → ACC                 |
| `VQUANT` | `0x40` | VPU        | ACC int32 → SPM_A int8/int4, requantize             |
| `VADD`   | `0x41` | VPU        | quantized elementwise add                           |
| `VPOOL`  | `0x42` | VPU        | max/avg pool                                        |
| `VCOPY`  | `0x43` | VPU        | 2D strided copy or constant fill                    |

Unassigned encodings are **illegal**, not reserved-NOP. `kea::keaValidate()` rejects them.

### 4.2 Textual form

Tools print instructions as:

```
<pc>  <unit>  <MNEMONIC>  field=value, field=value, ...
```

`kea::keaOpInfo(op)` returns the mnemonic plus a comma-separated, print-ordered list of field
names, so a disassembler needs no per-opcode printing code. Example:

```
0042  MXU   MATMUL  a_addr=352, a_inner_stride=16, a_outer_stride=160, m_inner=8, m_outer=8,
                    acc_addr=0, acc_inner_stride=16, acc_outer_stride=128, bank=1,
                    acc_mode=accumulate, dtype=int8
```

---

## 5. Control instructions

### 5.1 `NOP` (0x00)

| Bytes  | Field    | Type   | Notes                                   |
|--------|----------|--------|-----------------------------------------|
| 0–3    | header   |        | `aux` reserved                          |
| 4–7    | `cycles` | `u32`  | additional occupancy on `unit`, may be 0|
| 8–31   | reserved |        | zero                                    |

Consumes a queue slot on `unit` and occupies that unit's pipe for exactly `cycles` cycles — no
fixed setup cost is added. Exists for scheduling experiments and simulator self-tests; the
compiler should not need it.

### 5.2 `HALT` (0x01)

| Bytes | Field       | Type  | Notes                          |
|-------|-------------|-------|--------------------------------|
| 0–3   | header      |       | `unit` **must** be `CTRL`      |
| 4–7   | `exit_code` | `u32` | surfaced by the runtime; 0 = ok|
| 8–31  | reserved    |       | zero                           |

Stops instruction fetch. Must be the last instruction in the stream. Anything after it is
unreachable and should be rejected by the assembler.

### 5.3 `SIGNAL` (0x02) and `WAIT` (0x03)

Both share one layout:

| Bytes | Field    | Type  | Notes                                                  |
|-------|----------|-------|--------------------------------------------------------|
| 0–3   | header   |       | `aux` = event id, 0…31; `flags` reserved               |
| 4–7   | `value`  | `u32` | `SIGNAL`: increment. `WAIT`: threshold. ≤ `0x7FFFFFFF` |
| 8–31  | reserved |       | zero                                                   |

#### Semantics (normative — the simulator and the compiler both depend on every word here)

There are **32 counting semaphores**, `event[0..31]`, each an unsigned 32-bit counter, all zero
at reset. They are a global resource shared by all five queues.

**`SIGNAL e, inc`** executed by unit *U*:

1. Wait until every instruction that *U* issued before this `SIGNAL` has **retired** — i.e.
   all of their architectural writes are visible. `SIGNAL` is therefore a *local drain barrier*
   for its own unit; it does not affect any other unit.
2. Atomically `event[e] += inc`.
3. Retire. `SIGNAL` never blocks on the counter and never fails.

Step 1 is what makes the whole synchronization scheme sound: a `SIGNAL` observed by another
unit implies all of the signalling unit's prior work is complete. It also costs the unit's
pipeline depth (32 cycles for MXU, 12 for DWU, 8 for VPU, 0 for DMA), so place `SIGNAL`s where
that drain is hidden behind another unit's work.

**`WAIT e, thr`** executed by unit *U*:

1. Block at the head of *U*'s queue until `event[e] >= thr`. While blocked, *U* executes
   nothing else; its queue fills behind the `WAIT`.
2. On the cycle the predicate holds, atomically `event[e] -= thr`.
3. Retire in the same cycle. `WAIT` has no pipeline latency of its own.

`WAIT` is a **counting acquire**: it consumes exactly `thr` counts. Two units waiting on the
same event therefore split its counts rather than both being released.

Additional rules:

- `thr == 0` retires immediately and decrements nothing.
- If several units become unblocked on the same event in the same cycle and the counter cannot
  satisfy all of them, they are served in **ascending unit-id order** (MXU, DWU, VPU, DMA0,
  DMA1). This makes simulation deterministic.
- A `SIGNAL` and a `WAIT` on the same event in the same cycle: the `SIGNAL`'s increment is
  applied first, then waiters are evaluated. So a same-cycle signal can release a waiter.
- Exceeding `KEA_EVENT_MAX` (`0x7FFFFFFF`) is a program error. Simulators must report it. It
  means a producer is running unbounded ahead of its consumer, which also means the program is
  not actually double-buffered.
- If all queues are blocked or empty, no unit is busy, and at least one queue head is a blocked
  `WAIT`, the machine is **deadlocked**. Simulators must detect and report this with the
  offending PCs.

#### What does *not* need synchronization

- Two instructions in the **same** queue. Each unit is in-order and each functional resource is
  held for the full occupancy of an instruction, so back-to-back `MATMUL`s accumulating into
  the same ACC region are correctly ordered with no `SIGNAL`/`WAIT` at all. Likewise `LOAD_W`
  followed by a `MATMUL` using that bank.
- A `LOAD_W` into bank 1 issued while a `MATMUL` on bank 0 is still running: the MXU tracks the
  two banks separately and the `LOAD_W` proceeds concurrently. This is what makes weight
  double-buffering free.

#### What *does* need synchronization

Everything crossing a unit boundary. Specifically:

| Producer      | Consumer  | Resource            | Typical pattern                        |
|---------------|-----------|---------------------|----------------------------------------|
| `DMA_LD`      | MXU/DWU   | SPM_A, SPM_W        | fill-buffer → `SIGNAL` → `WAIT` → compute |
| MXU / DWU     | VPU       | ACC                 | `MATMUL`s → `SIGNAL` → `WAIT` → `VQUANT`  |
| VPU           | `DMA_ST`  | SPM_A               | `VQUANT` → `SIGNAL` → `WAIT` → store    |
| MXU/DWU/VPU   | `DMA_LD`  | SPM buffer reuse    | consumer → `SIGNAL` → `WAIT` → refill (WAR) |
| MXU           | DWU       | ACC (disjoint regions) | only if regions overlap             |

### 5.4 `TRACE` (0x04)

| Bytes | Field     | Type  | Notes                                    |
|-------|-----------|-------|------------------------------------------|
| 0–3   | header    |       | `flags` = `TraceKind`, `aux` reserved    |
| 4–7   | `tag`     | `u32` | user-defined region/marker id            |
| 8–11  | `payload` | `u32` | user-defined                             |
| 12–31 | reserved  |       | zero                                     |

`TraceKind`: `0 = MARKER`, `1 = REGION_BEGIN`, `2 = REGION_END`. Architecturally a `NOP` with
zero occupancy. When it retires in `unit`'s queue the simulator emits `(cycle, unit, kind, tag,
payload)`. Regions nest per `(unit, tag)`; `kea-sim` reports per-region occupancy, stall
cycles, and achieved GOPS. The compiler is expected to bracket every fused layer with a
`REGION_BEGIN`/`REGION_END` pair whose `tag` indexes into the KEAF metadata JSON.

### 5.5 Rule D — the dispatch-order rule (**read this before writing a scheduler**)

The dispatcher is in-order and its *only* stall condition is a full target queue. That creates
a deadlock hazard that has nothing to do with the events themselves:

> If unit *U*'s queue fills with instructions blocked behind a `WAIT`, the dispatcher stalls on
> the next *U*-targeted instruction, and **no instruction for any other unit can be dispatched**
> — including the `SIGNAL` that would have released *U*.

The compiler must therefore enforce:

> **Rule D.** For every `WAIT e, thr` at stream position *p*, the `SIGNAL`s that supply those
> `thr` counts must appear at stream positions **< *p***.

*Why this is sufficient:* dispatch is in-order, so every one of those `SIGNAL`s has already been
pushed into its own queue before the `WAIT` is pushed. Define a dependency edge from each
blocking instruction to the instructions that must retire to unblock it. Rule D makes every
such edge point backwards in stream order; a graph whose edges all point backwards in a total
order is acyclic, so some instruction is always runnable and the machine cannot deadlock.

Rule D is cheap to satisfy — it is the natural order anyway ("issue the prefetch, then wait for
it") — and it is easy to check. `keac`'s scheduler must verify it and `kea-sim` should assert
it. Note that Rule D permits the *producer* to still be executing when the `WAIT` is dispatched;
it only constrains stream order, not completion order. That is exactly what makes prefetching
work.

---

## 6. DMA instructions

### 6.1 `DMA_LD` (0x10) / `DMA_ST` (0x11)

`DMA_LD` moves DRAM → SPM. `DMA_ST` moves SPM → DRAM. Same encoding.

| Bytes | Field       | Type  | Notes                                                    |
|-------|-------------|-------|----------------------------------------------------------|
| 0–3   | header      |       | `unit` ∈ {`DMA0`,`DMA1`}; `flags` bit0 = SPM space (0=`SPM_A`, 1=`SPM_W`); `aux` = `n2`, 1…255 |
| 4–7   | `dram_addr` | `u32` | DRAM byte address                                        |
| 8–11  | `spm_addr`  | `u32` | SPM byte address                                         |
| 12–13 | `len0`      | `u16` | contiguous bytes per innermost run, 1…65535              |
| 14–15 | `n1`        | `u16` | dim-1 count, 1…65535                                     |
| 16–19 | `dram_s1`   | `i32` | DRAM byte stride between dim-1 elements                  |
| 20–23 | `dram_s2`   | `i32` | DRAM byte stride between dim-2 elements                  |
| 24–27 | `spm_s1`    | `i32` | SPM byte stride between dim-1 elements                   |
| 28–31 | `spm_s2`    | `i32` | SPM byte stride between dim-2 elements                   |

Semantics:

```
for i2 in [0, n2):
  for i1 in [0, n1):
    copy len0 bytes
      DRAM side: dram_addr + i2*dram_s2 + i1*dram_s1
      SPM  side: spm_addr  + i2*spm_s2  + i1*spm_s1
```

Total bytes moved = `len0 * n1 * n2`. Strides are signed and may be negative (reversed layouts)
or zero. A zero DRAM stride on `DMA_LD` broadcasts; a zero SPM stride on `DMA_LD` or a zero
DRAM stride on `DMA_ST` produces an undefined last-writer-wins result and should be rejected by
the assembler.

Addresses are byte-granular. 16-byte alignment of `dram_addr`, `spm_addr`, `len0` and the
strides is *recommended* (`KEA_ALIGN_DMA_RECOMMENDED`); the KEA-1 timing model does not penalize
misalignment, which is a documented modelling simplification (see MICROARCH.md §8).

#### Worked descriptor: extract an NHWC tile

Pull a `[16 rows][56 cols][32 ch]` int8 tile out of a `112×112×32` DRAM feature map at
`base`, into a dense SPM_A buffer:

| Field | Value            | Reasoning                              |
|-------|------------------|----------------------------------------|
| `len0` | `32`            | 32 contiguous channel bytes per pixel  |
| `n1`   | `56`            | 56 pixels along W                      |
| `n2`   | `16`            | 16 rows along H                        |
| `dram_s1` | `32`         | next pixel = next 32 bytes             |
| `dram_s2` | `112*32 = 3584` | next row                            |
| `spm_s1`  | `32`         | dense                                  |
| `spm_s2`  | `56*32 = 1792` | dense                                |
| bytes  | `32*56*16 = 28672` |                                    |

To load the same tile into a **zero-haloed** buffer of `58` columns (1 pixel of padding each
side), keep `len0/n1/n2` and set `spm_addr += 1*32` (skip the left halo), `spm_s2 = 58*32 =
1856`. Pre-zero the halo with a single `VCOPY` fill.

#### Why the SPM side gets two strides but the descriptor still fits

The naive 3D-strided descriptor (independent `dims[3]`, `src_stride[3]`, `dst_stride[3]`, elem
size) needs 3+3+3 fields plus two base addresses — about 44 bytes. KEA-1 collapses the innermost
dimension into a **contiguous byte run** (`len0`) rather than a strided element loop, which is
what every real tensor layout wants anyway, and drops the DRAM address to 32 bits. That leaves
exactly 32 bytes with two full 32-bit strides on each side. `n2 ≤ 255` is the one real casualty;
tile heights above 255 are not a thing on a 256 KiB scratchpad.

---

## 7. MXU instructions

### 7.1 Dataflow

The MXU is a **weight-stationary 16×16 systolic array**. A loaded tile holds
`W[k][n]`, `k ∈ [0,16)` the reduction (input-channel) index, `n ∈ [0,16)` the output-channel
index. Activation rows of 16 elements stream in, one per cycle (two per cycle in int4), and
16 int32 partial sums stream out per row.

The primitive is therefore a fixed `[M,16] × [16,16] → [M,16]` int32 GEMM tile.
Everything else — convolution, fully-connected layers, batched matmul — is the compiler
arranging `M`, the activation strides, and the ACC destination.

There are **two weight register banks per PE**. `LOAD_W` targets one bank while `MATMUL` streams
against the other, so weight load is free as long as the compiler alternates banks.

### 7.2 `LOAD_W` (0x20)

| Bytes | Field          | Type  | Notes                                            |
|-------|----------------|-------|--------------------------------------------------|
| 0–3   | header         |       | `flags` bit0 = int4, bit1 = target bank (0/1)    |
| 4–7   | `w_addr`       | `u32` | SPM_W byte address of tile row `k=0`             |
| 8–11  | `w_row_stride` | `i32` | bytes between consecutive `k` rows; ≥ 0          |
| 12    | `k_rows`       | `u8`  | valid reduction rows, 1…16                       |
| 13    | `n_cols`       | `u8`  | valid output columns, 1…16                       |
| 14–31 | reserved       |       | zero                                             |

```
int8:  W[k][n] = SPM_W[w_addr + k*w_row_stride + n]                        for k<k_rows, n<n_cols
int4:  W[k][n] = nibble (n & 1) of SPM_W[w_addr + k*w_row_stride + n/2]    for k<k_rows, n<n_cols
       W[k][n] = 0                                                          otherwise
```

Rows `k ≥ k_rows` and columns `n ≥ n_cols` are loaded as **zero**. This is how tail tiles work
for channel counts that are not multiples of 16 — no masking is needed in `MATMUL`.

Alignment: `w_addr` and `w_row_stride` must be multiples of 16 (int8) or 8 (int4).

### 7.3 `MATMUL` (0x21)

| Bytes | Field              | Type  | Notes                                                  |
|-------|--------------------|-------|--------------------------------------------------------|
| 0–3   | header             |       | `flags` bit0 = accumulate, bit1 = int4, bit2 = bank    |
| 4–7   | `a_addr`           | `u32` | SPM_A byte address of activation row (0,0)             |
| 8–11  | `acc_addr`         | `u32` | ACC **word** index of output row (0,0)                 |
| 12–13 | `m_inner`          | `u16` | inner row count, ≥ 1                                   |
| 14–15 | `m_outer`          | `u16` | outer row count, ≥ 1                                   |
| 16–19 | `a_inner_stride`   | `i32` | SPM_A bytes between inner rows                         |
| 20–23 | `a_outer_stride`   | `i32` | SPM_A bytes between outer rows                         |
| 24–27 | `acc_inner_stride` | `i32` | ACC words between inner rows, multiple of 16           |
| 28–31 | `acc_outer_stride` | `i32` | ACC words between outer rows, multiple of 16           |

```
for mo in [0, m_outer):
  for mi in [0, m_inner):
    a = a_addr   + mo*a_outer_stride   + mi*a_inner_stride      // SPM_A bytes
    q = acc_addr + mo*acc_outer_stride + mi*acc_inner_stride    // ACC words
    for n in [0, 16):
      ACC[q + n] = (accumulate ? ACC[q + n] : 0)
                 + Σ_{k=0}^{15} A[a + k] * W[k][n]
```

(In int4, `A[a + k]` is nibble `k` of the 8 bytes at `a`.)

**Invariants the compiler must guarantee:**

- The array always reads **16 activation elements** (16 bytes int8, 8 bytes int4) per row and
  always writes **16 int32 lanes** per row, regardless of the `k_rows`/`n_cols` of the resident
  weight tile. Unused reduction lanes are multiplied by the zeros `LOAD_W` installed, and unused
  output lanes receive zeros. So: the 16 activation bytes must be *readable* (pad the buffer)
  and the 16 ACC words must be *writable* (allocate 16-lane tiles).
- `m_inner * m_outer ≤ KEA_MXU_MAX_ROWS` (2048), because `M*16` int32 words must fit in ACC.
- `acc_addr`, `acc_inner_stride`, `acc_outer_stride` are multiples of 16 words.
- The 2D iteration is over the *activation* side; the `M` rows do not have to be contiguous, in
  either space. This is what lets one `MATMUL` cover a 2D output tile (§8).

**Ordering.** Consecutive `MATMUL`s from the MXU queue are fully ordered against each other, so
accumulating a chain of taps into the same ACC region needs no synchronization at all. Only the
hand-off to the VPU does.

---

## 8. Convolution lowering (**normative**)

> **There is no convolution instruction, and there never will be one.** A 2D convolution is
> lowered by the compiler into a sequence of `LOAD_W` + `MATMUL` pairs, **one pair per (kh, kw)
> kernel tap**, all accumulating into the same ACC region, with the activation base address
> shifted per tap.

This section defines exactly how. `keac`'s conv lowering must implement precisely what follows.

### 8.1 Setup and notation

Layouts (fixed by this ISA — the memory planner must produce them):

- **Activations in SPM_A** are NHWC with channels innermost:
  `A[ih][iw][ic]` at `A_base + ih*sr + iw*sp + ic`, where
  - `sp` = *pixel stride* in bytes = the number of channel bytes stored per pixel,
  - `sr` = *row stride* in bytes = `IW_spm * sp` for a dense buffer.
  `sp` and `sr` refer to the buffer as laid out in SPM, which may be wider than the tile (halo).
- **Weights in SPM_W** are pre-tiled by `keac` into dense 16×16 `int8` tiles of 256 bytes
  (128 bytes in int4), ordered `[oc0][ic0][kh][kw]`, so `w_row_stride = 16` always and the tile
  for tap `(kh,kw)` of group `(oc0, ic0)` is at
  `W_base + ((((oc0/16)*ceil(IC/16) + ic0/16)*KH + kh)*KW + kw) * 256`.
- **Accumulators in ACC** are `[oh][ow][16]` int32 per output-channel group:
  `Q_base + oh*qr + ow*16`, with `qr = OW_t * 16`.

Convolution parameters: kernel `KH×KW`, stride `S`, dilation `D`, output tile `OH_t × OW_t`,
input channels `IC`, output channels `OC`.

### 8.2 The addressing identity

Output pixel `(oh, ow)` for tap `(kh, kw)` reads input pixel `(oh*S + kh*D, ow*S + kw*D)`, whose
SPM_A address is

```
A_base + (oh*S + kh*D)*sr + (ow*S + kw*D)*sp
  ≡  [ A_base + kh*D*sr + kw*D*sp ]  +  oh*(S*sr)  +  ow*(S*sp)
       └──────── per-tap base ────┘     └─ outer ─┘   └─ inner ─┘
```

That decomposition is *exactly* the shape of `MATMUL`'s 2D activation walk. Therefore:

| `MATMUL` field     | Value for tap `(kh, kw)`     |
|--------------------|------------------------------|
| `a_addr`           | `A_base + kh*D*sr + kw*D*sp` |
| `a_inner_stride`   | `S * sp`                     |
| `a_outer_stride`   | `S * sr`                     |
| `m_inner`          | `OW_t`                       |
| `m_outer`          | `OH_t`                       |
| `acc_addr`         | `Q_base`                     |
| `acc_inner_stride` | `16`                         |
| `acc_outer_stride` | `OW_t * 16`                  |

Dilation `D` costs nothing: it is folded into `a_addr` and never appears again. A 1×1
convolution is the degenerate case `KH=KW=1`, one tap, and if the output tile is the whole
feature map you can collapse to `m_outer=1, m_inner=OH*OW, a_inner_stride=sp`.

### 8.3 The loop nest

```
for oc0 in range(0, OC, 16):                  # output-channel tiles
  for ic0 in range(0, IC, 16):                # reduction tiles
    for kh in range(KH):
      for kw in range(KW):
        t    = tap counter across (ic0, kh, kw)
        bank = t & 1                          # alternate → LOAD_W hides under MATMUL
        emit LOAD_W(w_addr = tile(oc0, ic0, kh, kw),
                    w_row_stride = 16,
                    k_rows = min(16, IC - ic0),
                    n_cols = min(16, OC - oc0),
                    bank = bank, int4 = ...)
        emit MATMUL(a_addr = A_base + ic0 + kh*D*sr + kw*D*sp,
                    a_inner_stride = S*sp, a_outer_stride = S*sr,
                    m_inner = OW_t, m_outer = OH_t,
                    acc_addr = Q_base + (oc0/16)*OH_t*OW_t*16,
                    acc_inner_stride = 16, acc_outer_stride = OW_t*16,
                    bank = bank,
                    accumulate = not (ic0 == 0 and kh == 0 and kw == 0),
                    int4 = ...)
```

Note `+ ic0` in `a_addr`: the reduction tile selects a 16-byte slice of the channel dimension,
and channels are innermost, so it is a flat byte offset.

**`accumulate` is false on exactly one instruction per ACC region** — the very first tap of the
very first reduction tile. Every other tap accumulates. There is no separate ACC-clear
instruction and none is needed.

### 8.4 Padding

`MATMUL` has no predication, so padding is handled entirely in SPM_A. Two options; option (a)
is recommended:

**(a) Zero halo (recommended).** Allocate the SPM_A tile with a `pad` -pixel border, `VCOPY`-fill
it with the input zero point once, and DMA the interior into the middle. `A_base` then points at
the *padded* origin, so output pixel `(0,0)` with tap `(0,0)` correctly reads the halo. Cost is
one `VCOPY` per tile and a slightly larger buffer.

> Note the zero point: for asymmetric int8 quantization the padding value is the input
> **zero point**, not 0. `VCOPY` fill takes an arbitrary `int8` value for exactly this reason.

**(b) Region splitting.** Emit separate `MATMUL`s for the interior (all taps) and the edges
(subset of taps), skipping taps that fall entirely in padding. Zero extra memory, more
instructions, more compiler complexity. Use it only if SPM pressure demands it.

### 8.5 Worked example — 3×3, stride 1, dilation 1

**Problem.** `IC = 16`, `OC = 32`, input `8×8×16` with `pad = 1`, output `8×8×32`, int8.

**SPM_A buffer.** Padded to `10×10×16`, so
`sp = 16`, `sr = 10*16 = 160`, `A_base = 0x0000`, buffer size `10*160 = 1600` bytes.
`VCOPY` fill 1600 bytes with the input zero point, then `DMA_LD` the `8×8×16` interior to
`0x0000 + 1*160 + 1*16 = 176`.

**ACC.** Two output-channel groups, each `8*8*16 = 1024` words.
`Q_base(oc0=0) = 0`, `Q_base(oc0=16) = 1024`. Total 2048 of 32768 words.

**Per-tap `a_addr` = `kh*160 + kw*16`:**

| tap `(kh,kw)` | `a_addr` | tap | `a_addr` | tap | `a_addr` |
|---------------|----------|-----|----------|-----|----------|
| (0,0)         | **0**    | (1,0) | **160** | (2,0) | **320** |
| (0,1)         | **16**   | (1,1) | **176** | (2,1) | **336** |
| (0,2)         | **32**   | (1,2) | **192** | (2,2) | **352** |

All nine `MATMUL`s share `a_inner_stride = 16`, `a_outer_stride = 160`, `m_inner = 8`,
`m_outer = 8`, `acc_inner_stride = 16`, `acc_outer_stride = 128`.

**Spot check.** Output `(7,7)`, tap `(2,2)` should read input `(9,9)` of the padded buffer,
i.e. the bottom-right halo pixel:

```
a_addr(2,2) + 7*a_outer_stride + 7*a_inner_stride
  = 352 + 7*160 + 7*16
  = 352 + 1120 + 112 = 1584
9*sr + 9*sp = 9*160 + 9*16 = 1440 + 144 = 1584   ✓
```

and `1584 + 16 = 1600` = exactly the end of the padded buffer, so the 16-byte read is in
bounds. ✓

**Emitted stream** (18 instruction pairs; `oc0=0` group shown):

```
pc  unit  instruction
--  ----  --------------------------------------------------------------------
 0  MXU   LOAD_W  w_addr=W+0,    w_row_stride=16, k_rows=16, n_cols=16, bank=0
 1  MXU   MATMUL  a_addr=0,   ... acc_addr=0,   acc_mode=overwrite,  bank=0
 2  MXU   LOAD_W  w_addr=W+256,  ...                                  bank=1
 3  MXU   MATMUL  a_addr=16,  ... acc_addr=0,   acc_mode=accumulate,  bank=1
 4  MXU   LOAD_W  w_addr=W+512,  ...                                  bank=0
 5  MXU   MATMUL  a_addr=32,  ... acc_addr=0,   acc_mode=accumulate,  bank=0
 6  MXU   LOAD_W  w_addr=W+768,  ...                                  bank=1
 7  MXU   MATMUL  a_addr=160, ... acc_addr=0,   acc_mode=accumulate,  bank=1
 ...                                          (taps (1,1) … (2,2))
17  MXU   MATMUL  a_addr=352, ... acc_addr=0,   acc_mode=accumulate,  bank=1
```

**Cost.** Each `MATMUL` occupies the array for `4 + 8*8 = 68` cycles; each `LOAD_W` occupies
the weight port for `2 + (16*16)/32 = 10` cycles, fully hidden under the previous `MATMUL`
because the banks alternate. Per output-channel group: `9 * 68 = 612` cycles. Two groups:
`1224` cycles.

Useful MACs = `8*8*9*16*32 = 294,912`. Ideal at 256 MAC/cycle = `1152` cycles.
**Efficiency = 1152/1224 = 94.1%**, the 5.9% being the fixed 4-cycle `MATMUL` setup, which
amortizes away for larger tiles.

### 8.6 Channel-packed lowering for the first layer

The `IC = 3` first layer of MobileNetV2 would use only 3 of 16 reduction lanes — 19%
utilization. Because NHWC stores channels innermost, **three consecutive pixels of a 3-channel
image are 9 contiguous bytes**, which is a legal 16-element activation row with 7 zero lanes.
So the compiler can fold an entire kernel *row* into one reduction tile:

```
K tile = { (kw=0,ic=0..2), (kw=1,ic=0..2), (kw=2,ic=0..2), 7 zero lanes }
```

- `LOAD_W` with `k_rows = 9`, the tile packing `W[kw][ic][oc]` in that order.
- `MATMUL` with `a_addr = A_base + kh*sr`, `a_inner_stride = S*sp = 2*3 = 6` (stride 2),
  `a_outer_stride = S*sr`.
- Three taps (`kh = 0,1,2`) instead of nine.

This turns 19% into 56% MXU utilization on the first layer and needs no new instruction — it is
purely a weight-layout choice. Generalizes to any layer where `KW * IC ≤ 16`.

---

## 9. DWU instruction

A systolic array is a poor fit for depthwise convolution: there is no reduction across channels,
so a `[M,16]×[16,16]` GEMM would use 1 of 16 columns. MobileNetV2 is ~50% depthwise layers by
count. Hence a dedicated unit: **16 lanes, one per channel, 8 MACs/lane/cycle**, which is
exactly enough to retire a 3×3 window in a bit over one cycle per pixel per lane.

### 9.1 `DWCONV` (0x30)

| Bytes | Field          | Type  | Notes                                                            |
|-------|----------------|-------|------------------------------------------------------------------|
| 0–3   | header         |       | `flags` bit0 = 5×5 (else 3×3), bit1 = stride 2 (else 1), bit2 = accumulate |
| 4–7   | `a_addr`       | `u32` | SPM_A byte address of input pixel (0,0)                          |
| 8–11  | `w_addr`       | `u32` | SPM_W byte address of tap plane (0,0)                            |
| 12–15 | `acc_addr`     | `u32` | ACC **word** index of output pixel (0,0), multiple of 16         |
| 16–17 | `out_h`        | `u16` | ≥ 1                                                              |
| 18–19 | `out_w`        | `u16` | ≥ 1                                                              |
| 20–21 | `channels`     | `u16` | ≥ 16, **multiple of 16**                                         |
| 22–23 | reserved       |       | zero                                                             |
| 24–27 | `a_row_stride` | `i32` | SPM_A bytes between input rows                                   |
| 28–31 | `a_pix_stride` | `i32` | SPM_A bytes between horizontally adjacent input pixels           |

```
K = flags.KERNEL5 ? 5 : 3
S = flags.STRIDE2 ? 2 : 1

for oh in [0, out_h):
  for ow in [0, out_w):
    for c in [0, channels):
      q = acc_addr + (oh*out_w + ow)*channels + c
      ACC[q] = (accumulate ? ACC[q] : 0)
             + Σ_{kh,kw < K}  A[a_addr + (oh*S+kh)*a_row_stride
                                       + (ow*S+kw)*a_pix_stride + c]
                            * W[w_addr + (kh*K + kw)*channels + c]
```

- **int8 only.** Weights are `[KH][KW][C]` int8, contiguous, `channels` bytes per tap plane.
- **VALID padding only.** The SPM_A tile must already contain any zero (zero-point) halo,
  exactly as in §8.4(a).
- **Dilation** is emulated by scaling `a_row_stride` / `a_pix_stride`, or by slicing.
- The ACC region is written **densely** as `[out_h][out_w][channels]` int32 starting at
  `acc_addr`. There is no ACC stride field; the compiler allocates a dense tile.
- `a_addr`, `a_row_stride`, `a_pix_stride` should be multiples of 16 (`KEA_ALIGN_DWU`).

Typical MobileNetV2 use: a `3×3` depthwise on a `56×56×144` feature map becomes
`ceil(144/16) = 9` `DWCONV`s over channel groups, or one `DWCONV` with `channels = 144` (the
unit iterates channel groups internally — same cycle count, one instruction).

---

## 10. VPU instructions

The VPU is 16 int8 lanes wide. It is the only unit that reads `ACC` and the only unit besides
DMA that writes `SPM_A`. Everything that turns an int32 accumulator into a storable activation
happens here.

### 10.1 `VQUANT` (0x40)

| Bytes | Field            | Type  | Notes                                                     |
|-------|------------------|-------|-----------------------------------------------------------|
| 0–3   | header           |       | `flags` bit0 = output int4; `aux` = `out_zp` (int8)       |
| 4–7   | `acc_addr`       | `u32` | ACC word index, multiple of 16                            |
| 8–11  | `out_addr`       | `u32` | SPM_A byte address                                        |
| 12–15 | `qparam_addr`    | `u32` | SPM_W byte address, 4-byte aligned                        |
| 16–19 | `num_pixels`     | `u32` | ≥ 1                                                       |
| 20–21 | `channels`       | `u16` | ≥ 16, multiple of 16                                      |
| 22    | `clamp_lo`       | `i8`  | applied **after** adding `out_zp`                         |
| 23    | `clamp_hi`       | `i8`  |                                                           |
| 24–27 | `acc_pix_stride` | `i32` | ACC words between pixels, multiple of 16                  |
| 28–31 | `out_pix_stride` | `i32` | SPM_A bytes between pixels                                |

```
qp = (KeaQuantParam*) SPM_W[qparam_addr]        // array of `channels` records

for p in [0, num_pixels):
  for c in [0, channels):
    v   = ACC[acc_addr + p*acc_pix_stride + c] + qp[c].bias
    out = keaRequantize(v, qp[c].mult, qp[c].shift, out_zp, clamp_lo, clamp_hi)
    store out at out_addr + p*out_pix_stride + c            // int8
      (int4: nibble c of out_addr + p*out_pix_stride + c/2)
```

`KeaQuantParam` is 12 bytes, packed: `{ int32 bias; int32 mult; int32 shift; }`.
`mult` is a **normalized Q31 multiplier** in `[2^30, 2^31)`. `shift > 0` is a right shift after
the multiply; `shift < 0` is a left shift *before* it (rare; the compiler should normalize it
away).

The requantization pipeline is defined **by reference implementation**, in
`kea::keaRequantize` / `kea::keaSrdhm` / `kea::keaRdpot`. It is gemmlowp/TFLite-compatible:

```cpp
int32 keaSrdhm(int32 a, int32 b);           // round(a*b / 2^31), saturating
int32 keaRdpot(int32 x, int exp);           // round-half-away-from-zero(x / 2^exp)

int32 keaRequantize(int32 v, int32 mult, int32 shift,
                    int32 out_zp, int32 clamp_lo, int32 clamp_hi) {
  if (shift < 0) v <<= -shift;
  int32 p = keaSrdhm(v, mult);
  if (shift > 0) p = keaRdpot(p, shift);
  return clamp(p + out_zp, clamp_lo, clamp_hi);
}
```

Do not reimplement this. Call the header.

**Fused activations** are expressed purely through the clamp:

| Activation | `clamp_lo`                     | `clamp_hi`                          |
|------------|--------------------------------|-------------------------------------|
| none       | `-128`                         | `127`                               |
| ReLU       | `out_zp`                       | `127`                               |
| ReLU6      | `out_zp`                       | `out_zp + round(6 / out_scale)`     |
| clamp(a,b) | `out_zp + round(a / out_scale)`| `out_zp + round(b / out_scale)`     |

For int4 output the clamps must additionally lie within `[-8, 7]`.

### 10.2 `VADD` (0x41)

| Bytes | Field        | Type  | Notes                                |
|-------|--------------|-------|--------------------------------------|
| 0–3   | header       |       | `flags`, `aux` reserved              |
| 4–7   | `a_addr`     | `u32` | SPM_A                                |
| 8–11  | `b_addr`     | `u32` | SPM_A                                |
| 12–15 | `out_addr`   | `u32` | SPM_A                                |
| 16–19 | `param_addr` | `u32` | SPM_W, 4-byte aligned                |
| 20–23 | `num_elems`  | `u32` | ≥ 1; channels are flattened in       |
| 24    | `clamp_lo`   | `i8`  |                                      |
| 25    | `clamp_hi`   | `i8`  |                                      |
| 26–31 | reserved     |       | zero                                 |

Per-tensor (not per-channel) quantized add, for MobileNetV2's inverted-residual skip
connections. Parameters come from a single 20-byte `KeaAddParam` record in SPM_W:

```cpp
struct KeaAddParam {
  int32 a_mult, b_mult, o_mult;
  int8  a_shift, b_shift, o_shift, _;
  int8  a_zp,    b_zp,    o_zp,    _;
};
```

Semantics, matching the TFLite reference kernel exactly (`kea::keaQuantizedAdd`):

```cpp
sa = (a - a_zp) << KEA_VADD_LEFT_SHIFT;     // KEA_VADD_LEFT_SHIFT == 20
sb = (b - b_zp) << KEA_VADD_LEFT_SHIFT;
xa = keaRdpot(keaSrdhm(sa, a_mult), a_shift);
xb = keaRdpot(keaSrdhm(sb, b_mult), b_shift);
o  = keaRdpot(keaSrdhm(xa + xb, o_mult), o_shift) + o_zp;
out = clamp(o, clamp_lo, clamp_hi);
```

In-place is legal (`out_addr == a_addr`); the VPU reads before it writes within a lane.

### 10.3 `VPOOL` (0x42)

| Bytes | Field            | Type  | Notes                                    |
|-------|------------------|-------|------------------------------------------|
| 0–3   | header           |       | `flags` bit0 = average (else max)        |
| 4–7   | `in_addr`        | `u32` | SPM_A                                    |
| 8–11  | `out_addr`       | `u32` | SPM_A                                    |
| 12–13 | `out_h`          | `u16` | ≥ 1                                      |
| 14–15 | `out_w`          | `u16` | ≥ 1                                      |
| 16–17 | `channels`       | `u16` | ≥ 1                                      |
| 18    | `kh`             | `u8`  | 1…32                                     |
| 19    | `kw`             | `u8`  | 1…32                                     |
| 20    | `stride_h`       | `u8`  | 1…8                                      |
| 21    | `stride_w`       | `u8`  | 1…8                                      |
| 22–23 | reserved         |       | zero                                     |
| 24–27 | `in_row_stride`  | `i32` | SPM_A bytes between input rows           |
| 28–31 | `out_row_stride` | `i32` | SPM_A bytes between output rows          |

Input and output pixel strides are implicitly `channels` bytes. VALID padding only.

```
window(oh,ow,c) = { A[in_addr + (oh*stride_h + i)*in_row_stride
                            + (ow*stride_w + j)*channels + c] : i<kh, j<kw }

max: out = max(window)
avg: out = keaPoolAverage(Σ window, kh*kw)      // round half away from zero
store at out_addr + oh*out_row_stride + ow*channels + c
```

> **Constraint: `VPOOL` requires the input and output quantization parameters to be identical**
> (same scale, same zero point). Then averaging raw int8 values is exactly correct, because
> `avg(x_i − zp) + zp = avg(x_i)`. TFLite always produces pooling ops this way. If a graph ever
> presents differing scales, `keac` must emit a separate rescale — `VPOOL` will not do it.

MobileNetV2 uses this once, for the `7×7` global average pool: `kh=kw=7`, `stride=1`,
`out_h=out_w=1`, `channels=1280`.

### 10.4 `VCOPY` (0x43)

| Bytes | Field            | Type  | Notes                                                        |
|-------|------------------|-------|--------------------------------------------------------------|
| 0–3   | header           |       | `flags` bit0 = fill, bit1 = src is SPM_W, bit2 = dst is SPM_W; `aux` = fill value (int8) |
| 4–7   | `src_addr`       | `u32` | ignored in fill mode                                         |
| 8–11  | `dst_addr`       | `u32` |                                                              |
| 12–15 | `row_bytes`      | `u32` | ≥ 1                                                          |
| 16–19 | `rows`           | `u32` | ≥ 1                                                          |
| 20–23 | `src_row_stride` | `i32` |                                                              |
| 24–27 | `dst_row_stride` | `i32` |                                                              |
| 28–31 | reserved         |       | zero                                                         |

```
for r in [0, rows):
  dst[dst_addr + r*dst_row_stride ... +row_bytes) =
      fill ? splat(fill_value)
           : src[src_addr + r*src_row_stride ... +row_bytes)
```

Fill mode is how conv halos get their zero point (§8.4). `ACC` is **not** reachable from
`VCOPY`; initialize accumulators with `MATMUL`/`DWCONV` `accumulate=0` instead. Overlapping
`src`/`dst` regions in copy mode are undefined.

---

## 11. Legality rules (summary)

`kea::keaValidate(const KeaInstr&)` checks all of these and returns a static error string.
Assemblers and the simulator's loader must call it on every instruction.

### 11.1 Alignment

| Field                                                  | Multiple of      |
|--------------------------------------------------------|------------------|
| `LOAD_W.w_addr`, `LOAD_W.w_row_stride` (int8)          | 16 bytes         |
| `LOAD_W.w_addr`, `LOAD_W.w_row_stride` (int4)          | 8 bytes          |
| `MATMUL.acc_addr`, `acc_inner_stride`, `acc_outer_stride` | 16 words      |
| `DWCONV.acc_addr`                                      | 16 words         |
| `DWCONV.channels`                                      | 16               |
| `DWCONV.a_addr`, `a_row_stride`, `a_pix_stride`        | 16 bytes (advisory) |
| `VQUANT.acc_addr`, `acc_pix_stride`                    | 16 words         |
| `VQUANT.channels`                                      | 16               |
| `VQUANT.qparam_addr`, `VADD.param_addr`                | 4 bytes          |
| everything else (SPM_A activation bases, DMA, `VCOPY`) | 1 byte           |

`MATMUL.a_addr` is deliberately byte-granular: the conv lowering shifts it by `kw*sp`, and `sp`
is 3 for the first layer of an RGB network.

### 11.2 Range

| Constraint                                              | Bound                     |
|---------------------------------------------------------|---------------------------|
| `MATMUL.m_inner * m_outer`                              | ≤ 2048 (`KEA_MXU_MAX_ROWS`) |
| `LOAD_W.k_rows`, `n_cols`                               | 1…16                      |
| `DMA.len0`, `DMA.n1`                                    | 1…65535                   |
| `DMA.n2` (`aux`)                                        | 1…255                     |
| event id                                                | 0…31                      |
| `SIGNAL`/`WAIT` value                                   | ≤ `0x7FFFFFFF`            |
| program length                                          | ≤ 32768 instructions      |
| every address                                           | inside its space          |

### 11.3 Structural

- Reserved flag bits and reserved bytes are zero.
- `HALT` targets `CTRL`; DMA opcodes target `DMA0` or `DMA1`; all other non-CTRL opcodes target
  their owning unit exactly.
- **Rule D** (§5.5) holds for every `WAIT`.
- `HALT` is the last instruction.

---

## 12. Programming pattern: double-buffered layer

This is the shape every `keac`-generated layer should have. Remember: it is **one linear
stream**; the `unit` column is a field, not a separate program.

```
event 0 : "SPM_A buffer 0 filled"      event 4 : "ACC tile ready"
event 1 : "SPM_A buffer 1 filled"      event 5 : "SPM_A output ready"
event 2 : "SPM_A buffer 0 free"        event 6 : "SPM_A output free"
event 3 : "SPM_A buffer 1 free"

pc  unit  instruction                                   comment
--  ----  ------------------------------------------    -------------------------------
 0  DMA0  DMA_LD  tile[0] -> A0                          prologue: prime buffer 0
 1  DMA0  SIGNAL  ev0, 1
 2  DMA1  DMA_LD  tile[1] -> A1                          prime buffer 1 on the other engine
 3  DMA1  SIGNAL  ev1, 1
 4  MXU   TRACE   begin, tag=layer_id
 5  MXU   WAIT    ev0, 1                                 --- iteration 0 ---
 6  MXU   LOAD_W  ... bank=0
 7  MXU   MATMUL  A0 -> ACC0, overwrite, bank=0
 8  MXU   LOAD_W  ... bank=1
 9  MXU   MATMUL  A0 -> ACC0, accumulate, bank=1
10  MXU   SIGNAL  ev2, 1                                 buffer 0 free  (drains MXU pipeline)
11  MXU   SIGNAL  ev4, 1                                 ACC0 ready
12  DMA0  WAIT    ev2, 1                                 refill buffer 0 for iteration 2
13  DMA0  DMA_LD  tile[2] -> A0
14  DMA0  SIGNAL  ev0, 1
15  VPU   WAIT    ev4, 1
16  VPU   VQUANT  ACC0 -> out, relu6 clamps
17  VPU   SIGNAL  ev5, 1
18  MXU   WAIT    ev1, 1                                 --- iteration 1, buffer 1 ---
19  MXU   LOAD_W  ... bank=0
20  MXU   MATMUL  A1 -> ACC1, overwrite, bank=0
    ...
    DMA1  WAIT    ev5, 1
    DMA1  DMA_ST  out -> DRAM
    DMA1  SIGNAL  ev6, 1
    MXU   TRACE   end, tag=layer_id
    CTRL  HALT
```

Things to notice:

- Every `WAIT` is preceded in stream order by the `SIGNAL`s that feed it (**Rule D**).
- `LOAD_W`/`MATMUL` alternate banks, so weight loads vanish under compute.
- `DMA0` and `DMA1` alternate buffers, so the load for iteration *n+2* overlaps compute on
  iteration *n*. This is the entire reason two DMA engines exist.
- The `SIGNAL` at pc 10 drains the MXU's 32-cycle pipeline. Putting the two `SIGNAL`s adjacent
  costs the drain once, not twice.
- `TRACE` brackets the region so `kea-sim` can report per-layer occupancy and roofline position.

---

## 13. Reserved for future revisions

The following are explicitly **not** in KEA-1 and must not be assumed:

- Opcodes `0x05–0x0F`, `0x12–0x1F`, `0x22–0x2F`, `0x31–0x3F`, `0x44–0xFE` are unassigned and
  illegal.
- int4 on `DWCONV`, `VADD`, `VPOOL` (flag bits reserved, must be zero).
- Any form of hardware loop, branch, predicate, or indirect addressing.
- Per-channel parameters on `VADD`.
- More than 255 in a DMA `n2`, more than 2048 rows in a `MATMUL`, DRAM beyond 4 GiB.
- Sub-byte scratchpad addressing.

A future `KEA-2` may widen the DRAM address (KEAF already stores 64-bit offsets) and add a
transpose/gather unit. Neither is in scope.
