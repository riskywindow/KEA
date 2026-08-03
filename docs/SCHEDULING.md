# `-kea-schedule` — the double-buffered DMA scheduler

**Status: implemented.** Everything here is exercised by
`compiler/test/kea-schedule{,-aliasing,-e2e,-tileloop}.mlir` and measured by
`compiler/test/kea-schedule-measure.py`; `bash scripts/build_compiler.sh` runs
the tests.

| | |
|---|---|
| Pass | `compiler/lib/Transforms/Schedule.cpp`, declared in `compiler/include/kea/Transforms/Passes.td` |
| In / out | Level 2 → Level 2 ([ADR-0002](adr/0002-two-level-kea-dialect.md)) |
| Runs | after `-kea-tile`, **before** `-kea-alloc` (ADR-0002's amendment) |
| Machine contract | `include/kea/hw_config.h` — every constant and every timing formula |
| Normative inputs | ISA.md §5.5 (Rule D), §7.3; ISA_ERRATA.md E1/E3/E4; DIALECT_L2.md §4, §6.1; MICROARCH.md §6, §8 |

Everything upstream of this pass produces a correct *sequential* program. This
pass makes it *overlap*. It is the only place in the compiler that knows the
machine has five queues.

```
-kea-tile                    -kea-schedule                     -kea-alloc
sequential L2           →    concurrent L2                →    addresses
no queues                    a queue on every instruction
no semaphores                kea.signal / kea.wait
no overlap                   loads hoisted, stores sunk
                             live ranges widened to match
```

---

## 1. What it does, in one page

1. **Assign a queue to every instruction.** `kea.load_w`/`kea.mm` → MXU,
   `kea.dwconv` → DWU, `kea.v*` → VPU, `kea.dma_*` → DMA0 or DMA1, `kea.halt`
   → CTRL. `kea.trace` goes on the queue that does its region's arithmetic.
2. **Build the dependence graph** from the per-operand memory effects Level 2
   already carries — RAW, WAR and WAW, on scratchpad *and* DRAM buffers.
3. **List-schedule it** against a cycle-approximate model of the machine built
   from `hw_config.h`. Prefetching is not a special case: tile *N+1*'s
   `DMA_LD` writes a fresh `kea.alloc`, so it depends on nothing tile *N* does,
   an idle engine can run it immediately, and the greedy choice puts it above
   tile *N*'s `MATMUL`s. Two guards keep the greed honest — the depth-16
   queues and the scratchpad high-water mark.
4. **Insert the minimum semaphores** the surviving cross-queue edges need, as
   per-(producer queue, consumer queue) token channels. Rule D holds by
   construction.
5. **Hoist each on-chip `kea.alloc`** to the earliest stream position that may
   run concurrently with one of its users. That is the whole of ADR-0002's
   soundness obligation, discharged.
6. Re-stamp the live ranges, re-run `verifyWeightBanks()`, re-prove Rule D.

```bash
kea-opt in.mlir -kea-schedule
kea-opt in.mlir -kea-schedule=report-schedule=true   # publish kea.schedule
kea-opt in.mlir -kea-schedule=mode=serial            # the A/B baseline, §8
kea-opt in.mlir "-kea-schedule=queue-depth=2 report-schedule=true"
kea-opt in.mlir -kea-schedule=annotate-units=false   # drop the kea.unit stamps
```

Pre-existing `kea.signal` / `kea.wait` are erased and recomputed, so the pass
is a fixed point on its own output (`compiler/test/kea-schedule.mlir`'s
`IDEMPOTENT` run line diffs the two).

---

## 2. Queues and engines

### 2.1 Where the `unit` attribute goes, and where it does not

`unit` is a real `KeaHead` field on every instruction, but it is only an
*attribute* in `KeaMachineOps.td` where the compiler has a choice:
`kea.dma_load` / `kea.dma_store` (which engine) and the queue-agnostic
`kea.signal` / `kea.wait` / `kea.trace` (which queue). For every other opcode
the unit follows from the opcode, which is stronger than an attribute because
nothing can disagree with it — `kea-as` rejects `VPU MATMUL` outright.

So this pass writes `unit` on the ops that have the field, and additionally
stamps the derived queue as a **discardable `kea.unit`** on the fixed-unit ops
so the assignment is readable and FileCheck-testable. That attribute is
advisory; `-kea-emit` must not need it, and `annotate-units=false` turns it
off.

### 2.2 Which DMA engine

Two engines share one 16 B/cycle DRAM port (MICROARCH.md §6.1), so a second
engine buys **no bandwidth**: two concurrent transfers get 8 B/cycle each. It
buys *concurrency* — a prefetch and a write-back in flight at the same time,
which is the entire reason ISA.md §12's hand-written layer alternates them.
That only happens if the work is actually spread.

Each descriptor is priced on both engines and takes the one that:

1. can **start** it soonest (the engine's queue is in-order, so this is the
   later of "engine free" and "dependences visible");
2. then, the one with the shorter **contended occupancy** — `keaDmaOccupancy()`
   plus the cycles the other engine's data phase overlaps this one's, which is
   the 16 → 8 B/cycle stretch;
3. then, the **less loaded** engine.

Rule 3 is not cosmetic. Without it every descriptor whose dependences are
already satisfied ties on rules 1 and 2 and piles onto DMA0, and the second
engine becomes decoration: on the MobileNetV2 block that was 10 descriptors /
1390 cycles on DMA0 against 4 / 240 on DMA1. With it, 8 / 894 against 6 / 734.

### 2.3 The one queue whose order is frozen

DIALECT_L2.md §6.1's weight-bank invariant — "for every `kea.mm`, the
`kea.load_w` that most recently targeted the same bank is the immediately
preceding MXU instruction" — is only an invariant if MXU instructions keep
their relative order. So the graph carries a chain between consecutive MXU
instructions. This costs nothing: the MXU queue is in-order and a chain of
`MATMUL`s accumulating into one ACC region needs no synchronization anyway
(ISA.md §7.3).

**Every other queue is free to be reordered**, and that freedom is worth a lot.
`-kea-tile` emits `… VQUANT(tile N) … VCOPY-fill(tile N+1) …`, both on the VPU.
Keeping that order pins tile *N+1*'s halo fill behind tile *N*'s requantize,
which pins tile *N+1*'s `DMA_LD` behind the fill, which pins its `MATMUL`s
behind the load — a recurrence that serializes the whole pipeline through the
VPU for no reason. Measured on `@pointwise_64_to_16`: 51,580 cycles with the
VPU order frozen, **33,071 without**. Reordering inside a queue needs no
synchronization either, because the emitted order *is* the execution order.

Two things do have to be kept out of the middle of a `LOAD_W`/`MATMUL` pair,
because a `kea.signal` is an MXU instruction too:

* a `WAIT` the `kea.mm` needs is emitted above the `kea.load_w` instead;
* a `SIGNAL` a `kea.load_w` owes (only possible via a WAR edge from a later DMA
  refilling that weight tile) is emitted below the `kea.mm` instead, and the
  graph gets a matching `mm → consumer` edge so Rule D still holds.

---

## 3. Dependences

Level 2 attaches its memory effects to the **operands**, not as an op-level
trait (DIALECT_L2.md §1.2), so MLIR's effect analysis already reports per-buffer
reads and writes. The graph is the ordinary closure over each buffer's use list
in `-kea-tile`'s order:

| edge | from | to |
|---|---|---|
| RAW | last writer | this reader |
| WAW | last writer | this writer |
| WAR | every reader since the last write | this writer |

Every edge points forwards in the sequential order, so the graph is acyclic and
*any* topological order of it executes the same program. DRAM buffers get
exactly the same treatment as scratchpads — a `DMA_ST` into an inter-layer
activation and the next layer's `DMA_LD` of it are a real dependence even
though nothing on chip connects them, and that is the edge that keeps a
three-layer block correct.

Ops with no buffer operands (`kea.trace`, `kea.halt`) are not scheduled: the
region markers are placed afterwards (§7) and `HALT` is pinned last.

---

## 4. The list scheduler and its two guards

At each step the ready set is priced with `hw_config.h`'s occupancy functions
and the candidate that can **start** soonest wins; ties go to the longest
remaining dependence path, then to `-kea-tile`'s order, so the result is
deterministic. The model tracks, per queue, when the functional resource frees
and which dispatched instructions have not started yet, plus a single
dispatcher clock advancing one instruction per cycle (`KEA_DISPATCH_PER_CYCLE`).

`SIGNAL`'s drain is modelled: a producer becomes *visible* to another queue at
`finish + pipelineDepth(queue)` — 32 cycles for MXU, 12 for DWU, 8 for VPU, 0
for DMA — which is exactly ISA.md §5.3 step 1.

### 4.1 Queue depth

The dispatcher is in-order and its **only** stall condition is a full target
queue. Dispatching a prefetch into a queue that already holds
`KEA_QUEUE_DEPTH` (16) unstarted instructions stalls *every* unit, which is
counterproductive — so the delay is charged to the candidate twice, once as a
later start and once as an explicit penalty, and a candidate that does not
stall wins. Errata E3 is honoured: a slot freed this cycle is usable this
cycle, so the occupancy test is "start strictly after now".

The pass publishes the resulting high-water mark per queue in `kea.schedule`,
and it is never above `queue_depth`. `compiler/test/kea-schedule.mlir`'s
`QDEPTH` run line shrinks the modelled queues to two slots and checks that the
schedule follows: the marks come down, and the program is still correct.

The model is approximate and knows it — `kea-sim` measures a real 321-cycle
dispatcher stall on the scheduled MobileNetV2 block against the model's zero,
because the model does not simulate a blocked `WAIT` at a queue head backing
instructions up behind it. What it does capture is the *first-order* effect,
which is enough to stop the scheduler running 40 instructions ahead on one
queue: the same block unscheduled stalls the dispatcher for 893 cycles.

### 4.2 Scratchpad high-water mark

Hoisting a load makes its tile live earlier — that is the entire point — but
four tile sets live at once do not fit in the half-scratchpad `-kea-tile`
reserved. A candidate that would push a space over capacity loses to any
candidate that fits. §6.2 explains why this guard alone is not sufficient and
what closes the gap.

---

## 5. Semaphores, and why Rule D is structural

### 5.1 Events are token channels

**One event per ordered pair of queues** — at most 5 × 4 = 20 of the 32, and in
practice fewer (18 for the whole MobileNetV2 block). Every `SIGNAL` increments
by 1 and every `WAIT` consumes 1, so the channel's counter is simply "signals
issued by *P* minus waits taken by *C*". Both queues are in-order, so *C*'s
*i*-th wait is released by *P*'s *i*-th signal: a counting semaphore used as a
FIFO of tokens between two totally ordered streams.

The alternative — one event per producer instruction, recycled — is a trap. An
event can only be safely reused once every wait on its previous generation has
*executed*, and stream order does not tell you that; a recycled event lets a
later waiter consume a token an earlier waiter had not taken yet. The
per-channel scheme has no generations to recycle.

### 5.2 Which waits get emitted

For each instruction *c*, `need[c][u]` is the latest stream position on queue
*u* that *c* transitively depends on, computed in one forward pass over the
graph. `front[C][u]` is what queue *C* has already waited for on *u*, directly
or transitively. A wait is emitted only when `need[c][u] > front[C][u]`, so:

* two instructions on the **same queue** are never synchronized (ISA.md §7.3 —
  each unit is in-order, so back-to-back `MATMUL`s accumulating into one ACC
  region need nothing at all);
* a dependence an **earlier wait on the same queue** already covers costs
  nothing, because the queue that waited is ordered after everything the
  producer was ordered after.

On the MobileNetV2 inverted-residual block that is 32 signals and 32 waits over
103 instructions, on 18 events.

### 5.3 Rule D, by construction

> **Rule D** (ISA.md §5.5). For every `WAIT e, thr` at stream position *p*, the
> `SIGNAL`s that supply those `thr` counts must appear at positions **< p**.

Three facts, and it falls out:

1. **`front[C][u]` only ever moves forwards**, so the producer positions queue
   *C* waits for on channel *(u, C)* are strictly increasing in *C*'s order.
2. **The signals on that channel are emitted adjacent to those same positions**,
   in the same increasing order. So the *i*-th signal on the channel and the
   *i*-th wait on it correspond to the same producer.
3. **The stream is a topological order of the dependence graph**, so that
   producer precedes its consumer, and the signal (right after the producer)
   precedes the wait (right before the consumer).

Hence for every *i* and every channel, the *i*-th signal sits at a lower stream
position than the *i*-th wait. That is Rule D, with nothing to repair
afterwards. The two fix-ups in §2.3 move a wait earlier and a signal later, and
both directions are checked in the code comments to preserve the property.

`checkRuleD()` walks the finished block and simulates the counters anyway,
because the assembler rejects a violating program and `kea-sim` reports one, and
a compiler that can only find out at that distance from the mistake is not much
use. It has never fired. `kea-as` and `kea-sim` are the independent check: every
measurement in §8 went through both, and `kea-sim --strict-hazards
--strict-poison` — which makes an unsynchronized cross-unit read fatal — passes
on the scheduled MobileNetV2 block.

### 5.4 What errata E1 and E4 change

E1 (a same-cycle `SIGNAL` releases a same-cycle `WAIT`) means the model does not
need a +1 between a producer becoming visible and a consumer starting; the
consumer's start is `max(visible(producer), …)`, not `visible + 1`.

E4 (a `REGION_END` closes after the issuing unit drains) is why the region
markers go on the queue that does the region's arithmetic — see §7.

---

## 6. ADR-0002's soundness obligation

> **After `-kea-schedule`, block order must be a sound over-approximation of
> temporal liveness. If two operations can execute concurrently — different
> queues, with no semaphore ordering them — then every buffer they touch must
> have overlapping block-order live ranges.**

Violating it is not a crash. `-kea-alloc` sees two disjoint ranges, does exactly
what it is supposed to do, gives the two buffers the same address, and two
hardware units then write the same bytes at the same time.

### 6.1 The discharge

**The `live` attribute is not the lever.** `-kea-alloc` does not trust it; it
re-derives the range from the SSA def-use chain as `[position of the kea.alloc,
position of its last user]` (MEMORY_PLANNING.md §2.1). Writing a wider attribute
would achieve nothing. What *does* move the range is moving the `kea.alloc`,
which is free — it is not an instruction.

Happens-before is exact and cheap here. Because each queue is in-order, "the set
of instructions on queue *u* that *o* is ordered after" is a **prefix** of *u*'s
stream. So the whole relation is five integers per instruction:

```
HB(q, o)  ⟺  pos(q) ≤ hbFront[o][queue(q)]
```

and `hbFront` is exactly the `front` vector §5.2 already maintains. The rule is
then:

> For every on-chip buffer *b*, move its `kea.alloc` to
>
> **lo(b)** = min over users *o* of *b*, of the earliest stream position
> *q* ≤ pos(*o*) such that ¬HB(*q*, *o*) and the instruction at *q* also
> touches an on-chip buffer in *b*'s address space.

Restricting to *b*'s own space is exact, because `-kea-alloc` only ever aliases
buffers within one space.

**Why hoisting the start is sufficient — the end never needs extending.** Take
any two instructions *X* at position *x* and *Y* at position *y > x* that may
run concurrently, with *X* touching *b₁* and *Y* touching *b₂*, both on chip in
the same space *S*. Then

* *x* ∈ range(*b₁*): the alloc precedes every user, so first(*b₁*) ≤ *x*, and
  *X* is a user so last(*b₁*) ≥ *x*;
* *x* ∈ range(*b₂*): first(*b₂*) = lo(*b₂*) ≤ *x*, because *X* sits at *x* ≤ *y*,
  is unordered with the *b₂*-user *Y*, and touches an *S* buffer — which is
  exactly what lo(*b₂*) minimizes over; and last(*b₂*) ≥ *y* > *x*.

Both ranges contain position *x*, so they overlap, so `-kea-alloc` must separate
them. ∎

It is also tight enough to be useful: happens-before is dense in a real program
(a tile's DMA is ordered before the compute that consumes it, which is ordered
before the requantize), so lo(*b*) typically reaches back exactly one tile —
precisely the window `spm-reserve-factor = 2` was sized for.

`compiler/test/kea-schedule-aliasing.mlir` is the test that would catch a
violation. It runs `-kea-alloc` on the *unscheduled* two-tile program and pins
the hazard in the output — `t.a0` and `t.a1` both at `addr = 0`, both ACC
regions at `acc:0`, both output tiles at `a:1040` — then runs
`-kea-schedule -kea-alloc` and pins all three pairs separated, with a
`-kea-alloc=verify-only=true` re-run proving from scratch that nothing with
overlapping ranges shares storage.

### 6.2 The buffers-in-flight bound

`-kea-tile` gives every tile a fresh `kea.alloc`, so nothing in the IR says
"there are only two activation buffers and they rotate". Left alone the
scheduler will run four tiles ahead, §6.1 will then — quite correctly — give all
four overlapping ranges, and `-kea-alloc` will need four tiles' worth of SPM and
refuse. Shaving the ranges is not an option: it trades a loud allocator failure
for a silent data race.

So the rotation is made explicit, exactly as ISA.md §12 writes it by hand. In
each address space, buffer *i*'s first use is ordered after buffer *i−K*'s last
use — an ordinary dependence, which §5 turns into an ordinary `WAIT`, and which
*is* the "buffer 0 free" handshake of the hand-written double-buffered layer.
*K* starts at how many of that space's buffers fit at once, computed from the
real extents; if the extended ranges still do not fit, *K* comes down and the
schedule is recomputed. *K* = 1 is the floor and always fits, because then no two
buffers of a space are ever live together.

The fixpoint converges in one iteration on the MobileNetV2 block (*K* = 7 for
SPM_A) and in three on `@pointwise_64_to_16` (6 → 4), which
`compiler/test/kea-schedule-tileloop.mlir` pins. `report-schedule=true`
publishes `buffers_in_flight`, `capacity_iters` and the resulting per-space
peaks.

---

## 7. Regions

`kea.trace` is queue-agnostic, so its unit is part of its meaning. Both markers
of a region go on the queue that does the region's arithmetic — MXU if it has
any `kea.mm`, else DWU, else VPU — because errata E4 says a `REGION_END` keeps
accumulating until the *issuing* unit's pipeline drains, and that is only the
honest attribution if the issuing unit is the one doing the work.

The markers bracket every instruction of the region on whichever queue it ended
up, because a region's counters cover every unit's activity in its cycle window
(SIMULATOR.md §5) — a layer must own the DMA that feeds it. The consequence
worth stating: **software pipelining makes consecutive regions overlap**, so a
prefetch issued for layer *n+1* during layer *n*'s window is attributed to layer
*n*. Errata E4 already anticipates this ("the tail of layer *n* really is
executing while layer *n+1* starts"); with cross-layer prefetch the overlap is
larger than a pipeline depth. It is an accounting artifact of the optimization,
not a modelling error, but a per-layer roofline should be read with it in mind.

---

## 8. Measured

`compiler/test/kea-schedule-measure.py` compiles one model twice —
`-kea-schedule=mode=serial` and `-kea-schedule=mode=overlap` — emits `.kasm`
with `kea-translate --sync=none` (or with its own transcription of
ASSEMBLY.md §5, `--emitter=builtin`), assembles with `kea-as` and runs
`kea-sim`. Both programs are produced by the same pass from the same tiled IR
and go through the same assembler, so the comparison is like for like.

**The baseline, stated precisely.** `mode=serial` is `-kea-tile`'s order, every
DMA on one engine, and a handshake at every cross-queue adjacency: the correct
sequential program, executed as a sequential program on a five-queue machine.
It is synchronized by the same machinery as the real schedule — same token
channels, same Rule D proof — and simply has nothing left to overlap. That is
the honest "before": not a different compiler, the same compiler with the
overlap turned off.

```
python3 compiler/test/kea-schedule-measure.py \
        compiler/test/kea-schedule-e2e.mlir --func mobilenet_v2_inverted_residual
python3 compiler/test/kea-schedule-measure.py \
        compiler/test/kea-schedule-tileloop.mlir --func pointwise_64_to_16 \
        --emitter=builtin
python3 compiler/test/kea-schedule-measure.py \
        compiler/test/kea-schedule-tileloop.mlir --func conv_tile_loop \
        --emitter=builtin
```

### 8.1 Headline

| program | shape | unscheduled | scheduled | **speedup** |
|---|---|---:|---:|---:|
| `@pointwise_64_to_16` | 1×1 conv 64→16, 64², 4 spatial tiles, DMA/compute balanced | 58,066 | **33,071** | **1.756×** |
| `@conv_tile_loop` | 3×3 conv 16→32, 64², 4 tile iterations, MXU bound | 120,642 | **92,750** | **1.301×** |
| `@mobilenet_v2_inverted_residual` | the integration test: 3 dependent layers, 1 tile each | 2,750 | **2,132** | **1.290×** |

All cycle counts are `kea-sim`'s `total_cycles`. `sim/tests/test_timing.cpp`
gets 1.709× on a hand-written double-buffered program; the pass gets 1.756× on
the layer whose balance matches that test.

### 8.2 Per-unit breakdown — `@pointwise_64_to_16`

The layer double buffering exists for: a 1×1 projection has an arithmetic
intensity of ~8 ops/DRAM byte, well under the int8 ridge point of 32, so DMA
and MXU cost about the same per tile and overlapping them is worth nearly 2×.

| | unscheduled | scheduled |
|---|---|---|
| **cycles** | **58,066** | **33,071** |
| MXU busy / sem-stall / res-stall / idle | 16,488 / 37,426 / 18 / 4,134 | 16,488 / 12,434 / 18 / 4,131 |
| VPU | 12,324 / 42,629 / 20 / 3,093 | 12,324 / 17,638 / 18 / 3,091 |
| DMA0 | 29,040 / 29,004 / 21 / 1 | 17,583 / 14,447 / 14 / 1,027 |
| DMA1 | 0 / 0 / 0 / 58,066 | 19,643 / 8,226 / 12 / 5,190 |
| dispatcher stalled | 37,264 | 21,570 |
| DRAM bytes | 328,896 | 328,896 |
| max queue depth (MXU / VPU / DMA0 / DMA1) | 16 / 14 / 14 / 0 | 16 / 12 / 8 / 10 |

Read three things off it:

* **Busy cycles are identical on every compute unit.** The pass does not remove
  work; it removes waiting. MXU semaphore stall falls from 37,426 to 12,434 and
  VPU's from 42,629 to 17,638 — 50,000 cycles of stall recovered out of a
  25,000-cycle saving, because the units now stall *concurrently* instead of in
  series.
* **The second engine goes from idle for the entire run to carrying 19,643
  busy cycles.** Total DMA busy rises (29,040 → 37,226) because two concurrent
  transfers each get 8 B/cycle rather than 16, exactly MICROARCH.md §6.3 — the
  second engine buys concurrency, not bandwidth, and the concurrency is worth
  far more than the contention costs.
* **DRAM bytes are unchanged**, which is the check that this is a scheduling
  result and not an accidental change of program.

### 8.3 Per-unit breakdown — `@conv_tile_loop`

| | unscheduled | scheduled |
|---|---|---|
| **cycles** | **120,642** | **92,750** |
| MXU | 73,912 / 38,482 / 18 / 8,230 | 73,912 / 10,593 / 19 / 8,226 |
| VPU | 12,716 / 84,709 / 20 / 23,197 | 12,716 / 65,848 / 16 / 14,170 |
| DMA0 | 33,800 / 78,552 / 21 / 8,269 | 20,052 / 29,596 / 12 / 43,090 |
| DMA1 | 0 / 0 / 0 / 120,642 | 20,052 / 34,830 / 12 / 37,856 |
| dispatcher stalled | 95,830 | 67,938 |

The ceiling here is low and it is not the scheduler's: **MXU busy is 73,912 of
the 92,750 cycles, so the layer is 80% MXU-occupied and no amount of overlap
can do much better.** The residual 10,593 cycles of MXU semaphore stall are the
`-kea-tile` decision described in §9: this tile's ACC region is 32,768 words,
the whole accumulator, so the two ACC regions cannot both be live and tile
*N+1*'s `MATMUL`s must wait for tile *N*'s `VQUANT`.

### 8.4 Per-unit breakdown — the MobileNetV2 inverted-residual block

| | unscheduled | scheduled |
|---|---|---|
| **cycles** | **2,750** | **2,132** |
| MXU | 290 / 1,849 / 15 / 596 | 290 / 1,508 / 18 / 316 |
| DWU | 152 / 1,187 / 8 / 1,403 | 152 / 905 / 10 / 1,065 |
| VPU | 611 / 1,955 / 19 / 165 | 611 / 1,341 / 16 / 164 |
| DMA0 | 1,504 / 1,228 / 17 / 1 | 894 / 1,223 / 14 / 1 |
| DMA1 | 0 / 0 / 0 / 2,750 | 734 / 906 / 11 / 481 |
| dispatcher stalled | 893 | 321 |
| instructions | 91 | 103 |

This is the hardest case for a scheduler and the most honest one to report:
three layers of 8×8×24, each a single tile, each reading its input from the
DRAM the previous one wrote (DIALECT_L2.md §9 — inter-layer SPM residency is not
implemented). The dependence chain
`DMA_LD → MATMUL → VQUANT → DMA_ST → DMA_LD → …` is almost the whole program,
so there is very little *within* a layer to overlap. The 1.29× comes from
prefetching each layer's **weights and quantization parameters** — which depend
on nothing — under the previous layer's compute, and from putting the store of
layer *n* and the weight load of layer *n+1* on different engines. Eleven of
the 91 instructions move earlier.

The scheduled program passes `kea-sim --strict-hazards --strict-poison`, which
makes an unsynchronized cross-unit read and a read of never-written scratchpad
fatal. That is an independent check that the semaphores are complete, not just
sufficient for the numbers to come out.

### 8.5 What the pass's own model predicts

`report-schedule=true` publishes `modelled_cycles`, and it is close enough to be
useful and far enough to be worth stating:

| program | modelled | measured | error |
|---|---:|---:|---:|
| MobileNetV2 block, scheduled | 2,098 | 2,132 | −1.6% |
| MobileNetV2 block, serial | 2,903 | 2,750 | +5.6% |
| `@pointwise_64_to_16`, scheduled | 30,022 | 33,071 | −9.2% |

The model is optimistic on the big cases because it does not simulate
instructions backing up behind a blocked `WAIT` at a queue head. It is used for
*choosing* between candidates, where only the ordering of the estimates matters,
never for reporting.

---

## 9. What is not implemented, and what this pass found

Stated plainly rather than stubbed.

* **No software pipelining across the `mode=serial` boundary.** `serial` is a
  measurement control, not a fallback strategy; the only time the pass emits it
  automatically is if the buffers-in-flight fixpoint bottoms out at *K* = 1 and
  the ranges still do not fit, which means `-kea-tile`'s own tiles do not fit
  and is that pass's diagnostic to give.
* **No MXU reordering, ever.** DIALECT_L2.md §6.1 makes bank assignment
  `-kea-tile`'s job and final. A scheduler that renumbered banks could shorten
  some MXU stalls; it would also have to re-derive `accumulate`, and the payoff
  is small because the MXU queue is already dense.
* **The queue model does not simulate head-of-line blocking.** §4.1.
* **`-kea-tile` does not reserve ACC.** `spm-reserve-factor` divides SPM_A and
  SPM_W but the ACC constraint in DIALECT_L2.md §5.2 is
  `OCG_t·OH_t·OW_t·16 ≤ KEA_ACC_WORDS` with no reserve, so the tiler routinely
  hands one tile the entire accumulator — and then §5.3's cost model prices the
  MXU, VPU and DMA terms as `max(…)`, i.e. it *assumes* the overlap that a
  full-ACC tile makes impossible. The scheduler cannot fix this from below: with
  one ACC region there is a genuine MXU→VPU→MXU serialization every tile. On
  `@conv_tile_loop` that is the entire residual 10,593 cycles of MXU semaphore
  stall. **The fix belongs in `-kea-tile`**: apply the reserve factor to
  `KEA_ACC_WORDS` as well, or add an `acc-reserve-factor`.
* **`-kea-tile` reuses one `name` for every spatial tile of a layer.** Both
  tiles of `@conv_tile_loop` get `kea.alloc {name = "conv_tile_loop.0.atile"}`,
  which DIALECT_L2.md §4.1 says must be unique in the module and which
  `kea-translate` rejects outright. It does not affect scheduling or allocation
  (both key on the SSA value), and it does not show up on single-tile layers
  like the MobileNetV2 block, but it stops any genuinely tiled layer from
  reaching `.kasm` through the real backend — which is why §8's two tiled
  measurements use the harness's own `--emitter=builtin` transcription.
  **The fix belongs in `-kea-tile`**: suffix the name with the tile index.
* **Regions overlap after software pipelining.** §7. Correct, and worth knowing
  before reading a per-layer roofline.
* **No cross-function scheduling.** The pass is per `func.func`, and a Level 2
  function is one block by construction.
