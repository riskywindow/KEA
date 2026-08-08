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
7. Check that regions on one queue do not interleave, so per-layer accounting
   stays partitionable (§7).
8. And, in `mode=auto`, do 2–7 a second time for a plan that does *not* reorder,
   and emit the reordered one only if it is predicted to beat that by a margin —
   otherwise emit nothing at all and leave `-kea-tile`'s program alone (§9).

```bash
kea-opt in.mlir -kea-schedule                        # mode=auto: §9
kea-opt in.mlir -kea-schedule=report-schedule=true   # publish kea.schedule
kea-opt in.mlir -kea-schedule=mode=overlap           # always reorder
kea-opt in.mlir -kea-schedule=mode=serial            # never -- a measurement control
kea-opt in.mlir -kea-schedule=decline-margin=0       # reorder on any predicted win
kea-opt in.mlir -kea-schedule=region-lookahead=4     # run further ahead, §4.3
kea-opt in.mlir -kea-schedule=fragmentation-margin=0 # spend the whole SPM, §6.3
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

### 2.2 Which DMA engine — spillover, not load balancing

Two engines share one 16 B/cycle DRAM port (MICROARCH.md §6.1), so a second
engine buys **no bandwidth**: two concurrent transfers get 8 B/cycle each. It
buys *concurrency* — a prefetch and a write-back in flight at the same time,
which is the entire reason ISA.md §12's hand-written layer alternates them.

Each descriptor is priced on both engines and takes the one that

1. can **start** it soonest (each engine's queue is in-order, so this is the
   later of "engine free" and "dependences visible"); then
2. has the shorter **contended occupancy** — `keaDmaOccupancy()` plus the cycles
   the other engine's data phase overlaps this one's, the 16 → 8 B/cycle stretch.

**And nothing else. There is deliberately no load-balancing tie-break.** If both
engines can start a descriptor in the same cycle, moving it to the idle one buys
nothing — it finishes no earlier — and costs real cycles as soon as anything
else wants the port. DMA1 is *spillover*: it gets a descriptor exactly when DMA0
is still busy and DMA1 would start sooner, which is the only situation in which
a second engine is worth anything.

This was originally a balancing rule ("on a tie, the less loaded engine"), added
because DMA0 otherwise took 10 of 14 descriptors on a small block. It is the
single worst thing the pass ever did. On a 28-convolution MobileNetV2 prefix,
which is DRAM-bandwidth bound (12.1 MB of traffic, DMA busy 948k cycles against
MXU 609k), balancing split a stream that had nothing to overlap with, doubled
every transfer's data-phase latency, and lengthened the `DMA → MXU → VPU → DMA`
critical chain:

These are historical figures, taken on a 28-convolution prefix when
`spm-reserve-factor` was still the knob that set tile sizes. They are kept
because they are why the policy is what it is; the current whole-network numbers
are in §8.1.1.

| 28-convolution prefix (historical) | cycles |
|---|---:|
| not scheduled | 2,130,318 |
| scheduled, balancing across both engines | 2,246,180 |
| scheduled, everything on one engine | **1,845,443** |

Spillover recovers all of that and costs nothing where a second engine does
help: `@conv_tile_loop` went from 1.301× to 1.511× on the same change, and
`@pointwise_64_to_16` was unaffected at 1.756×.

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

### 4.3 Region locality

An instruction may not be emitted more than `region-lookahead` (default 1)
`kea.trace` regions — layers, in practice — ahead of the oldest region that
still has unemitted work. Prefetching the *next* layer's weights is where the
cross-layer win comes from; prefetching the twentieth layer's 20-byte
`addparams` block is not. Left unbounded the scheduler did exactly that, which
cost nothing in cycles but held a scratchpad buffer live for the whole program
and dragged that layer's `TRACE` marker to the top of the stream with it (§7).

Measured on the MobileNetV2 feature extractor the bound is free — 3,176,072
cycles at lookahead 1 against 3,176,060 at lookahead 999 — so it is on by
default. It also makes §7's per-queue marker placement robust: a region's
instructions on its own queue cannot be separated by more than one region's
worth of other work, so regions on one queue cannot interleave.

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
use.

**It has fired, once, and the argument above was wrong in exactly one place.**
Every lowerable prefix of MobileNetV2 at or beyond node 99 failed it, while the
same prefix without `--schedule` compiled cleanly, and the threshold moved with
whatever changed the tile sizes — at the time, `spm-reserve-factor`
(`demo/regress/run_regressions.sh` case 3, since fixed). Program length, not any
particular layer. The mechanism, in three steps:

1. §6.2's rotation edges are added *inside* the capacity fixpoint, after the
   dependence graph was built — and **the last user of an SPM_W weight tile is a
   `kea.load_w`**. So a rotation edge makes the DMA that refills a later weight
   tile a direct consumer of a LOAD_W.
2. `protectWeightPairs()`, which redirects exactly such consumers onto the
   paired `kea.mm` so that no SIGNAL is ever slipped between a LOAD_W and its
   MATMUL, had already run and never saw those edges. So `need` pointed at the
   LOAD_W.
3. §2.3's fix-up then moved that SIGNAL down onto the MATMUL — where another
   SIGNAL on the same `(MXU, DMAx)` channel already sat — and the two collapsed
   into one flag. Two waits, one token. Fact 2 above ("the *i*-th signal and the
   *i*-th wait correspond to the same producer") had quietly become false,
   because a *position* could owe two tokens and only one was emitted.

Both halves are now closed. `protectWeightPairs()` runs again after the rotation
edges, so no cross-queue consumer ever depends on a LOAD_W directly; and a
SIGNAL carries a **count** rather than a flag, so even if two producers' tokens
did land on one position they would stay two tokens. Either fix alone is
sufficient; both are cheap. `compiler/test/kea-schedule-ruled.mlir` is the
regression test — three weight tiles that cannot all be resident, which is the
smallest program that forces a rotation edge out of a `kea.load_w` — and it
fails if either half is reverted. `kea-as` and `kea-sim` are the independent check: every
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
precisely the window a scratchpad with room for two tile sets provides.

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

At *K* > 1 one edge, from the previous generation's last user, is enough: *K*
buffers are *meant* to be live together, so the bound is about how many. **At
*K* = 1 one edge is not enough**, and the difference is why this loop needs a
floor. One edge orders the two buffers along one queue only, and §6.1's `lo()`
walks back along every *other* queue until it finds something unordered, so the
widened ranges cascade: measured on a 28-convolution prefix at
`spm-reserve-factor=1`, twenty 112 KB activation tiles came out mutually
overlapping — a 2.2 MB peak against a 256 KB scratchpad, which `-kea-alloc`
correctly refused. *K* = 1 means "these buffers may never be live together", so
the floor says exactly that: order the next generation's first use after *every*
user of the previous one, a handoff on every queue that touched it.

The fixpoint converges in one iteration on the MobileNetV2 block (*K* = 7 for
SPM_A) and in three on `@pointwise_64_to_16` (6 → 4), which
`compiler/test/kea-schedule-tileloop.mlir` pins. `report-schedule=true`
publishes `buffers_in_flight`, `capacity_iters` and the resulting per-space
peaks.

### 6.3 The anti-fragmentation margin

`spmPeak()` is `-kea-alloc`'s `maxlive`: a lower bound no allocator can beat.
It is not a guarantee, because that pass packs greedily by first fit
(MEMORY_PLANNING.md §3.1), and a plan whose maxlive fills 90% of a scratchpad
can still be refused for fragmentation. Measured: a 46-layer prefix at
`spm-reserve-factor=2` reached an SPM_W maxlive of 235,220 of 262,144 bytes and
`-kea-alloc` could not place an 11,520-byte qparam tile in what was left.

So the fixpoint keeps giving overlap back until every space is inside
`fragmentation-margin` percent of its capacity (default 6). The `fits` test is
still keyed to the real capacity, so the margin never *causes* a failure — it
only stops the loop one step earlier. It costs about 9% of the speedup on
programs that would have fitted anyway, which is the price of not turning a
compiling program into a hard allocator failure:

| | `fragmentation-margin=6` (default) | `=0` |
|---|---:|---:|
| `@pointwise_64_to_16` | 1.605× | **1.756×** |
| `@conv_tile_loop` | 1.388× | **1.511×** |
| 28-convolution prefix, reserve 2 | 1.484× | **1.603×** |
| 46-layer prefix, reserve 2 | compiles | **`-kea-alloc` fails** |

Set it to 0 if you know your program fits.

---

## 7. Regions — and why the markers cannot be allowed to float

`kea.trace` is queue-agnostic, so its unit is part of its meaning. Both markers
of a region go on the queue that does the region's arithmetic — MXU if it has
any `kea.mm`, else DWU, else VPU — because errata E4 says a `REGION_END` keeps
accumulating until the *issuing* unit's pipeline drains, and that is only the
honest attribution if the issuing unit is the one doing the work.

**The markers bracket that queue's instructions, not every queue's**, and that
distinction is the difference between a usable per-layer roofline and noise.

A `kea.trace` has no operands. Nothing in the dependence graph holds it
anywhere, so it is *placed*, and the obvious placement — before the earliest
stream position of anything in its region — makes it follow whatever floated
furthest. Something always does: a residual layer's 20-byte `addparams` DMA
depends on nothing at all, so the list scheduler issued it in the first hundred
instructions and the marker went with it. Measured on the MobileNetV2 feature
extractor: **11 of 52 region-begin markers opened inside the first 100 of 30,430
instructions, and the 52 regions summed to 23.6M cycles inside a 3.18M-cycle
program** — a 7.4× over-count, and a per-layer roofline that had to be taken
from the unscheduled build instead.

Two changes fix it, and they are independent:

1. **Keying the markers on the region's own queue** (`locateRegions()`). Work
   hoisted out of a layer is then attributed to whatever was executing when it
   actually ran, which is the standard convention for pipelined code and the one
   errata E4 already describes for a single pipeline depth. `REGION_END` is
   untouched: it still sits after the last instruction of the region on that
   queue, so E4's "retires immediately, keeps accumulating until the unit
   drains" semantics are preserved exactly.
2. **The region locality bound** (§4.3), which stops the underlying silliness
   rather than papering over it: that `addparams` DMA is now issued next to its
   own layer instead of 30,000 instructions early, where it held a scratchpad
   buffer live for the entire program for no benefit.

| MobileNetV2 feature extractor, 52 regions | Σ region cycles / total cycles |
|---|---:|
| unscheduled | 1.07 |
| scheduled, markers keyed on every queue | **7.43** |
| scheduled, markers keyed on the region's queue | **1.07** |

The scheduled build is now exactly as sound as the unscheduled one. The residual
7% is not an error: it is genuine cross-queue overlap between consecutive layers,
which errata E4 sanctions and which a per-layer roofline should show.

`checkRegions()` proves the structural property this rests on before the pass
returns: **regions on one queue must not interleave.** `kea-sim` nests regions
per `(unit, tag)`, so two windows that overlap without nesting double-count the
overlap instead of partitioning it. Markers on the arithmetic queue plus §4.3's
locality bound make interleaving impossible; the check is there because "impossible"
should be verified, not asserted.

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

All figures below are `kea-sim`'s `total_cycles`. The multi-layer ones are at
the defaults; the single-layer ones use `mode=overlap` so they measure the
reordering rather than the decision to reorder (§9).

| program | shape | unscheduled | scheduled | **speedup** |
|---|---|---:|---:|---:|
| `@pointwise_64_to_16` | 1×1 conv 64→16, 64², 4 spatial tiles, DMA/compute balanced | 58,070 | **36,190** | **1.605×** |
| `@conv_tile_loop` | 3×3 conv 16→32, 64², 4 tile iterations, MXU bound | 122,002 | **87,896** | **1.388×** |
| `@mobilenet_v2_inverted_residual` | the integration test: 3 dependent layers, 1 tile each | 2,755 | **2,135** | **1.290×** |

All cycle counts are `kea-sim`'s `total_cycles`, at the default
`fragmentation-margin=6`; §6.3 has the `=0` column, where the first two are
1.756× and 1.511×. `sim/tests/test_timing.cpp` gets 1.709× on a hand-written
double-buffered program.

### 8.1.1 Multi-layer — the numbers that matter most

The single-layer numbers above are each their own program: every one pays a cold
start and none overlaps a neighbour. The multi-layer measurements are the honest
ones. All of these are at the **defaults** — no flags — with `-kea-tile`
IMEM-aware and `spm-reserve-factor` at 1.

| MobileNetV2 feature extractor, nodes 0..178, 52 convolutions | cycles |
|---|---:|
| unscheduled | 3,727,589 |
| scheduled | **3,176,072** |
| **speedup** | **1.174×** |

DMA1 goes from 0% busy to 27.7%, DRAM traffic is unchanged, and the 52 TRACE
regions sum to 1.07× the program's cycles in both builds (§7).

**On the reported number.** 1.174× is `3,727,589 / 3,176,072` at
`imem-budget=20480`. An earlier draft of this document said 1.181×; that is
`3,696,622 / 3,129,389` at `imem-budget=20200`, which is a *different tiling*,
not a different division of the same counts. Both are real; the one to quote is
the one at the defaults the demo runs, which is 1.174×.

#### The budget curve, and why the tight end used to regress

`-kea-tile`'s instruction budget changes the tiling, and the tiling is what
decides whether there is anything to overlap. Measured across the feasible
window:

| `imem-budget` | unscheduled | scheduled | speedup | plan |
|---|---:|---:|---:|---|
| 19,100 | 3,721,080 | 3,721,080 | **1.000×** | declined |
| 19,500 | 3,665,429 | 3,665,429 | **1.000×** | declined |
| 20,200 | 3,696,622 | 3,129,389 | 1.181× | overlap |
| 20,480 (default) | 3,727,589 | 3,176,072 | **1.174×** | overlap |
| 21,300 | 3,649,751 | 3,179,786 | 1.148× | overlap |

At the two tightest budgets the tiler coarsens until an activation tile fills
SPM_A, `buffers_in_flight` for SPM_A is 1 before the capacity fixpoint even
starts, and there is nothing to double buffer — the same condition
DIALECT_L2.md §8 describes for `spm-reserve-factor=1`. Reordering there is pure
cost, and an earlier build measured it as such: **0.941×, a 6% regression.**

That is now 1.000×, exactly, because the pass declines and emits nothing at all
(§9), so the program is byte-identical to the unscheduled one. Two things had to
change for `auto` to see it:

* the in-order plan it compares against no longer carries §6.2's rotation edges,
  which it does not need — charging it for them made the estimate up to **8.2%
  pessimistic** (4,027,365 modelled against 3,721,080 measured), which was
  enough to make a plan that was really 6% worse look 0.8% better. It is now
  within 0.6% (3,745,218 against 3,721,080);
* the overlapped plan must clear a 5% margin, not merely win, because a 1%
  predicted difference is inside the model's error.

### 8.2 Per-unit breakdown — `@pointwise_64_to_16`

The layer double buffering exists for: a 1×1 projection has an arithmetic
intensity of ~8 ops/DRAM byte, well under the int8 ridge point of 32, so DMA and
MXU cost about the same per tile and overlapping them is worth nearly 2×.

| | unscheduled | scheduled |
|---|---|---|
| **cycles** | **58,070** | **36,190** |
| MXU busy / sem-stall / res-stall / idle | 16,488 / 37,430 / 18 / 4,134 | 16,488 / 15,549 / 21 / 4,132 |
| VPU | 12,324 / 42,633 / 20 / 3,093 | 12,324 / 20,757 / 19 / 3,090 |
| DMA0 | 29,040 / 29,008 / 21 / 1 | 22,784 / 13,387 / 15 / 4 |
| DMA1 | 0 / 0 / 0 / 58,070 | 14,444 / 12,388 / 10 / 9,348 |
| dispatcher stalled | 37,267 | 24,686 |
| DRAM bytes | 328,896 | 328,896 |

Read three things off it:

* **Busy cycles are identical on every compute unit.** The pass does not remove
  work; it removes waiting. MXU semaphore stall falls by 22,000 cycles and VPU's
  by 22,000, because the units now stall *concurrently* instead of in series.
* **The second engine goes from idle for the entire run to carrying 14,444 busy
  cycles**, and it earns them: DMA0 falls from 29,040 to 22,784, so the split is
  genuine concurrency rather than the same stream cut in half (§2.2).
* **DRAM bytes are unchanged**, which is the check that this is a scheduling
  result and not an accidental change of program.

### 8.3 Per-unit breakdown — `@conv_tile_loop`

| | unscheduled | scheduled |
|---|---|---|
| **cycles** | **122,002** | **87,896** |
| MXU | 74,096 / 43,738 / 34 / 4,134 | 74,096 / 9,638 / 32 / 4,130 |
| VPU | 13,016 / 88,157 / 40 / 20,789 | 13,016 / 65,580 / 33 / 9,267 |
| DMA0 | 34,456 / 79,160 / 41 / 8,345 | 34,126 / 53,742 / 24 / 4 |
| DMA1 | 0 / 0 / 0 / 122,002 | 7,086 / 907 / 6 / 79,897 |
| dispatcher stalled | 109,334 | 75,246 |

The ceiling here is low and it is not the scheduler's: **MXU busy is 74,096 of
the 87,896 cycles, so the layer is 84% MXU-occupied** and no amount of overlap
can do much better. MXU semaphore stall falls from 43,738 to 9,638 — most of
what was available has been taken. The residual is the `-kea-tile` decision in
§10: with `spm-reserve-factor` now applied to ACC this is much better than it
was, but a tile that fills the accumulator still forces tile *N+1*'s `MATMUL`s
to wait for tile *N*'s `VQUANT`.

Note DMA1 takes only 7,086 cycles of the 41,212 of DMA work. That is the
spillover policy doing its job: this layer is MXU bound, so a second engine has
nothing to overlap with and splitting the stream would only halve the port.

### 8.4 Per-unit breakdown — the MobileNetV2 inverted-residual block

| | unscheduled | scheduled |
|---|---|---|
| **cycles** | **2,755** | **2,135** |
| MXU | 290 / 1,853 / 15 / 597 | 290 / 1,513 / 16 / 316 |
| DWU | 152 / 1,190 / 8 / 1,405 | 152 / 936 / 9 / 1,038 |
| VPU | 611 / 1,960 / 19 / 165 | 611 / 1,348 / 14 / 162 |
| DMA0 | 1,504 / 1,233 / 17 / 1 | 1,259 / 860 / 12 / 4 |
| DMA1 | 0 / 0 / 0 / 2,755 | 363 / 14 / 5 / 1,753 |
| dispatcher stalled | 896 | 232 |
| DRAM bytes | 9,204 | 9,204 |

This is the hardest case for a scheduler and the most honest one to report:
three layers of 8×8×24, each a single tile, each reading its input from the DRAM
the previous one wrote (DIALECT_L2.md §9 — inter-layer SPM residency is not
implemented). The dependence chain
`DMA_LD → MATMUL → VQUANT → DMA_ST → DMA_LD → …` is almost the whole program, so
there is very little *within* a layer to overlap. The 1.29× comes from
prefetching each layer's **weights and quantization parameters** — which depend
on nothing — under the previous layer's compute, and from spilling 363 cycles of
that onto the second engine while DMA0 is busy.

The scheduled program passes `kea-sim --strict-hazards --strict-poison`, which
makes an unsynchronized cross-unit read and a read of never-written scratchpad
fatal. That is an independent check that the semaphores are complete, not just
sufficient for the numbers to come out.

### 8.5 What the pass's own model predicts

`report-schedule=true` publishes `modelled_cycles`, and its accuracy is what
makes `mode=auto` possible at all:

| program | modelled | measured | error |
|---|---:|---:|---:|
| 28-conv prefix, reserve 1, in-order plan | 2,124,565 | 2,126,037 | −0.07% |
| 28-conv prefix, reserve 1, overlap plan | 2,262,240 | 2,264,218 | −0.09% |
| MobileNetV2 block, scheduled | 2,098 | 2,135 | −1.7% |
| MobileNetV2 block, serial | 2,903 | 2,755 | +5.4% |

The two rows that matter are the first two: on the program where the decision is
close and wrong the wrong way round, the model is within 0.1% of `kea-sim` on
*both* plans and picks correctly. On small programs it is optimistic by a few
percent, because it does not simulate instructions backing up behind a blocked
`WAIT` at a queue head — but there the two plans differ by 20% or more, so the
error cannot flip the decision.

---

## 9. `mode=auto`: what it guarantees, and against which baseline

Reordering is not free. §6's soundness obligation is discharged with real
dependence edges, and on a program with no room to double buffer those edges are
pure cost. So the default costs two plans with one cost model and emits the
cheaper:

| mode | what it emits |
|---|---|
| `auto` (default) | the overlapped plan if it beats the in-order one by `decline-margin` percent (default 5); otherwise **nothing at all** |
| `overlap` | always the reordered plan, falling back to `serial` if it is not placeable |
| `serial` | never reorders and synchronizes every cross-queue adjacency — a measurement control, not a strategy |

**The baseline is the in-order plan, which models not scheduling at all** —
`-kea-tile`'s order, one DMA engine, minimal sync, no rotation edges, no widened
live ranges. It is deliberately *not* `mode=serial`: that one puts a handshake at
every cross-queue adjacency and measures **4,063,835 cycles on the feature
extractor, 9% slower than not scheduling**, so beating it would guarantee
nothing. Costing the in-order plan with the same cost model as the overlapped
one is what makes the comparison mean something, and it is accurate — within
0.6% of the measured unscheduled build at the budget where the decision is
closest.

**Declining means emitting nothing.** No reordering, no rotation edges, no
widened ranges, no `unit` attributes, no semaphores. `-kea-emit` then derives its
semaphores from post-allocation storage intervals exactly as it does without
`--schedule`, and `-kea-alloc` packs `-kea-tile`'s own ranges — so the output is
byte-identical to the unscheduled build. A pass that has decided not to help must
not still charge for the attempt; that is what turns the guarantee from "usually
better" into an exact 1.000× on the programs it declines.

So, precisely:

> **`mode=auto` guarantees that `--schedule` produces either a program its cost
> model predicts is at least 5% faster than not scheduling, or the identical
> program you would have got without it.** What it does not guarantee is that
> the prediction is right; the model's error is ~1% on the multi-layer programs
> measured here (§8.5), and the 5% margin is sized against that.

`auto` also enforces placeability: a plan whose widened ranges `-kea-alloc`
provably cannot place is never emitted, and if the overlapped plan is
unplaceable under `mode=overlap`, `serial` — which always fits — is emitted
instead.

**Should `--schedule` be on by default? Yes, with `mode=auto`.** It is worth
1.17× on the whole feature extractor at the defaults, up to 1.6× on layers with
room to double buffer, and exactly 1.000× where there is none.

## 10. What is not implemented, and what this pass found

Stated plainly rather than stubbed.

* **`serial` is a measurement control, not a strategy.** The pass emits it
  automatically only when neither the overlapped nor the in-order plan is
  placeable (§9), which means `-kea-tile`'s own tiles do not fit and is that
  pass's diagnostic to give.
* **No MXU reordering, ever.** DIALECT_L2.md §6.1 makes bank assignment
  `-kea-tile`'s job and final. A scheduler that renumbered banks could shorten
  some MXU stalls; it would also have to re-derive `accumulate`, and the payoff
  is small because the MXU queue is already dense.
* **The queue model does not simulate head-of-line blocking.** §4.1.
* **The instruction budget and the schedule are coupled, and nothing negotiates
  between them.** §8.1.1's curve is not monotonic: `-kea-tile` chooses a tiling
  against an IMEM budget without knowing whether that tiling leaves the
  scheduler anything to overlap, and below about 19,500 instructions it chooses
  one that does not. The scheduler then correctly declines, and the program ends
  up ~18% slower than it is 1,000 instructions of budget further up. Neither
  pass is wrong on its own terms; the search that would fix it — choosing the
  budget by what the *scheduled* program costs — spans both and belongs to
  neither. Today the demo finds the optimum by sweeping.
* **Regions overlap after software pipelining.** §7. Bounded now, but still
  worth knowing before reading a per-layer roofline.
* **The fit predicate is `maxlive`, not the allocator's own packing.** §6.3
  handles the gap with a margin rather than by reimplementing
  `-kea-alloc`'s greedy first fit. A scheduler that ran the real packer could
  drop the margin and take the ~9% back.
* **No cross-function scheduling.** The pass is per `func.func`, and a Level 2
  function is one block by construction.
