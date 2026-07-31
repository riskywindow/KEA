# `kea-sim` — the KEA-1 cycle-approximate simulator

Companion documents: [ISA.md](ISA.md) (the frozen instruction set),
[MICROARCH.md](MICROARCH.md) (the frozen timing model),
[ARTIFACT_FORMAT.md](ARTIFACT_FORMAT.md) (KEAF).

This document describes the *implementation*: how it is put together, exactly what its cycle
counts do and do not mean, how to read what it prints, and — most importantly — every place
where the frozen documents were silent or self-contradictory and a choice had to be made.

Source layout:

| Path | Contents |
|------|----------|
| `sim/include/kea/sim/memory.h` | the four address spaces, poison tracking, MXU weight banks |
| `sim/include/kea/sim/quant.h` | TOSA `apply_scale_32` (see §6) |
| `sim/include/kea/sim/stats.h` | every counter, and the roofline maths |
| `sim/include/kea/sim/simulator.h` | `Simulator`, `SimConfig`, `SimResult`, resource ids |
| `sim/include/kea/sim/program_builder.h` | build a `KeaProgram` in memory (tests, experiments) |
| `sim/src/functional.cpp` | the bit-exact functional model, one function per opcode |
| `sim/src/simulator.cpp` | the cycle-stepped timing model, Rule D, deadlock detection |
| `sim/src/stats.cpp` | text and JSON reports |
| `sim/tests/` | six test binaries, all registered with `add_test()` |
| `tools/kea-sim/` | the command line front end |

Every latency, occupancy, bandwidth and capacity is read from `include/kea/hw_config.h`. There
is no hardcoded cycle count anywhere in `sim/` — `grep` for a bare integer next to the word
"cycle" and you will find only `hw_config.h` symbols.

---

## 1. Architecture

Three layers, deliberately separable:

```
   KeaProgram  (from runtime/: .keaf reader or .kasm assembler, or ProgramBuilder)
        │
        ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ Simulator::run()   cycle-stepped, deterministic                  │
   │                                                                  │
   │  per cycle:  A  SIGNALs        (all units, ascending id)         │
   │              B  WAITs          (all units, ascending id)         │
   │              C  everything else starts                           │
   │              D  dispatch       (1 instr, program order)          │
   │              E  DMA engines advance, DRAM port arbitrated        │
   │              F  per-cycle counters, region bookkeeping           │
   │              G  termination / deadlock / overflow checks         │
   └───────────────┬──────────────────────────────┬───────────────────┘
                   │ at start-of-execution        │
                   ▼                              ▼
        functional.cpp (bit exact)          stats.cpp (counters)
                   │
                   ▼
        Machine { SPM_A, SPM_W, ACC, DRAM, wbank[2] }
```

**The functional model and the timing model are decoupled.** An instruction's entire
architectural effect is applied atomically at its *start* cycle; its *latency* only governs when
a `SIGNAL` on the same unit may retire and when the hazard detector considers the write
visible. This is sound because KEA-1 has no hardware dependency tracking: correctness is the
program's responsibility, expressed through `SIGNAL`/`WAIT` and through per-unit in-order
execution. Where a program gets that wrong, the simulator does not silently produce a
plausible answer — see the hazard detector in §4.

### 1.1 Resources, not units

The unit/resource split from MICROARCH.md §2 is modelled literally:

| Unit | Resources |
|------|-----------|
| MXU | `ARRAY`, `WPORT`, `BANK0`, `BANK1` |
| DWU | `DWPIPE` |
| VPU | `VPIPE` |
| DMA0 / DMA1 | `ENGINE0` / `ENGINE1` (+ a share of the global DRAM port) |

`LOAD_W` holds `WPORT` + `BANK[b]`; `MATMUL` holds `ARRAY` + `BANK[b]`. That is the whole
mechanism behind weight double buffering, and `sim/tests/test_timing.cpp` measures it: 20
`LOAD_W`/`MATMUL` pairs take **1403** cycles alternating banks and **1593** reusing one bank, a
**1.135×** difference against the model's `(68+10)/68 = 1.147×`.

### 1.2 The DMA engines

A DMA descriptor is not a fixed-occupancy instruction, because its duration depends on port
contention. It is a two-phase state machine per engine:

1. **overhead** — `KEA_DMA_DESC_OVERHEAD_CYCLES + n1*n2*KEA_DMA_ROW_OVERHEAD_CYCLES` cycles,
   requesting no DRAM bandwidth;
2. **data** — `len0*n1*n2` bytes at whatever the arbiter grants.

Arbitration, once per cycle, over engines in their data phase: one active engine gets all
16 B; two active engines get 8 B each; an odd remainder goes to the lower unit id. An engine
that finishes early does **not** donate its unused grant to the other engine within the same
cycle — see §5, approximation 5.

Uncontended, this reproduces `keaDmaOccupancy()` exactly, and
`test_timing.cpp::testDmaOccupancyExact` asserts it against the `static_assert`ed value in
`hw_config.h`.

---

## 2. What "cycle-approximate" means *here*

It means: **the cycle counts are the arithmetic of `hw_config.h`'s occupancy formulas, composed
by a faithful model of concurrency, queueing, and semaphores — and nothing else.**

### Modelled exactly

- One dispatcher, one instruction per cycle, strict program order, routed by `instr.unit`.
- Five depth-16 in-order queues. **A full target queue stalls the dispatcher**, which is the
  machine's only implicit backpressure and the mechanism by which a badly scheduled program
  serialises. `test_timing.cpp::testQueueBackpressure` measures 454 stall cycles on a program
  that overruns the VPU queue, and attributes all of them to the VPU.
- Per-resource occupancy, so intra-unit overlap (`LOAD_W` vs `MATMUL`) is real.
- Per-instruction latency, so a `SIGNAL`'s drain barrier is real. A `SIGNAL` after a `MATMUL`
  waits `occupancy + KEA_MXU_PIPELINE_DEPTH`; `test_sync.cpp` asserts the exact cycle (a
  `MATMUL` starting at cycle 1 with `M=64` retires at 101, and the `SIGNAL` retires at 101, not
  at 69 when the array frees).
- Counting semaphores with the exact acquire/release rules of ISA.md §5.3, including
  `thr == 0`, same-cycle release, and ascending-unit-id arbitration.
- Two DMA engines sharing one 16 B/cycle port, arbitrated every cycle.
- `TRACE` regions, nested per `(unit, tag)`.

### Not modelled (and therefore optimistic)

These are MICROARCH.md §10's list, restated with what this implementation actually does:

1. **Scratchpad port arbitration.** SPM_A and SPM_W are assumed infinitely banked. DMA, MXU,
   DWU and VPU can all hammer SPM_A in the same cycle with no penalty. This is the single
   biggest approximation. MICROARCH.md §6.4 floats a `--strict-spm` mode that would add
   bank-conflict stalls; **it is not implemented**, and no result produced by this simulator
   should be described as having used it.
2. **DRAM internals.** The DRAM port is a flat 16 B/cycle pipe. No row buffers, no page
   hits/misses, no refresh, no read/write turnaround, no CAS latency, no bank groups. The only
   nod to physical reality is `KEA_DMA_DESC_OVERHEAD_CYCLES` (a nominal row activate) and
   `KEA_DMA_ROW_OVERHEAD_CYCLES` per contiguous run. Real LPDDR would cost 10–30% more on
   scattered access.
3. **Alignment.** A misaligned `DMA` costs exactly what an aligned one costs.
4. **Instruction fetch.** Free, never misses, no IMEM bandwidth accounting.
5. **DRAM arbitration granularity.** The arbiter is fair-share-per-cycle with no rebalancing
   within a cycle and no notion of burst length or transaction reordering.
6. **`ACC` port arbitration.** None, by construction: MXU/DWU only write it and the VPU only
   reads it.
7. **Everything analog.** No power, no thermals, no DVFS, no clock gating.

There is also one place where the model is **pessimistic**: `SIGNAL` fully drains its unit's
pipeline, costing up to 32 cycles on the MXU. Real hardware could track per-instruction
completion and release earlier.

**Consequence: treat a `kea-sim` cycle count as a lower bound**, and say so in any reported
result.

### Determinism

Guaranteed, and tested (`test_sync.cpp::testDeterminism` runs a mixed MXU/DWU/VPU/DMA program
twice and compares total cycles, the full instruction trace, every region's cycle count, and an
FNV hash of SPM_A and ACC).

- Units are serviced in ascending unit id in every phase.
- Event increments are applied before any waiter is evaluated (§7, ambiguity 1).
- DRAM remainders go to the lower unit id.
- No `unordered_map`, no hashing, no wall-clock, no floating point anywhere in the scheduling
  path. Floating point appears only in the report formatter.

---

## 3. The functional model

All twelve non-`HALT` opcodes plus `HALT` are implemented bit-exactly, with no approximation
anywhere:

| Opcode | Notes |
|--------|-------|
| `NOP` | occupies the unit's main pipe for `cycles`; on `CTRL` it stalls the dispatcher |
| `HALT` | `exit_code` is surfaced in `SimResult` and as `kea-sim`'s exit status |
| `SIGNAL` / `WAIT` | full counting-semaphore semantics, drain barrier, overflow detection |
| `TRACE` | markers and nested regions, zero occupancy |
| `DMA_LD` / `DMA_ST` | 3D strided, signed and zero strides, both SPM spaces |
| `LOAD_W` | int8 and int4, `k_rows`/`n_cols` tail zeroing, both banks |
| `MATMUL` | 2D activation walk with independent inner/outer strides on both sides, accumulate/overwrite, int8 and int4 |
| `DWCONV` | 3×3 and 5×5, stride 1 and 2, arbitrary channel counts, accumulate/overwrite |
| `VQUANT` | per-channel bias/mult/shift, int8 and int4 output, `out_zp` then clamp |
| `VADD` | per-tensor requantized add, in-place legal |
| `VPOOL` | max and average, arbitrary window and stride |
| `VCOPY` | 2D strided copy and constant fill, SPM_A ⇄ SPM_W in both directions |

Three details that bite:

- **`ACC` is addressed in int32 words.** `AccMem` has no byte interface at all, so the mistake
  is not expressible. Out-of-range accesses produce a fault naming the word index and reminding
  you the space is word addressed.
- **int4 is little-nibble-first**, via `keaUnpackInt4`/`keaPackInt4` from `isa.h`. The simulator
  never does its own nibble arithmetic. `VQUANT`'s int4 output path read-modify-writes the byte
  and deliberately skips the poison check on the read, because a half-written byte is normal
  there.
- **Every piece of quantization arithmetic calls `isa.h`** — `keaRequantize`, `keaSrdhm`,
  `keaRdpot`, `keaQuantizedAdd`, `keaPoolAverage`, `keaClamp`. None of it is restated.

### Poison

`SPM_A`, `SPM_W` and `ACC` carry a per-element "has ever been written" bit, initialised to zero,
because ISA.md §2.3 declares them undefined at reset. A read of a never-written element is
reported (`--strict-poison` makes it fatal). MICROARCH.md §8.2 is right that this is worth the
effort: an unsynchronized `WAIT`-less read of a DMA target is the most likely compiler bug and
otherwise produces plausible-looking wrong numbers.

DRAM is **not** poison-tracked. Its contents are staged by the runtime from the KEAF `CONST`
section plus the input tensors, so "never written" is not a bug there; unwritten DRAM reads as
zero. DRAM is stored as 64 KiB pages allocated on first touch, so the 4 GiB space costs nothing.

---

## 4. Error detection

| Condition | How it is reported | Exit |
|-----------|--------------------|------|
| `keaValidate` failure, no `HALT`, code after `HALT` | load-time error with the pc | non-zero |
| **Rule D violation** (ISA.md §5.5) | static, before execution, with pc, unit, event, threshold, and the cumulative supply/demand that failed | non-zero |
| **Deadlock** | every blocked `(unit, pc, event, threshold)` plus every non-zero event counter | non-zero |
| Event counter over `KEA_EVENT_MAX` | pc of the offending `SIGNAL` | non-zero |
| Address outside its space | pc, space, interval, bound | non-zero |
| Read of never-written scratchpad | counted; first 8 reported; fatal under `--strict-poison` | 0 / non-zero |
| Cross-unit read of data whose producer has not retired | counted; first 8 reported; fatal under `--strict-hazards` | 0 / non-zero |
| `--max-cycles` exceeded | status `max-cycles` | non-zero |

**Rule D** is checked statically by walking the stream in dispatch order and tracking cumulative
`SIGNAL` supply and `WAIT` demand per event. Because the stream is straight-line and dispatch is
in order, this is an *exact* test, not a conservative one — it catches both "the `WAIT` comes
before its `SIGNAL`" and the subtler "two units both `WAIT e,1` but only one `SIGNAL e,1`
exists", since a counting acquire makes waiters split an event's counts. Disable it with
`--no-rule-d` if you want to watch the machine actually wedge.

**Deadlock detection** fires when, in one cycle, no unit started an instruction, nothing was
dispatched, no resource is occupied, no unit's pipeline is still draining, and at least one
queue head (or the dispatcher itself, on a `CTRL`-targeted `WAIT`) is a blocked `WAIT`. Both
shapes are tested: insufficient counts, and the pure Rule D wedge where the MXU queue fills
behind a blocked `WAIT` and the dispatcher then stalls on the next MXU instruction so the
releasing `SIGNAL` is never dispatched at all.

**The hazard detector** keeps a small list of in-flight write intervals per address space, each
with the retire cycle, pc and unit of its producer. A read that overlaps such an interval from a
*different* unit before the producer retires is a missing `SIGNAL`/`WAIT` pair. This subsumes
MICROARCH.md §8.2's "ACC written by MXU and DWU in overlapping regions" check and catches the
general case. It is range-based, not element-precise: a `MATMUL` registers the bounding interval
of its ACC walk, so a strided write with gaps is treated as covering the whole span. That makes
it conservative (it can warn where a truly disjoint access was fine), which is why it is a
warning by default.

---

## 5. Reading the stats

`kea-sim` prints a whole-program block, then one block per `TRACE` region, then markers.
`--stats-json FILE` writes the same content as `{"format": "kea.sim.stats", "version": 1, ...}`.

### Per-unit cycles

```
    MXU        292       373         9       180   busy  34.19%  instrs 15  avg-q 8.35  max-q 16
               busy   sem-stall  res-stall  idle
```

These four **partition every cycle of the run**, in that priority order:

- **busy** — at least one of the unit's resources was occupied. Note the priority: a unit that
  is draining a `MATMUL` while its next instruction sits on a `WAIT` counts as *busy*, not
  stalled, because it is still doing useful work. If you want the raw "how long was the head
  blocked" figure, use `--trace`: every instruction carries its own `stall_sem` and `stall_res`
  counts.
- **sem-stall** — otherwise, the queue head was a blocked `WAIT`, or a `SIGNAL` waiting for the
  unit's pipeline to drain.
- **res-stall** — otherwise, the queue head was ready but a resource (or the one-instruction-
  per-cycle issue slot) was busy.
- **idle** — otherwise; the queue was empty.

`avg-q` / `max-q` are queue depth. A `max-q` of 16 means that queue hit the depth limit and
therefore stalled the dispatcher at least once.

### Dispatcher

```
  dispatcher   stalled 0 cycles (0.00%), dispatched 39
    attributed to full queue: VPU=454
```

Stall cycles are attributed to the unit whose queue was full at the time. This is the direct
measurement of "my schedule is serialising because I emitted too many consecutive instructions
for one unit".

### Compute utilisation

```
  MXU    4 MATMUL, 4 LOAD_W, useful 65536 MAC, issued 65536 MAC
    MAC utilisation  29.98% of the 256 int8 MAC/cycle peak (issued 29.98%, padding efficiency 100.00%)
```

- **useful MAC** = `M × k_rows × n_cols`, using the extents of the weight tile actually resident
  in the named bank.
- **issued MAC** = `M × 256`, what the array physically ran.
- **MAC utilisation** = useful ÷ (cycles × 256).
- **padding efficiency** = useful ÷ issued. A layer can sit at 100% MXU occupancy while doing
  19% useful work because `IC = 3`; only the two counters together tell you which problem you
  have (MICROARCH.md §9.2 insists on this and it is right).

int4 `MATMUL`s are counted in `macs_useful_int4` as well, so a reader can renormalise against
the 512 MAC/cycle int4 peak. DWU and VPU get the same treatment against their own peaks
(128 MAC/cycle, 16 elem/cycle).

### DMA and DRAM

```
  DMA0   4 desc, 4800 B, 5.621 GB/s achieved, 427 data cycles (254 contended)
  DMA1   3 desc, 4096 B, 4.796 GB/s achieved, 383 data cycles (254 contended)
  DRAM   8896 B total (6848 load / 2048 store), 10.417 GB/s of 16.0 GB/s peak, port busy 65.11%
    lost to port sharing 254.0 engine-cycles of DRAM bandwidth
```

- **data cycles** — cycles the engine spent in its data phase actually requesting the port.
- **contended** — of those, cycles it had to share.
- **lost to port sharing** — `Σ over cycles of (1 − granted / 16)` across both engines, i.e.
  engine-cycles of bandwidth the engine asked for and did not get *purely because the other
  engine was also active*. This is the number that quantifies "the two engines exist for
  scheduling, not for bandwidth".
- Achieved GB/s is bytes ÷ (cycles ÷ `KEA_CLOCK_HZ`), so it is diluted by any cycles in which
  the engine was idle. Compare it against 16 GB/s only for regions where DMA is the intended
  bottleneck.

### Roofline

```
  roofline
    useful ops           131072   (issued 131072, padding efficiency 100.00%)
    arithmetic intensity 14.734 ops/DRAM byte  (int8 ridge point 32.0)
    achieved             153.480 GOPS of 235.741 GOPS attainable (65.11%); peak 512 GOPS
    bound                MEMORY (below the ridge point)
```

Computed exactly as MICROARCH.md §9.2 specifies, per region and for the whole program:

```
intensity  = ops_useful / dram_bytes                    [ops/byte]   (1 MAC = 2 ops)
attainable = min(PEAK_OPS, intensity * PEAK_DRAM_BW)    [ops/s]
achieved   = ops_useful / (cycles / KEA_CLOCK_HZ)       [ops/s]
efficiency = achieved / attainable
```

`PEAK_OPS` is the int8 roof (512 GOPS) unless the majority of useful MXU MACs in that block ran
in int4, in which case the int4 roof (1024 GOPS) is used; the JSON reports `peak_ops_per_s`
explicitly so a plotter never has to guess. `memory_bound` is true when the bandwidth roof lies
below the compute roof, i.e. when intensity is under 32 ops/byte.

To plot a roofline, take `intensity_ops_per_byte` and `achieved_ops_per_s` from each region of
the JSON — one point per layer — and draw the roof from `peak_ops_per_s`,
`peak_gb_per_s` and `ridge_point_ops_per_byte`.

**Caveat that matters for honest reporting:** `dram_bytes` is DMA traffic, counted at descriptor
start. A region whose activations stay in SPM_A has near-zero DRAM bytes and therefore a
near-infinite intensity; that is correct (it *is* compute bound) but the number is not
meaningful as a plotted x-coordinate. Regions with `dram_bytes == 0` report intensity 0 and are
treated as compute bound.

### Regions

A region is a `TRACE REGION_BEGIN`/`REGION_END` pair. Its counters cover **every unit's activity
inside its cycle window**, not just the unit that opened it — a layer opened on the MXU still
owns the DMA traffic issued to feed it, which is what per-layer attribution wants. Nested
regions each get the full set. Region `cycles` is `end − begin`, and the per-unit and per-cycle
counters inside a region sum to it exactly.

### Trace

`--trace` (to stdout) or `--trace=FILE` writes one row per queued instruction:

```
    pc unit opcode        issue      start     retire      sem      res
     0 MXU  MATMUL            0          1        101        0        0
     1 MXU  SIGNAL            1        101        102       99        0
```

`issue` is the cycle the dispatcher pushed it into a queue, `start` the cycle it began executing
(for `WAIT`/`SIGNAL`/`TRACE`, the cycle it retired), `retire` the cycle its last architectural
write is visible. `sem` and `res` are how many cycles it spent blocked at the queue head on a
semaphore and on a resource respectively — that is the per-instruction stall reason.

---

## 6. Requantization: two algorithms, one machine

**This is the most important thing in this document to get right, and the repository contains
two different functions that both look like "the requantizer".**

| | `kea::keaRequantize` (`include/kea/isa.h`) | TOSA `apply_scale_32` (`frontend/testdata/apply_scale_vectors.json`) |
|---|---|---|
| shape | `keaSrdhm` then `keaRdpot` — **rounds twice** | one 64-bit multiply-accumulate with a pre-added rounding term and **one** arithmetic shift |
| rounding | half **away from zero** (−1.5 → −2) | half **up**, toward +∞ (−1.5 → −1) |
| narrowing | saturating (`keaSrdhm` saturates the single overflow case) | **wraps** (`arith.trunci`) |
| `shift` | KEA convention: `>0` right shift after a Q31 multiply, `<0` left shift before | TOSA convention: `[0, 63]`, folds the Q31 normalisation in, so `tosa_shift = 31 + kea_shift` |
| double round | n/a | applied only when `shift > 31`, and its sign follows **`value`**, not `value * multiplier` |

ISA.md §10.1 defines `VQUANT` *by reference implementation* as `kea::keaRequantize`, and says in
so many words "Do not reimplement this. Call the header." **The simulator obeys that: `VQUANT`
calls `keaRequantize`.**

The frontend, meanwhile, is defined by `apply_scale_32`, and the vector file says explicitly:
"Do not substitute one for the other."

Both are therefore in the simulator's test suite, and the relationship between them is *tested*,
not assumed (`sim/tests/test_apply_scale.cpp`):

1. **`applyScale32` (in `sim/include/kea/sim/quant.h`) reproduces all 18,506 published vectors
   exactly** — including the 8,436 with `shift > 31`, the 9,256 with `double_round`, the 4,097
   negative-`value` double-round cases, and the 3,051 cases that overflow int32 before the
   wrapping narrow. `applyScale32` is the *only* piece of quantization arithmetic in `sim/`
   that is not a call into `isa.h`, and it exists precisely because it is a different algorithm
   from the one `isa.h` provides.
2. **On the domain KEA actually uses** — `mult ∈ [2^30, 2^31)`, `kea_shift ∈ [0, 31]`, i.e.
   `tosa_shift ∈ [31, 62]`, `value` uniform over int32 — the two agree on **0 of 400,000**
   sampled cases, reproducing the vector file's own 6.4M-case brute-force finding. The same
   sweep with `double_round = false` diverges on 6,108 cases, which proves the first assertion
   is not vacuous.

If that equivalence ever breaks, `sim.test_apply_scale` fails and says so. It is an empirical
result on a large sample, not a proof, and it says nothing about `shift < 31`.

---

## 7. Ambiguities in the frozen documents, and how they were resolved

Each of these is a place where ISA.md/MICROARCH.md were silent, or where two normative passages
disagreed. The choice made is stated, along with why and what it costs.

### 1. Same-cycle `SIGNAL` and `WAIT` ordering

ISA.md §5.3 says: *"A `SIGNAL` and a `WAIT` on the same event in the same cycle: the `SIGNAL`'s
increment is applied first, then waiters are evaluated."* MICROARCH.md §8's pseudocode does a
**single ascending-unit-id pass** in which `SIGNAL`s and `WAIT`s are interleaved — so a waiter on
unit 0 would miss a same-cycle signal from unit 4 and be released one cycle later.

**Resolved in favour of ISA.md** (it is the more specific and more normative statement). The
cycle is split into three passes: all eligible `SIGNAL`s, then all `WAIT`s in ascending unit id,
then everything else. Each unit still retires or starts at most one instruction per cycle, so
the split cannot let a unit run ahead. Cost: up to one cycle earlier release than MICROARCH.md's
literal pseudocode, only when the signalling unit's id exceeds the waiting unit's.

### 2. Where in the cycle the DMA engines advance

MICROARCH.md §8 puts DMA arbitration at step 1, before instruction starts. With that placement a
descriptor starting at cycle *t* gets its first advance at *t+1* and therefore occupies
`overhead + data + 1` cycles — one more than `keaDmaOccupancy()`, which is `static_assert`ed in
`hw_config.h` and is what the compiler costs against.

**Resolved by advancing the engines at the end of the cycle** (phase E), so the start cycle is
also the first cycle of the overhead phase and uncontended occupancy is exactly
`keaDmaOccupancy()`. Asserted by `test_timing.cpp::testDmaOccupancyExact`.

### 3. Whether a freed queue slot is usable in the same cycle

MICROARCH.md §8's step 3 is annotated *"(after execution, so a slot freed this cycle is usable
next cycle)"*, but the code it annotates places dispatch **after** the execution step, which
makes the slot usable **this** cycle. The comment and the code contradict each other.

**Resolved in favour of the code.** Dispatch happens after starts, so a slot freed in phase C is
available in phase D of the same cycle. Impact is one cycle per queue-full episode.

### 4. When a `TRACE REGION_END` closes its region

MICROARCH.md §7 gives `TRACE` occupancy 0 and latency 0, and ISA.md §5.4 calls it
"architecturally a `NOP` with zero occupancy". Taken literally, a `REGION_END` reaches the queue
head and retires **one cycle after the last `MATMUL` of the region merely started** — so a region
bracketing eighteen `MATMUL`s would stop counting 68+32 cycles before the array actually
finished, and every region in the report would be short by up to a pipeline depth. `TRACE`'s
entire documented purpose is per-region accounting, so that is not a usable reading.

**Resolved without perturbing timing:** `REGION_END` retires immediately, exactly as specified,
and costs nothing. But the region is only *marked* closing; it keeps accumulating until the
issuing unit's pipeline has drained, and its `end` cycle is that drain point. Nothing about the
schedule changes — no instruction is delayed — only the accounting window. (`REGION_BEGIN` opens
at its retire cycle with no such adjustment, so back-to-back regions on one unit can overlap by
up to a pipeline depth. That is the correct attribution: the tail of layer *n* really is
happening while layer *n+1* starts.)

### 5. int32 accumulator overflow in `MATMUL` and `DWCONV`

ISA.md is silent. `ACC` is int32 and a long reduction chain can exceed it.

**Resolved as wrapping two's complement**, implemented with explicit `uint32_t` arithmetic so
there is no C++ signed-overflow UB in the simulator. Saturation was rejected because nothing in
the ISA describes saturation on the accumulator path, because `keaSrdhm`'s saturation (which the
ISA *does* specify) is downstream in `VQUANT`, and because a wrapping accumulator is what a
plain systolic array does. **This is a real gap in the frozen ISA and should be stated
normatively in a future revision.**

### 6. Reset state of the MXU weight banks

ISA.md §2.3 says the weight banks are undefined at reset and "simulators should poison these
regions and trap on reads of poison". But a `MATMUL` does not read a bank byte-wise — it reads
the whole resident tile — and a program that runs `MATMUL` before any `LOAD_W` has a much more
basic problem.

**Resolved as: warn once per bank, simulate as zero.** A hard fault was rejected because it
would make the diagnostic worse than the warning, not better; the warning names the bank and the
pc. `k_rows`/`n_cols` for an unloaded bank are 0, so such a `MATMUL` contributes 0 useful MACs
and its padding efficiency correctly reads 0%.

### 7. Which resource a `NOP` occupies on the MXU

ISA.md §5.1 says a `NOP` "occupies that unit's pipe", but the MXU has four resources.

**Resolved as `ARRAY`.** `WPORT` and the banks stay free, which keeps `NOP` consistent with the
`LOAD_W`/`MATMUL` split and makes `NOP` usable as an array-time spacer, which is what it is for.
A `NOP` on `CTRL` (which has no queue) stalls the dispatcher for `cycles` cycles.

### 8. `CTRL`-targeted `WAIT`

ISA.md permits `WAIT` to name any unit including `CTRL`, and says `CTRL` instructions retire at
dispatch. A blocked `CTRL` `WAIT` therefore stalls the *dispatcher*, starving every unit.

**Resolved literally**: the dispatcher does not advance `pc` until the predicate holds, and the
deadlock detector reports the dispatcher's pc alongside any blocked queue heads. It is legal and
occasionally useful (a whole-machine barrier), but it is a foot-gun; the pattern is tested in
`test_sync.cpp::testCtrlWaitBlocksTheDispatcher`.

### 9. Unused DRAM grant is not redistributed

MICROARCH.md §6.3 defines the split over "engines currently in their data phase" but says
nothing about an engine whose remaining byte count is smaller than its grant.

**Resolved as: no redistribution within a cycle.** The other engine does not pick up the
leftover until the next cycle, when the finished engine is no longer in its data phase. The
error is bounded by 8 bytes per descriptor-completion.

### 10. `keaValidate` bounds only the base address

The frozen validator checks that base addresses are in range, but a strided walk can run off the
end of a space at runtime. ISA.md §11.2 says "every address inside its space" without saying
when that is checked.

**Resolved as a runtime fault** on the bounding interval of every instruction's access pattern,
reported with pc, space and interval. This is strictly stronger than load-time validation and
catches the stride mistakes that matter.

### 11. `VADD`'s intermediate sum

ISA.md §10.2 specifies `o = keaRdpot(keaSrdhm(xa + xb, o_mult), o_shift) + o_zp`, and `xa + xb`
is a plain int32 addition in `isa.h`. For the parameter ranges the frontend produces
(`|xa|, |xb| ≲ 1.3e8`) it cannot overflow, but the ISA does not *bound* the parameters, so a
hostile `KeaAddParam` could overflow it. **The simulator calls the header and inherits its
behaviour**, as instructed. A future revision should either bound `a_mult`/`a_shift` normatively
or specify saturation here.

### 12. Region attribution across units

ISA.md §5.4 says regions nest per `(unit, tag)` but does not say whose work a region owns.

**Resolved as: a region owns all activity in its cycle window, from every unit.** The
alternative — attributing only the opening unit's work — would make a layer's region exclude the
DMA traffic issued to feed it, which defeats the purpose. Nested regions all receive the same
contributions, so summing sibling regions is meaningful but summing a parent and its children is
double counting.

---

## 8. Performance

`sim/tests/test_bench.cpp` builds a MobileNetV2-scale synthetic program — 45 layers shaped like
MICROARCH.md §9.3(c)'s mid-network pointwise block (24 reduction tiles × 6 output tiles,
`M = 196`), plus a depthwise layer, `VQUANT`s and the DMA traffic to feed them:

| | |
|---|---|
| Instructions | 13,864 |
| Simulated cycles | 1,418,244 |
| Useful MACs actually multiplied out | 325,140,480 |
| Wallclock (RelWithDebInfo, Apple M-series, one core) | **0.14–0.16 s** |
| **Rate** | **≈ 9–10 M simulated cycles/s**, ≈ 2.2 G MAC/s functional throughput |

So a full MobileNetV2 inference simulates in well under a second, bit-exactly, with every
counter live. The functional model dominates: the timing loop alone runs far faster, but there
is no mode that skips the arithmetic, because the point of this simulator is that its numbers
are real.

---

## 9. Using it

```
kea-sim <program.keaf|program.kasm> [options]

  --map FILE             model map JSON (required for .kasm input)
  --const FILE           raw constant blob for .kasm input
  --input NAME=FILE      stage FILE into the DRAM arena at tensor NAME
  --output NAME=FILE     write tensor NAME back out after the run
  --stats-json FILE      machine-readable statistics
  --trace[=FILE]         per-instruction issue/start/retire trace
  --max-cycles N         abort after N simulated cycles
  --quiet                suppress the human-readable report
  --strict-poison        reads of never-written scratchpad are fatal
  --strict-hazards       cross-unit unsynchronized reads are fatal
  --no-rule-d            do not statically enforce Rule D (ISA.md §5.5)
  --list-tensors         print the tensor table and exit
```

Loading `.keaf` and `.kasm` is `runtime/`'s job (`kea::rt::loadProgramFile`); `kea-sim` only ever
sees a `kea::KeaProgram`. `tools/kea-sim/CMakeLists.txt` gates on `if(TARGET kea_runtime)`, so
the tool still builds in a configuration without `runtime/` — its loader just explains why it
cannot read an artifact. Outputs are written even on a failed run so a partial result can be
inspected.

Exit status is 0 on success, the `HALT` exit code (masked to 7 bits) if the program halted
non-zero, and 1 for any simulator-detected error.

To drive the simulator from C++ without a file, use `kea::sim::ProgramBuilder` (a thin wrapper
over the `keaMake*` constructors in `isa.h`) — that is how all of `sim/tests/` works.

### Tests

| Test | What it proves |
|------|----------------|
| `sim.test_functional` | every opcode against hand-computed or independently-written expected values, including int4 pack/unpack, `LOAD_W` tail-tile zeroing, 2D `MATMUL` tiling with independent strides, 3×3/5×5 × stride 1/2 `DWCONV`, int4 `VQUANT` output, in-place `VADD`, negative and broadcast DMA strides, ACC word-addressing faults, poison |
| `sim.test_conv3x3` | **the whole of ISA.md §8.5** built by hand as a real program — halo fill, DMA, 18 `LOAD_W`/`MATMUL` pairs, `VQUANT`, `DMA_ST` — checked against an independent reference convolution, under `--strict-poison` and `--strict-hazards`, with the per-tap `a_addr` values and the 294,912-useful-MAC / 1224-cycle cost model asserted against the document |
| `sim.test_timing` | bank overlap, double buffering, DRAM contention, queue backpressure, occupancy == `hw_config.h` |
| `sim.test_sync` | Rule D (both shapes), deadlock (both shapes), `SIGNAL` drain, `thr == 0`, event overflow, `CTRL` `WAIT`, determinism, hazard detection |
| `sim.test_apply_scale` | all 18,506 published vectors, plus the gemmlowp/TOSA equivalence sweep |
| `sim.test_bench` | MobileNetV2-scale program runs, and how fast |
| `sim.kea-sim.e2e` | the tool end to end: `.kasm` → runtime loader → simulation → report + JSON + trace |
