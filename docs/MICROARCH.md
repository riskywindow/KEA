# KEA-1 Microarchitecture and Timing Model

**Status: FROZEN.** Normative machine-readable form:
[`include/kea/hw_config.h`](../include/kea/hw_config.h).

This document specifies the **cycle-approximate** model that `kea-sim` implements and that
`keac`'s scheduler costs against. It is not an RTL specification — KEA-1 has no RTL — but every
number is chosen to be plausible for a 1 GHz edge NPU and, more importantly, the numbers are
*internally consistent* so that the roofline analysis in §9 means something.

Companion documents: [ISA.md](ISA.md), [ARTIFACT_FORMAT.md](ARTIFACT_FORMAT.md).

---

## 1. Top-level organization

```
        ┌───────────────────────────────────────────────────────────────┐
IMEM ──▶│ FETCH/DISPATCH   1 instr/cycle, strictly in program order     │
1 MiB   │ route by instr.unit; stall iff target queue full              │
        └──┬────────┬────────┬──────────┬──────────┬────────────────────┘
           │        │        │          │          │
        ┌──▼──┐  ┌──▼──┐  ┌──▼──┐   ┌───▼───┐  ┌───▼───┐
        │ Q0  │  │ Q1  │  │ Q2  │   │  Q3   │  │  Q4   │   depth 16, in-order
        └──┬──┘  └──┬──┘  └──┬──┘   └───┬───┘  └───┬───┘
        ┌──▼──────┐ │        │          │          │
        │ MXU     │ │        │          │          │
        │ ARRAY   │ │        │          │          │
        │ WPORT   │ │        │          │          │
        │ BANK0/1 │ │        │          │          │
        └─────────┘ │        │          │          │
                 ┌──▼──┐  ┌──▼──┐   ┌───▼───┐  ┌───▼───┐
                 │ DWU │  │ VPU │   │ DMA0  │  │ DMA1  │
                 └─────┘  └─────┘   └───┬───┘  └───┬───┘
                                        └────┬─────┘
                                    ┌────────▼────────┐
                                    │ DRAM port 16B/cy│
                                    └─────────────────┘
```

Five concurrent units, one queue each, one shared 16 B/cycle DRAM port, 32 global event
counters, three scratchpads. No caches, no coherence, no reorder, no speculation.

### 1.1 Fetch and dispatch

One instruction leaves IMEM per cycle and is pushed into `instr.unit`'s queue. Dispatch is
**strictly in program order** and the **only** stall condition is a full target queue.

That single rule is doing a lot of work. It gives you:

- **Program order is preserved per unit, for free.** Two `MATMUL`s reach the MXU in the order the
  compiler wrote them, so intra-unit dependencies never need synchronization.
- **The only implicit backpressure in the machine.** Everything else is explicit `SIGNAL`/`WAIT`.
- **One deadlock hazard**, addressed by Rule D in [ISA.md §5.5](ISA.md#55-rule-d--the-dispatch-order-rule-read-this-before-writing-a-scheduler).
  A blocked unit fills its queue, and the next instruction for that unit then wedges the
  dispatcher, starving every other unit. The compiler must guarantee that every `WAIT`'s
  producing `SIGNAL`s appear earlier in the stream.

CTRL-targeted instructions (`HALT`, and `NOP`/`SIGNAL`/`WAIT`/`TRACE` with `unit = CTRL`) retire
at dispatch and never enter a queue.

**Why a queue depth of 16?** A DMA descriptor takes 10³–10⁴ cycles; a `MATMUL` takes 10²–10³.
Sixteen slots is enough for the dispatcher to run several hundred cycles ahead of the slowest
unit — plenty to hide dispatch bandwidth — while staying a plausible flop-count for a real
queue. Making it much deeper would weaken Rule D's practical importance and hide scheduling
bugs the project exists to expose.

**Why 1 instr/cycle?** A 32-byte instruction at 1 GHz is 32 GB/s of IMEM read bandwidth, which
is already generous for an SRAM macro. A MobileNetV2 program is ~15–25k instructions against
~1.3M execution cycles, so dispatch bandwidth is ~2% utilized and never the bottleneck.

---

## 2. Timing vocabulary

Three distinct quantities, used with these exact meanings everywhere:

| Term          | Meaning                                                                        |
|---------------|--------------------------------------------------------------------------------|
| **issue cost** | Cycles before the same unit may *start* its next instruction. Always **1** (`KEA_ISSUE_COST_CYCLES`). |
| **occupancy**  | Cycles the instruction holds its functional **resource**. The unit's next instruction may start while a previous one still occupies a *different* resource, but never the same one. |
| **latency**    | Cycles from start-of-execution until the instruction's last architectural write is visible. Only observable through `SIGNAL`. |

The separation of *unit* from *resource* is what makes MXU weight double-buffering work: `LOAD_W`
holds `WPORT`, `MATMUL` holds `ARRAY`, and both hold the weight bank they name. A `LOAD_W` into
bank 1 issued right after a `MATMUL` on bank 0 starts one cycle later and runs concurrently.

Resources, by unit:

| Unit  | Resources                                  |
|-------|--------------------------------------------|
| MXU   | `ARRAY`, `WPORT`, `BANK0`, `BANK1`         |
| DWU   | `DWPIPE`                                   |
| VPU   | `VPIPE`                                    |
| DMA0  | `ENGINE0` + a share of the global `DRAM_PORT` |
| DMA1  | `ENGINE1` + a share of the global `DRAM_PORT` |

---

## 3. The MXU

### 3.1 Structure

A **weight-stationary 16×16 systolic array**, 256 PEs, one int8 MAC per PE per cycle.
`K = 16` rows (reduction), `N = 16` columns (output channels). Each PE holds **two** weight
registers (`BANK0`, `BANK1`).

- Activations enter from the left, one 16-element row per cycle, skewed across columns.
- Partial sums accumulate down the 16 rows.
- 16 int32 results leave the bottom per cycle and are written to `ACC`.

`ARRAY` writes 16 int32/cycle = 64 B/cycle into ACC (128 B/cycle in int4).

### 3.2 int4

int4 gives each PE two MACs/cycle. KEA-1 spends that on **two activation rows per cycle**, not
on a 32-deep reduction. Rationale: doubling `K` would change the tile shape, forcing the
compiler to maintain two different weight layouts and two different channel-padding rules for
what is a rarely used mode. Doubling row throughput keeps every address computation identical —
only `m_inner` pairing changes — and delivers the same 2× peak. Rows are paired along the
**inner** dimension, so an odd `m_inner` wastes half of the final cycle.

### 3.3 Costs

| Instruction | Resources held        | Occupancy                                        | Latency        |
|-------------|-----------------------|--------------------------------------------------|----------------|
| `LOAD_W`    | `WPORT`, `BANK[b]`    | `2 + ceil(k_rows * row_bytes / 32)`               | = occupancy    |
| `MATMUL`    | `ARRAY`, `BANK[b]`    | int8: `4 + m_outer * m_inner`<br>int4: `4 + m_outer * ceil(m_inner/2)` | occupancy + 32 |

`row_bytes` = 16 (int8) or 8 (int4). Normative implementations: `kea::keaLoadWOccupancy`,
`kea::keaMatmulOccupancy`, `kea::keaMatmulLatency`.

**Rationale.**

- `KEA_MXU_PIPELINE_DEPTH = 32`: activation skew across 16 columns plus partial-sum propagation
  down 16 rows. First result emerges after ~2·16 cycles; the array drains 32 cycles after the
  last row enters.
- `KEA_MXU_MATMUL_ISSUE_OVERHEAD = 4`: reprogramming the two-level activation address generator
  and switching the bank mux. Full-tile `MATMUL`s (M ≥ 64) amortize this below 6%.
- `KEA_MXU_LOADW_BYTES_PER_CYCLE = 32`: the SPM_W read port is 32 B wide, i.e. two array rows
  per cycle. A full int8 tile is 256 B → 8 cycles + 2 setup = **10 cycles**, which vanishes
  entirely under any `MATMUL` with `M ≥ 10` on the other bank.
- `KEA_MXU_MAX_ROWS = 2048`: `M × 16` int32 words must fit in the 32768-word ACC. This is an
  architectural bound, not a timing choice, and it is why large 1×1 convolutions are tiled into
  row bands.

### 3.4 Utilization arithmetic

Useful MACs per `MATMUL` = `M × k_rows × n_cols`, where `k_rows`/`n_cols` come from the resident
weight tile. Issued MACs = `M × 256`. So:

```
MXU utilization = (k_rows/16) × (n_cols/16) × M / occupancy(M)
```

Two lessons the compiler must internalize:

1. **Channel padding is expensive.** A layer with `IC = 3` runs at 3/16 = 19% unless the
   compiler uses the channel-packed lowering of [ISA.md §8.6](ISA.md#86-channel-packed-lowering-for-the-first-layer)
   (which reaches 9/16 = 56%).
2. **Small tiles are expensive.** `M = 8` costs `4 + 8 = 12` cycles for 8 rows of work: 67%.
   `M = 196` costs `200` for `196`: 98%.

---

## 4. The DWU

### 4.1 Why it exists

Depthwise convolution has no reduction across channels. Mapping it onto the MXU would use one of
16 array columns — 6% utilization — and MobileNetV2 is roughly half depthwise layers by count.
A separate unit that is *cheap* (128 MACs, half the MXU's PE count at a quarter of the MXU's
wiring complexity) and runs **concurrently with the MXU** is the right answer: while the DWU
grinds through a depthwise layer, the MXU can already be working the next pointwise layer if the
compiler software-pipelines across blocks.

### 4.2 Structure and cost

16 lanes, one per output channel, 8 MACs/lane/cycle. Each lane owns a small line buffer holding
the `K` input rows it needs, so the sliding window costs no re-fetch.

| Instruction | Resource | Occupancy                                                        | Latency        |
|-------------|----------|------------------------------------------------------------------|----------------|
| `DWCONV`    | `DWPIPE` | `8 + ceil(channels/16) * ceil(out_h * out_w * KH * KW / 8)`       | occupancy + 12 |

Normative: `kea::keaDwconvOccupancy`.

**Rationale.**

- Taps are consumed 8 at a time and the tap stream is **pipelined across output pixels**, so a
  3×3 kernel (9 taps) does *not* round up to 2 cycles per pixel. Over a whole tile the tap count
  rounds up once, not once per pixel. This is realistic — the window slides, it does not restart
  — and it is the difference between 56% and ~100% efficiency on 3×3.
- `KEA_DWU_ISSUE_OVERHEAD = 8`: priming the line buffers with the first `K` rows.
- `KEA_DWU_PIPELINE_DEPTH = 12`: multiplier stage, 8-input adder tree (3 levels), accumulator
  read-modify-write, ACC write.

Worked: 3×3 stride-1 on `112×112×32`:
`8 + 2 × ceil(112·112·9 / 8) = 8 + 2 × 14112 = 28,232` cycles for `3,612,672` MACs.
Ideal at 128 MAC/cycle is `28,224`. **Efficiency 99.97%.**

For 5×5 (25 taps) the rounding is likewise amortized. Efficiency is essentially `1 - 8/cycles`.

---

## 5. The VPU

16 int8 lanes. One element per lane per cycle for arithmetic; 32 B/cycle for pure byte movement.

| Instruction | Resource | Occupancy                                                | Latency       |
|-------------|----------|----------------------------------------------------------|---------------|
| `VQUANT`    | `VPIPE`  | `4 + ceil(num_pixels * channels / 16)`                   | occupancy + 8 |
| `VADD`      | `VPIPE`  | `4 + ceil(num_elems / 16)`                               | occupancy + 8 |
| `VPOOL`     | `VPIPE`  | `4 + ceil(out_h * out_w * channels * kh * kw / 16)`      | occupancy + 8 |
| `VCOPY`     | `VPIPE`  | `4 + rows * ceil(row_bytes / 32)`                        | occupancy + 8 |

Normative: `kea::keaVquantOccupancy`, `keaVaddOccupancy`, `keaVpoolOccupancy`,
`keaVcopyOccupancy`.

**Rationale.**

- `KEA_VPU_PIPELINE_DEPTH = 8`: the requantize path is a 32×32 → 64 multiply (`keaSrdhm`), a
  rounding right shift (`keaRdpot`), a zero-point add, and a clamp. Eight stages is comfortable.
- 16 lanes, not 32: the VPU consumes ACC at 16 int32/cycle, exactly matching the MXU's ACC
  *write* rate. A wider VPU could not be fed by the array anyway, and it would idle.
- `VCOPY` at 32 B/cycle: no arithmetic, so it is limited only by the SPM port width. Note the
  per-row `ceil`, which makes narrow-row halo fills relatively costly — fill a padded buffer in
  one big `rows=1` `VCOPY` when you can.

**The VPU is a real bottleneck candidate.** A `VQUANT` over the output of a large pointwise conv
processes `num_pixels × channels` elements at 16/cycle, while the MXU produced those same
elements at `256 / (k_rows·n_cols/16)` MAC/cycle. For a 1×1 conv with `IC = 384`, the MXU spends
`24 × M` cycles producing `M × 16` outputs, and the VPU spends `M × 16 / 16 = M` cycles
consuming them — 24× headroom. But for `IC = 16` the ratio is 1:1 and the VPU is exactly as busy
as the MXU. `keac` must schedule `VQUANT` against MXU work, not after it.

---

## 6. The DMA engines and the memory system

### 6.1 Two engines, one port

`DMA0` and `DMA1` are independent descriptor processors with independent queues, but they share
one **16 B/cycle DRAM port** (`KEA_DRAM_BYTES_PER_CYCLE`). Each engine can individually saturate
the port (`KEA_DMA_ENGINE_BYTES_PER_CYCLE = 16`).

The two engines exist **for scheduling, not for bandwidth**. Their purpose is to let the
compiler have a load in flight for tile *n+1* while a store drains for tile *n−1*, without one
serializing behind the other in a single queue. Aggregate bandwidth is unchanged.

### 6.2 Descriptor cost

A descriptor executes in two phases:

1. **Overhead phase** — `KEA_DMA_DESC_OVERHEAD_CYCLES (16) + n1*n2 * KEA_DMA_ROW_OVERHEAD_CYCLES (2)`
   cycles. Consumes no DRAM bandwidth. Models descriptor decode, DRAM row activation, and one
   address turnaround per contiguous run.
2. **Data phase** — `len0 * n1 * n2` bytes moved at the granted rate.

Uncontended occupancy (`kea::keaDmaOccupancy`, what the compiler costs against):

```
16 + n1*n2*2 + ceil(len0*n1*n2 / 16)
```

**Rationale.** `KEA_DMA_DESC_OVERHEAD_CYCLES = 16` is roughly a DRAM row activate. The
per-run cost of 2 cycles is the reason **`len0` matters enormously**: moving a `112×112×32`
feature map as `n1=112, n2=112, len0=32` costs `12544×2 = 25,088` overhead cycles on top of
`25,088` data cycles — a 2× penalty. Moving it as `n1=112, n2=1, len0=3584` (whole rows
contiguous) costs `224` overhead cycles. **Maximize `len0`.** This is a real property of real
DMA engines and the compiler's layout choices should respect it.

### 6.3 DRAM port arbitration (normative)

Cycle-by-cycle, over engines currently in their **data phase**:

- If one engine is in its data phase, it receives all 16 B that cycle.
- If both are, each receives `floor(16/2) = 8` B; any odd remainder goes to the lower unit id
  (`DMA0`).
- Engines in their overhead phase request nothing.

This is deterministic, which is required: two runs of `kea-sim` on the same artifact must
produce identical cycle counts.

`keaDmaOccupancy()` returns the **uncontended** figure. When both engines overlap, the simulator
stretches the data phase accordingly; the compiler's static cost model uses the uncontended
figure and should treat aggregate DRAM traffic as the real constraint (§9).

### 6.4 Scratchpad ports (modelling simplification — read this)

| Space   | Ports                                                                  |
|---------|------------------------------------------------------------------------|
| `SPM_A` | DMA read/write, MXU+DWU read, VPU read/write                            |
| `SPM_W` | DMA write, MXU weight read (32 B/cy), DWU+VPU read                       |
| `ACC`   | MXU+DWU write (64 B/cy), VPU read (64 B/cy)                              |

**The baseline KEA-1 timing model does not arbitrate scratchpad ports.** SPM_A and SPM_W are
assumed to be banked widely enough (16 banks each) that concurrent access by DMA, MXU/DWU, and
VPU never collides. This is the single biggest approximation in the model.

It is a defensible one — 16-way banked 256 KiB SRAM with three narrow clients really does have
low collision probability, and modelling it properly requires per-cycle bank-conflict tracking
that would slow the simulator by an order of magnitude for a few percent of accuracy. But it
means the model is **optimistic on layers where DMA, MXU and VPU are all saturating SPM_A at
once**. `kea-sim` may offer a `--strict-spm` mode that adds bank-conflict stalls; results
reported in the roofline analysis must state which mode was used.

`ACC` needs no arbitration: MXU/DWU only write it, VPU only reads it, and the two directions are
separate ports by construction.

---

## 7. Complete instruction timing table

`I` = issue cost, `O` = occupancy, `L` = latency. All in cycles.

| Opcode   | Unit    | Resource(s)          | I | O                                                        | L         |
|----------|---------|----------------------|---|----------------------------------------------------------|-----------|
| `NOP`    | any     | unit pipe            | 1 | `cycles` field                                            | `O`       |
| `HALT`   | CTRL    | —                    | 1 | 0                                                         | 0         |
| `SIGNAL` | any     | unit pipe (drain)    | 1 | 1                                                         | 1, **after** all prior instructions of that unit have retired |
| `WAIT`   | any     | —                    | 1 | 1 (plus unbounded blocking)                               | 1         |
| `TRACE`  | any     | —                    | 1 | 0                                                         | 0         |
| `DMA_LD` | DMA0/1  | `ENGINEn`, `DRAM_PORT` | 1 | `16 + n1*n2*2 + ceil(bytes/16)` (uncontended)            | `O`       |
| `DMA_ST` | DMA0/1  | `ENGINEn`, `DRAM_PORT` | 1 | same                                                     | `O`       |
| `LOAD_W` | MXU     | `WPORT`, `BANK[b]`   | 1 | `2 + ceil(k_rows*row_bytes/32)`                           | `O`       |
| `MATMUL` | MXU     | `ARRAY`, `BANK[b]`   | 1 | `4 + m_outer * (int4 ? ceil(m_inner/2) : m_inner)`        | `O + 32`  |
| `DWCONV` | DWU     | `DWPIPE`             | 1 | `8 + ceil(C/16) * ceil(out_h*out_w*KH*KW/8)`              | `O + 12`  |
| `VQUANT` | VPU     | `VPIPE`              | 1 | `4 + ceil(num_pixels*channels/16)`                        | `O + 8`   |
| `VADD`   | VPU     | `VPIPE`              | 1 | `4 + ceil(num_elems/16)`                                  | `O + 8`   |
| `VPOOL`  | VPU     | `VPIPE`              | 1 | `4 + ceil(out_h*out_w*C*kh*kw/16)`                        | `O + 8`   |
| `VCOPY`  | VPU     | `VPIPE`              | 1 | `4 + rows*ceil(row_bytes/32)`                             | `O + 8`   |

`SIGNAL`'s drain semantics are the reason latency exists at all in this model. A `SIGNAL` issued
immediately after a `MATMUL` cannot retire until `O + 32` cycles after that `MATMUL` started.
Two adjacent `SIGNAL`s pay the drain once.

---

## 8. Reference simulator algorithm

`kea-sim` is a cycle-stepped model. This is the normative structure; any implementation that
produces the same cycle counts is acceptable, but do not invent different formulas.

```
state:
  pc
  queue[5]                      # deque of instructions, cap KEA_QUEUE_DEPTH
  resource_free[R]              # cycle at which each resource becomes free
  unit_next_issue[5]            # earliest cycle the unit may start its next instruction
  unit_retire_all[5]            # cycle by which every instruction the unit has started
                                # will have fully retired  (running max of start+latency)
  event[32]                     # uint32
  dma_state[2]                  # phase (idle|overhead|data), bytes_remaining, ...
  halted

each cycle t:
  # 1. Retire / advance DMA data phases with arbitration
  active = { e in {0,1} : dma_state[e].phase == data }
  share  = |active| ? KEA_DRAM_BYTES_PER_CYCLE / |active| : 0
  for e in active in ascending unit order:
      grant = share + (remainder to the lowest id)
      dma_state[e].bytes_remaining -= min(grant, bytes_remaining)

  # 2. Start instructions, per unit, ascending unit id (determinism)
  for u in 0..4:
      if queue[u] empty: continue
      ins = queue[u].front()
      if t < unit_next_issue[u]: continue

      if ins is WAIT:
          if event[ins.event] >= ins.value:
              event[ins.event] -= ins.value          # ascending-unit-id arbitration
              retire; unit_next_issue[u] = t + 1
          continue                                   # else stay blocked

      if ins is SIGNAL:
          if t < unit_retire_all[u]: continue        # local drain barrier
          event[ins.event] += ins.value
          retire; unit_next_issue[u] = t + 1
          continue

      if any resource of ins is busy at t: continue
      start ins at t
      for each resource r of ins: resource_free[r] = t + occupancy(ins)
      unit_next_issue[u] = t + KEA_ISSUE_COST_CYCLES
      unit_retire_all[u] = max(unit_retire_all[u], t + latency(ins))
      pop queue[u]

  # 3. Dispatch (after execution, so a slot freed this cycle is usable next cycle)
  if not halted and dispatched_this_cycle < KEA_DISPATCH_PER_CYCLE:
      ins = imem[pc]
      if ins.unit == CTRL:
          execute immediately (HALT sets halted; TRACE emits; ...)
          pc += 1
      elif queue[ins.unit].size < KEA_QUEUE_DEPTH:
          queue[ins.unit].push(ins); pc += 1
      else:
          stall                                       # the only implicit backpressure

  # 4. Termination and error checks
  if halted and all queues empty and all units idle: done
  if no unit made progress and no dispatch is possible and some queue head is a
     blocked WAIT:  report DEADLOCK with the blocked (unit, pc, event, threshold) tuples
  if any event[e] > KEA_EVENT_MAX: report EVENT OVERFLOW
```

### 8.1 Determinism requirements

Two runs on the same artifact must produce byte-identical cycle counts and traces. That means:

- Units are always serviced in **ascending unit id** order within a cycle.
- Event decrements happen in that same order when several waiters are eligible.
- DRAM bandwidth remainders go to the lower unit id.
- No use of hash-map iteration order, wall-clock time, or unordered containers anywhere in the
  scheduling path.

### 8.2 Errors the simulator must detect and report

| Condition                                            | Report                                       |
|------------------------------------------------------|----------------------------------------------|
| Illegal opcode / failed `keaValidate`                | at load time, with pc                        |
| Address outside its space                            | pc, space, address, bound                    |
| Read of never-written scratchpad (poison)            | pc, space, address — a real compiler bug     |
| Deadlock                                             | every blocked `(unit, pc, event, threshold)` and every event value |
| Event counter over `KEA_EVENT_MAX`                   | pc of the offending `SIGNAL`                 |
| ACC written by MXU and DWU in overlapping regions without an intervening event | warning |
| Program without `HALT`, or instructions after `HALT` | at load time                                 |

The poison check is worth the effort: an unsynchronized `WAIT`-less read of a DMA target is the
single most likely compiler bug and it will otherwise silently produce plausible-looking wrong
numbers.

---

## 9. Peak numbers and roofline

### 9.1 Machine peaks

| Quantity                       | Value           | Derivation                       |
|--------------------------------|-----------------|----------------------------------|
| Clock                          | 1 GHz           | `KEA_CLOCK_HZ`                   |
| MXU int8                       | 256 MAC/cycle → **512 GOPS** | 256 PEs × 2 ops × 1 GHz |
| MXU int4                       | 512 MAC/cycle → **1024 GOPS (1 TOPS)** | 2 MAC/PE/cycle |
| DWU int8                       | 128 MAC/cycle → **256 GOPS**  | 16 lanes × 8 × 2 ops    |
| VPU                            | 16 elem/cycle → 16 Gelem/s     |                         |
| DRAM                           | 16 B/cycle → **16 GB/s**       | `KEA_DRAM_BYTES_PER_CYCLE` |
| **int8 ridge point**           | **32 ops/byte** | 512 GOPS ÷ 16 GB/s               |
| **int4 ridge point**           | **64 ops/byte** | 1024 GOPS ÷ 16 GB/s              |

An operation counts a multiply and an add separately: **1 MAC = 2 ops**. All of the above are
`constexpr` in `hw_config.h` and cross-checked by `static_assert`.

> **The number to remember: 32 ops/byte.** Any layer whose arithmetic intensity — useful ops
> divided by DRAM bytes it actually touches — is below 32 is memory-bound on KEA-1 and no amount
> of MXU tuning will help it. Any layer above 32 is compute-bound and the job is utilization.

### 9.2 How `kea-sim` computes the roofline

Per `TRACE`-delimited region, the simulator accumulates:

| Counter        | From                                                                       |
|----------------|----------------------------------------------------------------------------|
| `ops_useful`   | `MATMUL`: `2 · M · k_rows · n_cols` (from the resident weight tile)<br>`DWCONV`: `2 · out_h · out_w · C · KH · KW` |
| `ops_issued`   | `MATMUL`: `2 · M · 256`<br>`DWCONV`: `2 · out_h · out_w · ceil(C/16)·16 · KH · KW` |
| `dram_bytes`   | Σ `len0 · n1 · n2` over every `DMA_LD`/`DMA_ST` in the region               |
| `cycles`       | region end − region begin                                                   |
| `busy[unit]`   | Σ occupancy per unit                                                        |
| `stall[unit]`  | cycles blocked in `WAIT`, and cycles the dispatcher stalled on a full queue |

Then:

```
intensity  = ops_useful / dram_bytes                       [ops/byte]
attainable = min(PEAK_OPS, intensity × PEAK_DRAM_BW)       [ops/s]
achieved   = ops_useful / (cycles / CLOCK_HZ)              [ops/s]
efficiency = achieved / attainable
padding_loss = ops_useful / ops_issued
```

Reporting `ops_useful` **and** `ops_issued` separately is important. A layer can sit at 100% MXU
occupancy while doing 19% useful work because of channel padding; only the two counters together
tell you which problem you have.

### 9.3 Worked layer analyses (int8 MobileNetV2, 224×224)

**(a) First layer — 3×3×3 → 32, stride 2, out 112×112×32.**

| | naive (`IC` padded to 16) | channel-packed ([ISA.md §8.6](ISA.md#86-channel-packed-lowering-for-the-first-layer)) |
|---|---|---|
| taps per oc group | 9  | 3  |
| `k_rows`          | 3  | 9  |
| MXU cycles        | ~226,000 | ~75,400 |
| useful MACs       | 10,838,016 | 10,838,016 |
| ideal cycles      | 42,336 | 42,336 |
| **utilization**   | **19%** | **56%** |

The ceiling is `k_rows/16`. This layer is 3.6% of the model's MACs, so even 19% costs only
~180k cycles — but it is a good illustration of why `ops_issued` must be reported.

**(b) Depthwise 3×3 stride 1 on 112×112×32 (DWU).**

`28,232` cycles for `3,612,672` MACs → **99.97%** of DWU peak. DRAM traffic: 288 bytes of
weights. Arithmetic intensity is astronomically high *if the activations stay in SPM_A* — and
`112·112·32 = 401,408` bytes does **not** fit in a 256 KiB SPM_A, so this layer must be tiled
into row bands with double-buffered DMA. That is the whole reason the scratchpad-aware fusion
pass exists.

**(c) Pointwise 1×1, 14×14×384 → 14×14×96 (a mid-network block).**

- 24 reduction tiles × 6 output tiles = 144 `MATMUL`s, each `M = 196` → `144 × 200 = 28,800`
  cycles. Ideal `28,224`. **Utilization 98%.**
- DRAM: weights `384 × 96 = 36,864` B. Activations (`75,264` in, `18,816` out) fit in SPM_A and
  should never reach DRAM.
- Intensity = `2 × 7,225,344 / 36,864 = 392` ops/byte — **12× above the ridge, firmly compute
  bound.** This is the shape of most of MobileNetV2 and it is why KEA-1 is worth building.

**(d) The classifier — FC 1280 → 1001, batch 1.**

- Useful MACs `1,281,280`; ops `2,562,560`.
- Weights: `1,281,280` bytes — **1.28 MB**, larger than SPM_W, streamed from DRAM once.
- Intensity = `2,562,560 / 1,281,280 = 2.0` ops/byte — **16× below the ridge.**
- DMA cycles = `1,281,280 / 16 = 80,080`. MXU compute = `5,005` cycles.
- The layer takes **~80k cycles and the MXU is 6% busy.** No scheduling fixes this; it is
  bandwidth, by definition. (It is also the classic argument for weight compression, which
  KEA-1 does not have.)

Note also that batch-1 FC means `m_inner = 1`: 80 reduction tiles × 63 output tiles × `(4 + 1)`
= 25,200 cycles of pure `MATMUL` setup overhead. Still under the 80k DMA bound, so it is hidden
— but it shows why the 4-cycle setup constant matters for skinny GEMMs.

### 9.4 Whole-model estimate

| | |
|---|---|
| Total MACs (MobileNetV2 1.0/224) | ~300 M → ~600 MOPs |
| Of which depthwise (→ DWU)       | ~2% |
| MXU-eligible MACs                | ~294 M → **~1.15 M cycles** at peak |
| DWU MACs                         | ~6 M → ~47 k cycles, **concurrent** |
| Weights (int8)                   | ~3.4 MB |
| Unavoidable DRAM traffic         | ~3.6 MB (weights + input + a few large activation spills) |
| DRAM-bound floor                 | `3.6 MB / 16 B/cy` ≈ **225 k cycles** |
| Whole-model intensity            | `600 MOPs / 3.6 MB` ≈ **166 ops/byte — 5× above the ridge** |

So MobileNetV2 on KEA-1 is **compute bound overall**, with a hard floor around 1.15 M cycles and
a realistic target of **1.3–1.8 M cycles ≈ 1.3–1.8 ms**, i.e. **330–450 GOPS effective** and
roughly 550–750 inferences/s. The gap between 1.15 M and the realistic figure is exactly what
the compiler work is for: channel padding on narrow layers, `MATMUL` setup on skinny tiles,
`VQUANT` not overlapped with MXU work, `SIGNAL` pipeline drains, DMA descriptor overhead from
small `len0`, and imperfect double-buffering. Every one of those is visible as a distinct
counter in the `TRACE` region report, which is the point.

The one genuinely memory-bound piece is the final FC layer at ~80 k cycles (5% of runtime for
0.4% of the MACs).

---

## 10. Summary of modelling approximations

Stated honestly, because a cycle-approximate model that hides its assumptions is worthless:

| # | Approximation | Direction | Notes |
|---|---------------|-----------|-------|
| 1 | Scratchpad ports are never contended (§6.4) | **optimistic** | The largest one. Affects layers saturating DMA + MXU + VPU on SPM_A simultaneously. |
| 2 | DRAM is a flat 16 B/cycle pipe: no refresh, no bank conflicts, no read/write turnaround, no latency | **optimistic** | Real LPDDR would add 10–30% on scattered access. The per-run 2-cycle penalty is the only nod to this. |
| 3 | Misaligned DMA costs the same as aligned | **optimistic** | Real engines split misaligned bursts. Mitigated by requiring 16 B alignment by convention. |
| 4 | Instruction fetch is free and IMEM never misses | fair | 1 MiB dedicated SRAM, ~2% dispatch utilization. |
| 5 | `SIGNAL` fully drains its unit's pipeline | **pessimistic** | Real hardware could track per-instruction completion. Costs up to 32 cycles per MXU `SIGNAL`. |
| 6 | Fixed per-instruction setup constants (4/2/8/4) rather than modelled address-generator behaviour | fair | Chosen to be plausible; they matter only for small tiles. |
| 7 | DWU taps pipeline perfectly across pixels | slightly optimistic | Ignores line-buffer refill at row boundaries. |
| 8 | No clock gating, power, thermal, or DVFS modelling | n/a | Out of scope. |

Where the model is optimistic, treat simulated cycle counts as a **lower bound** and say so in
any reported result.
