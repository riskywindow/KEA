# Static memory planning — `-kea-alloc`

**Status: implemented.** Everything here is exercised by
`compiler/test/kea-alloc{,-invalid,-verify,-e2e}.mlir` and verified by
`bash scripts/build_compiler.sh`.

> **There is no allocator at run time, anywhere.** Every scratchpad and DRAM
> address in the final instruction stream is a compile-time constant chosen by
> this pass. The runtime's entire memory responsibility is to hand the device
> one contiguous arena of a size this pass computed, aligned to a boundary this
> pass computed. `runtime/tests/test_no_alloc.cpp` is the enforcement.

| | |
|---|---|
| Pass | `-kea-alloc`, `compiler/lib/Transforms/Alloc.cpp` |
| Level | L2 → L2 ([ADR-0002](adr/0002-two-level-kea-dialect.md)) |
| Input contract | [DIALECT_L2.md](DIALECT_L2.md) §4 — symbolic buffers and live ranges |
| Machine contract | `include/kea/hw_config.h`, [ISA.md](ISA.md) §7 and §11.1 (**frozen**) |
| Output contract | `KeafDramLayout` / `KeafSpmEntry` in `include/kea/keaf.h`, [ARTIFACT_FORMAT.md](ARTIFACT_FORMAT.md) §5, [ASSEMBLY.md](ASSEMBLY.md) §7.2 and §7.5 |
| Tests | `compiler/test/kea-alloc*.mlir` |

```
… ─-kea-tile──▶ L2 ─-kea-schedule──▶ L2 ─-kea-alloc──▶ L2 ─-kea-emit──▶ .kasm
                symbolic             queues +          absolute
                buffers              semaphores        addresses
```

---

## 1. What the pass is given, and what it decides

`-kea-tile` emits a `kea.alloc` for every buffer the program touches. Each one
carries a `name`, a `role`, an `alignment`, and — for on-chip buffers — a
`live = [first, last]` range. The result type carries the space and the size.
`-kea-schedule` then assigns queues and inserts `kea.signal` / `kea.wait`.

This pass adds exactly one thing: a **base address**.

```mlir
%t = kea.alloc {kea.addr = 2064 : i64, live = array<i64: 17, 26>,
                name = "block.0.atile", role = "scratch"} : !kea.buffer<272xi8, A>
```

### 1.1 Why the base goes on the `kea.alloc` and not into the ops

DIALECT_L2.md §1.1(a) defines an address in the instruction stream as a
`(buffer, displacement)` pair, and says `-kea-emit` writes

```
instr.X_addr = base(X) + X_addr
```

`kea.addr` is `base(X)`. The displacements are deliberately left alone.

That is not only the documented contract, it is forced. The Level 2 op verifiers
check that **every strided walk stays inside its buffer** — a `kea.mm` whose
activation walk runs past the end of its `!kea.buffer<272xi8, A>` is rejected.
Folding an absolute address into `a_addr` would make every walk in the program
look out of bounds, and would destroy the one check that catches a tiling bug.
Keeping the two separate means the bounds check stays meaningful right up to
`-kea-emit`, and the pass's output re-verifies as ordinary Level 2 IR.

`kea.addr` is a dialect-prefixed *discardable* attribute: `kea.alloc`'s ODS does
not declare it, and this pass does not own the op definition. Promoting it to a
declared optional attribute would be a small improvement and is noted in §9.

---

## 2. The live-range model

Two buffers may share storage **iff their live ranges are disjoint and they are
in the same address space**. Everything else in this pass is bookkeeping around
that one sentence.

A live range is a closed interval `[first, last]` of **positions in the
enclosing block's operation list**. Level 2 is a straight-line program in a
single block, so a position is a total order; the pass refuses a function with
more than one block containing a `kea.alloc`, because two ranges in different
blocks are not comparable and silently packing them against each other is
exactly the aliasing bug this pass exists to prevent.

### 2.1 On-chip buffers

`first` is the position of the `kea.alloc`; `last` is the position of its last
user in block order. This is DIALECT_L2.md §4.2's definition and
`mlir::kea::refreshLiveRanges()`'s.

**The stamped `live` attribute is not trusted.** ADR-0002's amendment puts
`-kea-schedule` *ahead* of this pass, and the scheduler inserts and moves ops —
which renumbers every block position a live range is expressed in. The pass
calls `refreshLiveRanges()` first and re-derives everything from the SSA
def-use chain, which is the authority. The attribute is a materialization of
that chain, not an independent source of truth.

### 2.2 Double buffering is not a concept in this pass

It is worth stating plainly, because it is the nicest property in the backend
and it would be easy to wreck by "helping".

`-kea-tile` emits a fresh `kea.alloc` per spatial tile, so consecutive tiles
have *disjoint* live ranges. An allocator running immediately after tiling would
correctly conclude they can share storage and give them the same address — at
which point double buffering is impossible, because prefetching tile *N+1*
would overwrite tile *N*.

Running the scheduler first dissolves this. Hoisting a DMA so it overlaps the
previous tile's compute **extends that buffer's live range** until it overlaps
its predecessor's. This pass then sees overlapping ranges and separates them for
the ordinary reason. The phrase "double buffering" appears nowhere in
`Alloc.cpp`, and should not.

**The obligation this places on `-kea-schedule`** is worth writing down, because
it is a real soundness condition and it is not checkable here: after scheduling,
block order must be a sound over-approximation of *temporal* liveness. If the
scheduler leaves two ops able to execute concurrently — different queues, no
semaphore between them — then the buffers they touch must have overlapping
block-order live ranges. If a scheduler ever ordered two accesses only by
program order while allowing them to run concurrently, this pass would be
entitled to alias the buffers and the result would be a data race that looks
like a numerical bug. §2.1's live ranges are only as good as that property.

### 2.3 DRAM buffers

DRAM live ranges are `[first use, last use]` — the position of the first
instruction that touches the buffer, not of the declaration.

This is both correct and necessary. A DRAM `kea.alloc` is a *symbol*, not a
store: it holds nothing until an instruction writes it. And `-kea-tile` hoists
every DRAM symbol to the top of the function, so keying on the declaration would
make every activation's range overlap every other one's and defeat the packing
entirely — which is precisely the case the packing exists for.

Only `role = "activation"` buffers are packed. A constant is live for the whole
program by definition, and the host binds an I/O tensor by name at an address it
is told once, at a time this pass cannot bound.

---

## 3. The algorithm, and what it costs

**Greedy by size, first fit by offset.** This is TensorFlow Lite Micro's
`GreedyMemoryPlanner`, and the shape of what production NPU compilers ship.

```
sort buffers by size descending
       (ties: longer-lived first, then by name, so the result is deterministic)

for each buffer b:
    conflicts = already-placed buffers whose live range overlaps b's
    sort conflicts by offset ascending
    candidate = 0
    for c in conflicts:
        if c.offset >= candidate + b.size:   break     # the gap below c fits
        candidate = max(candidate, alignUp(c.offset + c.size, b.align))
    if candidate + b.size > capacity:        fail loudly (§6)
    b.offset = candidate
```

### 3.1 Why this algorithm

The problem is **offline dynamic storage allocation**: place blocks of known
size and known lifetime in the smallest possible arena. It is NP-hard, so any
practical planner is an approximation and the honest question is *how good an
approximation*, not *is it optimal*.

Size-descending order is the part that does the work. The largest buffers are
the ones that cannot fit anywhere once the arena is chequered, so they go down
first into a clean arena; the small ones fill the gaps the big ones leave. The
inverse order — smallest first — reliably strands a large buffer behind a wall
of small ones. This is the same intuition as first-fit-decreasing for bin
packing, and it fails in the same rare ways.

The cost is `O(n²)` in the number of buffers per space, with a tiny constant.
That is nothing: the MobileNetV2 inverted residual has seven SPM_A buffers, and
even the full network has a few hundred. The pass does not need, and does not
have, a smarter data structure.

### 3.2 What it leaves on the table

The pass reports the gap itself rather than asking anyone to take its word.
For each space, `kea.occupancy` publishes:

| field | meaning |
|---|---|
| `peak` | the high-water mark, `max(offset + size)`. **This is what must fit.** |
| `maxlive` | the largest total size of simultaneously live buffers |
| `fragmentation` | `peak - maxlive` |
| `unpacked` | the sum of all sizes — what an allocator that never shared would need |
| `capacity` | from `hw_config.h` |

`maxlive` is a **lower bound on what any allocator could achieve**, optimal ones
included: those buffers all hold live data at the same instant, so no placement
can overlap them. Therefore `fragmentation` is an *upper bound on how far this
packer is from optimal*. When it is zero, the packing is provably optimal and no
cleverer algorithm could have done better. When it is not, the number is the
exact size of the prize.

That framing is why the greedy is defensible: it is not "probably fine", it
comes with a per-compile certificate of how close it landed.

### 3.3 Measured — the MobileNetV2 inverted residual

`kea-opt tests/mlir/tosa/mobilenet_block.mlir -tosa-to-kea -kea-fuse -kea-tile -kea-alloc`,
at the default `spm-reserve-factor = 2`:

| space | buffers | unpacked | **peak** | maxlive | fragmentation | capacity | used |
|---|---|---|---|---|---|---|---|
| SPM_A | 7 | 11,248 B | **5,280 B** | 5,280 B | **0** | 262,144 B | 2.0% |
| SPM_W | 7 | 2,292 B | **896 B** | 896 B | **0** | 262,144 B | 0.3% |
| ACC | 3 | 5,120 w | **2,048 w** | 2,048 w | **0** | 32,768 w | 6.3% |
| DRAM scratch | 2 | 3,072 B | **1,536 B** | 1,536 B | **0** | — | — |

And for the stride-2 variant in the same file:

| space | buffers | unpacked | **peak** | maxlive | fragmentation | capacity | used |
|---|---|---|---|---|---|---|---|
| SPM_A | 6 | 6,144 B | **3,136 B** | 3,136 B | **0** | 262,144 B | 1.2% |
| SPM_W | 6 | 2,272 B | **896 B** | 896 B | **0** | 262,144 B | 0.3% |
| ACC | 3 | 2,816 w | **2,048 w** | 2,048 w | **0** | 32,768 w | 6.3% |

Three things to read off these:

* **Fragmentation is zero everywhere, so the packing is optimal on this block.**
  Not "good" — optimal, in the sense that `peak == maxlive` and `maxlive` is a
  floor for every possible allocator.
* **Packing pays for itself.** SPM_A drops 11,248 → 5,280 (2.1×), ACC 5,120 →
  2,048 (2.5×), DRAM activations 3,072 → 1,536 (2.0× — two feature maps in one
  buffer). Those factors are the whole reason the pass exists.
* **Occupancy is low because the tensors are 8×8 and `-kea-tile` only spends
  half of each scratchpad.** These are not the numbers a 224×224 network
  produces; the *ratios* are the result, not the absolute bytes. DIALECT_L2.md
  §5.4 has the per-layer footprints for full MobileNetV2, where SPM_A tiles run
  to ~100 KB of the 131 KB budget and packing across layers is what keeps the
  whole program inside 256 KiB.

The zero-fragmentation result is not a coincidence of a small example. Greedy by
size hits `maxlive` exactly whenever the live ranges are "well nested" — which
is what a straight-line, layer-by-layer program produces, because a tile is born
and dies inside one layer. Fragmentation appears when lifetimes interleave
irregularly, which on this architecture means after aggressive software
pipelining. That is the case to watch when `-kea-schedule` gets more ambitious,
and the reported number is what will catch it.

---

## 4. Units and alignment

### 4.1 Units

SPM_A and SPM_W are **byte addressed**. ACC is addressed in **int32 words**, not
bytes (ISA.md §2.2). This is the mistake the ISA warns "trips everyone once".

Here it cannot, because the unit is carried by the type. An ACC buffer's element
type is `i32` and an SPM/DRAM buffer's is `i8`, `AllocOp::getExtent()` returns
the element count, and every size, offset, capacity and diagnostic in this pass
is in that same element. `KEA_ACC_WORDS` is 32768 *words*; a 20,000-word region
is over half of ACC even though it is only 80 KB. `kea-alloc-invalid.mlir`'s
`@acc_exhausted` is exactly that case: two 20,000-word buffers are rejected,
though 40,000 *bytes* would have fit trivially.

Every capacity comes from `hw_config.h`. Nothing is hardcoded.

### 4.2 Alignment

ISA.md §11.1 is the table. The op verifiers check the **displacement**; an
aligned displacement implies an aligned `base + displacement` only if the base
is aligned too, and the base is this pass's business.

Alignment is **derived from the ops that actually use the buffer**, not assumed
from the space. The `alignment` attribute on the `kea.alloc` is a floor, not the
answer:

| the buffer is used as | required base alignment | from |
|---|---|---|
| `kea.load_w`'s `$w`, int8 | 16 bytes | `KEA_ALIGN_LOADW_INT8` |
| `kea.load_w`'s `$w`, int4 | 8 bytes | `KEA_ALIGN_LOADW_INT4` |
| any ACC buffer | 16 words | `KEA_ALIGN_ACC_WORDS` |
| `kea.vquant`'s `$qparam`, `kea.vadd`'s `$param` | 4 bytes | `KEA_ALIGN_QPARAM` |
| `kea.dwconv`'s `$a` | 16 bytes (advisory) | `KEA_ALIGN_DWU` |
| a DRAM object | 16 bytes | `KEA_ALIGN_DMA_RECOMMENDED` |
| the DRAM arena itself | 64 bytes | `KEA_DRAM_BASE_ALIGN` |

A buffer used as both a weight tile and a qparam block gets the stronger of the
two. `compiler/test/kea-alloc.mlir`'s `@alignment_is_derived` is the proof: two
SPM_W buffers both declare `alignment = 4`, and the one fed to `LOAD_W` is
placed at 608 rather than 600 because the pass strengthened it to 16.

Then, after placement, **the absolute addresses are re-checked against the same
table**. This is not redundant with the derivation — it is the difference
between "we computed an alignment" and "the address that will be encoded into a
32-byte instruction is legal". `@misaligned_base` in `kea-alloc-verify.mlir`
shows both diagnostics: the base is rejected, *and* the `8 + 16 = 24` it would
have produced for a legal `w_addr = 16` is rejected.

---

## 5. The DRAM arena

The runtime allocates **one** contiguous arena and treats every DRAM address in
the instruction stream as an offset into it. `KeafDramLayout` (`include/kea/keaf.h`)
splits it into three disjoint regions, and this pass decides all seven numbers:

```
0                     const_bytes        io_bytes         scratch_bytes
├──────────────────────┼─────────────────┼────────────────────────────┤
   CONST                  I/O                ACTIVATION SCRATCH
   weights, qparams,      model inputs       inter-layer feature maps
   addparams              and outputs        — LIVE-RANGE PACKED
   staged before START    host-bound         written and read on chip
```

| region | roles | packed? | why |
|---|---|---|---|
| const | `weights`, `qparam`, `addparam` | no | live for the whole program |
| I/O | `input`, `output` | no | the host binds them by name and may read or write at any time relative to the program |
| scratch | `activation` | **yes** | an inter-layer feature map is dead the moment the next layer has read it |

Regions are laid out in that order, each starting on a `KEA_DRAM_BASE_ALIGN`
(64-byte) boundary, and `total_bytes` is the 64-byte-rounded end. Within the
const region, objects are placed **in program order**, which is also the order
`-kea-emit` must write the `CONST` blob — `const_bytes` includes the internal
alignment padding, and ASSEMBLY.md §7.2 requires `const_bytes` to equal the size
of the `--const` file exactly, so the padding bytes must be written.

Packing the scratch region is the point of it. On the inverted residual, layer
0's output and layer 1's output both land at arena offset 2816: one writes it,
the next reads it, and their ranges are disjoint. Two feature maps, one buffer.
On a fifty-layer network that is the difference between an arena holding a
couple of activation buffers and one holding fifty.

The arena for the whole inverted-residual block is **4,352 bytes**. That number,
and its alignment, is the entire memory interface between the compiler and the
runtime.

---

## 6. How failure is reported

A program that cannot fit gets a diagnostic that tells whoever has to fix the
tiling what to change. "Out of memory" alone is useless.

```
error: 'kea.alloc' op out of SPM_A: cannot place 'big.b' (200000 bytes,
       live [1, 4]) -- SPM_A holds 262144 bytes
 note: peak demand is 600000 bytes at block position 2, which is 337856 bytes
       over the 262144 bytes capacity. No allocator can fit this: those buffers
       hold live data at the same instant. Re-run -kea-tile with smaller tiles
 note: live at block position 2: 'big.a' 200000, 'big.b' 200000,
       'big.c' 200000 bytes
```

Four things, all of them actionable:

1. **The space**, named as the hardware names it, with its unit — `int32 words`
   for ACC, so nobody compares it against a byte figure.
2. **The buffer that could not be placed**, with its size and live range.
3. **The peak demand and the capacity**, *and which of the two failure modes
   this is*:
   * `maxlive > capacity` — the live data genuinely does not fit. **No allocator
     could place this**, so the fix is upstream: retile. Saying so stops the
     reader from hunting for a better packer.
   * `maxlive <= capacity` — the same data would fit under a better placement,
     so the shortfall is fragmentation this packer introduced. That is a bug
     report against `-kea-alloc`, not against the tiling, and it says so.
4. **The buffers alive at the peak**, largest first, capped at eight with a tail
   count. This is the list you need to know which layer to re-tile.

The peak is located by a difference-array sweep over block positions, so it is
the true maximum rather than the demand at the moment placement happened to
fail.

---

## 7. The verifier

After allocation the pass proves, from scratch, that

> no two buffers whose live ranges overlap were given overlapping storage.

This is a correctness property that is cheap to check and catastrophic to get
wrong. A silent aliasing bug does not look like an allocator bug: it looks like
a numerical bug inside a kernel, days from its cause, with nothing in the trace
pointing at memory planning. So the check runs on **every compile** (disable
with `-kea-alloc=verify-packing=false`, which exists only so the test suite can
prove the check fires).

It checks four things:

* **No aliasing.** Buffers are sorted by offset per space and each is compared
  only against those whose storage it actually reaches, so the quadratic term is
  the number of *storage overlaps*, not the number of buffer pairs.
* **Alignment**, of every base against the requirement derived in §4.2.
* **Capacity**, `offset + size <= capacity`, per space.
* **Absolute addresses**, `base + displacement`, against ISA.md §11.1 —
  independently of how the base was chosen.

`-kea-alloc=verify-only=true` runs only the verifier, over the `kea.addr`
already in the IR, allocating nothing. That makes it a checker for IR this pass
did not produce, and it is how a hand-corrupted allocation is tested
(`kea-alloc-verify.mlir`). `kea-alloc-e2e.mlir` pipes the finished MobileNetV2
block back through it, so the pipeline proves its own output free of aliasing.

---

## 8. Options and observability

```bash
kea-opt in.mlir -kea-alloc
kea-opt in.mlir -kea-alloc=verify-packing=false   # skip the overlap proof
kea-opt in.mlir -kea-alloc=verify-only=true       # check an existing placement
kea-opt in.mlir -kea-alloc=report-map=false       # omit the three attributes
```

Pass `Statistic`s are compiled out by this LLVM install (Release+NDEBUG ⇒
`NoopStatistic`; `compiler/README.md` gotcha 21a), so everything worth knowing
is published as attributes on the function instead — which also makes it
FileCheck-testable:

| attribute | mirrors | consumer |
|---|---|---|
| `kea.addr` (on each `kea.alloc`) | — | `-kea-emit`: `base(X)` |
| `kea.dram_layout` | `KeafDramLayout`, field for field | `-kea-emit` → `model.map.json` `dram` → the `DRAM_LAYOUT` section |
| `kea.spm_map` | `KeafSpmEntry`, field for field | `-kea-emit` → `model.map.json` `spm_map` → the `SPM_MAP` section |
| `kea.occupancy` | — | humans, and the regression tests |

### 8.1 A note on `first_pc` / `last_pc`

`KeafSpmEntry` defines these as the **instruction index** that first and last
touches the buffer. That is not the same number as `live`, which counts *block
positions*: `kea.alloc` occupies a block position but emits no instruction, and
neither do the `arith.constant`s that feed the weight symbols. The pass counts
PCs as "kea ops that are not a `kea.alloc`", which is exactly the set that
becomes a 32-byte instruction.

`SPM_MAP` is debug-only (`KEAF_SECF_DEBUG_ONLY`) and nothing at run time
consults it — every address in the stream is already absolute. It exists so
`kea-dis --annotate` can put a name to an address, and so scratchpad-occupancy
plots have something to plot.

---

## 9. What is not implemented

Stated plainly rather than stubbed.

* **`kea.addr` is a discardable attribute, not a declared one.** It should be an
  `OptionalAttr<Kea_I64>:$addr` on `kea.alloc`, which would let the op verifier
  enforce alignment and capacity on the base directly and would give
  `-kea-emit` a typed accessor instead of a string lookup. That is a change to
  `KeaMachineOps.td`, which this pass does not own.
* **One arena per function.** `kea.dram_layout` is attached per `func.func`. A
  module with several NPU entry points would need a module-level pass to merge
  the arenas, or would get one arena per function with colliding offsets.
  Every model compiled so far is one function.
* **No cross-space packing.** SPM_A, SPM_W and ACC are separate physical
  memories, so this is a hardware fact, not a limitation — but it does mean a
  program that overflows SPM_W cannot borrow the 99.7% of SPM_A it is not using.
  Only re-tiling can fix that.
* **No spilling.** If a program does not fit, the pass refuses; it does not
  insert DMA to spill a scratchpad tile to DRAM. That is the right call — a
  spill the tiler did not price would silently wreck the cost model that chose
  the tile — but it does mean the only recovery is to re-run `-kea-tile`.
* **No offset-preference or bank-conflict awareness.** MICROARCH.md §6.4 notes
  that SPM bank conflicts are not modelled by the simulator either, so there is
  nothing yet to optimize against. If `--strict-spm` ever lands, placement is
  where the mitigation would go: the packer already chooses among equally
  valid offsets and could prefer ones that spread hot buffers across banks.
* **Const-region deduplication.** Two layers with byte-identical weights get two
  copies in the `CONST` blob. Detecting that needs the materialized bytes, which
  only `-kea-emit` has.
