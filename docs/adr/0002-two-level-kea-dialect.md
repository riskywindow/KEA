# ADR-0002: The `kea` dialect has two levels, not one

**Status:** accepted
**Date:** 2026-08-01

## Context

The compiler has to get from `tosa.conv2d` on tensors down to a straight-line
stream of 32-byte KEA instructions with absolute scratchpad addresses and
hand-placed semaphores. That is an enormous drop. Doing it in one conversion
would mean a single pass that simultaneously matches TOSA patterns, picks tile
sizes, allocates scratchpad, and schedules DMA — untestable, and impossible for
several people to work on at once.

## Decision

`kea` is **one dialect with two clearly separated levels**, and every pass is
labelled with the level it consumes and the level it produces.

### Level 1 — tensor level (`kea` "op" ops)

Value-semantic ops on `tensor<...>`, carrying quantization as structured
attributes. No addresses, no tiles, no buffers, no events. This is a *normalised*
form of the input graph: the ~15 TOSA/linalg spellings we accept collapse into
one canonical NPU-shaped op each.

`kea.conv2d`, `kea.dwconv2d`, `kea.matmul`, `kea.fully_connected`, `kea.add`,
`kea.pool`, `kea.rescale`, `kea.clamp`, `kea.reshape`, `kea.transpose`

Each op carries a **fused epilogue** description (bias, requantize, activation
clamp, optional residual add) rather than leaving those as separate ops, because
the hardware performs them as one VPU pass.

### Level 2 — buffer level (`kea` "machine" ops)

Ops on `!kea.buffer<shape, elemtype, addrspace>` with **absolute integer
scratchpad addresses**, explicit DMA, explicit queue assignment, and explicit
semaphore signal/wait. There is a 1:1 correspondence between a Level 2 op and a
KEA instruction — this level is the ISA in SSA clothing.

`kea.dma_load`, `kea.dma_store`, `kea.load_w`, `kea.mm`, `kea.dwconv`,
`kea.vquant`, `kea.vadd`, `kea.vpool`, `kea.vcopy`, `kea.signal`, `kea.wait`

### The pass pipeline

| Pass | In | Out |
| --- | --- | --- |
| `-tosa-to-kea` / `-linalg-to-kea` | tosa / linalg | L1 |
| `-kea-fuse` | L1 | L1 (fewer ops, fatter epilogues) |
| `-kea-tile` | L1 | L2 (tiled to scratchpad capacity, symbolic buffers) |
| `-kea-schedule` | L2 | L2 (queues assigned, semaphores inserted, DMA double-buffered) |
| `-kea-alloc` | L2 | L2 (symbolic buffers → absolute addresses) |
| `-kea-emit` (kea-translate) | L2 | `.kasm` + weights + map (see ADR-0001) |

### Amendment (2026-08-01): scheduling runs *before* allocation

The table originally had `-kea-alloc` before `-kea-schedule`. That is wrong, and
the reason is worth recording because it is the nicest property in the backend.

`-kea-tile` emits a fresh `kea.alloc` per spatial tile, so consecutive tiles have
**disjoint** live ranges. An allocator run at that point would correctly conclude
they can share storage and give them the same address — at which point double
buffering is impossible, because prefetching tile *N+1* would overwrite tile *N*.

Run the scheduler first and the problem dissolves. Hoisting a DMA so it overlaps
the previous tile's compute *extends that buffer's live range* until it overlaps
its predecessor's. The allocator then sees overlapping ranges and separates them
for the ordinary reason. So:

> **Double buffering is expressed entirely as a live-range extension. The
> allocator needs no knowledge of it, and no concept of "double buffering"
> appears anywhere in `-kea-alloc`.**

`-kea-tile` reserves half of each scratchpad (`spm-reserve-factor`, default 2) so
the space for the second set of tiles is guaranteed to exist.

The cost is that `-kea-alloc` must tolerate `kea.signal`/`kea.wait` in its input
and must recompute live ranges rather than trusting the `live` attribute stamped
by `-kea-tile` — the scheduler moves ops, which shifts the block indices those
ranges are expressed in. `mlir::kea::refreshLiveRanges()` exists for exactly
this, and any pass that inserts, erases, or moves an L2 op must call it.

### The soundness obligation this places on `-kea-schedule` (normative)

Reordering into a *concurrent* program while liveness is still expressed in
*sequential* block order creates an obligation that is easy to miss and
catastrophic to violate:

> **After `-kea-schedule`, block order must be a sound over-approximation of
> temporal liveness. If two operations can execute concurrently — different
> queues, with no semaphore ordering them — then every buffer they touch must
> have overlapping block-order live ranges.**

If the scheduler leaves two operations able to run at the same time but their
buffers' block-order ranges are disjoint, `-kea-alloc` is *entitled* to place
those buffers at the same address. The result is a data race between two
hardware units, which surfaces as a wrong number in an output tensor with
nothing in the instruction stream to point at.

This cannot be checked from inside `-kea-alloc` — by the time it runs, the
information about what may execute concurrently has already been discarded. It
is the scheduler's obligation to establish and its tests' job to demonstrate.
The practical discharge is straightforward: whenever a DMA is hoisted to overlap
earlier work, extend the affected buffer's live range to cover everything it now
overlaps, rather than only its own producer and consumers.

## Consequences

- Each pass is independently testable with a small `.mlir` input and a FileCheck
  expectation, and different people can own different passes.
- The level boundary is where the interesting compiler work lives: **everything
  above it is graph normalisation, everything below it is hardware reality.**
  Tiling is the single pass that crosses it, so tiling is the pass that gets the
  most scrutiny.
- A verifier can enforce the invariant that Level 1 and Level 2 ops never appear
  in the same function after `-kea-tile`, catching partial lowering early.
- Cost: two sets of ops to define, and conv-like semantics expressed twice.
  Accepted — the alternative is one unmaintainable megapass.
