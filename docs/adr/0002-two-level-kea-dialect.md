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
| `-kea-alloc` | L2 | L2 (symbolic buffers → absolute addresses) |
| `-kea-schedule` | L2 | L2 (queues assigned, semaphores inserted, DMA double-buffered) |
| `-kea-emit` (kea-translate) | L2 | `.kasm` + weights + map (see ADR-0001) |

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
