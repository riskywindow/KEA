// RUN: kea-opt %s -split-input-file "-kea-schedule=report-schedule=true" | FileCheck %s
// RUN: kea-opt %s -split-input-file -kea-schedule | FileCheck %s --check-prefix=IR
// RUN: kea-opt %s -split-input-file -kea-schedule=mode=overlap | FileCheck %s --check-prefix=FORCED
//
//===----------------------------------------------------------------------===//
// `mode=auto`: what the default guarantees, and against which baseline
//===----------------------------------------------------------------------===//
//
// Reordering is not free. ADR-0002's soundness obligation is discharged with
// real dependence edges (SCHEDULING.md §6), and on a program with no room to
// double buffer those edges are pure cost. So the default costs two plans with
// one cost model and emits the cheaper:
//
//   * the **overlapped** plan -- the list schedule, both engines, hoisted
//     prefetch, widened live ranges;
//   * the **in-order** plan -- a model of the program the backend emits when
//     this pass does nothing at all: -kea-tile's order, one engine, semaphores
//     derived from storage. NOT `mode=serial`, which synchronizes every
//     cross-queue adjacency and is 9% slower than not scheduling at all, and
//     would be a baseline flattering enough to beat by accident.
//
// The overlapped plan must also clear `decline-margin` percent (default 5).
// Both estimates come from the same model so the difference is meaningful, but
// a 1% predicted win is inside the model's error. Measured on the MobileNetV2
// feature extractor at `imem-budget=19100`, where SPM_A has room for exactly
// one activation tile: the overlapped plan was predicted 0.8% better and
// measured 6% WORSE. With the margin it declines, and the program is then
// byte-identical to the unscheduled one -- 1.0000x, not 0.941x.
//
// Declining means emitting nothing. That is the part that makes the guarantee
// real: a pass that decided not to help must not still charge for the attempt.

//===--------------------------------------------------------------------===//
// 1. Nothing to overlap -> declined, and the IR comes out untouched
//===--------------------------------------------------------------------===//
//
// One DMA, one copy, one store, each depending on the last. There is no
// concurrency to find, so reordering can only add sync.

// CHECK-LABEL: func.func @nothing_to_overlap
// CHECK-SAME:  mode = "declined"
// Both plans were costed; the overlapped one did not clear the margin.
// CHECK-SAME:  modelled_in_order = {{[0-9]+}} : i64
// CHECK-SAME:  modelled_overlap = {{[0-9]+}} : i64

// Untouched means untouched: no `unit`, no `kea.unit`, no semaphores, and
// -kea-tile's own order.
// IR-LABEL: func.func @nothing_to_overlap
// IR-NOT:   kea.signal
// IR-NOT:   kea.wait
// IR-NOT:   unit =
// IR:       kea.dma_load
// IR-NEXT:  kea.vcopy
// IR-NEXT:  kea.dma_store
// IR-NEXT:  kea.halt

// `mode=overlap` overrides the judgement and schedules it anyway, which is how
// the two are compared and why the option exists.
// FORCED-LABEL: func.func @nothing_to_overlap
// FORCED: kea.dma_load {{.*}}unit = "DMA0"
// FORCED: kea.signal
// FORCED: kea.wait
func.func @nothing_to_overlap() {
  %din = kea.alloc {name = "n.in", role = "input"} : !kea.buffer<1024xi8, DRAM>
  %dout = kea.alloc {name = "n.out", role = "output"} : !kea.buffer<1024xi8, DRAM>
  %a = kea.alloc {name = "n.a", role = "scratch"} : !kea.buffer<1024xi8, A>
  %b = kea.alloc {name = "n.b", role = "scratch"} : !kea.buffer<1024xi8, A>
  kea.dma_load %din -> %a {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
      n2 = 1, dram_s1 = 1024, dram_s2 = 1024, spm_s1 = 1024, spm_s2 = 1024}
    : !kea.buffer<1024xi8, DRAM> -> !kea.buffer<1024xi8, A>
  kea.vcopy from %a : !kea.buffer<1024xi8, A>, to %b {src_addr = 0,
      dst_addr = 0, row_bytes = 1024, rows = 1, src_row_stride = 1024,
      dst_row_stride = 1024} : !kea.buffer<1024xi8, A>
  kea.dma_store %b -> %dout {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
      n2 = 1, dram_s1 = 1024, dram_s2 = 1024, spm_s1 = 1024, spm_s2 = 1024}
    : !kea.buffer<1024xi8, A> -> !kea.buffer<1024xi8, DRAM>
  kea.halt
  return
}

// -----

//===--------------------------------------------------------------------===//
// 2. Real overlap available -> engaged
//===--------------------------------------------------------------------===//
//
// Two independent tiles, each a DMA / MATMUL / VQUANT / DMA chain. Tile 1's
// load has no dependence on tile 0, so the reordered plan is far enough ahead
// of the in-order one to clear the margin comfortably.

// CHECK-LABEL: func.func @real_overlap_available
// CHECK-SAME:  mode = "overlap"
// Both activation loads are issued before either MATMUL: that is the overlap
// the decision bought. (kea-schedule.mlir's @prefetch_is_hoisted pins the
// operand identities; here the point is only that the plan engaged.)
// IR-LABEL: func.func @real_overlap_available
// IR:       kea.dma_load {{.*}}!kea.buffer<4096xi8, DRAM> -> !kea.buffer<1040xi8, A>
// IR:       kea.dma_load {{.*}}!kea.buffer<4096xi8, DRAM> -> !kea.buffer<1040xi8, A>
// IR:       kea.mm
// IR:       kea.mm
// And semaphores, which the declined function above has none of. (Engine
// spread is pinned by kea-schedule.mlir's ENGINES prefix.)
// IR:       kea.signal
// IR:       kea.wait
func.func @real_overlap_available() {
  %din = kea.alloc {name = "o.in", role = "input"} : !kea.buffer<4096xi8, DRAM>
  %dw = kea.alloc {name = "o.dw", role = "weights"} : !kea.buffer<256xi8, DRAM>
  %dout = kea.alloc {name = "o.out", role = "output"} : !kea.buffer<4096xi8, DRAM>
  %q = kea.alloc {name = "o.q", role = "scratch"} : !kea.buffer<192xi8, W>
  %w = kea.alloc {name = "o.w", role = "scratch"} : !kea.buffer<256xi8, W>
  kea.dma_load %dw -> %w {dram_addr = 0, spm_addr = 0, len0 = 256, n1 = 1,
      n2 = 1, dram_s1 = 256, dram_s2 = 256, spm_s1 = 256, spm_s2 = 256}
    : !kea.buffer<256xi8, DRAM> -> !kea.buffer<256xi8, W>
  kea.load_w %w {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
      bank = 0} : !kea.buffer<256xi8, W>

  %a0 = kea.alloc {name = "o.a0", role = "scratch"} : !kea.buffer<1040xi8, A>
  %c0 = kea.alloc {name = "o.c0", role = "scratch"} : !kea.buffer<1024xi32, ACC>
  %o0 = kea.alloc {name = "o.o0", role = "scratch"} : !kea.buffer<1040xi8, A>
  kea.dma_load %din -> %a0 {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
      n2 = 1, dram_s1 = 1024, dram_s2 = 1024, spm_s1 = 1024, spm_s2 = 1024}
    : !kea.buffer<4096xi8, DRAM> -> !kea.buffer<1040xi8, A>
  kea.mm %a0, %c0 {a_addr = 0, a_inner_stride = 16, a_outer_stride = 128,
      m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
      acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1040xi8, A>, !kea.buffer<1024xi32, ACC>
  kea.vquant %c0, %o0, %q {acc_addr = 0, out_addr = 0, qparam_addr = 0,
      num_pixels = 64, channels = 16, acc_pix_stride = 16, out_pix_stride = 16,
      out_zp = -5, clamp_lo = -128, clamp_hi = 127}
    : !kea.buffer<1024xi32, ACC>, !kea.buffer<1040xi8, A>, !kea.buffer<192xi8, W>
  kea.dma_store %o0 -> %dout {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
      n2 = 1, dram_s1 = 1024, dram_s2 = 1024, spm_s1 = 1024, spm_s2 = 1024}
    : !kea.buffer<1040xi8, A> -> !kea.buffer<4096xi8, DRAM>

  %a1 = kea.alloc {name = "o.a1", role = "scratch"} : !kea.buffer<1040xi8, A>
  %c1 = kea.alloc {name = "o.c1", role = "scratch"} : !kea.buffer<1024xi32, ACC>
  %o1 = kea.alloc {name = "o.o1", role = "scratch"} : !kea.buffer<1040xi8, A>
  kea.dma_load %din -> %a1 {dram_addr = 1024, spm_addr = 0, len0 = 1024,
      n1 = 1, n2 = 1, dram_s1 = 1024, dram_s2 = 1024, spm_s1 = 1024,
      spm_s2 = 1024}
    : !kea.buffer<4096xi8, DRAM> -> !kea.buffer<1040xi8, A>
  kea.mm %a1, %c1 {a_addr = 0, a_inner_stride = 16, a_outer_stride = 128,
      m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
      acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1040xi8, A>, !kea.buffer<1024xi32, ACC>
  kea.vquant %c1, %o1, %q {acc_addr = 0, out_addr = 0, qparam_addr = 0,
      num_pixels = 64, channels = 16, acc_pix_stride = 16, out_pix_stride = 16,
      out_zp = -5, clamp_lo = -128, clamp_hi = 127}
    : !kea.buffer<1024xi32, ACC>, !kea.buffer<1040xi8, A>, !kea.buffer<192xi8, W>
  kea.dma_store %o1 -> %dout {dram_addr = 1024, spm_addr = 0, len0 = 1024,
      n1 = 1, n2 = 1, dram_s1 = 1024, dram_s2 = 1024, spm_s1 = 1024,
      spm_s2 = 1024}
    : !kea.buffer<1040xi8, A> -> !kea.buffer<4096xi8, DRAM>
  kea.halt
  return
}
