// RUN: kea-opt %s -split-input-file -kea-schedule | FileCheck %s
// RUN: kea-opt %s -split-input-file -kea-schedule | kea-opt -split-input-file -kea-tile | FileCheck %s --check-prefix=REVALIDATE
// RUN: kea-opt %s -split-input-file -kea-schedule=report-schedule=true | FileCheck %s --check-prefix=REPORT
// RUN: kea-opt %s -split-input-file -kea-schedule | kea-opt -split-input-file -kea-schedule | FileCheck %s --check-prefix=IDEMPOTENT
// RUN: kea-opt %s -split-input-file -kea-schedule | FileCheck %s --check-prefix=ENGINES
// RUN: kea-opt %s -split-input-file "-kea-schedule=report-schedule=true queue-depth=2" | FileCheck %s --check-prefix=QDEPTH
//
// -kea-schedule: the correct *sequential* Level 2 program -kea-tile emits
// becomes a correctly synchronized *concurrent* one. docs/SCHEDULING.md is the
// write-up; docs/adr/0002 §"the soundness obligation" is the normative part.
//
// The second RUN line is worth reading twice: running `-kea-tile` on a
// function that is already Level 2 does nothing except re-run
// `verifyWeightBanks()` and `refreshLiveRanges()` (DIALECT_L2.md §2), so it is
// a free re-check that the scheduler did not break the weight-bank invariant
// or leave a stale live range. The fourth re-runs the scheduler on its own
// output: it drops the sync it finds and recomputes it, so the pass is a fixed
// point.

//===--------------------------------------------------------------------===//
// 1. Queue assignment
//===--------------------------------------------------------------------===//
//
// `unit` is a real ISA field only where the compiler has a choice: DMA (which
// engine) and the queue-agnostic SIGNAL / WAIT / TRACE (which queue). Every
// other opcode names its own unit, so `kea.load_w` / `kea.mm` / `kea.dwconv` /
// `kea.v*` have no `unit` operand at all -- the derived queue is stamped as the
// discardable `kea.unit` instead, so the assignment is readable and testable.

// Every instruction lands on a queue. The three DMA loads are spread over
// both engines, and the store lands on whichever one is free.
// CHECK-LABEL: func.func @unit_assignment
// CHECK-DAG: kea.vcopy {{.*}}kea.unit = "VPU"
// CHECK-DAG: kea.load_w {{.*}}kea.unit = "MXU"
// CHECK-DAG: kea.mm {{.*}}kea.unit = "MXU"
// CHECK-DAG: kea.vquant {{.*}}kea.unit = "VPU"
// CHECK-DAG: kea.dma_load {{.*}}unit = "DMA0"
// CHECK-DAG: kea.dma_load {{.*}}unit = "DMA1"
// CHECK-DAG: kea.dma_store {{.*}}unit = "DMA
// The store lands on DMA0 here, not DMA1: with one descriptor in flight at a
// time the second engine would start it no earlier and both would then share
// the 16 B/cycle DRAM port. DMA1 is spillover, not load balancing (§2.2).
// CHECK-NOT: unit = "DMA1"
// REPORT-LABEL: func.func @unit_assignment
// REPORT-SAME:  mode = "overlap"
// REPORT-SAME:  MXU = {busy = 78 : i64, instrs = 2 : i64
func.func @unit_assignment() {
  %dw = kea.alloc {name = "u.dw", role = "weights"} : !kea.buffer<256xi8, DRAM>
  %dq = kea.alloc {name = "u.dq", role = "qparam"} : !kea.buffer<192xi8, DRAM>
  %di = kea.alloc {name = "u.di", role = "input"} : !kea.buffer<1024xi8, DRAM>
  %do = kea.alloc {name = "u.do", role = "output"} : !kea.buffer<1024xi8, DRAM>

  %w = kea.alloc {name = "u.w", role = "scratch"} : !kea.buffer<256xi8, W>
  %q = kea.alloc {name = "u.q", role = "scratch"} : !kea.buffer<192xi8, W>
  %a = kea.alloc {name = "u.a", role = "scratch"} : !kea.buffer<1040xi8, A>
  %c = kea.alloc {name = "u.c", role = "scratch"} : !kea.buffer<1024xi32, ACC>
  %o = kea.alloc {name = "u.o", role = "scratch"} : !kea.buffer<1040xi8, A>

  kea.dma_load %dw -> %w {dram_addr = 0, spm_addr = 0, len0 = 256, n1 = 1,
      n2 = 1, dram_s1 = 256, dram_s2 = 256, spm_s1 = 256, spm_s2 = 256}
    : !kea.buffer<256xi8, DRAM> -> !kea.buffer<256xi8, W>
  kea.dma_load %dq -> %q {dram_addr = 0, spm_addr = 0, len0 = 192, n1 = 1,
      n2 = 1, dram_s1 = 192, dram_s2 = 192, spm_s1 = 192, spm_s2 = 192}
    : !kea.buffer<192xi8, DRAM> -> !kea.buffer<192xi8, W>
  kea.vcopy to %a {fill, fill_value = -5, dst_addr = 0, row_bytes = 1024,
      rows = 1, dst_row_stride = 1024, src_addr = 0, src_row_stride = 0}
    : !kea.buffer<1040xi8, A>
  kea.dma_load %di -> %a {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
      n2 = 1, dram_s1 = 1024, dram_s2 = 1024, spm_s1 = 1024, spm_s2 = 1024}
    : !kea.buffer<1024xi8, DRAM> -> !kea.buffer<1040xi8, A>
  kea.load_w %w {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
      bank = 0} : !kea.buffer<256xi8, W>
  kea.mm %a, %c {a_addr = 0, a_inner_stride = 16, a_outer_stride = 128,
      m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
      acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1040xi8, A>, !kea.buffer<1024xi32, ACC>
  kea.vquant %c, %o, %q {acc_addr = 0, out_addr = 0, qparam_addr = 0,
      num_pixels = 64, channels = 16, acc_pix_stride = 16, out_pix_stride = 16,
      out_zp = -5, clamp_lo = -128, clamp_hi = 127}
    : !kea.buffer<1024xi32, ACC>, !kea.buffer<1040xi8, A>, !kea.buffer<192xi8, W>
  kea.dma_store %o -> %do {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
      n2 = 1, dram_s1 = 1024, dram_s2 = 1024, spm_s1 = 1024, spm_s2 = 1024}
    : !kea.buffer<1040xi8, A> -> !kea.buffer<1024xi8, DRAM>
  kea.halt
  return
}

// -----

//===--------------------------------------------------------------------===//
// 2. No synchronization inside a queue; a semaphore on every cross-queue edge
//===--------------------------------------------------------------------===//
//
// ISA.md §7.3: "Two instructions in the same queue" need nothing, because each
// unit is in-order. So the four MXU instructions below -- two LOAD_W/MATMUL
// pairs accumulating into one ACC region -- get no `kea.wait` between them at
// all, and the only sync in the function is the one DMA0 -> MXU edge that
// really exists (the weight tile) and the one MXU -> VPU edge (the ACC).
//
// Exactly one wait and one signal per channel: the second MATMUL does not
// re-wait for the weight DMA, because the first already did and the MXU queue
// is in-order.

// Re-scheduling the scheduled program reproduces it exactly: the pass erases
// the sync it finds and recomputes it, so it is a fixed point.
// IDEMPOTENT-LABEL: func.func @no_sync_within_a_queue
// IDEMPOTENT:       kea.dma_load
// IDEMPOTENT:       kea.signal 0 {unit = "DMA0"}
// IDEMPOTENT:       kea.wait 0 {unit = "MXU"}
// IDEMPOTENT:       kea.load_w
// IDEMPOTENT-NOT:   kea.wait
// IDEMPOTENT:       kea.mm
// IDEMPOTENT:       kea.load_w
// IDEMPOTENT:       kea.mm
// IDEMPOTENT:       kea.signal 1 {unit = "MXU"}
// IDEMPOTENT:       kea.wait 1 {unit = "VPU"}
// IDEMPOTENT:       kea.vquant
// CHECK-LABEL: func.func @no_sync_within_a_queue
// CHECK:       kea.dma_load
// CHECK:       kea.signal 0 {unit = "DMA0"}
// CHECK:       kea.wait 0 {unit = "MXU"}
// Nothing at all between the four MXU instructions: not a wait, not a signal.
// CHECK:       kea.load_w
// CHECK-NOT:   kea.wait
// CHECK-NOT:   kea.signal
// CHECK:       kea.mm
// CHECK-NOT:   kea.wait
// CHECK-NOT:   kea.signal
// CHECK:       kea.load_w
// CHECK-NOT:   kea.wait
// CHECK-NOT:   kea.signal
// CHECK:       kea.mm
// CHECK:       kea.signal 1 {unit = "MXU"}
// CHECK:       kea.wait 1 {unit = "VPU"}
// CHECK:       kea.vquant
// CHECK-NOT:   kea.wait
// REPORT-LABEL: func.func @no_sync_within_a_queue
// REPORT-SAME:  events = 2 : i64
// REPORT-SAME:  signals = 2 : i64
// REPORT-SAME:  waits = 2 : i64
func.func @no_sync_within_a_queue() {
  %dw = kea.alloc {name = "s.dw", role = "weights"} : !kea.buffer<512xi8, DRAM>
  %w = kea.alloc {name = "s.w", role = "scratch"} : !kea.buffer<512xi8, W>
  %q = kea.alloc {name = "s.q", role = "scratch"} : !kea.buffer<192xi8, W>
  %a = kea.alloc {name = "s.a", role = "scratch"} : !kea.buffer<1040xi8, A>
  %c = kea.alloc {name = "s.c", role = "scratch"} : !kea.buffer<1024xi32, ACC>
  %o = kea.alloc {name = "s.o", role = "scratch"} : !kea.buffer<1040xi8, A>

  kea.dma_load %dw -> %w {dram_addr = 0, spm_addr = 0, len0 = 512, n1 = 1,
      n2 = 1, dram_s1 = 512, dram_s2 = 512, spm_s1 = 512, spm_s2 = 512}
    : !kea.buffer<512xi8, DRAM> -> !kea.buffer<512xi8, W>
  kea.load_w %w {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
      bank = 0} : !kea.buffer<512xi8, W>
  kea.mm %a, %c {a_addr = 0, a_inner_stride = 16, a_outer_stride = 128,
      m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
      acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1040xi8, A>, !kea.buffer<1024xi32, ACC>
  kea.load_w %w {w_addr = 256, w_row_stride = 16, k_rows = 16, n_cols = 16,
      bank = 1} : !kea.buffer<512xi8, W>
  kea.mm %a, %c {a_addr = 16, a_inner_stride = 16, a_outer_stride = 128,
      m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
      acc_outer_stride = 128, bank = 1, accumulate}
    : !kea.buffer<1040xi8, A>, !kea.buffer<1024xi32, ACC>
  kea.vquant %c, %o, %q {acc_addr = 0, out_addr = 0, qparam_addr = 0,
      num_pixels = 64, channels = 16, acc_pix_stride = 16, out_pix_stride = 16,
      out_zp = -5, clamp_lo = -128, clamp_hi = 127}
    : !kea.buffer<1024xi32, ACC>, !kea.buffer<1040xi8, A>, !kea.buffer<192xi8, W>
  kea.halt
  return
}

// -----

//===--------------------------------------------------------------------===//
// 3. Double buffering: the prefetch really is hoisted above the compute
//===--------------------------------------------------------------------===//
//
// Two spatial tiles, each with its own `kea.alloc` -- which is what -kea-tile
// emits and what makes them independent. Tile 1's `DMA_LD` depends on nothing
// tile 0 does, so it lands *above* tile 0's MATMUL, and the two engines take
// one each so the load and the store of different tiles run together.
//
// This is also the shape ADR-0002's amendment describes: after the hoist,
// %a1's `kea.alloc` sits inside %a0's live range, so their block-order ranges
// overlap and `-kea-alloc` separates them for the ordinary reason.

// BOTH activation loads are issued before EITHER MATMUL: tile 1 is being
// fetched while tile 0 is being multiplied. That is the double buffering.
// CHECK-LABEL: func.func @prefetch_is_hoisted
// CHECK:       kea.dma_load %{{[0-9]+}} -> %[[A0:[0-9]+]] {{.*}}<4096xi8, DRAM> -> !kea.buffer<1040xi8, A>
// CHECK:       kea.dma_load %{{[0-9]+}} -> %[[A1:[0-9]+]] {{.*}}<4096xi8, DRAM> -> !kea.buffer<1040xi8, A>
// CHECK:       kea.mm %[[A0]],
// CHECK:       kea.mm %[[A1]],
// The activation loads go to different engines, so tile 1 is being fetched
// while tile 0 is being multiplied (ISA.md §12). The stores stay on DMA0: by
// the time they are ready nothing else is in flight, and splitting them would
// only make both share the 16 B/cycle DRAM port -- see §2.2.
// ENGINES-LABEL: func.func @prefetch_is_hoisted
// ENGINES-DAG:   kea.dma_load {{.*}}unit = "DMA0"
// ENGINES-DAG:   kea.dma_load {{.*}}unit = "DMA1"
// ENGINES-DAG:   kea.dma_store {{.*}}unit = "DMA0"
// Both engines carry work, and the depth-16 queues are modelled and respected.
// REPORT-LABEL: func.func @prefetch_is_hoisted
// REPORT-SAME:  queue_depth = 16 : i64
// REPORT-SAME:  DMA0 = {busy = {{[1-9][0-9]*}} : i64, dram_bytes = {{[1-9][0-9]*}} : i64, instrs = {{[1-9]}} : i64, max_queue = {{[1-9]}} : i64}
// REPORT-SAME:  DMA1 = {busy = {{[1-9][0-9]*}} : i64, dram_bytes = {{[1-9][0-9]*}} : i64, instrs = {{[1-9]}} : i64, max_queue = {{[1-9]}} : i64}
//
// QUEUE DEPTH IS MODELLED, NOT ASSUMED. Shrink the modelled queues to two
// slots and the scheduler stops running as far ahead: the high-water mark on
// every queue comes down to the new bound, the dispatcher stall it would
// otherwise have paid is priced in, and the schedule is still correct.
// QDEPTH-LABEL: func.func @prefetch_is_hoisted
// QDEPTH-SAME:  queue_depth = 2 : i64
// QDEPTH-SAME:  DMA0 = {busy = {{[0-9]+}} : i64, dram_bytes = {{[0-9]+}} : i64, instrs = {{[0-9]+}} : i64, max_queue = {{[12]}} : i64}
// QDEPTH-SAME:  DMA1 = {busy = {{[0-9]+}} : i64, dram_bytes = {{[0-9]+}} : i64, instrs = {{[0-9]+}} : i64, max_queue = {{[12]}} : i64}
// QDEPTH-SAME:  MXU = {busy = {{[0-9]+}} : i64, instrs = {{[0-9]+}} : i64, max_queue = {{[12]}} : i64}
// QDEPTH-SAME:  VPU = {busy = {{[0-9]+}} : i64, instrs = {{[0-9]+}} : i64, max_queue = {{[12]}} : i64}
func.func @prefetch_is_hoisted() {
  %din = kea.alloc {name = "d.in", role = "input"} : !kea.buffer<4096xi8, DRAM>
  %dw = kea.alloc {name = "d.dw", role = "weights"} : !kea.buffer<256xi8, DRAM>
  %dout = kea.alloc {name = "d.out", role = "output"} : !kea.buffer<4096xi8, DRAM>
  %q = kea.alloc {name = "d.q", role = "scratch"} : !kea.buffer<192xi8, W>
  %w = kea.alloc {name = "d.w", role = "scratch"} : !kea.buffer<256xi8, W>
  kea.dma_load %dw -> %w {dram_addr = 0, spm_addr = 0, len0 = 256, n1 = 1,
      n2 = 1, dram_s1 = 256, dram_s2 = 256, spm_s1 = 256, spm_s2 = 256}
    : !kea.buffer<256xi8, DRAM> -> !kea.buffer<256xi8, W>
  kea.load_w %w {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
      bank = 0} : !kea.buffer<256xi8, W>

  // tile 0
  %a0 = kea.alloc {name = "d.a0", role = "scratch"} : !kea.buffer<1040xi8, A>
  %c0 = kea.alloc {name = "d.c0", role = "scratch"} : !kea.buffer<1024xi32, ACC>
  %o0 = kea.alloc {name = "d.o0", role = "scratch"} : !kea.buffer<1040xi8, A>
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

  // tile 1
  %a1 = kea.alloc {name = "d.a1", role = "scratch"} : !kea.buffer<1040xi8, A>
  %c1 = kea.alloc {name = "d.c1", role = "scratch"} : !kea.buffer<1024xi32, ACC>
  %o1 = kea.alloc {name = "d.o1", role = "scratch"} : !kea.buffer<1040xi8, A>
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

// -----

//===--------------------------------------------------------------------===//
// 4. The weight-bank invariant survives (DIALECT_L2.md §6.1)
//===--------------------------------------------------------------------===//
//
// Four one-tap output-channel groups with banks 0/1/0/1. MXU instructions are
// never reordered against each other, and nothing -- not even a `kea.signal`
// or a `kea.wait` -- is placed between a `kea.load_w` and the `kea.mm` that
// reads its bank. The one DMA0 -> MXU wait therefore sits *above* the first
// LOAD_W, not between it and its MATMUL.
//
// `-kea-tile` re-runs `verifyWeightBanks()` over this output (the REVALIDATE
// RUN line), so errata E7 is re-proved rather than assumed.

// CHECK-LABEL: func.func @weight_banks_preserved
// CHECK:       kea.wait {{.*}}unit = "MXU"
// CHECK:       kea.load_w {{.*}}bank = 0
// CHECK-NOT:   kea.wait
// CHECK-NOT:   kea.signal
// CHECK:       kea.mm {{.*}}bank = 0
// CHECK-NOT:   kea.wait
// CHECK-NOT:   kea.signal
// CHECK:       kea.load_w {{.*}}bank = 1
// CHECK-NOT:   kea.wait
// CHECK-NOT:   kea.signal
// CHECK:       kea.mm {{.*}}bank = 1
// CHECK-NOT:   kea.wait
// CHECK-NOT:   kea.signal
// CHECK:       kea.load_w {{.*}}bank = 0
// CHECK-NOT:   kea.wait
// CHECK-NOT:   kea.signal
// CHECK:       kea.mm {{.*}}bank = 0
// CHECK-NOT:   kea.wait
// CHECK-NOT:   kea.signal
// CHECK:       kea.load_w {{.*}}bank = 1
// CHECK-NOT:   kea.wait
// CHECK-NOT:   kea.signal
// CHECK:       kea.mm {{.*}}bank = 1
// REVALIDATE-LABEL: func.func @weight_banks_preserved
// REVALIDATE: kea.load_w
func.func @weight_banks_preserved() {
  %dw = kea.alloc {name = "b.dw", role = "weights"} : !kea.buffer<1024xi8, DRAM>
  %w = kea.alloc {name = "b.w", role = "scratch"} : !kea.buffer<1024xi8, W>
  %a = kea.alloc {name = "b.a", role = "scratch"} : !kea.buffer<1040xi8, A>
  %c = kea.alloc {name = "b.c", role = "scratch"} : !kea.buffer<4096xi32, ACC>
  kea.dma_load %dw -> %w {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
      n2 = 1, dram_s1 = 1024, dram_s2 = 1024, spm_s1 = 1024, spm_s2 = 1024}
    : !kea.buffer<1024xi8, DRAM> -> !kea.buffer<1024xi8, W>
  kea.load_w %w {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
      bank = 0} : !kea.buffer<1024xi8, W>
  kea.mm %a, %c {a_addr = 0, a_inner_stride = 16, a_outer_stride = 128,
      m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
      acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1040xi8, A>, !kea.buffer<4096xi32, ACC>
  kea.load_w %w {w_addr = 256, w_row_stride = 16, k_rows = 16, n_cols = 16,
      bank = 1} : !kea.buffer<1024xi8, W>
  kea.mm %a, %c {a_addr = 0, a_inner_stride = 16, a_outer_stride = 128,
      m_inner = 8, m_outer = 8, acc_addr = 1024, acc_inner_stride = 16,
      acc_outer_stride = 128, bank = 1}
    : !kea.buffer<1040xi8, A>, !kea.buffer<4096xi32, ACC>
  kea.load_w %w {w_addr = 512, w_row_stride = 16, k_rows = 16, n_cols = 16,
      bank = 0} : !kea.buffer<1024xi8, W>
  kea.mm %a, %c {a_addr = 0, a_inner_stride = 16, a_outer_stride = 128,
      m_inner = 8, m_outer = 8, acc_addr = 2048, acc_inner_stride = 16,
      acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1040xi8, A>, !kea.buffer<4096xi32, ACC>
  kea.load_w %w {w_addr = 768, w_row_stride = 16, k_rows = 16, n_cols = 16,
      bank = 1} : !kea.buffer<1024xi8, W>
  kea.mm %a, %c {a_addr = 0, a_inner_stride = 16, a_outer_stride = 128,
      m_inner = 8, m_outer = 8, acc_addr = 3072, acc_inner_stride = 16,
      acc_outer_stride = 128, bank = 1}
    : !kea.buffer<1040xi8, A>, !kea.buffer<4096xi32, ACC>
  kea.halt
  return
}

// -----

//===--------------------------------------------------------------------===//
// 5. `mode=serial` -- the A/B control docs/SCHEDULING.md measures against
//===--------------------------------------------------------------------===//
//
// -kea-tile's order, one engine, and a handshake at every cross-queue
// adjacency: the sequential program executed sequentially. It is synchronized
// by the same machinery as the real schedule (so it satisfies Rule D the same
// way) and simply has nothing left to overlap.

// SERIAL-LABEL: func.func @serial_baseline
// SERIAL: kea.dma_load {{.*}}unit = "DMA0"
// SERIAL: kea.signal 0 {unit = "DMA0"}
// SERIAL: kea.wait 0 {unit = "VPU"}
// SERIAL: kea.vcopy
// SERIAL: kea.signal 1 {unit = "VPU"}
// SERIAL: kea.wait 1 {unit = "DMA0"}
// SERIAL: kea.dma_store {{.*}}unit = "DMA0"
// SERIAL-NOT: unit = "DMA1"
// The overlapped schedule is free to use the second engine but does not need
// to here: one load, one copy, one store, nothing concurrent. What it does drop
// is the baseline's redundant handshake -- the vcopy reads %a and the store
// reads %b, so DMA0 never has to wait for the VPU twice.
// CHECK-LABEL: func.func @serial_baseline
// CHECK: kea.dma_store {{.*}}unit = "DMA0"

// RUN: kea-opt %s -split-input-file -kea-schedule=mode=serial | FileCheck %s --check-prefix=SERIAL
func.func @serial_baseline() {
  %din = kea.alloc {name = "z.in", role = "input"} : !kea.buffer<1024xi8, DRAM>
  %dout = kea.alloc {name = "z.out", role = "output"} : !kea.buffer<1024xi8, DRAM>
  %a = kea.alloc {name = "z.a", role = "scratch"} : !kea.buffer<1024xi8, A>
  %b = kea.alloc {name = "z.b", role = "scratch"} : !kea.buffer<1024xi8, A>
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
// 6. Region markers get a queue, and bracket the region on it
//===--------------------------------------------------------------------===//
//
// `kea.trace` is queue-agnostic, so its unit is part of its meaning and is
// this pass's decision. It goes on the queue that does the region's
// arithmetic, which is what makes errata E4 -- "the region keeps accumulating
// until the issuing unit's pipeline drains" -- attribute the tail of a layer
// to that layer.

// CHECK-LABEL: func.func @regions_get_a_queue
// CHECK: kea.trace "begin" 7 {unit = "MXU"}
// CHECK: kea.dma_load
// CHECK: kea.mm
// CHECK: kea.trace "end" 7 {unit = "MXU"}
// CHECK: kea.halt
func.func @regions_get_a_queue() {
  %dw = kea.alloc {name = "r.dw", role = "weights"} : !kea.buffer<256xi8, DRAM>
  %w = kea.alloc {name = "r.w", role = "scratch"} : !kea.buffer<256xi8, W>
  %a = kea.alloc {name = "r.a", role = "scratch"} : !kea.buffer<1040xi8, A>
  %c = kea.alloc {name = "r.c", role = "scratch"} : !kea.buffer<1024xi32, ACC>
  kea.trace "begin" 7
  kea.dma_load %dw -> %w {dram_addr = 0, spm_addr = 0, len0 = 256, n1 = 1,
      n2 = 1, dram_s1 = 256, dram_s2 = 256, spm_s1 = 256, spm_s2 = 256}
    : !kea.buffer<256xi8, DRAM> -> !kea.buffer<256xi8, W>
  kea.load_w %w {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
      bank = 0} : !kea.buffer<256xi8, W>
  kea.mm %a, %c {a_addr = 0, a_inner_stride = 16, a_outer_stride = 128,
      m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
      acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1040xi8, A>, !kea.buffer<1024xi32, ACC>
  kea.trace "end" 7
  kea.halt
  return
}
