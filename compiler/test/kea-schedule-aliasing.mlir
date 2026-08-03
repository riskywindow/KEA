// RUN: kea-opt %s -kea-alloc | FileCheck %s --check-prefix=UNSCHEDULED
// RUN: kea-opt %s -kea-schedule | FileCheck %s --check-prefix=RANGES
// RUN: kea-opt %s -kea-schedule -kea-alloc | FileCheck %s --check-prefix=SEPARATED
// RUN: kea-opt %s -kea-schedule -kea-alloc | kea-opt -kea-alloc=verify-only=true | FileCheck %s --check-prefix=SEPARATED
//
//===----------------------------------------------------------------------===//
// THE TEST THAT WOULD CATCH AN ALIASING VIOLATION
//===----------------------------------------------------------------------===//
//
// ADR-0002 makes this normative on `-kea-schedule`:
//
//   "If two operations can execute concurrently -- different queues, with no
//    semaphore ordering them -- then every buffer they touch must have
//    overlapping block-order live ranges."
//
// Violating it is not a crash. `-kea-alloc` sees two disjoint ranges, does
// exactly what it is supposed to do, gives the two buffers the same address --
// and two hardware units then write the same bytes at the same time. The
// symptom is a wrong number in an output tensor with nothing in the
// instruction stream to point at.
//
// The program below is the smallest thing that exhibits it: two spatial tiles,
// each with its own `kea.alloc`, exactly as `-kea-tile` emits them.
//
// FIRST RUN LINE -- the hazard, made visible. Running `-kea-alloc` on the
// *unscheduled* program is correct and gives `t.a0` and `t.a1` THE SAME
// ADDRESS, because in the sequential program their live ranges really are
// disjoint. That is the state of affairs a scheduler that hoisted tile 1's
// DMA without extending anything would hand the allocator, and it is a data
// race: DMA would be filling `t.a1` while the MXU was still reading `t.a0`.
//
// UNSCHEDULED-LABEL: func.func @two_tiles
// The two activation tiles, the two ACC regions and the two output tiles all
// come out on top of each other -- three separate races waiting to happen.
// UNSCHEDULED: kea.alloc {addr = 0 : i64, live = array<i64: 7, 11>, name = "t.a0"
// UNSCHEDULED: kea.alloc {addr = 0 : i64, live = array<i64: 8, 12>, name = "t.c0"
// UNSCHEDULED: kea.alloc {addr = 1040 : i64, live = array<i64: 9, 13>, name = "t.o0"
// UNSCHEDULED: kea.alloc {addr = 0 : i64, live = array<i64: 14, 18>, name = "t.a1"
// UNSCHEDULED: kea.alloc {addr = 0 : i64, live = array<i64: 15, 19>, name = "t.c1"
// UNSCHEDULED: kea.alloc {addr = 1040 : i64, live = array<i64: 16, 20>, name = "t.o1"
//
// SECOND RUN LINE -- what the scheduler actually hands over. After scheduling,
// `t.a1`'s `kea.alloc` sits *inside* `t.a0`'s live range, so the two ranges
// overlap. Nothing was annotated to make that true: the alloc was moved, which
// is what `refreshLiveRanges()` and `-kea-alloc` both re-derive the range from.
//
// RANGES-LABEL: func.func @two_tiles
// RANGES: kea.alloc {live = array<i64: 6, 17>, name = "t.a0"
// RANGES: kea.alloc {live = array<i64: 7, 22>, name = "t.a1"
// RANGES: kea.alloc {live = array<i64: 12, 35>, name = "t.o0"
// RANGES: kea.alloc {live = array<i64: 16, 27>, name = "t.c0"
// RANGES: kea.alloc {live = array<i64: 21, 31>, name = "t.c1"
// RANGES: kea.alloc {live = array<i64: 30, 40>, name = "t.o1"
//
// THIRD AND FOURTH RUN LINES -- the consequence. The allocator now separates
// them for the ordinary reason, and re-running it in `verify-only` mode
// re-proves from scratch that no two overlapping-live-range buffers were given
// overlapping storage.
//
// The price of the overlap, stated exactly: SPM_A `maxlive` goes from 2080 (one
// tile's activation plus one output) to 3120 (three of the four at once). That
// is what `-kea-tile`'s `spm-reserve-factor = 2` reserved the room for.
// SEPARATED-LABEL: func.func @two_tiles
// SEPARATED-SAME:  spm_a = {buffers = 4 : i64, capacity = 262144 : i64, fragmentation = 0 : i64, maxlive = 3120 : i64, peak = 3120 : i64, unpacked = 4160 : i64}
// SEPARATED: kea.alloc {addr = 2080 : i64, {{.*}}name = "t.a0"
// SEPARATED: kea.alloc {addr = 1040 : i64, {{.*}}name = "t.a1"
// The two ACC regions and the two output tiles are separated for the same
// reason, so nothing in the pipeline is aliased against something concurrent.
// SEPARATED: kea.alloc {addr = 0 : i64, {{.*}}name = "t.o0"
// SEPARATED: kea.alloc {addr = 0 : i64, {{.*}}name = "t.c0"
// SEPARATED: kea.alloc {addr = 1024 : i64, {{.*}}name = "t.c1"
// SEPARATED: kea.alloc {addr = 1040 : i64, {{.*}}name = "t.o1"


func.func @two_tiles() {
  %din = kea.alloc {name = "t.in", role = "input"} : !kea.buffer<4096xi8, DRAM>
  %dw = kea.alloc {name = "t.dw", role = "weights"} : !kea.buffer<256xi8, DRAM>
  %dout = kea.alloc {name = "t.out", role = "output"} : !kea.buffer<4096xi8, DRAM>
  %q = kea.alloc {name = "t.q", role = "scratch"} : !kea.buffer<192xi8, W>
  %w = kea.alloc {name = "t.w", role = "scratch"} : !kea.buffer<256xi8, W>
  kea.dma_load %dw -> %w {dram_addr = 0, spm_addr = 0, len0 = 256, n1 = 1,
      n2 = 1, dram_s1 = 256, dram_s2 = 256, spm_s1 = 256, spm_s2 = 256}
    : !kea.buffer<256xi8, DRAM> -> !kea.buffer<256xi8, W>
  kea.load_w %w {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
      bank = 0} : !kea.buffer<256xi8, W>

  // tile 0
  %a0 = kea.alloc {name = "t.a0", role = "scratch"} : !kea.buffer<1040xi8, A>
  %c0 = kea.alloc {name = "t.c0", role = "scratch"} : !kea.buffer<1024xi32, ACC>
  %o0 = kea.alloc {name = "t.o0", role = "scratch"} : !kea.buffer<1040xi8, A>
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
  %a1 = kea.alloc {name = "t.a1", role = "scratch"} : !kea.buffer<1040xi8, A>
  %c1 = kea.alloc {name = "t.c1", role = "scratch"} : !kea.buffer<1024xi32, ACC>
  %o1 = kea.alloc {name = "t.o1", role = "scratch"} : !kea.buffer<1040xi8, A>
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
