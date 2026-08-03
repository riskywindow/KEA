// RUN: kea-translate %s --function=raw       --no-labels --emit-kasm=- | FileCheck %s --check-prefix=RAW
// RUN: kea-translate %s --function=aliased   --no-labels --emit-kasm=- | FileCheck %s --check-prefix=ALIAS
// RUN: kea-translate %s --function=scheduled --no-labels --emit-kasm=- | FileCheck %s --check-prefix=SCHED
// RUN: kea-translate %s --function=raw --sync=none --no-labels --emit-kasm=- | FileCheck %s --check-prefix=NOSYNC
//
// Cross-unit synchronization, when `-kea-schedule` has not run.
//
// `-kea-tile` emits a correct SEQUENTIAL program (DIALECT_L2.md §4.6) and
// KEA-1 does not execute sequentially: five queues run concurrently and the
// only ordering between them is the 32 counting semaphores (ISA.md §5.3). So
// an unsynchronized stream is a set of data races. `--sync=auto` (the default)
// inserts exactly the edges the buffer dependencies imply -- and nothing else:
// no reordering, no hoisting, no double buffering. Those are `-kea-schedule`'s.

//===--------------------------------------------------------------------===//
// The RAW chain DMA -> MXU -> VPU -> DMA, and the WAR edge back
//===--------------------------------------------------------------------===//

func.func @raw() attributes {kea.dram_layout = {total_bytes = 65536 : i64,
    io_offset = 0 : i64, io_bytes = 8192 : i64, alignment = 64 : i64}} {
  %d = kea.alloc {name = "in", role = "input", addr = 0 : i64}
     : !kea.buffer<4096xi8, DRAM>
  %o = kea.alloc {name = "out", role = "output", addr = 4096 : i64}
     : !kea.buffer<1024xi8, DRAM>
  %a = kea.alloc {name = "at", role = "scratch", addr = 0 : i64}
     : !kea.buffer<2048xi8, A>
  %t = kea.alloc {name = "ot", role = "scratch", addr = 4096 : i64}
     : !kea.buffer<1040xi8, A>
  %w = kea.alloc {name = "wt", role = "scratch", addr = 0 : i64}
     : !kea.buffer<256xi8, W>
  %p = kea.alloc {name = "qt", role = "scratch", addr = 256 : i64}
     : !kea.buffer<192xi8, W>
  %q = kea.alloc {name = "acc", role = "scratch", addr = 0 : i64}
     : !kea.buffer<1024xi32, ACC>

  // One event per ordered unit pair, named after it, so the deadlock report
  // and the trace read in units rather than in numbers.
  // RAW: .event 0, "DMA0_to_MXU"
  // RAW: .event 1, "MXU_to_VPU"

  // RAW: DMA0  DMA_LD
  // RAW-NEXT: DMA0  SIGNAL  event=0, inc=1
  kea.dma_load %d -> %w {dram_addr = 0, spm_addr = 0, len0 = 256, n1 = 1,
                         n2 = 1, dram_s1 = 256, dram_s2 = 256, spm_s1 = 256,
                         spm_s2 = 256}
    : !kea.buffer<4096xi8, DRAM> -> !kea.buffer<256xi8, W>
  // RAW: DMA0  DMA_LD
  // RAW-NEXT: DMA0  SIGNAL  event=0, inc=1
  kea.dma_load %d -> %a {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
                         n2 = 1, dram_s1 = 1024, dram_s2 = 1024,
                         spm_s1 = 1024, spm_s2 = 1024}
    : !kea.buffer<4096xi8, DRAM> -> !kea.buffer<2048xi8, A>
  kea.dma_load %d -> %p {dram_addr = 0, spm_addr = 0, len0 = 192, n1 = 1,
                         n2 = 1, dram_s1 = 192, dram_s2 = 192, spm_s1 = 192,
                         spm_s2 = 192}
    : !kea.buffer<4096xi8, DRAM> -> !kea.buffer<192xi8, W>

  // The MXU waits for the weight load; then for the activation load. The
  // thresholds are CUMULATIVE over the pair's signal sequence, so the second
  // WAIT asks for 1 more count, not for 2 -- and a dependency an earlier WAIT
  // on the same queue already covered costs no instruction at all.
  // RAW: MXU   WAIT    event=0, threshold=1
  // RAW-NEXT: MXU   LOAD_W
  kea.load_w %w {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
                 bank = 0} : !kea.buffer<256xi8, W>
  // RAW: MXU   WAIT    event=0, threshold=1
  // RAW-NEXT: MXU   MATMUL
  // RAW-NEXT: MXU   SIGNAL  event=1, inc=1
  kea.mm %a, %q {a_addr = 0, a_inner_stride = 16, a_outer_stride = 128,
                 m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
                 acc_outer_stride = 128, bank = 0}
    : !kea.buffer<2048xi8, A>, !kea.buffer<1024xi32, ACC>

  // RAW: VPU   WAIT    event=1, threshold=1
  // RAW: VPU   VQUANT
  // RAW-NEXT: VPU   SIGNAL
  kea.vquant %q, %t, %p {acc_addr = 0, out_addr = 0, qparam_addr = 0,
                         num_pixels = 64, channels = 16, acc_pix_stride = 16,
                         out_pix_stride = 16, out_zp = -5, clamp_lo = -128,
                         clamp_hi = 127}
    : !kea.buffer<1024xi32, ACC>, !kea.buffer<1040xi8, A>, !kea.buffer<192xi8, W>

  // RAW: DMA0  WAIT
  // RAW-NEXT: DMA0  DMA_ST
  kea.dma_store %t -> %o {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
                          n2 = 1, dram_s1 = 1024, dram_s2 = 1024,
                          spm_s1 = 1024, spm_s2 = 1024}
    : !kea.buffer<1040xi8, A> -> !kea.buffer<1024xi8, DRAM>
  kea.halt
  return
}

// With `--sync=none` the same function comes out exactly as written: no
// semaphores, no `.event` block. That is the mode for diffing against the IR,
// and for IR `-kea-schedule` has already synchronized.
// NOSYNC-NOT: SIGNAL
// NOSYNC-NOT: WAIT
// NOSYNC-NOT: .event

//===--------------------------------------------------------------------===//
// Aliased storage: the case an SSA-keyed dependency analysis would miss
//===--------------------------------------------------------------------===//
//
// `q0` and `q1` are different `kea.alloc`s at the SAME ACC address, which is
// exactly what `-kea-alloc` does to two buffers whose block-order live ranges
// are disjoint (MEMORY_PLANNING.md §2.2). The DWCONV writing `q1` must not
// start before the VQUANT reading `q0` has finished, and nothing in the SSA
// graph says so -- only the addresses do.

func.func @aliased() attributes {kea.dram_layout = {total_bytes = 65536 : i64,
    alignment = 64 : i64}} {
  %a = kea.alloc {name = "aa", role = "scratch", addr = 0 : i64}
     : !kea.buffer<4096xi8, A>
  %t = kea.alloc {name = "at", role = "scratch", addr = 4096 : i64}
     : !kea.buffer<1040xi8, A>
  %w = kea.alloc {name = "aw", role = "scratch", addr = 0 : i64}
     : !kea.buffer<576xi8, W>
  %p = kea.alloc {name = "ap", role = "scratch", addr = 576 : i64}
     : !kea.buffer<192xi8, W>
  %q0 = kea.alloc {name = "acc0", role = "scratch", addr = 0 : i64}
      : !kea.buffer<1024xi32, ACC>
  %q1 = kea.alloc {name = "acc1", role = "scratch", addr = 0 : i64}
      : !kea.buffer<1024xi32, ACC>

  kea.load_w %w {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
                 bank = 0} : !kea.buffer<576xi8, W>
  kea.mm %a, %q0 {a_addr = 0, a_inner_stride = 16, a_outer_stride = 128,
                  m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
                  acc_outer_stride = 128, bank = 0}
    : !kea.buffer<4096xi8, A>, !kea.buffer<1024xi32, ACC>
  // ALIAS: VPU   WAIT    event={{[0-9]+}}, threshold=1
  // ALIAS-NEXT: VPU   VQUANT
  // The VPU tells the DWU it is done with those ACC words -- a WAR edge over
  // storage the two buffers share.
  // ALIAS-NEXT: VPU   SIGNAL
  kea.vquant %q0, %t, %p {acc_addr = 0, out_addr = 0, qparam_addr = 0,
                          num_pixels = 64, channels = 16, acc_pix_stride = 16,
                          out_pix_stride = 16, out_zp = 0, clamp_lo = -128,
                          clamp_hi = 127}
    : !kea.buffer<1024xi32, ACC>, !kea.buffer<1040xi8, A>, !kea.buffer<192xi8, W>
  // ALIAS: DWU   WAIT
  // ALIAS: DWU   DWCONV
  kea.dwconv %a, %w, %q1 {a_addr = 0, w_addr = 0, acc_addr = 0, out_h = 8,
                          out_w = 8, channels = 16, a_row_stride = 320,
                          a_pix_stride = 32, kernel = 3, stride = 1}
    : !kea.buffer<4096xi8, A>, !kea.buffer<576xi8, W>, !kea.buffer<1024xi32, ACC>
  kea.halt
  return
}

//===--------------------------------------------------------------------===//
// Already scheduled: pass the semaphores through untouched
//===--------------------------------------------------------------------===//
//
// `auto` inserts nothing when the IR already has `kea.signal` / `kea.wait`,
// because then `-kea-schedule` owns the synchronization and second-guessing it
// would break the very overlap it was hoisting DMA to create.

func.func @scheduled() attributes {kea.dram_layout = {total_bytes = 65536 : i64,
    io_offset = 0 : i64, io_bytes = 8192 : i64, alignment = 64 : i64}} {
  %d = kea.alloc {name = "sin", role = "input", addr = 0 : i64}
     : !kea.buffer<4096xi8, DRAM>
  %a = kea.alloc {name = "sa", role = "scratch", addr = 0 : i64}
     : !kea.buffer<2048xi8, A>
  %w = kea.alloc {name = "sw", role = "scratch", addr = 0 : i64}
     : !kea.buffer<256xi8, W>
  %q = kea.alloc {name = "sq", role = "scratch", addr = 0 : i64}
     : !kea.buffer<1024xi32, ACC>
  // SCHED: DMA1  DMA_LD
  kea.dma_load %d -> %a {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
                         n2 = 1, dram_s1 = 1024, dram_s2 = 1024,
                         spm_s1 = 1024, spm_s2 = 1024, unit = "DMA1"}
    : !kea.buffer<4096xi8, DRAM> -> !kea.buffer<2048xi8, A>
  // SCHED-NEXT: DMA1  SIGNAL  event=9, inc=1
  kea.signal 9 {unit = "DMA1"}
  // SCHED-NEXT: MXU   WAIT    event=9, threshold=1
  kea.wait 9 {unit = "MXU"}
  // SCHED-NEXT: MXU   LOAD_W
  kea.load_w %w {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
                 bank = 0} : !kea.buffer<256xi8, W>
  // SCHED-NEXT: MXU   MATMUL
  // SCHED-NEXT: CTRL  HALT
  kea.mm %a, %q {a_addr = 0, a_inner_stride = 16, a_outer_stride = 128,
                 m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
                 acc_outer_stride = 128, bank = 0}
    : !kea.buffer<2048xi8, A>, !kea.buffer<1024xi32, ACC>
  kea.halt
  return
}
