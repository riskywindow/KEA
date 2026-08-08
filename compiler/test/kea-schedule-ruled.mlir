// RUN: kea-opt %s -kea-schedule=mode=overlap | FileCheck %s
// RUN: kea-opt %s "-kea-schedule=mode=overlap report-schedule=true" | FileCheck %s --check-prefix=REPORT
// RUN: kea-opt %s -kea-schedule=mode=overlap | kea-opt -kea-tile | FileCheck %s --check-prefix=REVALIDATE
//
//===----------------------------------------------------------------------===//
// REGRESSION: Rule D on a program whose weight tiles have to rotate
//===----------------------------------------------------------------------===//
//
// The bug this pins was found by the end-to-end MobileNetV2 demo and reported
// as `demo/repro/run_repro.sh` defect 3: every lowerable prefix of the network
// at or beyond node 99 failed `checkRuleD()`, while the same prefix without
// `--schedule` compiled cleanly, and the threshold moved with
// `spm-reserve-factor`. docs/SCHEDULING.md §5.3 has the full account. The
// mechanism, in three steps:
//
//   1. When a space cannot hold all of its buffers at once, the scheduler adds
//      *rotation* edges (SCHEDULING.md §6.2) -- buffer i's first use is ordered
//      after buffer i-K's last use, the "buffer 0 free" handshake ISA.md §12
//      writes by hand. Those edges are added inside the capacity fixpoint,
//      after the dependence graph was built.
//   2. **The last user of an SPM_W weight tile is a `kea.load_w`.** So a
//      rotation edge makes the DMA that refills a later weight tile a direct
//      consumer of a LOAD_W -- and `protectWeightPairs()`, which normally
//      redirects such consumers onto the paired `kea.mm` so no SIGNAL is ever
//      slipped between a LOAD_W and its MATMUL, had already run and did not
//      see them.
//   3. The fix-up in `assignSync()` then moved that SIGNAL down onto the
//      MATMUL, where another SIGNAL on the same (MXU, DMAx) channel already
//      sat, and the two collapsed into one. Two waits, one token: Rule D
//      violated, and only on programs long enough for the rotation edges to
//      exist -- hence "it is program length, not any particular layer".
//
// Both halves of the fix are exercised here: `protectWeightPairs()` now runs
// again after the rotation edges are added, and a SIGNAL carries a *count*, so
// that even if two producers' tokens do land on one position they stay two
// tokens instead of one.
//
// Three 100000-byte weight tiles: any two fit in the 262144-byte SPM_W, three
// do not, so W's buffers-in-flight bound starts at 2, the capacity fixpoint
// tightens it to 1, and either way a later tile's `kea.dma_load` is ordered
// after an earlier tile's `kea.load_w` -- the edge that used to break Rule D.
// REPORT-LABEL: func.func @weight_tiles_rotate
// REPORT-SAME:  buffers_in_flight = [{{[0-9]+}}, 1, {{[0-9]+}}]
//
// The pass fails outright if Rule D does not hold -- `checkRuleD()` walks the
// finished block and simulates the counters -- so these RUN lines producing
// any output at all is the assertion. What the CHECK lines add is that the
// weight-bank invariant survived the repair: every `kea.load_w` is still
// immediately followed by its `kea.mm`, with no `kea.signal` or `kea.wait`
// between them, and the DMA0 -> MXU handshakes are intact.
//
// CHECK-LABEL: func.func @weight_tiles_rotate
// CHECK:       kea.load_w %{{.*}}bank = 0
// CHECK-NOT:   kea.signal
// CHECK-NOT:   kea.wait
// CHECK:       kea.mm %{{.*}}bank = 0
// CHECK:       kea.load_w %{{.*}}bank = 1
// CHECK-NOT:   kea.signal
// CHECK-NOT:   kea.wait
// CHECK:       kea.mm %{{.*}}bank = 1
// CHECK:       kea.load_w %{{.*}}bank = 0
// CHECK-NOT:   kea.signal
// CHECK-NOT:   kea.wait
// CHECK:       kea.mm %{{.*}}bank = 0
// CHECK:       kea.halt
//
// REVALIDATE re-runs `verifyWeightBanks()` over the scheduled output, so
// errata E7 is re-proved after all the motion (DIALECT_L2.md §2).
// REVALIDATE-LABEL: func.func @weight_tiles_rotate
// REVALIDATE: kea.mm

func.func @weight_tiles_rotate() {
  %dw = kea.alloc {name = "r.dw", role = "weights"} : !kea.buffer<300000xi8, DRAM>
  %din = kea.alloc {name = "r.in", role = "input"} : !kea.buffer<4096xi8, DRAM>
  %dout = kea.alloc {name = "r.out", role = "output"} : !kea.buffer<4096xi8, DRAM>
  %q = kea.alloc {name = "r.q", role = "scratch"} : !kea.buffer<192xi8, W>

  // Three weight tiles. Two fit in SPM_W at once; three do not.
  %w0 = kea.alloc {name = "r.w0", role = "scratch"} : !kea.buffer<100000xi8, W>
  %w1 = kea.alloc {name = "r.w1", role = "scratch"} : !kea.buffer<100000xi8, W>
  %w2 = kea.alloc {name = "r.w2", role = "scratch"} : !kea.buffer<100000xi8, W>

  %a0 = kea.alloc {name = "r.a0", role = "scratch"} : !kea.buffer<1040xi8, A>
  %c0 = kea.alloc {name = "r.c0", role = "scratch"} : !kea.buffer<1024xi32, ACC>
  %o0 = kea.alloc {name = "r.o0", role = "scratch"} : !kea.buffer<1040xi8, A>
  %a1 = kea.alloc {name = "r.a1", role = "scratch"} : !kea.buffer<1040xi8, A>
  %c1 = kea.alloc {name = "r.c1", role = "scratch"} : !kea.buffer<1024xi32, ACC>
  %o1 = kea.alloc {name = "r.o1", role = "scratch"} : !kea.buffer<1040xi8, A>
  %a2 = kea.alloc {name = "r.a2", role = "scratch"} : !kea.buffer<1040xi8, A>
  %c2 = kea.alloc {name = "r.c2", role = "scratch"} : !kea.buffer<1024xi32, ACC>
  %o2 = kea.alloc {name = "r.o2", role = "scratch"} : !kea.buffer<1040xi8, A>

  // ---- tile 0 ----
  kea.dma_load %dw -> %w0 {dram_addr = 0, spm_addr = 0, len0 = 50000, n1 = 2,
      n2 = 1, dram_s1 = 50000, dram_s2 = 0, spm_s1 = 50000, spm_s2 = 0}
    : !kea.buffer<300000xi8, DRAM> -> !kea.buffer<100000xi8, W>
  kea.dma_load %din -> %a0 {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
      n2 = 1, dram_s1 = 1024, dram_s2 = 1024, spm_s1 = 1024, spm_s2 = 1024}
    : !kea.buffer<4096xi8, DRAM> -> !kea.buffer<1040xi8, A>
  kea.load_w %w0 {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
      bank = 0} : !kea.buffer<100000xi8, W>
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

  // ---- tile 1 ----
  kea.dma_load %dw -> %w1 {dram_addr = 100000, spm_addr = 0, len0 = 50000,
      n1 = 2, n2 = 1, dram_s1 = 50000, dram_s2 = 0, spm_s1 = 50000, spm_s2 = 0}
    : !kea.buffer<300000xi8, DRAM> -> !kea.buffer<100000xi8, W>
  kea.dma_load %din -> %a1 {dram_addr = 1024, spm_addr = 0, len0 = 1024,
      n1 = 1, n2 = 1, dram_s1 = 1024, dram_s2 = 1024, spm_s1 = 1024,
      spm_s2 = 1024}
    : !kea.buffer<4096xi8, DRAM> -> !kea.buffer<1040xi8, A>
  kea.load_w %w1 {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
      bank = 1} : !kea.buffer<100000xi8, W>
  kea.mm %a1, %c1 {a_addr = 0, a_inner_stride = 16, a_outer_stride = 128,
      m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
      acc_outer_stride = 128, bank = 1}
    : !kea.buffer<1040xi8, A>, !kea.buffer<1024xi32, ACC>
  kea.vquant %c1, %o1, %q {acc_addr = 0, out_addr = 0, qparam_addr = 0,
      num_pixels = 64, channels = 16, acc_pix_stride = 16, out_pix_stride = 16,
      out_zp = -5, clamp_lo = -128, clamp_hi = 127}
    : !kea.buffer<1024xi32, ACC>, !kea.buffer<1040xi8, A>, !kea.buffer<192xi8, W>
  kea.dma_store %o1 -> %dout {dram_addr = 1024, spm_addr = 0, len0 = 1024,
      n1 = 1, n2 = 1, dram_s1 = 1024, dram_s2 = 1024, spm_s1 = 1024,
      spm_s2 = 1024}
    : !kea.buffer<1040xi8, A> -> !kea.buffer<4096xi8, DRAM>

  // ---- tile 2: its weight DMA is what the rotation edge orders after
  //      tile 0's kea.load_w ----
  kea.dma_load %dw -> %w2 {dram_addr = 200000, spm_addr = 0, len0 = 50000,
      n1 = 2, n2 = 1, dram_s1 = 50000, dram_s2 = 0, spm_s1 = 50000, spm_s2 = 0}
    : !kea.buffer<300000xi8, DRAM> -> !kea.buffer<100000xi8, W>
  kea.dma_load %din -> %a2 {dram_addr = 2048, spm_addr = 0, len0 = 1024,
      n1 = 1, n2 = 1, dram_s1 = 1024, dram_s2 = 1024, spm_s1 = 1024,
      spm_s2 = 1024}
    : !kea.buffer<4096xi8, DRAM> -> !kea.buffer<1040xi8, A>
  kea.load_w %w2 {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
      bank = 0} : !kea.buffer<100000xi8, W>
  kea.mm %a2, %c2 {a_addr = 0, a_inner_stride = 16, a_outer_stride = 128,
      m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
      acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1040xi8, A>, !kea.buffer<1024xi32, ACC>
  kea.vquant %c2, %o2, %q {acc_addr = 0, out_addr = 0, qparam_addr = 0,
      num_pixels = 64, channels = 16, acc_pix_stride = 16, out_pix_stride = 16,
      out_zp = -5, clamp_lo = -128, clamp_hi = 127}
    : !kea.buffer<1024xi32, ACC>, !kea.buffer<1040xi8, A>, !kea.buffer<192xi8, W>
  kea.dma_store %o2 -> %dout {dram_addr = 2048, spm_addr = 0, len0 = 1024,
      n1 = 1, n2 = 1, dram_s1 = 1024, dram_s2 = 1024, spm_s1 = 1024,
      spm_s2 = 1024}
    : !kea.buffer<1040xi8, A> -> !kea.buffer<4096xi8, DRAM>
  kea.halt
  return
}
