// RUN: kea-opt %s -split-input-file -kea-alloc | FileCheck %s
// RUN: kea-opt %s -split-input-file -kea-alloc | kea-opt -split-input-file -kea-alloc=verify-only=true | FileCheck %s --check-prefix=REVERIFY
//
// -kea-alloc: symbolic Level 2 buffers -> absolute, compile-time-constant
// addresses. docs/MEMORY_PLANNING.md is the write-up.
//
// The address is stamped as `kea.addr` on the `kea.alloc` rather than folded
// into the per-op displacements, because docs/DIALECT_L2.md §1.1(a) defines an
// address as a `(buffer, displacement)` pair and `-kea-emit` writes
// `base(X) + X_addr`. Folding would also break the Level 2 in-bounds verifiers,
// which check every strided walk against the *buffer*.
//
// The second RUN line is the point of the whole design: running the pass again
// in verify-only mode over its own output re-proves, from scratch, that no two
// overlapping-live-range buffers were given overlapping storage.

//===--------------------------------------------------------------------===//
// Live-range packing: disjoint ranges share storage, overlapping ones do not
//===--------------------------------------------------------------------===//
//
// Block positions and the live ranges they produce:
//
//   0  %a = alloc 1024           a: [0, 3]
//   1  %b = alloc 1024           b: [1, 3]
//   2  vcopy %a -> %b            a and b are both live here
//   3  vcopy %b -> %a
//   4  %c = alloc 1024           c: [4, 5]   -- starts after a and b are dead
//   5  vcopy fill %c
//   6  halt
//
// So {a, b} interfere and must not share; c interferes with neither and may
// reuse either one's storage. All three are the same size, so the greedy
// ordering falls through to longest-lived first: a, then b, then c.

// Peak is 2048, not the 3072 an allocator without packing would need, and it
// equals `maxlive` -- so on this function the greedy packer is provably optimal.
// CHECK-LABEL: func.func @share_and_separate
// CHECK-SAME:  spm_a = {buffers = 3 : i64, capacity = 262144 : i64, fragmentation = 0 : i64, maxlive = 2048 : i64, peak = 2048 : i64, unpacked = 3072 : i64}
func.func @share_and_separate() {
  // CHECK: kea.alloc {kea.addr = 0 : i64, live = array<i64: 0, 3>, name = "p.a"
  %a = kea.alloc {name = "p.a", role = "scratch"} : !kea.buffer<1024xi8, A>
  // %b overlaps %a, so it is pushed past the end of it: 0 + 1024 = 1024.
  // CHECK: kea.alloc {kea.addr = 1024 : i64, live = array<i64: 1, 3>, name = "p.b"
  %b = kea.alloc {name = "p.b", role = "scratch"} : !kea.buffer<1024xi8, A>

  kea.vcopy from %a : !kea.buffer<1024xi8, A>, to %b
    {src_addr = 0, dst_addr = 0, row_bytes = 1024, rows = 1,
     src_row_stride = 0, dst_row_stride = 0} : !kea.buffer<1024xi8, A>
  kea.vcopy from %b : !kea.buffer<1024xi8, A>, to %a
    {src_addr = 0, dst_addr = 0, row_bytes = 1024, rows = 1,
     src_row_stride = 0, dst_row_stride = 0} : !kea.buffer<1024xi8, A>

  // THE POINT: %c is born after %a died, so it lands on top of %a at 0. This
  // one line is the difference between a 3 KiB program and a 2 KiB one, and on
  // a real network it is the difference between fitting and not.
  // CHECK: kea.alloc {kea.addr = 0 : i64, live = array<i64: 4, 5>, name = "p.c"
  %c = kea.alloc {name = "p.c", role = "scratch"} : !kea.buffer<1024xi8, A>
  kea.vcopy to %c {fill, fill_value = 0, src_addr = 0, dst_addr = 0,
                   row_bytes = 1024, rows = 1, src_row_stride = 0,
                   dst_row_stride = 0} : !kea.buffer<1024xi8, A>
  kea.halt
  return
}

// REVERIFY-LABEL: func.func @share_and_separate
// REVERIFY: kea.addr = 0 : i64
// REVERIFY: kea.addr = 1024 : i64
// REVERIFY: kea.addr = 0 : i64

// -----

//===--------------------------------------------------------------------===//
// ACC is addressed in int32 WORDS, and every base is a multiple of 16 words
//===--------------------------------------------------------------------===//
//
// ISA.md §11.1: `MATMUL.acc_addr` and `DWCONV.acc_addr` are multiples of 16
// *words*. The op verifiers only see the displacement; an aligned displacement
// implies an aligned `base + displacement` only if the base is aligned too,
// which is this pass's job.
//
// `acc.big` is deliberately 1030 words -- NOT a multiple of 16 -- so that the
// rounding is observable. 1030 rounded up to a multiple of 16 is 1040. A pass
// that confused words with bytes, or that used the 16-*byte* SPM alignment
// here, could not produce 1040.

// Capacity and peak are in words too: 32768 words == 128 KiB. The 10 words of
// fragmentation are exactly the 1030 -> 1040 round up.
// CHECK-LABEL: func.func @acc_is_words
// CHECK-SAME:  acc = {buffers = 2 : i64, capacity = 32768 : i64, fragmentation = 10 : i64, maxlive = 1542 : i64, peak = 1552 : i64
func.func @acc_is_words() {
  %a = kea.alloc {name = "w.a", role = "scratch"} : !kea.buffer<1616xi8, A>
  %wt = kea.alloc {name = "w.wt", role = "scratch"} : !kea.buffer<256xi8, W>

  // CHECK: kea.alloc {kea.addr = 0 : i64, {{.*}}name = "w.big"{{.*}} : !kea.buffer<1030xi32, ACC>
  %big = kea.alloc {name = "w.big", role = "scratch"} : !kea.buffer<1030xi32, ACC>
  // 1030 is not a multiple of 16, so the next ACC base is 1040, not 1030.
  // CHECK: kea.alloc {kea.addr = 1040 : i64, {{.*}}name = "w.small"{{.*}} : !kea.buffer<512xi32, ACC>
  %small = kea.alloc {name = "w.small", role = "scratch"} : !kea.buffer<512xi32, ACC>

  kea.load_w %wt {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
                  bank = 0} : !kea.buffer<256xi8, W>
  kea.mm %a, %big {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                   m_inner = 8, m_outer = 8, acc_addr = 0,
                   acc_inner_stride = 16, acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1616xi8, A>, !kea.buffer<1030xi32, ACC>
  kea.mm %a, %small {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                     m_inner = 4, m_outer = 8, acc_addr = 0,
                     acc_inner_stride = 16, acc_outer_stride = 64, bank = 0}
    : !kea.buffer<1616xi8, A>, !kea.buffer<512xi32, ACC>
  kea.halt
  return
}

// -----

//===--------------------------------------------------------------------===//
// Alignment is DERIVED from the ops that use the buffer, not assumed
//===--------------------------------------------------------------------===//
//
// Both buffers below declare `alignment = 4`, the weakest thing SPM_W ever
// needs (`VQUANT.qparam_addr`, ISA.md §11.1). But `w.tile` is fed to `LOAD_W`,
// which requires 16, so the pass must strengthen it -- the declared alignment
// is a floor, not the answer.
//
// `w.params` is 600 bytes and goes down first (greedy by size). `w.tile` then
// has to clear it: at alignment 4 that would be 600, at the LOAD_W alignment of
// 16 it is 608. The 608 is the whole test.

// CHECK-LABEL: func.func @alignment_is_derived
func.func @alignment_is_derived() {
  %acc = kea.alloc {name = "d.acc", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  %out = kea.alloc {name = "d.out", role = "scratch"} : !kea.buffer<2064xi8, A>
  %act = kea.alloc {name = "d.act", role = "scratch"} : !kea.buffer<1616xi8, A>

  // Only ever a VQUANT qparam block: 4-byte alignment is enough, and 0 is.
  // CHECK: kea.alloc {alignment = 4 : i64, kea.addr = 0 : i64, {{.*}}name = "d.params"
  %params = kea.alloc {name = "d.params", role = "scratch", alignment = 4}
    : !kea.buffer<600xi8, W>
  // Fed to LOAD_W: 16 bytes, so 608 rather than the declared 4's 600.
  // CHECK: kea.alloc {alignment = 4 : i64, kea.addr = 608 : i64, {{.*}}name = "d.tile"
  %tile = kea.alloc {name = "d.tile", role = "scratch", alignment = 4}
    : !kea.buffer<512xi8, W>

  kea.load_w %tile {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
                    bank = 0} : !kea.buffer<512xi8, W>
  kea.mm %act, %acc {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                     m_inner = 8, m_outer = 8, acc_addr = 0,
                     acc_inner_stride = 16, acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
  kea.vquant %acc, %out, %params {acc_addr = 0, out_addr = 0, qparam_addr = 4,
                                  num_pixels = 64, channels = 16,
                                  acc_pix_stride = 16, out_pix_stride = 32,
                                  out_zp = -128, clamp_lo = -128,
                                  clamp_hi = 127}
    : !kea.buffer<2048xi32, ACC>, !kea.buffer<2064xi8, A>, !kea.buffer<600xi8, W>
  kea.halt
  return
}

// -----

//===--------------------------------------------------------------------===//
// The DRAM arena: three regions, and the activation region is packed too
//===--------------------------------------------------------------------===//
//
// `KeafDramLayout` (include/kea/keaf.h, ARTIFACT_FORMAT.md §5) splits the arena
// into constants, host I/O and intermediate-activation scratch. Constants and
// I/O are laid out once and never packed -- a constant is live for the whole
// program and the host binds an I/O tensor by name. Activations *are* packed,
// on their `[first use, last use]` range: a DRAM symbol holds nothing before
// the first instruction writes it, and `-kea-tile` hoists every symbol to the
// top of the function, so keying on the declaration would make every range
// overlap every other and defeat the packing entirely.
//
//   layer 0 writes @l0, layer 1 reads @l0 and writes @l1, layer 2 reads @l1.
//   So @l0 and @l1 have disjoint ranges and land on the SAME address. That is
//   the property that makes a 50-layer network need two activation buffers
//   rather than 50.

// The arena the runtime must provide, field for field with KeafDramLayout.
// Two 1 KiB activations occupy 1 KiB of it, not 2.
// CHECK-LABEL: func.func @dram_arena
// CHECK-SAME:  kea.dram_layout = {alignment = 64 : i64, const_bytes = 512 : i64, const_offset = 0 : i64, io_bytes = 512 : i64, io_offset = 512 : i64, scratch_bytes = 1024 : i64, scratch_offset = 1024 : i64, total_bytes = 2048 : i64}
// CHECK-SAME:  dram_scratch = {buffers = 2 : i64, capacity = 1024 : i64, fragmentation = 0 : i64, maxlive = 1024 : i64, peak = 1024 : i64, unpacked = 2048 : i64}
func.func @dram_arena() {
  // Region 2 (I/O) starts at 64-byte alignment past the constants: 512 -> 512.
  // CHECK: kea.alloc {kea.addr = 512 : i64, name = "m.in", role = "input"}
  %in = kea.alloc {name = "m.in", role = "input"} : !kea.buffer<256xi8, DRAM>
  // CHECK: kea.alloc {kea.addr = 768 : i64, name = "m.out", role = "output"}
  %out = kea.alloc {name = "m.out", role = "output"} : !kea.buffer<256xi8, DRAM>
  // Region 1 (constants), laid out in program order from 0.
  // CHECK: kea.alloc {kea.addr = 0 : i64, {{.*}}name = "m.weights", role = "weights"}
  %w = kea.alloc {name = "m.weights", role = "weights"} : !kea.buffer<512xi8, DRAM>

  // Region 3 (activation scratch), 64-byte aligned past the I/O region.
  // CHECK: kea.alloc {kea.addr = 1024 : i64, name = "m.l0", role = "activation"}
  %l0 = kea.alloc {name = "m.l0", role = "activation"} : !kea.buffer<1024xi8, DRAM>
  // SAME ADDRESS as m.l0: their live ranges do not overlap.
  // CHECK: kea.alloc {kea.addr = 1024 : i64, name = "m.l1", role = "activation"}
  %l1 = kea.alloc {name = "m.l1", role = "activation"} : !kea.buffer<1024xi8, DRAM>

  %sa = kea.alloc {name = "m.sa", role = "scratch"} : !kea.buffer<1024xi8, A>

  // layer 0: in -> l0
  kea.dma_load %in -> %sa {dram_addr = 0, spm_addr = 0, len0 = 256, n1 = 1,
                           n2 = 1, dram_s1 = 256, dram_s2 = 256, spm_s1 = 256,
                           spm_s2 = 256}
    : !kea.buffer<256xi8, DRAM> -> !kea.buffer<1024xi8, A>
  kea.dma_store %sa -> %l0 {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
                            n2 = 1, dram_s1 = 1024, dram_s2 = 1024,
                            spm_s1 = 1024, spm_s2 = 1024}
    : !kea.buffer<1024xi8, A> -> !kea.buffer<1024xi8, DRAM>
  // layer 1: l0 -> l1. m.l0's last use, m.l1's first.
  kea.dma_load %l0 -> %sa {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
                           n2 = 1, dram_s1 = 1024, dram_s2 = 1024,
                           spm_s1 = 1024, spm_s2 = 1024}
    : !kea.buffer<1024xi8, DRAM> -> !kea.buffer<1024xi8, A>
  kea.dma_store %sa -> %l1 {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
                            n2 = 1, dram_s1 = 1024, dram_s2 = 1024,
                            spm_s1 = 1024, spm_s2 = 1024}
    : !kea.buffer<1024xi8, A> -> !kea.buffer<1024xi8, DRAM>
  // layer 2: l1 -> out
  kea.dma_load %l1 -> %sa {dram_addr = 0, spm_addr = 0, len0 = 1024, n1 = 1,
                           n2 = 1, dram_s1 = 1024, dram_s2 = 1024,
                           spm_s1 = 1024, spm_s2 = 1024}
    : !kea.buffer<1024xi8, DRAM> -> !kea.buffer<1024xi8, A>
  kea.dma_store %sa -> %out {dram_addr = 0, spm_addr = 0, len0 = 256, n1 = 1,
                             n2 = 1, dram_s1 = 256, dram_s2 = 256,
                             spm_s1 = 256, spm_s2 = 256}
    : !kea.buffer<1024xi8, A> -> !kea.buffer<256xi8, DRAM>
  kea.halt
  return
}

// -----

//===--------------------------------------------------------------------===//
// The SPM allocation map (KeafSpmEntry / model.map.json `spm_map`)
//===--------------------------------------------------------------------===//
//
// ARTIFACT_FORMAT.md §5 and ASSEMBLY.md §7.5. Nothing at run time consults it --
// every address in the stream is already absolute -- it exists so that
// `kea-dis --annotate` can put a name to an address.
//
// `first_pc`/`last_pc` are INSTRUCTION indices, not the block positions `live`
// uses: `KeafSpmEntry` defines them as "instruction index that touches it", and
// `kea.alloc` is not an instruction. Here the two allocs occupy block positions
// 0 and 1 but the first instruction is pc 0.

// CHECK-LABEL: func.func @spm_map
// CHECK-SAME:  kea.spm_map = [
// CHECK-SAME:  {first_pc = 0 : i64, last_pc = 0 : i64, name = "s.src", offset = 0 : i64, size = 1024 : i64, space = "SPM_A"}
// CHECK-SAME:  {first_pc = 0 : i64, last_pc = 0 : i64, name = "s.dst", offset = 1024 : i64, size = 512 : i64, space = "SPM_A"}
func.func @spm_map() {
  %a = kea.alloc {name = "s.src", role = "scratch"} : !kea.buffer<1024xi8, A>
  %b = kea.alloc {name = "s.dst", role = "scratch"} : !kea.buffer<512xi8, A>
  kea.vcopy from %a : !kea.buffer<1024xi8, A>, to %b
    {src_addr = 0, dst_addr = 0, row_bytes = 512, rows = 1,
     src_row_stride = 0, dst_row_stride = 0} : !kea.buffer<512xi8, A>
  kea.trace "end" 0
  kea.halt
  return
}
