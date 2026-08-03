// RUN: kea-opt %s -split-input-file -kea-alloc -verify-diagnostics
//
// Programs -kea-alloc cannot fit, and the diagnostic each one produces.
//
// "Out of memory" alone is useless to whoever has to fix the tiling, so every
// refusal names the space, the peak demand, the capacity, and which buffers are
// holding the space at the peak -- and says whether the shortfall is
// fundamental (the live data genuinely does not fit, so retile) or
// fragmentation this packer introduced (the same data would fit under a better
// placement).
//
// The companion file kea-alloc-verify.mlir covers the other way this pass
// refuses: a placement that aliases.

//===--------------------------------------------------------------------===//
// 1. Out of capacity -- fundamental. No allocator could do this.
//===--------------------------------------------------------------------===//
//
// Three 200000-byte SPM_A buffers, chained so that all three are simultaneously
// live: 600000 bytes against a 262144-byte scratchpad. Packing cannot help, and
// the diagnostic says so rather than leaving the reader to work it out.

func.func @spm_a_exhausted() {
  %a = kea.alloc {name = "big.a", role = "scratch"} : !kea.buffer<200000xi8, A>
  // expected-error @+3 {{out of SPM_A: cannot place 'big.b' (200000 bytes, live [1, 4]) -- SPM_A holds 262144 bytes}}
  // expected-note @+2 {{peak demand is 600000 bytes at block position 2, which is 337856 bytes over the 262144 bytes capacity. No allocator can fit this: those buffers hold live data at the same instant. Re-run -kea-tile with smaller tiles}}
  // expected-note @+1 {{live at block position 2: 'big.a' 200000, 'big.b' 200000, 'big.c' 200000 bytes}}
  %b = kea.alloc {name = "big.b", role = "scratch"} : !kea.buffer<200000xi8, A>
  %c = kea.alloc {name = "big.c", role = "scratch"} : !kea.buffer<200000xi8, A>
  kea.vcopy from %a : !kea.buffer<200000xi8, A>, to %b
    {src_addr = 0, dst_addr = 0, row_bytes = 100, rows = 1,
     src_row_stride = 0, dst_row_stride = 0} : !kea.buffer<200000xi8, A>
  kea.vcopy from %b : !kea.buffer<200000xi8, A>, to %c
    {src_addr = 0, dst_addr = 0, row_bytes = 100, rows = 1,
     src_row_stride = 0, dst_row_stride = 0} : !kea.buffer<200000xi8, A>
  kea.vcopy from %c : !kea.buffer<200000xi8, A>, to %a
    {src_addr = 0, dst_addr = 0, row_bytes = 100, rows = 1,
     src_row_stride = 0, dst_row_stride = 0} : !kea.buffer<200000xi8, A>
  kea.halt
  return
}

// -----

//===--------------------------------------------------------------------===//
// 1b. Out of capacity in ACC -- and the numbers are WORDS, not bytes
//===--------------------------------------------------------------------===//
//
// ACC holds 32768 int32 words (128 KiB). Two 20000-word regions is 40000 words,
// over capacity -- but only 160000 bytes, which would fit comfortably if the
// unit were bytes. A pass that mixed the two up would accept this program.

func.func @acc_exhausted() {
  %a = kea.alloc {name = "acc.a", role = "scratch"} : !kea.buffer<1616xi8, A>
  %q0 = kea.alloc {name = "acc.q0", role = "scratch"} : !kea.buffer<20000xi32, ACC>
  // expected-error @+3 {{out of ACC: cannot place 'acc.q1' (20000 int32 words, live [2, 4]) -- ACC holds 32768 int32 words}}
  // expected-note @+2 {{peak demand is 40000 int32 words at block position 2, which is 7232 int32 words over the 32768 int32 words capacity}}
  // expected-note @+1 {{live at block position 2: 'acc.q0' 20000, 'acc.q1' 20000 int32 words}}
  %q1 = kea.alloc {name = "acc.q1", role = "scratch"} : !kea.buffer<20000xi32, ACC>
  kea.mm %a, %q0 {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                  m_inner = 8, m_outer = 8, acc_addr = 0,
                  acc_inner_stride = 16, acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1616xi8, A>, !kea.buffer<20000xi32, ACC>
  kea.mm %a, %q1 {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                  m_inner = 8, m_outer = 8, acc_addr = 0,
                  acc_inner_stride = 16, acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1616xi8, A>, !kea.buffer<20000xi32, ACC>
  kea.halt
  return
}
