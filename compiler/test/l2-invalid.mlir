// RUN: kea-opt %s -split-input-file -verify-diagnostics
//
// Every alignment and range rule the KEA-1 ISA imposes on a Level 2 op, one
// case each. These are the rules `kea::keaValidate()` in include/kea/isa.h
// enforces at assembly time; catching them here means the diagnostic points at
// a readable op instead of at a line of .kasm.
//
// NOTE: `expected-error {{...}}` is a literal SUBSTRING match, not a regex.

//===--------------------------------------------------------------------===//
// LOAD_W: w_addr / w_row_stride are multiples of 16 bytes (8 in int4)
//===--------------------------------------------------------------------===//

func.func @loadw_addr_misaligned() {
  %w = kea.alloc {name = "w", role = "scratch"} : !kea.buffer<4608xi8, W>
  // expected-error @+1 {{w_addr (8) must be a multiple of 16 bytes}}
  kea.load_w %w {w_addr = 8, w_row_stride = 16, k_rows = 16, n_cols = 16,
                 bank = 0} : !kea.buffer<4608xi8, W>
  return
}

// -----

func.func @loadw_stride_misaligned() {
  %w = kea.alloc {name = "w", role = "scratch"} : !kea.buffer<4608xi8, W>
  // expected-error @+1 {{w_row_stride (24) must be a multiple of 16 bytes}}
  kea.load_w %w {w_addr = 0, w_row_stride = 24, k_rows = 16, n_cols = 16,
                 bank = 0} : !kea.buffer<4608xi8, W>
  return
}

// -----

func.func @loadw_int4_addr_misaligned() {
  %w = kea.alloc {name = "w", role = "scratch"} : !kea.buffer<4608xi8, W>
  // int4 halves the requirement to 8 bytes -- but 4 is still wrong.
  // expected-error @+1 {{w_addr (4) must be a multiple of 8 bytes}}
  kea.load_w %w {w_addr = 4, w_row_stride = 8, k_rows = 16, n_cols = 16,
                 bank = 0, int4} : !kea.buffer<4608xi8, W>
  return
}

// -----

func.func @loadw_krows_range() {
  %w = kea.alloc {name = "w", role = "scratch"} : !kea.buffer<4608xi8, W>
  // expected-error @+1 {{k_rows must be in [1, 16], got 17}}
  kea.load_w %w {w_addr = 0, w_row_stride = 16, k_rows = 17, n_cols = 16,
                 bank = 0} : !kea.buffer<4608xi8, W>
  return
}

//===--------------------------------------------------------------------===//
// MATMUL: every ACC address and stride is a multiple of 16 WORDS
//===--------------------------------------------------------------------===//

// -----

func.func @mm_acc_addr_misaligned() {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<1616xi8, A>
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  // expected-error @+1 {{acc_addr (8) must be a multiple of 16 words}}
  kea.mm %a, %q {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                 m_inner = 8, m_outer = 8, acc_addr = 8, acc_inner_stride = 16,
                 acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
  return
}

// -----

func.func @mm_acc_inner_stride_misaligned() {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<1616xi8, A>
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  // expected-error @+1 {{acc_inner_stride (24) must be a multiple of 16 words}}
  kea.mm %a, %q {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                 m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 24,
                 acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
  return
}

// -----

func.func @mm_acc_outer_stride_misaligned() {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<1616xi8, A>
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  // expected-error @+1 {{acc_outer_stride (130) must be a multiple of 16 words}}
  kea.mm %a, %q {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                 m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
                 acc_outer_stride = 130, bank = 0}
    : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
  return
}

// -----

func.func @mm_rows_exceed_acc() {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<1616xi8, A>
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  // 64*64 = 4096 rows * 16 lanes = 65536 int32 words, twice the whole ACC.
  // expected-error @+1 {{m_inner * m_outer = 4096 exceeds KEA_MXU_MAX_ROWS (2048)}}
  kea.mm %a, %q {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                 m_inner = 64, m_outer = 64, acc_addr = 0,
                 acc_inner_stride = 16, acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
  return
}

// -----

func.func @mm_reads_past_activation_tile() {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<256xi8, A>
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  // An 8x8x4 dense tile is 256 B, and the last row starts at 7*32 + 7*4 = 252.
  // The array ALWAYS reads 16 activation bytes whatever the resident tile's
  // k_rows (ISA.md §7.3), so it reads to 268: this tile has to be
  // over-allocated to 272, which is exactly what -kea-tile does.
  // expected-error @+1 {{activation walk accesses [0, 268) of a SPM_A buffer of 256 bytes}}
  kea.mm %a, %q {a_addr = 0, a_inner_stride = 4, a_outer_stride = 32,
                 m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
                 acc_outer_stride = 128, bank = 0}
    : !kea.buffer<256xi8, A>, !kea.buffer<2048xi32, ACC>
  return
}

// -----

func.func @mm_wrong_space(%q: !kea.buffer<2048xi32, ACC>) {
  %w = kea.alloc {name = "w", role = "scratch"} : !kea.buffer<1616xi8, W>
  // The address space is an ODS type constraint, not a verifier check.
  // expected-error @+1 {{operand #0 must be buffer in the activation scratchpad}}
  kea.mm %w, %q {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                 m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
                 acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1616xi8, W>, !kea.buffer<2048xi32, ACC>
  return
}

//===--------------------------------------------------------------------===//
// DWCONV: channels a multiple of 16, acc_addr a multiple of 16 words
//===--------------------------------------------------------------------===//

// -----

func.func @dwconv_channels_not_multiple_of_16() {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<3216xi8, A>
  %w = kea.alloc {name = "w", role = "scratch"} : !kea.buffer<288xi8, W>
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  // expected-error @+1 {{channels (24) must be a multiple of 16 channels}}
  kea.dwconv %a, %w, %q {a_addr = 0, w_addr = 0, acc_addr = 0, out_h = 8,
                         out_w = 8, channels = 24, a_row_stride = 320,
                         a_pix_stride = 32, kernel = 3, stride = 1}
    : !kea.buffer<3216xi8, A>, !kea.buffer<288xi8, W>, !kea.buffer<2048xi32, ACC>
  return
}

// -----

func.func @dwconv_acc_addr_misaligned() {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<3216xi8, A>
  %w = kea.alloc {name = "w", role = "scratch"} : !kea.buffer<288xi8, W>
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<4096xi32, ACC>
  // expected-error @+1 {{acc_addr (4) must be a multiple of 16 words}}
  kea.dwconv %a, %w, %q {a_addr = 0, w_addr = 0, acc_addr = 4, out_h = 8,
                         out_w = 8, channels = 32, a_row_stride = 320,
                         a_pix_stride = 32, kernel = 3, stride = 1}
    : !kea.buffer<3216xi8, A>, !kea.buffer<288xi8, W>, !kea.buffer<4096xi32, ACC>
  return
}

// -----

func.func @dwconv_bad_kernel() {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<3216xi8, A>
  %w = kea.alloc {name = "w", role = "scratch"} : !kea.buffer<288xi8, W>
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  // expected-error @+1 {{kernel must be 3 or 5, got 7}}
  kea.dwconv %a, %w, %q {a_addr = 0, w_addr = 0, acc_addr = 0, out_h = 8,
                         out_w = 8, channels = 32, a_row_stride = 320,
                         a_pix_stride = 32, kernel = 7, stride = 1}
    : !kea.buffer<3216xi8, A>, !kea.buffer<288xi8, W>, !kea.buffer<2048xi32, ACC>
  return
}

//===--------------------------------------------------------------------===//
// VQUANT: channels % 16, acc_addr / acc_pix_stride % 16 words, qparam_addr % 4
//===--------------------------------------------------------------------===//

// -----

func.func @vquant_channels_not_multiple_of_16() {
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<2064xi8, A>
  %p = kea.alloc {name = "p", role = "scratch"} : !kea.buffer<384xi8, W>
  // expected-error @+1 {{channels (24) must be a multiple of 16 channels}}
  kea.vquant %q, %a, %p {acc_addr = 0, out_addr = 0, qparam_addr = 0,
                         num_pixels = 8, channels = 24, acc_pix_stride = 32,
                         out_pix_stride = 32, out_zp = -128, clamp_lo = -128,
                         clamp_hi = 127}
    : !kea.buffer<2048xi32, ACC>, !kea.buffer<2064xi8, A>, !kea.buffer<384xi8, W>
  return
}

// -----

func.func @vquant_acc_pix_stride_misaligned() {
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<2064xi8, A>
  %p = kea.alloc {name = "p", role = "scratch"} : !kea.buffer<384xi8, W>
  // expected-error @+1 {{acc_pix_stride (20) must be a multiple of 16 words}}
  kea.vquant %q, %a, %p {acc_addr = 0, out_addr = 0, qparam_addr = 0,
                         num_pixels = 8, channels = 16, acc_pix_stride = 20,
                         out_pix_stride = 16, out_zp = -128, clamp_lo = -128,
                         clamp_hi = 127}
    : !kea.buffer<2048xi32, ACC>, !kea.buffer<2064xi8, A>, !kea.buffer<384xi8, W>
  return
}

// -----

func.func @vquant_qparam_addr_misaligned() {
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<2064xi8, A>
  %p = kea.alloc {name = "p", role = "scratch"} : !kea.buffer<384xi8, W>
  // expected-error @+1 {{qparam_addr (6) must be a multiple of 4 bytes}}
  kea.vquant %q, %a, %p {acc_addr = 0, out_addr = 0, qparam_addr = 6,
                         num_pixels = 8, channels = 16, acc_pix_stride = 16,
                         out_pix_stride = 16, out_zp = -128, clamp_lo = -128,
                         clamp_hi = 127}
    : !kea.buffer<2048xi32, ACC>, !kea.buffer<2064xi8, A>, !kea.buffer<384xi8, W>
  return
}

// -----

func.func @vquant_clamps_inverted() {
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<2064xi8, A>
  %p = kea.alloc {name = "p", role = "scratch"} : !kea.buffer<384xi8, W>
  // expected-error @+1 {{clamp_lo (10) > clamp_hi (-10)}}
  kea.vquant %q, %a, %p {acc_addr = 0, out_addr = 0, qparam_addr = 0,
                         num_pixels = 8, channels = 16, acc_pix_stride = 16,
                         out_pix_stride = 16, out_zp = 0, clamp_lo = 10,
                         clamp_hi = -10}
    : !kea.buffer<2048xi32, ACC>, !kea.buffer<2064xi8, A>, !kea.buffer<384xi8, W>
  return
}

//===--------------------------------------------------------------------===//
// VADD / VPOOL / VCOPY
//===--------------------------------------------------------------------===//

// -----

func.func @vadd_param_addr_misaligned() {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<1024xi8, A>
  %b = kea.alloc {name = "b", role = "scratch"} : !kea.buffer<1024xi8, A>
  %p = kea.alloc {name = "p", role = "scratch"} : !kea.buffer<64xi8, W>
  // expected-error @+1 {{param_addr (2) must be a multiple of 4 bytes}}
  kea.vadd %a, %b, %a, %p {a_addr = 0, b_addr = 0, out_addr = 0,
                           param_addr = 2, num_elems = 1024, clamp_lo = -128,
                           clamp_hi = 127}
    : !kea.buffer<1024xi8, A>, !kea.buffer<1024xi8, A>, !kea.buffer<1024xi8, A>,
      !kea.buffer<64xi8, W>
  return
}

// -----

func.func @vpool_kernel_range() {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<4096xi8, A>
  %b = kea.alloc {name = "b", role = "scratch"} : !kea.buffer<4096xi8, A>
  // expected-error @+1 {{kh must be in [1, 32], got 33}}
  kea.vpool %a, %b {in_addr = 0, out_addr = 0, out_h = 1, out_w = 1,
                    channels = 16, kh = 33, kw = 7, stride_h = 1,
                    stride_w = 1, in_row_stride = 128, out_row_stride = 16}
    : !kea.buffer<4096xi8, A>, !kea.buffer<4096xi8, A>
  return
}

// -----

func.func @vcopy_fill_with_source() {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<1024xi8, A>
  %b = kea.alloc {name = "b", role = "scratch"} : !kea.buffer<1024xi8, A>
  // expected-error @+1 {{fill mode ignores the source}}
  kea.vcopy from %a : !kea.buffer<1024xi8, A>, to %b
    {fill, fill_value = -128, src_addr = 0, dst_addr = 0, row_bytes = 1024,
     rows = 1, src_row_stride = 0, dst_row_stride = 1024}
    : !kea.buffer<1024xi8, A>
  return
}

// -----

func.func @vcopy_copy_without_source() {
  %b = kea.alloc {name = "b", role = "scratch"} : !kea.buffer<1024xi8, A>
  // expected-error @+1 {{copy mode needs a source operand}}
  kea.vcopy to %b {src_addr = 0, dst_addr = 0, row_bytes = 1024, rows = 1,
                   src_row_stride = 0, dst_row_stride = 1024}
    : !kea.buffer<1024xi8, A>
  return
}

//===--------------------------------------------------------------------===//
// DMA
//===--------------------------------------------------------------------===//

// -----

func.func @dma_n2_out_of_range() {
  %d = kea.alloc {name = "x", role = "input"} : !kea.buffer<1048576xi8, DRAM>
  %a = kea.alloc {name = "t", role = "scratch"} : !kea.buffer<65536xi8, A>
  // n2 is the `aux` byte of the instruction header: 1..255.
  // expected-error @+1 {{n2 must be in [1, 255], got 256}}
  kea.dma_load %d -> %a {dram_addr = 0, spm_addr = 0, len0 = 16, n1 = 1,
                         n2 = 256, dram_s1 = 16, dram_s2 = 16, spm_s1 = 16,
                         spm_s2 = 16}
    : !kea.buffer<1048576xi8, DRAM> -> !kea.buffer<65536xi8, A>
  return
}

// -----

func.func @dma_store_zero_dram_stride() {
  %d = kea.alloc {name = "y", role = "output"} : !kea.buffer<4096xi8, DRAM>
  %a = kea.alloc {name = "t", role = "scratch"} : !kea.buffer<4096xi8, A>
  // expected-error @+1 {{a zero DRAM stride on the destination side of a DMA is last-writer-wins}}
  kea.dma_store %a -> %d {dram_addr = 0, spm_addr = 0, len0 = 16, n1 = 8,
                          n2 = 1, dram_s1 = 0, dram_s2 = 0, spm_s1 = 16,
                          spm_s2 = 0}
    : !kea.buffer<4096xi8, A> -> !kea.buffer<4096xi8, DRAM>
  return
}

//===--------------------------------------------------------------------===//
// kea.alloc and the CTRL ops
//===--------------------------------------------------------------------===//

// -----

func.func @alloc_bigger_than_the_scratchpad() {
  // SPM_A is 256 KiB (KEA_SPM_A_BYTES).
  // expected-error @+1 {{needs 262145 elements of SPM_A, which holds 262144}}
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<262145xi8, A>
  return
}

// -----

func.func @alloc_unknown_role() {
  // expected-error @+1 {{unknown role "spill"}}
  %a = kea.alloc {name = "a", role = "spill"} : !kea.buffer<1024xi8, A>
  return
}

// -----

func.func @alloc_dram_live_range() {
  // A DRAM buffer is a symbol resolved by the assembler, not an allocation.
  // expected-error @+1 {{a DRAM buffer has no live range}}
  %a = kea.alloc {name = "a", role = "weights", live = array<i64: 0, 1>}
     : !kea.buffer<1024xi8, DRAM>
  return
}

// -----

func.func @alloc_acc_must_be_i32() {
  // ACC is addressed in int32 words; an i8 ACC buffer would make every offset
  // in this dialect ambiguous. Caught by BufferType itself.
  // expected-error @+1 {{ACC buffers must have a 32-bit element type}}
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<1024xi8, ACC>
  return
}

// -----

func.func @halt_not_last() {
  // expected-error @+1 {{must be the last instruction in the block}}
  kea.halt
  kea.trace "marker" 0
  return
}

// -----

func.func @event_id_out_of_range() {
  // There are KEA_NUM_EVENTS = 32 counting semaphores.
  // expected-error @+1 {{event id must be 0..31, got 32}}
  kea.wait 32
  return
}

// -----

func.func @trace_bad_kind() {
  // expected-error @+1 {{kind must be "marker", "begin" or "end", got "region"}}
  kea.trace "region" 0
  return
}
