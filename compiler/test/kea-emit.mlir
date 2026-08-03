// RUN: kea-translate %s --function=dma  --sync=none --no-labels --emit-kasm=- | FileCheck %s --check-prefix=DMA
// RUN: kea-translate %s --function=mxu  --sync=none --no-labels --emit-kasm=- | FileCheck %s --check-prefix=MXU
// RUN: kea-translate %s --function=dwu  --sync=none --no-labels --emit-kasm=- | FileCheck %s --check-prefix=DWU
// RUN: kea-translate %s --function=vpu  --sync=none --no-labels --emit-kasm=- | FileCheck %s --check-prefix=VPU
// RUN: kea-translate %s --function=ctrl --sync=none --no-labels --emit-kasm=- | FileCheck %s --check-prefix=CTRL
//
// `-kea-emit`: one Level 2 op -> one line of docs/ASSEMBLY.md §5 assembly.
//
// Every instruction op appears once, in opcode order, so this file reads side
// by side with `runtime/src/op_fields.cpp` -- which is the normative operand
// order the CHECK lines are transcribed from. `--sync=none` keeps the stream
// exactly as written; kea-emit-sync.mlir covers the insertion path.
//
// The other property under test is the ONE piece of arithmetic emission does:
//
//     instr.X_addr = addr(buffer) + X_addr
//
// so every buffer below has a non-zero `addr` and most displacements are
// non-zero, and the CHECK lines spell out the sum. A backend that dropped the
// base, or added it twice, or converted ACC words to bytes on the way past,
// passes none of them.

//===--------------------------------------------------------------------===//
// 0x10 DMA_LD / 0x11 DMA_ST -- absolute SPM, symbolic DRAM (ADR-0001 rule 3)
//===--------------------------------------------------------------------===//

// Canonical form, ASSEMBLY.md §4: arch, revision, entry, then a blank line.
// DMA: .arch "KEA-1"
// DMA-NEXT: .isa_revision 1
// DMA-NEXT: .entry 0

func.func @dma() attributes {kea.dram_layout = {total_bytes = 65536 : i64,
    const_offset = 0 : i64, const_bytes = 0 : i64, io_offset = 0 : i64,
    io_bytes = 16384 : i64, scratch_offset = 16384 : i64,
    scratch_bytes = 49152 : i64, alignment = 64 : i64}} {
  %d = kea.alloc {name = "x", role = "input", addr = 1024 : i64}
     : !kea.buffer<8192xi8, DRAM>
  %a = kea.alloc {name = "at", role = "scratch", addr = 4096 : i64}
     : !kea.buffer<4096xi8, A>
  %w = kea.alloc {name = "wt", role = "scratch", addr = 512 : i64}
     : !kea.buffer<1024xi8, W>
  kea.trace "begin" 0

  // A DRAM displacement becomes `@symbol+offset`; the SPM one becomes an
  // absolute integer carrying its space prefix.
  // DMA: DMA0  DMA_LD  spm_space=SPM_A, dram_addr=@x+256, spm_addr=a:4160, len0=128, n1=8, n2=2, dram_s1=256, dram_s2=2048, spm_s1=128, spm_s2=1024
  kea.dma_load %d -> %a {dram_addr = 256, spm_addr = 64, len0 = 128, n1 = 8,
                         n2 = 2, dram_s1 = 256, dram_s2 = 2048, spm_s1 = 128,
                         spm_s2 = 1024}
    : !kea.buffer<8192xi8, DRAM> -> !kea.buffer<4096xi8, A>

  // `spm_space` follows the operand's address space, and the assembler
  // cross-checks it against the `w:` prefix -- ASSEMBLY.md §5.2. A `unit`
  // attribute, wherever `-kea-schedule` put one, is honoured.
  // DMA: DMA1  DMA_LD  spm_space=SPM_W, dram_addr=@x, spm_addr=w:512, len0=256, n1=1, n2=1, dram_s1=256, dram_s2=256, spm_s1=256, spm_s2=256
  kea.dma_load %d -> %w {dram_addr = 0, spm_addr = 0, len0 = 256, n1 = 1,
                         n2 = 1, dram_s1 = 256, dram_s2 = 256, spm_s1 = 256,
                         spm_s2 = 256, unit = "DMA1"}
    : !kea.buffer<8192xi8, DRAM> -> !kea.buffer<1024xi8, W>

  // DMA: DMA0  DMA_ST  spm_space=SPM_A, dram_addr=@x+4096, spm_addr=a:4096, len0=64, n1=4, n2=1, dram_s1=64, dram_s2=256, spm_s1=64, spm_s2=256
  kea.dma_store %a -> %d {dram_addr = 4096, spm_addr = 0, len0 = 64, n1 = 4,
                          n2 = 1, dram_s1 = 64, dram_s2 = 256, spm_s1 = 64,
                          spm_s2 = 256}
    : !kea.buffer<4096xi8, A> -> !kea.buffer<8192xi8, DRAM>
  kea.trace "end" 0
  kea.halt
  return
}

//===--------------------------------------------------------------------===//
// 0x20 LOAD_W / 0x21 MATMUL -- and the ACC word discipline (ASSEMBLY.md §1.2)
//===--------------------------------------------------------------------===//

func.func @mxu() attributes {kea.dram_layout = {total_bytes = 65536 : i64,
    alignment = 64 : i64}} {
  %w = kea.alloc {name = "mw", role = "scratch", addr = 1024 : i64}
     : !kea.buffer<2304xi8, W>
  %a = kea.alloc {name = "ma", role = "scratch", addr = 2048 : i64}
     : !kea.buffer<1616xi8, A>
  %q = kea.alloc {name = "mq", role = "scratch", addr = 1024 : i64}
     : !kea.buffer<2048xi32, ACC>

  // MXU: MXU   LOAD_W  w_addr=w:1280, w_row_stride=16, k_rows=16, n_cols=16, bank=1, dtype=int8
  kea.load_w %w {w_addr = 256, w_row_stride = 16, k_rows = 16, n_cols = 16,
                 bank = 1} : !kea.buffer<2304xi8, W>
  // MXU: MXU   LOAD_W  w_addr=w:1032, w_row_stride=8, k_rows=16, n_cols=16, bank=0, dtype=int4
  kea.load_w %w {w_addr = 8, w_row_stride = 8, k_rows = 16, n_cols = 16,
                 bank = 0, int4} : !kea.buffer<2304xi8, W>

  // The ACC base is 1024 WORDS and the ACC strides carry `w`; the SPM_A base
  // is 2048 BYTES and its strides do not. Getting that wrong is what
  // ASSEMBLY.md §1.2 exists to make impossible.
  // MXU: MXU   MATMUL  a_addr=a:2224, a_inner_stride=16, a_outer_stride=160, m_inner=8, m_outer=8, acc_addr=acc:1024, acc_inner_stride=16w, acc_outer_stride=128w, bank=0, acc_mode=overwrite, dtype=int8
  kea.mm %a, %q {a_addr = 176, a_inner_stride = 16, a_outer_stride = 160,
                 m_inner = 8, m_outer = 8, acc_addr = 0,
                 acc_inner_stride = 16, acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
  // MXU: MXU   MATMUL  a_addr=a:2048, a_inner_stride=16, a_outer_stride=160, m_inner=8, m_outer=8, acc_addr=acc:2048, acc_inner_stride=16w, acc_outer_stride=128w, bank=1, acc_mode=accumulate, dtype=int4
  kea.mm %a, %q {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                 m_inner = 8, m_outer = 8, acc_addr = 1024,
                 acc_inner_stride = 16, acc_outer_stride = 128, bank = 1,
                 accumulate, int4}
    : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
  kea.halt
  return
}

//===--------------------------------------------------------------------===//
// 0x30 DWCONV
//===--------------------------------------------------------------------===//

func.func @dwu() attributes {kea.dram_layout = {total_bytes = 65536 : i64,
    alignment = 64 : i64}} {
  %a = kea.alloc {name = "da", role = "scratch", addr = 512 : i64}
     : !kea.buffer<4096xi8, A>
  %w = kea.alloc {name = "dw", role = "scratch", addr = 256 : i64}
     : !kea.buffer<800xi8, W>
  %q = kea.alloc {name = "dq", role = "scratch", addr = 16 : i64}
     : !kea.buffer<2048xi32, ACC>
  // DWU: DWU   DWCONV  a_addr=a:512, w_addr=w:256, acc_addr=acc:16, out_h=8, out_w=8, channels=32, a_row_stride=320, a_pix_stride=32, kernel=3, stride=1, acc_mode=overwrite
  kea.dwconv %a, %w, %q {a_addr = 0, w_addr = 0, acc_addr = 0, out_h = 8,
                         out_w = 8, channels = 32, a_row_stride = 320,
                         a_pix_stride = 32, kernel = 3, stride = 1}
    : !kea.buffer<4096xi8, A>, !kea.buffer<800xi8, W>, !kea.buffer<2048xi32, ACC>
  // DWU: DWU   DWCONV  a_addr=a:512, w_addr=w:256, acc_addr=acc:16, out_h=4, out_w=4, channels=32, a_row_stride=320, a_pix_stride=32, kernel=5, stride=2, acc_mode=accumulate
  kea.dwconv %a, %w, %q {a_addr = 0, w_addr = 0, acc_addr = 0, out_h = 4,
                         out_w = 4, channels = 32, a_row_stride = 320,
                         a_pix_stride = 32, kernel = 5, stride = 2, accumulate}
    : !kea.buffer<4096xi8, A>, !kea.buffer<800xi8, W>, !kea.buffer<2048xi32, ACC>
  kea.halt
  return
}

//===--------------------------------------------------------------------===//
// 0x40 VQUANT / 0x41 VADD / 0x42 VPOOL / 0x43 VCOPY
//===--------------------------------------------------------------------===//

func.func @vpu() attributes {kea.dram_layout = {total_bytes = 65536 : i64,
    alignment = 64 : i64}} {
  %q = kea.alloc {name = "vq", role = "scratch", addr = 32 : i64}
     : !kea.buffer<2048xi32, ACC>
  %o = kea.alloc {name = "vo", role = "scratch", addr = 64 : i64}
     : !kea.buffer<4096xi8, A>
  %r = kea.alloc {name = "vr", role = "scratch", addr = 8192 : i64}
     : !kea.buffer<4096xi8, A>
  %p = kea.alloc {name = "vp", role = "scratch", addr = 128 : i64}
     : !kea.buffer<384xi8, W>

  // VPU: VPU   VQUANT  acc_addr=acc:48, out_addr=a:80, qparam_addr=w:140, num_pixels=64, channels=16, acc_pix_stride=16w, out_pix_stride=32, out_zp=-128, clamp_lo=-128, clamp_hi=127, dtype=int8
  kea.vquant %q, %o, %p {acc_addr = 16, out_addr = 16, qparam_addr = 12,
                         num_pixels = 64, channels = 16, acc_pix_stride = 16,
                         out_pix_stride = 32, out_zp = -128, clamp_lo = -128,
                         clamp_hi = 127}
    : !kea.buffer<2048xi32, ACC>, !kea.buffer<4096xi8, A>, !kea.buffer<384xi8, W>

  // A fused activation is expressed purely through the clamps (ISA.md §10.1),
  // and an int4 output narrows them to [-8, 7].
  // VPU: VPU   VQUANT  acc_addr=acc:32, out_addr=a:64, qparam_addr=w:128, num_pixels=16, channels=16, acc_pix_stride=16w, out_pix_stride=16, out_zp=-3, clamp_lo=-3, clamp_hi=7, dtype=int4
  kea.vquant %q, %o, %p {acc_addr = 0, out_addr = 0, qparam_addr = 0,
                         num_pixels = 16, channels = 16, acc_pix_stride = 16,
                         out_pix_stride = 16, out_zp = -3, clamp_lo = -3,
                         clamp_hi = 7, int4}
    : !kea.buffer<2048xi32, ACC>, !kea.buffer<4096xi8, A>, !kea.buffer<384xi8, W>

  // VPU: VPU   VADD    a_addr=a:64, b_addr=a:8192, out_addr=a:80, param_addr=w:132, num_elems=1024, clamp_lo=-128, clamp_hi=127
  kea.vadd %o, %r, %o, %p {a_addr = 0, b_addr = 0, out_addr = 16,
                           param_addr = 4, num_elems = 1024, clamp_lo = -128,
                           clamp_hi = 127}
    : !kea.buffer<4096xi8, A>, !kea.buffer<4096xi8, A>, !kea.buffer<4096xi8, A>,
      !kea.buffer<384xi8, W>

  // VPU: VPU   VPOOL   mode=max, in_addr=a:64, out_addr=a:8192, out_h=4, out_w=4, channels=16, kh=2, kw=2, stride_h=2, stride_w=2, in_row_stride=128, out_row_stride=64
  kea.vpool %o, %r {in_addr = 0, out_addr = 0, out_h = 4, out_w = 4,
                    channels = 16, kh = 2, kw = 2, stride_h = 2, stride_w = 2,
                    in_row_stride = 128, out_row_stride = 64}
    : !kea.buffer<4096xi8, A>, !kea.buffer<4096xi8, A>
  // VPU: VPU   VPOOL   mode=avg, in_addr=a:64, out_addr=a:8192, out_h=1, out_w=1, channels=16, kh=7, kw=7, stride_h=1, stride_w=1, in_row_stride=112, out_row_stride=16
  kea.vpool %o, %r {in_addr = 0, out_addr = 0, out_h = 1, out_w = 1,
                    channels = 16, kh = 7, kw = 7, stride_h = 1, stride_w = 1,
                    in_row_stride = 112, out_row_stride = 16, avg}
    : !kea.buffer<4096xi8, A>, !kea.buffer<4096xi8, A>

  // A copy carries both spaces; SPM_W -> SPM_A is legal and the address
  // prefixes have to agree with the flags.
  // VPU: VPU   VCOPY   mode=copy, src_space=SPM_W, dst_space=SPM_A, src_addr=w:128, dst_addr=a:64, row_bytes=128, rows=2, src_row_stride=128, dst_row_stride=256, fill_value=0
  kea.vcopy from %p : !kea.buffer<384xi8, W>, to %o
    {src_addr = 0, dst_addr = 0, row_bytes = 128, rows = 2,
     src_row_stride = 128, dst_row_stride = 256} : !kea.buffer<4096xi8, A>

  // In fill mode there is no source operand, but the source fields still
  // occupy encoding bits, so ASSEMBLY.md §5.5 fixes them at SPM_A / a:0 / 0.
  // VPU: VPU   VCOPY   mode=fill, src_space=SPM_A, dst_space=SPM_A, src_addr=a:0, dst_addr=a:64, row_bytes=272, rows=1, src_row_stride=0, dst_row_stride=272, fill_value=-5
  kea.vcopy to %o {src_addr = 0, dst_addr = 0, row_bytes = 272, rows = 1,
                   src_row_stride = 0, dst_row_stride = 272, fill_value = -5,
                   fill} : !kea.buffer<4096xi8, A>
  kea.halt
  return
}

//===--------------------------------------------------------------------===//
// 0x02 SIGNAL / 0x03 WAIT / 0x04 TRACE / 0x01 HALT
//===--------------------------------------------------------------------===//

func.func @ctrl() attributes {kea.dram_layout = {total_bytes = 65536 : i64,
    alignment = 64 : i64}} {
  // CTRL: CTRL  SIGNAL  event=4, inc=1
  kea.signal 4
  // CTRL: VPU   WAIT    event=4, threshold=2
  kea.wait 4 {value = 2 : i64, unit = "VPU"}
  // CTRL: DMA1  TRACE   kind=marker, tag=7, payload=99
  kea.trace "marker" 7 {payload = 99 : i64, unit = "DMA1"}
  // CTRL: CTRL  HALT    exit_code=3
  kea.halt {exit_code = 3 : i64}
  return
}
