// RUN: kea-opt %s | kea-opt | FileCheck %s
//
// Round trip for every Level 2 (machine level) op: parse, print, re-parse,
// re-print. One function per KEA-1 instruction, in opcode order, so this file
// can be read side by side with docs/ISA.md §4.1 and include/kea/isa.h.
//
// There is exactly one op per instruction (minus NOP, which ISA.md §5.1 says
// the compiler should not need), plus `kea.alloc`, which is the symbolic
// buffer identity `-kea-alloc` consumes.

//===--------------------------------------------------------------------===//
// kea.alloc -- not an instruction; the buffer/live-range interface
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @alloc
func.func @alloc(%w: tensor<24x1x1x4xi8>, %b: tensor<24xi32>) {
  // A DRAM constant, with the Level 1 tensor it is materialized from reachable
  // through the `source` operand.
  // CHECK: kea.alloc from %{{.*}} : tensor<24x1x1x4xi8> {layout = "mxu_tiles_16x16", name = "conv0.weights", role = "weights"} : !kea.buffer<1536xi8, DRAM>
  %0 = kea.alloc from %w : tensor<24x1x1x4xi8>
       {name = "conv0.weights", role = "weights", layout = "mxu_tiles_16x16"}
     : !kea.buffer<1536xi8, DRAM>

  // A KeaQuantParam block: bias first, then weights, so -kea-emit can fold the
  // zero-point correction `bias[c] - input_zp * sum_k w[c][k]`.
  // CHECK: kea.alloc from %{{.*}}, %{{.*}} : tensor<24xi32>, tensor<24x1x1x4xi8> {input_zp = -5 : i64, layout = "quant_params", name = "conv0.qparams"
  %1 = kea.alloc from %b, %w : tensor<24xi32>, tensor<24x1x1x4xi8>
       {name = "conv0.qparams", role = "qparam", layout = "quant_params",
        input_zp = -5 : i64,
        quant = #kea.quant<multiplier = [1073741824], shift = [36],
                           input_zp = 0, output_zp = -128, axis = -1,
                           rounding = DOUBLE>}
     : !kea.buffer<288xi8, DRAM>

  // The derived, machine-form KeaAddParam, in isa.h field order.
  // CHECK: kea.alloc {add_param = array<i64: 1610612736, 1073741824, 1503238553, 1, 0, 8, -5, -5, -5>
  %2 = kea.alloc {name = "conv0.addparams", role = "addparam",
                  layout = "add_params",
                  add_param = array<i64: 1610612736, 1073741824, 1503238553,
                                         1, 0, 8, -5, -5, -5>}
     : !kea.buffer<20xi8, DRAM>

  // On-chip scratch, with the advisory live range -kea-alloc sizes its
  // interference graph from.
  // CHECK: kea.alloc {live = array<i64: 4, 9>, name = "conv0.atile", role = "scratch"} : !kea.buffer<1616xi8, A>
  %3 = kea.alloc {name = "conv0.atile", role = "scratch",
                  live = array<i64: 4, 9>} : !kea.buffer<1616xi8, A>
  // CHECK: kea.alloc {alignment = 64 : i64, name = "conv0.acc", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  %4 = kea.alloc {name = "conv0.acc", role = "scratch", alignment = 64 : i64}
     : !kea.buffer<2048xi32, ACC>
  return
}

//===--------------------------------------------------------------------===//
// 0x10 DMA_LD / 0x11 DMA_ST -- KeaDma
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @dma
func.func @dma() {
  %d = kea.alloc {name = "x", role = "input"} : !kea.buffer<1024xi8, DRAM>
  %a = kea.alloc {name = "t", role = "scratch"} : !kea.buffer<1616xi8, A>
  %o = kea.alloc {name = "y", role = "output"} : !kea.buffer<1024xi8, DRAM>

  // The ISA.md §6.1 worked descriptor, into a zero-haloed buffer: keep
  // len0/n1/n2, offset spm_addr past the halo, widen spm_s1.
  // CHECK: kea.dma_load %{{.*}} -> %{{.*}} {dram_addr = 0 : i64, dram_s1 = 128 : i64, dram_s2 = 0 : i64, len0 = 128 : i64, n1 = 8 : i64, n2 = 1 : i64, spm_addr = 176 : i64, spm_s1 = 160 : i64, spm_s2 = 0 : i64} : !kea.buffer<1024xi8, DRAM> -> !kea.buffer<1616xi8, A>
  kea.dma_load %d -> %a {dram_addr = 0, spm_addr = 176, len0 = 128, n1 = 8,
                         n2 = 1, dram_s1 = 128, dram_s2 = 0, spm_s1 = 160,
                         spm_s2 = 0}
    : !kea.buffer<1024xi8, DRAM> -> !kea.buffer<1616xi8, A>

  // Negative strides are legal (reversed layouts) and must survive the round
  // trip as negative -- that is why the attributes are Kea_I64, not I64Attr.
  // CHECK: kea.dma_store %{{.*}} -> %{{.*}} {dram_addr = 896 : i64, dram_s1 = -128 : i64
  kea.dma_store %a -> %o {dram_addr = 896, spm_addr = 0, len0 = 128, n1 = 8,
                          n2 = 1, dram_s1 = -128, dram_s2 = 0, spm_s1 = 128,
                          spm_s2 = 0, unit = "DMA1"}
    : !kea.buffer<1616xi8, A> -> !kea.buffer<1024xi8, DRAM>
  return
}

//===--------------------------------------------------------------------===//
// 0x20 LOAD_W / 0x21 MATMUL -- KeaLoadW / KeaMatmul
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @mxu
func.func @mxu() {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<1616xi8, A>
  %w = kea.alloc {name = "w", role = "scratch"} : !kea.buffer<4608xi8, W>
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>

  // CHECK: kea.load_w %{{.*}} {bank = 0 : i64, k_rows = 16 : i64, n_cols = 16 : i64, w_addr = 0 : i64, w_row_stride = 16 : i64} : !kea.buffer<4608xi8, W>
  kea.load_w %w {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
                 bank = 0} : !kea.buffer<4608xi8, W>
  // int4 halves the LOAD_W alignment to 8 bytes.
  // CHECK: kea.load_w %{{.*}} {bank = 1 : i64, int4, k_rows = 9 : i64, n_cols = 8 : i64, w_addr = 8 : i64, w_row_stride = 8 : i64}
  kea.load_w %w {w_addr = 8, w_row_stride = 8, k_rows = 9, n_cols = 8,
                 bank = 1, int4} : !kea.buffer<4608xi8, W>

  // No weight operand: MATMUL streams against whatever LOAD_W left in `bank`.
  // CHECK: kea.mm %{{.*}}, %{{.*}} {a_addr = 352 : i64, a_inner_stride = 16 : i64, a_outer_stride = 160 : i64, acc_addr = 0 : i64, acc_inner_stride = 16 : i64, acc_outer_stride = 128 : i64, accumulate, bank = 1 : i64, m_inner = 8 : i64, m_outer = 8 : i64} : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
  kea.mm %a, %q {a_addr = 352, a_inner_stride = 16, a_outer_stride = 160,
                 m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
                 acc_outer_stride = 128, bank = 1, accumulate}
    : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
  return
}

//===--------------------------------------------------------------------===//
// 0x30 DWCONV -- KeaDwconv
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @dwu
func.func @dwu() {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<3216xi8, A>
  %w = kea.alloc {name = "w", role = "scratch"} : !kea.buffer<288xi8, W>
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  // CHECK: kea.dwconv %{{.*}}, %{{.*}}, %{{.*}} {a_addr = 0 : i64, a_pix_stride = 32 : i64, a_row_stride = 320 : i64, acc_addr = 0 : i64, channels = 32 : i64, kernel = 3 : i64, out_h = 8 : i64, out_w = 8 : i64, stride = 1 : i64, w_addr = 0 : i64} : !kea.buffer<3216xi8, A>, !kea.buffer<288xi8, W>, !kea.buffer<2048xi32, ACC>
  kea.dwconv %a, %w, %q {a_addr = 0, w_addr = 0, acc_addr = 0, out_h = 8,
                         out_w = 8, channels = 32, a_row_stride = 320,
                         a_pix_stride = 32, kernel = 3, stride = 1}
    : !kea.buffer<3216xi8, A>, !kea.buffer<288xi8, W>, !kea.buffer<2048xi32, ACC>
  return
}

//===--------------------------------------------------------------------===//
// 0x40 VQUANT / 0x41 VADD / 0x42 VPOOL / 0x43 VCOPY
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @vpu
func.func @vpu() {
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<2064xi8, A>
  %b = kea.alloc {name = "b", role = "scratch"} : !kea.buffer<2064xi8, A>
  %p = kea.alloc {name = "p", role = "scratch"} : !kea.buffer<384xi8, W>

  // ReLU6 in a zero point of -128 is expressed purely through the clamps.
  // CHECK: kea.vquant %{{.*}}, %{{.*}}, %{{.*}} {acc_addr = 0 : i64, acc_pix_stride = 16 : i64, channels = 16 : i64, clamp_hi = 127 : i64, clamp_lo = -128 : i64, num_pixels = 64 : i64, out_addr = 0 : i64, out_pix_stride = 32 : i64, out_zp = -128 : i64, qparam_addr = 0 : i64} : !kea.buffer<2048xi32, ACC>, !kea.buffer<2064xi8, A>, !kea.buffer<384xi8, W>
  kea.vquant %q, %a, %p {acc_addr = 0, out_addr = 0, qparam_addr = 0,
                         num_pixels = 64, channels = 16, acc_pix_stride = 16,
                         out_pix_stride = 32, out_zp = -128, clamp_lo = -128,
                         clamp_hi = 127}
    : !kea.buffer<2048xi32, ACC>, !kea.buffer<2064xi8, A>, !kea.buffer<384xi8, W>

  // In-place is legal: the VPU reads before it writes within a lane.
  // CHECK: kea.vadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {a_addr = 0 : i64, b_addr = 0 : i64, clamp_hi = 127 : i64, clamp_lo = -128 : i64, num_elems = 1024 : i64, out_addr = 0 : i64, param_addr = 0 : i64}
  kea.vadd %a, %b, %a, %p {a_addr = 0, b_addr = 0, out_addr = 0,
                           param_addr = 0, num_elems = 1024, clamp_lo = -128,
                           clamp_hi = 127}
    : !kea.buffer<2064xi8, A>, !kea.buffer<2064xi8, A>, !kea.buffer<2064xi8, A>,
      !kea.buffer<384xi8, W>

  // MobileNetV2's 7x7 global average pool shape.
  // CHECK: kea.vpool %{{.*}}, %{{.*}} {avg, channels = 16 : i64, in_addr = 0 : i64, in_row_stride = 128 : i64, kh = 7 : i64, kw = 7 : i64, out_addr = 0 : i64, out_h = 1 : i64, out_row_stride = 16 : i64, out_w = 1 : i64, stride_h = 1 : i64, stride_w = 1 : i64}
  kea.vpool %a, %b {in_addr = 0, out_addr = 0, out_h = 1, out_w = 1,
                    channels = 16, kh = 7, kw = 7, stride_h = 1, stride_w = 1,
                    in_row_stride = 128, out_row_stride = 16, avg}
    : !kea.buffer<2064xi8, A>, !kea.buffer<2064xi8, A>

  // Fill mode: the conv halo gets the input ZERO POINT, not 0 (ISA.md §8.4).
  // CHECK: kea.vcopy to %{{.*}} {dst_addr = 0 : i64, dst_row_stride = 1616 : i64, fill, fill_value = -128 : i64, row_bytes = 1616 : i64, rows = 1 : i64, src_addr = 0 : i64, src_row_stride = 0 : i64} : !kea.buffer<2064xi8, A>
  kea.vcopy to %a {fill, fill_value = -128, src_addr = 0, dst_addr = 0,
                   row_bytes = 1616, rows = 1, src_row_stride = 0,
                   dst_row_stride = 1616} : !kea.buffer<2064xi8, A>
  // Copy mode, SPM_A -> SPM_W.
  // CHECK: kea.vcopy from %{{.*}} : !kea.buffer<2064xi8, A>, to %{{.*}} {dst_addr = 0 : i64, dst_row_stride = 128 : i64, row_bytes = 128 : i64, rows = 2 : i64, src_addr = 0 : i64, src_row_stride = 256 : i64} : !kea.buffer<384xi8, W>
  kea.vcopy from %a : !kea.buffer<2064xi8, A>, to %p
    {src_addr = 0, dst_addr = 0, row_bytes = 128, rows = 2,
     src_row_stride = 256, dst_row_stride = 128} : !kea.buffer<384xi8, W>
  return
}

//===--------------------------------------------------------------------===//
// 0x01 HALT / 0x02 SIGNAL / 0x03 WAIT / 0x04 TRACE
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @ctrl
func.func @ctrl() {
  // CHECK: kea.trace "begin" 7
  kea.trace "begin" 7
  // CHECK: kea.trace "marker" 7 {payload = 42 : i64, unit = "MXU"}
  kea.trace "marker" 7 {payload = 42, unit = "MXU"}
  // The increment/threshold default to 1, so the spike's spelling still parses.
  // CHECK: kea.signal 0
  kea.signal 0
  // CHECK: kea.wait 3 {unit = "VPU", value = 2 : i64}
  kea.wait 3 {value = 2, unit = "VPU"}
  // CHECK: kea.trace "end" 7
  kea.trace "end" 7
  // CHECK: kea.halt
  kea.halt
  return
}
