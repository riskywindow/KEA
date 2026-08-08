// RUN: kea-opt %s -kea-tile | FileCheck %s
// RUN: kea-opt %s -kea-tile=report-tiles=true | FileCheck %s --check-prefix=TILES
//
// -kea-tile: fused KEA Level 1 -> Level 2 machine ops.
//
// The centrepiece is @conv3x3_s1, which reproduces the worked example of
// docs/ISA.md §8.5 instruction for instruction, including its per-tap `a_addr`
// table. That section is normative, so this test is the proof the lowering
// obeys it.

//===--------------------------------------------------------------------===//
// ISA.md §8.5 -- 3x3, stride 1, dilation 1, IC = 16, OC = 32, 8x8, pad 1
//===--------------------------------------------------------------------===//
//
// SPM_A buffer padded to 10x10x16, so sp = 16 and sr = 10*16 = 160. The ISA
// gives the buffer as 1600 bytes; -kea-tile allocates 1616, because `kea.mm`
// always reads 16 activation bytes per row whatever the resident k_rows and the
// read must be *defined*, not merely in bounds (ISA.md §7.3, §2.3).
//
// Per-tap a_addr = kh*160 + kw*16:
//   (0,0)   0   (1,0) 160   (2,0) 320
//   (0,1)  16   (1,1) 176   (2,1) 336
//   (0,2)  32   (1,2) 192   (2,2) 352

// CHECK-LABEL: func.func @conv3x3_s1
// CHECK-SAME:  (%{{[^:]*}}: tensor<1x8x8x16xi8>, %{{[^:]*}}: tensor<32x3x3x16xi8>)
// CHECK-NOT:   -> tensor

// The padded activation tile: 10*160 = 1600, plus one array row.
// CHECK: kea.alloc {{.*}}atile{{.*}} : !kea.buffer<1616xi8, A>
// The halo fill carries the input ZERO POINT (-5), not 0 -- ISA.md §8.4.
// CHECK-NEXT: kea.vcopy to %{{.*}} {dst_addr = 0 : i64, dst_row_stride = 1616 : i64, fill, fill_value = -5 : i64, row_bytes = 1616 : i64, rows = 1 : i64
// The 8x8x16 interior lands at 1*160 + 1*16 = 176, exactly as ISA.md §8.5 says.
// CHECK-NEXT: kea.dma_load %{{.*}} {dram_addr = 0 : i64, dram_s1 = 128 : i64, dram_s2 = 0 : i64, len0 = 128 : i64, n1 = 8 : i64, n2 = 1 : i64, spm_addr = 176 : i64, spm_s1 = 160 : i64
// Two output-channel groups of 8*8*16 = 1024 words each.
// CHECK-NEXT: kea.alloc {{.*}}acc{{.*}} : !kea.buffer<2048xi32, ACC>

// tap (0,0): the ONLY instruction of the ACC region without `accumulate`.
// CHECK: kea.load_w %{{.*}} {bank = 0 : i64, k_rows = 16 : i64, n_cols = 16 : i64, w_addr = 0 : i64, w_row_stride = 16 : i64}
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 0 : i64, a_inner_stride = 16 : i64, a_outer_stride = 160 : i64, acc_addr = 0 : i64, acc_inner_stride = 16 : i64, acc_outer_stride = 128 : i64, bank = 0 : i64, m_inner = 8 : i64, m_outer = 8 : i64}
// tap (0,1): bank flips to 1 so this LOAD_W hides under the previous MATMUL.
// CHECK-NEXT: kea.load_w %{{.*}} {bank = 1 : i64, k_rows = 16 : i64, n_cols = 16 : i64, w_addr = 256 : i64, w_row_stride = 16 : i64}
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 16 : i64, {{.*}}accumulate, bank = 1 : i64, m_inner = 8 : i64, m_outer = 8 : i64}
// tap (0,2)
// CHECK-NEXT: kea.load_w %{{.*}} {bank = 0 : i64, {{.*}}w_addr = 512 : i64
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 32 : i64, {{.*}}accumulate, bank = 0 : i64
// tap (1,0)
// CHECK-NEXT: kea.load_w %{{.*}} {bank = 1 : i64, {{.*}}w_addr = 768 : i64
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 160 : i64, {{.*}}accumulate, bank = 1 : i64
// tap (1,1)
// CHECK-NEXT: kea.load_w
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 176 : i64, {{.*}}accumulate
// tap (1,2)
// CHECK-NEXT: kea.load_w
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 192 : i64, {{.*}}accumulate
// tap (2,0)
// CHECK-NEXT: kea.load_w
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 320 : i64, {{.*}}accumulate
// tap (2,1)
// CHECK-NEXT: kea.load_w
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 336 : i64, {{.*}}accumulate
// tap (2,2)
// CHECK-NEXT: kea.load_w
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 352 : i64, {{.*}}accumulate

// The fused epilogue for output-channel group 0.
// CHECK-NEXT: kea.vquant %{{.*}} {acc_addr = 0 : i64, acc_pix_stride = 16 : i64, channels = 16 : i64, clamp_hi = 127 : i64, clamp_lo = -128 : i64, num_pixels = 64 : i64, out_addr = 0 : i64, out_pix_stride = 32 : i64, out_zp = -128 : i64, qparam_addr = 0 : i64}

// Group 1 starts a fresh ACC region at word 1024, so its first tap overwrites.
// Its 9 taps continue the bank alternation rather than restarting it: tap 8 of
// group 0 used bank 0, so tap 9 -- group 1's first -- uses bank 1, and the load
// still overlaps the matmul before it.
// CHECK: kea.load_w %{{.*}} {bank = 1 : i64, {{.*}}w_addr = 2304 : i64
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 0 : i64, a_inner_stride = 16 : i64, a_outer_stride = 160 : i64, acc_addr = 1024 : i64, acc_inner_stride = 16 : i64, acc_outer_stride = 128 : i64, bank = 1 : i64, m_inner = 8 : i64, m_outer = 8 : i64}
// CHECK: kea.vquant %{{.*}} {acc_addr = 1024 : i64, {{.*}}out_addr = 16 : i64, {{.*}}qparam_addr = 192 : i64}
// CHECK: kea.halt

// The chosen tile has to FIT: SPM_A 256 KiB, SPM_W 256 KiB, ACC 32768 words,
// and -kea-tile only spends half of each scratchpad so -kea-schedule can
// double buffer.
// TILES-LABEL: func.func @conv3x3_s1
// TILES-SAME: kea.tiling = [{acc = 2048 : i64, cycles = {{[0-9]+}} : i64, dram = {{[0-9]+}} : i64, instrs = {{[0-9]+}} : i64, layer = 0 : i64, oc_groups = 2 : i64, oh = 8 : i64, op = "kea.conv2d", ow = 8 : i64, spm_a = 3680 : i64, spm_w = 4992 : i64, taps = 144 : i64}]
func.func @conv3x3_s1(%in: tensor<1x8x8x16xi8>, %w: tensor<32x3x3x16xi8>)
    -> tensor<1x8x8x32xi8> {
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = -5, weight = 0>,
        strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
        dilations = array<i64: 1, 1>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [36], input_zp = 0, output_zp = -128, axis = -1,
                    rounding = DOUBLE>>}
    : (tensor<1x8x8x16xi8>, tensor<32x3x3x16xi8>) -> tensor<1x8x8x32xi8>
  return %0 : tensor<1x8x8x32xi8>
}

//===--------------------------------------------------------------------===//
// 1x1 pointwise: the degenerate one-tap case of the same identity (§8.2)
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @conv1x1
// No padding, so the input tile is exactly 8*8*32 = 2048 plus the array row.
// CHECK: kea.alloc {{.*}}atile{{.*}} : !kea.buffer<2064xi8, A>
// Two reduction tiles (IC = 32) x one tap; a_addr steps by 16 bytes, which is
// the `+ ic0` of §8.3: channels are innermost, so it is a flat byte offset.
// CHECK: kea.load_w %{{.*}} {bank = 0 : i64, k_rows = 16 : i64, n_cols = 16 : i64, w_addr = 0 : i64
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 0 : i64, a_inner_stride = 32 : i64, a_outer_stride = 256 : i64, acc_addr = 0 : i64, acc_inner_stride = 16 : i64, acc_outer_stride = 128 : i64, bank = 0 : i64, m_inner = 8 : i64, m_outer = 8 : i64}
// CHECK-NEXT: kea.load_w %{{.*}} {bank = 1 : i64, k_rows = 16 : i64, n_cols = 16 : i64, w_addr = 256 : i64
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 16 : i64, {{.*}}accumulate, bank = 1 : i64
// CHECK: kea.dma_store
// CHECK: kea.halt
func.func @conv1x1(%in: tensor<1x8x8x32xi8>, %w: tensor<16x1x1x32xi8>)
    -> tensor<1x8x8x16xi8> {
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
        strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
        dilations = array<i64: 1, 1>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [33], input_zp = 0, output_zp = 0, axis = -1,
                    rounding = DOUBLE>, clamp = [0, 127]>}
    : (tensor<1x8x8x32xi8>, tensor<16x1x1x32xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

//===--------------------------------------------------------------------===//
// Stride-2 depthwise -- the DWU, ISA.md §9
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @dwconv_s2
// 32 channels is already a multiple of KEA_DWU_LANES, so the input tile is
// dense: (4-1)*2 + 3 = 9 rows and columns of 32 channels = 9*288 = 2592.
// CHECK: kea.alloc {{.*}}atile{{.*}} : !kea.buffer<2608xi8, A>
// CHECK: kea.vcopy to %{{.*}}fill, fill_value = -128 : i64
// One contiguous run per row: the DMA lands the 8x8 input under the 1-pixel
// top/left halo at 1*288 + 1*32 = 320.
// CHECK: kea.dma_load %{{.*}} {dram_addr = 0 : i64, dram_s1 = 256 : i64, dram_s2 = 0 : i64, len0 = 256 : i64, n1 = 8 : i64, n2 = 1 : i64, spm_addr = 320 : i64, spm_s1 = 288 : i64
// CHECK: kea.dwconv %{{.*}} {a_addr = 0 : i64, a_pix_stride = 32 : i64, a_row_stride = 288 : i64, acc_addr = 0 : i64, channels = 32 : i64, kernel = 3 : i64, out_h = 4 : i64, out_w = 4 : i64, stride = 2 : i64, w_addr = 0 : i64}
// The ACC region is dense [out_h][out_w][channels] -- there is no ACC stride
// field on DWCONV, so acc_pix_stride on the VQUANT is `channels`.
// CHECK-NEXT: kea.vquant %{{.*}} {acc_addr = 0 : i64, acc_pix_stride = 32 : i64, channels = 32 : i64, clamp_hi = 127 : i64, clamp_lo = -128 : i64, num_pixels = 16 : i64, out_addr = 0 : i64, out_pix_stride = 32 : i64, out_zp = -128 : i64, qparam_addr = 0 : i64}

// TILES-LABEL: func.func @dwconv_s2
// TILES-SAME: {acc = 512 : i64, channels = 32 : i64, dram = {{[0-9]+}} : i64, instrs = {{[0-9]+}} : i64, layer = 0 : i64, oh = 4 : i64, op = "kea.dwconv2d", ow = 4 : i64, spm_a = 3136 : i64, spm_w = 672 : i64, taps = 9 : i64}
func.func @dwconv_s2(%in: tensor<1x8x8x32xi8>, %w: tensor<32x3x3x1xi8>)
    -> tensor<1x4x4x32xi8> {
  %0 = kea.dwconv2d %in, %w {zero_points = #kea.zp<input = -128, weight = 0>,
        strides = array<i64: 2, 2>, pads = array<i64: 1, 1, 1, 1>,
        dilations = array<i64: 1, 1>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [35], input_zp = 0, output_zp = -128, axis = -1,
                    rounding = DOUBLE>>}
    : (tensor<1x8x8x32xi8>, tensor<32x3x3x1xi8>) -> tensor<1x4x4x32xi8>
  return %0 : tensor<1x4x4x32xi8>
}

//===--------------------------------------------------------------------===//
// A layer too big for the scratchpad, so the tiler has to split it
//===--------------------------------------------------------------------===//
//
// 112x112x32 is 401,408 bytes of activations. MICROARCH.md §9.3(b) calls this
// out by name: it "does NOT fit in a 256 KiB SPM_A, so this layer must be tiled
// into row bands with double-buffered DMA".

// CHECK-LABEL: func.func @too_big_for_spm
// TILES-LABEL: func.func @too_big_for_spm
// The tile is an 8-row band of the full 112-wide image. Every on-chip
// footprint stays under half of its capacity -- SPM_A and SPM_W under half of
// 262144 bytes, ACC under half of 32768 words -- which is the reserve that lets
// -kea-schedule put a second tile in flight. ACC is held to the same reserve as
// the scratchpads deliberately: a tile sized to fill ACC cannot be overlapped
// with its successor, which is precisely the overlap the cost model assumes
// when it prices max(MXU,VPU,DMA) rather than the sum.
// TILES-SAME: acc = 14336 : i64
// TILES-SAME: oh = 8 : i64, op = "kea.conv2d", ow = 112 : i64
// TILES-SAME: spm_a = 43040 : i64, spm_w = 704 : i64
func.func @too_big_for_spm(%in: tensor<1x112x112x32xi8>, %w: tensor<16x1x1x32xi8>)
    -> tensor<1x112x112x16xi8> {
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
        strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
        dilations = array<i64: 1, 1>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [33], input_zp = 0, output_zp = 0, axis = -1,
                    rounding = DOUBLE>>}
    : (tensor<1x112x112x32xi8>, tensor<16x1x1x32xi8>) -> tensor<1x112x112x16xi8>
  return %0 : tensor<1x112x112x16xi8>
}

//===--------------------------------------------------------------------===//
// ISA.md §8.6 -- channel-packed first layer, IC = 3
//===--------------------------------------------------------------------===//
//
// KW * IC = 9 <= 16, so a whole kernel ROW is one reduction tile: k_rows = 9,
// three taps instead of nine, a_inner_stride = S * sp = 2 * 3 = 6. That is 56%
// MXU utilization instead of 19% (MICROARCH.md §9.3a). The per-tap base is
// kh * sr with sr = 33 * 3 = 99, i.e. 0, 99, 198 -- and there is no kw term at
// all, because kw now lives inside the reduction tile.

// CHECK-LABEL: func.func @first_layer_packed
// CHECK: kea.alloc from %{{.*}} {layout = "mxu_tiles_16x16_packed"
// CHECK: kea.load_w %{{.*}} {bank = 0 : i64, k_rows = 9 : i64, n_cols = 16 : i64, w_addr = 0 : i64, w_row_stride = 16 : i64}
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 0 : i64, a_inner_stride = 6 : i64, {{.*}}bank = 0 : i64
// CHECK-NEXT: kea.load_w %{{.*}} {bank = 1 : i64, k_rows = 9 : i64, {{.*}}w_addr = 256 : i64
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 99 : i64, {{.*}}accumulate, bank = 1 : i64
// CHECK-NEXT: kea.load_w %{{.*}} {bank = 0 : i64, k_rows = 9 : i64, {{.*}}w_addr = 512 : i64
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 198 : i64, {{.*}}accumulate, bank = 0 : i64
// CHECK-NOT: kea.mm
// CHECK: kea.vquant

// TILES-LABEL: func.func @first_layer_packed
// Three taps, not nine: `taps` is the reduction chain length KH*KW*IC = 27.
// TILES-SAME: taps = 27 : i64
func.func @first_layer_packed(%in: tensor<1x32x32x3xi8>, %w: tensor<16x3x3x3xi8>)
    -> tensor<1x16x16x16xi8> {
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = -128, weight = 0>,
        strides = array<i64: 2, 2>, pads = array<i64: 0, 1, 0, 1>,
        dilations = array<i64: 1, 1>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [34], input_zp = 0, output_zp = -128, axis = -1,
                    rounding = DOUBLE>>}
    : (tensor<1x32x32x3xi8>, tensor<16x3x3x3xi8>) -> tensor<1x16x16x16xi8>
  return %0 : tensor<1x16x16x16xi8>
}

//===--------------------------------------------------------------------===//
// Fully connected -- the classifier, MICROARCH.md §9.3(d)
//===--------------------------------------------------------------------===//
//
// A batch-1 FC is `OH = 1, OW = 1, KH = KW = 1`: one MATMUL row per reduction
// tile, which is the `m_inner = 1` shape that makes the 4-cycle setup constant
// visible.

// CHECK-LABEL: func.func @fully_connected
// CHECK: kea.mm %{{.*}} {a_addr = 0 : i64, {{.*}}m_inner = 1 : i64, m_outer = 1 : i64}
// CHECK: kea.mm %{{.*}} {a_addr = 16 : i64, {{.*}}accumulate
// CHECK: kea.vquant
// CHECK: kea.halt
func.func @fully_connected(%in: tensor<1x64xi8>, %w: tensor<16x64xi8>,
                           %b: tensor<16xi32>) -> tensor<1x16xi8> {
  %0 = kea.fully_connected %in, %w bias %b
       {zero_points = #kea.zp<input = -128, weight = 0>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [38], input_zp = 0, output_zp = 0, axis = -1,
                    rounding = DOUBLE>>}
     : (tensor<1x64xi8>, tensor<16x64xi8>, tensor<16xi32>) -> tensor<1x16xi8>
  return %0 : tensor<1x16xi8>
}

//===--------------------------------------------------------------------===//
// Global average pool -- VPOOL, ISA.md §10.3
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @global_avg_pool
// CHECK: kea.vpool %{{.*}} {avg, channels = 32 : i64, in_addr = 0 : i64, in_row_stride = 224 : i64, kh = 7 : i64, kw = 7 : i64, out_addr = 0 : i64, out_h = 1 : i64, out_row_stride = 32 : i64, out_w = 1 : i64, stride_h = 1 : i64, stride_w = 1 : i64}
func.func @global_avg_pool(%in: tensor<1x7x7x32xi8>) -> tensor<1x1x1x32xi8> {
  %0 = kea.pool %in {kind = #kea.pool_kind<AVG>, kernel = array<i64: 7, 7>,
                     strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>}
     : (tensor<1x7x7x32xi8>) -> tensor<1x1x1x32xi8>
  return %0 : tensor<1x1x1x32xi8>
}

//===--------------------------------------------------------------------===//
// Standalone quantized add -- VADD, and the errata E6 derivation
//===--------------------------------------------------------------------===//
//
// The Level 1 quants are in TOSA form; KEA_VADD applies a fixed
// KEA_VADD_LEFT_SHIFT = 20 to both inputs and absorbs a >>31 in keaSrdhm, so
// the exact input shifts are `tosa_shift - 11`. Here that is 11-11 = 0 and
// 10-11 = -1; a negative keaRdpot exponent is silently ignored, so both inputs
// are shifted down by d = 1 and d is taken back out of the output shift:
//   a_shift = 1, b_shift = 0, o_shift = 40 - 31 - 1 = 8.

// CHECK-LABEL: func.func @quantized_add
// CHECK: kea.alloc {add_param = array<i64: 1610612736, 1073741824, 1503238553, 1, 0, 8, -5, -5, -5>, layout = "add_params"
// CHECK: kea.vadd %{{.*}} {a_addr = 0 : i64, b_addr = 0 : i64, clamp_hi = 127 : i64, clamp_lo = -128 : i64, num_elems = 256 : i64
func.func @quantized_add(%a: tensor<1x8x8x4xi8>, %b: tensor<1x8x8x4xi8>)
    -> tensor<1x8x8x4xi8> {
  %0 = kea.add %a, %b {
    lhs_quant = #kea.quant<multiplier = [1610612736], shift = [11],
                           input_zp = -5, output_zp = 0, axis = -1,
                           rounding = DOUBLE>,
    rhs_quant = #kea.quant<multiplier = [1073741824], shift = [10],
                           input_zp = -5, output_zp = 0, axis = -1,
                           rounding = DOUBLE>,
    out_quant = #kea.quant<multiplier = [1503238553], shift = [40],
                           input_zp = 0, output_zp = -5, axis = -1,
                           rounding = DOUBLE>
  } : (tensor<1x8x8x4xi8>, tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi8>
  return %0 : tensor<1x8x8x4xi8>
}

//===--------------------------------------------------------------------===//
// Buffers stay SYMBOLIC, and carry live ranges -- docs/DIALECT_L2.md §4
//===--------------------------------------------------------------------===//

// Not one absolute scratchpad address anywhere: every address field is a
// displacement inside a `kea.alloc`, and `-kea-alloc` supplies the base.
// CHECK-LABEL: func.func @symbolic_buffers
// CHECK: kea.alloc {{.*}}role = "input"{{.*}} : !kea.buffer<2048xi8, DRAM>
// CHECK: kea.alloc {{.*}}role = "output"{{.*}} : !kea.buffer<1024xi8, DRAM>
// CHECK: kea.alloc from %{{.*}} {layout = "mxu_tiles_16x16", {{.*}}role = "weights"}
// Every on-chip buffer carries a live range, and only on-chip buffers do.
// CHECK: kea.alloc {live = array<i64: {{[0-9]+}}, {{[0-9]+}}>, {{.*}}role = "scratch"} : !kea.buffer<{{[0-9]+}}xi8, W>
// CHECK: kea.alloc {live = array<i64: {{[0-9]+}}, {{[0-9]+}}>, {{.*}}role = "scratch"} : !kea.buffer<{{[0-9]+}}xi32, ACC>
// No semaphores and no queue assignment either -- that is -kea-schedule's job.
// CHECK-NOT: kea.signal
// CHECK-NOT: kea.wait
// CHECK-NOT: unit =
func.func @symbolic_buffers(%in: tensor<1x8x8x32xi8>, %w: tensor<16x1x1x32xi8>)
    -> tensor<1x8x8x16xi8> {
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
        strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
        dilations = array<i64: 1, 1>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [33], input_zp = 0, output_zp = 0, axis = -1,
                    rounding = DOUBLE>>}
    : (tensor<1x8x8x32xi8>, tensor<16x1x1x32xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

//===--------------------------------------------------------------------===//
// Weight banks alternate on EVERY consecutive LOAD_W/MATMUL pair
//===--------------------------------------------------------------------===//
//
// The sharpest case: a 1x1 convolution with a single reduction tile and four
// output-channel groups is four pairs of exactly one tap each. Restarting the
// tap counter per group -- the literal reading of ISA.md §8.3's pseudocode --
// would emit bank 0 four times, and since LOAD_W occupies the bank it targets,
// every load would stall until the matmul before it released that bank. The
// MXU models the two banks as separate resources specifically so that does not
// have to happen (ISA.md §5.3, §7.1), so the counter runs over the whole MXU
// stream instead.
//
// Errata E7 is unaffected and stays trivially true: every kea.mm reads the bank
// the kea.load_w immediately before it wrote. `mlir::kea::verifyWeightBanks()`
// re-checks that over the whole function on every run of -kea-tile.

// CHECK-LABEL: func.func @banks_alternate_across_groups
// CHECK: kea.load_w %{{.*}} {bank = 0 : i64, k_rows = 16 : i64, n_cols = 16 : i64, w_addr = 0 : i64
// CHECK-NEXT: kea.mm %{{.*}} {{{.*}}acc_addr = 0 : i64, {{.*}}bank = 0 : i64
// CHECK: kea.load_w %{{.*}} {bank = 1 : i64, {{.*}}w_addr = 256 : i64
// CHECK-NEXT: kea.mm %{{.*}} {{{.*}}acc_addr = 1024 : i64, {{.*}}bank = 1 : i64
// CHECK: kea.load_w %{{.*}} {bank = 0 : i64, {{.*}}w_addr = 512 : i64
// CHECK-NEXT: kea.mm %{{.*}} {{{.*}}acc_addr = 2048 : i64, {{.*}}bank = 0 : i64
// CHECK: kea.load_w %{{.*}} {bank = 1 : i64, {{.*}}w_addr = 768 : i64
// CHECK-NEXT: kea.mm %{{.*}} {{{.*}}acc_addr = 3072 : i64, {{.*}}bank = 1 : i64
func.func @banks_alternate_across_groups(%in: tensor<1x8x8x16xi8>,
                                         %w: tensor<64x1x1x16xi8>)
    -> tensor<1x8x8x64xi8> {
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
        strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
        dilations = array<i64: 1, 1>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [33], input_zp = 0, output_zp = 0, axis = -1,
                    rounding = DOUBLE>>}
    : (tensor<1x8x8x16xi8>, tensor<64x1x1x16xi8>) -> tensor<1x8x8x64xi8>
  return %0 : tensor<1x8x8x64xi8>
}

//===--------------------------------------------------------------------===//
// A standalone rescale -- an identity MATMUL into ACC, then VQUANT
//===--------------------------------------------------------------------===//
//
// `VQUANT` *is* a rescale, but it can only read ACC (isa.h `KeaVquant`:
// `acc_addr` is an ACC word index), and ACC is writable only by the MXU and
// the DWU -- ISA.md §10.4 says so outright: "ACC is not reachable from VCOPY;
// initialize accumulators with MATMUL/DWCONV accumulate=0 instead." A tensor
// sitting in SPM_A therefore cannot be requantized in place.
//
// The way in is to multiply it by the identity: `LOAD_W` a 16x16 identity and
// `MATMUL` streams a 16-channel slice of SPM_A into ACC unchanged, because
// `acc[n] = sum_k A[k]*I[k][n] = A[n]`. The zero-point fold -kea-emit already
// does for every contraction, `bias[c] = bias_l1[c] - input_zp*sum_k w[c][k]`,
// then becomes exactly `-input_zp` because the identity's row sum is 1 -- which
// is what a rescale needs. So this needs no new emitter contract at all.
//
// This is what unblocks a scale-changing pool: MobileNetV2's head converts
// 0.023529 -> 0.016946 across its global average pool, which the frontend
// emits as avg_pool2d + tosa.rescale, and `kea.pool`'s `quant` is restricted
// by its own verifier to a zero-point rebase.

// CHECK-LABEL: func.func @standalone_rescale
// The identity, laid out through the ordinary weight path.
// CHECK: kea.alloc from %{{.*}} : tensor<16x1x1x16xi8> {layout = "mxu_tiles_16x16", {{.*}}role = "weights"} : !kea.buffer<256xi8, DRAM>
// One KeaQuantParam block per 16-channel group; per tensor, so they share the
// same `quant` and differ only in where they land in SPM_W.
// CHECK: kea.alloc from %{{.*}} {input_zp = -128 : i64, layout = "quant_params", {{.*}}role = "qparam"} : !kea.buffer<192xi8, DRAM>
// CHECK: kea.trace "begin"
// 64 channels = 4 groups. Each is one identity LOAD_W, one MATMUL that copies
// the group into ACC (`accumulate` absent -- it overwrites), and one VQUANT.
// CHECK: kea.load_w %{{.*}} {bank = 0 : i64, k_rows = 16 : i64, n_cols = 16 : i64, w_addr = 0 : i64, w_row_stride = 16 : i64}
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 0 : i64, a_inner_stride = 64 : i64, a_outer_stride = 64 : i64, acc_addr = 0 : i64, acc_inner_stride = 16 : i64, acc_outer_stride = 16 : i64, bank = 0 : i64, m_inner = 1 : i64, m_outer = 1 : i64}
// CHECK-NEXT: kea.vquant %{{.*}} {acc_addr = 0 : i64, acc_pix_stride = 16 : i64, channels = 16 : i64, clamp_hi = 127 : i64, clamp_lo = -128 : i64, num_pixels = 1 : i64, out_addr = 0 : i64, out_pix_stride = 64 : i64, out_zp = -128 : i64, qparam_addr = 0 : i64}
// The next group reads the next 16 channels of the same pixel and writes the
// next 16 bytes of the output, with the bank alternating as everywhere else.
// CHECK-NEXT: kea.load_w %{{.*}} {bank = 1 : i64,
// CHECK-NEXT: kea.mm %{{.*}} {a_addr = 16 : i64, {{.*}}bank = 1 : i64
// CHECK-NEXT: kea.vquant %{{.*}} {{{.*}}out_addr = 16 : i64, {{.*}}qparam_addr = 192 : i64}
// CHECK: kea.dma_store
// CHECK: kea.halt
func.func @standalone_rescale(%x: tensor<1x1x1x64xi8>) -> tensor<1x1x1x64xi8> {
  %0 = kea.rescale %x {quant = #kea.quant<multiplier = [1490907399],
        shift = [30], input_zp = -128, output_zp = -128, axis = -1,
        rounding = DOUBLE>}
    : (tensor<1x1x1x64xi8>) -> tensor<1x1x1x64xi8>
  return %0 : tensor<1x1x1x64xi8>
}
