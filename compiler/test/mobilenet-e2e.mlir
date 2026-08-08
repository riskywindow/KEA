// RUN: kea-opt %s -tosa-to-kea -kea-fuse | FileCheck %s
// RUN: kea-opt %s -tosa-to-kea -kea-fuse=report-stats=true | FileCheck %s --check-prefix=STATS
// RUN: kea-opt %s -tosa-to-kea -kea-fuse -kea-tile=report-tiles=true | FileCheck %s --check-prefix=TILEFIT
// RUN: kea-opt %s -tosa-to-kea -kea-fuse -kea-tile | kea-opt | FileCheck %s --check-prefix=L2
//
// END-TO-END INTEGRATION TEST: a complete MobileNetV2 inverted-residual block
// in quantized TOSA, all the way to fully-fused KEA Level 1.
//
// The source below is a verbatim copy of tests/mlir/tosa/mobilenet_block.mlir,
// which is read-only reference material and lives outside compiler/test, where
// run_tests.sh looks. Structure:
//
//   1x1 expand conv (4 -> 24)  -> per-channel rescale -> ReLU6
//   3x3 depthwise s1 (24)      -> per-channel rescale -> ReLU6
//   1x1 project conv (24 -> 4) -> per-channel rescale        [linear bottleneck]
//   residual add (rescale both sides to a common i32, add, rescale to i8)
//
// TWELVE tosa compute ops (3 convs + 6 rescales + 2 clamps + 1 add) collapse to
// THREE kea ops. The residual add lands in the projection conv's epilogue, so
// the whole block becomes three MXU/DWU passes each with one fused VPU
// epilogue -- which is exactly what the hardware executes.

//===--------------------------------------------------------------------===//
// The fully fused result.
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @mobilenet_v2_inverted_residual
// CHECK-SAME:  (%[[X:[^:]*]]: tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi8>

// 1. Expand: 1x1 conv 4 -> 24, per-channel requantize to zp -128, ReLU6 clamp.
// CHECK-NOT: tosa.
// CHECK: %[[EXP:[^ ]*]] = kea.conv2d %[[X]], %{{[^ ]*}} bias %{{[^ ]*}} {dilations = array<i64: 1, 1>,
// CHECK-SAME: epilogue = #kea.epilogue<requant = <multiplier = [1073741824, 1181116006, 1288490189, 1395864371, 1503238553, 1610612736, 1717986918, 1825361101, 1932735283, 2040109466, 1073741824, 1181116006, 1288490189, 1395864371, 1503238553, 1610612736, 1717986918, 1825361101, 1932735283, 2040109466, 1073741824, 1181116006, 1288490189, 1395864371], shift = [36, 36, 36, 36, 37, 37, 37, 37, 36, 36, 36, 36, 37, 37, 37, 37, 36, 36, 36, 36, 37, 37, 37, 37], input_zp = 0, output_zp = -128, axis = 3, rounding = DOUBLE>, clamp = [-128, 127]>,
// CHECK-SAME: zero_points = #kea.zp<input = -5, weight = 0>}
// CHECK-SAME: -> tensor<1x8x8x24xi8>

// 2. Depthwise 3x3 s1 pad 1. The HWCM constant weights were relaid out to the
//    canonical [24, 3, 3, 1] at conversion time, so no transpose survives.
// CHECK-NOT: kea.rescale
// CHECK-NOT: kea.clamp
// CHECK-NOT: kea.transpose
// CHECK: %[[DW:[^ ]*]] = kea.dwconv2d %[[EXP]], %{{[^ ]*}} bias %{{[^ ]*}} {dilations = array<i64: 1, 1>,
// CHECK-SAME: clamp = [-128, 127]>, pads = array<i64: 1, 1, 1, 1>, strides = array<i64: 1, 1>,
// CHECK-SAME: zero_points = #kea.zp<input = -128, weight = 0>}
// CHECK-SAME: (tensor<1x8x8x24xi8>, tensor<24x3x3x1xi8>, tensor<24xi32>) -> tensor<1x8x8x24xi8>

// 3. Project 24 -> 4, linear bottleneck (no activation clamp), with the
//    residual add folded in: `residual %[[X]]` is the block input, and
//    accum/residual/output are the three KEA_VADD requantizations.
// CHECK-NOT: kea.rescale
// CHECK-NOT: kea.clamp
// CHECK-NOT: kea.add
// CHECK: kea.conv2d %[[DW]], %{{[^ ]*}} bias %{{[^ ]*}} residual %[[X]] {dilations = array<i64: 1, 1>,
// CHECK-SAME: requant = <multiplier = [1288490189, 1395864371, 1503238553, 1610612736], shift = [37, 37, 38, 38], input_zp = 0, output_zp = -5, axis = 3, rounding = DOUBLE>,
// CHECK-SAME: accum = <multiplier = [1610612736], shift = [11], input_zp = -5, output_zp = 0, axis = -1, rounding = DOUBLE>,
// CHECK-SAME: residual = <multiplier = [1073741824], shift = [10], input_zp = -5, output_zp = 0, axis = -1, rounding = DOUBLE>,
// CHECK-SAME: output = <multiplier = [1503238553], shift = [40], input_zp = 0, output_zp = -5, axis = -1, rounding = DOUBLE>>,
// CHECK-SAME: -> tensor<1x8x8x4xi8>
// CHECK-NOT: kea.rescale
// CHECK-NOT: kea.clamp
// CHECK-NOT: kea.add
// CHECK: return

// The stride-2 variant: no residual (the spatial size changes), so the
// projection conv keeps a bare requant epilogue and the block is three ops
// with nothing else between them.
// CHECK-LABEL: func.func @mobilenet_v2_inverted_residual_stride2
// CHECK: kea.conv2d {{.*}}clamp = [-128, 127]{{.*}} -> tensor<1x8x8x24xi8>
// CHECK-NOT: kea.rescale
// CHECK-NOT: kea.clamp
// CHECK: kea.dwconv2d {{.*}}pads = array<i64: 0, 1, 0, 1>, strides = array<i64: 2, 2>{{.*}} -> tensor<1x4x4x24xi8>
// CHECK-NOT: kea.rescale
// CHECK-NOT: kea.clamp
// CHECK-NOT: residual
// CHECK: kea.conv2d {{.*}} -> tensor<1x4x4x8xi8>
// CHECK-NOT: kea.rescale
// CHECK: return

//===--------------------------------------------------------------------===//
// Fusion counts. `-mlir-pass-statistics` is inert on this toolchain (the
// Homebrew LLVM is Release+NDEBUG, so llvm::Statistic is NoopStatistic and even
// upstream mlir-opt prints an empty report), hence the report-stats option --
// it publishes the same counters as an attribute. See Transforms/Passes.td.
//===--------------------------------------------------------------------===//

// STATS-LABEL: func.func @mobilenet_v2_inverted_residual
// STATS-SAME: kea.fusion_stats = {bias = 0 : i64, clamp = 2 : i64, quant_add = 1 : i64, requant = 3 : i64, rescale_composed = 0 : i64, rescale_refused = 0 : i64, rescale_removed = 0 : i64, residual = 1 : i64, shape_folded = 0 : i64}

// STATS-LABEL: func.func @mobilenet_v2_inverted_residual_stride2
// STATS-SAME: kea.fusion_stats = {bias = 0 : i64, clamp = 2 : i64, quant_add = 0 : i64, requant = 3 : i64, rescale_composed = 0 : i64, rescale_refused = 0 : i64, rescale_removed = 0 : i64, residual = 0 : i64, shape_folded = 0 : i64}


//===--------------------------------------------------------------------===//
// ...and all the way down to Level 2 machine ops.
//
// -kea-tile turns the three fused Level 1 ops into a straight-line KEA-1
// instruction stream on symbolic buffers: a MATMUL chain for the expand conv,
// a DWCONV for the depthwise, a MATMUL chain plus VADD for the projection and
// its residual. The second RUN line pipes the result back through kea-opt, so
// everything below is verified L2 that parses, verifies and round-trips.
//===--------------------------------------------------------------------===//

// The function is now a machine program: it returns nothing and ends in HALT.
// L2-LABEL: func.func @mobilenet_v2_inverted_residual
// L2-SAME:  (%{{[^:]*}}: tensor<1x8x8x4xi8>)
// L2-NOT:   -> tensor

// Not one Level 1 op survives. (The `= ` anchors these to an SSA definition;
// the tiling report names the ops it costed, which is not the same thing.)
// L2-NOT: = kea.conv2d
// L2-NOT: = kea.dwconv2d
// L2-NOT: = kea.rescale
// L2-NOT: = kea.clamp
// L2-NOT: = kea.add

// The block input is a DRAM symbol; so is every inter-layer feature map.
// L2: kea.alloc {name = "mobilenet_v2_inverted_residual.input0", role = "input"} : !kea.buffer<256xi8, DRAM>

// 1. Expand 4 -> 24. Two output-channel groups, so two ACC regions of
//    8*8*16 = 1024 words and two VQUANTs. IC = 4 means k_rows = 4 and the
//    other 12 reduction lanes are the zeros LOAD_W installs.
// L2: kea.trace "begin" 0
// L2: kea.load_w %{{.*}} {bank = 0 : i64, k_rows = 4 : i64, n_cols = 16 : i64
// L2: kea.mm %{{.*}} {a_addr = 0 : i64, a_inner_stride = 4 : i64, a_outer_stride = 32 : i64, acc_addr = 0 : i64, acc_inner_stride = 16 : i64, acc_outer_stride = 128 : i64, bank = 0 : i64, m_inner = 8 : i64, m_outer = 8 : i64}
// L2: kea.vquant %{{.*}} {acc_addr = 0 : i64, {{.*}}out_pix_stride = 32 : i64, out_zp = -128 : i64
//    The 24-channel tail group: n_cols = 8, and its ACC region starts at 1024.
//    NOTE THE BANK. This group is one LOAD_W/MATMUL pair, as is the group
//    before it, so restarting the tap counter per output-channel group (the
//    literal reading of ISA.md §8.3) would put BOTH loads in bank 0 and
//    serialise this one behind the previous MATMUL. The counter is monotonic
//    over the whole MXU stream instead, so it lands in the idle bank.
// L2: kea.load_w %{{.*}} {bank = 1 : i64, k_rows = 4 : i64, n_cols = 8 : i64
// L2: kea.mm %{{.*}} {a_addr = 0 : i64, {{.*}}acc_addr = 1024 : i64, {{.*}}bank = 1 : i64
// L2: kea.dma_store
// L2: kea.trace "end" 0

// 2. Depthwise 3x3 s1 pad 1 on 24 channels. DWCONV needs a multiple of 16, so
//    the tile is padded to 32 channels and the extra 8 weight lanes are zero.
//    The halo (and the padded lanes) get the input zero point, -128.
// L2: kea.trace "begin" 1
// L2: kea.vcopy to %{{.*}}fill, fill_value = -128 : i64
// L2: kea.dwconv %{{.*}} {a_addr = 0 : i64, a_pix_stride = 32 : i64, a_row_stride = 320 : i64, acc_addr = 0 : i64, channels = 32 : i64, kernel = 3 : i64, out_h = 8 : i64, out_w = 8 : i64, stride = 1 : i64, w_addr = 0 : i64}
// L2: kea.vquant %{{.*}} {acc_addr = 0 : i64, acc_pix_stride = 32 : i64, channels = 32 : i64
// L2: kea.trace "end" 1

// 3. Project 24 -> 4 plus the inverted-residual add. Two reduction tiles
//    (k_rows 16 then 8), then VQUANT, then VADD against the block input.
//    The derived KeaAddParam is on the addparam allocation, in isa.h field
//    order, and satisfies errata E6.
// L2: kea.alloc {add_param = array<i64: 1610612736, 1073741824, 1503238553, 1, 0, 8, -5, -5, -5>, layout = "add_params"
// L2: kea.trace "begin" 2
// L2: kea.load_w %{{.*}} {bank = 0 : i64, k_rows = 16 : i64, n_cols = 4 : i64, w_addr = 0 : i64
// L2: kea.mm %{{.*}} {a_addr = 0 : i64, a_inner_stride = 24 : i64, a_outer_stride = 192 : i64, acc_addr = 0 : i64, {{.*}}bank = 0 : i64
// L2: kea.load_w %{{.*}} {bank = 1 : i64, k_rows = 8 : i64, n_cols = 4 : i64, w_addr = 256 : i64
// L2: kea.mm %{{.*}} {a_addr = 16 : i64, {{.*}}accumulate, bank = 1 : i64
// L2: kea.vquant
// L2: kea.vadd %{{.*}} {a_addr = 0 : i64, b_addr = 0 : i64, clamp_hi = 127 : i64, clamp_lo = -128 : i64, num_elems = 1024 : i64
// L2: kea.dma_store %{{.*}} : !kea.buffer<1040xi8, A> -> !kea.buffer<256xi8, DRAM>
// L2: kea.trace "end" 2
// L2: kea.halt

// -kea-tile assigns NO addresses, NO queues and NO semaphores: that is
// -kea-alloc's and -kea-schedule's work.
// L2-NOT: kea.signal
// L2-NOT: kea.wait

//===--------------------------------------------------------------------===//
// Every chosen tile fits the real scratchpad: SPM_A and SPM_W are 256 KiB
// (KEA_SPM_A_BYTES / KEA_SPM_W_BYTES) and -kea-tile spends at most half of
// each so -kea-schedule can double buffer; ACC is 32768 int32 words.
//===--------------------------------------------------------------------===//

// TILEFIT-LABEL: func.func @mobilenet_v2_inverted_residual
// TILEFIT-SAME: kea.tiling = [{acc = 2048 : i64, cycles = {{[0-9]+}} : i64, dram = {{[0-9]+}} : i64, instrs = {{[0-9]+}} : i64, layer = 0 : i64, oc_groups = 2 : i64, oh = 8 : i64, op = "kea.conv2d", ow = 8 : i64, spm_a = 2336 : i64, spm_w = 896 : i64, taps = 16 : i64},
// TILEFIT-SAME: {acc = 2048 : i64, channels = 32 : i64, dram = {{[0-9]+}} : i64, instrs = {{[0-9]+}} : i64, layer = 1 : i64, oh = 8 : i64, op = "kea.dwconv2d", ow = 8 : i64, spm_a = 5280 : i64, spm_w = 672 : i64, taps = 9 : i64},
// TILEFIT-SAME: {acc = 1024 : i64, cycles = {{[0-9]+}} : i64, dram = {{[0-9]+}} : i64, instrs = {{[0-9]+}} : i64, layer = 2 : i64, oc_groups = 1 : i64, oh = 8 : i64, op = "kea.conv2d", ow = 8 : i64, spm_a = 2592 : i64, spm_w = 704 : i64, taps = 32 : i64}]

// TILEFIT-LABEL: func.func @mobilenet_v2_inverted_residual_stride2
// TILEFIT-SAME: kea.tiling = [{acc = 2048 : i64, cycles = {{[0-9]+}} : i64, dram = {{[0-9]+}} : i64, instrs = {{[0-9]+}} : i64, layer = 0 : i64, oc_groups = 2 : i64, oh = 8 : i64, op = "kea.conv2d", ow = 8 : i64, spm_a = 2336 : i64, spm_w = 896 : i64, taps = 16 : i64},
// TILEFIT-SAME: {acc = 512 : i64, channels = 32 : i64, dram = {{[0-9]+}} : i64, instrs = {{[0-9]+}} : i64, layer = 1 : i64, oh = 4 : i64, op = "kea.dwconv2d", ow = 4 : i64, spm_a = 3136 : i64, spm_w = 672 : i64, taps = 9 : i64},
// TILEFIT-SAME: {acc = 256 : i64, cycles = {{[0-9]+}} : i64, dram = {{[0-9]+}} : i64, instrs = {{[0-9]+}} : i64, layer = 2 : i64, oc_groups = 1 : i64, oh = 4 : i64, op = "kea.conv2d", ow = 4 : i64, spm_a = 672 : i64, spm_w = 704 : i64, taps = 32 : i64}]

// The stride-2 depthwise: 8x8 -> 4x4 with pad [0,1,0,1].
// L2-LABEL: func.func @mobilenet_v2_inverted_residual_stride2
// L2: kea.dwconv %{{.*}}stride = 2 : i64
// L2: kea.halt

func.func @mobilenet_v2_inverted_residual(%x: tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi8> {

  //===------------------------------------------------------------------===//
  // 1. Expand: 1x1 pointwise convolution, 4 -> 24 channels.
  //===------------------------------------------------------------------===//
  %w_exp = "tosa.const"() {value = dense<2> : tensor<24x1x1x4xi8>}
      : () -> tensor<24x1x1x4xi8>
  %b_exp = "tosa.const"() {value = dense<128> : tensor<24xi32>}
      : () -> tensor<24xi32>

  %exp_acc = tosa.conv2d %x, %w_exp, %b_exp {
    pad = array<i64: 0, 0, 0, 0>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -5, weight_zp = 0>
  } : (tensor<1x8x8x4xi8>, tensor<24x1x1x4xi8>, tensor<24xi32>)
      -> tensor<1x8x8x24xi32>

  // Per-channel requantization back to i8. 24 multipliers / 24 shifts,
  // one per output channel (the last dimension).
  %exp_q = tosa.rescale %exp_acc {
    input_zp = 0 : i32,
    output_zp = -128 : i32,
    multiplier = array<i32: 1073741824, 1181116006, 1288490189, 1395864371,
                            1503238553, 1610612736, 1717986918, 1825361101,
                            1932735283, 2040109466, 1073741824, 1181116006,
                            1288490189, 1395864371, 1503238553, 1610612736,
                            1717986918, 1825361101, 1932735283, 2040109466,
                            1073741824, 1181116006, 1288490189, 1395864371>,
    shift = array<i8: 36, 36, 36, 36, 37, 37, 37, 37,
                       36, 36, 36, 36, 37, 37, 37, 37,
                       36, 36, 36, 36, 37, 37, 37, 37>,
    scale32 = true,
    double_round = true,
    per_channel = true
  } : (tensor<1x8x8x24xi32>) -> tensor<1x8x8x24xi8>

  // ReLU6 in the i8 domain (zero point -128).
  %exp_relu = tosa.clamp %exp_q {
    min_int = -128 : i64,
    max_int = 127 : i64,
    min_fp = 0.0 : f32,
    max_fp = 6.0 : f32
  } : (tensor<1x8x8x24xi8>) -> tensor<1x8x8x24xi8>

  //===------------------------------------------------------------------===//
  // 2. Depthwise: 3x3, stride 1, pad 1, channel multiplier 1, 24 channels.
  //     NOTE the HWCM weight layout [KH, KW, C, M] -- different from conv2d.
  //===------------------------------------------------------------------===//
  %w_dw = "tosa.const"() {value = dense<1> : tensor<3x3x24x1xi8>}
      : () -> tensor<3x3x24x1xi8>
  %b_dw = "tosa.const"() {value = dense<64> : tensor<24xi32>}
      : () -> tensor<24xi32>

  %dw_acc = tosa.depthwise_conv2d %exp_relu, %w_dw, %b_dw {
    pad = array<i64: 1, 1, 1, 1>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x8x8x24xi8>, tensor<3x3x24x1xi8>, tensor<24xi32>)
      -> tensor<1x8x8x24xi32>

  %dw_q = tosa.rescale %dw_acc {
    input_zp = 0 : i32,
    output_zp = -128 : i32,
    multiplier = array<i32: 1503238553, 1610612736, 1717986918, 1825361101,
                            1932735283, 2040109466, 1073741824, 1181116006,
                            1288490189, 1395864371, 1503238553, 1610612736,
                            1717986918, 1825361101, 1932735283, 2040109466,
                            1073741824, 1181116006, 1288490189, 1395864371,
                            1503238553, 1610612736, 1717986918, 1825361101>,
    shift = array<i8: 37, 37, 37, 37, 36, 36, 36, 36,
                       37, 37, 37, 37, 36, 36, 36, 36,
                       37, 37, 37, 37, 36, 36, 36, 36>,
    scale32 = true,
    double_round = true,
    per_channel = true
  } : (tensor<1x8x8x24xi32>) -> tensor<1x8x8x24xi8>

  %dw_relu = tosa.clamp %dw_q {
    min_int = -128 : i64,
    max_int = 127 : i64,
    min_fp = 0.0 : f32,
    max_fp = 6.0 : f32
  } : (tensor<1x8x8x24xi8>) -> tensor<1x8x8x24xi8>

  //===------------------------------------------------------------------===//
  // 3. Project: 1x1 pointwise convolution, 24 -> 4 channels.
  //     Linear bottleneck -- NO activation after this rescale.
  //===------------------------------------------------------------------===//
  %w_proj = "tosa.const"() {value = dense<3> : tensor<4x1x1x24xi8>}
      : () -> tensor<4x1x1x24xi8>
  %b_proj = "tosa.const"() {value = dense<-32> : tensor<4xi32>}
      : () -> tensor<4xi32>

  %proj_acc = tosa.conv2d %dw_relu, %w_proj, %b_proj {
    pad = array<i64: 0, 0, 0, 0>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x8x8x24xi8>, tensor<4x1x1x24xi8>, tensor<4xi32>)
      -> tensor<1x8x8x4xi32>

  %proj_q = tosa.rescale %proj_acc {
    input_zp = 0 : i32,
    output_zp = -5 : i32,
    multiplier = array<i32: 1288490189, 1395864371, 1503238553, 1610612736>,
    shift = array<i8: 37, 37, 38, 38>,
    scale32 = true,
    double_round = true,
    per_channel = true
  } : (tensor<1x8x8x4xi32>) -> tensor<1x8x8x4xi8>

  //===------------------------------------------------------------------===//
  // 4. Residual add. TOSA has no quantized-add attribute, so both operands
  //    must be rescaled into a shared i32 domain, added, then rescaled back.
  //===------------------------------------------------------------------===//

  // Identity path: block input, zero point -5, into the common i32 domain.
  %id_i32 = tosa.rescale %x {
    input_zp = -5 : i32,
    output_zp = 0 : i32,
    multiplier = array<i32: 1073741824>,
    shift = array<i8: 10>,
    scale32 = true,
    double_round = true,
    per_channel = false
  } : (tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi32>

  // Residual path: projection output, zero point -5, same common i32 domain.
  %res_i32 = tosa.rescale %proj_q {
    input_zp = -5 : i32,
    output_zp = 0 : i32,
    multiplier = array<i32: 1610612736>,
    shift = array<i8: 11>,
    scale32 = true,
    double_round = true,
    per_channel = false
  } : (tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi32>

  %sum_i32 = tosa.add %id_i32, %res_i32
      : (tensor<1x8x8x4xi32>, tensor<1x8x8x4xi32>) -> tensor<1x8x8x4xi32>

  // Back to i8 at the block output scale / zero point.
  %y = tosa.rescale %sum_i32 {
    input_zp = 0 : i32,
    output_zp = -5 : i32,
    multiplier = array<i32: 1503238553>,
    shift = array<i8: 40>,
    scale32 = true,
    double_round = true,
    per_channel = false
  } : (tensor<1x8x8x4xi32>) -> tensor<1x8x8x4xi8>

  return %y : tensor<1x8x8x4xi8>
}

//===--------------------------------------------------------------------===//
// Stride-2 variant: spatial size changes, so there is NO residual connection.
// This is the other half of the MobileNetV2 block vocabulary.
//===--------------------------------------------------------------------===//
func.func @mobilenet_v2_inverted_residual_stride2(%x: tensor<1x8x8x4xi8>)
    -> tensor<1x4x4x8xi8> {
  %w_exp = "tosa.const"() {value = dense<2> : tensor<24x1x1x4xi8>}
      : () -> tensor<24x1x1x4xi8>
  %b_exp = "tosa.const"() {value = dense<128> : tensor<24xi32>} : () -> tensor<24xi32>
  %exp_acc = tosa.conv2d %x, %w_exp, %b_exp {
    pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -5, weight_zp = 0>
  } : (tensor<1x8x8x4xi8>, tensor<24x1x1x4xi8>, tensor<24xi32>)
      -> tensor<1x8x8x24xi32>
  %exp_q = tosa.rescale %exp_acc {
    input_zp = 0 : i32, output_zp = -128 : i32,
    multiplier = array<i32: 1073741824>, shift = array<i8: 36>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x8x8x24xi32>) -> tensor<1x8x8x24xi8>
  %exp_relu = tosa.clamp %exp_q {
    min_int = -128 : i64, max_int = 127 : i64,
    min_fp = 0.0 : f32, max_fp = 6.0 : f32
  } : (tensor<1x8x8x24xi8>) -> tensor<1x8x8x24xi8>

  // Stride-2 depthwise: 8x8 -> 4x4, asymmetric pad [top=0,bot=1,left=0,right=1].
  %w_dw = "tosa.const"() {value = dense<1> : tensor<3x3x24x1xi8>}
      : () -> tensor<3x3x24x1xi8>
  %b_dw = "tosa.const"() {value = dense<64> : tensor<24xi32>} : () -> tensor<24xi32>
  %dw_acc = tosa.depthwise_conv2d %exp_relu, %w_dw, %b_dw {
    pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>,
    dilation = array<i64: 1, 1>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x8x8x24xi8>, tensor<3x3x24x1xi8>, tensor<24xi32>)
      -> tensor<1x4x4x24xi32>
  %dw_q = tosa.rescale %dw_acc {
    input_zp = 0 : i32, output_zp = -128 : i32,
    multiplier = array<i32: 1503238553>, shift = array<i8: 37>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x4x4x24xi32>) -> tensor<1x4x4x24xi8>
  %dw_relu = tosa.clamp %dw_q {
    min_int = -128 : i64, max_int = 127 : i64,
    min_fp = 0.0 : f32, max_fp = 6.0 : f32
  } : (tensor<1x4x4x24xi8>) -> tensor<1x4x4x24xi8>

  // Project 24 -> 8, linear (no activation), no residual.
  %w_proj = "tosa.const"() {value = dense<3> : tensor<8x1x1x24xi8>}
      : () -> tensor<8x1x1x24xi8>
  %b_proj = "tosa.const"() {value = dense<-32> : tensor<8xi32>} : () -> tensor<8xi32>
  %proj_acc = tosa.conv2d %dw_relu, %w_proj, %b_proj {
    pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x4x4x24xi8>, tensor<8x1x1x24xi8>, tensor<8xi32>)
      -> tensor<1x4x4x8xi32>
  %y = tosa.rescale %proj_acc {
    input_zp = 0 : i32, output_zp = -5 : i32,
    multiplier = array<i32: 1288490189>, shift = array<i8: 37>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %y : tensor<1x4x4x8xi8>
}
