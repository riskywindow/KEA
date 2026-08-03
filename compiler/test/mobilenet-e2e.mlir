// RUN: kea-opt %s -tosa-to-kea -kea-fuse | FileCheck %s
// RUN: kea-opt %s -tosa-to-kea -kea-fuse=report-stats=true | FileCheck %s --check-prefix=STATS
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
