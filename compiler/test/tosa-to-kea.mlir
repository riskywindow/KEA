// RUN: kea-opt %s -tosa-to-kea | FileCheck %s
//
// Per-pattern coverage of -tosa-to-kea. Syntax is TOSA as it exists in
// LLVM/MLIR 20.1.6 (docs/TOSA_NOTES.md): zero points are attributes,
// tosa.fully_connected still exists, tosa.reshape's shape is an attribute and
// tosa.transpose's permutation is an operand.
//
// The pass NORMALISES ONLY. conv + rescale + clamp stays three ops here; see
// kea-fuse.mlir for the fusion.

//===--------------------------------------------------------------------===//
// tosa.const -> arith.constant
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @const
func.func @const() -> tensor<16xi32> {
  // CHECK: arith.constant dense<100> : tensor<16xi32>
  // CHECK-NOT: tosa.const
  %0 = "tosa.const"() {value = dense<100> : tensor<16xi32>} : () -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

//===--------------------------------------------------------------------===//
// tosa.conv2d -> kea.conv2d (weights are OHWI on both sides: no relayout)
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @conv2d
func.func @conv2d(%input: tensor<1x8x8x4xi8>, %weight: tensor<16x3x3x4xi8>,
                  %bias: tensor<16xi32>) -> tensor<1x8x8x16xi32> {
  // CHECK: kea.conv2d %arg0, %arg1 bias %arg2 {dilations = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>, strides = array<i64: 1, 1>, zero_points = #kea.zp<input = -3, weight = 0>} : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi32>) -> tensor<1x8x8x16xi32>
  %0 = tosa.conv2d %input, %weight, %bias {
    pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -3, weight_zp = 0>
  } : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi32>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// Stride 2 with asymmetric pad, and dilation 2, both carried across verbatim.
// CHECK-LABEL: func.func @conv2d_stride2
func.func @conv2d_stride2(%input: tensor<1x16x16x8xi8>, %weight: tensor<32x3x3x8xi8>,
                          %bias: tensor<32xi32>) -> tensor<1x8x8x32xi32> {
  // CHECK: kea.conv2d {{.*}}pads = array<i64: 0, 1, 0, 1>, strides = array<i64: 2, 2>, zero_points = #kea.zp<input = -128, weight = 0>
  %0 = tosa.conv2d %input, %weight, %bias {
    pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>,
    dilation = array<i64: 1, 1>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x16x16x8xi8>, tensor<32x3x3x8xi8>, tensor<32xi32>) -> tensor<1x8x8x32xi32>
  return %0 : tensor<1x8x8x32xi32>
}

// CHECK-LABEL: func.func @conv2d_dilated
func.func @conv2d_dilated(%input: tensor<1x8x8x4xi8>, %weight: tensor<16x3x3x4xi8>,
                          %bias: tensor<16xi32>) -> tensor<1x8x8x16xi32> {
  // CHECK: kea.conv2d {{.*}}dilations = array<i64: 2, 2>
  %0 = tosa.conv2d %input, %weight, %bias {
    pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>,
    dilation = array<i64: 2, 2>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = 7, weight_zp = 0>
  } : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi32>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// NOTE: `quantization_info` cannot actually be omitted on a quantized
// tosa.conv2d -- the TOSA verifier rejects it with "quantizationattr is
// required for quantized type" -- so the "absent means 0/0" path is only
// reachable via tosa.matmul / tosa.avg_pool2d, both covered below.

//===--------------------------------------------------------------------===//
// tosa.depthwise_conv2d -> kea.dwconv2d, weights HWCM -> canonical
//===--------------------------------------------------------------------===//

// Non-constant weights: the relayout is materialized as kea.transpose (HWCM ->
// CMHW) followed by an element-order-preserving kea.reshape.
// CHECK-LABEL: func.func @depthwise_dynamic_weights
func.func @depthwise_dynamic_weights(%input: tensor<1x8x8x16xi8>,
                                     %weight: tensor<3x3x16x1xi8>,
                                     %bias: tensor<16xi32>) -> tensor<1x8x8x16xi32> {
  // CHECK: %[[T:.*]] = kea.transpose %arg1 {perms = array<i64: 2, 3, 0, 1>} : (tensor<3x3x16x1xi8>) -> tensor<16x1x3x3xi8>
  // CHECK: %[[R:.*]] = kea.reshape %[[T]] {new_shape = array<i64: 16, 3, 3, 1>} : (tensor<16x1x3x3xi8>) -> tensor<16x3x3x1xi8>
  // CHECK: kea.dwconv2d %arg0, %[[R]] bias %arg2 {dilations = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>, strides = array<i64: 1, 1>, zero_points = #kea.zp<input = -7, weight = 0>} : (tensor<1x8x8x16xi8>, tensor<16x3x3x1xi8>, tensor<16xi32>) -> tensor<1x8x8x16xi32>
  %0 = tosa.depthwise_conv2d %input, %weight, %bias {
    pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -7, weight_zp = 0>
  } : (tensor<1x8x8x16xi8>, tensor<3x3x16x1xi8>, tensor<16xi32>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// Constant weights: the relayout is constant-folded, so no ops are left behind
// and the original HWCM constant is dead-code eliminated.
// CHECK-LABEL: func.func @depthwise_const_weights
func.func @depthwise_const_weights(%input: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi32> {
  // CHECK-NOT: kea.transpose
  // CHECK: %[[W:.*]] = arith.constant dense<1> : tensor<16x3x3x1xi8>
  // CHECK: kea.dwconv2d %arg0, %[[W]] bias
  %w = "tosa.const"() {value = dense<1> : tensor<3x3x16x1xi8>} : () -> tensor<3x3x16x1xi8>
  %b = "tosa.const"() {value = dense<0> : tensor<16xi32>} : () -> tensor<16xi32>
  %0 = tosa.depthwise_conv2d %input, %w, %b {
    pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x8x8x16xi8>, tensor<3x3x16x1xi8>, tensor<16xi32>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// Channel multiplier 2: HWCM [3, 3, 8, 2] becomes canonical [16, 3, 3, 1].
// CHECK-LABEL: func.func @depthwise_m2
func.func @depthwise_m2(%input: tensor<1x8x8x8xi8>, %weight: tensor<3x3x8x2xi8>,
                        %bias: tensor<16xi32>) -> tensor<1x8x8x16xi32> {
  // CHECK: kea.transpose %arg1 {perms = array<i64: 2, 3, 0, 1>} : (tensor<3x3x8x2xi8>) -> tensor<8x2x3x3xi8>
  // CHECK: kea.reshape {{.*}} {new_shape = array<i64: 16, 3, 3, 1>} : (tensor<8x2x3x3xi8>) -> tensor<16x3x3x1xi8>
  // CHECK: kea.dwconv2d
  %0 = tosa.depthwise_conv2d %input, %weight, %bias {
    pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x8x8x8xi8>, tensor<3x3x8x2xi8>, tensor<16xi32>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

//===--------------------------------------------------------------------===//
// tosa.matmul / tosa.fully_connected
//===--------------------------------------------------------------------===//

// matmul uses #tosa.matmul_quant<a_zp, b_zp>, which maps onto #kea.zp's
// input/weight fields in that order.
// CHECK-LABEL: func.func @matmul
func.func @matmul(%a: tensor<1x4x8xi8>, %b: tensor<1x8x16xi8>) -> tensor<1x4x16xi32> {
  // CHECK: kea.matmul %arg0, %arg1 {zero_points = #kea.zp<input = -2, weight = 1>} : (tensor<1x4x8xi8>, tensor<1x8x16xi8>) -> tensor<1x4x16xi32>
  %0 = tosa.matmul %a, %b {
    quantization_info = #tosa.matmul_quant<a_zp = -2, b_zp = 1>
  } : (tensor<1x4x8xi8>, tensor<1x8x16xi8>) -> tensor<1x4x16xi32>
  return %0 : tensor<1x4x16xi32>
}

// CHECK-LABEL: func.func @matmul_batched_no_quant
func.func @matmul_batched_no_quant(%a: tensor<4x32x64xi8>, %b: tensor<4x64x32xi8>)
    -> tensor<4x32x32xi32> {
  // CHECK: kea.matmul {{.*}}zero_points = #kea.zp<input = 0, weight = 0>
  %0 = tosa.matmul %a, %b : (tensor<4x32x64xi8>, tensor<4x64x32xi8>) -> tensor<4x32x32xi32>
  return %0 : tensor<4x32x32xi32>
}

// fully_connected uses #tosa.conv_quant and keeps the [OC, IC] weight layout.
// CHECK-LABEL: func.func @fully_connected
func.func @fully_connected(%a: tensor<4x8xi8>, %w: tensor<16x8xi8>, %b: tensor<16xi32>)
    -> tensor<4x16xi32> {
  // CHECK: kea.fully_connected %arg0, %arg1 bias %arg2 {zero_points = #kea.zp<input = -2, weight = 0>} : (tensor<4x8xi8>, tensor<16x8xi8>, tensor<16xi32>) -> tensor<4x16xi32>
  %0 = tosa.fully_connected %a, %w, %b {
    quantization_info = #tosa.conv_quant<input_zp = -2, weight_zp = 0>
  } : (tensor<4x8xi8>, tensor<16x8xi8>, tensor<16xi32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

// The classifier tail: fully_connected then rescale, still two ops here.
// CHECK-LABEL: func.func @fc_rescale
func.func @fc_rescale(%a: tensor<1x64xi8>) -> tensor<1x10xi8> {
  // CHECK: %[[FC:.*]] = kea.fully_connected
  // CHECK: kea.rescale %[[FC]] {quant = #kea.quant<multiplier = [1503238553], shift = [37], input_zp = 0, output_zp = -3, axis = -1, rounding = DOUBLE>}
  %w = "tosa.const"() {value = dense<3> : tensor<10x64xi8>} : () -> tensor<10x64xi8>
  %b = "tosa.const"() {value = dense<7> : tensor<10xi32>} : () -> tensor<10xi32>
  %acc = tosa.fully_connected %a, %w, %b {
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x64xi8>, tensor<10x64xi8>, tensor<10xi32>) -> tensor<1x10xi32>
  %out = tosa.rescale %acc {
    input_zp = 0 : i32, output_zp = -3 : i32,
    multiplier = array<i32: 1503238553>, shift = array<i8: 37>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x10xi32>) -> tensor<1x10xi8>
  return %out : tensor<1x10xi8>
}

//===--------------------------------------------------------------------===//
// tosa.rescale -> kea.rescale
//===--------------------------------------------------------------------===//

// per_channel = true becomes axis = rank - 1 (TOSA always indexes the LAST
// dimension); per_channel = false becomes axis = -1.
// CHECK-LABEL: func.func @rescale_per_channel
func.func @rescale_per_channel(%a: tensor<1x4x4x4xi32>) -> tensor<1x4x4x4xi8> {
  // CHECK: kea.rescale %arg0 {quant = #kea.quant<multiplier = [1073741824, 1234567890, 2000000000, 1500000000], shift = [30, 31, 32, 33], input_zp = 0, output_zp = -5, axis = 3, rounding = DOUBLE>} : (tensor<1x4x4x4xi32>) -> tensor<1x4x4x4xi8>
  %0 = tosa.rescale %a {
    input_zp = 0 : i32, output_zp = -5 : i32,
    multiplier = array<i32: 1073741824, 1234567890, 2000000000, 1500000000>,
    shift = array<i8: 30, 31, 32, 33>,
    scale32 = true, double_round = true, per_channel = true
  } : (tensor<1x4x4x4xi32>) -> tensor<1x4x4x4xi8>
  return %0 : tensor<1x4x4x4xi8>
}

// double_round = false becomes rounding = SINGLE.
// CHECK-LABEL: func.func @rescale_single_round
func.func @rescale_single_round(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  // CHECK: kea.rescale {{.*}}rounding = SINGLE
  %0 = tosa.rescale %a {
    input_zp = 0 : i32, output_zp = 0 : i32,
    multiplier = array<i32: 1073741824>, shift = array<i8: 33>,
    scale32 = true, double_round = false, per_channel = false
  } : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// The widening i8 -> i32 rescale that opens a quantized add.
// CHECK-LABEL: func.func @rescale_widening
func.func @rescale_widening(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32> {
  // CHECK: kea.rescale {{.*}}input_zp = -5, output_zp = 0{{.*}} : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32>
  %0 = tosa.rescale %a {
    input_zp = -5 : i32, output_zp = 0 : i32,
    multiplier = array<i32: 1073741824>, shift = array<i8: 20>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

//===--------------------------------------------------------------------===//
// tosa.clamp -> kea.clamp (min_fp / max_fp are dropped, as TOSA's own integer
// lowering drops them)
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @clamp_relu6
func.func @clamp_relu6(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {
  // CHECK: kea.clamp %arg0 {max = 91 : i64, min = -5 : i64} : tensor<1x4x4x8xi8>
  %0 = tosa.clamp %a {
    min_int = -5 : i64, max_int = 91 : i64,
    min_fp = 0.0 : f32, max_fp = 6.0 : f32
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

//===--------------------------------------------------------------------===//
// tosa.add -> kea.add (no quantization; TOSA has none on add by design)
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @add
func.func @add(%a: tensor<1x4x4x8xi32>, %b: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32> {
  // CHECK: kea.add %arg0, %arg1 : (tensor<1x4x4x8xi32>, tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %0 = tosa.add %a, %b : (tensor<1x4x4x8xi32>, tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// Rank-equal broadcasting survives.
// CHECK-LABEL: func.func @add_broadcast
func.func @add_broadcast(%a: tensor<1x4x4x8xi32>, %bias: tensor<8xi32>)
    -> tensor<1x4x4x8xi32> {
  // CHECK: kea.reshape {{.*}} {new_shape = array<i64: 1, 1, 1, 8>}
  // CHECK: kea.add {{.*}} : (tensor<1x4x4x8xi32>, tensor<1x1x1x8xi32>) -> tensor<1x4x4x8xi32>
  %r = tosa.reshape %bias {new_shape = array<i64: 1, 1, 1, 8>}
      : (tensor<8xi32>) -> tensor<1x1x1x8xi32>
  %0 = tosa.add %a, %r : (tensor<1x4x4x8xi32>, tensor<1x1x1x8xi32>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

//===--------------------------------------------------------------------===//
// Pooling
//===--------------------------------------------------------------------===//

// avg_pool2d uses #tosa.unary_quant<input_zp, output_zp>. Averaging is affine,
// so the zero-point rebase is exact and is carried as an identity #kea.quant
// (multiplier 2^30, shift 30).
// CHECK-LABEL: func.func @avg_pool
func.func @avg_pool(%a: tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8> {
  // CHECK: kea.pool %arg0 {kernel = array<i64: 2, 2>, kind = #kea.pool_kind<AVG>, pads = array<i64: 0, 0, 0, 0>, quant = #kea.quant<multiplier = [1073741824], shift = [30], input_zp = -5, output_zp = -5, axis = -1, rounding = SINGLE>, strides = array<i64: 2, 2>} : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  %0 = tosa.avg_pool2d %a {
    kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>,
    pad = array<i64: 0, 0, 0, 0>, acc_type = i32,
    quantization_info = #tosa.unary_quant<input_zp = -5, output_zp = -5>
  } : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  return %0 : tensor<1x4x4x16xi8>
}

// Both zero points zero, or no quantization_info at all: no quant attribute.
// CHECK-LABEL: func.func @avg_pool_zero_zp
func.func @avg_pool_zero_zp(%a: tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8> {
  // CHECK-NOT: quant =
  // CHECK: kea.pool %arg0 {kernel = array<i64: 2, 2>, kind = #kea.pool_kind<AVG>, pads = array<i64: 0, 0, 0, 0>, strides = array<i64: 2, 2>}
  %0 = tosa.avg_pool2d %a {
    kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>,
    pad = array<i64: 0, 0, 0, 0>, acc_type = i32,
    quantization_info = #tosa.unary_quant<input_zp = 0, output_zp = 0>
  } : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  return %0 : tensor<1x4x4x16xi8>
}

// Global average pool.
// CHECK-LABEL: func.func @global_avg_pool
func.func @global_avg_pool(%a: tensor<1x7x7x32xi8>) -> tensor<1x1x1x32xi8> {
  // CHECK: kea.pool {{.*}}kernel = array<i64: 7, 7>{{.*}}kind = #kea.pool_kind<AVG>
  %0 = tosa.avg_pool2d %a {
    kernel = array<i64: 7, 7>, stride = array<i64: 1, 1>,
    pad = array<i64: 0, 0, 0, 0>, acc_type = i32,
    quantization_info = #tosa.unary_quant<input_zp = -128, output_zp = -128>
  } : (tensor<1x7x7x32xi8>) -> tensor<1x1x1x32xi8>
  return %0 : tensor<1x1x1x32xi8>
}

// max_pool2d carries no quantization.
// CHECK-LABEL: func.func @max_pool
func.func @max_pool(%a: tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8> {
  // CHECK: kea.pool %arg0 {kernel = array<i64: 2, 2>, kind = #kea.pool_kind<MAX>, pads = array<i64: 0, 0, 0, 0>, strides = array<i64: 2, 2>} : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  %0 = tosa.max_pool2d %a {
    kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>,
    pad = array<i64: 0, 0, 0, 0>
  } : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  return %0 : tensor<1x4x4x16xi8>
}

// TOSA silently accepts unknown attributes on every op (TOSA_NOTES 13.17), so
// a `quantization_info` written on max_pool2d parses and round-trips while
// being meaningless. We read the real ODS operand list, so it is ignored --
// exactly as every upstream pass ignores it.
// CHECK-LABEL: func.func @max_pool_bogus_quant_attr
func.func @max_pool_bogus_quant_attr(%a: tensor<1x16x16x8xi8>) -> tensor<1x8x8x8xi8> {
  // CHECK-NOT: quant =
  // CHECK: kea.pool {{.*}}kind = #kea.pool_kind<MAX>{{.*}}pads = array<i64: 0, 1, 0, 1>
  %0 = tosa.max_pool2d %a {
    kernel = array<i64: 3, 3>, stride = array<i64: 2, 2>,
    pad = array<i64: 0, 1, 0, 1>,
    quantization_info = #tosa.unary_quant<input_zp = -5, output_zp = -5>,
    bogus_attr = 42 : i64
  } : (tensor<1x16x16x8xi8>) -> tensor<1x8x8x8xi8>
  return %0 : tensor<1x8x8x8xi8>
}

//===--------------------------------------------------------------------===//
// Shape manipulation
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @reshape_flatten
func.func @reshape_flatten(%a: tensor<1x7x7x32xi8>) -> tensor<1x1568xi8> {
  // CHECK: kea.reshape %arg0 {new_shape = array<i64: 1, 1568>} : (tensor<1x7x7x32xi8>) -> tensor<1x1568xi8>
  %0 = tosa.reshape %a {new_shape = array<i64: 1, 1568>}
      : (tensor<1x7x7x32xi8>) -> tensor<1x1568xi8>
  return %0 : tensor<1x1568xi8>
}

// The -1 placeholder is resolved from the (static) result type: Level 1 forbids it.
// CHECK-LABEL: func.func @reshape_infer
func.func @reshape_infer(%a: tensor<1x7x7x32xi8>) -> tensor<1x1568xi8> {
  // CHECK: kea.reshape %arg0 {new_shape = array<i64: 1, 1568>}
  %0 = tosa.reshape %a {new_shape = array<i64: 1, -1>}
      : (tensor<1x7x7x32xi8>) -> tensor<1x1568xi8>
  return %0 : tensor<1x1568xi8>
}

// In 20.1.6 the permutation is an OPERAND. It is folded into an attribute and
// the now-dead constant is cleaned up.
// CHECK-LABEL: func.func @transpose_nhwc_to_nchw
func.func @transpose_nhwc_to_nchw(%a: tensor<1x8x8x16xi8>) -> tensor<1x16x8x8xi8> {
  // CHECK-NOT: arith.constant
  // CHECK: kea.transpose %arg0 {perms = array<i64: 0, 3, 1, 2>} : (tensor<1x8x8x16xi8>) -> tensor<1x16x8x8xi8>
  %perm = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = tosa.transpose %a, %perm
      : (tensor<1x8x8x16xi8>, tensor<4xi32>) -> tensor<1x16x8x8xi8>
  return %0 : tensor<1x16x8x8xi8>
}

// HWIO -> OHWI weight conversion, i.e. what an exporter emits to feed conv2d.
// CHECK-LABEL: func.func @transpose_weight_hwio_to_ohwi
func.func @transpose_weight_hwio_to_ohwi(%w: tensor<3x3x4x16xi8>) -> tensor<16x3x3x4xi8> {
  // CHECK: kea.transpose %arg0 {perms = array<i64: 3, 0, 1, 2>} : (tensor<3x3x4x16xi8>) -> tensor<16x3x3x4xi8>
  %perm = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = tosa.transpose %w, %perm
      : (tensor<3x3x4x16xi8>, tensor<4xi32>) -> tensor<16x3x3x4xi8>
  return %0 : tensor<16x3x3x4xi8>
}

// Channel shuffle: reshape, rank-5 transpose, reshape.
// CHECK-LABEL: func.func @reshape_transpose_combined
func.func @reshape_transpose_combined(%a: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8> {
  // CHECK: kea.reshape {{.*}} {new_shape = array<i64: 1, 8, 8, 2, 8>}
  // CHECK: kea.transpose {{.*}} {perms = array<i64: 0, 1, 2, 4, 3>}
  // CHECK: kea.reshape {{.*}} {new_shape = array<i64: 1, 8, 8, 16>}
  %r = tosa.reshape %a {new_shape = array<i64: 1, 8, 8, 2, 8>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x8x8x2x8xi8>
  %perm = "tosa.const"() {value = dense<[0, 1, 2, 4, 3]> : tensor<5xi32>} : () -> tensor<5xi32>
  %t = tosa.transpose %r, %perm
      : (tensor<1x8x8x2x8xi8>, tensor<5xi32>) -> tensor<1x8x8x8x2xi8>
  %o = tosa.reshape %t {new_shape = array<i64: 1, 8, 8, 16>}
      : (tensor<1x8x8x8x2xi8>) -> tensor<1x8x8x16xi8>
  return %o : tensor<1x8x8x16xi8>
}

//===--------------------------------------------------------------------===//
// The whole quantized-add idiom, still unfused
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @add_requantized
func.func @add_requantized(%a: tensor<1x4x4x8xi8>, %b: tensor<1x4x4x8xi8>)
    -> tensor<1x4x4x8xi8> {
  // CHECK: kea.rescale
  // CHECK: kea.rescale
  // CHECK: kea.add
  // CHECK: kea.rescale
  %la = tosa.rescale %a {
    input_zp = -5 : i32, output_zp = 0 : i32,
    multiplier = array<i32: 1073741824>, shift = array<i8: 10>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32>
  %lb = tosa.rescale %b {
    input_zp = 3 : i32, output_zp = 0 : i32,
    multiplier = array<i32: 1932735283>, shift = array<i8: 11>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32>
  %sum = tosa.add %la, %lb
      : (tensor<1x4x4x8xi32>, tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %out = tosa.rescale %sum {
    input_zp = 0 : i32, output_zp = -7 : i32,
    multiplier = array<i32: 1288490189>, shift = array<i8: 31>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %out : tensor<1x4x4x8xi8>
}
