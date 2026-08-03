// RUN: kea-opt %s | kea-opt | FileCheck %s
//
// Per-op round trip for every Level 1 (tensor level) op: parse, print,
// re-parse, re-print. This is what catches a wrong assemblyFormat.
//
// Spelling notes:
//  * `#kea.quant`'s multiplier/shift are DenseI{32,8}ArrayAttr *parameters*, so
//    they print in the "stripped" form `[1073741824]`, not `array<i32: ...>`.
//    Both spellings parse; the stripped one is what round-trips.
//  * a nested `#kea.quant` inside `#kea.epilogue` likewise prints without its
//    `#kea.quant` prefix.

//===--------------------------------------------------------------------===//
// kea.conv2d
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @conv2d_raw_accumulator
func.func @conv2d_raw_accumulator(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>)
    -> tensor<1x8x8x16xi32> {
  // CHECK: kea.conv2d %{{.*}}, %{{.*}} {dilations = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>, strides = array<i64: 1, 1>, zero_points = #kea.zp<input = -3, weight = 0>}
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = -3, weight = 0>,
                           strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                           dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// Dilation 2 with pad 2 keeps the spatial size (effective kernel 5x5).
// CHECK-LABEL: func.func @conv2d_dilated_biased
func.func @conv2d_dilated_biased(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>,
                                 %b: tensor<16xi32>) -> tensor<1x8x8x16xi32> {
  // CHECK: kea.conv2d %{{.*}}, %{{.*}} bias %{{.*}} {dilations = array<i64: 2, 2>
  %0 = kea.conv2d %in, %w bias %b {zero_points = #kea.zp<input = 7, weight = 0>,
                                    strides = array<i64: 1, 1>, pads = array<i64: 2, 2, 2, 2>,
                                    dilations = array<i64: 2, 2>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi32>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// The fully fused form: bias + per-channel requantize + ReLU6 clamp.
// CHECK-LABEL: func.func @conv2d_fused_epilogue
func.func @conv2d_fused_epilogue(%in: tensor<1x8x8x4xi8>, %w: tensor<2x1x1x4xi8>,
                                 %b: tensor<2xi32>) -> tensor<1x8x8x2xi8> {
  // CHECK: kea.conv2d %{{.*}}, %{{.*}} bias %{{.*}} {dilations = array<i64: 1, 1>, epilogue = #kea.epilogue<requant = <multiplier = [1073741824, 1181116006], shift = [36, 37], input_zp = 0, output_zp = -128, axis = 3, rounding = DOUBLE>, clamp = [-128, 127]>
  %0 = kea.conv2d %in, %w bias %b {
      zero_points = #kea.zp<input = -5, weight = 0>,
      strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
      dilations = array<i64: 1, 1>,
      epilogue = #kea.epilogue<
        requant = #kea.quant<multiplier = [1073741824, 1181116006], shift = [36, 37],
                             input_zp = 0, output_zp = -128, axis = 3, rounding = DOUBLE>,
        clamp = [-128, 127]>}
      : (tensor<1x8x8x4xi8>, tensor<2x1x1x4xi8>, tensor<2xi32>) -> tensor<1x8x8x2xi8>
  return %0 : tensor<1x8x8x2xi8>
}

// The MobileNetV2 inverted residual: a residual operand plus the epilogue's
// accum/residual/output triple.
// CHECK-LABEL: func.func @conv2d_residual
func.func @conv2d_residual(%in: tensor<1x8x8x24xi8>, %w: tensor<4x1x1x24xi8>,
                           %b: tensor<4xi32>, %skip: tensor<1x8x8x4xi8>)
    -> tensor<1x8x8x4xi8> {
  // CHECK: kea.conv2d %{{.*}}, %{{.*}} bias %{{.*}} residual %{{.*}} {dilations
  // CHECK-SAME: accum = <multiplier = [1610612736], shift = [11], input_zp = -5, output_zp = 0, axis = -1, rounding = DOUBLE>
  // CHECK-SAME: residual = <multiplier = [1073741824], shift = [10], input_zp = -5, output_zp = 0, axis = -1, rounding = DOUBLE>
  // CHECK-SAME: output = <multiplier = [1503238553], shift = [40], input_zp = 0, output_zp = -5, axis = -1, rounding = DOUBLE>
  %0 = kea.conv2d %in, %w bias %b residual %skip {
      zero_points = #kea.zp<input = -128, weight = 0>,
      strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
      dilations = array<i64: 1, 1>,
      epilogue = #kea.epilogue<
        requant = #kea.quant<multiplier = [1288490189], shift = [37],
                             input_zp = 0, output_zp = -5, axis = -1, rounding = DOUBLE>,
        accum = #kea.quant<multiplier = [1610612736], shift = [11],
                           input_zp = -5, output_zp = 0, axis = -1, rounding = DOUBLE>,
        residual = #kea.quant<multiplier = [1073741824], shift = [10],
                              input_zp = -5, output_zp = 0, axis = -1, rounding = DOUBLE>,
        output = #kea.quant<multiplier = [1503238553], shift = [40],
                            input_zp = 0, output_zp = -5, axis = -1, rounding = DOUBLE>>}
      : (tensor<1x8x8x24xi8>, tensor<4x1x1x24xi8>, tensor<4xi32>, tensor<1x8x8x4xi8>)
        -> tensor<1x8x8x4xi8>
  return %0 : tensor<1x8x8x4xi8>
}

//===--------------------------------------------------------------------===//
// kea.dwconv2d -- canonical [OC, KH, KW, 1] weights
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @dwconv2d_m1
func.func @dwconv2d_m1(%in: tensor<1x8x8x24xi8>, %w: tensor<24x3x3x1xi8>,
                       %b: tensor<24xi32>) -> tensor<1x8x8x24xi32> {
  // CHECK: kea.dwconv2d %{{.*}}, %{{.*}} bias %{{.*}} {dilations = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>, strides = array<i64: 1, 1>
  %0 = kea.dwconv2d %in, %w bias %b {zero_points = #kea.zp<input = -128, weight = 0>,
                                      strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                                      dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x24xi8>, tensor<24x3x3x1xi8>, tensor<24xi32>) -> tensor<1x8x8x24xi32>
  return %0 : tensor<1x8x8x24xi32>
}

// Channel multiplier 2: 8 input channels, 16 canonical weights, 16 outputs.
// CHECK-LABEL: func.func @dwconv2d_m2
func.func @dwconv2d_m2(%in: tensor<1x8x8x8xi8>, %w: tensor<16x3x3x1xi8>)
    -> tensor<1x8x8x16xi32> {
  // CHECK: kea.dwconv2d
  %0 = kea.dwconv2d %in, %w {zero_points = #kea.zp<input = -128, weight = 0>,
                             strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                             dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x8xi8>, tensor<16x3x3x1xi8>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// Stride 2 with the asymmetric MobileNetV2 pad.
// CHECK-LABEL: func.func @dwconv2d_stride2
func.func @dwconv2d_stride2(%in: tensor<1x8x8x24xi8>, %w: tensor<24x3x3x1xi8>)
    -> tensor<1x4x4x24xi32> {
  // CHECK: kea.dwconv2d {{.*}}pads = array<i64: 0, 1, 0, 1>, strides = array<i64: 2, 2>
  %0 = kea.dwconv2d %in, %w {zero_points = #kea.zp<input = -128, weight = 0>,
                             strides = array<i64: 2, 2>, pads = array<i64: 0, 1, 0, 1>,
                             dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x24xi8>, tensor<24x3x3x1xi8>) -> tensor<1x4x4x24xi32>
  return %0 : tensor<1x4x4x24xi32>
}

//===--------------------------------------------------------------------===//
// kea.matmul / kea.fully_connected
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @matmul
func.func @matmul(%a: tensor<1x4x8xi8>, %b: tensor<1x8x16xi8>) -> tensor<1x4x16xi32> {
  // CHECK: kea.matmul %{{.*}}, %{{.*}} {zero_points = #kea.zp<input = -2, weight = 1>} : (tensor<1x4x8xi8>, tensor<1x8x16xi8>) -> tensor<1x4x16xi32>
  %0 = kea.matmul %a, %b {zero_points = #kea.zp<input = -2, weight = 1>}
      : (tensor<1x4x8xi8>, tensor<1x8x16xi8>) -> tensor<1x4x16xi32>
  return %0 : tensor<1x4x16xi32>
}

// CHECK-LABEL: func.func @matmul_biased_batched
func.func @matmul_biased_batched(%a: tensor<4x32x64xi8>, %b: tensor<4x64x32xi8>,
                                 %bias: tensor<32xi32>) -> tensor<4x32x32xi32> {
  // CHECK: kea.matmul %{{.*}}, %{{.*}} bias %{{.*}}
  %0 = kea.matmul %a, %b bias %bias {zero_points = #kea.zp<input = -5, weight = 0>}
      : (tensor<4x32x64xi8>, tensor<4x64x32xi8>, tensor<32xi32>) -> tensor<4x32x32xi32>
  return %0 : tensor<4x32x32xi32>
}

// CHECK-LABEL: func.func @fully_connected
func.func @fully_connected(%a: tensor<1x64xi8>, %w: tensor<10x64xi8>, %b: tensor<10xi32>)
    -> tensor<1x10xi32> {
  // CHECK: kea.fully_connected %{{.*}}, %{{.*}} bias %{{.*}} {zero_points = #kea.zp<input = -128, weight = 0>} : (tensor<1x64xi8>, tensor<10x64xi8>, tensor<10xi32>) -> tensor<1x10xi32>
  %0 = kea.fully_connected %a, %w bias %b {zero_points = #kea.zp<input = -128, weight = 0>}
      : (tensor<1x64xi8>, tensor<10x64xi8>, tensor<10xi32>) -> tensor<1x10xi32>
  return %0 : tensor<1x10xi32>
}

// CHECK-LABEL: func.func @fully_connected_requantized
func.func @fully_connected_requantized(%a: tensor<1x64xi8>, %w: tensor<10x64xi8>,
                                       %b: tensor<10xi32>) -> tensor<1x10xi8> {
  // CHECK: kea.fully_connected {{.*}}epilogue = #kea.epilogue<requant = <multiplier = [1503238553], shift = [37], input_zp = 0, output_zp = -3, axis = -1, rounding = DOUBLE>>
  %0 = kea.fully_connected %a, %w bias %b {
      zero_points = #kea.zp<input = -128, weight = 0>,
      epilogue = #kea.epilogue<requant = #kea.quant<multiplier = [1503238553], shift = [37],
                                                    input_zp = 0, output_zp = -3,
                                                    axis = -1, rounding = DOUBLE>>}
      : (tensor<1x64xi8>, tensor<10x64xi8>, tensor<10xi32>) -> tensor<1x10xi8>
  return %0 : tensor<1x10xi8>
}

//===--------------------------------------------------------------------===//
// kea.add
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @add_plain
func.func @add_plain(%a: tensor<1x4x4x8xi32>, %b: tensor<1x4x4x8xi32>)
    -> tensor<1x4x4x8xi32> {
  // CHECK: kea.add %{{.*}}, %{{.*}} : (tensor<1x4x4x8xi32>, tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %0 = kea.add %a, %b : (tensor<1x4x4x8xi32>, tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// TOSA-style broadcasting of size-1 dimensions at equal rank.
// CHECK-LABEL: func.func @add_broadcast
func.func @add_broadcast(%a: tensor<1x4x4x8xi32>, %b: tensor<1x1x1x8xi32>)
    -> tensor<1x4x4x8xi32> {
  // CHECK: kea.add %{{.*}}, %{{.*}} : (tensor<1x4x4x8xi32>, tensor<1x1x1x8xi32>) -> tensor<1x4x4x8xi32>
  %0 = kea.add %a, %b : (tensor<1x4x4x8xi32>, tensor<1x1x1x8xi32>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// The KEA_VADD form: three per-tensor requantizations plus a fused clamp.
// CHECK-LABEL: func.func @add_quantized
func.func @add_quantized(%a: tensor<1x4x4x8xi8>, %b: tensor<1x4x4x8xi8>)
    -> tensor<1x4x4x8xi8> {
  // CHECK: kea.add %{{.*}}, %{{.*}} {clamp = array<i64: -128, 127>, lhs_quant = #kea.quant<multiplier = [1073741824], shift = [10], input_zp = -5, output_zp = 0, axis = -1, rounding = DOUBLE>
  %0 = kea.add %a, %b {
      lhs_quant = #kea.quant<multiplier = [1073741824], shift = [10],
                             input_zp = -5, output_zp = 0, axis = -1, rounding = DOUBLE>,
      rhs_quant = #kea.quant<multiplier = [1932735283], shift = [11],
                             input_zp = 3, output_zp = 0, axis = -1, rounding = DOUBLE>,
      out_quant = #kea.quant<multiplier = [1288490189], shift = [31],
                             input_zp = 0, output_zp = -7, axis = -1, rounding = DOUBLE>,
      clamp = array<i64: -128, 127>}
      : (tensor<1x4x4x8xi8>, tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

//===--------------------------------------------------------------------===//
// kea.clamp / kea.rescale
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @clamp
func.func @clamp(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {
  // CHECK: kea.clamp %{{.*}} {max = 91 : i64, min = -5 : i64} : tensor<1x4x4x8xi8>
  %0 = kea.clamp %a {min = -5 : i64, max = 91 : i64} : tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// CHECK-LABEL: func.func @rescale_per_tensor
func.func @rescale_per_tensor(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  // CHECK: kea.rescale %{{.*}} {quant = #kea.quant<multiplier = [1073741824], shift = [30], input_zp = 0, output_zp = -5, axis = -1, rounding = DOUBLE>} : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  %0 = kea.rescale %a {quant = #kea.quant<multiplier = [1073741824], shift = [30],
                                          input_zp = 0, output_zp = -5,
                                          axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// CHECK-LABEL: func.func @rescale_per_channel_single_round
func.func @rescale_per_channel_single_round(%a: tensor<1x4x4x4xi32>) -> tensor<1x4x4x4xi8> {
  // CHECK: kea.rescale {{.*}}axis = 3, rounding = SINGLE
  %0 = kea.rescale %a {quant = #kea.quant<multiplier = [1073741824, 1181116006, 1288490189, 1395864371],
                                          shift = [30, 31, 32, 33],
                                          input_zp = 0, output_zp = -5,
                                          axis = 3, rounding = SINGLE>}
      : (tensor<1x4x4x4xi32>) -> tensor<1x4x4x4xi8>
  return %0 : tensor<1x4x4x4xi8>
}

//===--------------------------------------------------------------------===//
// kea.pool
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @pool_max
func.func @pool_max(%a: tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8> {
  // CHECK: kea.pool %{{.*}} {kernel = array<i64: 2, 2>, kind = #kea.pool_kind<MAX>, pads = array<i64: 0, 0, 0, 0>, strides = array<i64: 2, 2>} : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  %0 = kea.pool %a {kind = #kea.pool_kind<MAX>, kernel = array<i64: 2, 2>,
                    strides = array<i64: 2, 2>, pads = array<i64: 0, 0, 0, 0>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  return %0 : tensor<1x4x4x16xi8>
}

// Average pooling carries at most a zero-point rebase (identity multiplier).
// CHECK-LABEL: func.func @pool_avg_rebase
func.func @pool_avg_rebase(%a: tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8> {
  // CHECK: kea.pool {{.*}}kind = #kea.pool_kind<AVG>{{.*}}quant = #kea.quant<multiplier = [1073741824], shift = [30], input_zp = -5, output_zp = -5, axis = -1, rounding = SINGLE>
  %0 = kea.pool %a {kind = #kea.pool_kind<AVG>, kernel = array<i64: 2, 2>,
                    strides = array<i64: 2, 2>, pads = array<i64: 0, 0, 0, 0>,
                    quant = #kea.quant<multiplier = [1073741824], shift = [30],
                                       input_zp = -5, output_zp = -5,
                                       axis = -1, rounding = SINGLE>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  return %0 : tensor<1x4x4x16xi8>
}

// Global average pool: kernel == full spatial extent.
// CHECK-LABEL: func.func @pool_global_avg
func.func @pool_global_avg(%a: tensor<1x7x7x32xi8>) -> tensor<1x1x1x32xi8> {
  // CHECK: kea.pool {{.*}}kernel = array<i64: 7, 7>
  %0 = kea.pool %a {kind = #kea.pool_kind<AVG>, kernel = array<i64: 7, 7>,
                    strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>}
      : (tensor<1x7x7x32xi8>) -> tensor<1x1x1x32xi8>
  return %0 : tensor<1x1x1x32xi8>
}

//===--------------------------------------------------------------------===//
// kea.reshape / kea.transpose
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @reshape
func.func @reshape(%a: tensor<1x7x7x32xi8>) -> tensor<1x1568xi8> {
  // CHECK: kea.reshape %{{.*}} {new_shape = array<i64: 1, 1568>} : (tensor<1x7x7x32xi8>) -> tensor<1x1568xi8>
  %0 = kea.reshape %a {new_shape = array<i64: 1, 1568>}
      : (tensor<1x7x7x32xi8>) -> tensor<1x1568xi8>
  return %0 : tensor<1x1568xi8>
}

// CHECK-LABEL: func.func @transpose
func.func @transpose(%a: tensor<1x8x8x16xi8>) -> tensor<1x16x8x8xi8> {
  // CHECK: kea.transpose %{{.*}} {perms = array<i64: 0, 3, 1, 2>} : (tensor<1x8x8x16xi8>) -> tensor<1x16x8x8xi8>
  %0 = kea.transpose %a {perms = array<i64: 0, 3, 1, 2>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x16x8x8xi8>
  return %0 : tensor<1x16x8x8xi8>
}

// The HWCM -> canonical depthwise permutation, spelled explicitly.
// CHECK-LABEL: func.func @transpose_hwcm_to_cmhw
func.func @transpose_hwcm_to_cmhw(%w: tensor<3x3x24x1xi8>) -> tensor<24x1x3x3xi8> {
  // CHECK: kea.transpose %{{.*}} {perms = array<i64: 2, 3, 0, 1>}
  %0 = kea.transpose %w {perms = array<i64: 2, 3, 0, 1>}
      : (tensor<3x3x24x1xi8>) -> tensor<24x1x3x3xi8>
  return %0 : tensor<24x1x3x3xi8>
}
