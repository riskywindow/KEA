// RUN: kea-opt %s -kea-fuse | FileCheck %s
//
// Per-fusion coverage of -kea-fuse. Every rewrite here is BIT-EXACT; the
// refusal cases at the bottom are the ones where it would not be.

//===--------------------------------------------------------------------===//
// 1. contraction + kea.rescale -> epilogue.requant
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fuse_requant
func.func @fuse_requant(%in: tensor<1x8x8x4xi8>, %w: tensor<16x1x1x4xi8>,
                        %b: tensor<16xi32>) -> tensor<1x8x8x16xi8> {
  // CHECK-NOT: kea.rescale
  // CHECK: kea.conv2d {{.*}}epilogue = #kea.epilogue<requant = <multiplier = [1073741824], shift = [30], input_zp = 0, output_zp = -5, axis = -1, rounding = DOUBLE>>{{.*}} -> tensor<1x8x8x16xi8>
  %acc = kea.conv2d %in, %w bias %b {zero_points = #kea.zp<input = -5, weight = 0>,
                                     strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                                     dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x1x1x4xi8>, tensor<16xi32>) -> tensor<1x8x8x16xi32>
  %0 = kea.rescale %acc {quant = #kea.quant<multiplier = [1073741824], shift = [30],
                                            input_zp = 0, output_zp = -5,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// Per-channel requantization fuses the same way; the axis stays 3.
// CHECK-LABEL: func.func @fuse_requant_per_channel
func.func @fuse_requant_per_channel(%in: tensor<1x8x8x4xi8>, %w: tensor<2x1x1x4xi8>)
    -> tensor<1x8x8x2xi8> {
  // CHECK: kea.conv2d {{.*}}multiplier = [1073741824, 1181116006], shift = [36, 37]{{.*}}axis = 3
  %acc = kea.conv2d %in, %w {zero_points = #kea.zp<input = -5, weight = 0>,
                             strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                             dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<2x1x1x4xi8>) -> tensor<1x8x8x2xi32>
  %0 = kea.rescale %acc {quant = #kea.quant<multiplier = [1073741824, 1181116006],
                                            shift = [36, 37],
                                            input_zp = 0, output_zp = -128,
                                            axis = 3, rounding = DOUBLE>}
      : (tensor<1x8x8x2xi32>) -> tensor<1x8x8x2xi8>
  return %0 : tensor<1x8x8x2xi8>
}

// Depthwise, matmul and fully_connected fuse through the same path.
// CHECK-LABEL: func.func @fuse_requant_dwconv
func.func @fuse_requant_dwconv(%in: tensor<1x8x8x24xi8>, %w: tensor<24x3x3x1xi8>)
    -> tensor<1x8x8x24xi8> {
  // CHECK: kea.dwconv2d {{.*}}epilogue = #kea.epilogue<requant =
  %acc = kea.dwconv2d %in, %w {zero_points = #kea.zp<input = -128, weight = 0>,
                               strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                               dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x24xi8>, tensor<24x3x3x1xi8>) -> tensor<1x8x8x24xi32>
  %0 = kea.rescale %acc {quant = #kea.quant<multiplier = [1073741824], shift = [30],
                                            input_zp = 0, output_zp = -128,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x8x8x24xi32>) -> tensor<1x8x8x24xi8>
  return %0 : tensor<1x8x8x24xi8>
}

// CHECK-LABEL: func.func @fuse_requant_fully_connected
func.func @fuse_requant_fully_connected(%a: tensor<1x64xi8>, %w: tensor<10x64xi8>,
                                        %b: tensor<10xi32>) -> tensor<1x10xi8> {
  // CHECK-NOT: kea.rescale
  // CHECK: kea.fully_connected {{.*}}epilogue = #kea.epilogue<requant =
  %acc = kea.fully_connected %a, %w bias %b {zero_points = #kea.zp<input = -128, weight = 0>}
      : (tensor<1x64xi8>, tensor<10x64xi8>, tensor<10xi32>) -> tensor<1x10xi32>
  %0 = kea.rescale %acc {quant = #kea.quant<multiplier = [1503238553], shift = [37],
                                            input_zp = 0, output_zp = -3,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x10xi32>) -> tensor<1x10xi8>
  return %0 : tensor<1x10xi8>
}

// A rescale whose producer has more than one use must stay: fusing it would
// change the other consumer's operand.
// CHECK-LABEL: func.func @requant_multi_use_producer
func.func @requant_multi_use_producer(%in: tensor<1x8x8x4xi8>, %w: tensor<16x1x1x4xi8>)
    -> (tensor<1x8x8x16xi8>, tensor<1x8x8x16xi32>) {
  // CHECK: kea.conv2d {{.*}} -> tensor<1x8x8x16xi32>
  // CHECK: kea.rescale
  %acc = kea.conv2d %in, %w {zero_points = #kea.zp<input = -5, weight = 0>,
                             strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                             dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x1x1x4xi8>) -> tensor<1x8x8x16xi32>
  %0 = kea.rescale %acc {quant = #kea.quant<multiplier = [1073741824], shift = [30],
                                            input_zp = 0, output_zp = -5,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi8>
  return %0, %acc : tensor<1x8x8x16xi8>, tensor<1x8x8x16xi32>
}

//===--------------------------------------------------------------------===//
// 2. contraction + kea.clamp -> epilogue.clamp
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fuse_requant_and_clamp
func.func @fuse_requant_and_clamp(%in: tensor<1x8x8x4xi8>, %w: tensor<16x1x1x4xi8>)
    -> tensor<1x8x8x16xi8> {
  // CHECK-NOT: kea.rescale
  // CHECK-NOT: kea.clamp
  // CHECK: kea.conv2d {{.*}}epilogue = #kea.epilogue<requant = <{{.*}}>, clamp = [-128, 127]>
  %acc = kea.conv2d %in, %w {zero_points = #kea.zp<input = -5, weight = 0>,
                             strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                             dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x1x1x4xi8>) -> tensor<1x8x8x16xi32>
  %q = kea.rescale %acc {quant = #kea.quant<multiplier = [1073741824], shift = [30],
                                            input_zp = 0, output_zp = -128,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi8>
  %0 = kea.clamp %q {min = -128 : i64, max = 127 : i64} : tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// A clamp on a RAW accumulator has no epilogue slot (the clamp field is stage
// B, after requant), so it stays a separate op.
// CHECK-LABEL: func.func @clamp_on_raw_accumulator
func.func @clamp_on_raw_accumulator(%in: tensor<1x8x8x4xi8>, %w: tensor<16x1x1x4xi8>)
    -> tensor<1x8x8x16xi32> {
  // CHECK: kea.conv2d
  // CHECK: kea.clamp
  %acc = kea.conv2d %in, %w {zero_points = #kea.zp<input = -5, weight = 0>,
                             strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                             dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x1x1x4xi8>) -> tensor<1x8x8x16xi32>
  %0 = kea.clamp %acc {min = 0 : i64, max = 2147483647 : i64} : tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

//===--------------------------------------------------------------------===//
// 3. contraction + unquantized constant kea.add -> the bias operand
//===--------------------------------------------------------------------===//

// The tosa.matmul + broadcast-bias-add idiom becomes a matmul with a bias.
// CHECK-LABEL: func.func @fuse_bias
func.func @fuse_bias(%a: tensor<1x4x8xi8>, %b: tensor<1x8x16xi8>) -> tensor<1x4x16xi32> {
  // CHECK-NOT: kea.add
  // CHECK: %[[B:.*]] = arith.constant dense<11> : tensor<16xi32>
  // CHECK: kea.matmul %arg0, %arg1 bias %[[B]]
  %bias = arith.constant dense<11> : tensor<1x1x16xi32>
  %mm = kea.matmul %a, %b {zero_points = #kea.zp<input = -2, weight = 0>}
      : (tensor<1x4x8xi8>, tensor<1x8x16xi8>) -> tensor<1x4x16xi32>
  %0 = kea.add %mm, %bias
      : (tensor<1x4x16xi32>, tensor<1x1x16xi32>) -> tensor<1x4x16xi32>
  return %0 : tensor<1x4x16xi32>
}

// Bias then requantize then clamp, all folded into one op.
// CHECK-LABEL: func.func @fuse_bias_requant_clamp
func.func @fuse_bias_requant_clamp(%a: tensor<1x4x8xi8>, %b: tensor<1x8x16xi8>)
    -> tensor<1x4x16xi8> {
  // CHECK-NOT: kea.add
  // CHECK-NOT: kea.rescale
  // CHECK-NOT: kea.clamp
  // CHECK: kea.matmul %arg0, %arg1 bias %{{.*}} {epilogue = #kea.epilogue<requant = <{{.*}}>, clamp = [-128, 127]>
  %bias = arith.constant dense<11> : tensor<1x1x16xi32>
  %mm = kea.matmul %a, %b {zero_points = #kea.zp<input = -2, weight = 0>}
      : (tensor<1x4x8xi8>, tensor<1x8x16xi8>) -> tensor<1x4x16xi32>
  %biased = kea.add %mm, %bias
      : (tensor<1x4x16xi32>, tensor<1x1x16xi32>) -> tensor<1x4x16xi32>
  %q = kea.rescale %biased {quant = #kea.quant<multiplier = [1073741824], shift = [30],
                                               input_zp = 0, output_zp = -128,
                                               axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x16xi32>) -> tensor<1x4x16xi8>
  %0 = kea.clamp %q {min = -128 : i64, max = 127 : i64} : tensor<1x4x16xi8>
  return %0 : tensor<1x4x16xi8>
}

// A non-constant add is not a bias.
// CHECK-LABEL: func.func @add_of_value_is_not_bias
func.func @add_of_value_is_not_bias(%a: tensor<1x4x8xi8>, %b: tensor<1x8x16xi8>,
                                    %other: tensor<1x4x16xi32>) -> tensor<1x4x16xi32> {
  // CHECK: kea.matmul
  // CHECK: kea.add
  %mm = kea.matmul %a, %b {zero_points = #kea.zp<input = -2, weight = 0>}
      : (tensor<1x4x8xi8>, tensor<1x8x16xi8>) -> tensor<1x4x16xi32>
  %0 = kea.add %mm, %other
      : (tensor<1x4x16xi32>, tensor<1x4x16xi32>) -> tensor<1x4x16xi32>
  return %0 : tensor<1x4x16xi32>
}

//===--------------------------------------------------------------------===//
// 4a. rescale / add / rescale sandwich -> one quantized kea.add
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @form_quantized_add
func.func @form_quantized_add(%a: tensor<1x4x4x8xi8>, %b: tensor<1x4x4x8xi8>)
    -> tensor<1x4x4x8xi8> {
  // CHECK-NOT: kea.rescale
  // CHECK: kea.add %arg0, %arg1 {lhs_quant = #kea.quant<multiplier = [1073741824], shift = [10], input_zp = -5, output_zp = 0, axis = -1, rounding = DOUBLE>, out_quant = #kea.quant<multiplier = [1288490189], shift = [31], input_zp = 0, output_zp = -7, axis = -1, rounding = DOUBLE>, rhs_quant = #kea.quant<multiplier = [1932735283], shift = [11], input_zp = 3, output_zp = 0, axis = -1, rounding = DOUBLE>} : (tensor<1x4x4x8xi8>, tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
  %la = kea.rescale %a {quant = #kea.quant<multiplier = [1073741824], shift = [10],
                                           input_zp = -5, output_zp = 0,
                                           axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32>
  %lb = kea.rescale %b {quant = #kea.quant<multiplier = [1932735283], shift = [11],
                                           input_zp = 3, output_zp = 0,
                                           axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32>
  %sum = kea.add %la, %lb
      : (tensor<1x4x4x8xi32>, tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %0 = kea.rescale %sum {quant = #kea.quant<multiplier = [1288490189], shift = [31],
                                            input_zp = 0, output_zp = -7,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

//===--------------------------------------------------------------------===//
// 4b. quantized kea.add folded into the producing contraction's epilogue
//===--------------------------------------------------------------------===//

// The MobileNetV2 inverted residual, hand-written at Level 1.
// CHECK-LABEL: func.func @fuse_residual
func.func @fuse_residual(%x: tensor<1x8x8x4xi8>, %in: tensor<1x8x8x24xi8>,
                         %w: tensor<4x1x1x24xi8>) -> tensor<1x8x8x4xi8> {
  // CHECK-NOT: kea.add
  // CHECK-NOT: kea.rescale
  // CHECK: kea.conv2d %arg1, %arg2 residual %arg0 {{{.*}}epilogue = #kea.epilogue<requant = <multiplier = [1288490189], shift = [37], input_zp = 0, output_zp = -5, axis = -1, rounding = DOUBLE>, accum = <multiplier = [1610612736], shift = [11], input_zp = -5, output_zp = 0, axis = -1, rounding = DOUBLE>, residual = <multiplier = [1073741824], shift = [10], input_zp = -5, output_zp = 0, axis = -1, rounding = DOUBLE>, output = <multiplier = [1503238553], shift = [40], input_zp = 0, output_zp = -5, axis = -1, rounding = DOUBLE>>
  %acc = kea.conv2d %in, %w {zero_points = #kea.zp<input = -128, weight = 0>,
                             strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                             dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x24xi8>, tensor<4x1x1x24xi8>) -> tensor<1x8x8x4xi32>
  %proj = kea.rescale %acc {quant = #kea.quant<multiplier = [1288490189], shift = [37],
                                               input_zp = 0, output_zp = -5,
                                               axis = -1, rounding = DOUBLE>}
      : (tensor<1x8x8x4xi32>) -> tensor<1x8x8x4xi8>
  %id32 = kea.rescale %x {quant = #kea.quant<multiplier = [1073741824], shift = [10],
                                             input_zp = -5, output_zp = 0,
                                             axis = -1, rounding = DOUBLE>}
      : (tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi32>
  %res32 = kea.rescale %proj {quant = #kea.quant<multiplier = [1610612736], shift = [11],
                                                 input_zp = -5, output_zp = 0,
                                                 axis = -1, rounding = DOUBLE>}
      : (tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi32>
  %sum = kea.add %id32, %res32
      : (tensor<1x8x8x4xi32>, tensor<1x8x8x4xi32>) -> tensor<1x8x8x4xi32>
  %0 = kea.rescale %sum {quant = #kea.quant<multiplier = [1503238553], shift = [40],
                                            input_zp = 0, output_zp = -5,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x8x8x4xi32>) -> tensor<1x8x8x4xi8>
  return %0 : tensor<1x8x8x4xi8>
}

// The residual operand must already be available where the contraction is. Here
// it is produced AFTER the conv by an op that is not itself a contraction, so
// neither side of the add can absorb it -- folding would require reordering,
// which we refuse. (When BOTH sides are contractions the later one absorbs the
// earlier one as its residual, which is the @fuse_residual case above.)
// CHECK-LABEL: func.func @residual_not_available
func.func @residual_not_available(%in: tensor<1x8x8x24xi8>, %w: tensor<4x1x1x24xi8>,
                                  %late: tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi8> {
  // CHECK: kea.conv2d
  // CHECK: kea.clamp
  // CHECK: kea.add {{.*}}lhs_quant
  %acc = kea.conv2d %in, %w {zero_points = #kea.zp<input = -128, weight = 0>,
                             strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                             dilations = array<i64: 1, 1>,
                             epilogue = #kea.epilogue<requant = #kea.quant<
                                 multiplier = [1288490189], shift = [37],
                                 input_zp = 0, output_zp = -5, axis = -1,
                                 rounding = DOUBLE>>}
      : (tensor<1x8x8x24xi8>, tensor<4x1x1x24xi8>) -> tensor<1x8x8x4xi8>
  // Defined after %acc, so it does not dominate it, and its producer is not a
  // contraction so it cannot absorb %acc either.
  %other = kea.clamp %late {min = -5 : i64, max = 127 : i64} : tensor<1x8x8x4xi8>
  %0 = kea.add %acc, %other {
      lhs_quant = #kea.quant<multiplier = [1610612736], shift = [11],
                             input_zp = -5, output_zp = 0, axis = -1, rounding = DOUBLE>,
      rhs_quant = #kea.quant<multiplier = [1073741824], shift = [10],
                             input_zp = -5, output_zp = 0, axis = -1, rounding = DOUBLE>,
      out_quant = #kea.quant<multiplier = [1503238553], shift = [40],
                             input_zp = 0, output_zp = -5, axis = -1, rounding = DOUBLE>}
      : (tensor<1x8x8x4xi8>, tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi8>
  return %0 : tensor<1x8x8x4xi8>
}

//===--------------------------------------------------------------------===//
// 5. rescale algebra -- and where it REFUSES
//===--------------------------------------------------------------------===//

// A bit-exact no-op rescale (identity scale, same zero point, same type) is
// removed outright.
// CHECK-LABEL: func.func @drop_noop_rescale
func.func @drop_noop_rescale(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32> {
  // CHECK-NOT: kea.rescale
  // CHECK: return %arg0
  %0 = kea.rescale %a {quant = #kea.quant<multiplier = [1073741824], shift = [30],
                                          input_zp = -5, output_zp = -5,
                                          axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// EXACT COMPOSITION, case (a): the FIRST rescale only rebases the zero point
// and its result is i32, so it cannot clamp. The pair collapses into the second
// rescale with input_zp' = a.izp - a.ozp + b.izp = 7 - 0 + 3 = 10.
// CHECK-LABEL: func.func @compose_identity_then_scale
func.func @compose_identity_then_scale(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  // CHECK: %[[R:.*]] = kea.rescale %arg0 {quant = #kea.quant<multiplier = [1288490189], shift = [31], input_zp = 10, output_zp = -7, axis = -1, rounding = DOUBLE>} : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  // CHECK-NEXT: return %[[R]]
  %mid = kea.rescale %a {quant = #kea.quant<multiplier = [1073741824], shift = [30],
                                            input_zp = 7, output_zp = 0,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %0 = kea.rescale %mid {quant = #kea.quant<multiplier = [1288490189], shift = [31],
                                            input_zp = 3, output_zp = -7,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// EXACT COMPOSITION, case (b): the SECOND rescale only rebases the zero point.
// The pair collapses into the first with output_zp' = a.ozp - b.izp + b.ozp
// = 0 - 4 + (-7) = -11, and the second's result type (whose clamp is then the
// only clamp, exactly as in the two-op form).
// CHECK-LABEL: func.func @compose_scale_then_identity
func.func @compose_scale_then_identity(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  // CHECK: %[[R:.*]] = kea.rescale %arg0 {quant = #kea.quant<multiplier = [1288490189], shift = [31], input_zp = 3, output_zp = -11, axis = -1, rounding = DOUBLE>} : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  // CHECK-NEXT: return %[[R]]
  %mid = kea.rescale %a {quant = #kea.quant<multiplier = [1288490189], shift = [31],
                                            input_zp = 3, output_zp = 0,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %0 = kea.rescale %mid {quant = #kea.quant<multiplier = [1073741824], shift = [30],
                                            input_zp = 4, output_zp = -7,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

//===--------------------------------------------------------------------===//
// THE REFUSAL CASES. These are the whole point of doing this carefully.
//===--------------------------------------------------------------------===//

// REFUSED: the intermediate is i8. The first rescale SATURATES to [-128, 127]
// and rounds to an integer there; both effects are observable and neither can
// be recovered by a single multiply-shift. This is the ubiquitous
// i32 -> i8 -> i32 pattern at the input of a quantized add, and collapsing it
// would silently change the network's numerics.
// CHECK-LABEL: func.func @refuse_compose_through_i8
func.func @refuse_compose_through_i8(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32> {
  // CHECK: %[[A:.*]] = kea.rescale %arg0 {{.*}} -> tensor<1x4x4x8xi8>
  // CHECK: kea.rescale %[[A]] {{.*}} -> tensor<1x4x4x8xi32>
  %mid = kea.rescale %a {quant = #kea.quant<multiplier = [1288490189], shift = [37],
                                            input_zp = 0, output_zp = -5,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  %0 = kea.rescale %mid {quant = #kea.quant<multiplier = [1610612736], shift = [11],
                                            input_zp = -5, output_zp = 0,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// REFUSED: both rescales carry a real multiplier. Even with an i32 intermediate
// (so no saturation), each applies its own half-up rounding, and
// round(round(x*m1)*m2) != round(x*m1*m2) in general.
// CHECK-LABEL: func.func @refuse_compose_two_real_scales
func.func @refuse_compose_two_real_scales(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  // CHECK: kea.rescale
  // CHECK: kea.rescale
  %mid = kea.rescale %a {quant = #kea.quant<multiplier = [1288490189], shift = [31],
                                            input_zp = 0, output_zp = 0,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %0 = kea.rescale %mid {quant = #kea.quant<multiplier = [1610612736], shift = [30],
                                            input_zp = 0, output_zp = -5,
                                            axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

//===--------------------------------------------------------------------===//
// 6. layout no-ops
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_identity_transpose
func.func @fold_identity_transpose(%a: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8> {
  // CHECK-NOT: kea.transpose
  // CHECK: return %arg0
  %0 = kea.transpose %a {perms = array<i64: 0, 1, 2, 3>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// Only unit dimensions move, so no element changes its row-major position: the
// transpose is a pure metadata change and becomes a reshape.
// CHECK-LABEL: func.func @fold_unit_dim_transpose
func.func @fold_unit_dim_transpose(%a: tensor<1x8x8x1xi8>) -> tensor<1x1x8x8xi8> {
  // CHECK-NOT: kea.transpose
  // CHECK: kea.reshape %arg0 {new_shape = array<i64: 1, 1, 8, 8>}
  %0 = kea.transpose %a {perms = array<i64: 0, 3, 1, 2>}
      : (tensor<1x8x8x1xi8>) -> tensor<1x1x8x8xi8>
  return %0 : tensor<1x1x8x8xi8>
}

// A real transpose (two non-unit dims swap) is left alone.
// CHECK-LABEL: func.func @keep_real_transpose
func.func @keep_real_transpose(%a: tensor<1x8x8x16xi8>) -> tensor<1x16x8x8xi8> {
  // CHECK: kea.transpose %arg0 {perms = array<i64: 0, 3, 1, 2>}
  %0 = kea.transpose %a {perms = array<i64: 0, 3, 1, 2>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x16x8x8xi8>
  return %0 : tensor<1x16x8x8xi8>
}

// CHECK-LABEL: func.func @fold_identity_reshape
func.func @fold_identity_reshape(%a: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8> {
  // CHECK-NOT: kea.reshape
  // CHECK: return %arg0
  %0 = kea.reshape %a {new_shape = array<i64: 1, 8, 8, 16>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// reshape(reshape(x)) is one reshape.
// CHECK-LABEL: func.func @fold_reshape_chain
func.func @fold_reshape_chain(%a: tensor<1x7x7x32xi8>) -> tensor<1568xi8> {
  // CHECK: %[[R:.*]] = kea.reshape %arg0 {new_shape = array<i64: 1568>} : (tensor<1x7x7x32xi8>) -> tensor<1568xi8>
  // CHECK-NEXT: return %[[R]]
  %mid = kea.reshape %a {new_shape = array<i64: 1, 1568>}
      : (tensor<1x7x7x32xi8>) -> tensor<1x1568xi8>
  %0 = kea.reshape %mid {new_shape = array<i64: 1568>}
      : (tensor<1x1568xi8>) -> tensor<1568xi8>
  return %0 : tensor<1568xi8>
}
