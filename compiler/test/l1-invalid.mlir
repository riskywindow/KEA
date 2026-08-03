// RUN: kea-opt %s -split-input-file -verify-diagnostics
//
// Verifier-rejection coverage for every Level 1 op. `-verify-diagnostics` makes
// kea-opt exit 0 iff every emitted diagnostic was matched, so no `not` tool is
// needed (Homebrew LLVM does not ship one).
//
// REMINDER: `expected-error {{...}}` is a literal SUBSTRING match, not a regex.
// Paste the exact diagnostic text.

//===--------------------------------------------------------------------===//
// kea.conv2d
//===--------------------------------------------------------------------===//

func.func @conv2d_rank(%in: tensor<8x8x4xi8>, %w: tensor<16x3x3x4xi8>)
    -> tensor<1x8x8x16xi32> {
  // expected-error @+1 {{expects a rank-4 NHWC input}}
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
                           strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                           dilations = array<i64: 1, 1>}
      : (tensor<8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// -----

func.func @conv2d_channels(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x8xi8>)
    -> tensor<1x8x8x16xi32> {
  // expected-error @+1 {{input channel count 4 does not match weight channel count 8}}
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
                           strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                           dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x8xi8>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// -----

func.func @conv2d_out_channels(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>)
    -> tensor<1x8x8x32xi32> {
  // expected-error @+1 {{output channel count 32 does not match filter count 16}}
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
                           strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                           dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x32xi32>
  return %0 : tensor<1x8x8x32xi32>
}

// -----

// THE SHAPE CHECK THAT MATTERS: a 3x3 kernel with pad 0 over an 8x8 input is
// 6x6, not 8x8. A verifier that only checked types would let this through.
func.func @conv2d_spatial_pad(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>)
    -> tensor<1x8x8x16xi32> {
  // expected-error @+1 {{output spatial size 8x8 disagrees with pad/stride/dilation, which give 6x6}}
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
                           strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                           dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// -----

// Same, driven by the stride: 8x8 / 2 with pad 1 and a 3x3 kernel is 4x4.
func.func @conv2d_spatial_stride(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>)
    -> tensor<1x8x8x16xi32> {
  // expected-error @+1 {{output spatial size 8x8 disagrees with pad/stride/dilation, which give 4x4}}
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
                           strides = array<i64: 2, 2>, pads = array<i64: 1, 1, 1, 1>,
                           dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// -----

// And by the dilation: dilation 2 makes a 3x3 kernel effectively 5x5, so pad 1
// gives 6x6, not 8x8.
func.func @conv2d_spatial_dilation(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>)
    -> tensor<1x8x8x16xi32> {
  // expected-error @+1 {{output spatial size 8x8 disagrees with pad/stride/dilation, which give 6x6}}
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
                           strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                           dilations = array<i64: 2, 2>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// -----

func.func @conv2d_bias_length(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>,
                              %b: tensor<8xi32>) -> tensor<1x8x8x16xi32> {
  // expected-error @+1 {{bias must have 16 elements, got 8}}
  %0 = kea.conv2d %in, %w bias %b {zero_points = #kea.zp<input = 0, weight = 0>,
                                    strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                                    dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<8xi32>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// -----

func.func @conv2d_bias_element_type(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>,
                                    %b: tensor<16xi8>) -> tensor<1x8x8x16xi32> {
  // expected-error @+1 {{bias element type must be i32, got 'i8'}}
  %0 = kea.conv2d %in, %w bias %b {zero_points = #kea.zp<input = 0, weight = 0>,
                                    strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                                    dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi8>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// -----

// Without a requant stage the result IS the accumulator, so it must be i32.
func.func @conv2d_unrequantized_i8(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>)
    -> tensor<1x8x8x16xi8> {
  // expected-error @+1 {{without an epilogue requant stage the result is the raw accumulator and must be i32}}
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
                           strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                           dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// -----

// THE PER-CHANNEL SCALE LENGTH CHECK: 4 multipliers for 16 output channels.
func.func @conv2d_per_channel_length(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>)
    -> tensor<1x8x8x16xi8> {
  // expected-error @+1 {{epilogue requant has 4 scales but dimension 3 of the tensor is 16}}
  %0 = kea.conv2d %in, %w {
      zero_points = #kea.zp<input = 0, weight = 0>,
      strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
      dilations = array<i64: 1, 1>,
      epilogue = #kea.epilogue<requant = #kea.quant<
          multiplier = [1073741824, 1073741824, 1073741824, 1073741824],
          shift = [30, 30, 30, 30], input_zp = 0, output_zp = -5,
          axis = 3, rounding = DOUBLE>>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// -----

func.func @conv2d_residual_without_quants(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>,
                                          %r: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8> {
  // expected-error @+1 {{a residual operand requires the epilogue's accum/residual/output requantizations}}
  %0 = kea.conv2d %in, %w residual %r {
      zero_points = #kea.zp<input = 0, weight = 0>,
      strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
      dilations = array<i64: 1, 1>,
      epilogue = #kea.epilogue<requant = #kea.quant<multiplier = [1073741824], shift = [30],
                                                    input_zp = 0, output_zp = -5,
                                                    axis = -1, rounding = DOUBLE>>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// -----

func.func @conv2d_clamp_out_of_range(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>)
    -> tensor<1x8x8x16xi8> {
  // expected-error @+1 {{epilogue clamp [-128, 500] does not fit the result element type range [-128, 127]}}
  %0 = kea.conv2d %in, %w {
      zero_points = #kea.zp<input = 0, weight = 0>,
      strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
      dilations = array<i64: 1, 1>,
      epilogue = #kea.epilogue<requant = #kea.quant<multiplier = [1073741824], shift = [30],
                                                    input_zp = 0, output_zp = -5,
                                                    axis = -1, rounding = DOUBLE>,
                               clamp = [-128, 500]>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

//===--------------------------------------------------------------------===//
// kea.dwconv2d -- the canonical layout is enforced here
//===--------------------------------------------------------------------===//

// -----

// TOSA's HWCM weights [3, 3, 24, 1] must be normalised to [24, 3, 3, 1] before
// they reach Level 1. Feeding HWCM straight in is rejected.
func.func @dwconv2d_hwcm_weights(%in: tensor<1x8x8x24xi8>, %w: tensor<3x3x24x1xi8>)
    -> tensor<1x8x8x24xi32> {
  // expected-error @+1 {{output channel count 24 does not match weight count 3}}
  %0 = kea.dwconv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
                             strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                             dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x24xi8>, tensor<3x3x24x1xi8>) -> tensor<1x8x8x24xi32>
  return %0 : tensor<1x8x8x24xi32>
}

// -----

func.func @dwconv2d_trailing_dim(%in: tensor<1x8x8x24xi8>, %w: tensor<24x3x3x2xi8>)
    -> tensor<1x8x8x24xi32> {
  // expected-error @+1 {{canonical depthwise weights are [OC, KH, KW, 1]; the trailing dimension must be 1}}
  %0 = kea.dwconv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
                             strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                             dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x24xi8>, tensor<24x3x3x2xi8>) -> tensor<1x8x8x24xi32>
  return %0 : tensor<1x8x8x24xi32>
}

// -----

// 20 output channels is not a whole multiple of 8 input channels.
func.func @dwconv2d_multiplier(%in: tensor<1x8x8x8xi8>, %w: tensor<20x3x3x1xi8>)
    -> tensor<1x8x8x20xi32> {
  // expected-error @+1 {{output channel count 20 must be a multiple of the input channel count 8 (the channel multiplier)}}
  %0 = kea.dwconv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
                             strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                             dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x8xi8>, tensor<20x3x3x1xi8>) -> tensor<1x8x8x20xi32>
  return %0 : tensor<1x8x8x20xi32>
}

//===--------------------------------------------------------------------===//
// kea.matmul / kea.fully_connected
//===--------------------------------------------------------------------===//

// -----

func.func @matmul_rank(%a: tensor<4x8xi8>, %b: tensor<8x16xi8>) -> tensor<4x16xi32> {
  // expected-error @+1 {{expects strictly rank-3 [B, M, K] x [B, K, N] operands and a rank-3 result}}
  %0 = kea.matmul %a, %b {zero_points = #kea.zp<input = 0, weight = 0>}
      : (tensor<4x8xi8>, tensor<8x16xi8>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

// -----

func.func @matmul_k(%a: tensor<1x4x8xi8>, %b: tensor<1x16x16xi8>) -> tensor<1x4x16xi32> {
  // expected-error @+1 {{contraction dimension mismatch: a K=8 but b K=16}}
  %0 = kea.matmul %a, %b {zero_points = #kea.zp<input = 0, weight = 0>}
      : (tensor<1x4x8xi8>, tensor<1x16x16xi8>) -> tensor<1x4x16xi32>
  return %0 : tensor<1x4x16xi32>
}

// -----

func.func @matmul_batch(%a: tensor<2x4x8xi8>, %b: tensor<1x8x16xi8>) -> tensor<2x4x16xi32> {
  // expected-error @+1 {{batch dimensions disagree: 2, 1, 2}}
  %0 = kea.matmul %a, %b {zero_points = #kea.zp<input = 0, weight = 0>}
      : (tensor<2x4x8xi8>, tensor<1x8x16xi8>) -> tensor<2x4x16xi32>
  return %0 : tensor<2x4x16xi32>
}

// -----

func.func @matmul_bias_length(%a: tensor<1x4x8xi8>, %b: tensor<1x8x16xi8>,
                              %bias: tensor<4xi32>) -> tensor<1x4x16xi32> {
  // expected-error @+1 {{bias must have 16 elements, got 4}}
  %0 = kea.matmul %a, %b bias %bias {zero_points = #kea.zp<input = 0, weight = 0>}
      : (tensor<1x4x8xi8>, tensor<1x8x16xi8>, tensor<4xi32>) -> tensor<1x4x16xi32>
  return %0 : tensor<1x4x16xi32>
}

// -----

// The weight is [OC, IC], already transposed; [IC, OC] is rejected.
func.func @fully_connected_transposed_weight(%a: tensor<4x8xi8>, %w: tensor<8x16xi8>)
    -> tensor<4x16xi32> {
  // expected-error @+1 {{input feature count 8 does not match weight feature count 16 (weights are [OC, IC], already transposed)}}
  %0 = kea.fully_connected %a, %w {zero_points = #kea.zp<input = 0, weight = 0>}
      : (tensor<4x8xi8>, tensor<8x16xi8>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

// -----

func.func @fully_connected_rows(%a: tensor<4x8xi8>, %w: tensor<16x8xi8>)
    -> tensor<2x16xi32> {
  // expected-error @+1 {{result rows 2 do not match input rows 4}}
  %0 = kea.fully_connected %a, %w {zero_points = #kea.zp<input = 0, weight = 0>}
      : (tensor<4x8xi8>, tensor<16x8xi8>) -> tensor<2x16xi32>
  return %0 : tensor<2x16xi32>
}

//===--------------------------------------------------------------------===//
// kea.add
//===--------------------------------------------------------------------===//

// -----

func.func @add_not_broadcastable(%a: tensor<1x4x4x8xi32>, %b: tensor<1x2x4x8xi32>)
    -> tensor<1x4x4x8xi32> {
  // expected-error @+1 {{dimension 1 (4, 2) is not broadcastable to 4}}
  %0 = kea.add %a, %b : (tensor<1x4x4x8xi32>, tensor<1x2x4x8xi32>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// -----

func.func @add_rank(%a: tensor<1x4x4x8xi32>, %b: tensor<8xi32>) -> tensor<1x4x4x8xi32> {
  // expected-error @+1 {{operands and result must have equal rank}}
  %0 = kea.add %a, %b : (tensor<1x4x4x8xi32>, tensor<8xi32>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// -----

func.func @add_partial_quants(%a: tensor<1x4x4x8xi8>, %b: tensor<1x4x4x8xi8>)
    -> tensor<1x4x4x8xi8> {
  // expected-error @+1 {{lhs_quant/rhs_quant/out_quant are all-or-nothing, got 1 of 3}}
  %0 = kea.add %a, %b {
      lhs_quant = #kea.quant<multiplier = [1073741824], shift = [10],
                             input_zp = -5, output_zp = 0, axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi8>, tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

//===--------------------------------------------------------------------===//
// kea.clamp / kea.rescale
//===--------------------------------------------------------------------===//

// -----

func.func @clamp_inverted(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {
  // expected-error @+1 {{min 100 exceeds max 10}}
  %0 = kea.clamp %a {min = 100 : i64, max = 10 : i64} : tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// -----

func.func @clamp_out_of_range(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {
  // expected-error @+1 {{clamp bounds [-128, 300] do not fit the element type range [-128, 127]}}
  %0 = kea.clamp %a {min = -128 : i64, max = 300 : i64} : tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// -----

func.func @rescale_shape(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x16xi8> {
  // expected-error @+1 {{rescale preserves shape}}
  %0 = kea.rescale %a {quant = #kea.quant<multiplier = [1073741824], shift = [30],
                                          input_zp = 0, output_zp = 0,
                                          axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x16xi8>
  return %0 : tensor<1x4x4x16xi8>
}

// -----

func.func @rescale_per_channel_length(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  // expected-error @+1 {{quant has 2 scales but dimension 3 of the tensor is 8}}
  %0 = kea.rescale %a {quant = #kea.quant<multiplier = [1073741824, 1073741824],
                                          shift = [30, 30],
                                          input_zp = 0, output_zp = 0,
                                          axis = 3, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

//===--------------------------------------------------------------------===//
// The #kea.quant / #kea.epilogue attribute verifiers themselves
//===--------------------------------------------------------------------===//

// -----

func.func @quant_length_mismatch(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  // expected-error @+1 {{kea.quant multiplier and shift must have the same length, got 2 and 1}}
  %0 = kea.rescale %a {quant = #kea.quant<multiplier = [1073741824, 1073741824],
                                          shift = [30],
                                          input_zp = 0, output_zp = 0,
                                          axis = 3, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// -----

func.func @quant_per_tensor_multiple_scales(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  // expected-error @+1 {{kea.quant with axis = -1 is per tensor and needs exactly one multiplier/shift pair, got 2}}
  %0 = kea.rescale %a {quant = #kea.quant<multiplier = [1073741824, 1073741824],
                                          shift = [30, 30],
                                          input_zp = 0, output_zp = 0,
                                          axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// -----

func.func @quant_negative_multiplier(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  // expected-error @+1 {{kea.quant multipliers must be positive, got -1}}
  %0 = kea.rescale %a {quant = #kea.quant<multiplier = [-1], shift = [30],
                                          input_zp = 0, output_zp = 0,
                                          axis = -1, rounding = DOUBLE>}
      : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// -----

func.func @epilogue_partial_residual(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>)
    -> tensor<1x8x8x16xi8> {
  %0 = kea.conv2d %in, %w {
      zero_points = #kea.zp<input = 0, weight = 0>,
      strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
      dilations = array<i64: 1, 1>,
      // An AttrDef verifier runs at PARSE time, so the diagnostic is located at
      // the attribute rather than at the op.
      // expected-error @+1 {{kea.epilogue accum/residual/output are all-or-nothing, got 1 of 3}}
      epilogue = #kea.epilogue<requant = #kea.quant<multiplier = [1073741824], shift = [30],
                                                    input_zp = 0, output_zp = -5,
                                                    axis = -1, rounding = DOUBLE>,
                               accum = #kea.quant<multiplier = [1073741824], shift = [30],
                                                  input_zp = 0, output_zp = 0,
                                                  axis = -1, rounding = DOUBLE>>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// -----

func.func @epilogue_clamp_without_requant(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>)
    -> tensor<1x8x8x16xi32> {
  %0 = kea.conv2d %in, %w {
      zero_points = #kea.zp<input = 0, weight = 0>,
      strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
      dilations = array<i64: 1, 1>,
      // expected-error @+1 {{kea.epilogue without a requant stage cannot carry any later stage}}
      epilogue = #kea.epilogue<clamp = [-128, 127]>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

//===--------------------------------------------------------------------===//
// kea.pool
//===--------------------------------------------------------------------===//

// -----

func.func @pool_spatial(%a: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8> {
  // expected-error @+1 {{output spatial size 8x8 disagrees with pad/stride/dilation, which give 4x4}}
  %0 = kea.pool %a {kind = #kea.pool_kind<MAX>, kernel = array<i64: 2, 2>,
                    strides = array<i64: 2, 2>, pads = array<i64: 0, 0, 0, 0>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// -----

func.func @pool_channels(%a: tensor<1x8x8x16xi8>) -> tensor<1x4x4x8xi8> {
  // expected-error @+1 {{pooling preserves channels, got 16 -> 8}}
  %0 = kea.pool %a {kind = #kea.pool_kind<MAX>, kernel = array<i64: 2, 2>,
                    strides = array<i64: 2, 2>, pads = array<i64: 0, 0, 0, 0>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// -----

// Max pooling is monotonic, so quantization on it is meaningless. TOSA silently
// accepts a quantization_info on tosa.max_pool2d (TOSA_NOTES 13.17); we do not.
func.func @pool_max_with_quant(%a: tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8> {
  // expected-error @+1 {{max pooling is monotonic and carries no quantization}}
  %0 = kea.pool %a {kind = #kea.pool_kind<MAX>, kernel = array<i64: 2, 2>,
                    strides = array<i64: 2, 2>, pads = array<i64: 0, 0, 0, 0>,
                    quant = #kea.quant<multiplier = [1073741824], shift = [30],
                                       input_zp = -5, output_zp = -5,
                                       axis = -1, rounding = SINGLE>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  return %0 : tensor<1x4x4x16xi8>
}

// -----

// Average pooling may only rebase the zero point; a real scale change has to be
// a separate kea.rescale because VPOOL has no multiplier.
func.func @pool_avg_real_scale(%a: tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8> {
  // expected-error @+1 {{average pooling can only rebase the zero point}}
  %0 = kea.pool %a {kind = #kea.pool_kind<AVG>, kernel = array<i64: 2, 2>,
                    strides = array<i64: 2, 2>, pads = array<i64: 0, 0, 0, 0>,
                    quant = #kea.quant<multiplier = [1503238553], shift = [30],
                                       input_zp = -5, output_zp = -5,
                                       axis = -1, rounding = SINGLE>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  return %0 : tensor<1x4x4x16xi8>
}

//===--------------------------------------------------------------------===//
// kea.reshape / kea.transpose
//===--------------------------------------------------------------------===//

// -----

func.func @reshape_element_count(%a: tensor<1x7x7x32xi8>) -> tensor<1x1000xi8> {
  // expected-error @+1 {{reshape changes the element count, 1568 -> 1000}}
  %0 = kea.reshape %a {new_shape = array<i64: 1, 1000>}
      : (tensor<1x7x7x32xi8>) -> tensor<1x1000xi8>
  return %0 : tensor<1x1000xi8>
}

// -----

// Level 1 requires the inferred dimension to have been resolved already.
func.func @reshape_placeholder(%a: tensor<1x7x7x32xi8>) -> tensor<1x1568xi8> {
  // expected-error @+1 {{new_shape must be fully resolved at Level 1; a -1 placeholder is not allowed}}
  %0 = kea.reshape %a {new_shape = array<i64: 1, -1>}
      : (tensor<1x7x7x32xi8>) -> tensor<1x1568xi8>
  return %0 : tensor<1x1568xi8>
}

// -----

func.func @reshape_disagrees_with_type(%a: tensor<1x7x7x32xi8>) -> tensor<1x1568xi8> {
  // expected-error @+1 {{new_shape does not match the result type}}
  %0 = kea.reshape %a {new_shape = array<i64: 1568, 1>}
      : (tensor<1x7x7x32xi8>) -> tensor<1x1568xi8>
  return %0 : tensor<1x1568xi8>
}

// -----

func.func @transpose_duplicate_perm(%a: tensor<1x8x8x16xi8>) -> tensor<1x8x8x8xi8> {
  // expected-error @+1 {{perms entry 1 appears more than once}}
  %0 = kea.transpose %a {perms = array<i64: 0, 1, 1, 2>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x8x8x8xi8>
  return %0 : tensor<1x8x8x8xi8>
}

// -----

func.func @transpose_out_of_range(%a: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8> {
  // expected-error @+1 {{perms entry 4 is out of range}}
  %0 = kea.transpose %a {perms = array<i64: 0, 1, 2, 4>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// -----

func.func @transpose_shape(%a: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8> {
  // expected-error @+1 {{result dimension 1 is 8 but input dimension 3 is 16}}
  %0 = kea.transpose %a {perms = array<i64: 0, 3, 1, 2>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}
