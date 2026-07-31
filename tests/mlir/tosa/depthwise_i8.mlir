// Quantized int8 depthwise conv2d. Verified against LLVM/MLIR 20.1.6.
//
// KEY FACTS:
//   * Weight layout is HWCM: [KH, KW, in_channels, channel_multiplier].
//     This is DIFFERENT from tosa.conv2d, whose weight is OHWI.
//   * Output channel count = in_channels * channel_multiplier.
//   * Zero points via #tosa.conv_quant<input_zp, weight_zp> attribute (20.1.6).
//   * acc_type is mandatory.

// Channel multiplier 1 -- the MobileNet case.
func.func @depthwise_i8_m1(%input: tensor<1x8x8x16xi8>,
                           %weight: tensor<3x3x16x1xi8>,
                           %bias: tensor<16xi32>) -> tensor<1x8x8x16xi32> {
  %0 = tosa.depthwise_conv2d %input, %weight, %bias {
    pad = array<i64: 1, 1, 1, 1>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -7, weight_zp = 0>
  } : (tensor<1x8x8x16xi8>, tensor<3x3x16x1xi8>, tensor<16xi32>)
      -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// Channel multiplier 2: 8 input channels -> 16 output channels.
// Note the bias length must be in_channels * multiplier = 16.
func.func @depthwise_i8_m2(%input: tensor<1x8x8x8xi8>,
                           %weight: tensor<3x3x8x2xi8>,
                           %bias: tensor<16xi32>) -> tensor<1x8x8x16xi32> {
  %0 = tosa.depthwise_conv2d %input, %weight, %bias {
    pad = array<i64: 1, 1, 1, 1>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x8x8x8xi8>, tensor<3x3x8x2xi8>, tensor<16xi32>)
      -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// Stride 2 depthwise -- the MobileNetV2 downsampling block.
func.func @depthwise_i8_stride2(%input: tensor<1x16x16x24xi8>,
                                %weight: tensor<3x3x24x1xi8>,
                                %bias: tensor<24xi32>) -> tensor<1x8x8x24xi32> {
  %0 = tosa.depthwise_conv2d %input, %weight, %bias {
    pad = array<i64: 0, 1, 0, 1>,
    stride = array<i64: 2, 2>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x16x16x24xi8>, tensor<3x3x24x1xi8>, tensor<24xi32>)
      -> tensor<1x8x8x24xi32>
  return %0 : tensor<1x8x8x24xi32>
}

// With constant weights, as an exporter emits.
func.func @depthwise_i8_const(%input: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi32> {
  %w = "tosa.const"() {value = dense<1> : tensor<3x3x16x1xi8>} : () -> tensor<3x3x16x1xi8>
  %b = "tosa.const"() {value = dense<0> : tensor<16xi32>} : () -> tensor<16xi32>
  %0 = tosa.depthwise_conv2d %input, %w, %b {
    pad = array<i64: 1, 1, 1, 1>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x8x8x16xi8>, tensor<3x3x16x1xi8>, tensor<16xi32>)
      -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}
