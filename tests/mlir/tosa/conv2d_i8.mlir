// Quantized int8 conv2d, NHWC. Verified against LLVM/MLIR 20.1.6.
//
// KEY FACTS FOR 20.1.6 (differ from LLVM 21+ TOSA docs you may find online):
//   * Zero points live in an ATTRIBUTE: #tosa.conv_quant<input_zp, weight_zp>.
//     They are NOT operands in this release.
//   * `acc_type` is a mandatory TypeAttr (i32 / i48 / f16 / f32).
//   * Layouts: input NHWC, weight OHWI (out_ch, KH, KW, in_ch), bias [OC].
//   * pad is [top, bottom, left, right]; stride/dilation are [h, w].
//   * Generic assembly is "operands attr-dict : functional-type(operands, results)".

// Basic 3x3, stride 1, pad 1, dilation 1.
func.func @conv2d_i8_basic(%input: tensor<1x8x8x4xi8>,
                           %weight: tensor<16x3x3x4xi8>,
                           %bias: tensor<16xi32>) -> tensor<1x8x8x16xi32> {
  %0 = tosa.conv2d %input, %weight, %bias {
    pad = array<i64: 1, 1, 1, 1>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -3, weight_zp = 0>
  } : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi32>)
      -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// Stride 2 downsample, asymmetric pad.
func.func @conv2d_i8_stride2(%input: tensor<1x16x16x8xi8>,
                             %weight: tensor<32x3x3x8xi8>,
                             %bias: tensor<32xi32>) -> tensor<1x8x8x32xi32> {
  %0 = tosa.conv2d %input, %weight, %bias {
    pad = array<i64: 0, 1, 0, 1>,
    stride = array<i64: 2, 2>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x16x16x8xi8>, tensor<32x3x3x8xi8>, tensor<32xi32>)
      -> tensor<1x8x8x32xi32>
  return %0 : tensor<1x8x8x32xi32>
}

// Dilation 2 (atrous). Effective kernel 5x5, so pad 2 keeps spatial size.
func.func @conv2d_i8_dilated(%input: tensor<1x8x8x4xi8>,
                             %weight: tensor<16x3x3x4xi8>,
                             %bias: tensor<16xi32>) -> tensor<1x8x8x16xi32> {
  %0 = tosa.conv2d %input, %weight, %bias {
    pad = array<i64: 2, 2, 2, 2>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 2, 2>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = 7, weight_zp = 0>
  } : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi32>)
      -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// 1x1 pointwise convolution with constant weights/bias, as an exporter emits it.
func.func @conv2d_i8_pointwise_const(%input: tensor<1x8x8x4xi8>) -> tensor<1x8x8x16xi32> {
  %w = "tosa.const"() {value = dense<2> : tensor<16x1x1x4xi8>} : () -> tensor<16x1x1x4xi8>
  %b = "tosa.const"() {value = dense<100> : tensor<16xi32>} : () -> tensor<16xi32>
  %0 = tosa.conv2d %input, %w, %b {
    pad = array<i64: 0, 0, 0, 0>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -5, weight_zp = 0>
  } : (tensor<1x8x8x4xi8>, tensor<16x1x1x4xi8>, tensor<16xi32>)
      -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// Fully generic form -- what `mlir-opt --mlir-print-op-generic` produces.
// Useful as the canonical target for a textual exporter.
func.func @conv2d_i8_generic_form(%input: tensor<1x4x4x2xi8>,
                                  %weight: tensor<8x3x3x2xi8>,
                                  %bias: tensor<8xi32>) -> tensor<1x4x4x8xi32> {
  %0 = "tosa.conv2d"(%input, %weight, %bias) {
    pad = array<i64: 1, 1, 1, 1>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -3, weight_zp = 0>
  } : (tensor<1x4x4x2xi8>, tensor<8x3x3x2xi8>, tensor<8xi32>)
      -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}
