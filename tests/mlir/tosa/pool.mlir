// avg_pool2d and max_pool2d on i8 -- verified against LLVM/MLIR 20.1.6.
//
// KEY FACTS:
//   * tosa.avg_pool2d IS quantized: it takes acc_type (mandatory) and an
//     OPTIONAL #tosa.unary_quant<input_zp, output_zp> attribute. Note this is
//     unary_quant, NOT conv_quant -- it carries input_zp and OUTPUT_zp.
//   * tosa.max_pool2d is NOT quantized at all: no acc_type, no quant attr.
//     Max pooling is monotonic, so it needs no zero-point correction.
//     DANGER: writing acc_type/quantization_info on max_pool2d anyway does NOT
//     error -- MLIR keeps unrecognized entries as discardable attributes, so it
//     parses and round-trips while being silently ignored by every pass.
//   * Attribute order in both: kernel [h,w], stride [h,w], pad [t,b,l,r].
//   * avg_pool2d output element type equals the input element type (i8 -> i8);
//     acc_type only describes the internal accumulator.

// Quantized 2x2/2 average pool, i8 -> i8, i32 accumulator.
func.func @avg_pool_i8(%a: tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8> {
  %0 = tosa.avg_pool2d %a {
    kernel = array<i64: 2, 2>,
    stride = array<i64: 2, 2>,
    pad = array<i64: 0, 0, 0, 0>,
    acc_type = i32,
    quantization_info = #tosa.unary_quant<input_zp = -5, output_zp = -5>
  } : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  return %0 : tensor<1x4x4x16xi8>
}

// Global average pool: kernel == full spatial extent, output 1x1.
func.func @global_avg_pool_i8(%a: tensor<1x7x7x32xi8>) -> tensor<1x1x1x32xi8> {
  %0 = tosa.avg_pool2d %a {
    kernel = array<i64: 7, 7>,
    stride = array<i64: 1, 1>,
    pad = array<i64: 0, 0, 0, 0>,
    acc_type = i32,
    quantization_info = #tosa.unary_quant<input_zp = -128, output_zp = -128>
  } : (tensor<1x7x7x32xi8>) -> tensor<1x1x1x32xi8>
  return %0 : tensor<1x1x1x32xi8>
}

// avg_pool2d without the quantization attribute (it is optional). Valid, and
// means both zero points are zero.
func.func @avg_pool_i8_noquant(%a: tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8> {
  %0 = tosa.avg_pool2d %a {
    kernel = array<i64: 2, 2>,
    stride = array<i64: 2, 2>,
    pad = array<i64: 0, 0, 0, 0>,
    acc_type = i32
  } : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  return %0 : tensor<1x4x4x16xi8>
}

// Max pool 2x2/2. No acc_type and no quantization_info -- neither exists on
// this op (and neither would be diagnosed if you wrote one; see header).
func.func @max_pool_i8(%a: tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8> {
  %0 = tosa.max_pool2d %a {
    kernel = array<i64: 2, 2>,
    stride = array<i64: 2, 2>,
    pad = array<i64: 0, 0, 0, 0>
  } : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  return %0 : tensor<1x4x4x16xi8>
}

// Max pool 3x3 stride 2 with padding, plus the optional nan_mode attribute.
func.func @max_pool_i8_3x3(%a: tensor<1x16x16x8xi8>) -> tensor<1x8x8x8xi8> {
  %0 = tosa.max_pool2d %a {
    kernel = array<i64: 3, 3>,
    stride = array<i64: 2, 2>,
    pad = array<i64: 0, 1, 0, 1>,
    nan_mode = "PROPAGATE"
  } : (tensor<1x16x16x8xi8>) -> tensor<1x8x8x8xi8>
  return %0 : tensor<1x8x8x8xi8>
}

// Average pool with explicit zero zero-points.
//
// NOTE: although the ODS type constraint Tosa_AccType allows i32/i48/f16/f32,
// the verifier REJECTS acc_type = i48 on an integer tensor:
//   "'tosa.avg_pool2d' op accumulator type for integer tensor is not i32"
// For any integer avg_pool2d the accumulator must be exactly i32.
func.func @avg_pool_i8_zero_zp(%a: tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8> {
  %0 = tosa.avg_pool2d %a {
    kernel = array<i64: 2, 2>,
    stride = array<i64: 2, 2>,
    pad = array<i64: 0, 0, 0, 0>,
    acc_type = i32,
    quantization_info = #tosa.unary_quant<input_zp = 0, output_zp = 0>
  } : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  return %0 : tensor<1x4x4x16xi8>
}
