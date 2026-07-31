// tosa.clamp on i8 -- verified against LLVM/MLIR 20.1.6.
//
// KEY FACTS:
//   * In 20.1.6 clamp carries FOUR required attributes: min_int, max_int
//     (both I64Attr) and min_fp, max_fp (both F32-ish FloatAttr).
//     You must supply the float pair EVEN FOR AN INTEGER TENSOR -- they are
//     simply ignored by the integer lowering. Omitting them is a parse error.
//     (In LLVM 21+ these were merged into a single min_val/max_val pair.)
//   * nan_mode is optional, defaults to "PROPAGATE".
//   * No zero-point subtraction is performed. To clamp at the zero point,
//     pass the zero point itself as min_int.

// Plain ReLU on i8 with zero point -5: clamp lower bound AT the zero point.
func.func @relu_i8(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {
  %0 = tosa.clamp %a {
    min_int = -5 : i64,
    max_int = 127 : i64,
    min_fp = 0.0 : f32,
    max_fp = 3.4028234663852886E+38 : f32
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// ReLU6 on i8. With scale s and zero point zp, 6.0 maps to
// round(6.0 / s) + zp; here that has been precomputed to 91.
func.func @relu6_i8(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {
  %0 = tosa.clamp %a {
    min_int = -5 : i64,
    max_int = 91 : i64,
    min_fp = 0.0 : f32,
    max_fp = 6.0 : f32
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// ReLU6 in the common "zero point = -128" activation encoding, where the
// quantized range is effectively unsigned. Saturates to the full i8 range.
func.func @relu6_i8_zp128(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {
  %0 = tosa.clamp %a {
    min_int = -128 : i64,
    max_int = 127 : i64,
    min_fp = 0.0 : f32,
    max_fp = 6.0 : f32
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// Explicit nan_mode (only meaningful for floats, but it parses on ints).
func.func @clamp_i8_nanmode(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {
  %0 = tosa.clamp %a {
    min_int = -128 : i64,
    max_int = 127 : i64,
    min_fp = 0.0 : f32,
    max_fp = 6.0 : f32,
    nan_mode = "PROPAGATE"
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// Clamp applied directly to an i32 accumulator, before the rescale.
func.func @clamp_i32_acc(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32> {
  %0 = tosa.clamp %a {
    min_int = 0 : i64,
    max_int = 2147483647 : i64,
    min_fp = 0.0 : f32,
    max_fp = 3.4028234663852886E+38 : f32
  } : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// The idiomatic quantized ReLU6 sequence: i32 accumulator -> rescale to i8 ->
// clamp in the i8 domain.
func.func @conv_rescale_relu6(%acc: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  %0 = tosa.rescale %acc {
    input_zp = 0 : i32,
    output_zp = -128 : i32,
    multiplier = array<i32: 1073741824>,
    shift = array<i8: 30>,
    scale32 = true,
    double_round = true,
    per_channel = false
  } : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  %1 = tosa.clamp %0 {
    min_int = -128 : i64,
    max_int = 127 : i64,
    min_fp = 0.0 : f32,
    max_fp = 6.0 : f32
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
  return %1 : tensor<1x4x4x8xi8>
}
