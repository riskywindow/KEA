// tosa.rescale -- verified against LLVM/MLIR 20.1.6.
//
// SIGNATURE IN 20.1.6 (all quantization state is in ATTRIBUTES, not operands):
//   tosa.rescale %input {
//     input_zp     : i32          (I32Attr, required)
//     output_zp    : i32          (I32Attr, required)
//     multiplier   : array<i32>   (DenseI32ArrayAttr, required)
//     shift        : array<i8>    (DenseI8ArrayAttr, required)   <-- i8, not i32
//     scale32      : bool         (BoolAttr, required)
//     double_round : bool         (BoolAttr, required)
//     per_channel  : bool         (BoolAttr, required)
//     input_unsigned  : bool      (optional, default false)
//     output_unsigned : bool      (optional, default false)
//   } : (tensor<...xi32>) -> tensor<...xi8>
//
// per_channel = true  => multiplier/shift length == size of the LAST dimension.
// per_channel = false => multiplier/shift length == 1.
//
// Exact math (extracted from this build's own --tosa-to-arith lowering):
//   value = i32(input) - input_zp
//   prod  = sext_i64(value) * sext_i64(multiplier)
//   round = (i64(1) << shift) >> 1
//   acc   = prod + round
//   if (double_round && shift > 31)
//       acc += (value >= 0) ? (1 << 30) : -(1 << 30)
//   res   = i32(acc >> shift)          // arithmetic shift right
//   res   = res + output_zp
//   out   = clamp(res, -128, 127)      // for an i8 result

// Per-channel i32 -> i8. 8 channels, so 8 multipliers and 8 shifts.
func.func @rescale_per_channel_i32_to_i8(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  %0 = tosa.rescale %a {
    input_zp = 0 : i32,
    output_zp = -5 : i32,
    multiplier = array<i32: 1073741824, 1234567890, 2000000000, 1500000000,
                            1073741824, 1900000000, 1610612736, 1288490189>,
    shift = array<i8: 30, 31, 32, 33, 30, 31, 30, 31>,
    scale32 = true,
    double_round = true,
    per_channel = true
  } : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// Per-tensor i32 -> i8. Exactly one multiplier and one shift.
func.func @rescale_per_tensor_i32_to_i8(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  %0 = tosa.rescale %a {
    input_zp = 0 : i32,
    output_zp = -5 : i32,
    multiplier = array<i32: 1073741824>,
    shift = array<i8: 30>,
    scale32 = true,
    double_round = true,
    per_channel = false
  } : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// double_round = false. Only differs from the above when shift > 31.
func.func @rescale_no_double_round(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  %0 = tosa.rescale %a {
    input_zp = 0 : i32,
    output_zp = 0 : i32,
    multiplier = array<i32: 1073741824>,
    shift = array<i8: 33>,
    scale32 = true,
    double_round = false,
    per_channel = false
  } : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// scale32 = false: 16-bit multiplier path. The multiplier must fit in i16
// and shift is restricted; the accumulate is done in i32 rather than i64.
func.func @rescale_scale16(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  %0 = tosa.rescale %a {
    input_zp = 0 : i32,
    output_zp = 0 : i32,
    multiplier = array<i32: 16384>,
    shift = array<i8: 15>,
    scale32 = false,
    double_round = false,
    per_channel = false
  } : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// i8 -> i32 widening rescale. This is how you bring an i8 tensor into a common
// i32 domain before a quantized elementwise add.
func.func @rescale_i8_to_i32(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32> {
  %0 = tosa.rescale %a {
    input_zp = -5 : i32,
    output_zp = 0 : i32,
    multiplier = array<i32: 1073741824>,
    shift = array<i8: 20>,
    scale32 = true,
    double_round = true,
    per_channel = false
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// i8 -> i8 requantization (change of scale/zero-point without widening).
func.func @rescale_i8_to_i8(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {
  %0 = tosa.rescale %a {
    input_zp = -5 : i32,
    output_zp = 12 : i32,
    multiplier = array<i32: 1717986918>,
    shift = array<i8: 31>,
    scale32 = true,
    double_round = true,
    per_channel = false
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// Unsigned input (uint8 -> int8), using the optional input_unsigned flag.
func.func @rescale_u8_to_i8(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {
  %0 = tosa.rescale %a {
    input_zp = 128 : i32,
    output_zp = 0 : i32,
    multiplier = array<i32: 1073741824>,
    shift = array<i8: 30>,
    scale32 = true,
    double_round = true,
    per_channel = false,
    input_unsigned = true,
    output_unsigned = false
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}
