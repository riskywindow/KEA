// Quantized elementwise add -- verified against LLVM/MLIR 20.1.6.
//
// KEY FACT: tosa.add has NO quantization attribute at all. There is no
// "BinaryOpQuantizationAttr" in TOSA. The quantization is expressed entirely
// by surrounding tosa.rescale ops, which is what makes the idiomatic pattern:
//
//   lhs_i32 = rescale(lhs_i8, input_zp=lhs_zp, output_zp=0, <lhs scale>)
//   rhs_i32 = rescale(rhs_i8, input_zp=rhs_zp, output_zp=0, <rhs scale>)
//   sum_i32 = add(lhs_i32, rhs_i32)
//   out_i8  = rescale(sum_i32, input_zp=0, output_zp=out_zp, <out scale>)
//
// Both rescales must bring the operands into the SAME intermediate scale.
// The usual choice is 2^-20 relative to max(lhs_scale, rhs_scale), giving the
// shift headroom that avoids i32 overflow in the add.

// Raw i8 add with no rescale plumbing. Legal TOSA, but only correct when both
// operands already share a scale and have zero point 0. Included so the
// exporter can tell the two cases apart.
func.func @add_i8_raw(%a: tensor<1x4x4x8xi8>, %b: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {
  %0 = tosa.add %a, %b : (tensor<1x4x4x8xi8>, tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// The idiomatic fully quantized add: rescale both inputs into a common i32
// domain, add there, then rescale back down to i8.
func.func @add_i8_requantized(%a: tensor<1x4x4x8xi8>,
                              %b: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {
  // lhs: zero point -5, brought to the common i32 scale.
  %la = tosa.rescale %a {
    input_zp = -5 : i32, output_zp = 0 : i32,
    multiplier = array<i32: 1073741824>, shift = array<i8: 10>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32>

  // rhs: zero point 3, different scale, same common i32 domain.
  %lb = tosa.rescale %b {
    input_zp = 3 : i32, output_zp = 0 : i32,
    multiplier = array<i32: 1932735283>, shift = array<i8: 11>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32>

  %sum = tosa.add %la, %lb : (tensor<1x4x4x8xi32>, tensor<1x4x4x8xi32>)
      -> tensor<1x4x4x8xi32>

  // Back down to i8 at the output scale / zero point.
  %out = tosa.rescale %sum {
    input_zp = 0 : i32, output_zp = -7 : i32,
    multiplier = array<i32: 1288490189>, shift = array<i8: 31>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>

  return %out : tensor<1x4x4x8xi8>
}

// Quantized add fused with a ReLU on the output, as fused in most PTQ graphs.
func.func @add_i8_requantized_relu(%a: tensor<1x4x4x8xi8>,
                                   %b: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {
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
  %sum = tosa.add %la, %lb : (tensor<1x4x4x8xi32>, tensor<1x4x4x8xi32>)
      -> tensor<1x4x4x8xi32>
  %out = tosa.rescale %sum {
    input_zp = 0 : i32, output_zp = -128 : i32,
    multiplier = array<i32: 1288490189>, shift = array<i8: 31>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  %relu = tosa.clamp %out {
    min_int = -128 : i64, max_int = 127 : i64,
    min_fp = 0.0 : f32, max_fp = 6.0 : f32
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
  return %relu : tensor<1x4x4x8xi8>
}

// Broadcast add of a per-channel bias-like i32 term. TOSA broadcasting
// requires equal RANK, so the bias must be reshaped to 1x1x1xC first, not
// left as a bare 1-D tensor.
func.func @add_i32_broadcast(%a: tensor<1x4x4x8xi32>,
                             %bias: tensor<8xi32>) -> tensor<1x4x4x8xi32> {
  %r = tosa.reshape %bias {new_shape = array<i64: 1, 1, 1, 8>}
      : (tensor<8xi32>) -> tensor<1x1x1x8xi32>
  %0 = tosa.add %a, %r : (tensor<1x4x4x8xi32>, tensor<1x1x1x8xi32>)
      -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// tosa.sub follows exactly the same no-attribute pattern.
func.func @sub_i32(%a: tensor<1x4x4x8xi32>, %b: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32> {
  %0 = tosa.sub %a, %b : (tensor<1x4x4x8xi32>, tensor<1x4x4x8xi32>)
      -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// GOTCHA: tosa.mul takes THREE operands in 20.1.6. The shift is an OPERAND --
// a rank-1 single-element i8 tensor -- not a `shift = N : i8` attribute as in
// older TOSA. Writing it as an attribute fails with
// "'tosa.mul' op expected 3 operands, but found 2".
func.func @mul_i32(%a: tensor<1x4x4x8xi32>, %b: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32> {
  %shift = "tosa.const"() {value = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = tosa.mul %a, %b, %shift
      : (tensor<1x4x4x8xi32>, tensor<1x4x4x8xi32>, tensor<1xi8>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// NOTE on the WIDENING form of tosa.mul (i8 x i8 -> i32): it is valid TOSA and
// round-trips through mlir-opt, but its tosa-to-linalg lowering is BROKEN in
// 20.1.6 -- it builds a linalg.generic that yields i8 into an i32 body and
// fails verification with
//   "'linalg.yield' op type of yield operand 1 ('i8') doesn't match the
//    element type of the enclosing linalg.generic op ('i32')".
// It is therefore deliberately NOT included as an example here, so that this
// suite stays a clean CI gate. Repro is in docs/TOSA_NOTES.md. Emit a
// same-type mul plus an explicit tosa.cast/tosa.rescale instead.
