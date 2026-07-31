// Quantized matmul / fully-connected -- verified against LLVM/MLIR 20.1.6.
//
// ANSWER TO "DOES tosa.fully_connected STILL EXIST IN 20.1.6?": YES.
// It is alive and well in this release (TosaOps.td line 226). It was removed
// upstream LATER, in the LLVM 21 development cycle. So on 20.1.6 you may emit
// either tosa.fully_connected or tosa.matmul.
//
// tosa.fully_connected:
//   operands: input 2-D [N, IC], weight 2-D [OC, IC], bias 1-D [OC]
//   attr:     optional #tosa.conv_quant<input_zp, weight_zp>
//   result:   2-D [N, OC]
//   NOTE the weight is [OC, IC] -- i.e. already transposed relative to matmul.
//
// tosa.matmul:
//   operands: a 3-D [B, M, K], b 3-D [B, K, N]     (both strictly rank 3)
//   attr:     optional #tosa.matmul_quant<a_zp, b_zp>
//   result:   3-D [B, M, N]
//   NOTE there is NO bias operand on matmul -- add it separately.
//
// Forward-compatibility recommendation: prefer tosa.matmul, since
// fully_connected disappears in LLVM 21+ and matmul does not.

// Quantized fully-connected: 4 rows, 8 in-features, 16 out-features.
func.func @fully_connected_i8(%a: tensor<4x8xi8>,
                              %w: tensor<16x8xi8>,
                              %b: tensor<16xi32>) -> tensor<4x16xi32> {
  %0 = tosa.fully_connected %a, %w, %b {
    quantization_info = #tosa.conv_quant<input_zp = -2, weight_zp = 0>
  } : (tensor<4x8xi8>, tensor<16x8xi8>, tensor<16xi32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

// fully_connected with constant weights/bias, as an exporter emits.
func.func @fully_connected_i8_const(%a: tensor<1x64xi8>) -> tensor<1x10xi32> {
  %w = "tosa.const"() {value = dense<3> : tensor<10x64xi8>} : () -> tensor<10x64xi8>
  %b = "tosa.const"() {value = dense<7> : tensor<10xi32>} : () -> tensor<10xi32>
  %0 = tosa.fully_connected %a, %w, %b {
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x64xi8>, tensor<10x64xi8>, tensor<10xi32>) -> tensor<1x10xi32>
  return %0 : tensor<1x10xi32>
}

// Quantized batched matmul. Both operands must be rank 3.
func.func @matmul_i8(%a: tensor<1x4x8xi8>, %b: tensor<1x8x16xi8>) -> tensor<1x4x16xi32> {
  %0 = tosa.matmul %a, %b {
    quantization_info = #tosa.matmul_quant<a_zp = -2, b_zp = 1>
  } : (tensor<1x4x8xi8>, tensor<1x8x16xi8>) -> tensor<1x4x16xi32>
  return %0 : tensor<1x4x16xi32>
}

// Batch > 1.
func.func @matmul_i8_batched(%a: tensor<4x32x64xi8>,
                             %b: tensor<4x64x32xi8>) -> tensor<4x32x32xi32> {
  %0 = tosa.matmul %a, %b {
    quantization_info = #tosa.matmul_quant<a_zp = -5, b_zp = 0>
  } : (tensor<4x32x64xi8>, tensor<4x64x32xi8>) -> tensor<4x32x32xi32>
  return %0 : tensor<4x32x32xi32>
}

// matmul without quantization info (attribute is optional).
func.func @matmul_i8_noquant(%a: tensor<1x4x8xi8>, %b: tensor<1x8x16xi8>) -> tensor<1x4x16xi32> {
  %0 = tosa.matmul %a, %b : (tensor<1x4x8xi8>, tensor<1x8x16xi8>) -> tensor<1x4x16xi32>
  return %0 : tensor<1x4x16xi32>
}

// The forward-compatible way to express a fully-connected layer using matmul:
// reshape the 2-D activation up to rank 3, matmul against the [1, IC, OC]
// weight, add the broadcast bias, then reshape back down.
func.func @fc_via_matmul_i8(%a: tensor<4x8xi8>) -> tensor<4x16xi32> {
  %w = "tosa.const"() {value = dense<2> : tensor<1x8x16xi8>} : () -> tensor<1x8x16xi8>
  %bias = "tosa.const"() {value = dense<11> : tensor<1x1x16xi32>} : () -> tensor<1x1x16xi32>
  %a3 = tosa.reshape %a {new_shape = array<i64: 1, 4, 8>}
      : (tensor<4x8xi8>) -> tensor<1x4x8xi8>
  %mm = tosa.matmul %a3, %w {
    quantization_info = #tosa.matmul_quant<a_zp = -2, b_zp = 0>
  } : (tensor<1x4x8xi8>, tensor<1x8x16xi8>) -> tensor<1x4x16xi32>
  %biased = tosa.add %mm, %bias
      : (tensor<1x4x16xi32>, tensor<1x1x16xi32>) -> tensor<1x4x16xi32>
  %out = tosa.reshape %biased {new_shape = array<i64: 4, 16>}
      : (tensor<1x4x16xi32>) -> tensor<4x16xi32>
  return %out : tensor<4x16xi32>
}

// Full quantized classifier tail: fully-connected then rescale to i8.
func.func @fc_rescale_i8(%a: tensor<1x64xi8>) -> tensor<1x10xi8> {
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
