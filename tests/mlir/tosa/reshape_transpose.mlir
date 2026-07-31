// reshape / transpose / slice / concat / pad -- verified against LLVM/MLIR 20.1.6.
//
// THIS IS THE AREA WITH THE MOST VERSION SKEW. In 20.1.6 the shape migration
// to !tosa.shape is HALF DONE:
//
//   tosa.reshape   -> shape is an ATTRIBUTE: new_shape = array<i64: ...>
//                     (NOT a !tosa.shape operand; that lands in LLVM 21)
//   tosa.transpose -> permutation is an OPERAND: a rank-1 i32 tensor value,
//                     normally a tosa.const. (NOT a `perms` attribute; that
//                     also lands in LLVM 21)
//   tosa.slice     -> start/size are ATTRIBUTES: DenseI64ArrayAttr
//   tosa.tile      -> multiples IS a !tosa.shape operand, built with
//                     tosa.const_shape. This one HAS already migrated.
//   tosa.pad       -> padding IS a !tosa.shape operand, built with
//                     tosa.const_shape. This one HAS already migrated.
//
// So: pad and tile use !tosa.shape; reshape, transpose and slice do not.

// Reshape: flatten NHWC to 2-D for a classifier head.
func.func @reshape_flatten(%a: tensor<1x7x7x32xi8>) -> tensor<1x1568xi8> {
  %0 = tosa.reshape %a {new_shape = array<i64: 1, 1568>}
      : (tensor<1x7x7x32xi8>) -> tensor<1x1568xi8>
  return %0 : tensor<1x1568xi8>
}

// Reshape with an inferred (-1) dimension.
func.func @reshape_infer(%a: tensor<1x7x7x32xi8>) -> tensor<1x1568xi8> {
  %0 = tosa.reshape %a {new_shape = array<i64: 1, -1>}
      : (tensor<1x7x7x32xi8>) -> tensor<1x1568xi8>
  return %0 : tensor<1x1568xi8>
}

// Rank-increasing reshape (2-D -> 3-D), needed to feed tosa.matmul.
func.func @reshape_expand_rank(%a: tensor<4x8xi8>) -> tensor<1x4x8xi8> {
  %0 = tosa.reshape %a {new_shape = array<i64: 1, 4, 8>}
      : (tensor<4x8xi8>) -> tensor<1x4x8xi8>
  return %0 : tensor<1x4x8xi8>
}

// Transpose NHWC -> NCHW. The permutation is an OPERAND (rank-1 i32 tensor).
func.func @transpose_nhwc_to_nchw(%a: tensor<1x8x8x16xi8>) -> tensor<1x16x8x8xi8> {
  %perm = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>}
      : () -> tensor<4xi32>
  %0 = tosa.transpose %a, %perm
      : (tensor<1x8x8x16xi8>, tensor<4xi32>) -> tensor<1x16x8x8xi8>
  return %0 : tensor<1x16x8x8xi8>
}

// Transpose NCHW -> NHWC, the inverse permutation.
func.func @transpose_nchw_to_nhwc(%a: tensor<1x16x8x8xi8>) -> tensor<1x8x8x16xi8> {
  %perm = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>}
      : () -> tensor<4xi32>
  %0 = tosa.transpose %a, %perm
      : (tensor<1x16x8x8xi8>, tensor<4xi32>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// Weight layout conversion HWIO -> OHWI, i.e. what a PyTorch/TF exporter needs
// to do to feed tosa.conv2d.
func.func @transpose_weight_hwio_to_ohwi(%w: tensor<3x3x4x16xi8>) -> tensor<16x3x3x4xi8> {
  %perm = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi32>}
      : () -> tensor<4xi32>
  %0 = tosa.transpose %w, %perm
      : (tensor<3x3x4x16xi8>, tensor<4xi32>) -> tensor<16x3x3x4xi8>
  return %0 : tensor<16x3x3x4xi8>
}

// Combined reshape + transpose, as in a channel-shuffle.
func.func @reshape_transpose_combined(%a: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8> {
  %r = tosa.reshape %a {new_shape = array<i64: 1, 8, 8, 2, 8>}
      : (tensor<1x8x8x16xi8>) -> tensor<1x8x8x2x8xi8>
  %perm = "tosa.const"() {value = dense<[0, 1, 2, 4, 3]> : tensor<5xi32>}
      : () -> tensor<5xi32>
  %t = tosa.transpose %r, %perm
      : (tensor<1x8x8x2x8xi8>, tensor<5xi32>) -> tensor<1x8x8x8x2xi8>
  %o = tosa.reshape %t {new_shape = array<i64: 1, 8, 8, 16>}
      : (tensor<1x8x8x8x2xi8>) -> tensor<1x8x8x16xi8>
  return %o : tensor<1x8x8x16xi8>
}

// Slice: start/size are attributes.
func.func @slice_i8(%a: tensor<1x8x8x16xi8>) -> tensor<1x4x4x8xi8> {
  %0 = tosa.slice %a {
    start = array<i64: 0, 2, 2, 0>,
    size  = array<i64: 1, 4, 4, 8>
  } : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// Concat along the channel axis.
func.func @concat_i8(%a: tensor<1x8x8x8xi8>, %b: tensor<1x8x8x8xi8>) -> tensor<1x8x8x16xi8> {
  %0 = tosa.concat %a, %b {axis = 3 : i32}
      : (tensor<1x8x8x8xi8>, tensor<1x8x8x8xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// Tile: multiples is a !tosa.shape OPERAND in 20.1.6, not an attribute.
// (tile and pad are the two ops that have already migrated in this release.)
func.func @tile_i8(%a: tensor<1x4x4x8xi8>) -> tensor<2x4x4x8xi8> {
  %mult = tosa.const_shape {value = dense<[2, 1, 1, 1]> : tensor<4xindex>}
      : () -> !tosa.shape<4>
  %0 = tosa.tile %a, %mult
      : (tensor<1x4x4x8xi8>, !tosa.shape<4>) -> tensor<2x4x4x8xi8>
  return %0 : tensor<2x4x4x8xi8>
}

// PAD -- the one op that already uses !tosa.shape in 20.1.6.
// The shape operand holds [low0, high0, low1, high1, ...], so its length is
// 2 * rank. It is produced by tosa.const_shape with a tensor<Nxindex> value.
// The quantization attribute is #tosa.pad_quant<input_zp = ...>, which is what
// the padding is filled with for a quantized tensor.
func.func @pad_i8(%a: tensor<1x8x8x16xi8>) -> tensor<1x10x10x16xi8> {
  %padding = tosa.const_shape {value = dense<[0, 0, 1, 1, 1, 1, 0, 0]> : tensor<8xindex>}
      : () -> !tosa.shape<8>
  %0 = tosa.pad %a, %padding {
    quantization_info = #tosa.pad_quant<input_zp = -5>
  } : (tensor<1x8x8x16xi8>, !tosa.shape<8>) -> tensor<1x10x10x16xi8>
  return %0 : tensor<1x10x10x16xi8>
}

// Pad with an explicit pad_const operand instead of relying on the
// quantization attribute. NOTE: pad_const must be a 0-D tensor -- tensor<i8>,
// NOT tensor<1xi8>. A rank-1 single-element tensor is rejected by the verifier.
func.func @pad_i8_explicit_const(%a: tensor<1x8x8x16xi8>) -> tensor<1x10x10x16xi8> {
  %padding = tosa.const_shape {value = dense<[0, 0, 1, 1, 1, 1, 0, 0]> : tensor<8xindex>}
      : () -> !tosa.shape<8>
  %pc = "tosa.const"() {value = dense<-5> : tensor<i8>} : () -> tensor<i8>
  %0 = tosa.pad %a, %padding, %pc
      : (tensor<1x8x8x16xi8>, !tosa.shape<8>, tensor<i8>) -> tensor<1x10x10x16xi8>
  return %0 : tensor<1x10x10x16xi8>
}

// tosa.cast: pure element-type conversion, no scaling and no zero point.
// Use tosa.rescale, never tosa.cast, to requantize.
func.func @cast_i8_to_i32(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32> {
  %0 = tosa.cast %a : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}
