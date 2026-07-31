// linalg named convolution ops, i8 -> i32. Verified on LLVM/MLIR 20.1.6.
//
// WHICH OPS EXIST IN 20.1.6 (all confirmed by mlir-opt on this machine):
//   linalg.conv_2d_nhwc_hwcf     filter HWCF [KH,KW,IC,OC]   -- exists
//   linalg.conv_2d_nhwc_hwcf_q   same + zero-point operands  -- exists
//   linalg.conv_2d_nhwc_fhwc     filter FHWC [OC,KH,KW,IC]   -- exists
//   linalg.conv_2d_nhwc_fhwc_q   same + zero-point operands  -- exists
//
// IMPORTANT: the `_q` variants take the zero points as TRAILING SCALAR
// OPERANDS of type i32 inside ins(...), NOT as attributes:
//     ins(%input, %filter, %input_zp, %filter_zp : T, T, i32, i32)
// This is exactly what --tosa-to-linalg-named generates from a quantized
// tosa.conv2d, and it is the natural quantized entry point for a backend.
//
// Note also: linalg conv ops do NOT have padding. Padding must be materialized
// as an explicit tensor.pad / memref copy BEFORE the conv, filled with the
// input zero point. The output shape is therefore the already-padded result.
//
// The `outs` operand is the accumulator initializer: for a quantized conv it
// is normally pre-broadcast with the bias, or filled with 0 via linalg.fill.

//===--------------------------------------------------------------------===//
// Tensors
//===--------------------------------------------------------------------===//

// Unquantized i8 -> i32, HWCF filter. 8x8 input, 3x3 filter, no pad -> 6x6.
func.func @conv_2d_nhwc_hwcf_tensor(%input: tensor<1x8x8x4xi8>,
                                    %filter: tensor<3x3x4x16xi8>,
                                    %init: tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32> {
  %0 = linalg.conv_2d_nhwc_hwcf {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : tensor<1x8x8x4xi8>, tensor<3x3x4x16xi8>)
    outs(%init : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  return %0 : tensor<1x6x6x16xi32>
}

// Quantized variant: zero points as trailing i32 scalar operands.
func.func @conv_2d_nhwc_hwcf_q_tensor(%input: tensor<1x8x8x4xi8>,
                                      %filter: tensor<3x3x4x16xi8>,
                                      %init: tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32> {
  %izp = arith.constant -3 : i32
  %fzp = arith.constant 0 : i32
  %0 = linalg.conv_2d_nhwc_hwcf_q {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %izp, %fzp
        : tensor<1x8x8x4xi8>, tensor<3x3x4x16xi8>, i32, i32)
    outs(%init : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  return %0 : tensor<1x6x6x16xi32>
}

// FHWC filter layout -- this is what --tosa-to-linalg-named emits BY DEFAULT
// from tosa.conv2d, because TOSA weights are already OHWI == FHWC.
// Pass --prefer-conv2d-kernel-layout-hwcf to get the HWCF form instead.
func.func @conv_2d_nhwc_fhwc_q_tensor(%input: tensor<1x8x8x4xi8>,
                                      %filter: tensor<16x3x3x4xi8>,
                                      %init: tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32> {
  %izp = arith.constant -3 : i32
  %fzp = arith.constant 0 : i32
  %0 = linalg.conv_2d_nhwc_fhwc_q {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %izp, %fzp
        : tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, i32, i32)
    outs(%init : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  return %0 : tensor<1x6x6x16xi32>
}

// Stride 2 and dilation 2 both expressed as dense<...> tensor attributes.
func.func @conv_2d_nhwc_hwcf_strided(%input: tensor<1x16x16x4xi8>,
                                     %filter: tensor<3x3x4x16xi8>,
                                     %init: tensor<1x7x7x16xi32>) -> tensor<1x7x7x16xi32> {
  %0 = linalg.conv_2d_nhwc_hwcf {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%input, %filter : tensor<1x16x16x4xi8>, tensor<3x3x4x16xi8>)
    outs(%init : tensor<1x7x7x16xi32>) -> tensor<1x7x7x16xi32>
  return %0 : tensor<1x7x7x16xi32>
}

func.func @conv_2d_nhwc_hwcf_dilated(%input: tensor<1x12x12x4xi8>,
                                     %filter: tensor<3x3x4x16xi8>,
                                     %init: tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi32> {
  %0 = linalg.conv_2d_nhwc_hwcf {
    dilations = dense<2> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : tensor<1x12x12x4xi8>, tensor<3x3x4x16xi8>)
    outs(%init : tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// Realistic quantized conv on tensors: explicit zero-point padding, bias
// broadcast into the accumulator, then the quantized conv.
func.func @conv_2d_nhwc_hwcf_q_padded_biased(%input: tensor<1x8x8x4xi8>,
                                             %filter: tensor<3x3x4x16xi8>,
                                             %bias: tensor<16xi32>) -> tensor<1x8x8x16xi32> {
  %izp_i8 = arith.constant -3 : i8
  %padded = tensor.pad %input low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %izp_i8 : i8
  } : tensor<1x8x8x4xi8> to tensor<1x10x10x4xi8>

  %empty = tensor.empty() : tensor<1x8x8x16xi32>
  %init = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%bias : tensor<16xi32>) outs(%empty : tensor<1x8x8x16xi32>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %in : i32
  } -> tensor<1x8x8x16xi32>

  %izp = arith.constant -3 : i32
  %fzp = arith.constant 0 : i32
  %0 = linalg.conv_2d_nhwc_hwcf_q {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%padded, %filter, %izp, %fzp
        : tensor<1x10x10x4xi8>, tensor<3x3x4x16xi8>, i32, i32)
    outs(%init : tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

//===--------------------------------------------------------------------===//
// Memrefs -- note there is NO result and NO `-> type` on the op.
//===--------------------------------------------------------------------===//

func.func @conv_2d_nhwc_hwcf_memref(%input: memref<1x8x8x4xi8>,
                                    %filter: memref<3x3x4x16xi8>,
                                    %output: memref<1x6x6x16xi32>) {
  linalg.conv_2d_nhwc_hwcf {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : memref<1x8x8x4xi8>, memref<3x3x4x16xi8>)
    outs(%output : memref<1x6x6x16xi32>)
  return
}

func.func @conv_2d_nhwc_hwcf_q_memref(%input: memref<1x8x8x4xi8>,
                                      %filter: memref<3x3x4x16xi8>,
                                      %output: memref<1x6x6x16xi32>) {
  %izp = arith.constant -3 : i32
  %fzp = arith.constant 0 : i32
  linalg.conv_2d_nhwc_hwcf_q {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %izp, %fzp
        : memref<1x8x8x4xi8>, memref<3x3x4x16xi8>, i32, i32)
    outs(%output : memref<1x6x6x16xi32>)
  return
}

func.func @conv_2d_nhwc_fhwc_q_memref(%input: memref<1x8x8x4xi8>,
                                      %filter: memref<16x3x3x4xi8>,
                                      %output: memref<1x6x6x16xi32>) {
  %izp = arith.constant -3 : i32
  %fzp = arith.constant 0 : i32
  linalg.conv_2d_nhwc_fhwc_q {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %izp, %fzp
        : memref<1x8x8x4xi8>, memref<16x3x3x4xi8>, i32, i32)
    outs(%output : memref<1x6x6x16xi32>)
  return
}

// Zero-initializing the accumulator in the memref world.
func.func @conv_2d_nhwc_hwcf_memref_filled(%input: memref<1x8x8x4xi8>,
                                           %filter: memref<3x3x4x16xi8>,
                                           %output: memref<1x6x6x16xi32>) {
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%output : memref<1x6x6x16xi32>)
  linalg.conv_2d_nhwc_hwcf {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : memref<1x8x8x4xi8>, memref<3x3x4x16xi8>)
    outs(%output : memref<1x6x6x16xi32>)
  return
}
