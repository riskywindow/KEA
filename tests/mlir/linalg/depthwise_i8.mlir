// linalg named depthwise convolution ops, i8 -> i32. Verified on LLVM/MLIR 20.1.6.
//
// WHICH OPS EXIST IN 20.1.6 (all confirmed by mlir-opt on this machine):
//   linalg.depthwise_conv_2d_nhwc_hwc      filter [KH,KW,C],   result rank 4 -- exists
//   linalg.depthwise_conv_2d_nhwc_hwc_q    same + zero points, result rank 4 -- exists
//   linalg.depthwise_conv_2d_nhwc_hwcm     filter [KH,KW,C,M], result rank 5 -- exists
//   linalg.depthwise_conv_2d_nhwc_hwcm_q   same + zero points, result rank 5 -- exists
//
// CRITICAL SHAPE DIFFERENCE:
//   *_hwc  variants: filter is rank 3 [KH,KW,C]   -> output rank 4 [N,H,W,C]
//                    (implicit channel multiplier 1)
//   *_hwcm variants: filter is rank 4 [KH,KW,C,M] -> output rank 5 [N,H,W,C,M]
//                    You must tensor.collapse_shape the rank-5 result back to
//                    rank 4 yourself. This is exactly what --tosa-to-linalg-named
//                    does when lowering tosa.depthwise_conv2d, which always
//                    picks the hwcm_q form even when M == 1.
//
// Zero points on the `_q` forms are trailing i32 scalar operands, same as conv.
// As with conv, linalg depthwise has NO padding attribute.

//===--------------------------------------------------------------------===//
// Tensors
//===--------------------------------------------------------------------===//

// hwc form, multiplier 1 implicitly. 8x8 -> 6x6 with a 3x3 filter, no pad.
func.func @depthwise_hwc_tensor(%input: tensor<1x8x8x16xi8>,
                                %filter: tensor<3x3x16xi8>,
                                %init: tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwc {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : tensor<1x8x8x16xi8>, tensor<3x3x16xi8>)
    outs(%init : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  return %0 : tensor<1x6x6x16xi32>
}

// hwc quantized form.
func.func @depthwise_hwc_q_tensor(%input: tensor<1x8x8x16xi8>,
                                  %filter: tensor<3x3x16xi8>,
                                  %init: tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32> {
  %izp = arith.constant -7 : i32
  %fzp = arith.constant 0 : i32
  %0 = linalg.depthwise_conv_2d_nhwc_hwc_q {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %izp, %fzp
        : tensor<1x8x8x16xi8>, tensor<3x3x16xi8>, i32, i32)
    outs(%init : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  return %0 : tensor<1x6x6x16xi32>
}

// hwcm form: rank-4 filter, RANK-5 result [N, H, W, C, M].
func.func @depthwise_hwcm_tensor(%input: tensor<1x8x8x16xi8>,
                                 %filter: tensor<3x3x16x2xi8>,
                                 %init: tensor<1x6x6x16x2xi32>) -> tensor<1x6x6x16x2xi32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : tensor<1x8x8x16xi8>, tensor<3x3x16x2xi8>)
    outs(%init : tensor<1x6x6x16x2xi32>) -> tensor<1x6x6x16x2xi32>
  return %0 : tensor<1x6x6x16x2xi32>
}

// hwcm quantized -- this is precisely the op --tosa-to-linalg-named emits for
// a quantized tosa.depthwise_conv2d, including the M=1 rank-5 result and the
// collapse_shape that follows it.
func.func @depthwise_hwcm_q_tensor(%input: tensor<1x10x10x16xi8>,
                                   %filter: tensor<3x3x16x1xi8>,
                                   %bias: tensor<16xi32>) -> tensor<1x8x8x16xi32> {
  %zero = arith.constant 0 : i32
  %empty5 = tensor.empty() : tensor<1x8x8x16x1xi32>
  %init5 = linalg.fill ins(%zero : i32) outs(%empty5 : tensor<1x8x8x16x1xi32>)
      -> tensor<1x8x8x16x1xi32>

  %izp = arith.constant -7 : i32
  %fzp = arith.constant 0 : i32
  %conv5 = linalg.depthwise_conv_2d_nhwc_hwcm_q {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %izp, %fzp
        : tensor<1x10x10x16xi8>, tensor<3x3x16x1xi8>, i32, i32)
    outs(%init5 : tensor<1x8x8x16x1xi32>) -> tensor<1x8x8x16x1xi32>

  // Rank 5 -> rank 4: fold the channel-multiplier dimension into the channels.
  %collapsed = tensor.collapse_shape %conv5 [[0], [1], [2], [3, 4]]
      : tensor<1x8x8x16x1xi32> into tensor<1x8x8x16xi32>

  // Bias add as a separate elementwise generic.
  %empty4 = tensor.empty() : tensor<1x8x8x16xi32>
  %out = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%bias, %collapsed : tensor<16xi32>, tensor<1x8x8x16xi32>)
    outs(%empty4 : tensor<1x8x8x16xi32>) {
  ^bb0(%b: i32, %c: i32, %out0: i32):
    %s = arith.addi %b, %c : i32
    linalg.yield %s : i32
  } -> tensor<1x8x8x16xi32>
  return %out : tensor<1x8x8x16xi32>
}

// Stride 2 depthwise.
func.func @depthwise_hwc_stride2_tensor(%input: tensor<1x16x16x24xi8>,
                                        %filter: tensor<3x3x24xi8>,
                                        %init: tensor<1x7x7x24xi32>) -> tensor<1x7x7x24xi32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwc {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%input, %filter : tensor<1x16x16x24xi8>, tensor<3x3x24xi8>)
    outs(%init : tensor<1x7x7x24xi32>) -> tensor<1x7x7x24xi32>
  return %0 : tensor<1x7x7x24xi32>
}

//===--------------------------------------------------------------------===//
// Memrefs
//===--------------------------------------------------------------------===//

func.func @depthwise_hwc_memref(%input: memref<1x8x8x16xi8>,
                                %filter: memref<3x3x16xi8>,
                                %output: memref<1x6x6x16xi32>) {
  linalg.depthwise_conv_2d_nhwc_hwc {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : memref<1x8x8x16xi8>, memref<3x3x16xi8>)
    outs(%output : memref<1x6x6x16xi32>)
  return
}

func.func @depthwise_hwc_q_memref(%input: memref<1x8x8x16xi8>,
                                  %filter: memref<3x3x16xi8>,
                                  %output: memref<1x6x6x16xi32>) {
  %izp = arith.constant -7 : i32
  %fzp = arith.constant 0 : i32
  linalg.depthwise_conv_2d_nhwc_hwc_q {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %izp, %fzp
        : memref<1x8x8x16xi8>, memref<3x3x16xi8>, i32, i32)
    outs(%output : memref<1x6x6x16xi32>)
  return
}

func.func @depthwise_hwcm_q_memref(%input: memref<1x8x8x16xi8>,
                                   %filter: memref<3x3x16x2xi8>,
                                   %output: memref<1x6x6x16x2xi32>) {
  %izp = arith.constant -7 : i32
  %fzp = arith.constant 0 : i32
  linalg.depthwise_conv_2d_nhwc_hwcm_q {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %izp, %fzp
        : memref<1x8x8x16xi8>, memref<3x3x16x2xi8>, i32, i32)
    outs(%output : memref<1x6x6x16x2xi32>)
  return
}
