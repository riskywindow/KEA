// RUN: kea-opt %s -split-input-file -linalg-to-kea -verify-diagnostics
//
// -linalg-to-kea is deliberately narrower than -tosa-to-kea. Anything
// contraction- or pooling-shaped that it does not model is reported by name
// rather than silently left half-lowered. Plumbing ops (linalg.generic,
// linalg.fill, linalg.transpose) are left alone and are NOT diagnosed.

// The HWCF filter layout is not the one TOSA's lowering emits.
func.func @conv_hwcf(%input: tensor<1x8x8x4xi8>, %filter: tensor<3x3x4x16xi8>,
                     %init: tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32> {
  // expected-error @+1 {{-linalg-to-kea does not handle 'linalg.conv_2d_nhwc_hwcf'; the supported named ops are linalg.conv_2d_nhwc_fhwc_q, linalg.depthwise_conv_2d_nhwc_hwcm_q, linalg.quantized_matmul and linalg.matmul, all in tensor form with constant zero points}}
  %0 = linalg.conv_2d_nhwc_hwcf {
    dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : tensor<1x8x8x4xi8>, tensor<3x3x4x16xi8>)
    outs(%init : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  return %0 : tensor<1x6x6x16xi32>
}

// -----

// The rank-3 filter depthwise form (implicit multiplier 1) is not modelled;
// only the hwcm_q form that --tosa-to-linalg-named actually emits is.
func.func @depthwise_hwc(%input: tensor<1x8x8x16xi8>, %filter: tensor<3x3x16xi8>,
                         %init: tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32> {
  // expected-error @+1 {{-linalg-to-kea does not handle 'linalg.depthwise_conv_2d_nhwc_hwc'}}
  %0 = linalg.depthwise_conv_2d_nhwc_hwc {
    dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : tensor<1x8x8x16xi8>, tensor<3x3x16xi8>)
    outs(%init : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  return %0 : tensor<1x6x6x16xi32>
}

// -----

func.func @batch_matmul(%a: tensor<2x4x8xi8>, %b: tensor<2x8x16xi8>,
                        %init: tensor<2x4x16xi32>) -> tensor<2x4x16xi32> {
  // expected-error @+1 {{-linalg-to-kea does not handle 'linalg.batch_matmul'}}
  %0 = linalg.batch_matmul ins(%a, %b : tensor<2x4x8xi8>, tensor<2x8x16xi8>)
                           outs(%init : tensor<2x4x16xi32>) -> tensor<2x4x16xi32>
  return %0 : tensor<2x4x16xi32>
}

// -----

// Memref (destination-passing) form: KEA Level 1 is value semantic, so the
// pattern declines and the leftover check reports it.
func.func @quantized_matmul_memref(%a: memref<4x8xi8>, %b: memref<8x16xi8>,
                                   %c: memref<4x16xi32>) {
  %azp = arith.constant -2 : i32
  %bzp = arith.constant 0 : i32
  // expected-error @+1 {{-linalg-to-kea does not handle 'linalg.quantized_matmul'}}
  linalg.quantized_matmul
      ins(%a, %b, %azp, %bzp : memref<4x8xi8>, memref<8x16xi8>, i32, i32)
      outs(%c : memref<4x16xi32>)
  return
}

// -----

// Zero points must be constants to become a #kea.zp attribute.
func.func @quantized_matmul_dynamic_zp(%a: tensor<4x8xi8>, %b: tensor<8x16xi8>,
                                       %azp: i32, %bzp: i32) -> tensor<4x16xi32> {
  %empty = tensor.empty() : tensor<4x16xi32>
  // expected-error @+1 {{-linalg-to-kea does not handle 'linalg.quantized_matmul'}}
  %0 = linalg.quantized_matmul
      ins(%a, %b, %azp, %bzp : tensor<4x8xi8>, tensor<8x16xi8>, i32, i32)
      outs(%empty : tensor<4x16xi32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

// -----

// linalg.generic / linalg.fill / linalg.transpose are plumbing and are left
// alone without complaint: this function converts cleanly.
func.func @plumbing_is_left_alone(%a: tensor<4x8xi8>, %w: tensor<16x8xi8>,
                                  %bias: tensor<16xi32>) -> tensor<4x16xi32> {
  %wt_empty = tensor.empty() : tensor<8x16xi8>
  %wt = linalg.transpose ins(%w : tensor<16x8xi8>)
                         outs(%wt_empty : tensor<8x16xi8>) permutation = [1, 0]
  %empty = tensor.empty() : tensor<4x16xi32>
  %init = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%bias : tensor<16xi32>) outs(%empty : tensor<4x16xi32>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %in : i32
  } -> tensor<4x16xi32>
  %azp = arith.constant -2 : i32
  %bzp = arith.constant 0 : i32
  %0 = linalg.quantized_matmul
      ins(%a, %wt, %azp, %bzp : tensor<4x8xi8>, tensor<8x16xi8>, i32, i32)
      outs(%init : tensor<4x16xi32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}
