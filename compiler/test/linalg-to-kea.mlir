// RUN: kea-opt %s -linalg-to-kea | FileCheck %s
//
// The secondary ingest path: the four quantized linalg named ops that
// --tosa-to-linalg-named produces (docs/TOSA_NOTES.md 11.3, 12).
//
// Structural differences from TOSA that show up in every check below:
//   * linalg convolutions have NO padding, so `pads` is always [0, 0, 0, 0]
//     and any preceding tensor.pad is left in place;
//   * `outs` is the accumulator INITIALIZER. When it is provably zero
//     (tensor.empty, or a linalg.fill of 0) it is dropped; otherwise an
//     explicit kea.add of the initializer is emitted, which is exactly what
//     linalg's `outs + A*B` means.

//===--------------------------------------------------------------------===//
// linalg.conv_2d_nhwc_fhwc_q  (FHWC == OHWI, so no weight relayout)
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func.func @conv_fhwc_q
func.func @conv_fhwc_q(%input: tensor<1x8x8x4xi8>, %filter: tensor<16x3x3x4xi8>,
                       %init: tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32> {
  // The init is an opaque block argument, so it becomes an explicit kea.add.
  // CHECK: %[[C:.*]] = kea.conv2d %arg0, %arg1 {dilations = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>, strides = array<i64: 1, 1>, zero_points = #kea.zp<input = -3, weight = 0>} : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x6x6x16xi32>
  // CHECK: kea.add %[[C]], %arg2 : (tensor<1x6x6x16xi32>, tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  %izp = arith.constant -3 : i32
  %fzp = arith.constant 0 : i32
  %0 = linalg.conv_2d_nhwc_fhwc_q {
    dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %izp, %fzp
        : tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, i32, i32)
    outs(%init : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  return %0 : tensor<1x6x6x16xi32>
}

// A zero-filled accumulator is dropped entirely.
// CHECK-LABEL: func.func @conv_fhwc_q_zero_init
func.func @conv_fhwc_q_zero_init(%input: tensor<1x8x8x4xi8>,
                                 %filter: tensor<16x3x3x4xi8>) -> tensor<1x6x6x16xi32> {
  // CHECK-NOT: kea.add
  // CHECK: kea.conv2d %arg0, %arg1 {dilations = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>, strides = array<i64: 1, 1>
  %zero = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<1x6x6x16xi32>
  %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<1x6x6x16xi32>)
      -> tensor<1x6x6x16xi32>
  %izp = arith.constant -3 : i32
  %fzp = arith.constant 0 : i32
  %0 = linalg.conv_2d_nhwc_fhwc_q {
    dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %izp, %fzp
        : tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, i32, i32)
    outs(%init : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  return %0 : tensor<1x6x6x16xi32>
}

// Strides and dilations come across from the dense<...> : tensor<2xi64> attrs.
// CHECK-LABEL: func.func @conv_fhwc_q_strided
func.func @conv_fhwc_q_strided(%input: tensor<1x16x16x4xi8>,
                               %filter: tensor<16x3x3x4xi8>) -> tensor<1x6x6x16xi32> {
  // CHECK: kea.conv2d {{.*}}dilations = array<i64: 2, 2>{{.*}}strides = array<i64: 2, 2>
  %empty = tensor.empty() : tensor<1x6x6x16xi32>
  %izp = arith.constant -3 : i32
  %fzp = arith.constant 0 : i32
  %0 = linalg.conv_2d_nhwc_fhwc_q {
    dilations = dense<2> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>
  } ins(%input, %filter, %izp, %fzp
        : tensor<1x16x16x4xi8>, tensor<16x3x3x4xi8>, i32, i32)
    outs(%empty : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  return %0 : tensor<1x6x6x16xi32>
}

// Explicit zero-point padding stays a tensor.pad in front of the conv; the
// conv's own `pads` are zero, and the input is the already-padded tensor.
// CHECK-LABEL: func.func @conv_fhwc_q_after_tensor_pad
func.func @conv_fhwc_q_after_tensor_pad(%input: tensor<1x8x8x4xi8>,
                                        %filter: tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi32> {
  // CHECK: %[[P:.*]] = tensor.pad
  // CHECK: kea.conv2d %[[P]], %arg1 {dilations = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>
  %izp_i8 = arith.constant -3 : i8
  %padded = tensor.pad %input low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %izp_i8 : i8
  } : tensor<1x8x8x4xi8> to tensor<1x10x10x4xi8>
  %empty = tensor.empty() : tensor<1x8x8x16xi32>
  %izp = arith.constant -3 : i32
  %fzp = arith.constant 0 : i32
  %0 = linalg.conv_2d_nhwc_fhwc_q {
    dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
  } ins(%padded, %filter, %izp, %fzp
        : tensor<1x10x10x4xi8>, tensor<16x3x3x4xi8>, i32, i32)
    outs(%empty : tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

//===--------------------------------------------------------------------===//
// linalg.depthwise_conv_2d_nhwc_hwcm_q  (rank-5 result, HWCM weights)
//===--------------------------------------------------------------------===//

// The rank-5 [N, H, W, C, M] result becomes a rank-4 kea.dwconv2d plus a
// kea.reshape, and the HWCM weights are normalised to [C*M, KH, KW, 1].
// CHECK-LABEL: func.func @depthwise_hwcm_q
func.func @depthwise_hwcm_q(%input: tensor<1x10x10x16xi8>,
                            %filter: tensor<3x3x16x1xi8>) -> tensor<1x8x8x16x1xi32> {
  // CHECK: %[[T:.*]] = kea.transpose %arg1 {perms = array<i64: 2, 3, 0, 1>} : (tensor<3x3x16x1xi8>) -> tensor<16x1x3x3xi8>
  // CHECK: %[[W:.*]] = kea.reshape %[[T]] {new_shape = array<i64: 16, 3, 3, 1>}
  // CHECK: %[[D:.*]] = kea.dwconv2d %arg0, %[[W]] {dilations = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>, strides = array<i64: 1, 1>, zero_points = #kea.zp<input = -7, weight = 0>} : (tensor<1x10x10x16xi8>, tensor<16x3x3x1xi8>) -> tensor<1x8x8x16xi32>
  // CHECK: kea.reshape %[[D]] {new_shape = array<i64: 1, 8, 8, 16, 1>} : (tensor<1x8x8x16xi32>) -> tensor<1x8x8x16x1xi32>
  %zero = arith.constant 0 : i32
  %empty5 = tensor.empty() : tensor<1x8x8x16x1xi32>
  %init5 = linalg.fill ins(%zero : i32) outs(%empty5 : tensor<1x8x8x16x1xi32>)
      -> tensor<1x8x8x16x1xi32>
  %izp = arith.constant -7 : i32
  %fzp = arith.constant 0 : i32
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm_q {
    dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %izp, %fzp
        : tensor<1x10x10x16xi8>, tensor<3x3x16x1xi8>, i32, i32)
    outs(%init5 : tensor<1x8x8x16x1xi32>) -> tensor<1x8x8x16x1xi32>
  return %0 : tensor<1x8x8x16x1xi32>
}

// Channel multiplier 2: rank-5 [1, 6, 6, 16, 2] collapses to 32 channels.
// CHECK-LABEL: func.func @depthwise_hwcm_q_m2
func.func @depthwise_hwcm_q_m2(%input: tensor<1x8x8x16xi8>,
                               %filter: tensor<3x3x16x2xi8>) -> tensor<1x6x6x16x2xi32> {
  // CHECK: kea.dwconv2d {{.*}} : (tensor<1x8x8x16xi8>, tensor<32x3x3x1xi8>) -> tensor<1x6x6x32xi32>
  // CHECK: kea.reshape {{.*}} {new_shape = array<i64: 1, 6, 6, 16, 2>}
  %empty = tensor.empty() : tensor<1x6x6x16x2xi32>
  %izp = arith.constant -7 : i32
  %fzp = arith.constant 0 : i32
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm_q {
    dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %izp, %fzp
        : tensor<1x8x8x16xi8>, tensor<3x3x16x2xi8>, i32, i32)
    outs(%empty : tensor<1x6x6x16x2xi32>) -> tensor<1x6x6x16x2xi32>
  return %0 : tensor<1x6x6x16x2xi32>
}

//===--------------------------------------------------------------------===//
// linalg.quantized_matmul / linalg.matmul
//===--------------------------------------------------------------------===//

// kea.matmul is strictly rank 3 (matching tosa.matmul), so the rank-2 linalg
// form is bracketed by reshapes.
// CHECK-LABEL: func.func @quantized_matmul
func.func @quantized_matmul(%a: tensor<4x8xi8>, %b: tensor<8x16xi8>) -> tensor<4x16xi32> {
  // CHECK: %[[A:.*]] = kea.reshape %arg0 {new_shape = array<i64: 1, 4, 8>} : (tensor<4x8xi8>) -> tensor<1x4x8xi8>
  // CHECK: %[[B:.*]] = kea.reshape %arg1 {new_shape = array<i64: 1, 8, 16>} : (tensor<8x16xi8>) -> tensor<1x8x16xi8>
  // CHECK: %[[M:.*]] = kea.matmul %[[A]], %[[B]] {zero_points = #kea.zp<input = -2, weight = 0>} : (tensor<1x4x8xi8>, tensor<1x8x16xi8>) -> tensor<1x4x16xi32>
  // CHECK: kea.reshape %[[M]] {new_shape = array<i64: 4, 16>} : (tensor<1x4x16xi32>) -> tensor<4x16xi32>
  %zero = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<4x16xi32>
  %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<4x16xi32>) -> tensor<4x16xi32>
  %azp = arith.constant -2 : i32
  %bzp = arith.constant 0 : i32
  %0 = linalg.quantized_matmul
      ins(%a, %b, %azp, %bzp : tensor<4x8xi8>, tensor<8x16xi8>, i32, i32)
      outs(%init : tensor<4x16xi32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

// The unquantized named op means both zero points are 0.
// CHECK-LABEL: func.func @matmul
func.func @matmul(%a: tensor<4x8xi8>, %b: tensor<8x16xi8>, %init: tensor<4x16xi32>)
    -> tensor<4x16xi32> {
  // CHECK: kea.matmul {{.*}}{zero_points = #kea.zp<input = 0, weight = 0>}
  // CHECK: kea.add
  %0 = linalg.matmul ins(%a, %b : tensor<4x8xi8>, tensor<8x16xi8>)
                     outs(%init : tensor<4x16xi32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

// A bias-broadcast accumulator initializer (what tosa.fully_connected lowers
// to) is preserved verbatim as a kea.add of the linalg.generic result. We do
// not try to reverse-engineer it back into a bias operand -- the TOSA path is
// the one that carries bias structurally.
// CHECK-LABEL: func.func @quantized_matmul_biased_init
func.func @quantized_matmul_biased_init(%a: tensor<4x8xi8>, %wt: tensor<8x16xi8>,
                                        %bias: tensor<16xi32>) -> tensor<4x16xi32> {
  // CHECK: %[[G:.*]] = linalg.generic
  // CHECK: kea.matmul
  // CHECK: kea.add {{.*}}%[[G]]
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
