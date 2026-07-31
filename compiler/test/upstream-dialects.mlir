// RUN: kea-opt %s | kea-opt | FileCheck %s
// RUN: kea-opt %s --pass-pipeline='builtin.module(func.func(tosa-to-linalg-named))' | FileCheck %s --check-prefix=LOWERED
//
// NOTE: tosa-to-linalg-named is an OperationPass<func::FuncOp>, so passing it
// bare as `--tosa-to-linalg-named` fails with "unable to schedule pass ... on a
// PassManager intended to run on 'builtin.module'". Nest it explicitly.
//
// kea-opt must understand tosa/linalg/tensor as well as kea, because the real
// pipeline is tosa -> linalg -> kea. The second RUN line proves the upstream
// conversion passes are registered and usable in this binary, not just the
// parsers.

// CHECK-LABEL: func.func @tosa_conv
// LOWERED-LABEL: func.func @tosa_conv
func.func @tosa_conv(%input: tensor<1x8x8x4xf32>, %weights: tensor<16x3x3x4xf32>,
                     %bias: tensor<16xf32>) -> tensor<1x8x8x16xf32> {
  // NOTE: this is the MLIR 20.1.6 spelling of tosa.conv2d -- three operands and
  // no explicit zero-point operands. LLVM 21 moves to TOSA 1.0, which adds
  // %input_zp / %weight_zp operands; this test will need updating then.
  // CHECK: tosa.conv2d
  // LOWERED: linalg.conv_2d_nhwc_fhwc
  %0 = tosa.conv2d %input, %weights, %bias {
         acc_type = f32,
         dilation = array<i64: 1, 1>,
         pad = array<i64: 1, 1, 1, 1>,
         stride = array<i64: 1, 1>}
       : (tensor<1x8x8x4xf32>, tensor<16x3x3x4xf32>, tensor<16xf32>)
       -> tensor<1x8x8x16xf32>
  return %0 : tensor<1x8x8x16xf32>
}

// CHECK-LABEL: func.func @linalg_matmul
func.func @linalg_matmul(%a: tensor<8x16xf32>, %b: tensor<16x8xf32>,
                         %c: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: linalg.matmul
  %0 = linalg.matmul ins(%a, %b : tensor<8x16xf32>, tensor<16x8xf32>)
                     outs(%c : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// A mixed module: tosa/linalg and kea coexisting is what the lowering passes
// will produce mid-pipeline.
// CHECK-LABEL: func.func @mixed
func.func @mixed(%src: memref<8x16xi8>, %dst: !kea.buffer<8x16xi8, A>,
                 %a: tensor<8x16xf32>, %b: tensor<16x8xf32>, %c: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  // CHECK: kea.dma_load
  kea.dma_load %src to %dst : memref<8x16xi8> to !kea.buffer<8x16xi8, A>
  // CHECK: linalg.matmul
  %0 = linalg.matmul ins(%a, %b : tensor<8x16xf32>, tensor<16x8xf32>)
                     outs(%c : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
