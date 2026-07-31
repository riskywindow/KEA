// linalg named matmul ops, i8 -> i32. Verified on LLVM/MLIR 20.1.6.
//
// WHICH OPS EXIST IN 20.1.6 (all confirmed by mlir-opt on this machine):
//   linalg.matmul                  -- exists
//   linalg.quantized_matmul        -- exists  (YES, it is present in 20.1.6)
//   linalg.batch_matmul            -- exists
//   linalg.quantized_batch_matmul  -- exists
//   linalg.matmul_transpose_b      -- exists (deprecated upstream, still parses)
//
// Quantized forms take zero points as trailing i32 scalar operands:
//   ins(%a, %b, %a_zp, %b_zp : tensor<MxK>, tensor<KxN>, i32, i32)
//
// linalg.matmul in 20.1.6 also accepts an OPTIONAL `indexing_maps` attribute
// that generalizes it (transposed/broadcast operands) without a separate named
// op. When absent it means plain (M,K)x(K,N)->(M,N).
//
// Both tosa.fully_connected and tosa.matmul lower into these:
//   tosa.fully_connected -> linalg.transpose (weight [OC,IC] -> [IC,OC])
//                           then linalg.quantized_matmul
//   tosa.matmul          -> linalg.quantized_batch_matmul

//===--------------------------------------------------------------------===//
// Tensors
//===--------------------------------------------------------------------===//

func.func @matmul_tensor(%a: tensor<4x8xi8>, %b: tensor<8x16xi8>,
                         %init: tensor<4x16xi32>) -> tensor<4x16xi32> {
  %0 = linalg.matmul ins(%a, %b : tensor<4x8xi8>, tensor<8x16xi8>)
                     outs(%init : tensor<4x16xi32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

func.func @quantized_matmul_tensor(%a: tensor<4x8xi8>, %b: tensor<8x16xi8>,
                                   %init: tensor<4x16xi32>) -> tensor<4x16xi32> {
  %azp = arith.constant -2 : i32
  %bzp = arith.constant 0 : i32
  %0 = linalg.quantized_matmul
      ins(%a, %b, %azp, %bzp : tensor<4x8xi8>, tensor<8x16xi8>, i32, i32)
      outs(%init : tensor<4x16xi32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

func.func @batch_matmul_tensor(%a: tensor<2x4x8xi8>, %b: tensor<2x8x16xi8>,
                               %init: tensor<2x4x16xi32>) -> tensor<2x4x16xi32> {
  %0 = linalg.batch_matmul ins(%a, %b : tensor<2x4x8xi8>, tensor<2x8x16xi8>)
                           outs(%init : tensor<2x4x16xi32>) -> tensor<2x4x16xi32>
  return %0 : tensor<2x4x16xi32>
}

func.func @quantized_batch_matmul_tensor(%a: tensor<2x4x8xi8>, %b: tensor<2x8x16xi8>,
                                         %init: tensor<2x4x16xi32>) -> tensor<2x4x16xi32> {
  %azp = arith.constant -2 : i32
  %bzp = arith.constant 1 : i32
  %0 = linalg.quantized_batch_matmul
      ins(%a, %b, %azp, %bzp : tensor<2x4x8xi8>, tensor<2x8x16xi8>, i32, i32)
      outs(%init : tensor<2x4x16xi32>) -> tensor<2x4x16xi32>
  return %0 : tensor<2x4x16xi32>
}

// Zero-initialized accumulator, the normal way to start a matmul on tensors.
func.func @quantized_matmul_filled(%a: tensor<4x8xi8>,
                                   %b: tensor<8x16xi8>) -> tensor<4x16xi32> {
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

// A fully-connected layer the way --tosa-to-linalg-named builds it from
// tosa.fully_connected: transpose the [OC, IC] weight into [IC, OC], broadcast
// the bias into the accumulator, then linalg.quantized_matmul.
func.func @fully_connected_via_quantized_matmul(%a: tensor<4x8xi8>,
                                                %w: tensor<16x8xi8>,
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

// Transposed-B matmul: B is [N, K] instead of [K, N].
//
// WARNING -- DO NOT use the `linalg.matmul indexing_maps = [...]` spelling on
// 20.1.6. It PARSES, but it does not ROUND-TRIP: the printer emits the
// indexing_maps clause AFTER the result type, while the parser requires it
// BEFORE ins(...). Re-parsing mlir-opt's own output fails with
//   "custom op 'indexing_maps' is unknown".
// This is a genuine printer/parser asymmetry in this release. Use the
// dedicated named op below instead, which round-trips correctly.
func.func @matmul_transpose_b_tensor(%a: tensor<4x8xi8>, %b: tensor<16x8xi8>,
                                     %init: tensor<4x16xi32>) -> tensor<4x16xi32> {
  %0 = linalg.matmul_transpose_b
      ins(%a, %b : tensor<4x8xi8>, tensor<16x8xi8>)
      outs(%init : tensor<4x16xi32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

//===--------------------------------------------------------------------===//
// Memrefs -- no result, no `-> type`.
//===--------------------------------------------------------------------===//

func.func @matmul_memref(%a: memref<4x8xi8>, %b: memref<8x16xi8>,
                         %c: memref<4x16xi32>) {
  linalg.matmul ins(%a, %b : memref<4x8xi8>, memref<8x16xi8>)
                outs(%c : memref<4x16xi32>)
  return
}

func.func @quantized_matmul_memref(%a: memref<4x8xi8>, %b: memref<8x16xi8>,
                                   %c: memref<4x16xi32>) {
  %azp = arith.constant -2 : i32
  %bzp = arith.constant 0 : i32
  linalg.quantized_matmul
      ins(%a, %b, %azp, %bzp : memref<4x8xi8>, memref<8x16xi8>, i32, i32)
      outs(%c : memref<4x16xi32>)
  return
}

func.func @batch_matmul_memref(%a: memref<2x4x8xi8>, %b: memref<2x8x16xi8>,
                               %c: memref<2x4x16xi32>) {
  linalg.batch_matmul ins(%a, %b : memref<2x4x8xi8>, memref<2x8x16xi8>)
                      outs(%c : memref<2x4x16xi32>)
  return
}

func.func @quantized_batch_matmul_memref(%a: memref<2x4x8xi8>, %b: memref<2x8x16xi8>,
                                         %c: memref<2x4x16xi32>) {
  %azp = arith.constant -2 : i32
  %bzp = arith.constant 1 : i32
  linalg.quantized_batch_matmul
      ins(%a, %b, %azp, %bzp : memref<2x4x8xi8>, memref<2x8x16xi8>, i32, i32)
      outs(%c : memref<2x4x16xi32>)
  return
}

// Zero-fill then accumulate, memref style.
func.func @quantized_matmul_memref_filled(%a: memref<4x8xi8>, %b: memref<8x16xi8>,
                                          %c: memref<4x16xi32>) {
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%c : memref<4x16xi32>)
  %azp = arith.constant -2 : i32
  %bzp = arith.constant 0 : i32
  linalg.quantized_matmul
      ins(%a, %b, %azp, %bzp : memref<4x8xi8>, memref<8x16xi8>, i32, i32)
      outs(%c : memref<4x16xi32>)
  return
}
