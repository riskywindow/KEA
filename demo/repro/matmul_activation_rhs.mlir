// BACKEND DEFECT 2 -- `-kea-tile` assumes a `kea.matmul`'s second operand is a
// compile-time constant (it becomes a `role = "weights"` DRAM buffer sourced
// from the constant blob).  When it is an activation the pass builds a
// `kea.alloc` with a null source operand instead of reporting the limitation,
// and the verifier reports it as `null operand found`, which points at nothing
// useful.
//
// This matters beyond this demo: models/tiny_vit_int8.kgraph.json has 12
// rank-3 matmuls with two activation operands (attention QK^T and PV).
//
// Reproduce:
//   build/native/bin/keac demo/repro/matmul_activation_rhs.mlir \
//       --function mm_act -o /tmp/m.keaf
//
// Observed (kea-opt):
//   error: null operand found
//   note: see current operation: %3 = "kea.alloc"(<<NULL VALUE>>)
//         <{... name = "mm_act.1.weights", role = "weights"}>
//
// Wanted: "-kea-tile does not lower kea.matmul with a non-constant second
// operand", naming the op.

func.func @mm_act(%x: tensor<1x7x7x64xi8>) -> tensor<1x1x64xi8> {
  %r = tosa.reshape %x {new_shape = array<i64: 1, 49, 64>}
     : (tensor<1x7x7x64xi8>) -> tensor<1x49x64xi8>
  %ones = "tosa.const"() {value = dense<1> : tensor<1x1x49xi8>} : () -> tensor<1x1x49xi8>
  %acc = tosa.matmul %ones, %r {
    quantization_info = #tosa.matmul_quant<a_zp = 0, b_zp = -128>
  } : (tensor<1x1x49xi8>, tensor<1x49x64xi8>) -> tensor<1x1x64xi32>
  %y = tosa.rescale %acc {
    input_zp = 0 : i32, output_zp = -128 : i32,
    multiplier = array<i32: 1947307623>, shift = array<i8: 36>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x1x64xi32>) -> tensor<1x1x64xi8>
  return %y : tensor<1x1x64xi8>
}
