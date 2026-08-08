// REGRESSION -- an activation-RHS `kea.matmul` must be *refused*, not crashed
// on.
//
// `-kea-tile` cannot lower a matmul whose second operand is an activation: the
// MXU is weight stationary and the second operand becomes a DRAM weight blob.
// That limitation stands.  What was wrong was the diagnostic -- the pass built
// a `kea.alloc` with a null source and the verifier said:
//
//   error: null operand found
//   note: see current operation: %3 = "kea.alloc"(<<NULL VALUE>>)
//
// which named neither the op nor the reason.  It now says so directly, naming
// `kea.matmul` and the weight-stationary constraint.  This file asserts the
// good diagnostic, not the crash.
//
// Still relevant beyond this demo: models/tiny_vit_int8.kgraph.json has 12
// rank-3 matmuls with two activation operands (attention QK^T and PV).
//
//   build/native/bin/keac demo/regress/matmul_activation_rhs.mlir \
//       --function mm_act -o /tmp/m.keaf

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
