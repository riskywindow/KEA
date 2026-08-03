// RUN: kea-opt %s -split-input-file -tosa-to-kea -verify-diagnostics
//
// What -tosa-to-kea deliberately refuses, and the diagnostic it gives.
//
// Two flavours:
//   * a hard `emitError` for TOSA that is representable but has no KEA
//     equivalent, so the user learns WHY;
//   * the dialect-conversion target's own "failed to legalize operation" for
//     TOSA ops that are simply out of Level 1's scope.

// KEA_VQUANT's multiplier is a normalized Q31 value (docs/ISA.md 10.1), so
// TOSA's 16-bit multiplier path has nowhere to go.
func.func @rescale_scale16(%a: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8> {
  // expected-error @+2 {{scale32 = false (the 16-bit multiplier path) has no KEA Level 1 equivalent}}
  // expected-error @+1 {{failed to legalize operation 'tosa.rescale'}}
  %0 = tosa.rescale %a {
    input_zp = 0 : i32, output_zp = 0 : i32,
    multiplier = array<i32: 16384>, shift = array<i8: 15>,
    scale32 = false, double_round = false, per_channel = false
  } : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// -----

func.func @rescale_unsigned(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {
  // expected-error @+2 {{unsigned rescale operands are not modelled at KEA Level 1}}
  // expected-error @+1 {{failed to legalize operation 'tosa.rescale'}}
  %0 = tosa.rescale %a {
    input_zp = 128 : i32, output_zp = 0 : i32,
    multiplier = array<i32: 1073741824>, shift = array<i8: 30>,
    scale32 = true, double_round = true, per_channel = false,
    input_unsigned = true, output_unsigned = false
  } : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// -----

// In 20.1.6 the permutation is an operand. A non-constant one cannot become the
// kea.transpose 'perms' attribute.
func.func @transpose_dynamic_perms(%a: tensor<1x8x8x16xi8>, %perm: tensor<4xi32>)
    -> tensor<1x16x8x8xi8> {
  // expected-error @+2 {{tosa.transpose's permutation operand must be constant}}
  // expected-error @+1 {{failed to legalize operation 'tosa.transpose'}}
  %0 = tosa.transpose %a, %perm
      : (tensor<1x8x8x16xi8>, tensor<4xi32>) -> tensor<1x16x8x8xi8>
  return %0 : tensor<1x16x8x8xi8>
}

// -----

// tosa.sub has no Level 1 op. Out of scope, reported by the conversion target.
func.func @sub(%a: tensor<1x4x4x8xi32>, %b: tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32> {
  // expected-error @+1 {{failed to legalize operation 'tosa.sub'}}
  %0 = tosa.sub %a, %b : (tensor<1x4x4x8xi32>, tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// -----

func.func @slice(%a: tensor<1x8x8x16xi8>) -> tensor<1x4x4x8xi8> {
  // expected-error @+1 {{failed to legalize operation 'tosa.slice'}}
  %0 = tosa.slice %a {
    start = array<i64: 0, 2, 2, 0>, size = array<i64: 1, 4, 4, 8>
  } : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x8xi8>
  return %0 : tensor<1x4x4x8xi8>
}

// -----

func.func @concat(%a: tensor<1x8x8x8xi8>, %b: tensor<1x8x8x8xi8>) -> tensor<1x8x8x16xi8> {
  // expected-error @+1 {{failed to legalize operation 'tosa.concat'}}
  %0 = tosa.concat %a, %b {axis = 3 : i32}
      : (tensor<1x8x8x8xi8>, tensor<1x8x8x8xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// -----

// tosa.pad already uses !tosa.shape in 20.1.6. Explicit padding is expressed by
// the conv's own `pads` at Level 1, so a standalone pad has no Level 1 form.
func.func @pad(%a: tensor<1x8x8x16xi8>) -> tensor<1x10x10x16xi8> {
  // The shape operand is reported first, since the conversion walks in order.
  // expected-error @+1 {{failed to legalize operation 'tosa.const_shape'}}
  %padding = tosa.const_shape {value = dense<[0, 0, 1, 1, 1, 1, 0, 0]> : tensor<8xindex>}
      : () -> !tosa.shape<8>
  %0 = tosa.pad %a, %padding {
    quantization_info = #tosa.pad_quant<input_zp = -5>
  } : (tensor<1x8x8x16xi8>, !tosa.shape<8>) -> tensor<1x10x10x16xi8>
  return %0 : tensor<1x10x10x16xi8>
}

// -----

// tosa.cast is a bare element-type change with no scale and no zero point.
// Refusing it is deliberate: at Level 1 every type change is a kea.rescale.
func.func @cast(%a: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32> {
  // expected-error @+1 {{failed to legalize operation 'tosa.cast'}}
  %0 = tosa.cast %a : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}
