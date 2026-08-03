// RUN: kea-opt %s -split-input-file -verify-diagnostics
//
// Exercises the hand-written verifiers. `-verify-diagnostics` makes kea-opt
// exit 0 iff every emitted diagnostic was matched by an `expected-error`, so
// no `not` tool is required.

func.func @bad_channels(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x8xi8>)
    -> tensor<1x8x8x16xi32> {
  // expected-error @+1 {{input channel count 4 does not match weight channel count 8}}
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
                           strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                           dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x8xi8>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// -----

func.func @bad_strides(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>)
    -> tensor<1x8x8x16xi32> {
  // expected-error @+1 {{expects exactly 2 strides}}
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
                           strides = array<i64: 1, 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                           dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// -----

func.func @bad_mm_k(%a: !kea.buffer<8x16xi8, A>,
                    %w: !kea.buffer<32x8xi8, W>,
                    %acc: !kea.buffer<8x8xi32, ACC>) {
  // expected-error @+1 {{contraction dimension mismatch: lhs K=16 but rhs K=32}}
  kea.mm %a, %w, %acc : !kea.buffer<8x16xi8, A>, !kea.buffer<32x8xi8, W>, !kea.buffer<8x8xi32, ACC>
  return
}

// -----

func.func @mm_wrong_space(%a: !kea.buffer<8x16xi8, A>,
                          %w: !kea.buffer<16x8xi8, A>,
                          %acc: !kea.buffer<8x8xi32, ACC>) {
  // The address space is enforced by the ODS type constraint, not the verifier.
  // expected-error @+1 {{operand #1 must be buffer in the weight scratchpad}}
  kea.mm %a, %w, %acc : !kea.buffer<8x16xi8, A>, !kea.buffer<16x8xi8, A>, !kea.buffer<8x8xi32, ACC>
  return
}

// -----

func.func @dma_to_dram(%src: memref<8x16xi8>, %dst: !kea.buffer<8x16xi8, DRAM>) {
  // expected-error @+1 {{destination must be an on-chip buffer}}
  kea.dma_load %src to %dst : memref<8x16xi8> to !kea.buffer<8x16xi8, DRAM>
  return
}

// -----

// expected-error @+1 {{ACC buffers must have a 32-bit element type}}
func.func @bad_acc_width(%acc: !kea.buffer<8x8xi8, ACC>) {
  return
}

// -----

// expected-error @+1 {{kea.buffer dimensions must be positive}}
func.func @bad_dim(%b: !kea.buffer<0x8xi8, A>) {
  return
}

// -----

func.func @negative_event() {
  // NOTE: `expected-error {{...}}` is a literal *substring* match, not a regex.
  // expected-error @+1 {{failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  kea.signal -1
  return
}
