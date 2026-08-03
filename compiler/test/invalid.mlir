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

// The Level 2 verifiers get a file of their own (l2-invalid.mlir), which walks
// every ISA.md §11.1 alignment rule. One case here keeps this file covering
// both levels.
func.func @mm_acc_misaligned(%a: !kea.buffer<1616xi8, A>,
                             %acc: !kea.buffer<2048xi32, ACC>) {
  // ACC is addressed in int32 WORDS and every address is a multiple of 16.
  // expected-error @+1 {{acc_addr (8) must be a multiple of 16 words}}
  kea.mm %a, %acc {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                   m_inner = 8, m_outer = 8, acc_addr = 8,
                   acc_inner_stride = 16, acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
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
