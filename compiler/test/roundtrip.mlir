// RUN: kea-opt %s | kea-opt | FileCheck %s
//
// The canonical out-of-tree smoke test: parse, print, re-parse, re-print and
// check the result. If a custom assembly format is wrong this is what catches
// it.

// NB: the buffer-level ops moved on since the de-risking spike. `kea.mm` is
// now the KEA-1 `MATMUL` instruction exactly -- no weight operand (the weights
// are whatever the last `kea.load_w` put in `bank`), and every field named
// after its `KeaMatmul` counterpart in include/kea/isa.h. The full Level 2 op
// set round-trips in l2-roundtrip.mlir; this file keeps one of each so the
// canonical smoke test still covers both levels.

// CHECK-LABEL: func.func @mm_buffers
func.func @mm_buffers(%a: !kea.buffer<1616xi8, A>,
                      %acc: !kea.buffer<2048xi32, ACC>) {
  // CHECK: kea.mm %{{.*}}, %{{.*}} {a_addr = 0 : i64, {{.*}}} : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
  kea.mm %a, %acc {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                   m_inner = 8, m_outer = 8, acc_addr = 0,
                   acc_inner_stride = 16, acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
  return
}

// CHECK-LABEL: func.func @dma
func.func @dma(%src: !kea.buffer<1024xi8, DRAM>, %dst: !kea.buffer<1616xi8, A>) {
  // CHECK: kea.dma_load %{{.*}} -> %{{.*}} {dram_addr = 0 : i64, {{.*}}} : !kea.buffer<1024xi8, DRAM> -> !kea.buffer<1616xi8, A>
  kea.dma_load %src -> %dst {dram_addr = 0, spm_addr = 176, len0 = 128, n1 = 8,
                             n2 = 1, dram_s1 = 128, dram_s2 = 0, spm_s1 = 160,
                             spm_s2 = 0}
    : !kea.buffer<1024xi8, DRAM> -> !kea.buffer<1616xi8, A>
  return
}

// CHECK-LABEL: func.func @events
func.func @events() {
  // CHECK: kea.signal 0
  kea.signal 0
  // CHECK: kea.wait 0
  kea.wait 0
  return
}

// Level 1 conv2d: see l1-roundtrip.mlir for the full Level 1 op coverage.
// CHECK-LABEL: func.func @conv
func.func @conv(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>, %b: tensor<16xi32>)
    -> tensor<1x8x8x16xi32> {
  // CHECK: kea.conv2d %{{.*}}, %{{.*}} bias %{{.*}} {dilations = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>, strides = array<i64: 1, 1>, zero_points = #kea.zp<input = 7, weight = 0>} : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi32>) -> tensor<1x8x8x16xi32>
  %0 = kea.conv2d %in, %w bias %b {zero_points = #kea.zp<input = 7, weight = 0>,
                                    strides = array<i64: 1, 1>,
                                    pads = array<i64: 1, 1, 1, 1>,
                                    dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi32>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// CHECK-LABEL: func.func @conv_no_bias
func.func @conv_no_bias(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>)
    -> tensor<1x4x4x16xi32> {
  // CHECK: kea.conv2d %{{.*}}, %{{.*}} {dilations = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>, strides = array<i64: 2, 2>, zero_points = #kea.zp<input = 0, weight = 0>} : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x4x4x16xi32>
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
                           strides = array<i64: 2, 2>,
                           pads = array<i64: 1, 1, 1, 1>,
                           dilations = array<i64: 1, 1>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x4x4x16xi32>
  return %0 : tensor<1x4x4x16xi32>
}

// Rank-0 and DRAM buffers, plus the attribute spelling of the address space.
// CHECK-LABEL: func.func @exotic_types
// CHECK-SAME: attributes {kea.home = #kea.address_space<DRAM>}
func.func @exotic_types(%s: !kea.buffer<f32, DRAM>, %d: !kea.buffer<1x224x224x3xi8, DRAM>)
    attributes {kea.home = #kea.address_space<DRAM>} {
  return
}
