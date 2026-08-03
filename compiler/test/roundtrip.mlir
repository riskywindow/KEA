// RUN: kea-opt %s | kea-opt | FileCheck %s
//
// The canonical out-of-tree smoke test: parse, print, re-parse, re-print and
// check the result. If a custom assembly format is wrong this is what catches
// it.

// NB: the buffer-level matrix multiply is `kea.mm`, not `kea.matmul`. The
// de-risking spike called it `kea.matmul`, but ADR-0002 gives that name to the
// Level 1 tensor op and names this one `kea.mm`; the definition is otherwise
// unchanged and now lives in KeaMachineOps.td.

// CHECK-LABEL: func.func @mm_buffers
func.func @mm_buffers(%a: !kea.buffer<8x16xi8, A>,
                      %w: !kea.buffer<16x8xi8, W>,
                      %acc: !kea.buffer<8x8xi32, ACC>) {
  // CHECK: kea.mm %{{.*}}, %{{.*}}, %{{.*}} : !kea.buffer<8x16xi8, A>, !kea.buffer<16x8xi8, W>, !kea.buffer<8x8xi32, ACC>
  kea.mm %a, %w, %acc : !kea.buffer<8x16xi8, A>, !kea.buffer<16x8xi8, W>, !kea.buffer<8x8xi32, ACC>

  // CHECK: kea.mm %{{.*}}, %{{.*}}, %{{.*}} {tile = #kea.tile_config<16, 16, 32>}
  kea.mm %a, %w, %acc {tile = #kea.tile_config<16, 16, 32>} : !kea.buffer<8x16xi8, A>, !kea.buffer<16x8xi8, W>, !kea.buffer<8x8xi32, ACC>
  return
}

// CHECK-LABEL: func.func @mm_transposed
func.func @mm_transposed(%a: !kea.buffer<8x16xi8, A>,
                         %w: !kea.buffer<8x16xi8, W>,
                         %acc: !kea.buffer<8x8xi32, ACC>) {
  // CHECK: kea.mm %{{.*}}, %{{.*}}, %{{.*}} {transpose_rhs}
  kea.mm %a, %w, %acc {transpose_rhs} : !kea.buffer<8x16xi8, A>, !kea.buffer<8x16xi8, W>, !kea.buffer<8x8xi32, ACC>
  return
}

// CHECK-LABEL: func.func @dma
func.func @dma(%src: memref<8x16xi8>, %dst: !kea.buffer<8x16xi8, A>) {
  // CHECK: kea.dma_load %{{.*}} to %{{.*}} {event = 3 : i64} : memref<8x16xi8> to !kea.buffer<8x16xi8, A>
  kea.dma_load %src to %dst {event = 3 : i64} : memref<8x16xi8> to !kea.buffer<8x16xi8, A>
  // CHECK: kea.dma_load %{{.*}} to %{{.*}} : memref<8x16xi8> to !kea.buffer<8x16xi8, A>
  kea.dma_load %src to %dst : memref<8x16xi8> to !kea.buffer<8x16xi8, A>
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
