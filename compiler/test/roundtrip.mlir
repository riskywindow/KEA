// RUN: kea-opt %s | kea-opt | FileCheck %s
//
// The canonical out-of-tree smoke test: parse, print, re-parse, re-print and
// check the result. If a custom assembly format is wrong this is what catches
// it.

// CHECK-LABEL: func.func @matmul_buffers
func.func @matmul_buffers(%a: !kea.buffer<8x16xi8, A>,
                          %w: !kea.buffer<16x8xi8, W>,
                          %acc: !kea.buffer<8x8xi32, ACC>) {
  // CHECK: kea.matmul %{{.*}}, %{{.*}}, %{{.*}} : !kea.buffer<8x16xi8, A>, !kea.buffer<16x8xi8, W>, !kea.buffer<8x8xi32, ACC>
  kea.matmul %a, %w, %acc : !kea.buffer<8x16xi8, A>, !kea.buffer<16x8xi8, W>, !kea.buffer<8x8xi32, ACC>

  // CHECK: kea.matmul %{{.*}}, %{{.*}}, %{{.*}} {tile = #kea.tile_config<16, 16, 32>}
  kea.matmul %a, %w, %acc {tile = #kea.tile_config<16, 16, 32>} : !kea.buffer<8x16xi8, A>, !kea.buffer<16x8xi8, W>, !kea.buffer<8x8xi32, ACC>
  return
}

// CHECK-LABEL: func.func @matmul_transposed
func.func @matmul_transposed(%a: !kea.buffer<8x16xi8, A>,
                             %w: !kea.buffer<8x16xi8, W>,
                             %acc: !kea.buffer<8x8xi32, ACC>) {
  // CHECK: kea.matmul %{{.*}}, %{{.*}}, %{{.*}} {transpose_rhs}
  kea.matmul %a, %w, %acc {transpose_rhs} : !kea.buffer<8x16xi8, A>, !kea.buffer<8x16xi8, W>, !kea.buffer<8x8xi32, ACC>
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

// CHECK-LABEL: func.func @conv
func.func @conv(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>, %b: tensor<16xi32>)
    -> tensor<1x8x8x16xi32> {
  // CHECK: kea.conv2d %{{.*}}, %{{.*}}, %{{.*}} {pads = array<i64: 1, 1, 1, 1>, scale = 1.250000e-01 : f64, strides = array<i64: 1, 1>, zero_point = 7 : i32} : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi32>) -> tensor<1x8x8x16xi32>
  %0 = kea.conv2d %in, %w, %b {strides = array<i64: 1, 1>,
                               pads = array<i64: 1, 1, 1, 1>,
                               scale = 0.125 : f64,
                               zero_point = 7 : i32}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi32>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// CHECK-LABEL: func.func @conv_no_bias
func.func @conv_no_bias(%in: tensor<1x8x8x4xi8>, %w: tensor<16x3x3x4xi8>)
    -> tensor<1x8x8x16xi32> {
  // Defaulted attributes (scale/zero_point) are elided on print.
  // CHECK: kea.conv2d %{{.*}}, %{{.*}} {pads = array<i64: 0, 0, 0, 0>, strides = array<i64: 2, 2>} : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi32>
  %0 = kea.conv2d %in, %w {strides = array<i64: 2, 2>, pads = array<i64: 0, 0, 0, 0>}
      : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// Rank-0 and DRAM buffers, plus the attribute spelling of the address space.
// CHECK-LABEL: func.func @exotic_types
// CHECK-SAME: attributes {kea.home = #kea.address_space<DRAM>}
func.func @exotic_types(%s: !kea.buffer<f32, DRAM>, %d: !kea.buffer<1x224x224x3xi8, DRAM>)
    attributes {kea.home = #kea.address_space<DRAM>} {
  return
}
