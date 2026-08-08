// RUN: kea-opt %s -tosa-to-kea -kea-fuse -kea-tile -kea-schedule -kea-alloc | FileCheck %s
// RUN: kea-opt %s -tosa-to-kea -kea-fuse -kea-tile -kea-schedule=report-schedule=true | FileCheck %s --check-prefix=REPORT
// RUN: kea-opt %s -tosa-to-kea -kea-fuse -kea-tile "-kea-schedule=mode=serial report-schedule=true" | FileCheck %s --check-prefix=SERIAL
//
// THE TWO LAYERS docs/SCHEDULING.md §8 MEASURES, and the reason they are here
// rather than in a scratch directory: the cycle counts in that document are
// reproducible from this file with
//
//     python3 compiler/test/kea-schedule-measure.py \
//             compiler/test/kea-schedule-tileloop.mlir --func <name>
//
// Both are big enough that `-kea-tile` splits them into a loop of spatial
// tiles, which is the shape double buffering exists for -- consecutive tiles
// have disjoint dependences, so tile N+1's `DMA_LD` can run under tile N's
// compute. `compiler/test/kea-schedule-e2e.mlir` covers the other end of the
// range: three dependent layers with one tile each, where the win comes from
// prefetching the *next layer's* weights instead.
//
// @pointwise_64_to_16 is deliberately DMA/compute balanced (a 1x1 projection
// has an arithmetic intensity of about 8 ops/DRAM byte, well under the int8
// ridge point of 32), so overlapping DMA against MXU is worth close to 2x.
// @conv_tile_loop is MXU bound, where the ceiling is much lower and the
// remaining stall is `-kea-tile` handing one tile the whole accumulator.

// CHECK-LABEL: func.func @pointwise_64_to_16(
// Four spatial tiles; the fourth tile's activation load is issued before the
// first tile's MATMULs are done, and the engines alternate.
// CHECK-DAG: kea.dma_load {{.*}}unit = "DMA0"
// CHECK-DAG: kea.dma_load {{.*}}unit = "DMA1"
// CHECK-DAG: kea.dma_store {{.*}}unit = "DMA0"
// CHECK-DAG: kea.dma_store {{.*}}unit = "DMA1"
// CHECK-DAG: kea.signal
// CHECK-DAG: kea.wait
// CHECK: kea.halt

// The buffers-in-flight bound is what stops the scheduler running so far ahead
// that -kea-alloc cannot fit the widened live ranges. It starts at what the
// extents allow and the capacity fixpoint walks it down, because the
// *extended* ranges need more room than the emission-order ones and because
// the loop keeps going until each space is inside its anti-fragmentation
// margin (§6.3). The exact values follow -kea-tile's tile sizes, so they are
// not pinned here.
// REPORT-LABEL: func.func @pointwise_64_to_16(
// REPORT-SAME:  buffers_in_flight = [{{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}]
// REPORT-SAME:  capacity_iters = {{[0-9]+}} : i64
// REPORT-SAME:  queue_depth = 16 : i64
// REPORT-SAME:  DMA0 = {busy = {{[1-9][0-9]*}} : i64
// REPORT-SAME:  DMA1 = {busy = {{[1-9][0-9]*}} : i64

// The baseline puts everything on one engine and overlaps nothing, which is
// what "the sequential program, executed" means on a five-queue machine.
// SERIAL-LABEL: func.func @pointwise_64_to_16(
// SERIAL-SAME:  hoisted = 0 : i64
// SERIAL-SAME:  mode = "serial"
// SERIAL-SAME:  DMA1 = {busy = 0 : i64, dram_bytes = 0 : i64, instrs = 0 : i64

// A 1x1 pointwise projection, 64 -> 16 channels over a 64x64 feature map --
// a MobileNetV2-shaped layer whose arithmetic intensity is low enough that
// DMA and MXU cost about the same per tile. -kea-tile splits it into four
// 16-row spatial tiles (SPM_A is what binds), so consecutive tiles are
// independent and this is exactly the shape double buffering exists for.
func.func @pointwise_64_to_16(%x: tensor<1x64x64x64xi8>) -> tensor<1x64x64x16xi8> {
  %w = "tosa.const"() {value = dense<2> : tensor<16x1x1x64xi8>}
      : () -> tensor<16x1x1x64xi8>
  %b = "tosa.const"() {value = dense<64> : tensor<16xi32>} : () -> tensor<16xi32>

  %acc = tosa.conv2d %x, %w, %b {
    pad = array<i64: 0, 0, 0, 0>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -5, weight_zp = 0>
  } : (tensor<1x64x64x64xi8>, tensor<16x1x1x64xi8>, tensor<16xi32>)
      -> tensor<1x64x64x16xi32>

  %q = tosa.rescale %acc {
    input_zp = 0 : i32, output_zp = -128 : i32,
    multiplier = array<i32: 1073741824>, shift = array<i8: 38>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x64x64x16xi32>) -> tensor<1x64x64x16xi8>

  %r = tosa.clamp %q {
    min_int = -128 : i64, max_int = 127 : i64,
    min_fp = 0.0 : f32, max_fp = 0.0 : f32
  } : (tensor<1x64x64x16xi8>) -> tensor<1x64x64x16xi8>

  return %r : tensor<1x64x64x16xi8>
}

// A single 3x3 stride-1 int8 convolution whose output does not fit in half of
// SPM_A, so -kea-tile splits it into a loop of spatial tiles. This is the
// shape -kea-schedule's double buffering exists for: consecutive tiles are
// independent, so tile N+1's DMA_LD can run under tile N's MATMULs.
func.func @conv_tile_loop(%x: tensor<1x64x64x16xi8>) -> tensor<1x64x64x32xi8> {
  %w = "tosa.const"() {value = dense<2> : tensor<32x3x3x16xi8>}
      : () -> tensor<32x3x3x16xi8>
  %b = "tosa.const"() {value = dense<64> : tensor<32xi32>} : () -> tensor<32xi32>

  %acc = tosa.conv2d %x, %w, %b {
    pad = array<i64: 1, 1, 1, 1>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -5, weight_zp = 0>
  } : (tensor<1x64x64x16xi8>, tensor<32x3x3x16xi8>, tensor<32xi32>)
      -> tensor<1x64x64x32xi32>

  %q = tosa.rescale %acc {
    input_zp = 0 : i32, output_zp = -128 : i32,
    multiplier = array<i32: 1073741824>, shift = array<i8: 38>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x64x64x32xi32>) -> tensor<1x64x64x32xi8>

  %r = tosa.clamp %q {
    min_int = -128 : i64, max_int = 127 : i64,
    min_fp = 0.0 : f32, max_fp = 0.0 : f32
  } : (tensor<1x64x64x32xi8>) -> tensor<1x64x64x32xi8>

  return %r : tensor<1x64x64x32xi8>
}
