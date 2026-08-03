// RUN: kea-opt %s -tosa-to-kea -kea-fuse -kea-tile -kea-alloc | FileCheck %s
// RUN: kea-opt %s -tosa-to-kea -kea-fuse -kea-tile -kea-alloc | kea-opt -kea-alloc=verify-only=true | FileCheck %s --check-prefix=REVERIFY
//
// END-TO-END: quantized TOSA -> a program in which every scratchpad and DRAM
// address is a compile-time constant. There is no allocator at run time.
//
// The source below is a verbatim copy of tests/mlir/tosa/mobilenet_block.mlir,
// which is read-only reference material and lives outside compiler/test, where
// run_tests.sh looks. compiler/test/mobilenet-e2e.mlir carries the same source
// and asserts the Level 1 fusion and the tiling; this file picks up where that
// one stops and asserts the memory plan.
//
// The second RUN line re-runs the overlap verifier over the finished program
// from scratch, so the pipeline proves its own output free of aliasing.

// MobileNetV2 inverted-residual block, fully quantized int8, in TOSA.
// Verified to round-trip through mlir-opt on LLVM/MLIR 20.1.6.
//
// This is THE integration test for the compiler. Structure:
//
//   x (i8, 1x8x8x4)
//     |-------------------------------.            (identity / residual path)
//     v                               |
//   1x1 conv expand   4 -> 24 ch       |
//   rescale i32 -> i8 (per-channel)    |
//   clamp ReLU6                        |
//     v                               |
//   3x3 depthwise, stride 1, pad 1     |
//   rescale i32 -> i8 (per-channel)    |
//   clamp ReLU6                        |
//     v                               |
//   1x1 conv project  24 -> 4 ch       |
//   rescale i32 -> i8 (per-channel)    |
//     v                               |
//   +<------------------------------- '
//   (quantized add: rescale both sides to a common i32 domain, add, rescale)
//     v
//   y (i8, 1x8x8x4)
//
// Quantization conventions used here:
//   * Activation zero point after ReLU6 is -128 (the "unsigned-like" encoding),
//     so clamp min_int = -128 clamps exactly at the quantized zero.
//   * Block input/output zero point is -5, a general signed-affine activation.
//   * Weight zero point is 0 throughout (symmetric weight quantization), which
//     is what per-channel PTQ produces and what most NPUs require.
//   * Weight rescales are PER-CHANNEL, matching per-channel weight scales.
//   * The projection output is NOT followed by a ReLU -- that is the defining
//     "linear bottleneck" property of MobileNetV2.

//===--------------------------------------------------------------------===//
// The memory plan for the inverted residual block
//===--------------------------------------------------------------------===//
//
// Everything on this line is decided at compile time and nothing is decided
// again later. Reading the occupancy dictionary, per space:
//
//   SPM_A   peak 5,280 of 262,144 bytes   (2.0% of capacity)
//   SPM_W   peak   896 of 262,144 bytes   (0.3%)
//   ACC     peak 2,048 of  32,768 words   (6.3%)
//
// `unpacked` is what an allocator that never shared storage would have needed,
// so packing turns 11,248 bytes of SPM_A into 5,280 and 5,120 ACC words into
// 2,048 -- a 2.1x and 2.5x reduction. `fragmentation` is `peak - maxlive`, and
// it is ZERO in every space: on this block the greedy packer is provably
// optimal, because no allocator can beat the largest amount of simultaneously
// live data.
//
// The tiles are small relative to capacity because -kea-tile only ever spends
// half of each scratchpad (spm-reserve-factor, default 2) and this block's
// tensors are 8x8; the ratios, not the absolute numbers, are the result.

// The attributes print in alphabetical order, so kea.dram_layout comes first:
// the DRAM arena the runtime must hand the device, and the only allocation that
// happens anywhere in the system -- one contiguous block of 4,352 bytes.
// CHECK-LABEL: func.func @mobilenet_v2_inverted_residual(
// CHECK-SAME:  kea.dram_layout = {alignment = 64 : i64, const_bytes = 2292 : i64, const_offset = 0 : i64, io_bytes = 512 : i64, io_offset = 2304 : i64, scratch_bytes = 1536 : i64, scratch_offset = 2816 : i64, total_bytes = 4352 : i64}
// CHECK-SAME:  acc = {buffers = 3 : i64, capacity = 32768 : i64, fragmentation = 0 : i64, maxlive = 2048 : i64, peak = 2048 : i64, unpacked = 5120 : i64}
// CHECK-SAME:  dram_scratch = {buffers = 2 : i64, capacity = 1536 : i64, fragmentation = 0 : i64, maxlive = 1536 : i64, peak = 1536 : i64, unpacked = 3072 : i64}
// CHECK-SAME:  spm_a = {buffers = 7 : i64, capacity = 262144 : i64, fragmentation = 0 : i64, maxlive = 5280 : i64, peak = 5280 : i64, unpacked = 11248 : i64}
// CHECK-SAME:  spm_w = {buffers = 7 : i64, capacity = 262144 : i64, fragmentation = 0 : i64, maxlive = 896 : i64, peak = 896 : i64, unpacked = 2292 : i64}

// The block input and output are in the I/O region; the host binds them by name
// and they are never packed against anything.
// CHECK: kea.alloc {kea.addr = 2304 : i64, name = "{{.*}}.input0", role = "input"}

// THE INTER-LAYER ACTIVATIONS SHARE ONE BUFFER. Layer 0 writes @0.out and
// layer 1 reads it; layer 1 writes @1.out and layer 2 reads it. The two ranges
// are disjoint, so both land at 2816. On a 50-layer network this is the
// difference between an arena of a few buffers and an arena of fifty.
// CHECK: kea.alloc {kea.addr = 2816 : i64, name = "{{.*}}.0.out", role = "activation"}
// CHECK: kea.alloc {kea.addr = 2816 : i64, name = "{{.*}}.1.out", role = "activation"}

// Weights and quantization parameters are in the const region from offset 0, in
// the order -kea-emit must write the CONST blob.
// CHECK: kea.alloc {{.*}}{kea.addr = 0 : i64, {{.*}}role = "weights"}

// Every ACC base is a multiple of 16 WORDS. The two layer-0/layer-1 ACC regions
// have disjoint live ranges, so both sit at 0.
// CHECK: kea.alloc {kea.addr = 0 : i64, {{.*}}.0.acc"{{.*}} : !kea.buffer<2048xi32, ACC>
// CHECK: kea.alloc {kea.addr = 0 : i64, {{.*}}.1.acc"{{.*}} : !kea.buffer<2048xi32, ACC>

// REVERIFY-LABEL: func.func @mobilenet_v2_inverted_residual(
// REVERIFY: kea.addr

func.func @mobilenet_v2_inverted_residual(%x: tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi8> {

  //===------------------------------------------------------------------===//
  // 1. Expand: 1x1 pointwise convolution, 4 -> 24 channels.
  //===------------------------------------------------------------------===//
  %w_exp = "tosa.const"() {value = dense<2> : tensor<24x1x1x4xi8>}
      : () -> tensor<24x1x1x4xi8>
  %b_exp = "tosa.const"() {value = dense<128> : tensor<24xi32>}
      : () -> tensor<24xi32>

  %exp_acc = tosa.conv2d %x, %w_exp, %b_exp {
    pad = array<i64: 0, 0, 0, 0>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -5, weight_zp = 0>
  } : (tensor<1x8x8x4xi8>, tensor<24x1x1x4xi8>, tensor<24xi32>)
      -> tensor<1x8x8x24xi32>

  // Per-channel requantization back to i8. 24 multipliers / 24 shifts,
  // one per output channel (the last dimension).
  %exp_q = tosa.rescale %exp_acc {
    input_zp = 0 : i32,
    output_zp = -128 : i32,
    multiplier = array<i32: 1073741824, 1181116006, 1288490189, 1395864371,
                            1503238553, 1610612736, 1717986918, 1825361101,
                            1932735283, 2040109466, 1073741824, 1181116006,
                            1288490189, 1395864371, 1503238553, 1610612736,
                            1717986918, 1825361101, 1932735283, 2040109466,
                            1073741824, 1181116006, 1288490189, 1395864371>,
    shift = array<i8: 36, 36, 36, 36, 37, 37, 37, 37,
                       36, 36, 36, 36, 37, 37, 37, 37,
                       36, 36, 36, 36, 37, 37, 37, 37>,
    scale32 = true,
    double_round = true,
    per_channel = true
  } : (tensor<1x8x8x24xi32>) -> tensor<1x8x8x24xi8>

  // ReLU6 in the i8 domain (zero point -128).
  %exp_relu = tosa.clamp %exp_q {
    min_int = -128 : i64,
    max_int = 127 : i64,
    min_fp = 0.0 : f32,
    max_fp = 6.0 : f32
  } : (tensor<1x8x8x24xi8>) -> tensor<1x8x8x24xi8>

  //===------------------------------------------------------------------===//
  // 2. Depthwise: 3x3, stride 1, pad 1, channel multiplier 1, 24 channels.
  //     NOTE the HWCM weight layout [KH, KW, C, M] -- different from conv2d.
  //===------------------------------------------------------------------===//
  %w_dw = "tosa.const"() {value = dense<1> : tensor<3x3x24x1xi8>}
      : () -> tensor<3x3x24x1xi8>
  %b_dw = "tosa.const"() {value = dense<64> : tensor<24xi32>}
      : () -> tensor<24xi32>

  %dw_acc = tosa.depthwise_conv2d %exp_relu, %w_dw, %b_dw {
    pad = array<i64: 1, 1, 1, 1>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x8x8x24xi8>, tensor<3x3x24x1xi8>, tensor<24xi32>)
      -> tensor<1x8x8x24xi32>

  %dw_q = tosa.rescale %dw_acc {
    input_zp = 0 : i32,
    output_zp = -128 : i32,
    multiplier = array<i32: 1503238553, 1610612736, 1717986918, 1825361101,
                            1932735283, 2040109466, 1073741824, 1181116006,
                            1288490189, 1395864371, 1503238553, 1610612736,
                            1717986918, 1825361101, 1932735283, 2040109466,
                            1073741824, 1181116006, 1288490189, 1395864371,
                            1503238553, 1610612736, 1717986918, 1825361101>,
    shift = array<i8: 37, 37, 37, 37, 36, 36, 36, 36,
                       37, 37, 37, 37, 36, 36, 36, 36,
                       37, 37, 37, 37, 36, 36, 36, 36>,
    scale32 = true,
    double_round = true,
    per_channel = true
  } : (tensor<1x8x8x24xi32>) -> tensor<1x8x8x24xi8>

  %dw_relu = tosa.clamp %dw_q {
    min_int = -128 : i64,
    max_int = 127 : i64,
    min_fp = 0.0 : f32,
    max_fp = 6.0 : f32
  } : (tensor<1x8x8x24xi8>) -> tensor<1x8x8x24xi8>

  //===------------------------------------------------------------------===//
  // 3. Project: 1x1 pointwise convolution, 24 -> 4 channels.
  //     Linear bottleneck -- NO activation after this rescale.
  //===------------------------------------------------------------------===//
  %w_proj = "tosa.const"() {value = dense<3> : tensor<4x1x1x24xi8>}
      : () -> tensor<4x1x1x24xi8>
  %b_proj = "tosa.const"() {value = dense<-32> : tensor<4xi32>}
      : () -> tensor<4xi32>

  %proj_acc = tosa.conv2d %dw_relu, %w_proj, %b_proj {
    pad = array<i64: 0, 0, 0, 0>,
    stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x8x8x24xi8>, tensor<4x1x1x24xi8>, tensor<4xi32>)
      -> tensor<1x8x8x4xi32>

  %proj_q = tosa.rescale %proj_acc {
    input_zp = 0 : i32,
    output_zp = -5 : i32,
    multiplier = array<i32: 1288490189, 1395864371, 1503238553, 1610612736>,
    shift = array<i8: 37, 37, 38, 38>,
    scale32 = true,
    double_round = true,
    per_channel = true
  } : (tensor<1x8x8x4xi32>) -> tensor<1x8x8x4xi8>

  //===------------------------------------------------------------------===//
  // 4. Residual add. TOSA has no quantized-add attribute, so both operands
  //    must be rescaled into a shared i32 domain, added, then rescaled back.
  //===------------------------------------------------------------------===//

  // Identity path: block input, zero point -5, into the common i32 domain.
  %id_i32 = tosa.rescale %x {
    input_zp = -5 : i32,
    output_zp = 0 : i32,
    multiplier = array<i32: 1073741824>,
    shift = array<i8: 10>,
    scale32 = true,
    double_round = true,
    per_channel = false
  } : (tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi32>

  // Residual path: projection output, zero point -5, same common i32 domain.
  %res_i32 = tosa.rescale %proj_q {
    input_zp = -5 : i32,
    output_zp = 0 : i32,
    multiplier = array<i32: 1610612736>,
    shift = array<i8: 11>,
    scale32 = true,
    double_round = true,
    per_channel = false
  } : (tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi32>

  %sum_i32 = tosa.add %id_i32, %res_i32
      : (tensor<1x8x8x4xi32>, tensor<1x8x8x4xi32>) -> tensor<1x8x8x4xi32>

  // Back to i8 at the block output scale / zero point.
  %y = tosa.rescale %sum_i32 {
    input_zp = 0 : i32,
    output_zp = -5 : i32,
    multiplier = array<i32: 1503238553>,
    shift = array<i8: 40>,
    scale32 = true,
    double_round = true,
    per_channel = false
  } : (tensor<1x8x8x4xi32>) -> tensor<1x8x8x4xi8>

  return %y : tensor<1x8x8x4xi8>
}

//===--------------------------------------------------------------------===//
// Stride-2 variant: spatial size changes, so there is NO residual connection.
// This is the other half of the MobileNetV2 block vocabulary.
//===--------------------------------------------------------------------===//
//===--------------------------------------------------------------------===//
// The stride-2 variant: no residual, so one fewer SPM_A tile to place
//===--------------------------------------------------------------------===//
//
//   SPM_A   peak 3,136 of 262,144 bytes
//   SPM_W   peak   896 of 262,144 bytes
//   ACC     peak 2,048 of  32,768 words
//
// Again zero fragmentation in every space, and again every peak is far inside
// capacity -- which is the assertion that matters: a program that did not fit
// would not have compiled at all, so these CHECK lines are a regression guard
// on the packing quality rather than on legality.

// CHECK-LABEL: func.func @mobilenet_v2_inverted_residual_stride2(
// CHECK-SAME:  acc = {buffers = 3 : i64, capacity = 32768 : i64, fragmentation = 0 : i64, maxlive = 2048 : i64, peak = 2048 : i64, unpacked = 2816 : i64}
// CHECK-SAME:  spm_a = {buffers = 6 : i64, capacity = 262144 : i64, fragmentation = 0 : i64, maxlive = 3136 : i64, peak = 3136 : i64, unpacked = 6144 : i64}
// CHECK-SAME:  spm_w = {buffers = 6 : i64, capacity = 262144 : i64, fragmentation = 0 : i64, maxlive = 896 : i64, peak = 896 : i64, unpacked = 2272 : i64}

// REVERIFY-LABEL: func.func @mobilenet_v2_inverted_residual_stride2(
// REVERIFY: kea.addr

func.func @mobilenet_v2_inverted_residual_stride2(%x: tensor<1x8x8x4xi8>)
    -> tensor<1x4x4x8xi8> {
  %w_exp = "tosa.const"() {value = dense<2> : tensor<24x1x1x4xi8>}
      : () -> tensor<24x1x1x4xi8>
  %b_exp = "tosa.const"() {value = dense<128> : tensor<24xi32>} : () -> tensor<24xi32>
  %exp_acc = tosa.conv2d %x, %w_exp, %b_exp {
    pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -5, weight_zp = 0>
  } : (tensor<1x8x8x4xi8>, tensor<24x1x1x4xi8>, tensor<24xi32>)
      -> tensor<1x8x8x24xi32>
  %exp_q = tosa.rescale %exp_acc {
    input_zp = 0 : i32, output_zp = -128 : i32,
    multiplier = array<i32: 1073741824>, shift = array<i8: 36>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x8x8x24xi32>) -> tensor<1x8x8x24xi8>
  %exp_relu = tosa.clamp %exp_q {
    min_int = -128 : i64, max_int = 127 : i64,
    min_fp = 0.0 : f32, max_fp = 6.0 : f32
  } : (tensor<1x8x8x24xi8>) -> tensor<1x8x8x24xi8>

  // Stride-2 depthwise: 8x8 -> 4x4, asymmetric pad [top=0,bot=1,left=0,right=1].
  %w_dw = "tosa.const"() {value = dense<1> : tensor<3x3x24x1xi8>}
      : () -> tensor<3x3x24x1xi8>
  %b_dw = "tosa.const"() {value = dense<64> : tensor<24xi32>} : () -> tensor<24xi32>
  %dw_acc = tosa.depthwise_conv2d %exp_relu, %w_dw, %b_dw {
    pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>,
    dilation = array<i64: 1, 1>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x8x8x24xi8>, tensor<3x3x24x1xi8>, tensor<24xi32>)
      -> tensor<1x4x4x24xi32>
  %dw_q = tosa.rescale %dw_acc {
    input_zp = 0 : i32, output_zp = -128 : i32,
    multiplier = array<i32: 1503238553>, shift = array<i8: 37>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x4x4x24xi32>) -> tensor<1x4x4x24xi8>
  %dw_relu = tosa.clamp %dw_q {
    min_int = -128 : i64, max_int = 127 : i64,
    min_fp = 0.0 : f32, max_fp = 6.0 : f32
  } : (tensor<1x4x4x24xi8>) -> tensor<1x4x4x24xi8>

  // Project 24 -> 8, linear (no activation), no residual.
  %w_proj = "tosa.const"() {value = dense<3> : tensor<8x1x1x24xi8>}
      : () -> tensor<8x1x1x24xi8>
  %b_proj = "tosa.const"() {value = dense<-32> : tensor<8xi32>} : () -> tensor<8xi32>
  %proj_acc = tosa.conv2d %dw_relu, %w_proj, %b_proj {
    pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>
  } : (tensor<1x4x4x24xi8>, tensor<8x1x1x24xi8>, tensor<8xi32>)
      -> tensor<1x4x4x8xi32>
  %y = tosa.rescale %proj_acc {
    input_zp = 0 : i32, output_zp = -5 : i32,
    multiplier = array<i32: 1288490189>, shift = array<i8: 37>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %y : tensor<1x4x4x8xi8>
}
