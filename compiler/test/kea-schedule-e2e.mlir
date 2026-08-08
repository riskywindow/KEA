// RUN: kea-opt %s -tosa-to-kea -kea-fuse -kea-tile -kea-schedule -kea-alloc | FileCheck %s
// RUN: kea-opt %s -tosa-to-kea -kea-fuse -kea-tile -kea-schedule -kea-alloc | kea-opt -kea-alloc=verify-only=true | FileCheck %s --check-prefix=REVERIFY
// RUN: kea-opt %s -tosa-to-kea -kea-fuse -kea-tile -kea-schedule=report-schedule=true | FileCheck %s --check-prefix=REPORT
// RUN: kea-opt %s -tosa-to-kea -kea-fuse -kea-tile -kea-schedule | kea-opt -kea-tile | FileCheck %s --check-prefix=REVALIDATE
//
// END-TO-END: a complete MobileNetV2 inverted-residual block in quantized
// TOSA, all the way through `-tosa-to-kea -kea-fuse -kea-tile -kea-schedule
// -kea-alloc`. The source is a verbatim copy of
// tests/mlir/tosa/mobilenet_block.mlir, which lives outside compiler/test
// where run_tests.sh looks; compiler/test/kea-alloc-e2e.mlir carries the same
// copy for the pass before this one.
//
// What each RUN line proves:
//
//   1. the scheduled program still allocates -- the live ranges the scheduler
//      widened to discharge ADR-0002's soundness obligation still fit, which
//      is exactly what `-kea-tile`'s `spm-reserve-factor = 2` reserved room
//      for;
//   2. re-running `-kea-alloc` in verify-only mode re-proves from scratch that
//      no two overlapping-live-range buffers share storage;
//   3. the schedule report: both DMA engines carry work, the depth-16 queues
//      are respected, the whole block needs 15 of the 32 events, and the
//      modelled cost of overlapping (2098) beats the modelled cost of not
//      (2522), which is why `mode=auto` emits the overlapped plan;
//   4. running `-kea-tile` over already-Level-2 IR re-runs `verifyWeightBanks`
//      and `refreshLiveRanges` (DIALECT_L2.md §2), so errata E7 and the
//      live-range stamps are re-checked after all the motion.
//
// Note what is NOT here: any mention of double buffering in `-kea-alloc`'s
// output. The allocator separates the tiles because their ranges overlap, full
// stop -- ADR-0002's amendment in one line of test output.

// The three layers keep their TRACE brackets, on the queue that does their
// arithmetic: MXU for the two 1x1 convolutions, DWU for the depthwise.
// CHECK-LABEL: func.func @mobilenet_v2_inverted_residual(
// CHECK-DAG: kea.trace "begin" 0 {unit = "MXU"}
// CHECK-DAG: kea.trace "begin" 1 {unit = "DWU"}
// CHECK-DAG: kea.trace "begin" 2 {unit = "MXU"}
// CHECK-DAG: kea.trace "end" 2 {unit = "MXU"}
// Both DMA engines carry loads -- the second one takes a descriptor exactly
// when the first is still busy, which is when a second engine buys anything
// (SCHEDULING.md §2.2) -- every instruction has a queue, and the semaphores
// are there.
// CHECK-DAG: kea.dma_load {{.*}}unit = "DMA0"
// CHECK-DAG: kea.dma_load {{.*}}unit = "DMA1"
// CHECK-DAG: kea.dma_store {{.*}}unit = "DMA0"
// CHECK-DAG: kea.dwconv {{.*}}kea.unit = "DWU"
// CHECK-DAG: kea.vadd {{.*}}kea.unit = "VPU"
// CHECK-DAG: kea.signal
// CHECK-DAG: kea.wait
// Every buffer got an address and the whole block fits with room to spare.
// CHECK: addr =
// CHECK: kea.halt

// REVERIFY-LABEL: func.func @mobilenet_v2_inverted_residual(
// REVERIFY: addr =

// REPORT-LABEL: func.func @mobilenet_v2_inverted_residual(
// One pass of the capacity fixpoint: the buffers-in-flight bound each space
// started from was admissible first time.
// REPORT-SAME:  capacity_iters = 1 : i64
// REPORT-SAME:  events = 15 : i64
// The two plans are costed with the same model and the overlapped one wins,
// so that is what is emitted (SCHEDULING.md §9).
// REPORT-SAME:  mode = "overlap"
// REPORT-SAME:  modelled_in_order = 2522 : i64
// REPORT-SAME:  modelled_overlap = 2098 : i64
// REPORT-SAME:  queue_depth = 16 : i64
// Both engines carry real DRAM traffic, and neither queue reaches its depth.
// REPORT-SAME:  DMA0 = {busy = {{[1-9][0-9]*}} : i64, dram_bytes = {{[1-9][0-9]*}} : i64
// REPORT-SAME:  DMA1 = {busy = {{[1-9][0-9]*}} : i64, dram_bytes = {{[1-9][0-9]*}} : i64
// The whole block needs well under either scratchpad even with the widened
// ranges: 11248 of 262144 bytes of SPM_A.
// REPORT-SAME:  spm_a_peak = 11248 : i64
// REPORT-SAME:  spm_w_peak = 2292 : i64

// REVALIDATE-LABEL: func.func @mobilenet_v2_inverted_residual(
// REVALIDATE: kea.load_w
// REVALIDATE: kea.mm

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
