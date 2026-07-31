// AUTO-GENERATED. Do not hand-edit; regenerate with:
//
//   /usr/local/opt/llvm/bin/mlir-opt \
//     --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg))" \
//     tests/mlir/tosa/mobilenet_block.mlir
//
// This is tests/mlir/tosa/mobilenet_block.mlir lowered to linalg by
// LLVM/MLIR 20.1.6. Everything lowers successfully. Things to note:
//
//  * tosa.conv2d (quantized)          -> linalg.conv_2d_nhwc_fhwc_q
//        Zero points become trailing SCALAR i32 operands of the linalg op,
//        i.e. ins(%input, %weight, %input_zp, %weight_zp).
//  * tosa.depthwise_conv2d (quantized)-> linalg.depthwise_conv_2d_nhwc_hwcm_q
//        Produces a rank-5 [N,H,W,C,M] result that is then collapse_shape'd
//        back to rank 4, with the bias added by a separate linalg.generic.
//  * Explicit padding                 -> tensor.pad filled with the INPUT
//        ZERO POINT (not 0). This is the correct quantized padding semantics.
//  * tosa.rescale                     -> linalg.generic containing
//        tosa.apply_scale, which SURVIVES this pipeline. Add
//        tosa-to-arith{include-apply-rescale=true} to expand it into pure
//        arith (i64 multiply / round / shift).
//  * tosa.clamp                       -> arith.maxsi / arith.minsi
//  * tosa.add                         -> linalg.generic with arith.addi
//  * tosa.const                       -> stays as tosa.const here; it is
//        folded away by tosa-to-arith or --tosa-layerwise-constant-fold.
//
#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @mobilenet_v2_inverted_residual(%arg0: tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi8> {
    %0 = "tosa.const"() <{value = dense<2> : tensor<24x1x1x4xi8>}> : () -> tensor<24x1x1x4xi8>
    %1 = "tosa.const"() <{value = dense<128> : tensor<24xi32>}> : () -> tensor<24xi32>
    %2 = tensor.empty() : tensor<1x8x8x24xi32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<24xi32>) outs(%2 : tensor<1x8x8x24xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x8x8x24xi32>
    %c-5_i32 = arith.constant -5 : i32
    %c0_i32 = arith.constant 0 : i32
    %4 = linalg.conv_2d_nhwc_fhwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %0, %c-5_i32, %c0_i32 : tensor<1x8x8x4xi8>, tensor<24x1x1x4xi8>, i32, i32) outs(%3 : tensor<1x8x8x24xi32>) -> tensor<1x8x8x24xi32>
    %cst = arith.constant dense<[1073741824, 1181116006, 1288490189, 1395864371, 1503238553, 1610612736, 1717986918, 1825361101, 1932735283, 2040109466, 1073741824, 1181116006, 1288490189, 1395864371, 1503238553, 1610612736, 1717986918, 1825361101, 1932735283, 2040109466, 1073741824, 1181116006, 1288490189, 1395864371]> : tensor<24xi32>
    %cst_0 = arith.constant dense<[36, 36, 36, 36, 37, 37, 37, 37, 36, 36, 36, 36, 37, 37, 37, 37, 36, 36, 36, 36, 37, 37, 37, 37]> : tensor<24xi8>
    %5 = tensor.empty() : tensor<1x8x8x24xi8>
    %6 = linalg.generic {indexing_maps = [#map1, #map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %cst, %cst_0 : tensor<1x8x8x24xi32>, tensor<24xi32>, tensor<24xi8>) outs(%5 : tensor<1x8x8x24xi8>) {
    ^bb0(%in: i32, %in_9: i32, %in_10: i8, %out: i8):
      %c0_i32_11 = arith.constant 0 : i32
      %c-128_i32_12 = arith.constant -128 : i32
      %35 = arith.subi %in, %c0_i32_11 : i32
      %36 = tosa.apply_scale %35, %in_9, %in_10 {double_round = true} : (i32, i32, i8) -> i32
      %37 = arith.addi %36, %c-128_i32_12 : i32
      %c-128_i32_13 = arith.constant -128 : i32
      %c127_i32 = arith.constant 127 : i32
      %38 = arith.maxsi %c-128_i32_13, %37 : i32
      %39 = arith.minsi %c127_i32, %38 : i32
      %40 = arith.trunci %39 : i32 to i8
      linalg.yield %40 : i8
    } -> tensor<1x8x8x24xi8>
    %7 = tensor.empty() : tensor<1x8x8x24xi8>
    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x8x8x24xi8>) outs(%7 : tensor<1x8x8x24xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c-128_i8_9 = arith.constant -128 : i8
      %c127_i8 = arith.constant 127 : i8
      %35 = arith.maxsi %c-128_i8_9, %in : i8
      %36 = arith.minsi %c127_i8, %35 : i8
      linalg.yield %36 : i8
    } -> tensor<1x8x8x24xi8>
    %9 = "tosa.const"() <{value = dense<1> : tensor<3x3x24x1xi8>}> : () -> tensor<3x3x24x1xi8>
    %10 = "tosa.const"() <{value = dense<64> : tensor<24xi32>}> : () -> tensor<24xi32>
    %c-128_i8 = arith.constant -128 : i8
    %padded = tensor.pad %8 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %c-128_i8 : i8
    } : tensor<1x8x8x24xi8> to tensor<1x10x10x24xi8>
    %11 = tensor.empty() : tensor<1x8x8x24x1xi32>
    %c0_i32_1 = arith.constant 0 : i32
    %12 = linalg.fill ins(%c0_i32_1 : i32) outs(%11 : tensor<1x8x8x24x1xi32>) -> tensor<1x8x8x24x1xi32>
    %13 = tensor.empty() : tensor<1x8x8x24xi32>
    %c-128_i32 = arith.constant -128 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %14 = linalg.depthwise_conv_2d_nhwc_hwcm_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %9, %c-128_i32, %c0_i32_2 : tensor<1x10x10x24xi8>, tensor<3x3x24x1xi8>, i32, i32) outs(%12 : tensor<1x8x8x24x1xi32>) -> tensor<1x8x8x24x1xi32>
    %collapsed = tensor.collapse_shape %14 [[0], [1], [2], [3, 4]] : tensor<1x8x8x24x1xi32> into tensor<1x8x8x24xi32>
    %15 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10, %collapsed : tensor<24xi32>, tensor<1x8x8x24xi32>) outs(%13 : tensor<1x8x8x24xi32>) {
    ^bb0(%in: i32, %in_9: i32, %out: i32):
      %35 = arith.addi %in, %in_9 : i32
      linalg.yield %35 : i32
    } -> tensor<1x8x8x24xi32>
    %cst_3 = arith.constant dense<[1503238553, 1610612736, 1717986918, 1825361101, 1932735283, 2040109466, 1073741824, 1181116006, 1288490189, 1395864371, 1503238553, 1610612736, 1717986918, 1825361101, 1932735283, 2040109466, 1073741824, 1181116006, 1288490189, 1395864371, 1503238553, 1610612736, 1717986918, 1825361101]> : tensor<24xi32>
    %cst_4 = arith.constant dense<[37, 37, 37, 37, 36, 36, 36, 36, 37, 37, 37, 37, 36, 36, 36, 36, 37, 37, 37, 37, 36, 36, 36, 36]> : tensor<24xi8>
    %16 = tensor.empty() : tensor<1x8x8x24xi8>
    %17 = linalg.generic {indexing_maps = [#map1, #map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15, %cst_3, %cst_4 : tensor<1x8x8x24xi32>, tensor<24xi32>, tensor<24xi8>) outs(%16 : tensor<1x8x8x24xi8>) {
    ^bb0(%in: i32, %in_9: i32, %in_10: i8, %out: i8):
      %c0_i32_11 = arith.constant 0 : i32
      %c-128_i32_12 = arith.constant -128 : i32
      %35 = arith.subi %in, %c0_i32_11 : i32
      %36 = tosa.apply_scale %35, %in_9, %in_10 {double_round = true} : (i32, i32, i8) -> i32
      %37 = arith.addi %36, %c-128_i32_12 : i32
      %c-128_i32_13 = arith.constant -128 : i32
      %c127_i32 = arith.constant 127 : i32
      %38 = arith.maxsi %c-128_i32_13, %37 : i32
      %39 = arith.minsi %c127_i32, %38 : i32
      %40 = arith.trunci %39 : i32 to i8
      linalg.yield %40 : i8
    } -> tensor<1x8x8x24xi8>
    %18 = tensor.empty() : tensor<1x8x8x24xi8>
    %19 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17 : tensor<1x8x8x24xi8>) outs(%18 : tensor<1x8x8x24xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c-128_i8_9 = arith.constant -128 : i8
      %c127_i8 = arith.constant 127 : i8
      %35 = arith.maxsi %c-128_i8_9, %in : i8
      %36 = arith.minsi %c127_i8, %35 : i8
      linalg.yield %36 : i8
    } -> tensor<1x8x8x24xi8>
    %20 = "tosa.const"() <{value = dense<3> : tensor<4x1x1x24xi8>}> : () -> tensor<4x1x1x24xi8>
    %21 = "tosa.const"() <{value = dense<-32> : tensor<4xi32>}> : () -> tensor<4xi32>
    %22 = tensor.empty() : tensor<1x8x8x4xi32>
    %23 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21 : tensor<4xi32>) outs(%22 : tensor<1x8x8x4xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x8x8x4xi32>
    %c-128_i32_5 = arith.constant -128 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %24 = linalg.conv_2d_nhwc_fhwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%19, %20, %c-128_i32_5, %c0_i32_6 : tensor<1x8x8x24xi8>, tensor<4x1x1x24xi8>, i32, i32) outs(%23 : tensor<1x8x8x4xi32>) -> tensor<1x8x8x4xi32>
    %cst_7 = arith.constant dense<[1288490189, 1395864371, 1503238553, 1610612736]> : tensor<4xi32>
    %cst_8 = arith.constant dense<[37, 37, 38, 38]> : tensor<4xi8>
    %25 = tensor.empty() : tensor<1x8x8x4xi8>
    %26 = linalg.generic {indexing_maps = [#map1, #map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %cst_7, %cst_8 : tensor<1x8x8x4xi32>, tensor<4xi32>, tensor<4xi8>) outs(%25 : tensor<1x8x8x4xi8>) {
    ^bb0(%in: i32, %in_9: i32, %in_10: i8, %out: i8):
      %c0_i32_11 = arith.constant 0 : i32
      %c-5_i32_12 = arith.constant -5 : i32
      %35 = arith.subi %in, %c0_i32_11 : i32
      %36 = tosa.apply_scale %35, %in_9, %in_10 {double_round = true} : (i32, i32, i8) -> i32
      %37 = arith.addi %36, %c-5_i32_12 : i32
      %c-128_i32_13 = arith.constant -128 : i32
      %c127_i32 = arith.constant 127 : i32
      %38 = arith.maxsi %c-128_i32_13, %37 : i32
      %39 = arith.minsi %c127_i32, %38 : i32
      %40 = arith.trunci %39 : i32 to i8
      linalg.yield %40 : i8
    } -> tensor<1x8x8x4xi8>
    %c1073741824_i32 = arith.constant 1073741824 : i32
    %c10_i8 = arith.constant 10 : i8
    %27 = tensor.empty() : tensor<1x8x8x4xi32>
    %28 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x8x8x4xi8>) outs(%27 : tensor<1x8x8x4xi32>) {
    ^bb0(%in: i8, %out: i32):
      %c-5_i32_9 = arith.constant -5 : i32
      %c0_i32_10 = arith.constant 0 : i32
      %35 = arith.extsi %in : i8 to i32
      %36 = arith.subi %35, %c-5_i32_9 : i32
      %37 = tosa.apply_scale %36, %c1073741824_i32, %c10_i8 {double_round = false} : (i32, i32, i8) -> i32
      %38 = arith.addi %37, %c0_i32_10 : i32
      %c-2147483648_i32 = arith.constant -2147483648 : i32
      %c2147483647_i32 = arith.constant 2147483647 : i32
      %39 = arith.maxsi %c-2147483648_i32, %38 : i32
      %40 = arith.minsi %c2147483647_i32, %39 : i32
      linalg.yield %40 : i32
    } -> tensor<1x8x8x4xi32>
    %c1610612736_i32 = arith.constant 1610612736 : i32
    %c11_i8 = arith.constant 11 : i8
    %29 = tensor.empty() : tensor<1x8x8x4xi32>
    %30 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26 : tensor<1x8x8x4xi8>) outs(%29 : tensor<1x8x8x4xi32>) {
    ^bb0(%in: i8, %out: i32):
      %c-5_i32_9 = arith.constant -5 : i32
      %c0_i32_10 = arith.constant 0 : i32
      %35 = arith.extsi %in : i8 to i32
      %36 = arith.subi %35, %c-5_i32_9 : i32
      %37 = tosa.apply_scale %36, %c1610612736_i32, %c11_i8 {double_round = false} : (i32, i32, i8) -> i32
      %38 = arith.addi %37, %c0_i32_10 : i32
      %c-2147483648_i32 = arith.constant -2147483648 : i32
      %c2147483647_i32 = arith.constant 2147483647 : i32
      %39 = arith.maxsi %c-2147483648_i32, %38 : i32
      %40 = arith.minsi %c2147483647_i32, %39 : i32
      linalg.yield %40 : i32
    } -> tensor<1x8x8x4xi32>
    %31 = tensor.empty() : tensor<1x8x8x4xi32>
    %32 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %30 : tensor<1x8x8x4xi32>, tensor<1x8x8x4xi32>) outs(%31 : tensor<1x8x8x4xi32>) {
    ^bb0(%in: i32, %in_9: i32, %out: i32):
      %35 = arith.addi %in, %in_9 : i32
      linalg.yield %35 : i32
    } -> tensor<1x8x8x4xi32>
    %c1503238553_i32 = arith.constant 1503238553 : i32
    %c40_i8 = arith.constant 40 : i8
    %33 = tensor.empty() : tensor<1x8x8x4xi8>
    %34 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32 : tensor<1x8x8x4xi32>) outs(%33 : tensor<1x8x8x4xi8>) {
    ^bb0(%in: i32, %out: i8):
      %c0_i32_9 = arith.constant 0 : i32
      %c-5_i32_10 = arith.constant -5 : i32
      %35 = arith.subi %in, %c0_i32_9 : i32
      %36 = tosa.apply_scale %35, %c1503238553_i32, %c40_i8 {double_round = true} : (i32, i32, i8) -> i32
      %37 = arith.addi %36, %c-5_i32_10 : i32
      %c-128_i32_11 = arith.constant -128 : i32
      %c127_i32 = arith.constant 127 : i32
      %38 = arith.maxsi %c-128_i32_11, %37 : i32
      %39 = arith.minsi %c127_i32, %38 : i32
      %40 = arith.trunci %39 : i32 to i8
      linalg.yield %40 : i8
    } -> tensor<1x8x8x4xi8>
    return %34 : tensor<1x8x8x4xi8>
  }
  func.func @mobilenet_v2_inverted_residual_stride2(%arg0: tensor<1x8x8x4xi8>) -> tensor<1x4x4x8xi8> {
    %0 = "tosa.const"() <{value = dense<2> : tensor<24x1x1x4xi8>}> : () -> tensor<24x1x1x4xi8>
    %1 = "tosa.const"() <{value = dense<128> : tensor<24xi32>}> : () -> tensor<24xi32>
    %2 = tensor.empty() : tensor<1x8x8x24xi32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<24xi32>) outs(%2 : tensor<1x8x8x24xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x8x8x24xi32>
    %c-5_i32 = arith.constant -5 : i32
    %c0_i32 = arith.constant 0 : i32
    %4 = linalg.conv_2d_nhwc_fhwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %0, %c-5_i32, %c0_i32 : tensor<1x8x8x4xi8>, tensor<24x1x1x4xi8>, i32, i32) outs(%3 : tensor<1x8x8x24xi32>) -> tensor<1x8x8x24xi32>
    %c1073741824_i32 = arith.constant 1073741824 : i32
    %c36_i8 = arith.constant 36 : i8
    %5 = tensor.empty() : tensor<1x8x8x24xi8>
    %6 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<1x8x8x24xi32>) outs(%5 : tensor<1x8x8x24xi8>) {
    ^bb0(%in: i32, %out: i8):
      %c0_i32_5 = arith.constant 0 : i32
      %c-128_i32_6 = arith.constant -128 : i32
      %27 = arith.subi %in, %c0_i32_5 : i32
      %28 = tosa.apply_scale %27, %c1073741824_i32, %c36_i8 {double_round = true} : (i32, i32, i8) -> i32
      %29 = arith.addi %28, %c-128_i32_6 : i32
      %c-128_i32_7 = arith.constant -128 : i32
      %c127_i32 = arith.constant 127 : i32
      %30 = arith.maxsi %c-128_i32_7, %29 : i32
      %31 = arith.minsi %c127_i32, %30 : i32
      %32 = arith.trunci %31 : i32 to i8
      linalg.yield %32 : i8
    } -> tensor<1x8x8x24xi8>
    %7 = tensor.empty() : tensor<1x8x8x24xi8>
    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x8x8x24xi8>) outs(%7 : tensor<1x8x8x24xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c-128_i8_5 = arith.constant -128 : i8
      %c127_i8 = arith.constant 127 : i8
      %27 = arith.maxsi %c-128_i8_5, %in : i8
      %28 = arith.minsi %c127_i8, %27 : i8
      linalg.yield %28 : i8
    } -> tensor<1x8x8x24xi8>
    %9 = "tosa.const"() <{value = dense<1> : tensor<3x3x24x1xi8>}> : () -> tensor<3x3x24x1xi8>
    %10 = "tosa.const"() <{value = dense<64> : tensor<24xi32>}> : () -> tensor<24xi32>
    %c-128_i8 = arith.constant -128 : i8
    %padded = tensor.pad %8 low[0, 0, 0, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %c-128_i8 : i8
    } : tensor<1x8x8x24xi8> to tensor<1x9x9x24xi8>
    %11 = tensor.empty() : tensor<1x4x4x24x1xi32>
    %c0_i32_0 = arith.constant 0 : i32
    %12 = linalg.fill ins(%c0_i32_0 : i32) outs(%11 : tensor<1x4x4x24x1xi32>) -> tensor<1x4x4x24x1xi32>
    %13 = tensor.empty() : tensor<1x4x4x24xi32>
    %c-128_i32 = arith.constant -128 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %14 = linalg.depthwise_conv_2d_nhwc_hwcm_q {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded, %9, %c-128_i32, %c0_i32_1 : tensor<1x9x9x24xi8>, tensor<3x3x24x1xi8>, i32, i32) outs(%12 : tensor<1x4x4x24x1xi32>) -> tensor<1x4x4x24x1xi32>
    %collapsed = tensor.collapse_shape %14 [[0], [1], [2], [3, 4]] : tensor<1x4x4x24x1xi32> into tensor<1x4x4x24xi32>
    %15 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10, %collapsed : tensor<24xi32>, tensor<1x4x4x24xi32>) outs(%13 : tensor<1x4x4x24xi32>) {
    ^bb0(%in: i32, %in_5: i32, %out: i32):
      %27 = arith.addi %in, %in_5 : i32
      linalg.yield %27 : i32
    } -> tensor<1x4x4x24xi32>
    %c1503238553_i32 = arith.constant 1503238553 : i32
    %c37_i8 = arith.constant 37 : i8
    %16 = tensor.empty() : tensor<1x4x4x24xi8>
    %17 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15 : tensor<1x4x4x24xi32>) outs(%16 : tensor<1x4x4x24xi8>) {
    ^bb0(%in: i32, %out: i8):
      %c0_i32_5 = arith.constant 0 : i32
      %c-128_i32_6 = arith.constant -128 : i32
      %27 = arith.subi %in, %c0_i32_5 : i32
      %28 = tosa.apply_scale %27, %c1503238553_i32, %c37_i8 {double_round = true} : (i32, i32, i8) -> i32
      %29 = arith.addi %28, %c-128_i32_6 : i32
      %c-128_i32_7 = arith.constant -128 : i32
      %c127_i32 = arith.constant 127 : i32
      %30 = arith.maxsi %c-128_i32_7, %29 : i32
      %31 = arith.minsi %c127_i32, %30 : i32
      %32 = arith.trunci %31 : i32 to i8
      linalg.yield %32 : i8
    } -> tensor<1x4x4x24xi8>
    %18 = tensor.empty() : tensor<1x4x4x24xi8>
    %19 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17 : tensor<1x4x4x24xi8>) outs(%18 : tensor<1x4x4x24xi8>) {
    ^bb0(%in: i8, %out: i8):
      %c-128_i8_5 = arith.constant -128 : i8
      %c127_i8 = arith.constant 127 : i8
      %27 = arith.maxsi %c-128_i8_5, %in : i8
      %28 = arith.minsi %c127_i8, %27 : i8
      linalg.yield %28 : i8
    } -> tensor<1x4x4x24xi8>
    %20 = "tosa.const"() <{value = dense<3> : tensor<8x1x1x24xi8>}> : () -> tensor<8x1x1x24xi8>
    %21 = "tosa.const"() <{value = dense<-32> : tensor<8xi32>}> : () -> tensor<8xi32>
    %22 = tensor.empty() : tensor<1x4x4x8xi32>
    %23 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21 : tensor<8xi32>) outs(%22 : tensor<1x4x4x8xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x4x4x8xi32>
    %c-128_i32_2 = arith.constant -128 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %24 = linalg.conv_2d_nhwc_fhwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%19, %20, %c-128_i32_2, %c0_i32_3 : tensor<1x4x4x24xi8>, tensor<8x1x1x24xi8>, i32, i32) outs(%23 : tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
    %c1288490189_i32 = arith.constant 1288490189 : i32
    %c37_i8_4 = arith.constant 37 : i8
    %25 = tensor.empty() : tensor<1x4x4x8xi8>
    %26 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24 : tensor<1x4x4x8xi32>) outs(%25 : tensor<1x4x4x8xi8>) {
    ^bb0(%in: i32, %out: i8):
      %c0_i32_5 = arith.constant 0 : i32
      %c-5_i32_6 = arith.constant -5 : i32
      %27 = arith.subi %in, %c0_i32_5 : i32
      %28 = tosa.apply_scale %27, %c1288490189_i32, %c37_i8_4 {double_round = true} : (i32, i32, i8) -> i32
      %29 = arith.addi %28, %c-5_i32_6 : i32
      %c-128_i32_7 = arith.constant -128 : i32
      %c127_i32 = arith.constant 127 : i32
      %30 = arith.maxsi %c-128_i32_7, %29 : i32
      %31 = arith.minsi %c127_i32, %30 : i32
      %32 = arith.trunci %31 : i32 to i8
      linalg.yield %32 : i8
    } -> tensor<1x4x4x8xi8>
    return %26 : tensor<1x4x4x8xi8>
  }
}

