// RUN: kea-opt %s "-kea-tile=report-tiles=true imem-budget=200" | FileCheck %s
// RUN: kea-opt %s "-kea-tile=imem-budget=8" -verify-diagnostics
//
// THE WHOLE-FUNCTION INSTRUCTION BUDGET.
//
// KEA-1 is branchless (ISA.md §1), so every tile of every layer is written out
// as straight-line code and the tile count IS the program size. IMEM holds
// KEA_MAX_INSTRUCTIONS = 32768 instructions and is not paged, which makes
// program size a capacity constraint exactly like SPM_A or ACC -- except that
// it is GLOBAL, so no per-layer greedy tile choice can respect it. Choosing
// each layer's cycle-optimal tile overruns IMEM by 26% on MobileNetV2's
// feature extractor.
//
// So `-kea-tile` plans the whole function at once: every layer publishes its
// Pareto frontier of (instructions, cycles), and a Lagrangian price in cycles
// per instruction is bisected for the cheapest one whose total fits. See
// docs/DIALECT_L2.md §5.3.1, and §5.5 for the measured trade-off curve.
//
// This file needs its own RUN lines because `imem-budget` is a pass option and
// therefore applies to every function in the module -- it cannot share
// kea-tile-invalid.mlir's `-split-input-file` run.

//===--------------------------------------------------------------------===//
// A budget the cycle-optimal plan already fits inside
//===--------------------------------------------------------------------===//
//
// One 8x8 spatial tile, 4 output-channel groups, one reduction tile: 2 TRACE +
// 2 (weight and qparam DMA) + 3 (fill, load, store) + 4*(2+1) = 19
// instructions, well under 200. `price = 0` is the observable signal that
// nothing was traded away -- the budget never became binding.

// CHECK-LABEL: func.func @fits_easily
// CHECK-SAME: kea.imem = {budget = 200 : i64, cycles = {{[0-9]+}} : i64, cycles_unconstrained = {{[0-9]+}} : i64, dram = {{[0-9]+}} : i64, dram_unconstrained = {{[0-9]+}} : i64, emitted = 20 : i64, planned = 19 : i64, price = 0.000000e+00 : f64, smallest = {{[0-9]+}} : i64}
//
// `emitted` counts what the lowering actually produced and `planned` what the
// instruction model predicted; they differ by exactly the ops the model does
// not attribute to any layer, which here is the single `kea.halt`. The model
// may only ever UNDERcount -- the budget is spent against it, so overcounting
// would mean the budget bought less than it was charged for, and `run()` fails
// loudly if that ever happens.

// The whole 8x8 output in one tile: four groups, four LOAD_W/MATMUL pairs,
// four VQUANTs, one store.
// CHECK: kea.mm %{{.*}} {{{.*}}acc_addr = 0 : i64
// CHECK: kea.mm %{{.*}} {{{.*}}acc_addr = 1024 : i64
// CHECK: kea.mm %{{.*}} {{{.*}}acc_addr = 2048 : i64
// CHECK: kea.mm %{{.*}} {{{.*}}acc_addr = 3072 : i64
// CHECK-NOT: kea.mm
// CHECK: kea.halt
// A budget of 8 cannot hold even the coarsest tiling of this one layer. The
// diagnostic says so with the number, rather than letting the assembler reject
// the program at the far end of the pipeline with nothing to blame. It is
// reported on the function because the budget is a whole-function resource.
// expected-error @+1 {{cannot fit this function in the instruction budget: even the coarsest tiling of every layer needs 19 instructions against a budget of 8}}
func.func @fits_easily(%in: tensor<1x8x8x16xi8>, %w: tensor<64x1x1x16xi8>)
    -> tensor<1x8x8x64xi8> {
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
        strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
        dilations = array<i64: 1, 1>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [33], input_zp = 0, output_zp = 0, axis = -1,
                    rounding = DOUBLE>>}
    : (tensor<1x8x8x16xi8>, tensor<64x1x1x16xi8>) -> tensor<1x8x8x64xi8>
  return %0 : tensor<1x8x8x64xi8>
}
