// RUN: kea-opt %s -kea-tile -kea-schedule -kea-alloc | FileCheck %s
// RUN: kea-opt %s -kea-tile -kea-schedule -kea-alloc | kea-opt -kea-alloc=verify-only=true | FileCheck %s --check-prefix=REVERIFY
//
// REGRESSION: a genuinely TILED layer, scheduled, then allocated.
//
// A 112x112x32 -> 16 pointwise does not fit on chip in one go, so -kea-tile
// splits it into row bands and -kea-schedule double buffers them. That produces
// the shape this pass exists for and the shape it originally got wrong: a long
// chain of equal-sized ACC tiles whose live ranges overlap only their immediate
// neighbours.
//
//   acc    [ 35,  55]
//   acc#1  [ 52,  79]     <- overlaps acc and acc#2, nothing else
//   acc#2  [ 76,  97]
//   ...
//   acc#13 [327, 344]
//
// Fourteen 14,336-word tiles, never more than two live at once. `maxlive` is
// 28,672 against a 32,768-word ACC, so a placement exists: alternate the two
// tiles between offsets 0 and 14,336.
//
// THE BUG THIS PINS. Greedy-by-size ordered equal-sized buffers by live-range
// *length*, which is unrelated to their position in the chain. It therefore
// coloured the chain out of temporal order -- acc#10 was placed at 0 long
// before acc#9 was considered -- and by the time acc#9 came up, its two
// neighbours acc#8 and acc#10 sat at different offsets even though they do not
// interfere with each other. Both slots taken, no third slot available, and the
// pass rejected a program that fits. See docs/MEMORY_PLANNING.md §3.1.
//
// The fix is that equal-sized buffers are now offered in order of live-range
// START, which is the classic optimal colouring order for an interval graph.
// This test fails if that ordering regresses.

// CHECK-LABEL: func.func @tiled_pointwise
//
// peak == maxlive and fragmentation == 0: the placement is not merely legal,
// it is optimal, because no allocator can beat the largest amount of
// simultaneously live data.
// CHECK-SAME:  acc = {buffers = 14 : i64, capacity = 32768 : i64, fragmentation = 0 : i64, maxlive = 28672 : i64, peak = 28672 : i64, unpacked = 200704 : i64}
//
// Packing turns 200,704 words of ACC demand into 28,672 -- a 7x reduction, and
// the difference between compiling and not.

// The chain alternates between the only two offsets that fit. Two 14,336-word
// tiles at 0 and 14,336 reach 28,672; a third would need 43,008 and ACC holds
// 32,768.
// CHECK:      kea.alloc {addr = 0 : i64, {{.*}}name = "tiled_pointwise.0.acc"{{.*}} !kea.buffer<14336xi32, ACC>
// CHECK:      kea.alloc {addr = 14336 : i64, {{.*}}name = "tiled_pointwise.0.acc#1"{{.*}} !kea.buffer<14336xi32, ACC>
// CHECK:      kea.alloc {addr = 0 : i64, {{.*}}name = "tiled_pointwise.0.acc#2"{{.*}} !kea.buffer<14336xi32, ACC>
// CHECK:      kea.alloc {addr = 14336 : i64, {{.*}}name = "tiled_pointwise.0.acc#3"{{.*}} !kea.buffer<14336xi32, ACC>
// The one that used to fail:
// CHECK:      kea.alloc {addr = {{0|14336}} : i64, {{.*}}name = "tiled_pointwise.0.acc#9"{{.*}} !kea.buffer<14336xi32, ACC>
// CHECK:      kea.alloc {addr = {{0|14336}} : i64, {{.*}}name = "tiled_pointwise.0.acc#13"{{.*}} !kea.buffer<14336xi32, ACC>

// Every ACC base is a multiple of 16 WORDS (ISA.md §11.1), which 0 and 14336
// both are -- 14336 = 896 * 16.

// The second RUN line re-runs the overlap verifier from scratch over the
// finished program: no two buffers whose live ranges overlap were given
// overlapping storage. It exits non-zero if they were, so reaching this CHECK
// at all is the assertion.
// REVERIFY-LABEL: func.func @tiled_pointwise
// REVERIFY:       kea.alloc {addr = 0 : i64
// REVERIFY:       !kea.buffer<14336xi32, ACC>

func.func @tiled_pointwise(%in: tensor<1x112x112x32xi8>, %w: tensor<16x1x1x32xi8>)
    -> tensor<1x112x112x16xi8> {
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
        strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
        dilations = array<i64: 1, 1>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
          shift = [38], input_zp = 0, output_zp = 0, axis = -1, rounding = DOUBLE>>}
      : (tensor<1x112x112x32xi8>, tensor<16x1x1x32xi8>) -> tensor<1x112x112x16xi8>
  return %0 : tensor<1x112x112x16xi8>
}
