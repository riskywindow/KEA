// REGRESSION -- a standalone `kea.rescale` must lower.
//
// `kea.pool`'s optional `quant` is restricted by its own verifier to a
// zero-point rebase (`multiplier[i] == 1 << shift[i]`), because averaging is
// affine and only that much is exact. docs/DIALECT_L1.md's PoolOp therefore
// says a general rescale after a pool "must be written as a separate
// `kea.rescale`" -- and that used to be a dead end, because `-kea-fuse` only
// folds a rescale into a *contraction* and `-kea-tile` then refused the
// leftover:
//
//   error: 'kea.rescale' op has no Level 2 lowering. ... kea.rescale,
//   kea.clamp and kea.transpose must be eliminated by -kea-fuse
//
// A standalone `kea.rescale` now lowers through a 16x16 identity matmul --
// `VQUANT` can only read ACC, so the values have to be put there first. That
// is what lets MobileNetV2's head pool run on the NPU at all: its pool
// converts between two activation scales (0.023529 -> 0.016946), so the
// frontend emits avg_pool2d followed by a rescale of 1490907399 >> 30.
//
//   build/native/bin/keac demo/regress/pool_rescale.mlir \
//       --function pool_rescale -o /tmp/pr.keaf
//   build/native/bin/kea-sim /tmp/pr.keaf --quiet --strict-poison

func.func @pool_rescale(%x: tensor<1x7x7x64xi8>) -> tensor<1x1x1x64xi8> {
  %0 = tosa.avg_pool2d %x {
    kernel = array<i64: 7, 7>, stride = array<i64: 1, 1>,
    pad = array<i64: 0, 0, 0, 0>, acc_type = i32,
    quantization_info = #tosa.unary_quant<input_zp = -128, output_zp = -128>
  } : (tensor<1x7x7x64xi8>) -> tensor<1x1x1x64xi8>
  %1 = tosa.rescale %0 {
    input_zp = -128 : i32, output_zp = -128 : i32,
    multiplier = array<i32: 1490907399>, shift = array<i8: 30>,
    scale32 = true, double_round = true, per_channel = false
  } : (tensor<1x1x1x64xi8>) -> tensor<1x1x1x64xi8>
  return %1 : tensor<1x1x1x64xi8>
}
