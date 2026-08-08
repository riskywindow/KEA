// KNOWN LIMITATION -- a general `tosa.rescale` after a pool has no lowering.
//
// `kea.pool`'s optional `quant` is restricted by its own verifier to a
// zero-point rebase (`multiplier[i] == 1 << shift[i]`), because averaging is
// affine and only that much is exact.  docs/DIALECT_L1.md's PoolOp says "a
// general rescale after a pool must be written as a separate `kea.rescale`" --
// but `-kea-fuse` only fuses a `kea.rescale` into a *contraction*
// (conv2d / dwconv2d / matmul / fully_connected), and `-kea-tile` then refuses
// the leftover:
//
//   error: 'kea.rescale' op has no Level 2 lowering. ... kea.rescale,
//   kea.clamp and kea.transpose must be eliminated by -kea-fuse
//
// So the L1 dialect can express a scale-changing pool that the backend cannot
// compile.  This is what keeps MobileNetV2 from being one program: its head
// pool converts between two activation scales (0.023529 -> 0.016946), so the
// frontend emits avg_pool2d followed by a rescale of 1490907399 >> 30, and
// that rescale survives -kea-fuse.
//
// Two ways out, neither taken here: fuse a rescale into a `kea.pool` epilogue
// (the VPU's pool already requantizes), or fuse it forward into the *consumer*
// contraction's input quantization.
//
//   build/native/bin/keac demo/regress/pool_rescale_unsupported.mlir \
//       --function pool_rescale -o /tmp/pr.keaf

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
