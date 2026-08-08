// REGRESSION -- `kea.pool` must translate.
//
// This used to fail: `-kea-tile`'s lowerPool gave the pool's SPM_A tile the
// same name as the DRAM buffer backing the pool's result, and `kea-translate`
// rejected the duplicate:
//
//   error: 'kea.alloc' op duplicate buffer name "gap_pool.0.out"; a DRAM name
//   is also the .kasm symbol and must be unique (docs/DIALECT_L2.md 4.1)
//
// Fixed by moving uniquification into makeBuffer() so it covers every buffer,
// not just scratch ones.  `demo/regress/run_regressions.sh` now asserts that
// this compiles, assembles and runs.
//
//   build/native/bin/keac demo/regress/pool_translates.mlir \
//       --function gap_pool -o /tmp/p.keaf
//   build/native/bin/kea-sim /tmp/p.keaf --quiet          # 460 cycles

func.func @gap_pool(%x: tensor<1x7x7x64xi8>) -> tensor<1x1x1x64xi8> {
  %0 = tosa.avg_pool2d %x {
    kernel = array<i64: 7, 7>, stride = array<i64: 1, 1>,
    pad = array<i64: 0, 0, 0, 0>, acc_type = i32,
    quantization_info = #tosa.unary_quant<input_zp = -128, output_zp = -128>
  } : (tensor<1x7x7x64xi8>) -> tensor<1x1x1x64xi8>
  return %0 : tensor<1x1x1x64xi8>
}
