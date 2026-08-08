// BACKEND DEFECT 1 -- `-kea-tile` gives a pool's SPM_A tile the same name as
// the DRAM buffer backing the pool's result, and `kea-translate` rejects
// duplicate buffer names.  Every `kea.pool` in every program hits this, so
// `tosa.avg_pool2d` / `tosa.max_pool2d` are unusable end to end.
//
//   Tile.cpp lowerPool():
//     Value outBuf = dram(op.getOutput(), "activation", layerName("out"));
//     Value oTile  = scratch(loc, ..., AddressSpace::A, layerName("out"));
//                                                       ^^^^^^^^^^^^^^^^
//   The conv path avoids it by calling its SPM tile "otile"; scratch()'s
//   uniquifier only disambiguates against other *scratch* names, not against
//   DRAM ones.
//
// Reproduce:
//   build/native/bin/keac demo/repro/pool_dram_name_collision.mlir \
//       --function gap_pool -o /tmp/p.keaf
//
// Observed (kea-translate):
//   error: 'kea.alloc' op duplicate buffer name "gap_pool.0.out"; a DRAM name
//   is also the .kasm symbol and must be unique (docs/DIALECT_L2.md 4.1)

func.func @gap_pool(%x: tensor<1x7x7x64xi8>) -> tensor<1x1x1x64xi8> {
  %0 = tosa.avg_pool2d %x {
    kernel = array<i64: 7, 7>, stride = array<i64: 1, 1>,
    pad = array<i64: 0, 0, 0, 0>, acc_type = i32,
    quantization_info = #tosa.unary_quant<input_zp = -128, output_zp = -128>
  } : (tensor<1x7x7x64xi8>) -> tensor<1x1x1x64xi8>
  return %0 : tensor<1x1x1x64xi8>
}
