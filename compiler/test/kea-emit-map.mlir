// RUN: kea-translate %s --emit-map=- | FileCheck %s
// RUN: kea-translate %s --io-quant=in=0.0078125,-128 --io-quant=out=0.25 --emit-map=- | FileCheck %s --check-prefix=OVERRIDE
//
// `model.map.json`, docs/ASSEMBLY.md §7: everything the assembly deliberately
// does not carry. The DRAM arena geometry and the symbol table come straight
// from `-kea-alloc`; the I/O descriptors need shape and quantization facts the
// Level 2 IR does not have, and each one is either DERIVED from the
// instruction stream (and says how) or defaulted and overridable.

func.func @m(%x: tensor<1x8x8x4xi8>) attributes {kea.dram_layout = {
    total_bytes = 4352 : i64, const_offset = 0 : i64, const_bytes = 256 : i64,
    io_offset = 2304 : i64, io_bytes = 512 : i64, scratch_offset = 2816 : i64,
    scratch_bytes = 1536 : i64, alignment = 64 : i64}} {

  // CHECK: "dram": {
  // CHECK-NEXT: "total_bytes": 4352
  // CHECK-NEXT: "const_offset": 0
  // CHECK-NEXT: "const_bytes": 256
  // CHECK-NEXT: "io_offset": 2304
  // CHECK-NEXT: "io_bytes": 512
  // CHECK-NEXT: "scratch_offset": 2816
  // CHECK-NEXT: "scratch_bytes": 1536
  // CHECK-NEXT: "alignment": 64

  %w = arith.constant dense<2> : tensor<16x1x1x4xi8>
  // Constants and inter-layer activations are plain DRAM symbols: `@name`
  // resolves against these, and `size` is what lets the disassembler
  // re-symbolize an address that lands inside one.
  // CHECK: "symbols": [
  // CHECK-NEXT: "name": "m.weights", "offset": 0, "size": 256
  %cw = kea.alloc from %w : tensor<16x1x1x4xi8>
        {name = "m.weights", role = "weights", layout = "mxu_tiles_16x16",
         addr = 0 : i64} : !kea.buffer<256xi8, DRAM>
  // CHECK-NEXT: "name": "m.mid", "offset": 2816, "size": 1024
  %mid = kea.alloc {name = "m.mid", role = "activation", addr = 2816 : i64}
       : !kea.buffer<1024xi8, DRAM>

  %in = kea.alloc {name = "in", role = "input", addr = 2304 : i64}
      : !kea.buffer<256xi8, DRAM>
  %out = kea.alloc {name = "out", role = "output", addr = 2560 : i64}
       : !kea.buffer<256xi8, DRAM>
  %a = kea.alloc {name = "at", role = "scratch", addr = 0 : i64}
     : !kea.buffer<1024xi8, A>
  %o = kea.alloc {name = "ot", role = "scratch", addr = 1024 : i64}
     : !kea.buffer<1024xi8, A>

  // The input's shape is the block argument's, because `-kea-tile` names a
  // model input `<func>.input<argno>` -- this one is named `in`, so the name
  // carries no argument number and the map falls back to FLAT rather than
  // inventing a shape.
  // CHECK: "tensors": [
  // CHECK-NEXT: "name": "in", "kind": "input", "index": 0,
  // CHECK-NEXT: "offset": 2304, "size_bytes": 256, "dtype": "int8", "layout": "FLAT",
  // The zero point is the value the tile is VCOPY-filled with before the input
  // is DMA'd into it: ISA.md §8.4(a) requires that to be the input zero point.
  // CHECK-NEXT: "shape": [256], "scale": 1, "zero_point": -5
  // OVERRIDE: "name": "in", "kind": "input"
  // OVERRIDE: "scale": 0.0078125, "zero_point": -128
  kea.vcopy to %a {src_addr = 0, dst_addr = 0, row_bytes = 1024, rows = 1,
                   src_row_stride = 0, dst_row_stride = 1024, fill_value = -5,
                   fill} : !kea.buffer<1024xi8, A>
  kea.dma_load %in -> %a {dram_addr = 0, spm_addr = 0, len0 = 256, n1 = 1,
                          n2 = 1, dram_s1 = 256, dram_s2 = 256, spm_s1 = 256,
                          spm_s2 = 256}
    : !kea.buffer<256xi8, DRAM> -> !kea.buffer<1024xi8, A>

  %q = kea.alloc {name = "acc", role = "scratch", addr = 0 : i64}
     : !kea.buffer<1024xi32, ACC>
  %p = kea.alloc {name = "qp", role = "scratch", addr = 0 : i64}
     : !kea.buffer<192xi8, W>
  // The output's zero point is the `out_zp` of the VPU op that produced the
  // tile stored to it.
  kea.vquant %q, %o, %p {acc_addr = 0, out_addr = 0, qparam_addr = 0,
                         num_pixels = 64, channels = 16, acc_pix_stride = 16,
                         out_pix_stride = 16, out_zp = -9, clamp_lo = -128,
                         clamp_hi = 127}
    : !kea.buffer<1024xi32, ACC>, !kea.buffer<1024xi8, A>, !kea.buffer<192xi8, W>

  // One contiguous whole-tensor store, so the shape is `[1, n2, n1, len0]`.
  // Anything else leaves the shape unknown and the map says FLAT.
  // CHECK: "name": "out", "kind": "output", "index": 0,
  // CHECK-NEXT: "offset": 2560, "size_bytes": 256, "dtype": "int8", "layout": "NHWC",
  // CHECK-NEXT: "shape": [1, 8, 8, 4], "scale": 1, "zero_point": -9
  // A `--io-quant` without a zero point overrides only the scale; the zero
  // point stays the one derived from the stream.
  // OVERRIDE: "name": "out", "kind": "output"
  // OVERRIDE: "scale": 0.25, "zero_point": -9
  kea.dma_store %o -> %out {dram_addr = 0, spm_addr = 0, len0 = 4, n1 = 8,
                            n2 = 8, dram_s1 = 4, dram_s2 = 32, spm_s1 = 4,
                            spm_s2 = 32}
    : !kea.buffer<1024xi8, A> -> !kea.buffer<256xi8, DRAM>

  // `spm_map` is debug only -- nothing at run time consults it, every address
  // in the stream is already absolute. The PCs are the FINAL instruction
  // indices, counted after synchronization is woven in, which is why they are
  // recomputed here rather than copied from `-kea-alloc`'s attribute.
  // CHECK: "spm_map": [
  // CHECK-NEXT: "name": "at", "space": "SPM_A", "offset": 0, "size": 1024, "first_pc": 0
  // CHECK: "name": "acc", "space": "ACC", "offset": 0, "size": 1024
  // CHECK: "metadata": {
  // CHECK: "function": "m"
  kea.halt
  return
}
