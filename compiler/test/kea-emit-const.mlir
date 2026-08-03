// RUN: kea-translate %s --emit-const-listing=- | FileCheck %s
// RUN: kea-translate %s --emit-const=%t.bin --emit-const-listing=%t.txt
//
// The weight/constant blob: every `layout` docs/DIALECT_L2.md §4.5 defines,
// materialized into the exact bytes ISA.md §8.1 / §8.6 / §9.1 specify.
//
// The tensors below are deliberately tiny and every element is distinct, so a
// transposed, mis-strided or mis-grouped layout produces different bytes. A
// splat weight tensor would pass under every wrong permutation, which is why
// there is not one here.
//
// `--emit-const-listing` is the same bytes as `--emit-const`, printed: raw
// int8 for the weight layouts (16 per line, i.e. one `k` row of one 16x16 MXU
// tile), and decoded records for `quant_params` / `add_params`.

func.func @constants() attributes {kea.dram_layout = {
    total_bytes = 65536 : i64, const_offset = 0 : i64, const_bytes = 1320 : i64,
    io_offset = 1344 : i64, io_bytes = 0 : i64, scratch_offset = 1344 : i64,
    scratch_bytes = 64192 : i64, alignment = 64 : i64}} {

  //--- mxu_tiles_16x16 : [OC, KH, KW, IC] -> dense 16x16 tiles ------------
  // OC = 2, IC = 3, one tap: one 256-byte tile holding W[k][n] at k*16 + n,
  // i.e. the weight array TRANSPOSED, with the k >= IC and n >= OC lanes
  // zeroed (that is how LOAD_W's k_rows / n_cols tail tiles work, ISA.md §7.2).
  %w1 = arith.constant dense<[[[[1, 2, 3]]], [[[4, 5, 6]]]]> : tensor<2x1x1x3xi8>
  // CHECK: ; conv.weights  role=weights layout=mxu_tiles_16x16 offset=0 size=256
  // CHECK-NEXT:      0:    1    4    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  // CHECK-NEXT:     16:    2    5    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  // CHECK-NEXT:     32:    3    6    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  // CHECK-NEXT:     48:    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  %0 = kea.alloc from %w1 : tensor<2x1x1x3xi8>
       {name = "conv.weights", role = "weights", layout = "mxu_tiles_16x16",
        addr = 0 : i64} : !kea.buffer<256xi8, DRAM>

  //--- mxu_tiles_16x16_packed : ISA.md §8.6 -------------------------------
  // OC = 2, KH = KW = 2, IC = 2. A whole kernel ROW is one reduction tile, so
  // there are KH = 2 tiles instead of KH*KW = 4, and the row index is
  // k = kw*IC + ic. w[oc][kh][kw][ic] = oc*8 + kh*4 + kw*2 + ic + 1.
  %w2 = arith.constant dense<[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                              [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]>
      : tensor<2x2x2x2xi8>
  // CHECK: ; conv.packed  role=weights layout=mxu_tiles_16x16_packed offset=256 size=512
  // CHECK-NEXT:      0:    1    9    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  // CHECK-NEXT:     16:    2   10    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  // CHECK-NEXT:     32:    3   11    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  // CHECK-NEXT:     48:    4   12    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  // The second tile is kh = 1, at byte 256 of this object.
  // CHECK:    256:    5   13    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  // CHECK-NEXT:    272:    6   14    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  // CHECK-NEXT:    288:    7   15    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  // CHECK-NEXT:    304:    8   16    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  %1 = kea.alloc from %w2 : tensor<2x2x2x2xi8>
       {name = "conv.packed", role = "weights",
        layout = "mxu_tiles_16x16_packed", addr = 256 : i64}
     : !kea.buffer<512xi8, DRAM>

  //--- mxu_tiles_16x16_kn : [B, K, N], the batched-matmul rhs -------------
  // The one layout whose output channel is the LAST dimension, so the tile is
  // the array as written rather than its transpose.
  %w3 = arith.constant dense<[[[1, 2], [3, 4], [5, 6]]]> : tensor<1x3x2xi8>
  // CHECK: ; mm.rhs  role=weights layout=mxu_tiles_16x16_kn offset=768 size=256
  // CHECK-NEXT:      0:    1    2    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  // CHECK-NEXT:     16:    3    4    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  // CHECK-NEXT:     32:    5    6    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  %2 = kea.alloc from %w3 : tensor<1x3x2xi8>
       {name = "mm.rhs", role = "weights", layout = "mxu_tiles_16x16_kn",
        addr = 768 : i64} : !kea.buffer<256xi8, DRAM>

  //--- dwu_planes : [OC, KH, KW, 1] -> [KH][KW][C_pad] --------------------
  // C = 3 pads to C_pad = 16, and the padding lanes MUST be zero: DWCONV's
  // `channels` is rounded up to a multiple of 16, so those lanes are read, and
  // a zero weight is what makes reading them harmless (DIALECT_L2.md §6.4).
  // w[c][kh][kw] = c*4 + kh*2 + kw + 1.
  %w4 = arith.constant dense<[[[[1], [2]], [[3], [4]]],
                              [[[5], [6]], [[7], [8]]],
                              [[[9], [10]], [[11], [12]]]]> : tensor<3x2x2x1xi8>
  // CHECK: ; dw.weights  role=weights layout=dwu_planes offset=1024 size=64
  // CHECK-NEXT:      0:    1    5    9    0    0    0    0    0    0    0    0    0    0    0    0    0
  // CHECK-NEXT:     16:    2    6   10    0    0    0    0    0    0    0    0    0    0    0    0    0
  // CHECK-NEXT:     32:    3    7   11    0    0    0    0    0    0    0    0    0    0    0    0    0
  // CHECK-NEXT:     48:    4    8   12    0    0    0    0    0    0    0    0    0    0    0    0    0
  %3 = kea.alloc from %w4 : tensor<3x2x2x1xi8>
       {name = "dw.weights", role = "weights", layout = "dwu_planes",
        addr = 1024 : i64} : !kea.buffer<64xi8, DRAM>

  //--- quant_params : DIALECT_L2.md §4.3, INCLUDING the zero-point fold ---
  // bias[c] = bias_l1[c] - input_zp * sum_k w[c][k]
  //   c=0: 100 - (-5)*(1+2+3) =  130
  //   c=1: 200 - (-5)*(4+5+6) =  275
  // shift = tosa_shift - 31 (ADR-0003), so 36 -> 5 and 40 -> 9.
  // Channels 2..15 are the padding VQUANT reads because `channels` must be a
  // multiple of 16; their records are zero and are never stored.
  %b = arith.constant dense<[100, 200]> : tensor<2xi32>
  // CHECK: ; conv.qparams  role=qparam layout=quant_params offset=1088 size=192
  // CHECK-NEXT:   [0] bias=130 mult=1073741824 shift=5
  // CHECK-NEXT:   [1] bias=275 mult=1181116006 shift=9
  // CHECK-NEXT:   [2] bias=0 mult=0 shift=0
  %4 = kea.alloc from %b, %w1 : tensor<2xi32>, tensor<2x1x1x3xi8>
       {name = "conv.qparams", role = "qparam", layout = "quant_params",
        input_zp = -5 : i64, addr = 1088 : i64,
        quant = #kea.quant<multiplier = [1073741824, 1181116006],
                           shift = [36, 40], input_zp = 0, output_zp = -128,
                           axis = 3, rounding = DOUBLE>}
     : !kea.buffer<192xi8, DRAM>

  //--- add_params : DIALECT_L2.md §4.4, the 20-byte KeaAddParam ----------
  // Written verbatim in isa.h field order. These are the values §7.3 derives
  // for the MobileNetV2 inverted residual.
  // CHECK: ; add.params  role=addparam layout=add_params offset=1280 size=20
  // CHECK-NEXT:   a_mult=1610612736 b_mult=1073741824 o_mult=1503238553
  // CHECK-NEXT:   a_shift=1 b_shift=0 o_shift=8
  // CHECK-NEXT:   a_zp=-5 b_zp=-5 o_zp=-5
  %5 = kea.alloc {name = "add.params", role = "addparam",
                  layout = "add_params", addr = 1280 : i64,
                  add_param = array<i64: 1610612736, 1073741824, 1503238553,
                                         1, 0, 8, -5, -5, -5>}
     : !kea.buffer<20xi8, DRAM>

  //--- nhwc : a dense activation constant, byte for byte -----------------
  %a = arith.constant dense<[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]>
      : tensor<1x2x2x2xi8>
  // CHECK: ; act.const  role=weights layout=nhwc offset=1312 size=8
  // CHECK-NEXT:      0:    1    2    3    4    5    6    7    8
  %6 = kea.alloc from %a : tensor<1x2x2x2xi8>
       {name = "act.const", role = "weights", layout = "nhwc",
        addr = 1312 : i64} : !kea.buffer<8xi8, DRAM>

  kea.halt
  return
}
