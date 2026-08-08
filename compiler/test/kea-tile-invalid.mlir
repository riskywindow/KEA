// RUN: kea-opt %s -split-input-file -kea-tile -verify-diagnostics
//
// The constraints -kea-tile must ACTIVELY DISCHARGE rather than assume:
// errata E5 (int32 ACC wraps), E6 (KeaAddParam bounds), ADR-0003 (the
// normalised-multiplier invariant), plus the ops that have no Level 2 form.
//
// "Fail loudly" is the requirement in every case: silently emitting a stream
// the golden model cannot reproduce is far worse than refusing to compile.

//===--------------------------------------------------------------------===//
// ADR-0003 -- tosa_shift >= 31, or the accumulator bound proves |acc| < 2^shift
//===--------------------------------------------------------------------===//
//
// MobileNetV2's real rescale shifts run from 22 to 48, so `tosa_shift < 31` is
// live, not hypothetical, and cannot simply be banned. The bound that decides
// it is `255 * sum_k |w[oc][k]| + |bias[oc]|`, computed from the actual
// constant weights -- the worst-case `K * 127 * 127` is true but so loose it
// would reject working layers.

// Weights of 127 over IC = 64: sum|w| = 8128, bound = 255*8128 = 2,072,640,
// which is NOT < 2^20 = 1,048,576.
func.func @adr0003_bound_too_large(%in: tensor<1x8x8x64xi8>) -> tensor<1x8x8x16xi8> {
  %w = arith.constant dense<127> : tensor<16x1x1x64xi8>
  // expected-error @+1 {{channel 0 requantizes with tosa_shift = 20 (< 31) but the accumulator range bound is 2072640, which is not < 2^20 = 1048576}}
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
        strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
        dilations = array<i64: 1, 1>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [20], input_zp = 0, output_zp = 0, axis = -1,
                    rounding = DOUBLE>>}
    : (tensor<1x8x8x64xi8>, tensor<16x1x1x64xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// -----

// The same layer with weights of 1: sum|w| = 64, bound = 16,320 < 2^20. The
// invariant IS discharged, so this compiles -- which is the point of using the
// real weights rather than the worst case.
func.func @adr0003_bound_discharged(%in: tensor<1x8x8x64xi8>) -> tensor<1x8x8x16xi8> {
  %w = arith.constant dense<1> : tensor<16x1x1x64xi8>
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
        strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
        dilations = array<i64: 1, 1>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [20], input_zp = 0, output_zp = 0, axis = -1,
                    rounding = DOUBLE>>}
    : (tensor<1x8x8x64xi8>, tensor<16x1x1x64xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// -----

// Non-constant weights: there is no tight bound to compute, so the fallback is
// the same worst case E5 uses, `K * 127 * 127`. With IC = 128 that is
// 128 * 16129 = 2,064,512, which does not fit under 2^20 either -- and with
// nothing better to appeal to the pass refuses rather than guessing.
func.func @adr0003_unknown_weights(%in: tensor<1x8x8x128xi8>,
                                   %w: tensor<16x1x1x128xi8>)
    -> tensor<1x8x8x16xi8> {
  // expected-error @+1 {{requantizes with tosa_shift = 20 (< 31) but the accumulator range bound is 2064512}}
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
        strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
        dilations = array<i64: 1, 1>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [20], input_zp = 0, output_zp = 0, axis = -1,
                    rounding = DOUBLE>>}
    : (tensor<1x8x8x128xi8>, tensor<16x1x1x128xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// -----

//===--------------------------------------------------------------------===//
// Errata E5 -- int32 ACC wraps, it does not saturate
//===--------------------------------------------------------------------===//
//
// The bound is `K * 127 * 127 < 2^31`, i.e. K <= 133,144 accumulated taps.
// A 5x5 convolution over 8192 input channels is 8192*25 = 204,800 taps.

func.func @e5_reduction_chain_overflows(%in: tensor<1x8x8x8192xi8>,
                                        %w: tensor<16x5x5x8192xi8>)
    -> tensor<1x8x8x16xi8> {
  // expected-error @+1 {{reduction chain of 204800 taps can overflow the int32 accumulator (errata E5: ACC wraps, it does not saturate; the bound is K * 127 * 127 < 2^31, i.e. K < 133144)}}
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
        strides = array<i64: 1, 1>, pads = array<i64: 2, 2, 2, 2>,
        dilations = array<i64: 1, 1>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [40], input_zp = 0, output_zp = 0, axis = -1,
                    rounding = DOUBLE>>}
    : (tensor<1x8x8x8192xi8>, tensor<16x5x5x8192xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// -----

//===--------------------------------------------------------------------===//
// Errata E6 -- keaQuantizedAdd sums xa + xb in plain int32
//===--------------------------------------------------------------------===//
//
// A left shift on the *output* rescale cannot be expressed: keaRdpot returns
// its input unchanged for a non-positive exponent, so the scale would silently
// vanish.

func.func @e6_output_needs_a_left_shift(%a: tensor<1x8x8x4xi8>,
                                        %b: tensor<1x8x8x4xi8>)
    -> tensor<1x8x8x4xi8> {
  // expected-error @+1 {{cannot express this quantized add as KEA_VADD: the output rescale would need a left shift KEA_VADD cannot express}}
  %0 = kea.add %a, %b {
    lhs_quant = #kea.quant<multiplier = [1610612736], shift = [11],
                           input_zp = 0, output_zp = 0, axis = -1,
                           rounding = DOUBLE>,
    rhs_quant = #kea.quant<multiplier = [1073741824], shift = [11],
                           input_zp = 0, output_zp = 0, axis = -1,
                           rounding = DOUBLE>,
    out_quant = #kea.quant<multiplier = [1503238553], shift = [20],
                           input_zp = 0, output_zp = 0, axis = -1,
                           rounding = DOUBLE>
  } : (tensor<1x8x8x4xi8>, tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xi8>
  return %0 : tensor<1x8x8x4xi8>
}

// -----

//===--------------------------------------------------------------------===//
// Ops with no Level 2 form
//===--------------------------------------------------------------------===//

// A raw i32 accumulator has nowhere to live: SPM_A holds int8/int4, and ACC is
// not DMA-reachable at all.
func.func @no_epilogue(%in: tensor<1x8x8x32xi8>, %w: tensor<16x1x1x32xi8>)
    -> tensor<1x8x8x16xi32> {
  // expected-error @+1 {{needs a requantized epilogue to reach Level 2}}
  %0 = kea.conv2d %in, %w {zero_points = #kea.zp<input = 0, weight = 0>,
        strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
        dilations = array<i64: 1, 1>}
    : (tensor<1x8x8x32xi8>, tensor<16x1x1x32xi8>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}

// -----

// KEA-1 has no transpose unit and no plan to grow one (ISA.md §13).
func.func @transpose_has_no_instruction(%in: tensor<1x8x8x16xi8>)
    -> tensor<1x16x8x8xi8> {
  // expected-error @+1 {{has no Level 2 lowering}}
  %0 = kea.transpose %in {perms = array<i64: 0, 3, 1, 2>}
     : (tensor<1x8x8x16xi8>) -> tensor<1x16x8x8xi8>
  return %0 : tensor<1x16x8x8xi8>
}

// -----

// VPOOL is VALID padding only, and an average pool over a padded window also
// needs TOSA's padding-aware divisor, which the unit does not have.
func.func @padded_pool(%in: tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8> {
  // expected-error @+1 {{VPOOL is VALID padding only}}
  %0 = kea.pool %in {kind = #kea.pool_kind<AVG>, kernel = array<i64: 3, 3>,
                     strides = array<i64: 2, 2>, pads = array<i64: 1, 0, 1, 0>}
     : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
  return %0 : tensor<1x4x4x16xi8>
}

// -----

// The DWU has one lane per channel and no output fan-out.
func.func @dwconv_channel_multiplier(%in: tensor<1x8x8x16xi8>,
                                     %w: tensor<32x3x3x1xi8>)
    -> tensor<1x8x8x32xi8> {
  // expected-error @+1 {{channel multiplier != 1 is not lowered}}
  %0 = kea.dwconv2d %in, %w {zero_points = #kea.zp<input = -128, weight = 0>,
        strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
        dilations = array<i64: 1, 1>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [35], input_zp = 0, output_zp = -128, axis = -1,
                    rounding = DOUBLE>>}
    : (tensor<1x8x8x16xi8>, tensor<32x3x3x1xi8>) -> tensor<1x8x8x32xi8>
  return %0 : tensor<1x8x8x32xi8>
}

// -----

// DWCONV has no dilation field.
func.func @dwconv_dilated(%in: tensor<1x12x12x16xi8>, %w: tensor<16x3x3x1xi8>)
    -> tensor<1x8x8x16xi8> {
  // expected-error @+1 {{DWCONV has no dilation field}}
  %0 = kea.dwconv2d %in, %w {zero_points = #kea.zp<input = -128, weight = 0>,
        strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
        dilations = array<i64: 2, 2>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [35], input_zp = 0, output_zp = -128, axis = -1,
                    rounding = DOUBLE>>}
    : (tensor<1x12x12x16xi8>, tensor<16x3x3x1xi8>) -> tensor<1x8x8x16xi8>
  return %0 : tensor<1x8x8x16xi8>
}

// -----

//===--------------------------------------------------------------------===//
// Errata E7 -- never MATMUL against a weight bank no LOAD_W has written
//===--------------------------------------------------------------------===//
//
// MXU weight banks are undefined at reset. The simulator warns once and treats
// an unloaded bank as zero, and the errata says outright: do not rely on that.
// This is not a property of any single op, so it is a whole-function check
// (`mlir::kea::verifyWeightBanks`) that -kea-tile runs over its own output --
// and over an already-Level-2 function, which is what this case is.

func.func @e7_unwritten_weight_bank() {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<1616xi8, A>
  %w = kea.alloc {name = "w", role = "scratch"} : !kea.buffer<4608xi8, W>
  %q = kea.alloc {name = "q", role = "scratch"} : !kea.buffer<2048xi32, ACC>
  kea.load_w %w {w_addr = 0, w_row_stride = 16, k_rows = 16, n_cols = 16,
                 bank = 0} : !kea.buffer<4608xi8, W>
  // expected-error @+1 {{reads MXU weight bank 1 but no kea.load_w has written it (errata E7}}
  kea.mm %a, %q {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                 m_inner = 8, m_outer = 8, acc_addr = 0, acc_inner_stride = 16,
                 acc_outer_stride = 128, bank = 1}
    : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
  return
}

//===--------------------------------------------------------------------===//
// Activation-by-activation matmul
//===--------------------------------------------------------------------===//
//
// The MXU is weight stationary: -kea-tile turns a matmul's second operand into
// a DRAM weight blob that -kea-emit lays out as 16x16 tiles, so it has to be a
// constant. Before this was diagnosed the pass built a kea.alloc with a null
// source and the failure surfaced as "null operand found" from the verifier,
// pointing at nothing useful.

func.func @matmul_activation_rhs(%a: tensor<1x8x16xi8>, %b: tensor<1x16x32xi8>)
    -> tensor<1x8x32xi8> {
  // expected-error @+1 {{cannot lower a matmul whose right-hand side is an activation rather than a compile-time constant}}
  %0 = kea.matmul %a, %b {zero_points = #kea.zp<input = 0, weight = 0>,
        epilogue = #kea.epilogue<requant = <multiplier = [1073741824],
                    shift = [36], input_zp = 0, output_zp = -128, axis = -1,
                    rounding = DOUBLE>>}
    : (tensor<1x8x16xi8>, tensor<1x16x32xi8>) -> tensor<1x8x32xi8>
  return %0 : tensor<1x8x32xi8>
}

// -----

//===--------------------------------------------------------------------===//
// A rescale of an i32 accumulator
//===--------------------------------------------------------------------===//
//
// An int8 -> int8 rescale lowers (identity MATMUL into ACC, then VQUANT), but
// an i32 operand is an accumulator that never left ACC, and there is no way to
// put one back: DMA cannot reach ACC and neither can VCOPY (ISA.md §10.4). It
// has to be fused into whatever produced it.

func.func @rescale_from_i32(%x: tensor<1x1x1x64xi32>) -> tensor<1x1x1x64xi8> {
  // expected-error @+1 {{only an int8 -> int8 rescale is lowered}}
  %0 = kea.rescale %x {quant = #kea.quant<multiplier = [1073741824],
        shift = [36], input_zp = 0, output_zp = -128, axis = -1,
        rounding = DOUBLE>}
    : (tensor<1x1x1x64xi32>) -> tensor<1x1x1x64xi8>
  return %0 : tensor<1x1x1x64xi8>
}
