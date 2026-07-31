// SPDX-License-Identifier: Apache-2.0
//
// kea/isa.h --- KEA-1 instruction set architecture (FROZEN CONTRACT)
//
// Fixed-width 32-byte, little-endian instruction encoding for the KEA-1 edge
// NPU, plus inline encoders/decoders, the normative requantization reference
// implementation, an opcode metadata table for disassemblers, and a static
// validator.
//
// Header-only, C++17.  Depends only on <cstdint>, <cstddef>, and
// "kea/hw_config.h".
//
// Nobody should hand-roll bit twiddling against this format.  Use the
// keaMake*/keaDecode* helpers.  See docs/ISA.md for the full field tables and
// the convolution lowering rule.

#pragma once

#include <cstddef>
#include <cstdint>

#include "kea/hw_config.h"

namespace kea {

// ===========================================================================
// 1. Enumerations
// ===========================================================================

/// Instruction opcodes.  The high nibble names the owning unit class, which
/// makes `defaultUnitFor()` a shift and lets a disassembler bucket opcodes
/// without a table lookup.
///
///   0x0_  CTRL   0x1_  DMA   0x2_  MXU   0x3_  DWU   0x4_  VPU
enum class Opcode : std::uint8_t {
  NOP    = 0x00,
  HALT   = 0x01,
  SIGNAL = 0x02,
  WAIT   = 0x03,
  TRACE  = 0x04,

  DMA_LD = 0x10,
  DMA_ST = 0x11,

  LOAD_W = 0x20,
  MATMUL = 0x21,

  DWCONV = 0x30,

  VQUANT = 0x40,
  VADD   = 0x41,
  VPOOL  = 0x42,
  VCOPY  = 0x43,

  INVALID = 0xFF,
};

/// Target queue / execution unit.  Mirrors KEA_UNIT_* in hw_config.h.
enum class Unit : std::uint8_t {
  MXU  = 0,
  DWU  = 1,
  VPU  = 2,
  DMA0 = 3,
  DMA1 = 4,
  CTRL = 5,
};

/// Datatypes visible to the ISA.
enum class DType : std::uint8_t {
  INT8  = 0,
  INT4  = 1,
  INT32 = 2,  // accumulator only; never appears in an instruction flag bit
};

/// Address spaces.
enum class Space : std::uint8_t {
  SPM_A = 0,  // byte-addressed, 256 KiB
  SPM_W = 1,  // byte-addressed, 256 KiB
  ACC   = 2,  // int32-WORD-addressed, 32768 words
  DRAM  = 3,  // byte-addressed, 4 GiB
};

/// TRACE record kinds.
enum class TraceKind : std::uint8_t {
  MARKER       = 0,  // point event
  REGION_BEGIN = 1,  // opens a profiling region identified by `tag`
  REGION_END   = 2,  // closes the innermost region with the same `tag`
};

// ---------------------------------------------------------------------------
// Flag bit positions (per opcode).  Bits not listed are reserved and MUST be
// written as zero; a decoder may reject non-zero reserved bits.
// ---------------------------------------------------------------------------

// DMA_LD / DMA_ST, byte[2]
inline constexpr std::uint8_t KEA_DMA_F_SPM_W = 1u << 0;  // 0 => SPM_A, 1 => SPM_W

// LOAD_W, byte[2]
inline constexpr std::uint8_t KEA_LDW_F_INT4 = 1u << 0;
inline constexpr std::uint8_t KEA_LDW_F_BANK1 = 1u << 1;

// MATMUL, byte[2]
inline constexpr std::uint8_t KEA_MM_F_ACCUMULATE = 1u << 0;  // 0 => overwrite ACC
inline constexpr std::uint8_t KEA_MM_F_INT4 = 1u << 1;
inline constexpr std::uint8_t KEA_MM_F_BANK1 = 1u << 2;

// DWCONV, byte[2]
inline constexpr std::uint8_t KEA_DW_F_KERNEL5 = 1u << 0;     // 0 => 3x3, 1 => 5x5
inline constexpr std::uint8_t KEA_DW_F_STRIDE2 = 1u << 1;     // 0 => stride 1
inline constexpr std::uint8_t KEA_DW_F_ACCUMULATE = 1u << 2;  // 0 => overwrite ACC

// VQUANT, byte[2]
inline constexpr std::uint8_t KEA_VQ_F_OUT_INT4 = 1u << 0;

// VPOOL, byte[2]
inline constexpr std::uint8_t KEA_VP_F_AVG = 1u << 0;  // 0 => max pool

// VCOPY, byte[2]
inline constexpr std::uint8_t KEA_VC_F_FILL = 1u << 0;   // ignore src, splat fill_value
inline constexpr std::uint8_t KEA_VC_F_SRC_W = 1u << 1;  // 0 => SPM_A, 1 => SPM_W
inline constexpr std::uint8_t KEA_VC_F_DST_W = 1u << 2;  // 0 => SPM_A, 1 => SPM_W

// ===========================================================================
// 2. Instruction container and per-opcode overlays
// ===========================================================================

#pragma pack(push, 1)

/// The raw 32-byte instruction word.  Little-endian.  Byte 0 is always the
/// opcode, byte 1 the target unit, byte 2 the opcode-specific flag byte, and
/// byte 3 an opcode-specific auxiliary byte.
struct KeaInstr {
  std::uint8_t b[32];
};

/// Common prefix shared by every instruction.
struct KeaHead {
  std::uint8_t opcode;  // Opcode
  std::uint8_t unit;    // Unit
  std::uint8_t flags;   // opcode-specific bit flags
  std::uint8_t aux;     // opcode-specific byte
};

// --- CTRL ------------------------------------------------------------------

/// NOP: consumes a queue slot and `cycles` occupancy cycles on `unit`.
/// Useful as a scheduling spacer and for simulator self-tests.
struct KeaNop {
  KeaHead h;                  // aux reserved
  std::uint32_t cycles;       // additional occupancy, may be 0
  std::uint8_t reserved[24];
};

/// HALT: stops instruction fetch.  `unit` MUST be Unit::CTRL.
struct KeaHalt {
  KeaHead h;                  // flags/aux reserved
  std::uint32_t exit_code;    // reported by the runtime; 0 == success
  std::uint8_t reserved[24];
};

/// SIGNAL / WAIT share one layout.
///   SIGNAL: event[aux] += value            (never blocks)
///   WAIT:   block until event[aux] >= value, then event[aux] -= value
struct KeaEventOp {
  KeaHead h;                  // aux = event id, 0..KEA_NUM_EVENTS-1
  std::uint32_t value;        // SIGNAL: increment.  WAIT: threshold.
  std::uint8_t reserved[24];
};

/// TRACE: architecturally a NOP.  Emits a debug marker when it retires in
/// `unit`'s queue.  flags = TraceKind.
struct KeaTrace {
  KeaHead h;                  // flags = TraceKind, aux reserved
  std::uint32_t tag;          // user-defined region/marker id
  std::uint32_t payload;      // user-defined
  std::uint8_t reserved[20];
};

// --- DMA -------------------------------------------------------------------

/// DMA_LD (DRAM -> SPM) / DMA_ST (SPM -> DRAM), 3D strided descriptor.
///
/// The transfer is a 3-level nest whose innermost level is a contiguous run
/// of `len0` bytes:
///
///   for i2 in [0, n2):
///     for i1 in [0, n1):
///       copy len0 bytes
///         DRAM side: dram_addr + i2*dram_s2 + i1*dram_s1
///         SPM  side: spm_addr  + i2*spm_s2  + i1*spm_s1
///
/// Total bytes moved = len0 * n1 * n2.  Strides are signed byte offsets and
/// may be negative (for flipped layouts) or zero (broadcast on load only).
struct KeaDma {
  KeaHead h;                  // flags: KEA_DMA_F_*, aux = n2 (1..255)
  std::uint32_t dram_addr;    // DRAM byte address
  std::uint32_t spm_addr;     // SPM_A or SPM_W byte address (per flags)
  std::uint16_t len0;         // contiguous bytes per run, 1..65535
  std::uint16_t n1;           // dim-1 count, 1..65535
  std::int32_t dram_s1;       // DRAM byte stride between dim-1 elements
  std::int32_t dram_s2;       // DRAM byte stride between dim-2 elements
  std::int32_t spm_s1;        // SPM byte stride between dim-1 elements
  std::int32_t spm_s2;        // SPM byte stride between dim-2 elements
};

// --- MXU -------------------------------------------------------------------

/// LOAD_W: load one 16x16 weight tile from SPM_W into weight bank 0 or 1.
///
/// Tile element W[k][n] (k = reduction index, n = output column) is read from
///   w_addr + k*w_row_stride + n*elem_bytes           (int8: elem_bytes = 1)
///   w_addr + k*w_row_stride + (n/2) byte, nibble n%2 (int4)
/// Rows k >= k_rows and columns n >= n_cols are loaded as ZERO.
struct KeaLoadW {
  KeaHead h;                  // flags: KEA_LDW_F_*, aux reserved
  std::uint32_t w_addr;       // SPM_W byte address of tile row 0
  std::int32_t w_row_stride;  // bytes between consecutive k rows
  std::uint8_t k_rows;        // valid reduction rows, 1..16
  std::uint8_t n_cols;        // valid output columns, 1..16
  std::uint8_t reserved[18];
};

/// MATMUL: stream a 2D block of activation rows through the array into ACC.
///
///   for mo in [0, m_outer):
///     for mi in [0, m_inner):
///       a = a_addr + mo*a_outer_stride + mi*a_inner_stride   (SPM_A bytes)
///       q = acc_addr + mo*acc_outer_stride + mi*acc_inner_stride  (ACC words)
///       for n in [0, 16):
///         acc[q + n] = (accumulate ? acc[q + n] : 0)
///                    + sum over k in [0,16) of A[a + k] * W[k][n]
///
/// The array always reads 16 activation elements (16 bytes int8, 8 bytes
/// int4) per row and always writes 16 int32 lanes per row, regardless of
/// LOAD_W's k_rows/n_cols; unused lanes are multiplied by the zeros LOAD_W
/// installed.  The compiler must guarantee the 16 activation bytes are
/// readable and the 16 ACC words are writable.
struct KeaMatmul {
  KeaHead h;                     // flags: KEA_MM_F_*, aux reserved
  std::uint32_t a_addr;          // SPM_A byte address of row (0,0)
  std::uint32_t acc_addr;        // ACC word index of output row (0,0)
  std::uint16_t m_inner;         // inner row count, >= 1
  std::uint16_t m_outer;         // outer row count, >= 1
  std::int32_t a_inner_stride;   // SPM_A bytes between inner rows
  std::int32_t a_outer_stride;   // SPM_A bytes between outer rows
  std::int32_t acc_inner_stride; // ACC words between inner rows (mult. of 16)
  std::int32_t acc_outer_stride; // ACC words between outer rows (mult. of 16)
};

// --- DWU -------------------------------------------------------------------

/// DWCONV: depthwise convolution, SPM_A x SPM_W -> ACC.  int8 only.
///
///   for oh in [0, out_h):
///     for ow in [0, out_w):
///       for c in [0, channels):
///         q = acc_addr + (oh*out_w + ow)*channels + c
///         acc[q] = (accumulate ? acc[q] : 0) + sum over kh,kw of
///             A[a_addr + (oh*S+kh)*a_row_stride + (ow*S+kw)*a_pix_stride + c]
///           * W[w_addr + (kh*K + kw)*channels + c]
///
/// where K = 3 or 5 (flags) and S = 1 or 2 (flags).  Dilation is always 1;
/// the compiler emulates dilation by scaling a_row_stride / a_pix_stride when
/// it slices dilated inputs.  Padding is VALID only: the input tile in SPM_A
/// must already include any zero halo.
///
/// The ACC region is written DENSELY as [out_h][out_w][channels] int32.
struct KeaDwconv {
  KeaHead h;                  // flags: KEA_DW_F_*, aux reserved
  std::uint32_t a_addr;       // SPM_A byte address of input pixel (0,0)
  std::uint32_t w_addr;       // SPM_W byte address of tap plane (0,0)
  std::uint32_t acc_addr;     // ACC word index of output pixel (0,0)
  std::uint16_t out_h;        // >= 1
  std::uint16_t out_w;        // >= 1
  std::uint16_t channels;     // multiple of 16, <= 4096
  std::uint16_t reserved0;
  std::int32_t a_row_stride;  // SPM_A bytes between input rows
  std::int32_t a_pix_stride;  // SPM_A bytes between horizontally adjacent pixels
};

// --- VPU -------------------------------------------------------------------

/// VQUANT: requantize an int32 ACC tile down to int8/int4 in SPM_A.
///
///   for p in [0, num_pixels):
///     for c in [0, channels):
///       v   = acc[acc_addr + p*acc_pix_stride + c] + qp[c].bias
///       out = keaRequantize(v, qp[c].mult, qp[c].shift, out_zp,
///                           clamp_lo, clamp_hi)
///       store out at out_addr + p*out_pix_stride + c   (int8)
///                 or nibble c of out_addr + p*out_pix_stride + c/2 (int4)
///
/// qp is an array of `channels` KeaQuantParam records at qparam_addr in SPM_W.
struct KeaVquant {
  KeaHead h;                     // flags: KEA_VQ_F_*, aux = out_zp (int8)
  std::uint32_t acc_addr;        // ACC word index
  std::uint32_t out_addr;        // SPM_A byte address
  std::uint32_t qparam_addr;     // SPM_W byte address, 4-byte aligned
  std::uint32_t num_pixels;      // >= 1
  std::uint16_t channels;        // >= 1, multiple of 16
  std::int8_t clamp_lo;          // applied after adding out_zp
  std::int8_t clamp_hi;
  std::int32_t acc_pix_stride;   // ACC words between pixels (multiple of 16)
  std::int32_t out_pix_stride;   // SPM_A bytes between pixels
};

/// VADD: quantized elementwise add of two SPM_A tensors into SPM_A.
/// Per-tensor (not per-channel) parameters, read from a single KeaAddParam
/// record at param_addr in SPM_W.  See keaQuantizedAdd().
struct KeaVadd {
  KeaHead h;                  // flags/aux reserved
  std::uint32_t a_addr;       // SPM_A
  std::uint32_t b_addr;       // SPM_A
  std::uint32_t out_addr;     // SPM_A
  std::uint32_t param_addr;   // SPM_W, 4-byte aligned
  std::uint32_t num_elems;    // element count (channels are flattened in)
  std::int8_t clamp_lo;
  std::int8_t clamp_hi;
  std::uint8_t reserved[6];
};

/// VPOOL: max or average pooling, SPM_A -> SPM_A, int8 only, VALID padding.
///
/// REQUIRES input and output quantization parameters to be IDENTICAL (same
/// scale, same zero point).  Average pooling is then exactly
/// round-half-away-from-zero(sum / (kh*kw)) over the raw int8 values.
/// If scales differ the compiler must emit a separate rescale.
///
///   for oh, ow, c:
///     window over in_addr + (oh*stride_h + i)*in_row_stride
///                         + (ow*stride_w + j)*channels + c
///     store at out_addr + oh*out_row_stride + ow*channels + c
struct KeaVpool {
  KeaHead h;                     // flags: KEA_VP_F_*, aux reserved
  std::uint32_t in_addr;         // SPM_A
  std::uint32_t out_addr;        // SPM_A
  std::uint16_t out_h;           // >= 1
  std::uint16_t out_w;           // >= 1
  std::uint16_t channels;        // >= 1
  std::uint8_t kh;               // 1..32
  std::uint8_t kw;               // 1..32
  std::uint8_t stride_h;         // 1..8
  std::uint8_t stride_w;         // 1..8
  std::uint16_t reserved0;
  std::int32_t in_row_stride;    // SPM_A bytes between input rows
  std::int32_t out_row_stride;   // SPM_A bytes between output rows
};

/// VCOPY: 2D strided byte copy or constant fill within SPM_A/SPM_W.
///
///   for r in [0, rows):
///     dst[dst_addr + r*dst_row_stride .. +row_bytes) =
///        fill ? fill_value : src[src_addr + r*src_row_stride .. +row_bytes)
///
/// Fill mode is how the compiler zeroes conv halos.  ACC is NOT reachable by
/// VCOPY; use MATMUL/DWCONV with accumulate=0 to initialize accumulators.
struct KeaVcopy {
  KeaHead h;                     // flags: KEA_VC_F_*, aux = fill_value (int8)
  std::uint32_t src_addr;
  std::uint32_t dst_addr;
  std::uint32_t row_bytes;       // >= 1
  std::uint32_t rows;            // >= 1
  std::int32_t src_row_stride;
  std::int32_t dst_row_stride;
  std::uint8_t reserved[4];
};

// ---------------------------------------------------------------------------
// Parameter blocks (live in SPM_W, staged there by DMA)
// ---------------------------------------------------------------------------

/// Per-output-channel requantization parameters consumed by VQUANT.
/// An array of `channels` of these, 4-byte aligned, packed, in SPM_W.
struct KeaQuantParam {
  std::int32_t bias;   // int32 bias added to the accumulator
  std::int32_t mult;   // normalized Q31 multiplier, in [2^30, 2^31)
  std::int32_t shift;  // >0 right shift after the multiply; <0 left shift before
};

/// Per-tensor parameters consumed by VADD.
struct KeaAddParam {
  std::int32_t a_mult;
  std::int32_t b_mult;
  std::int32_t o_mult;
  std::int8_t a_shift;
  std::int8_t b_shift;
  std::int8_t o_shift;
  std::int8_t reserved0;
  std::int8_t a_zp;
  std::int8_t b_zp;
  std::int8_t o_zp;
  std::int8_t reserved1;
};

#pragma pack(pop)

// ---------------------------------------------------------------------------
// Size / layout guarantees
// ---------------------------------------------------------------------------

static_assert(sizeof(KeaInstr) == 32, "instruction must be 32 bytes");
static_assert(sizeof(KeaInstr) == KEA_INSTR_BYTES, "KEA_INSTR_BYTES mismatch");
static_assert(sizeof(KeaHead) == 4, "header must be 4 bytes");
static_assert(sizeof(KeaNop) == 32, "KeaNop must be 32 bytes");
static_assert(sizeof(KeaHalt) == 32, "KeaHalt must be 32 bytes");
static_assert(sizeof(KeaEventOp) == 32, "KeaEventOp must be 32 bytes");
static_assert(sizeof(KeaTrace) == 32, "KeaTrace must be 32 bytes");
static_assert(sizeof(KeaDma) == 32, "KeaDma must be 32 bytes");
static_assert(sizeof(KeaLoadW) == 32, "KeaLoadW must be 32 bytes");
static_assert(sizeof(KeaMatmul) == 32, "KeaMatmul must be 32 bytes");
static_assert(sizeof(KeaDwconv) == 32, "KeaDwconv must be 32 bytes");
static_assert(sizeof(KeaVquant) == 32, "KeaVquant must be 32 bytes");
static_assert(sizeof(KeaVadd) == 32, "KeaVadd must be 32 bytes");
static_assert(sizeof(KeaVpool) == 32, "KeaVpool must be 32 bytes");
static_assert(sizeof(KeaVcopy) == 32, "KeaVcopy must be 32 bytes");
static_assert(sizeof(KeaQuantParam) == 12, "KeaQuantParam must be 12 bytes");
static_assert(sizeof(KeaAddParam) == 20, "KeaAddParam must be 20 bytes");
static_assert(sizeof(std::int32_t) == 4 && sizeof(std::uint16_t) == 2,
              "unexpected fundamental type sizes");

// Field offsets that the docs, the assembler and any external tool depend on.
static_assert(offsetof(KeaDma, dram_addr) == 4, "KeaDma layout drift");
static_assert(offsetof(KeaDma, spm_addr) == 8, "KeaDma layout drift");
static_assert(offsetof(KeaDma, len0) == 12, "KeaDma layout drift");
static_assert(offsetof(KeaDma, n1) == 14, "KeaDma layout drift");
static_assert(offsetof(KeaDma, dram_s1) == 16, "KeaDma layout drift");
static_assert(offsetof(KeaDma, spm_s2) == 28, "KeaDma layout drift");
static_assert(offsetof(KeaMatmul, a_addr) == 4, "KeaMatmul layout drift");
static_assert(offsetof(KeaMatmul, acc_addr) == 8, "KeaMatmul layout drift");
static_assert(offsetof(KeaMatmul, m_inner) == 12, "KeaMatmul layout drift");
static_assert(offsetof(KeaMatmul, acc_outer_stride) == 28, "KeaMatmul layout drift");
static_assert(offsetof(KeaLoadW, k_rows) == 12, "KeaLoadW layout drift");
static_assert(offsetof(KeaDwconv, a_pix_stride) == 28, "KeaDwconv layout drift");
static_assert(offsetof(KeaVquant, out_pix_stride) == 28, "KeaVquant layout drift");
static_assert(offsetof(KeaVpool, in_row_stride) == 24, "KeaVpool layout drift");
static_assert(offsetof(KeaVcopy, src_row_stride) == 20, "KeaVcopy layout drift");

// ===========================================================================
// 3. Encode / decode
// ===========================================================================

namespace detail {
inline void byteCopy(void* dst, const void* src, std::size_t n) {
  auto* d = static_cast<std::uint8_t*>(dst);
  const auto* s = static_cast<const std::uint8_t*>(src);
  for (std::size_t i = 0; i < n; ++i) d[i] = s[i];
}
}  // namespace detail

/// Reinterpret a typed instruction overlay as the raw 32-byte word.
/// Any trailing bytes of `T` shorter than 32 are zero-filled (all overlays
/// are exactly 32 bytes, so this is a straight copy).
template <typename T>
inline KeaInstr keaEncode(const T& in) {
  static_assert(sizeof(T) <= 32, "overlay larger than an instruction");
  KeaInstr out{};
  for (std::size_t i = 0; i < 32; ++i) out.b[i] = 0;
  detail::byteCopy(out.b, &in, sizeof(T));
  return out;
}

/// Reinterpret a raw instruction word as a typed overlay.  Does not validate;
/// call keaValidate() first if the bytes came from an untrusted artifact.
template <typename T>
inline T keaDecode(const KeaInstr& in) {
  static_assert(sizeof(T) <= 32, "overlay larger than an instruction");
  T out{};
  detail::byteCopy(&out, in.b, sizeof(T));
  return out;
}

/// Field accessors for the common header, usable before the opcode is known.
inline Opcode keaOpcode(const KeaInstr& i) { return static_cast<Opcode>(i.b[0]); }
inline Unit keaUnit(const KeaInstr& i) { return static_cast<Unit>(i.b[1]); }
inline std::uint8_t keaFlags(const KeaInstr& i) { return i.b[2]; }
inline std::uint8_t keaAux(const KeaInstr& i) { return i.b[3]; }

/// The unit that owns an opcode.  DMA opcodes may target DMA0 or DMA1 and
/// report DMA0 here; CTRL opcodes may target any unit and report CTRL.
constexpr Unit defaultUnitFor(Opcode op) {
  switch (static_cast<std::uint8_t>(op) >> 4) {
    case 0x0: return Unit::CTRL;
    case 0x1: return Unit::DMA0;
    case 0x2: return Unit::MXU;
    case 0x3: return Unit::DWU;
    case 0x4: return Unit::VPU;
    default:  return Unit::CTRL;
  }
}

/// True if `op` may be placed in any queue (CTRL-class opcodes) rather than
/// only in its owning unit's queue.
constexpr bool keaIsQueueAgnostic(Opcode op) {
  return op == Opcode::NOP || op == Opcode::SIGNAL || op == Opcode::WAIT ||
         op == Opcode::TRACE;
}

// ---------------------------------------------------------------------------
// Convenience constructors.  These are the sanctioned way to build code.
// ---------------------------------------------------------------------------

inline KeaInstr keaMakeNop(Unit unit, std::uint32_t cycles = 0) {
  KeaNop n{};
  n.h.opcode = static_cast<std::uint8_t>(Opcode::NOP);
  n.h.unit = static_cast<std::uint8_t>(unit);
  n.cycles = cycles;
  return keaEncode(n);
}

inline KeaInstr keaMakeHalt(std::uint32_t exit_code = 0) {
  KeaHalt n{};
  n.h.opcode = static_cast<std::uint8_t>(Opcode::HALT);
  n.h.unit = static_cast<std::uint8_t>(Unit::CTRL);
  n.exit_code = exit_code;
  return keaEncode(n);
}

inline KeaInstr keaMakeSignal(Unit unit, std::uint8_t event, std::uint32_t inc = 1) {
  KeaEventOp n{};
  n.h.opcode = static_cast<std::uint8_t>(Opcode::SIGNAL);
  n.h.unit = static_cast<std::uint8_t>(unit);
  n.h.aux = event;
  n.value = inc;
  return keaEncode(n);
}

inline KeaInstr keaMakeWait(Unit unit, std::uint8_t event, std::uint32_t threshold = 1) {
  KeaEventOp n{};
  n.h.opcode = static_cast<std::uint8_t>(Opcode::WAIT);
  n.h.unit = static_cast<std::uint8_t>(unit);
  n.h.aux = event;
  n.value = threshold;
  return keaEncode(n);
}

inline KeaInstr keaMakeTrace(Unit unit, TraceKind kind, std::uint32_t tag,
                             std::uint32_t payload = 0) {
  KeaTrace n{};
  n.h.opcode = static_cast<std::uint8_t>(Opcode::TRACE);
  n.h.unit = static_cast<std::uint8_t>(unit);
  n.h.flags = static_cast<std::uint8_t>(kind);
  n.tag = tag;
  n.payload = payload;
  return keaEncode(n);
}

/// Build a DMA_LD or DMA_ST.  `is_load` picks the direction; `unit` must be
/// Unit::DMA0 or Unit::DMA1; `spm_is_w` selects SPM_W over SPM_A.
inline KeaInstr keaMakeDma(bool is_load, Unit unit, bool spm_is_w,
                           std::uint32_t dram_addr, std::uint32_t spm_addr,
                           std::uint16_t len0, std::uint16_t n1, std::uint8_t n2,
                           std::int32_t dram_s1, std::int32_t dram_s2,
                           std::int32_t spm_s1, std::int32_t spm_s2) {
  KeaDma n{};
  n.h.opcode = static_cast<std::uint8_t>(is_load ? Opcode::DMA_LD : Opcode::DMA_ST);
  n.h.unit = static_cast<std::uint8_t>(unit);
  n.h.flags = spm_is_w ? KEA_DMA_F_SPM_W : 0u;
  n.h.aux = n2;
  n.dram_addr = dram_addr;
  n.spm_addr = spm_addr;
  n.len0 = len0;
  n.n1 = n1;
  n.dram_s1 = dram_s1;
  n.dram_s2 = dram_s2;
  n.spm_s1 = spm_s1;
  n.spm_s2 = spm_s2;
  return keaEncode(n);
}

inline KeaInstr keaMakeLoadW(std::uint32_t w_addr, std::int32_t w_row_stride,
                             std::uint8_t k_rows, std::uint8_t n_cols,
                             std::uint8_t bank, bool int4) {
  KeaLoadW n{};
  n.h.opcode = static_cast<std::uint8_t>(Opcode::LOAD_W);
  n.h.unit = static_cast<std::uint8_t>(Unit::MXU);
  n.h.flags = static_cast<std::uint8_t>((int4 ? KEA_LDW_F_INT4 : 0u) |
                                        (bank ? KEA_LDW_F_BANK1 : 0u));
  n.w_addr = w_addr;
  n.w_row_stride = w_row_stride;
  n.k_rows = k_rows;
  n.n_cols = n_cols;
  return keaEncode(n);
}

inline KeaInstr keaMakeMatmul(std::uint32_t a_addr, std::int32_t a_inner_stride,
                              std::int32_t a_outer_stride, std::uint16_t m_inner,
                              std::uint16_t m_outer, std::uint32_t acc_addr,
                              std::int32_t acc_inner_stride,
                              std::int32_t acc_outer_stride, std::uint8_t bank,
                              bool accumulate, bool int4) {
  KeaMatmul n{};
  n.h.opcode = static_cast<std::uint8_t>(Opcode::MATMUL);
  n.h.unit = static_cast<std::uint8_t>(Unit::MXU);
  n.h.flags = static_cast<std::uint8_t>((accumulate ? KEA_MM_F_ACCUMULATE : 0u) |
                                        (int4 ? KEA_MM_F_INT4 : 0u) |
                                        (bank ? KEA_MM_F_BANK1 : 0u));
  n.a_addr = a_addr;
  n.acc_addr = acc_addr;
  n.m_inner = m_inner;
  n.m_outer = m_outer;
  n.a_inner_stride = a_inner_stride;
  n.a_outer_stride = a_outer_stride;
  n.acc_inner_stride = acc_inner_stride;
  n.acc_outer_stride = acc_outer_stride;
  return keaEncode(n);
}

inline KeaInstr keaMakeDwconv(std::uint32_t a_addr, std::uint32_t w_addr,
                              std::uint32_t acc_addr, std::uint16_t out_h,
                              std::uint16_t out_w, std::uint16_t channels,
                              std::int32_t a_row_stride, std::int32_t a_pix_stride,
                              std::uint8_t kernel /*3 or 5*/,
                              std::uint8_t stride /*1 or 2*/, bool accumulate) {
  KeaDwconv n{};
  n.h.opcode = static_cast<std::uint8_t>(Opcode::DWCONV);
  n.h.unit = static_cast<std::uint8_t>(Unit::DWU);
  n.h.flags = static_cast<std::uint8_t>((kernel == 5 ? KEA_DW_F_KERNEL5 : 0u) |
                                        (stride == 2 ? KEA_DW_F_STRIDE2 : 0u) |
                                        (accumulate ? KEA_DW_F_ACCUMULATE : 0u));
  n.a_addr = a_addr;
  n.w_addr = w_addr;
  n.acc_addr = acc_addr;
  n.out_h = out_h;
  n.out_w = out_w;
  n.channels = channels;
  n.a_row_stride = a_row_stride;
  n.a_pix_stride = a_pix_stride;
  return keaEncode(n);
}

inline KeaInstr keaMakeVquant(std::uint32_t acc_addr, std::uint32_t out_addr,
                              std::uint32_t qparam_addr, std::uint32_t num_pixels,
                              std::uint16_t channels, std::int32_t acc_pix_stride,
                              std::int32_t out_pix_stride, std::int8_t out_zp,
                              std::int8_t clamp_lo, std::int8_t clamp_hi,
                              bool out_int4 = false) {
  KeaVquant n{};
  n.h.opcode = static_cast<std::uint8_t>(Opcode::VQUANT);
  n.h.unit = static_cast<std::uint8_t>(Unit::VPU);
  n.h.flags = out_int4 ? KEA_VQ_F_OUT_INT4 : 0u;
  n.h.aux = static_cast<std::uint8_t>(out_zp);
  n.acc_addr = acc_addr;
  n.out_addr = out_addr;
  n.qparam_addr = qparam_addr;
  n.num_pixels = num_pixels;
  n.channels = channels;
  n.clamp_lo = clamp_lo;
  n.clamp_hi = clamp_hi;
  n.acc_pix_stride = acc_pix_stride;
  n.out_pix_stride = out_pix_stride;
  return keaEncode(n);
}

inline KeaInstr keaMakeVadd(std::uint32_t a_addr, std::uint32_t b_addr,
                            std::uint32_t out_addr, std::uint32_t param_addr,
                            std::uint32_t num_elems, std::int8_t clamp_lo,
                            std::int8_t clamp_hi) {
  KeaVadd n{};
  n.h.opcode = static_cast<std::uint8_t>(Opcode::VADD);
  n.h.unit = static_cast<std::uint8_t>(Unit::VPU);
  n.a_addr = a_addr;
  n.b_addr = b_addr;
  n.out_addr = out_addr;
  n.param_addr = param_addr;
  n.num_elems = num_elems;
  n.clamp_lo = clamp_lo;
  n.clamp_hi = clamp_hi;
  return keaEncode(n);
}

inline KeaInstr keaMakeVpool(bool avg, std::uint32_t in_addr, std::uint32_t out_addr,
                             std::uint16_t out_h, std::uint16_t out_w,
                             std::uint16_t channels, std::uint8_t kh, std::uint8_t kw,
                             std::uint8_t stride_h, std::uint8_t stride_w,
                             std::int32_t in_row_stride, std::int32_t out_row_stride) {
  KeaVpool n{};
  n.h.opcode = static_cast<std::uint8_t>(Opcode::VPOOL);
  n.h.unit = static_cast<std::uint8_t>(Unit::VPU);
  n.h.flags = avg ? KEA_VP_F_AVG : 0u;
  n.in_addr = in_addr;
  n.out_addr = out_addr;
  n.out_h = out_h;
  n.out_w = out_w;
  n.channels = channels;
  n.kh = kh;
  n.kw = kw;
  n.stride_h = stride_h;
  n.stride_w = stride_w;
  n.in_row_stride = in_row_stride;
  n.out_row_stride = out_row_stride;
  return keaEncode(n);
}

inline KeaInstr keaMakeVcopy(std::uint32_t src_addr, std::uint32_t dst_addr,
                             std::uint32_t row_bytes, std::uint32_t rows,
                             std::int32_t src_row_stride, std::int32_t dst_row_stride,
                             bool src_is_w = false, bool dst_is_w = false) {
  KeaVcopy n{};
  n.h.opcode = static_cast<std::uint8_t>(Opcode::VCOPY);
  n.h.unit = static_cast<std::uint8_t>(Unit::VPU);
  n.h.flags = static_cast<std::uint8_t>((src_is_w ? KEA_VC_F_SRC_W : 0u) |
                                        (dst_is_w ? KEA_VC_F_DST_W : 0u));
  n.src_addr = src_addr;
  n.dst_addr = dst_addr;
  n.row_bytes = row_bytes;
  n.rows = rows;
  n.src_row_stride = src_row_stride;
  n.dst_row_stride = dst_row_stride;
  return keaEncode(n);
}

inline KeaInstr keaMakeVfill(std::uint32_t dst_addr, std::uint32_t row_bytes,
                             std::uint32_t rows, std::int32_t dst_row_stride,
                             std::int8_t value, bool dst_is_w = false) {
  KeaVcopy n{};
  n.h.opcode = static_cast<std::uint8_t>(Opcode::VCOPY);
  n.h.unit = static_cast<std::uint8_t>(Unit::VPU);
  n.h.flags = static_cast<std::uint8_t>(KEA_VC_F_FILL | (dst_is_w ? KEA_VC_F_DST_W : 0u));
  n.h.aux = static_cast<std::uint8_t>(value);
  n.dst_addr = dst_addr;
  n.row_bytes = row_bytes;
  n.rows = rows;
  n.dst_row_stride = dst_row_stride;
  return keaEncode(n);
}

// ===========================================================================
// 4. int4 packing
// ===========================================================================
//
// A signed 4-bit vector is stored two elements per byte: element 2*i in the
// LOW nibble (bits 0..3) of byte i, element 2*i+1 in the HIGH nibble.  An
// int4 vector always starts on a byte boundary, i.e. at an even element
// index.  Values are sign-extended from 4 bits, range [-8, 7].

inline std::int8_t keaUnpackInt4(std::uint8_t byte, unsigned which) {
  const std::uint8_t nib = (which & 1u) ? static_cast<std::uint8_t>(byte >> 4)
                                        : static_cast<std::uint8_t>(byte & 0x0Fu);
  return static_cast<std::int8_t>((nib & 0x08u) ? static_cast<std::int8_t>(nib | 0xF0u)
                                                : static_cast<std::int8_t>(nib));
}

inline std::uint8_t keaPackInt4(std::int8_t lo, std::int8_t hi) {
  return static_cast<std::uint8_t>((static_cast<std::uint8_t>(lo) & 0x0Fu) |
                                   ((static_cast<std::uint8_t>(hi) & 0x0Fu) << 4));
}

inline constexpr std::int32_t KEA_INT4_MIN = -8;
inline constexpr std::int32_t KEA_INT4_MAX = 7;

// ===========================================================================
// 5. Normative arithmetic reference
// ===========================================================================
//
// The simulator, the compiler's constant folder, and any reference kernel
// MUST use these exact routines.  They match gemmlowp / TFLite semantics.

/// Saturating rounding doubling high multiply (gemmlowp SRDHM).
/// Computes round(a * b / 2^31) with saturation at the single overflow case.
inline std::int32_t keaSrdhm(std::int32_t a, std::int32_t b) {
  const std::int32_t kMin = static_cast<std::int32_t>(0x80000000u);
  if (a == kMin && b == kMin) return 0x7FFFFFFF;
  const std::int64_t ab = static_cast<std::int64_t>(a) * static_cast<std::int64_t>(b);
  const std::int64_t nudge = (ab >= 0) ? (1LL << 30) : (1LL - (1LL << 30));
  return static_cast<std::int32_t>((ab + nudge) / (1LL << 31));
}

/// Rounding divide by a power of two, round half away from zero
/// (gemmlowp RoundingDivideByPOT).  `exp` must be >= 0.
inline std::int32_t keaRdpot(std::int32_t x, int exp) {
  if (exp <= 0) return x;
  const std::int32_t mask = static_cast<std::int32_t>((1u << exp) - 1u);
  const std::int32_t remainder = x & mask;
  const std::int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
  return (x >> exp) + ((remainder > threshold) ? 1 : 0);
}

inline std::int32_t keaClamp(std::int32_t v, std::int32_t lo, std::int32_t hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

/// The KEA-1 requantization pipeline, exactly as VQUANT implements it.
/// `v` is (accumulator + bias).
inline std::int32_t keaRequantize(std::int32_t v, std::int32_t mult, std::int32_t shift,
                                  std::int32_t out_zp, std::int32_t clamp_lo,
                                  std::int32_t clamp_hi) {
  if (shift < 0) v = static_cast<std::int32_t>(static_cast<std::uint32_t>(v) << (-shift));
  std::int32_t p = keaSrdhm(v, mult);
  if (shift > 0) p = keaRdpot(p, shift);
  return keaClamp(p + out_zp, clamp_lo, clamp_hi);
}

/// The KEA-1 quantized add, exactly as VADD implements it.
inline std::int32_t keaQuantizedAdd(std::int32_t a, std::int32_t b,
                                    const KeaAddParam& p, std::int32_t clamp_lo,
                                    std::int32_t clamp_hi) {
  const std::int32_t sa =
      static_cast<std::int32_t>(static_cast<std::uint32_t>(a - p.a_zp) << KEA_VADD_LEFT_SHIFT);
  const std::int32_t sb =
      static_cast<std::int32_t>(static_cast<std::uint32_t>(b - p.b_zp) << KEA_VADD_LEFT_SHIFT);
  const std::int32_t xa = keaRdpot(keaSrdhm(sa, p.a_mult), p.a_shift);
  const std::int32_t xb = keaRdpot(keaSrdhm(sb, p.b_mult), p.b_shift);
  const std::int32_t s = xa + xb;
  const std::int32_t o = keaRdpot(keaSrdhm(s, p.o_mult), p.o_shift) + p.o_zp;
  return keaClamp(o, clamp_lo, clamp_hi);
}

/// Average pooling reduction used by VPOOL (round half away from zero).
/// `sum` is the raw int8 sum over the window, `count` = kh*kw.
inline std::int32_t keaPoolAverage(std::int32_t sum, std::int32_t count) {
  if (count <= 0) return 0;
  return (sum >= 0) ? ((sum + count / 2) / count) : -(((-sum) + count / 2) / count);
}

// ===========================================================================
// 6. Names and disassembly metadata
// ===========================================================================

/// Static description of an opcode, sufficient to drive a text disassembler
/// without any per-opcode code in the printer.
struct KeaOpInfo {
  const char* mnemonic;   ///< e.g. "MATMUL"
  const char* operands;   ///< comma-separated field names, in print order
  Unit unit;              ///< owning unit (Unit::CTRL for queue-agnostic ops)
  bool valid;             ///< false for unassigned encodings
};

inline KeaOpInfo keaOpInfo(Opcode op) {
  switch (op) {
    case Opcode::NOP:
      return {"NOP", "unit,cycles", Unit::CTRL, true};
    case Opcode::HALT:
      return {"HALT", "exit_code", Unit::CTRL, true};
    case Opcode::SIGNAL:
      return {"SIGNAL", "unit,event,inc", Unit::CTRL, true};
    case Opcode::WAIT:
      return {"WAIT", "unit,event,threshold", Unit::CTRL, true};
    case Opcode::TRACE:
      return {"TRACE", "unit,kind,tag,payload", Unit::CTRL, true};
    case Opcode::DMA_LD:
      return {"DMA_LD",
              "unit,spm_space,dram_addr,spm_addr,len0,n1,n2,dram_s1,dram_s2,spm_s1,spm_s2",
              Unit::DMA0, true};
    case Opcode::DMA_ST:
      return {"DMA_ST",
              "unit,spm_space,dram_addr,spm_addr,len0,n1,n2,dram_s1,dram_s2,spm_s1,spm_s2",
              Unit::DMA0, true};
    case Opcode::LOAD_W:
      return {"LOAD_W", "w_addr,w_row_stride,k_rows,n_cols,bank,dtype", Unit::MXU, true};
    case Opcode::MATMUL:
      return {"MATMUL",
              "a_addr,a_inner_stride,a_outer_stride,m_inner,m_outer,acc_addr,"
              "acc_inner_stride,acc_outer_stride,bank,acc_mode,dtype",
              Unit::MXU, true};
    case Opcode::DWCONV:
      return {"DWCONV",
              "a_addr,w_addr,acc_addr,out_h,out_w,channels,a_row_stride,a_pix_stride,"
              "kernel,stride,acc_mode",
              Unit::DWU, true};
    case Opcode::VQUANT:
      return {"VQUANT",
              "acc_addr,out_addr,qparam_addr,num_pixels,channels,acc_pix_stride,"
              "out_pix_stride,out_zp,clamp_lo,clamp_hi,dtype",
              Unit::VPU, true};
    case Opcode::VADD:
      return {"VADD", "a_addr,b_addr,out_addr,param_addr,num_elems,clamp_lo,clamp_hi",
              Unit::VPU, true};
    case Opcode::VPOOL:
      return {"VPOOL",
              "mode,in_addr,out_addr,out_h,out_w,channels,kh,kw,stride_h,stride_w,"
              "in_row_stride,out_row_stride",
              Unit::VPU, true};
    case Opcode::VCOPY:
      return {"VCOPY",
              "mode,src_space,dst_space,src_addr,dst_addr,row_bytes,rows,"
              "src_row_stride,dst_row_stride,fill_value",
              Unit::VPU, true};
    default:
      return {"<invalid>", "", Unit::CTRL, false};
  }
}

inline const char* opcodeName(Opcode op) { return keaOpInfo(op).mnemonic; }

inline const char* unitName(Unit u) {
  switch (u) {
    case Unit::MXU:  return "MXU";
    case Unit::DWU:  return "DWU";
    case Unit::VPU:  return "VPU";
    case Unit::DMA0: return "DMA0";
    case Unit::DMA1: return "DMA1";
    case Unit::CTRL: return "CTRL";
  }
  return "<invalid-unit>";
}

inline const char* dtypeName(DType d) {
  switch (d) {
    case DType::INT8:  return "int8";
    case DType::INT4:  return "int4";
    case DType::INT32: return "int32";
  }
  return "<invalid-dtype>";
}

inline const char* spaceName(Space s) {
  switch (s) {
    case Space::SPM_A: return "SPM_A";
    case Space::SPM_W: return "SPM_W";
    case Space::ACC:   return "ACC";
    case Space::DRAM:  return "DRAM";
  }
  return "<invalid-space>";
}

inline const char* traceKindName(TraceKind k) {
  switch (k) {
    case TraceKind::MARKER:       return "marker";
    case TraceKind::REGION_BEGIN: return "begin";
    case TraceKind::REGION_END:   return "end";
  }
  return "<invalid-trace-kind>";
}

// ===========================================================================
// 7. Static validator
// ===========================================================================

/// Structural validation of a single instruction against the KEA-1 rules in
/// docs/ISA.md.  Returns nullptr if the instruction is well formed, otherwise
/// a static string naming the first violated rule.
///
/// This checks only what is decidable from one instruction: opcode existence,
/// unit legality, field ranges, alignment, and address-space bounds.  It does
/// NOT check aliasing, event protocol correctness, or schedule validity.
inline const char* keaValidate(const KeaInstr& in) {
  const Opcode op = keaOpcode(in);
  const KeaOpInfo info = keaOpInfo(op);
  if (!info.valid) return "unknown opcode";

  const std::uint8_t unit_raw = in.b[1];
  if (unit_raw >= static_cast<std::uint8_t>(KEA_NUM_UNITS)) return "unit id out of range";
  const Unit unit = static_cast<Unit>(unit_raw);

  // Unit legality.
  if (op == Opcode::HALT) {
    if (unit != Unit::CTRL) return "HALT must target CTRL";
  } else if (keaIsQueueAgnostic(op)) {
    // NOP/SIGNAL/WAIT/TRACE may target any unit, including CTRL.
  } else if (op == Opcode::DMA_LD || op == Opcode::DMA_ST) {
    if (unit != Unit::DMA0 && unit != Unit::DMA1) return "DMA must target DMA0 or DMA1";
  } else {
    if (unit != info.unit) return "opcode targets the wrong unit";
  }

  switch (op) {
    case Opcode::NOP:
    case Opcode::HALT:
    case Opcode::TRACE:
      return nullptr;

    case Opcode::SIGNAL:
    case Opcode::WAIT: {
      if (in.b[3] >= static_cast<std::uint8_t>(KEA_NUM_EVENTS)) return "event id out of range";
      const KeaEventOp e = keaDecode<KeaEventOp>(in);
      if (e.value > KEA_EVENT_MAX) return "event value too large";
      return nullptr;
    }

    case Opcode::DMA_LD:
    case Opcode::DMA_ST: {
      const KeaDma d = keaDecode<KeaDma>(in);
      if (d.len0 == 0) return "DMA len0 must be >= 1";
      if (d.n1 == 0) return "DMA n1 must be >= 1";
      if (d.h.aux == 0) return "DMA n2 must be >= 1";
      if (d.h.flags & ~KEA_DMA_F_SPM_W) return "DMA reserved flag bits set";
      const std::size_t spm_bytes =
          (d.h.flags & KEA_DMA_F_SPM_W) ? KEA_SPM_W_BYTES : KEA_SPM_A_BYTES;
      if (d.spm_addr >= spm_bytes) return "DMA spm_addr out of range";
      return nullptr;
    }

    case Opcode::LOAD_W: {
      const KeaLoadW w = keaDecode<KeaLoadW>(in);
      if (w.h.flags & ~(KEA_LDW_F_INT4 | KEA_LDW_F_BANK1)) return "LOAD_W reserved flags set";
      if (w.k_rows == 0 || w.k_rows > KEA_MXU_K) return "LOAD_W k_rows must be 1..16";
      if (w.n_cols == 0 || w.n_cols > KEA_MXU_N) return "LOAD_W n_cols must be 1..16";
      const std::uint32_t align =
          (w.h.flags & KEA_LDW_F_INT4) ? KEA_ALIGN_LOADW_INT4 : KEA_ALIGN_LOADW_INT8;
      if (!keaIsAligned(w.w_addr, align)) return "LOAD_W w_addr misaligned";
      if (w.w_row_stride < 0 || !keaIsAligned(static_cast<std::uint32_t>(w.w_row_stride), align))
        return "LOAD_W w_row_stride misaligned or negative";
      if (w.w_addr >= KEA_SPM_W_BYTES) return "LOAD_W w_addr out of range";
      return nullptr;
    }

    case Opcode::MATMUL: {
      const KeaMatmul m = keaDecode<KeaMatmul>(in);
      if (m.h.flags & ~(KEA_MM_F_ACCUMULATE | KEA_MM_F_INT4 | KEA_MM_F_BANK1))
        return "MATMUL reserved flags set";
      if (m.m_inner == 0 || m.m_outer == 0) return "MATMUL m_inner/m_outer must be >= 1";
      if (static_cast<std::uint32_t>(m.m_inner) * m.m_outer > KEA_MXU_MAX_ROWS)
        return "MATMUL row count exceeds ACC capacity";
      if (m.a_addr >= KEA_SPM_A_BYTES) return "MATMUL a_addr out of range";
      if (m.acc_addr >= KEA_ACC_WORDS) return "MATMUL acc_addr out of range";
      if (!keaIsAligned(m.acc_addr, KEA_ALIGN_ACC_WORDS)) return "MATMUL acc_addr misaligned";
      if (m.acc_inner_stride % static_cast<std::int32_t>(KEA_ALIGN_ACC_WORDS))
        return "MATMUL acc_inner_stride must be a multiple of 16 words";
      if (m.acc_outer_stride % static_cast<std::int32_t>(KEA_ALIGN_ACC_WORDS))
        return "MATMUL acc_outer_stride must be a multiple of 16 words";
      return nullptr;
    }

    case Opcode::DWCONV: {
      const KeaDwconv d = keaDecode<KeaDwconv>(in);
      if (d.h.flags & ~(KEA_DW_F_KERNEL5 | KEA_DW_F_STRIDE2 | KEA_DW_F_ACCUMULATE))
        return "DWCONV reserved flags set";
      if (d.out_h == 0 || d.out_w == 0) return "DWCONV out_h/out_w must be >= 1";
      if (d.channels == 0) return "DWCONV channels must be >= 1";
      if (!keaIsAligned(d.channels, KEA_DWU_LANES)) return "DWCONV channels must be a multiple of 16";
      if (d.a_addr >= KEA_SPM_A_BYTES) return "DWCONV a_addr out of range";
      if (d.w_addr >= KEA_SPM_W_BYTES) return "DWCONV w_addr out of range";
      if (d.acc_addr >= KEA_ACC_WORDS) return "DWCONV acc_addr out of range";
      if (!keaIsAligned(d.acc_addr, KEA_ALIGN_ACC_WORDS)) return "DWCONV acc_addr misaligned";
      {
        const std::uint64_t need =
            static_cast<std::uint64_t>(d.out_h) * d.out_w * d.channels;
        if (d.acc_addr + need > KEA_ACC_WORDS) return "DWCONV output exceeds ACC";
      }
      return nullptr;
    }

    case Opcode::VQUANT: {
      const KeaVquant v = keaDecode<KeaVquant>(in);
      if (v.h.flags & ~KEA_VQ_F_OUT_INT4) return "VQUANT reserved flags set";
      if (v.num_pixels == 0) return "VQUANT num_pixels must be >= 1";
      if (v.channels == 0) return "VQUANT channels must be >= 1";
      if (!keaIsAligned(v.channels, KEA_VPU_LANES)) return "VQUANT channels must be a multiple of 16";
      if (v.acc_addr >= KEA_ACC_WORDS) return "VQUANT acc_addr out of range";
      if (!keaIsAligned(v.acc_addr, KEA_ALIGN_ACC_WORDS)) return "VQUANT acc_addr misaligned";
      if (v.acc_pix_stride % static_cast<std::int32_t>(KEA_ALIGN_ACC_WORDS))
        return "VQUANT acc_pix_stride must be a multiple of 16 words";
      if (v.out_addr >= KEA_SPM_A_BYTES) return "VQUANT out_addr out of range";
      if (!keaIsAligned(v.qparam_addr, KEA_ALIGN_QPARAM)) return "VQUANT qparam_addr misaligned";
      if (v.qparam_addr >= KEA_SPM_W_BYTES) return "VQUANT qparam_addr out of range";
      if (v.clamp_lo > v.clamp_hi) return "VQUANT clamp_lo > clamp_hi";
      return nullptr;
    }

    case Opcode::VADD: {
      const KeaVadd v = keaDecode<KeaVadd>(in);
      if (v.h.flags != 0) return "VADD reserved flags set";
      if (v.num_elems == 0) return "VADD num_elems must be >= 1";
      if (v.a_addr >= KEA_SPM_A_BYTES || v.b_addr >= KEA_SPM_A_BYTES ||
          v.out_addr >= KEA_SPM_A_BYTES)
        return "VADD address out of range";
      if (!keaIsAligned(v.param_addr, KEA_ALIGN_QPARAM)) return "VADD param_addr misaligned";
      if (v.param_addr >= KEA_SPM_W_BYTES) return "VADD param_addr out of range";
      if (v.clamp_lo > v.clamp_hi) return "VADD clamp_lo > clamp_hi";
      return nullptr;
    }

    case Opcode::VPOOL: {
      const KeaVpool v = keaDecode<KeaVpool>(in);
      if (v.h.flags & ~KEA_VP_F_AVG) return "VPOOL reserved flags set";
      if (v.out_h == 0 || v.out_w == 0 || v.channels == 0) return "VPOOL extent must be >= 1";
      if (v.kh == 0 || v.kw == 0 || v.kh > 32 || v.kw > 32) return "VPOOL kernel must be 1..32";
      if (v.stride_h == 0 || v.stride_w == 0 || v.stride_h > 8 || v.stride_w > 8)
        return "VPOOL stride must be 1..8";
      if (v.in_addr >= KEA_SPM_A_BYTES || v.out_addr >= KEA_SPM_A_BYTES)
        return "VPOOL address out of range";
      return nullptr;
    }

    case Opcode::VCOPY: {
      const KeaVcopy v = keaDecode<KeaVcopy>(in);
      if (v.h.flags & ~(KEA_VC_F_FILL | KEA_VC_F_SRC_W | KEA_VC_F_DST_W))
        return "VCOPY reserved flags set";
      if (v.row_bytes == 0 || v.rows == 0) return "VCOPY extent must be >= 1";
      const std::size_t src_cap = (v.h.flags & KEA_VC_F_SRC_W) ? KEA_SPM_W_BYTES : KEA_SPM_A_BYTES;
      const std::size_t dst_cap = (v.h.flags & KEA_VC_F_DST_W) ? KEA_SPM_W_BYTES : KEA_SPM_A_BYTES;
      if (!(v.h.flags & KEA_VC_F_FILL) && v.src_addr >= src_cap) return "VCOPY src_addr out of range";
      if (v.dst_addr >= dst_cap) return "VCOPY dst_addr out of range";
      return nullptr;
    }

    default:
      return "unhandled opcode";
  }
}

// ---------------------------------------------------------------------------
// Encode/decode round-trip self-checks (compile-time where possible)
// ---------------------------------------------------------------------------

static_assert(static_cast<std::uint8_t>(Unit::MXU) == KEA_UNIT_MXU, "unit id drift");
static_assert(static_cast<std::uint8_t>(Unit::DWU) == KEA_UNIT_DWU, "unit id drift");
static_assert(static_cast<std::uint8_t>(Unit::VPU) == KEA_UNIT_VPU, "unit id drift");
static_assert(static_cast<std::uint8_t>(Unit::DMA0) == KEA_UNIT_DMA0, "unit id drift");
static_assert(static_cast<std::uint8_t>(Unit::DMA1) == KEA_UNIT_DMA1, "unit id drift");
static_assert(static_cast<std::uint8_t>(Unit::CTRL) == KEA_UNIT_CTRL, "unit id drift");
static_assert(defaultUnitFor(Opcode::MATMUL) == Unit::MXU, "opcode/unit mapping drift");
static_assert(defaultUnitFor(Opcode::DWCONV) == Unit::DWU, "opcode/unit mapping drift");
static_assert(defaultUnitFor(Opcode::VCOPY) == Unit::VPU, "opcode/unit mapping drift");
static_assert(defaultUnitFor(Opcode::DMA_ST) == Unit::DMA0, "opcode/unit mapping drift");
static_assert(defaultUnitFor(Opcode::WAIT) == Unit::CTRL, "opcode/unit mapping drift");

}  // namespace kea
