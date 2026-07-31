// SPDX-License-Identifier: Apache-2.0
//
// kea/hw_config.h --- KEA-1 machine parameters (FROZEN CONTRACT)
//
// Every architectural constant of the KEA-1 edge NPU lives here: array
// dimensions, scratchpad sizes, queue depths, bandwidths, and the
// cycle-approximate timing model.  The simulator, the compiler backend, the
// runtime, and the roofline tooling MUST all derive their numbers from this
// header.  Nothing here may be redefined locally.
//
// Header-only, C++17, no dependencies beyond <cstdint>/<cstddef>.
//
// See docs/MICROARCH.md for the rationale behind every number.

#pragma once

#include <cstddef>
#include <cstdint>

namespace kea {

// ---------------------------------------------------------------------------
// 0. Identity
// ---------------------------------------------------------------------------

/// Architecture revision.  Bumped on any incompatible ISA/encoding change.
inline constexpr std::uint32_t KEA_ISA_REVISION = 1;

/// Human-readable architecture name, used in tool banners and KEAF metadata.
inline constexpr const char* KEA_ARCH_NAME = "KEA-1";

/// Nominal clock.  A "cycle" is the unit of the timing model; the frequency
/// only matters when converting cycles to GOPS / GB/s for reporting.
inline constexpr std::uint64_t KEA_CLOCK_HZ = 1000000000ULL;  // 1 GHz

// ---------------------------------------------------------------------------
// 1. Small integer helpers (constexpr, shared by sim + compiler)
// ---------------------------------------------------------------------------

/// Ceiling division for non-negative integers.
constexpr std::uint64_t keaCeilDiv(std::uint64_t a, std::uint64_t b) {
  return (b == 0) ? 0 : (a + b - 1) / b;
}

/// Round `a` up to the next multiple of `b` (b must be > 0).
constexpr std::uint64_t keaRoundUp(std::uint64_t a, std::uint64_t b) {
  return (b == 0) ? a : keaCeilDiv(a, b) * b;
}

/// True if `a` is a multiple of `b`.
constexpr bool keaIsAligned(std::uint64_t a, std::uint64_t b) {
  return (b == 0) ? true : (a % b) == 0;
}

// ---------------------------------------------------------------------------
// 2. Execution units and queues
// ---------------------------------------------------------------------------
//
// A single in-order fetch/dispatch stage reads the linear instruction stream
// and pushes each instruction into the queue named by its `unit` field.
// Units 0..4 have queues and execute concurrently.  Unit 5 (CTRL) is the
// dispatcher itself and has no queue: CTRL instructions retire at dispatch.

inline constexpr int KEA_UNIT_MXU  = 0;
inline constexpr int KEA_UNIT_DWU  = 1;
inline constexpr int KEA_UNIT_VPU  = 2;
inline constexpr int KEA_UNIT_DMA0 = 3;
inline constexpr int KEA_UNIT_DMA1 = 4;
inline constexpr int KEA_UNIT_CTRL = 5;

/// Total number of unit ids, including CTRL.
inline constexpr int KEA_NUM_UNITS = 6;

/// Number of units that own an instruction queue (MXU, DWU, VPU, DMA0, DMA1).
inline constexpr int KEA_NUM_QUEUES = 5;

/// Number of independent DMA engines.  Two exist precisely so the compiler
/// can double-buffer loads against stores / compute.
inline constexpr int KEA_NUM_DMA_ENGINES = 2;

/// Instructions the dispatcher can push per cycle.
inline constexpr int KEA_DISPATCH_PER_CYCLE = 1;

/// Depth of each per-unit instruction queue, in instructions.  A full target
/// queue stalls the dispatcher; that is the ONLY implicit backpressure in the
/// machine.
inline constexpr int KEA_QUEUE_DEPTH = 16;

/// Cycles a unit needs between starting two consecutive instructions.
/// Every unit can start at most one instruction per cycle.
inline constexpr int KEA_ISSUE_COST_CYCLES = 1;

// ---------------------------------------------------------------------------
// 3. Synchronization
// ---------------------------------------------------------------------------

/// Number of hardware counting semaphores ("events").  Ids 0..31.
inline constexpr int KEA_NUM_EVENTS = 32;

/// Event counters are 32-bit unsigned.  Exceeding this value is a program
/// error; simulators must flag it.
inline constexpr std::uint32_t KEA_EVENT_MAX = 0x7FFFFFFFu;

// ---------------------------------------------------------------------------
// 4. Instruction memory
// ---------------------------------------------------------------------------

/// Every instruction is exactly 32 bytes.  Fixed width because there is no
/// I-cache and decode must be a fixed-offset field extract.
inline constexpr std::size_t KEA_INSTR_BYTES = 32;

/// Dedicated instruction memory, staged by the runtime before START.  It is
/// NOT part of the DRAM address space and consumes no DRAM bandwidth.
inline constexpr std::size_t KEA_IMEM_BYTES = 1024 * 1024;  // 1 MiB

/// Maximum program length, in instructions.
inline constexpr std::uint32_t KEA_MAX_INSTRUCTIONS =
    static_cast<std::uint32_t>(KEA_IMEM_BYTES / KEA_INSTR_BYTES);  // 32768

// ---------------------------------------------------------------------------
// 5. Address spaces
// ---------------------------------------------------------------------------
//
// Four distinct, disjoint address spaces, each base-0.  There are no caches,
// no TLB, no runtime allocator, and no aliasing between spaces.

/// Activation scratchpad, byte-addressed.  Holds int8/int4 feature maps.
inline constexpr std::size_t KEA_SPM_A_BYTES = 256 * 1024;  // 256 KiB

/// Weight / parameter scratchpad, byte-addressed.  Holds MXU weight tiles,
/// DWU kernels, and VPU quantization parameter blocks.
inline constexpr std::size_t KEA_SPM_W_BYTES = 256 * 1024;  // 256 KiB

/// Accumulator scratchpad.  Addressed in *int32 words*, not bytes.
inline constexpr std::size_t KEA_ACC_WORDS = 32768;              // 32768 x int32
inline constexpr std::size_t KEA_ACC_BYTES = KEA_ACC_WORDS * 4;  // 128 KiB

/// DRAM address space.  Flat, byte-addressed, 32-bit => 4 GiB.  This is an
/// edge part; 4 GiB is far beyond any model we target and keeps the DMA
/// descriptor inside 32 bytes.
inline constexpr std::uint64_t KEA_DRAM_BYTES = 0x100000000ULL;  // 4 GiB

// ---------------------------------------------------------------------------
// 6. Alignment contract
// ---------------------------------------------------------------------------

/// LOAD_W source address and row stride must be multiples of this (int8).
inline constexpr std::uint32_t KEA_ALIGN_LOADW_INT8 = 16;
/// LOAD_W source address and row stride must be multiples of this (int4).
inline constexpr std::uint32_t KEA_ALIGN_LOADW_INT4 = 8;

/// MXU/DWU/VQUANT accumulator addresses and strides, in int32 words.
inline constexpr std::uint32_t KEA_ALIGN_ACC_WORDS = 16;

/// VPU quantization / add parameter blocks in SPM_W.
inline constexpr std::uint32_t KEA_ALIGN_QPARAM = 4;

/// DWU activation base / strides.
inline constexpr std::uint32_t KEA_ALIGN_DWU = 16;

/// DMA addresses are byte-granular.  This is the *recommended* alignment for
/// both the DRAM and SPM sides; the timing model does not penalize misaligned
/// transfers, but real silicon would.
inline constexpr std::uint32_t KEA_ALIGN_DMA_RECOMMENDED = 16;

/// Base alignment the runtime must honour when allocating the DRAM arena.
inline constexpr std::uint32_t KEA_DRAM_BASE_ALIGN = 64;

// ---------------------------------------------------------------------------
// 7. MXU --- weight-stationary systolic array
// ---------------------------------------------------------------------------

/// Reduction depth of the array (rows of PEs) == K tile size.
inline constexpr int KEA_MXU_K = 16;
/// Output width of the array (columns of PEs) == N tile size.
inline constexpr int KEA_MXU_N = 16;
/// Total PEs.
inline constexpr int KEA_MXU_PES = KEA_MXU_K * KEA_MXU_N;  // 256

/// Weight register banks per PE.  Two, so a tile can be loaded into the
/// inactive bank while the active bank is streaming.
inline constexpr int KEA_MXU_WEIGHT_BANKS = 2;

/// Peak MACs/cycle by datatype.
inline constexpr int KEA_MXU_INT8_MACS_PER_CYCLE = 256;
inline constexpr int KEA_MXU_INT4_MACS_PER_CYCLE = 512;

/// Activation rows consumed per cycle.  int4 packs two MACs per PE per cycle,
/// which KEA-1 spends on a second activation row (not on a deeper K tile).
inline constexpr int KEA_MXU_ROWS_PER_CYCLE_INT8 = 1;
inline constexpr int KEA_MXU_ROWS_PER_CYCLE_INT4 = 2;

/// Fill + drain latency of the 16-deep array, in cycles.  Added to the
/// *latency* of a MATMUL (when its last ACC write lands), not to occupancy.
inline constexpr int KEA_MXU_PIPELINE_DEPTH = 32;

/// Fixed per-MATMUL setup (address generator programming, bank select).
inline constexpr int KEA_MXU_MATMUL_ISSUE_OVERHEAD = 4;

/// Fixed per-LOAD_W setup.
inline constexpr int KEA_MXU_LOADW_ISSUE_OVERHEAD = 2;

/// SPM_W read port width used by LOAD_W, bytes/cycle.
inline constexpr int KEA_MXU_LOADW_BYTES_PER_CYCLE = 32;

/// Maximum total activation rows in a single MATMUL (M_inner * M_outer).
/// Bounded by ACC capacity: M * 16 int32 words must fit in ACC.
inline constexpr std::uint32_t KEA_MXU_MAX_ROWS =
    static_cast<std::uint32_t>(KEA_ACC_WORDS / KEA_MXU_N);  // 2048

// ---------------------------------------------------------------------------
// 8. DWU --- depthwise convolution unit
// ---------------------------------------------------------------------------

/// Channel lanes processed in parallel.  One lane per output channel.
inline constexpr int KEA_DWU_LANES = 16;
/// MACs per lane per cycle (taps consumed per cycle per channel).
inline constexpr int KEA_DWU_MACS_PER_LANE = 8;
/// Peak MACs/cycle.
inline constexpr int KEA_DWU_MACS_PER_CYCLE = KEA_DWU_LANES * KEA_DWU_MACS_PER_LANE;  // 128

/// Fixed per-DWCONV setup.
inline constexpr int KEA_DWU_ISSUE_OVERHEAD = 8;
/// Pipeline latency from last input read to last ACC write.
inline constexpr int KEA_DWU_PIPELINE_DEPTH = 12;

/// Supported kernel sizes (square only) and strides.
inline constexpr int KEA_DWU_KERNEL_3 = 3;
inline constexpr int KEA_DWU_KERNEL_5 = 5;
inline constexpr int KEA_DWU_MAX_STRIDE = 2;

// ---------------------------------------------------------------------------
// 9. VPU --- post-processing / elementwise unit
// ---------------------------------------------------------------------------

/// Lanes.  One int8 element per lane per cycle for arithmetic ops.
inline constexpr int KEA_VPU_LANES = 16;
inline constexpr int KEA_VPU_ELEMS_PER_CYCLE = 16;

/// VCOPY moves bytes through a wider path than the arithmetic datapath.
inline constexpr int KEA_VPU_COPY_BYTES_PER_CYCLE = 32;

/// Fixed per-instruction setup.
inline constexpr int KEA_VPU_ISSUE_OVERHEAD = 4;
/// Pipeline latency from last read to last write.
inline constexpr int KEA_VPU_PIPELINE_DEPTH = 8;

/// Fixed left shift applied to both operands of VADD before requantization,
/// matching the TFLite reference kernel.  See ISA.md "VADD semantics".
inline constexpr int KEA_VADD_LEFT_SHIFT = 20;

// ---------------------------------------------------------------------------
// 10. DMA and DRAM
// ---------------------------------------------------------------------------

/// Peak bytes/cycle a single DMA engine can move.
inline constexpr int KEA_DMA_ENGINE_BYTES_PER_CYCLE = 16;

/// Aggregate DRAM port width, bytes/cycle, SHARED by both engines.
/// 16 B/cycle @ 1 GHz == 16 GB/s.
inline constexpr int KEA_DRAM_BYTES_PER_CYCLE = 16;

/// Fixed cost of accepting one descriptor (address generator programming,
/// DRAM page open).
inline constexpr int KEA_DMA_DESC_OVERHEAD_CYCLES = 16;

/// Per-inner-run overhead: one address turnaround per contiguous run.
inline constexpr int KEA_DMA_ROW_OVERHEAD_CYCLES = 2;

/// Descriptor field limits (see the DMA encoding in isa.h).
inline constexpr std::uint32_t KEA_DMA_MAX_LEN0 = 65535;  // contiguous bytes
inline constexpr std::uint32_t KEA_DMA_MAX_N1   = 65535;  // dim-1 count
inline constexpr std::uint32_t KEA_DMA_MAX_N2   = 255;    // dim-2 count

// ---------------------------------------------------------------------------
// 11. Derived peak numbers (for reporting and roofline)
// ---------------------------------------------------------------------------
//
// "ops" counts multiply and add separately, i.e. 1 MAC == 2 ops.

inline constexpr double KEA_PEAK_INT8_OPS_PER_SEC =
    2.0 * KEA_MXU_INT8_MACS_PER_CYCLE * static_cast<double>(KEA_CLOCK_HZ);  // 512 GOPS

inline constexpr double KEA_PEAK_INT4_OPS_PER_SEC =
    2.0 * KEA_MXU_INT4_MACS_PER_CYCLE * static_cast<double>(KEA_CLOCK_HZ);  // 1024 GOPS

inline constexpr double KEA_PEAK_DWU_OPS_PER_SEC =
    2.0 * KEA_DWU_MACS_PER_CYCLE * static_cast<double>(KEA_CLOCK_HZ);  // 256 GOPS

inline constexpr double KEA_PEAK_DRAM_BYTES_PER_SEC =
    static_cast<double>(KEA_DRAM_BYTES_PER_CYCLE) * static_cast<double>(KEA_CLOCK_HZ);  // 16 GB/s

/// Roofline ridge points, in ops per DRAM byte.
inline constexpr double KEA_RIDGE_INT8_OPS_PER_BYTE =
    KEA_PEAK_INT8_OPS_PER_SEC / KEA_PEAK_DRAM_BYTES_PER_SEC;  // 32.0
inline constexpr double KEA_RIDGE_INT4_OPS_PER_BYTE =
    KEA_PEAK_INT4_OPS_PER_SEC / KEA_PEAK_DRAM_BYTES_PER_SEC;  // 64.0

// ---------------------------------------------------------------------------
// 12. Cycle-approximate timing model
// ---------------------------------------------------------------------------
//
// Terminology, used identically by kea-sim and keac:
//
//   issue cost  Cycles before the SAME unit may start its next instruction.
//               Always KEA_ISSUE_COST_CYCLES (1).
//   occupancy   Cycles the instruction holds its functional resource.  The
//               unit's next instruction may issue while a previous one is
//               still occupying a *different* resource (this is how LOAD_W
//               overlaps MATMUL), but never the same resource.
//   latency     Cycles from start-of-execution to the instruction's last
//               architectural write being visible.  Only matters for the
//               SIGNAL that follows it: a SIGNAL retires after the preceding
//               instructions in its queue have fully retired (occupancy +
//               pipeline depth).
//
// All functions below are the normative definition.  Simulators must not
// invent their own formulas.

/// MXU MATMUL occupancy of the systolic array.
/// int8: one activation row per cycle.  int4: two rows per cycle, paired
/// along the inner dimension (an odd M_inner wastes half of the last cycle).
constexpr std::uint64_t keaMatmulOccupancy(std::uint32_t m_inner,
                                           std::uint32_t m_outer,
                                           bool int4) {
  return static_cast<std::uint64_t>(KEA_MXU_MATMUL_ISSUE_OVERHEAD) +
         static_cast<std::uint64_t>(m_outer) *
             (int4 ? keaCeilDiv(m_inner, 2) : static_cast<std::uint64_t>(m_inner));
}

/// MXU MATMUL total latency (occupancy + array fill/drain).
constexpr std::uint64_t keaMatmulLatency(std::uint32_t m_inner,
                                         std::uint32_t m_outer,
                                         bool int4) {
  return keaMatmulOccupancy(m_inner, m_outer, int4) + KEA_MXU_PIPELINE_DEPTH;
}

/// MXU LOAD_W occupancy of the weight port (and of the target weight bank).
/// `k_rows` is the number of valid reduction rows (1..16).
constexpr std::uint64_t keaLoadWOccupancy(std::uint32_t k_rows, bool int4) {
  const std::uint64_t row_bytes = int4 ? 8u : 16u;
  return static_cast<std::uint64_t>(KEA_MXU_LOADW_ISSUE_OVERHEAD) +
         keaCeilDiv(static_cast<std::uint64_t>(k_rows) * row_bytes,
                    static_cast<std::uint64_t>(KEA_MXU_LOADW_BYTES_PER_CYCLE));
}

/// DWU DWCONV occupancy.
/// Channels are processed 16 at a time; within a 16-channel group each lane
/// retires KEA_DWU_MACS_PER_LANE taps per cycle, pipelined across output
/// pixels (so the tap count does NOT round up per pixel).
constexpr std::uint64_t keaDwconvOccupancy(std::uint32_t out_h,
                                           std::uint32_t out_w,
                                           std::uint32_t channels,
                                           std::uint32_t kh,
                                           std::uint32_t kw) {
  const std::uint64_t groups = keaCeilDiv(channels, KEA_DWU_LANES);
  const std::uint64_t taps = static_cast<std::uint64_t>(out_h) * out_w * kh * kw;
  return static_cast<std::uint64_t>(KEA_DWU_ISSUE_OVERHEAD) +
         groups * keaCeilDiv(taps, KEA_DWU_MACS_PER_LANE);
}

constexpr std::uint64_t keaDwconvLatency(std::uint32_t out_h,
                                         std::uint32_t out_w,
                                         std::uint32_t channels,
                                         std::uint32_t kh,
                                         std::uint32_t kw) {
  return keaDwconvOccupancy(out_h, out_w, channels, kh, kw) + KEA_DWU_PIPELINE_DEPTH;
}

/// VPU VQUANT occupancy.  One int32 accumulator lane per cycle per lane.
constexpr std::uint64_t keaVquantOccupancy(std::uint32_t num_pixels,
                                           std::uint32_t channels) {
  return static_cast<std::uint64_t>(KEA_VPU_ISSUE_OVERHEAD) +
         keaCeilDiv(static_cast<std::uint64_t>(num_pixels) * channels,
                    static_cast<std::uint64_t>(KEA_VPU_ELEMS_PER_CYCLE));
}

/// VPU VADD occupancy.
constexpr std::uint64_t keaVaddOccupancy(std::uint32_t num_elems) {
  return static_cast<std::uint64_t>(KEA_VPU_ISSUE_OVERHEAD) +
         keaCeilDiv(num_elems, static_cast<std::uint64_t>(KEA_VPU_ELEMS_PER_CYCLE));
}

/// VPU VPOOL occupancy.  Every window element costs one lane-cycle.
constexpr std::uint64_t keaVpoolOccupancy(std::uint32_t out_h,
                                          std::uint32_t out_w,
                                          std::uint32_t channels,
                                          std::uint32_t kh,
                                          std::uint32_t kw) {
  const std::uint64_t work =
      static_cast<std::uint64_t>(out_h) * out_w * channels * kh * kw;
  return static_cast<std::uint64_t>(KEA_VPU_ISSUE_OVERHEAD) +
         keaCeilDiv(work, static_cast<std::uint64_t>(KEA_VPU_ELEMS_PER_CYCLE));
}

/// VPU VCOPY occupancy.  `rows` runs of `row_bytes` bytes each.
constexpr std::uint64_t keaVcopyOccupancy(std::uint32_t row_bytes,
                                          std::uint32_t rows) {
  return static_cast<std::uint64_t>(KEA_VPU_ISSUE_OVERHEAD) +
         static_cast<std::uint64_t>(rows) *
             keaCeilDiv(row_bytes, static_cast<std::uint64_t>(KEA_VPU_COPY_BYTES_PER_CYCLE));
}

/// DMA engine occupancy for one descriptor, ASSUMING the engine gets its full
/// KEA_DMA_ENGINE_BYTES_PER_CYCLE share of the DRAM port.  When both engines
/// are active the simulator must stretch the byte term by the arbitration
/// rule in MICROARCH.md (fair split, 8 B/cycle each).
constexpr std::uint64_t keaDmaOccupancy(std::uint32_t len0,
                                        std::uint32_t n1,
                                        std::uint32_t n2) {
  const std::uint64_t runs = static_cast<std::uint64_t>(n1) * n2;
  const std::uint64_t bytes = runs * len0;
  return static_cast<std::uint64_t>(KEA_DMA_DESC_OVERHEAD_CYCLES) +
         runs * KEA_DMA_ROW_OVERHEAD_CYCLES +
         keaCeilDiv(bytes, static_cast<std::uint64_t>(KEA_DMA_ENGINE_BYTES_PER_CYCLE));
}

/// Total bytes moved by a DMA descriptor.
constexpr std::uint64_t keaDmaBytes(std::uint32_t len0,
                                    std::uint32_t n1,
                                    std::uint32_t n2) {
  return static_cast<std::uint64_t>(len0) * n1 * n2;
}

// ---------------------------------------------------------------------------
// 13. Internal consistency checks
// ---------------------------------------------------------------------------

static_assert(KEA_MXU_PES == 256, "MXU must be 16x16");
static_assert(KEA_MXU_INT8_MACS_PER_CYCLE == KEA_MXU_PES, "1 int8 MAC/PE/cycle");
static_assert(KEA_MXU_INT4_MACS_PER_CYCLE == 2 * KEA_MXU_PES, "2 int4 MAC/PE/cycle");
static_assert(KEA_ACC_BYTES == 128 * 1024, "ACC is 128 KiB");
static_assert(KEA_ACC_WORDS * 4 == KEA_ACC_BYTES, "ACC word/byte mismatch");
static_assert(KEA_MAX_INSTRUCTIONS == 32768, "IMEM holds 32768 instructions");
static_assert(KEA_NUM_QUEUES + 1 == KEA_NUM_UNITS, "CTRL is the only queueless unit");
static_assert(KEA_UNIT_DMA1 - KEA_UNIT_DMA0 + 1 == KEA_NUM_DMA_ENGINES,
              "DMA unit ids must be contiguous");
static_assert(KEA_PEAK_INT8_OPS_PER_SEC == 512e9, "peak int8 must be 512 GOPS");
static_assert(KEA_PEAK_INT4_OPS_PER_SEC == 1024e9, "peak int4 must be 1 TOPS");
static_assert(KEA_PEAK_DRAM_BYTES_PER_SEC == 16e9, "peak DRAM must be 16 GB/s");
static_assert(KEA_RIDGE_INT8_OPS_PER_BYTE == 32.0, "int8 ridge point must be 32 ops/byte");
static_assert(KEA_RIDGE_INT4_OPS_PER_BYTE == 64.0, "int4 ridge point must be 64 ops/byte");
static_assert(KEA_MXU_MAX_ROWS == 2048, "MATMUL row bound follows from ACC size");
static_assert(keaMatmulOccupancy(8, 8, false) == 68, "timing model drift (MATMUL int8)");
static_assert(keaMatmulOccupancy(8, 8, true) == 36, "timing model drift (MATMUL int4)");
static_assert(keaLoadWOccupancy(16, false) == 10, "timing model drift (LOAD_W int8)");
static_assert(keaLoadWOccupancy(16, true) == 6, "timing model drift (LOAD_W int4)");
static_assert(keaDmaOccupancy(64, 8, 1) == 64, "timing model drift (DMA)");

}  // namespace kea
