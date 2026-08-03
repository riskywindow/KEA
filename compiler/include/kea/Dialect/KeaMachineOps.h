//===- KeaMachineOps.h - KEA Level 2 (machine level) ops --------*- C++ -*-===//
//
// Declarations for the ops defined in KeaMachineOps.td. Level 1 ops are in
// KeaOps.h; the two headers are independent so the two halves of the compiler
// do not collide. Including this header also pulls in KeaOps.h because both
// levels share the dialect, types and attributes.
//
// It also declares the handful of *whole-function* checks that cannot live in
// an op verifier because they are not properties of a single op -- notably
// errata E7 (no `MATMUL` against an unwritten weight bank) and the live-range
// bookkeeping `-kea-alloc` consumes. See docs/DIALECT_L2.md §4 and §7.
//
//===----------------------------------------------------------------------===//
#ifndef KEA_DIALECT_KEAMACHINEOPS_H
#define KEA_DIALECT_KEAMACHINEOPS_H

#include "kea/Dialect/KeaAttrs.h"
#include "kea/Dialect/KeaDialect.h"
#include "kea/Dialect/KeaTypes.h"

// MLIR >= 17 gives every ODS op a BytecodeOpInterface::Trait, so this include
// is mandatory even though nothing here mentions bytecode. Leaving it out
// produces a wall of "no member named 'BytecodeOpInterface' in namespace
// 'mlir'" errors from the generated header.
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <cstdint>

#define GET_OP_CLASSES
#include "kea/Dialect/KeaMachineOps.h.inc"

namespace mlir {
namespace kea {

//===----------------------------------------------------------------------===//
// Buffer helpers
//===----------------------------------------------------------------------===//

/// Number of *elements* in `v`'s buffer type: bytes for SPM_A / SPM_W / DRAM
/// (whose Level 2 element type is `i8`) and int32 words for ACC. This is the
/// unit every Level 2 offset and stride is expressed in.
int64_t bufferExtent(Value v);

/// Half-open span `[lo, hi)` of elements a strided walk touches, given a base
/// displacement, up to two `(count, stride)` levels, and the extent read or
/// written at each visited point. Strides may be negative.
struct Span {
  int64_t lo;
  int64_t hi;
};
Span stridedSpan(int64_t base, int64_t count0, int64_t stride0, int64_t count1,
                 int64_t stride1, int64_t extent);

//===----------------------------------------------------------------------===//
// Live ranges -- the `-kea-alloc` interface (docs/DIALECT_L2.md §4)
//===----------------------------------------------------------------------===//

/// Recompute `kea.alloc`'s `live = [first, last]` attribute for every on-chip
/// allocation in `root` from the SSA def-use chain plus block order, and stamp
/// it. The indices are positions in the enclosing block's operation list.
///
/// The SSA range is authoritative; the attribute is a materialization of it so
/// that `-kea-alloc` (and a human reading the IR) does not have to recompute
/// it. Any pass that inserts, erases or moves Level 2 ops must call this again.
void refreshLiveRanges(Operation *root);

//===----------------------------------------------------------------------===//
// Whole-function legality checks
//===----------------------------------------------------------------------===//

/// Errata E7: "never issue a MATMUL against a bank no LOAD_W has written. Do
/// not rely on the zero-fill." Walks `root` in program order tracking which of
/// the two MXU weight banks have been written by a `kea.load_w`, and reports
/// the first `kea.mm` that reads an unwritten one.
LogicalResult verifyWeightBanks(Operation *root);

/// Errata E6: `keaQuantizedAdd` sums `xa + xb` in plain int32, and ISA.md does
/// not bound `KeaAddParam`. Returns the worst-case `|xa| + |xb|` for the given
/// parameters, over the whole int8 input domain. The caller must reject
/// anything that is not `< 2^31`.
///
///   sa  = (a - a_zp) << KEA_VADD_LEFT_SHIFT,  |sa| <= 255 << 20
///   xa  = keaRdpot(keaSrdhm(sa, a_mult), a_shift)
///   |xa| <= (|sa| * a_mult) >> 31 >> max(a_shift, 0)
int64_t addParamWorstCaseSum(int64_t aMult, int64_t aShift, int64_t bMult,
                             int64_t bShift);

} // namespace kea
} // namespace mlir

#endif // KEA_DIALECT_KEAMACHINEOPS_H
