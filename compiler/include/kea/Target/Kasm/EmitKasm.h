//===- EmitKasm.h - Level 2 MLIR -> .kasm / .bin / .map.json ----*- C++ -*-===//
//
// The `-kea-emit` backend, as a library. ADR-0001 puts text assembly between
// the MLIR compiler and the binary artifact, so this is where the compiler
// stops: it produces the three files `kea-as` consumes and never touches
// `keaf.h`.
//
//   model.kasm          docs/ASSEMBLY.md   -- one instruction per Level 2 op
//   model.weights.bin   docs/DIALECT_L2.md §4.5 -- the CONST blob
//   model.map.json      docs/ASSEMBLY.md §7 -- DRAM layout, symbols, tensors
//
// See docs/CODEGEN.md.
//
//===----------------------------------------------------------------------===//

#ifndef KEA_TARGET_KASM_EMITKASM_H
#define KEA_TARGET_KASM_EMITKASM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mlir {
namespace kea {

/// How to synthesize the `unit` assignment and the cross-unit semaphores that
/// `-kea-schedule` would otherwise have provided.
enum class SyncMode {
  /// Emit whatever the IR says. If the IR has no `kea.signal`/`kea.wait`, this
  /// produces a program that is only correct if the five units did not run
  /// concurrently -- useful for diffing, not for running.
  None,
  /// Insert the minimal cross-unit `SIGNAL`/`WAIT` pairs implied by the
  /// buffer dependencies of the *sequential* program. Not scheduling: nothing
  /// is reordered, hoisted or double buffered.
  Insert,
  /// `Insert` when the IR carries no semaphores, `None` when it does.
  Auto,
};

/// Host-visible quantization for one model input/output, which the Level 2 IR
/// does not carry (TOSA i8 tensors have no scale). Supplied by `keac` from the
/// frontend's `.kgraph.json`, or on the command line.
struct IoQuant {
  double scale = 1.0;
  int64_t zeroPoint = 0;
  bool hasScale = false;
  bool hasZeroPoint = false;
};

struct EmitOptions {
  /// Which `func.func` to emit. Empty selects the only Level 2 function; an
  /// ambiguous module is an error rather than a guess.
  std::string functionName;
  SyncMode sync = SyncMode::Auto;
  /// Append `; pc=N` and buffer-name comments. Not canonical form.
  bool annotate = false;
  /// Emit a label at the first instruction of each traced region.
  bool labels = true;
  std::string producer = "kea-translate 0.1.0";
  /// Path recorded in `metadata.source`, purely provenance.
  std::string sourceName;
  /// Overrides keyed by tensor name.
  std::map<std::string, IoQuant> ioQuant;
};

struct EmitResult {
  std::string kasm;
  std::string mapJson;
  std::vector<uint8_t> constBlob;
  /// A readable dump of `constBlob`: one section per DRAM constant, with
  /// `KeaQuantParam` / `KeaAddParam` records decoded field by field. Never
  /// consumed by the assembler -- it is for FileCheck and for the human
  /// debugging a miscompile (docs/CODEGEN.md §8).
  std::string constListing;
  /// Instruction count, for the driver's summary line.
  int64_t numInstructions = 0;
  /// Semaphore pairs this emitter inserted (0 when the IR was scheduled).
  int64_t numInsertedSync = 0;
};

/// Emit the three files for one Level 2 function. Diagnostics are reported
/// against the offending op through MLIR's diagnostic engine.
LogicalResult emitKasm(ModuleOp module, const EmitOptions &options,
                       EmitResult &result);

//===----------------------------------------------------------------------===//
// Constant materialization (docs/DIALECT_L2.md §4.5), exposed for testing
//===----------------------------------------------------------------------===//

/// Lay out the constants a `kea.alloc` with a DRAM `role` declares, in the byte
/// layout its `layout` attribute names. `out` is resized to the alloc's extent.
/// Fails, with a diagnostic on the op, if the source is not a constant or the
/// declared extent disagrees with the layout's size.
LogicalResult materializeConstant(Operation *allocOp, std::vector<uint8_t> &out);

} // namespace kea
} // namespace mlir

#endif // KEA_TARGET_KASM_EMITKASM_H
