//===- CanonicalizeEvents.cpp - -kea-canonicalize-events -------*- C++ -*-===//
//
// Local (per-block) cleanup of kea.signal / kea.wait pairs.
//
//===----------------------------------------------------------------------===//

#include "kea/Dialect/KeaDialect.h"
#include "kea/Dialect/KeaMachineOps.h"
#include "kea/Dialect/KeaOps.h"
#include "kea/Transforms/Passes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace kea {
#define GEN_PASS_DEF_KEACANONICALIZEEVENTS
#include "kea/Transforms/Passes.h.inc"
} // namespace kea
} // namespace mlir

using namespace mlir;
using namespace mlir::kea;

namespace {

// See Annotate.cpp: `impl` alone is ambiguous between mlir::impl and
// mlir::kea::impl once both namespaces are `using`-ed.
struct KeaCanonicalizeEventsPass
    : public mlir::kea::impl::KeaCanonicalizeEventsBase<
          KeaCanonicalizeEventsPass> {
  using mlir::kea::impl::KeaCanonicalizeEventsBase<
      KeaCanonicalizeEventsPass>::KeaCanonicalizeEventsBase;

  void runOnOperation() override {
    SmallVector<Operation *> toErase;

    getOperation()->walk([&](Block *block) {
      llvm::DenseSet<int64_t> signalled;
      int64_t lastSignal = -1;
      bool lastWasSignal = false;

      for (Operation &op : *block) {
        if (auto signal = dyn_cast<SignalOp>(op)) {
          int64_t event = signal.getEvent();
          if (lastWasSignal && lastSignal == event) {
            // Back-to-back signal of the same event: the second is redundant.
            toErase.push_back(&op);
            ++numSignalsRemoved;
            continue;
          }
          signalled.insert(event);
          lastSignal = event;
          lastWasSignal = true;
          continue;
        }

        if (auto wait = dyn_cast<WaitOp>(op)) {
          if (!signalled.contains(wait.getEvent())) {
            // Nothing in this block ever raises the event: the wait cannot be
            // satisfied locally, so drop it.
            toErase.push_back(&op);
            ++numWaitsRemoved;
          }
        }

        lastWasSignal = false;
      }
    });

    for (Operation *op : toErase)
      op->erase();
  }
};

} // namespace
