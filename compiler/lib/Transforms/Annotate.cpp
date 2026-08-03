//===- Annotate.cpp - -kea-annotate ----------------------------*- C++ -*-===//
//
// Trivial module-level pass: proves out pass registration, pass options and
// dependentDialects.
//
//===----------------------------------------------------------------------===//

#include "kea/Dialect/KeaDialect.h"
#include "kea/Dialect/KeaMachineOps.h"
#include "kea/Dialect/KeaOps.h"
#include "kea/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace kea {
// Emits impl::KeaAnnotateBase<T>, which already parses the pass options and
// implements getArgument()/getName()/getDependentDialects() for us.
#define GEN_PASS_DEF_KEAANNOTATE
#include "kea/Transforms/Passes.h.inc"
} // namespace kea
} // namespace mlir

using namespace mlir;
using namespace mlir::kea;

namespace {

// GOTCHA: `impl` on its own is ambiguous here -- both `mlir::impl` and
// `mlir::kea::impl` exist and both namespaces are `using`-ed above. Always
// spell the generated base class fully qualified.
struct KeaAnnotatePass : public mlir::kea::impl::KeaAnnotateBase<KeaAnnotatePass> {
  // Pulls in the generated constructors, including the one taking the
  // KeaAnnotateOptions struct. Without this line `createKeaAnnotate(options)`
  // does not compile.
  using mlir::kea::impl::KeaAnnotateBase<KeaAnnotatePass>::KeaAnnotateBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder b(module.getContext());

    int64_t opCount = 0;
    int64_t eventCount = 0;
    module.walk([&](Operation *op) {
      if (op->getDialect() != module->getContext()->getLoadedDialect("kea"))
        return;
      ++opCount;
      if (isa<SignalOp, WaitOp>(op))
        ++eventCount;
    });

    module->setAttr("kea.op_count", b.getI64IntegerAttr(opCount));
    module->setAttr("kea.event_count", b.getI64IntegerAttr(eventCount));
    module->setAttr(marker + ".version", b.getStringAttr("0.1.0"));
  }
};

} // namespace
