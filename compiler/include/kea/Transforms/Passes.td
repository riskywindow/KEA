//===- Passes.td - KEA transform passes --------------------*- tablegen -*-===//
//
// Every KEA pass is declared here and nowhere else. `-gen-pass-decls -name Kea`
// turns this into Passes.h.inc, which provides:
//   * GEN_PASS_DECL_<UPPERCASENAME>  -> createXxx() declaration + Options struct
//   * GEN_PASS_DEF_<UPPERCASENAME>   -> impl::XxxBase<T> CRTP base + createXxx()
//   * GEN_PASS_REGISTRATION          -> registerKeaPasses() and friends
//
//===----------------------------------------------------------------------===//

#ifndef KEA_TRANSFORMS_PASSES
#define KEA_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def KeaAnnotate : Pass<"kea-annotate", "::mlir::ModuleOp"> {
  let summary = "Tag the module with a summary of its KEA content";
  let description = [{
    Walks the module and attaches `kea.op_count`, `kea.event_count` and
    `<marker>.version` attributes. Deliberately trivial -- it exists to prove
    the pass registration, pass options and `dependentDialects` plumbing all
    work end to end.
  }];

  let options = [
    Option<"marker", "marker", "std::string", /*default=*/"\"kea\"",
           "Attribute namespace used for the version marker">,
  ];

  let dependentDialects = ["::mlir::kea::KeaDialect"];
}

def KeaCanonicalizeEvents : Pass<"kea-canonicalize-events"> {
  let summary = "Remove redundant kea.signal / kea.wait pairs";
  let description = [{
    Within each block:
      * a `kea.wait` on an event that has not been signalled earlier in the
        block by any op is dropped (nothing to wait for), and
      * back-to-back `kea.signal`s of the same event collapse to one.

    This is intentionally a local, single-block analysis; the real scheduler
    will replace it. It exists so that there is one pass in the tree that
    actually mutates IR and has a FileCheck test.
  }];

  // NB: the field is `statistics` (plural) but `options` (plural too) --
  // getting either singular is a bare "Value 'x' unknown!" TableGen error.
  let statistics = [
    Statistic<"numWaitsRemoved", "num-waits-removed", "Waits erased">,
    Statistic<"numSignalsRemoved", "num-signals-removed", "Signals erased">,
  ];

  let dependentDialects = ["::mlir::kea::KeaDialect"];
}

#endif // KEA_TRANSFORMS_PASSES
