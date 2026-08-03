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

// The graph-ingest conversions are declared in their own file but generated
// into this one, so that `registerKeaPasses()` stays the single registration
// entry point that tools/kea-opt calls. See Conversion/Passes.td's header.
include "kea/Conversion/Passes.td"

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

def KeaFuse : Pass<"kea-fuse", "::mlir::func::FuncOp"> {
  let summary = "Fuse KEA Level 1 epilogues (bias / requantize / clamp / residual)";
  let description = [{
    The scratchpad-aware fusion pass. It rewrites Level 1 into Level 1 with
    fewer ops and fatter `#kea.epilogue`s, because the KEA VPU performs bias,
    requantize, activation clamp and the residual add in a single pass
    (`KEA_VQUANT` / `KEA_VADD`, docs/ISA.md §10).

    Every rewrite here is **bit-exact**: it reassociates work between ops but
    never changes the arithmetic. In particular a `kea.rescale` chain is only
    collapsed in the two cases where composition is provably exact (see
    docs/DIALECT_L1.md §6.3); the general `i32 -> i8 -> i32` chain is refused,
    because the intermediate `i8` saturation and rounding are observable.

    Fusions, in the order the greedy driver reaches them:

    1. contraction + `kea.rescale`  -> `epilogue.requant`
    2. contraction + `kea.clamp`    -> `epilogue.clamp`
    3. contraction + `kea.add` of a constant bias -> the `bias` operand
    4. contraction + quantized `kea.add` -> `epilogue.{accum,residual,output}`
       plus the `residual` operand (the MobileNetV2 inverted residual)
    5. exact `kea.rescale` o `kea.rescale` composition, and removal of
       no-op rescales
    6. no-op `kea.reshape` / `kea.transpose` folding
  }];

  let options = [
    // The Homebrew LLVM 20.1.6 bottle is Release+NDEBUG, so `llvm::Statistic`
    // is `NoopStatistic` and `-mlir-pass-statistics` prints an empty report
    // (verified: upstream `mlir-opt --symbol-dce -mlir-pass-statistics` prints
    // nothing either). This option attaches the same counters to the function
    // as a `kea.fusion_stats` dictionary so they are observable -- and
    // FileCheck-testable -- on this toolchain.
    Option<"reportStats", "report-stats", "bool", /*default=*/"false",
           "Attach the fusion counts to the function as kea.fusion_stats">,
  ];

  let statistics = [
    Statistic<"numRequantFused", "num-requant-fused",
              "kea.rescale ops folded into a producer epilogue">,
    Statistic<"numClampFused", "num-clamp-fused",
              "kea.clamp ops folded into a producer epilogue">,
    Statistic<"numBiasFused", "num-bias-fused",
              "constant kea.add ops folded into a producer bias operand">,
    Statistic<"numQuantAddFormed", "num-quant-add-formed",
              "rescale/add/rescale sandwiches collapsed into a quantized kea.add">,
    Statistic<"numResidualFused", "num-residual-fused",
              "quantized kea.add ops folded into a producer epilogue">,
    Statistic<"numRescaleComposed", "num-rescale-composed",
              "rescale pairs collapsed where composition is exact">,
    Statistic<"numRescaleRemoved", "num-rescale-removed",
              "rescale ops removed because they were bit-exact no-ops">,
    Statistic<"numRescaleRefused", "num-rescale-refused",
              "rescale pairs left alone because composition is not exact">,
    Statistic<"numShapeOpsFolded", "num-shape-ops-folded",
              "layout-no-op kea.reshape / kea.transpose ops removed">,
  ];

  let dependentDialects = ["::mlir::kea::KeaDialect"];
}

#endif // KEA_TRANSFORMS_PASSES
