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

def KeaTile : Pass<"kea-tile", "::mlir::func::FuncOp"> {
  let summary = "Lower fused KEA Level 1 tensor ops to Level 2 machine ops";
  let description = [{
    The pass that crosses the level boundary (ADR-0002): everything above it is
    graph normalisation, everything below it is hardware reality. It takes
    `-kea-fuse`'d Level 1 ops on tensors and produces a straight-line Level 2
    program of KEA-1 instructions on **symbolic** buffers.

    What it decides:

    * **Tile sizes**, from an explicit cost model built on `hw_config.h`'s
      occupancy formulas -- see docs/DIALECT_L2.md §5. Candidate output tiles
      are enumerated over the divisors of `OH`/`OW` and the output-channel group
      count, filtered by SPM_A / SPM_W / ACC capacity and by the ISA's field
      limits, and scored by
      `max(MXU, VPU, DMA)` cycles per tile times the tile count.
    * **The §8.5 convolution lowering**: one `kea.load_w` + `kea.mm` pair per
      `(kh, kw)` tap, all accumulating into one ACC region, activation base
      shifted per tap, `accumulate` absent only on `(ic0 == 0 && kh == 0 &&
      kw == 0)`, weight bank alternating `t & 1` so the load hides under the
      previous matmul.
    * **Padding**, as a `kea.vcopy` zero-point fill of the halo, because
      `kea.mm` has no predication (ISA.md §8.4a).

    What it deliberately does NOT decide:

    * **Addresses.** Buffers are `kea.alloc` values with a size, a role and a
      live range; `-kea-alloc` places them.
    * **Queues and semaphores.** The output is a correct *sequential* program
      with no `kea.signal` / `kea.wait` and no DMA engine assignment;
      `-kea-schedule` makes it parallel.

    Constraints it actively discharges, failing loudly rather than assuming
    (docs/DIALECT_L2.md §7): errata E5 (int32 ACC cannot overflow -- it wraps,
    it does not saturate), E6 (`KeaAddParam` bounds), E7 (no `kea.mm` against an
    unwritten weight bank) and ADR-0003 (`tosa_shift >= 31` or
    `|acc + bias| < 2^tosa_shift`, using the same accumulator range bound E5
    needs).
  }];

  let options = [
    // -kea-schedule needs room to double-buffer; if -kea-tile filled the
    // scratchpad exactly, nothing downstream could overlap DMA with compute.
    // 2 means "half of each scratchpad", i.e. one spare set of tiles.
    Option<"spmReserveFactor", "spm-reserve-factor", "int64_t", /*default=*/"2",
           "Divide each scratchpad's capacity by this before tiling, leaving "
           "room for -kea-schedule to double-buffer">,
    // Statistics are inert on this LLVM (Release+NDEBUG => NoopStatistic; see
    // compiler/README.md gotcha 21a), so publish the chosen tiles as an
    // attribute instead. This is what the tests assert against.
    Option<"reportTiles", "report-tiles", "bool", /*default=*/"false",
           "Attach the chosen tile shapes and their scratchpad footprints to "
           "the function as kea.tiling">,
  ];

  let statistics = [
    Statistic<"numLayers", "num-layers", "Level 1 ops lowered">,
    Statistic<"numTiles", "num-tiles", "output tiles emitted">,
    Statistic<"numMatmuls", "num-matmuls", "kea.mm ops emitted">,
  ];

  let dependentDialects = ["::mlir::kea::KeaDialect", "::mlir::arith::ArithDialect"];
}

def KeaAlloc : Pass<"kea-alloc", "::mlir::func::FuncOp"> {
  let summary = "Assign every KEA Level 2 buffer a compile-time-constant address";
  let description = [{
    The exact static memory planner. **There is no allocator at run time,
    anywhere**: after this pass every scratchpad and DRAM address in the program
    is a compile-time constant, and the runtime's only memory job is to hand the
    device one contiguous arena of a size this pass computed.

    In:  Level 2 with symbolic `kea.alloc` buffers (docs/DIALECT_L2.md §4).
    Out: the same program with a `kea.addr` base stamped on every `kea.alloc`,
         plus `kea.dram_layout`, `kea.spm_map` and `kea.occupancy` on the
         function.

    Per docs/DIALECT_L2.md §1.1(a) an address in the instruction stream is a
    `(buffer, displacement)` pair and `-kea-emit` writes
    `instr.X_addr = base(X) + X_addr`. `base(X)` is exactly the `kea.addr` this
    pass assigns. The displacements are deliberately left alone: they are what
    the Level 2 in-bounds verifiers check a strided walk against, so folding an
    absolute address into them would make every walk look out of bounds.

    What it decides, and how -- the long form is docs/MEMORY_PLANNING.md:

    * **Scratchpad placement by live-range packing.** Two on-chip buffers may
      share storage iff their `live = [first, last]` intervals are disjoint.
      The packer is greedy-by-size with first-fit-by-offset, the algorithm
      TensorFlow Lite Micro's `GreedyMemoryPlanner` uses; offline dynamic
      storage allocation is NP-hard, so this is an approximation and the pass
      reports how far it landed from the `maxlive` lower bound rather than
      pretending it is optimal.
    * **Units.** SPM_A/SPM_W are byte addressed, ACC is addressed in int32
      *words*. Sizes and offsets are always in the space's own unit, which is
      also the element type of the buffer (`i8` vs `i32`), so the two can never
      be mixed up silently.
    * **Alignment**, derived per buffer from the ops that actually use it
      (ISA.md §11.1): `LOAD_W` bases 16 B (8 B in int4), ACC bases 16 words,
      `VQUANT.qparam` / `VADD.param` blocks 4 B. The absolute addresses are
      re-checked against the same table afterwards.
    * **The DRAM arena**, as the three-region `KeafDramLayout` split: constants
      (weights / qparams / addparams), the host-visible I/O tensors, and the
      intermediate-activation scratch region -- which is live-range packed too,
      so a deep network reuses a handful of buffers instead of one per layer.

    Failure is loud: a program that cannot fit names the space, the peak
    demand, the capacity and the buffers alive at the peak, and says whether
    the shortfall is fundamental (`maxlive` alone exceeds capacity, so no
    allocator could do it -- retile) or fragmentation the packer introduced.
  }];

  let options = [
    // A silent aliasing bug -- two overlapping-live-range buffers handed
    // overlapping storage -- would surface as a numerical bug inside a kernel,
    // days away from its cause. The check is O(n log n + overlaps) on a few
    // dozen buffers, so it is on by default and there is no reason to turn it
    // off except to test that it fires.
    Option<"verifyPacking", "verify-packing", "bool", /*default=*/"true",
           "Prove no two overlapping-live-range buffers were given "
           "overlapping storage, and fail the pass if so">,
    // Runs the verifier over an existing placement without producing one. This
    // is how a hand-corrupted allocation is tested, and it makes the pass
    // usable as a checker for IR some other tool wrote.
    Option<"verifyOnly", "verify-only", "bool", /*default=*/"false",
           "Do not allocate; verify the kea.addr placement already in the IR">,
    // Statistics are inert on this LLVM install (Release+NDEBUG => NoopStatistic,
    // see compiler/README.md gotcha 21a), so the numbers that matter are
    // published as attributes. The SPM map is not debug output -- ARTIFACT_FORMAT.md
    // §5 makes it a KEAF section -- so it is on by default.
    Option<"reportMap", "report-map", "bool", /*default=*/"true",
           "Attach kea.spm_map, kea.dram_layout and kea.occupancy to the "
           "function">,
  ];

  let statistics = [
    Statistic<"numBuffers", "num-buffers", "kea.alloc buffers placed">,
    Statistic<"numShared", "num-shared",
              "buffers that reused storage another buffer also uses">,
  ];

  let dependentDialects = ["::mlir::kea::KeaDialect"];
}

#endif // KEA_TRANSFORMS_PASSES
