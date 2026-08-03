# KEA compiler — out-of-tree MLIR dialect

`kea` is the NPU-facing MLIR dialect of this project. This directory is a
**standalone CMake project** (there is deliberately no root `CMakeLists.txt`)
that builds against the system LLVM/MLIR install and produces `kea-opt`, a
`mlir-opt` clone that knows about `kea` in addition to every upstream dialect.

Status: the **Level 1 (tensor level) half is implemented** -- ops, verifiers,
`-tosa-to-kea`, `-linalg-to-kea` and `-kea-fuse`; see
[docs/DIALECT_L1.md](../docs/DIALECT_L1.md). The Level 2 (buffer level) half is
still the de-risking spike: `kea.dma_load` / `kea.mm` / `kea.signal` /
`kea.wait` in `KeaMachineOps.td`, with the tiling / allocation / scheduling
passes still to be written. **This directory is the pattern the rest of the
compiler copies**, so if you change a convention here, change it everywhere.

---

## 1. Build

```bash
bash scripts/build_compiler.sh          # configure + build + test
bash scripts/build_compiler.sh --clean  # from scratch
bash scripts/build_compiler.sh --no-test
```

Or by hand:

```bash
cmake -S compiler -B build/compiler \
      -DMLIR_DIR=/usr/local/opt/llvm/lib/cmake/mlir \
      -DLLVM_DIR=/usr/local/opt/llvm/lib/cmake/llvm \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build/compiler -j10
bash compiler/test/run_tests.sh build/compiler/bin
```

Environment overrides honoured by the script: `LLVM_PREFIX` (default
`/usr/local/opt/llvm`), `BUILD_TYPE` (default `Release`), `JOBS` (default 10),
`CMAKE_GENERATOR` (auto: Ninja if installed, else Unix Makefiles).

**Measured on this machine** (macOS, x86_64 under Rosetta, 12 cores, Apple
clang 16, Unix Makefiles, Release):

| step | time |
|---|---|
| `cmake` configure (cold) | ~19 s |
| full build `-j10` (cold) | ~49 s |
| no-op rebuild | ~1 s |
| test suite | ~1.7 s |
| **clean `build_compiler.sh` end to end** | **~75 s** |

`kea-opt` is a ~108 MB statically linked binary — that is normal, it links
every upstream MLIR dialect and conversion. Link time dominates incremental
builds; if that becomes annoying, trim the library list in
`tools/kea-opt/CMakeLists.txt` rather than switching to the `libMLIR.dylib`
shared build (mixing the two causes duplicate-registration crashes).

### Toolchain facts this was validated against

```
LLVM/MLIR              20.1.6 (Homebrew, /usr/local/opt/llvm)
LLVM_ENABLE_RTTI       ON
LLVM_ENABLE_EH         OFF
LLVM_ENABLE_ASSERTIONS OFF
C++ standard           17
cmake                  4.2.3
generator              Unix Makefiles (ninja not installed)
lit / llvm-lit         NOT shipped
FileCheck              shipped (/usr/local/opt/llvm/bin/FileCheck)
```

---

## 2. Layout

```
compiler/
  CMakeLists.txt                     top-level standalone project
  include/kea/
    Dialect/
      KeaDialect.td                  Dialect def + Kea_Op/Kea_Type/Kea_Attr bases
      KeaAttrs.td                    AddressSpace enum, #kea.address_space, #kea.tile_config
      KeaTypes.td                    !kea.buffer + address-space type constraints
      KeaOps.td                      LEVEL 1 ops (TableGen entry point for add_mlir_dialect)
      KeaMachineOps.td               LEVEL 2 ops -- its own -gen-op-{decls,defs} target
      Kea{Dialect,Attrs,Types,Ops,MachineOps}.h  thin headers wrapping the .inc files
      CMakeLists.txt                 tablegen rules
    Conversion/
      Passes.td                      -tosa-to-kea / -linalg-to-kea declarations
      Passes.h                       the populate*Patterns API
    Transforms/
      Passes.td                      every pass is declared or included here
      Passes.h                       GEN_PASS_DECL + GEN_PASS_REGISTRATION
      CMakeLists.txt
  lib/
    Dialect/{KeaDialect,KeaAttrs,KeaTypes,KeaOps,KeaMachineOps}.cpp
    Conversion/{TosaToKea,LinalgToKea,WeightLayout}.cpp
    Transforms/{Annotate,CanonicalizeEvents,Fuse}.cpp
    Target/Kasm/{EmitKasm,ConstBlob}.cpp   the backend (ADR-0001)
  tools/kea-opt/kea-opt.cpp          the driver
  tools/kea-translate/               Level 2 -> .kasm + .bin + .map.json
  test/                              .mlir tests + run_tests.sh
```

Generated headers land in `build/compiler/include/kea/...` and are included
with the same `kea/Dialect/Foo.h.inc` spelling as the hand-written ones,
because both `compiler/include` and `build/compiler/include` are on the include
path.

### What exists today

| thing | spelling |
|---|---|
| type | `!kea.buffer<8x16xi8, A>`, `!kea.buffer<f32, DRAM>` |
| enum attr | `#kea.address_space<ACC>`, `#kea.pool_kind<AVG>` |
| attr | `#kea.tile_config<16, 16, 32>` |
| attr (L1 quant) | `#kea.zp<input = -5, weight = 0>` |
| attr (L1 quant) | `#kea.quant<multiplier = [1073741824], shift = [30], input_zp = 0, output_zp = -5, axis = -1, rounding = DOUBLE>` |
| attr (L1 quant) | `#kea.epilogue<requant = <…>, clamp = [-128, 127], accum = <…>, residual = <…>, output = <…>>` |
| op (L1 tensor) | `kea.conv2d`, `kea.dwconv2d`, `kea.matmul`, `kea.fully_connected`, `kea.add`, `kea.pool`, `kea.rescale`, `kea.clamp`, `kea.reshape`, `kea.transpose` |
| op (L2 buffer) | `kea.dma_load %src to %dst : memref<…> to !kea.buffer<…>` |
| op (L2 buffer) | `kea.mm %a, %w, %acc` |
| op (L2 sync) | `kea.signal 3` / `kea.wait 3` |
| pass (L1 in) | `-tosa-to-kea`, `-linalg-to-kea` |
| pass (L1 → L1) | `-kea-fuse[=report-stats=true]` |
| pass (misc) | `-kea-annotate[=marker=npu]`, `-kea-canonicalize-events` |
| tool (backend) | `kea-translate` -- see [docs/CODEGEN.md](../docs/CODEGEN.md) |

**NB: the buffer-level matrix multiply is `kea.mm`, not `kea.matmul`.** The
spike called it `kea.matmul`; ADR-0002 gives that name to the Level 1 tensor op
and names the Level 2 one `kea.mm`. The definition is otherwise unchanged.

Address spaces: `A` (activation scratchpad), `W` (weight scratchpad), `ACC`
(accumulator bank, 32-bit only), `DRAM` (off chip).

---

## 3. How to add a new op

1. **Declare it in `include/kea/Dialect/KeaOps.td`**, deriving from `Kea_Op`:

   ```tablegen
   def Kea_ReluOp : Kea_Op<"relu", [Pure]> {
     let summary = "...";
     let arguments = (ins AnyRankedTensor:$input);
     let results   = (outs AnyRankedTensor:$output);
     let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
     let hasVerifier = 1;   // only if you actually write one
   }
   ```

2. **If `hasVerifier = 1`, implement it in `lib/Dialect/KeaOps.cpp`**:

   ```cpp
   LogicalResult ReluOp::verify() {
     if (...) return emitOpError("...");
     return success();
   }
   ```

3. Rebuild. Nothing else is needed — the op is added to the dialect
   automatically via the `GET_OP_LIST` include in `KeaDialect.cpp`.

4. **Add a round-trip line to `test/roundtrip.mlir`** and, if it has a
   verifier, a case in `test/invalid.mlir`.

Conventions:

* Operand/attribute names are `snake_case` in ODS (`zero_point`); the generated
  accessor is `getZeroPoint()`.
* Use `Arg<Type, "desc", [MemRead]>` for memory effects on operands. **Do not
  additionally put a `MemoryEffects<[...]>` trait on the same op** — ODS already
  adds `MemoryEffectOpInterface` and a `getEffects()` when it sees `Arg`
  effects, and you would get two definitions. Use the trait form only for ops
  with no operands (`kea.signal`, `kea.wait`).
* Mark side-effect-free ops `Pure` so `-canonicalize` / `-cse` can DCE them.
  This is verified in practice: a dead `kea.conv2d` is removed by
  `-canonicalize` while a dead `kea.dma_load` is not.

## 4. How to add a new pass

1. **Declare it in `include/kea/Transforms/Passes.td`**:

   ```tablegen
   def KeaFuseBias : Pass<"kea-fuse-bias", "::mlir::func::FuncOp"> {
     let summary = "...";
     let options    = [Option<"aggressive", "aggressive", "bool", "false", "...">];
     let statistics = [Statistic<"numFused", "num-fused", "...">];
     let dependentDialects = ["::mlir::kea::KeaDialect"];
   }
   ```

2. **Add `lib/Transforms/FuseBias.cpp`**:

   ```cpp
   #include "kea/Transforms/Passes.h"

   namespace mlir { namespace kea {
   #define GEN_PASS_DEF_KEAFUSEBIAS
   #include "kea/Transforms/Passes.h.inc"
   }}

   namespace {
   struct KeaFuseBiasPass
       : public mlir::kea::impl::KeaFuseBiasBase<KeaFuseBiasPass> {
     using mlir::kea::impl::KeaFuseBiasBase<KeaFuseBiasPass>::KeaFuseBiasBase;
     void runOnOperation() override { ... }
   };
   }
   ```

3. Add the file to `lib/Transforms/CMakeLists.txt`.

4. Add a test with a `// RUN: kea-opt %s -kea-fuse-bias | FileCheck %s` line.

Registration is automatic: `Passes.td` → `registerKeaPasses()` → called from
`tools/kea-opt/kea-opt.cpp`. Confirm with
`build/compiler/bin/kea-opt --help | grep kea-`.

## 5. Testing

`lit` is **not** shipped by the Homebrew LLVM bottle (`FileCheck` is), so
`compiler/test/CMakeLists.txt` deliberately does not call `add_lit_testsuite()`
— that would create a target nobody can run. Instead `test/run_tests.sh` is a
~100-line lit work-alike: it scans each `.mlir` for `// RUN:` lines, applies the
`%s` / `%t` substitutions, and runs them with `kea-opt` and `FileCheck` on
`PATH`. Tests are therefore written in **ordinary lit syntax** and will run
unmodified under real lit.

```bash
bash compiler/test/run_tests.sh build/compiler/bin   # direct
cmake --build build/compiler --target check-kea      # via cmake
ctest --test-dir build/compiler                      # via ctest
```

Unsupported lit features: `RUN:` line continuations (`\`), `REQUIRES:`,
`XFAIL:`, and the `not` tool (which Homebrew does not ship either — use
`-verify-diagnostics` for negative tests instead).

**If you ever get lit** (`pip install lit`), replace `test/CMakeLists.txt` with
the standard `configure_lit_site_cfg()` + `add_lit_testsuite()` pair and set
`-DLLVM_EXTERNAL_LIT=$(which lit)`. No test file needs to change.

---

## 6. Platform-specific gotchas (all of these actually bit)

### Build system

1. **`mlir-headers` / `mlir-doc` targets.** `add_mlir_dialect()` does
   `add_dependencies(mlir-headers ...)`. Out-of-tree that target does not exist
   — but `MLIRConfig.cmake` creates stubs for `mlir-headers`,
   `mlir-generic-headers`, `mlir-tablegen-targets` and `mlir-doc` if they are
   missing, so this works *provided you `find_package(MLIR)` before
   `include(AddMLIR)`*.

2. **There is no `add_mlir_pass_incgen()`** in MLIR 20's `AddMLIR.cmake`,
   despite it being widely referenced. Passes are wired up with a plain
   `mlir_tablegen(Passes.h.inc -gen-pass-decls -name Kea)` +
   `add_public_tablegen_target(...)`.

3. **RTTI/EH.** Brew LLVM 20.1.6 is built with `LLVM_ENABLE_RTTI=ON` and
   `LLVM_ENABLE_EH=OFF`, so no `-fno-rtti` mismatch here. Do not hand-roll
   these flags — `include(HandleLLVMOptions)` plus `llvm_update_compile_flags()`
   on each target propagates exactly what the install was built with, and will
   keep working if someone switches to an RTTI-less LLVM. Symptom of getting it
   wrong: `undefined symbol: typeinfo for mlir::Pass` at link time.

4. **Assertions.** The install has `LLVM_ENABLE_ASSERTIONS=OFF`. That is safe to
   mix with a local `Debug` build because `llvm/Config/abi-breaking.h` is
   installed and pins `LLVM_ENABLE_ABI_BREAKING_CHECKS` for us. Still, prefer
   `Release`: a `Debug` `kea-opt` links in minutes, not seconds.

5. **`enable_testing()` must be in the top-level `CMakeLists.txt`**, not in
   `test/`. Calling it only in the subdirectory produces `ctest: No tests were
   found!!!` even though `add_test()` ran.

6. **ninja is not installed** on this machine. The build script falls back to
   Unix Makefiles automatically; `brew install ninja` and it will pick Ninja up
   with no other change.

7. **macOS ships bash 3.2.** No `mapfile`, no associative arrays, and `set -u`
   trips on `${#arr[@]}` for empty arrays. `run_tests.sh` is written to that
   constraint.

### C++ / generated code

8. **`#include "mlir/Bytecode/BytecodeOpInterface.h"` is mandatory** in any
   header that includes a `-gen-op-decls` output, even if you never touch
   bytecode: MLIR ≥ 17 gives every ODS op a `BytecodeOpInterface::Trait`.
   Symptom: a cascade of
   `no member named 'BytecodeOpInterface' in namespace 'mlir'`, followed by
   dozens of bogus `use of undeclared identifier 'getOperation'` /
   `'getProperties'` errors — **the later errors are noise, always fix the
   first one.**

9. **`impl` is ambiguous.** With both `using namespace mlir;` and
   `using namespace mlir::kea;` in scope, `impl::KeaAnnotateBase` fails with
   `reference to 'impl' is ambiguous` (`mlir::impl` vs `mlir::kea::impl`).
   Always write `mlir::kea::impl::XxxBase<...>` in pass files.

10. **`Statistic` vs `statistics`.** The `Pass` ODS field is `statistics`
    (plural) and `options` (plural). Getting it wrong yields the unhelpful
    `error: Value 'statistic' unknown!`.

11. **`F64Attr` accessors return `llvm::APFloat`, not `double`.** It has neither
    `operator<=(double)` nor a `Diagnostic` streaming operator; call
    `.convertToDouble()`.

12. Generated file names from `add_mlir_dialect(KeaOps kea)` are
    `KeaOps.h.inc`, `KeaOpsTypes.h.inc`, `KeaOpsDialect.h.inc` — the *first*
    argument is a filename prefix, not the dialect name. That is why
    `KeaTypes.h` includes `KeaOpsTypes.h.inc`.

13. `add_mlir_dialect()` does **not** generate enum or `AttrDef` code; those need
    their own `mlir_tablegen(... -gen-enum-decls / -gen-attrdef-decls)` calls
    (see `include/kea/Dialect/CMakeLists.txt`).

### ODS assembly formats

14. **`qualified(type($x))` is required for your own dialect's types in an op's
    assembly format.** Without it MLIR elides the dialect prefix when the type
    and the op belong to the same dialect, printing
    `kea.matmul ... : <8x16xi8, A>` instead of `!kea.buffer<8x16xi8, A>`. It
    still round-trips, but it is unreadable and defeats FileCheck patterns.

15. **You cannot use `` `x` `` as a literal separator between integers.** The
    MLIR lexer reads `16x16x32` as the integer `16` followed by the bare
    identifier `x16x32`, so a `` `x` `` literal never matches. Hence
    `#kea.tile_config<16, 16, 32>` uses commas. The only way to get the
    `AxBxC` spelling is `AsmParser::parseDimensionList(shape, allowDynamic,
    withTrailingX)` — see `custom<Shape>` in `lib/Dialect/KeaTypes.cpp`.

16. **Custom directive signatures for `TypeDef`/`AttrDef` formats** (these differ
    from the op-level ones, which take `OpAsmParser`):

    ```cpp
    // custom<Shape>($shape, $elementType) with
    //   ArrayRefParameter<"int64_t">:$shape, "::mlir::Type":$elementType
    ParseResult parseShape(AsmParser &, SmallVectorImpl<int64_t> &, Type &);
    void        printShape(AsmPrinter &, ArrayRef<int64_t>, Type);
    ```

    They must be declared *inside `namespace mlir::kea`* and *before* the
    `#include "...Types.cpp.inc"`, because the generated code calls them
    unqualified. Parse gets the parameter's **storage** type by mutable
    reference, print gets the **cpp** type by value.

17. **`EnumParameter<...>` prints as a bare keyword** inside a type
    (`!kea.buffer<…, ACC>`), while the `EnumAttr` wrapper prints as
    `#kea.address_space<ACC>`. Both spellings exist on purpose; they are
    different things.

18. `useDefaultTypePrinterParser = 1` / `useDefaultAttributePrinterParser = 1` on
    the `Dialect` are what connect `assemblyFormat` on a `TypeDef`/`AttrDef` to
    the dialect's parse/print hooks. Without them your type parses as
    "unknown type in dialect kea".

19. **`expected-error {{...}}` in `-verify-diagnostics` is a literal substring
    match, not a regex.** Paste the exact diagnostic text. Note also that an
    `AttrDef` verifier (`genVerifyDecl`) fires at *parse* time and its
    diagnostic is located at the **attribute**, not at the op, so the
    `expected-error` comment must sit above the attribute's line.

19a. **A comma-led optional operand group commits on the comma.**
    `(`,` `bias` $bias^)? (`,` `residual` $residual^)?` cannot parse
    `%in, %w, residual %r` -- it sees the comma, enters the bias group and
    fails with `expected 'bias'`. Key optional groups on their keyword instead:
    `(`bias` $bias^)? (`residual` $residual^)?`, giving `%in, %w bias %b
    residual %r`.

19b. **An attribute cannot directly follow an SSA operand in an assembly
    format.** MLIR lexes `%b #kea.zp<…>` as "result `#…` of value `%b`" and
    reports `invalid SSA value result number`. Put the attribute in
    `attr-dict`, or separate it with a keyword literal.

19c. **`SameOperandsAndResultType` lives in
    `mlir/Interfaces/InferTypeOpInterface.td`, not `OpBase.td`**, and ODS
    silently adds `InferTypeOpInterface::Trait` to any op that uses it -- so
    the corresponding `.h` include becomes mandatory (see gotcha 8 for the
    identical failure mode).

### Upstream dialect versions

20. **`tosa.conv2d` in MLIR 20.1.6 takes three operands** (`input`, `weight`,
    `bias`) and no zero-point operands. LLVM 21 switches to TOSA 1.0, which adds
    `%input_zp` / `%weight_zp` operands and renames `tosa.const`'s `value`
    attribute to `values`. When we upgrade LLVM,
    `test/upstream-dialects.mlir` is the first thing that will break — that is
    intentional, it is the canary.

21a. **Pass `Statistic`s are compiled out by this LLVM install.** The bottle is
    Release with `NDEBUG` and without `LLVM_FORCE_ENABLE_STATS`, so
    `llvm::Statistic` is `NoopStatistic` and `-mlir-pass-statistics` prints an
    empty report -- for upstream passes too (verify with
    `mlir-opt --symbol-dce -mlir-pass-statistics`). It cannot be fixed from out
    of tree, because the printer lives in the prebuilt `libMLIRPass` and
    forcing the macro on locally would change `Pass::Statistic`'s layout
    relative to it. `-kea-fuse` therefore keeps plain counters and publishes
    them with `-kea-fuse=report-stats=true` in addition to declaring the
    `Statistic`s.

21. **`--tosa-to-linalg-named` is an `OperationPass<func::FuncOp>`** and cannot be
    given directly on the command line at module scope:
    `unable to schedule pass 'TosaToLinalgNamed' on a PassManager intended to
    run on 'builtin.module'`. Nest it:
    `--pass-pipeline='builtin.module(func.func(tosa-to-linalg-named))'`.

---

## 7. Constraints for everyone else

* **Do not create a root `CMakeLists.txt`.** `compiler/` is intentionally a
  standalone project so the runtime/sim/frontend trees can use their own build
  systems.
* **Everything the pipeline needs is already in `kea-opt`**:
  `registerAllDialects` + `registerAllExtensions` + `registerAllPasses` + KEA.
  `tosa`, `linalg`, `tensor`, `memref`, `arith`, `func`, `scf` all parse, and
  upstream conversion passes run. There is no need for a second driver binary.
* **Pin to MLIR 20.1.6 for now.** Gotchas 20 and 21 are version-specific and the
  TOSA 1.0 migration in LLVM 21 is a real (small) porting job.
* `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` is set by the build script, so
  `build/compiler/compile_commands.json` exists — symlink it to the repo root
  for clangd.
