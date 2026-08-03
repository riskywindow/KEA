# Pass registration TableGen.
#
# NB: there is no `add_mlir_pass_incgen()` helper in MLIR 20's AddMLIR.cmake --
# passes are wired up with a plain mlir_tablegen() + add_public_tablegen_target()
# pair, exactly as upstream does in mlir/include/mlir/Dialect/*/Transforms.
#
# `-name Kea` controls the generated symbol names: registerKeaPasses(),
# GEN_PASS_DECL_KEAANNOTATE, impl::KeaAnnotateBase, ...
set(LLVM_TARGET_DEFINITIONS Passes.td)
# Passes.td `include`s ../Conversion/Passes.td. Under the Makefile generator
# LLVM's tablegen() only globs *.td in THIS directory, so name the cross-
# directory input explicitly or edits to it will not trigger a re-run.
set(LLVM_TARGET_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../Conversion/Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Kea)
add_public_tablegen_target(MLIRKeaTransformsIncGen)
add_dependencies(mlir-headers MLIRKeaTransformsIncGen)
