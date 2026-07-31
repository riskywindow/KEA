# MLIRConfig.cmake populates these global properties with every installed
# upstream library, so the same idiom as in-tree mlir-opt works out-of-tree.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRKeaDialect
  MLIRKeaTransforms
  MLIROptLib
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRIR
)

add_llvm_executable(kea-opt kea-opt.cpp)

# Applies the LLVM install's compile flags (-fno-rtti/-fno-exceptions/-std=...)
# to a target created with add_llvm_executable. Skipping this is the classic
# cause of "undefined symbol: typeinfo for mlir::Pass"-style link failures.
llvm_update_compile_flags(kea-opt)

target_link_libraries(kea-opt PRIVATE ${LIBS})

# Diagnoses libraries linked into more than one aggregate, which silently
# produces duplicate pass/dialect registration at runtime.
mlir_check_all_link_libraries(kea-opt)
