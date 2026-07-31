# add_mlir_dialect_library registers the target in the global MLIR_DIALECT_LIBS
# property, which is how tools/kea-opt picks it up, and implicitly adds a
# dependency on the `mlir-headers` target created by MLIRConfig.cmake.
add_mlir_dialect_library(MLIRKeaDialect
  KeaAttrs.cpp
  KeaDialect.cpp
  KeaOps.cpp
  KeaTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/kea/Dialect

  DEPENDS
  MLIRKeaOpsIncGen
  MLIRKeaAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
)
