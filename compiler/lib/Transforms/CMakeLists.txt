add_mlir_library(MLIRKeaTransforms
  Annotate.cpp
  CanonicalizeEvents.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/kea/Transforms

  DEPENDS
  MLIRKeaTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRKeaDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
)
