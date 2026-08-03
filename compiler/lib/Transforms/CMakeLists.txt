add_mlir_library(MLIRKeaTransforms
  Alloc.cpp
  Annotate.cpp
  CanonicalizeEvents.cpp
  Fuse.cpp
  Schedule.cpp
  Tile.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/kea/Transforms

  DEPENDS
  MLIRKeaTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRKeaDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRTransforms
)
