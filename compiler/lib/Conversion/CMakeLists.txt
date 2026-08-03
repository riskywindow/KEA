# add_mlir_conversion_library registers the target in the global
# MLIR_CONVERSION_LIBS property, which is how tools/kea-opt picks it up without
# having to name it explicitly (it already does
# `get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)`).
add_mlir_conversion_library(MLIRKeaConversion
  LinalgToKea.cpp
  TosaToKea.cpp
  WeightLayout.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/kea/Conversion

  DEPENDS
  MLIRKeaTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRKeaDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
  MLIRTosaDialect
  MLIRTransformUtils
  MLIRTransforms
)
