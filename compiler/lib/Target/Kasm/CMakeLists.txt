add_mlir_library(MLIRKeaTargetKasm
  ConstBlob.cpp
  EmitKasm.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/kea/Target/Kasm

  DEPENDS
  MLIRKeaMachineOpsIncGen

  LINK_LIBS PUBLIC
  MLIRKeaDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)
