# kea-translate links only what it needs: the KEA dialect, the emission
# library, and the two upstream dialects a Level 2 module can contain (arith
# for the weight constants, func for the enclosing function). No conversion or
# transform libraries, so this binary is a fraction of kea-opt's size and its
# link is not on the critical path of an ordinary rebuild.
set(LIBS
  MLIRKeaDialect
  MLIRKeaTargetKasm
  MLIRArithDialect
  MLIRFuncDialect
  MLIRParser
  MLIRSupport
  MLIRIR
)

add_llvm_executable(kea-translate kea-translate.cpp)
llvm_update_compile_flags(kea-translate)
target_link_libraries(kea-translate PRIVATE ${LIBS})
mlir_check_all_link_libraries(kea-translate)
