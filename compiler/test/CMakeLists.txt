# NOTE: this deliberately does NOT use add_lit_testsuite(). The Homebrew LLVM
# 20.1.6 bottle does not ship lit/llvm-lit, so a lit test suite would be a
# target that can never be executed. run_tests.sh is a minimal lit work-alike
# that understands the `// RUN:` lines in the .mlir files; see its header
# comment and compiler/README.md.
#
# If lit ever becomes available (`pip install lit`), the tests themselves need
# no changes -- replace this file with the standard configure_lit_site_cfg() +
# add_lit_testsuite() incantation and point LLVM_EXTERNAL_LIT at it.

find_program(LLVM_EXTERNAL_LIT NAMES llvm-lit lit lit.py)
if(LLVM_EXTERNAL_LIT)
  message(STATUS "lit found at ${LLVM_EXTERNAL_LIT}; "
                 "the KEA test suite still runs through run_tests.sh "
                 "(see compiler/test/CMakeLists.txt)")
else()
  message(STATUS "lit not found in the LLVM install; using run_tests.sh")
endif()

add_custom_target(check-kea
  COMMAND ${CMAKE_COMMAND} -E env
          "LLVM_BIN_DIR=${LLVM_TOOLS_BINARY_DIR}"
          bash ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh $<TARGET_FILE_DIR:kea-opt>
  DEPENDS kea-opt
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running the KEA regression tests"
  USES_TERMINAL
)

# enable_testing() lives in the top-level CMakeLists, not here.
add_test(NAME kea-tests
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh $<TARGET_FILE_DIR:kea-opt>)
set_tests_properties(kea-tests PROPERTIES
  ENVIRONMENT "LLVM_BIN_DIR=${LLVM_TOOLS_BINARY_DIR}")
