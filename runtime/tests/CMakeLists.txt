# Runtime tests. Each file is a standalone executable and one ctest entry, so a
# failure names itself and there is no framework to learn.

set(KEA_RUNTIME_TESTS
  test_json
  test_op_fields
  test_roundtrip_kasm
  test_keaf_io
  test_golden_encoding
  test_asm_errors
  test_no_alloc
)

foreach(_t ${KEA_RUNTIME_TESTS})
  add_executable(${_t} ${_t}.cpp)
  target_link_libraries(${_t} PRIVATE kea_runtime kea_flags)
  target_include_directories(${_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${_t} PRIVATE KEA_KASM_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kasm")
  add_test(NAME runtime.${_t} COMMAND ${_t})
endforeach()

# The end-to-end tool tests are registered from tools/kea-dis/CMakeLists.txt,
# because the root project configures runtime/ before tools/ and the tool
# targets do not exist yet at this point. Export what those tests need.
set(KEA_KASM_CORPUS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kasm CACHE INTERNAL "")
set(KEA_TOOLS_TEST_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/tools_roundtrip.cmake CACHE INTERNAL "")
set(KEA_RT_TEST_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/tools_rt.cmake CACHE INTERNAL "")
