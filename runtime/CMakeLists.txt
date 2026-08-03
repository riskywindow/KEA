# The KEA runtime: KEAF reader/writer, the `.kasm` assembler and
# disassembler, the model-map/JSON layer and the DRAM staging arena.
#
# Target name is `kea_runtime` and is depended on by name by tools/ and by the
# simulator (which gates on `if(TARGET kea_runtime)`).

add_library(kea_runtime STATIC
  src/json.cpp
  src/diagnostics.cpp
  src/op_fields.cpp
  src/model_map.cpp
  src/assembler.cpp
  src/disassembler.cpp
  src/keaf_io.cpp
  src/dram_arena.cpp
  src/load.cpp
)

target_include_directories(kea_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kea_runtime PUBLIC kea_isa PRIVATE kea_flags)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
  add_subdirectory(tests)
endif()
