add_executable(kea-sim kea-sim.cpp)
target_link_libraries(kea-sim PRIVATE kea_sim kea_flags)

# Loading .keaf / .kasm is runtime/'s job. If runtime/ is not part of this
# configuration, kea-sim still builds; its loader just explains why it cannot
# read an artifact.
if(TARGET kea_runtime)
  target_link_libraries(kea-sim PRIVATE kea_runtime)
  target_compile_definitions(kea-sim PRIVATE KEA_SIM_HAVE_RUNTIME=1)
else()
  target_compile_definitions(kea-sim PRIVATE KEA_SIM_HAVE_RUNTIME=0)
  message(STATUS "kea-sim: building without runtime/; artifact loading disabled")
endif()

# End-to-end smoke test: assemble-and-run a .kasm through the runtime loader
# and check the tool produces a report plus machine-readable stats. The map
# declares 704 bytes of constants, so synthesize a filler blob -- the test is
# about the toolchain path, not about the weight values.
if(TARGET kea_runtime AND EXISTS ${CMAKE_SOURCE_DIR}/runtime/tests/kasm/double_buffered.kasm)
  string(REPEAT "A" 704 _kea_sim_filler)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/filler.const.bin "${_kea_sim_filler}")
  add_test(NAME sim.kea-sim.e2e
           COMMAND kea-sim
                   ${CMAKE_SOURCE_DIR}/runtime/tests/kasm/double_buffered.kasm
                   --map ${CMAKE_SOURCE_DIR}/runtime/tests/kasm/double_buffered.map.json
                   --const ${CMAKE_CURRENT_BINARY_DIR}/filler.const.bin
                   --stats-json ${CMAKE_CURRENT_BINARY_DIR}/double_buffered.stats.json
                   --trace=${CMAKE_CURRENT_BINARY_DIR}/double_buffered.trace.txt
                   --max-cycles 10000000)
endif()
