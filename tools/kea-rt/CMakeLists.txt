# kea-rt is the runtime driver: it loads an artifact, allocates the DRAM arena,
# stages inputs, runs the program on a device, and extracts outputs.
#
# The "device" is the simulator, so the tool only exists when `sim/` was
# configured. Gating here (rather than failing to link) is what lets the
# runtime half of the project build and test on its own.
if(TARGET kea_sim)
  add_executable(kea-rt kea-rt.cpp)
  target_link_libraries(kea-rt PRIVATE kea_runtime kea_sim kea_flags)

  if(TARGET kea-as AND KEA_RT_TEST_SCRIPT)
    add_test(NAME runtime.tools_rt
             COMMAND ${CMAKE_COMMAND}
                     -DKEA_AS=$<TARGET_FILE:kea-as>
                     -DKEA_RT=$<TARGET_FILE:kea-rt>
                     -DKASM_DIR=${KEA_KASM_CORPUS_DIR}
                     -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/tools_rt
                     -P ${KEA_RT_TEST_SCRIPT})
  endif()
else()
  message(STATUS "kea-rt: skipped (needs the kea_sim target)")
endif()
