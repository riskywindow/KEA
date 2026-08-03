add_executable(kea-dis kea-dis.cpp)
target_link_libraries(kea-dis PRIVATE kea_runtime kea_flags)

# End-to-end exercise of the assembler/disassembler command lines. Registered
# here rather than in runtime/tests/ because the root project configures
# runtime/ before tools/, so the tool targets do not exist there yet.
if(TARGET kea-as AND KEA_TOOLS_TEST_SCRIPT)
  add_test(NAME runtime.tools_roundtrip
           COMMAND ${CMAKE_COMMAND}
                   -DKEA_AS=$<TARGET_FILE:kea-as>
                   -DKEA_DIS=$<TARGET_FILE:kea-dis>
                   -DKASM_DIR=${KEA_KASM_CORPUS_DIR}
                   -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/tools_roundtrip
                   -P ${KEA_TOOLS_TEST_SCRIPT})
endif()
