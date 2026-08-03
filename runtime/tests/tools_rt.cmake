# End-to-end check of the kea-rt command line:
#
#   kea-as echo.kasm --map echo.map.json -o echo.keaf
#   kea-rt echo.keaf --input input=in.bin --output output=out.bin
#
# echo.kasm copies its 256-byte input tensor through SPM_A into the first half
# of its 512-byte output tensor and fills the second half with the byte 42
# ('*'). Checking both halves proves that the host input reached the device,
# that the device actually executed, and that the output came back out of the
# DRAM arena rather than being left zeroed.
#
# Run as a ctest entry via `cmake -P`.

function(fail msg)
  message(FATAL_ERROR "tools_rt: ${msg}")
endfunction()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

# 256 bytes of recognisable input. 256 = 4 * 64.
set(_chunk "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-")
set(_in "")
foreach(_i RANGE 1 4)
  string(APPEND _in "${_chunk}")
endforeach()
file(WRITE "${WORK_DIR}/in.bin" "${_in}")
file(SIZE "${WORK_DIR}/in.bin" _sz)
if(NOT _sz EQUAL 256)
  fail("could not build a 256-byte input (got ${_sz})")
endif()

execute_process(COMMAND "${KEA_AS}" "${KASM_DIR}/echo.kasm" --map "${KASM_DIR}/echo.map.json"
                        -o "${WORK_DIR}/echo.keaf"
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
  fail("kea-as failed: ${_err}")
endif()

# --list-tensors must describe the artifact's I/O without running anything.
execute_process(COMMAND "${KEA_RT}" "${WORK_DIR}/echo.keaf" --list-tensors
                RESULT_VARIABLE _rc OUTPUT_VARIABLE _tensors)
if(NOT _rc EQUAL 0 OR NOT _tensors MATCHES "output" OR NOT _tensors MATCHES "input")
  fail("kea-rt --list-tensors did not report the tensor table")
endif()

execute_process(COMMAND "${KEA_RT}" "${WORK_DIR}/echo.keaf"
                        --input input=${WORK_DIR}/in.bin
                        --output output=${WORK_DIR}/out.bin
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
  fail("kea-rt failed: ${_err}")
endif()

file(SIZE "${WORK_DIR}/out.bin" _osz)
if(NOT _osz EQUAL 512)
  fail("expected a 512-byte output tensor, got ${_osz}")
endif()

file(READ "${WORK_DIR}/out.bin" _out)
string(SUBSTRING "${_out}" 0 256 _echoed)
string(SUBSTRING "${_out}" 256 256 _filled)
if(NOT _echoed STREQUAL _in)
  fail("the first half of the output is not the input that was staged")
endif()
string(REPEAT "*" 256 _expected_fill)
if(NOT _filled STREQUAL _expected_fill)
  fail("the VCOPY fill half of the output is wrong")
endif()

# kea-rt must reject an input file whose size disagrees with the tensor.
file(WRITE "${WORK_DIR}/short.bin" "too short")
execute_process(COMMAND "${KEA_RT}" "${WORK_DIR}/echo.keaf" --input input=${WORK_DIR}/short.bin
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(_rc EQUAL 0 OR NOT _err MATCHES "expects 256")
  fail("kea-rt accepted a wrongly sized input tensor: ${_err}")
endif()

# ... and an unknown tensor name.
execute_process(COMMAND "${KEA_RT}" "${WORK_DIR}/echo.keaf" --input nope=${WORK_DIR}/in.bin
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(_rc EQUAL 0 OR NOT _err MATCHES "no tensor named 'nope'")
  fail("kea-rt accepted an unknown tensor name: ${_err}")
endif()

# kea-rt reads `.kasm` directly too, given the map.
execute_process(COMMAND "${KEA_RT}" "${KASM_DIR}/echo.kasm" --map "${KASM_DIR}/echo.map.json"
                        --input input=${WORK_DIR}/in.bin
                        --output output=${WORK_DIR}/out2.bin
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
  fail("kea-rt could not run a .kasm input: ${_err}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${WORK_DIR}/out.bin" "${WORK_DIR}/out2.bin"
                RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  fail("running the .kasm and the .keaf produced different outputs")
endif()

message(STATUS "tools_rt: OK")
