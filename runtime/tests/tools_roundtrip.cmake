# End-to-end check of the kea-as / kea-dis command lines:
#
#   kea-as  in.kasm --map in.map.json --const in.bin -o out.keaf
#   kea-dis out.keaf --map in.map.json        ->  byte-identical .kasm
#
# Run as a ctest entry via `cmake -P`.

function(fail msg)
  message(FATAL_ERROR "tools_roundtrip: ${msg}")
endfunction()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

# The 704-byte constant blob that double_buffered.map.json declares. The
# contents are irrelevant to the round trip; only the length has to match
# dram.const_bytes. 704 = 11 * 64.
set(_chunk "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
set(_blob "")
foreach(_i RANGE 1 11)
  string(APPEND _blob "${_chunk}")
endforeach()
file(WRITE "${WORK_DIR}/weights.bin" "${_blob}")
file(SIZE "${WORK_DIR}/weights.bin" _sz)
if(NOT _sz EQUAL 704)
  fail("could not build a 704-byte constant blob (got ${_sz})")
endif()

foreach(_stem all_opcodes double_buffered)
  set(_kasm "${KASM_DIR}/${_stem}.kasm")
  set(_map "${KASM_DIR}/${_stem}.map.json")
  set(_keaf "${WORK_DIR}/${_stem}.keaf")
  set(_out "${WORK_DIR}/${_stem}.kasm")

  set(_const_args)
  file(READ "${_map}" _map_text)
  if(_map_text MATCHES "\"const_bytes\"[ \t]*:[ \t]*704")
    set(_const_args --const "${WORK_DIR}/weights.bin")
  endif()

  execute_process(COMMAND "${KEA_AS}" "${_kasm}" --map "${_map}" ${_const_args} -o "${_keaf}"
                  RESULT_VARIABLE _rc ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 0)
    fail("kea-as failed on ${_stem}: ${_err}")
  endif()

  execute_process(COMMAND "${KEA_DIS}" "${_keaf}" --map "${_map}" -o "${_out}"
                  RESULT_VARIABLE _rc ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 0)
    fail("kea-dis failed on ${_stem}: ${_err}")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${_kasm}" "${_out}"
                  RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    fail("kea-dis(kea-as(${_stem}.kasm)) differs from the source")
  endif()

  # --annotate and --emit-map must at least run and produce output.
  execute_process(COMMAND "${KEA_DIS}" "${_keaf}" --map "${_map}" --annotate
                  RESULT_VARIABLE _rc OUTPUT_VARIABLE _ann)
  if(NOT _rc EQUAL 0 OR _ann STREQUAL "")
    fail("kea-dis --annotate failed on ${_stem}")
  endif()
  execute_process(COMMAND "${KEA_DIS}" "${_keaf}" --emit-map
                  RESULT_VARIABLE _rc OUTPUT_VARIABLE _emitted)
  if(NOT _rc EQUAL 0 OR NOT _emitted MATCHES "\"dram\"")
    fail("kea-dis --emit-map failed on ${_stem}")
  endif()
endforeach()

# kea-as must reject a broken listing with a non-zero exit status and a
# diagnostic naming the file and line.
file(WRITE "${WORK_DIR}/bad.kasm"
     ".arch \"KEA-1\"\n  MXU   MATMULL a_addr=a:0\n  CTRL  HALT    exit_code=0\n")
execute_process(COMMAND "${KEA_AS}" "${WORK_DIR}/bad.kasm" -o "${WORK_DIR}/bad.keaf"
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(_rc EQUAL 0)
  fail("kea-as accepted an unknown mnemonic")
endif()
if(NOT _err MATCHES "bad.kasm:2:" OR NOT _err MATCHES "unknown mnemonic")
  fail("kea-as diagnostic is missing a location or a message: ${_err}")
endif()

# --check parses without writing an artifact.
execute_process(COMMAND "${KEA_AS}" "${KASM_DIR}/messy.kasm" --check
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
  fail("kea-as --check failed on messy.kasm: ${_err}")
endif()

# kea-dis reads .kasm directly, which is the simulator's other front door.
execute_process(COMMAND "${KEA_DIS}" "${KASM_DIR}/messy.kasm"
                RESULT_VARIABLE _rc OUTPUT_VARIABLE _out)
if(NOT _rc EQUAL 0 OR NOT _out MATCHES "MATMUL")
  fail("kea-dis could not read a .kasm input")
endif()

message(STATUS "tools_roundtrip: OK")
