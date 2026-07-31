set(KEA_SIM_TESTS
  test_functional
  test_conv3x3
  test_timing
  test_sync
  test_bench
)

foreach(_t ${KEA_SIM_TESTS})
  add_executable(${_t} ${_t}.cpp)
  target_link_libraries(${_t} PRIVATE kea_sim kea_flags)
  target_include_directories(${_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME sim.${_t} COMMAND ${_t})
endforeach()

# The requantization conformance sweep needs the frontend's golden vectors.
# They are produced by frontend/gen_apply_scale_vectors.py; if they are not
# checked in yet the test is simply not registered rather than failing the
# build.
set(_scale_vectors ${CMAKE_SOURCE_DIR}/frontend/testdata/apply_scale_vectors.json)
add_executable(test_apply_scale test_apply_scale.cpp)
target_link_libraries(test_apply_scale PRIVATE kea_sim kea_flags)
target_include_directories(test_apply_scale PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(EXISTS ${_scale_vectors})
  add_test(NAME sim.test_apply_scale COMMAND test_apply_scale ${_scale_vectors})
else()
  message(WARNING
    "frontend/testdata/apply_scale_vectors.json is missing; the requantization "
    "conformance sweep will not run")
endif()
