# The KEA-1 cycle-approximate simulator: a bit-exact functional model plus a
# concurrent timing model, both driven entirely by include/kea/hw_config.h.

add_library(kea_sim STATIC
  src/functional.cpp
  src/simulator.cpp
  src/stats.cpp
)

target_include_directories(kea_sim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kea_sim PUBLIC kea_isa)

add_subdirectory(tests)
