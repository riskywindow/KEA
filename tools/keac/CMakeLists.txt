# keac links nothing: it is a process driver (ADR-0001, "a thin driver that
# runs this pipeline end to end"). kea_flags only carries the project's warning
# and standard settings.
add_executable(keac keac.cpp)
target_link_libraries(keac PRIVATE kea_flags)
