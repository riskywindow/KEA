cmake_minimum_required(VERSION 3.20)
project(kea LANGUAGES CXX)

# ---------------------------------------------------------------------------
# This is the *native* half of the KEA project: the ISA headers, the
# cycle-approximate simulator, the runtime, and the command line tools. It has
# no dependency on LLVM or MLIR and builds in a few seconds.
#
# The MLIR compiler (`compiler/`) is a separate, standalone CMake project
# because it needs the out-of-tree MLIR CMake machinery. Build it with
# `scripts/build_compiler.sh`.
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(KEA_WERROR "Treat warnings as errors" OFF)
add_library(kea_flags INTERFACE)
target_compile_options(kea_flags INTERFACE -Wall -Wextra -Wno-unused-parameter)
if(KEA_WERROR)
  target_compile_options(kea_flags INTERFACE -Werror)
endif()

# The frozen hardware contract: ISA encoding, machine parameters, artifact
# format. Header-only, shared by the simulator, the runtime, the compiler
# backend and every tool.
add_library(kea_isa INTERFACE)
target_include_directories(kea_isa INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kea_isa INTERFACE kea_flags)

enable_testing()

foreach(_sub sim runtime tools tests)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_sub}/CMakeLists.txt)
    add_subdirectory(${_sub})
  endif()
endforeach()
