#===----------------------------------------------------------------------===#
# KEA NPU compiler -- standalone (out-of-tree) MLIR project.
#
# This is a TOP-LEVEL CMake project on purpose: there is deliberately no root
# CMakeLists.txt for the npu repo. Configure this directory directly:
#
#   cmake -S compiler -B build/compiler \
#         -DMLIR_DIR=/usr/local/opt/llvm/lib/cmake/mlir \
#         -DLLVM_DIR=/usr/local/opt/llvm/lib/cmake/llvm \
#         -DCMAKE_BUILD_TYPE=Release
#
# or just run scripts/build_compiler.sh.
#===----------------------------------------------------------------------===#

cmake_minimum_required(VERSION 3.20)

project(kea-compiler LANGUAGES C CXX)

#===----------------------------------------------------------------------===#
# Find MLIR / LLVM
#===----------------------------------------------------------------------===#

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM version           : ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM_ENABLE_RTTI       : ${LLVM_ENABLE_RTTI}")
message(STATUS "LLVM_ENABLE_EH         : ${LLVM_ENABLE_EH}")
message(STATUS "LLVM_ENABLE_ASSERTIONS : ${LLVM_ENABLE_ASSERTIONS}")

# MLIR 20 is C++17. Set this BEFORE HandleLLVMOptions so it does not get
# clobbered; HandleLLVMOptions also honours LLVM_REQUIRED_CXX_STANDARD.
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Put binaries/libs where add_llvm_executable / add_mlir_library expect them.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
# add_mlir_doc and friends reference MLIR_BINARY_DIR; point it at ours.
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
# HandleLLVMOptions propagates -fno-rtti / -fno-exceptions / -std=... exactly as
# the LLVM install was built. Including it is what keeps us ABI-compatible with
# the Homebrew libMLIR*.a; do not drop it and hand-roll flags.
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#===----------------------------------------------------------------------===#
# Subdirectories
#===----------------------------------------------------------------------===#

# enable_testing() must be called from the *top-level* CMakeLists, otherwise no
# CTestTestfile is emitted in the root binary dir and `ctest` reports
# "No tests were found!!!" even though add_test() ran in a subdirectory.
enable_testing()

add_subdirectory(include/kea/Dialect)
add_subdirectory(include/kea/Transforms)
add_subdirectory(lib/Dialect)
add_subdirectory(lib/Transforms)
add_subdirectory(tools/kea-opt)
add_subdirectory(test)
