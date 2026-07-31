#!/usr/bin/env bash
#===----------------------------------------------------------------------===#
# Configure, build and test the KEA MLIR compiler.
#
# Idempotent: safe to re-run; CMake reconfigures only when needed.
#
#   bash scripts/build_compiler.sh              # configure + build + test
#   bash scripts/build_compiler.sh --clean      # wipe build/compiler first
#   bash scripts/build_compiler.sh --no-test    # skip the test run
#
# Exits non-zero on any configure/build/test failure.
#===----------------------------------------------------------------------===#

set -e
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/compiler"
BUILD_DIR="${REPO_ROOT}/build/compiler"

# Homebrew LLVM/MLIR 20.1.6. Override with LLVM_PREFIX=... if you relocate it.
LLVM_PREFIX="${LLVM_PREFIX:-/usr/local/opt/llvm}"
MLIR_DIR="${LLVM_PREFIX}/lib/cmake/mlir"
LLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm"

BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-10}"

RUN_TESTS=1
CLEAN=0
for arg in "$@"; do
  case "${arg}" in
    --clean)   CLEAN=1 ;;
    --no-test) RUN_TESTS=0 ;;
    -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

#===----------------------------------------------------------------------===#
# Preflight
#===----------------------------------------------------------------------===#

for f in "${MLIR_DIR}/MLIRConfig.cmake" "${LLVM_DIR}/LLVMConfig.cmake"; do
  if [[ ! -f "${f}" ]]; then
    echo "error: ${f} not found." >&2
    echo "       Install LLVM+MLIR (brew install llvm) or set LLVM_PREFIX." >&2
    exit 1
  fi
done

if [[ ${CLEAN} -eq 1 ]]; then
  echo ">>> removing ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

# `ninja` is not installed on this machine, so use Unix Makefiles unless a
# generator is forced (CMAKE_GENERATOR=Ninja works fine if you brew install it).
if [[ -z "${CMAKE_GENERATOR}" ]]; then
  if command -v ninja >/dev/null 2>&1; then
    CMAKE_GENERATOR="Ninja"
  else
    CMAKE_GENERATOR="Unix Makefiles"
  fi
fi
export CMAKE_GENERATOR

#===----------------------------------------------------------------------===#
# Configure
#===----------------------------------------------------------------------===#

echo ">>> configuring (${CMAKE_GENERATOR}, ${BUILD_TYPE})"
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -G "${CMAKE_GENERATOR}" \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

#===----------------------------------------------------------------------===#
# Build
#===----------------------------------------------------------------------===#

echo ">>> building with -j${JOBS}"
START=$(date +%s)
cmake --build "${BUILD_DIR}" -j"${JOBS}"
END=$(date +%s)
echo ">>> build finished in $((END - START))s"

if [[ ! -x "${BUILD_DIR}/bin/kea-opt" ]]; then
  echo "error: ${BUILD_DIR}/bin/kea-opt was not produced" >&2
  exit 1
fi

#===----------------------------------------------------------------------===#
# Test
#===----------------------------------------------------------------------===#

if [[ ${RUN_TESTS} -eq 1 ]]; then
  echo ">>> running tests"
  LLVM_BIN_DIR="${LLVM_PREFIX}/bin" \
    bash "${SRC_DIR}/test/run_tests.sh" "${BUILD_DIR}/bin"
fi

echo
echo ">>> OK: ${BUILD_DIR}/bin/kea-opt"
