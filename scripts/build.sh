#!/usr/bin/env bash
# Configure and build the native half of KEA (ISA headers, simulator, runtime,
# tools) and run its tests. Idempotent.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build/native"
BUILD_TYPE="${BUILD_TYPE:-RelWithDebInfo}"

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" "$@"
cmake --build "$BUILD" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
ctest --test-dir "$BUILD" --output-on-failure --no-tests=ignore
