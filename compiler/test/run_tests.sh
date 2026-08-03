#!/usr/bin/env bash
#===----------------------------------------------------------------------===#
# A ~100 line stand-in for `llvm-lit`.
#
# WHY THIS EXISTS
# ---------------
# The Homebrew `llvm` 20.1.6 bottle ships FileCheck, split-file and the tblgens,
# but it does NOT ship `lit` / `llvm-lit` / `lit.py` (verified:
#   ls /usr/local/opt/llvm/bin | grep -E 'FileCheck|lit'   -> FileCheck only
#   find /usr/local/opt/llvm -name 'lit.py' -o -name 'llvm-lit'  -> nothing
# ). Because of that, wiring up `add_lit_testsuite()` would produce a build
# target that can never run, so we do not use it.
#
# Instead this script implements the small subset of lit that our tests need:
# it scans each .mlir file for `// RUN:` lines and executes them, with the
# usual `%s` / `%S` / `%t` substitutions. The tests are therefore written in ordinary
# lit syntax and will run unmodified under real lit the day someone does
# `pip install lit` and switches compiler/test/CMakeLists.txt over to
# add_lit_testsuite(); see compiler/README.md.
#
# Deliberate limitations: no `RUN:` line continuations (`\`), no `REQUIRES:`,
# no `XFAIL:`, no `not` tool (use mlir-opt's `-verify-diagnostics` instead).
#
# macOS ships bash 3.2, so: no `mapfile`, no associative arrays, and `set -u`
# is left off because `${#arr[@]}` on an empty array trips it in 3.2.
#
# Usage: run_tests.sh [<dir containing kea-opt>]
#===----------------------------------------------------------------------===#

set -o pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Where to find kea-opt: argv[1], then $KEA_BIN_DIR, then the default build dir.
KEA_BIN_DIR="${1:-${KEA_BIN_DIR:-$(cd "${TEST_DIR}/../.." && pwd)/build/compiler/bin}}"
LLVM_BIN_DIR="${LLVM_BIN_DIR:-/usr/local/opt/llvm/bin}"

if [[ ! -x "${KEA_BIN_DIR}/kea-opt" ]]; then
  echo "error: kea-opt not found at ${KEA_BIN_DIR}/kea-opt" >&2
  echo "       build it first (scripts/build_compiler.sh) or pass its directory" >&2
  exit 1
fi

if [[ ! -x "${LLVM_BIN_DIR}/FileCheck" ]]; then
  echo "error: FileCheck not found at ${LLVM_BIN_DIR}/FileCheck" >&2
  exit 1
fi

export PATH="${KEA_BIN_DIR}:${LLVM_BIN_DIR}:${PATH}"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/kea-test.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

pass=0
fail=0
failed_tests=()

shopt -s nullglob
for test_file in "${TEST_DIR}"/*.mlir; do
  name="$(basename "${test_file}")"

  # Collect the RUN: lines (bash 3.2: no mapfile).
  run_lines=()
  while IFS= read -r line; do
    run_lines+=("${line}")
  done < <(sed -n 's|^[[:space:]]*//[[:space:]]*RUN:[[:space:]]*||p' "${test_file}")

  if [[ ${#run_lines[@]} -eq 0 ]]; then
    echo "SKIP: ${name} (no RUN: line)"
    continue
  fi

  ok=1
  output=""
  for run_line in "${run_lines[@]}"; do
    cmd="${run_line//%s/${test_file}}"
    # %S is lit's "directory containing the test", which end-to-end tests need
    # to reach the shared fixtures in tests/mlir and the native build.
    cmd="${cmd//%S/${TEST_DIR}}"
    cmd="${cmd//%t/${TMP_ROOT}/${name}.tmp}"
    if ! output="$(bash -o pipefail -c "${cmd}" 2>&1)"; then
      ok=0
      echo "FAIL: ${name}"
      echo "  RUN: ${cmd}"
      # Indent the failure output so it is obvious what belongs to what.
      printf '%s\n' "${output}" | sed 's/^/    /'
      break
    fi
  done

  if [[ ${ok} -eq 1 ]]; then
    echo "PASS: ${name}"
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    failed_tests+=("${name}")
  fi
done

echo
echo "-------------------------------------------------"
echo "kea tests: ${pass} passed, ${fail} failed"
if [[ ${fail} -ne 0 ]]; then
  printf '  failed: %s\n' "${failed_tests[@]}"
  exit 1
fi
exit 0
