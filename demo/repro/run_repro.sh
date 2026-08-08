#!/usr/bin/env bash
# Run every recorded backend defect and print the diagnostic each one produces.
# Exit status is 0 when every case still fails the way docs/RESULTS.md says it
# does -- i.e. this script is a regression guard for the bug reports, not a
# test of the compiler.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEAC="${ROOT}/build/native/bin/keac"
PY="${ROOT}/.venv/bin/python"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/kea-repro.XXXXXX")"
trap 'rm -rf "${TMP}"' EXIT
fail=0

expect_error() {   # <label> <needle> <cmd...>
  local label="$1" needle="$2"; shift 2
  local out; out="$("$@" 2>&1)"
  if grep -qF -- "${needle}" <<<"${out}"; then
    echo "STILL BROKEN  ${label}"
    grep -m1 -F -- "${needle}" <<<"${out}" | sed 's/^/    /'
  else
    echo "CHANGED       ${label}  (expected substring not found)"
    sed 's/^/    /' <<<"${out}" | head -6
    fail=1
  fi
}

echo "=== defect 1: kea.pool DRAM/SPM buffer name collision"
expect_error "avg_pool2d" 'duplicate buffer name' \
  "${KEAC}" "${ROOT}/demo/repro/pool_dram_name_collision.mlir" \
  --function gap_pool -o "${TMP}/p.keaf"

echo
echo "=== defect 2: kea.matmul with a non-constant right-hand side"
expect_error "matmul(act, act)" 'null operand found' \
  "${KEAC}" "${ROOT}/demo/repro/matmul_activation_rhs.mlir" \
  --function mm_act -o "${TMP}/m.keaf"

echo
echo "=== defect 3: --kea-schedule violates Rule D on long programs"
if [[ -x "${PY}" ]]; then
  ( cd "${ROOT}/frontend" && "${PY}" -m kea_frontend.tosa_emit \
      "${ROOT}/models/mobilenetv2_int8.kgraph.json" -o "${TMP}/p99.mlir" \
      --function p99 --last-index 99 ) >/dev/null 2>&1
  echo "  (nodes 0..99 of models/mobilenetv2_int8.kgraph.json, --spm-reserve 1)"
  expect_error "mobilenetv2 prefix 0..99" 'violates Rule D' \
    "${KEAC}" "${TMP}/p99.mlir" --function p99 -o "${TMP}/s.keaf" \
    --spm-reserve 1 --schedule
  echo "  the same prefix WITHOUT --schedule:"
  if "${KEAC}" "${TMP}/p99.mlir" --function p99 -o "${TMP}/u.keaf" \
       --spm-reserve 1 >/dev/null 2>&1; then
    echo "    compiles cleanly"
  else
    echo "    ALSO FAILS -- the defect is not scheduler-specific"; fail=1
  fi
else
  echo "  skipped: ${PY} not found"
fi

echo
echo "=== defect 4: IMEM overrun at the default SPM reserve factor"
if [[ -x "${PY}" ]]; then
  ( cd "${ROOT}/frontend" && "${PY}" -m kea_frontend.tosa_emit \
      "${ROOT}/models/mobilenetv2_int8.kgraph.json" -o "${TMP}/feat.mlir" \
      --function feat --last-index 178 ) >/dev/null 2>&1
  expect_error "features, --spm-reserve 2 (default)" 'exceeds IMEM' \
    "${KEAC}" "${TMP}/feat.mlir" --function feat -o "${TMP}/f2.keaf"
  echo "  the same slice at --spm-reserve 1:"
  if "${KEAC}" "${TMP}/feat.mlir" --function feat -o "${TMP}/f1.keaf" \
       --spm-reserve 1 >/dev/null 2>&1; then
    echo "    fits"
  else
    echo "    ALSO FAILS"; fail=1
  fi
fi
exit ${fail}
