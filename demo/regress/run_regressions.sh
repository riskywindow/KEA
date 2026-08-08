#!/usr/bin/env bash
#===----------------------------------------------------------------------===#
# Regression tests for the four backend defects found during MobileNetV2
# bring-up, plus the two limitations that are still open.
#
# This used to be demo/repro/run_repro.sh, which asserted that each defect was
# still *broken*. All four are fixed, so the assertions are inverted: each case
# now pins the fixed behaviour so it cannot silently regress.
#
#   PASS  the fixed behaviour still holds
#   FAIL  it regressed  -> exit 1
#   OPEN  a known limitation, still limited in exactly the documented way
#         (informational; does not fail the run)
#
#   bash demo/regress/run_regressions.sh
#===----------------------------------------------------------------------===#
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEAC="${ROOT}/build/native/bin/keac"
KEA_SIM="${ROOT}/build/native/bin/kea-sim"
PY="${ROOT}/.venv/bin/python"
HERE="${ROOT}/demo/regress"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/kea-regress.XXXXXX")"
trap 'rm -rf "${TMP}"' EXIT
fail=0

pass() { echo "PASS  $1"; }
bad()  { echo "FAIL  $1"; shift; sed 's/^/    /' <<<"$*" | head -4; fail=1; }
open_() { echo "OPEN  $1"; shift; sed 's/^/    /' <<<"$*" | head -2; }

# <label> <cmd...> -- the command must succeed
expect_ok() {
  local label="$1"; shift
  local out; out="$("$@" 2>&1)"
  if [[ $? -eq 0 ]]; then pass "${label}"; else bad "${label}" "${out}"; fi
}

# <label> <needle> <cmd...> -- must fail, with <needle> in the diagnostic
expect_error() {
  local label="$1" needle="$2"; shift 2
  local out; out="$("$@" 2>&1)"
  if [[ $? -eq 0 ]]; then
    bad "${label}" "expected a diagnostic, but the command succeeded"
  elif grep -qF -- "${needle}" <<<"${out}"; then
    pass "${label}"
    grep -m1 -oF -- "${needle}" <<<"${out}" | sed 's/^/    /'
  else
    bad "${label}" "expected \"${needle}\"; got:" "${out}"
  fi
}

echo "=== 1. kea.pool translates, assembles and runs"
expect_ok "bare 7x7 tosa.avg_pool2d compiles" \
  "${KEAC}" "${HERE}/pool_translates.mlir" --function gap_pool \
  -o "${TMP}/p.keaf"
expect_ok "  and runs on kea-sim" \
  "${KEA_SIM}" "${TMP}/p.keaf" --quiet --strict-poison --strict-hazards

echo
echo "=== 2. an activation-RHS kea.matmul is refused with a real diagnostic"
expect_error "matmul(act, act) names the op and the reason" \
  "'kea.matmul' op cannot lower a matmul whose right-hand side is an activation" \
  "${KEAC}" "${HERE}/matmul_activation_rhs.mlir" --function mm_act \
  -o "${TMP}/m.keaf"
out="$("${KEAC}" "${HERE}/matmul_activation_rhs.mlir" --function mm_act \
      -o "${TMP}/m.keaf" 2>&1)"
if grep -qF 'null operand found' <<<"${out}"; then
  bad "  and no longer says 'null operand found'" "${out}"
else
  pass "  and no longer says 'null operand found'"
fi

echo
echo "=== 3. the whole 52-convolution feature extractor schedules (Rule D)"
if [[ -x "${PY}" ]]; then
  ( cd "${ROOT}/frontend" && "${PY}" -m kea_frontend.tosa_emit \
      "${ROOT}/models/mobilenetv2_int8.kgraph.json" -o "${TMP}/feat.mlir" \
      --function feat --last-index 178 ) >/dev/null 2>&1
  expect_ok "nodes 0..178, --schedule, at the defaults" \
    "${KEAC}" "${TMP}/feat.mlir" --function feat -o "${TMP}/s.keaf" --schedule
  expect_ok "  the scheduled program runs clean" \
    "${KEA_SIM}" "${TMP}/s.keaf" --quiet --strict-poison --strict-hazards

  echo
  echo "=== 4. the feature extractor fits IMEM at the defaults"
  expect_ok "nodes 0..178, no flags at all" \
    "${KEAC}" "${TMP}/feat.mlir" --function feat -o "${TMP}/u.keaf" \
    --keep-intermediates
  n=$(grep -cE '^[[:space:]]+(MXU|DWU|VPU|DMA0|DMA1|CTRL)[[:space:]]+[A-Z_]+' \
      "${TMP}/u.kasm" 2>/dev/null || echo 0)
  if [[ "${n}" -gt 0 && "${n}" -le 32768 ]]; then
    pass "  ${n} instructions, IMEM holds 32768"
  else
    bad "  instruction count ${n} does not fit IMEM (32768)"
  fi
else
  echo "  skipped: ${PY} not found"
fi

echo
echo "=== still open (informational, not failures)"
out="$("${KEAC}" "${HERE}/pool_rescale_unsupported.mlir" --function pool_rescale \
      -o "${TMP}/pr.keaf" 2>&1)"
if grep -qF "'kea.rescale' op has no Level 2 lowering" <<<"${out}"; then
  open_ "a general rescale after a pool still has no lowering" \
        "$(grep -m1 -oE 'error: .kea.rescale. op has no Level 2 lowering' \
           <<<"${out}")"
else
  echo "CHANGED  pool+rescale now behaves differently -- update docs/RESULTS.md"
  sed 's/^/    /' <<<"${out}" | head -3
fi

exit ${fail}
