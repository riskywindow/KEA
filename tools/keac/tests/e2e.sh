#!/usr/bin/env bash
#===----------------------------------------------------------------------===#
# The whole loop, on the MobileNetV2 inverted residual.
#
#   TOSA MLIR --keac--> .keaf --kea-sim--> cycles, utilisation, output tensor
#                                       --> compared against a numpy reference
#
# Requires BOTH halves of the build, which is the point:
#
#   bash scripts/build_compiler.sh     # kea-opt, kea-translate
#   bash scripts/build.sh              # kea-as, kea-dis, kea-sim, keac
#   bash tools/keac/tests/e2e.sh
#
# `compiler/test/kea-emit-e2e.mlir` is the version that runs inside the
# compiler's own test suite and skips the native half when it is not built.
#===----------------------------------------------------------------------===#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="${ROOT}/build/native/bin"
PY="${PYTHON:-${ROOT}/.venv/bin/python}"
SRC="${ROOT}/tests/mlir/tosa/mobilenet_block.mlir"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/keac-e2e.XXXXXX")"
trap 'rm -rf "${OUT}"' EXIT

for t in keac kea-as kea-sim kea-dis; do
  [[ -x "${BIN}/${t}" ]] || { echo "error: ${BIN}/${t} not built; run scripts/build.sh" >&2; exit 1; }
done
[[ -x "${ROOT}/build/compiler/bin/kea-translate" ]] || {
  echo "error: build/compiler/bin/kea-translate not built; run scripts/build_compiler.sh" >&2
  exit 1
}

for FN in mobilenet_v2_inverted_residual mobilenet_v2_inverted_residual_stride2; do
  echo "=== ${FN}"
  "${BIN}/keac" "${SRC}" --function "${FN}" -o "${OUT}/${FN}.keaf" \
                --keep-intermediates -v

  # `kea-as` already ran inside keac and validated every field, alignment, the
  # ACC-word discipline and Rule D. Prove the round trip on top of that:
  # assemble(disassemble(p)) == p, byte for byte.
  "${BIN}/kea-dis" "${OUT}/${FN}.keaf" --map "${OUT}/${FN}.map.json" \
      > "${OUT}/${FN}.rt.kasm"
  "${BIN}/kea-as" "${OUT}/${FN}.rt.kasm" --map "${OUT}/${FN}.map.json" \
      --const "${OUT}/${FN}.weights.bin" -o "${OUT}/${FN}.rt.keaf"
  cmp "${OUT}/${FN}.keaf" "${OUT}/${FN}.rt.keaf"
  echo "round trip: assemble(disassemble(p)) == p"

  "${BIN}/kea-sim" "${OUT}/${FN}.keaf" --stats-json "${OUT}/${FN}.stats.json" \
      | sed -n '1,12p'
done

#===----------------------------------------------------------------------===#
# The numerical check: does the program compute the right numbers?
#===----------------------------------------------------------------------===#
FN=mobilenet_v2_inverted_residual
if [[ -x "${PY}" ]]; then
  "${PY}" "$(dirname "${BASH_SOURCE[0]}")/block_check.py" write \
      "${OUT}/in.bin" "${OUT}/expected.bin"
  "${BIN}/kea-sim" "${OUT}/${FN}.keaf" --quiet \
      --input "${FN}.input0=${OUT}/in.bin" \
      --output "${FN}.2.out=${OUT}/got.bin"
  "${PY}" "$(dirname "${BASH_SOURCE[0]}")/block_check.py" compare \
      "${OUT}/expected.bin" "${OUT}/got.bin"
  echo
  echo "=== per-layout numerical check (distinct weights)"
  "${PY}" "$(dirname "${BASH_SOURCE[0]}")/numeric_check.py"
else
  echo "note: ${PY} not found; skipping the numerical check"
fi
