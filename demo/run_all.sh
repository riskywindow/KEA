#!/usr/bin/env bash
#===----------------------------------------------------------------------===#
# The whole demo: MobileNetV2 int8 .kgraph.json -> TOSA -> .keaf -> kea-sim,
# validated against the frontend's golden integer reference and analysed on a
# roofline.
#
# Requires both halves of the build:
#   bash scripts/build_compiler.sh      # kea-opt, kea-translate  (~6 min)
#   bash scripts/build.sh               # kea-as, kea-sim, keac   (~1 min)
#
# Everything it writes lands in demo/build/ (intermediates, ~230 MB with the A/B
# sweep) and demo/results/ (the numbers, committed).
#
# Runtime: about 90 seconds, dominated by the 52-layer scheduled/unscheduled
# A/B sweep. Pass --quick to skip it.
#===----------------------------------------------------------------------===#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

for t in build/native/bin/keac build/native/bin/kea-sim \
         build/compiler/bin/kea-opt build/compiler/bin/kea-translate; do
  [[ -x "${ROOT}/${t}" ]] || { echo "error: ${t} not built" >&2; exit 1; }
done
[[ -x "${PY}" ]] || { echo "error: ${PY} not found (see docs/PLATFORM.md)" >&2; exit 1; }

echo "############ 1. emit TOSA and compile"
"${PY}" "${ROOT}/demo/compile_mobilenetv2.py"

echo
echo "############ 2. numerical validation against the golden vectors"
"${PY}" "${ROOT}/demo/validate_mobilenetv2.py"

echo
echo "############ 3. roofline"
"${PY}" "${ROOT}/demo/roofline.py"

if [[ ${QUICK} -eq 0 ]]; then
  echo
  echo "############ 4. --kea-schedule A/B, every layer"
  "${PY}" "${ROOT}/demo/ab_schedule.py"
fi

echo
echo "############ 5. backend defects, still reproducing?"
bash "${ROOT}/demo/repro/run_repro.sh" || true

# A canonical excerpt for README.md: the scheduled stream where a DMA of the
# next tile sits between two MATMULs of the current one.
SRC="${ROOT}/demo/build/ab/slice_6_7.scheduled.kasm"
if [[ -f "${SRC}" ]]; then
  mkdir -p "${ROOT}/demo/results"
  cp "${SRC}" "${ROOT}/demo/results/schedule_excerpt.kasm"
  cp "${ROOT}/demo/build/ab/slice_6_7.unscheduled.kasm" \
     "${ROOT}/demo/results/schedule_excerpt.unscheduled.kasm"
fi

echo
echo "results in demo/results/, plot in demo/roofline_mobilenetv2.png"
