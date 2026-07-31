#!/usr/bin/env bash
#
# check_tosa_examples.sh -- verify every MLIR example in tests/mlir/ against the
# pinned toolchain.
#
# For each .mlir file this runs two checks:
#   1. ROUND-TRIP: `mlir-opt file.mlir` must parse, verify and print. The result
#      is then re-parsed to confirm the printed form is itself valid.
#   2. LOWERING (TOSA files only, informational): the tosa->linalg pipeline is
#      run and reported, but a lowering failure is NOT fatal unless --strict.
#
# Exits non-zero if any round-trip fails. Idempotent, builds nothing, writes
# nothing outside a temporary directory.
#
# Usage:
#   scripts/check_tosa_examples.sh            # round-trip is fatal, lowering informational
#   scripts/check_tosa_examples.sh --strict   # lowering failures are fatal too
#   scripts/check_tosa_examples.sh --quiet    # only print failures and the summary
#   MLIR_OPT=/path/to/mlir-opt scripts/check_tosa_examples.sh

set -uo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TEST_DIR="${REPO_ROOT}/tests/mlir"

# Pinned toolchain. Override with the MLIR_OPT environment variable.
MLIR_OPT="${MLIR_OPT:-/usr/local/opt/llvm/bin/mlir-opt}"
readonly EXPECTED_VERSION="20.1.6"

# The full TOSA -> linalg pipeline. These passes are FUNCTION-scoped, so they
# must be nested inside func.func in an explicit --pass-pipeline; passing them
# as top-level flags fails with "unable to schedule pass ... on a PassManager
# intended to run on 'builtin.module'".
readonly TOSA_TO_LINALG_PIPELINE='builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith{include-apply-rescale=true}))'

STRICT=0
QUIET=0
for arg in "$@"; do
  case "$arg" in
    --strict) STRICT=1 ;;
    --quiet)  QUIET=1 ;;
    -h|--help)
      sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ -t 1 ]]; then
  RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BOLD=''; RESET=''
fi

log()  { [[ $QUIET -eq 1 ]] || printf '%s\n' "$*"; }
fail() { printf '%s\n' "$*" >&2; }

if ! command -v "$MLIR_OPT" >/dev/null 2>&1 && [[ ! -x "$MLIR_OPT" ]]; then
  fail "${RED}error${RESET}: mlir-opt not found at '${MLIR_OPT}'."
  fail "Install LLVM ${EXPECTED_VERSION} or set MLIR_OPT=/path/to/mlir-opt."
  exit 2
fi

if [[ ! -d "$TEST_DIR" ]]; then
  fail "${RED}error${RESET}: test directory not found: ${TEST_DIR}"
  exit 2
fi

version_line="$("$MLIR_OPT" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
log "${BOLD}mlir-opt${RESET}: ${MLIR_OPT}"
log "${BOLD}version ${RESET}: ${version_line:-unknown}"
if [[ "$version_line" != "$EXPECTED_VERSION" ]]; then
  log "${YELLOW}warning${RESET}: expected LLVM ${EXPECTED_VERSION}, found ${version_line:-unknown}."
  log "          TOSA syntax differs substantially between releases; these"
  log "          examples are pinned to ${EXPECTED_VERSION} (see docs/TOSA_NOTES.md)."
fi
log ""

TMPDIR_RUN="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_RUN"' EXIT

total=0; rt_failed=0; lower_failed=0; lower_ok=0
declare -a RT_FAILURES=() LOWER_FAILURES=()

# Collect files deterministically. Note: `mapfile`/`readarray` is bash 4+, and
# macOS ships bash 3.2, so use a portable read loop instead.
FILES=()
while IFS= read -r _f; do
  [[ -n "$_f" ]] && FILES+=("$_f")
done < <(find "$TEST_DIR" -name '*.mlir' -type f | LC_ALL=C sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
  fail "${RED}error${RESET}: no .mlir files found under ${TEST_DIR}"
  exit 2
fi

for file in "${FILES[@]}"; do
  rel="${file#"${REPO_ROOT}/"}"
  total=$((total + 1))

  # --- 1. Round-trip -------------------------------------------------------
  printed="${TMPDIR_RUN}/printed.mlir"
  err="${TMPDIR_RUN}/err.txt"
  if ! "$MLIR_OPT" "$file" >"$printed" 2>"$err"; then
    rt_failed=$((rt_failed + 1))
    RT_FAILURES+=("$rel")
    fail "${RED}FAIL${RESET}  ${rel}  (parse/verify)"
    sed 's/^/        /' "$err" | head -12 >&2
    continue
  fi

  # Re-parse the printed output: catches printer/parser asymmetry.
  if ! "$MLIR_OPT" "$printed" >/dev/null 2>"$err"; then
    rt_failed=$((rt_failed + 1))
    RT_FAILURES+=("$rel")
    fail "${RED}FAIL${RESET}  ${rel}  (reparse of printed output)"
    sed 's/^/        /' "$err" | head -12 >&2
    continue
  fi

  # --- 2. Lowering (TOSA files only, informational) ------------------------
  note=""
  if [[ "$file" == *"/tosa/"* ]]; then
    if "$MLIR_OPT" --pass-pipeline="$TOSA_TO_LINALG_PIPELINE" "$file" \
         >"${TMPDIR_RUN}/lowered.mlir" 2>"$err"; then
      lower_ok=$((lower_ok + 1))
      residual="$(grep -oE 'tosa\.[a-z_0-9]+' "${TMPDIR_RUN}/lowered.mlir" \
                  | LC_ALL=C sort -u | tr '\n' ' ')"
      if [[ -n "$residual" ]]; then
        note="  ${YELLOW}[lowers; residual tosa: ${residual%% }]${RESET}"
      else
        note="  [lowers cleanly]"
      fi
    else
      lower_failed=$((lower_failed + 1))
      LOWER_FAILURES+=("$rel")
      note="  ${YELLOW}[LOWERING FAILED]${RESET}"
      if [[ $STRICT -eq 1 ]]; then
        fail "${RED}FAIL${RESET}  ${rel}  (tosa->linalg lowering, --strict)"
        sed 's/^/        /' "$err" | head -12 >&2
      fi
    fi
  fi

  log "${GREEN}ok${RESET}    ${rel}${note}"
done

log ""
log "${BOLD}--------------------------------------------------------${RESET}"
log "checked ${total} file(s)"
log "round-trip: $((total - rt_failed)) passed, ${rt_failed} failed"
log "tosa->linalg lowering: ${lower_ok} lowered, ${lower_failed} failed"

exit_code=0

if [[ $rt_failed -gt 0 ]]; then
  fail ""
  fail "${RED}${BOLD}round-trip failures:${RESET}"
  for f in "${RT_FAILURES[@]}"; do fail "  - $f"; done
  exit_code=1
fi

if [[ $lower_failed -gt 0 ]]; then
  if [[ $STRICT -eq 1 ]]; then
    fail ""
    fail "${RED}${BOLD}lowering failures (--strict):${RESET}"
    for f in "${LOWER_FAILURES[@]}"; do fail "  - $f"; done
    exit_code=1
  else
    log ""
    log "${YELLOW}note${RESET}: some files did not lower to linalg. Re-run with"
    log "      --strict to treat that as an error."
  fi
fi

if [[ $exit_code -eq 0 ]]; then
  log ""
  log "${GREEN}${BOLD}all checks passed${RESET}"
fi

exit $exit_code
