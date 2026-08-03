#!/usr/bin/env bash
#===----------------------------------------------------------------------===#
# The native half of the kea-emit end-to-end test.
#
# ADR-0001 splits the stack in two: the MLIR compiler and the native
# assembler/simulator build separately and neither links the other. So this
# script is the seam. Given the basename of a `kea-translate` run
# (<base>.kasm, <base>.weights.bin, <base>.map.json) it:
#
#   1. assembles it with the REAL kea-as, which range-checks every field,
#      alignment-checks every address, enforces the ACC-word discipline and
#      enforces Rule D. If kea-as rejects the compiler's output that is a
#      backend bug, and this test is where it surfaces;
#   2. proves `assemble(disassemble(p)) == p` byte for byte, which is the
#      round-trip property ADR-0001 asks for;
#   3. runs it on kea-sim and requires exit code 0 and a completed program.
#
# If build/native/bin is absent it prints a SKIP note and succeeds: the two
# halves are independently buildable on purpose, and `scripts/build_compiler.sh`
# must not go red because the other half was not built. Run
# `bash scripts/build.sh` and re-run to get the real check;
# tools/keac/tests/e2e.sh is the version that insists on both.
#===----------------------------------------------------------------------===#
set -o pipefail

BASE="$1"
if [[ -z "${BASE}" ]]; then
  echo "usage: native-tools.sh <basename>" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${KEA_NATIVE_BIN:-${HERE}/../../build/native/bin}"

if [[ ! -x "${BIN}/kea-as" ]]; then
  echo "note: ${BIN}/kea-as not built -- skipping the assemble/simulate half"
  echo "      of this test. Run 'bash scripts/build.sh' to enable it."
  exit 0
fi

set -e

# 1. The assembler validates what the compiler emitted.
"${BIN}/kea-as" "${BASE}.kasm" --map "${BASE}.map.json" \
                --const "${BASE}.weights.bin" -o "${BASE}.keaf"

# 2. assemble(disassemble(p)) == p, over the whole artifact.
#
# The TEXT is not required to be identical: `-kea-alloc` packs two DRAM
# activation symbols onto the same address with the same size, so an address
# inside them has two equally tight covering symbols and the disassembler picks
# one. That renames a symbol; it does not change a byte.
"${BIN}/kea-dis" "${BASE}.keaf" --map "${BASE}.map.json" > "${BASE}.rt.kasm"
"${BIN}/kea-as" "${BASE}.rt.kasm" --map "${BASE}.map.json" \
                --const "${BASE}.weights.bin" -o "${BASE}.rt.keaf"
cmp "${BASE}.keaf" "${BASE}.rt.keaf"

# 3. It runs. `kea-sim` fails the exit code on a deadlock, an out-of-range
#    access, an event overflow or a non-zero HALT.
"${BIN}/kea-sim" "${BASE}.keaf" --quiet --stats-json "${BASE}.stats.json"
grep -q '"status"' "${BASE}.stats.json"

echo "native: assembled, round-tripped and simulated ${BASE}.keaf"
