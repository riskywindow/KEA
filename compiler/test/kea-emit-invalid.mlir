// RUN: (kea-translate %s --function=no_addr --emit-kasm=- 2>&1 || true) | FileCheck %s --check-prefix=NOADDR
// RUN: (kea-translate %s --function=bad_layout --emit-const-listing=- 2>&1 || true) | FileCheck %s --check-prefix=LAYOUT
// RUN: (kea-translate %s --function=wrong_size --emit-const-listing=- 2>&1 || true) | FileCheck %s --check-prefix=SIZE
// RUN: (kea-translate %s --function=not_constant --emit-const-listing=- 2>&1 || true) | FileCheck %s --check-prefix=NOTCONST
// RUN: (kea-translate %s --function=no_layout --emit-map=- 2>&1 || true) | FileCheck %s --check-prefix=NOLAYOUT
// RUN: (kea-translate %s --emit-kasm=- 2>&1 || true) | FileCheck %s --check-prefix=AMBIGUOUS
// RUN: (kea-translate %s --function=nope --emit-kasm=- 2>&1 || true) | FileCheck %s --check-prefix=NOFUNC
//
// What the backend refuses. Every one of these is a condition that would
// otherwise produce a plausible-looking program with wrong numbers, so each
// diagnostic names the file and section that defines the rule.
//
// (`kea-translate` is not an `mlir-opt` tool, so there is no
// `-verify-diagnostics`; the `|| true` keeps the pipeline's exit status from
// masking the FileCheck, which is what checks the message.)

// NOADDR: has no `addr`
// NOADDR-SAME: -kea-alloc
func.func @no_addr() attributes {kea.dram_layout = {total_bytes = 4096 : i64}} {
  %a = kea.alloc {name = "a", role = "scratch"} : !kea.buffer<1024xi8, A>
  kea.vcopy to %a {src_addr = 0, dst_addr = 0, row_bytes = 16, rows = 1,
                   src_row_stride = 0, dst_row_stride = 16, fill}
    : !kea.buffer<1024xi8, A>
  kea.halt
  return
}

//===--------------------------------------------------------------------===//

// LAYOUT: unknown layout "mxu_tiles_8x8"
// LAYOUT-SAME: DIALECT_L2.md
func.func @bad_layout() attributes {kea.dram_layout = {total_bytes = 4096 : i64,
    const_bytes = 256 : i64}} {
  %w = arith.constant dense<1> : tensor<2x1x1x3xi8>
  %0 = kea.alloc from %w : tensor<2x1x1x3xi8>
       {name = "w", role = "weights", layout = "mxu_tiles_8x8", addr = 0 : i64}
     : !kea.buffer<256xi8, DRAM>
  kea.halt
  return
}

//===--------------------------------------------------------------------===//

// A declared extent that disagrees with the layout is the stale-weight-blob
// bug, and it is fatal rather than a silent truncation.
// SIZE: materializes 256 bytes but the buffer declares 128
func.func @wrong_size() attributes {kea.dram_layout = {total_bytes = 4096 : i64,
    const_bytes = 128 : i64}} {
  %w = arith.constant dense<1> : tensor<2x1x1x3xi8>
  %0 = kea.alloc from %w : tensor<2x1x1x3xi8>
       {name = "w", role = "weights", layout = "mxu_tiles_16x16",
        addr = 0 : i64} : !kea.buffer<128xi8, DRAM>
  kea.halt
  return
}

//===--------------------------------------------------------------------===//

// NOTCONST: is not a compile-time constant
func.func @not_constant(%w: tensor<2x1x1x3xi8>)
    attributes {kea.dram_layout = {total_bytes = 4096 : i64,
                                   const_bytes = 256 : i64}} {
  %0 = kea.alloc from %w : tensor<2x1x1x3xi8>
       {name = "w", role = "weights", layout = "mxu_tiles_16x16",
        addr = 0 : i64} : !kea.buffer<256xi8, DRAM>
  kea.halt
  return
}

//===--------------------------------------------------------------------===//

// NOLAYOUT: has no `kea.dram_layout`
// NOLAYOUT-SAME: MEMORY_PLANNING.md
func.func @no_layout() {
  kea.halt
  return
}

//===--------------------------------------------------------------------===//

// A KEAF artifact holds exactly one program, so an ambiguous module is an
// error that lists the candidates rather than a silent pick.
// AMBIGUOUS: Level 2 functions and a KEAF artifact holds exactly one program
// AMBIGUOUS: candidates: no_addr, bad_layout
// NOFUNC: no Level 2 function named 'nope'
// NOFUNC: candidates: no_addr, bad_layout
