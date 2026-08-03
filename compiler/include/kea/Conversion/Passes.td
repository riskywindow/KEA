//===- Passes.td - Conversions into KEA Level 1 ------------*- tablegen -*-===//
//
// The graph-ingest passes: `tosa` / `linalg` -> KEA Level 1.
//
// WHY THIS FILE IS `include`d RATHER THAN TABLEGEN'd ON ITS OWN
// ------------------------------------------------------------
// `-gen-pass-decls -name Kea` emits exactly one `registerKeaPasses()`, and
// `tools/kea-opt/kea-opt.cpp` calls exactly that one function. Giving this
// directory its own `-name KeaConversion` target would produce a second
// registration entry point that nobody calls, so instead
// `include/kea/Transforms/Passes.td` includes this file and the conversions
// land in the same generated `Passes.h.inc`. `include/kea/Conversion/Passes.h`
// is still the header you include from `lib/Conversion/*.cpp`.
//
// (The Transforms CMakeLists sets LLVM_TARGET_DEPENDS so that editing this
// file re-runs TableGen under the Makefile generator, which only globs .td
// files in its own directory.)
//
//===----------------------------------------------------------------------===//

#ifndef KEA_CONVERSION_PASSES
#define KEA_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TosaToKea : Pass<"tosa-to-kea", "::mlir::func::FuncOp"> {
  let summary = "Convert the quantized TOSA subset into KEA Level 1";
  let description = [{
    The primary graph-ingest path. Rewrites the ops that a quantized TOSA
    exporter emits into their normalised KEA Level 1 form:

    | TOSA | KEA Level 1 |
    | --- | --- |
    | `tosa.const` | `arith.constant` |
    | `tosa.conv2d` | `kea.conv2d` |
    | `tosa.depthwise_conv2d` | `kea.dwconv2d` (weights HWCM -> OHWI/IC=1) |
    | `tosa.matmul` | `kea.matmul` |
    | `tosa.fully_connected` | `kea.fully_connected` |
    | `tosa.add` | `kea.add` |
    | `tosa.rescale` | `kea.rescale` |
    | `tosa.clamp` | `kea.clamp` |
    | `tosa.avg_pool2d` / `tosa.max_pool2d` | `kea.pool` |
    | `tosa.reshape` | `kea.reshape` (`-1` resolved) |
    | `tosa.transpose` | `kea.transpose` (perms operand folded to an attribute) |

    This pass performs **no fusion**: a `tosa.conv2d` followed by a
    `tosa.rescale` becomes a `kea.conv2d` followed by a `kea.rescale`, and
    `-kea-fuse` merges them afterwards.

    Anything else in the `tosa` dialect is rejected by the conversion target,
    which reports `failed to legalize operation 'tosa.xxx'`.
  }];

  let dependentDialects = [
    "::mlir::kea::KeaDialect",
    "::mlir::arith::ArithDialect",
  ];
}

def LinalgToKea : Pass<"linalg-to-kea", "::mlir::func::FuncOp"> {
  let summary = "Convert the quantized linalg named ops into KEA Level 1";
  let description = [{
    The secondary graph-ingest path, for front ends that hand us
    `--tosa-to-linalg-named` output rather than TOSA. Deliberately narrower
    than `-tosa-to-kea`; it handles exactly

    | linalg named op | KEA Level 1 |
    | --- | --- |
    | `linalg.conv_2d_nhwc_fhwc_q` | `kea.conv2d` |
    | `linalg.depthwise_conv_2d_nhwc_hwcm_q` | `kea.dwconv2d` + `kea.reshape` |
    | `linalg.quantized_matmul` | `kea.reshape` x2 + `kea.matmul` + `kea.reshape` |
    | `linalg.matmul` | same, with zero points 0 |

    and fails with a diagnostic naming the op on any other contraction- or
    pooling-flavoured linalg op. `linalg.generic`, `linalg.fill`,
    `linalg.transpose` and everything outside `linalg` are left alone.

    Two structural differences from TOSA are handled explicitly:

    * linalg convolutions have **no padding**, so the produced `kea` op always
      has `pads = [0, 0, 0, 0]`; a preceding `tensor.pad` is left in place
      rather than being absorbed.
    * linalg's `outs` operand is an accumulator *initializer*, not a shape
      hint. When it is provably zero (`tensor.empty`, or a `linalg.fill` of a
      zero constant) it is dropped; otherwise an explicit `kea.add` of the
      initializer is emitted, which is exactly what linalg means.
  }];

  let dependentDialects = [
    "::mlir::kea::KeaDialect",
    "::mlir::arith::ArithDialect",
  ];
}

#endif // KEA_CONVERSION_PASSES
