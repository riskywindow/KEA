//===- kea-opt.cpp - KEA optimizer driver -----------------------*- C++ -*-===//
//
// `mlir-opt` plus the kea dialect and the kea passes. It registers *all*
// upstream dialects and passes on purpose: the production pipeline is
// tosa -> linalg -> kea, so this binary must be able to read and write tosa,
// linalg, tensor, memref, arith, func, ... as well as kea.
//
//===----------------------------------------------------------------------===//

#include "kea/Dialect/KeaDialect.h"
#include "kea/Transforms/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Upstream dialects + the extensions that attach external interface
  // implementations (bufferization, transform-dialect ops, ...).
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  // KEA.
  registry.insert<mlir::kea::KeaDialect>();

  // Upstream passes (-canonicalize, -cse, --tosa-to-linalg, ...) + KEA passes.
  mlir::registerAllPasses();
  mlir::kea::registerKeaPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "KEA NPU compiler optimizer driver\n", registry));
}
