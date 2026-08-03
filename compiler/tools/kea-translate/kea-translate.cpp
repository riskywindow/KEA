//===- kea-translate.cpp - the KEA backend driver ---------------*- C++ -*-===//
//
// Reads scheduled, allocated Level 2 MLIR and writes the three files
// docs/ASSEMBLY.md specifies:
//
//   <base>.kasm          the instruction stream
//   <base>.weights.bin   the CONST blob staged at dram.const_offset
//   <base>.map.json      DRAM layout, symbols, I/O tensors, SPM map, metadata
//
// WHY A SEPARATE BINARY AND NOT A PASS
// ------------------------------------
// `kea-opt` is an IR-to-IR tool: one input, one output, on stdout. This
// produces THREE outputs, one of which is binary, and consumes IR without
// producing any -- which is exactly the shape upstream calls a *translation*
// and gives its own driver (`mlir-translate`). A `-kea-emit` pass would have to
// write files as a side effect of running, which makes it untestable through
// `kea-opt`'s normal plumbing and makes `kea-opt in.mlir -kea-emit` print
// nothing useful. It also keeps `kea-opt` free of the file-writing and CLI
// surface, and lets this binary link only the dialect and the emission library
// rather than every upstream conversion.
//
//===----------------------------------------------------------------------===//

#include "kea/Dialect/KeaDialect.h"
#include "kea/hw_config.h"
#include "kea/Target/Kasm/EmitKasm.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input .mlir>"),
                                         llvm::cl::init("-"));

llvm::cl::opt<std::string>
    outputBase("o",
               llvm::cl::desc("Output basename; writes <base>.kasm, "
                              "<base>.weights.bin and <base>.map.json"),
               llvm::cl::value_desc("base"));

llvm::cl::opt<std::string> kasmFile("emit-kasm",
                                    llvm::cl::desc("Write the assembly here"),
                                    llvm::cl::value_desc("file"));
llvm::cl::opt<std::string>
    constFile("emit-const", llvm::cl::desc("Write the constant blob here"),
              llvm::cl::value_desc("file"));
llvm::cl::opt<std::string>
    mapFile("emit-map", llvm::cl::desc("Write model.map.json here"),
            llvm::cl::value_desc("file"));
llvm::cl::opt<std::string> listingFile(
    "emit-const-listing",
    llvm::cl::desc("Write a readable dump of the constant blob here, with "
                   "KeaQuantParam/KeaAddParam records decoded. Not consumed by "
                   "kea-as; this is for reading and for tests"),
    llvm::cl::value_desc("file"));

llvm::cl::opt<std::string>
    functionName("function",
                 llvm::cl::desc("Which Level 2 func.func to emit; required "
                                "when the module has more than one"),
                 llvm::cl::value_desc("name"));

llvm::cl::opt<mlir::kea::SyncMode> syncMode(
    "sync", llvm::cl::desc("Cross-unit semaphores (see docs/CODEGEN.md §5)"),
    llvm::cl::init(mlir::kea::SyncMode::Auto),
    llvm::cl::values(
        clEnumValN(mlir::kea::SyncMode::Auto, "auto",
                   "insert them iff the IR has none (default)"),
        clEnumValN(mlir::kea::SyncMode::Insert, "insert",
                   "always insert, even over existing ones"),
        clEnumValN(mlir::kea::SyncMode::None, "none",
                   "emit exactly what the IR says")));

llvm::cl::opt<bool>
    annotate("annotate",
             llvm::cl::desc("Append `; pc=N` comments. NOT canonical form, so "
                            "the output no longer round-trips byte for byte"),
             llvm::cl::init(false));

llvm::cl::opt<bool> noLabels("no-labels",
                             llvm::cl::desc("Omit the per-region labels"),
                             llvm::cl::init(false));

llvm::cl::list<std::string> ioQuant(
    "io-quant",
    llvm::cl::desc("<tensor>=<scale>[,<zero_point>] for model.map.json. The "
                   "Level 2 IR has no scales; without this the map says 1.0 "
                   "and the zero point derived from the stream"),
    llvm::cl::value_desc("spec"));

llvm::cl::opt<bool> verbose("v", llvm::cl::desc("Print a summary to stderr"),
                            llvm::cl::init(false));

LogicalResult writeFile(StringRef path, StringRef bytes) {
  std::string err;
  auto out = openOutputFile(path, &err);
  if (!out) {
    llvm::errs() << "kea-translate: " << err << "\n";
    return failure();
  }
  out->os() << bytes;
  out->keep();
  return success();
}

LogicalResult parseIoQuant(std::map<std::string, mlir::kea::IoQuant> &out) {
  for (const std::string &spec : ioQuant) {
    StringRef s(spec);
    auto [name, rest] = s.split('=');
    if (name.empty() || rest.empty()) {
      llvm::errs() << "kea-translate: --io-quant expects "
                      "<tensor>=<scale>[,<zero_point>], got '"
                   << spec << "'\n";
      return failure();
    }
    auto [scaleStr, zpStr] = rest.split(',');
    mlir::kea::IoQuant q;
    if (scaleStr.getAsDouble(q.scale)) {
      llvm::errs() << "kea-translate: '" << scaleStr << "' is not a scale\n";
      return failure();
    }
    q.hasScale = true;
    if (!zpStr.empty()) {
      if (zpStr.getAsInteger(10, q.zeroPoint)) {
        llvm::errs() << "kea-translate: '" << zpStr
                     << "' is not a zero point\n";
        return failure();
      }
      q.hasZeroPoint = true;
    }
    out[name.str()] = q;
  }
  return success();
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::SetVersionPrinter([](llvm::raw_ostream &os) {
    os << "kea-translate 0.1.0 (" << ::kea::KEA_ARCH_NAME
       << ", ISA revision " << ::kea::KEA_ISA_REVISION << ")\n";
  });
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "kea-translate -- Level 2 kea MLIR to .kasm + .weights.bin + "
      ".map.json\n\nADR-0001: the compiler emits text; kea-as makes the "
      "binary.\n");

  if (outputBase.empty() && kasmFile.empty() && mapFile.empty() &&
      constFile.empty() && listingFile.empty()) {
    llvm::errs() << "kea-translate: nothing to write; pass -o <base> or at "
                    "least one of --emit-kasm / --emit-const / --emit-map\n";
    return 1;
  }

  MLIRContext context;
  context.getOrLoadDialect<mlir::kea::KeaDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  std::string err;
  auto file = openInputFile(inputFilename, &err);
  if (!file) {
    llvm::errs() << "kea-translate: " << err << "\n";
    return 1;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module)
    return 1;

  mlir::kea::EmitOptions opts;
  opts.functionName = functionName;
  opts.sync = syncMode;
  opts.annotate = annotate;
  opts.labels = !noLabels;
  opts.sourceName = inputFilename;
  if (failed(parseIoQuant(opts.ioQuant)))
    return 1;

  mlir::kea::EmitResult result;
  if (failed(mlir::kea::emitKasm(*module, opts, result)))
    return 1;

  const std::string base = outputBase;
  std::string kasmPath = kasmFile.empty() && !base.empty() ? base + ".kasm"
                                                           : std::string(kasmFile);
  std::string constPath = constFile.empty() && !base.empty()
                              ? base + ".weights.bin"
                              : std::string(constFile);
  std::string mapPath =
      mapFile.empty() && !base.empty() ? base + ".map.json" : std::string(mapFile);

  if (!kasmPath.empty() && failed(writeFile(kasmPath, result.kasm)))
    return 1;
  if (!mapPath.empty() && failed(writeFile(mapPath, result.mapJson)))
    return 1;
  if (!listingFile.empty() && failed(writeFile(listingFile, result.constListing)))
    return 1;
  if (!constPath.empty() &&
      failed(writeFile(constPath,
                       StringRef(reinterpret_cast<const char *>(
                                     result.constBlob.data()),
                                 result.constBlob.size()))))
    return 1;

  // With no -o and no --emit-kasm, behave like a filter: assembly on stdout.
  if (kasmPath.empty() && mapPath.empty() && constPath.empty() &&
      listingFile.empty())
    llvm::outs() << result.kasm;

  if (verbose)
    llvm::errs() << "kea-translate: " << result.numInstructions
                 << " instructions (" << result.numInsertedSync
                 << " inserted SIGNAL/WAIT), " << result.constBlob.size()
                 << " bytes of constants\n";
  return 0;
}
