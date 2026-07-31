//===- KeaDialect.td - KEA dialect definition --------------*- tablegen -*-===//
//
// The KEA dialect models an NPU with an explicitly managed scratchpad:
// activations (A), weights (W) and an accumulator (ACC) bank, fed from DRAM by
// DMA. Everything else in the compiler is built on top of the definitions
// below, so keep this file minimal and dependency-free.
//
//===----------------------------------------------------------------------===//

#ifndef KEA_DIALECT
#define KEA_DIALECT

include "mlir/IR/OpBase.td"

def Kea_Dialect : Dialect {
  let name = "kea";
  let cppNamespace = "::mlir::kea";

  let summary = "Ops and types for the KEA NPU backend";
  let description = [{
    `kea` is the NPU-facing dialect of the KEA compiler. It spans two levels:

    * a *tensor* level (`kea.conv2d`, ...) that a `tosa` -> `linalg` -> `kea`
      lowering pipeline targets, and
    * a *buffer* level (`!kea.buffer`, `kea.dma_load`, `kea.matmul`,
      `kea.signal` / `kea.wait`) that maps 1:1 onto the hardware ISA.
  }];

  // Required for `assemblyFormat` on TypeDef/AttrDef to be wired into the
  // dialect's parseType/printType and parseAttribute/printAttribute hooks.
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Defined in lib/Dialect/KeaTypes.cpp.
    void registerTypes();
    /// Defined in lib/Dialect/KeaAttrs.cpp.
    void registerAttributes();
  }];
}

/// Base class for every op in this dialect. Always derive from this, never
/// from `Op<Kea_Dialect, ...>` directly, so that dialect-wide traits can be
/// added in one place later.
class Kea_Op<string mnemonic, list<Trait> traits = []>
    : Op<Kea_Dialect, mnemonic, traits>;

/// Base class for KEA types.
class Kea_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Kea_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

/// Base class for KEA attributes.
class Kea_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Kea_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // KEA_DIALECT
