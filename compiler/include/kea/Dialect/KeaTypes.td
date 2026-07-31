//===- KeaTypes.td - KEA types -----------------------------*- tablegen -*-===//
#ifndef KEA_TYPES
#define KEA_TYPES

include "mlir/IR/AttrTypeBase.td"
include "kea/Dialect/KeaDialect.td"
include "kea/Dialect/KeaAttrs.td"

//===----------------------------------------------------------------------===//
// !kea.buffer
//===----------------------------------------------------------------------===//

def Kea_BufferType : Kea_Type<"Buffer", "buffer"> {
  let summary = "A scratchpad-resident buffer";
  let description = [{
    A statically shaped chunk of on-chip (or DRAM) memory owned by the NPU.
    Unlike `memref` there is no layout map and no dynamic dimension: the
    address space is part of the type and drives allocation.

    ```mlir
    !kea.buffer<16x16xi8, A>
    !kea.buffer<16x16xi32, ACC>
    !kea.buffer<1x224x224x3xi8, DRAM>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "buffer shape">:$shape,
    "::mlir::Type":$elementType,
    EnumParameter<Kea_AddressSpace>:$addressSpace
  );

  // `custom<Shape>` is implemented by parseShape/printShape in
  // lib/Dialect/KeaTypes.cpp. Shape and element type are handled by a single
  // directive so that `16x16xi8` prints with no stray whitespace.
  let assemblyFormat = "`<` custom<Shape>($shape, $elementType) `,` $addressSpace `>`";

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Total number of elements; 1 for a rank-0 buffer.
    int64_t getNumElements() const;
    /// Size in bytes, or std::nullopt if the element type is not byte-sized.
    std::optional<int64_t> getSizeInBytes() const;
    /// True if the buffer lives on chip (i.e. anything but DRAM).
    bool isOnChip() const {
      return getAddressSpace() != ::mlir::kea::AddressSpace::DRAM;
    }
  }];
}

// Convenience constraints used by the ops.
class Kea_BufferInSpace<string space, string desc>
    : Type<And<[Kea_BufferType.predicate,
                CPred<"::llvm::cast<::mlir::kea::BufferType>($_self)"
                      ".getAddressSpace() == ::mlir::kea::AddressSpace::" # space>]>,
           desc, "::mlir::kea::BufferType">;

def Kea_ABuffer   : Kea_BufferInSpace<"A",   "buffer in the activation scratchpad">;
def Kea_WBuffer   : Kea_BufferInSpace<"W",   "buffer in the weight scratchpad">;
def Kea_AccBuffer : Kea_BufferInSpace<"ACC", "buffer in the accumulator bank">;

#endif // KEA_TYPES
