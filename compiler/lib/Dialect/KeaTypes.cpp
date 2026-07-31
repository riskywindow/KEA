//===- KeaTypes.cpp - KEA type implementations ------------------*- C++ -*-===//

#include "kea/Dialect/KeaTypes.h"
#include "kea/Dialect/KeaDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::kea;

//===----------------------------------------------------------------------===//
// custom<Shape> directive
//===----------------------------------------------------------------------===//
//
// TableGen turns `custom<Shape>($shape, $elementType)` in an assemblyFormat into
// unqualified calls to parseShape()/printShape() from inside `namespace
// mlir::kea`, so these must be declared *before* KeaOpsTypes.cpp.inc is
// included. The parameter types are the ODS "storage" types of the parameters:
//   ArrayRefParameter<"int64_t">  -> SmallVector<int64_t>& on parse,
//                                    ArrayRef<int64_t> on print
//   "::mlir::Type"                -> Type& on parse, Type on print
// Parse functions return ParseResult; the generated code checks them with
// mlir::failed().

namespace mlir {
namespace kea {

static ParseResult parseShape(AsmParser &parser, SmallVectorImpl<int64_t> &shape,
                              Type &elementType) {
  // withTrailingX consumes the final 'x' in "16x16x", leaving the element type.
  if (parser.parseDimensionList(shape, /*allowDynamic=*/false,
                                /*withTrailingX=*/true))
    return failure();
  return parser.parseType(elementType);
}

static void printShape(AsmPrinter &printer, ArrayRef<int64_t> shape,
                       Type elementType) {
  raw_ostream &os = printer.getStream();
  for (int64_t dim : shape)
    os << dim << 'x';
  printer.printType(elementType);
}

} // namespace kea
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "kea/Dialect/KeaOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

LogicalResult BufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<int64_t> shape, Type elementType,
                                 AddressSpace addressSpace) {
  if (shape.size() > 4)
    return emitError() << "kea.buffer supports at most rank 4, got "
                       << shape.size();

  for (int64_t dim : shape)
    if (dim <= 0)
      return emitError() << "kea.buffer dimensions must be positive, got "
                         << dim;

  if (!elementType.isIntOrFloat())
    return emitError() << "kea.buffer element type must be an integer or "
                          "float type, got "
                       << elementType;

  // The accumulator bank is 32-bit wide only.
  if (addressSpace == AddressSpace::ACC &&
      elementType.getIntOrFloatBitWidth() != 32)
    return emitError() << "ACC buffers must have a 32-bit element type, got "
                       << elementType;

  return success();
}

int64_t BufferType::getNumElements() const {
  int64_t n = 1;
  for (int64_t dim : getShape())
    n *= dim;
  return n;
}

std::optional<int64_t> BufferType::getSizeInBytes() const {
  unsigned bits = getElementType().getIntOrFloatBitWidth();
  if (bits % 8 != 0)
    return std::nullopt;
  return getNumElements() * static_cast<int64_t>(bits / 8);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void KeaDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "kea/Dialect/KeaOpsTypes.cpp.inc"
      >();
}
