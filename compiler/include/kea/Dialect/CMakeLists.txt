# TableGen for the kea dialect.
#
# add_mlir_dialect(<Foo> <ns>) expects a file named <Foo>.td and emits
# <Foo>.h.inc, <Foo>.cpp.inc, <Foo>Types.{h,cpp}.inc and
# <Foo>Dialect.{h,cpp}.inc into CMAKE_CURRENT_BINARY_DIR. Hence the slightly
# odd generated names "KeaOpsTypes.h.inc" / "KeaOpsDialect.h.inc" -- that is
# upstream's naming, not a typo.
add_mlir_dialect(KeaOps kea)

# add_mlir_dialect does NOT cover enums or AttrDefs, so drive those by hand.
set(LLVM_TARGET_DEFINITIONS KeaAttrs.td)
mlir_tablegen(KeaEnums.h.inc -gen-enum-decls)
mlir_tablegen(KeaEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(KeaAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=kea)
mlir_tablegen(KeaAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=kea)
add_public_tablegen_target(MLIRKeaAttrsIncGen)
add_dependencies(mlir-headers MLIRKeaAttrsIncGen)
