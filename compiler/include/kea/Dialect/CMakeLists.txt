# TableGen for the kea dialect.
#
# add_mlir_dialect(<Foo> <ns>) expects a file named <Foo>.td and emits
# <Foo>.h.inc, <Foo>.cpp.inc, <Foo>Types.{h,cpp}.inc and
# <Foo>Dialect.{h,cpp}.inc into CMAKE_CURRENT_BINARY_DIR. Hence the slightly
# odd generated names "KeaOpsTypes.h.inc" / "KeaOpsDialect.h.inc" -- that is
# upstream's naming, not a typo.
add_mlir_dialect(KeaOps kea)

# Level 2 (buffer level) ops get their OWN op-decls/op-defs target so that the
# two halves of the dialect never edit the same TableGen file. There is no
# add_mlir_dialect() call here: the dialect, types and enum/attr code are
# already generated from KeaOps.td / KeaAttrs.td and must not be duplicated.
set(LLVM_TARGET_DEFINITIONS KeaMachineOps.td)
mlir_tablegen(KeaMachineOps.h.inc -gen-op-decls)
mlir_tablegen(KeaMachineOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRKeaMachineOpsIncGen)
add_dependencies(mlir-headers MLIRKeaMachineOpsIncGen)

# add_mlir_dialect does NOT cover enums or AttrDefs, so drive those by hand.
set(LLVM_TARGET_DEFINITIONS KeaAttrs.td)
mlir_tablegen(KeaEnums.h.inc -gen-enum-decls)
mlir_tablegen(KeaEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(KeaAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=kea)
mlir_tablegen(KeaAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=kea)
add_public_tablegen_target(MLIRKeaAttrsIncGen)
add_dependencies(mlir-headers MLIRKeaAttrsIncGen)
