// RUN: kea-opt %s -kea-annotate | FileCheck %s
// RUN: kea-opt %s -kea-annotate=marker=npu | FileCheck %s --check-prefix=OPTION

// CHECK: module attributes {kea.event_count = 2 : i64, kea.op_count = 3 : i64, kea.version = "0.1.0"}
// OPTION: module attributes {kea.event_count = 2 : i64, kea.op_count = 3 : i64, npu.version = "0.1.0"}

func.func @body(%a: !kea.buffer<8x16xi8, A>,
                %w: !kea.buffer<16x8xi8, W>,
                %acc: !kea.buffer<8x8xi32, ACC>) {
  kea.signal 0
  kea.matmul %a, %w, %acc : !kea.buffer<8x16xi8, A>, !kea.buffer<16x8xi8, W>, !kea.buffer<8x8xi32, ACC>
  kea.wait 0
  return
}
