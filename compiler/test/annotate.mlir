// RUN: kea-opt %s -kea-annotate | FileCheck %s
// RUN: kea-opt %s -kea-annotate=marker=npu | FileCheck %s --check-prefix=OPTION

// CHECK: module attributes {kea.event_count = 2 : i64, kea.op_count = 3 : i64, kea.version = "0.1.0"}
// OPTION: module attributes {kea.event_count = 2 : i64, kea.op_count = 3 : i64, npu.version = "0.1.0"}

func.func @body(%a: !kea.buffer<1616xi8, A>, %acc: !kea.buffer<2048xi32, ACC>) {
  kea.signal 0
  kea.mm %a, %acc {a_addr = 0, a_inner_stride = 16, a_outer_stride = 160,
                   m_inner = 8, m_outer = 8, acc_addr = 0,
                   acc_inner_stride = 16, acc_outer_stride = 128, bank = 0}
    : !kea.buffer<1616xi8, A>, !kea.buffer<2048xi32, ACC>
  kea.wait 0
  return
}
