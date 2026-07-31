// RUN: kea-opt %s -kea-canonicalize-events | FileCheck %s

// A wait whose event is never signalled in the same block is dropped, and
// back-to-back signals of the same event collapse.

// CHECK-LABEL: func.func @dedup
func.func @dedup() {
  // CHECK-NEXT: kea.signal 1
  // CHECK-NEXT: kea.wait 1
  // CHECK-NEXT: return
  kea.signal 1
  kea.signal 1
  kea.wait 1
  return
}

// CHECK-LABEL: func.func @drop_orphan_wait
func.func @drop_orphan_wait() {
  // CHECK-NOT: kea.wait
  // CHECK: return
  kea.wait 7
  return
}

// CHECK-LABEL: func.func @keep_distinct
func.func @keep_distinct() {
  // CHECK-NEXT: kea.signal 1
  // CHECK-NEXT: kea.signal 2
  // CHECK-NEXT: kea.wait 1
  // CHECK-NEXT: kea.wait 2
  // CHECK-NEXT: return
  kea.signal 1
  kea.signal 2
  kea.wait 1
  kea.wait 2
  return
}
