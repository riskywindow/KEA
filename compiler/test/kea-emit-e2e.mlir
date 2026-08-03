// RUN: kea-opt %S/../../tests/mlir/tosa/mobilenet_block.mlir --tosa-to-kea --kea-fuse --kea-tile --kea-alloc | kea-translate --function=mobilenet_v2_inverted_residual -o %t | FileCheck %s --allow-empty
// RUN: FileCheck %s --check-prefix=KASM --input-file=%t.kasm
// RUN: FileCheck %s --check-prefix=MAP --input-file=%t.map.json
// RUN: bash %S/native-tools.sh %t
//
// The whole loop, on the file DIALECT_L2.md calls "THE integration test":
//
//   mobilenet_block.mlir --kea-opt--> Level 2 --kea-translate--> .kasm + .bin
//                        --kea-as--> .keaf --kea-sim--> a run
//
// The first three RUN lines are pure compiler and always run. The fourth calls
// out to the NATIVE half (kea-as, kea-dis, kea-sim), which lives in a separate
// build; native-tools.sh skips with a note when that build is absent, so this
// file never turns `scripts/build_compiler.sh` red for a reason that has
// nothing to do with the compiler. Building both halves is what makes it a
// real end-to-end test, and tools/keac/tests/e2e.sh is the version that
// requires them.

// CHECK-NOT: error

//===--------------------------------------------------------------------===//
// The assembly
//===--------------------------------------------------------------------===//

// KASM: .arch "KEA-1"
// KASM: .isa_revision 1

// Three fused layers, each a named region, so `kea-sim`'s per-region roofline
// is labelled with the layer it came from.
// KASM: .region 0, "mobilenet_v2_inverted_residual.0"
// KASM: .region 1, "mobilenet_v2_inverted_residual.1"
// KASM: .region 2, "mobilenet_v2_inverted_residual.2"

// Layer 0, the 4 -> 24 expand: two output-channel groups, so two LOAD_W /
// MATMUL pairs on ALTERNATING banks (ISA.md §7.1), each into its own ACC
// region, each requantized by its own slice of the KeaQuantParam block.
// KASM: layer0:
// KASM: DMA0  DMA_LD  spm_space=SPM_W, dram_addr=@mobilenet_v2_inverted_residual.0.weights, spm_addr=w:0
// KASM: MXU   LOAD_W  w_addr=w:0, w_row_stride=16, k_rows=4, n_cols=16, bank=0, dtype=int8
// KASM: MXU   MATMUL  a_addr=a:2064, a_inner_stride=4, a_outer_stride=32, m_inner=8, m_outer=8, acc_addr=acc:0, acc_inner_stride=16w, acc_outer_stride=128w, bank=0, acc_mode=overwrite, dtype=int8
// KASM: MXU   LOAD_W  w_addr=w:256, w_row_stride=16, k_rows=4, n_cols=8, bank=1, dtype=int8
// KASM: VPU   VQUANT  acc_addr=acc:1024, out_addr=a:16, qparam_addr=w:704, num_pixels=64, channels=16, acc_pix_stride=16w, out_pix_stride=32, out_zp=-128, clamp_lo=-128, clamp_hi=127, dtype=int8

// Layer 1, the depthwise: one DWCONV over 32 padded channel lanes, on the DWU.
// KASM: layer1:
// KASM: DWU   DWCONV  a_addr=a:0, w_addr=w:384, acc_addr=acc:0, out_h=8, out_w=8, channels=32, a_row_stride=320, a_pix_stride=32, kernel=3, stride=1, acc_mode=overwrite

// Layer 2, the 24 -> 4 project and the residual: two reduction tiles into ONE
// ACC region (the second accumulates), then the VADD against the block input.
// KASM: layer2:
// KASM: MXU   MATMUL  a_addr=a:0, {{.*}} acc_addr=acc:0, {{.*}} bank=0, acc_mode=overwrite
// KASM: MXU   MATMUL  a_addr=a:16, {{.*}} acc_addr=acc:0, {{.*}} bank=1, acc_mode=accumulate
// KASM: VPU   VADD    a_addr=a:1552, b_addr=a:0, out_addr=a:1552, param_addr=w:704, num_elems=1024, clamp_lo=-128, clamp_hi=127
// KASM: DMA0  DMA_ST  spm_space=SPM_A, dram_addr=@mobilenet_v2_inverted_residual.2.out
// KASM: CTRL  HALT    exit_code=0

//===--------------------------------------------------------------------===//
// The map
//===--------------------------------------------------------------------===//

// The arena is the 4,352 bytes MEMORY_PLANNING.md §5 measured, and
// `const_bytes` must equal the size of the weight blob exactly or `kea-as`
// refuses the pair -- which is what catches a stale blob.
// MAP: "total_bytes": 4352
// MAP: "const_bytes": 2292

// Both inter-layer feature maps are packed into the same 1,536 bytes of DRAM
// scratch: two symbols, one address.
// MAP: "name": "mobilenet_v2_inverted_residual.0.out", "offset": 2816
// MAP: "name": "mobilenet_v2_inverted_residual.1.out", "offset": 2816

// The block's zero point is -5 on both ends: on the input it is the VCOPY fill
// the padded tile is initialized with, on the output it is the KeaAddParam's
// `o_zp`. Neither is guessed.
// MAP: "name": "mobilenet_v2_inverted_residual.input0", "kind": "input"
// MAP: "shape": [1, 8, 8, 4], "scale": 1, "zero_point": -5
// MAP: "name": "mobilenet_v2_inverted_residual.2.out", "kind": "output"
// MAP: "shape": [1, 8, 8, 4], "scale": 1, "zero_point": -5
