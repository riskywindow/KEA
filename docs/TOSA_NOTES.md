# TOSA / linalg cheat-sheet for the NPU compiler

**Everything in this document was verified by running `mlir-opt` on this machine.**
Nothing here is from memory or from upstream docs alone.

| | |
|---|---|
| Toolchain | `/usr/local/opt/llvm/bin/mlir-opt` |
| Version | **LLVM/MLIR 20.1.6** (Homebrew, optimized build) |
| Verified on | 2026-08-01 |
| Examples | `tests/mlir/tosa/*.mlir`, `tests/mlir/linalg/*.mlir` |
| Gate | `scripts/check_tosa_examples.sh` (exit 0 = all round-trip) |

---

## 0. READ THIS FIRST: the premise correction

The project brief assumed that this build had already taken the big TOSA
breaking changes ("zero points moved from attributes to operands,
`tosa.fully_connected` was removed, `tosa.rescale` signature changed, shapes
moved to `!tosa.shape`"). **On 20.1.6 that is mostly NOT true.** Those changes
landed in the **LLVM 21** development cycle. Concretely, on this build:

| Change you may have read about | Status in **20.1.6** |
|---|---|
| Zero points are **operands** | ❌ No. They are **attributes** (`quantization_info = #tosa.conv_quant<...>`) |
| `tosa.fully_connected` removed | ❌ No. **It still exists** and verifies |
| `tosa.rescale` takes zp operands | ❌ No. `input_zp` / `output_zp` are `I32Attr` **attributes** |
| `tosa.reshape` takes `!tosa.shape` | ❌ No. Shape is the `new_shape` **`DenseI64ArrayAttr`** |
| `tosa.transpose` takes a `perms` attr | ❌ No. Permutation is an **operand** (rank-1 `i32` tensor) |
| `tosa.pad` takes `!tosa.shape` | ✅ **Yes** — this one HAS migrated |
| `tosa.tile` takes `!tosa.shape` | ✅ **Yes** — this one HAS migrated |
| `tosa.clamp` has `min_val`/`max_val` | ❌ No. It has **four** attrs: `min_int`/`max_int`/`min_fp`/`max_fp` |
| `tosa.mul` shift is an attribute | ❌ No. Shift is a **third operand** (rank-1 `i8` tensor) |

So 20.1.6 is a **transitional** release: `pad` and `tile` use `!tosa.shape`;
everything else still uses attributes. Do not copy syntax from the current
mlir.llvm.org TOSA page — it documents LLVM 21+ and will not parse here.

---

## 1. Where zero points live in 20.1.6

All conv/pool/matmul zero points are **attributes**, not operands. There are
four distinct quantization attributes, and using the wrong one is an error:

| Attribute | Parameters | Used by |
|---|---|---|
| `#tosa.conv_quant<input_zp = A, weight_zp = B>` | 2 × `int64_t` | `conv2d`, `conv3d`, `depthwise_conv2d`, `transpose_conv2d`, `fully_connected` |
| `#tosa.matmul_quant<a_zp = A, b_zp = B>` | 2 × `int64_t` | `matmul` |
| `#tosa.unary_quant<input_zp = A, output_zp = B>` | 2 × `int64_t` | `avg_pool2d` |
| `#tosa.pad_quant<input_zp = A>` | 1 × `int64_t` | `pad` |

Notes:

- The attribute is spelled `quantization_info = #tosa....` on the op.
- It is **optional** on every op that accepts it. Absent means all zero points
  are 0.
- The parameters are `int64_t` in the attribute even though the tensors are i8.
  Write them as plain integers: `input_zp = -128`.
- `tosa.rescale` is the exception: its zero points are **not** in one of these
  attributes, they are two separate `I32Attr`s named `input_zp` / `output_zp`
  (see §4), and they are **mandatory**.
- `tosa.add` / `tosa.sub` / `tosa.mul` have **no** quantization attribute at
  all. There is no "binary op quantization attr" in TOSA by design — you
  express it with surrounding `tosa.rescale` ops (see §6).

---

## 2. `tosa.conv2d` — exact syntax

```mlir
%out = tosa.conv2d %input, %weight, %bias {
  pad      = array<i64: 1, 1, 1, 1>,   // [top, bottom, left, right]
  stride   = array<i64: 1, 1>,         // [h, w]
  dilation = array<i64: 1, 1>,         // [h, w]
  acc_type = i32,                      // MANDATORY TypeAttr
  quantization_info = #tosa.conv_quant<input_zp = -3, weight_zp = 0>
} : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi32>)
    -> tensor<1x8x8x16xi32>
```

Generic form (identical semantics, useful as a textual-exporter target):

```mlir
%out = "tosa.conv2d"(%input, %weight, %bias) {
  pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>,
  dilation = array<i64: 1, 1>, acc_type = i32,
  quantization_info = #tosa.conv_quant<input_zp = -3, weight_zp = 0>
} : (tensor<1x8x8x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi32>)
    -> tensor<1x8x8x16xi32>
```

**Layouts (get these wrong and the verifier will not always catch it):**

| Operand | Layout | Meaning |
|---|---|---|
| `input` | **NHWC**, rank 4 | `[batch, H, W, in_channels]` |
| `weight` | **OHWI**, rank 4 | `[out_channels, KH, KW, in_channels]` |
| `bias` | rank 1 | `[out_channels]`, element type **i32** |
| `output` | **NHWC**, rank 4 | `[batch, H', W', out_channels]` |

- `acc_type` must be one of `i32`, `i48`, `f16`, `f32`. For i8 conv use `i32`.
- `local_bound` is an optional `BoolAttr`, default `false`. Irrelevant for int.
- Attribute **order in the printed form is alphabetical** (`acc_type`,
  `dilation`, `pad`, `quantization_info`, `stride`) — do not rely on source
  order when diffing.
- Output spatial size:
  `H' = floor((H + pad_top + pad_bottom - dilation_h*(KH-1) - 1) / stride_h) + 1`

See `tests/mlir/tosa/conv2d_i8.mlir`.

---

## 3. `tosa.depthwise_conv2d` — exact syntax

```mlir
%out = tosa.depthwise_conv2d %input, %weight, %bias {
  pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>,
  dilation = array<i64: 1, 1>, acc_type = i32,
  quantization_info = #tosa.conv_quant<input_zp = -7, weight_zp = 0>
} : (tensor<1x8x8x16xi8>, tensor<3x3x16x1xi8>, tensor<16xi32>)
    -> tensor<1x8x8x16xi32>
```

**The weight layout is different from `conv2d`. This is the single most common
mistake.**

| Op | Weight layout |
|---|---|
| `tosa.conv2d` | **OHWI** `[OC, KH, KW, IC]` |
| `tosa.depthwise_conv2d` | **HWCM** `[KH, KW, C, M]` |

- `M` is the channel multiplier. Output channels = `C * M`.
- `bias` length must be `C * M`, not `C`.
- MobileNet always uses `M = 1`, so the weight is `[KH, KW, C, 1]`.

See `tests/mlir/tosa/depthwise_i8.mlir`.

---

## 4. `tosa.rescale` — exact syntax and **bit-exact** semantics

```mlir
%out = tosa.rescale %input {
  input_zp     = 0 : i32,                      // I32Attr, REQUIRED
  output_zp    = -5 : i32,                     // I32Attr, REQUIRED
  multiplier   = array<i32: 1073741824, ...>,  // DenseI32ArrayAttr, REQUIRED
  shift        = array<i8: 30, ...>,           // DenseI8ArrayAttr  <-- i8!
  scale32      = true,                         // BoolAttr, REQUIRED
  double_round = true,                         // BoolAttr, REQUIRED
  per_channel  = true,                         // BoolAttr, REQUIRED
  input_unsigned  = false,                     // optional, default false
  output_unsigned = false                      // optional, default false
} : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
```

Gotchas:

- `shift` is **`array<i8:...>`**, not `array<i32:...>`. Using i32 is a parse error.
- `scale32`, `double_round`, `per_channel` are **all mandatory**. Omitting any
  one is an error, even though they look like they should have defaults.
- `per_channel = true` ⇒ `multiplier`/`shift` length == size of the **last**
  dimension of the tensor. `per_channel = false` ⇒ length exactly **1**.
- The op is *not* `InferShapedType`-driven for element type: you state the
  output element type in the functional type, and that is what selects the
  clamp range.

### 4.1 Exact arithmetic

This was extracted from **this build's own lowering**
(`--tosa-to-arith{include-apply-rescale=true}`), so it is authoritative for
bit-exact simulation. For `scale32 = true`:

```
  value = i32(input) - input_zp                  // i32 arithmetic
  prod  = sext_i64(value) * sext_i64(multiplier) // i64
  round = (i64(1) << shift) >> 1                 // i.e. 2^(shift-1)
  acc   = prod + round                           // i64

  if (double_round && shift > 31)
      acc += (value >= 0) ? (i64(1) << 30) : -(i64(1) << 30)

  res = i32(acc >> shift)                        // ARITHMETIC shift right
  res = res + output_zp
  out = clamp(res, MIN_T, MAX_T)                 // MIN/MAX of the output type
```

Key points for the simulator:

- **`double_round` only has any effect when `shift > 31`.** For `shift <= 31`
  the `double_round` flag is a no-op — the lowering does not even emit the
  branch. This is a very common source of mismatch against naive
  reimplementations, which apply the extra rounding unconditionally.
- The double-round correction is **sign-dependent**, keyed off the sign of
  `value` (the zero-point-corrected input), **not** the sign of `prod`.
- The shift is an **arithmetic** (sign-propagating) right shift on the i64
  accumulator, applied *after* adding the rounding terms.
- The intermediate is i64 for `scale32 = true`. Do not truncate to i32 before
  the shift.
- Clamp bounds come from the **output element type**: i8 ⇒ `[-128, 127]`,
  i16 ⇒ `[-32768, 32767]`. Saturating, not wrapping.
- For `scale32 = false`, the multiplier must fit in i16 and the accumulate is
  done in i32 rather than i64. Same rounding structure.

### 4.2 `tosa.apply_scale`

`tosa.rescale` lowers to a `linalg.generic` whose body contains
**`tosa.apply_scale`**, which is *not* removed by `--tosa-to-linalg`. It is a
separate util op:

```mlir
%r = tosa.apply_scale %value, %multiplier, %shift {double_round = true}
     : (i32, i32, i8) -> i32
```

To eliminate it you must additionally run
`tosa-to-arith{include-apply-rescale=true}`. See §9.

See `tests/mlir/tosa/rescale.mlir`.

---

## 5. `tosa.clamp` (ReLU / ReLU6)

```mlir
%out = tosa.clamp %input {
  min_int = -128 : i64,   // I64Attr
  max_int = 127  : i64,   // I64Attr
  min_fp  = 0.0  : f32,   // FloatAttr -- REQUIRED EVEN FOR INTEGER TENSORS
  max_fp  = 6.0  : f32,   // FloatAttr -- REQUIRED EVEN FOR INTEGER TENSORS
  nan_mode = "PROPAGATE"  // optional, default "PROPAGATE"
} : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
```

- **You must supply `min_fp`/`max_fp` even on an i8 tensor.** They are ignored
  by the integer lowering but omitting them is a parse error. This is the #1
  surprise in this op.
- **No zero-point subtraction is performed.** To clamp at the quantized zero,
  pass the zero point itself as `min_int`. E.g. for a ReLU on a tensor with
  `zp = -5`, use `min_int = -5`, not `min_int = 0`.
- Lowers to `arith.maxsi` + `arith.minsi` (i.e. `max(min_int, min(max_int, x))`).
- `tosa.clamp` is `SameOperandsAndResultElementType`: it cannot change type.

See `tests/mlir/tosa/clamp_relu.mlir`.

---

## 6. Quantized elementwise add

`tosa.add` has **no quantization attribute**. The idiomatic fully-quantized
add is a rescale sandwich:

```mlir
// Bring both operands into a common i32 domain (zero points removed).
%a32 = tosa.rescale %a { input_zp = -5 : i32, output_zp = 0 : i32,
         multiplier = array<i32: 1073741824>, shift = array<i8: 10>,
         scale32 = true, double_round = true, per_channel = false }
       : (tensor<...xi8>) -> tensor<...xi32>
%b32 = tosa.rescale %b { input_zp = 3 : i32, output_zp = 0 : i32,
         multiplier = array<i32: 1932735283>, shift = array<i8: 11>,
         scale32 = true, double_round = true, per_channel = false }
       : (tensor<...xi8>) -> tensor<...xi32>

%s   = tosa.add %a32, %b32 : (tensor<...xi32>, tensor<...xi32>) -> tensor<...xi32>

// Back down to i8 at the output scale/zero point.
%out = tosa.rescale %s { input_zp = 0 : i32, output_zp = -7 : i32,
         multiplier = array<i32: 1288490189>, shift = array<i8: 31>,
         scale32 = true, double_round = true, per_channel = false }
       : (tensor<...xi32>) -> tensor<...xi8>
```

Both input rescales must target the **same** intermediate scale, with enough
shift headroom that the i32 add cannot overflow.

**Broadcasting:** TOSA requires **equal rank** for broadcast operands. A
per-channel `tensor<8xi32>` bias must be reshaped to `tensor<1x1x1x8xi32>`
first; a bare rank-1 tensor will not broadcast against a rank-4 tensor.

**`tosa.mul` gotcha:** it takes **three** operands on 20.1.6 — the shift is an
operand (rank-1 `i8` tensor), not a `shift = N : i8` attribute:

```mlir
%s = "tosa.const"() {value = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
%0 = tosa.mul %a, %b, %s : (tensor<4xi32>, tensor<4xi32>, tensor<1xi8>) -> tensor<4xi32>
```

See `tests/mlir/tosa/add_i8.mlir`.

---

## 7. Pooling

```mlir
// Average pool IS quantized: acc_type + unary_quant (input_zp AND output_zp).
%0 = tosa.avg_pool2d %a {
  kernel = array<i64: 2, 2>,   // [h, w]
  stride = array<i64: 2, 2>,   // [h, w]
  pad    = array<i64: 0, 0, 0, 0>,  // [top, bottom, left, right]
  acc_type = i32,
  quantization_info = #tosa.unary_quant<input_zp = -5, output_zp = -5>
} : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>

// Max pool is NOT quantized: no acc_type, no quantization_info.
%1 = tosa.max_pool2d %a {
  kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>,
  pad = array<i64: 0, 0, 0, 0>,
  nan_mode = "PROPAGATE"      // optional
} : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>
```

- `avg_pool2d` uses **`#tosa.unary_quant`** (input_zp + **output**_zp), not
  `conv_quant`. Its output element type equals its input element type
  (i8 → i8); `acc_type` only describes the internal accumulator.
- For an **integer** `avg_pool2d` the accumulator must be exactly **i32**.
  Even though the ODS constraint `Tosa_AccType` permits `i32/i48/f16/f32`, the
  verifier rejects `acc_type = i48` on an integer tensor with
  `'tosa.avg_pool2d' op accumulator type for integer tensor is not i32`.
- `max_pool2d` has **neither** `acc_type` nor `quantization_info` in its ODS
  definition. Max is monotonic, so no zero-point correction is needed.
  **DANGER:** writing them anyway does *not* error. MLIR treats unrecognized
  entries as *discardable* attributes, so `tosa.max_pool2d ... {acc_type = i32,
  quantization_info = #tosa.unary_quant<...>, bogus_attr = 42}` parses, verifies
  and round-trips — the attributes are simply carried along and **silently
  ignored** by every pass. An exporter that emits quantization info on
  `max_pool2d` will look correct and be wrong. Validate against the ODS operand
  list, not against whether `mlir-opt` complains.
- Quantized avg-pool semantics, per this build's lowering: accumulate the
  window in i32 (`linalg.pooling_nhwc_sum`), subtract `count * input_zp`,
  then divide by the **actual** window size — which is recomputed per output
  position so that padded edges divide by the smaller real count — via a
  reciprocal-and-`apply_scale` sequence, then add `output_zp`, clamp and
  truncate. If your NPU divides by the nominal `KH*KW` everywhere, you will
  mismatch TOSA on padded borders.

See `tests/mlir/tosa/pool.mlir`.

---

## 8. Matmul / fully-connected

**`tosa.fully_connected` still exists in 20.1.6** (`TosaOps.td` line 226). It
is removed in LLVM 21+, so prefer `tosa.matmul` for forward compatibility.

```mlir
// input [N, IC], weight [OC, IC]  <-- note: weight is ALREADY transposed
// bias [OC], result [N, OC]
%0 = tosa.fully_connected %a, %w, %b {
  quantization_info = #tosa.conv_quant<input_zp = -2, weight_zp = 0>
} : (tensor<4x8xi8>, tensor<16x8xi8>, tensor<16xi32>) -> tensor<4x16xi32>

// a [B, M, K], b [B, K, N], result [B, M, N]. All operands STRICTLY rank 3.
// There is NO bias operand on matmul -- add it separately.
%1 = tosa.matmul %a, %b {
  quantization_info = #tosa.matmul_quant<a_zp = -2, b_zp = 1>
} : (tensor<1x4x8xi8>, tensor<1x8x16xi8>) -> tensor<1x4x16xi32>
```

`fully_connected` uses `conv_quant` (`input_zp`/`weight_zp`); `matmul` uses
`matmul_quant` (`a_zp`/`b_zp`). They are not interchangeable.

See `tests/mlir/tosa/matmul_fc_i8.mlir`.

---

## 9. Shape-manipulation ops (the messiest area)

| Op | How the shape/param is passed in **20.1.6** |
|---|---|
| `tosa.reshape` | `new_shape = array<i64: ...>` **attribute** (supports one `-1`) |
| `tosa.transpose` | permutation is an **operand**: rank-1 `i32` tensor, normally `tosa.const` |
| `tosa.slice` | `start` / `size` **attributes** (`DenseI64ArrayAttr`) |
| `tosa.concat` | `axis = N : i32` **attribute** |
| `tosa.tile` | `!tosa.shape` **operand** (migrated) |
| `tosa.pad` | `!tosa.shape` **operand** (migrated) |

```mlir
%0 = tosa.reshape %a {new_shape = array<i64: 1, -1>}
     : (tensor<1x7x7x32xi8>) -> tensor<1x1568xi8>

%perm = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
%1 = tosa.transpose %a, %perm : (tensor<1x8x8x16xi8>, tensor<4xi32>) -> tensor<1x16x8x8xi8>

// pad: the shape operand is [low0, high0, low1, high1, ...], length 2*rank.
%padding = tosa.const_shape {value = dense<[0,0,1,1,1,1,0,0]> : tensor<8xindex>}
           : () -> !tosa.shape<8>
%2 = tosa.pad %a, %padding {quantization_info = #tosa.pad_quant<input_zp = -5>}
     : (tensor<1x8x8x16xi8>, !tosa.shape<8>) -> tensor<1x10x10x16xi8>

// tile: multiples is also a !tosa.shape operand.
%mult = tosa.const_shape {value = dense<[2,1,1,1]> : tensor<4xindex>} : () -> !tosa.shape<4>
%3 = tosa.tile %a, %mult : (tensor<1x4x4x8xi8>, !tosa.shape<4>) -> tensor<2x4x4x8xi8>
```

- `tosa.const_shape` value must be `tensor<Nxindex>` — **`index`**, not i64.
- `tosa.pad`'s optional `pad_const` operand must be a **0-D** tensor
  (`tensor<i8>`), **not** `tensor<1xi8>`. Rank-1 is rejected.
- `tosa.cast` is a pure element-type conversion with **no** scale and **no**
  zero point. Never use it to requantize — use `tosa.rescale`.

See `tests/mlir/tosa/reshape_transpose.mlir`.

---

## 10. The MobileNetV2 integration test

`tests/mlir/tosa/mobilenet_block.mlir` contains a complete, fully-quantized
inverted-residual block that round-trips and lowers cleanly:

```
1x1 expand conv (4→24) → per-channel rescale → ReLU6
  → 3x3 depthwise s1 (24) → per-channel rescale → ReLU6
  → 1x1 project conv (24→4) → per-channel rescale   [linear bottleneck, no ReLU]
  → residual add (rescale both sides to common i32, add, rescale to i8)
```

It also contains the stride-2 variant (no residual, spatial size changes).
Conventions used: activation zero point `-128` after ReLU6, block I/O zero
point `-5`, symmetric weights (`weight_zp = 0`), per-channel weight rescales.

---

## 11. TOSA → linalg lowering: what works

### 11.1 The passes are FUNCTION-scoped

This is the first thing that will bite you:

```console
$ mlir-opt --tosa-to-linalg-named --tosa-to-linalg f.mlir
error: unable to schedule pass 'TosaToLinalgNamed' on a PassManager intended
       to run on 'builtin.module'!
```

You **must** nest them explicitly:

```bash
mlir-opt --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg))" f.mlir
```

### 11.2 The pipeline that actually eliminates all TOSA

```bash
mlir-opt --pass-pipeline="builtin.module(func.func(\
tosa-to-linalg-named,\
tosa-to-linalg,\
tosa-to-tensor,\
tosa-to-arith{include-apply-rescale=true}))" f.mlir
```

Verified: this leaves **zero** residual `tosa.*` ops for every example in
`tests/mlir/tosa/` except `reshape_transpose.mlir`, which retains dead
`tosa.const_shape` / `!tosa.shape` values left behind after `pad`/`tile`
lowering (harmless; DCE removes them).

Order matters, and each pass is load-bearing:

| Pass | Removes |
|---|---|
| `tosa-to-linalg-named` | conv2d, depthwise_conv2d, matmul, fully_connected, max_pool2d, transpose → linalg **named** ops |
| `tosa-to-linalg` | rescale, clamp, add/sub/mul, avg_pool2d → `linalg.generic` |
| `tosa-to-tensor` | reshape, slice, concat, pad → `tensor.*` |
| `tosa-to-arith{include-apply-rescale=true}` | `tosa.apply_scale`, `tosa.const` |

**Do NOT rely on the built-in `--tosa-to-linalg-pipeline`.** It exists, and it
runs, but it is *incomplete*: on `mobilenet_block.mlir` it leaves
`tosa.apply_scale`, `tosa.const` and `tosa.reshape` behind. Use the explicit
pipeline above.

### 11.3 What each op lowers to

| TOSA op | linalg result |
|---|---|
| `tosa.conv2d` (quantized) | **`linalg.conv_2d_nhwc_fhwc_q`** |
| `tosa.conv2d` + `--prefer-conv2d-kernel-layout-hwcf` | `linalg.conv_2d_nhwc_hwcf_q` |
| `tosa.depthwise_conv2d` (quantized) | **`linalg.depthwise_conv_2d_nhwc_hwcm_q`** (rank-5 result + `tensor.collapse_shape`) |
| `tosa.fully_connected` | `linalg.transpose` + **`linalg.quantized_matmul`** |
| `tosa.matmul` | **`linalg.quantized_batch_matmul`** |
| `tosa.avg_pool2d` | `linalg.pooling_nhwc_sum` + generic (zp correction, real-count divide) |
| `tosa.max_pool2d` | `linalg.pooling_nhwc_max` (+ `linalg.fill` with type min) |
| `tosa.rescale` | `linalg.generic` containing `tosa.apply_scale` |
| `tosa.clamp` | `linalg.generic` with `arith.maxsi`/`arith.minsi` |
| `tosa.add` | `linalg.generic` with `arith.addi` |
| `tosa.transpose` | `linalg.transpose` |
| explicit conv padding | **`tensor.pad` filled with the input zero point** (not 0) |

**Zero points become trailing scalar `i32` operands on the linalg `_q` ops:**

```mlir
linalg.conv_2d_nhwc_fhwc_q {dilations = ..., strides = ...}
  ins(%padded, %weight, %input_zp, %weight_zp
      : tensor<1x10x10x4xi8>, tensor<16x3x3x4xi8>, i32, i32)
  outs(%init : tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi32>
```

The full lowered MobileNet block is checked in at
`tests/mlir/linalg/mobilenet_block_lowered.mlir`.

---

## 12. linalg named ops: which exist in 20.1.6

All of the following were confirmed to parse and verify on this machine, on
**both tensors and memrefs**, i8 → i32:

| Op | Exists | Notes |
|---|---|---|
| `linalg.conv_2d_nhwc_hwcf` | ✅ | filter `[KH,KW,IC,OC]` |
| `linalg.conv_2d_nhwc_hwcf_q` | ✅ | + `i32` zp operands |
| `linalg.conv_2d_nhwc_fhwc` | ✅ | filter `[OC,KH,KW,IC]` |
| `linalg.conv_2d_nhwc_fhwc_q` | ✅ | **what TOSA lowering emits** |
| `linalg.depthwise_conv_2d_nhwc_hwc` | ✅ | filter rank 3 `[KH,KW,C]`, result rank 4 |
| `linalg.depthwise_conv_2d_nhwc_hwc_q` | ✅ | + zp operands |
| `linalg.depthwise_conv_2d_nhwc_hwcm` | ✅ | filter rank 4, **result rank 5** |
| `linalg.depthwise_conv_2d_nhwc_hwcm_q` | ✅ | **what TOSA lowering emits** |
| `linalg.matmul` | ✅ | |
| `linalg.quantized_matmul` | ✅ | **yes, it exists in 20.1.6** |
| `linalg.batch_matmul` | ✅ | |
| `linalg.quantized_batch_matmul` | ✅ | |
| `linalg.matmul_transpose_b` | ✅ | deprecated upstream, still parses here |

Key differences from TOSA:

- **linalg conv ops have no padding.** Padding must be materialized as an
  explicit `tensor.pad` / memref copy *before* the conv, filled with the input
  zero point.
- `outs` is the **accumulator initializer**, not just a shape hint. For a
  quantized conv it is normally pre-broadcast with the bias (via a
  `linalg.generic`), or zero-filled with `linalg.fill`.
- `strides`/`dilations` are `dense<...> : tensor<2xi64>` attributes.
- The `*_hwcm*` depthwise ops produce a **rank-5** `[N,H,W,C,M]` result that you
  must `tensor.collapse_shape` back to rank 4 yourself.
- On memrefs the op has **no result** and no `-> type` suffix.

See `tests/mlir/linalg/`.

---

## 13. Things that DON'T work / differ from expectation

Ordered roughly by how much time they will cost you.

1. **`--tosa-to-linalg-named` as a top-level flag fails.** The passes are
   function-scoped; use `--pass-pipeline="builtin.module(func.func(...))"`. (§11.1)
2. **`--tosa-to-linalg-pipeline` is incomplete.** It leaves `tosa.apply_scale`,
   `tosa.const` and `tosa.reshape`. Use the explicit 4-pass pipeline. (§11.2)
3. **`tosa.apply_scale` survives `--tosa-to-linalg`.** You need
   `tosa-to-arith{include-apply-rescale=true}` on top.
4. **`tosa.clamp` requires `min_fp`/`max_fp` even for integer tensors.** Omitting
   them is a parse error.
5. **`tosa.rescale`'s `shift` is `array<i8:...>`**, not `array<i32:...>`.
6. **`scale32`, `double_round`, `per_channel` are all mandatory** on
   `tosa.rescale` — they look optional but are not.
7. **`double_round` is a no-op when `shift <= 31`.** Naive simulators that
   always apply the extra rounding will mismatch. (§4.1)
8. **`tosa.mul` takes 3 operands** (shift is an operand, not an attribute).
   Error: `'tosa.mul' op expected 3 operands, but found 2`.
9. **Widening `tosa.mul` (i8×i8→i32) does not lower.** It is valid TOSA and
   round-trips, but `--tosa-to-linalg` builds an invalid `linalg.generic`:
   ```
   'linalg.yield' op type of yield operand 1 ('i8') doesn't match the
    element type of the enclosing linalg.generic op ('i32')
   ```
   Repro:
   ```mlir
   %s = "tosa.const"() {value = dense<7> : tensor<1xi8>} : () -> tensor<1xi8>
   %0 = tosa.mul %a, %b, %s : (tensor<4xi8>, tensor<4xi8>, tensor<1xi8>) -> tensor<4xi32>
   ```
   Same-type mul (i32×i32→i32) lowers fine. Emit a same-type mul plus an
   explicit cast/rescale instead.
10. **`linalg.matmul` with `indexing_maps = [...]` does not round-trip.** It
    parses, but the printer emits the clause *after* the result type while the
    parser requires it *before* `ins(...)`. Re-parsing mlir-opt's own output
    fails with `custom op 'indexing_maps' is unknown`. Genuine printer/parser
    asymmetry in 20.1.6 — use `linalg.matmul_transpose_b` or a
    `linalg.generic` instead.
11. **`tosa.pad`'s `pad_const` must be 0-D** (`tensor<i8>`); `tensor<1xi8>` is
    rejected with `operand #2 must be 0D tensor of number values`.
12. **`tosa.tile`'s `multiples` is a `!tosa.shape` operand**, not an attribute.
    Error: `'tosa.tile' op expected 2 operands, but found 1`.
13. **`tosa.const_shape` values must be `tensor<Nxindex>`**, not `tensor<Nxi64>`.
14. **TOSA broadcasting requires equal rank.** Reshape rank-1 biases up to the
    full rank first.
15. **Depthwise weight layout is HWCM, conv2d weight layout is OHWI.** Easy to
    conflate; the verifier will not always catch a transposed weight.
16. **`avg_pool2d` uses `unary_quant`, `conv2d` uses `conv_quant`.** Different
    attribute names for what looks like the same concept.
17. **Unknown attributes are silently accepted on every TOSA op.** MLIR treats
    them as discardable, so typos and misplaced quantization attributes
    (e.g. `quantization_info` on `tosa.max_pool2d`, which has no such operand)
    parse, verify, round-trip and are then ignored. `mlir-opt` accepting your
    output is **not** proof the attribute is meaningful. Conversely, genuinely
    required attributes *are* enforced: omitting `acc_type` on `tosa.conv2d`
    gives `'tosa.conv2d' op requires attribute 'acc_type'`.
18. **Quantized avg-pool divides by the real (padding-aware) window count**, not
    the nominal `KH*KW`.
19. **Printed attribute order is alphabetical**, which will not match your
    source order when diffing exporter output.

---

## 14. Accumulator / overflow / rounding rules for quantized conv

- The accumulator type is whatever `acc_type` says; for i8 convolution it is
  **i32**. Products are `i8 × i8 → i32`; TOSA requires the implementation to
  behave as if no intermediate overflow occurs within the accumulator width.
- The bias is **i32** and is added into the accumulator directly, at the
  accumulator's scale — it is *not* rescaled first.
- The zero-point correction is conceptually
  `sum((input - input_zp) * (weight - weight_zp))`. With symmetric weights
  (`weight_zp = 0`, which is what per-channel PTQ produces and what this
  compiler should assume) this collapses to
  `sum(input*weight) - input_zp * sum(weight)`, and the second term can be
  folded into the bias at compile time.
- **Padding is filled with the input zero point**, not with 0. This build's
  lowering makes that explicit as `tensor.pad ... tensor.yield %input_zp`.
- **Convolution itself does no rounding and no saturation.** All rounding and
  clamping happens later, in `tosa.rescale` (§4.1). The conv output is a raw
  i32 accumulator.
- Ordering of the reduction is unspecified; since the accumulation is exact in
  i32, any order gives the same result.

---

## 15. Regenerating / checking

```bash
# Verify every example round-trips (exit 0 = all good).
scripts/check_tosa_examples.sh

# Also require that every TOSA example lowers to linalg.
scripts/check_tosa_examples.sh --strict

# Point at a different toolchain.
MLIR_OPT=/path/to/mlir-opt scripts/check_tosa_examples.sh

# Regenerate the lowered MobileNet artifact.
/usr/local/opt/llvm/bin/mlir-opt \
  --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg))" \
  tests/mlir/tosa/mobilenet_block.mlir \
  > tests/mlir/linalg/mobilenet_block_lowered.mlir
```

The script warns loudly if `mlir-opt` is not 20.1.6, because TOSA syntax
differs substantially between releases.

---

## 16. Version pinning

**Everything in this document and in `tests/mlir/` is verified against LLVM
20.1.6 and only 20.1.6.**

If the toolchain is upgraded to LLVM 21 or later, expect **all** of the
following to break at once and require a coordinated update of the exporter,
the tests and this document:

- zero points move from attributes to operands on conv/matmul/rescale/pool
- `tosa.fully_connected` is deleted (migrate to `tosa.matmul`)
- `tosa.reshape` takes a `!tosa.shape` operand instead of `new_shape`
- `tosa.transpose` takes a `perms` attribute instead of an operand
- `tosa.clamp`'s four min/max attributes collapse to `min_val`/`max_val`
- `tosa.rescale`'s `multiplier`/`shift` become tensor operands
- `double_round` is replaced by a `rounding_mode` enum

Pin the toolchain in CI. Do not float it.
