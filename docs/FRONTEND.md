# KEA model-ingest frontend

Turns a float PyTorch or ONNX model into `.kgraph.json` + `.npz`: a small,
explicit, **integer-only** graph that the MLIR compiler consumes.

Requantization semantics live in **`docs/QUANTIZATION.md`** and are normative.
This document specifies the file format, the op set, and what the frontend does
and does not support.

```
frontend/
  kea_frontend/
    ir.py            the .kgraph.json / .npz format: dataclasses, IO, validator
    apply_scale.py   the requantization primitive (normative -- see QUANTIZATION.md)
    reference.py     bit-exact integer reference interpreter (the GOLDEN MODEL)
    quant.py         observers, scale derivation, Conv+BN folding
    builder.py       dual-backend graph builder (calibrate / emit)
    nets.py          MobileNetV2 extraction + the tiny ViT
    onnx_ingest.py   ONNX -> KGraph
    tosa_emit.py     KGraph -> TOSA MLIR for MLIR 20.1.6 (section 5)
    pipeline.py      calibrate-then-emit driver
    data.py          Imagenette loading / preprocessing
  export_mobilenetv2.py     end-to-end MobileNetV2 + accuracy measurement
  export_tiny_vit.py        the ViT stress-test model
  export_onnx.py            quantize an arbitrary float ONNX model
  gen_apply_scale_vectors.py    -> testdata/apply_scale_vectors.json
  gen_golden_io.py              -> testdata/golden_io_*.npz  (simulator cross-check)
  tests/
```

Run everything with `.venv/bin/python` after `source scripts/env.sh`.
Tests: `.venv/bin/python -m pytest frontend/tests` (92 tests, ~24 s).

**Added to the venv by this component:** `pytest` (9.1.1) and its dependencies
(`pluggy`, `iniconfig`, `packaging`, `pygments`). Nothing else; the frontend
runs on the stock `torch` / `torchvision` / `onnx` / `numpy` / `pillow` from
`docs/PLATFORM.md`.

---

## 1. Design in one page

- **Two files.** `.kgraph.json` holds topology and metadata; the sibling `.npz`
  holds every weight and bias array. The JSON never embeds bulk tensor data.
- **Explicit quantization.** Every tensor states dtype, shape, layout, scale(s),
  zero point, and whether quantization is per-tensor or per-channel (and along
  which axis).
- **Integer-only execution.** The only floats in the file are the `scale` fields,
  which are *provenance*. Every op that requantizes carries its
  already-derived integer `multiplier`/`shift`. A consumer executes the graph
  without a single floating-point operation.
- **Accumulate, then rescale.** `conv2d`, `depthwise_conv2d`, `fully_connected`,
  `matmul` and `layernorm` produce **int32**. A separate `rescale` node narrows
  to int8. This mirrors TOSA exactly and keeps every op's contract single-purpose.
- **Topologically ordered, SSA.** `nodes` is in execution order; every tensor is
  written exactly once. A consumer can execute by walking the list.
- **NHWC everywhere**, with TOSA's weight layouts (`OHWI`, `HWCM`), so a TOSA
  emitter needs no layout rewriting.

---

## 2. File format

### 2.1 Top level

```json
{
 "kgraph_version": "1.0",
 "name": "mobilenetv2_int8",
 "producer": "kea-frontend",
 "weights_file": "mobilenetv2_int8.npz",
 "requant": {"algorithm": "tosa_apply_scale_32", "double_round": true},
 "metadata": { ... free-form provenance ... },
 "inputs": ["input"],
 "outputs": ["fc"],
 "tensors": [ ... ],
 "nodes": [ ... ]
}
```

`kgraph_version` is checked on load; a mismatch is a hard error, never a
best-effort parse. `requant.double_round` is the graph-wide default and is also
repeated on every node that uses it — nothing is implied.

`weights_file` is resolved **relative to the JSON's directory**.

### 2.2 Tensors

```json
{
 "name": "stem.w",
 "dtype": "int8",
 "shape": [32, 3, 3, 3],
 "layout": "OHWI",
 "kind": "const",
 "const_key": "stem.w",
 "quant": {"kind": "per_channel", "axis": 0,
           "scale": [4.23e-05, ...], "zero_point": [0, ...]}
}
```

| Field | Values |
|---|---|
| `dtype` | `int8`, `int32` |
| `kind` | `input`, `output`, `activation`, `const` |
| `layout` | `NHWC`, `OHWI`, `HWCM`, `OI`, `C`, `*` (unconstrained) |
| `quant.kind` | `per_tensor` (axis `null`, 1 scale/zp) or `per_channel` (axis set, `shape[axis]` scales/zps) |
| `const_key` | key into the `.npz`; present iff `kind == "const"` |

Semantics: `real = scale * (q - zero_point)`.

Invariants the validator enforces:

- per-channel quantization is **always symmetric** (`zero_point == 0`)
- `scale > 0` everywhere
- a `const`'s npz array matches its declared shape and dtype exactly
- `zero_point` fits the tensor's dtype

### 2.3 Nodes

```json
{"op": "conv2d", "name": "conv2d#1",
 "inputs": ["input", "stem.w", "stem.b"], "outputs": ["stem.acc"],
 "attrs": {"pad": [1,1,1,1], "stride": [2,2], "dilation": [1,1],
           "input_zp": -14, "weight_zp": 0}}
```

`name` is unique and diagnostic only. Order is significant: nodes are
topologically sorted.

### 2.4 The `.npz`

A standard `numpy.savez` archive. Keys are the `const_key`s. Arrays are `int8`
(weights, LUTs) or `int32` (biases, exp tables), C-contiguous, in the layout the
owning tensor declares.

---

## 3. Op set

16 ops. All indices are 0-based; all pads are TOSA order `[top, bottom, left, right]`.

| Op | Inputs | Output | Key attrs |
|---|---|---|---|
| `conv2d` | x `NHWC` i8, w `OHWI` i8, b `[OC]` i32 | `NHWC` **i32** | `pad`, `stride`, `dilation`, `input_zp`, `weight_zp` |
| `depthwise_conv2d` | x `NHWC` i8, w `HWCM` i8, b `[C*M]` i32 | `NHWC` **i32** | as above |
| `fully_connected` | x `[..., IC]` i8, w `[OC, IC]` i8, b `[OC]` i32 | `[..., OC]` **i32** | `input_zp`, `weight_zp` |
| `matmul` | a `[B,M,K]` i8, b `[B,K,N]` i8 | `[B,M,N]` **i32** | `a_zp`, `b_zp` |
| `add` | 2+ i32 | i32 | — |
| `clamp` | x | same dtype | `min_val`, `max_val` |
| `rescale` | x | `out_dtype` | see §3.2 |
| `avg_pool2d` | x `NHWC` i8 | `NHWC` i8 | `kernel`, `stride`, `pad`, zps, `multiplier`, `shift` |
| `global_avg_pool` | x `NHWC` i8 | `[N,1,1,C]` i8 | zps, `multiplier`, `shift` |
| `reshape` | x | x's dtype | `new_shape` (fully explicit, no `-1`) |
| `transpose` | x | x's dtype | `perms` |
| `concat` | 2+ same dtype & quant | same | `axis` |
| `pad` | x | same | `padding`, `pad_const` |
| `table` | x i8, tab `[256]` i8 | i8 | — |
| `softmax` | x i8, exp `[256]` i32 | **i32** (Q15) | `axis`, `frac_bits` |
| `layernorm` | x i8, gamma i8, beta i32 | **i32** | `axis`, `input_zp`, `frac_bits`, `eps_term` |

`reshape`, `transpose`, `concat` and `pad` **must not change quantization
parameters** — they are pure data movement. The builder inherits the input's
`quant` for their outputs.

### 3.1 Convolution

```
out[n,oy,ox,oc] = sum over ky,kx,ic of
      (x[n, oy*sh + ky*dh - pad_top, ox*sw + kx*dw - pad_left, ic] - input_zp)
    * (w[oc, ky, kx, ic] - weight_zp)
  + bias[oc]
```

Out-of-bounds reads take the value `input_zp`, so padded contributions are
exactly zero. Accumulator is **int32**; the frontend guarantees by construction
that it does not overflow (MobileNetV2's worst case is 3×3×960 taps → ~2.8e8,
well inside int32). The reference raises `OverflowError` rather than wrapping.

`depthwise_conv2d` is the same with weights in `HWCM` and
`out[..., c*M + m] = sum_{ky,kx} (x[...,c] - izp) * (w[ky,kx,c,m] - wzp)`.
Only `M == 1` is emitted.

### 3.2 `rescale`

```json
{"input_zp": 0, "output_zp": -128,
 "multiplier": [1804491496, ...], "shift": [45, ...],
 "per_channel": true, "channel_axis": 3,
 "out_dtype": "int8", "double_round": true}
```

Semantics are in `docs/QUANTIZATION.md` §3. Constraints the validator enforces,
all so the node maps 1:1 onto `tosa.rescale`:

- `2 <= shift <= 62` and `0 <= multiplier < 2^31` (TOSA's `REQUIRE`s)
- `per_channel = true` ⇒ `channel_axis == rank - 1` and
  `len(multiplier) == shape[-1]`
- `per_channel = false` ⇒ exactly one multiplier/shift

### 3.3 `add`

Operates on **int32** only, and **saturates** rather than wrapping. Quantized
addition is therefore a rescale sandwich, exactly as TOSA does it:

```
a32 = rescale(a_i8 -> i32 @ common_scale)
b32 = rescale(b_i8 -> i32 @ common_scale)
s   = add(a32, b32)
out = rescale(s -> i8 @ output_scale)
```

The frontend picks `common_scale = min(scale_a, scale_b) / 256`, i.e. 8 bits of
headroom below the finer operand, which keeps the int32 add far from overflow.

### 3.4 Pooling

`avg_pool2d` **requires zero padding** and the validator rejects anything else.
TOSA divides each output position by its *actual* window count, so a padded
average pool has a position-dependent divisor that a single constant multiplier
cannot express. Rather than silently disagree with TOSA on padded borders, KEA
makes it unrepresentable. With `pad = 0` the divisor is the constant `kh*kw`,
folded into `multiplier`/`shift`.

```
acc = sum over window of (x - input_zp)                    # int32
out = clamp(int32_wrap(apply_scale_32(acc, mult, shift) + output_zp), -128, 127)
```

`global_avg_pool` reduces `NHWC` over H and W to `[N,1,1,C]`, with `1/(H*W)`
folded into the multiplier.

### 3.5 `softmax`

Fused, and defined exactly. Input int8, output **int32 in Q15** (values
`0..32768`, i.e. scale `2^-15`, zero point 0); a `rescale` follows to reach int8.

```
m = max(x, axis)                          # int32, keepdims
d = int32(x) - m                          # in [-255, 0]
e = exp_table[d + 255]                    # int32 in [0, 32768]
S = sum(e, axis)                          # int32, keepdims
out = idiv_trunc(e << 15, S)              # int32 in [0, 32768]
```

The 256-entry table is a graph constant, built at export time as
`exp_table[i] = clamp(round(exp((i - 255) * input_scale) * 2^15), 0, 2^15)`.
`exp_table[255] = 32768` always, so `S >= 32768 > 0` and the division is safe.
`e << 15 <= 2^30` fits int32. `idiv_trunc` is specified in
`docs/QUANTIZATION.md` §8.

Measured error: ~2% of full scale (max ≈ 5.5 steps on the 1/256 output grid) —
the cost of a 256-entry LUT at int8 input resolution.

### 3.6 `layernorm`

Fused, normalizing over the **last** axis (enforced). Input int8, output int32;
a `rescale` follows.

```
N    = shape[axis]
xc   = int32(x) - input_zp
A    = sum(xc, axis)                      # int64
B    = sum(xc * xc, axis)                 # int64  <-- 64-bit, see below
D    = N*B - A*A                          # int64, == N^2 * variance, >= 0
den  = max(isqrt(D + eps_term), 1)        # int64
num  = N*xc - A                           # int64
t    = clamp_i32(idiv_trunc(num << 15, den))
out  = clamp_i32(t * gamma + beta)        # int32
```

`eps_term = round(N^2 * eps / input_scale^2)` is precomputed at export time, so
`eps` never appears at inference. `gamma` is per-tensor symmetric int8 with
scale `s_g`; `beta` is int32 at scale `2^-15 * s_g`, which is also the output
accumulator's scale — the same "int32 accumulator + int32 bias" shape as a
convolution.

**The sum-of-squares accumulator is int64, not int32.** With `N = 192` and
`|xc| <= 255`, `N*B` reaches ~2.4e9, past int32. This is the one place KEA needs
a 64-bit accumulator, and it is called out here because it is easy to miss.

### 3.7 `table`

`out[i] = tab[x[i] + 128]`, a 256-entry int8 LUT. Used for GELU. The table
already encodes both the function and the input/output quantization, so no
rescale is needed.

---

## 4. The reference interpreter is the golden model

`frontend/kea_frontend/reference.py::execute` executes a `.kgraph.json` in pure
integer numpy. **This is what the C++ simulator is validated against.**

- No float appears anywhere in the inference path. `test_every_intermediate_stays_integral`
  asserts every intermediate tensor has an integer dtype no wider than int32.
- Deterministic and batch-invariant: `test_execute_is_deterministic_and_integral`,
  `test_batch_invariance`.
- Where the spec says "int32 accumulator", the interpreter computes in int64 and
  **raises** if a value escapes int32, instead of silently wrapping. A conforming
  implementation may therefore use plain int32.

One implementation note that is *not* a semantic difference: convolution and
matmul are computed with float64 BLAS when the operand magnitudes prove it
exact. float64 represents every integer below 2^53 exactly, and every partial
sum here stays far below that, so the result is bit-identical to integer
arithmetic and independent of reduction order. The code falls back to integer
numpy when the bound cannot be proven. This is a ~40x speedup and changes
nothing.

`execute_float` runs the same graph in float64 via the quantization metadata.
It exists only for testing and is **not** part of the compiler contract.

### 4.1 Golden I/O vectors — cross-validating the simulator

`frontend/testdata/golden_io_<model>.npz` + `golden_io_manifest.json`
(regenerate with `frontend/gen_golden_io.py`).

Each npz holds, for 4 fixed inputs, the **already-quantized int8 input tensor**
and the **exact integer output** of the reference interpreter:

```
input_0 .. input_3     int8 [1,224,224,3]
output_0 .. output_3   int8 [1,1000]
```

To validate a C++ implementation: load the npz, feed `input_<i>` to your
implementation of the matching `.kgraph.json`, and assert **exact integer
equality** with `output_<i>`. There is no tolerance.

Inputs are stored post-quantization deliberately — that removes image decoding,
resizing and the float→int8 boundary from the comparison, so a mismatch can only
mean a disagreement about integer graph semantics.

Reproducibility was verified by running the interpreter in separate processes
and hashing the outputs. Current values (also in the manifest):

| Model | sha256 of concatenated outputs |
|---|---|
| `mobilenetv2_int8` | `5f6473718508e1e89dae94cc65d9f115bbd636537a77bcff693efe62ed72ee1b` |
| `tiny_vit_int8` | `ca1fd26a1074352c90e5a988ad355209a4afddb73b32eb72722ee7e5c8662e08` |

`test_golden_io_vectors_reproduce` guards them. **If that test ever fails, the
reference semantics changed** — regenerate the vectors, update
`docs/QUANTIZATION.md`, and tell the simulator and hardware teams.

---

## 5. How this maps onto TOSA

The IR was designed against the verified behaviour in `docs/TOSA_NOTES.md`
(MLIR 20.1.6), so a `tosa` / `kea` dialect emitter is a mechanical walk with no
redesign. Zero points are attributes on 20.1.6, not operands.

**Implemented** in `kea_frontend/tosa_emit.py`, tested in
`tests/test_tosa_emit.py` (every emitted fixture is gated through `mlir-opt`,
and several through the whole `kea-opt` / `kea-translate` backend):

```sh
cd frontend && ../.venv/bin/python -m kea_frontend.tosa_emit \
    ../models/mobilenetv2_int8.kgraph.json -o /tmp/m.tosa.mlir --function mnv2
# --first-index / --last-index emit a contiguous node slice as one function
```

Constants are written in MLIR's hex `dense<"0x…">` byte form; MobileNetV2's
3.5 MB of weights become 7.4 MB of MLIR rather than ~25 MB of decimal. The
measured end-to-end outcome is in `docs/RESULTS.md`.

| KEA node | TOSA op | Notes |
|---|---|---|
| `conv2d` | `tosa.conv2d` | weights already `OHWI`; `quantization_info = #tosa.conv_quant<input_zp, weight_zp>`; `acc_type = i32` |
| `depthwise_conv2d` | `tosa.depthwise_conv2d` | weights already `HWCM` |
| `fully_connected` | `tosa.fully_connected` | `#tosa.conv_quant`. Rank > 2 needs a `tosa.reshape` to `[prod(leading), IC]` and back — free, and the only shape adaptation the emitter has to do |
| `matmul` | `tosa.matmul` | `#tosa.matmul_quant<a_zp, b_zp>`; operands already strictly rank 3 (the validator enforces it) |
| `rescale` | `tosa.rescale` | fields line up 1:1; `shift` must be emitted as `array<i8: ...>`, not `array<i32: ...>` |
| `add` | `tosa.add` | operands are already equal-rank i32 |
| `clamp` | `tosa.clamp` | `min_int`/`max_int` from the node; **must also emit `min_fp`/`max_fp`** or it will not parse |
| `avg_pool2d` | `tosa.avg_pool2d` | `#tosa.unary_quant`; KEA forbids padding, so KEA's constant divisor and TOSA's per-position divisor always agree |
| `global_avg_pool` | `tosa.avg_pool2d` with `kernel = [H, W]` | **only exact when the folded `multiplier`/`shift` is precisely `1/(H·W)`.** KEA's pool folds an arbitrary requantization; TOSA's can only divide by the window count. The emitter falls back to `avg_pool2d` at the input scale + a `tosa.rescale`, records it in the file it writes, and says it is not bit-exact. MobileNetV2's head needs this (its pool also changes scale by 1.3885×) |
| `reshape` | `tosa.reshape` | `new_shape` is a `DenseI64ArrayAttr` on 20.1.6 |
| `transpose` | `tosa.transpose` | permutation is an **operand** (rank-1 i32 tensor) on 20.1.6, not an attribute |
| `concat` | `tosa.concat` | inputs already share scale and zero point |
| `pad` | `tosa.pad` | `!tosa.shape` operand on 20.1.6; `#tosa.pad_quant<input_zp>` |
| `table` | `tosa.table` | 256-entry i8 LUT |
| `softmax` | **no TOSA op** | needs a `kea.softmax` custom op, or expansion into reduce_max / sub / table / reduce_sum / divide |
| `layernorm` | **no TOSA op** | needs a `kea.layernorm` custom op, or expansion |

`softmax` and `layernorm` are deliberately fused: they correspond to dedicated
hardware units, and their exact integer semantics (§3.5, §3.6) are specified
here precisely so a `kea` dialect op can carry them without ambiguity. Every
other node has a direct TOSA counterpart.

---

## 6. ONNX ingest

`frontend/export_onnx.py` / `kea_frontend/onnx_ingest.py`. A **float** ONNX
model in, the same `.kgraph.json` out. The importer lowers ONNX to the same
backend-agnostic op list the torch path uses, so both go through identical
calibration and quantization code and cannot drift.

### Supported

| ONNX op | Support |
|---|---|
| `Conv` | full, incl. grouped/depthwise; explicit `pads` only |
| `BatchNormalization` | folded into a **directly preceding** `Conv` |
| `Relu` | fused into the producing Conv/Gemm, else rejected as a standalone |
| `Clip` | **only `Clip(0, 6)`** (ReLU6), as attribute or initializer operands |
| `Add` | activation + activation only |
| `Gemm` | incl. `transB`, `alpha`, `beta`; `transA=1` rejected |
| `MatMul` | **only with a constant right-hand side** (i.e. a dense layer) |
| `GlobalAveragePool` | full |
| `AveragePool` | unpadded only |
| `Flatten`, `Squeeze`, `Reshape` | treated as flatten-to-2D of a pooled tensor |
| `Transpose` | full |
| `Identity`, `Dropout` | no-ops |
| `Constant` | folded into the initializer table |

### Not supported (rejected loudly, with the node name)

Any other op type; `auto_pad` other than `NOTSET`/`VALID`; standalone
`BatchNormalization`; `Clip` with bounds other than `(0, 6)`; `Add` with a
constant operand; `MatMul` of two activations; padded `AveragePool`; models with
more than one input. Nothing is silently approximated — every one of these
raises `UnsupportedOnnxOp` naming the offending node.

`MatMul` of two activations is the notable gap: it means attention-style ONNX
models must come through the torch path. Adding it needs activation-operand
quantization plumbing in the importer, not new IR.

### Coverage achieved

MobileNetV2 exported from torch (`opset 13`) parses and quantizes completely:
52 `Conv`, 35 `Clip`, 10 `Add`, 1 `GlobalAveragePool`, 1 `Flatten`, 1 `Gemm`,
70 `Constant`. The result is **the same graph the torch path produces** — same
183 nodes, same op sequence, same shapes, and quantized constants agreeing to
within 1 LSB (`test_onnx_and_torch_paths_produce_identical_graphs`). int8 weight
tensors are bit-identical; 12 of 106 bias tensors differ by exactly 1 on 1–2
elements, because ONNX stores BN `epsilon` as float32 while torch has the exact
Python double.

---

## 7. Models

### MobileNetV2 — `models/mobilenetv2_int8.kgraph.json`

`torchvision` `IMAGENET1K_V1`. 52 convolutions with BN folded in.

```
183 nodes, 290 tensors, 106 consts (3.5 MB npz)
conv2d x35, depthwise_conv2d x17, rescale x83, clamp x35,
add x10, global_avg_pool x1, fully_connected x1, reshape x1
```

Both observer variants are also written
(`mobilenetv2_int8_{minmax,percentile}.kgraph.json`); the canonical file is a
copy of the percentile one.

### Tiny ViT — `models/tiny_vit_int8.kgraph.json`

**Random-initialised, not trained** (seed 20240601), and the graph's metadata
says `"trained": false`. It exists purely as a compiler/ISA stress test: it
exercises rank-3 `matmul` with two activation operands, integer `softmax`,
integer `layernorm`, and `table`, none of which MobileNetV2 touches. **Any
classification accuracy from it would be chance and none is reported.**

6 layers, dim 192, 3 heads, patch 16, 196 tokens.

```
272 nodes, 388 tensors, 115 consts (2.2 MB npz)
fully_connected x37, rescale x108, reshape x51, transpose x24,
matmul x12, layernorm x13, add x13, softmax x6, table x6,
conv2d x1, global_avg_pool x1
```

Deliberate simplifications, all to stay inside the op set: no class token
(classification reads a mean over tokens via `global_avg_pool`), GELU via an
int8 LUT, and the attention `1/sqrt(head_dim)` folded into the QK^T rescale so
it costs nothing at runtime. Multi-head attention folds batch and head into one
axis (`[N,T,D] → [N,T,H,Dh] → [N,H,T,Dh] → [N*H,T,Dh]`) because `tosa.matmul` is
strictly rank 3.

What it measures instead of accuracy (16 real images, percentile observer):

| Metric | Value |
|---|---|
| int8-vs-float logit cosine similarity | mean **0.9983**, min 0.9941 |
| argmax agreement with float | 15/16 |
| int8 argmax within float top-5 | 16/16 |

---

## 8. Measured accuracy

**Read the caveats.** These are honest numbers from a small, specific
evaluation set, not ImageNet-1k results.

- **Eval set:** 1000 images sampled from `imagenette2-160/val`. Imagenette is a
  **10-class subset** of ImageNet-1k; the classifier is still 1000-way, so
  errors include genuine ImageNet class ambiguity. `church` in particular scores
  ~9% top-1 because ImageNet also contains monastery, dome, mosque, bell cote
  and palace. Top-5 is the more informative metric here and is reported too.
- **This is not ImageNet-1k top-1** (MobileNetV2's published figure is 71.9%,
  measured on a different set at full resolution). Imagenette images are
  pre-downscaled to 160px shortest side and then upscaled to 256/cropped to 224,
  which costs real accuracy.
- **Calibration:** 384 images from `imagenette2-160/train` — disjoint from the
  eval split.
- **Float baseline** is the unmodified `torchvision` model, not a
  re-implementation. **int8** is the bit-exact integer reference interpreter
  running the emitted `.kgraph.json`.

### MobileNetV2, 1000 eval images, 384 calibration images

| Model | top-1 | top-5 | argmax agreement with float |
|---|---|---|---|
| float32 (torchvision) | **67.0%** (670/1000) | **90.8%** (908/1000) | — |
| int8, `minmax` observer | **67.0%** (670/1000) | 90.4% (904/1000) | 87.8% |
| int8, `percentile` 99.99 | **67.9%** (679/1000) | 89.5% (895/1000) | **91.3%** |

Reproduce with:

```sh
source scripts/env.sh
.venv/bin/python frontend/export_mobilenetv2.py --calib-images 384 --eval-images 1000
```

Machine-readable copy: `models/mobilenetv2_int8_accuracy.json`.

**Reading these numbers honestly.** int8 top-1 is within ±1 point of float for
both observers — at n=1000 that difference is inside sampling noise (1 s.e. ≈
1.5 points), so the correct conclusion is *"int8 quantization costs no
measurable top-1 accuracy on this set"*, not *"percentile beats float"*. The
`percentile` observer's higher **argmax agreement** (91.3% vs 87.8%) is the
sharper signal: it is a paired per-image comparison, and it says percentile
tracks the float model more faithfully. That, not the top-1 column, is why
percentile is the default.

Per-image logit fidelity of the shipped artifact against the float model,
measured over 12 images: mean correlation **0.987**, worst 0.954.

---

## 9. Calibration data

`scripts/fetch_calibration_data.sh` downloads Imagenette v2 (160px, ~94 MB) into
`models/data/`. It is idempotent (a marker file short-circuits re-runs),
verifies the extracted layout, deletes the tarball afterwards, and writes
`models/data/.gitignore` so nothing is ever committed.

**If the network is unavailable it prints a clear message and exits 0**, so it
can sit in a build script. The export scripts then fall back to synthetic
gaussian calibration, say so loudly, and **skip accuracy evaluation entirely
rather than report a meaningless number**.

Imagenette's 10 WordNet ids are mapped to their real ImageNet-1k class indices in
`kea_frontend/data.py`. Preprocessing (resize shortest side to 256, center-crop
224, ImageNet mean/std, NHWC) was checked against
`MobileNet_V2_Weights.IMAGENET1K_V1.transforms()` and agrees to within 1 image
in 200.

---

## 10. Tests

```sh
.venv/bin/python -m pytest frontend/tests          # 92 tests, ~24 s
.venv/bin/python -m pytest frontend/tests -m "not slow"
```

| File | Covers |
|---|---|
| `test_apply_scale.py` | the 18,506 published vectors; values measured from MLIR 20.1.6; half-up rounding; the `shift > 31` gate; wrapping; `quantize_multiplier`; the gemmlowp relationship in both directions |
| `test_ir_roundtrip.py` | save/load fidelity, byte-stable re-serialization, no bulk data in JSON, and 11 validator rejection cases |
| `test_reference_ops.py` | every op, integer vs float, bounded in **quantization steps of the output tensor** (RMS *and* max, limits set from measurement with ~2x headroom); `idiv_trunc`; `isqrt` |
| `test_end_to_end.py` | BN folding exactness; MobileNetV2 int8 vs float; the shipped artifacts; **the golden I/O vectors**; integrality of every intermediate; tiny ViT; ONNX-vs-torch graph equivalence; ONNX rejection paths |
| `test_tosa_emit.py` | the TOSA emitter (§5): every op fixture gated through `mlir-opt` twice, four of them through the whole `kea-opt`/`kea-translate` backend; the 20.1.6 spellings asserted individually; hex constants decoded back by `mlir-opt`; node-slice closure; the inexact-pool fallback; the whole 183-node MobileNetV2 module |

Measured per-op quantization error (max / RMS, in output quantization steps):

| Op | max | RMS |
|---|---|---|
| `conv2d` | 1.18 | 0.44 |
| `conv2d` + relu6 | 2.88 | 0.65 |
| `matmul` | 1.71 | 0.65 |
| `layernorm` | 0.83 | 0.35 |
| `table` (GELU) | 1.55 | 0.61 |
| `global_avg_pool` | 0.77 | 0.46 |
| `softmax` | 5.45 | 1.57 |

`softmax` is the outlier, as expected for a 256-entry LUT at int8 input
resolution.

---

## 11. Known limitations

- **Static shapes only.** Batch is fixed at graph build time. The builder can
  emit any batch size; the shipped artifacts are batch 1.
- **`avg_pool2d` cannot be padded** (§3.4). Deliberate.
- **Depthwise `channel_multiplier > 1` is rejected.** MobileNetV2 never uses it,
  and the per-channel scale axis in `HWCM` is only unambiguous when `M == 1`.
- **ONNX `MatMul` of two activations is unsupported** (§6).
- **`concat` and `pad` are specified and implemented but unused** by the two
  shipped models.
- **int16 activations are not implemented.** The dtype table has room for it;
  `apply_scale_32` is already width-agnostic.
- **The tiny ViT is untrained**, so it validates numerics and the op set, not
  accuracy.
