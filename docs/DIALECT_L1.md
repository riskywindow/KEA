# `kea` Level 1 — the tensor level

**Status: implemented.** Everything in this document is exercised by
`compiler/test/*.mlir` and verified by `bash scripts/build_compiler.sh`.

Level 1 is the *normalised graph*. It is value-semantic, works on `tensor<…>`,
and carries quantization as structured attributes. It has no addresses, no
tiles, no buffers and no events — those are Level 2
([ADR-0002](adr/0002-two-level-kea-dialect.md)).

| | |
|---|---|
| Ops | `compiler/include/kea/Dialect/KeaOps.td` |
| Attributes | `compiler/include/kea/Dialect/KeaAttrs.td` |
| Verifiers | `compiler/lib/Dialect/KeaOps.cpp`, `KeaAttrs.cpp` |
| Conversions | `compiler/lib/Conversion/{TosaToKea,LinalgToKea,WeightLayout}.cpp` |
| Fusion | `compiler/lib/Transforms/Fuse.cpp` |
| Tests | `compiler/test/{l1-roundtrip,l1-invalid,tosa-to-kea*,linalg-to-kea*,kea-fuse,mobilenet-e2e}.mlir` |
| Toolchain | LLVM/MLIR **20.1.6** only (see [TOSA_NOTES.md](TOSA_NOTES.md) §16) |

```
tosa ──-tosa-to-kea──┐
                     ├──> Level 1 ──-kea-fuse──> Level 1 ──-kea-tile──> Level 2
linalg ─-linalg-to-kea┘      (fewer ops, fatter epilogues)
```

---

## 1. The op set

Ten ops. Every one is `Pure`, every one has a hand-written verifier that checks
*shapes*, not just types.

| op | signature |
|---|---|
| `kea.conv2d` | `(input: tensor<NxHxWxICxT>, weights: tensor<OCxKHxKWxICxT>, bias?: tensor<OCxi32>, residual?: tensor<NxH'xW'xOCxT'>) -> tensor<NxH'xW'xOCxT'>` |
| `kea.dwconv2d` | `(input: tensor<NxHxWxCxT>, weights: tensor<OCxKHxKWx1xT>, bias?: tensor<OCxi32>, residual?) -> tensor<NxH'xW'xOCxT'>`, `OC = C*M` |
| `kea.matmul` | `(a: tensor<BxMxKxT>, b: tensor<BxKxNxT>, bias?: tensor<Nxi32>, residual?) -> tensor<BxMxNxT'>` |
| `kea.fully_connected` | `(input: tensor<NxICxT>, weights: tensor<OCxICxT>, bias?: tensor<OCxi32>, residual?) -> tensor<NxOCxT'>` |
| `kea.add` | `(lhs, rhs) -> out`, equal rank, size-1 broadcasting |
| `kea.pool` | `(input: tensor<NxHxWxCxT>) -> tensor<NxH'xW'xCxT>` |
| `kea.rescale` | `(input: tensor<SxT>) -> tensor<SxT'>` |
| `kea.clamp` | `(input: tensor<SxT>) -> tensor<SxT>` |
| `kea.reshape` | `(input: tensor<SxT>) -> tensor<S'xT>` |
| `kea.transpose` | `(input: tensor<SxT>) -> tensor<perm(S)xT>` |

Attributes per op:

| op | attributes |
|---|---|
| `kea.conv2d` / `kea.dwconv2d` | `zero_points: #kea.zp` (req), `strides`, `pads`, `dilations` (req `DenseI64ArrayAttr`), `epilogue: #kea.epilogue` (opt) |
| `kea.matmul` / `kea.fully_connected` | `zero_points: #kea.zp` (req), `epilogue` (opt) |
| `kea.add` | `lhs_quant`, `rhs_quant`, `out_quant: #kea.quant` (all-or-nothing), `clamp: DenseI64ArrayAttr` (opt) |
| `kea.pool` | `kind: #kea.pool_kind` (req), `kernel`, `strides`, `pads` (req), `quant: #kea.quant` (opt) |
| `kea.rescale` | `quant: #kea.quant` (req) |
| `kea.clamp` | `min`, `max: I64Attr` (req) |
| `kea.reshape` | `new_shape: DenseI64ArrayAttr` (req) |
| `kea.transpose` | `perms: DenseI64ArrayAttr` (req) |

### 1.1 Assembly

```mlir
// A raw accumulator: no epilogue, so the result must be i32.
%acc = kea.conv2d %in, %w bias %b {
    zero_points = #kea.zp<input = -5, weight = 0>,
    strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
    dilations = array<i64: 1, 1>}
  : (tensor<1x8x8x4xi8>, tensor<24x1x1x4xi8>, tensor<24xi32>) -> tensor<1x8x8x24xi32>

// The same conv after -kea-fuse: per-channel requantize + ReLU6 clamp folded in.
%act = kea.conv2d %in, %w bias %b {
    zero_points = #kea.zp<input = -5, weight = 0>,
    strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
    dilations = array<i64: 1, 1>,
    epilogue = #kea.epilogue<
        requant = <multiplier = [1073741824, 1181116006], shift = [36, 37],
                   input_zp = 0, output_zp = -128, axis = 3, rounding = DOUBLE>,
        clamp = [-128, 127]>}
  : (tensor<1x8x8x4xi8>, tensor<2x1x1x4xi8>, tensor<2xi32>) -> tensor<1x8x8x2xi8>

// The MobileNetV2 inverted residual, all in one op.
%y = kea.conv2d %dw, %wp bias %bp residual %x {
    zero_points = #kea.zp<input = -128, weight = 0>,
    strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
    dilations = array<i64: 1, 1>,
    epilogue = #kea.epilogue<
        requant  = <multiplier = [1288490189], shift = [37], input_zp = 0,
                    output_zp = -5,  axis = -1, rounding = DOUBLE>,
        accum    = <multiplier = [1610612736], shift = [11], input_zp = -5,
                    output_zp = 0,   axis = -1, rounding = DOUBLE>,
        residual = <multiplier = [1073741824], shift = [10], input_zp = -5,
                    output_zp = 0,   axis = -1, rounding = DOUBLE>,
        output   = <multiplier = [1503238553], shift = [40], input_zp = 0,
                    output_zp = -5,  axis = -1, rounding = DOUBLE>>}
  : (tensor<1x8x8x24xi8>, tensor<4x1x1x24xi8>, tensor<4xi32>, tensor<1x8x8x4xi8>)
    -> tensor<1x8x8x4xi8>

%q  = kea.rescale %acc {quant = #kea.quant<multiplier = [1073741824], shift = [30],
                                           input_zp = 0, output_zp = -5,
                                           axis = -1, rounding = DOUBLE>}
    : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
%r  = kea.clamp %q {min = -5 : i64, max = 91 : i64} : tensor<1x4x4x8xi8>

%mm = kea.matmul %a, %b bias %bias {zero_points = #kea.zp<input = -2, weight = 1>}
    : (tensor<1x4x8xi8>, tensor<1x8x16xi8>, tensor<16xi32>) -> tensor<1x4x16xi32>
%fc = kea.fully_connected %a, %w bias %b {zero_points = #kea.zp<input = -128, weight = 0>}
    : (tensor<1x64xi8>, tensor<10x64xi8>, tensor<10xi32>) -> tensor<1x10xi32>

// KEA_VADD: three per-tensor requantizations and a fused clamp.
%s  = kea.add %a, %b {
    lhs_quant = #kea.quant<multiplier = [1073741824], shift = [10], input_zp = -5,
                           output_zp = 0, axis = -1, rounding = DOUBLE>,
    rhs_quant = #kea.quant<multiplier = [1932735283], shift = [11], input_zp = 3,
                           output_zp = 0, axis = -1, rounding = DOUBLE>,
    out_quant = #kea.quant<multiplier = [1288490189], shift = [31], input_zp = 0,
                           output_zp = -7, axis = -1, rounding = DOUBLE>,
    clamp = array<i64: -128, 127>}
  : (tensor<1x4x4x8xi8>, tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>

%p  = kea.pool %a {kind = #kea.pool_kind<AVG>, kernel = array<i64: 2, 2>,
                   strides = array<i64: 2, 2>, pads = array<i64: 0, 0, 0, 0>,
                   quant = #kea.quant<multiplier = [1073741824], shift = [30],
                                      input_zp = -5, output_zp = -5,
                                      axis = -1, rounding = SINGLE>}
    : (tensor<1x8x8x16xi8>) -> tensor<1x4x4x16xi8>

%f  = kea.reshape %a {new_shape = array<i64: 1, 1568>}
    : (tensor<1x7x7x32xi8>) -> tensor<1x1568xi8>
%t  = kea.transpose %a {perms = array<i64: 0, 3, 1, 2>}
    : (tensor<1x8x8x16xi8>) -> tensor<1x16x8x8xi8>
```

**Two assembly gotchas, both load-bearing:**

1. The optional `bias` / `residual` operands take **no leading comma**
   (`%in, %w bias %b residual %r`). A comma-led optional group in ODS commits to
   the group the moment it sees the comma, so `%in, %w, residual %r` — bias
   absent, residual present, which `-kea-fuse` produces for a bias-less conv —
   would fail to parse with `expected 'bias'`.
2. An attribute cannot follow an SSA operand directly in a declarative format:
   MLIR lexes `%b #kea.zp<…>` as "result `#…` of value `%b`". That is why
   `zero_points` lives in `attr-dict` rather than being printed inline.

---

## 2. The quantization attributes

Three attributes, each with one job. The whole reason ADR-0002 says to ingest
TOSA rather than linalg is to keep this information, so it is modelled
explicitly rather than as loose integers.

### 2.1 `#kea.zp` — operand zero points

```mlir
#kea.zp<input = -5, weight = 0>
```

| op | `input` | `weight` |
|---|---|---|
| `kea.conv2d`, `kea.dwconv2d`, `kea.fully_connected` | activation zp | weight zp |
| `kea.matmul` | `a` zp | `b` zp |

Zero points on the *input* side are per tensor **by construction**: TOSA models
them as scalars (`#tosa.conv_quant` / `#tosa.matmul_quant`), linalg models them
as trailing scalar `i32` operands, and no PTQ toolchain emits a per-channel
*activation* zero point. Per-channel information lives in `#kea.quant`.

Symmetric weights (`weight = 0`) are what per-channel PTQ produces and what the
KEA MXU assumes. A non-zero `weight` is accepted and carried, but the backend
has to materialize the `input_zp · Σw` correction term itself
([TOSA_NOTES §14](TOSA_NOTES.md)).

### 2.2 `#kea.quant` — a requantization

```mlir
#kea.quant<multiplier = [1073741824], shift = [30],
           input_zp = 0, output_zp = -128, axis = -1, rounding = DOUBLE>
```

| field | type | meaning |
|---|---|---|
| `multiplier` | `DenseI32ArrayAttr` | Q0.31 multipliers, one per channel or exactly one |
| `shift` | `DenseI8ArrayAttr` | matching exponents, same length |
| `input_zp` | `int32` | subtracted before the multiply |
| `output_zp` | `int32` | added after the shift, before the clamp |
| `axis` | `int64` | `-1` = per tensor; `>= 0` = the tensor dimension the arrays index |
| `rounding` | `SINGLE` \| `DOUBLE` | TOSA's `double_round` flag |

Semantics, per element, bit-exactly TOSA's (`docs/QUANTIZATION.md` §1,
`TOSA_NOTES.md` §4.1), where `i` indexes `axis` when `axis >= 0`:

```
v     = i32(in) - input_zp
prod  = sext_i64(v) * sext_i64(multiplier[i])
acc   = prod + ((i64(1) << shift[i]) >> 1)          // half-up, toward +inf
if (rounding == DOUBLE && shift[i] > 31)
    acc += (v >= 0) ? (i64(1) << 30) : -(i64(1) << 30)
out   = clamp(i32(acc >> shift[i]) + output_zp, MIN(out_ty), MAX(out_ty))
```

Verifier: `multiplier.size() == shift.size() >= 1`; `axis >= -1`; per tensor ⇒
exactly one pair; multipliers strictly positive (TOSA requires it, and a
negative multiplier breaks the half-up rounding — `QUANTIZATION.md` §1.3);
`0 <= shift <= 62`. TOSA's own range is `2..62`; `0` and `1` are additionally
permitted so the exact identity `multiplier = 1 << shift` can be spelled at
small shifts.

**Why `rounding` is an enum and not a `bool`.** A bare `bool` AttrDef parameter
goes through `FieldParser`'s generic integral path and round-trips as `0`/`1`,
which is unreadable in the IR. The enum prints as a bare keyword.

**Why `multiplier`/`shift` print as `[…]` and not `array<i32: …>`.** They are
`DenseI{32,8}ArrayAttr` *parameters*, and ODS prints attribute-typed parameters
with `printStrippedAttrOrType`, whose counterpart parser expects the bracketed
short form. Both spellings parse; the stripped one is what round-trips. The same
mechanism strips the `#kea.quant` prefix when a quant is nested inside a
`#kea.epilogue`.

### 2.3 `#kea.epilogue` — the fused VPU pass

The KEA VPU turns an i32 accumulator into a storable i8 activation in one pass
(`KEA_VQUANT`, [ISA.md §10.1](ISA.md)) and can fold a quantized residual add on
top (`KEA_VADD`, §10.2). `#kea.epilogue` is the structured description of that
pass. Every field is optional and `-kea-fuse` fills them in **progressively**:
the conversions emit ops with no epilogue at all, then fusion adds `requant`,
then `clamp`, then the residual triple.

Stages, applied in order to the raw accumulator (`bias` is an *operand*, not
part of the attribute):

| field | stage |
|---|---|
| — | `acc = MAC(input, weights) + bias` — i32, exact, no rounding |
| `requant` | `t = requant(acc)`, saturating into the primary domain |
| `clamp` | `t = clamp(t, clamp[0], clamp[1])` — the fused activation |
| `accum` | `u = accum(t)`, into the shared add domain (i32) |
| `residual` | `v = residual(%residual)`, into the same shared add domain |
| `output` | `out = output(u + v)`, back to the result element type |

Verifier rules:

* `accum` / `residual` / `output` are **all-or-nothing** and require the op's
  optional `residual` operand; the operand likewise requires the triple. That
  triple is exactly `KeaAddParam`'s `{a_mult,a_shift,a_zp}` / `{b_…}` /
  `{o_…}` (ISA.md §10.2).
* Without `requant` no later stage may be set, and the op's result **must be
  `i32`** — it is the raw accumulator.
* `clamp` is `[lo, hi]` with `lo <= hi`, and must fit the result element type's
  signed range.
* A per-channel `requant` must have exactly `OC` scales, indexed by a real
  dimension of the result.

**Why the residual needs three requantizations and not one.** TOSA expresses a
quantized add as `out = R_out(R_a(a) + R_b(b))`, and the KEA VADD instruction
implements literally that (`a_mult/a_shift/a_zp`, `b_…`, `o_…`). Collapsing
`R_a ∘ requant` into a single multiply-shift would drop the intermediate i8
saturation and rounding, which are observable. So the epilogue keeps them
separate and stays bit-exact. See §6.3.

---

## 3. Canonical layouts

Level 1 has **one** layout per operand kind, enforced by the verifiers. The
point is that every pass below this level has exactly one case to handle.

| operand | layout | rationale |
|---|---|---|
| activations | **NHWC** `[N, H, W, C]` | what TOSA, linalg-named and the KEA scratchpad all use; channels are contiguous, which is what `KEA_VQUANT`'s `channels`-major loop wants (ISA.md §10.1) |
| `kea.conv2d` weights | **OHWI** `[OC, KH, KW, IC]` | identical to `tosa.conv2d` and `linalg.conv_2d_nhwc_fhwc_q`, so the common path needs no relayout at all |
| `kea.dwconv2d` weights | **OHWI with IC = 1** `[OC, KH, KW, 1]`, `OC = C·M` | see §3.2 |
| bias | rank 1 `i32`, length `OC` | added at accumulator scale, never rescaled first |
| `pads` | `[top, bottom, left, right]` | TOSA's order |
| `strides`, `dilations` | `[h, w]` | TOSA's order |
| `kea.fully_connected` weights | `[OC, IC]`, already transposed | `tosa.fully_connected`'s layout, and what `nn.Linear` exports |
| `kea.matmul` | strictly rank 3 `[B, M, K] × [B, K, N]` | `tosa.matmul`'s shape; rank-2 inputs are bracketed by reshapes |

Output spatial extent is TOSA's formula, checked by the verifier on every
conv-like and pool op:

```
H' = floor((H + pad_top + pad_bottom - dilation_h·(KH - 1) - 1) / stride_h) + 1
```

### 3.1 Why NHWC and not NCHW

The MXU is fed `K = 16` contiguous input channels per cycle and writes `N = 16`
contiguous output channels (MICROARCH §3.1); `KEA_VQUANT` walks pixels on the
outside and channels on the inside; `KEA_VPOOL`'s pixel stride is implicitly
`channels` bytes. All three want the channel dimension innermost. NCHW would
force a transpose in front of every single op.

### 3.2 Depthwise weights: HWCM → canonical, and why

TOSA and linalg both spell depthwise weights **HWCM** `[KH, KW, C, M]` (this is
the single most common mistake in the TOSA API — `tosa.conv2d` is OHWI while
`tosa.depthwise_conv2d` is HWCM, and the verifier does not always catch a
transposed weight). Level 1 normalises them to `[C·M, KH, KW, 1]` so that:

* both conv ops have the same weight rank *and* the same meaning for dimension
  0 (output channels) — the tiling pass writes one address computation;
* the channel-multiplier dimension disappears into `OC`, which is what the
  hardware sees anyway (the DWU has no notion of `M`);
* `M = 1` (every MobileNet depthwise) is not a special case.

The index algebra:

```
w'[c·M + m, kh, kw, 0]  ==  w[kh, kw, c, m]
```

which is a permutation `[2, 3, 0, 1]` (HWCM → CMHW) followed by an
element-order-preserving reshape, because the row-major linear index of
`[C, M, KH, KW]` at `(c, m, kh, kw)` is `((c·M + m)·KH + kh)·KW + kw`, and the
row-major linear index of `[C·M, KH, KW, 1]` at `(c·M + m, kh, kw, 0)` is the
same expression.

`-tosa-to-kea` and `-linalg-to-kea` share one implementation
(`lib/Conversion/WeightLayout.cpp`). When the weight is a **constant** — which
is what every exporter emits — the relayout is done at compile time and leaves
**no ops at all**; the original HWCM constant is then dead and is cleaned up.
Otherwise a `kea.transpose` + `kea.reshape` pair is materialized. Both paths are
tested (`tosa-to-kea.mlir` `@depthwise_const_weights` /
`@depthwise_dynamic_weights`).

The verifier rejects a raw HWCM weight: with `[3, 3, 24, 1]` on 24 channels it
reports `output channel count 24 does not match weight count 3`.

---

## 4. `-tosa-to-kea`

A full dialect conversion (`tosa` illegal, `kea` + `arith` + everything else
legal) plus a trivially-dead-op sweep. It **normalises only**: a `tosa.conv2d`
followed by a `tosa.rescale` becomes a `kea.conv2d` followed by a `kea.rescale`,
and `-kea-fuse` merges them afterwards.

### 4.1 The mapping table

| TOSA (20.1.6) | KEA Level 1 | notes |
|---|---|---|
| `tosa.const` | `arith.constant` | keeps Level 1 free of any `tosa` residue |
| `tosa.conv2d` | `kea.conv2d` | OHWI on both sides, no relayout; `pad`→`pads`, `stride`→`strides`, `dilation`→`dilations`; `acc_type` dropped (Level 1 is always i32) |
| `tosa.depthwise_conv2d` | `kea.dwconv2d` (+ folded weight relayout) | HWCM → `[C·M, KH, KW, 1]`, §3.2 |
| `tosa.matmul` | `kea.matmul` | `#tosa.matmul_quant<a_zp, b_zp>` → `#kea.zp<input, weight>` |
| `tosa.fully_connected` | `kea.fully_connected` | `[OC, IC]` weight kept as-is |
| `tosa.add` | `kea.add`, unquantized | TOSA has no add quantization by design; the surrounding rescales carry it and `-kea-fuse` pulls them in |
| `tosa.rescale` | `kea.rescale` | `per_channel = true` → `axis = rank - 1`; `false` → `axis = -1`; `double_round` → `rounding` |
| `tosa.clamp` | `kea.clamp` | `min_int`/`max_int` kept, `min_fp`/`max_fp` dropped exactly as TOSA's own integer lowering drops them |
| `tosa.avg_pool2d` | `kea.pool<AVG>` (+ identity `#kea.quant`) | `#tosa.unary_quant<input_zp, output_zp>` becomes a zero-point rebase; omitted when both are 0 |
| `tosa.max_pool2d` | `kea.pool<MAX>` | no quantization; any `quantization_info` written on it is a discardable attribute and is ignored |
| `tosa.reshape` | `kea.reshape` | the `-1` placeholder is resolved from the (static) result type |
| `tosa.transpose` | `kea.transpose` | the rank-1 `i32` **operand** is constant-folded into the `perms` attribute; the now-dead constant is removed |

### 4.2 The 20.1.6 traps this pass steps around

Every one of these is from [TOSA_NOTES.md](TOSA_NOTES.md) §13 and each is
exercised by `compiler/test/tosa-to-kea.mlir`.

* **Zero points are attributes, not operands.** Read via
  `getQuantizationInfo()`, which returns `std::optional<…>`; absent means 0/0.
  (In practice `quantization_info` cannot be omitted on a quantized
  `tosa.conv2d` — the TOSA verifier demands it — but it *is* omittable on
  `tosa.matmul` and `tosa.avg_pool2d`.)
* **`tosa.rescale`'s `shift` is `array<i8>`**, and `getInputZp()` returns
  `uint32_t` (it is an `I32Attr`), so it is cast to `int32_t` rather than
  widened.
* **OHWI vs HWCM.** §3.2.
* **`tosa.transpose`'s perms is an operand.** Folded with `m_Constant` on the
  *original* operand, so the pattern does not depend on whether the producing
  `tosa.const` has already been converted.
* **`tosa.reshape`'s `new_shape` is an attribute and may contain `-1`.**
  Resolved; Level 1 rejects a `-1`.
* **Unknown attributes are silently accepted on every TOSA op.** The patterns
  read the real ODS operand list only. `@max_pool_bogus_quant_attr` writes both
  a `quantization_info` and a `bogus_attr` on a `tosa.max_pool2d` (which has
  neither in its ODS) and checks that neither reaches Level 1.
* **`tosa.pad`/`tosa.tile` use `!tosa.shape` while `tosa.reshape` does not.**
  Both are simply out of scope and are refused by the conversion target.

### 4.3 What it refuses, and how

Hard `emitError` (the user is told *why*):

| refused | reason |
|---|---|
| `tosa.rescale` with `scale32 = false` | the 16-bit multiplier path has no KEA form; `KEA_VQUANT`'s multiplier is a normalized Q31 value (ISA.md §10.1) |
| `tosa.rescale` with `input_unsigned` / `output_unsigned` | all KEA activations are signed int8 |
| `tosa.transpose` with a non-constant permutation | it cannot become an attribute |

Reported by the conversion target as `failed to legalize operation 'tosa.X'`:
`tosa.sub`, `tosa.mul`, `tosa.slice`, `tosa.concat`, `tosa.pad`,
`tosa.const_shape`, `tosa.tile`, `tosa.cast`, and everything else in `tosa`.
See `compiler/test/tosa-to-kea-invalid.mlir`.

---

## 5. `-linalg-to-kea`

The secondary entry point, for front ends that hand us `--tosa-to-linalg-named`
output. Deliberately narrower than the TOSA path.

| linalg named op | KEA Level 1 |
|---|---|
| `linalg.conv_2d_nhwc_fhwc_q` | `kea.conv2d` (FHWC == OHWI) |
| `linalg.depthwise_conv_2d_nhwc_hwcm_q` | `kea.dwconv2d` + `kea.reshape` back to the rank-5 result |
| `linalg.quantized_matmul` | `kea.reshape` ×2 → `kea.matmul` → `kea.reshape` |
| `linalg.matmul` | the same, with both zero points 0 |

Zero points arrive as trailing scalar `i32` operands inside `ins(…)` and must be
constants to become a `#kea.zp`.

### 5.1 No padding

linalg convolutions have no padding — it is a separate `tensor.pad` filled with
the input zero point. The produced `kea` op therefore always has
`pads = [0, 0, 0, 0]`, and the `tensor.pad` is left in place rather than being
absorbed: recovering it would need a proof that the pad value equals the input
zero point, which is exactly the information the TOSA path preserves natively.
This is a deliberate asymmetry, not an oversight — the TOSA path is primary.

### 5.2 `outs` is an initializer, not a shape hint

`linalg` conv/matmul compute `outs + A·B`. When `outs` is provably zero
(`tensor.empty`, or a `linalg.fill` of a zero constant) it is dropped; otherwise
an explicit `kea.add` of the initializer is emitted, which is exactly what
linalg means. In particular the bias-broadcast `linalg.generic` that
`tosa.fully_connected` lowers to is preserved verbatim and added — we do not
try to reverse-engineer it back into a `bias` operand.

`tensor.empty` is formally uninitialized; treating it as zero is the universal
convention among producers of quantized linalg and is documented here so the
assumption is explicit.

### 5.3 Failing cleanly

The pass is a partial (greedy) conversion, so plumbing — `linalg.generic`,
`linalg.fill`, `linalg.transpose`, `tensor.*` — is left alone silently. After
the rewrite it walks for any surviving linalg op whose name contains `conv`,
`matmul` or `pooling` and reports it by name:

```
error: -linalg-to-kea does not handle 'linalg.conv_2d_nhwc_hwcf'; the supported
named ops are linalg.conv_2d_nhwc_fhwc_q, linalg.depthwise_conv_2d_nhwc_hwcm_q,
linalg.quantized_matmul and linalg.matmul, all in tensor form with constant zero
points
```

That covers the HWCF filter layout, the rank-3-filter depthwise form,
`linalg.batch_matmul`, the memref (destination-passing) forms, and non-constant
zero points. `linalg.matmul` carrying a user-defined `indexing_maps` is also
declined — it generalizes the op, and it does not round-trip in 20.1.6
(TOSA_NOTES §13.10).

---

## 6. `-kea-fuse`

Level 1 → Level 1. **Every rewrite in this pass is bit-exact.** It reassociates
work between ops but never changes the arithmetic, because the entire value of
ingesting TOSA rather than linalg is that the quantization survives exactly.

### 6.1 The fusions

| # | pattern | result |
|---|---|---|
| 1 | contraction + `kea.rescale` | `epilogue.requant`; the op's result type becomes the rescale's |
| 2 | contraction + `kea.clamp` | `epilogue.clamp` |
| 3 | contraction + unquantized `kea.add` of a broadcast constant | the `bias` operand |
| 4a | `rescale` / `rescale` / `add` / `rescale` sandwich | one quantized `kea.add` |
| 4b | contraction + quantized `kea.add` | `epilogue.{accum,residual,output}` + the `residual` operand |
| 5 | `rescale ∘ rescale`, no-op `rescale` | composed / removed, only when exact (§6.3) |
| 6 | layout-no-op `kea.reshape` / `kea.transpose` | folded away |

All of them require the producer to have **exactly one use** — otherwise fusing
would change the other consumer's operand.

Fusion 1 additionally requires the producer to have no epilogue yet (a second
requantization after a first is a real op, not a fusable one). Fusion 2 refuses
when the epilogue already has a `clamp`, and when the residual triple is already
populated — the `clamp` field is stage B, immediately after `requant`, so
folding a later clamp into it would reorder the arithmetic. Fusion 4b refuses
when the `kea.add` carries its own `clamp`, for the same reason, and requires
the residual value to already dominate the contraction so that no reordering is
needed. When *both* operands of the add are contractions the later one absorbs
the earlier one as its residual, which is the correct and desirable outcome.

### 6.2 Statistics

The pass declares MLIR `Statistic`s: `num-requant-fused`, `num-clamp-fused`,
`num-bias-fused`, `num-quant-add-formed`, `num-residual-fused`,
`num-rescale-composed`, `num-rescale-removed`, `num-rescale-refused`,
`num-shape-ops-folded`.

**`-mlir-pass-statistics` prints nothing on this toolchain.** The Homebrew LLVM
20.1.6 bottle is a Release build with `NDEBUG` and without
`LLVM_FORCE_ENABLE_STATS`, so `llvm::Statistic` is `NoopStatistic`
(`llvm/ADT/Statistic.h` line 38). This is not specific to `kea`: upstream
`mlir-opt --symbol-dce -mlir-pass-statistics` prints an empty report on this
machine too, and it cannot be fixed from out of tree because the *printer* lives
in the prebuilt `libMLIRPass`. So the pass also offers

```bash
kea-opt f.mlir -tosa-to-kea -kea-fuse=report-stats=true
```

which attaches the same counters to each function as a `kea.fusion_stats`
dictionary. The `Statistic`s light up unchanged on a stats-enabled LLVM.

On `tests/mlir/tosa/mobilenet_block.mlir`:

| counter | `…_inverted_residual` | `…_inverted_residual_stride2` |
|---|---|---|
| `requant` | 3 | 3 |
| `clamp` | 2 | 2 |
| `quant_add` | 1 | 0 |
| `residual` | 1 | 0 |
| `bias` | 0 | 0 |
| `rescale_composed` | 0 | 0 |
| `rescale_removed` | 0 | 0 |
| `rescale_refused` | 0 | 0 |
| `shape_folded` | 0 | 0 |

`bias` is 0 because `tosa.conv2d` carries bias structurally, so it is already an
operand after conversion — the bias fusion exists for the `matmul` + broadcast
add idiom (`tosa.matmul` has no bias operand), which the MobileNet block does
not use. Twelve TOSA compute ops become **three** `kea` ops per function.

### 6.3 Rescale algebra: exactly when composition is legal

This is the part that has to be right.

Write a rescale as `R(x) = clamp_T(⌊(x − izp)·m + 2^(s−1)⌋ >> s) + ozp)`, with
the `DOUBLE` correction when `s > 31`. For a pair `b(a(x))`:

**Both exact cases require the intermediate tensor to be `i32`**, so that `a`
cannot clamp — an i32 result clamped to the i32 range is the identity.

* **(a) `a` only rebases the zero point** (`multiplier[i] == 1 << shift[i]` for
  every `i`). Then `a(x) = x − a.izp + a.ozp` exactly: for `m = 2^s`,
  `(v·2^s + 2^(s−1)) >> s == v` for every sign of `v`, and the `DOUBLE`
  correction of ±2^30 cannot cross a multiple of `2^s` because `s > 31` implies
  `2^(s−1) ≥ 2^31 > 2^30`. The pair collapses into `b` with
  `input_zp' = a.izp − a.ozp + b.izp`.

* **(b) `b` only rebases the zero point.** `a` produces an unclamped i32, so the
  pair collapses into `a` with `output_zp' = a.ozp − b.izp + b.ozp` and `b`'s
  result type — whose clamp is then the only clamp, exactly as in the two-op
  form.

* **No-op removal.** A rescale with identity scale, `input_zp == output_zp` and
  the same in/out element type is the identity function and is deleted.

**Everything else is refused**, and the refusals are counted in
`num-rescale-refused` and tested in `kea-fuse.mlir`:

* **`i32 → i8 → i32`** — the ubiquitous pattern at the input of a quantized add.
  The first rescale *saturates* to `[−128, 127]` and *rounds* to an integer
  there. Both effects are observable and neither can be recovered from a single
  multiply-shift. Collapsing it would silently change the network's numerics on
  every saturating activation. This is why the residual epilogue keeps four
  separate `#kea.quant`s (§2.3) instead of pre-multiplying them.
* **Two real multipliers, even with an i32 intermediate.** Each applies its own
  half-up rounding, and `round(round(x·m₁)·m₂) ≠ round(x·m₁·m₂)` in general.
* Any per-channel pair on different axes, and any pair where the producer has
  more than one use.

### 6.4 Layout no-ops

* An identity permutation is deleted.
* A permutation that moves **only unit dimensions** — i.e. the dimensions of
  extent ≠ 1 keep their relative order — changes no element's row-major
  position, so it becomes a `kea.reshape` (pure metadata) rather than a data
  movement. `[1,8,8,1]` with perms `[0,3,1,2]` folds; `[1,8,8,16]` with the same
  perms does not.
* A reshape whose result type equals its operand type is deleted, and
  `reshape(reshape(x))` collapses to one reshape.

---

## 7. What Level 1 deliberately does not model

| not modelled | why |
|---|---|
| float element types | KEA is an integer-only NPU; the verifiers require signless integers |
| `scale32 = false` (16-bit multiplier) | `KEA_VQUANT`'s multiplier is Q31 (ISA.md §10.1) |
| unsigned quantization | all KEA activations are signed int8 |
| `tosa.slice` / `concat` / `pad` / `tile` / `cast` / `sub` / `mul` | no Level 1 op; they are refused, not silently dropped |
| dynamic shapes | every Level 1 tensor is statically shaped; tiling needs the extents |
| per-channel activation zero points | no PTQ toolchain emits them, and neither TOSA nor linalg can express them |
| a real scale change fused into `kea.pool` | `KEA_VPOOL` has no multiplier; only a zero-point rebase is representable, and the verifier enforces it |
| recovering `tensor.pad` into `pads` on the linalg path | §5.1 |
| recovering a bias-broadcast `linalg.generic` into `bias` | §5.2 |

---

## 8. Relationship to Level 2

Level 1 carries everything the tiling pass needs and nothing it does not.
Specifically, KEA-1 has **no convolution instruction**: a conv2d lowers to one
`LOAD_W` + `MATMUL` pair per `(kh, kw)` tap, all accumulating into one `ACC`
region (ISA.md §8.5). That is why `kea.conv2d` keeps `KH`, `KW`, `pads`,
`strides` and `dilations` as first-class structure rather than pre-expanding
anything: `-kea-tile` needs the tap loop bounds and the per-tap input offsets.

Level 1 is also careful not to force a reordering later. The fusion pass only
folds a residual whose value already dominates the consuming contraction, so the
Level 2 scheduler is never handed a graph whose data flow contradicts program
order — which matters because of Rule D (ISA.md §5.5): every `WAIT`'s producing
`SIGNAL`s must appear earlier in the instruction stream, or the single in-order
dispatcher wedges.

The Level 2 ops live in `compiler/include/kea/Dialect/KeaMachineOps.td`, which
has its own TableGen target so that the two halves of the dialect never edit the
same file.
