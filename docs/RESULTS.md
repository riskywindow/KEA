# End-to-end results: MobileNetV2 int8 on KEA-1

**What this document is.** The measured outcome of running a real quantized
network through the whole stack — `.kgraph.json` → TOSA MLIR → `kea-opt` →
`kea-translate` → `kea-as` → `kea-sim` — with the numerical validation, the
roofline analysis, the scheduler A/B, and every place the stack did not work.

**Every number here came out of a command in this repository.** Reproduce all
of it with:

```sh
source scripts/env.sh
bash scripts/build_compiler.sh && bash scripts/build.sh
bash demo/run_all.sh
```

which writes `demo/results/{compile,validation,roofline,ab_schedule}.json`,
`demo/results/roofline_layers.csv` and `demo/roofline_mobilenetv2.png`. The
tables below are those files.

| | |
|---|---|
| Model | `models/mobilenetv2_int8.kgraph.json` — torchvision `IMAGENET1K_V1`, percentile observer, BN folded, 183 nodes / 52 convolutions |
| Input | 224×224×3 int8, batch 1. **Full resolution — no width multiplier, no reduced input size.** |
| Toolchain | LLVM/MLIR 20.1.6, `kea-opt`/`kea-translate` from `compiler/`, `keac`/`kea-as`/`kea-sim` from `tools/` |
| Date | 2026-08-01 |

---

## 1. The missing link: `.kgraph.json` → TOSA MLIR

`frontend/kea_frontend/tosa_emit.py` implements the mapping
`docs/FRONTEND.md` §5 specifies. It covers `conv2d`, `depthwise_conv2d`,
`fully_connected`, `matmul`, `add`, `clamp`, `rescale`, `avg_pool2d`,
`global_avg_pool`, `reshape`, `transpose`, `concat`, `pad` and `table`. It
refuses `softmax` and `layernorm` by name, because those are fused KEA ops with
no TOSA counterpart.

**It emits TOSA for MLIR 20.1.6 and nothing else**, following the
machine-verified spellings in `docs/TOSA_NOTES.md`: zero points as
`#tosa.conv_quant` / `#tosa.matmul_quant` / `#tosa.unary_quant` / `#tosa.pad_quant`
*attributes*; `acc_type = i32` mandatory; `tosa.rescale`'s `shift` as
`array<i8:…>` with `scale32`/`double_round`/`per_channel` all present;
`min_fp`/`max_fp` on `tosa.clamp` even for integer tensors; `new_shape` as an
attribute on `tosa.reshape`; the permutation as an *operand* on
`tosa.transpose`; `!tosa.shape` on `tosa.pad`; OHWI conv weights next to HWCM
depthwise weights.

Constants go out in MLIR's hex `dense<"0x…">` byte form. MobileNetV2's 3.5 MB
of weights are 7.4 MB of MLIR that way; decimal would be about 25 MB.

**Round-trip status: passing.** `frontend/tests/test_tosa_emit.py` (18 tests)
pushes every emitted fixture through `mlir-opt` twice — parse, print, re-parse —
and asserts the second parse succeeds. Four fixtures additionally go through
the real backend (`--tosa-to-kea --kea-fuse --kea-tile --kea-schedule
--kea-alloc`, then `kea-translate`) and must reach `HALT` in `.kasm`. The whole
183-node MobileNetV2 module round-trips (`test_mobilenetv2_whole_graph_round_trips`).

```sh
.venv/bin/python -m pytest frontend/tests            # 92 passed
```

### 1.1 The one place the emitter is not exact, and why

`global_avg_pool` in the KEA IR folds an **arbitrary** requantization into
`multiplier`/`shift`. `tosa.avg_pool2d` can only divide by the window count and
rebase the zero point. For MobileNetV2's head the folded factor is
`1947307623 >> 36 = 0.0283372`, i.e. `(1/49) × 1.3885` — the pool also converts
between two different activation scales (`head` is 0.023529, `gap` is
0.016946).

The emitter therefore emits `tosa.avg_pool2d` at the *input* scale followed by
a `tosa.rescale` of `1490907399 >> 30`, records the substitution in the
module header and in `EmitResult.notes`, and states in both places that it is
**not bit-exact** (two roundings where the reference does one). Nothing in the
feature-extractor slice needs this — `test_mobilenetv2_feature_extractor_round_trips`
asserts `notes == []` for nodes 0..178.

It turns out not to matter for the demo, because **the pool does not compile at
all** (§3, defect 1) and is run on the host with the reference kernel.

---

## 2. What compiled, and what ran

`demo/results/compile.json`.

| attempt | nodes | result | instructions |
|---|---|---|---|
| whole graph, one function | 183 | **failed** — `kea.rescale` after the pool has no Level 2 lowering | — |
| feature extractor, default `spm-reserve-factor = 2` | 179 | **failed** — IMEM | **41,409** |
| feature extractor, `--spm-reserve 1` | 179 | **ok** | **30,773** |
| classifier (`reshape` + `fully_connected` + `rescale`) | 3 | **ok** | **10,833** |
| `global_avg_pool` alone | 1 | **failed** | — |

**182 of 183 nodes (99.5%) compile and run. All 52 of 52 convolutions compile
and run, at full 224×224 resolution.** The one node that does not is the global
average pool.

The network therefore runs as **two `.keaf` programs** with one host step
between them:

```
input (1,224,224,3) i8
  └─ features.keaf   nodes 0..178, 52 convolutions      → head (1,7,7,1280) i8
       └─ global_avg_pool                    ON THE HOST, numpy reference
            └─ classifier.keaf  nodes 180..182          → fc (1,1000) i8
```

This is stated wherever a number is reported. The pooled node is 62,720
additions — 0.01% of the network's arithmetic — but it is **not running on the
NPU**, and "MobileNetV2 runs end to end on kea-sim" would be an overclaim. The
honest statement is: *the 52-convolution feature extractor and the classifier
each run as a single unrolled KEA-1 program, and one reduction between them
does not compile.*

### 2.1 IMEM: the real capacity story

`KEA_MAX_INSTRUCTIONS` is 32,768 and there is no loop instruction, so the whole
network is one straight line of instructions.

At the compiler's default `spm-reserve-factor = 2` the feature extractor is
**41,409 instructions — 26% over IMEM**. That is not a marginal overrun.

`--spm-reserve 1` gives `-kea-tile` the full 256 KiB of SPM_A and all 32,768
ACC words per tile instead of half, which produces larger tiles and **30,773
instructions — 94% of IMEM**. That is what the demo ships. The tradeoff is real
and worth stating: the reserve factor exists to leave headroom for
double-buffering across tiles, and spending it buys the instruction budget back.

Per-layer instruction counts at reserve 2 show where the budget goes: layer 3
(the 16→96 expand at 112×112) alone is **6,392** instructions, and the first ten
layers are **15,783** of the 41,409 — the early high-resolution layers dominate,
because a 112×112×96 activation is 1.2 MB against a 256 KiB scratchpad and has
to be cut into many bands. At 94% of IMEM with the reserve factor already spent,
**there is very little headroom left** — this is not measured for other
resolutions or width multipliers, but the tiling of the early layers scales with
spatial area, so a larger input would need a different tiling strategy (or a
loop instruction) rather than a different flag.

---

## 3. Backend defects found

All four are reproduced by `bash demo/repro/run_repro.sh`, which exits 0 while
they still behave as described. **None of them were worked around in
`compiler/`** — the demo is shaped around them.

### Defect 1 — every `kea.pool` fails to translate (blocking)

`-kea-tile`'s `lowerPool` names the pool's SPM_A output tile with
`layerName("out")`, the same string it just gave the DRAM buffer backing the
pool's result:

```c++
Value outBuf = dram(op.getOutput(), "activation", layerName("out"));   // DRAM
Value oTile  = scratch(loc, ..., AddressSpace::A, layerName("out"));   // SPM_A
```

`scratch()`'s uniquifier only disambiguates against other *scratch* names, so
the two collide and `kea-translate` rejects the module:

```
error: 'kea.alloc' op duplicate buffer name "gap_pool.0.out"; a DRAM name is
also the .kasm symbol and must be unique (docs/DIALECT_L2.md §4.1)
```

The convolution path avoids this only by accident — it calls its SPM tile
`otile`. The collision is unconditional: it does not depend on the pool being a
graph output, on the pool kind, or on the tiling. **`tosa.avg_pool2d` and
`tosa.max_pool2d` are currently unusable end to end**, which is worth knowing
independently of MobileNetV2.

Reproducer: `demo/repro/pool_dram_name_collision.mlir` (a bare 7×7
`tosa.avg_pool2d`, 8 lines).

### Defect 2 — `kea.matmul` with a non-constant second operand emits a null operand

`-kea-tile` assumes a matmul's `b` operand is a compile-time constant, and
builds a `kea.alloc` with a **null** source when it is not:

```
error: null operand found
note: see current operation: %3 = "kea.alloc"(<<NULL VALUE>>)
      <{... name = "mm_act.1.weights", role = "weights"}>
```

The diagnostic points at nothing actionable. It should say "`-kea-tile` does
not lower `kea.matmul` with a non-constant second operand" and name the op, the
way the `kea.transpose` path does.

This is not hypothetical: `models/tiny_vit_int8.kgraph.json` has **12** rank-3
matmuls with two activation operands (attention QKᵀ and PV). It also closed off
the most promising workaround for defect 1 — expressing the 7×7 global
reduction as `matmul(ones[1,1,49], x[1,49,1280])`.

Reproducer: `demo/repro/matmul_activation_rhs.mlir`.

### Defect 3 — `--kea-schedule` violates Rule D on long programs

```
error: 'kea.wait' op violates Rule D (ISA.md §5.5): only 0 of the 1 counts this
WAIT consumes are signalled earlier in the stream, so a full queue behind it
would wedge the in-order dispatcher
```

The scheduler emits a `WAIT` whose producing `SIGNAL` is placed *later* in the
stream. `-kea-tile`'s own output does not have this problem — the same module
compiles cleanly without `--schedule`.

It is **length dependent, not layer dependent**. Measured by exhaustive scan
over every prefix of this network at `--spm-reserve 1`:

* prefixes ending at node ≤ 97 (28 convolutions): **schedule fine**
* every lowerable prefix ending at node ≥ 99: **Rule D violation**
* a *middle* slice, nodes 60..99, schedules fine on its own — so it is not any
  individual layer

At `--spm-reserve 2` the 0..99 prefix schedules but the full 0..178 one does
not, so the threshold moves with the tiling and the failure is not tied to one
reserve factor.

Consequence for this report: **the scheduled/unscheduled A/B could not be run
on one whole-network program.** It is measured per layer instead (§6), which is
a weaker experiment and is labelled as such.

Reproducer: `demo/repro/run_repro.sh`, defect 3 (prefix 0..99, `--spm-reserve 1`).

### Not a defect, but a limit worth recording — the fused `kea.add` domain

The KEA `VADD` cannot express every `(lhs, rhs, out)` requantization triple:

```
error: 'kea.add' op cannot express this quantized add as KEA_VADD: the output
rescale would need a left shift KEA_VADD cannot express (keaRdpot ignores a
negative exponent)
```

MobileNetV2's ten residual adds are all inside the representable domain, so
this never fires on the shipped model. It fired on a hand-written test fixture
with an implausible scale ratio, and the diagnostic named the problem exactly.
Recorded because it is a real constraint on what a frontend may emit.

---

## 4. Numerical validation — **bit-exact**

`demo/results/validation.json`. `frontend/testdata/golden_io_mobilenetv2_int8.npz`
holds four fixed int8 inputs and the exact int8 outputs of the numpy reference
interpreter. `docs/FRONTEND.md` §4.1: *"assert exact integer equality. There is
no tolerance."*

| comparison | elements per vector | mismatches (4 vectors) |
|---|---|---|
| `features.keaf` output vs the reference's `head` | 62,720 | **0** |
| host pool vs the reference's `gap` | 1,280 | **0** |
| `classifier.keaf` fed the reference `gap`, vs the golden output | 1,000 | **0** |
| **end to end** (sim → host pool → sim) vs the golden output | 1,000 | **0** |

Across the four vectors that is **258,880 simulator-produced int8 values**
(4 × 62,720 feature values, 4 × 2,000 logits) compared against the reference,
plus 5,120 host-pool values. **Zero mismatches anywhere.** argmax agreement 4/4.

The middle row is the host step and is reported separately on purpose: the
classifier is validated *both* against the reference's own `gap` (which
isolates it) and against the simulator's `head` pushed through the reference
pool (which is the real chain). Both are exact, which is expected — the
simulator's `head` is bit-identical to the reference's, so the pool receives
identical input either way.

Both programs also pass the simulator's strict gates:

```sh
kea-sim demo/build/features.keaf   --quiet --strict-poison --strict-hazards   # exit 0
kea-sim demo/build/classifier.keaf --quiet --strict-poison --strict-hazards   # exit 0
```

i.e. **no read of never-written scratchpad and no cross-unit read of data whose
producer had not retired** — no missing `SIGNAL`/`WAIT` anywhere in 41,606
instructions. `diagnostics` in both stats files is empty.

That the requantization matches at all is the payoff of
[ADR-0003](adr/0003-requantization-equivalence-invariant.md): `VQUANT` runs
gemmlowp-style `keaRequantize` while the golden model runs TOSA
`apply_scale_32`, two different functions that agree only on the normalised
domain. **300,774,272 MACs' worth of results agree exactly** across this model.

One small correction to ADR-0003 while we are here: it says MobileNetV2's
rescale shifts "range over **22** to 48". Measured on the shipped
`models/mobilenetv2_int8.kgraph.json`, the range is **21 to 48** — `rescale#134`
(`b12_c2` → `b12_res.b32`, per-tensor) has `shift = 21`, one below the quoted
floor. It is still inside the invariant (`tosa_shift < 31` is exact while
`|acc| < 2^21`, and this is a residual-branch rescale of an i8 input, so
`|v| ≤ 255`), and the results are bit-exact, so nothing is wrong — but the
number in the ADR is off by one and should be updated.

---

## 5. Roofline

`demo/results/roofline.json`, `demo/results/roofline_layers.csv`,
`demo/roofline_mobilenetv2.png`. All figures are `kea-sim --stats-json` output;
the roofline arithmetic is the simulator's, per `docs/MICROARCH.md` §9.2:

```
intensity  = ops_useful / dram_bytes                 (1 MAC = 2 ops)
attainable = min(512 GOPS, intensity × 16 GB/s)
achieved   = ops_useful / (cycles / 1 GHz)
```

### 5.1 Whole network

| | features (52 conv) | classifier | **whole compiled network** |
|---|---|---|---|
| cycles | 3,231,748 | 143,425 | **3,375,173** (3.375 ms @ 1 GHz) |
| useful ops | 598,988,544 | 2,560,000 | **601,548,544** |
| issued ops | 642,340,608 | 2,580,480 | 644,921,088 |
| padding efficiency | 93.25% | 99.21% | **93.27%** |
| DRAM bytes | 17,309,080 | 1,330,216 | **18,639,296** (11.9 MB in / 6.7 MB out) |
| arithmetic intensity | 34.61 ops/B | **1.92 ops/B** | **32.27 ops/B** |
| achieved | 185.3 GOPS | 17.9 GOPS | **178.2 GOPS** |
| attainable | 512 GOPS | 30.8 GOPS | **512 GOPS** |
| % of attainable | 36.2% | 58.0% | **34.8%** |
| bound | COMPUTE | **MEMORY** | COMPUTE (barely) |
| MXU MAC utilisation | 33.70% | 3.49% | **32.41%** of the 256 int8 MAC/cycle peak |
| DWU MAC utilisation | 5.01% | — | of the 128 MAC/cycle peak |
| VPU utilisation | 30.18% | — | of 16 elem/cycle |
| achieved DRAM bandwidth | 5.36 GB/s | 9.28 GB/s | of 16 GB/s peak |

300.8 M MACs matches MobileNetV2's published ~300 M multiply-accumulates, which
is a useful independent check that the compiler is not silently skipping work.

**The network sits within 1% of the ridge point** (32.27 vs 32.0 ops/byte). It
is nominally compute bound, but only just: it is the single least comfortable
place on a roofline to be, and it is why neither more bandwidth nor more MACs
would help much on their own.

Where the 3.23 M cycles go in the feature extractor:

| unit | busy | semaphore stall | resource stall | idle |
|---|---|---|---|---|
| MXU | 37.87% | 38.6% | 0.09% | 23.4% |
| DWU | 5.07% | 21.7% | 0.02% | 73.2% |
| VPU | 22.04% | 61.3% | 0.12% | 16.6% |
| DMA0 | 39.96% | 28.5% | 0.06% | 31.5% |
| DMA1 | **0%** | — | — | 100% |

The dispatcher is stalled 98.97% of cycles, 2,277,859 of them behind a full MXU
queue. **DMA1 is completely unused in the unscheduled build** — allocating the
second engine is something `--kea-schedule` does, and it is a large part of the
per-layer speedups in §6 (1.267× aggregate, up to 1.971×).

### 5.2 Per layer

Full table in `demo/results/roofline_layers.csv` (53 rows). Grouped:

| family | layers | intensity | achieved | share of ops | share of layer-cycles | share of DRAM |
|---|---|---|---|---|---|---|
| 1×1 pointwise conv (MXU) | 35 | 19.99 – 89.96 ops/B | 117 – 324 GOPS | 91.5% | 72.1% | 58.8% |
| 3×3 depthwise (DWU) | 17 | **3.66 – 9.97 ops/B** | 35 – 78 GOPS | **8.1%** | **23.9%** | **34.6%** |
| classifier `fully_connected` | 1 | **1.92 ops/B** | 17.9 GOPS | 0.4% | 4.0% | 6.6% |

(shares are of the per-layer totals: 621,992,576 ops, 3,618,115 layer-cycles,
20,294,816 DRAM bytes — see the overlap note at the end of this section.)

**24 of 53 layers are memory bound** — all 17 depthwise layers, the classifier,
and six pointwise convolutions (layers 2, 3, 5, 11, 20, 41) whose activations
are large relative to their weights.

Why the families look so different, concretely:

* A **3×3 depthwise** does exactly 9 MACs per input element, whatever the
  channel count. Intensity is therefore pinned by the kernel: with one i8 read
  and one i8 write per element it can never exceed ~9 ops/byte, and the
  measured range is 3.66–9.97 (the stride-2 layers are lower because they read
  4× the pixels they write). Every one of them lands under the ridge. They do
  **8.1%** of the network's arithmetic, occupy **23.9%** of its layer-cycles,
  and move **34.6%** of its DRAM traffic. The DWU is idle 73% of the time not
  because it is slow but because it is starved.
* A **1×1 pointwise** does `IC` MACs per input element, so intensity grows with
  channel count: layer 2 (32→16 at 112×112) is at 20.0 ops/byte, while layer 51
  (320→1280 at 7×7) is at 81.0 and reaches 323.9 GOPS, 63.3% of attainable.
  Late layers with many channels and few pixels are exactly what this machine
  is built for.
* The **classifier** reads 1.28 MB of weights to do 1.28 M MACs — one weight
  byte per two ops, hence 1.92 ops/byte, an attainable ceiling of 30.8 GOPS,
  and 3.49% MAC utilisation. It reaches 58% of that ceiling, which is a *good*
  result for a layer that is pure bandwidth. This is the clearest case in the
  network of a layer that no scheduler can help.

One measurement note: per-layer TRACE regions sum to 3,474,690 cycles against a
3,231,748-cycle program — a **7.5% overlap**. That is real: consecutive layers
genuinely overlap (the DWU region for layer 1 opens at cycle 152,599 while the
MXU is still finishing layer 0 at 159,002). Region cycles must not be summed
and presented as a total.

---

## 6. `--kea-schedule` A/B

`demo/results/ab_schedule.json`. Each layer is emitted as a standalone TOSA
function, compiled twice (with and without `--schedule`, `--spm-reserve 1`
both times) and run on `kea-sim`.

### 6.1 The whole network could not be measured

```
whole feature extractor (nodes 0..178):
  unscheduled  3,231,748 cycles
  scheduled    FAILED: 'kea.wait' op violates Rule D (ISA.md §5.5)
```

That is defect 3. The largest prefix of MobileNetV2 that `--kea-schedule`
accepts at `--spm-reserve 1` ends at **node 97** — 28 of 52 convolutions.

On that prefix — the only genuine multi-layer A/B available — **the scheduler
loses**:

| nodes 0..97, 28 convolutions | unscheduled | scheduled |
|---|---|---|
| cycles | **2,130,318** | **2,240,691** (**0.951×**) |
| instructions | 11,808 | 11,571 |
| achieved | 138.0 GOPS | 131.2 GOPS |
| MXU busy | 28.61% | 27.20% |

A 5% regression, not a speedup. This is the single most important caveat on the
scheduler's headline number: it is measured on isolated layers, and the one
place where cross-layer effects are in play it goes the other way. The likely
mechanism is visible in §5.2 — consecutive layers already overlap by 7.5% in
the unscheduled build, and the list scheduler's per-layer reordering appears to
break some of that overlap. It has not been diagnosed further here; it is
reported as measured.

### 6.2 Per layer

53 layers compiled and run twice each, **0 failures**:

| | unscheduled | scheduled | speedup |
|---|---|---|---|
| **all 53 layers, summed** | **3,376,277** | **2,664,995** | **1.267×** |
| 17 depthwise (DWU) | 841,877 | 551,518 | **1.526×** |
| 35 pointwise conv (MXU) | 2,393,537 | 1,973,253 | **1.213×** |
| 1 classifier `fully_connected` | 140,863 | 140,224 | 1.005× |

Distribution: **31 layers gain more than 10%**, 12 land within ±2%, and **2
regress** — `conv2d#23` at 0.962× and `depthwise_conv2d#32` at 0.905×
(62,065 → 68,571). Best: `depthwise_conv2d#20` at **1.971×** (131,941 →
66,949).

Full table in `demo/results/ab_schedule.json`. Depthwise layers gain most,
which follows from §5.2: they are memory bound with one idle DMA engine, so
putting loads and stores on both engines is directly worth cycles. The
classifier gains nothing, which also follows — it is at 1.92 ops/byte and
already saturating what one DRAM port can deliver.

### 6.3 Reading it honestly

* The per-layer aggregate is **not** a whole-network speedup. Each layer is its
  own program here, so every one pays a cold start and none of them overlaps
  its neighbours — which the whole-network unscheduled run demonstrably does
  (§5.2, 7.5% region overlap). The correct claim is "the scheduler is worth
  1.267× on the layers it was measured on, in isolation, and −5% on the one
  28-layer program it could be measured on".
* The mechanism is visible in the assembly, not inferred. The unscheduled
  stream issues every DMA on **DMA0**, never touches DMA1, and puts a `WAIT`
  between each producer and consumer. The scheduled stream spreads loads and
  stores across both engines, hoists the next tile's `DMA_LD` between the
  current tile's two `MATMUL`s, and alternates `LOAD_W` between weight banks
  so a weight load never waits for the array. `demo/results/schedule_excerpt.kasm`
  is the exact text.
* Where the speedup is largest it is because the layer was DMA bound with one
  idle engine; where it is ~1.00× the layer was already limited by something
  the scheduler cannot move.

---

## 7. What did not work

Listed so it is on the record, not buried.

1. **The global average pool does not run on the NPU** (defect 1). It is
   executed on the host by `frontend/kea_frontend/reference.py`'s own kernel.
   Every workaround was tried and each was blocked by something real:
   * `tosa.avg_pool2d` → defect 1;
   * a 7×7 `tosa.depthwise_conv2d` of ones → `'kea.dwconv2d' op DWCONV
     supports square 3x3 or 5x5 kernels only, got 7x7` (a correct, documented
     ISA limit);
   * `tosa.matmul(ones[1,1,49], x[1,49,1280])` → defect 2;
   * `tosa.transpose` + `matmul(x^T, ones)` → `'kea.transpose' op has no
     Level 2 lowering` (correct: KEA-1 has no transpose unit);
   * a 1280×1280 conv2d over the reshaped tensor → 80 MB of weights, absurd.
2. **The whole network is not one program**, for the same reason.
3. **`--kea-schedule` on the whole network** (defect 3). The scheduler's
   central claim is therefore measured per layer, not on the artifact that
   ships.
4. **The default SPM reserve factor does not fit IMEM** (§2.1). Not a defect —
   a genuine capacity result — but it means the shipped configuration is
   `--spm-reserve 1`, and there is very little room left.
5. **The tiny ViT was not attempted end to end.** Its 12 activation×activation
   matmuls hit defect 2, and its `softmax` and `layernorm` have no TOSA
   spelling at all (`docs/FRONTEND.md` §5 says so; the emitter refuses them by
   name rather than approximating them). Compiling it needs `kea.softmax` and
   `kea.layernorm` custom ops plus a matmul path for non-constant operands.
6. **`tosa.avg_pool2d` with an inexact folded scale is not bit-exact** (§1.1).
   Unexercised here because the only such node in this model does not compile,
   but it is a latent correctness gap for any model whose pooling changes
   scale, and the emitter says so in the file it writes.

---

## 8. Reproducing a single number

```sh
# instruction count and IMEM fit
build/native/bin/keac demo/build/mobilenetv2_features.tosa.mlir \
    --function mnv2_features -o /tmp/f.keaf --spm-reserve 1 --keep-intermediates
grep -cE '^\s+(MXU|DWU|VPU|DMA0|DMA1|CTRL)\s' /tmp/f.kasm     # 30773

# whole-program roofline
build/native/bin/kea-sim demo/build/features.keaf --stats-json /tmp/s.json
.venv/bin/python -c "import json;print(json.load(open('/tmp/s.json'))['global']['roofline'])"

# bit-exactness on one golden vector
.venv/bin/python demo/validate_mobilenetv2.py

# every defect, still failing the same way
bash demo/repro/run_repro.sh
```
