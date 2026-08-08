# End-to-end results: MobileNetV2 int8 on KEA-1

**What this document is.** The measured outcome of running a real quantized
network through the whole stack — `.kgraph.json` → TOSA MLIR → `kea-opt` →
`kea-translate` → `kea-as` → `kea-sim` — with the numerical validation, the
roofline analysis, the scheduler A/B, and every place the stack did not work.

**Every number here came out of a command in this repository, at the
compiler's defaults.** Nothing is pinned: `spm-reserve-factor = 1`,
`-kea-tile=imem-budget = 20480`, `-kea-schedule mode = auto`. Reproduce all of
it with:

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
| Suites | compiler lit 34/34, native ctest 17/17, frontend pytest 92 passed, `numeric_check` 7/7 |
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

### 1.1 The one place the emitter is not exact — and the one that matters

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

**That trailing rescale is now the single node that keeps MobileNetV2 from
being one program** (§3.1). It is not a bit-exactness problem in practice,
because the pool never reaches the machine at all.

---

## 2. What compiled, and what ran

`demo/results/compile.json`. All at the defaults.

| attempt | nodes | result | instructions |
|---|---|---|---|
| whole graph, one function, defaults | 183 | **failed** — coarsest tiling needs **29,241** instructions against the 20,480 budget | — |
| whole graph, `--imem-budget 29241` | 183 | **failed** — `kea.rescale` after the head pool has no Level 2 lowering | — |
| *structural probe:* same graph, pool rescale deleted, `--imem-budget 29241` | 183 | **failed** — **36,633** instructions, IMEM holds 32,768 | 36,633 |
| *structural probe*, `--imem-budget 32768` | 183 | **failed** — `-kea-tile` itself refuses: "lowers to 32,772 KEA-1 instructions" | — |
| feature extractor, no `--schedule` | 179 | **ok** | 30,498 |
| feature extractor, `--schedule` | 179 | **ok** | **30,430** |
| classifier, no `--schedule` | 3 | **ok** | 10,539 |
| classifier, `--schedule` | 3 | **ok** | **10,507** |
| head pool as the model spells it (`avg_pool2d` + `rescale`) | 1 | **failed** — the rescale | — |
| a bare 7×7 `tosa.avg_pool2d`, no scale change | 1 | **ok**, runs in **460 cycles**, VPU 42.6% | 10 |

**182 of 183 nodes (99.5%) compile and run. All 52 of 52 convolutions compile,
schedule and run, at full 224×224 resolution.** The one node that does not is
the global average pool.

### 2.1 MobileNetV2 is not one program, for two independent reasons

This was worth re-testing, because the buffer-name collision that used to make
*every* `kea.pool` untranslatable is fixed — a bare pool now compiles,
assembles and runs (last row above). Two things still stand in the way, and
they are different in kind:

1. **A lowering gap.** MobileNetV2's head pool changes activation scale, so the
   frontend emits it as `avg_pool2d` + `rescale`. `-kea-fuse` only fuses a
   `kea.rescale` into a *contraction* (`conv2d`, `dwconv2d`, `matmul`,
   `fully_connected`); `kea.pool`'s own `quant` is restricted by its verifier to
   a zero-point rebase, because only that much is exact for an average. So the
   rescale survives `-kea-fuse` and `-kea-tile` refuses it:

   ```
   error: 'kea.rescale' op has no Level 2 lowering. -kea-tile lowers
   kea.conv2d, kea.dwconv2d, kea.matmul, kea.fully_connected, kea.pool,
   quantized kea.add and kea.reshape; kea.rescale, kea.clamp and
   kea.transpose must be eliminated by -kea-fuse
   ```

   `docs/DIALECT_L1.md`'s `PoolOp` says in as many words that "a general
   rescale after a pool must be written as a separate `kea.rescale`" — and a
   separate `kea.rescale` is exactly what has no lowering. **The L1 dialect can
   express a scale-changing pool that the backend cannot compile.** That is a
   real gap, reproduced by `demo/regress/pool_rescale_unsupported.mlir`.

2. **A hard capacity result, which is the bigger one.** Deleting that rescale
   (a *structural* probe — numerically wrong, never run) lets the whole graph
   tile. It then lowers to **36,633 KEA-1 instructions against a 32,768-entry
   IMEM: 11.8% over, at the coarsest tiling of every layer.** Pushed to
   `--imem-budget 32768`, `-kea-tile` refuses before `kea-as` gets a chance
   ("lowers to 32,772 … and `-kea-schedule` still has to add its SIGNAL/WAIT
   pairs"). So **even if the rescale lowered, MobileNetV2 would not fit in one
   KEA-1 program.** Fixing the lowering gap would not change the split; it would
   only move where the split can be made.

The network therefore still runs as **two `.keaf` programs** with one host step
between them:

```
input (1,224,224,3) i8
  └─ features.keaf   nodes 0..178, 52 convolutions      → head (1,7,7,1280) i8
       └─ global_avg_pool                    ON THE HOST, numpy reference
            └─ classifier.keaf  nodes 180..182          → fc (1,1000) i8
```

The pooled node is 62,720 additions — 0.01% of the network's arithmetic — but
it is **not running on the NPU**, and "MobileNetV2 runs end to end on kea-sim"
would be an overclaim. The honest statement is: *the 52-convolution feature
extractor and the classifier each run as a single scheduled KEA-1 program, and
one reduction between them does not compile.*

### 2.2 IMEM: the real capacity story

`KEA_MAX_INSTRUCTIONS` is 32,768 and there is no loop instruction, so the whole
network is one straight line of instructions. `-kea-tile` is IMEM-aware: each
layer publishes a Pareto frontier of (instructions, cycles) and a Lagrangian
price is bisected for the cheapest plan fitting `imem-budget`, default 20,480.
The budget is on the *tiled* program; `-kea-schedule` and `kea-translate` then
add their SIGNAL/WAIT pairs, which is what the remaining 12,288 instructions of
headroom are for.

Measured on this network, the headroom is well calibrated but the usable window
is **narrow at both ends**:

| | |
|---|---|
| smallest budget `-kea-tile` accepts | **19,066** — its own coarsest tiling of all 52 layers |
| largest budget that still assembles | between **21,250** and **21,500** — above it the finished program exceeds IMEM |
| the default, 20,480 | 30,430 scheduled instructions, **92.9% of IMEM** |

So the whole feasible window is about −7%/+4% around the default, and the
default sits comfortably inside it. `spm-reserve-factor` now defaults to 1 and the demo does
not touch it: the tiling cost model prices double-buffering directly, so the
old divisor is redundant.

Two programs at the defaults come to **40,937 instructions** (30,430 + 10,507),
against a single 32,768-entry IMEM — which is the same capacity story from the
other side.

---

## 3. Bugs found and fixed during bring-up

Bringing this network up found four backend defects. **All four are fixed**, in
`compiler/`, and `bash demo/regress/run_regressions.sh` now asserts the *fixed*
behaviour so it cannot silently come back. That script replaces the old
`demo/repro/run_repro.sh`, which asserted the opposite.

| | what was wrong | how it showed up | now |
|---|---|---|---|
| 1 | `-kea-tile`'s `lowerPool` gave the pool's SPM_A tile the same name as the DRAM buffer holding its result; `scratch()`'s uniquifier only disambiguated against other scratch names | `error: 'kea.alloc' op duplicate buffer name "gap_pool.0.out"` — **every** `kea.pool`, in every program | uniquification moved into `makeBuffer()`, so it covers every buffer. A bare 7×7 `tosa.avg_pool2d` compiles and runs in **460 cycles** at 42.6% VPU utilisation |
| 2 | `-kea-tile` assumed a `kea.matmul`'s second operand was a compile-time constant and built a `kea.alloc` with a null source when it was not | `error: null operand found` / `%3 = "kea.alloc"(<<NULL VALUE>>)` — pointing at nothing actionable | a real diagnostic naming the op and the weight-stationary reason. The limitation itself stands: the MXU cannot do activation×activation |
| 3 | rotation edges added *after* the dependence graph made a DMA a direct consumer of a `LOAD_W`, so a `SIGNAL` collapsed with an existing one and two waits shared one token | `error: 'kea.wait' op violates Rule D (ISA.md §5.5)` on every lowerable prefix past node 97 | the whole 52-convolution feature extractor schedules, and `kea-sim --strict-poison --strict-hazards` is clean on it |
| 4 | tiling was greedy per layer, so program size — a *global* capacity limit on a branchless machine — was nobody's constraint | the feature extractor was 41,409 instructions against a 32,768 IMEM, and the demo had to be pinned to `--spm-reserve 1` | `-kea-tile` is IMEM-aware (§2.2). The feature extractor fits at the defaults with **no flags at all**, and the pin is gone |

Two of the four were *diagnosis* bugs rather than capability bugs — defect 1
made a working code path unreachable, defect 2 hid a legitimate refusal behind
a verifier crash — which is the pattern worth taking away: the compiler could do
more than it appeared to, and the reproducers were what showed that.

### 3.1 What is still open

**A general rescale after a pool has no lowering.** §2.1(1). Reproducer:
`demo/regress/pool_rescale_unsupported.mlir`. Two ways out, neither taken here:
fuse a rescale into a `kea.pool` epilogue (the VPU's pool already requantizes),
or fuse it forward into the consumer contraction's input quantization.

**`-kea-schedule` hoists TRACE region-begin markers.** A `kea.trace` begin
marker has no operands, so nothing stops the list scheduler from placing it at
the earliest free slot on its queue. On the shipped scheduled feature extractor,
**10 of 52 region-begin markers land in the first hundred instructions of a
30,430-instruction program** while their `end` markers stay with their layer:

```
$ grep -nE 'TRACE' demo/build/features.kasm | head -12
59:  MXU   TRACE   kind=begin, tag=0, payload=0
68:  MXU   TRACE   kind=begin, tag=8, payload=0
74:  MXU   TRACE   kind=begin, tag=14, payload=0
77:  MXU   TRACE   kind=begin, tag=17, payload=0
   …  begins for tags 23 26 29 35 38 44 47 follow, all before line 102
782:  DWU   TRACE   kind=begin, tag=1, payload=0    <- where tag 8 belongs
790:  MXU   TRACE   kind=end,   tag=0, payload=0
```

The simulator then reports those regions as spanning from cycle ~0, and the 52
regions sum to 23,606,140 cycles inside a 3,176,061-cycle program. The
unscheduled build is clean (max nesting depth 1, regions sum to 5.4% over the
program total, which is genuine layer overlap). `demo/common.py`'s
`unsound_regions()` detects the condition — a region that begins before a
lower-tagged one — and `demo/roofline.py` **refuses to report those layers**
rather than summing nonsense. **The consequence for this report: every per-layer
number in §5.2 comes from the unscheduled build.** Intensity, ops and DRAM bytes
are schedule-invariant, so only the cycle and GOPS columns are affected.

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

**Both programs under test are the scheduled builds at the defaults**, so this
validates `-kea-schedule`'s reordered, double-buffered, two-DMA-engine output —
not just the tiler's sequential one.

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
producer had not retired** — no missing `SIGNAL`/`WAIT` anywhere in 40,937
scheduled instructions. `diagnostics` in both stats files is empty. That is the
strongest single statement in this report: the scheduler moved instructions
across queues, split DMA over two engines and pipelined tiles, and the answer
did not change by one bit.

That the requantization matches at all is the payoff of
[ADR-0003](adr/0003-requantization-equivalence-invariant.md): `VQUANT` runs
gemmlowp-style `keaRequantize` while the golden model runs TOSA
`apply_scale_32`, two different functions that agree only on the normalised
domain. **300,774,272 MACs' worth of results agree exactly** across this model.
(An earlier revision of this document reported that ADR-0003's quoted rescale
shift range was off by one — 21, not 22, because of `rescale#134`. That
correction has been applied to the ADR.)

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
| cycles, scheduled | 3,176,061 | 140,280 | **3,316,341** (3.316 ms @ 1 GHz) |
| cycles, unscheduled | 3,727,589 | 140,863 | 3,868,452 (3.868 ms) |
| useful ops | 598,988,544 | 2,560,000 | **601,548,544** |
| issued ops | 642,340,608 | 2,580,480 | 644,921,088 |
| padding efficiency | 93.25% | 99.21% | **93.27%** |
| DRAM bytes | 21,591,756 | 1,312,296 | **22,904,052** (16.2 MB in / 6.7 MB out) |
| arithmetic intensity | 27.74 ops/B | **1.95 ops/B** | **26.26 ops/B** |
| achieved (scheduled) | 188.6 GOPS | 18.2 GOPS | **181.4 GOPS** |
| attainable | 443.9 GOPS | 31.2 GOPS | **420.2 GOPS** |
| % of attainable | 42.5% | 58.5% | **43.2%** |
| bound | MEMORY | MEMORY | **MEMORY** |
| MXU MAC utilisation | 34.29% | 3.56% | **32.99%** of the 256 int8 MAC/cycle peak |
| DWU MAC utilisation | 5.10% | — | of the 128 MAC/cycle peak |
| VPU utilisation | 39.14% | — | of 16 elem/cycle |
| achieved DRAM bandwidth | 6.80 GB/s | 9.35 GB/s | 6.91 GB/s of 16 GB/s peak |

300.8 M MACs (13,742 `MATMUL`s and 187 `DWCONV`s) matches MobileNetV2's
published ~300 M multiply-accumulates, which is a useful independent check that
the compiler is not silently skipping work.

**The network is memory bound, at 26.26 against a ridge point of 32.0
ops/byte.** The previous revision of this document reported 32.27 ops/byte and
"compute bound, barely", on a greedily-tiled `--spm-reserve 1` build that
today's compiler can no longer produce. The move below the ridge is a direct
consequence of §2.2, and the trade is worth stating plainly — this is a
cross-version comparison against the recorded old numbers, not two builds of
one compiler:

| | old (greedy tiling, `--spm-reserve 1`, unscheduled) | now (IMEM-aware, defaults) |
|---|---|---|
| instructions | 41,606 | **40,937** (−1.6%) |
| DRAM bytes | 18,639,296 | 22,904,052 (**+22.9%**) |
| intensity | 32.27 ops/B (compute bound) | 26.26 ops/B (**memory bound**) |
| cycles, unscheduled | 3,375,173 | 3,868,452 (**+14.6%**) |
| cycles, scheduled | *could not be built* | **3,316,341** (−1.7% vs old) |

So the IMEM-aware tiler is, on its own, a **cycle and bandwidth regression** on
this network: it spends DRAM traffic to buy instruction budget. What it buys is
that the program fits with no flags at all *and* schedules — and scheduling
more than pays the tiling back, netting 1.7% ahead of the number that used to
ship. That is a real result and it is also a warning: the tiler is optimising
instructions against a modelled cycle cost, and the modelled cost is evidently
not tracking DRAM traffic closely enough. Bandwidth is now the thing to attack.

Where the 3.18 M cycles go in the scheduled feature extractor, against the
unscheduled build:

| unit | busy (sched) | busy (unsched) | semaphore stall | resource stall | idle |
|---|---|---|---|---|---|
| MXU | **38.30%** | 32.63% | 42.2% | 0.08% | 19.5% |
| DWU | 5.17% | 4.40% | 20.5% | 0.03% | 74.3% |
| VPU | 26.62% | 22.68% | 54.2% | 0.11% | 19.1% |
| DMA0 | 38.23% | 48.13% | 38.6% | 0.05% | 23.2% |
| DMA1 | **27.71%** | **0%** | 36.7% | 0.04% | 35.5% |

(stall/idle columns are the scheduled build.) The dispatcher is stalled 98.99%
of cycles. **DMA1 goes from completely idle to 27.7% busy** — allocating the
second engine is the largest single thing `--kea-schedule` does, and §6 prices
it.

### 5.2 Per layer

Full table in `demo/results/roofline_layers.csv` (53 rows). **These rows come
from the unscheduled build**, because `-kea-schedule` corrupts 10 of the 52
TRACE regions (§3.1); the CSV carries a `scheduled_cycles` column that is
filled only for the 42 sound feature-extractor regions, plus the classifier
(which is its own program and has no region problem). Grouped:

| family | layers | intensity | achieved | share of ops | share of layer-cycles | share of DRAM |
|---|---|---|---|---|---|---|
| pointwise/3×3 conv (MXU) | 35 | 18.6 – 74.5 ops/B | 99 – 316 GOPS | 90.9% | 73.2% | 62.8% |
| 3×3 depthwise (DWU) | 17 | **3.80 – 10.17 ops/B** | 35 – 78 GOPS | **8.7%** | **23.3%** | **31.8%** |
| classifier `fully_connected` | 1 | **1.95 ops/B** | 18.2 GOPS | 0.4% | 3.5% | 5.4% |

(shares are of the per-layer totals: 621,005,184 ops, 4,069,681 layer-cycles,
24,233,764 DRAM bytes — see the overlap note at the end of this section.)

**28 of 53 layers are memory bound** — all 17 depthwise layers, the classifier,
and ten convolutions (layers 0, 2, 3, 5, 6, 8, 9, 11, 15, 18), all of them in
the high-resolution front half where activations dwarf weights.

Why the families look so different, concretely:

* A **3×3 depthwise** does exactly 9 MACs per input element, whatever the
  channel count. Intensity is therefore pinned by the kernel: with one i8 read
  and one i8 write per element it can never exceed ~9 ops/byte, and the
  measured range is 3.80–10.17 (the stride-2 layers are lower because they read
  4× the pixels they write). Every one of them lands under the ridge. They do
  **8.7%** of the network's arithmetic, occupy **23.3%** of its layer-cycles,
  and move **31.8%** of its DRAM traffic. The DWU is idle 74% of the time not
  because it is slow but because it is starved.
* A **1×1 pointwise** does `IC` MACs per input element, so intensity grows with
  channel count: layer 2 (32→16 at 112×112) is at 20.6 ops/byte and 120.8
  GOPS, while layer 51 (320→1280 at 7×7) is at **74.5 ops/byte and 315.9 GOPS,
  61.7% of attainable** — the best layer in the network. Late layers with many
  channels and few pixels are exactly what this machine is built for. The
  3×3 stem (layer 0) sits with the memory-bound group at 30.6 ops/byte and
  110.6 GOPS: three input channels is not enough reuse.
* The **classifier** reads 1.28 MB of weights to do 1.28 M MACs — one weight
  byte per two ops, hence 1.95 ops/byte, an attainable ceiling of 31.2 GOPS,
  and 3.56% MAC utilisation. It reaches 58% of that ceiling, which is a *good*
  result for a layer that is pure bandwidth. This is the clearest case in the
  network of a layer that no scheduler can help — and §6 measures exactly that:
  1.004×.

One measurement note: per-layer TRACE regions in the unscheduled feature
extractor sum to 3,928,818 cycles against a 3,727,589-cycle program — a **5.4%
overlap**. That is real: consecutive layers genuinely overlap. Region cycles
must not be summed and presented as a total.

---

## 6. `--kea-schedule` A/B

`demo/results/ab_schedule.json`. Everything at the defaults.

### 6.1 The whole network, which is now measurable

Defect 3 used to make this impossible; the entire 52-convolution feature
extractor now schedules.

| nodes 0..178, 52 convolutions | cycles | instructions | MXU busy | DMA0 | DMA1 |
|---|---|---|---|---|---|
| no `--schedule` | 3,727,589 | 30,498 | 32.63% | 48.13% | **0%** |
| `--schedule` (`mode=auto`) | **3,176,061** | 30,430 | **38.30%** | 38.23% | **27.71%** |
| `-kea-schedule mode=serial` | 4,063,835 | 29,128 | 29.93% | 44.15% | 0% |
| `-kea-schedule mode=overlap` | 3,176,061 | 30,430 | — | — | — |

* **`--schedule` vs not scheduling: 1.174×** on the feature extractor,
  **1.166×** on the whole compiled network (3,868,452 → 3,316,341 cycles).
* `mode=auto` produced a bit-identical program to `mode=overlap`, i.e. on this
  network it costed the overlapped plan as cheaper and took it.
* **`mode=serial` is not the same program as "no `--schedule`"** and is 9%
  *slower* than it (4,063,835 vs 3,727,589). Without the pass, `kea-translate`
  inserts the synchronization itself, and on this network it does a better job
  than the scheduler's deliberately-sequential control. `mode=auto`'s "cannot
  lose" property is relative to `mode=serial`, which is the weaker baseline —
  §6.2 finds a configuration where auto does lose to the real one.

The classifier gains 1.004× (140,863 → 140,280), as predicted in §5.2.

### 6.2 The `imem-budget` curve is narrow and not monotonic

The budget picks the tiling and the tiling picks the cycle count, so this knob
is worth sweeping. It is reachable as `keac --imem-budget`.

| `imem-budget` | unscheduled instrs / cycles | scheduled instrs / cycles | speedup |
|---|---|---|---|
| 19,066 | 26,104 / 3,713,412 | 25,909 / 3,947,847 | **0.941×** |
| 19,200 | 26,468 / 3,765,330 | 26,336 / 4,002,032 | **0.941×** |
| 19,500 | 27,455 / 3,692,535 | 25,904 / 3,682,850 | 1.003× |
| 19,750 | 28,239 / 3,704,708 | 26,552 / 3,697,401 | 1.002× |
| 20,000 | 28,703 / 3,700,071 | 28,634 / 3,133,342 | 1.181× |
| **20,200** | 29,567 / 3,703,839 | 29,498 / **3,130,202** | 1.183× |
| **20,480 (default)** | 30,498 / 3,727,589 | 30,430 / 3,176,061 | 1.174× |
| 20,750 | 30,498 / 3,727,589 | 30,430 / 3,176,061 | 1.174× (same plan) |
| 21,000 | 31,718 / 3,732,977 | 31,648 / 3,179,147 | 1.174× |
| 21,250 | 32,312 / 3,736,157 | 32,242 / 3,176,563 | 1.176× |
| 21,500 | — over IMEM — | — over IMEM — | — |

Three things fall out of that, all measured:

1. **The optimum is not the default.** `imem-budget = 20,200` gives
   **3,130,202 cycles, 1.44% under the default's 3,176,061**, with 932 fewer
   instructions. The default is a good choice, not the best one. (This confirms
   an earlier independent measurement that `20,000` beat `20,480`; sweeping
   finer finds 20,200 slightly better still.)
2. **The curve is not monotonic in either direction.** Scheduled cycles go
   3.95 M → 4.00 M → 3.68 M → 3.70 M → 3.13 M → 3.13 M → 3.18 M as the budget
   rises. A bigger instruction budget buys finer tiles, finer tiles cut DRAM
   traffic *and* lengthen the program, and the two do not trade off smoothly.
   The cost model is not wrong so much as it is optimising a proxy.
3. **At the two tightest budgets the scheduler makes things worse** — 0.941×,
   a 6% regression against not scheduling at all. At those budgets the tiles are
   coarse enough that there is little to overlap, and the scheduler's extra
   SIGNAL/WAIT traffic is not paid for. `mode=auto` does not catch this because
   it costs against `mode=serial`, and it does beat `mode=serial` there
   (3,947,847 vs 4,019,811 at 19,066). **The auto-mode guarantee is real but it
   is guarding the wrong baseline.**

### 6.3 Per layer

53 layers compiled and run twice each, standalone, **0 failures**:

| | unscheduled | scheduled | speedup |
|---|---|---|---|
| **all 53 layers, summed** | **3,554,122** | **2,513,148** | **1.414×** |
| 17 depthwise (DWU) | 879,083 | 512,182 | **1.716×** |
| 35 conv (MXU) | 2,534,176 | 1,860,686 | **1.362×** |
| 1 classifier `fully_connected` | 140,863 | 140,280 | 1.004× |

Distribution: **43 layers gain more than 10%**, 5 land within ±2%, and **2
regress** — `conv2d#67` at 0.998× (15,355 → 15,381) and `conv2d#143` at 0.999×
(28,369 → 28,406), both under 40 cycles. Best: `depthwise_conv2d#20` at
**1.983×** (135,685 → 68,431).

The per-layer aggregate (1.414×) is **higher** than the whole-network number
(1.174×), and the gap is the honest part of this measurement: each layer is its
own program here, so every one pays a cold start with nothing to overlap it,
which flatters the scheduler. The whole-network figure is the one that ships.
Depthwise layers gain most, which follows from §5.2 — they are memory bound
with one idle DMA engine, so putting loads and stores on both engines is
directly worth cycles.

### 6.4 The mechanism, in the assembly

Not inferred — read off the text. The unscheduled stream issues every DMA on
**DMA0**, never touches DMA1, and puts a `WAIT` between each producer and
consumer. The scheduled stream spreads loads and stores across both engines,
hoists the next tile's `DMA_LD` between the current tile's two `MATMUL`s, and
alternates `LOAD_W` between weight banks so a weight load never waits for the
array. `demo/results/schedule_excerpt.kasm` is the exact text; the same layer
takes 102,764 cycles unscheduled and 53,114 scheduled — **1.935×**.

---

## 7. What did not work

Listed so it is on the record, not buried.

1. **MobileNetV2 does not compile as one program**, for two independent
   reasons, and the second one is not fixable in software: a scale-changing
   pool has no lowering (§2.1.1), *and* the graph is 36,633 instructions at the
   coarsest tiling against a 32,768-entry IMEM (§2.1.2).
2. **The global average pool does not run on the NPU.** It is executed on the
   host by `frontend/kea_frontend/reference.py`'s own kernel. `kea.pool` itself
   works now; it is the trailing rescale that does not. Every alternative
   spelling was tried and each is blocked by something real:
   * a 7×7 `tosa.depthwise_conv2d` of ones → `'kea.dwconv2d' op DWCONV
     supports square 3x3 or 5x5 kernels only, got 7x7` (a correct, documented
     ISA limit);
   * `tosa.matmul(ones[1,1,49], x[1,49,1280])` → refused, correctly: the MXU is
     weight stationary and cannot take an activation right-hand side;
   * `tosa.transpose` + `matmul(x^T, ones)` → `'kea.transpose' op has no
     Level 2 lowering` (correct: KEA-1 has no transpose unit);
   * a 1280×1280 conv2d over the reshaped tensor → 80 MB of weights, absurd.
3. **Per-layer cycles cannot be measured on a scheduled build** (§3.1), so the
   per-layer roofline is the unscheduled one. This is a live compiler bug, not
   a design limit.
4. **The tiny ViT was not attempted end to end.** Its 12 activation×activation
   matmuls are outside what a weight-stationary MXU can do — the diagnostic is
   now clear about that, which does not make it compile — and its `softmax` and
   `layernorm` have no TOSA spelling at all (`docs/FRONTEND.md` §5 says so; the
   emitter refuses them by name rather than approximating them). Compiling it
   needs `kea.softmax` and `kea.layernorm` custom ops plus a matmul path that
   streams both operands.
5. **`tosa.avg_pool2d` with an inexact folded scale is not bit-exact** (§1.1).
   Unexercised here because the only such node in this model does not compile,
   but it is a latent correctness gap for any model whose pooling changes
   scale, and the emitter says so in the file it writes.
6. **`docs/SCHEDULING.md` is stale** in three places — §5.3 (the Rule D
   violation, since fixed), §8.1.1 and §10 (both still describe the demo as
   pinned to `--spm-reserve 1` and the IMEM ceiling as "what is costing the
   whole-network speedup"). All three were fixed in the compiler; the
   whole-network A/B in §6.1 supersedes those paragraphs. Only the dangling
   `demo/repro/` paths were corrected here — the prose needs a pass from
   whoever owns that document.

---

## 8. Reproducing a single number

```sh
# instruction count and IMEM fit, at the defaults
build/native/bin/keac demo/build/mobilenetv2_features.tosa.mlir \
    --function mnv2_features -o /tmp/f.keaf --schedule --keep-intermediates
grep -cE '^\s+(MXU|DWU|VPU|DMA0|DMA1|CTRL)\s' /tmp/f.kasm     # 30430

# the whole graph does not fit, even at the coarsest tiling
build/native/bin/keac demo/build/mobilenetv2_full.tosa.mlir \
    --function mobilenetv2 -o /tmp/w.keaf --imem-budget 29241  # kea.rescale

# whole-program roofline
build/native/bin/kea-sim demo/build/features.keaf --stats-json /tmp/s.json
.venv/bin/python -c "import json;print(json.load(open('/tmp/s.json'))['global']['roofline'])"

# the scheduler A/B, one number
build/native/bin/kea-sim demo/build/features_unsched.keaf --quiet --stats-json /tmp/u.json
.venv/bin/python -c "import json;print(json.load(open('/tmp/u.json'))['total_cycles'] \
    / json.load(open('/tmp/s.json'))['total_cycles'])"          # 1.1737

# bit-exactness on the golden vectors
.venv/bin/python demo/validate_mobilenetv2.py

# the four fixed defects, still fixed
bash demo/regress/run_regressions.sh
```
