# End-to-end results: MobileNetV2 int8 on KEA-1

**What this document is.** The measured outcome of running a real quantized
network through the whole stack — `.kgraph.json` → TOSA MLIR → `kea-opt` →
`kea-translate` → `kea-as` → `kea-sim` — with the numerical validation, the
roofline analysis, the scheduler A/B, and every place the stack did not work.

**Every number here came out of a command in this repository, at the
compiler's defaults.** Nothing is pinned -- the demo passes no tuning flag
anywhere except where a measurement is explicitly about a flag. Reproduce all
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
| Suites | compiler lit 35/35, native ctest 17/17, frontend pytest 92 passed, `numeric_check` 7/7, `demo/regress` 11/11 |
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

### 1.1 The one place the emitter is not exact, and what it now costs

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

This used to be unexercised, because the pool did not compile. It compiles now,
so the substitution is on the critical path and its cost is **measured** in §4:
running the pool on the NPU puts 1,660 of 5,120 pooled values and 455 of 4,000
logits off by exactly ±1, with the argmax unchanged on all four vectors. That
is the price of the split, and it is a *frontend* limitation — the backend
reproduces what it was given exactly.

---

## 2. What compiled, and what ran

`demo/results/compile.json`. All at the defaults, no flags.

| attempt | nodes | result | instructions |
|---|---|---|---|
| whole graph, one function, defaults | 183 | **failed** — coarsest tiling needs **29,241** instructions against the default budget | — |
| whole graph, `--imem-budget 29241` | 183 | **failed** — **37,295** instructions, IMEM holds 32,768 | 37,295 |
| whole graph, `--imem-budget 32768` | 183 | **failed** — `-kea-tile` itself refuses: "lowers to 32,786 KEA-1 instructions" | — |
| feature extractor (0..178), no `--schedule` | 179 | **ok** | 28,730 |
| feature extractor, `--schedule` | 179 | **ok** | **28,675** |
| feature extractor **+ head pool** (0..179), no `--schedule` | 180 | **ok** | 29,398 |
| feature extractor **+ head pool**, `--schedule` | 180 | **ok** | **29,346** |
| classifier (180..182), no `--schedule` | 3 | **ok** | 10,833 |
| classifier, `--schedule` | 3 | **ok** | **10,753** |
| head pool alone, as the model spells it (`avg_pool2d` + `rescale`) | 1 | **ok** | 666 |
| a bare 7×7 `tosa.avg_pool2d`, no scale change | 1 | **ok**, runs in **460 cycles**, VPU 42.6% | 10 |

**All 183 of 183 nodes compile and run on the NPU.** Every convolution, the
residual adds, the head pool and the classifier. Nothing runs on the host any
more — which is new, and is the single biggest change in this report.

### 2.1 It is still two programs, and that is a capacity result

MobileNetV2 does not fit in one KEA-1 program, and the reason is no longer a
missing lowering — every op in the graph lowers. It is size:

* At the default budget `-kea-tile` refuses up front: **the coarsest tiling of
  every layer already needs 29,241 instructions**, more than the budget it is
  allowed to plan against.
* Given that budget explicitly, it tiles — and the finished program is
  **37,295 KEA-1 instructions against a 32,768-entry IMEM, 13.8% over**.
  `kea-as` rejects it.
* Pushed to `--imem-budget 32768`, `-kea-tile` refuses before `kea-as` gets a
  chance: "lowers to 32,786 KEA-1 instructions, more than IMEM holds (32768),
  and `-kea-schedule` still has to add its SIGNAL/WAIT pairs".

That is three independent statements of the same fact. KEA-1 is branchless and
IMEM is not paged, so program size is a hard capacity limit, and this network
is over it by about 4,500 instructions at the *most* favourable tiling
available. No flag fixes that.

### 2.2 Two ways to split it, and they are not equivalent

Because the pool compiles now, the cut can go on either side of it:

```
bit-exact                                       all on the NPU
  features.keaf     nodes 0..178                  featpool.keaf   nodes 0..179
    -> head (1,7,7,1280) i8                         -> gap (1,1,1,1280) i8
  global_avg_pool   ON THE HOST                   classifier.keaf nodes 180..182
  classifier.keaf   nodes 180..182
     39,428 instructions, bit-exact                 40,099 instructions, ±1
```

The all-NPU split needs no host step and moves 1,280 bytes between the two
programs instead of 62,720. It is **not** bit-exact, for the frontend reason in
§1.1. Both are built, both are validated, and §4 reports both. The demo ships
the bit-exact one as the primary artifact because this repository's claim is
exactness; the all-NPU one is the more interesting engineering result.

### 2.3 IMEM: the real capacity story

`KEA_MAX_INSTRUCTIONS` is 32,768 and there is no loop instruction, so the whole
network is one straight line of instructions. `-kea-tile` is IMEM-aware: each
layer publishes a Pareto frontier of (instructions, cycles), and a Lagrangian
price is bisected for the cheapest plan that fits `imem-budget`. The budget is
on the *tiled* program; `-kea-schedule` and `kea-translate` then add their
SIGNAL/WAIT pairs, which is what the headroom below 32,768 is for.

Measured on this network, the usable window is **narrow at both ends**:

| | |
|---|---|
| smallest budget `-kea-tile` accepts | **19,066** — its own coarsest tiling of all 52 layers |
| largest budget that still assembles | between **21,000** and **21,500** |
| at the default | 28,675 scheduled instructions, **87.5% of IMEM** |

The whole feasible window is about −6%/+4% around the default. `spm-reserve-factor`
defaults to 1 and the demo does not touch it: the tiling cost model prices
double-buffering directly, so the old divisor is redundant.

---

## 3. Bugs found and fixed during bring-up

Bringing this network up found **six** backend defects. All six are fixed, and
`bash demo/regress/run_regressions.sh` asserts the *fixed* behaviour — 11
assertions, exit non-zero on any regression. That script replaces the old
`demo/repro/run_repro.sh`, which asserted the opposite.

| | what was wrong | how it showed up | now |
|---|---|---|---|
| 1 | `-kea-tile`'s `lowerPool` gave the pool's SPM_A tile the same name as the DRAM buffer holding its result; `scratch()`'s uniquifier only disambiguated against other scratch names | `error: 'kea.alloc' op duplicate buffer name "gap_pool.0.out"` — **every** `kea.pool`, in every program | uniquification moved into `makeBuffer()`. A bare 7×7 pool runs in **460 cycles** at 42.6% VPU |
| 2 | `-kea-tile` assumed a `kea.matmul`'s second operand was a compile-time constant and built a `kea.alloc` with a null source when it was not | `error: null operand found` / `%3 = "kea.alloc"(<<NULL VALUE>>)` — pointing at nothing actionable | a real diagnostic naming the op and the weight-stationary reason. The limitation itself stands |
| 3 | rotation edges added *after* the dependence graph made a DMA a direct consumer of a `LOAD_W`, so a `SIGNAL` collapsed and two waits shared one token | `error: 'kea.wait' op violates Rule D (ISA.md §5.5)` on every lowerable prefix past node 97 | the whole feature extractor schedules, and validates bit-exactly |
| 4 | tiling was greedy per layer, so program size — a *global* limit on a branchless machine — was nobody's constraint | 41,409 instructions against a 32,768 IMEM; the demo was pinned to `--spm-reserve 1` | `-kea-tile` is IMEM-aware (§2.3). Fits at the defaults with no flags |
| 5 | a standalone `kea.rescale` had no Level 2 lowering, so `-kea-fuse` had to eliminate every one — and it only folds a rescale into a *contraction* | `error: 'kea.rescale' op has no Level 2 lowering` on MobileNetV2's head pool, which changes activation scale | lowered through a 16×16 identity matmul, because `VQUANT` can only read ACC. **This is what puts the pool on the NPU** |
| 6 | `-kea-schedule` placed a region marker before the earliest stream position of anything in its region, so it followed whatever floated furthest — and a residual layer's 20-byte `addparams` DMA depends on nothing | 10 of 52 TRACE regions opened at cycle ~0; the 52 regions summed to **7.43×** the program length | markers key on the region's own queue, and a `region-lookahead` bound stops an instruction being emitted more than one region ahead |

Three of the six were *diagnosis* or *instrumentation* bugs rather than
capability bugs: defect 1 made a working code path unreachable, defect 2 hid a
legitimate refusal behind a verifier crash, and defect 6 corrupted the
measurement without touching the program. The compiler could do more than it
appeared to, and small reproducers were what showed that.

### 3.1 Verifying defect 6 independently

Defect 6 is the one that invalidated a whole section of an earlier revision of
this document, so it is worth saying how it was re-checked rather than taking
the fix on trust. `demo/common.py`'s `unsound_regions()` flags any region that
opens **materially** before a lower-tagged one — materially meaning more than 1%
of the program length, because two adjacent regions on different queues can
legitimately overlap by a pipelining lip of a few tens of cycles. That
tolerance is not a fudge: the bug measured 3.3% of program length and seven
whole regions spanned, so 1% separates the two cases by a factor of three.

`demo/roofline.py` runs it over **every** build it touches and aborts rather
than reporting a table it cannot vouch for, with a second, independent gate on
the aggregate (Σ region cycles / total > 1.5, or nesting depth > 2). Measured
now:

| build | regions | unsound | max depth | Σ regions / total |
|---|---|---|---|---|
| features, scheduled | 52 | **0** | 1 | **1.072** |
| features, unscheduled | 52 | 0 | 1 | 1.070 |
| featpool, scheduled | 54 | 0 | 1 | 1.075 |
| featpool, unscheduled | 54 | 0 | 1 | 1.073 |
| *(features, scheduled, before the fix)* | 52 | *10* | *11* | *7.43* |

The scheduled build is now as sound as the unscheduled one, so **the per-layer
roofline in §5.2 comes from the scheduled build** — the shipped artifact — and
the workaround that took it from the unscheduled build is gone.

One thing the strict version of the check did flag, correctly, and which is
*not* a bug: in `featpool` the rescale layer's region opens ~2,700 cycles before
the pool layer's, because the rescale's first MXU instruction is an identity
`LOAD_W` that depends on nothing. That is real overlap between two adjacent
layers on different queues, it appears in the unscheduled build too, and it is
0.09% of the program rather than 3.3%.

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

## 4. Numerical validation

`demo/results/validation.json`. `frontend/testdata/golden_io_mobilenetv2_int8.npz`
holds four fixed int8 inputs and the exact int8 outputs of the numpy reference
interpreter. `docs/FRONTEND.md` §4.1: *"assert exact integer equality. There is
no tolerance."*

**Every program under test is the scheduled build at the defaults**, so this
validates `-kea-schedule`'s reordered, double-buffered, two-DMA-engine output —
not just the tiler's sequential one.

### 4.1 The bit-exact split — **zero mismatches**

`features.keaf` → host `global_avg_pool` → `classifier.keaf`.

| comparison | elements per vector | mismatches (4 vectors) |
|---|---|---|
| `features.keaf` output vs the reference's `head` | 62,720 | **0** |
| host pool vs the reference's `gap` | 1,280 | **0** |
| `classifier.keaf` fed the reference `gap`, vs the golden output | 1,000 | **0** |
| **end to end** (sim → host pool → sim) vs the golden output | 1,000 | **0** |

Across the four vectors that is **258,880 simulator-produced int8 values**
compared against the reference, plus 5,120 host-pool values. **Zero mismatches
anywhere.** argmax agreement 4/4.

### 4.2 The all-NPU split — ±1, and the argmax holds

`featpool.keaf` → `classifier.keaf`, nothing on the host.

| comparison | elements (4 vectors) | mismatches | max abs diff |
|---|---|---|---|
| `featpool.keaf`'s `gap` vs the reference's | 5,120 | **1,660** (32.4%) | **1** |
| end to end vs the golden output | 4,000 | **455** (11.4%) | **1** |
| argmax | 4 | — | **4/4 agree** |

Every difference is exactly one quantization step, and no prediction changes.
The cause is entirely in §1.1: the frontend spells a scale-changing pool as
`avg_pool2d` + `rescale`, which rounds twice where the reference rounds once.
The backend is not approximating anything — feed it the reference's own `gap`
and the classifier is bit-exact (row 3 of §4.1).

This is the honest trade: **you can have all 183 nodes on the NPU, or you can
have bit-exactness, and today you cannot have both.** Closing it needs the
frontend to be able to express a pool with a folded requantization — i.e. a
`kea.pool` whose `quant` is not restricted to a zero-point rebase, which is a
Level 1 dialect change, not a backend one.

### 4.3 Both pass the simulator's strict gates

```sh
kea-sim demo/build/features.keaf   --quiet --strict-poison --strict-hazards   # exit 0
kea-sim demo/build/featpool.keaf   --quiet --strict-poison --strict-hazards   # exit 0
kea-sim demo/build/classifier.keaf --quiet --strict-poison --strict-hazards   # exit 0
```

i.e. **no read of never-written scratchpad and no cross-unit read of data whose
producer had not retired** — no missing `SIGNAL`/`WAIT` anywhere in 40,099
scheduled instructions. `diagnostics` in every stats file is empty. That is the
strongest single statement in this report: the scheduler moved instructions
across queues, split DMA over two engines and pipelined tiles, and on the
bit-exact split the answer did not change by one bit.

That the requantization matches at all is the payoff of
[ADR-0003](adr/0003-requantization-equivalence-invariant.md): `VQUANT` runs
gemmlowp-style `keaRequantize` while the golden model runs TOSA
`apply_scale_32`, two different functions that agree only on the normalised
domain. **300,774,272 MACs' worth of results agree exactly** across this model.
(An earlier revision reported that ADR-0003's quoted rescale shift range was off
by one — 21, not 22, because of `rescale#134`. That correction has been applied
to the ADR.)

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

The headline is the **all-NPU** split — every one of the 183 nodes on the
machine. The bit-exact split is within 0.4% of it on every figure and is given
underneath.

| | featpool (0..179) | classifier | **whole network on the NPU** |
|---|---|---|---|
| cycles, scheduled | 3,142,636 | 87,714 | **3,230,350** (3.230 ms @ 1 GHz) |
| cycles, unscheduled | 3,711,432 | 143,425 | 3,854,857 (3.855 ms) |
| useful ops | 599,029,504 | 2,560,000 | **601,589,504** |
| issued ops | 642,381,568 | 2,580,480 | 644,962,048 |
| padding efficiency | 93.25% | 99.21% | **93.28%** |
| DRAM bytes | 21,336,204 | 1,330,216 | **22,666,420** (16.0 MB in / 6.7 MB out) |
| arithmetic intensity | 28.08 ops/B | **1.92 ops/B** | **26.54 ops/B** |
| achieved (scheduled) | 190.6 GOPS | 29.2 GOPS | **186.2 GOPS** |
| attainable | 449.2 GOPS | 30.8 GOPS | **424.7 GOPS** |
| % of attainable | 42.4% | **94.8%** | **43.9%** |
| bound | MEMORY | MEMORY | **MEMORY** |
| MXU MAC utilisation | 34.65% | 5.70% | **33.87%** of the 256 int8 MAC/cycle peak |
| DWU MAC utilisation | 5.15% | — | of the 128 MAC/cycle peak |
| VPU utilisation | 38.96% | — | of 16 elem/cycle |
| achieved DRAM bandwidth | 6.79 GB/s | **15.17 GB/s** | 7.02 GB/s of 16 GB/s peak |

The bit-exact split (`features` + host pool + `classifier`): **3,217,103
cycles**, 26.64 ops/byte, **187.0 GOPS**, 43.9% of attainable, 22,584,244 DRAM
bytes, MXU MAC utilisation 34.01%. Putting the pool on the NPU costs **13,247
cycles, 0.41%** — which is the real answer to "what does the pool cost", and it
is small because a 62,720-element reduction is 0.01% of the network's
arithmetic.

300.8 M MACs matches MobileNetV2's published ~300 M multiply-accumulates, which
is a useful independent check that the compiler is not silently skipping work.

**The network is memory bound at 26.54 against a ridge point of 32.0
ops/byte**, and it reaches **43.9% of what that intensity allows**. The
classifier is the standout: it runs at **94.8% of attainable and 15.17 GB/s of
a 16 GB/s port**, which is as close to saturating the DRAM interface as
anything in this repository gets.

Where the 3.13 M cycles go in the scheduled feature extractor, against the
unscheduled build:

| unit | busy (sched) | busy (unsched) | semaphore stall | resource stall | idle |
|---|---|---|---|---|---|
| MXU | **38.85%** | 32.89% | 43.0% | 0.08% | 18.1% |
| DWU | 5.23% | 4.43% | 20.0% | 0.03% | 74.7% |
| VPU | 26.63% | 22.55% | 54.0% | 0.10% | 19.2% |
| DMA0 | 37.89% | 47.83% | 38.2% | 0.05% | 23.9% |
| DMA1 | **27.36%** | **0%** | 36.3% | 0.03% | 36.3% |

(stall/idle columns are the scheduled build.) The dispatcher is stalled 99.03%
of cycles. **DMA1 goes from completely idle to 27.4% busy** — allocating the
second engine is the largest single thing `--kea-schedule` does.

### 5.2 Per layer

Full table in `demo/results/roofline_layers.csv` (53 rows), **from the
scheduled build**, licensed by the region-soundness check in §3.1. The CSV also
carries an `unscheduled_cycles` column, so the per-layer A/B and the roofline
position are in one table. Grouped:

| family | layers | intensity | achieved | share of ops | share of layer-cycles | share of DRAM |
|---|---|---|---|---|---|---|
| conv (MXU) | 35 | 16.5 – 92.0 ops/B | 118 – 316 GOPS | 90.7% | 74.5% | 62.3% |
| 3×3 depthwise (DWU) | 17 | **3.55 – 10.04 ops/B** | 38 – 101 GOPS | **8.9%** | **23.0%** | **32.3%** |
| classifier `fully_connected` | 1 | **1.92 ops/B** | 29.2 GOPS | 0.4% | 2.5% | 5.3% |

(shares are of the per-layer totals: 625,271,936 ops, 3,443,064 layer-cycles,
24,962,316 DRAM bytes — see the overlap note at the end of this section.)

**31 of 53 layers are memory bound** — all 17 depthwise layers, the classifier,
and 13 convolutions (layers 0, 2, 3, 5, 6, 8, 9, 11, 12, 15, 18, 20, 41),
concentrated in the high-resolution front half where activations dwarf weights.

Why the families look so different, concretely:

* A **3×3 depthwise** does exactly 9 MACs per input element, whatever the
  channel count. Intensity is pinned by the kernel: with one i8 read and one i8
  write per element it can never exceed ~9 ops/byte, and the measured range is
  3.55–10.04 (the stride-2 layers are lower because they read 4× the pixels
  they write). Every one lands under the ridge. They do **8.9%** of the
  network's arithmetic, occupy **23.0%** of its layer-cycles and move **32.3%**
  of its DRAM traffic. The DWU is idle 75% of the time not because it is slow
  but because it is starved.
* A **1×1 pointwise** does `IC` MACs per input element, so intensity grows with
  channel count: layer 2 (32→16 at 112×112) is at 20.1 ops/byte and 184 GOPS,
  while layer 51 (320→1280 at 7×7) is at **92.0 ops/byte and 316.4 GOPS** — the
  fastest layer in the network. Late layers with many channels and few pixels
  are exactly what this machine is built for. The 3×3 stem (layer 0) sits with
  the memory-bound group at 30.7 ops/byte: three input channels is not enough
  reuse.
* The **classifier** reads 1.28 MB of weights to do 1.28 M MACs — one weight
  byte per two ops, hence 1.92 ops/byte and an attainable ceiling of 30.8 GOPS.
  It reaches **94.8%** of that ceiling. It is the least *efficient* layer by MAC
  utilisation (5.70%) and the most efficient by the only measure that applies
  to it. This is the clearest case in the network of a layer whose limit is the
  DRAM port and nothing else.

One measurement note: per-layer TRACE regions sum to 3,355,350 cycles against a
3,129,389-cycle program — a **7.2% overlap**. That is real: consecutive layers
genuinely overlap, and more so in a scheduled build than an unscheduled one
(7.2% against 7.0%). Region cycles must not be summed and presented as a total.

---

## 6. `--kea-schedule` A/B

`demo/results/ab_schedule.json`. Everything at the defaults.

### 6.1 The whole network

| nodes 0..178, 52 convolutions | cycles | instructions | MXU busy | DMA0 | DMA1 |
|---|---|---|---|---|---|
| no `--schedule` | 3,696,622 | 28,730 | 32.89% | 47.83% | **0%** |
| `--schedule` (`mode=auto`) | **3,129,389** | 28,675 | **38.85%** | 37.89% | **27.36%** |
| `-kea-schedule mode=serial` | 4,021,417 | 27,523 | 30.23% | 43.97% | 0% |
| `-kea-schedule mode=overlap` | 3,129,389 | 28,675 | — | — | — |

* **`--schedule` vs not scheduling: 1.181×** on the feature extractor,
  **1.193×** on the whole network (3,854,857 → 3,230,350 cycles).
* The classifier gains **1.635×** (143,425 → 87,714) — up from 1.004× before
  the tiler learned to price DRAM. It is now the layer that gains *most* after
  the depthwise ones, which inverts an earlier finding in this document.
* `mode=auto` produced a bit-identical program to `mode=overlap`, i.e. on this
  network it costed the overlapped plan as cheaper and took it.
* `mode=serial` remains 8.8% *slower* than not scheduling at all — without the
  pass, `kea-translate` inserts the synchronization itself, and it does that
  better than the scheduler's deliberately-sequential control. That is a
  property of the control, not a defect.

### 6.2 The `imem-budget` curve, and `mode=auto`'s new floor

The budget picks the tiling and the tiling picks the cycle count. It is
reachable as `keac --imem-budget`.

| `imem-budget` | unscheduled instrs / cycles | scheduled instrs / cycles | speedup | |
|---|---|---|---|---|
| 19,000 | — below the tiler's floor — | — | — | |
| 19,066 | 26,104 / 3,713,412 | 26,104 / 3,713,412 | 1.000× | identical `.kasm` |
| 19,200 | 26,445 / 3,738,034 | 26,445 / 3,738,034 | 1.000× | identical `.kasm` |
| 19,500 | 27,412 / 3,665,429 | 27,412 / 3,665,429 | 1.000× | identical `.kasm` |
| 19,750 | 28,324 / 3,674,256 | 28,324 / 3,674,256 | 1.000× | identical `.kasm` |
| 20,000 | 28,730 / 3,696,622 | 28,675 / **3,129,389** | **1.181×** | |
| **default** | 28,730 / 3,696,622 | 28,675 / **3,129,389** | **1.181×** | |
| 20,480 | 30,498 / 3,727,589 | 30,430 / 3,176,072 | 1.174× | |
| 20,750 | 30,498 / 3,727,589 | 30,430 / 3,176,072 | 1.174× | |
| 21,000 | 31,718 / 3,732,977 | 31,648 / 3,179,158 | 1.174× | |
| 21,500 and up | — over IMEM — | — | — | |

Three things fall out, all measured:

1. **The default is the optimum.** Nothing in the feasible window beats
   3,129,389 cycles, and the default reaches it. An earlier revision of this
   document found the default 1.44% off the best; the tiler's cost model was
   retuned and that gap is now zero.
2. **`mode=auto` cannot lose any more.** Below 20,000 it declines to reorder
   and emits output **byte-identical** to omitting `--schedule` — verified with
   `filecmp`, not inferred from equal cycle counts, because equal cycles are
   consistent with a different program. The 6% regressions an earlier revision
   measured at these budgets are now exactly 1.000×.
3. **The "cliff" is `mode=auto` switching on, and the measurement says so
   plainly.** Scheduled cycles go 3.71 M → 3.74 M → 3.67 M → 3.67 M → **3.13 M**
   as the budget rises: a step, not a slope, between 19,750 and 20,000. That
   step is exactly where `mode=auto` stops declining. Per-unit duty across the
   sweep, all from `kea-sim`:

   | budget | cycles | DMA0 busy | DMA1 busy | DRAM port | MXU semaphore stall |
   |---|---|---|---|---|---|
   | 19,066 | 3,713,412 | 48.25% | **0%** | 34.85% | 46.36% |
   | 19,500 | 3,665,429 | 47.40% | **0%** | 35.52% | 45.47% |
   | 19,750 | 3,674,256 | 47.59% | **0%** | 35.69% | 45.10% |
   | 20,000 | **3,129,389** | 37.89% | **27.36%** | 42.45% | 43.01% |
   | 20,480 | 3,176,072 | 38.23% | 27.71% | 42.49% | 41.79% |
   | 21,000 | 3,179,158 | 38.16% | 27.76% | 42.45% | 41.87% |

   Below the step DMA1 is **completely idle**, because the emitted program *is*
   the unscheduled one. Above it, both engines are working. The budget is
   really a knob on **how many tiles there are**: below 20,000 the tiler
   produces too few and too large tiles for the scheduler's model to predict a
   5% win, so it correctly declines, and the machine runs on one DMA engine.

   Worth recording because two plausible explanations were wrong. It is **not**
   the tiler being blind to DRAM traffic — `-kea-tile` now prices DRAM
   explicitly and the chosen plan did not change, and the port never exceeds
   **42.5% of 16 GB/s** anywhere in the window, so bandwidth never binds. Nor
   is it MXU semaphore stall, which moves monotonically (46.4% → 41.8%) across
   a cycle count that does not. The residual non-monotonicity *above* the step
   — 20,000 is 1.47% faster than 20,480 on 1,755 fewer instructions — is
   small and unexplained.

### 6.3 Per layer

53 layers compiled and run twice each, standalone, **0 failures**:

| | unscheduled | scheduled | speedup |
|---|---|---|---|
| **all 53 layers, summed** | **3,545,222** | **2,390,852** | **1.483×** |
| 17 depthwise (DWU) | 879,083 | 512,165 | **1.716×** |
| 35 conv (MXU) | 2,522,714 | 1,790,973 | **1.409×** |
| 1 classifier `fully_connected` | 143,425 | 87,714 | **1.635×** |

Best: `depthwise_conv2d#20` at **1.983×** (135,685 → 68,431). **One** layer
regresses, `conv2d#143` at 0.9987× — 36 cycles on 28,369.

The per-layer aggregate (1.483×) is **higher** than the whole-network figure
(1.181×), and the gap is the honest part of this measurement: each layer is its
own program here, so every one pays a cold start with nothing to overlap it,
which flatters the scheduler. The whole-network figure is the one that ships.

### 6.4 The mechanism, in the assembly

Not inferred — read off the text. The unscheduled stream issues every DMA on
**DMA0**, never touches DMA1, and puts a `WAIT` between each producer and
consumer. The scheduled stream spreads loads and stores across both engines,
hoists the next tile's `DMA_LD` between the current tile's two `MATMUL`s, and
alternates `LOAD_W` between weight banks so a weight load never waits for the
array. `demo/results/schedule_excerpt.kasm` is the exact text.

---

## 7. What did not work

Listed so it is on the record, not buried.

1. **MobileNetV2 does not fit in one program**, and no flag changes that: at
   the coarsest tiling available it is **37,295 instructions against a
   32,768-entry IMEM** (§2.1). Every op in it lowers; it is purely size. A
   loop instruction, or paged IMEM, is what this would need.
2. **All-NPU and bit-exact are mutually exclusive today** (§4.2). Running the
   head pool on the machine puts 11.4% of the logits off by ±1, because the
   frontend can only spell a scale-changing pool as `avg_pool2d` + `rescale`.
   The fix is a Level 1 dialect change — a `kea.pool` whose `quant` is not
   restricted to a zero-point rebase — not a backend one.
3. **The tiny ViT was not attempted end to end.** Its 12 activation×activation
   matmuls are outside what a weight-stationary MXU can do — the diagnostic is
   clear about that, which does not make it compile — and its `softmax` and
   `layernorm` have no TOSA spelling at all (`docs/FRONTEND.md` §5; the emitter
   refuses them by name rather than approximating them). Compiling it needs
   `kea.softmax` and `kea.layernorm` custom ops plus a matmul path that streams
   both operands.
4. **`kea-sim` cycle counts are a lower bound.** Scratchpad port arbitration,
   DRAM row buffers, refresh and turnaround, and misalignment costs are all
   unmodelled (`docs/SIMULATOR.md` §2). Real LPDDR would cost 10–30% more on
   scattered access, and the classifier — which is at 94.8% of attainable and
   15.17 GB/s of a 16 GB/s port — is exactly the layer where that would show
   first.
5. **The `imem-budget` cliff is understood but not smoothed** (§6.2). Below
   20,000 the tiler produces too few, too large tiles for the scheduler to
   overlap, and `mode=auto` correctly declines rather than making things worse.
   Nothing surfaces that to the user except the cycle count.

---

## 8. Reproducing a single number

```sh
# instruction count and IMEM fit, at the defaults, no flags
build/native/bin/keac demo/build/mobilenetv2_features.tosa.mlir \
    --function mnv2_features -o /tmp/f.keaf --schedule --keep-intermediates
grep -cE '^\s+(MXU|DWU|VPU|DMA0|DMA1|CTRL)\s' /tmp/f.kasm     # 28675

# the whole graph lowers, and does not fit
build/native/bin/keac demo/build/mobilenetv2_full.tosa.mlir \
    --function mobilenetv2 -o /tmp/w.keaf --imem-budget 29241   # 37295, over IMEM

# whole-program roofline
build/native/bin/kea-sim demo/build/featpool.keaf --stats-json /tmp/s.json
.venv/bin/python -c "import json;print(json.load(open('/tmp/s.json'))['global']['roofline'])"

# the scheduler A/B, one number
build/native/bin/kea-sim demo/build/features_unsched.keaf --quiet --stats-json /tmp/u.json
build/native/bin/kea-sim demo/build/features.keaf --quiet --stats-json /tmp/c.json
.venv/bin/python -c "import json;print(json.load(open('/tmp/u.json'))['total_cycles'] \
    / json.load(open('/tmp/c.json'))['total_cycles'])"          # 1.1813

# bit-exactness, both splits
.venv/bin/python demo/validate_mobilenetv2.py

# the six fixed defects, still fixed
bash demo/regress/run_regressions.sh
```
