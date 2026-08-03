# KEA-1 ISA errata

`docs/ISA.md` and `docs/MICROARCH.md` are frozen: they are the contract several
components were built against in parallel, and editing them retroactively would
invalidate work already validated against them. Where they turned out to be
silent or self-contradictory, the resolution is recorded here instead.

**This file is normative.** Where it disagrees with ISA.md or MICROARCH.md, this
file wins. Every item was found by building a real implementation against the
spec, not by reading it.

`docs/SIMULATOR.md` §7 carries the full reasoning and the cost of each choice;
this is the summary plus the parts that constrain the *compiler* rather than the
simulator.

## E1 — Same-cycle `SIGNAL`/`WAIT` ordering

ISA.md §5.3 (increments first, then waiters) and MICROARCH.md §8 (one
interleaved ascending-unit-id pass) disagree. Under §8, a waiter on a low unit
id misses a same-cycle signal from a higher one.

**Resolution: ISA.md §5.3 wins.** A cycle is three phases — all eligible
`SIGNAL`s, then all `WAIT`s in ascending unit id, then everything else.

## E2 — DMA engines advance at end of cycle

MICROARCH.md §8 arbitrates DMA at step 1, which makes a descriptor occupy
`overhead + data + 1` cycles — one more than `keaDmaOccupancy()`, which is
`static_assert`ed in `hw_config.h` and is the number the compiler costs against.

**Resolution: engines advance at end of cycle**, so uncontended occupancy is
exactly `keaDmaOccupancy()`. The compiler's cost model is correct as written.

## E3 — A queue slot freed this cycle is usable this cycle

MICROARCH.md §8's prose and its own pseudocode contradict each other.

**Resolution: the code wins** — dispatch follows starts, so the slot is
available in the same cycle. Worth one cycle per queue-full episode.

## E4 — `TRACE REGION_END` closes after the issuing unit drains

Read literally, a zero-latency `REGION_END` closes a region up to a pipeline
depth before the work it brackets finishes, which would make every per-layer
number in the roofline report short. `TRACE` exists for exactly that accounting.

**Resolution:** `REGION_END` still retires immediately and costs nothing — the
schedule is unchanged — but the region keeps accumulating until the issuing
unit's pipeline drains. Consecutive regions on one unit may therefore overlap by
up to a pipeline depth, which is the honest attribution: the tail of layer *n*
really is executing while layer *n+1* starts.

## E5 — int32 ACC overflow wraps

ISA.md is silent, and a long reduction chain can exceed int32.

**Resolution: wrapping two's complement.**

**Constraint on the compiler:** the tiling pass must bound reduction chains so
that accumulation cannot overflow int32 for the quantization ranges the frontend
produces, and must not rely on saturation to rescue it. For int8 × int8 inputs
the worst case is `K * 127 * 127`, so `K < 2^31 / 16129 ≈ 133,000` accumulated
taps — comfortable for every layer in MobileNetV2, but it is a real bound and
the tiler should assert it rather than assume it.

## E6 — `VADD` parameter bounds are unspecified

`keaQuantizedAdd()` sums `xa + xb` in plain int32. That is safe for the
parameters the frontend actually produces (`|xa|, |xb| ≲ 1.3e8`), but ISA.md
does not *bound* `a_mult`/`a_shift`, so a hostile or careless `KeaAddParam` can
overflow it.

**Constraint on the compiler:** `KeaAddParam` must be emitted such that
`|xa| + |xb| < 2^31`. A future ISA revision should either bound the parameters
normatively or specify saturation.

## E7 — MXU weight banks are undefined at reset

**Resolution:** the simulator warns once and treats an unloaded bank as zero
(`k_rows = 0`, contributing no useful MACs).

**Constraint on the compiler:** never issue a `MATMUL` against a bank no
`LOAD_W` has written. Do not rely on the zero-fill.

## E8 — Minor resolutions

- A `NOP` targeting the MXU occupies `ARRAY`.
- A `CTRL`-targeted `WAIT` stalls the dispatcher; implemented literally and
  reported by the deadlock detector.
- An unused DRAM grant is not redistributed within a cycle.
- `keaValidate()` bounds only base addresses; strided accesses are bounds
  checked at run time against their bounding interval.

## Known modelling gaps (not spec defects)

Recorded so nobody mistakes them for verified behaviour:

- **Scratchpad ports are never arbitrated.** SPM_A/SPM_W/ACC bank conflicts are
  not modelled, so the simulator is optimistic when several units hammer one
  space in the same cycle. This is the single largest optimism in the timing
  model. MICROARCH.md §6.4 floats a `--strict-spm` mode; it is not implemented.
- **The ACC/SPM hazard detector is range-based, not element-precise**, so a
  strided access registers its bounding interval and can warn on genuinely
  disjoint accesses. Hence it warns by default; `--strict-hazards` opts in.
