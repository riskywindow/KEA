# The KEA-1 runtime

**Status: normative for `runtime/` and `tools/`.**

`kea_runtime` is the native library that turns artifacts into something executable and back
again. It is the only component that touches files: the simulator and the compiler both talk to
it in terms of `kea::KeaProgram`.

```
model.kasm ─┐
model.map.json ─┼─ kea-as ──▶ model.keaf ─┬─ kea-rt  ──▶ outputs
model.weights.bin ─┘                      ├─ kea-sim ──▶ cycles + outputs
                                          └─ kea-dis ──▶ model.kasm
```

Companion documents: [ASSEMBLY.md](ASSEMBLY.md) (the `.kasm` format and the `model.map.json`
schema), [ARTIFACT_FORMAT.md](ARTIFACT_FORMAT.md) (the binary), [ISA.md](ISA.md) (semantics),
[ADR-0001](adr/0001-text-assembly-as-the-compiler-backend-boundary.md).

---

## 1. What the runtime is, and what it is deliberately not

ARTIFACT_FORMAT.md §11 ends with:

> Nothing in this sequence allocates scratchpad, resolves symbols, or patches instructions. The
> runtime is a loader and a doorbell; that is the whole design.

So `kea_runtime` does exactly five things:

1. **Read** a `.keaf` into a `KeaProgram`, skipping section types it does not know.
2. **Assemble** a `.kasm` + `.map.json` + `.bin` into the same `KeaProgram`.
3. **Write** a `KeaProgram` back out as a valid `.keaf`.
4. **Disassemble** a `KeaProgram` back to canonical `.kasm`.
5. **Stage** one DRAM arena: allocate it once, copy the constants in, place inputs, extract
   outputs.

It contains no allocator for SPM, SPM_W or ACC, no relocation, no symbol patching of the
instruction stream, and no code path that allocates once execution has started.

---

## 2. Headers and entry points

Everything lives under `runtime/include/kea/rt/` and in `namespace kea::rt`. Link against the
CMake target `kea_runtime`; it propagates its include directory and `kea_isa`.

### 2.1 `kea/rt/load.h` — the front door

This is what the simulator calls. It sniffs the KEAF magic and either decodes the binary or runs
the assembler, producing the same `KeaProgram` either way — ADR-0001 rule 4, "`.kasm` is the
simulator's other front door".

```cpp
struct LoadOptions {
  const ModelMap* map = nullptr;                     // .kasm only: @symbol resolution
  const std::vector<std::uint8_t>* const_data = nullptr;  // .kasm only: the weight blob
  bool check_crc = true;
  bool rule_d_is_error = true;
};

bool looksLikeKeaf(const void* data, std::size_t size);
bool loadProgramFile(const std::string& path, const LoadOptions&, KeaProgram&, Diagnostics&);
bool loadProgram(const std::vector<std::uint8_t>& bytes, const std::string& filename,
                 const LoadOptions&, KeaProgram&, Diagnostics&);
```

### 2.2 `kea/rt/assembler.h`

```cpp
struct AsmOptions {
  const ModelMap* map = nullptr;
  const std::vector<std::uint8_t>* const_data = nullptr;
  bool rule_d_is_error = true;
};

bool assemble(const std::string& source, const std::string& filename, const AsmOptions&,
              KeaProgram&, Diagnostics&);
bool assembleFile(const std::string& path, const AsmOptions&, KeaProgram&, Diagnostics&);
```

Both return `false` on any error and leave the explanation in `Diagnostics`. `program.code` is
populated only from the text; `dram`, `tensors`, `spm_map` and `metadata_json` come from the map;
`const_data` is copied from `AsmOptions::const_data`.

### 2.3 `kea/rt/disassembler.h`

```cpp
struct DisasmOptions {
  const ModelMap* map = nullptr;   // symbolize DRAM addresses
  bool annotate = false;           // "; pc=N" + SPM_MAP buffer names
  bool emit_directives = true;
};

std::string disassemble(const KeaProgram&, const DisasmOptions&);
std::string disassembleInstruction(const KeaInstr&, const ModelMap* map = nullptr);
```

`disassembleInstruction` is the one a simulator trace wants: it renders a single instruction in
canonical `UNIT MNEMONIC field=value, ...` form with no trailing newline.

### 2.4 `kea/rt/keaf_io.h`

```cpp
struct KeafWriteOptions {
  bool include_spm_map = true;   // false also sets KEAF_FILEF_STRIPPED
  bool include_metadata = true;
  bool compute_crc = true;
};

bool writeKeaf(const KeaProgram&, std::vector<std::uint8_t>& out, std::string& error,
               const KeafWriteOptions& = {});
bool writeKeafFile(const KeaProgram&, const std::string& path, std::string& error,
                   const KeafWriteOptions& = {});
bool readKeaf(const void* data, std::size_t size, KeaProgram&, std::string& error,
              bool check_crc = true);
bool readKeafFile(const std::string& path, KeaProgram&, std::string& error,
                  bool check_crc = true);

bool readWholeFile(const std::string& path, std::vector<std::uint8_t>& out, std::string& error);
bool writeWholeFile(const std::string& path, const void* data, std::size_t size,
                    std::string& error);
```

### 2.5 `kea/rt/model_map.h`

```cpp
struct MapSymbol { std::string name; std::uint64_t offset, size; bool from_tensor; };

struct ModelMap {
  KeafDramLayout dram;
  std::vector<MapSymbol> symbols;      // symbols[] followed by tensors[]
  std::vector<TensorBinding> tensors;
  std::vector<SpmBinding> spm_map;
  std::uint32_t entry_pc;
  std::string metadata_json;

  const MapSymbol* find(const std::string& name) const;
  const MapSymbol* covering(std::uint64_t addr) const;   // tightest enclosing range
};

bool parseModelMap(const std::string& text, const std::string& filename, ModelMap&, Diagnostics&);
bool loadModelMapFile(const std::string& path, ModelMap&, Diagnostics&);
std::string dumpModelMap(const ModelMap&);
ModelMap modelMapFromProgram(const KeaProgram&);   // symbolize a .keaf with no side-car
```

### 2.6 `kea/rt/dram_arena.h`

```cpp
class DramArena {
 public:
  bool reset(const KeaProgram&, std::string& error);   // the ONLY allocation
  void release() noexcept;

  std::uint8_t*       data() noexcept;
  std::size_t         size() const noexcept;
  std::size_t         alignment() const noexcept;
  const KeaProgram*   program() const noexcept;

  std::uint8_t*       tensorData(const TensorBinding&) noexcept;
  bool writeTensor(const TensorBinding&, const void* src, std::size_t n) noexcept;
  bool readTensor (const TensorBinding&, void* dst, std::size_t n) const noexcept;
  bool restageConstants() noexcept;
};
```

### 2.7 `kea/rt/diagnostics.h`, `kea/rt/json.h`, `kea/rt/op_fields.h`

- `Diagnostics` collects `{file, line, col, length, message, notes}` and renders them with the
  source line and a caret. `Diagnostics::contains()` exists for tests.
- `Json` is a ~480-line RFC 8259 reader/writer with no dependencies: order-preserving objects,
  int64-vs-double number distinction, UTF-8 from `\u` escapes including surrogate pairs, and a
  nesting cap so a hostile file cannot smash the stack.
- `op_fields.h` is the table that defines `.kasm` operand syntax, derived from `keaOpInfo()`. A
  tool that wants to iterate the ISA's fields generically — a trace formatter, a coverage
  report — should read this rather than re-deriving it.

---

## 3. The DRAM arena contract

The runtime allocates **one** contiguous block of `dram.total_bytes`, aligned to
`dram.alignment`, and every DRAM address in the instruction stream is an offset into it. The
artifact is therefore position independent and needs no relocation.

```
DRAM arena
0            const_offset        io_offset            scratch_offset      total_bytes
├────────────┼───────────────────┼────────────────────┼───────────────────┤
│            │ weights, quant    │ inputs + outputs   │ activation spills │
│            │ param blocks      │                    │ double-buffer     │
└────────────┴───────────────────┴────────────────────┴───────────────────┘
                    ▲                     ▲                     ▲
              staged from the       written by the        never touched
              CONST section         host before START     by the host
```

`DramArena::reset()`:

1. validates `alignment ≥ 64` and a power of two, `total_bytes ≤ 4 GiB`, the three regions
   inside the arena, and `const_data.size() == dram.const_bytes`;
2. allocates the block with aligned `operator new`;
3. zero fills it;
4. copies the `CONST` image to `const_offset`.

Everything after that is `noexcept` and allocation free. `restageConstants()` exists so a second
inference run can reset the weight region without reallocating.

The arena is the *host's* image of device memory. Against a real device a driver would DMA it
across the bus; against the simulator, `kea-rt` pushes it into `kea::sim::Machine::dram` in one
write before the run and pulls the output tensor ranges back after it.

### 3.1 The no-runtime-allocation guarantee, and how it is enforced

The guarantee is: **once the program starts, the runtime performs no heap allocation.**

It is enforced by a test, not by a convention. `runtime/tests/test_no_alloc.cpp` replaces the
global `operator new` / `operator delete` — including the C++17 aligned and sized overloads —
with counting versions, then runs the whole execution-phase API inside a counting window:

```cpp
g_counting = true;
arena.writeTensor(*in, host_in.data(), host_in.size());
arena.tensorData(*in);
arena.restageConstants();
arena.readTensor(*out, host_out.data(), host_out.size());
g_counting = false;
CHECK(g_new_count == 0);
CHECK(g_delete_count == 0);
```

If someone adds a `std::string` to an error path in `writeTensor`, the test fails. That is the
whole mechanism, and it is why every method past `reset()` returns `bool` and takes raw pointers
rather than doing anything friendlier.

---

## 4. Loading

The sequence is ARTIFACT_FORMAT.md §11, implemented in `kea-rt`:

```
1. loadProgramFile()          mmap-equivalent read, keafValidate(), CRC check
2. DramArena::reset()         allocate total_bytes; stage CONST at const_offset
3. arena.writeTensor()        host inputs into the io region
4. push the arena to the device, set PC = entry_pc, ring the doorbell
5. wait for machine idle      HALT dispatched AND all queues empty AND all units idle
6. arena.readTensor()         host outputs out of the io region
```

Step 5 waits for **idle**, never for `HALT` itself: `HALT` only stops fetch, and the final
`DMA_ST` of the output tensor is typically still draining when it retires (ISA.md §2.3).

### 4.1 Forward compatibility

`readKeaf()` iterates the section table and switches on the type. The `default` branch does
nothing, on purpose:

> **Unknown section types must be skipped silently.** A loader that rejects them cannot read
> artifacts produced by a newer `keac`, which defeats the purpose of the section table.

`runtime/tests/test_keaf_io.cpp` builds artifacts carrying section types 7, 42 and 0xFFFF and
asserts they load to a program identical to the original. It also asserts that a higher
*minor* version loads and a higher *major* version does not.

### 4.2 Integrity

Both CRC layers are checked. The whole-file CRC comes from `keafValidate()`; the per-section
CRCs are checked as the sections are walked, so a corrupt payload names its own section. A CRC
of 0 means "not computed" and is not an error.

`readKeaf(..., check_crc=false)` skips both and does structural validation only — the documented
trade-off for a hot load path whose artifact was verified at install time. Truncation and bad
magic are still caught, because they are structural.

---

## 5. Authoring `.kasm` by hand

Hand-written `.kasm` is how the simulator got tested before the compiler existed, and it stays
the cheapest way to reproduce a bug. The corpus lives in `runtime/tests/kasm/` and is meant to be
copied from:

| File | What it is for |
|------|----------------|
| `all_opcodes.kasm` | every opcode, every field, both spellings of every symbolic value |
| `double_buffered.kasm` | the worked example from ASSEMBLY.md §6 |
| `echo.kasm` | a 10-instruction end-to-end program with real input and output tensors |
| `messy.kasm` | comments, labels, hex, `b` suffixes, continuation lines — all normalized away |

A minimal complete program:

```
.arch "KEA-1"
.isa_revision 1

  VPU   VCOPY   mode=fill, src_space=SPM_A, dst_space=SPM_A, src_addr=a:0, dst_addr=a:0, row_bytes=64, rows=1, src_row_stride=0, dst_row_stride=64, fill_value=7
  CTRL  HALT    exit_code=0
```

The loop to work in:

```sh
kea-as  scratch.kasm --check                    # syntax + ISA validation, writes nothing
kea-as  scratch.kasm --map m.json -o s.keaf
kea-dis s.keaf --map m.json --annotate          # read it back with pc + buffer names
kea-rt  s.keaf --input x=x.bin --output y=y.bin # run it
kea-sim s.keaf --map m.json --trace             # run it and look at the schedule
```

Three things bite people, in order of frequency:

1. **ACC is word-addressed.** `acc:16` is byte 64. The `acc:` prefix and the `w` suffix on
   strides exist so the assembler catches this; write `acc_inner_stride=16w`, not `=64`.
2. **Every operand is mandatory.** There are no defaults. If you forget `acc_mode`, you get an
   error naming it rather than a silently overwritten accumulator.
3. **Rule D.** A `WAIT` must be preceded in *stream order* by the `SIGNAL`s that release it. It
   is the natural order anyway ("issue the prefetch, then wait for it"), and the assembler
   refuses the program otherwise because the alternative is a deadlock.

---

## 6. Tools

### `kea-as` — assembler

```
kea-as <input.kasm> [-o <output.keaf>] [options]
  -o, --output <file>   artifact to write (default: the input with .keaf)
      --map <file>      model.map.json: DRAM symbols, tensors, SPM map, metadata
      --const <file>    constant/weight blob staged at dram.const_offset
      --strip           omit the SPM_MAP and METADATA debug sections
      --no-crc          store 0 instead of the whole-file CRC-32
      --no-rule-d       downgrade Rule D violations to warnings
      --check           parse and validate only; write nothing
```

Exits non-zero on any error, with diagnostics on stderr.

### `kea-dis` — disassembler

```
kea-dis <input.keaf|input.kasm> [options]
      --map <file>      model.map.json, for full DRAM symbolization
      --annotate        append '; pc=N' and SPM_MAP buffer names
      --no-symbols      print raw dram:<address> literals
      --no-directives   omit the .arch/.isa_revision/.entry prologue
      --emit-map        print the implied model.map.json instead of .kasm
      --no-crc          skip CRC verification while loading
  -o, --output <file>   write here instead of stdout
```

Without `--map`, DRAM addresses are symbolized against the artifact's own `TENSORS` section, so a
`.keaf` alone still disassembles readably; `--map` adds the compiler's full symbol table.

### `kea-rt` — run an artifact

```
kea-rt <program.keaf|program.kasm> [options]
      --input NAME=FILE   stage FILE into input tensor NAME
      --output NAME=FILE  write output tensor NAME to FILE after the run
      --map FILE          model.map.json (required for .kasm input)
      --const FILE        constant blob (required for .kasm with const_bytes)
      --list-tensors      print the tensor table and exit
      --check             load and stage only; do not run
      --max-cycles N      abort the run after N simulated cycles
      --no-crc            skip KEAF CRC verification
  -q, --quiet             suppress the completion summary
```

`kea-rt` is the deployment-shaped driver: load, stage, run, extract, and exit with the program's
own `HALT exit_code`. Input files must match `size_bytes` exactly. The "device" is `kea_sim`, so
`kea-rt` is built only when the simulator target exists; the runtime library and `kea-as` /
`kea-dis` build and test without it.

For cycle counts, per-region roofline, traces and `--stats-json`, use `kea-sim` instead — same
artifact, analysis-shaped front end.

---

## 7. Tests

`ctest -R runtime`:

| Test | What it pins down |
|------|-------------------|
| `runtime.test_json` | malformed input, `\u` escapes and surrogates, 300-deep nesting, int64/double edges |
| `runtime.test_op_fields` | the `.kasm` field tables reproduce `keaOpInfo()` exactly, for every opcode |
| `runtime.test_roundtrip_kasm` | `disassemble(assemble(x)) == x` over the corpus; non-canonical input normalizes to a fixed point; the corpus really does cover every opcode, field and symbolic value |
| `runtime.test_keaf_io` | `read(write(p)) == p`; section alignment and CRCs; stripped artifacts; CRC corruption, truncation and bad magic; unknown section types skipped; minor/major version policy |
| `runtime.test_golden_encoding` | 11 instructions compared against byte arrays derived by hand from ISA.md, so an encoding bug cannot hide behind a self-consistent round trip |
| `runtime.test_asm_errors` | ~70 specific diagnostics: bad mnemonic, wrong unit, out-of-range field, misaligned ACC address, missing `w` suffix, unresolved `@symbol`, HALT discipline, Rule D, and the caret's column |
| `runtime.test_no_alloc` | zero heap operations in the execution path (§3.1) |
| `runtime.tools_roundtrip` | the `kea-as` / `kea-dis` command lines, including that a bad listing exits non-zero with a located diagnostic |
| `runtime.tools_rt` | `kea-rt` end to end: host file → arena → device → arena → host file |
