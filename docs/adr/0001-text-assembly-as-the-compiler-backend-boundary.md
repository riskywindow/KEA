# ADR-0001: Text assembly is the boundary between the compiler and the binary

**Status:** accepted
**Date:** 2026-08-01

## Context

The KEA stack has two halves that would otherwise be forced into one build:

- The **compiler** (`compiler/`) is an out-of-tree MLIR project. It needs LLVM 20's
  CMake machinery, TableGen, `-fno-rtti` matching, and takes minutes to build.
- The **native** half (`sim/`, `runtime/`, `tools/`) is plain C++17 against
  `include/kea/*.h` and builds in seconds.

The compiler backend has to produce a KEAF artifact: an instruction stream plus a
weight blob plus a DRAM memory map. The obvious approach is to link `keaf.h` and
`isa.h` into an MLIR translation target and write the binary directly.

## Decision

**The MLIR compiler never writes binary. It emits three text/data files, and a
separate native assembler turns them into a KEAF artifact.**

```
model.py ──frontend──▶ model.tosa.mlir
                           │
                        kea-opt   (tosa/linalg ─▶ kea ─▶ fuse ─▶ tile ─▶ alloc ─▶ schedule)
                           │
                        kea-translate
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       model.kasm    model.weights.bin  model.map.json
        (assembly)     (constants)     (DRAM map, I/O
             │             │            tensors, metadata)
             └─────────────┼─────────────┘
                           ▼
                        kea-as
                           ▼
                       model.keaf  ──▶ kea-rt / kea-sim
```

`keac` is a thin driver that runs this pipeline end to end.

## Consequences

Good:

- **Every stage is human-inspectable.** You can read the scheduled, allocated,
  double-buffered instruction stream as text and see exactly which DMA overlaps
  which MATMUL. For a project whose entire thesis is "the compiler controls the
  hardware", being able to *read the compiler's output* is the point.
- The MLIR half never links native code, and the native half never links MLIR.
  Either can be rebuilt, tested, and worked on alone.
- The assembler and disassembler are trivially unit-testable, and
  `disassemble(assemble(x)) == x` is a cheap, strong round-trip property.
- Hand-written `.kasm` becomes the test vehicle for the simulator, so simulator
  work does not block on the compiler existing.

Bad:

- Two encodings of the ISA to keep in sync (`isa.h` structs, and the assembly
  syntax). Mitigated by deriving the assembly syntax mechanically from `isa.h`
  and by the round-trip test.
- A text serialize/parse round trip on every compile. Irrelevant at our scale.

## Rules this imposes

1. `docs/ASSEMBLY.md` specifies `.kasm` and is owned by the assembler. The
   backend emitter conforms to it; it does not get to invent syntax.
2. **Scratchpad addresses in `.kasm` are absolute integers.** The compiler has
   already done exact static memory planning; there is no allocator anywhere
   downstream, at assembly time or at run time. The assembler must never
   relocate an SPM/ACC address.
3. **DRAM addresses in `.kasm` are symbolic** (`@conv1.weights`, `@input`,
   `@output`) and resolved by the assembler against `model.map.json`. DRAM
   layout is still statically planned by the compiler and recorded in the map;
   the symbols exist purely so the assembly is readable and diffable.
4. `.kasm` is the simulator's *other* front door. `kea-sim` accepts either a
   `.keaf` or a `.kasm` directly.
