# KEAF — the KEA Executable Artifact Format

**Status: FROZEN.** Format version **1.0**, magic `KEA1`.
Normative machine-readable form: [`include/kea/keaf.h`](../include/kea/keaf.h).

A `.keaf` file is what `keac` emits and what `kea-rt` and `kea-sim` consume. It is the only
interface between the compiler and the runtime.

Companion documents: [ISA.md](ISA.md), [MICROARCH.md](MICROARCH.md).

---

## 1. Design goals

1. **mmap-and-go.** A loader maps the file and casts pointers into it. No parsing pass, no
   allocation, no endianness conversion, no relocation. The instruction stream in the file is
   byte-identical to what gets DMA'd into IMEM.
2. **Everything offset-based.** No pointers, no self-referential sizes, no variable-length
   prefixes. Every structure is fixed-size POD with a `static_assert` on its size.
3. **Extensible without breaking loaders.** New section types are additive; unknown section
   types must be skipped, not rejected. Trailing reserved fields absorb minor additions.
4. **Verifiable.** Whole-file CRC plus a per-section CRC, so a truncated or bit-rotted artifact
   fails loudly at load rather than producing garbage inference results.
5. **Self-describing at the tensor level.** The runtime can bind inputs and outputs by name,
   with shape, dtype, scale and zero point, without any side-channel metadata.

Constraints inherited from the ISA: **little-endian only**, `KEA_ISA_REVISION` must match
exactly, and the `CODE` section must be ≤ 32768 instructions (1 MiB IMEM).

---

## 2. File layout

```
offset 0   ┌────────────────────────────────────────┐
           │ KeafHeader                    64 bytes │
           ├────────────────────────────────────────┤
           │ (padding to 64)                        │
           ├────────────────────────────────────────┤
           │ section payloads, in any order,        │
           │ each 64-byte aligned:                  │
           │                                        │
           │   CODE         KeaInstr[n]             │
           │   CONST        raw weight bytes        │
           │   DRAM_LAYOUT  KeafDramLayout          │
           │   TENSORS      KeafTensorEntry[n]      │
           │   SPM_MAP      KeafSpmEntry[n]         │
           │   METADATA     UTF-8 JSON              │
           │                                        │
           ├────────────────────────────────────────┤
           │ KeafSection[section_count]  40 B each  │
           └────────────────────────────────────────┘
                                          = file_bytes
```

The section table conventionally goes last (so the writer can stream payloads and backpatch only
the table and the header), but its position is defined entirely by
`KeafHeader::section_table_offset` and a reader must not assume it.

All section payloads start at a multiple of `KEAF_SECTION_ALIGN` (**64**). That is enough for
any structure in the format and for a 64-byte cache line on the host.

---

## 3. `KeafHeader` — 64 bytes, at offset 0

| Off | Size | Field                  | Type   | Notes                                              |
|-----|------|------------------------|--------|----------------------------------------------------|
| 0   | 4    | `magic`                | `u8[4]`| `'K'`,`'E'`,`'A'`,`'1'`                            |
| 4   | 2    | `version_major`        | `u16`  | `1` — a mismatch is a hard load failure            |
| 6   | 2    | `version_minor`        | `u16`  | `0` — a higher value must still load               |
| 8   | 4    | `header_bytes`         | `u32`  | `64`                                               |
| 12  | 8    | `file_bytes`           | `u64`  | total file size; must equal the mapping length     |
| 20  | 4    | `section_count`        | `u32`  | ≥ 1                                                |
| 24  | 8    | `section_table_offset` | `u64`  | byte offset of `KeafSection[section_count]`        |
| 32  | 4    | `isa_revision`         | `u32`  | must equal `KEA_ISA_REVISION` (1)                  |
| 36  | 4    | `flags`                | `u32`  | `KEAF_FILEF_*`                                     |
| 40  | 4    | `crc32`                | `u32`  | whole-file CRC-32, this field treated as 0; `0` = "not computed" |
| 44  | 4    | `entry_pc`             | `u32`  | index of the first instruction, normally `0`       |
| 48  | 16   | `reserved`             | `u32[4]` | zero                                             |

File flags:

| Bit | Name                 | Meaning                                    |
|-----|----------------------|--------------------------------------------|
| 0   | `KEAF_FILEF_STRIPPED`| debug-only sections have been removed       |

---

## 4. `KeafSection` — 40 bytes per entry

| Off | Size | Field         | Type  | Notes                                                        |
|-----|------|---------------|-------|--------------------------------------------------------------|
| 0   | 4    | `type`        | `u32` | `KeafSectionType`                                             |
| 4   | 4    | `flags`       | `u32` | `KEAF_SECF_*`                                                 |
| 8   | 8    | `offset`      | `u64` | byte offset from file start; multiple of 64                   |
| 16  | 8    | `size`        | `u64` | payload bytes                                                 |
| 24  | 4    | `entry_count` | `u32` | record count, or `0` for opaque blobs                         |
| 28  | 4    | `entry_size`  | `u32` | bytes per record, or `0` for opaque blobs                     |
| 32  | 4    | `crc32`       | `u32` | CRC-32 of the payload                                         |
| 36  | 4    | `reserved`    | `u32` | zero                                                          |

If `entry_size != 0` then `size == entry_count * entry_size` — enforced by `keafValidate`.

| Section type  | id | `entry_size`             | Required | Contents                          |
|---------------|----|--------------------------|----------|-----------------------------------|
| `CODE`        | 1  | 32                       | **yes**  | `KeaInstr[entry_count]`           |
| `CONST`       | 2  | 0                        | no       | raw bytes staged into DRAM        |
| `DRAM_LAYOUT` | 3  | 0 (`size == 64`)         | **yes**  | one `KeafDramLayout`              |
| `TENSORS`     | 4  | 112                      | **yes**  | `KeafTensorEntry[entry_count]`    |
| `SPM_MAP`     | 5  | 64                       | no       | `KeafSpmEntry[entry_count]`       |
| `METADATA`    | 6  | 0                        | no       | UTF-8 JSON, **not** NUL-terminated|

Section flags:

| Bit | Name                       | Meaning                                                     |
|-----|----------------------------|-------------------------------------------------------------|
| 0   | `KEAF_SECF_STAGE_TO_DRAM`  | the loader copies this payload into DRAM at load time       |
| 1   | `KEAF_SECF_DEBUG_ONLY`     | may be stripped without affecting execution                 |
| 2   | `KEAF_SECF_ZERO_FILL`      | no payload in the file; `size` bytes of zeros. `offset` must be 0 |

**Unknown section types must be skipped silently.** A loader that rejects them cannot read
artifacts produced by a newer `keac`, which defeats the purpose of the section table.

---

## 5. `DRAM_LAYOUT` — `KeafDramLayout`, 64 bytes

The runtime allocates **one contiguous DRAM arena** of `total_bytes`, aligned to `alignment`.
Every DRAM address appearing in the instruction stream is an **offset into that arena**, so the
artifact is position-independent and needs no relocation.

| Off | Size | Field            | Type  | Notes                                          |
|-----|------|------------------|-------|------------------------------------------------|
| 0   | 8    | `total_bytes`    | `u64` | arena size; ≤ 4 GiB (`KEA_DRAM_BYTES`)         |
| 8   | 8    | `const_offset`   | `u64` | where `CONST` is staged                        |
| 16  | 8    | `const_bytes`    | `u64` |                                                |
| 24  | 8    | `io_offset`      | `u64` | input/output tensor arena                      |
| 32  | 8    | `io_bytes`       | `u64` |                                                |
| 40  | 8    | `scratch_offset` | `u64` | intermediate activation spill arena            |
| 48  | 8    | `scratch_bytes`  | `u64` |                                                |
| 56  | 4    | `alignment`      | `u32` | ≥ `KEA_DRAM_BASE_ALIGN` (64)                   |
| 60  | 4    | `reserved`       | `u32` | zero                                           |

The three regions are disjoint and together cover `[0, total_bytes)`. Conventional order is
`CONST`, then `IO`, then `SCRATCH`, but nothing depends on it.

```
DRAM arena
0                const_bytes      +io_bytes                     total_bytes
├────────────────┼────────────────┼───────────────────────────────────────┤
│ weights, quant │ inputs +       │ activation spills, double-buffer      │
│ param blocks   │ outputs        │ staging                               │
└────────────────┴────────────────┴───────────────────────────────────────┘
```

---

## 6. `TENSORS` — `KeafTensorEntry`, 112 bytes each

One entry per host-visible tensor. Inputs, outputs, and (optionally, for debugging) named
constants.

| Off | Size | Field         | Type      | Notes                                                |
|-----|------|---------------|-----------|------------------------------------------------------|
| 0   | 48   | `name`        | `char[48]`| NUL-padded; **not** guaranteed NUL-terminated. Use `keafNameEquals`. |
| 48  | 8    | `dram_offset` | `u64`     | offset into the DRAM arena                           |
| 56  | 8    | `size_bytes`  | `u64`     | bytes occupied                                       |
| 64  | 24   | `shape`       | `i32[6]`  | `shape[0..rank)`; remainder zero                     |
| 88  | 1    | `rank`        | `u8`      | 0…6                                                  |
| 89  | 1    | `dtype`       | `u8`      | `KeafDType`                                          |
| 90  | 1    | `kind`        | `u8`      | `KeafTensorKind`: 0 input, 1 output, 2 const, 3 scratch |
| 91  | 1    | `layout`      | `u8`      | `KeafLayout`: 0 NHWC, 1 NCHW, 2 FLAT                 |
| 92  | 4    | `scale`       | `f32`     | per-tensor quantization scale                        |
| 96  | 4    | `zero_point`  | `i32`     | per-tensor zero point                                |
| 100 | 4    | `index`       | `u32`     | ordinal within its `kind` (input 0, input 1, …)      |
| 104 | 8    | `reserved`    | `u32[2]`  | zero                                                 |

`KeafDType`: `0 int8`, `1 uint8`, `2 int4`, `3 int32`, `4 fp32`, `5 int16`.

The host dequantizes with `real = scale * (quantized - zero_point)`. I/O tensors are per-tensor
quantized; per-channel parameters live in `CONST` as `KeaQuantParam` blocks and are never
host-visible.

`size_bytes` is authoritative, not derived: for `int4` it is `ceil(prod(shape)/2)`, and it may
include trailing padding the compiler added for alignment.

---

## 7. `SPM_MAP` — `KeafSpmEntry`, 64 bytes each (debug only)

The output of the static memory planner, purely for tooling. Mark the section
`KEAF_SECF_DEBUG_ONLY`; a stripped artifact drops it and runs identically.

| Off | Size | Field      | Type      | Notes                                                    |
|-----|------|------------|-----------|----------------------------------------------------------|
| 0   | 40   | `name`     | `char[40]`| planner buffer name, e.g. `block_5/dw/act[buf0]`         |
| 40  | 4    | `space`    | `u32`     | `KeafSpace`: 0 `SPM_A`, 1 `SPM_W`, 2 `ACC`               |
| 44  | 4    | `offset`   | `u32`     | **bytes** for SPM_A/SPM_W, **int32 words** for ACC       |
| 48  | 4    | `size`     | `u32`     | same units as `offset`                                    |
| 52  | 4    | `first_pc` | `u32`     | first instruction index that touches the buffer          |
| 56  | 4    | `last_pc`  | `u32`     | last instruction index that touches the buffer            |
| 60  | 4    | `reserved` | `u32`     | zero                                                     |

`[first_pc, last_pc]` is the buffer's live range. Two entries in the same space with overlapping
address ranges **and** overlapping live ranges are a memory-planner bug; `kea-sim --check-spm`
should flag it. This table is also what a scratchpad-occupancy visualization is drawn from.

---

## 8. `METADATA` — JSON

Opaque to the runtime; consumed by `kea-sim`, profiling tools, and humans. UTF-8, not
NUL-terminated, length given by the section `size`. Unknown keys must be ignored.

Recommended schema:

```json
{
  "keaf_version": "1.0",
  "arch": "KEA-1",
  "producer": "keac 0.1.0",
  "producer_commit": "a1b2c3d",
  "built_at": "2026-08-01T09:15:00Z",

  "model": {
    "name": "mobilenet_v2_1.0_224",
    "source": "tflite",
    "source_hash": "sha256:…",
    "quantization": "int8 per-channel weights, per-tensor activations"
  },

  "compile_options": {
    "opt_level": 2,
    "fusion": true,
    "double_buffer": true,
    "prefetch_distance": 2,
    "weight_dtype": "int8"
  },

  "static_estimate": {
    "cycles": 1345678,
    "dram_bytes": 3612345,
    "instructions": 18432,
    "mxu_busy_cycles": 1180000,
    "dwu_busy_cycles": 52000,
    "vpu_busy_cycles": 310000
  },

  "peak_usage": { "spm_a_bytes": 245760, "spm_w_bytes": 262144, "acc_words": 32768 },

  "events": [
    { "id": 0, "name": "act_buf0_full"  },
    { "id": 2, "name": "act_buf0_free"  }
  ],

  "regions": [
    { "tag": 17, "name": "block_5/expand", "op": "conv2d_1x1",
      "shape_in": [1,14,14,384], "shape_out": [1,14,14,96],
      "macs": 7225344, "dram_bytes": 36864, "est_cycles": 28800 }
  ]
}
```

`regions[].tag` keys into the `TRACE` instruction `tag` field, which is how `kea-sim` labels its
per-region roofline report. `events[].name` makes deadlock reports readable.

---

## 9. Checksums

CRC-32, IEEE 802.3: reflected, polynomial `0xEDB88320`, init `0xFFFFFFFF`, final XOR
`0xFFFFFFFF`. `keafCrc32("123456789", 9) == 0xCBF43926`.

- `KeafSection::crc32` covers that section's payload bytes.
- `KeafHeader::crc32` covers the **entire file** with the four bytes of `crc32` itself treated as
  zero (`kea::keafFileCrc32`).
- A `crc32` of `0` means "not computed" and is not an error.

Verify the whole-file CRC once at install time. Skipping it on every load is fine and expected —
`keafValidate(file, n, /*check_crc=*/false)` still does all the structural checks.

---

## 10. Writing a KEAF file

```
1. reserve 64 bytes for the header
2. for each section:
     pad to a 64-byte boundary; record the offset
     append the payload
     record size, entry_count, entry_size, payload CRC
3. pad to 64; record section_table_offset; append KeafSection[]
4. fill in the header (magic, versions, file_bytes, section_count,
   section_table_offset, isa_revision, entry_pc), with crc32 = 0
5. compute keafFileCrc32() over the finished image, store it in the header
```

Never emit a `CODE` section that has not had `kea::keaValidate()` run over every instruction, and
never emit one whose last instruction is not `HALT`.

## 11. Loading a KEAF file

```
1. mmap the file; call keafValidate(base, size, check_crc)
     → rejects bad magic, wrong major version, wrong ISA revision, out-of-bounds
       sections, missing CODE/DRAM_LAYOUT/TENSORS, bad entry_size, an entry_pc
       outside CODE, and (optionally) a CRC mismatch
2. read DRAM_LAYOUT; allocate total_bytes aligned to `alignment`
3. for each section with KEAF_SECF_STAGE_TO_DRAM (normally just CONST):
     copy the payload to arena + const_offset
   for each section with KEAF_SECF_ZERO_FILL: zero the region
4. walk TENSORS; expose inputs and outputs to the host by name, with
   arena + dram_offset, size_bytes, shape, dtype, scale, zero_point
5. DMA the CODE section into IMEM (entry_count * 32 bytes)
6. host writes input tensors into the arena
7. set PC = entry_pc, start the machine
8. wait for done  (HALT dispatched AND all queues empty AND all units idle)
9. host reads output tensors out of the arena
```

Step 8 waits for **machine idle**, never for `HALT` itself — `HALT` only stops fetch, and the
final `DMA_ST` of the output tensor is typically still draining when it retires. See
[ISA.md §2.3](ISA.md#23-reset-state).

Nothing in this sequence allocates scratchpad, resolves symbols, or patches instructions. The
runtime is a loader and a doorbell; that is the whole design.

---

## 12. Versioning policy

| Change                                                | Bump          |
|-------------------------------------------------------|---------------|
| New section type                                      | minor         |
| New field in reserved space, ignorable by old readers  | minor         |
| New `KEAF_SECF_*` bit that old readers may ignore      | minor         |
| Any change to an existing struct's size or field offsets | **major**   |
| New required section                                   | **major**     |
| ISA encoding change                                    | `isa_revision` (independent of KEAF version) |

`isa_revision` and the KEAF version are independent. A loader must reject an artifact whose
`isa_revision` differs from the one it was built against — the instruction encodings would be
silently misinterpreted otherwise.
