# `.kasm` — the KEA-1 assembly language

**Status: normative.** This document specifies the text format that
[ADR-0001](adr/0001-text-assembly-as-the-compiler-backend-boundary.md) puts between the MLIR
compiler and the binary artifact. The compiler backend emits `.kasm`; `kea-as` turns it into a
`.keaf`; `kea-dis` turns a `.keaf` back into `.kasm`.

> **Rule 1 of ADR-0001: this document is owned by the assembler. The backend emitter conforms to
> it; it does not get to invent syntax.**

Normative machine-readable form: [`runtime/include/kea/rt/op_fields.h`](../runtime/include/kea/rt/op_fields.h)
and [`runtime/src/op_fields.cpp`](../runtime/src/op_fields.cpp). Every operand name and order in
§5 is derived mechanically from `kea::keaOpInfo()` in [`include/kea/isa.h`](../include/kea/isa.h),
and `runtime/tests/test_op_fields.cpp` fails the build if the two ever disagree.

Companion documents: [ISA.md](ISA.md) (semantics), [ARTIFACT_FORMAT.md](ARTIFACT_FORMAT.md)
(the binary), [RUNTIME.md](RUNTIME.md) (loading and running).

---

## 1. The three rules that matter

If you read nothing else, read these. They are the reason the format looks the way it does.

### 1.1 Scratchpad and accumulator addresses are absolute, and are never relocated

The compiler has already done exact static memory planning. There is no allocator anywhere
downstream — not at assembly time, not at load time, not at run time. `a:1600` in the source is
byte 1600 of SPM_A in the encoding, always.

The assembler will range-check and alignment-check an SPM/ACC address, and it will refuse one
that is out of bounds, but it will never *move* one.

### 1.2 `ACC` is addressed in **int32 words**, not bytes

`acc:16` is the 17th int32, i.e. byte 64. ISA.md §2.2 says this "trips everyone once", so the
syntax is built to make it trip you at assembly time instead:

- an ACC **address** must be written `acc:<word index>` — a bare integer or an `a:`/`w:` prefix
  is a hard error;
- an ACC **stride** must carry a `w` suffix — `acc_inner_stride=16w`. Writing `16` is a hard
  error, and writing `16w` for a *byte* stride is also a hard error.

So a line always shows you the units:

```
  MXU   MATMUL  a_addr=a:0, a_inner_stride=16, ..., acc_addr=acc:0, acc_inner_stride=16w, ...
                     └─ SPM_A bytes ─┘                  └── ACC int32 words ──┘
```

### 1.3 DRAM addresses are symbolic

DRAM layout is still statically planned by the compiler; the symbols exist so the assembly is
readable and diffable. Write `@conv1.weights`, `@conv1.weights+256`, `@input`. The assembler
resolves them against `model.map.json` (§7). An unresolved symbol is an error, never a guess.

A `dram:<byte address>` escape exists for hand-written tests that have no map. The compiler
backend must not emit it.

---

## 2. Lexical structure

| Element | Form |
|---------|------|
| comment | `;` or `#` to end of line |
| identifier | `[A-Za-z_][A-Za-z0-9_]*` |
| DRAM symbol | `@` followed by `[A-Za-z0-9_./]+`, e.g. `@block_5/dw.weights` |
| integer | `123`, `0x7B`, optional leading `-` or `+` |
| unit suffix | `b` = bytes (optional), `w` = ACC int32 words (mandatory where it applies) |
| string | `"..."`, with `\"`, `\\`, `\n`, `\t` |
| label | identifier followed by `:` at statement position |

Whitespace is insignificant except that a statement ends at a newline. **A statement whose line
ends with `,` continues onto the next line**, which is the only continuation rule:

```
    DMA0 DMA_LD spm_space=SPM_A, dram_addr=@input, spm_addr=a:0,
                len0=256, n1=8, n2=1, dram_s1=1024, dram_s2=8192,
                spm_s1=256, spm_s2=2048
```

> **Hex + `b` suffix is ambiguous**: `0x1b` is the hex number 27, not `0x1` with a byte suffix,
> because `b` is a hex digit. Use decimal when you want an explicit `b`. The canonical form never
> emits either, so this only affects hand-written input.

Mnemonics and unit names are **upper case**; field names and symbolic values are **lower case**
(except the space names `SPM_A` / `SPM_W`). This is not cosmetic: it lets you scan the unit and
opcode columns of a scheduled stream without reading the operands.

---

## 3. Grammar

```
file        := { line }
line        := [ label ] [ directive | instruction ] [ comment ] NEWLINE
label       := IDENT ":"
directive   := "." IDENT args
instruction := UNIT MNEMONIC [ field { "," field } ]
field       := IDENT "=" value
value       := integer | symbolic | spm-addr | acc-addr | dram-addr
spm-addr    := ( "a" | "w" ) ":" integer
acc-addr    := "acc" ":" integer
dram-addr   := "@" SYMBOL [ ("+"|"-") integer ] | "dram" ":" integer
UNIT        := "MXU" | "DWU" | "VPU" | "DMA0" | "DMA1" | "CTRL"
```

**Every operand must be written explicitly.** There are no defaults and no optional fields. A
missing field is an error naming the field; this is deliberate, because a defaulted `acc_mode`
or `bank` is exactly the kind of bug that produces plausible-looking wrong numbers.

Fields may appear in any order. The canonical form (§4) uses `keaOpInfo()` order.

### 3.1 The unit column

Every statement begins with the queue that executes it. For `NOP`, `SIGNAL`, `WAIT`, `TRACE`,
`DMA_LD` and `DMA_ST` this really is an operand — `keaOpInfo()` lists `unit` first — and it is
hoisted into a fixed column so a scheduled stream lines up:

```
  DMA0  DMA_LD  ...        ; the load for iteration n+2
  MXU   MATMUL  ...        ; ... overlaps the compute for iteration n
```

For every other opcode the unit is fixed by the opcode and must still be written; a mismatch is
an error (`MATMUL must be issued on MXU, not VPU`). `HALT` must target `CTRL`; DMA opcodes must
target `DMA0` or `DMA1`.

### 3.2 Directives

| Directive | Meaning |
|-----------|---------|
| `.arch "KEA-1"` | checked against `KEA_ARCH_NAME`; optional but recommended |
| `.isa_revision 1` | checked against `KEA_ISA_REVISION`; a mismatch is a hard error |
| `.entry <int \| label>` | first instruction index; defaults to `0`, or to the map's `entry_pc` |
| `.event <id>, "<name>"` | names a semaphore; lands in the artifact's METADATA `events[]` |
| `.region <tag>, "<name>"` | names a `TRACE` region; lands in METADATA `regions[]` |

`.event` and `.region` names are what make `kea-sim`'s deadlock reports and per-region roofline
readable, and they survive a `.kasm → .keaf → .kasm` round trip because they are stored in
METADATA. So do labels (METADATA `labels[]`).

---

## 4. Canonical form

`kea-dis` emits exactly one spelling of any given program, and that spelling re-assembles to the
same bytes. Formally: `disassemble(assemble(x)) == x` for any `x` already in canonical form, and
`assemble(disassemble(p)) == p` for any program `p`. This is the round-trip property ADR-0001
asks for, and `runtime/tests/test_roundtrip_kasm.cpp` enforces it over the whole corpus in
`runtime/tests/kasm/`.

Canonical form is:

1. `.arch "KEA-1"`, `.isa_revision 1`, `.entry <n>`, then a blank line.
2. Any `.event` / `.region` declarations, then a blank line.
3. One instruction per line, never wrapped, formatted as

   ```
   ␣␣UNIT␣␣MNEMONIC␣␣field=value, field=value, ...
      └4┘    └─6──┘
   ```

   two spaces, the unit padded to 4 columns, two spaces, the mnemonic padded to 6 columns, two
   spaces, then the fields in `keaOpInfo()` order separated by `", "`.
4. Labels appear on their own line at column 0, immediately before the instruction they name.
5. Integers are decimal. Byte quantities carry no suffix; ACC word quantities carry `w`.
   Addresses carry their space prefix. DRAM addresses are symbolized when a map is available.
6. No comments.

`kea-dis --annotate` appends `; pc=N` plus the `SPM_MAP` buffer name covering each scratchpad
operand. Annotated output is **not** canonical — it is for reading, not for re-assembling.

---

## 5. The instruction set

Ranges below are what the assembler enforces before it hands the instruction to
`kea::keaValidate()`. Semantics are in ISA.md; this section is only syntax.

Notation: `a:<byte>` = SPM_A byte address, `w:<byte>` = SPM_W byte address, `acc:<word>` = ACC
int32 word index, `<n>w` = a stride counted in ACC int32 words, `×k` = must be a multiple of k.

### 5.1 CTRL — `NOP`, `HALT`, `SIGNAL`, `WAIT`, `TRACE`

#### `NOP` (0x00) — any queue

| # | field | value syntax | range |
|---|-------|--------------|-------|
| 1 | `cycles` | integer | 0…4294967295 |

#### `HALT` (0x01) — must target `CTRL`

| # | field | value syntax | range |
|---|-------|--------------|-------|
| 1 | `exit_code` | integer | 0…4294967295 |

`HALT` must be the **last** instruction in the stream, and there must be exactly one. Anything
after it is unreachable and is rejected.

#### `SIGNAL` (0x02) — any queue

| # | field | value syntax | range |
|---|-------|--------------|-------|
| 1 | `event` | integer | 0…31 |
| 2 | `inc` | integer | 0…2147483647 |

#### `WAIT` (0x03) — any queue

| # | field | value syntax | range |
|---|-------|--------------|-------|
| 1 | `event` | integer | 0…31 |
| 2 | `threshold` | integer | 0…2147483647 |

#### `TRACE` (0x04) — any queue

| # | field | value syntax | range |
|---|-------|--------------|-------|
| 1 | `kind` | `marker` \| `begin` \| `end` | — |
| 2 | `tag` | integer | 0…4294967295 |
| 3 | `payload` | integer | 0…4294967295 |

`tag` keys into the METADATA `regions[]` array (§7.6), which is how `kea-sim` labels its
per-region report. Bracket every fused layer with a `begin`/`end` pair.

### 5.2 DMA — `DMA_LD`, `DMA_ST`

#### `DMA_LD` (0x10) / `DMA_ST` (0x11) — must target `DMA0` or `DMA1`

| # | field | value syntax | range |
|---|-------|--------------|-------|
| 1 | `spm_space` | `SPM_A` \| `SPM_W` | — |
| 2 | `dram_addr` | `@sym[+off]` or `dram:<byte>` | inside the arena |
| 3 | `spm_addr` | `a:` or `w:`, matching `spm_space` | 0…262143 |
| 4 | `len0` | integer (bytes) | 1…65535 |
| 5 | `n1` | integer | 1…65535 |
| 6 | `n2` | integer | 1…255 |
| 7 | `dram_s1` | integer (bytes) | −2147483648…2147483647 |
| 8 | `dram_s2` | integer (bytes) | −2147483648…2147483647 |
| 9 | `spm_s1` | integer (bytes) | −2147483648…2147483647 |
| 10 | `spm_s2` | integer (bytes) | −2147483648…2147483647 |

`spm_space` and the `a:`/`w:` prefix on `spm_addr` are redundant **on purpose**: the assembler
cross-checks them, so a backend that computes the flag and the address from different variables
finds out here rather than in silicon.

Per ISA.md §6.1, a zero stride on the *destination* side is last-writer-wins and therefore
undefined; the assembler rejects it (`spm_s1=0` with `n1>1` on a load, `dram_s1=0` with `n1>1` on
a store). A zero stride on the *source* side is a legal broadcast.

### 5.3 MXU — `LOAD_W`, `MATMUL`

#### `LOAD_W` (0x20) — `MXU`

| # | field | value syntax | range |
|---|-------|--------------|-------|
| 1 | `w_addr` | `w:<byte>` | 0…262143, ×16 (int8) or ×8 (int4) |
| 2 | `w_row_stride` | integer (bytes) | 0…2147483647, ×16 (int8) or ×8 (int4) |
| 3 | `k_rows` | integer | 1…16 |
| 4 | `n_cols` | integer | 1…16 |
| 5 | `bank` | `0` \| `1` | — |
| 6 | `dtype` | `int8` \| `int4` | — |

The alignment of `w_addr` and `w_row_stride` depends on `dtype`, and the diagnostic says so.

#### `MATMUL` (0x21) — `MXU`

| # | field | value syntax | range |
|---|-------|--------------|-------|
| 1 | `a_addr` | `a:<byte>` | 0…262143 |
| 2 | `a_inner_stride` | integer (bytes) | −2147483648…2147483647 |
| 3 | `a_outer_stride` | integer (bytes) | −2147483648…2147483647 |
| 4 | `m_inner` | integer | 1…65535 |
| 5 | `m_outer` | integer | 1…65535 |
| 6 | `acc_addr` | `acc:<word>` | 0…32767, ×16 |
| 7 | `acc_inner_stride` | `<n>w` **ACC int32 words** | ×16 |
| 8 | `acc_outer_stride` | `<n>w` **ACC int32 words** | ×16 |
| 9 | `bank` | `0` \| `1` | — |
| 10 | `acc_mode` | `overwrite` \| `accumulate` | — |
| 11 | `dtype` | `int8` \| `int4` | — |

`m_inner * m_outer ≤ 2048` (`KEA_MXU_MAX_ROWS`) is checked. `a_addr` is deliberately
byte-granular — the conv lowering shifts it by `kw*sp`, and `sp` is 3 for an RGB first layer.

### 5.4 DWU — `DWCONV`

#### `DWCONV` (0x30) — `DWU`

| # | field | value syntax | range |
|---|-------|--------------|-------|
| 1 | `a_addr` | `a:<byte>` | 0…262143 |
| 2 | `w_addr` | `w:<byte>` | 0…262143 |
| 3 | `acc_addr` | `acc:<word>` | 0…32767, ×16 |
| 4 | `out_h` | integer | 1…65535 |
| 5 | `out_w` | integer | 1…65535 |
| 6 | `channels` | integer | 16…4096, ×16 |
| 7 | `a_row_stride` | integer (bytes) | −2147483648…2147483647 |
| 8 | `a_pix_stride` | integer (bytes) | −2147483648…2147483647 |
| 9 | `kernel` | `3` \| `5` | — |
| 10 | `stride` | `1` \| `2` | — |
| 11 | `acc_mode` | `overwrite` \| `accumulate` | — |

The ACC output is written densely as `[out_h][out_w][channels]` int32; the assembler checks that
`acc_addr + out_h*out_w*channels ≤ 32768`.

### 5.5 VPU — `VQUANT`, `VADD`, `VPOOL`, `VCOPY`

#### `VQUANT` (0x40) — `VPU`

| # | field | value syntax | range |
|---|-------|--------------|-------|
| 1 | `acc_addr` | `acc:<word>` | 0…32767, ×16 |
| 2 | `out_addr` | `a:<byte>` | 0…262143 |
| 3 | `qparam_addr` | `w:<byte>` | 0…262143, ×4 |
| 4 | `num_pixels` | integer | 1…4294967295 |
| 5 | `channels` | integer | 16…65535, ×16 |
| 6 | `acc_pix_stride` | `<n>w` **ACC int32 words** | ×16 |
| 7 | `out_pix_stride` | integer (bytes) | −2147483648…2147483647 |
| 8 | `out_zp` | integer | −128…127 |
| 9 | `clamp_lo` | integer | −128…127 |
| 10 | `clamp_hi` | integer | −128…127 |
| 11 | `dtype` | `int8` \| `int4` | — |

`clamp_lo ≤ clamp_hi` is checked, and with `dtype=int4` both clamps must lie inside `[-8, 7]`.
Fused activations are expressed purely through the clamps (ISA.md §10.1): ReLU is
`clamp_lo=out_zp`, ReLU6 is `clamp_lo=out_zp, clamp_hi=out_zp+round(6/out_scale)`.

#### `VADD` (0x41) — `VPU`

| # | field | value syntax | range |
|---|-------|--------------|-------|
| 1 | `a_addr` | `a:<byte>` | 0…262143 |
| 2 | `b_addr` | `a:<byte>` | 0…262143 |
| 3 | `out_addr` | `a:<byte>` | 0…262143 |
| 4 | `param_addr` | `w:<byte>` | 0…262143, ×4 |
| 5 | `num_elems` | integer | 1…4294967295 |
| 6 | `clamp_lo` | integer | −128…127 |
| 7 | `clamp_hi` | integer | −128…127 |

#### `VPOOL` (0x42) — `VPU`

| # | field | value syntax | range |
|---|-------|--------------|-------|
| 1 | `mode` | `max` \| `avg` | — |
| 2 | `in_addr` | `a:<byte>` | 0…262143 |
| 3 | `out_addr` | `a:<byte>` | 0…262143 |
| 4 | `out_h` | integer | 1…65535 |
| 5 | `out_w` | integer | 1…65535 |
| 6 | `channels` | integer | 1…65535 |
| 7 | `kh` | integer | 1…32 |
| 8 | `kw` | integer | 1…32 |
| 9 | `stride_h` | integer | 1…8 |
| 10 | `stride_w` | integer | 1…8 |
| 11 | `in_row_stride` | integer (bytes) | −2147483648…2147483647 |
| 12 | `out_row_stride` | integer (bytes) | −2147483648…2147483647 |

#### `VCOPY` (0x43) — `VPU`

| # | field | value syntax | range |
|---|-------|--------------|-------|
| 1 | `mode` | `copy` \| `fill` | — |
| 2 | `src_space` | `SPM_A` \| `SPM_W` | — |
| 3 | `dst_space` | `SPM_A` \| `SPM_W` | — |
| 4 | `src_addr` | `a:` or `w:`, matching `src_space` | 0…262143 |
| 5 | `dst_addr` | `a:` or `w:`, matching `dst_space` | 0…262143 |
| 6 | `row_bytes` | integer (bytes) | 1…4294967295 |
| 7 | `rows` | integer | 1…4294967295 |
| 8 | `src_row_stride` | integer (bytes) | −2147483648…2147483647 |
| 9 | `dst_row_stride` | integer (bytes) | −2147483648…2147483647 |
| 10 | `fill_value` | integer | −128…127 |

`src_addr`, `src_space` and `src_row_stride` are ignored in `fill` mode but must still be
written, because they occupy encoding bits that have to be deterministic. Emit
`src_space=SPM_A, src_addr=a:0, src_row_stride=0`.

`ACC` is not reachable from `VCOPY`; initialize accumulators with `acc_mode=overwrite` on a
`MATMUL`/`DWCONV` instead.

---

## 6. Worked example — a double-buffered DMA + MATMUL layer

A 1×1 convolution (`IC=32`, `OC=16`) over 8×8 output tiles, the shape MobileNetV2's expand and
project layers have. Two activation buffers alternate across the two DMA engines, and the two
MXU weight banks alternate across the two reduction tiles, so weight loads vanish under compute
and the load for tile *n+2* overlaps the MATMULs for tile *n*.

Memory plan (all of it decided by the compiler, all of it absolute in the text):

| Buffer | Space | Address | Size |
|--------|-------|---------|------|
| `act[buf0]` | SPM_A | `a:0` | 2048 B = 8×8×32 |
| `act[buf1]` | SPM_A | `a:2048` | 2048 B |
| `out[buf0]` | SPM_A | `a:8192` | 1024 B = 8×8×16 |
| `out[buf1]` | SPM_A | `a:9216` | 1024 B |
| `weights` | SPM_W | `w:0` | 512 B = 2 tiles × 256 |
| `qparams` | SPM_W | `w:4096` | 192 B = 16 ch × 12 |
| `acc[0]` | ACC | `acc:0` | 1024 **words** = 8×8×16 |
| `acc[1]` | ACC | `acc:1024` | 1024 **words** |

Activation addressing: channels innermost, so the pixel stride `sp` is 32 bytes and the row
stride `sr` is `8*32 = 256`. Reduction tile `ic0` selects a 16-byte channel slice, which is a
flat byte offset — hence `a_addr = A_base + ic0` (`a:0` then `a:16`).

```
.arch "KEA-1"
.isa_revision 1
.entry 0

.event 0, "a0_filled"
.event 1, "a1_filled"
.event 2, "a0_free"
.event 3, "a1_free"
.event 4, "acc0_ready"
.event 5, "acc1_ready"
.event 6, "out_ready"
.event 7, "params_ready"
.region 1, "conv1/1x1s1"

  DMA0  DMA_LD  spm_space=SPM_W, dram_addr=@conv1.weights, spm_addr=w:0, len0=512, n1=1, n2=1, dram_s1=512, dram_s2=512, spm_s1=512, spm_s2=512
  DMA0  DMA_LD  spm_space=SPM_W, dram_addr=@conv1.qparams, spm_addr=w:4096, len0=192, n1=1, n2=1, dram_s1=192, dram_s2=192, spm_s1=192, spm_s2=192
  DMA0  SIGNAL  event=7, inc=2
  DMA0  DMA_LD  spm_space=SPM_A, dram_addr=@input, spm_addr=a:0, len0=256, n1=8, n2=1, dram_s1=1024, dram_s2=8192, spm_s1=256, spm_s2=2048
  DMA0  SIGNAL  event=0, inc=1
  DMA1  DMA_LD  spm_space=SPM_A, dram_addr=@input+256, spm_addr=a:2048, len0=256, n1=8, n2=1, dram_s1=1024, dram_s2=8192, spm_s1=256, spm_s2=2048
  DMA1  SIGNAL  event=1, inc=1
  MXU   TRACE   kind=begin, tag=1, payload=0
  MXU   WAIT    event=7, threshold=1
  MXU   WAIT    event=0, threshold=1
  MXU   LOAD_W  w_addr=w:0, w_row_stride=16, k_rows=16, n_cols=16, bank=0, dtype=int8
  MXU   MATMUL  a_addr=a:0, a_inner_stride=32, a_outer_stride=256, m_inner=8, m_outer=8, acc_addr=acc:0, acc_inner_stride=16w, acc_outer_stride=128w, bank=0, acc_mode=overwrite, dtype=int8
  MXU   LOAD_W  w_addr=w:256, w_row_stride=16, k_rows=16, n_cols=16, bank=1, dtype=int8
  MXU   MATMUL  a_addr=a:16, a_inner_stride=32, a_outer_stride=256, m_inner=8, m_outer=8, acc_addr=acc:0, acc_inner_stride=16w, acc_outer_stride=128w, bank=1, acc_mode=accumulate, dtype=int8
  MXU   SIGNAL  event=2, inc=1
  MXU   SIGNAL  event=4, inc=1
  DMA0  WAIT    event=2, threshold=1
  DMA0  DMA_LD  spm_space=SPM_A, dram_addr=@input+512, spm_addr=a:0, len0=256, n1=8, n2=1, dram_s1=1024, dram_s2=8192, spm_s1=256, spm_s2=2048
  DMA0  SIGNAL  event=0, inc=1
  VPU   WAIT    event=7, threshold=1
  VPU   WAIT    event=4, threshold=1
  VPU   VQUANT  acc_addr=acc:0, out_addr=a:8192, qparam_addr=w:4096, num_pixels=64, channels=16, acc_pix_stride=16w, out_pix_stride=16, out_zp=-10, clamp_lo=-10, clamp_hi=127, dtype=int8
  VPU   SIGNAL  event=6, inc=1
  MXU   WAIT    event=1, threshold=1
  MXU   LOAD_W  w_addr=w:0, w_row_stride=16, k_rows=16, n_cols=16, bank=0, dtype=int8
  MXU   MATMUL  a_addr=a:2048, a_inner_stride=32, a_outer_stride=256, m_inner=8, m_outer=8, acc_addr=acc:1024, acc_inner_stride=16w, acc_outer_stride=128w, bank=0, acc_mode=overwrite, dtype=int8
  MXU   LOAD_W  w_addr=w:256, w_row_stride=16, k_rows=16, n_cols=16, bank=1, dtype=int8
  MXU   MATMUL  a_addr=a:2064, a_inner_stride=32, a_outer_stride=256, m_inner=8, m_outer=8, acc_addr=acc:1024, acc_inner_stride=16w, acc_outer_stride=128w, bank=1, acc_mode=accumulate, dtype=int8
  MXU   SIGNAL  event=3, inc=1
  MXU   SIGNAL  event=5, inc=1
  DMA1  WAIT    event=6, threshold=1
  DMA1  DMA_ST  spm_space=SPM_A, dram_addr=@output, spm_addr=a:8192, len0=128, n1=8, n2=1, dram_s1=512, dram_s2=4096, spm_s1=128, spm_s2=1024
  VPU   WAIT    event=5, threshold=1
  VPU   VQUANT  acc_addr=acc:1024, out_addr=a:9216, qparam_addr=w:4096, num_pixels=64, channels=16, acc_pix_stride=16w, out_pix_stride=16, out_zp=-10, clamp_lo=-10, clamp_hi=127, dtype=int8
  VPU   SIGNAL  event=6, inc=1
  DMA1  WAIT    event=6, threshold=1
  DMA1  DMA_ST  spm_space=SPM_A, dram_addr=@output+128, spm_addr=a:9216, len0=128, n1=8, n2=1, dram_s1=512, dram_s2=4096, spm_s1=128, spm_s2=1024
  MXU   TRACE   kind=end, tag=1, payload=0
  CTRL  HALT    exit_code=0
```

This is `runtime/tests/kasm/double_buffered.kasm` verbatim, with
`runtime/tests/kasm/double_buffered.map.json` alongside it. Things to read off the text:

- **The overlap is visible.** `DMA0`'s prefetch of tile 2 (pc 17) sits between the MXU's
  `SIGNAL`s for iteration 0 and the VPU's `VQUANT` of ACC 0. While the MXU is running iteration
  1's four instructions, DMA0 is refilling `act[buf0]` and the VPU is requantizing `acc[0]`.
  All three units are busy at once, and you can see it without a simulator.
- **`bank` alternates** across the `LOAD_W`/`MATMUL` pairs, so the weight load for the second
  reduction tile happens while the first tile's `MATMUL` is still streaming (ISA.md §5.3).
- **`acc_mode=overwrite` appears exactly once per ACC region**, on the first tap. Every later
  tap accumulates. There is no ACC-clear instruction and none is needed.
- **The strided DMA descriptor** pulls an 8×8 tile out of a 32×32×32 DRAM feature map:
  `len0=256` is 8 pixels × 32 channels of contiguous bytes, `n1=8` walks the rows, `dram_s1=1024`
  is one full DRAM row (32 × 32), and `spm_s1=256` packs them densely.
- **Rule D holds**: every `WAIT` is preceded in stream order by the `SIGNAL`s that supply it.
  The assembler checks this and refuses the program otherwise (§8).

---

## 7. `model.map.json`

The side-car the compiler emits with the `.kasm`. It carries everything the text deliberately
does not: the DRAM layout, the symbol table `@name` resolves against, the host-visible tensor
descriptors, the SPM allocation map, and free-form metadata. `kea-as` folds all of it into the
KEAF artifact.

```json
{
  "arch": "KEA-1",
  "isa_revision": 1,
  "entry_pc": 0,

  "dram": {
    "total_bytes": 1048576,
    "const_offset": 0,
    "const_bytes": 704,
    "io_offset": 65536,
    "io_bytes": 65536,
    "scratch_offset": 131072,
    "scratch_bytes": 917504,
    "alignment": 64
  },

  "symbols": [
    { "name": "conv1.weights", "offset": 0,   "size": 512 },
    { "name": "conv1.qparams", "offset": 512, "size": 192 }
  ],

  "tensors": [
    { "name": "input",  "kind": "input",  "index": 0, "offset": 65536, "size_bytes": 32768,
      "shape": [1, 32, 32, 32], "dtype": "int8", "layout": "NHWC",
      "scale": 0.0078125, "zero_point": -128 },
    { "name": "output", "kind": "output", "index": 0, "offset": 98304, "size_bytes": 16384,
      "shape": [1, 32, 32, 16], "dtype": "int8", "layout": "NHWC",
      "scale": 0.023529412, "zero_point": -10 }
  ],

  "spm_map": [
    { "name": "conv1/act[buf0]", "space": "SPM_A", "offset": 0, "size": 2048,
      "first_pc": 3, "last_pc": 17 }
  ],

  "metadata": { "producer": "keac 0.1.0" }
}
```

### 7.1 Top level

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| `arch` | string | no | must be `"KEA-1"` if present |
| `isa_revision` | int | no | must equal `KEA_ISA_REVISION` (1) if present |
| `entry_pc` | int | no | default entry point; `.entry` in the `.kasm` overrides it |
| `dram` | object | **yes** | §7.2 |
| `symbols` | array | no | §7.3 |
| `tensors` | array | no | §7.4 |
| `spm_map` | array | no | §7.5 |
| `metadata` | object | no | §7.6 |

Unknown keys are ignored, at every level.

### 7.2 `dram` — the arena geometry

Mirrors `KeafDramLayout` exactly (ARTIFACT_FORMAT.md §5). `total_bytes` is required; every other
field defaults to 0 except `alignment`, which defaults to 64.

| Key | Type | Meaning |
|-----|------|---------|
| `total_bytes` | int | arena size; ≤ 4 GiB |
| `const_offset`, `const_bytes` | int | where the `--const` blob is staged |
| `io_offset`, `io_bytes` | int | input/output tensor region |
| `scratch_offset`, `scratch_bytes` | int | intermediate activation spill region |
| `alignment` | int | ≥ 64, power of two |

The three regions must be disjoint and must fit inside `total_bytes`; the assembler checks this,
and checks that every resolved DRAM address lands inside the arena.

`const_bytes` must equal the size of the `--const` file. A mismatch is an error naming both
numbers — this catches a stale weight blob, which is otherwise a silent wrong-answer bug.

### 7.3 `symbols` — the DRAM symbol table

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| `name` | string | yes | what `@name` resolves to; may contain `.` and `/` |
| `offset` | int | yes | byte offset into the arena |
| `size` | int | no | extent, used by the disassembler to re-symbolize `@name+off` |

Names must be unique across `symbols` and `tensors` together. Give every symbol a `size`: the
disassembler symbolizes an address by finding the tightest covering `[offset, offset+size)`
range, so a missing size means an address one byte in prints as a raw `dram:` literal.

### 7.4 `tensors` — the host-visible I/O

Mirrors `KeafTensorEntry` (ARTIFACT_FORMAT.md §6). Every tensor is also a DRAM symbol, so
`@input` works without a duplicate `symbols` entry.

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| `name` | string | yes | ≤ 47 characters |
| `kind` | string | yes | `input` \| `output` \| `const` \| `scratch` |
| `offset` | int | yes | byte offset into the arena |
| `size_bytes` | int | yes | authoritative, not derived; for int4 it is `ceil(prod(shape)/2)` |
| `dtype` | string | yes | `int8` \| `uint8` \| `int4` \| `int32` \| `fp32` \| `int16` |
| `shape` | int[] | no | rank ≤ 6 |
| `layout` | string | no | `NHWC` (default) \| `NCHW` \| `FLAT` |
| `scale` | number | no | per-tensor quantization scale, default 1.0 |
| `zero_point` | int | no | per-tensor zero point, default 0 |
| `index` | int | no | ordinal within its kind; auto-assigned in file order |

The host dequantizes with `real = scale * (quantized - zero_point)`. Per-channel parameters are
never host-visible; they live in the `CONST` blob as `KeaQuantParam` records.

### 7.5 `spm_map` — the static memory planner's output (debug)

Mirrors `KeafSpmEntry`. Nothing at run time consults it — every address in the stream is already
absolute. It exists so `kea-dis --annotate`, the simulator's trace, and scratchpad-occupancy
plots can put names to addresses.

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| `name` | string | yes | planner buffer name, ≤ 39 characters |
| `space` | string | yes | `SPM_A` \| `SPM_W` \| `ACC` |
| `offset` | int | yes | **bytes** for SPM_A/SPM_W, **int32 words** for ACC |
| `size` | int | yes | same units as `offset` |
| `first_pc`, `last_pc` | int | no | the buffer's live range |

### 7.6 `metadata` — free-form JSON

Copied verbatim into the artifact's `METADATA` section. The recommended schema is in
ARTIFACT_FORMAT.md §8. `kea-as` merges the `.kasm`'s `.event`, `.region` and label declarations
into it as the `events`, `regions` and `labels` arrays, overwriting any same-named keys, so the
two sources of names cannot disagree.

---

## 8. What the assembler rejects

Beyond every field range and alignment in §5, and everything `kea::keaValidate()` checks:

| Rejected | Why |
|----------|-----|
| a mnemonic, unit, field or symbolic value it does not know | typo; the diagnostic lists the alternatives |
| a missing or duplicated field | no defaults, ever |
| an opcode on the wrong queue | ISA.md §2.1 |
| an SPM/ACC address outside its space, or misaligned | §1.1 |
| an ACC stride without `w`, or a byte quantity with `w` | §1.2 |
| an address prefix disagreeing with its space flag | the redundancy is the check |
| an unresolved `@symbol`, or one outside the arena | ADR-0001 rule 3 |
| a program that does not end with exactly one `HALT` | ISA.md §5.2 |
| a zero destination stride on a multi-run DMA | ISA.md §6.1, last-writer-wins |
| **a Rule D violation** | ISA.md §5.5; the machine would deadlock |

Rule D deserves its own note. The assembler walks the stream keeping a running balance per
event, and for every `WAIT e, thr` it requires that earlier `SIGNAL`s have already supplied
`thr` uncommitted counts. Because dispatch is in-order and its only stall is a full queue, a
`WAIT` whose `SIGNAL`s come later can wedge the whole machine. This is an **error**, not a
warning. `kea-as --no-rule-d` downgrades it, which exists so the simulator's deadlock detector
can be tested against a deliberately broken program.

Diagnostics carry a file, line, column, the width of the offending token, an explanation of what
was expected, and often a note:

```
conv1.kasm:2:103: error: 'acc_addr' must be a multiple of 16 int32 words, got 8
    MXU   MATMUL  a_addr=a:0, a_inner_stride=16, ..., acc_addr=acc:8, acc_inner_stride=16w, ...
                                                                ^
note: ACC is addressed in int32 WORDS, not bytes: acc:8 is byte offset 32
```

(the caret is shown against an abbreviated line here; in real output the whole source line is
reproduced and the column is exact)

---

## 9. Tools

```
kea-as  model.kasm --map model.map.json --const model.weights.bin -o model.keaf
kea-dis model.keaf [--map model.map.json] [--annotate] [--emit-map]
kea-rt  model.keaf --input input=in.bin --output output=out.bin
kea-sim model.keaf --map model.map.json          # or a .kasm directly
```

`kea-as --check` parses and validates without writing an artifact, which is the fastest way for a
backend test to assert that its output is well formed. `kea-dis` and `kea-rt` accept a `.kasm`
directly, as ADR-0001 rule 4 requires.
