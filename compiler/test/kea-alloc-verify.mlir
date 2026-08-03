// RUN: kea-opt %s -split-input-file -kea-alloc=verify-only=true -verify-diagnostics
//
// The overlap verifier, run standalone over a placement this pass did not
// produce (`verify-only=true` reads the `kea.addr` already in the IR instead of
// creating one).
//
// The property is: no two buffers whose live ranges overlap may have been given
// overlapping storage. Getting it wrong is a silent aliasing bug -- it surfaces
// as a wrong number out of a kernel, with nothing in the trace pointing at the
// allocator -- so it is checked on every compile as well as here, and a
// hand-corrupted allocation must be caught.

//===--------------------------------------------------------------------===//
// 2. A hand-corrupted allocation: overlapping storage, overlapping lifetimes
//===--------------------------------------------------------------------===//
//
// `bad.a` occupies [0, 1024) and `bad.b` occupies [512, 1536), so they share
// bytes 512..1024. Both are live at the `kea.vcopy`, so every write through one
// silently corrupts the other. This is the exact shape of bug the verifier
// exists to catch, and it is what a subtly wrong packer would produce.
//
// The allocating path recomputes the placement and overwrites the bad
// `kea.addr`, so this function is only interesting under `verify-only=true` --
// hence the check prefix.

func.func @aliasing_placement() {
  // expected-error @+1 {{aliases 'bad.b': 'bad.a' occupies [0, 1024) of SPM_A over live range [0, 2] and 'bad.b' occupies [512, 1536) over [1, 2]; the ranges overlap, so the storage must not}}
  %a = kea.alloc {name = "bad.a", role = "scratch", kea.addr = 0 : i64}
    : !kea.buffer<1024xi8, A>
  %b = kea.alloc {name = "bad.b", role = "scratch", kea.addr = 512 : i64}
    : !kea.buffer<1024xi8, A>
  kea.vcopy from %a : !kea.buffer<1024xi8, A>, to %b
    {src_addr = 0, dst_addr = 0, row_bytes = 512, rows = 1,
     src_row_stride = 0, dst_row_stride = 0} : !kea.buffer<1024xi8, A>
  kea.halt
  return
}

// -----

//===--------------------------------------------------------------------===//
// 2b. A misaligned base, and the absolute address it produces
//===--------------------------------------------------------------------===//
//
// `bad.w` is fed to `LOAD_W`, so its base must be a multiple of 16 bytes
// (ISA.md §11.1). At 8 it is not -- and note the second diagnostic: the
// displacement `w_addr = 16` is legal on its own, which is exactly why the
// absolute `8 + 16 = 24` has to be re-checked rather than assumed.

func.func @misaligned_base() {
  // expected-error @+1 {{base address 8 is not a multiple of 16 bytes, which SPM_W requires for this buffer (ISA.md §11.1)}}
  %w = kea.alloc {name = "bad.w", role = "scratch", kea.addr = 8 : i64}
    : !kea.buffer<512xi8, W>
  // expected-error @+1 {{absolute w_addr = 8 + 16 = 24 is not a multiple of 16 bytes (ISA.md §11.1)}}
  kea.load_w %w {w_addr = 16, w_row_stride = 16, k_rows = 16, n_cols = 16,
                 bank = 0} : !kea.buffer<512xi8, W>
  kea.halt
  return
}
