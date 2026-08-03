#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end numerical check for the KEA compiler backend.

Generates a small TOSA fixture with NON-UNIFORM weights, computes the expected
int8 output with an independent numpy reference (docs/QUANTIZATION.md section 1
and 3, using the frontend's normative `apply_scale_32`), compiles the fixture
with `keac`, runs it on `kea-sim`, and asserts bit-exact equality.

The weights matter.  A splat weight tensor passes even when the MXU tile
ordering, the k/n indexing inside a tile or the per-channel zero-point fold are
wrong, because every wrong permutation of an all-2s array is still an all-2s
array.  Every weight here is distinct along every axis, so the layout formulas
in docs/CODEGEN.md section 4 are actually under test:

  conv3x3        -> mxu_tiles_16x16        (IC = 8: one reduction tile, 9 taps)
  conv3x3_ic3_s2 -> mxu_tiles_16x16_packed (ISA.md section 8.6 channel packing)
  depthwise3x3   -> dwu_planes             (C = 24: padded lanes are read)
  fc             -> mxu_tiles_16x16, the degenerate KH = KW = 1 case
  bmm            -> mxu_tiles_16x16_kn     (output channel is the LAST dim)
  qadd           -> add_params
  fc_input_zp    -> regression test for the -kea-tile input-zero-point fix

Every scale is chosen so the outputs land in the interior of int8 rather than
against the clamps, so the comparison is against real values and not against a
tensor of saturated -128s.

Run:  .venv/bin/python tools/keac/tests/numeric_check.py [--keep]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "frontend"))
from kea_frontend.apply_scale import apply_scale_32, quantize_multiplier  # noqa: E402

KEAC = os.path.join(ROOT, "build", "native", "bin", "keac")


# --------------------------------------------------------------------------
# The reference.  Nothing here is shared with the compiler.
# --------------------------------------------------------------------------
def rescale_per_channel(acc, mult, shift, out_zp, lo=-128, hi=127):
    out = np.empty(acc.shape, dtype=np.int64)
    flat = acc.reshape(-1, acc.shape[-1])
    o = out.reshape(-1, acc.shape[-1])
    for c in range(acc.shape[-1]):
        for i in range(flat.shape[0]):
            o[i, c] = apply_scale_32(np.int32(flat[i, c]), np.int32(mult[c]),
                                     np.int32(shift[c]), True)
    return np.clip(out + out_zp, lo, hi).astype(np.int8)


def conv2d_ref(x, w, b, in_zp, pad, stride):
    """x NHWC int8, w OHWI int8, b [OC] int32 -> NHWC int32 accumulator."""
    N, H, W, IC = x.shape
    OC, KH, KW, _ = w.shape
    pt, pb, pl, pr = pad
    xp = np.full((N, H + pt + pb, W + pl + pr, IC), in_zp, dtype=np.int32)
    xp[:, pt:pt + H, pl:pl + W, :] = x.astype(np.int32)
    xc = xp - in_zp                       # TOSA pads with the input zero point
    OH = (H + pt + pb - KH) // stride[0] + 1
    OW = (W + pl + pr - KW) // stride[1] + 1
    out = np.zeros((N, OH, OW, OC), dtype=np.int64)
    for oh in range(OH):
        for ow in range(OW):
            win = xc[:, oh * stride[0]:oh * stride[0] + KH,
                     ow * stride[1]:ow * stride[1] + KW, :]
            out[:, oh, ow, :] = np.tensordot(win, w.astype(np.int32),
                                             axes=([1, 2, 3], [1, 2, 3]))
    return (out + b.astype(np.int64)).astype(np.int32)


def depthwise_ref(x, w, b, in_zp, pad, stride):
    """x NHWC int8, w HWCM int8 (M = 1), b [C] int32 -> NHWC int32."""
    N, H, W, C = x.shape
    KH, KW, _, _ = w.shape
    pt, pb, pl, pr = pad
    xp = np.full((N, H + pt + pb, W + pl + pr, C), in_zp, dtype=np.int32)
    xp[:, pt:pt + H, pl:pl + W, :] = x.astype(np.int32)
    xc = xp - in_zp
    OH = (H + pt + pb - KH) // stride[0] + 1
    OW = (W + pl + pr - KW) // stride[1] + 1
    out = np.zeros((N, OH, OW, C), dtype=np.int64)
    for oh in range(OH):
        for ow in range(OW):
            for kh in range(KH):
                for kw in range(KW):
                    out[:, oh, ow, :] += (
                        xc[:, oh * stride[0] + kh, ow * stride[1] + kw, :]
                        * w[kh, kw, :, 0].astype(np.int32))
    return (out + b.astype(np.int64)).astype(np.int32)


# --------------------------------------------------------------------------
# Fixture generation
# --------------------------------------------------------------------------
def i8_array(rng, shape, lo=-7, hi=8):
    return rng.integers(lo, hi, size=shape, dtype=np.int64).astype(np.int8)


def dense(a, ty):
    return "dense<[%s]> : %s" % (_nest(a), ty)


def _nest(a):
    if a.ndim == 1:
        return ", ".join(str(int(v)) for v in a)
    return ", ".join("[%s]" % _nest(v) for v in a)


def pick_scales(acc, target=90):
    """Per-channel (multiplier, shift) that map this accumulator onto roughly
    +/- `target`, so the comparison is against interior values rather than
    against a tensor of saturated clamps."""
    mults, shifts = [], []
    for c in range(acc.shape[-1]):
        peak = max(1, int(np.abs(acc[..., c]).max()))
        m, s = quantize_multiplier(target / peak)
        mults.append(int(m))
        shifts.append(int(s))
    return mults, shifts


def rescale_i32(x, mult, shift, in_zp):
    flat = x.reshape(-1).astype(np.int32)
    out = np.array([apply_scale_32(np.int32(int(v) - in_zp), np.int32(mult),
                                   np.int32(shift), True) for v in flat])
    return out.reshape(x.shape).astype(np.int32)


def fc_mlir(name, in_zp, w, b, m, s):
    return f"""
func.func @{name}(%x: tensor<4x32xi8>) -> tensor<4x48xi8> {{
  %w = "tosa.const"() {{value = {dense(w, "tensor<48x32xi8>")}}} : () -> tensor<48x32xi8>
  %b = "tosa.const"() {{value = {dense(b, "tensor<48xi32>")}}} : () -> tensor<48xi32>
  %acc = tosa.fully_connected %x, %w, %b {{
    quantization_info = #tosa.conv_quant<input_zp = {in_zp}, weight_zp = 0>
  }} : (tensor<4x32xi8>, tensor<48x32xi8>, tensor<48xi32>) -> tensor<4x48xi32>
  %y = tosa.rescale %acc {{
    input_zp = 0 : i32, output_zp = 6 : i32,
    multiplier = array<i32: {", ".join(map(str, m))}>,
    shift = array<i8: {", ".join(map(str, s))}>,
    scale32 = true, double_round = true, per_channel = true
  }} : (tensor<4x48xi32>) -> tensor<4x48xi8>
  return %y : tensor<4x48xi8>
}}
"""


def bmm_mlir(name, a_zp, w, m, s):
    return f"""
func.func @{name}(%x: tensor<1x6x32xi8>) -> tensor<1x6x16xi8> {{
  %w = "tosa.const"() {{value = {dense(w, "tensor<1x32x16xi8>")}}} : () -> tensor<1x32x16xi8>
  %acc = tosa.matmul %x, %w {{
    quantization_info = #tosa.matmul_quant<a_zp = {a_zp}, b_zp = 0>
  }} : (tensor<1x6x32xi8>, tensor<1x32x16xi8>) -> tensor<1x6x16xi32>
  %y = tosa.rescale %acc {{
    input_zp = 0 : i32, output_zp = -3 : i32,
    multiplier = array<i32: {", ".join(map(str, m))}>,
    shift = array<i8: {", ".join(map(str, s))}>,
    scale32 = true, double_round = true, per_channel = true
  }} : (tensor<1x6x16xi32>) -> tensor<1x6x16xi8>
  return %y : tensor<1x6x16xi8>
}}
"""


class Case:
    def __init__(self, name, mlir, x, expect, xfail=None):
        self.name, self.mlir, self.x, self.expect = name, mlir, x, expect
        self.xfail = xfail


def build_cases(rng):
    cases = []

    # --- 1. conv 3x3 s1 pad 1, IC 8 -> OC 16 : mxu_tiles_16x16 -------------
    x = i8_array(rng, (1, 8, 8, 8), -128, 128)
    w = i8_array(rng, (16, 3, 3, 8))
    b = rng.integers(-500, 500, size=(16,), dtype=np.int64).astype(np.int32)
    acc = conv2d_ref(x, w, b, -3, (1, 1, 1, 1), (1, 1))
    m, s = pick_scales(acc)
    y = rescale_per_channel(acc, m, s, -7)
    cases.append(Case("conv3x3", f"""
func.func @conv3x3(%x: tensor<1x8x8x8xi8>) -> tensor<1x8x8x16xi8> {{
  %w = "tosa.const"() {{value = {dense(w, "tensor<16x3x3x8xi8>")}}} : () -> tensor<16x3x3x8xi8>
  %b = "tosa.const"() {{value = {dense(b, "tensor<16xi32>")}}} : () -> tensor<16xi32>
  %acc = tosa.conv2d %x, %w, %b {{
    pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -3, weight_zp = 0>
  }} : (tensor<1x8x8x8xi8>, tensor<16x3x3x8xi8>, tensor<16xi32>) -> tensor<1x8x8x16xi32>
  %y = tosa.rescale %acc {{
    input_zp = 0 : i32, output_zp = -7 : i32,
    multiplier = array<i32: {", ".join(map(str, m))}>,
    shift = array<i8: {", ".join(map(str, s))}>,
    scale32 = true, double_round = true, per_channel = true
  }} : (tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi8>
  return %y : tensor<1x8x8x16xi8>
}}
""", x, y))

    # --- 2. conv 3x3 s2, IC 3 -> OC 16 : mxu_tiles_16x16_packed ------------
    # The MobileNetV2 first-layer shape: KW*IC = 9 <= 16, so a whole kernel
    # row folds into one reduction tile (ISA.md section 8.6).
    x2 = i8_array(rng, (1, 8, 8, 3), -128, 128)
    w2 = i8_array(rng, (16, 3, 3, 3))
    b2 = rng.integers(-300, 300, size=(16,), dtype=np.int64).astype(np.int32)
    acc2 = conv2d_ref(x2, w2, b2, -11, (0, 1, 0, 1), (2, 2))
    m2, s2 = pick_scales(acc2)
    y2 = rescale_per_channel(acc2, m2, s2, -20)
    cases.append(Case("conv3x3_ic3_s2", f"""
func.func @conv3x3_ic3_s2(%x: tensor<1x8x8x3xi8>) -> tensor<1x4x4x16xi8> {{
  %w = "tosa.const"() {{value = {dense(w2, "tensor<16x3x3x3xi8>")}}} : () -> tensor<16x3x3x3xi8>
  %b = "tosa.const"() {{value = {dense(b2, "tensor<16xi32>")}}} : () -> tensor<16xi32>
  %acc = tosa.conv2d %x, %w, %b {{
    pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>,
    dilation = array<i64: 1, 1>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = -11, weight_zp = 0>
  }} : (tensor<1x8x8x3xi8>, tensor<16x3x3x3xi8>, tensor<16xi32>) -> tensor<1x4x4x16xi32>
  %y = tosa.rescale %acc {{
    input_zp = 0 : i32, output_zp = -20 : i32,
    multiplier = array<i32: {", ".join(map(str, m2))}>,
    shift = array<i8: {", ".join(map(str, s2))}>,
    scale32 = true, double_round = true, per_channel = true
  }} : (tensor<1x4x4x16xi32>) -> tensor<1x4x4x16xi8>
  return %y : tensor<1x4x4x16xi8>
}}
""", x2, y2))

    # --- 3. depthwise 3x3 s1 pad 1, C 24 : dwu_planes ----------------------
    # C = 24 is deliberately NOT a multiple of 16, so the padded channel lanes
    # and their zero weight planes are exercised.
    x3 = i8_array(rng, (1, 8, 8, 24), -128, 128)
    w3 = i8_array(rng, (3, 3, 24, 1))
    b3 = rng.integers(-400, 400, size=(24,), dtype=np.int64).astype(np.int32)
    acc3 = depthwise_ref(x3, w3, b3, 12, (1, 1, 1, 1), (1, 1))
    m3, s3 = pick_scales(acc3)
    y3 = rescale_per_channel(acc3, m3, s3, 4)
    cases.append(Case("depthwise3x3", f"""
func.func @depthwise3x3(%x: tensor<1x8x8x24xi8>) -> tensor<1x8x8x24xi8> {{
  %w = "tosa.const"() {{value = {dense(w3, "tensor<3x3x24x1xi8>")}}} : () -> tensor<3x3x24x1xi8>
  %b = "tosa.const"() {{value = {dense(b3, "tensor<24xi32>")}}} : () -> tensor<24xi32>
  %acc = tosa.depthwise_conv2d %x, %w, %b {{
    pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>, acc_type = i32,
    quantization_info = #tosa.conv_quant<input_zp = 12, weight_zp = 0>
  }} : (tensor<1x8x8x24xi8>, tensor<3x3x24x1xi8>, tensor<24xi32>) -> tensor<1x8x8x24xi32>
  %y = tosa.rescale %acc {{
    input_zp = 0 : i32, output_zp = 4 : i32,
    multiplier = array<i32: {", ".join(map(str, m3))}>,
    shift = array<i8: {", ".join(map(str, s3))}>,
    scale32 = true, double_round = true, per_channel = true
  }} : (tensor<1x8x8x24xi32>) -> tensor<1x8x8x24xi8>
  return %y : tensor<1x8x8x24xi8>
}}
""", x3, y3))

    # --- 4. fully connected 32 -> 48 : mxu_tiles_16x16, KH = KW = 1 --------
    # NOTE the input zero point is 0 here, and case 6 says why.
    x4 = i8_array(rng, (4, 32), -128, 128)
    w4 = i8_array(rng, (48, 32))
    b4 = rng.integers(-600, 600, size=(48,), dtype=np.int64).astype(np.int32)
    acc4 = (x4.astype(np.int32) @ w4.astype(np.int32).T
            + b4.astype(np.int32)).astype(np.int32)
    m4, s4 = pick_scales(acc4)
    y4 = rescale_per_channel(acc4, m4, s4, 6)
    cases.append(Case("fc", fc_mlir("fc", 0, w4, b4, m4, s4), x4, y4))

    # --- 5. batched matmul [1,6,32] x [1,32,16] : mxu_tiles_16x16_kn -------
    # The one layout whose output channel is the LAST weight dimension.
    x5 = i8_array(rng, (1, 6, 32), -128, 128)
    w5 = i8_array(rng, (1, 32, 16))
    acc5 = (x5[0].astype(np.int32) @ w5[0].astype(np.int32))[None]
    m5, s5 = pick_scales(acc5)
    y5 = rescale_per_channel(acc5, m5, s5, -3)
    cases.append(Case("bmm", bmm_mlir("bmm", 0, w5, m5, s5), x5, y5))

    # --- 6. the same fully connected with a NON-ZERO input zero point ------
    # Regression test. `-kea-tile` used to set `ConvShape::inputZp` only on the
    # conv2d path, so fully_connected and matmul left it at 0 and the zero-point
    # fold `bias[c] - input_zp * sum_k w[c][k]` (DIALECT_L2.md section 4.3)
    # silently dropped its correction term -- every FC and matmul in a real PTQ
    # export was wrong by a per-channel constant. Cases 4 and 5 could not catch
    # it because they use input_zp = 0, which is exactly the value the bug
    # assumed. This case differs from case 4 only in the zero point.
    acc6 = ((x4.astype(np.int32) + 9) @ w4.astype(np.int32).T
            + b4.astype(np.int32)).astype(np.int32)
    m6, s6 = pick_scales(acc6)
    y6 = rescale_per_channel(acc6, m6, s6, 6)
    cases.append(Case("fc_input_zp", fc_mlir("fc_input_zp", -9, w4, b4, m6, s6),
                      x4, y6))


    # --- 7. quantized residual add : add_params ---------------------------
    # Two model inputs, rescaled into a common i32 domain, added, rescaled
    # back -- the TOSA idiom `-kea-fuse` turns into one VADD reading a single
    # KeaAddParam record (docs/DIALECT_L2.md section 4.4).
    xa = i8_array(rng, (1, 4, 4, 8), -128, 128)
    xb = i8_array(rng, (1, 4, 4, 8), -128, 128)
    la = rescale_i32(xa, 1073741824, 10, -5)
    lb = rescale_i32(xb, 1932735283, 11, 3)
    ssum = (la.astype(np.int64) + lb.astype(np.int64))
    ssum = np.clip(ssum, -2**31, 2**31 - 1).astype(np.int32)
    mo, so = quantize_multiplier(100.0 / max(1, int(np.abs(ssum).max())))
    ya = np.clip(np.array([[apply_scale_32(np.int32(v), np.int32(mo),
                                           np.int32(so), True)]
                           for v in ssum.reshape(-1)]).reshape(ssum.shape) - 9,
                 -128, 127).astype(np.int8)
    cases.append(Case("qadd", f"""
func.func @qadd(%a: tensor<1x4x4x8xi8>, %b: tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8> {{
  %la = tosa.rescale %a {{
    input_zp = -5 : i32, output_zp = 0 : i32,
    multiplier = array<i32: 1073741824>, shift = array<i8: 10>,
    scale32 = true, double_round = true, per_channel = false
  }} : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32>
  %lb = tosa.rescale %b {{
    input_zp = 3 : i32, output_zp = 0 : i32,
    multiplier = array<i32: 1932735283>, shift = array<i8: 11>,
    scale32 = true, double_round = true, per_channel = false
  }} : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi32>
  %sum = tosa.add %la, %lb : (tensor<1x4x4x8xi32>, tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %y = tosa.rescale %sum {{
    input_zp = 0 : i32, output_zp = -9 : i32,
    multiplier = array<i32: {mo}>, shift = array<i8: {so}>,
    scale32 = true, double_round = true, per_channel = false
  }} : (tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi8>
  return %y : tensor<1x4x4x8xi8>
}}
""", [xa, xb], ya))

    return cases


# --------------------------------------------------------------------------
def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.stderr.write(" ".join(cmd) + "\n" + r.stdout + r.stderr)
        raise SystemExit("command failed: " + cmd[0])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--write-fixture", metavar="FILE",
                    help="also write the generated TOSA fixture here")
    args = ap.parse_args()

    if not os.path.exists(KEAC):
        raise SystemExit("keac not built; run bash scripts/build.sh")

    rng = np.random.default_rng(args.seed)
    cases = build_cases(rng)

    work = tempfile.mkdtemp(prefix="kea-numeric-")
    src = os.path.join(work, "fixture.mlir")
    with open(src, "w") as f:
        f.write("// Generated by tools/keac/tests/numeric_check.py --seed %d\n"
                % args.seed)
        for c in cases:
            f.write(c.mlir)
    if args.write_fixture:
        # A readable copy of what was compiled, for eyeballing or for feeding
        # to kea-opt by hand. Regenerated, never edited.
        with open(args.write_fixture, "w") as f:
            f.write("// GENERATED by numeric_check.py --seed %d. Do not edit.\n"
                    % args.seed)
            for c in cases:
                f.write(c.mlir)

    failures = 0
    for c in cases:
        base = os.path.join(work, c.name)
        run([KEAC, src, "--function", c.name, "-o", base + ".keaf",
             "--keep-intermediates"])
        xs = c.x if isinstance(c.x, list) else [c.x]
        cmd = [os.path.join(ROOT, "build", "native", "bin", "kea-sim"),
               base + ".keaf", "--quiet"]
        for i, x in enumerate(xs):
            x.tofile("%s.in%d" % (base, i))
            cmd += ["--input", "%s.input%d=%s.in%d" % (c.name, i, base, i)]
        cmd += ["--output",
                "%s=%s.out" % (_output_name(base + ".map.json"), base)]
        run(cmd)
        got = np.fromfile(base + ".out", dtype=np.int8).reshape(c.expect.shape)
        bad = int((got != c.expect).sum())
        interior = int(((c.expect > -128) & (c.expect < 127)).sum())
        if c.xfail:
            status = "XFAIL" if bad else "XPASS"
        else:
            status = "OK   " if bad == 0 else "FAIL "
        print("%s %-16s %6d values, %5d mismatches, %d interior (%.0f%%)"
              % (status, c.name, c.expect.size, bad, interior,
                 100.0 * interior / c.expect.size))
        if c.xfail:
            print("      known failure: %s" % c.xfail)
            failures += bad == 0          # an XPASS means the note is stale
        else:
            failures += bad != 0

    if not args.keep:
        import shutil
        shutil.rmtree(work)
    else:
        print("work directory:", work)
    return 1 if failures else 0


def _output_name(map_path):
    import json
    with open(map_path) as f:
        m = json.load(f)
    for t in m["tensors"]:
        if t["kind"] == "output":
            return t["name"]
    raise SystemExit("no output tensor in " + map_path)


if __name__ == "__main__":
    raise SystemExit(main())
