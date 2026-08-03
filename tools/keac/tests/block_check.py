#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Whole-block numerical check for tests/mlir/tosa/mobilenet_block.mlir.

An independent numpy reference for the MobileNetV2 inverted residual in that
file, written from the TOSA graph and docs/QUANTIZATION.md §1/§3.  It shares
nothing with the compiler except the frontend's normative `apply_scale_32`,
which QUANTIZATION.md §10 says every implementation must agree with bit for
bit.

    block_check.py write <in.bin> <expected.bin> [--seed N]
    block_check.py compare <expected.bin> <got.bin>

CAVEAT, stated because it limits what a match proves: every weight in that
fixture is a splat (all 2s / all 1s / all 3s), so a match does not exercise the
MXU tile ORDERING -- every permutation of an all-2s array is an all-2s array.
`numeric_check.py` is the test that pins the layouts, with distinct weights.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "frontend"))
from kea_frontend.apply_scale import apply_scale_32  # noqa: E402


def rescale(acc, mult, shift, in_zp, out_zp, per_channel, to_int8=True):
    acc = acc.astype(np.int64)
    out = np.empty(acc.shape, dtype=np.int64)
    flat = acc.reshape(-1, acc.shape[-1])
    o = out.reshape(-1, acc.shape[-1])
    for c in range(acc.shape[-1]):
        m = mult[c] if per_channel else mult[0]
        s = shift[c] if per_channel else shift[0]
        for i in range(flat.shape[0]):
            o[i, c] = apply_scale_32(np.int32(int(flat[i, c]) - in_zp),
                                     np.int32(m), np.int32(s), True)
    out = out + out_zp
    if to_int8:
        return np.clip(out, -128, 127).astype(np.int8)
    return out.astype(np.int32)


def conv2d(x, w, b, in_zp, pad, stride):
    N, H, W, IC = x.shape
    OC, KH, KW, _ = w.shape
    pt, pb, pl, pr = pad
    xp = np.full((N, H + pt + pb, W + pl + pr, IC), in_zp, dtype=np.int32)
    xp[:, pt:pt + H, pl:pl + W, :] = x.astype(np.int32)
    xc = xp - in_zp
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


def depthwise(x, w, b, in_zp, pad, stride):
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


EXP_M = [1073741824, 1181116006, 1288490189, 1395864371, 1503238553,
         1610612736, 1717986918, 1825361101, 1932735283, 2040109466,
         1073741824, 1181116006, 1288490189, 1395864371, 1503238553,
         1610612736, 1717986918, 1825361101, 1932735283, 2040109466,
         1073741824, 1181116006, 1288490189, 1395864371]
EXP_S = [36, 36, 36, 36, 37, 37, 37, 37, 36, 36, 36, 36,
         37, 37, 37, 37, 36, 36, 36, 36, 37, 37, 37, 37]
DW_M = [1503238553, 1610612736, 1717986918, 1825361101, 1932735283,
        2040109466, 1073741824, 1181116006, 1288490189, 1395864371,
        1503238553, 1610612736, 1717986918, 1825361101, 1932735283,
        2040109466, 1073741824, 1181116006, 1288490189, 1395864371,
        1503238553, 1610612736, 1717986918, 1825361101]
DW_S = [37, 37, 37, 37, 36, 36, 36, 36, 37, 37, 37, 37,
        36, 36, 36, 36, 37, 37, 37, 37, 36, 36, 36, 36]


def block(x):
    # 1. expand, 1x1, 4 -> 24, then per-channel rescale and the ReLU6 clamp.
    acc = conv2d(x, np.full((24, 1, 1, 4), 2, np.int8),
                 np.full((24,), 128, np.int32), -5, (0, 0, 0, 0), (1, 1))
    exp = rescale(acc, EXP_M, EXP_S, 0, -128, True)

    # 2. depthwise 3x3 s1 pad 1, zero point -128 in the halo.
    acc = depthwise(exp, np.full((3, 3, 24, 1), 1, np.int8),
                    np.full((24,), 64, np.int32), -128, (1, 1, 1, 1), (1, 1))
    dw = rescale(acc, DW_M, DW_S, 0, -128, True)

    # 3. project, 1x1, 24 -> 4, linear bottleneck (no activation).
    acc = conv2d(dw, np.full((4, 1, 1, 24), 3, np.int8),
                 np.full((4,), -32, np.int32), -128, (0, 0, 0, 0), (1, 1))
    proj = rescale(acc, [1288490189, 1395864371, 1503238553, 1610612736],
                   [37, 37, 38, 38], 0, -5, True)

    # 4. residual add, TOSA style: both operands into a common i32 domain.
    idn = rescale(x.astype(np.int32), [1073741824], [10], -5, 0, False, False)
    res = rescale(proj.astype(np.int32), [1610612736], [11], -5, 0, False, False)
    ssum = np.clip(idn.astype(np.int64) + res.astype(np.int64),
                   -2**31, 2**31 - 1).astype(np.int32)
    return rescale(ssum, [1503238553], [40], 0, -5, False)


def main(argv):
    if len(argv) >= 4 and argv[1] == "write":
        seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 0
        rng = np.random.default_rng(seed)
        x = rng.integers(-128, 128, size=(1, 8, 8, 4),
                         dtype=np.int64).astype(np.int8)
        x.tofile(argv[2])
        block(x).tofile(argv[3])
        return 0
    if len(argv) == 4 and argv[1] == "compare":
        e = np.fromfile(argv[2], dtype=np.int8)
        g = np.fromfile(argv[3], dtype=np.int8)
        if e.shape != g.shape:
            print("size mismatch: expected %d bytes, got %d" % (e.size, g.size))
            return 1
        bad = int((e != g).sum())
        interior = int(((e > -128) & (e < 127)).sum())
        print("block: %d values, %d mismatches, %d not clamped"
              % (e.size, bad, interior))
        return 1 if bad else 0
    sys.stderr.write(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
