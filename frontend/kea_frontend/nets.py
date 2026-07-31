"""Network descriptions written once against :class:`~kea_frontend.builder.NetOps`.

Each ``build_*`` function is run twice with different backends: once with
:class:`~kea_frontend.builder.TorchTracer` to calibrate, once with
:class:`~kea_frontend.builder.KGraphBuilder` to emit.  They must therefore be
free of data-dependent control flow -- the op sequence and the tensor names have
to be identical on both walks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .quant import fold_conv_bn

__all__ = [
    "ConvDesc",
    "extract_mobilenetv2",
    "build_mobilenetv2",
    "TinyViTConfig",
    "make_tiny_vit_params",
    "build_tiny_vit",
]


# ---------------------------------------------------------------------------
# MobileNetV2
# ---------------------------------------------------------------------------


@dataclass
class ConvDesc:
    w: np.ndarray            # OIHW float64, BN already folded in
    b: np.ndarray            # [OC] float64
    stride: List[int]
    pad: List[int]           # TOSA order: [top, bottom, left, right]
    groups: int
    act: Optional[str]


@dataclass
class MobileNetV2Desc:
    stem: ConvDesc
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    head_conv: ConvDesc = None
    fc_w: np.ndarray = None
    fc_b: np.ndarray = None


def _fold_cna(cna, act: Optional[str]) -> ConvDesc:
    """Fold a torchvision ``Conv2dNormActivation`` (Conv, BN, [act]) into one conv."""
    conv, bn = cna[0], cna[1]
    w = conv.weight.detach().numpy().astype(np.float64)
    b = None if conv.bias is None else conv.bias.detach().numpy().astype(np.float64)
    folded = fold_conv_bn(
        w, b,
        bn.weight.detach().numpy(), bn.bias.detach().numpy(),
        bn.running_mean.detach().numpy(), bn.running_var.detach().numpy(),
        float(bn.eps),
    )
    ph, pw = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
    return ConvDesc(
        w=folded.weight, b=folded.bias,
        stride=list(conv.stride), pad=[int(ph), int(ph), int(pw), int(pw)],
        groups=int(conv.groups), act=act,
    )


def _fold_conv_bn_pair(conv, bn) -> ConvDesc:
    w = conv.weight.detach().numpy().astype(np.float64)
    b = None if conv.bias is None else conv.bias.detach().numpy().astype(np.float64)
    folded = fold_conv_bn(
        w, b,
        bn.weight.detach().numpy(), bn.bias.detach().numpy(),
        bn.running_mean.detach().numpy(), bn.running_var.detach().numpy(),
        float(bn.eps),
    )
    ph, pw = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
    return ConvDesc(
        w=folded.weight, b=folded.bias,
        stride=list(conv.stride), pad=[int(ph), int(ph), int(pw), int(pw)],
        groups=int(conv.groups), act=None,
    )


def extract_mobilenetv2(model) -> MobileNetV2Desc:
    """Pull BN-folded float weights out of a ``torchvision.models.MobileNetV2``."""
    import torch.nn as nn

    feats = model.features
    desc = MobileNetV2Desc(stem=_fold_cna(feats[0], "relu6"))

    for i in range(1, len(feats) - 1):
        blk = feats[i]
        seq = blk.conv
        convs: List[ConvDesc] = []
        # every element but the last two is a Conv2dNormActivation (relu6);
        # the tail is Conv2d + BatchNorm2d (the linear bottleneck)
        n = len(seq)
        for j in range(n - 2):
            convs.append(_fold_cna(seq[j], "relu6"))
        convs.append(_fold_conv_bn_pair(seq[n - 2], seq[n - 1]))
        desc.blocks.append({"convs": convs, "residual": bool(blk.use_res_connect)})

    desc.head_conv = _fold_cna(feats[len(feats) - 1], "relu6")
    fc = model.classifier[-1]
    desc.fc_w = fc.weight.detach().numpy().astype(np.float64)
    desc.fc_b = fc.bias.detach().numpy().astype(np.float64)
    return desc


def build_mobilenetv2(net, x, desc: MobileNetV2Desc):
    """The MobileNetV2 topology, backend-agnostic.  ``x`` is NHWC."""
    d = desc.stem
    h = net.conv2d("stem", x, d.w, d.b, d.stride, d.pad, (1, 1), d.groups, d.act)

    for bi, blk in enumerate(desc.blocks):
        inp = h
        y = h
        for ci, c in enumerate(blk["convs"]):
            y = net.conv2d(f"b{bi}_c{ci}", y, c.w, c.b, c.stride, c.pad, (1, 1), c.groups, c.act)
        if blk["residual"]:
            y = net.add(f"b{bi}_res", inp, y)
        h = y

    d = desc.head_conv
    h = net.conv2d("head", h, d.w, d.b, d.stride, d.pad, (1, 1), d.groups, d.act)
    h = net.global_avg_pool("gap", h)
    h = net.reshape("gap_flat", h, [int(h.shape[0]), 1280])
    h = net.linear("fc", h, desc.fc_w, desc.fc_b)
    return h


# ---------------------------------------------------------------------------
# Tiny ViT  --  a compiler/ISA stress test, NOT a trained model
# ---------------------------------------------------------------------------


@dataclass
class TinyViTConfig:
    image_size: int = 224
    patch: int = 16
    dim: int = 192
    depth: int = 6
    heads: int = 3
    mlp_ratio: int = 2
    num_classes: int = 1000
    seed: int = 20240601

    @property
    def tokens(self) -> int:
        g = self.image_size // self.patch
        return g * g

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads


def make_tiny_vit_params(cfg: TinyViTConfig) -> Dict[str, np.ndarray]:
    """Deterministic **random-initialised** weights.

    There is no pretraining here.  The tiny ViT exists to exercise the
    ``matmul`` / ``softmax`` / ``layernorm`` / ``table`` paths that MobileNetV2
    never touches.  Any accuracy figure computed from it is meaningless and is
    reported as such.
    """
    rng = np.random.default_rng(cfg.seed)
    d, h = cfg.dim, cfg.head_dim
    hid = cfg.dim * cfg.mlp_ratio
    p: Dict[str, np.ndarray] = {}

    def lin(name: str, out_f: int, in_f: int, gain: float = 1.0) -> None:
        p[f"{name}.w"] = (rng.standard_normal((out_f, in_f)) * (gain / np.sqrt(in_f))).astype(np.float64)
        p[f"{name}.b"] = (rng.standard_normal(out_f) * 0.02).astype(np.float64)

    p["patch.w"] = (
        rng.standard_normal((d, 3, cfg.patch, cfg.patch)) / np.sqrt(3 * cfg.patch * cfg.patch)
    ).astype(np.float64)
    p["patch.b"] = (rng.standard_normal(d) * 0.02).astype(np.float64)
    p["pos"] = (rng.standard_normal((1, cfg.tokens, d)) * 0.02).astype(np.float64)

    for i in range(cfg.depth):
        p[f"blk{i}.ln1.g"] = np.ones(d)
        p[f"blk{i}.ln1.b"] = np.zeros(d)
        p[f"blk{i}.ln2.g"] = np.ones(d)
        p[f"blk{i}.ln2.b"] = np.zeros(d)
        lin(f"blk{i}.q", d, d)
        lin(f"blk{i}.k", d, d)
        lin(f"blk{i}.v", d, d)
        lin(f"blk{i}.proj", d, d)
        lin(f"blk{i}.fc1", hid, d)
        lin(f"blk{i}.fc2", d, hid)
    p["ln_f.g"] = np.ones(d)
    p["ln_f.b"] = np.zeros(d)
    lin("head", cfg.num_classes, d)
    return p


def build_tiny_vit(net, x, cfg: TinyViTConfig, p: Dict[str, np.ndarray]):
    """Tiny ViT topology.  ``x`` is NHWC ``[1, 224, 224, 3]``.

    Deliberate simplifications, all so the graph stays inside the KEA op set:

    * no class token (avoids ``concat``); classification reads a mean over
      tokens via ``global_avg_pool``
    * fixed-function GELU via an int8 ``table`` lookup
    * the attention ``1/sqrt(head_dim)`` is folded into the QK^T rescale, so it
      costs no runtime work
    """
    t, d, nh, hd = cfg.tokens, cfg.dim, cfg.heads, cfg.head_dim
    n = int(x.shape[0])
    bh = n * nh

    h = net.conv2d("patch", x, p["patch.w"], p["patch.b"],
                   (cfg.patch, cfg.patch), (0, 0, 0, 0), (1, 1), 1, None)
    h = net.reshape("patch_tok", h, [n, t, d])

    h = net.add("pos_add", h, net.constant("pos", p["pos"]))

    inv_sqrt = 1.0 / np.sqrt(hd)
    for i in range(cfg.depth):
        s = f"blk{i}"
        n1 = net.layernorm(f"{s}.ln1", h, p[f"{s}.ln1.g"], p[f"{s}.ln1.b"])
        q = net.linear(f"{s}.q", n1, p[f"{s}.q.w"], p[f"{s}.q.b"])
        k = net.linear(f"{s}.k", n1, p[f"{s}.k.w"], p[f"{s}.k.b"])
        v = net.linear(f"{s}.v", n1, p[f"{s}.v.w"], p[f"{s}.v.b"])

        # [N,T,D] -> [N,T,H,Dh] -> [N,H,T,Dh] -> [N*H,T,Dh].  tosa.matmul is
        # strictly rank 3, so batch and head must be folded into one axis.
        def heads(nm, z, perm, shape3):
            z = net.reshape(f"{nm}r", z, [n, t, nh, hd])
            z = net.transpose(f"{nm}t", z, perm)
            return net.reshape(f"{nm}h", z, shape3)

        q = heads(f"{s}.q", q, [0, 2, 1, 3], [bh, t, hd])
        k = heads(f"{s}.k", k, [0, 2, 3, 1], [bh, hd, t])
        v = heads(f"{s}.v", v, [0, 2, 1, 3], [bh, t, hd])

        a = net.matmul(f"{s}.qk", q, k, extra_scale=inv_sqrt)
        a = net.softmax(f"{s}.sm", a, axis=-1)
        c = net.matmul(f"{s}.ctx", a, v)
        c = net.reshape(f"{s}.c4", c, [n, nh, t, hd])
        c = net.transpose(f"{s}.ct", c, [0, 2, 1, 3])
        c = net.reshape(f"{s}.cf", c, [n, t, d])
        c = net.linear(f"{s}.proj", c, p[f"{s}.proj.w"], p[f"{s}.proj.b"])
        h = net.add(f"{s}.res1", h, c)

        n2 = net.layernorm(f"{s}.ln2", h, p[f"{s}.ln2.g"], p[f"{s}.ln2.b"])
        m = net.linear(f"{s}.fc1", n2, p[f"{s}.fc1.w"], p[f"{s}.fc1.b"])
        m = net.gelu(f"{s}.act", m)
        m = net.linear(f"{s}.fc2", m, p[f"{s}.fc2.w"], p[f"{s}.fc2.b"])
        h = net.add(f"{s}.res2", h, m)

    h = net.layernorm("ln_f", h, p["ln_f.g"], p["ln_f.b"])
    # mean over tokens, expressed as an NHWC global average pool
    h = net.reshape("tok_nhwc", h, [n, t, 1, d])
    h = net.global_avg_pool("pool", h)
    h = net.reshape("pool_flat", h, [n, d])
    h = net.linear("head", h, p["head.w"], p["head.b"])
    return h
