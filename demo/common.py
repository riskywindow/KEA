#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for the KEA end-to-end demo.

Paths, subprocess wrappers, the kgraph->layer grouping used by the per-layer
A/B, and the region-tag -> layer-name mapping used by the roofline analysis.

Nothing here computes a number that is later reported; it only runs tools and
parses their output.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo")
RESULTS = os.path.join(DEMO, "results")
BUILD = os.path.join(DEMO, "build")

PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
KEAC = os.path.join(ROOT, "build", "native", "bin", "keac")
KEA_SIM = os.path.join(ROOT, "build", "native", "bin", "kea-sim")
KEA_DIS = os.path.join(ROOT, "build", "native", "bin", "kea-dis")
MLIR_OPT = os.environ.get("MLIR_OPT", "/usr/local/opt/llvm/bin/mlir-opt")

KGRAPH = os.path.join(ROOT, "models", "mobilenetv2_int8.kgraph.json")
GOLDEN = os.path.join(ROOT, "frontend", "testdata",
                      "golden_io_mobilenetv2_int8.npz")

sys.path.insert(0, os.path.join(ROOT, "frontend"))

#: ops that open a `-kea-tile` layer (and therefore a TRACE region)
CONTRACTIONS = ("conv2d", "depthwise_conv2d", "fully_connected", "matmul")


def ensure_dirs() -> None:
    for d in (RESULTS, BUILD):
        os.makedirs(d, exist_ok=True)


def require_tools() -> None:
    missing = [p for p in (KEAC, KEA_SIM, PYTHON) if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "missing tools: %s\nrun: bash scripts/build_compiler.sh && "
            "bash scripts/build.sh" % ", ".join(missing))


class Run:
    """The result of a subprocess, with the pieces the demo actually reports."""

    def __init__(self, cmd: Sequence[str], proc: subprocess.CompletedProcess):
        self.cmd = list(cmd)
        self.returncode = proc.returncode
        self.stdout = proc.stdout
        self.stderr = proc.stderr

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def first_error(self) -> str:
        for line in (self.stderr + self.stdout).splitlines():
            if "error:" in line:
                return line.strip()
        return (self.stderr or self.stdout).strip().splitlines()[-1:] and \
            (self.stderr or self.stdout).strip().splitlines()[-1] or ""


def run(cmd: Sequence[str], check: bool = True, **kw) -> Run:
    p = subprocess.run(list(cmd), capture_output=True, text=True, **kw)
    r = Run(cmd, p)
    if check and not r.ok:
        sys.stderr.write("$ %s\n%s%s\n" % (" ".join(cmd), p.stdout, p.stderr))
        raise SystemExit("command failed: %s" % cmd[0])
    return r


# ---------------------------------------------------------------------------
# emitting / compiling
# ---------------------------------------------------------------------------


_GRAPHS: Dict[str, object] = {}


def load_kgraph(path: str = KGRAPH):
    """Load (and memoize) a `.kgraph.json` + its `.npz`.

    Memoized because the A/B sweep emits a hundred node slices out of the same
    3.5 MB graph, and re-reading it each time dominates the run.
    """
    if path not in _GRAPHS:
        from kea_frontend.ir import KGraph
        _GRAPHS[path] = KGraph.load(path)
    return _GRAPHS[path]


def emit_tosa(out_path: str, function: str, first: Optional[int] = None,
              last: Optional[int] = None, kgraph: str = KGRAPH,
              cache: bool = False):
    """Emit TOSA for a node slice.  ``cache`` skips the work when the output is
    already newer than the graph."""
    if cache and os.path.exists(out_path) and \
            os.path.getmtime(out_path) > os.path.getmtime(kgraph):
        return None
    from kea_frontend.tosa_emit import emit_module

    g = load_kgraph(kgraph)
    lo = 0 if first is None else first
    hi = len(g.nodes) - 1 if last is None else last
    res = emit_module(g, function=function, nodes=g.nodes[lo:hi + 1])
    with open(out_path, "w") as f:
        f.write(res.text)
    return res


def compile_keaf(mlir_path: str, function: str, out_keaf: str,
                 schedule: bool = False, spm_reserve: Optional[int] = None,
                 imem_budget: Optional[int] = None,
                 schedule_mode: Optional[str] = None,
                 check: bool = True, emit: Optional[str] = None) -> Run:
    """Run `keac`.

    Everything is at the compiler's defaults unless a keyword says otherwise:
    `spm-reserve-factor = 1`, `imem-budget = 20480`, `-kea-schedule mode=auto`.
    `schedule_mode` forces the mode and implies `--schedule`; it exists so the
    A/B can measure `mode=serial`, which is the pass's own control.
    """
    cmd = [KEAC, mlir_path, "--function", function, "-o", out_keaf,
           "--keep-intermediates"]
    if schedule_mode is not None:
        # `--schedule` hard-codes a bare `--kea-schedule`; forcing a mode needs
        # the explicit pass list, so build the whole thing here instead.
        opts = []
        if spm_reserve is not None:
            opts.append("spm-reserve-factor=%d" % spm_reserve)
        if imem_budget is not None:
            opts.append("imem-budget=%d" % imem_budget)
        tile = "--kea-tile" + ("=" + " ".join(opts) if opts else "")
        cmd += ["--pipeline",
                "--tosa-to-kea --kea-fuse %s --kea-schedule=mode=%s "
                "--kea-alloc" % (tile, schedule_mode)]
    else:
        if spm_reserve is not None:
            cmd += ["--spm-reserve", str(spm_reserve)]
        if imem_budget is not None:
            cmd += ["--imem-budget", str(imem_budget)]
        if schedule:
            cmd += ["--schedule"]
    if emit:
        cmd += ["--emit", emit]
    return run(cmd, check=check)


_INSTR_RE = re.compile(r"^\s+(MXU|DWU|VPU|DMA0|DMA1|CTRL)\s+[A-Z_]+")


def count_instructions(kasm_path: str) -> int:
    """Instructions in a `.kasm`, i.e. the IMEM footprint."""
    with open(kasm_path) as f:
        return sum(1 for line in f if _INSTR_RE.match(line))


def simulate(keaf: str, stats_json: Optional[str] = None,
             inputs: Optional[Dict[str, str]] = None,
             outputs: Optional[Dict[str, str]] = None,
             check: bool = True) -> Run:
    cmd = [KEA_SIM, keaf, "--quiet"]
    if stats_json:
        cmd += ["--stats-json", stats_json]
    for name, path in (inputs or {}).items():
        cmd += ["--input", "%s=%s" % (name, path)]
    for name, path in (outputs or {}).items():
        cmd += ["--output", "%s=%s" % (name, path)]
    return run(cmd, check=check)


def tensor_table(keaf: str) -> List[Dict[str, str]]:
    """`kea-sim --list-tensors`, parsed."""
    r = run([KEA_SIM, keaf, "--list-tensors"])
    rows = []
    for line in r.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 5:
            rows.append({"name": parts[0], "kind": parts[1], "dtype": parts[2],
                         "dram_offset": int(parts[3]), "size": int(parts[4])})
    return rows


def io_names(keaf: str) -> Tuple[List[str], List[str]]:
    rows = tensor_table(keaf)
    return ([r["name"] for r in rows if r["kind"] == "input"],
            [r["name"] for r in rows if r["kind"] == "output"])


# ---------------------------------------------------------------------------
# kgraph -> layers
# ---------------------------------------------------------------------------


class Layer:
    """One `-kea-tile` layer: a contraction plus its fused epilogue nodes."""

    def __init__(self, index: int, first: int, last: int, nodes) -> None:
        self.index = index
        self.first = first
        self.last = last
        self.nodes = nodes
        self.op = nodes[0].op
        self.name = nodes[0].name

    def __repr__(self) -> str:  # pragma: no cover
        return "Layer(%d, %s, %s..%s)" % (self.index, self.op, self.first,
                                          self.last)


def layer_groups(g) -> List[Layer]:
    """Group the graph's nodes the way `-kea-fuse` + `-kea-tile` do.

    A group starts at a contraction and runs to just before the next one, so
    the trailing rescale / clamp / residual sandwich lands in the group whose
    contraction absorbs it.  Nodes before the first contraction, and pooling
    nodes (which open their own layer), start their own group.
    """
    groups: List[Layer] = []
    start = None
    for i, n in enumerate(g.nodes):
        if n.op in CONTRACTIONS or n.op in ("avg_pool2d", "global_avg_pool"):
            if start is not None:
                groups.append(Layer(len(groups), start, i - 1,
                                    g.nodes[start:i]))
            start = i
    if start is not None:
        groups.append(Layer(len(groups), start, len(g.nodes) - 1,
                            g.nodes[start:]))
    return groups


def tiling_report(mlir_path: str, function: str,
                  imem_budget: Optional[int] = None) -> List[Dict]:
    """`-kea-tile=report-tiles=true`'s `kea.tiling`, as a list of dicts.

    This is the only way to learn which `-kea-tile` layer id corresponds to
    which op kind, which is what turns a TRACE region tag into a layer name.
    Runs at the pass defaults so the tiling it reports is the shipped one.
    """
    kea_opt = os.path.join(ROOT, "build", "compiler", "bin", "kea-opt")
    out = os.path.join(BUILD, os.path.basename(mlir_path) + ".tiled.mlir")
    opts = ["report-tiles=true"]
    if imem_budget is not None:
        opts.append("imem-budget=%d" % imem_budget)
    run([kea_opt, mlir_path, "--tosa-to-kea", "--kea-fuse",
         "--kea-tile=" + " ".join(opts), "-o", out])
    text = open(out).read()
    m = re.search(r"kea\.tiling = \[(.*?)\]\}?\s*\{", text, re.S)
    if not m:
        m = re.search(r"kea\.tiling = \[(.*)", text)
    body = m.group(1)
    entries = []
    for chunk in re.findall(r"\{([^{}]*)\}", body):
        d: Dict[str, object] = {}
        for k, v in re.findall(r"(\w+) = ([^,]+?)(?: : i64)?(?=,|$)", chunk):
            d[k] = v.strip().strip('"')
        if "layer" in d:
            entries.append(d)
    return entries


def unsound_regions(stats: Dict) -> List[int]:
    """TRACE regions whose `begin` was not issued where its layer starts.

    `-kea-schedule` hoists a `kea.trace` begin marker to the top of its queue
    when nothing depends on it, so in a scheduled build some regions open at
    cycle ~0 and their `cycles` measure from program start instead of from the
    layer.  A region is called unsound here when it strictly contains another
    region that starts earlier -- i.e. `depth > 0` for a region that is not
    itself nested by construction.  Concretely: any region that begins before
    the region with the previous tag ends *and* spans it entirely.

    Returns the tags, so a caller can refuse to report those layers rather
    than quietly summing nonsense.
    """
    rs = sorted(stats.get("regions", []), key=lambda r: r["tag"])
    bad = []
    for i, r in enumerate(rs):
        earlier = [q for q in rs[:i] if q["begin_cycle"] > r["begin_cycle"]]
        if earlier:
            bad.append(r["tag"])
    return bad


def load_stats(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    ensure_dirs()
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
        f.write("\n")
