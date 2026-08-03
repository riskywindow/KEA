#!/usr/bin/env python3
"""Measure -kea-schedule on kea-sim: the A/B that docs/SCHEDULING.md reports.

Compiles one model twice -- `-kea-schedule=mode=serial` (the unscheduled
program executed as a sequential program: one DMA engine, a handshake at every
cross-queue adjacency, nothing overlapped) and `-kea-schedule=mode=overlap`
(the real schedule) -- runs both on `kea-sim`, and prints the cycle counts and
the per-unit busy / stall breakdown side by side.

Two ways to get from Level 2 IR to `.kasm`:

  --emitter=translate  `kea-translate --sync=none`, the real backend. Default
                       when the binary exists. `--sync=none` matters: it emits
                       exactly the semaphores `-kea-schedule` put in the IR
                       rather than inventing its own.
  --emitter=builtin    a ~150-line transcription of docs/ASSEMBLY.md §5 kept
                       here so the measurement stays reproducible if
                       kea-translate is unavailable or in flux. It writes the
                       DRAM constant blob as zeros -- cycle counts do not
                       depend on constant values -- so its numbers are cycle
                       counts and never numerical results.

Usage:
    kea-schedule-measure.py MODEL.mlir [--func NAME] [--tile-opts OPTS]
                            [--emitter {auto,translate,builtin}]
                            [--kea-opt PATH] [--bin-dir PATH] [--keep DIR]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------------------------------
# A very small reader for the subset of MLIR that Level 2 is
# --------------------------------------------------------------------------

ALLOC_RE = re.compile(r"^\s*(%\S+) = kea\.alloc\b(.*)$")
OP_RE = re.compile(r"^\s*kea\.(\w+)\b(.*)$")
FUNC_RE = re.compile(r"^\s*func\.func @(\w+)")
BUF_RE = re.compile(r"!kea\.buffer<(\d+)x(i8|i32),\s*(A|W|ACC|DRAM)>")


def split_attrs(text):
    """Pull `key = value` pairs out of the last balanced {...} group on a line.

    Nested `<...>` (quant attributes) and `[...]` are skipped over, which is
    all Level 2 needs: no instruction op carries a nested attribute.
    """
    start = text.find("{")
    if start < 0:
        return {}
    depth, end = 0, len(text)
    for i in range(start, len(text)):
        if text[i] in "{<[(":
            depth += 1
        elif text[i] in "}>])":
            depth -= 1
            if depth == 0:
                end = i
                break
    body, out, i, n = text[start + 1:end], {}, 0, 0
    key = None
    tok = ""
    depth = 0
    for ch in body:
        if ch in "<[(":
            depth += 1
        elif ch in ">])":
            depth -= 1
        if ch == "," and depth == 0:
            k, _, v = tok.partition("=")
            out[k.strip()] = v.strip()
            tok = ""
            continue
        tok += ch
    if tok.strip():
        k, _, v = tok.partition("=")
        out[k.strip()] = v.strip()
    return out


def attr_int(attrs, key, default=None):
    v = attrs.get(key)
    if v is None:
        if default is None:
            raise KeyError(key)
        return default
    return int(v.split(":")[0].strip())


def attr_str(attrs, key, default=None):
    v = attrs.get(key)
    if v is None:
        return default
    return v.strip().strip('"')


def operands_of(text):
    """SSA operand names of an instruction, in source order."""
    head = text.split("{")[0] if "{" in text else text.split(" : ")[0]
    return re.findall(r"%[A-Za-z0-9_]+", head)


class Buf:
    def __init__(self, name, addr, space, size, role):
        self.name, self.addr, self.space, self.size, self.role = (
            name, addr, space, size, role)


def parse_function(ir, want):
    """Return (bufs_by_ssa, instructions, dram_layout) for one function."""
    lines = ir.splitlines()
    start = None
    for i, ln in enumerate(lines):
        m = FUNC_RE.match(ln)
        if m and (want is None or m.group(1) == want):
            start = i
            want = m.group(1)
            break
    if start is None:
        sys.exit("no matching func.func in the IR")

    layout = {}
    m = re.search(r"kea\.dram_layout = \{([^}]*)\}", lines[start])
    if m:
        for part in m.group(1).split(","):
            k, _, v = part.partition("=")
            layout[k.strip()] = int(v.split(":")[0].strip())

    bufs, instrs = {}, []
    for ln in lines[start + 1:]:
        if ln.startswith("  }") or re.match(r"^\s*(func\.)?return", ln):
            break
        m = ALLOC_RE.match(ln)
        if m:
            ssa, rest = m.group(1), m.group(2)
            attrs = split_attrs(rest)
            bt = BUF_RE.search(rest)
            bufs[ssa] = Buf(attr_str(attrs, "name"), attr_int(attrs, "addr", 0),
                            bt.group(3), int(bt.group(1)),
                            attr_str(attrs, "role"))
            continue
        m = OP_RE.match(ln)
        if m:
            instrs.append((m.group(1), m.group(2)))
    return want, bufs, instrs, layout


# --------------------------------------------------------------------------
# Level 2 -> .kasm  (docs/ASSEMBLY.md §5)
# --------------------------------------------------------------------------

SPACE_PREFIX = {"A": "a", "W": "w", "ACC": "acc"}
SPM_SPACE = {"A": "SPM_A", "W": "SPM_W"}


def to_kasm(bufs, instrs):
    out = ['.arch "KEA-1"', ".isa_revision 1", ".entry 0", ""]
    regions = sorted({int(t[1].split()[1]) for t in instrs if t[0] == "trace"})
    for r in regions:
        out.append('.region %d, "layer%d"' % (r, r))
    if regions:
        out.append("")

    def addr(ssa, disp):
        b = bufs[ssa]
        return "%s:%d" % (SPACE_PREFIX[b.space], b.addr + disp)

    def dram(ssa, disp):
        b = bufs[ssa]
        return "@%s" % b.name if disp == 0 else "@%s+%d" % (b.name, disp)

    def emit(unit, mnem, fields):
        out.append("  %-4s  %-6s  %s" % (unit, mnem, ", ".join(fields)))

    for opname, rest in instrs:
        a = split_attrs(rest)
        ops = operands_of(rest)
        unit = attr_str(a, "unit") or attr_str(a, "kea.unit")
        dtype = "int4" if "int4" in a else "int8"

        if opname in ("dma_load", "dma_store"):
            load = opname == "dma_load"
            d, s = (ops[0], ops[1]) if load else (ops[1], ops[0])
            emit(unit, "DMA_LD" if load else "DMA_ST", [
                "spm_space=" + SPM_SPACE[bufs[s].space],
                "dram_addr=" + dram(d, attr_int(a, "dram_addr")),
                "spm_addr=" + addr(s, attr_int(a, "spm_addr")),
                "len0=%d" % attr_int(a, "len0"), "n1=%d" % attr_int(a, "n1"),
                "n2=%d" % attr_int(a, "n2"),
                "dram_s1=%d" % attr_int(a, "dram_s1"),
                "dram_s2=%d" % attr_int(a, "dram_s2"),
                "spm_s1=%d" % attr_int(a, "spm_s1"),
                "spm_s2=%d" % attr_int(a, "spm_s2")])
        elif opname == "load_w":
            emit(unit, "LOAD_W", [
                "w_addr=" + addr(ops[0], attr_int(a, "w_addr")),
                "w_row_stride=%d" % attr_int(a, "w_row_stride"),
                "k_rows=%d" % attr_int(a, "k_rows"),
                "n_cols=%d" % attr_int(a, "n_cols"),
                "bank=%d" % attr_int(a, "bank"), "dtype=" + dtype])
        elif opname == "mm":
            emit(unit, "MATMUL", [
                "a_addr=" + addr(ops[0], attr_int(a, "a_addr")),
                "a_inner_stride=%d" % attr_int(a, "a_inner_stride"),
                "a_outer_stride=%d" % attr_int(a, "a_outer_stride"),
                "m_inner=%d" % attr_int(a, "m_inner"),
                "m_outer=%d" % attr_int(a, "m_outer"),
                "acc_addr=" + addr(ops[1], attr_int(a, "acc_addr")),
                "acc_inner_stride=%dw" % attr_int(a, "acc_inner_stride"),
                "acc_outer_stride=%dw" % attr_int(a, "acc_outer_stride"),
                "bank=%d" % attr_int(a, "bank"),
                "acc_mode=" + ("accumulate" if "accumulate" in a else "overwrite"),
                "dtype=" + dtype])
        elif opname == "dwconv":
            emit(unit, "DWCONV", [
                "a_addr=" + addr(ops[0], attr_int(a, "a_addr")),
                "w_addr=" + addr(ops[1], attr_int(a, "w_addr")),
                "acc_addr=" + addr(ops[2], attr_int(a, "acc_addr")),
                "out_h=%d" % attr_int(a, "out_h"),
                "out_w=%d" % attr_int(a, "out_w"),
                "channels=%d" % attr_int(a, "channels"),
                "a_row_stride=%d" % attr_int(a, "a_row_stride"),
                "a_pix_stride=%d" % attr_int(a, "a_pix_stride"),
                "kernel=%d" % attr_int(a, "kernel"),
                "stride=%d" % attr_int(a, "stride"),
                "acc_mode=" + ("accumulate" if "accumulate" in a else "overwrite")])
        elif opname == "vquant":
            emit(unit, "VQUANT", [
                "acc_addr=" + addr(ops[0], attr_int(a, "acc_addr")),
                "out_addr=" + addr(ops[1], attr_int(a, "out_addr")),
                "qparam_addr=" + addr(ops[2], attr_int(a, "qparam_addr")),
                "num_pixels=%d" % attr_int(a, "num_pixels"),
                "channels=%d" % attr_int(a, "channels"),
                "acc_pix_stride=%dw" % attr_int(a, "acc_pix_stride"),
                "out_pix_stride=%d" % attr_int(a, "out_pix_stride"),
                "out_zp=%d" % attr_int(a, "out_zp"),
                "clamp_lo=%d" % attr_int(a, "clamp_lo"),
                "clamp_hi=%d" % attr_int(a, "clamp_hi"), "dtype=" + dtype])
        elif opname == "vadd":
            emit(unit, "VADD", [
                "a_addr=" + addr(ops[0], attr_int(a, "a_addr")),
                "b_addr=" + addr(ops[1], attr_int(a, "b_addr")),
                "out_addr=" + addr(ops[2], attr_int(a, "out_addr")),
                "param_addr=" + addr(ops[3], attr_int(a, "param_addr")),
                "num_elems=%d" % attr_int(a, "num_elems"),
                "clamp_lo=%d" % attr_int(a, "clamp_lo"),
                "clamp_hi=%d" % attr_int(a, "clamp_hi")])
        elif opname == "vpool":
            emit(unit, "VPOOL", [
                "mode=" + ("avg" if "avg" in a else "max"),
                "in_addr=" + addr(ops[0], attr_int(a, "in_addr")),
                "out_addr=" + addr(ops[1], attr_int(a, "out_addr")),
                "out_h=%d" % attr_int(a, "out_h"),
                "out_w=%d" % attr_int(a, "out_w"),
                "channels=%d" % attr_int(a, "channels"),
                "kh=%d" % attr_int(a, "kh"), "kw=%d" % attr_int(a, "kw"),
                "stride_h=%d" % attr_int(a, "stride_h"),
                "stride_w=%d" % attr_int(a, "stride_w"),
                "in_row_stride=%d" % attr_int(a, "in_row_stride"),
                "out_row_stride=%d" % attr_int(a, "out_row_stride")])
        elif opname == "vcopy":
            fill = "fill" in a
            dst = ops[-1]
            src = ops[0] if len(ops) > 1 else None
            emit(unit, "VCOPY", [
                "mode=" + ("fill" if fill else "copy"),
                "src_space=" + (SPM_SPACE[bufs[src].space] if src else "SPM_A"),
                "dst_space=" + SPM_SPACE[bufs[dst].space],
                "src_addr=" + (addr(src, attr_int(a, "src_addr")) if src else "a:0"),
                "dst_addr=" + addr(dst, attr_int(a, "dst_addr")),
                "row_bytes=%d" % attr_int(a, "row_bytes"),
                "rows=%d" % attr_int(a, "rows"),
                "src_row_stride=%d" % attr_int(a, "src_row_stride"),
                "dst_row_stride=%d" % attr_int(a, "dst_row_stride"),
                "fill_value=%d" % attr_int(a, "fill_value", 0)])
        elif opname == "signal":
            emit(unit, "SIGNAL", ["event=%s" % rest.split()[0],
                                  "inc=%d" % attr_int(a, "value", 1)])
        elif opname == "wait":
            emit(unit, "WAIT", ["event=%s" % rest.split()[0],
                                "threshold=%d" % attr_int(a, "value", 1)])
        elif opname == "trace":
            kind = rest.split()[0].strip('"')
            emit(unit, "TRACE", ["kind=%s" % kind, "tag=%s" % rest.split()[1],
                                 "payload=%d" % attr_int(a, "payload", 0)])
        elif opname == "halt":
            emit("CTRL", "HALT", ["exit_code=%d" % attr_int(a, "exit_code", 0)])
        else:
            sys.exit("harness does not know kea.%s" % opname)
    return "\n".join(out) + "\n"


def to_map(bufs, layout):
    syms, tensors = [], []
    seen = set()
    for b in bufs.values():
        if b.space != "DRAM" or b.name in seen:
            continue
        seen.add(b.name)
        if b.role in ("input", "output"):
            tensors.append({"name": b.name, "kind": b.role,
                            "index": len([t for t in tensors
                                          if t["kind"] == b.role]),
                            "offset": b.addr, "size_bytes": b.size,
                            "shape": [b.size], "dtype": "int8",
                            "layout": "NHWC", "scale": 1.0, "zero_point": 0})
        else:
            syms.append({"name": b.name, "offset": b.addr, "size": b.size})
    return {
        "arch": "KEA-1", "isa_revision": 1, "entry_pc": 0,
        "dram": {"total_bytes": layout["total_bytes"],
                 "const_offset": layout["const_offset"],
                 "const_bytes": layout["const_bytes"],
                 "io_offset": layout["io_offset"],
                 "io_bytes": layout["io_bytes"],
                 "scratch_offset": layout["scratch_offset"],
                 "scratch_bytes": layout["scratch_bytes"],
                 "alignment": layout["alignment"]},
        "symbols": sorted(syms, key=lambda s: s["offset"]),
        "tensors": tensors,
    }


# --------------------------------------------------------------------------

PIPE = ("builtin.module(func.func(tosa-to-kea,kea-fuse,kea-tile{%s},"
        "kea-schedule{mode=%s},kea-alloc))")


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        sys.exit("FAILED: %s\n%s\n%s" % (" ".join(cmd), p.stdout, p.stderr))
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--func", default=None)
    ap.add_argument("--tile-opts", default="")
    ap.add_argument("--kea-opt",
                    default=os.path.join(REPO, "build/compiler/bin/kea-opt"))
    ap.add_argument("--bin-dir", default=os.path.join(REPO, "build/native/bin"))
    ap.add_argument("--kea-translate",
                    default=os.path.join(REPO,
                                         "build/compiler/bin/kea-translate"))
    ap.add_argument("--emitter", default="auto",
                    choices=["auto", "translate", "builtin"])
    ap.add_argument("--keep", default=None)
    args = ap.parse_args()
    if args.emitter == "auto":
        args.emitter = ("translate" if os.path.exists(args.kea_translate)
                        else "builtin")

    workdir = args.keep or tempfile.mkdtemp(prefix="kea-sched-")
    os.makedirs(workdir, exist_ok=True)
    results = {}
    for mode in ("serial", "overlap"):
        ir = run([args.kea_opt, args.model, "--pass-pipeline",
                  PIPE % (args.tile_opts, mode)])
        fn, bufs, instrs, layout = parse_function(ir, args.func)
        base = os.path.join(workdir, "%s.%s" % (fn, mode))
        with open(base + ".mlir", "w") as f:
            f.write(ir)
        if args.emitter == "translate":
            run([args.kea_translate, base + ".mlir", "--function=" + fn,
                 "--sync=none", "--emit-kasm=" + base + ".kasm",
                 "--emit-map=" + base + ".map.json",
                 "--emit-const=" + base + ".const.bin"])
        else:
            with open(base + ".kasm", "w") as f:
                f.write(to_kasm(bufs, instrs))
            with open(base + ".map.json", "w") as f:
                json.dump(to_map(bufs, layout), f, indent=2)
            with open(base + ".const.bin", "wb") as f:
                f.write(b"\0" * layout["const_bytes"])
        run([os.path.join(args.bin_dir, "kea-as"), base + ".kasm",
             "--map", base + ".map.json", "-o", base + ".keaf",
             "--const", base + ".const.bin"])
        run([os.path.join(args.bin_dir, "kea-sim"), base + ".keaf",
             "--stats-json", base + ".stats.json", "--quiet"])
        with open(base + ".stats.json") as f:
            results[mode] = json.load(f)
        results[mode]["_instrs"] = len(instrs)

    print("model     : %s  (func @%s)" % (args.model, fn))
    print("emitter   : %s" % args.emitter)
    print("workdir   : %s" % workdir)
    print()
    hdr = "%-28s %12s %12s %9s" % ("", "unscheduled", "scheduled", "speedup")
    print(hdr)
    print("-" * len(hdr))
    a, b = results["serial"]["global"], results["overlap"]["global"]

    def row(label, x, y, ratio=False):
        s = "%-28s %12s %12s" % (label, x, y)
        if ratio and y:
            s += " %9.3f" % (x / float(y))
        print(s)

    row("instructions", results["serial"]["_instrs"],
        results["overlap"]["_instrs"])
    row("CYCLES", a["cycles"], b["cycles"], True)
    print()
    print("%-28s %12s %12s" % ("per unit: busy / sem-stall / res-stall / idle",
                               "", ""))
    for u in ("MXU", "DWU", "VPU", "DMA0", "DMA1"):
        ua, ub = a["units"][u], b["units"][u]
        fmt = lambda d: "%d/%d/%d/%d" % (d["busy_cycles"],
                                         d["stall_semaphore_cycles"],
                                         d["stall_resource_cycles"],
                                         d["idle_cycles"])
        print("  %-26s %12s %12s" % (u, fmt(ua), fmt(ub)))
        print("  %-26s %12s %12s" % ("  max queue depth",
                                     ua["max_queue_depth"],
                                     ub["max_queue_depth"]))
    row("dispatcher stall cycles", a["dispatcher"]["stall_cycles"],
        b["dispatcher"]["stall_cycles"])
    row("DRAM bytes", a["dram"]["bytes_total"], b["dram"]["bytes_total"])
    print()
    print(json.dumps({"unscheduled_cycles": a["cycles"],
                      "scheduled_cycles": b["cycles"],
                      "speedup": round(a["cycles"] / float(b["cycles"]), 4)}))
    if args.keep is None:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
