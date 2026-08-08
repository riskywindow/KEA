#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-layer and whole-network roofline analysis, from `kea-sim --stats-json`.

Every number here is read out of the simulator's own JSON.  Nothing is
recomputed by hand except the two derived columns the JSON does not carry
(``% of attainable`` is in the JSON as ``efficiency``; ``cycles`` share is
arithmetic on ``cycles``).

Inputs   demo/build/features.keaf, demo/build/classifier.keaf
Outputs  demo/results/roofline.json, demo/results/roofline_layers.csv,
         demo/roofline_mobilenetv2.png

    .venv/bin/python demo/roofline.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from kea_frontend.ir import KGraph  # noqa: E402

GOPS = 1e9

# categorical slots 1-3 of the validated default palette (light mode); the
# all-pairs cap is three series, which is exactly what this chart needs.
COLOR = {"conv2d": "#2a78d6", "depthwise_conv2d": "#eb6834",
         "fully_connected": "#1baf7a"}
MARKER = {"conv2d": "o", "depthwise_conv2d": "^", "fully_connected": "s"}
LABEL = {"conv2d": "conv2d (MXU)", "depthwise_conv2d": "depthwise 3x3 (DWU)",
         "fully_connected": "fully_connected (classifier)"}

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdbd6"
SURFACE = "#fcfcfb"


def collect_layers(stats, kgraph_layers, tiling):
    """Join TRACE regions to kgraph layer names.

    `-kea-tile` numbers its layers in the order it lowers the Level 1
    contractions, and emits one TRACE region per layer tagged with that number.
    The Level 1 contractions are in the same order as the kgraph's contraction
    nodes, so region tag *i* is kgraph contraction *i*.  The tiling report's
    `op` field is used as a consistency check, not as the source of the name.
    """
    rows = []
    for r in sorted(stats["regions"], key=lambda x: x["tag"]):
        tag = r["tag"]
        lay = kgraph_layers[tag]
        kind = {"kea.conv2d": "conv2d", "kea.dwconv2d": "depthwise_conv2d",
                "kea.fully_connected": "fully_connected",
                "kea.matmul": "matmul", "kea.pool": "pool"}
        if tag < len(tiling):
            want = kind.get(tiling[tag].get("op", ""), "?")
            if want != lay.op:
                raise SystemExit(
                    "region %d: tiling report says %s but kgraph layer %d is %s"
                    % (tag, want, tag, lay.op))
        rf, dram = r["roofline"], r["dram"]
        rows.append({
            "layer": tag,
            "node": lay.name,
            "op": lay.op,
            "cycles": r["cycles"],
            "ops_useful": rf["ops_useful"],
            "ops_issued": rf["ops_issued"],
            "padding_efficiency": rf["padding_efficiency"],
            "dram_bytes": dram["bytes_total"],
            "intensity_ops_per_byte": rf["intensity_ops_per_byte"],
            "achieved_gops": rf["achieved_ops_per_s"] / GOPS,
            "attainable_gops": rf["attainable_ops_per_s"] / GOPS,
            "pct_of_attainable": 100.0 * rf["efficiency"],
            "bound": "MEMORY" if rf["memory_bound"] else "COMPUTE",
            "mxu_mac_utilization": r["mxu"]["mac_utilization"],
            "dwu_mac_utilization": r["dwu"]["mac_utilization"],
        })
    return rows


def whole(stats, label):
    rf = stats["global"]["roofline"]
    return {
        "program": label,
        "cycles": stats["total_cycles"],
        "seconds": stats["seconds"],
        "ops_useful": rf["ops_useful"],
        "ops_issued": rf["ops_issued"],
        "padding_efficiency": rf["padding_efficiency"],
        "dram_bytes": rf["dram_bytes"],
        "dram_bytes_load": stats["global"]["dram"]["bytes_load"],
        "dram_bytes_store": stats["global"]["dram"]["bytes_store"],
        "intensity_ops_per_byte": rf["intensity_ops_per_byte"],
        "achieved_gops": rf["achieved_ops_per_s"] / GOPS,
        "attainable_gops": rf["attainable_ops_per_s"] / GOPS,
        "peak_gops": rf["peak_ops_per_s"] / GOPS,
        "pct_of_attainable": 100.0 * rf["efficiency"],
        "ridge_point_ops_per_byte": rf["ridge_point_ops_per_byte"],
        "bound": "MEMORY" if rf["memory_bound"] else "COMPUTE",
        "mxu_mac_utilization": stats["global"]["mxu"]["mac_utilization"],
        "dwu_mac_utilization": stats["global"]["dwu"]["mac_utilization"],
        "vpu_utilization": stats["global"]["vpu"]["utilization"],
        "dram_gb_per_s": stats["global"]["dram"]["gb_per_s"],
        "peak_dram_gb_per_s": stats["global"]["dram"]["peak_gb_per_s"],
        "matmuls": stats["global"]["mxu"]["matmuls"],
        "dwconvs": stats["global"]["dwu"]["dwconvs"],
    }


def plot(rows, nets, path, peak_gops, peak_bw, ridge):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 5.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xs = [r["intensity_ops_per_byte"] for r in rows if r["intensity_ops_per_byte"] > 0]
    xlo, xhi = min(xs) / 2.2, max(xs) * 2.2
    xlo = min(xlo, 1.0)

    # the machine's roof: min(peak compute, intensity * peak bandwidth)
    import numpy as np
    gx = np.geomspace(xlo, xhi, 400)
    gy = np.minimum(peak_gops, gx * peak_bw)
    ax.plot(gx, gy, color=INK, lw=2.0, zorder=3)
    ax.fill_between(gx, gy, 1e-3, color=INK, alpha=0.035, zorder=0)

    ax.axvline(ridge, color=GRID, lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.annotate("ridge %g ops/byte" % ridge, xy=(ridge, peak_gops * 0.028),
                xytext=(3, 0), textcoords="offset points", rotation=90,
                va="bottom", ha="left", color=INK2, fontsize=8)
    ax.annotate("peak %g GOPS int8 (256 MAC/cycle @ 1 GHz)" % peak_gops,
                xy=(xhi, peak_gops), xytext=(-4, 5), textcoords="offset points",
                ha="right", va="bottom", color=INK, fontsize=8.5)
    lx = xlo * 1.25
    ax.annotate("DRAM roof %g GB/s" % peak_bw, xy=(lx, lx * peak_bw),
                xytext=(4, -3), textcoords="offset points", rotation=33,
                rotation_mode="anchor", ha="left", va="top", color=INK2,
                fontsize=8.5)

    seen = set()
    for r in rows:
        op = r["op"]
        ax.scatter(r["intensity_ops_per_byte"], r["achieved_gops"],
                   s=46, marker=MARKER.get(op, "o"),
                   facecolor=COLOR.get(op, INK2), edgecolor=SURFACE,
                   linewidth=1.2, zorder=5,
                   label=LABEL[op] if op not in seen else None)
        seen.add(op)

    for n in nets:
        ax.scatter(n["intensity_ops_per_byte"], n["achieved_gops"], s=230,
                   marker="*", facecolor="#ffffff", edgecolor=INK,
                   linewidth=1.6, zorder=6, label=n["label"])
        ax.annotate(n["label"],
                    xy=(n["intensity_ops_per_byte"], n["achieved_gops"]),
                    xytext=(-15, -2), textcoords="offset points", ha="right",
                    va="center", fontsize=8.5, color=INK)

    # direct labels on the extremes, which is where the story is
    lo = min(rows, key=lambda r: r["intensity_ops_per_byte"])
    hi = max(rows, key=lambda r: r["achieved_gops"])
    for r, dx, dy, ha in ((lo, 7, 6, "left"), (hi, -7, 6, "right")):
        ax.annotate("%s L%d" % (r["op"].replace("depthwise_conv2d", "dwconv"),
                                r["layer"]),
                    xy=(r["intensity_ops_per_byte"], r["achieved_gops"]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=8, color=INK2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(peak_gops / 60.0, peak_gops * 2.6)
    ax.set_xlabel("arithmetic intensity  (useful ops / DRAM byte)", fontsize=9.5,
                  color=INK2)
    ax.set_ylabel("achieved  (GOPS)", fontsize=9.5, color=INK2)
    ax.set_title("KEA-1 roofline — MobileNetV2 int8, 224x224, per layer",
                 fontsize=11.5, color=INK, loc="left", pad=10)
    ax.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.35, alpha=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    leg = ax.legend(loc="lower right", frameon=True, fontsize=8.5,
                    facecolor=SURFACE, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.008, 0.012,
             "cycle-approximate; kea-sim counts are a lower bound "
             "(docs/SIMULATOR.md section 2). 1 MAC = 2 ops.",
             fontsize=7.2, color=INK2)
    fig.tight_layout(rect=(0, 0.028, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    print("wrote %s" % path)


def main() -> int:
    C.require_tools()
    C.ensure_dirs()
    feat = os.path.join(C.BUILD, "features.keaf")
    clf = os.path.join(C.BUILD, "classifier.keaf")
    for p in (feat, clf):
        if not os.path.exists(p):
            raise SystemExit("run demo/compile_mobilenetv2.py first")

    fstats_p = os.path.join(C.BUILD, "features.stats.json")
    cstats_p = os.path.join(C.BUILD, "classifier.stats.json")
    C.simulate(feat, stats_json=fstats_p)
    C.simulate(clf, stats_json=cstats_p)
    fstats, cstats = C.load_stats(fstats_p), C.load_stats(cstats_p)

    g = KGraph.load(C.KGRAPH)
    groups = [l for l in C.layer_groups(g) if l.op in C.CONTRACTIONS]
    tiling = C.tiling_report(
        os.path.join(C.BUILD, "mobilenetv2_features.tosa.mlir"),
        "mnv2_features", spm_reserve=1)

    rows = collect_layers(fstats, groups, tiling)

    # the classifier is its own program, so it contributes one whole-program
    # point rather than a region.
    crf = cstats["global"]["roofline"]
    rows.append({
        "layer": len(rows), "node": "fc#182", "op": "fully_connected",
        "cycles": cstats["total_cycles"],
        "ops_useful": crf["ops_useful"], "ops_issued": crf["ops_issued"],
        "padding_efficiency": crf["padding_efficiency"],
        "dram_bytes": crf["dram_bytes"],
        "intensity_ops_per_byte": crf["intensity_ops_per_byte"],
        "achieved_gops": crf["achieved_ops_per_s"] / GOPS,
        "attainable_gops": crf["attainable_ops_per_s"] / GOPS,
        "pct_of_attainable": 100.0 * crf["efficiency"],
        "bound": "MEMORY" if crf["memory_bound"] else "COMPUTE",
        "mxu_mac_utilization": cstats["global"]["mxu"]["mac_utilization"],
        "dwu_mac_utilization": cstats["global"]["dwu"]["mac_utilization"],
    })

    programs = [whole(fstats, "features (52 conv layers)"),
                whole(cstats, "classifier (fully_connected)")]

    # The two programs run back to back on the same machine, so the compiled
    # network's aggregate is their sum.  The global average pool between them
    # does not compile and is NOT counted here (62,720 additions, 0.01% of the
    # network's arithmetic -- see docs/RESULTS.md section 3).
    comb = {
        "program": "whole compiled network = features + classifier",
        "label": "whole network",
        "cycles": sum(p["cycles"] for p in programs),
        "ops_useful": sum(p["ops_useful"] for p in programs),
        "ops_issued": sum(p["ops_issued"] for p in programs),
        "dram_bytes": sum(p["dram_bytes"] for p in programs),
        "peak_gops": programs[0]["peak_gops"],
        "peak_dram_gb_per_s": programs[0]["peak_dram_gb_per_s"],
        "ridge_point_ops_per_byte": programs[0]["ridge_point_ops_per_byte"],
    }
    comb["seconds"] = comb["cycles"] / fstats["clock_hz"]
    comb["intensity_ops_per_byte"] = comb["ops_useful"] / comb["dram_bytes"]
    comb["achieved_gops"] = comb["ops_useful"] / comb["seconds"] / GOPS
    comb["attainable_gops"] = min(
        comb["peak_gops"],
        comb["intensity_ops_per_byte"] * comb["peak_dram_gb_per_s"])
    comb["pct_of_attainable"] = 100.0 * comb["achieved_gops"] / comb["attainable_gops"]
    comb["bound"] = ("MEMORY" if comb["intensity_ops_per_byte"]
                     < comb["ridge_point_ops_per_byte"] else "COMPUTE")
    comb["padding_efficiency"] = comb["ops_useful"] / comb["ops_issued"]
    comb["mxu_mac_utilization"] = (
        (fstats["global"]["mxu"]["macs_useful"]
         + cstats["global"]["mxu"]["macs_useful"])
        / (comb["cycles"] * 256.0))
    nets = [comb]

    region_cycles = sum(r["cycles"] for r in rows[:-1])
    out = {
        "whole_network": comb,
        "programs": programs,
        "layers": rows,
        "layer_cycle_sum": region_cycles,
        "layer_cycle_overlap_pct":
            100.0 * (region_cycles - fstats["total_cycles"]) / fstats["total_cycles"],
        "memory_bound_layers": [r["layer"] for r in rows if r["bound"] == "MEMORY"],
        "note": "TRACE regions cover every unit active inside their cycle "
                "window, and consecutive layers overlap, so per-layer cycles "
                "sum to more than the program total.",
    }
    C.write_json(os.path.join(C.RESULTS, "roofline.json"), out)

    csv = os.path.join(C.RESULTS, "roofline_layers.csv")
    cols = ["layer", "node", "op", "cycles", "ops_useful", "ops_issued",
            "padding_efficiency", "dram_bytes", "intensity_ops_per_byte",
            "achieved_gops", "attainable_gops", "pct_of_attainable", "bound",
            "mxu_mac_utilization", "dwu_mac_utilization"]
    with open(csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(
                ("%.4f" % r[c]) if isinstance(r[c], float) else str(r[c])
                for c in cols) + "\n")

    print("%-4s %-22s %-18s %9s %9s %8s %8s %7s %s"
          % ("L", "node", "op", "cycles", "DRAM B", "int.", "GOPS", "%att",
             "bound"))
    for r in rows:
        print("%-4d %-22s %-18s %9d %9d %8.2f %8.2f %6.1f%% %s"
              % (r["layer"], r["node"], r["op"], r["cycles"], r["dram_bytes"],
                 r["intensity_ops_per_byte"], r["achieved_gops"],
                 r["pct_of_attainable"], r["bound"]))
    for n in programs + nets:
        print("\n%s: %d cycles (%.3f ms), %.2f ops/byte, %.1f GOPS of %.1f "
              "attainable (%.1f%%), %s bound, %d DRAM bytes, MXU MAC util %.2f%%"
              % (n["program"], n["cycles"], n["seconds"] * 1e3,
                 n["intensity_ops_per_byte"], n["achieved_gops"],
                 n["attainable_gops"], n["pct_of_attainable"], n["bound"],
                 n["dram_bytes"], 100 * n["mxu_mac_utilization"]))

    plot(rows, nets, os.path.join(C.DEMO, "roofline_mobilenetv2.png"),
         comb["peak_gops"], comb["peak_dram_gb_per_s"],
         comb["ridge_point_ops_per_byte"])
    print("wrote demo/results/roofline.json and roofline_layers.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
