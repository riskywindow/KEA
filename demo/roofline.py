#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-layer and whole-network roofline analysis, from `kea-sim --stats-json`.

Every number here is read out of the simulator's own JSON.  Nothing is
recomputed by hand except the two derived columns the JSON does not carry
(``% of attainable`` is in the JSON as ``efficiency``; ``cycles`` share is
arithmetic on ``cycles``).

**Per-layer numbers come from the scheduled build**, which is what ships.  That
was not always safe: `-kea-schedule` used to hoist `kea.trace` begin markers
away from their region, so some regions opened at cycle ~0 and 52 of them
summed to 7.4x the program length.  It is fixed, but the check that caught it
stays: `common.unsound_regions` is run over **every** build here and the script
aborts rather than reporting a per-layer table it cannot vouch for.

Inputs   demo/build/{features,featpool,classifier}.keaf and their
         _unsched counterparts
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


def collect_layers(stats, kgraph_layers, tiling, unsched_stats=None):
    """Join TRACE regions to kgraph layer names.

    `-kea-tile` numbers its layers in the order it lowers the Level 1
    contractions, and emits one TRACE region per layer tagged with that number.
    The Level 1 contractions are in the same order as the kgraph's contraction
    nodes, so region tag *i* is kgraph contraction *i*.  The tiling report's
    `op` field is used as a consistency check, not as the source of the name.

    `unsched_stats`, when given, contributes an `unscheduled_cycles` column, so
    the per-layer A/B is visible in the same table as the roofline position.
    """
    unsched = {r["tag"]: r for r in (unsched_stats or {}).get("regions", [])}
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
            "unscheduled_cycles": (unsched[tag]["cycles"] if tag in unsched
                                   else ""),
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


def combine(programs, clock_hz, label, name):
    """The two programs run back to back on one machine, so the compiled
    network's aggregate is their sum."""
    c = {"program": name, "label": label,
         "cycles": sum(p["cycles"] for p in programs),
         "ops_useful": sum(p["ops_useful"] for p in programs),
         "ops_issued": sum(p["ops_issued"] for p in programs),
         "dram_bytes": sum(p["dram_bytes"] for p in programs),
         "dram_bytes_load": sum(p["dram_bytes_load"] for p in programs),
         "dram_bytes_store": sum(p["dram_bytes_store"] for p in programs),
         "peak_gops": programs[0]["peak_gops"],
         "peak_dram_gb_per_s": programs[0]["peak_dram_gb_per_s"],
         "ridge_point_ops_per_byte": programs[0]["ridge_point_ops_per_byte"]}
    c["seconds"] = c["cycles"] / clock_hz
    c["intensity_ops_per_byte"] = c["ops_useful"] / c["dram_bytes"]
    c["achieved_gops"] = c["ops_useful"] / c["seconds"] / GOPS
    c["attainable_gops"] = min(
        c["peak_gops"], c["intensity_ops_per_byte"] * c["peak_dram_gb_per_s"])
    c["pct_of_attainable"] = 100.0 * c["achieved_gops"] / c["attainable_gops"]
    c["bound"] = ("MEMORY" if c["intensity_ops_per_byte"]
                  < c["ridge_point_ops_per_byte"] else "COMPUTE")
    c["padding_efficiency"] = c["ops_useful"] / c["ops_issued"]
    c["dram_gb_per_s"] = c["dram_bytes"] / c["seconds"] / 1e9
    return c


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
    # kept hard left: the lowest-intensity layer sits on the roof, and the
    # label would otherwise run under its marker
    lx = xlo * 1.02
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

    # the whole network, before and after -kea-schedule, with the move drawn
    if len(nets) == 2:
        ax.annotate("", xy=(nets[1]["intensity_ops_per_byte"],
                            nets[1]["achieved_gops"]),
                    xytext=(nets[0]["intensity_ops_per_byte"],
                            nets[0]["achieved_gops"]),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2,
                                    shrinkA=8, shrinkB=9), zorder=5)
    for n, face, lab in zip(nets, ("#d8d7d2", "#ffffff"),
                            (None, "whole network, unscheduled → scheduled")):
        ax.scatter(n["intensity_ops_per_byte"], n["achieved_gops"], s=230,
                   marker="*", facecolor=face, edgecolor=INK,
                   linewidth=1.6, zorder=6, label=lab)
    # the two stars share an x and are only 17% apart vertically, so they get
    # one label between them rather than two that collide
    lo_net = min(nets, key=lambda n: n["achieved_gops"])
    hi_net = max(nets, key=lambda n: n["achieved_gops"])
    ax.annotate("whole network\n%.2f → %.2f ms  (%.3f×)"
                % (lo_net["seconds"] * 1e3, hi_net["seconds"] * 1e3,
                   lo_net["cycles"] / float(hi_net["cycles"])),
                xy=(hi_net["intensity_ops_per_byte"], hi_net["achieved_gops"]),
                xytext=(-26, 26), textcoords="offset points", ha="right",
                va="bottom", fontsize=8.5, color=INK, linespacing=1.4,
                zorder=7,
                # the compute roof passes through here; knock it out behind
                # the glyphs rather than moving the label into the points
                bbox=dict(boxstyle="round,pad=0.28", facecolor=SURFACE,
                          edgecolor="none"))

    # direct labels on the extremes, which is where the story is
    lo = min(rows, key=lambda r: r["intensity_ops_per_byte"])
    hi = max(rows, key=lambda r: r["achieved_gops"])
    # the lowest-intensity layer sits ON the DRAM roof, whose rotated label
    # runs through the space to its upper right -- so label it downwards
    for r, dx, dy, ha, va in ((lo, 8, -9, "left", "top"),
                              (hi, -7, 6, "right", "bottom")):
        ax.annotate("%s L%d" % (r["op"].replace("depthwise_conv2d", "dwconv"),
                                r["layer"]),
                    xy=(r["intensity_ops_per_byte"], r["achieved_gops"]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
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
             "(docs/SIMULATOR.md §2). 1 MAC = 2 ops.\n"
             "Layer points and both network points are the scheduled build at "
             "the compiler defaults; all 183 nodes run on the NPU.",
             fontsize=7.2, color=INK2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    print("wrote %s" % path)


def main() -> int:
    C.require_tools()
    C.ensure_dirs()
    keafs = {k: os.path.join(C.BUILD, k + ".keaf")
             for k in ("features", "featpool", "classifier",
                       "features_unsched", "featpool_unsched",
                       "classifier_unsched")}
    for p in keafs.values():
        if not os.path.exists(p):
            raise SystemExit("run demo/compile_mobilenetv2.py first (%s)" % p)

    st = {}
    for k, p in keafs.items():
        j = os.path.join(C.BUILD, k + ".stats.json")
        C.simulate(p, stats_json=j)
        st[k] = C.load_stats(j)

    # TRACE regions used to come back scrambled from a scheduled build. Fixed
    # -- but verified here, on every build, before anything per-layer is
    # reported. `ratio` is the region sum over the program length: consecutive
    # layers genuinely overlap a little, so it sits just above 1.0; the bug
    # drove it to 7.4.
    region_health = {}
    for k, v in st.items():
        rs = v.get("regions", [])
        if not rs:
            continue
        region_health[k] = {
            "regions": len(rs),
            "unsound_tags": C.unsound_regions(v),
            "max_depth": max(r["depth"] for r in rs),
            "sum_over_total": sum(r["cycles"] for r in rs) / v["total_cycles"],
        }
    for k, h in sorted(region_health.items()):
        print("regions %-20s n=%-3d unsound=%-4s max_depth=%d  "
              "sum/total=%.3f" % (k, h["regions"], len(h["unsound_tags"]),
                                  h["max_depth"], h["sum_over_total"]))
    # Two independent gates. The per-region one catches a marker detached from
    # its layer; the aggregate one catches the same thing from the other side
    # and would fire even if the ordering check were tuned wrong. The bug this
    # guards against measured 7.43 and depth 11.
    bad = sorted(k for k, h in region_health.items()
                 if h["unsound_tags"] or h["sum_over_total"] > 1.5
                 or h["max_depth"] > 2)
    if bad:
        raise SystemExit(
            "TRACE regions are not sound in %s (%s); refusing to report a "
            "per-layer table from them"
            % (", ".join(bad), {k: region_health[k] for k in bad}))

    g = KGraph.load(C.KGRAPH)
    groups = [l for l in C.layer_groups(g) if l.op in C.CONTRACTIONS]
    tiling = C.tiling_report(
        os.path.join(C.BUILD, "mobilenetv2_features.tosa.mlir"),
        "mnv2_features")

    rows = collect_layers(st["features"], groups, tiling,
                          unsched_stats=st["features_unsched"])

    # the classifier is its own program, so it contributes one whole-program
    # point rather than a region.
    crf = st["classifier"]["global"]["roofline"]
    rows.append({
        "layer": len(rows), "node": "fc#182", "op": "fully_connected",
        "cycles": st["classifier"]["total_cycles"],
        "unscheduled_cycles": st["classifier_unsched"]["total_cycles"],
        "ops_useful": crf["ops_useful"], "ops_issued": crf["ops_issued"],
        "padding_efficiency": crf["padding_efficiency"],
        "dram_bytes": crf["dram_bytes"],
        "intensity_ops_per_byte": crf["intensity_ops_per_byte"],
        "achieved_gops": crf["achieved_ops_per_s"] / GOPS,
        "attainable_gops": crf["attainable_ops_per_s"] / GOPS,
        "pct_of_attainable": 100.0 * crf["efficiency"],
        "bound": "MEMORY" if crf["memory_bound"] else "COMPUTE",
        "mxu_mac_utilization":
            st["classifier"]["global"]["mxu"]["mac_utilization"],
        "dwu_mac_utilization":
            st["classifier"]["global"]["dwu"]["mac_utilization"],
    })

    programs = [whole(st["featpool"], "features + head pool, scheduled"),
                whole(st["classifier"], "classifier (fully_connected), "
                                        "scheduled"),
                whole(st["featpool_unsched"], "features + head pool, "
                                              "unscheduled"),
                whole(st["classifier_unsched"], "classifier, unscheduled"),
                whole(st["features"], "features only (52 conv), scheduled"),
                whole(st["features_unsched"], "features only, unscheduled")]

    hz = st["featpool"]["clock_hz"]
    # Every one of the 183 nodes is on the NPU now: featpool.keaf carries the
    # global average pool, so nothing runs on the host between the two
    # programs. `whole_network_bit_exact` is the alternative split that keeps
    # the pool on the host to stay bit-exact -- see validate_mobilenetv2.py.
    comb = combine(programs[:2], hz, "whole network (scheduled)",
                   "whole network on the NPU = featpool + classifier, "
                   "scheduled")
    comb_u = combine(programs[2:4], hz, "unscheduled",
                     "whole network on the NPU, unscheduled")
    comb_be = combine([programs[4], programs[1]], hz,
                      "bit-exact split (host pool)",
                      "features + host pool + classifier, scheduled")
    for c, a, b in ((comb, st["featpool"], st["classifier"]),
                    (comb_u, st["featpool_unsched"],
                     st["classifier_unsched"]),
                    (comb_be, st["features"], st["classifier"])):
        c["mxu_mac_utilization"] = (
            (a["global"]["mxu"]["macs_useful"]
             + b["global"]["mxu"]["macs_useful"]) / (c["cycles"] * 256.0))
        c["dwu_mac_utilization"] = (
            (a["global"]["dwu"]["macs_useful"]
             + b["global"]["dwu"]["macs_useful"]) / (c["cycles"] * 128.0))
    comb["speedup_over_unscheduled"] = comb_u["cycles"] / float(comb["cycles"])
    nets = [comb_u, comb]

    region_cycles = sum(r["cycles"] for r in rows[:-1])
    out = {
        "whole_network": comb,
        "whole_network_unscheduled": comb_u,
        "whole_network_bit_exact": comb_be,
        "programs": programs,
        "layers": rows,
        "layer_source_build": "scheduled",
        "region_health": region_health,
        "layer_cycle_sum": region_cycles,
        "layer_cycle_overlap_pct":
            100.0 * (region_cycles - st["features"]["total_cycles"])
            / st["features"]["total_cycles"],
        "memory_bound_layers": [r["layer"] for r in rows if r["bound"] == "MEMORY"],
        "note": "TRACE regions cover every unit active inside their cycle "
                "window, and consecutive layers overlap, so per-layer cycles "
                "sum to a little more than the program total. Per-layer rows "
                "come from the scheduled build; region_health records the "
                "soundness check that licenses that.",
    }
    C.write_json(os.path.join(C.RESULTS, "roofline.json"), out)

    csv = os.path.join(C.RESULTS, "roofline_layers.csv")
    cols = ["layer", "node", "op", "cycles", "unscheduled_cycles",
            "ops_useful", "ops_issued", "padding_efficiency", "dram_bytes",
            "intensity_ops_per_byte", "achieved_gops", "attainable_gops",
            "pct_of_attainable", "bound", "mxu_mac_utilization",
            "dwu_mac_utilization"]
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
