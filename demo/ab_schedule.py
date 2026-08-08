#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A/B the `-kea-schedule` pass, on the whole network and per layer.

The compiler's central claim is that reordering the straight-line instruction
stream so DMA overlaps compute is worth real cycles.  This measures it, at the
compiler's defaults -- nothing pinned.

Four experiments, all reported:

1. **the whole 52-convolution feature extractor**, compiled with and without
   `--schedule` and run on `kea-sim`.  This is the headline: it is the artifact
   that ships.
2. **`-kea-schedule mode=serial` and `mode=overlap`**, the pass's own controls.
   `mode=auto` promises either a program its model predicts >=5% faster than
   not scheduling, or output byte-identical to omitting `--schedule`.  The
   sweep is where that bites, and the byte-identity is checked with `cmp`, not
   inferred from equal cycle counts.
3. **an `-kea-tile=imem-budget` sweep**, because the budget picks the tiling
   and the tiling picks the cycle count.  The feasible window is narrow at both
   ends and the curve inside it is not monotonic.
4. **per layer** -- each of MobileNetV2's 52 convolution layers, plus the
   classifier, compiled standalone twice.  A weaker experiment than 1, kept
   because it separates the families.

    .venv/bin/python demo/ab_schedule.py [--layers N] [--no-sweep]
"""
from __future__ import annotations

import argparse
import filecmp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from kea_frontend.ir import KGraph  # noqa: E402

CLASSIFIER_SLICE = (180, 182)
LAST_FEATURE_NODE = 178
DEFAULT_BUDGET = 20200

#: `-kea-tile` refuses below 19,066 -- its own coarsest tiling of all 52 layers
#: of this network -- and `kea-as` refuses once the finished program passes the
#: 32,768-entry IMEM. The sweep spans that whole window and one step past each
#: end, so both walls are measured rather than assumed.
SWEEP = [19000, 19066, 19200, 19500, 19750, 20000, 20200, 20480, 20750, 21000,
         21500, 22000, 22500, 23000, 24000]


def build_and_run(tag, mlir, fn, **kw):
    """Compile+run one configuration.  Returns a record; never raises."""
    keaf = os.path.join(C.BUILD, "ab", "%s.keaf" % tag)
    os.makedirs(os.path.dirname(keaf), exist_ok=True)
    r = C.compile_keaf(mlir, fn, keaf, check=False, **kw)
    if not r.ok:
        return {"ok": False, "error": r.first_error()}
    stats = os.path.join(C.BUILD, "ab", "%s.json" % tag)
    s = C.simulate(keaf, stats_json=stats, check=False)
    if not s.ok:
        return {"ok": False, "error": s.first_error()}
    st = C.load_stats(stats)
    return {
        "ok": True,
        "cycles": st["total_cycles"],
        "instructions": C.count_instructions(keaf[: -len(".keaf")] + ".kasm"),
        "achieved_gops": st["global"]["roofline"]["achieved_ops_per_s"] / 1e9,
        "mxu_busy_pct": 100 * st["global"]["units"]["MXU"]["utilization"],
        "dma0_busy_pct": 100 * st["global"]["units"]["DMA0"]["utilization"],
        "dma1_busy_pct": 100 * st["global"]["units"]["DMA1"]["utilization"],
        "dispatcher_stall_pct":
            100.0 * st["global"]["dispatcher"]["stall_cycles"]
            / st["total_cycles"],
    }


def ab(name, first, last, **kw):
    """Compile+run a node slice twice, unscheduled and scheduled."""
    slice_id = "%s_%s" % ("b" if first is None else first,
                          "e" if last is None else last)
    fn = "slice_%s" % slice_id
    mlir = os.path.join(C.BUILD, "ab", "%s.tosa.mlir" % fn)
    os.makedirs(os.path.dirname(mlir), exist_ok=True)
    C.emit_tosa(mlir, fn, first=first, last=last, cache=True)
    rec = {"name": name, "function": fn, "first_node": first,
           "last_node": last}
    for mode in ("unscheduled", "scheduled"):
        rec[mode] = build_and_run("%s.%s" % (fn, mode), mlir, fn,
                                  schedule=(mode == "scheduled"), **kw)
    if rec["unscheduled"].get("ok") and rec["scheduled"].get("ok"):
        rec["speedup"] = (rec["unscheduled"]["cycles"]
                          / float(rec["scheduled"]["cycles"]))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None,
                    help="only the first N convolution layers (debugging)")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the imem-budget sweep")
    args = ap.parse_args()

    C.require_tools()
    C.ensure_dirs()
    # DEFAULT_BUDGET only labels the sweep row that matches the compiler's
    # own default; the A/B itself passes no flags.
    out = {"flags": "none except where a row names one",
           "default_imem_budget_assumed": DEFAULT_BUDGET,
           "per_layer": []}

    feat = os.path.join(C.BUILD, "mobilenetv2_features.tosa.mlir")
    C.emit_tosa(feat, "mnv2_features", last=LAST_FEATURE_NODE, cache=True)

    # ---- 1/2. the whole feature extractor, three ways ----------------------
    print("############ whole feature extractor (nodes 0..%d), defaults"
          % LAST_FEATURE_NODE)
    whole = {}
    whole["unscheduled"] = build_and_run("features.unscheduled", feat,
                                         "mnv2_features")
    whole["scheduled_auto"] = build_and_run("features.auto", feat,
                                            "mnv2_features", schedule=True)
    for mode in ("serial", "overlap"):
        whole["mode_" + mode] = build_and_run(
            "features.%s" % mode, feat, "mnv2_features", schedule_mode=mode)
    print("%-22s %10s %8s %8s %8s %8s"
          % ("configuration", "cycles", "instrs", "MXU%", "DMA0%", "DMA1%"))
    for k, v in whole.items():
        if v.get("ok"):
            print("%-22s %10d %8d %8.2f %8.2f %8.2f"
                  % (k, v["cycles"], v["instructions"], v["mxu_busy_pct"],
                     v["dma0_busy_pct"], v["dma1_busy_pct"]))
        else:
            print("%-22s FAILED: %s" % (k, v.get("error", "")[:120]))
    if whole["unscheduled"].get("ok") and whole["scheduled_auto"].get("ok"):
        whole["speedup_vs_unscheduled"] = (
            whole["unscheduled"]["cycles"]
            / float(whole["scheduled_auto"]["cycles"]))
        print("\n--schedule vs no --schedule: %.4fx"
              % whole["speedup_vs_unscheduled"])
    if whole["mode_serial"].get("ok") and whole["scheduled_auto"].get("ok"):
        whole["speedup_vs_serial"] = (
            whole["mode_serial"]["cycles"]
            / float(whole["scheduled_auto"]["cycles"]))
        print("mode=auto vs mode=serial:    %.4fx"
              % whole["speedup_vs_serial"])
    out["feature_extractor"] = whole

    print("\n############ classifier (nodes %d..%d)" % CLASSIFIER_SLICE)
    clf = os.path.join(C.BUILD, "mobilenetv2_classifier.tosa.mlir")
    C.emit_tosa(clf, "mnv2_classifier", first=CLASSIFIER_SLICE[0], cache=True)
    cl = {"unscheduled": build_and_run("clf.unscheduled", clf,
                                       "mnv2_classifier"),
          "scheduled": build_and_run("clf.scheduled", clf, "mnv2_classifier",
                                     schedule=True)}
    if cl["unscheduled"].get("ok") and cl["scheduled"].get("ok"):
        cl["speedup"] = (cl["unscheduled"]["cycles"]
                         / float(cl["scheduled"]["cycles"]))
        print("  %d -> %d cycles (%.4fx)"
              % (cl["unscheduled"]["cycles"], cl["scheduled"]["cycles"],
                 cl["speedup"]))
    out["classifier"] = cl

    # ---- 3. the imem-budget sweep ------------------------------------------
    if not args.no_sweep:
        print("\n############ -kea-tile=imem-budget sweep, nodes 0..%d"
              % LAST_FEATURE_NODE)
        print("%8s %10s %10s %10s %10s %8s"
              % ("budget", "u.instrs", "u.cycles", "s.instrs", "s.cycles",
                 "speedup"))
        sweep = []
        serial_done = False
        for b in SWEEP:
            u = build_and_run("sweep.%d.unscheduled" % b, feat,
                              "mnv2_features", imem_budget=b)
            s = build_and_run("sweep.%d.scheduled" % b, feat, "mnv2_features",
                              imem_budget=b, schedule=True)
            row = {"imem_budget": b, "unscheduled": u, "scheduled": s,
                   "default": b == DEFAULT_BUDGET}
            if u.get("ok") and s.get("ok"):
                # `mode=auto`'s fallback is "emit exactly what no-schedule
                # emits". Equal cycle counts are consistent with that but do
                # not establish it -- diff the assembly.
                row["identical_to_unscheduled"] = filecmp.cmp(
                    os.path.join(C.BUILD, "ab",
                                 "sweep.%d.unscheduled.kasm" % b),
                    os.path.join(C.BUILD, "ab", "sweep.%d.scheduled.kasm" % b),
                    shallow=False)
            if s.get("ok") and not serial_done:
                # At the tightest feasible budget, cost mode=serial too. This
                # is where `mode=auto` used to lose to not scheduling at all;
                # its guarantee is now "beat no-schedule by 5% or emit exactly
                # what no-schedule emits", and this row is where that bites.
                row["serial"] = build_and_run("sweep.%d.serial" % b, feat,
                                              "mnv2_features", imem_budget=b,
                                              schedule_mode="serial")
                serial_done = True
            if u.get("ok") and s.get("ok"):
                row["speedup"] = u["cycles"] / float(s["cycles"])
            sweep.append(row)
            print("%8d %10s %10s %10s %10s %8s%s"
                  % (b,
                     u.get("instructions", "-"), u.get("cycles", "FAIL"),
                     s.get("instructions", "-"), s.get("cycles", "FAIL"),
                     "%.3fx" % row["speedup"] if "speedup" in row else "-",
                     ("   <- default" if b == DEFAULT_BUDGET else "")
                     + ("  (identical .kasm)"
                        if row.get("identical_to_unscheduled") else "")))
        out["imem_budget_sweep"] = sweep
        ok = [r for r in sweep if r["scheduled"].get("ok")]
        if ok:
            best = min(ok, key=lambda r: r["scheduled"]["cycles"])
            dflt = [r for r in ok if r["default"]]
            out["imem_budget_best"] = {
                "imem_budget": best["imem_budget"],
                "cycles": best["scheduled"]["cycles"],
                "instructions": best["scheduled"]["instructions"]}
            if dflt:
                out["imem_budget_best"]["vs_default_pct"] = 100.0 * (
                    dflt[0]["scheduled"]["cycles"]
                    - best["scheduled"]["cycles"]) / float(
                        dflt[0]["scheduled"]["cycles"])
            out["imem_budget_feasible"] = [r["imem_budget"] for r in ok]
            out["imem_budget_monotonic"] = all(
                ok[i]["scheduled"]["cycles"] <= ok[i + 1]["scheduled"]["cycles"]
                for i in range(len(ok) - 1)) or all(
                ok[i]["scheduled"]["cycles"] >= ok[i + 1]["scheduled"]["cycles"]
                for i in range(len(ok) - 1))
            print("\nbest scheduled: imem-budget=%d at %d cycles "
                  "(%.2f%% under the default); curve monotonic: %s"
                  % (best["imem_budget"], best["scheduled"]["cycles"],
                     out["imem_budget_best"].get("vs_default_pct", 0.0),
                     out["imem_budget_monotonic"]))

    # ---- 4. per layer ------------------------------------------------------
    print("\n############ per layer")
    g = KGraph.load(C.KGRAPH)
    groups = [l for l in C.layer_groups(g)
              if l.op in C.CONTRACTIONS and l.first < CLASSIFIER_SLICE[0]]
    todo = groups if args.layers is None else groups[: args.layers]
    print("%-4s %-22s %-18s %10s %10s %8s" % ("L", "node", "op", "unsched",
                                              "sched", "speedup"))
    for i, lay in enumerate(todo):
        rec = ab("layer%02d" % i, lay.first, lay.last)
        rec["layer"] = i
        rec["node"] = lay.name
        rec["op"] = lay.op
        out["per_layer"].append(rec)
        u, s = rec.get("unscheduled", {}), rec.get("scheduled", {})
        print("%-4d %-22s %-18s %10s %10s %8s"
              % (i, lay.name, lay.op,
                 u.get("cycles", "FAIL"), s.get("cycles", "FAIL"),
                 "%.3fx" % rec["speedup"] if "speedup" in rec else "-"))

    if args.layers is None:
        rec = ab("classifier", *CLASSIFIER_SLICE)
        rec["layer"] = len(todo)
        rec["node"] = "fc#182"
        rec["op"] = "fully_connected"
        out["per_layer"].append(rec)
        u, s = rec.get("unscheduled", {}), rec.get("scheduled", {})
        print("%-4d %-22s %-18s %10s %10s %8s"
              % (rec["layer"], "fc#182", "fully_connected",
                 u.get("cycles", "FAIL"), s.get("cycles", "FAIL"),
                 "%.3fx" % rec["speedup"] if "speedup" in rec else "-"))

    ok = [r for r in out["per_layer"] if "speedup" in r]
    if ok:
        tu = sum(r["unscheduled"]["cycles"] for r in ok)
        ts = sum(r["scheduled"]["cycles"] for r in ok)
        byop = {}
        for r in ok:
            d = byop.setdefault(r["op"], {"layers": 0, "u": 0, "s": 0})
            d["layers"] += 1
            d["u"] += r["unscheduled"]["cycles"]
            d["s"] += r["scheduled"]["cycles"]
        for d in byop.values():
            d["speedup"] = d["u"] / float(d["s"])
        out["per_layer_totals"] = {
            "layers_compared": len(ok),
            "layers_failed": len(out["per_layer"]) - len(ok),
            "unscheduled_cycles": tu,
            "scheduled_cycles": ts,
            "aggregate_speedup": tu / float(ts),
            "by_op": byop,
            "best": max(ok, key=lambda r: r["speedup"])["name"],
            "best_speedup": max(r["speedup"] for r in ok),
            "worst": min(ok, key=lambda r: r["speedup"])["name"],
            "worst_speedup": min(r["speedup"] for r in ok),
            "regressed": [r["node"] for r in ok if r["speedup"] < 1.0],
        }
        print("\nper-layer aggregate: %d cycles unscheduled -> %d scheduled "
              "(%.3fx) over %d layers" % (tu, ts, tu / float(ts), len(ok)))
        for op, d in sorted(byop.items()):
            print("  %-18s %2d layers  %9d -> %9d  (%.3fx)"
                  % (op, d["layers"], d["u"], d["s"], d["speedup"]))
        print("  %d regressed: %s" % (len(out["per_layer_totals"]["regressed"]),
                                      out["per_layer_totals"]["regressed"]))

    C.write_json(os.path.join(C.RESULTS, "ab_schedule.json"), out)
    print("wrote demo/results/ab_schedule.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
