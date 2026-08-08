#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A/B the `-kea-schedule` pass: every layer compiled twice, run twice.

The compiler's central claim is that reordering the straight-line instruction
stream so DMA overlaps compute is worth real cycles.  This measures it.

Three experiments, all reported:

1. **per layer** -- each of MobileNetV2's 52 convolution layers, plus the
   classifier, compiled standalone with and without `--schedule` and run on
   `kea-sim`.
2. **the largest multi-layer subgraph that schedules** -- found by bisection,
   because `--kea-schedule` fails Rule D on longer prefixes of this network
   (recorded as a defect in `docs/RESULTS.md`).
3. **the whole feature extractor** -- attempted, and the failure recorded
   verbatim.

    .venv/bin/python demo/ab_schedule.py [--layers N]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from kea_frontend.ir import KGraph  # noqa: E402

CLASSIFIER_SLICE = (180, 182)


def ab(name, first, last, reserve=1):
    """Compile+run `name` twice.  Returns a record; never raises on failure."""
    slice_id = "%s_%s" % ("b" if first is None else first,
                          "e" if last is None else last)
    fn = "slice_%s" % slice_id
    mlir = os.path.join(C.BUILD, "ab", "%s.tosa.mlir" % fn)
    os.makedirs(os.path.dirname(mlir), exist_ok=True)
    C.emit_tosa(mlir, fn, first=first, last=last, cache=True)
    rec = {"name": name, "function": fn, "first_node": first,
           "last_node": last}
    for mode in ("unscheduled", "scheduled"):
        keaf = os.path.join(C.BUILD, "ab", "%s.%s.keaf" % (fn, mode))
        r = C.compile_keaf(mlir, fn, keaf, schedule=(mode == "scheduled"),
                           spm_reserve=reserve, check=False)
        if not r.ok:
            rec[mode] = {"ok": False, "error": r.first_error()}
            continue
        stats = os.path.join(C.BUILD, "ab", "%s.%s.json" % (fn, mode))
        s = C.simulate(keaf, stats_json=stats, check=False)
        if not s.ok:
            rec[mode] = {"ok": False, "error": s.first_error()}
            continue
        st = C.load_stats(stats)
        rec[mode] = {
            "ok": True,
            "cycles": st["total_cycles"],
            "instructions": C.count_instructions(
                keaf[: -len(".keaf")] + ".kasm"),
            "achieved_gops": st["global"]["roofline"]["achieved_ops_per_s"] / 1e9,
            "mxu_busy_pct": 100 * st["global"]["units"]["MXU"]["utilization"],
            "dispatcher_stall_pct":
                100.0 * st["global"]["dispatcher"]["stall_cycles"]
                / st["total_cycles"],
        }
    if rec.get("unscheduled", {}).get("ok") and rec.get("scheduled", {}).get("ok"):
        rec["speedup"] = (rec["unscheduled"]["cycles"]
                          / float(rec["scheduled"]["cycles"]))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None,
                    help="only the first N convolution layers (debugging)")
    args = ap.parse_args()

    C.require_tools()
    C.ensure_dirs()
    g = KGraph.load(C.KGRAPH)
    # every contraction that is part of the feature extractor; the classifier
    # is handled separately below because the shipped program includes the
    # reshape in front of its fully_connected.
    groups = [l for l in C.layer_groups(g)
              if l.op in C.CONTRACTIONS and l.first < CLASSIFIER_SLICE[0]]

    out = {"per_layer": [], "notes": []}

    todo = groups if args.layers is None else groups[: args.layers]
    print("%-4s %-22s %-18s %10s %10s %8s" % ("L", "node", "op", "unsched",
                                              "sched", "speedup"))
    for i, lay in enumerate(todo):
        rec = ab("layer%02d" % i, lay.first, lay.last)
        rec["layer"] = i
        rec["node"] = lay.name
        rec["op"] = lay.op
        out["per_layer"].append(rec)
        u = rec.get("unscheduled", {})
        s = rec.get("scheduled", {})
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
        out["per_layer_totals"] = {
            "layers_compared": len(ok),
            "layers_failed": len(out["per_layer"]) - len(ok),
            "unscheduled_cycles": tu,
            "scheduled_cycles": ts,
            "aggregate_speedup": tu / float(ts),
            "best": max(ok, key=lambda r: r["speedup"])["name"],
            "best_speedup": max(r["speedup"] for r in ok),
            "worst_speedup": min(r["speedup"] for r in ok),
        }
        print("\nper-layer aggregate: %d cycles unscheduled -> %d scheduled "
              "(%.3fx) over %d layers"
              % (tu, ts, tu / float(ts), len(ok)))

    # ---- 2/3: multi-layer and whole-network ------------------------------
    print("\nwhole feature extractor (nodes 0..178):")
    rec = ab("features_ab", None, 178)
    out["feature_extractor"] = rec
    for mode in ("unscheduled", "scheduled"):
        m = rec.get(mode, {})
        print("  %-12s %s" % (mode, m.get("cycles") if m.get("ok")
                              else "FAILED: " + m.get("error", "")[:150]))

    print("\nlargest schedulable prefix (binary search):")
    # A cut in the middle of a conv+rescale pair legitimately fails to lower at
    # all -- those are not candidates, so each probe walks down to the nearest
    # index that compiles unscheduled.  The search assumes the "schedules"
    # predicate is monotonic in the prefix length, which the exhaustive scan in
    # docs/RESULTS.md confirms for this network.
    cache: dict = {}

    def probe(last):
        if last in cache:
            return cache[last]
        for cand in range(last, 0, -1):
            r = ab("prefix", None, cand)
            if r.get("unscheduled", {}).get("ok"):
                r["last_node_index"] = cand
                cache[last] = r
                return r
        cache[last] = None
        return None

    lo, hi, best = 1, 178, None
    while lo <= hi:
        mid = (lo + hi) // 2
        r = probe(mid)
        if r is None:
            lo = mid + 1
            continue
        idx = r["last_node_index"]
        # `idx <= mid` because probe walks *down* to a lowerable cut, so the
        # bounds are clamped against `mid` as well: without that the interval
        # can stop shrinking and the loop never terminates.
        if r.get("scheduled", {}).get("ok"):
            best = r
            lo = max(idx + 1, mid + 1)
        else:
            hi = min(idx - 1, mid - 1)
    if best:
        print("  nodes 0..%d: %d -> %d cycles (%.3fx)"
              % (best["last_node_index"], best["unscheduled"]["cycles"],
                 best["scheduled"]["cycles"], best["speedup"]))
        out["largest_schedulable_prefix"] = best
    else:
        out["largest_schedulable_prefix"] = None
        print("  none found")

    C.write_json(os.path.join(C.RESULTS, "ab_schedule.json"), out)
    print("wrote demo/results/ab_schedule.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
