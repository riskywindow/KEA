#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit TOSA for MobileNetV2 int8 and compile it, recording exactly what fits.

Everything runs at the compiler's **defaults**: this script passes no tuning
flag anywhere except where a step is explicitly about a flag.

Six things are attempted, in this order, and every outcome -- including the
failures -- is written to ``demo/results/compile.json``:

1. **the whole 183-node graph** as one function.  Every op in it lowers; it
   does not fit.  Attempted at the default budget, at the tiler's own
   coarsest-tiling floor, and at a budget equal to all of IMEM, so the capacity
   result is measured from three directions rather than assumed.
2. **the 179-node feature extractor** (all 52 convolutions, input -> the last
   1x1 conv's ReLU6 output), unscheduled and scheduled.
3. **the 180-node feature extractor + head pool** (nodes 0..179).  This is new:
   a standalone ``kea.rescale`` lowers now, so the pool no longer has to run on
   the host.  It costs bit-exactness -- see ``demo/validate_mobilenetv2.py``.
4. **the 3-node classifier** (reshape + fully_connected + rescale), unscheduled
   and scheduled.
5. **the head pool as the model actually spells it** (node 179 alone).
6. **a pool with no scale change** (`demo/regress/pool_translates.mlir`), which
   is the case that used to be untranslatable.

The scheduled builds are what ships as ``features.keaf`` / ``featpool.keaf`` /
``classifier.keaf``.

Nothing here is estimated.  The instruction counts come from counting
instruction lines in the ``.kasm`` the compiler actually wrote.

    .venv/bin/python demo/compile_mobilenetv2.py
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from kea_frontend.ir import KGraph  # noqa: E402

# node indices, from the shipped models/mobilenetv2_int8.kgraph.json
LAST_FEATURE_NODE = 178     # clamp#179 -> "head", [1,7,7,1280]
POOL_NODE = 179             # gap#180
FIRST_CLASSIFIER_NODE = 180  # reshape#181

IMEM = 32768


def attempt(label, mlir, fn, keaf, **kw):
    r = C.compile_keaf(mlir, fn, keaf, check=False, **kw)
    kasm = keaf[: -len(".keaf")] + ".kasm"
    rec = {"label": label, "ok": r.ok, "returncode": r.returncode}
    for k in ("spm_reserve", "imem_budget", "schedule"):
        if k in kw:
            rec[k] = kw[k]
    if os.path.exists(kasm):
        rec["instructions"] = C.count_instructions(kasm)
    if not r.ok:
        rec["error"] = r.first_error()
    print("  %-46s %s%s" % (
        label, "ok" if r.ok else "FAILED",
        "" if "instructions" not in rec else
        "  (%d instructions)" % rec["instructions"]))
    if not r.ok:
        print("      %s" % rec["error"][:220])
    return rec


def floor_from_diagnostic(msg: str):
    """`-kea-tile`'s own number for the coarsest tiling of every layer."""
    m = re.search(r"needs (\d+) instructions against a budget of (\d+)", msg)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def main() -> int:
    C.require_tools()
    C.ensure_dirs()
    g = KGraph.load(C.KGRAPH)
    print(g.summary())

    out = {"kgraph": os.path.relpath(C.KGRAPH, C.ROOT),
           "nodes_total": len(g.nodes),
           "imem_capacity": IMEM,
           # The demo passes no tuning flags at all, so the values in force are
           # whatever `-kea-tile` / `-kea-schedule` default to. Recorded as
           # "unset" rather than restated, so this file cannot go stale when a
           # default moves.
           "flags": "none: compiler defaults for spm-reserve-factor, "
                    "imem-budget and -kea-schedule mode",
           "attempts": []}

    # ---- 1. the whole graph, at the defaults --------------------------------
    print("\n[1] whole graph, %d nodes, at the compiler defaults"
          % len(g.nodes))
    full = os.path.join(C.BUILD, "mobilenetv2_full.tosa.mlir")
    C.emit_tosa(full, "mobilenetv2")
    rt = C.run([C.MLIR_OPT, full, "-o", os.path.join(C.BUILD, "rt.mlir")],
               check=False)
    print("  mlir-opt round-trip: %s" % ("ok" if rt.ok else "FAILED"))
    out["full_graph_mlir_roundtrip"] = rt.ok
    out["full_graph_mlir_bytes"] = os.path.getsize(full)

    rec = attempt("whole graph (183 nodes), defaults", full, "mobilenetv2",
                  os.path.join(C.BUILD, "mobilenetv2_full.keaf"))
    rec["nodes"] = len(g.nodes)
    need, budget = floor_from_diagnostic(rec.get("error", ""))
    rec["coarsest_tiling_instructions"] = need
    rec["budget"] = budget
    out["attempts"].append(rec)

    # Raise the budget to the tiler's own floor, and then to all of IMEM: every
    # op lowers, so what comes back is the real program size, not a proxy.
    out["capacity"] = {"imem": IMEM,
                       "coarsest_tiling_budget": need}
    for b in ([need] if need else []) + [IMEM]:
        rec = attempt("whole graph, --imem-budget %d" % b, full, "mobilenetv2",
                      os.path.join(C.BUILD, "full_%d.keaf" % b),
                      imem_budget=b)
        rec["nodes"] = len(g.nodes)
        out["attempts"].append(rec)
        if rec.get("instructions") and b == need:
            out["capacity"]["instructions_at_coarsest_tiling"] = \
                rec["instructions"]
            out["capacity"]["over_imem_by"] = rec["instructions"] - IMEM
            out["capacity"]["over_imem_pct"] = \
                100.0 * (rec["instructions"] - IMEM) / IMEM
            out["capacity"]["fits_imem"] = rec["instructions"] <= IMEM

    # ---- 2. the feature extractor ------------------------------------------
    print("\n[2] feature extractor, nodes 0..%d (52 convolutions), defaults"
          % LAST_FEATURE_NODE)
    feat = os.path.join(C.BUILD, "mobilenetv2_features.tosa.mlir")
    C.emit_tosa(feat, "mnv2_features", last=LAST_FEATURE_NODE)
    for sched in (False, True):
        rec = attempt("features, %s" % ("--schedule" if sched
                                        else "no --schedule"),
                      feat, "mnv2_features",
                      os.path.join(C.BUILD, "features_%s.keaf"
                                   % ("sched" if sched else "unsched")),
                      schedule=sched)
        rec["nodes"] = LAST_FEATURE_NODE + 1
        rec["fits_imem"] = rec.get("instructions", 1 << 30) <= IMEM
        out["attempts"].append(rec)

    # ---- 3. the feature extractor WITH the head pool -----------------------
    print("\n[3] feature extractor + head pool, nodes 0..%d, defaults"
          % POOL_NODE)
    featpool = os.path.join(C.BUILD, "mobilenetv2_featpool.tosa.mlir")
    C.emit_tosa(featpool, "mnv2_featpool", last=POOL_NODE)
    for sched in (False, True):
        rec = attempt("features + pool, %s" % ("--schedule" if sched
                                               else "no --schedule"),
                      featpool, "mnv2_featpool",
                      os.path.join(C.BUILD, "featpool_%s.keaf"
                                   % ("sched" if sched else "unsched")),
                      schedule=sched)
        rec["nodes"] = POOL_NODE + 1
        rec["fits_imem"] = rec.get("instructions", 1 << 30) <= IMEM
        out["attempts"].append(rec)

    # ---- 4. the classifier -------------------------------------------------
    print("\n[4] classifier, nodes %d..%d, defaults"
          % (FIRST_CLASSIFIER_NODE, len(g.nodes) - 1))
    clf = os.path.join(C.BUILD, "mobilenetv2_classifier.tosa.mlir")
    C.emit_tosa(clf, "mnv2_classifier", first=FIRST_CLASSIFIER_NODE)
    for sched in (False, True):
        rec = attempt("classifier, %s" % ("--schedule" if sched
                                          else "no --schedule"),
                      clf, "mnv2_classifier",
                      os.path.join(C.BUILD, "classifier_%s.keaf"
                                   % ("sched" if sched else "unsched")),
                      schedule=sched)
        rec["nodes"] = len(g.nodes) - FIRST_CLASSIFIER_NODE
        rec["fits_imem"] = rec.get("instructions", 1 << 30) <= IMEM
        out["attempts"].append(rec)

    # the shipped artifacts are the scheduled builds
    for src, dst in (("features_sched", "features"),
                     ("featpool_sched", "featpool"),
                     ("classifier_sched", "classifier")):
        for ext in (".keaf", ".kasm", ".map.json", ".weights.bin", ".l2.mlir"):
            p = os.path.join(C.BUILD, src + ext)
            if os.path.exists(p):
                os.replace(p, os.path.join(C.BUILD, dst + ext))
    out["shipped"] = {
        "features.keaf": "nodes 0..178, --schedule, defaults",
        "featpool.keaf": "nodes 0..179 (pool on the NPU), --schedule, defaults",
        "classifier.keaf": "nodes 180..182, --schedule, defaults"}

    # ---- 5. the head pool, as this model spells it -------------------------
    print("\n[5] the head pool alone (node %d), as the emitter spells it"
          % POOL_NODE)
    pool = os.path.join(C.BUILD, "mobilenetv2_pool.tosa.mlir")
    C.emit_tosa(pool, "mnv2_pool", first=POOL_NODE, last=POOL_NODE)
    rec = attempt("head pool: avg_pool2d + rescale (1 node)", pool,
                  "mnv2_pool", os.path.join(C.BUILD, "pool.keaf"))
    rec["nodes"] = 1
    out["attempts"].append(rec)

    # ---- 6. a pool with no scale change ------------------------------------
    print("\n[6] a pool with no scale change (the old collision case)")
    bare = os.path.join(C.ROOT, "demo", "regress", "pool_translates.mlir")
    rec = attempt("bare 7x7 tosa.avg_pool2d", bare, "gap_pool",
                  os.path.join(C.BUILD, "gap_pool.keaf"))
    rec["nodes"] = 1
    if rec["ok"]:
        st = os.path.join(C.BUILD, "gap_pool.stats.json")
        C.simulate(os.path.join(C.BUILD, "gap_pool.keaf"), stats_json=st)
        s = C.load_stats(st)
        rec["cycles"] = s["total_cycles"]
        rec["vpu_utilization"] = s["global"]["vpu"]["utilization"]
        print("      runs: %d cycles, VPU %.1f%% utilised"
              % (rec["cycles"], 100 * rec["vpu_utilization"]))
    out["attempts"].append(rec)

    by_label = {a["label"]: a for a in out["attempts"] if a["ok"]}
    covered = (by_label["features + pool, --schedule"]["nodes"]
               + by_label["classifier, --schedule"]["nodes"])
    out["nodes_compiled"] = covered
    out["fraction_of_graph"] = covered / len(g.nodes)
    out["single_program"] = False
    out["split"] = {
        "bit_exact": ["features.keaf (0..178)", "host global_avg_pool",
                      "classifier.keaf (180..182)"],
        "all_on_npu": ["featpool.keaf (0..179)",
                       "classifier.keaf (180..182)"],
        "note": "the all-NPU split needs no host step but is not bit-exact -- "
                "the frontend's avg_pool2d+rescale substitution rounds twice "
                "where the reference rounds once. demo/validate_mobilenetv2.py "
                "measures both.",
    }
    print("\n%d of %d nodes (%.1f%%) compile and run on the NPU, as two "
          "programs" % (covered, len(g.nodes), 100.0 * covered / len(g.nodes)))

    C.write_json(os.path.join(C.RESULTS, "compile.json"), out)
    print("wrote demo/results/compile.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
