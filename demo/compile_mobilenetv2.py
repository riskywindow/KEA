#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit TOSA for MobileNetV2 int8 and compile it, recording exactly what fits.

Three things are attempted, in this order, and every outcome -- including the
failures -- is written to ``demo/results/compile.json``:

1. **the whole 183-node graph** as one function.  Expected to fail: the
   ``global_avg_pool`` cannot be lowered (see the two backend defects recorded
   in ``docs/RESULTS.md``).
2. **the 179-node feature extractor** (all 52 convolutions, input -> the last
   1x1 conv's ReLU6 output), at the default SPM reserve factor and at
   ``--spm-reserve 1``.  The default overruns IMEM; reserve 1 fits.
3. **the 3-node classifier** (reshape + fully_connected + rescale).

Nothing here is estimated.  The instruction counts come from counting
instruction lines in the ``.kasm`` the compiler actually wrote.

    .venv/bin/python demo/compile_mobilenetv2.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from kea_frontend.ir import KGraph  # noqa: E402

# node indices, from the shipped models/mobilenetv2_int8.kgraph.json
LAST_FEATURE_NODE = 178     # clamp#179 -> "head", [1,7,7,1280]
POOL_NODE = 179             # gap#180
FIRST_CLASSIFIER_NODE = 180  # reshape#181


def attempt(label, mlir, fn, keaf, **kw):
    r = C.compile_keaf(mlir, fn, keaf, check=False, **kw)
    kasm = keaf[: -len(".keaf")] + ".kasm"
    rec = {"label": label, "ok": r.ok, "returncode": r.returncode}
    if os.path.exists(kasm):
        rec["instructions"] = C.count_instructions(kasm)
    if not r.ok:
        rec["error"] = r.first_error()
    print("  %-46s %s%s" % (
        label, "ok" if r.ok else "FAILED",
        "" if "instructions" not in rec else
        "  (%d instructions)" % rec["instructions"]))
    if not r.ok:
        print("      %s" % rec["error"][:200])
    return rec


def main() -> int:
    C.require_tools()
    C.ensure_dirs()
    g = KGraph.load(C.KGRAPH)
    print(g.summary())

    out = {"kgraph": os.path.relpath(C.KGRAPH, C.ROOT),
           "nodes_total": len(g.nodes),
           "imem_capacity": 32768,
           "attempts": []}

    # ---- 1. the whole graph ------------------------------------------------
    print("\n[1] whole graph, %d nodes" % len(g.nodes))
    full = os.path.join(C.BUILD, "mobilenetv2_full.tosa.mlir")
    C.emit_tosa(full, "mobilenetv2")
    rt = C.run([C.MLIR_OPT, full, "-o", os.path.join(C.BUILD, "rt.mlir")],
               check=False)
    print("  mlir-opt round-trip: %s" % ("ok" if rt.ok else "FAILED"))
    out["full_graph_mlir_roundtrip"] = rt.ok
    out["full_graph_mlir_bytes"] = os.path.getsize(full)
    rec = attempt("whole graph (183 nodes)", full, "mobilenetv2",
                  os.path.join(C.BUILD, "mobilenetv2_full.keaf"),
                  spm_reserve=1)
    rec["nodes"] = len(g.nodes)
    out["attempts"].append(rec)

    # ---- 2. the feature extractor -----------------------------------------
    print("\n[2] feature extractor, nodes 0..%d (52 convolutions)"
          % LAST_FEATURE_NODE)
    feat = os.path.join(C.BUILD, "mobilenetv2_features.tosa.mlir")
    C.emit_tosa(feat, "mnv2_features", last=LAST_FEATURE_NODE)
    for reserve in (2, 1):
        rec = attempt("features, --spm-reserve %d" % reserve, feat,
                      "mnv2_features",
                      os.path.join(C.BUILD, "features_r%d.keaf" % reserve),
                      spm_reserve=reserve)
        rec["nodes"] = LAST_FEATURE_NODE + 1
        rec["spm_reserve"] = reserve
        rec["fits_imem"] = rec.get("instructions", 1 << 30) <= 32768
        out["attempts"].append(rec)

    # the shipped artifact
    if os.path.exists(os.path.join(C.BUILD, "features_r1.keaf")):
        os.replace(os.path.join(C.BUILD, "features_r1.keaf"),
                   os.path.join(C.BUILD, "features.keaf"))
        for ext in (".kasm", ".map.json", ".weights.bin", ".l2.mlir"):
            src = os.path.join(C.BUILD, "features_r1" + ext)
            if os.path.exists(src):
                os.replace(src, os.path.join(C.BUILD, "features" + ext))

    # ---- 3. the classifier -------------------------------------------------
    print("\n[3] classifier, nodes %d..%d" % (FIRST_CLASSIFIER_NODE,
                                              len(g.nodes) - 1))
    clf = os.path.join(C.BUILD, "mobilenetv2_classifier.tosa.mlir")
    C.emit_tosa(clf, "mnv2_classifier", first=FIRST_CLASSIFIER_NODE)
    rec = attempt("classifier (3 nodes)", clf, "mnv2_classifier",
                  os.path.join(C.BUILD, "classifier.keaf"))
    rec["nodes"] = len(g.nodes) - FIRST_CLASSIFIER_NODE
    out["attempts"].append(rec)

    # ---- 4. the pool on its own, to pin the defect down --------------------
    print("\n[4] the global average pool alone (node %d)" % POOL_NODE)
    pool = os.path.join(C.BUILD, "mobilenetv2_pool.tosa.mlir")
    C.emit_tosa(pool, "mnv2_pool", first=POOL_NODE, last=POOL_NODE)
    rec = attempt("global_avg_pool (1 node)", pool, "mnv2_pool",
                  os.path.join(C.BUILD, "pool.keaf"))
    rec["nodes"] = 1
    out["attempts"].append(rec)

    ran = [a for a in out["attempts"] if a["ok"]]
    covered = sum(a["nodes"] for a in ran if a["label"].startswith(
        ("features, --spm-reserve 1", "classifier")))
    out["nodes_compiled"] = covered
    out["fraction_of_graph"] = covered / len(g.nodes)
    print("\ncompiled %d of %d nodes (%.1f%%)"
          % (covered, len(g.nodes), 100.0 * covered / len(g.nodes)))

    C.write_json(os.path.join(C.RESULTS, "compile.json"), out)
    print("wrote demo/results/compile.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
