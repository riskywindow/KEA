#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare the simulator against the frontend's golden integer reference.

``frontend/testdata/golden_io_mobilenetv2_int8.npz`` holds four fixed int8
input tensors and the exact int8 outputs of the numpy reference interpreter
(``docs/FRONTEND.md`` section 4.1: *"assert exact integer equality; there is no
tolerance"*).

The network runs on the simulator as two programs because its global average
pool does not compile (see ``docs/RESULTS.md``).  So three things are measured
separately and reported separately:

* **features** -- the simulator's `head` tensor vs the reference's `head`;
* **classifier** -- the simulator fed the *reference's* `gap` vs the golden
  output, which isolates the classifier from the pool;
* **end-to-end** -- simulator features, then the reference's `global_avg_pool`
  kernel on the host, then the simulator classifier, vs the golden output.
  The middle step is **not** running on the NPU and the report says so.

    .venv/bin/python demo/validate_mobilenetv2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from kea_frontend import reference  # noqa: E402
from kea_frontend.ir import KGraph  # noqa: E402


def compare(name, got, want):
    bad = int((got != want).sum())
    d = got.astype(np.int64) - want.astype(np.int64)
    rec = {"tensor": name, "elements": int(want.size), "mismatches": bad,
           "max_abs_diff": int(np.abs(d).max()) if bad else 0,
           "exact": bad == 0}
    print("    %-12s %7d elements, %5d mismatches%s"
          % (name, want.size, bad,
             "" if bad == 0 else "  max |diff| %d" % rec["max_abs_diff"]))
    return rec


def main() -> int:
    C.require_tools()
    C.ensure_dirs()

    feat_keaf = os.path.join(C.BUILD, "features.keaf")
    clf_keaf = os.path.join(C.BUILD, "classifier.keaf")
    for p in (feat_keaf, clf_keaf):
        if not os.path.exists(p):
            raise SystemExit("missing %s; run demo/compile_mobilenetv2.py "
                             "first" % p)

    g = KGraph.load(C.KGRAPH)
    pool_node = [n for n in g.nodes if n.op == "global_avg_pool"][0]
    gio = np.load(C.GOLDEN)

    feat_in, feat_out = C.io_names(feat_keaf)
    clf_in, clf_out = C.io_names(clf_keaf)

    results = {"golden_npz": os.path.relpath(C.GOLDEN, C.ROOT), "vectors": []}
    tmp = os.path.join(C.BUILD, "io")
    os.makedirs(tmp, exist_ok=True)

    n_vec = sum(1 for k in gio.files if k.startswith("input_"))
    for i in range(n_vec):
        x = gio["input_%d" % i]
        want = gio["output_%d" % i]
        print("vector %d  input %s -> golden output %s" % (i, x.shape,
                                                           want.shape))

        env = reference.execute(g, {"input": x}, keep=["head", "gap"])
        assert (env["fc"] == want).all(), "reference disagrees with the npz"

        # --- features on the simulator ---
        xin = os.path.join(tmp, "in%d.bin" % i)
        x.tofile(xin)
        hout = os.path.join(tmp, "head%d.bin" % i)
        C.simulate(feat_keaf, inputs={feat_in[0]: xin},
                   outputs={feat_out[0]: hout})
        head = np.fromfile(hout, dtype=np.int8).reshape(env["head"].shape)
        rec_feat = compare("head", head, env["head"])

        # --- the pool, on the host, with the reference kernel ---
        gap = reference._KERNELS["global_avg_pool"](g, pool_node, [head])[0]
        rec_pool = compare("gap(host)", gap, env["gap"])

        # --- classifier on the simulator, fed the reference gap ---
        gref = os.path.join(tmp, "gapref%d.bin" % i)
        env["gap"].tofile(gref)
        fout = os.path.join(tmp, "fc%d.bin" % i)
        C.simulate(clf_keaf, inputs={clf_in[0]: gref},
                   outputs={clf_out[0]: fout})
        fc_isolated = np.fromfile(fout, dtype=np.int8).reshape(want.shape)
        rec_clf = compare("fc|ref gap", fc_isolated, want)

        # --- end to end ---
        gsim = os.path.join(tmp, "gapsim%d.bin" % i)
        gap.tofile(gsim)
        fout2 = os.path.join(tmp, "fce2e%d.bin" % i)
        C.simulate(clf_keaf, inputs={clf_in[0]: gsim},
                   outputs={clf_out[0]: fout2})
        fc_e2e = np.fromfile(fout2, dtype=np.int8).reshape(want.shape)
        rec_e2e = compare("fc(e2e)", fc_e2e, want)

        results["vectors"].append({
            "index": i,
            "features": rec_feat,
            "pool_on_host": rec_pool,
            "classifier_given_reference_pool": rec_clf,
            "end_to_end": rec_e2e,
            "argmax_golden": int(want.argmax()),
            "argmax_sim": int(fc_e2e.argmax()),
            "argmax_agrees": int(want.argmax()) == int(fc_e2e.argmax()),
        })

    # The simulator's own correctness gates: a read of never-written
    # scratchpad, or a cross-unit read of data whose producer has not retired,
    # is a missing SIGNAL/WAIT and is made fatal here rather than counted.
    strict = {}
    for label, keaf in (("features", feat_keaf), ("classifier", clf_keaf)):
        r = C.run([C.KEA_SIM, keaf, "--quiet", "--strict-poison",
                   "--strict-hazards"], check=False)
        strict[label] = r.ok
        print("strict poison/hazard check, %-11s %s"
              % (label, "clean" if r.ok else "FAILED: " + r.first_error()))
    results["strict_checks"] = strict

    every = results["vectors"]
    results["summary"] = {
        "vectors": len(every),
        "features_exact": all(v["features"]["exact"] for v in every),
        "classifier_exact": all(
            v["classifier_given_reference_pool"]["exact"] for v in every),
        "end_to_end_exact": all(v["end_to_end"]["exact"] for v in every),
        "argmax_agreement": sum(v["argmax_agrees"] for v in every),
        "caveat": "the global average pool runs on the host with the "
                  "frontend's reference kernel; it does not compile for the "
                  "NPU (see docs/RESULTS.md)",
    }
    print("\nsummary: features exact=%s, classifier exact=%s, "
          "end-to-end exact=%s, argmax agreement %d/%d"
          % (results["summary"]["features_exact"],
             results["summary"]["classifier_exact"],
             results["summary"]["end_to_end_exact"],
             results["summary"]["argmax_agreement"], len(every)))

    C.write_json(os.path.join(C.RESULTS, "validation.json"), results)
    print("wrote demo/results/validation.json")
    return 0 if results["summary"]["end_to_end_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
