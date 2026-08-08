#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare the simulator against the frontend's golden integer reference.

``frontend/testdata/golden_io_mobilenetv2_int8.npz`` holds four fixed int8
input tensors and the exact int8 outputs of the numpy reference interpreter
(``docs/FRONTEND.md`` section 4.1: *"assert exact integer equality; there is no
tolerance"*).

Every program here is the **scheduled** build at the compiler defaults, so this
validates `-kea-schedule`'s output, not just the tiler's.

The network runs as two programs because it does not fit in IMEM (see
``docs/RESULTS.md`` section 2.1), and there are now **two ways to split it**,
which this script measures against each other:

* **bit-exact** -- `features.keaf` (0..178), the global average pool on the
  host, `classifier.keaf` (180..182).
* **all on the NPU** -- `featpool.keaf` (0..179, pool included),
  `classifier.keaf`.  Nothing runs on the host.  This is newly possible: a
  standalone `kea.rescale` lowers now.  It is **not** bit-exact, and the reason
  is in the frontend, not the backend: `docs/RESULTS.md` section 1.1 records
  that `tosa.avg_pool2d` + `tosa.rescale` rounds twice where the reference
  rounds once.  How much that costs is measured here rather than asserted.

For the bit-exact split, three things are measured separately and reported
separately:

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
    fp_keaf = os.path.join(C.BUILD, "featpool.keaf")
    clf_keaf = os.path.join(C.BUILD, "classifier.keaf")
    for p in (feat_keaf, fp_keaf, clf_keaf):
        if not os.path.exists(p):
            raise SystemExit("missing %s; run demo/compile_mobilenetv2.py "
                             "first" % p)

    g = KGraph.load(C.KGRAPH)
    pool_node = [n for n in g.nodes if n.op == "global_avg_pool"][0]
    gio = np.load(C.GOLDEN)

    feat_in, feat_out = C.io_names(feat_keaf)
    fp_in, fp_out = C.io_names(fp_keaf)
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

        # --- the all-NPU split: featpool.keaf, then the classifier ---
        gnpu = os.path.join(tmp, "gapnpu%d.bin" % i)
        C.simulate(fp_keaf, inputs={fp_in[0]: xin}, outputs={fp_out[0]: gnpu})
        gap_npu = np.fromfile(gnpu, dtype=np.int8).reshape(env["gap"].shape)
        rec_gnpu = compare("gap(NPU)", gap_npu, env["gap"])
        fout3 = os.path.join(tmp, "fcnpu%d.bin" % i)
        C.simulate(clf_keaf, inputs={clf_in[0]: gnpu},
                   outputs={clf_out[0]: fout3})
        fc_npu = np.fromfile(fout3, dtype=np.int8).reshape(want.shape)
        rec_npu = compare("fc(all-NPU)", fc_npu, want)

        results["vectors"].append({
            "index": i,
            "features": rec_feat,
            "pool_on_host": rec_pool,
            "classifier_given_reference_pool": rec_clf,
            "end_to_end": rec_e2e,
            "pool_on_npu": rec_gnpu,
            "end_to_end_all_npu": rec_npu,
            "argmax_golden": int(want.argmax()),
            "argmax_sim": int(fc_e2e.argmax()),
            "argmax_agrees": int(want.argmax()) == int(fc_e2e.argmax()),
            "argmax_sim_all_npu": int(fc_npu.argmax()),
            "argmax_agrees_all_npu":
                int(want.argmax()) == int(fc_npu.argmax()),
        })

    # The simulator's own correctness gates: a read of never-written
    # scratchpad, or a cross-unit read of data whose producer has not retired,
    # is a missing SIGNAL/WAIT and is made fatal here rather than counted.
    strict = {}
    for label, keaf in (("features", feat_keaf), ("featpool", fp_keaf),
                        ("classifier", clf_keaf)):
        r = C.run([C.KEA_SIM, keaf, "--quiet", "--strict-poison",
                   "--strict-hazards"], check=False)
        strict[label] = r.ok
        print("strict poison/hazard check, %-11s %s"
              % (label, "clean" if r.ok else "FAILED: " + r.first_error()))
    results["strict_checks"] = strict

    every = results["vectors"]
    npu_fc_bad = sum(v["end_to_end_all_npu"]["mismatches"] for v in every)
    npu_gap_bad = sum(v["pool_on_npu"]["mismatches"] for v in every)
    fc_elems = sum(v["end_to_end_all_npu"]["elements"] for v in every)
    gap_elems = sum(v["pool_on_npu"]["elements"] for v in every)
    results["summary"] = {
        "vectors": len(every),
        "features_exact": all(v["features"]["exact"] for v in every),
        "classifier_exact": all(
            v["classifier_given_reference_pool"]["exact"] for v in every),
        "end_to_end_exact": all(v["end_to_end"]["exact"] for v in every),
        "argmax_agreement": sum(v["argmax_agrees"] for v in every),
        "all_npu_exact": all(v["end_to_end_all_npu"]["exact"] for v in every),
        "all_npu_gap_mismatches": npu_gap_bad,
        "all_npu_gap_elements": gap_elems,
        "all_npu_fc_mismatches": npu_fc_bad,
        "all_npu_fc_elements": fc_elems,
        "all_npu_max_abs_diff": max(
            [v["end_to_end_all_npu"]["max_abs_diff"] for v in every]
            + [v["pool_on_npu"]["max_abs_diff"] for v in every]),
        "all_npu_argmax_agreement": sum(
            v["argmax_agrees_all_npu"] for v in every),
        "programs": "scheduled builds at the compiler defaults",
        "caveat": "the bit-exact split runs the global average pool on the "
                  "host; the all-NPU split runs all 183 nodes on the machine "
                  "but inherits the frontend's avg_pool2d+rescale double "
                  "rounding (docs/RESULTS.md 1.1)",
    }
    s = results["summary"]
    print("\nbit-exact split:  features exact=%s, classifier exact=%s, "
          "end-to-end exact=%s, argmax %d/%d"
          % (s["features_exact"], s["classifier_exact"],
             s["end_to_end_exact"], s["argmax_agreement"], len(every)))
    print("all-NPU split:    gap %d/%d differ, logits %d/%d differ, "
          "max |diff| %d, argmax %d/%d"
          % (npu_gap_bad, gap_elems, npu_fc_bad, fc_elems,
             s["all_npu_max_abs_diff"], s["all_npu_argmax_agreement"],
             len(every)))

    C.write_json(os.path.join(C.RESULTS, "validation.json"), results)
    print("wrote demo/results/validation.json")
    return 0 if results["summary"]["end_to_end_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
