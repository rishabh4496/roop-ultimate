"""Calibrate the aux canary's floors (roop/aux_canary.py SPECS) against real faces and against a wrong-picture control.

    env/Scripts/python.exe tests/aux_canary_calibration.py            # needs output/trt_precision_fidelity/capture.npz
                                                                      # (tests/trt_precision_fidelity.py capture)
    env/Scripts/python.exe tests/aux_canary_calibration.py shipped    # the SHIPPED per-model precisions, built by the app's own
                                                                      # loaders: proves each session runs the precision the table
                                                                      # says, then measures it on the same real faces

Writes docs/perf/aux_canary_calibration.json and .md. For each of xseg, w600k_r50, 2d106det, 1k3d68 it measures the CANARY'S OWN
metric (aux_canary.compare), three ways, with the CUDA FP32 (TF32 off) session as the reference:

  good    the shipped TensorRT 'mixed' engine and the TensorRT FP32 engine on the ~500 REAL captured faces - what a healthy
          engine scores at its worst, i.e. the number a floor must stay clear of;
  canary  the same two engines on the two synthetic canary inputs - what a healthy engine scores on the inputs the canary
          really uses (these must pass, with room);
  wrong   the reference's output for face i scored against the reference's output for face i+1 - an engine that answers for a
          DIFFERENT face (the swap-canary failure class: builds, runs at speed, wrong picture), plus an all-zero output - what
          a floor must catch.

A floor is acceptable when  max(good, canary)  <  floor  <  min(wrong, zeros)  with a margin on both sides; the report prints
both sides and the ratio so the choice is checkable. The mixed arm is built through the app's own loaders with the per-model
precision pinned to mixed (ROOP_TRT_MODEL_PRECISION), because the shipped table moves three of these models to FP32.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODE = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
if MODE == "calibrate":
    # Must be set before the loaders run: the shipped table would otherwise build xseg / 2d106det / 1k3d68 as FP32.
    os.environ.setdefault("ROOP_TRT_MODEL_PRECISION", "xseg:mixed;2d106det:mixed;1k3d68:mixed;w600k_r50:mixed")

import trt_precision_fidelity as fid          # noqa: E402


def _stats(v):
    v = np.asarray([x for x in v if np.isfinite(x)], np.float64)
    if not len(v):
        return {}
    return {"n": int(len(v)), "min": float(v.min()), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def main():
    fid._init_pipeline()
    from roop import aux_canary as ac
    data = np.load(os.path.join(fid.OUT, "capture.npz"), allow_pickle=True)
    mixed, _xs = fid.mixed_sessions()
    al, _bx = fid.xseg_inputs(data)
    real = {"xseg": al}
    for s in fid.AUX:
        real[s] = data[s + "_x"]
    out = {"date": time.strftime("%Y-%m-%d %H:%M"), "version": ac.VERSION, "models": {}}
    for stem in ("xseg", "w600k_r50", "2d106det", "1k3d68"):
        spec = ac.SPECS[stem]
        ref, t32 = fid.ref_session(stem), fid.trt32_session(stem)
        x = real[stem]
        r, m, t = fid.run_all(ref, x), fid.run_all(mixed[stem], x), fid.run_all(t32, x)
        rec = {"floors": dict(spec.floors), "good_real": {}, "canary_synth": {}, "wrong": {}}
        metrics = [k for k, _ in spec.floors]
        for arm, o in (("mixed", m), ("trt32", t)):
            rows = [ac.compare(stem, o[i:i + 1], r[i:i + 1]) for i in range(len(x))]
            rec["good_real"][arm] = {k: _stats([row[k] for row in rows]) for k in metrics}
        # wrong picture: another real face's output; zeros: a collapsed output
        rows = [ac.compare(stem, r[(i + 1) % len(r):(i + 1) % len(r) + 1], r[i:i + 1]) for i in range(len(r))]
        rec["wrong"]["other_face"] = {k: _stats([row[k] for row in rows]) for k in metrics}
        rows = [ac.compare(stem, np.zeros_like(r[i:i + 1]), r[i:i + 1]) for i in range(len(r))]
        rec["wrong"]["zeros"] = {k: _stats([row[k] for row in rows]) for k in metrics}
        feeds = ac.build_feeds(stem, mixed[stem])
        crop = ac._shape_hw(mixed[stem])[0]
        for arm, sess in (("mixed", mixed[stem]), ("trt32", t32)):
            cases = {}
            for (kind, seed), feed in zip(ac._CASES, feeds):
                got = sess.run(None, feed)[0]
                want = ref.run(None, feed)[0]
                cases["%s%d" % (kind, seed)] = ac.compare(stem, got, want, crop)
            rec["canary_synth"][arm] = cases
        out["models"][stem] = rec
        print("[calib] %-10s done" % stem, flush=True)
        del ref, t32
    path = os.path.join(REPO, "docs", "perf", "aux_canary_calibration.json")
    json.dump(out, open(path, "w"), indent=1)
    write_md(out, os.path.join(REPO, "docs", "perf", "aux_canary_calibration.md"))
    print("wrote", path)
    return 0


def write_md(out, path):
    lines = ["# Aux canary floor calibration", "",
             "Generated by `app/tests/aux_canary_calibration.py` (%s). Metric = `aux_canary.compare`, reference = CUDA FP32 with TF32 "
             "off. `good` = healthy engines on ~500 real faces; `canary` = the same engines on the two synthetic canary inputs; "
             "`wrong` = an engine that answers for a different face / returns zeros. A floor must sit above `good`/`canary` and "
             "below `wrong`." % out["date"], ""]
    for stem, rec in out["models"].items():
        lines += ["## %s" % stem, "", "Floors: %s" % ", ".join("`%s` %s %g" % (k, ">=" if k in ("cosine", "iou") else "<=", v)
                                                              for k, v in rec["floors"].items()), "",
                  "| metric | mixed real (median / p95 / WORST) | trt32 real (WORST) | canary synth mixed / trt32 (worst case) | "
                  "other face (p5 / median)* | zeros (BEST) |", "|---|---|---|---|---|---|"]
        for k in rec["floors"]:
            low = k in ("cosine", "iou")
            pick = min if low else max
            best = max if low else min
            g = rec["good_real"]
            mx = g["mixed"][k]
            worst = lambda s: s["min" if low else "max"]
            syn = {arm: pick(c[k] for c in cases.values()) for arm, cases in rec["canary_synth"].items()}
            w, z = rec["wrong"]["other_face"][k], rec["wrong"]["zeros"][k]
            tail = "p95" if low else "p5"
            lines.append("| %s | %.5g / %.5g / **%.5g** | %.5g | %.5g / %.5g | %.5g / %.5g | %.5g |" % (
                k, mx["median"], mx["p95"], worst(mx), worst(g["trt32"][k]), syn["mixed"], syn["trt32"],
                w[tail], w["median"], z["max" if low else "min"]))
        lines.append("")
    lines += ["\\* neighbouring captured faces are near-duplicates, so the minimum (maximum for cosine / IoU) over the 500 pairs is "
              "0 (1) by construction; the 5th (95th) percentile and the median are the honest 'wrong face' figures.", ""]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def shipped():
    """The live sessions of the shipped config: assert each runs the precision the table says, then measure it."""
    fid._init_pipeline()
    import roop.globals as g
    from roop import face_util, precision_policy as pp
    from roop.processors.Mask_XSeg import Mask_XSeg
    xs = Mask_XSeg()
    xs.Initialize({"devicename": "cuda"})
    face_util._ensure_face_analyser()
    fa = face_util.get_face_analyser()
    live = {"xseg": xs.model_xseg}
    for stem, task in fid.AUX.items():
        live[stem] = (fa.models.get(task) or getattr(fa, "lm68_model", None)).session
    glob = str(getattr(g.CFG, "trt_precision", "mixed"))
    data = np.load(os.path.join(fid.OUT, "capture.npz"), allow_pickle=True)
    al, _bx = fid.xseg_inputs(data)
    inputs = {"xseg": al}
    for s in fid.AUX:
        inputs[s] = data[s + "_x"]
    out = {"date": time.strftime("%Y-%m-%d %H:%M"), "global_precision": glob, "models": {}}
    for stem, sess in live.items():
        want = pp.precision_override_for(fid.KEYS[stem], fid.model_path(stem)) or glob
        opts = (sess.get_provider_options() or {}).get("TensorrtExecutionProvider") or {}
        fp16 = str(opts.get("trt_fp16_enable")).lower() in ("1", "true")
        if sess.get_providers()[0] != "TensorrtExecutionProvider" or fp16 != (want in ("fp16", "mixed")):
            raise SystemExit("%s: table says %s but the live session is %s fp16=%s" % (stem, want, sess.get_providers()[0], fp16))
        ref = fid.ref_session(stem)
        x = inputs[stem]
        r, c = fid.run_all(ref, x), fid.run_all(sess, x)
        rec = {"precision": want, "trt_fp16_enable": fp16, "cache": os.path.basename(str(opts.get("trt_engine_cache_path"))),
               "timing_ms": fid.time_gpu_io(sess, x[0:1])}
        if stem == "xseg":
            ms = fid.mask_stats(c, r)
            rec["iou"] = fid.summ(ms["iou"])
            rec["iou_below_0.995"] = int(sum(1 for v in ms["iou"] if v < 0.995))
            rec["iou_below_0.95"] = int(sum(1 for v in ms["iou"] if v < 0.95))
            rec["boundary_p95_px"] = fid.summ(ms["boundary_p95_px"])
            rec["soft_mean_abs"] = fid.summ(ms["soft_mean_abs"])
        elif stem == "w600k_r50":
            cs = fid.cos_stats(c, r)
            rec["cosine"] = fid.summ(cs)
            rec["below_0.999"] = int(sum(1 for v in cs if v < 0.999))
        else:
            ls = fid.lm_stats(stem, c, r, data[stem + "_bbox"])
            rec["landmarks"] = {k: fid.summ(v) for k, v in ls.items()}
            rec["over_0.3px_faces"] = int(sum(1 for v in ls["mean_px"] if v > 0.3))
        out["models"][stem] = rec
        print("[shipped] %-10s %-5s fp16=%s ok" % (stem, want, fp16), flush=True)
        del ref
    path = os.path.join(REPO, "docs", "perf", "trt_precision_shipped_check.json")
    json.dump(out, open(path, "w"), indent=1)
    lines = ["# Shipped per-model precision: live sessions vs the CUDA FP32 reference", "",
             "Generated by `app/tests/aux_canary_calibration.py shipped` (%s), global `trt_precision` = `%s`. Each session was built by "
             "the app's own loader (`Mask_XSeg.Initialize`, the buffalo_l bundle), its `trt_fp16_enable` was read back and compared "
             "with `precision_policy`'s table, and it was then run on the same %d captured real faces as `trt_precision_fidelity.md`."
             % (out["date"], glob, len(al)), "",
             "| model | precision | engine cache | ms/call | result |", "|---|---|---|---|---|"]
    for stem, r in out["models"].items():
        if stem == "xseg":
            res = "IoU mean %.4f / min %.4f; %d faces < 0.995, %d < 0.95; boundary p95 mean %.3f px" % (
                r["iou"]["mean"], r["iou"]["min"], r["iou_below_0.995"], r["iou_below_0.95"], r["boundary_p95_px"]["mean"])
        elif stem == "w600k_r50":
            res = "cosine mean %.6f / min %.6f; %d faces < 0.999" % (r["cosine"]["mean"], r["cosine"]["min"], r["below_0.999"])
        else:
            m = r["landmarks"]["mean_px"]
            res = "frame px mean %.3f / p95 %.3f / max %.3f; %d faces > 0.3 px" % (m["mean"], m["p95"], m["max"], r["over_0.3px_faces"])
            if "kps5_mean_px" in r["landmarks"]:
                k = r["landmarks"]["kps5_mean_px"]
                res += "; refined kps mean %.3f / p95 %.3f px" % (k["mean"], k["p95"])
        lines.append("| %s | %s | `...%s` | %.3f | %s |" % (stem, r["precision"], r["cache"][-30:], r["timing_ms"]["median_ms"], res))
    open(os.path.join(REPO, "docs", "perf", "trt_precision_shipped_check.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(shipped() if MODE == "shipped" else main())
