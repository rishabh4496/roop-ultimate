"""What do the restorer's FP16 / island variants cost on the FINAL composited output, not on the 512 crop?

    app/env/Scripts/python.exe tools/restorer_composite_eval.py --clips d4,s7 --frames 300 \\
        --reference "rf_fp32|env.ROOP_TRT_MODEL_PRECISION=restoreformer_plus_plus:fp32" \\
        --variant "prod" \\
        --variant "isl_enc_quant|env.ROOP_RESTORER_NATIVE_ENGINE=<abs path to engine>"

The reference differs from every variant in ONE thing: the restorer's precision (RestoreFormer++ on TensorRT FP32, via the
per-model precision table). Detector, masks, swapper, stabiliser, paste-back are the live config.yaml in all of them, the
target people come from the reference's saved fixture (pinned), the stabiliser block size is pinned, and the encode is
lossless x264 - so a pixel difference between two renders IS the restorer. This reuses tools/quality_harness.py's
`render()` (the app's real path) and `Scorer.score()` (SSIM / PSNR of the composited face, AdaFace identity to the source,
landmark drift), but not its `run()`: that loop is built to compare swap models and its guards would reject a comparison
whose swapper is the same by design.

Each render's own session log is checked: the restorer session must exist and have the precision the variant claims
(FP32 reference: trt_fp16 off; prod: on; native: islands). A variant that did not run its engine fails the run instead of
reporting the production number under another name. Frame rate is not reported: a lossless x264 encode is CPU-bound.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APP = os.path.join(ROOT, "app")
for _p in (APP, os.path.join(APP, "tests"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import quality_harness as qh          # noqa: E402

TAG = "enhancer:restoreformer++"


def _variant(spec, swap_model):
    if "swap_model=" not in spec:
        spec = spec + "|swap_model=" + swap_model
    return qh.parse_candidate(spec)


def _restorer_session(meta):
    rows = [r for r in meta["sessions"] if r["tag"] == TAG]
    return rows[0] if rows else None


def _expect(meta, want):
    row = _restorer_session(meta)
    if row is None:
        return ["no [Session] %s in the render log: the restorer did not run (or ran under another name)" % TAG]
    if want == "fp32" and row["trt_fp16"] != "off":
        return ["restorer session is trt_fp16=%s, expected off (%s)" % (row["trt_fp16"], row["provider"])]
    if want == "mixed" and (row["trt_fp16"] != "on" or "Tensorrt" not in row["provider"]):
        return ["restorer session is %s trt_fp16=%s, expected TensorRT fp16 on" % (row["provider"], row["trt_fp16"])]
    if want == "native" and row["provider"] != "TensorrtNativeEngine":
        return ["restorer session is %s, expected the native engine" % row["provider"]]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", default="d4,s7")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--variant", action="append", default=[], required=True)
    ap.add_argument("--out", default=os.path.join(APP, "output", "restorer_fp16", "composite"))
    ap.add_argument("--chunk-mb", type=int, default=1500)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    table = qh.clip_table()
    clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    threads = int(cfg.max_threads)
    swap_model = str(cfg.swap_model)
    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)
    ref_v = _variant(args.reference, swap_model)
    variants = [_variant(v, swap_model) for v in args.variant]
    names = [ref_v.name] + [v.name for v in variants]
    if len(set(names)) != len(names):
        raise SystemExit("variant names must be unique")

    t0 = time.time()
    metas = {n: {} for n in names}
    for clip in clips:
        fx = os.path.join(out_root, "%s.fixture.pkl" % clip)
        metas[ref_v.name][clip] = qh.render(ref_v, clip, table[clip], args.frames, cfg, out_root, threads, args.chunk_mb,
                                            args.reuse, fixture_out=fx)
        for v in variants:
            metas[v.name][clip] = qh.render(v, clip, table[clip], args.frames, cfg, out_root, threads, args.chunk_mb,
                                            args.reuse, fixture_in=fx)
    problems = []
    for name, per in metas.items():
        want = "fp32" if name == ref_v.name else ("native" if any(
            "NATIVE_ENGINE" in k for k in next(v for v in variants if v.name == name).env) else "mixed")
        for clip, m in per.items():
            if not m["lossless_encode"]:
                problems.append("%s/%s: not a lossless encode" % (name, clip))
            problems += ["%s/%s: %s" % (name, clip, p) for p in _expect(m, want)]
    if problems:
        raise SystemExit("the renders are not what they claim:\n  " + "\n  ".join(problems))

    scorer = qh.Scorer(cfg, os.path.join(APP, "facesets"))
    report = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "frames": args.frames, "clips": clips, "reference": ref_v.name,
              "reference_env": ref_v.env, "variants": {}}
    for v in variants:
        per_clip = {}
        for clip in clips:
            sc = scorer.score(clip, table[clip], metas[ref_v.name][clip], metas[v.name][clip], args.frames,
                              os.path.join(out_root, "scores", "%s__%s.csv" % (v.name, clip)))
            per_clip[clip] = sc
        report["variants"][v.name] = {"env": v.env, "restorer_session": {c: _restorer_session(metas[v.name][c]) for c in clips},
                                      "metrics": qh.flatten_metrics(per_clip),
                                      "per_clip": {c: qh.flatten_metrics({c: per_clip[c]}) for c in clips},
                                      "identical_to_reference": {c: metas[v.name][c]["output_sha256"] == metas[ref_v.name][c]["output_sha256"]
                                                                 for c in clips}}
    report["wall_minutes"] = round((time.time() - t0) / 60.0, 1)
    path = os.path.join(out_root, "composite_report.json")
    json.dump(report, open(path, "w", encoding="utf-8"), indent=1, default=str)
    for name, r in report["variants"].items():
        m = r["metrics"]
        print("[composite] %-18s %s" % (name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
