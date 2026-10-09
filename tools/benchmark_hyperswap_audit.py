"""Stage 5 benchmark: HyperSwap / InSwapper / RealSwap quality and performance audit - rebuilt on tools/quality_harness.py.

WHY THIS FILE WAS REWRITTEN (2026-10-09)
----------------------------------------
The previous version did not measure what its table said. It printed:
  * "identity similarity" typed in per model (0.88 / 0.84 / 0.81 / 0.865 / 0.86) plus Gaussian noise;
  * "eye / mouth alignment error" as a keypoint set compared with ITSELF plus noise;
  * "profile quality", "occlusion robustness", "temporal consistency" as constants (94.5 / 88.0, 92.0 / 85.5, 96.2 / 95.8);
  * results from ONE crop, with random embeddings, on a grey frame;
  * and ran `hyperswap_1a_256.onnx` under the names hyperswap_1b, hyperswap_1c and realswap, so the "different
    models" were one network.
`benchmark_stage5_hyperswap.json` and `STAGE5_HYPERSWAP_AUDIT_REPORT.md` were produced by that code and are INVALID.

WHAT IT DOES NOW
----------------
Every quality number comes from `tools/quality_harness.py`: real renders of real footage, scored against a full-FP32
reference with an independent recogniser (AdaFace), masked SSIM / PSNR on the face region, keypoint drift between the
plate and the swapped output, high-frequency detail, frame-to-frame identity jitter, and yaw-binned identity. The
harness FAILS the run if two different models produce an identical metric, or if two names load the same network
files. A variant whose ONNX file is not on disk is SKIPPED and reported as skipped; it is never run under another
model's file (`--download-missing` lets the pipeline fetch it). The one thing kept from the old file is genuine: the
source-latent cache timing, now measured on a real source embedding taken from a real faceset.

NOT MEASURED, and said so: "occlusion robustness". There is no per-face occluder ground truth in the footage, so there
is nothing to score it against; the swap-audit counts of faces "partly behind an object" are reported per render in
the harness report instead of a made-up percentage.

    env/Scripts/python.exe tools/benchmark_hyperswap_audit.py [--clips d4,s7] [--frames 300] [--out DIR] [--reuse]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APP = os.path.join(ROOT, "app")
for _p in (APP, os.path.join(APP, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location("quality_harness", os.path.join(HERE, "quality_harness.py"))
qh = importlib.util.module_from_spec(_spec)
sys.modules["quality_harness"] = qh
_spec.loader.exec_module(qh)

# (candidate name, SWAP_MODELS key). The files each one needs are read from SWAP_MODELS, not typed here.
VARIANTS: List[Tuple[str, str]] = [
    ("hyperswap_1a", "hyperswap"),
    ("hyperswap_1b", "hyperswap_1b"),
    ("hyperswap_1c", "hyperswap_1c"),
    ("inswapper_128", "inswapper"),
    ("realswap", "realswap"),
]
# realswap is hyperswap_1a plus a HiFiFace eyelid/eyelash band: it needs these as well.
EXTRA_FILES = {"realswap": ["hififace_unofficial_256.onnx", "crossface_hififace.onnx"]}
OUT_JSON = os.path.join(ROOT, "benchmark_stage5_hyperswap.json")


def availability(download_missing: bool) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """Which variants can really run. A missing ONNX is a SKIP with the reason, never a silent stand-in."""
    from roop.processors.FaceSwapInsightFace import SWAP_MODELS
    models = os.path.join(APP, "models")
    runnable, skipped = [], []
    for name, key in VARIANTS:
        spec = SWAP_MODELS.get(key)
        if spec is None:
            skipped.append({"variant": name, "reason": "swap model key %r is not registered" % key})
            continue
        need = [spec["file"]] + EXTRA_FILES.get(name, [])
        absent = [f for f in need if not os.path.exists(os.path.join(models, f))]
        if absent and not download_missing:
            skipped.append({"variant": name, "reason": "%s not on disk (the pipeline would download it from %s); "
                            "pass --download-missing to allow that" % (", ".join(absent), spec.get("url", "?"))})
            continue
        runnable.append((name, key))
    return runnable, skipped


def source_cache_microbenchmark(source_name: str = "harjot", iterations: int = 3000) -> Dict[str, Any]:
    """Latent preparation with and without the Stage 5 source cache, on a REAL source embedding.

    This times one function (`HyperSwapSourceCache.get_latent`), not a swap: it says how much the cache saves per
    frame in latent preparation and nothing about quality or end-to-end speed."""
    import two_face_video as tfv
    from roop.hyperswap_optimizer import HyperSwapSourceCache
    fs = tfv.load_library_faceset(source_name)
    emb = np.asarray(fs.faces[0].embedding, np.float32)          # the vector the swapper actually consumes
    t0 = time.perf_counter()
    for _ in range(iterations):
        HyperSwapSourceCache().get_latent({"embedding": emb.copy()}, model_key="hyperswap", embedding_mode="normed")
    uncached = (time.perf_counter() - t0) / iterations
    cache, face = HyperSwapSourceCache(), {"embedding": emb}
    cache.get_latent(face, model_key="hyperswap", embedding_mode="normed")          # fill
    t0 = time.perf_counter()
    for _ in range(iterations):
        cache.get_latent(face, model_key="hyperswap", embedding_mode="normed")
    cached = (time.perf_counter() - t0) / iterations
    return {"source": source_name, "embedding_dim": int(emb.size), "iterations": iterations,
            "uncached_us_per_call": round(uncached * 1e6, 3), "cached_us_per_call": round(cached * 1e6, 3),
            "speedup": round(uncached / cached, 2) if cached > 0 else None,
            "scope": "latent preparation only; not a swap, not quality, not end-to-end"}


def table(report: Dict[str, Any], skipped: List[Dict[str, str]]) -> None:
    print("\n" + "=" * 150)
    print("%-15s | %-5s | %-7s | %-8s | %-7s | %-6s | %-8s | %-8s | %-7s | %-8s | %s" % (
        "VARIANT", "CLIP", "FACES", "ID COS", "D ID", "SSIM", "EYE PX", "MOUTH PX", "DETAIL", "JITTER", "FPS*"))
    print("=" * 150)
    for name, res in report["candidates"].items():
        for clip, m in res["per_clip"].items():
            g = lambda k: (m.get(k) or {}).get("mean")                                       # noqa: E731
            f = lambda x, d=4: "-" if x is None else ("%.*f" % (d, x))                        # noqa: E731
            print("%-15s | %-5s | %-7d | %-8s | %-7s | %-6s | %-8s | %-8s | %-7s | %-8s | %s" % (
                name, clip, m["faces_scored"], f(g("identity_cos")), f(g("identity_delta_vs_reference")), f(g("ssim"), 4),
                f(g("eye_drift_px"), 2), f(g("mouth_drift_px"), 2),
                f((m.get("skin_detail_ratio_vs_reference") or {}).get("mean"), 3), f(g("identity_jitter")),
                f(m["render"]["fps"], 2)))
    print("=" * 150)
    for s in skipped:
        print("SKIPPED %-15s %s" % (s["variant"], s["reason"]))
    print("NOT MEASURED  occlusion robustness: no per-face occluder ground truth exists in the footage")
    print("* FPS is the frame loop of a render with a LOSSLESS x264 encode (CPU-bound, ~56 Mb/s): it says nothing about "
          "model or render speed. Measure speed with a counterbalanced ABBA (tests/ab_stab_dedup.py).")


def main() -> int:
    ap = argparse.ArgumentParser(description="HyperSwap quality & performance audit (real renders, FP32 reference)")
    ap.add_argument("--clips", default="d4,s7")
    ap.add_argument("--frames", type=int, default=qh.DEFAULT_FRAMES)
    ap.add_argument("--out", default=os.path.join(APP, "output", "hyperswap_audit_" + time.strftime("%Y-%m-%d")))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--chunk-mb", type=int, default=1500)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--download-missing", action="store_true")
    ap.add_argument("--strict-all-metrics", action="store_true")
    ap.add_argument("--from-report", action="store_true",
                    help="skip rendering/scoring: rebuild the JSON and table from <out>/quality_report.json")
    args = ap.parse_args()
    # The harness chdir()s into app/, so a relative --out must be made absolute BEFORE anything runs.
    out_abs = os.path.abspath(args.out)
    report_path = os.path.join(out_abs, "quality_report.json")

    runnable, skipped = availability(args.download_missing)
    if not runnable:
        print("nothing to audit: no variant's ONNX files are on disk")
        return 1
    rc = 0
    if args.from_report:
        if not os.path.exists(report_path):
            print("--from-report: %s does not exist" % report_path)
            return 1
        print("=== HYPERSWAP AUDIT: rebuilt from %s (no render, no scoring) ===" % report_path)
    else:
        candidates = ["%s|swap_model=%s|trt_precision=mixed" % (n, k) for n, k in runnable]
        print("=== HYPERSWAP AUDIT: %d variants x clips %s x %d frames, vs a full-FP32 reference ===" % (
            len(runnable), args.clips, args.frames))
        for s in skipped:
            print("  skipping %s: %s" % (s["variant"], s["reason"]))
        rc = qh.run(argparse.Namespace(clips=args.clips, frames=args.frames, candidate=candidates,
                                       reference_swap_model=None, out=out_abs, threads=args.threads,
                                       chunk_mb=args.chunk_mb, reuse=args.reuse,
                                       strict_all_metrics=args.strict_all_metrics))
    report = json.load(open(report_path, encoding="utf-8"))
    micro = None
    try:
        micro = source_cache_microbenchmark()
    except Exception as exc:                                              # a failed micro-benchmark is reported, not faked
        micro = {"error": "%s: %s" % (type(exc).__name__, exc)}
    result = {
        "schema_version": 2, "generated_by": "tools/benchmark_hyperswap_audit.py via tools/quality_harness.py",
        "supersedes": "the pre-2026-10-09 file of this name, whose quality values were typed in or random (see the "
                      "module docstring); that output was INVALID",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "frames": args.frames, "clips": args.clips.split(","),
        "reference": report["reference"], "skipped_variants": skipped,
        "variants": report["candidates"], "xseg_iou_by_precision": report["xseg_iou_by_precision"],
        "per_model_runtime_log": report["per_model_runtime_log"], "instrument_controls": report["instrument_controls"],
        "source_cache_microbenchmark": micro,
        "not_measured": {"occlusion_robustness": "no per-face occluder ground truth in the footage; the swap-audit "
                                                 "'partly behind an object' counts are in variants[*].per_clip[*].swap_audit"},
        "guard": report["guard"], "harness_report": report_path,
    }
    json.dump(result, open(OUT_JSON, "w", encoding="utf-8"), indent=2, default=str)
    table(report, skipped)
    print("\nsource-latent cache micro-benchmark:", json.dumps(micro))
    print("wrote", OUT_JSON)
    return rc


if __name__ == "__main__":
    sys.exit(main())
