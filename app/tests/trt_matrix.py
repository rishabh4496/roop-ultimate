"""TensorRT build matrix: heuristics {on, off} x builder level {3, 5} per small model.

    env/Scripts/python.exe tests/trt_matrix.py prepare                 # real inputs + CUDA-FP32 references, once
    env/Scripts/python.exe tests/trt_matrix.py run [--models a,b]      # every (model, config, build); resumable
    env/Scripts/python.exe tests/trt_matrix.py report                  # docs/perf/trt_matrix.md (+ .json)

WHY. core.py gives EVERY engine `trt_build_heuristics_enable` (when precision is mixed) and builder level 3. Heuristics decide
which tactics a build picks and that is not neutral (swap_canary.py: the same options minus heuristics built a swapper engine
that ran at full speed and painted the wrong face). For the small models it is a per-model measurement.
`precision_policy.apply_build_override` (called from `_finalize`) is the mechanism: this harness drives it with
ROOP_TRT_MODEL_BUILD, so what is measured is the code path an adopted override takes.

WHAT IS MEASURED, per (model, config, build #1/#2 - two builds, each in its own engine AND timing cache):
  * build time (session creation + first inference, cold cache - the engine is built on the first run),
  * median of 200 runs with GPU-resident IO (torch CUDA tensors bound with IOBinding; every output bound on the device),
  * fidelity against a CUDA session of a GENUINE FP32 graph (every float16 tensor upcast; the shipped files may be FP16) on
    40 real crops/frames from d4, d1, d6, Love and s7: restoreformer_pp SSIM, xseg mask IoU, retinaface_r50 box IoU +
    landmark px + face counts, w600k embedding cosine, 2d106 / 1k3d68 landmark px (in the 192 crop).
  * proof the override ran: the live session's provider options (heuristics and level) are read back and recorded.

GATE. A build PASSES when none of its fidelity components is worse than the WORSE of the two baseline (heuristics on,
level 3) builds, widened by the baseline's own build-to-build difference on that component: the shipped engine is one draw
from that distribution, so a candidate inside it is not distinguishable from what is running now. The swapper is not in this
matrix and stays on heuristics until the explicit-FP32-island work is done (apply_build_override refuses 'face_swap').
"""
import argparse
import glob
import json
import os
import statistics as st
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.path.join(APP, "output", "trt_matrix")
N_INPUTS = 40
RUNS = 200
WARM = 30
CONFIGS = [(1, 3), (0, 3), (1, 5), (0, 5)]            # (heuristics, level); (1, 3) is today's global setting
REPS = (1, 2)
CLIPS = (("double/d4.mp4", 8), ("double/d1.mp4", 8), ("double/d6.mp4", 8), ("Love.mp4", 8), ("single/s7.mp4", 8))

#: stem -> how it is built in production and how its input/fidelity are made
MODELS = {
    "restoreformer_plus_plus": {"file": "restoreformer_plus_plus.onnx", "key": "restoreformer_pp", "bundle": False,
                                "metric": "ssim"},
    "xseg": {"file": "xseg.onnx", "key": "masking:xseg", "bundle": False, "metric": "mask_iou"},
    "retinaface_r50": {"file": "retinaface_r50.onnx", "key": "face_detection:r50", "bundle": False, "metric": "boxes"},
    # the buffalo_l bundle is built by FaceAnalysis with ONE chain and no model path (face_util.py:167); the harness gives
    # each its own path so a per-model override is measurable. Adoption there needs a loader change (see the report).
    "w600k_r50": {"file": "buffalo_l/w600k_r50.onnx", "key": "recognition:buffalo_l", "bundle": True, "metric": "cosine"},
    "2d106det": {"file": "buffalo_l/2d106det.onnx", "key": "recognition:buffalo_l", "bundle": True, "metric": "lm2d"},
    "1k3d68": {"file": "buffalo_l/1k3d68.onnx", "key": "recognition:buffalo_l", "bundle": True, "metric": "lm3d"},
}


def model_path(stem):
    return os.path.join(APP, "models", MODELS[stem]["file"])


#: parse_build_overrides caps a tag at 24 chars; 'restoreformer_plus_plus-h0l3-r1' is 31 and was REFUSED (first run: eight
#: builds all silently ran the shipped engine). build_one now also proves the override took effect.
SHORT = {"restoreformer_plus_plus": "rfpp"}


def tag_of(stem, cfg, rep):
    tag = "%s-h%dl%d-r%d" % (SHORT.get(stem, stem), cfg[0], cfg[1], rep)
    assert len(tag) <= 24, tag
    return tag


# ── inputs and references ─────────────────────────────────────────────────────────────────────────
def _init_pipeline():
    from settings import Settings
    import angle_bench as ab
    cfg = Settings(os.path.join(APP, "config.yaml"))
    return ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "None", sync_config=True)


def _real_faces(n):
    """[(frame, face)] - n real detections spread over the clips (largest face per sampled frame)."""
    import cv2
    import fixtures
    from roop.face_util import get_all_faces
    out = []
    for rel, k in CLIPS:
        path = fixtures.clip(rel)
        if not os.path.exists(path):
            continue
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frac in np.linspace(0.06, 0.94, k + 3):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
            ok, frame = cap.read()
            if not ok:
                continue
            faces = list(get_all_faces(frame) or [])
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                out.append((frame, face))
        cap.release()
    if len(out) < n:
        raise SystemExit("only %d real detections found, need %d (clips missing?)" % (len(out), n))
    step = len(out) / float(n)
    return [out[int(i * step)] for i in range(n)]


def make_inputs(stem, pairs):
    """float32 arrays, one per sample, shaped exactly as the production code feeds the model (B=1)."""
    import cv2
    from roop.face_util import align_crop
    xs = []
    for frame, face in pairs:
        kps = np.asarray(face.kps, np.float32)
        if stem == "restoreformer_plus_plus":
            crop, _ = align_crop(frame, kps, 512, mode="ffhq_512")
            rgb = crop[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
            xs.append((rgb / 127.5 - 1.0)[None])                                   # BUFFER_POOL.prepare_model_input
        elif stem == "xseg":
            crop, _ = align_crop(frame, kps, 256, mode="arcface")
            xs.append((crop.astype(np.float32) / 255.0)[None])                      # Mask_XSeg.Run: NHWC BGR / 255
        elif stem == "retinaface_r50":
            img = cv2.resize(frame, (640, 640))                                     # squash: what the pipeline feeds r50
            xs.append(cv2.dnn.blobFromImage(img, 1.0, (640, 640), (104.0, 117.0, 123.0), swapRB=False))
        elif stem == "w600k_r50":
            from insightface.utils import face_align
            aimg = face_align.norm_crop(frame, kps, image_size=112)
            xs.append(cv2.dnn.blobFromImage(aimg, 1.0 / 127.5, (112, 112), (127.5,) * 3, swapRB=True))
        else:                                                                         # 2d106det / 1k3d68: insightface Landmark.get
            from insightface.utils import face_align
            x1, y1, x2, y2 = [float(v) for v in face.bbox[:4]]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            scale = 192 / (max(x2 - x1, y2 - y1) * 1.5)
            aimg, _ = face_align.transform(frame, center, 192, scale, 0.0)
            xs.append(cv2.dnn.blobFromImage(aimg, 1.0, (192, 192), (0.0, 0.0, 0.0), swapRB=True))
    return np.concatenate(xs, axis=0).astype(np.float32)


def fp32_session(stem):
    import onnxruntime
    from roop.utilities import get_onnx_session_options
    from test_swap_batch_equivalence import fp32_graph
    return onnxruntime.InferenceSession(
        fp32_graph(model_path(stem)), get_onnx_session_options(),
        providers=[("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"])


def run_all(sess, xs, n_out):
    outs = [[] for _ in range(n_out)]
    name = sess.get_inputs()[0].name
    for i in range(len(xs)):
        res = sess.run(None, {name: np.ascontiguousarray(xs[i:i + 1])})
        for k in range(n_out):
            outs[k].append(np.asarray(res[k][0], np.float32))
    return [np.stack(o) for o in outs]


def n_outputs_used(stem):
    return 3 if stem == "retinaface_r50" else 1


def cmd_prepare(args):
    os.makedirs(OUT, exist_ok=True)
    _init_pipeline()
    pairs = _real_faces(N_INPUTS)
    for stem in MODELS:
        path = os.path.join(OUT, "ref_%s.npz" % stem)
        if os.path.exists(path) and not args.fresh:
            print("[prepare] %s cached" % stem, flush=True)
            continue
        xs = make_inputs(stem, pairs)
        sess = fp32_session(stem)
        outs = run_all(sess, xs, n_outputs_used(stem))
        del sess
        np.savez_compressed(path, x=xs, **{"o%d" % k: o for k, o in enumerate(outs)})
        print("[prepare] %s: inputs %s outputs %s" % (stem, xs.shape, [o.shape for o in outs]), flush=True)
    return 0


# ── fidelity ──────────────────────────────────────────────────────────────────────────────────────
def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _r50_faces(loc, conf, landms):
    from roop.retinaface import _decode_boxes, _decode_landmarks, _generate_priors
    from roop.nms import nms_keep
    scores = conf[:, 1]
    pos = np.where(scores >= 0.5)[0]
    pri = _generate_priors((640, 640))[pos]
    boxes = _decode_boxes(loc[pos], pri) * 640
    kps = _decode_landmarks(landms[pos], pri).reshape(-1, 5, 2) * 640
    det = np.hstack([boxes, scores[pos][:, None]]).astype(np.float32)
    order = det[:, 4].argsort()[::-1]
    det, kps = det[order], kps[order]
    keep = nms_keep(det, 0.4, offset=1.0) if len(det) else []
    return det[keep], kps[keep]


def fidelity(stem, got, ref):
    """Higher-is-better components, each as {'mean','worst'} so the gate can compare like with like."""
    metric = MODELS[stem]["metric"]
    n = len(ref[0])
    if metric == "ssim":
        from roop import swap_canary
        s = []
        for i in range(n):
            a = np.clip((got[0][i].transpose(1, 2, 0) + 1.0) * 127.5, 0, 255)
            b = np.clip((ref[0][i].transpose(1, 2, 0) + 1.0) * 127.5, 0, 255)
            s.append(swap_canary.ssim(a, b, 255.0))
        return {"ssim": {"mean": float(np.mean(s)), "worst": float(np.min(s))}}
    if metric == "mask_iou":
        s = []
        for i in range(n):
            a, b = got[0][i] > 0.5, ref[0][i] > 0.5
            u = np.logical_or(a, b).sum()
            s.append(1.0 if u == 0 else float(np.logical_and(a, b).sum() / u))
        return {"mask_iou": {"mean": float(np.mean(s)), "worst": float(np.min(s))}}
    if metric == "cosine":
        s = [float(np.dot(got[0][i], ref[0][i]) / (np.linalg.norm(got[0][i]) * np.linalg.norm(ref[0][i]) + 1e-12))
             for i in range(n)]
        return {"cosine": {"mean": float(np.mean(s)), "worst": float(np.min(s))}}
    if metric in ("lm2d", "lm3d"):
        def pts(v):
            if metric == "lm2d":
                p = v.reshape(-1, 2)
            else:
                p = v.reshape(-1, 3)[-68:, :2]
            return (p + 1.0) * 96.0
        e = [float(np.linalg.norm(pts(got[0][i]) - pts(ref[0][i]), axis=1).mean()) for i in range(n)]
        w = [float(np.linalg.norm(pts(got[0][i]) - pts(ref[0][i]), axis=1).max()) for i in range(n)]
        return {"lm_px_neg": {"mean": -float(np.mean(e)), "worst": -float(np.max(w))}}
    # boxes: per sample, match reference faces to the engine's by IoU
    ious, kp_err, missing, extra, n_ref = [], [], 0, 0, 0
    for i in range(n):
        rb, rk = _r50_faces(ref[0][i], ref[1][i], ref[2][i])
        gb, gk = _r50_faces(got[0][i], got[1][i], got[2][i])
        n_ref += len(rb)
        used = set()
        for a, ka in zip(rb, rk):
            best, bj = 0.0, None
            for j, b in enumerate(gb):
                if j not in used and _iou(a, b) > best:
                    best, bj = _iou(a, b), j
            if bj is None or best < 0.5:
                missing += 1
            else:
                used.add(bj)
                ious.append(best)
                kp_err.append(float(np.linalg.norm(ka - gk[bj], axis=1).mean() * 1.0))
        extra += len(gb) - len(used)
    return {"box_iou": {"mean": float(np.mean(ious)) if ious else 0.0, "worst": float(np.min(ious)) if ious else 0.0},
            "kps_px_neg": {"mean": -float(np.mean(kp_err)) if kp_err else 0.0,
                           "worst": -float(np.max(kp_err)) if kp_err else 0.0},
            "faces_kept_frac": {"mean": 1.0 - missing / max(1, n_ref), "worst": 1.0 - missing / max(1, n_ref)},
            "no_extra_faces": {"mean": -float(extra), "worst": -float(extra)}}


# ── one build ─────────────────────────────────────────────────────────────────────────────────────
def time_gpu_io(sess, x, runs=RUNS, warm=WARM):
    import torch
    name = sess.get_inputs()[0].name
    t = torch.from_numpy(np.ascontiguousarray(x)).cuda().contiguous()
    io = sess.io_binding()
    io.bind_input(name, "cuda", 0, np.float32, tuple(t.shape), t.data_ptr())
    for o in sess.get_outputs():
        io.bind_output(o.name, "cuda", 0)
    for _ in range(warm):
        sess.run_with_iobinding(io)
    ts = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sess.run_with_iobinding(io)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {"median_ms": round(st.median(ts), 4), "p10_ms": round(ts[len(ts) // 10], 4),
            "p90_ms": round(ts[(len(ts) * 9) // 10], 4), "runs": len(ts)}


def build_one(stem, cfg, rep, data):
    import onnxruntime
    import roop.globals as g
    from roop import precision_policy as pp
    from roop.utilities import get_onnx_session_options
    spec, path = MODELS[stem], model_path(stem)
    tag = tag_of(stem, cfg, rep)
    os.environ[pp.BUILD_OVERRIDE_ENV] = "%s:h=%d,l=%d,tag=%s" % (stem, cfg[0], cfg[1], tag)
    del pp.override_log[:]
    pp._override_announced.clear()
    if spec["bundle"]:
        chain, _ = pp.providers_for(spec["key"], g.execution_providers)           # what FaceAnalysis gets: no model path
        chain = pp.apply_build_override(chain, spec["key"], path)                  # what a per-model loader would add
    else:
        chain, _ = pp.providers_for(spec["key"], g.execution_providers, path)
    trt = next((p for p in chain if isinstance(p, tuple) and "tensorrt" in p[0].lower()), None)
    if trt is None:
        raise SystemExit("no TensorRT provider in the chain: this machine cannot run the matrix")
    cache = trt[1]["trt_engine_cache_path"]
    cold = not glob.glob(os.path.join(cache, "*.engine"))
    x = data["x"]
    name = None
    t0 = time.perf_counter()
    sess = onnxruntime.InferenceSession(path, get_onnx_session_options(), providers=chain)
    t1 = time.perf_counter()
    name = sess.get_inputs()[0].name
    sess.run(None, {name: np.ascontiguousarray(x[0:1])})                          # TensorRT builds the engine HERE
    t2 = time.perf_counter()
    live = (sess.get_provider_options() or {}).get("TensorrtExecutionProvider") or {}
    if sess.get_providers()[0] != "TensorrtExecutionProvider":
        raise SystemExit("%s: session is on %s, not TensorRT" % (tag, sess.get_providers()[0]))
    # PROVE THE CONFIG RAN. The live session's options, the override log and the cache directory must all say what was asked.
    applied = [e for e in pp.override_log if e.get("stem") == stem.lower() and "refused" not in e]
    want = {"trt_build_heuristics_enable": str(cfg[0]), "trt_builder_optimization_level": str(cfg[1])}
    got_live = {k: str(live.get(k)) for k in want}
    if not applied or got_live != want or ("_ovh%dl%d_%s" % (cfg[0], cfg[1], tag)) not in cache or not cold:
        raise RuntimeError("%s: override NOT in effect (applied=%s live=%s want=%s cache=%s cold=%s log=%s)" % (
            tag, bool(applied), got_live, want, cache, cold, pp.override_log))
    got = run_all(sess, x, n_outputs_used(stem))
    ref = [data["o%d" % k] for k in range(n_outputs_used(stem))]
    rec = {"model": stem, "heuristics": cfg[0], "level": cfg[1], "rep": rep, "tag": tag, "cold": cold,
           "create_s": round(t1 - t0, 2), "first_run_s": round(t2 - t1, 2), "build_s": round(t2 - t0, 2),
           "override_log": list(pp.override_log),
           "live_options": {k: live.get(k) for k in ("trt_build_heuristics_enable", "trt_builder_optimization_level",
                                                       "trt_engine_cache_path")},
           "fidelity": fidelity(stem, got, ref)}
    rec["timing"] = time_gpu_io(sess, x[0:1])
    del sess
    return rec


def cmd_run(args):
    os.makedirs(os.path.join(OUT, "builds"), exist_ok=True)
    _init_pipeline()
    stems = [s for s in (args.models.split(",") if args.models else MODELS) if s]
    for stem in stems:
        data = dict(np.load(os.path.join(OUT, "ref_%s.npz" % stem)))
        for rep in REPS:
            for cfg in CONFIGS:
                path = os.path.join(OUT, "builds", tag_of(stem, cfg, rep) + ".json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                rec = build_one(stem, cfg, rep, data)
                json.dump(rec, open(path, "w"), indent=1)
                print("[matrix] %-26s build %6.1fs  median %.3f ms  live heur=%s level=%s  %s  (%.0f min)" % (
                    rec["tag"], rec["build_s"], rec["timing"]["median_ms"],
                    rec["live_options"]["trt_build_heuristics_enable"], rec["live_options"]["trt_builder_optimization_level"],
                    {k: round(v["mean"], 5) for k, v in rec["fidelity"].items()}, (time.time() - t0) / 60), flush=True)
    print("[matrix] ALL_DONE", flush=True)
    return 0


# ── report ────────────────────────────────────────────────────────────────────────────────────────
def load_builds():
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "builds", "*.json")))]


def gate(builds):
    """Per model: baseline floor per component (worse of the two baseline builds, widened by their difference)."""
    out = {}
    for stem in MODELS:
        base = [b for b in builds if b["model"] == stem and (b["heuristics"], b["level"]) == (1, 3)]
        if len(base) < 2:
            continue
        floor = {}
        for comp in base[0]["fidelity"]:
            for kind in ("mean", "worst"):
                vals = [b["fidelity"][comp][kind] for b in base]
                floor[(comp, kind)] = min(vals) - (max(vals) - min(vals))
        out[stem] = floor
    return out


def cmd_report(args):
    builds = load_builds()
    floors = gate(builds)
    rows, md = [], []
    for stem in MODELS:
        bs = [b for b in builds if b["model"] == stem]
        if not bs or stem not in floors:
            continue
        for b in bs:
            b["pass"] = all(b["fidelity"][c][k] >= floors[stem][(c, k)] - 1e-9 for (c, k) in floors[stem])
        base = [b for b in bs if (b["heuristics"], b["level"]) == (1, 3)]
        base_ok = [b for b in base if b["pass"]]
        base_best = min(b["timing"]["median_ms"] for b in (base_ok or base))
        base_spread = (max(b["timing"]["median_ms"] for b in base) - min(b["timing"]["median_ms"] for b in base)) / base_best
        entry = {"model": stem, "baseline_ms": base_best, "baseline_spread_pct": round(100 * base_spread, 1),
                 "baseline_build_s": [b["build_s"] for b in base], "configs": []}
        for cfg in CONFIGS:
            cb = [b for b in bs if (b["heuristics"], b["level"]) == cfg]
            passing = [b for b in cb if b["pass"]]
            keep = min(passing, key=lambda b: b["timing"]["median_ms"]) if passing else None
            noise = max(2.0, 100 * base_spread)
            gain = 100.0 * (base_best - keep["timing"]["median_ms"]) / base_best if keep else None
            entry["configs"].append({
                "heuristics": cfg[0], "level": cfg[1], "builds": len(cb), "passing": len(passing),
                "median_ms": [b["timing"]["median_ms"] for b in cb], "build_s": [b["build_s"] for b in cb],
                "kept_ms": keep["timing"]["median_ms"] if keep else None, "kept_rep": keep["rep"] if keep else None,
                "gain_pct_vs_baseline": round(gain, 1) if gain is not None else None, "noise_pct": round(noise, 1),
                "fidelity_mean": {c: [round(b["fidelity"][c]["mean"], 5) for b in cb] for c in cb[0]["fidelity"]} if cb else {},
                "fidelity_worst": {c: [round(b["fidelity"][c]["worst"], 5) for b in cb] for c in cb[0]["fidelity"]} if cb else {},
                "adopt": bool(keep and cfg != (1, 3) and gain is not None and gain > noise)})
        rows.append(entry)
    lines = ["# TensorRT build matrix: heuristics x builder level, per model", "",
             "Generated by `app/tests/trt_matrix.py report` from `app/output/trt_matrix/builds/*.json` (kept as ",
             "`docs/perf/trt_matrix.json`). Gate and method: see the module docstring.", ""]
    for e in rows:
        lines += ["## %s" % e["model"], "",
                  "baseline (heuristics on, level 3): %.3f ms (builds differ by %.1f%%), build %s s" % (
                      e["baseline_ms"], e["baseline_spread_pct"], e["baseline_build_s"]), "",
                  "| heur | level | builds (pass) | median ms per build | kept ms | vs baseline | noise | build s | fidelity mean per build | adopt |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c in e["configs"]:
            lines.append("| %s | %d | %d (%d) | %s | %s | %s | ±%.1f%% | %s | %s | %s |" % (
                "on" if c["heuristics"] else "off", c["level"], c["builds"], c["passing"], c["median_ms"], c["kept_ms"],
                ("%+.1f%%" % c["gain_pct_vs_baseline"]) if c["gain_pct_vs_baseline"] is not None else "-", c["noise_pct"],
                c["build_s"], json.dumps(c["fidelity_mean"]), "**yes**" if c["adopt"] else "no"))
        lines.append("")
    path = os.path.join(REPO, "docs", "perf", "trt_matrix.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump({"rows": rows, "builds": builds}, open(os.path.join(REPO, "docs", "perf", "trt_matrix.json"), "w"),
              indent=1, default=str)
    print("\n".join(lines))
    print("wrote", path)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prepare", "run", "report"))
    ap.add_argument("--models", default="")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    return {"prepare": cmd_prepare, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
