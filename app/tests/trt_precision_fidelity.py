"""Shipped TensorRT 'mixed' vs a genuine FP32 reference, on ~500 REAL faces, for the four small models that ride the global
trt_precision: XSeg, w600k_r50, 2d106det, 1k3d68.

    env/Scripts/python.exe tests/trt_precision_fidelity.py capture [--faces 500]   # detect real faces, store the exact model inputs
    env/Scripts/python.exe tests/trt_precision_fidelity.py evaluate                # three arms on those inputs + timing
    env/Scripts/python.exe tests/trt_precision_fidelity.py report                  # docs/perf/trt_precision_fidelity.md

ARMS (every one on the SAME stored inputs, batch 1, the processors' own pre/post-processing):
  ref    CUDA EP, TF32 off, the shipped ONNX (all four are FP32 files - asserted, so this IS an FP32 reference).
  mixed  the session the app's own loader builds under trt_precision=mixed (Mask_XSeg.Initialize / the FaceAnalysis bundle).
  trt32  TensorRT with fp16 off (precision_policy.providers_for(..., requested='fp32')): the engine a per-model FP32 choice builds.

INPUTS. The three buffalo models' inputs are CAPTURED from the live sessions while get_all_faces runs on real clips (a wrapper on
session.run records the exact blob production fed, so crop, kps and normalisation are production's, not re-derived). XSeg's
inputs are built the way procmgr_masking feeds it: the aligned arcface-256 crop of the final kps, and (second population) the
unwarped bbox crop the non-frontal path uses. Faces are spread evenly over d4, d1, d6, Love and s7.

METRICS.
  xseg    raw sigmoid output thresholded at 0.5: IoU; BOUNDARY error = mean / p95 / max distance (crop px) from each boundary pixel
          of one mask to the nearest boundary pixel of the other, both directions; and the soft keep-mask the compositor really uses
          (1 - clip(out) with out<0.1 zeroed): mean and max abs difference.
  w600k   cosine of the 512-d embeddings (the brief's gate: every face >= 0.999).
  lm106 / lm68  per-face mean landmark error in FRAME pixels (crop-space error / (192 / (1.5 x face size)) - a 400 px face turns
          0.1 crop px into 0.5 px; same definition as docs/perf/aux_batch_2026-10-05.md), and for lm68 the 5 kps that
          face_util._refine_kps_from_68 writes into face.kps (eye centres, nose tip, mouth corners): error in frame px and as a
          fraction of the inter-ocular distance.

GATES (taken from the earlier briefs, not invented here): xseg IoU >= 0.995 on >= 99% of faces and no face below 0.95
(stage_precision_verify_lighting_2026-10-05.md); embeddings >= 0.999 on every face; landmarks p95 <= 0.3 frame px
(aux_batch_2026-10-05.md).
"""
import argparse
import json
import os
import statistics as st
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures                                                   # noqa: E402  (module level: test_fixture_paths pins it)

OUT = os.path.join(APP, "output", "trt_precision_fidelity")
CLIPS = ("double/d4.mp4", "double/d1.mp4", "double/d6.mp4", "Love.mp4", "single/s7.mp4")
AUX = {"w600k_r50": "recognition", "2d106det": "landmark_2d_106", "1k3d68": "landmark_3d_68"}
FILES = {"xseg": "xseg.onnx", "w600k_r50": "buffalo_l/w600k_r50.onnx", "2d106det": "buffalo_l/2d106det.onnx",
         "1k3d68": "buffalo_l/1k3d68.onnx"}
KEYS = {"xseg": "masking:xseg", "w600k_r50": "recognition:buffalo_l", "2d106det": "recognition:buffalo_l",
        "1k3d68": "recognition:buffalo_l"}
RUNS, WARM = 200, 30


def _init_pipeline():
    from settings import Settings
    import angle_bench as ab
    cfg = Settings(os.path.join(APP, "config.yaml"))
    return ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "None", sync_config=True)


def model_path(stem):
    return os.path.join(APP, "models", FILES[stem])


# ── capture ───────────────────────────────────────────────────────────────────────────────────────
def cmd_capture(args):
    import cv2
    _init_pipeline()
    from roop import face_util
    from roop.face_util import align_crop, get_all_faces
    os.makedirs(OUT, exist_ok=True)
    face_util._ensure_face_analyser()
    pool = list(face_util.FACE_ANALYSER_POOL or [face_util.get_face_analyser()])
    tl = threading.local()
    tl.collect, tl.bbox = False, None
    rec = {s: [] for s in AUX}

    def wrap(fa):
        for stem, task in AUX.items():
            m = fa.models.get(task) or (getattr(fa, "lm68_model", None) if task == "landmark_3d_68" else None)
            if m is None:
                raise SystemExit("buffalo_l has no %s model loaded" % task)
            og, orun = m.get, m.session.run

            def get(img, face, _o=og):
                tl.bbox = np.asarray(face.bbox[:4], np.float64).copy()
                try:
                    return _o(img, face)
                finally:
                    tl.bbox = None

            def run(names, feed, opts=None, _o=orun, _s=stem):
                out = _o(names, feed, opts)
                if getattr(tl, "collect", False) and tl.bbox is not None:
                    x = np.asarray(next(iter(feed.values())))
                    rec[_s].append((x.copy(), tl.bbox.copy(), tl.clip))
                return out
            m.get, m.session.run = get, run

    for fa in pool:
        wrap(fa)
    per = int(np.ceil(args.faces / float(len(CLIPS))))
    xs_aligned, xs_box, xs_clip = [], [], []
    for ci, rel in enumerate(CLIPS):
        path = fixtures.clip(rel)
        if not os.path.exists(path):
            print("[capture] missing clip", rel, flush=True)
            continue
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tl.clip = ci
        n_xs = n_aux = 0
        start = {s: len(rec[s]) for s in AUX}
        for frac in np.linspace(0.03, 0.97, 600):
            if n_xs >= per and len(rec["1k3d68"]) - start["1k3d68"] >= per:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
            ok, frame = cap.read()
            if not ok:
                continue
            tl.collect = len(rec["1k3d68"]) - start["1k3d68"] < per
            faces = list(get_all_faces(frame) or [])
            tl.collect = False
            for f in faces:
                if n_xs >= per:
                    break
                kps = np.asarray(f.kps, np.float32)
                crop, _ = align_crop(frame, kps, 256, mode="arcface")
                x0, y0, x1, y1 = [float(v) for v in f.bbox[:4]]
                side = max(x1 - x0, y1 - y0) * 1.25
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                bx0, by0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
                bx1, by1 = int(min(frame.shape[1], cx + side / 2)), int(min(frame.shape[0], cy + side / 2))
                if bx1 - bx0 < 32 or by1 - by0 < 32:
                    continue
                xs_aligned.append(np.ascontiguousarray(crop))
                xs_box.append(np.ascontiguousarray(frame[by0:by1, bx0:bx1]))
                xs_clip.append(ci)
                n_xs += 1
        cap.release()
        for s in AUX:                          # trim this clip to its quota
            mine = [i for i in range(start[s], len(rec[s]))]
            for i in reversed(mine[per:]):
                rec[s].pop(i)
        print("[capture] %-16s xseg %3d  aux %s" % (rel, n_xs, {s: sum(1 for r in rec[s] if r[2] == ci) for s in AUX}), flush=True)
    data = {"xseg_aligned": np.stack(xs_aligned), "xseg_clip": np.asarray(xs_clip)}
    # the unwarped-box crops differ in size: store as an object array
    box = np.empty(len(xs_box), dtype=object)
    for i, b in enumerate(xs_box):
        box[i] = b
    data["xseg_box"] = box
    for s in AUX:
        data[s + "_x"] = np.concatenate([r[0] for r in rec[s]], axis=0)
        data[s + "_bbox"] = np.stack([r[1] for r in rec[s]])
        data[s + "_clip"] = np.asarray([r[2] for r in rec[s]])
    np.savez_compressed(os.path.join(OUT, "capture.npz"), **data)
    print("[capture] saved", {k: (v.shape if hasattr(v, "shape") else len(v)) for k, v in data.items()}, flush=True)
    return 0


# ── sessions ──────────────────────────────────────────────────────────────────────────────────────
def assert_fp32_file(path):
    import onnx
    from onnx import TensorProto as T
    m = onnx.load(path)
    bad = [i.name for i in m.graph.initializer if i.data_type == T.FLOAT16]
    if bad:
        raise SystemExit("%s has %d float16 initializers: a CUDA session of it is not an FP32 reference" % (path, len(bad)))


def ref_session(stem):
    import onnxruntime
    from roop.utilities import get_onnx_session_options
    assert_fp32_file(model_path(stem))
    return onnxruntime.InferenceSession(
        model_path(stem), get_onnx_session_options(),
        providers=[("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"])


def trt32_session(stem):
    import onnxruntime
    import roop.globals as g
    from roop import precision_policy as pp
    from roop.utilities import get_onnx_session_options
    chain, dec = pp.providers_for(KEYS[stem], g.execution_providers, model_path(stem), requested="fp32")
    sess = onnxruntime.InferenceSession(model_path(stem), get_onnx_session_options(), providers=chain)
    if sess.get_providers()[0] != "TensorrtExecutionProvider":
        raise SystemExit("%s trt32 session is on %s" % (stem, sess.get_providers()[0]))
    live = (sess.get_provider_options() or {}).get("TensorrtExecutionProvider") or {}
    if str(live.get("trt_fp16_enable")).lower() not in ("0", "false"):
        raise SystemExit("%s trt32 session has trt_fp16_enable=%s" % (stem, live.get("trt_fp16_enable")))
    return sess


def mixed_sessions():
    """The live sessions of the app's own loaders: {'xseg': (Mask_XSeg, session), stem: session}."""
    from roop import face_util
    from roop.processors.Mask_XSeg import Mask_XSeg
    xs = Mask_XSeg()
    xs.Initialize({"devicename": "cuda"})
    face_util._ensure_face_analyser()
    fa = face_util.get_face_analyser()
    out = {"xseg": xs.model_xseg}
    for stem, task in AUX.items():
        m = fa.models.get(task) or getattr(fa, "lm68_model", None)
        out[stem] = m.session
    for stem, s in out.items():
        live = (s.get_provider_options() or {}).get("TensorrtExecutionProvider") or {}
        if s.get_providers()[0] != "TensorrtExecutionProvider" or str(live.get("trt_fp16_enable")).lower() not in ("1", "true"):
            raise SystemExit("%s mixed session is not TensorRT fp16: %s %s" % (stem, s.get_providers(), live.get("trt_fp16_enable")))
    return out, xs


def run_all(sess, xs):
    name = sess.get_inputs()[0].name
    return np.concatenate([np.asarray(sess.run(None, {name: np.ascontiguousarray(xs[i:i + 1])})[0], np.float32)
                           for i in range(len(xs))], axis=0)


def xseg_inputs(data):
    """NHWC float32 [0,1], exactly Mask_XSeg.Run's preprocessing, for the aligned and the unwarped-box populations."""
    import cv2

    def prep(img):
        t = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        return (t.astype("float32") / 255.0)[None, ...]
    al = np.concatenate([prep(i) for i in data["xseg_aligned"]], axis=0)
    bx = np.concatenate([prep(i) for i in data["xseg_box"]], axis=0)
    return al, bx


# ── metrics ───────────────────────────────────────────────────────────────────────────────────────
def mask_stats(a_out, b_out):
    """a = candidate, b = reference; raw sigmoid arrays (N,256,256,1)."""
    import cv2
    ious, bmean, b95, bmax, soft_mean, soft_max, empty = [], [], [], [], [], [], 0
    k = np.ones((3, 3), np.uint8)
    for a, b in zip(a_out, b_out):
        ma, mb = (a[..., 0] > 0.5), (b[..., 0] > 0.5)
        u = np.logical_or(ma, mb).sum()
        ious.append(1.0 if u == 0 else float(np.logical_and(ma, mb).sum() / u))
        d = []
        for x, y in ((ma, mb), (mb, ma)):
            bx = x & ~cv2.erode(x.astype(np.uint8), k).astype(bool)
            by = y & ~cv2.erode(y.astype(np.uint8), k).astype(bool)
            if bx.any() and by.any():
                dt = cv2.distanceTransform((~by).astype(np.uint8), cv2.DIST_L2, 5)
                d.append(dt[bx])
        if d:
            d = np.concatenate(d)
            bmean.append(float(d.mean()))
            b95.append(float(np.percentile(d, 95)))
            bmax.append(float(d.max()))
        elif ma.any() != mb.any():
            bmean.append(float("nan")); b95.append(float("nan")); bmax.append(float("nan")); empty += 1
        else:
            bmean.append(0.0); b95.append(0.0); bmax.append(0.0)

        def keep(o):
            r = np.clip(o[..., 0], 0, 1.0)
            r = np.where(r < 0.1, 0, r)
            return 1.0 - r
        diff = np.abs(keep(a) - keep(b))
        soft_mean.append(float(diff.mean()))
        soft_max.append(float(diff.max()))
    return {"iou": ious, "boundary_mean_px": bmean, "boundary_p95_px": b95, "boundary_max_px": bmax,
            "soft_mean_abs": soft_mean, "soft_max_abs": soft_max, "one_side_empty": empty}


def cos_stats(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return [float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)) for x, y in zip(a, b)]


def _pts(stem, v):
    v = np.asarray(v, np.float64).reshape(-1)
    if stem == "2d106det":
        return (v.reshape(-1, 2) + 1.0) * 96.0
    return (v.reshape(-1, 3)[-68:, :2] + 1.0) * 96.0


def _kps5(p):
    return np.array([p[36:42].mean(axis=0), p[42:48].mean(axis=0), p[30], p[48], p[54]])


def lm_stats(stem, cand, ref, bbox):
    """Frame-pixel landmark error per face (+ refined 5 kps for the 68 model)."""
    mean_e, max_e, kps_e, kps_rel = [], [], [], []
    for c, r, bb in zip(cand, ref, bbox):
        size = max(bb[2] - bb[0], bb[3] - bb[1])
        f = (size * 1.5) / 192.0                         # frame px per crop px
        pc, pr = _pts(stem, c), _pts(stem, r)
        e = np.linalg.norm(pc - pr, axis=1) * f
        mean_e.append(float(e.mean()))
        max_e.append(float(e.max()))
        if stem == "1k3d68":
            kc, kr = _kps5(pc), _kps5(pr)
            ke = np.linalg.norm(kc - kr, axis=1) * f
            kps_e.append(float(ke.mean()))
            iod = float(np.linalg.norm(kr[0] - kr[1]) * f)
            kps_rel.append(float(ke.mean() / max(iod, 1e-6)))
    out = {"mean_px": mean_e, "max_px": max_e}
    if kps_e:
        out.update({"kps5_mean_px": kps_e, "kps5_rel_iod": kps_rel})
    return out


def summ(v):
    v = np.asarray([x for x in v if x == x], np.float64)
    if not len(v):
        return {}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)), "p5": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)), "min": float(v.min()), "max": float(v.max())}


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
    return {"median_ms": round(st.median(ts), 4), "p10_ms": round(ts[len(ts) // 10], 4), "p90_ms": round(ts[(len(ts) * 9) // 10], 4)}


# ── evaluate ──────────────────────────────────────────────────────────────────────────────────────
def cmd_evaluate(args):
    _init_pipeline()
    data = np.load(os.path.join(OUT, "capture.npz"), allow_pickle=True)
    mixed, _xs = mixed_sessions()
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "n": {}, "models": {}}
    al, bx = xseg_inputs(data)
    inputs = {"xseg_aligned": al, "xseg_box": bx}
    for s in AUX:
        inputs[s] = data[s + "_x"]
    res["n"] = {k: int(len(v)) for k, v in inputs.items()}
    outs = {}
    for stem in FILES:
        ref, t32 = ref_session(stem), trt32_session(stem)
        keys = ("xseg_aligned", "xseg_box") if stem == "xseg" else (stem,)
        for key in keys:
            x = inputs[key]
            r, m, t = run_all(ref, x), run_all(mixed[stem], x), run_all(t32, x)
            outs[key] = (r, m, t)
            print("[eval] %-13s n=%d ran ref / mixed / trt32" % (key, len(x)), flush=True)
        probe = inputs[keys[0]][0:1]
        res["models"][stem] = {"timing": {"cuda_fp32": time_gpu_io(ref, probe), "trt_mixed": time_gpu_io(mixed[stem], probe),
                                          "trt_fp32": time_gpu_io(t32, probe)}}
        # determinism of the mixed arm: the same inputs twice must be identical
        again = run_all(mixed[stem], inputs[keys[0]][:20])
        res["models"][stem]["mixed_repeat_max_abs"] = float(np.abs(again - outs[keys[0]][1][:20]).max())
        del ref, t32
    for stem in FILES:
        rec = res["models"][stem]
        for arm, ai in (("mixed", 1), ("trt32", 2)):
            if stem == "xseg":
                for key in ("xseg_aligned", "xseg_box"):
                    r, m, t = outs[key]
                    ms = mask_stats((m, t)[ai - 1], r)
                    rec.setdefault(key, {})[arm] = {k: (summ(v) if isinstance(v, list) else v) for k, v in ms.items()}
                    rec[key][arm]["_raw"] = {k: v for k, v in ms.items() if isinstance(v, list) and k in ("iou",)}
            elif stem == "w600k_r50":
                r, m, t = outs[stem]
                c = cos_stats((m, t)[ai - 1], r)
                rec.setdefault("cosine", {})[arm] = summ(c)
                rec["cosine"][arm]["below_0.999"] = int(sum(1 for v in c if v < 0.999))
                rec["cosine"][arm]["_raw"] = c
            else:
                r, m, t = outs[stem]
                ls = lm_stats(stem, (m, t)[ai - 1], r, data[stem + "_bbox"])
                rec.setdefault("landmarks", {})[arm] = {k: summ(v) for k, v in ls.items()}
                rec["landmarks"][arm]["over_0.3px_faces"] = int(sum(1 for v in ls["mean_px"] if v > 0.3))
                rec["landmarks"][arm]["_raw"] = {k: v for k, v in ls.items()}
        if stem in ("2d106det", "1k3d68"):
            r, m, t = outs[stem]
            rec["trt32_vs_cuda_fp32_floor"] = {k: summ(v) for k, v in lm_stats(stem, t, r, data[stem + "_bbox"]).items()}
        if stem == "xseg":
            sizes = np.asarray([(max(b[2] - b[0], b[3] - b[1])) for b in data["2d106det_bbox"]])
        res["models"][stem]["clip_of_sample"] = (data[stem + "_clip"].tolist() if stem in AUX else data["xseg_clip"].tolist())
    for stem in AUX:
        bb = data[stem + "_bbox"]
        res["models"][stem]["face_size_px"] = summ([max(b[2] - b[0], b[3] - b[1]) for b in bb])
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print("[eval] wrote", os.path.join(OUT, "results.json"), flush=True)
    return 0


def cmd_tf32(args):
    """trt32 with NVIDIA_TF32_OVERRIDE=0 (set by the caller before launch) in its OWN engine cache: is the TRT-FP32 vs CUDA-FP32
    gap TF32? Writes tf32_results.json next to results.json."""
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise SystemExit("run with NVIDIA_TF32_OVERRIDE=0 in the environment (it must be set before CUDA/TensorRT load)")
    import onnxruntime
    import roop.globals as g
    from roop import precision_policy as pp
    from roop.utilities import get_onnx_session_options
    _init_pipeline()
    data = np.load(os.path.join(OUT, "capture.npz"), allow_pickle=True)
    al, bx = xseg_inputs(data)
    inputs = {"xseg": al}
    for st_ in AUX:
        inputs[st_] = data[st_ + "_x"]
    res = {}
    for stem in FILES:
        chain, _ = pp.providers_for(KEYS[stem], g.execution_providers, model_path(stem), requested="fp32")
        new = []
        for p_ in chain:
            if isinstance(p_, tuple) and "tensorrt" in p_[0].lower():
                o = dict(p_[1])
                d = o["trt_engine_cache_path"] + "_tf32off"
                os.makedirs(d, exist_ok=True)
                o["trt_engine_cache_path"] = d
                o["trt_timing_cache_path"] = d
                new.append((p_[0], o))
            else:
                new.append(p_)
        sess = onnxruntime.InferenceSession(model_path(stem), get_onnx_session_options(), providers=new)
        ref = ref_session(stem)
        x = inputs[stem]
        r, t = run_all(ref, x), run_all(sess, x)
        if stem == "xseg":
            ms = mask_stats(t, r)
            res[stem] = {"iou": summ(ms["iou"]), "boundary_mean_px": summ(ms["boundary_mean_px"])}
        elif stem == "w600k_r50":
            res[stem] = {"cosine": summ(cos_stats(t, r))}
        else:
            ls = lm_stats(stem, t, r, data[stem + "_bbox"])
            res[stem] = {k: summ(v) for k, v in ls.items()}
        res[stem]["timing_ms"] = time_gpu_io(sess, x[0:1])
        print("[tf32off] %-10s %s" % (stem, json.dumps(res[stem])[:400]), flush=True)
    json.dump(res, open(os.path.join(OUT, "tf32_results.json"), "w"), indent=1)
    return 0


# ── report ────────────────────────────────────────────────────────────────────────────────────────
def cmd_report(args):
    res = json.load(open(os.path.join(OUT, "results.json")))
    M = res["models"]
    L = ["# TensorRT mixed vs FP32 reference on real faces: XSeg, w600k_r50, 2d106det, 1k3d68", "",
         "Generated by `app/tests/trt_precision_fidelity.py report` from `app/output/trt_precision_fidelity/results.json` (%s). "
         "Method, arms and gates: the module docstring. Samples: %s." % (res["date"], res["n"]), ""]

    def t(stem):
        x = M[stem]["timing"]
        return "%.3f / %.3f / %.3f" % (x["cuda_fp32"]["median_ms"], x["trt_mixed"]["median_ms"], x["trt_fp32"]["median_ms"])
    L += ["## Timing (GPU-resident IO, batch 1, median of %d; CUDA FP32 / TRT mixed / TRT FP32, ms)" % RUNS, "",
          "| model | ms |", "|---|---|"] + ["| %s | %s |" % (s, t(s)) for s in FILES] + [""]
    L += ["## XSeg", "", "| population | arm | IoU mean / p1(p5) / min | faces IoU<0.995 | faces IoU<0.95 | boundary mean / p95 / max px (per-face mean, p95, max) | soft keep-mask mean / max abs |",
          "|---|---|---|---|---|---|---|"]
    for key in ("xseg_aligned", "xseg_box"):
        for arm in ("mixed", "trt32"):
            d = M["xseg"][key][arm]
            iou = d["_raw"]["iou"]
            L.append("| %s | %s | %.4f / %.4f / %.4f | %d | %d | %.3f (p95 %.3f, max %.3f) / %.3f / %.2f | %.5f / %.4f |" % (
                key, arm, d["iou"]["mean"], d["iou"]["p5"], d["iou"]["min"], sum(1 for v in iou if v < 0.995),
                sum(1 for v in iou if v < 0.95), d["boundary_mean_px"]["mean"], d["boundary_mean_px"]["p95"],
                d["boundary_mean_px"]["max"], d["boundary_p95_px"]["p95"], d["boundary_max_px"]["max"],
                d["soft_mean_abs"]["mean"], d["soft_max_abs"]["max"]))
    c = M["w600k_r50"]["cosine"]
    L += ["", "## w600k_r50 embedding cosine vs FP32", "", "| arm | mean | p1-ish (p5) | min | faces < 0.999 |", "|---|---|---|---|---|"]
    for arm in ("mixed", "trt32"):
        L.append("| %s | %.6f | %.6f | %.6f | %d |" % (arm, c[arm]["mean"], c[arm]["p5"], c[arm]["min"], c[arm]["below_0.999"]))
    for stem in ("2d106det", "1k3d68"):
        lm = M[stem]["landmarks"]
        L += ["", "## %s landmark error vs FP32 (frame px; per-face mean over the points)" % stem, "",
              "| arm | mean | median | p95 | p99 | max | faces > 0.3 px | per-face worst point (max of max) |", "|---|---|---|---|---|---|---|---|"]
        for arm in ("mixed", "trt32"):
            d = lm[arm]
            L.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %d | %.2f |" % (
                arm, d["mean_px"]["mean"], d["mean_px"]["median"], d["mean_px"]["p95"], d["mean_px"]["p99"], d["mean_px"]["max"],
                d["over_0.3px_faces"], d["max_px"]["max"]))
        if stem == "1k3d68":
            L += ["", "Refined kps (what `_refine_kps_from_68` writes into `face.kps`):", "",
                  "| arm | mean px | median | p95 | max | mean as % of inter-ocular | p95 % IOD |", "|---|---|---|---|---|---|---|"]
            for arm in ("mixed", "trt32"):
                d, r = lm[arm]["kps5_mean_px"], lm[arm]["kps5_rel_iod"]
                L.append("| %s | %.3f | %.3f | %.3f | %.3f | %.2f%% | %.2f%% |" % (
                    arm, d["mean"], d["median"], d["p95"], d["max"], 100 * r["mean"], 100 * r["p95"]))
        fl = M[stem]["trt32_vs_cuda_fp32_floor"]["mean_px"]
        L += ["", "TRT-FP32 vs CUDA-FP32 floor (engine-to-engine, same precision): mean %.4f px, max %.4f px." % (fl["mean"], fl["max"])]
    L += ["", "Mixed arm repeat (same inputs twice, max abs output difference): " + ", ".join(
        "%s %.3g" % (s, M[s]["mixed_repeat_max_abs"]) for s in FILES), ""]
    path = os.path.join(REPO, "docs", "perf", "trt_precision_fidelity.md")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    slim = json.loads(json.dumps(res))
    json.dump(slim, open(os.path.join(REPO, "docs", "perf", "trt_precision_fidelity.json"), "w"))
    print("\n".join(L))
    print("wrote", path)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("capture", "evaluate", "report", "tf32"))
    ap.add_argument("--faces", type=int, default=500)
    args = ap.parse_args()
    return {"capture": cmd_capture, "evaluate": cmd_evaluate, "report": cmd_report, "tf32": cmd_tf32}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
