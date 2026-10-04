"""Per-face aux models vs the batched GPU-crop engine (roop/aux_batch.py), on real footage.

    env/Scripts/python.exe tests/ab_aux_batch.py --phase equiv   [--faces 500]
    env/Scripts/python.exe tests/ab_aux_batch.py --phase prepass [--clips d4,d1,Love,d6] [--frames 600]

PHASE equiv
  (a) MODEL LEVEL. N real faces (the live detector, aux=False, so both arms get the SAME
      bbox/kps), then recognition / landmark_2d_106 / landmark_3d_68 two ways: the original
      per-face `model.get` (cv2.warpAffine crop, batch-1 ORT call) and the batched engine,
      fed a frame at a time AND coalesced across threads. Acceptance: embedding cosine
      >= 0.999 and landmarks within 0.3 px (106 and 68 points, in frame pixels) on every
      face. A NULL CONTROL (per-face path against itself, a second call on another pooled
      analyser) comes first, so the comparison is read against the noise floor.
  (b) FUNCTION LEVEL. `_detect_faces(frame, expected_count=2)` with and without
      `aux_batch_scope`, to see the whole detect+rescue+enrich chain agree.

PHASE prepass
  `_precompute_tracks` over a clip window with ROOP_AUX_BATCH=0 / 1, counterbalanced
  (A B B A). Reports pre-pass fps, `track_detect` ms/call (the stage the aux models run
  inside), whether the tracks and the per-frame source assignments are IDENTICAL, and the
  engine's own batch statistics (so "it ran" is read off counters, not inferred).
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("ROOP_PROFILE", "1")

import fixtures                      # noqa: E402

CLIPS = {"d4": ("double/d4.mp4", 0), "d1": ("double/d1.mp4", 0), "Love": ("Love.mp4", 1100),
         "d6": ("double/d6.mp4", 0)}


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def init(args):
    from settings import Settings
    import angle_bench as ab
    cfg = Settings(os.path.join(APP, "config.yaml"))
    # The pool widths the app reads from config.yaml at startup.
    try:
        from settings import apply_env
        apply_env(cfg, os.environ)
    except Exception as exc:                                  # pragma: no cover - harness only
        print("[ab] apply_env failed (%s): pool widths may differ from the app" % exc)
    g = ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "None", sync_config=True)
    g.g_desired_face_analysis = ["landmark_3d_68", "landmark_2d_106", "detection", "recognition"]
    g.lm68_lazy = False
    g.detector_engine = str(cfg.detector_engine)
    g.processing = True
    return g, cfg


def read_frames(name, n, stride=1):
    rel, lo = CLIPS[name]
    cap = cv2.VideoCapture(fixtures.clip(rel, required=True))
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    out = []
    for i in range(n * stride):
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            out.append((lo + i, fr))
    cap.release()
    return out


def clone_face(f):
    from insightface.app.common import Face
    return Face(bbox=np.array(f.bbox, copy=True), kps=np.array(f.kps, copy=True), det_score=float(f.det_score))


def per_face_aux(fa, frame, face):
    for task, model in fa.models.items():
        if task != "detection":
            model.get(frame, face)


def compare(a, b):
    return {"emb_cos": cos(a.embedding, b.embedding),
            "lm106": float(np.max(np.abs(a.landmark_2d_106 - b.landmark_2d_106))),
            "lm68": float(np.max(np.abs(a.landmark_3d_68[:, :2] - b.landmark_3d_68[:, :2]))),
            "lm68_z": float(np.max(np.abs(a.landmark_3d_68[:, 2] - b.landmark_3d_68[:, 2]))),
            "pose": float(np.max(np.abs(np.asarray(a.pose) - np.asarray(b.pose))))}


def summarize(rows, label):
    def mx(k):
        return max(r[k] for r in rows)

    def p99(k):
        return float(np.percentile([r[k] for r in rows], 99))
    return {"label": label, "faces": len(rows), "min_emb_cos": min(r["emb_cos"] for r in rows),
            "faces_below_0.999": sum(1 for r in rows if r["emb_cos"] < 0.999),
            "max_lm106_px": mx("lm106"), "p99_lm106_px": p99("lm106"),
            "max_lm68_px": mx("lm68"), "p99_lm68_px": p99("lm68"), "max_lm68_z": mx("lm68_z"),
            "max_pose_rad": mx("pose"),
            "faces_lm106_over_0.3px": sum(1 for r in rows if r["lm106"] > 0.3),
            "faces_lm68_over_0.3px": sum(1 for r in rows if r["lm68"] > 0.3)}


def fp32_truth(models, fa_models_unused=None):
    """Per-face aux models on an FP32 CUDA session: the reference both TensorRT paths are
    measured against (the live per-face path is TensorRT FP16 and is NOT a ground truth)."""
    import onnxruntime as ort
    out = {}
    for task, model in models.items():
        # A FRESH object of the same class: with ROOP_PROFILE=1 the live model carries an
        # instance-level `get` wrapper (baseline_probe) bound to the live model, and a
        # copy.copy of it would silently run the live session again -- which is how this
        # reference first came out identical to the thing it was meant to check.
        sess = ort.InferenceSession(model.model_file, providers=["CUDAExecutionProvider"])
        out[task] = type(model)(model_file=model.model_file, session=sess)
        assert out[task].session is sess and out[task].get.__self__ is out[task]
    return out


def run_truth(truth, frame, face):
    for task in ("recognition", "landmark_2d_106", "landmark_3d_68"):
        truth[task].get(frame, face)


def face_size(f):
    return float(max(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]))


def table(label, sets_a, sets_b, sizes):
    rows = [compare(x, y) for a, b in zip(sets_a, sets_b) for x, y in zip(a, b)]
    out = summarize(rows, label)
    sz = np.array(sizes)
    for name, lo, hi in (("<250px", 0, 250), ("250-400px", 250, 400), (">400px", 400, 1e9)):
        sel = [r for r, z in zip(rows, sz) if lo <= z < hi]
        if sel:
            out["by_size_" + name] = {"faces": len(sel),
                                      "max_lm106_px": max(r["lm106"] for r in sel),
                                      "max_lm68_px": max(r["lm68"] for r in sel),
                                      "min_emb_cos": min(r["emb_cos"] for r in sel)}
    return out


def phase_equiv(args):
    g, cfg = init(args)
    from roop import face_util as fu
    from roop import aux_batch
    # --- collect faces -------------------------------------------------------
    jobs = []
    n_faces = 0
    for name in args.clips.split(","):
        for idx, fr in read_frames(name, args.frames, args.stride):
            faces = fu._detect_faces_raw(fr, aux=False, unclamped=True) or []
            if faces:
                jobs.append((name, idx, fr, faces))
                n_faces += len(faces)
            if n_faces >= args.faces:
                break
        if n_faces >= args.faces:
            break
    sizes = [face_size(f) for j in jobs for f in j[3]]
    print("[equiv] %d faces / %d frames; face size px median %.0f p90 %.0f max %.0f"
          % (n_faces, len(jobs), np.median(sizes), np.percentile(sizes, 90), max(sizes)), flush=True)

    with fu.lease_face_analyser() as fa:
        models = dict(fa.models)
    truth = fp32_truth({t: models[t] for t in ("recognition", "landmark_2d_106", "landmark_3d_68")})

    def per_face(runner):
        out = []
        for _l, _i, fr, faces in jobs:
            c = [clone_face(f) for f in faces]
            for f in c:
                runner(fr, f)
            out.append(c)
        return out

    def live(fr, f):
        with fu.lease_face_analyser() as fa:
            per_face_aux(fa, fr, f)
    ref = per_face(live)
    ref2 = per_face(live)                                  # null: the same path again
    tru = per_face(lambda fr, f: run_truth(truth, fr, f))
    sets = {}
    stats = {}
    for label, fp16 in (("fp16", True), ("fp32", False)):
        t0 = time.time()
        eng = aux_batch.AuxBatchEngine(models, device_id=g.cuda_device_id, fp16=fp16)
        print("[equiv] engine %s built in %.1f s" % (label, time.time() - t0), flush=True)
        got = []
        for _l, _i, fr, faces in jobs:
            c = [clone_face(f) for f in faces]
            eng.run(fr, c)
            got.append(c)
        sets[label] = got
        # coalesced across 4 threads must equal the one-call-per-frame result exactly
        eng.reset_stats()
        got2 = [None] * len(jobs)
        lock, nxt = threading.Lock(), [0]

        def worker():
            while True:
                with lock:
                    i = nxt[0]
                    nxt[0] += 1
                if i >= len(jobs):
                    return
                c = [clone_face(f) for f in jobs[i][3]]
                eng.run(jobs[i][2], c)
                got2[i] = c
        ts = [threading.Thread(target=worker) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        stats[label] = eng.summary()
        same = all(np.array_equal(x.embedding, y.embedding) and np.array_equal(x.landmark_2d_106, y.landmark_2d_106)
                   and np.array_equal(x.landmark_3d_68, y.landmark_3d_68)
                   for a, b in zip(got, got2) for x, y in zip(a, b))
        stats[label + "_coalesced_equals_single"] = bool(same)
        eng.close()
    res = {"faces": n_faces, "face_size_px": [float(np.median(sizes)), float(np.percentile(sizes, 90)), max(sizes)],
           "NULL live vs live": table("null", ref, ref2, sizes),
           "live(TRT-FP16, per face) vs FP32 truth": table("live vs truth", tru, ref, sizes),
           "batched FP16 vs FP32 truth": table("b16 vs truth", tru, sets["fp16"], sizes),
           "batched FP32 vs FP32 truth": table("b32 vs truth", tru, sets["fp32"], sizes),
           "batched FP16 vs live": table("b16 vs live", ref, sets["fp16"], sizes),
           "batched FP32 vs live": table("b32 vs live", ref, sets["fp32"], sizes),
           "engine_stats": stats}
    print(json.dumps(res, indent=1))

    # --- function level -------------------------------------------------------
    fn_rows, n_diff, nf = [], 0, 0
    with fu.aux_batch_scope(True):
        batched = []
        for _lbl, _idx, fr, _faces in jobs[: args.fn_frames]:
            batched.append(fu._detect_faces(fr, expected_count=2))
    plain = [fu._detect_faces(fr, expected_count=2) for _l, _i, fr, _f in jobs[: args.fn_frames]]
    for a, b in zip(plain, batched):
        nf += 1
        if len(a) != len(b):
            n_diff += 1
            continue
        for x, y in zip(a, b):
            fn_rows.append({**compare(x, y), "bbox": float(np.max(np.abs(x.bbox - y.bbox))),
                            "kps": float(np.max(np.abs(x.kps - y.kps)))})
    out = {"frames": nf, "frames_with_different_face_count": n_diff,
           "function_level": summarize(fn_rows, "_detect_faces off vs on") if fn_rows else None,
           "max_kps_px": max((r["kps"] for r in fn_rows), default=None),
           "max_bbox_px": max((r["bbox"] for r in fn_rows), default=None)}
    print(json.dumps(out, indent=1))
    path = os.path.join(REPO, "docs", "perf", "aux_batch_equiv_%s.json" % args.date)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({**res, "function": out}, fh, indent=1)
    print("wrote", path)


# ── pre-pass ─────────────────────────────────────────────────────────────────────

def make_mgr(g, targets):
    from roop.procmgr_tracking import TrackingMixin

    class Cap:
        def __init__(self, e):
            self.embedding = e

    class Opt:
        face_distance_threshold = 0.75
        selected_index = 0

    class Mgr(TrackingMixin):
        def __init__(self):
            self.target_face_datas = [Cap(e) for e in targets]
            self.target_face_groups = [0] * len(targets)
            self.options = Opt()
            self.progress_gradio = None
            self._track_assignments = {}
            self._track_scanned = 0

        def _publish_live(self, frame):
            pass

    return Mgr()


def pick_targets(fu, clip, lo, n_people=2):
    """The first distinct embeddings the live pipeline sees: arm-independent targets."""
    embs = []
    for _idx, fr in read_frames(clip, 120, 3):
        for f in fu.get_all_faces(fr) or []:
            e = np.asarray(f.embedding, np.float32)
            e = e / np.linalg.norm(e)
            if all(1 - float(e @ o) > 0.45 for o in embs):
                embs.append(e)
        if len(embs) >= n_people:
            break
    return embs[:max(1, n_people)]


def digest(tracks, mgr):
    """What 'identical tracks and source assignments' is compared on."""
    t = []
    for tr in tracks or []:
        obs = tr.get("obs") or {}
        t.append({"n_obs": len(obs), "frames": (min(obs), max(obs)) if obs else None,
                  "last_seen": int(tr.get("last_seen", -1)),
                  "src": tr.get("src", tr.get("source")),
                  "emb_mean": np.asarray(tr.get("emb_mean"), np.float32) if tr.get("emb_mean") is not None else None,
                  "bbox": [np.asarray(o.bbox, np.float32) for _k, o in sorted(obs.items())]})
    asg = {int(k): [(np.asarray(c, np.float32), (-1 if s is None else int(s))) for c, s, *_ in v]
           for k, v in sorted(mgr._track_assignments.items())}
    return t, asg


def diff_digest(a, b):
    (ta, aa), (tb, ab_) = a, b
    out = {"tracks": (len(ta), len(tb)), "tracks_identical_shape": len(ta) == len(tb)}
    if len(ta) != len(tb):
        return out
    out["frames_and_obs_equal"] = all(x["n_obs"] == y["n_obs"] and x["frames"] == y["frames"] and
                                      x["last_seen"] == y["last_seen"] for x, y in zip(ta, tb))
    out["max_bbox_px"] = max((float(np.max(np.abs(p - q))) for x, y in zip(ta, tb)
                              for p, q in zip(x["bbox"], y["bbox"])), default=0.0)
    out["min_emb_mean_cos"] = min((cos(x["emb_mean"], y["emb_mean"]) for x, y in zip(ta, tb)
                                   if x["emb_mean"] is not None and y["emb_mean"] is not None), default=1.0)
    keys_a, keys_b = set(aa), set(ab_)
    out["assignment_frames_equal"] = keys_a == keys_b
    sa = sb = 0
    for k in keys_a & keys_b:
        sa += 1
        if [s for _c, s in aa[k]] != [s for _c, s in ab_[k]]:
            sb += 1
    out["assignment_frames"] = sa
    out["frames_with_different_source_assignment"] = sb
    out["assignments_identical"] = out["assignment_frames_equal"] and sb == 0
    return out


def phase_prepass(args):
    g, cfg = init(args)
    from roop import face_util as fu
    from roop import procmgr_runtime as rt
    results = []
    for name in args.clips.split(","):
        rel, lo = CLIPS[name]
        path = fixtures.clip(rel, required=True)
        targets = pick_targets(fu, name, lo)
        print("[prepass] %s: %d target embeddings" % (name, len(targets)), flush=True)
        n = args.frames
        arms = args.arms.split(",")
        recs = []
        base_digest = None
        for i, arm in enumerate(arms):
            # A = per-face aux models (today), B = batched FP16, C = batched FP32
            os.environ["ROOP_AUX_BATCH"] = "0" if arm == "A" else "1"
            os.environ["ROOP_AUX_BATCH_FP16"] = "0" if arm in ("C", "E") else "1"
            os.environ["ROOP_AUX_BATCH_CROPS"] = "cpu" if arm in ("D", "E") else "gpu"
            rt._prof_reset()
            mgr = make_mgr(g, targets)
            g.processing = True
            t0 = time.perf_counter()
            tracks = mgr._precompute_tracks(path, lo, lo + n, n, step=1, collect_obs=True,
                                            desc="ab %s %s" % (name, arm))
            wall = time.perf_counter() - t0
            td, tc = rt._prof_times.get("track_detect", 0.0), rt._prof_counts.get("track_detect", 0)
            dg = digest(tracks, mgr)
            if base_digest is None:
                base_digest = dg
            rec = {"clip": name, "arm": arm, "pos": i, "wall_s": round(wall, 2),
                   "fps": round(n / wall, 2), "track_detect_ms": round(1000 * td / max(1, tc), 2),
                   "track_detect_calls": tc, "tracks": len(dg[0]),
                   "aux_batch": getattr(mgr, "_aux_batch_summary", None),
                   "vs_first_arm": diff_digest(base_digest, dg) if i else None}
            print(json.dumps(rec), flush=True)
            recs.append(rec)
        results.append(recs)
    os.environ.pop("ROOP_AUX_BATCH", None)
    os.environ.pop("ROOP_AUX_BATCH_FP16", None)
    os.environ.pop("ROOP_AUX_BATCH_CROPS", None)
    path = os.path.join(REPO, "docs", "perf", "aux_batch_prepass_%s.json" % args.date)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", path)
    for recs in results:
        by = {}
        for r in recs:
            by.setdefault(r["arm"], []).append(r)
        line = {"clip": recs[0]["clip"]}
        for arm, rs in by.items():
            line[arm] = {"fps": round(statistics.mean(r["fps"] for r in rs), 2),
                         "track_detect_ms": round(statistics.mean(r["track_detect_ms"] for r in rs), 2)}
        print(json.dumps(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("equiv", "prepass"), required=True)
    ap.add_argument("--date", default="2026-10-05")
    ap.add_argument("--clips", default="d4,d1,Love")
    ap.add_argument("--faces", type=int, default=500)
    ap.add_argument("--frames", type=int, default=600, help="equiv: frames to read per clip; prepass: window")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fn-frames", type=int, default=150)
    ap.add_argument("--arms", default="A,B,B,A", help="A off, B batched FP16 GPU crops, C FP32 GPU crops, D FP16 CPU crops, E FP32 CPU crops")
    args = ap.parse_args()
    {"equiv": phase_equiv, "prepass": phase_prepass}[args.phase](args)


if __name__ == "__main__":
    main()
