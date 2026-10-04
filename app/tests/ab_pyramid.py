"""Old vs new MultiScaleFaceDetector.detect: same boxes, fewer detector inferences?

    env/Scripts/python.exe tests/ab_pyramid.py --base HEAD --phase equiv
    env/Scripts/python.exe tests/ab_pyramid.py --base HEAD --phase speed

What changed (roop/face_detector.py):
  1. when the adaptive single pass triggers the pyramid, its result is the pyramid's
     scale-1.0 level (same image, same det_size/det_thresh) and is reused, so only the
     OTHER scales run: 4 detector inferences -> 3 per triggered frame;
  2. the scale passes run in the caller's thread when it is one of the pool workers
     (it holds a pooled FaceAnalysis lease), otherwise on ONE persistent executor
     instead of a ThreadPoolExecutor built and torn down per call.
should_trigger_pyramid and its thresholds are untouched.

PHASE equiv  -- the old module is `git show <base>:app/roop/face_detector.py` executed as a
second module. Both get the SAME leased retinaface detect_fn (r50 here: the only
pyramid-capable detector model present). For every frame of
    * d6 (3840x2160, every frame),
    * Love 1100..1700 (the kiss: natural close-ups),
    * d4 0..600 every 2nd frame,
    * SYNTHETIC close-ups: a face from Love/d4 cropped so it fills ~85% of a 1280x720 frame
      (some cut by the frame edge, to exercise the padding),
the merged boxes/kps are compared in order (|diff| <= 0.5 px required), once from a plain
thread (shared executor) and, on the frames that trigger, once from a thread flagged as a
pool worker (sequential). Detector inferences are counted at `detect_fn`, the unit the
"detect calls per frame" figure is made of.

PHASE speed  -- throughput of the SAME frames old vs new, ABBA, from 1 plain thread (pool
idle) and from N flagged threads (what the render does: N workers hold the pool).
"""
import argparse
import concurrent.futures as cf
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import types

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

TOL_PX = 0.5


def load_base(ref):
    src = subprocess.check_output(["git", "show", "%s:app/roop/face_detector.py" % ref], cwd=REPO)
    mod = types.ModuleType("roop.face_detector_base")
    mod.__file__ = os.path.join(APP, "roop", "face_detector_base.py")
    sys.modules["roop.face_detector_base"] = mod
    exec(compile(src, "face_detector@%s" % ref, "exec"), mod.__dict__)
    return mod


class Counted:
    """detect_fn that counts detector inferences."""

    def __init__(self, model_type):
        from roop.retinaface import _detect_single_instance
        self._fn, self.model_type, self.n = _detect_single_instance, model_type, 0
        self._lock = threading.Lock()

    def __call__(self, f, ds, dt):
        with self._lock:
            self.n += 1
        return self._fn(f, det_size=ds, det_thresh=dt, model_type=self.model_type)


def diff(a, b):
    (ba, ka), (bb, kb) = a, b
    if len(ba) != len(bb):
        return {"n": (len(ba), len(bb)), "box": float("inf"), "kps": float("inf"), "score": float("inf")}
    if len(ba) == 0:
        return {"n": (0, 0), "box": 0.0, "kps": 0.0, "score": 0.0}
    return {"n": (len(ba), len(bb)),
            "box": float(np.max(np.abs(ba[:, :4] - bb[:, :4]))),
            "kps": float(np.max(np.abs(ka - kb))) if ka is not None and kb is not None else 0.0,
            "score": float(np.max(np.abs(ba[:, 4] - bb[:, 4])))}


def frames(spec):
    """(label, idx, frame)"""
    for name, rel, lo, hi, stride in spec:
        cap = cv2.VideoCapture(fixtures.clip(rel, required=True))
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        for idx in range(lo, hi):
            ok, fr = cap.read()
            if not ok:
                break
            if (idx - lo) % stride == 0:
                yield name, idx, fr
        cap.release()


def synthetic_closeups(new, fn, n_per_clip, seed=7):
    """Face-filling crops (and edge-cut ones) from frames where r50 finds a face."""
    rng = np.random.RandomState(seed)
    out = []
    for name, rel, lo, hi in (("Love", "Love.mp4", 1100, 1700), ("d4", "double/d4.mp4", 0, 600)):
        cap = cv2.VideoCapture(fixtures.clip(rel, required=True))
        got, tries = 0, 0
        while got < n_per_clip and tries < 400:
            tries += 1
            idx = int(rng.randint(lo, hi))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if not ok:
                continue
            boxes, _ = fn(fr, 640, 0.5)
            if len(boxes) == 0:
                continue
            x0, y0, x1, y1 = [float(v) for v in boxes[0][:4]]
            fh = max(8.0, y1 - y0)
            ch = fh / 0.85                                  # crop height so the face is ~85% of it
            cw = ch * 16.0 / 9.0
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            if got % 3 == 2:                                # every third one is cut by the frame edge
                cx += 0.45 * cw
            ax0, ay0 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
            H, W = fr.shape[:2]
            ax0 = max(0, min(ax0, W - 2)); ay0 = max(0, min(ay0, H - 2))
            ax1, ay1 = min(W, int(ax0 + cw)), min(H, int(ay0 + ch))
            crop = fr[ay0:ay1, ax0:ax1]
            if crop.shape[0] < 32 or crop.shape[1] < 32:
                continue
            out.append(("synthetic-%s" % name, idx, cv2.resize(crop, (1280, 720), interpolation=cv2.INTER_CUBIC)))
            got += 1
        cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HEAD")
    ap.add_argument("--phase", choices=("equiv", "speed"), default="equiv")
    ap.add_argument("--date", default="2026-10-04")
    ap.add_argument("--model-type", default="r50")
    ap.add_argument("--limit", type=int, default=0, help="debug: frames per source")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    from settings import Settings
    import angle_bench as ab
    from roop import baseline_probe as bp
    cfg = Settings(os.path.join(APP, "config.yaml"))
    g = ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "None", sync_config=True)
    g.g_desired_face_analysis = ["landmark_3d_68", "landmark_2d_106", "detection", "recognition"]
    g.lm68_lazy = False
    g.detector_engine = "retinaface_r50" if args.model_type == "r50" else "retinaface"
    thresh = float(g.face_detector_threshold)
    from roop import face_detector as new
    old = load_base(args.base)
    fn_old, fn_new = Counted(args.model_type), Counted(args.model_type)
    d_old = old.MultiScaleFaceDetector(detect_fn=fn_old)
    d_new = new.MultiScaleFaceDetector(detect_fn=fn_new)
    fn_plain = Counted(args.model_type)
    print("[ab] model=%s thresh=%s base=%s" % (args.model_type, thresh, args.base), flush=True)

    sources = [("d6", "double/d6.mp4", 0, 488, 1), ("Love", "Love.mp4", 1100, 1700, 1),
               ("d4", "double/d4.mp4", 0, 600, 2)]
    if args.limit:
        sources = [(n, r, lo, min(hi, lo + args.limit), 1) for n, r, lo, hi, _ in sources]

    def triggered_count():
        return bp.total("pyramid.path.pyramid_executed")

    if args.phase == "equiv":
        rows, per = [], {}
        pool = cf.ThreadPoolExecutor(max_workers=2, initializer=new.pool_worker_enter)

        def run_pair(label, idx, fr):
            res = {"label": label, "idx": idx}
            n_old0, n_new0 = fn_old.n, fn_new.n

            def call(det):
                t0 = triggered_count()
                out = det.detect(fr, det_size=640, det_thresh=thresh)
                return out, triggered_count() - t0

            # alternate who goes first, so a first-call cost never lands on one arm
            if len(rows) % 2:
                (o, trig_o), (n, trig) = call(d_old), call(d_new)
            else:
                (n, trig), (o, trig_o) = call(d_new), call(d_old)
            res.update({"triggered": bool(trig), "triggered_old": bool(trig_o),
                        "inf_old": fn_old.n - n_old0, "inf_new": fn_new.n - n_new0,
                        **{"plain_" + k: v for k, v in diff(o, n).items()}})
            # same new code from a flagged pool-worker thread (sequential scale passes)
            if trig:
                w = pool.submit(d_new.detect, fr, 640, thresh).result()
                res.update({"worker_" + k: v for k, v in diff(o, w).items()})
                # what the pyramid adds on top of a plain single pass at scale 1.0
                single = d_new.detect(fr, det_size=640, det_thresh=thresh, scales=[1.0])
                res.update({"vs_single_" + k: v for k, v in diff(single, n).items()})
            rows.append(res)
            c = per.setdefault(label, {"frames": 0, "triggered": 0, "inf_old": 0, "inf_new": 0})
            c["frames"] += 1; c["triggered"] += int(res["triggered"])
            c["inf_old"] += res["inf_old"]; c["inf_new"] += res["inf_new"]
            if c["frames"] % 100 == 0:
                print("[ab] %s %d frames, %d triggered, inferences old/new %d/%d"
                      % (label, c["frames"], c["triggered"], c["inf_old"], c["inf_new"]), flush=True)

        for label, idx, fr in frames(sources):
            run_pair(label, idx, fr)
        for label, idx, fr in synthetic_closeups(new, fn_plain, 40 if not args.limit else 3):
            run_pair(label, idx, fr)
        pool.shutdown()

        def worst(key, sel=lambda r: True):
            v = [r[key] for r in rows if key in r and sel(r) and r[key] == r[key]]
            return max(v) if v else None
        trig_rows = [r for r in rows if r["triggered"]]
        res = {
            "base": args.base, "model_type": args.model_type, "tolerance_px": TOL_PX,
            "per_source": per,
            "frames": len(rows), "triggered_frames": len(trig_rows),
            "frames_where_old_and_new_disagree_on_triggering": sum(1 for r in rows if r["triggered"] != r["triggered_old"]),
            "plain_thread": {"max_box_px": worst("plain_box"), "max_kps_px": worst("plain_kps"),
                             "max_score": worst("plain_score"),
                             "frames_over_tolerance": sum(1 for r in rows if r["plain_box"] > TOL_PX or r["plain_kps"] > TOL_PX),
                             "frames_with_different_box_count": sum(1 for r in rows if r["plain_n"][0] != r["plain_n"][1])},
            "pool_worker_thread": {"max_box_px": worst("worker_box"), "max_kps_px": worst("worker_kps"),
                                   "frames_over_tolerance": sum(1 for r in trig_rows if r.get("worker_box", 0) > TOL_PX or r.get("worker_kps", 0) > TOL_PX),
                                   "frames_with_different_box_count": sum(1 for r in trig_rows if r["worker_n"][0] != r["worker_n"][1])},
            # summed from the per-frame counts around the plain call only: fn_new.n also
            # holds the extra calls this harness makes for its own analysis
            "detector_inferences": {"old": sum(r["inf_old"] for r in rows),
                                    "new": sum(r["inf_new"] for r in rows),
                                    "old_per_frame": round(sum(r["inf_old"] for r in rows) / max(1, len(rows)), 3),
                                    "new_per_frame": round(sum(r["inf_new"] for r in rows) / max(1, len(rows)), 3),
                                    "old_per_triggered_frame": round(sum(r["inf_old"] for r in trig_rows) / max(1, len(trig_rows)), 3),
                                    "new_per_triggered_frame": round(sum(r["inf_new"] for r in trig_rows) / max(1, len(trig_rows)), 3)},
            "pyramid_vs_plain_single_pass": {
                "triggered_frames": len(trig_rows),
                "frames_where_merged_differs_over_tolerance": sum(1 for r in trig_rows if r.get("vs_single_box", 0) > TOL_PX or r.get("vs_single_kps", 0) > TOL_PX),
                "frames_with_different_box_count": sum(1 for r in trig_rows if r["vs_single_n"][0] != r["vs_single_n"][1]),
                "max_box_px": worst("vs_single_box", lambda r: r["triggered"])},
            "counters": {k: sum(v.values()) for k, v in bp.snapshot().items() if k.startswith("pyramid.")},
        }
        print(json.dumps(res, indent=1))
        path = os.path.join(REPO, "docs", "perf", "pyramid_ab_%s.json" % args.date)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"summary": res, "rows": rows}, fh, indent=1, default=str)
        print("wrote", path)
        return 0

    # ── speed ────────────────────────────────────────────────────────────────
    trig_frames = []
    for label, idx, fr in frames([("d6", "double/d6.mp4", 0, 488 if not args.limit else args.limit, 3)]):
        t0 = triggered_count()
        d_new.detect(fr, det_size=640, det_thresh=thresh)
        if triggered_count() > t0:
            trig_frames.append(fr)
        if len(trig_frames) >= 40:
            break
    print("[ab] speed set: %d triggered d6 frames" % len(trig_frames), flush=True)

    def throughput(det, nthreads, flagged, reps=2, force_executor=False):
        work = trig_frames * reps
        # isolates change 2: same reuse (change 1), but the pool-worker shortcut disabled
        saved_flag = new.in_pool_worker
        if force_executor:
            new.in_pool_worker = lambda: False
        def one(fr):
            det.detect(fr, det_size=640, det_thresh=thresh)
        init = new.pool_worker_enter if flagged else None
        t0 = time.perf_counter()
        if nthreads == 1:
            if flagged:
                new.pool_worker_enter()
            for fr in work:
                one(fr)
            if flagged:
                new.pool_worker_exit()
        else:
            with cf.ThreadPoolExecutor(max_workers=nthreads, initializer=init) as ex:
                list(ex.map(one, work))
        rate = len(work) / (time.perf_counter() - t0)
        new.in_pool_worker = saved_flag
        return rate

    results = {}
    for label, nthreads, flagged in (("1 plain thread (pool idle)", 1, False),
                                     ("%d pool-worker threads (render-like)" % args.threads, args.threads, True)):
        # warm both, then ABBA
        throughput(d_old, nthreads, flagged, reps=1); throughput(d_new, nthreads, flagged, reps=1)
        # A B C C B A: position in the sequence cancels out
        order = [("old", d_old, False), ("new", d_new, False), ("new_reuse_only", d_new, True),
                 ("new_reuse_only", d_new, True), ("new", d_new, False), ("old", d_old, False)]
        fps = {"old": [], "new": [], "new_reuse_only": []}
        for name, det, force in order:
            fps[name].append(throughput(det, nthreads, flagged, force_executor=force))
        results[label] = {k: [round(x, 3) for x in v] for k, v in fps.items()}
        for k in fps:
            results[label]["mean_" + k] = round(statistics.mean(fps[k]), 3)
        base = statistics.mean(fps["old"])
        results[label]["new_vs_old_pct"] = round(100.0 * (statistics.mean(fps["new"]) / base - 1), 1)
        results[label]["reuse_only_vs_old_pct"] = round(100.0 * (statistics.mean(fps["new_reuse_only"]) / base - 1), 1)
        print("[ab]", label, json.dumps(results[label]), flush=True)
    path = os.path.join(REPO, "docs", "perf", "pyramid_speed_%s.json" % args.date)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"frames": len(trig_frames), "model_type": args.model_type, "results": results}, fh, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
