"""Old vs new rotated rescues: is every face identical, and how many aux calls went away?

    env/Scripts/python.exe tests/ab_rescue_aux.py --base f9aaf96

`_rescue_rotated` and `_detect_faces`'s partial-miss rescue used to detect on a rotated
frame WITH the aux models (recognition, 106- and 68-point landmarks) and then throw away
the aux work of every face that turned out to be a duplicate. They now detect without
aux, run the duplicate test on an un-rotated copy of the coordinates, and run the aux
models only on the survivors (face_util._rotated_pass).

This runs BOTH implementations on the same real frames in one process: the new code is
the live `roop.face_util`; the old code is `git show <base>:app/roop/face_util.py`
executed as a second module with its own analyser pool. Per frame it calls
`_detect_faces(frame, expected_count=N)` on each and compares, face by face and in
order: bbox, kps, det_score, embedding (cosine), 106- and 68-point landmarks.

A NULL CONTROL comes first in the report: the old implementation compared with ITSELF
on the same frames (a second call, which leases another pooled instance). That is the
noise floor of "same code, same frame, different TensorRT context"; the old-vs-new
comparison only means something against it.

Aux calls are the model-level `get` counters (baseline_probe), differenced around each
call, so they count the models' own executions, not the wrappers.
"""
import argparse
import json
import os
import subprocess
import sys
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
os.environ.setdefault("ROOP_PROFILE", "1")       # the model-level counters are gated on it

import fixtures                      # noqa: E402

# (name, file, first frame, last frame, stride). d6 is left out on purpose: its baseline
# counters show no partial-miss / rotated rescue activity at all, so there is nothing
# of this change to compare there.
CLIPS = [("d4", "double/d4.mp4", 0, 600, 2),
         ("Love", "Love.mp4", 1100, 1700, 2),
         ("d1", "double/d1.mp4", 0, 418, 1)]
AUX_KEYS = ("aux.buffalo_l.recognition.get", "aux.buffalo_l.landmark_2d_106.get",
            "aux.buffalo_l.landmark_3d_68.get")


def load_base(ref):
    src = subprocess.check_output(["git", "show", "%s:app/roop/face_util.py" % ref], cwd=REPO)
    mod = types.ModuleType("roop.face_util_base")
    mod.__file__ = os.path.join(APP, "roop", "face_util_base.py")
    sys.modules["roop.face_util_base"] = mod
    exec(compile(src, "face_util@%s" % ref, "exec"), mod.__dict__)
    return mod


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else float("nan")


def maxabs(a, b):
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")


def compare(fa, fb):
    """Face-by-face stats for two `_detect_faces` results on one frame."""
    out = {"n_a": len(fa), "n_b": len(fb), "faces": []}
    for a, b in zip(fa, fb):
        out["faces"].append({
            "bbox": maxabs(a.bbox, b.bbox), "kps": maxabs(a.kps, b.kps),
            "det_score": abs(float(a.det_score) - float(b.det_score)),
            "emb_cos": cos(a.embedding, b.embedding) if getattr(a, "embedding", None) is not None
            and getattr(b, "embedding", None) is not None else float("nan"),
            "lm106": maxabs(getattr(a, "landmark_2d_106", None), getattr(b, "landmark_2d_106", None)),
            "lm68": maxabs(getattr(a, "landmark_3d_68", None), getattr(b, "landmark_3d_68", None)),
        })
    return out


def summarize(rows, label):
    faces = [f for r in rows for f in r["faces"]]
    frames_diff_n = sum(1 for r in rows if r["n_a"] != r["n_b"])
    def mx(k):
        v = [f[k] for f in faces if f[k] == f[k]]
        return max(v) if v else None
    emb = [f["emb_cos"] for f in faces if f["emb_cos"] == f["emb_cos"]]
    return {"label": label, "frames": len(rows), "faces_compared": len(faces),
            "frames_with_different_face_count": frames_diff_n,
            "max_bbox_abs": mx("bbox"), "max_kps_abs": mx("kps"), "max_det_score_abs": mx("det_score"),
            "min_embedding_cosine": min(emb) if emb else None,
            "max_lm106_abs": mx("lm106"), "max_lm68_abs": mx("lm68"),
            "faces_exactly_equal": sum(1 for f in faces if f["bbox"] == 0 and f["kps"] == 0
                                       and f["emb_cos"] >= 0.99999999),
            "faces_below_0.9999_cosine": sum(1 for f in faces if f["emb_cos"] == f["emb_cos"]
                                             and f["emb_cos"] < 0.9999)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="f9aaf96")
    ap.add_argument("--date", default="2026-10-04")
    ap.add_argument("--expected", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="debug: frames per clip")
    args = ap.parse_args()

    from settings import Settings
    import angle_bench as ab
    from roop import baseline_probe as bp
    cfg = Settings(os.path.join(APP, "config.yaml"))
    g = ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "None", sync_config=True)
    g.g_desired_face_analysis = ["landmark_3d_68", "landmark_2d_106", "detection", "recognition"]
    g.lm68_lazy = False
    g.detector_engine = str(cfg.detector_engine)
    from roop import face_util as new
    old = load_base(args.base)
    print("[ab] engine=%s expected_count=%d base=%s" % (g.detector_engine, args.expected, args.base), flush=True)

    def aux_total():
        return {k: bp.total(k) for k in AUX_KEYS}

    def timed(mod, frame, ec):
        before = aux_total()
        t0 = time.perf_counter()
        faces = mod._detect_faces(frame, expected_count=ec)
        dt = time.perf_counter() - t0
        after = aux_total()
        return faces, dt, {k: after[k] - before[k] for k in AUX_KEYS}

    null_rows, ab_rows = [], []
    t_old = t_new = 0.0
    aux_old = {k: 0 for k in AUX_KEYS}
    aux_new = {k: 0 for k in AUX_KEYS}
    per_clip = {}
    for name, rel, lo, hi, stride in CLIPS:
        cap = cv2.VideoCapture(fixtures.clip(rel, required=True))
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        n = 0
        c = per_clip.setdefault(name, {"frames": 0, "rescue_frames": 0, "aux_old": 0, "aux_new": 0})
        for idx in range(lo, hi):
            ok, fr = cap.read()
            if not ok:
                break
            if (idx - lo) % stride:
                continue
            if args.limit and n >= args.limit:
                break
            n += 1
            # Alternate who goes first so a first-call cost (cold caches, a TensorRT
            # context's first use of a shape) does not always land on the same arm.
            if n % 2:
                f_old, d_old, a_old = timed(old, fr, args.expected)
                f_new, d_new, a_new = timed(new, fr, args.expected)
            else:
                f_new, d_new, a_new = timed(new, fr, args.expected)
                f_old, d_old, a_old = timed(old, fr, args.expected)
            f_old2, _, _ = timed(old, fr, args.expected)          # null control: old vs old
            t_old += d_old
            t_new += d_new
            for k in AUX_KEYS:
                aux_old[k] += a_old[k]
                aux_new[k] += a_new[k]
            c["frames"] += 1
            c["aux_old"] += sum(a_old.values())
            c["aux_new"] += sum(a_new.values())
            c["rescue_frames"] += int(sum(a_old.values()) != sum(a_new.values()))
            null_rows.append(compare(f_old, f_old2))
            ab_rows.append(compare(f_old, f_new))
            if n % 50 == 0:
                print("[ab] %s %d frames  aux old/new %d/%d" % (name, n, c["aux_old"], c["aux_new"]), flush=True)
        cap.release()

    res = {"base": args.base, "expected_count": args.expected, "engine": g.detector_engine,
           "null_old_vs_old": summarize(null_rows, "old vs old (same code, second call)"),
           "old_vs_new": summarize(ab_rows, "old vs new"),
           "aux_calls_old": aux_old, "aux_calls_new": aux_new,
           "aux_calls_saved": {k: aux_old[k] - aux_new[k] for k in AUX_KEYS},
           "detect_faces_seconds_old": round(t_old, 2), "detect_faces_seconds_new": round(t_new, 2),
           "per_clip": per_clip,
           "worst_old_vs_new": sorted(
               ({"i": i, **f} for i, r in enumerate(ab_rows) for f in r["faces"]),
               key=lambda f: (f["emb_cos"] if f["emb_cos"] == f["emb_cos"] else 2.0))[:5]}
    print(json.dumps({k: v for k, v in res.items() if k != "worst_old_vs_new"}, indent=1))
    path = os.path.join(REPO, "docs", "perf", "rescue_aux_ab_%s.json" % args.date)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
