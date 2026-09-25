"""Does the Identity Blender change the SWAPPED face, or only the vector?

Real frames go through ``roop.core.live_swap`` with the user's live config
(``init_pipeline(sync_config=True)``); only ``roop.globals.identity_blend``
changes between arms. Every output face is re-measured by an independent
buffalo_l instance (detection + w600k + genderage + 68 landmarks):

    cos_A / cos_B   w600k cosine of the swapped face to source A / B
    age             genderage's CONTINUOUS age (pred[2] * 100, unrounded)
    male            genderage's male-minus-female score (pred[1] - pred[0])
    jaw             |p4-p12| / |p0-p16| on the output's 68 landmarks
    mouth           mouth opening + smile width / interocular

Faces are paired across arms: a face is graded only if it is found (IoU >= 0.5
to the original detection) in EVERY arm and the base arm actually swapped it
(cos to A rose by >= 0.15 over the original).  The ``hook`` column counts
``ResolvedBlend.transform`` calls, i.e. that the render path ran the recipe --
an arm that reads "no change" with hook 0 measured nothing.

    app/env/Scripts/python.exe app/tests/identity_algebra_bench.py --a akansha --b raunak
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import fixtures                       # noqa: E402
import angle_bench as ab              # noqa: E402
from two_face_video import load_library_faceset, map_mask_engine  # noqa: E402

DEFAULT_CLIPS = ['single/s%d.mp4' % i for i in (1, 2, 3, 4, 5, 6)] + \
                ['double/d1.mp4', 'double/d6.mp4']


def unit(v):
    v = np.asarray(v, np.float64).reshape(-1)
    return v / (np.linalg.norm(v) or 1.0)


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Meter:
    def __init__(self):
        import insightface
        self.fa = insightface.app.FaceAnalysis(
            name="buffalo_l", root=os.path.join(APP, ".."),
            allowed_modules=["detection", "recognition", "genderage", "landmark_3d_68"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
        self.ga = self.fa.models["genderage"]

    def genderage_raw(self, img, face):
        from insightface.utils import face_align
        b = face.bbox
        w, h = b[2] - b[0], b[3] - b[1]
        center = ((b[2] + b[0]) / 2, (b[3] + b[1]) / 2)
        size = self.ga.input_size[0]
        aimg, _ = face_align.transform(img, center, size, size / (max(w, h) * 1.5), 0)
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.ga.input_std, (size, size),
                                     (self.ga.input_mean,) * 3, swapRB=True)
        pred = self.ga.session.run(self.ga.output_names, {self.ga.input_name: blob})[0][0]
        return float(pred[2] * 100.0), float(pred[1] - pred[0])

    def measure(self, img):
        out = []
        for f in self.fa.get(img) or []:
            age, male = self.genderage_raw(img, f)
            lm = np.asarray(f.landmark_3d_68, np.float64)[:, :2]
            d = lambda a, b: float(np.linalg.norm(lm[a] - lm[b]))
            iod = d(36, 45) or 1.0
            out.append(dict(bbox=[float(x) for x in f.bbox], emb=unit(f.normed_embedding),
                            age=age, male=male, jaw=d(4, 12) / (d(0, 16) or 1.0),
                            mouth=d(62, 66) / iod + d(48, 54) / iod))
        return out


def sweep_arms():
    """Raw tangent-step sweep, bypassing the dial calibration: the dial is
    set in the direction's own units after `patch_units` makes 1 unit = 1
    tangent step. Answers "does the RENDER respond at all, and how much"."""
    def r(**dials):
        return {"enabled": True, "components": [], "dials": dials, "min_cosine": 0.5}
    out = [("base (A)", None)]
    for name in ("age", "gender", "jawline"):
        for t in (-0.75, -0.4, 0.4, 0.75):
            out.append((f"{name} t={t:+.2f}", r(**{name: t})))
    return out


def patch_units(ia):
    """Make every dial read in raw tangent steps (age: 1 'year' = 1 step,
    others: dial 1 = 1 step) so sweep_arms can address the step directly."""
    d = ia.load_directions()
    d.units_per_step[:] = 1.0
    d.spread[:] = 1.0 / ia.DIAL_SIGMAS
    return d


def arms(a_id, b_id):
    def r(components=(), **dials):
        return {"enabled": True, "components": [{"source_id": s, "weight": w} for s, w in components],
                "dials": dials, "min_cosine": 0.8}
    out = [("base (A)", None)]
    for wa in (75, 50, 25, 0):
        out.append((f"blend A{wa}/B{100 - wa}", r([(a_id, wa), (b_id, 100 - wa)])))
    for yrs in (-30, -15, 15, 30):
        out.append((f"age {yrs:+d}y", r(age=yrs)))
    for name in ("gender", "jawline", "expression"):
        for v in (-1, 1):
            out.append((f"{name} {v:+d}", r(**{name: v})))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    ap.add_argument("--a", default="akansha")
    ap.add_argument("--b", default="raunak")
    ap.add_argument("--start", type=int, default=60)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--out", default="")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="raw tangent-step sweep instead of the calibrated dial arms")
    args = ap.parse_args()

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    swap_model, mask_engine = str(cfg.swap_model), str(cfg.mask_engine)
    g = ab.init_pipeline(str(cfg.provider), swap_model, str(cfg.selected_enhancer),
                         mask_engine, sync_config=True)
    options = ab.build_options(g, swap_model, map_mask_engine(mask_engine))
    fs_a, fs_b = load_library_faceset(args.a), load_library_faceset(args.b)
    fs_a._source_id, fs_b._source_id = f"bench:{args.a}", f"bench:{args.b}"
    g.INPUT_FACESETS = [fs_a, fs_b]
    from roop.core import live_swap
    from roop import identity_algebra as ia
    print(f"[ia] swap={swap_model} enhancer={cfg.selected_enhancer} mask={mask_engine}")

    calls = {"n": 0}
    orig_transform = ia.ResolvedBlend.transform

    def counted(self, *a, **k):
        out = orig_transform(self, *a, **k)
        if out is not None:
            calls["n"] += 1
        return out
    ia.ResolvedBlend.transform = counted

    meter = Meter()
    za, zb = unit(ia.faceset_identity_vector(fs_a)), unit(ia.faceset_identity_vector(fs_b))
    print(f"[ia] cos(A,B)={za @ zb:.3f}")
    if args.sweep:
        patch_units(ia)     # cached object: every resolve_recipe sees it
        arm_list = sweep_arms()
    else:
        arm_list = arms(fs_a._source_id, fs_b._source_id)
    rows = {name: [] for name, _ in arm_list}
    hooks = {name: 0 for name, _ in arm_list}

    frames = []
    for clip in args.clips:
        path = fixtures.clip(clip)
        if not os.path.isfile(path):
            print(f"[ia] skip {clip}: missing")
            continue
        cap = cv2.VideoCapture(path)
        for k in range(args.frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start + k * args.stride)
            ok, fr = cap.read()
            if ok:
                frames.append((clip, fr))
        cap.release()

    graded = 0
    why = {"no_match_in_some_arm": 0, "base_not_swapped": 0}
    for fi, (clip, fr) in enumerate(frames):
        orig = meter.measure(fr)
        if not orig:
            continue
        per_arm = {}
        for name, recipe in arm_list:
            g.identity_blend = recipe
            before = calls["n"]
            out = live_swap(fr.copy(), options, input_facesets=[fs_a])
            hooks[name] += calls["n"] - before
            per_arm[name] = meter.measure(out) if out is not None else []
        g.identity_blend = None
        for o in orig:
            match = {}
            for name in per_arm:
                best = max(per_arm[name], key=lambda f: iou(f["bbox"], o["bbox"]), default=None)
                if best is None or iou(best["bbox"], o["bbox"]) < 0.5:
                    break
                match[name] = best
            if len(match) != len(per_arm):
                why["no_match_in_some_arm"] += 1
                continue
            if float(match["base (A)"]["emb"] @ za) < float(o["emb"] @ za) + 0.15:
                why["base_not_swapped"] += 1
                if args.debug:
                    print(f"[ia]   not swapped: orig cos_A {float(o['emb'] @ za):.3f} "
                          f"base cos_A {float(match['base (A)']['emb'] @ za):.3f} "
                          f"cos(base,orig) {float(match['base (A)']['emb'] @ o['emb']):.3f}")
                continue      # the base arm did not swap this face
            graded += 1
            for name, f in match.items():
                rows[name].append(dict(cos_A=float(f["emb"] @ za), cos_B=float(f["emb"] @ zb),
                                       age=f["age"], male=f["male"], jaw=f["jaw"], mouth=f["mouth"]))
        print(f"[ia] frame {fi + 1}/{len(frames)} {clip}: {len(orig)} faces, "
              f"{graded} graded so far, rejected {why}", flush=True)

    if not graded:
        raise SystemExit("no face was swapped in the base arm -- nothing graded")
    base = rows["base (A)"]
    keys = ["cos_A", "cos_B", "age", "male", "jaw", "mouth"]
    print(f"\n{graded} paired faces over {len(frames)} frames")
    print(f"{'arm':18s} {'hook':>5s} " + " ".join(f"{k:>8s}" for k in keys) + "   d_age (paired, mean+-sd)")
    report = {}
    for name, _ in arm_list:
        rs = rows[name]
        means = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        dage = np.array([r["age"] - b["age"] for r, b in zip(rs, base)])
        report[name] = {**means, "hook": hooks[name], "d_age_mean": float(dage.mean()),
                        "d_age_sd": float(dage.std()),
                        "d_male_mean": float(np.mean([r["male"] - b["male"] for r, b in zip(rs, base)])),
                        "d_jaw_mean": float(np.mean([r["jaw"] - b["jaw"] for r, b in zip(rs, base)])),
                        "d_mouth_mean": float(np.mean([r["mouth"] - b["mouth"] for r, b in zip(rs, base)]))}
        print(f"{name:18s} {hooks[name]:5d} " + " ".join(f"{means[k]:8.3f}" for k in keys)
              + f"   {dage.mean():+.2f}+-{dage.std():.2f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"graded": graded, "frames": len(frames), "a": args.a, "b": args.b,
                       "swap_model": swap_model, "enhancer": str(cfg.selected_enhancer),
                       "arms": report}, fh, indent=1)


if __name__ == "__main__":
    main()
