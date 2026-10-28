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

Stage 20 adds two things:

``--algebra``   no clip, no GPU, no model: the invariants of the vector algebra itself on random, correlated and
                (when ``--a/--b`` load) real library embeddings -- Slerp and spherical-mean unit norm, constant angular
                speed, keyframe-track unit norm / smoothness / cut counts per curve, and the de-identification bound.
                Exit status 1 on any violation, so it can gate a commit.
``--stage20``   adds render arms: ``slerp`` vs ``linear`` at the same weights, ``deid`` against A at two thresholds
                (does the SWAPPED face's cos_A fall, not just the input vector's?), and a ``keyframe A->B`` walk whose
                frame number reaches the render the way /api/preview's does (``roto_preview_frame``). The keyframe
                table reports mean cos_B per sampled frame, the largest step between neighbours and whether it rises.
                These are measurements of the output face, reported as measured -- no pass/fail is invented for them.
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


def _corr_pair(rng, c):
    """Two unit vectors with cosine c."""
    a = unit(rng.standard_normal(512))
    p = rng.standard_normal(512)
    p -= (p @ a) * a
    p = unit(p)
    return a, c * a + np.sqrt(1 - c * c) * p


def algebra_check(real_vectors=None, seed=0, trials=400):
    """Invariants of roop.identity_algebra, no model. Returns the number of violations."""
    from roop import identity_algebra as ia
    rng = np.random.default_rng(seed)
    bad = []

    def check(name, value, limit, note=""):
        ok = value <= limit
        print(f"  {'ok  ' if ok else 'FAIL'} {name:58s} {value:12.3e}  (<= {limit:.0e}) {note}")
        if not ok:
            bad.append(name)

    def ang(a, b):
        return float(np.arccos(np.clip(unit(a) @ unit(b), -1, 1)))

    families = {"random (different people, cos~0)": lambda: (unit(rng.standard_normal(512)), unit(rng.standard_normal(512))),
                "correlated (cos 0.6-0.95)": lambda: _corr_pair(rng, float(rng.uniform(0.6, 0.95))),
                "near-identical (cos>0.999)": lambda: _corr_pair(rng, float(rng.uniform(0.999, 0.99999)))}
    if real_vectors is not None:
        families["real library pair"] = lambda: real_vectors

    print("Slerp: unit norm and constant angular speed")
    for fam, make in families.items():
        worst_norm = worst_speed = 0.0
        for _ in range(trials if fam != "real library pair" else 1):
            a, b = make()
            theta = ang(a, b)
            ts = np.linspace(0, 1, 9)
            zs = [ia.slerp(a, b, t) for t in ts]
            worst_norm = max(worst_norm, max(abs(float(np.linalg.norm(z.astype(np.float64))) - 1) for z in zs))
            if theta > 1e-3:
                worst_speed = max(worst_speed, max(abs(ang(zs[i], zs[i + 1]) - theta / 8) for i in range(8)))
        check(f"[{fam}] max | |slerp| - 1 |", worst_norm, 1e-6)
        check(f"[{fam}] max | step - theta/8 | (rad)", worst_speed, 1e-4)

    print("\nSpherical mean of 3-5 identities: unit norm, converged, Karcher optimality")
    for n in (3, 4, 5):
        worst_norm = worst_grad = 0.0
        not_converged = 0
        for _ in range(trials // 4):
            zs = [unit(rng.standard_normal(512)) for _ in range(n)]
            w = rng.uniform(0.1, 1.0, n)
            m, info = ia.spherical_mean(zs, w)
            if not info["converged"]:
                not_converged += 1
                continue
            m64 = m.astype(np.float64)
            worst_norm = max(worst_norm, abs(float(np.linalg.norm(m64)) - 1))
            wn = w / w.sum()
            g = np.zeros(512)
            for wi, z in zip(wn, zs):
                th = ang(m64, z)
                g += wi * th / np.sin(th) * (z - np.cos(th) * m64)
            worst_grad = max(worst_grad, float(np.linalg.norm(g)))
        check(f"[{n} sources] max | |mean| - 1 |", worst_norm, 1e-6)
        check(f"[{n} sources] max |Riemannian gradient|", worst_grad, 1e-5)
        check(f"[{n} sources] runs that did not converge", float(not_converged), 0)

    print("\nKeyframe track: unit norm everywhere, smoothness, cuts")
    for curve in ia.CURVES:
        a, b = _corr_pair(rng, 0.2)
        c, d = _corr_pair(rng, 0.1)
        frames = (0, 40, 75, 140)
        track = ia.KeyframeTrack(ia.IdentityKeyframe(f, v, curve) for f, v in zip(frames, (a, b, c, d)))
        sm = track.smoothness()
        check(f"[{curve}] max | |v(f)| - 1 | over {sm['frames']} frames", max(abs(sm["min_norm"] - 1), abs(sm["max_norm"] - 1)), 1e-6)
        expect_cuts = len(frames) - 1 if curve == ia.CURVE_HOLD else 0
        check(f"[{curve}] cuts != expected ({expect_cuts})", float(abs(sm["cuts"] - expect_cuts)), 0)
        if curve == ia.CURVE_LINEAR:
            fastest = max(ang(a, b) / 40, ang(b, c) / 35, ang(c, d) / 65)
            check("[linear] max step - fastest segment's theta/dF (rad)", abs(sm["max_step"] - fastest), 1e-4,
                  f"(max step {np.degrees(sm['max_step']):.3f} deg/frame)")
        if curve == ia.CURVE_CUBIC_BEZIER:
            m = track.vectors(range(0, 141)).astype(np.float64)
            total = sum(ang(m[i], m[i + 1]) for i in range(140))
            seg = ang(a, b) + ang(b, c) + ang(c, d)
            check("[cubic_bezier] total arc vs sum of segment arcs (rad)", abs(total - seg), 2e-3,
                  f"(max step {np.degrees(sm['max_step']):.3f} deg/frame vs linear mean "
                  f"{np.degrees(seg / 140):.3f})")

    print("\nDe-identification: the cosine bound, closed form")
    for fam in ("one reference", "three correlated references"):
        worst_excess = worst_closed = 0.0
        unsat = 0
        for _ in range(trials // 4):
            if fam == "one reference":
                ref, p = _corr_pair(rng, float(rng.uniform(0.5, 0.97)))
                refs = [ref]
            else:
                base = unit(rng.standard_normal(512))
                refs = [unit(base + 0.5 * rng.standard_normal(512)) for _ in range(3)]
                p = unit(sum(refs) + 0.5 * rng.standard_normal(512))
            tau = float(rng.uniform(0.0, 0.4))
            res = ia.deidentify(p, refs, tau)
            if not res.satisfied:
                unsat += 1
                continue
            worst_excess = max(worst_excess, max(res.cosine_after) - tau)
        check(f"[{fam}] max (cos_after - tau)", max(worst_excess, 0.0), 1e-5)
        check(f"[{fam}] unsatisfied", float(unsat), 0)

    print(f"\n{len(bad)} violation(s)" + (": " + "; ".join(bad) if bad else ""))
    return len(bad)


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


def stage20_arms(a_id, b_id, first_frame, last_frame):
    """Slerp vs linear at the SAME weights, de-identification against A, and a keyframed A -> B walk.
    Keyframe frame_index is 0-based absolute, which is the cv2 position the bench seeks to."""
    def comps(*pairs):
        return [{"source_id": s, "weight": w} for s, w in pairs]
    out = []
    for wa in (75, 25):
        for mode in ("slerp", "linear"):
            out.append((f"{mode} A{wa}/B{100 - wa}", {"enabled": True, "mode": mode, "min_cosine": 0.8,
                                                      "components": comps((a_id, wa), (b_id, 100 - wa))}))
    for tau in (0.30, 0.15):
        out.append((f"deid vs A tau={tau:.2f}", {"enabled": True, "min_cosine": 0.8, "components": comps((a_id, 100)),
                                                 "deidentify": {"enabled": True, "references": [a_id], "max_cosine": tau}}))
    out.append(("keyframe A->B", {"enabled": True, "min_cosine": 0.8, "components": comps((a_id, 50), (b_id, 50)),
                                  "keyframes": [{"frame_index": first_frame, "weights": {a_id: 100, b_id: 0}, "curve": "linear"},
                                                {"frame_index": last_frame, "weights": {a_id: 0, b_id: 100}}]}))
    return out


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
    ap.add_argument("--stage20", action="store_true",
                    help="add the Slerp / de-identification / keyframe arms (see the module docstring)")
    ap.add_argument("--algebra", action="store_true",
                    help="vector-algebra invariants only: no clip, no GPU, no model; exit 1 on a violation")
    args = ap.parse_args()

    if args.algebra:
        real = None
        try:
            fa, fb = load_library_faceset(args.a), load_library_faceset(args.b)
            from roop import identity_algebra as _ia
            real = (_ia.faceset_identity_vector(fa), _ia.faceset_identity_vector(fb))
            print(f"[ia] real library pair {args.a}/{args.b}: cos = {float(unit(real[0]) @ unit(real[1])):.3f}")
        except BaseException as exc:                      # SystemExit when the faceset is absent
            print(f"[ia] real library embeddings unavailable ({type(exc).__name__}: {exc}); random + correlated only")
        raise SystemExit(1 if algebra_check(real) else 0)

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
    last_frame = args.start + (args.frames - 1) * args.stride
    if args.sweep:
        patch_units(ia)     # cached object: every resolve_recipe sees it
        arm_list = sweep_arms()
    else:
        arm_list = arms(fs_a._source_id, fs_b._source_id)
        if args.stage20:
            arm_list += stage20_arms(fs_a._source_id, fs_b._source_id, args.start, last_frame)
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
                frames.append((clip, fr, args.start + k * args.stride, k))
        cap.release()

    graded = 0
    why = {"no_match_in_some_arm": 0, "base_not_swapped": 0}
    for fi, (clip, fr, frame_no, kpos) in enumerate(frames):
        orig = meter.measure(fr)
        if not orig:
            continue
        per_arm = {}
        # What /api/preview does: the UI's 1-based frame number reaches the render as roto_preview_frame, and a
        # keyframed recipe reads it (ProcessMgr._roto_frame_ui). Set for every arm; only keyframes consume it.
        g.roto_preview_frame = frame_no + 1
        for name, recipe in arm_list:
            g.identity_blend = recipe
            before = calls["n"]
            out = live_swap(fr.copy(), options, input_facesets=[fs_a])
            hooks[name] += calls["n"] - before
            per_arm[name] = meter.measure(out) if out is not None else []
        g.identity_blend = None
        g.roto_preview_frame = None
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
                                       age=f["age"], male=f["male"], jaw=f["jaw"], mouth=f["mouth"], k=kpos,
                                       in_=float(o["emb"] @ za)))
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
    if args.stage20 and "keyframe A->B" in rows:
        kf = rows["keyframe A->B"]
        ks = sorted({r["k"] for r in kf})
        print(f"\nkeyframe A->B: linear walk from frame {args.start} (A) to {last_frame} (B); output face, "
              f"mean over paired faces ({len(ks)} sampled positions)")
        print(f"{'k':>3s} {'frame':>6s} {'n':>3s} {'cos_A':>8s} {'cos_B':>8s} {'cos_B-cos_A':>12s}")
        walk = []
        for k in ks:
            sel = [r for r in kf if r["k"] == k]
            ca, cb = float(np.mean([r["cos_A"] for r in sel])), float(np.mean([r["cos_B"] for r in sel]))
            walk.append((cb - ca))
            print(f"{k:3d} {args.start + k * args.stride:6d} {len(sel):3d} {ca:8.3f} {cb:8.3f} {cb - ca:12.3f}")
        if len(walk) >= 3:
            steps = np.diff(walk)
            rho = float(np.corrcoef(np.arange(len(walk)), walk)[0, 1])
            report["keyframe walk"] = {"cos_B_minus_cos_A": walk, "max_step": float(np.abs(steps).max()),
                                       "pearson_vs_position": rho, "rises": bool(walk[-1] > walk[0])}
            print(f"  rises A->B: {walk[-1] > walk[0]}  largest step between neighbours: {np.abs(steps).max():.3f}  "
                  f"correlation with position: {rho:+.3f}  (reported, not gated: {len(ks)} positions only)")
    if args.stage20:
        for name in [n for n, _ in arm_list if n.startswith("deid")]:
            print(f"{name}: swapped-face cos_A {np.mean([r['cos_A'] for r in rows[name]]):.3f} "
                  f"(base arm {np.mean([r['cos_A'] for r in rows['base (A)']]):.3f}; hook calls {hooks[name]})")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"graded": graded, "frames": len(frames), "a": args.a, "b": args.b,
                       "swap_model": swap_model, "enhancer": str(cfg.selected_enhancer),
                       "arms": report}, fh, indent=1)


if __name__ == "__main__":
    main()
