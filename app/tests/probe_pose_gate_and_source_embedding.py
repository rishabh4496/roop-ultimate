"""Two Stage 9 questions, answered on real heads instead of argued.

    env/Scripts/python.exe tests/probe_pose_gate_and_source_embedding.py --section gates
    env/Scripts/python.exe tests/probe_pose_gate_and_source_embedding.py --section source

Measurement only: it never edits the code under test. Needs `faceset-v1-backup-2026-09-03`
(36 studio banks, 5 photos each, a yaw sweep of one person) under the clip root.

  gates   Would a swap-time "no frontal facial plane" gate refuse the heads the swap handles well?
          For all real banks, the pose the pipeline reads (solve_pose_jaw_5pt) and each candidate
          predicate: |yaw| > 72, |pitch| > 55, interocular distance < 12% of box width, signed
          eye-line x nose-to-mouth cross product <= 0, detector score < 0.55 -- counted by |yaw| bin.
  source  Does a pose-adaptive source embedding beat the centroid the product uses today?
          Real yaw-sweep plates of one person are swapped with a source built three ways from the
          OTHER person's photos -- the production mean, the nearest-pose photo, a SLERP between the two
          bracketing photos -- and scored with AdaFace (independent of the swapper's own embedding).
          Leave-one-out: the photo at the target's pose is HELD OUT of the arms and is the score
          reference, so "nearest" cannot simply copy the photo it is graded against. The swapper
          caches its latent ON the face (`_latent_<model>`); it is deleted per arm, or every arm
          silently swaps with the first one's identity (an earlier version of this probe did).

Results of 2026-10-19 (RTX 4070, realswap, retinaface_r50):

  gates   180 faces, 71 of them |yaw| >= 72 (real profiles): ALL 71 refused -- 66 by the yaw cutoff,
          63 by IOD < 12% (a profile head's IOD / box width has median 0.070: one eye is hidden).
          0 of the 109 faces under 58 degrees refused by anything. The sweep bench
          (tests/angle_bench.py --swap-model realswap --rolls 0) swaps those same +-90 plates with
          feature drift < 0.09 interocular distances and identity 0.64-0.76 vs 0.67-0.79 frontal.
  source  60 swaps, AdaFace cosine to the held-out pose-matched photo:
              mean (production) 0.627   SLERP 0.603   nearest 0.575
          paired SLERP - mean -0.024 (19/60 wins), nearest - mean -0.052 (13/60), SLERP - nearest +0.028 (37/60).
          By target yaw: frontal SLERP +0.016 over mean; 45 degrees -0.010; 90 degrees -0.058.
"""
import argparse
import collections
import itertools
import os
import sys
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures                                                      # noqa: E402

BANK_DIR = "faceset-v1-backup-2026-09-03"
SWEEP_BANKS = ["sadaf", "anshita", "mahima", "mahek"]     # proper -90..+90 sweeps (5-point yaw within ~8 deg)


def unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def slerp(a, b, t):
    a, b = unit(a), unit(b)
    th = np.arccos(float(np.clip(a @ b, -1, 1)))
    if th < 1e-4:
        return unit((1 - t) * a + t * b)
    return unit((np.sin((1 - t) * th) * a + np.sin(t * th) * b) / np.sin(th))


def yaw_of(face, jaw_pose, plain_pose):
    k = np.asarray(face.kps, np.float64)
    p = jaw_pose(k) or plain_pose(k)
    return float(p[0])


def section_gates():
    import angle_bench as ab
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "None", sync_config=True)
    from roop.face_util import get_first_face, solve_pose_jaw_5pt, solve_pose_5pt
    root = fixtures.clip_dir(BANK_DIR, required=True)
    rows = []
    for path in sorted(os.listdir(root)):
        if not path.endswith(".fsz") or "_v2" in path:
            continue
        z = zipfile.ZipFile(os.path.join(root, path))
        for name in sorted(n for n in z.namelist() if n.lower().endswith(".png")):
            img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
            f = get_first_face(img)
            if f is None:
                continue
            k = np.asarray(f.kps, np.float64)
            pose = solve_pose_jaw_5pt(k) or solve_pose_5pt(k)
            if pose is None:
                continue
            bw = float(f.bbox[2] - f.bbox[0])
            eye, n2m = k[1] - k[0], (k[3] + k[4]) / 2 - k[2]
            rows.append((abs(float(pose[0])), abs(float(pose[1])), float(np.linalg.norm(eye)) / bw,
                         float(eye[0] * n2m[1] - eye[1] * n2m[0]), float(f.det_score)))
    a = np.array(rows)
    print("faces: %d" % len(a))
    print("|yaw| bin     n | refused by: yaw>72  pitch>55  IOD<12%  cross<=0  det<0.55 | any")
    for lo, hi in ((0, 58), (58, 72), (72, 88), (88, 181)):
        s = a[(a[:, 0] >= lo) & (a[:, 0] < hi)]
        if not len(s):
            print("%3d-%-3d  %5d" % (lo, hi, 0))
            continue
        g = [s[:, 0] > 72, s[:, 1] > 55, s[:, 2] < 0.12, s[:, 3] <= 0, s[:, 4] < 0.55]
        print("%3d-%-3d  %5d | %19d %9d %8d %9d %8d | %d" % (
            lo, hi, len(s), *[int(x.sum()) for x in g], int(np.any(np.stack(g), axis=0).sum())))
    prof = a[a[:, 0] >= 72]
    print("IOD / box width at |yaw| >= 72: median %.3f  min %.3f  max %.3f ; at |yaw| < 30: median %.3f" % (
        np.median(prof[:, 2]), prof[:, 2].min(), prof[:, 2].max(), np.median(a[a[:, 0] < 30][:, 2])))


def section_source():
    import angle_bench as ab
    from roop import recognizer_adaface as rf
    from roop.core import live_swap
    from roop.face_util import solve_pose_jaw_5pt, solve_pose_5pt
    ab.init_pipeline("cuda", "realswap", "None", "None")
    g = __import__("roop.globals", fromlist=["x"])
    options = ab.build_options(g, "realswap", "None", None)
    try:
        rf.download()
    except Exception as exc:                                         # pragma: no cover - network
        print("adaface download:", exc)
    root = fixtures.clip_dir(BANK_DIR, required=True)
    paths = {n: os.path.join(root, n + ".fsz") for n in SWEEP_BANKS}

    banks = {}
    for n in SWEEP_BANKS:
        fs = ab.load_faceset(paths[n])
        raw = [np.asarray(fs.embeddings_backup if i == 0 and fs.embeddings_backup is not None else f.embedding,
                          np.float64) for i, f in enumerate(fs.faces)]
        ada = []
        for img in fs.ref_images:
            f0 = ab.biggest_face(img)
            e = rf.face_embedding(f0, img) if f0 is not None else None
            ada.append(unit(e) if e is not None else None)
        banks[n] = dict(fs=fs, raw=raw, ada=ada, yaws=[yaw_of(f, solve_pose_jaw_5pt, solve_pose_5pt) for f in fs.faces])

    def arms(raw, yaws, ty):
        order = np.argsort(yaws)
        ys = np.asarray(yaws)[order]
        es = [raw[i] for i in order]
        near = unit(raw[int(np.argmin(np.abs(np.asarray(yaws) - ty)))])
        if ty <= ys[0]:
            sl = unit(es[0])
        elif ty >= ys[-1]:
            sl = unit(es[-1])
        else:
            j = int(np.searchsorted(ys, ty))
            sl = slerp(es[j - 1], es[j], (ty - ys[j - 1]) / max(ys[j] - ys[j - 1], 1e-6))
        return {"mean": unit(np.mean([unit(e) for e in raw], axis=0)), "nearest": near, "slerp": sl}

    res = collections.defaultdict(list)
    for src, tgt in itertools.permutations(SWEEP_BANKS, 2):
        S = banks[src]
        fs = S["fs"]
        for plate in ab.plates(paths[tgt]):
            square, bg = ab.prepare_plate(plate)
            if square is None:
                continue
            rolled = ab.roll_frame(square, bg, 0)
            pf = ab.biggest_face(rolled)
            if pf is None:
                continue
            ty = yaw_of(pf, solve_pose_jaw_5pt, solve_pose_5pt)
            yw = np.asarray(S["yaws"])
            h = int(np.argmin(np.abs(yw - ty)))                       # held out: the photo at the target's pose
            if S["ada"][h] is None:
                continue
            keep = [i for i in range(len(yw)) if i != h]
            vecs = arms([S["raw"][i] for i in keep], [S["yaws"][i] for i in keep], ty)
            for arm, v in vecs.items():
                fs.faces[0].embedding = v
                for k in [k for k in list(fs.faces[0].keys()) if str(k).startswith("_latent_")]:
                    del fs.faces[0][k]                                # else every arm swaps with the first one's identity
                out = live_swap(rolled.copy(), options, input_facesets=[fs])
                of = ab.biggest_face(out) if out is not None else None
                e = rf.face_embedding(of, out) if of is not None else None
                if e is not None:
                    res[(abs(round(ty / 45) * 45), arm)].append(float(unit(e) @ S["ada"][h]))
    print("AdaFace cosine to the HELD-OUT pose-matched photo (arms built from the other 4 photos)")
    print("target |yaw|    n    mean  nearest   slerp")
    for yb in (0, 45, 90):
        print("%8d  %4d  %6.3f  %7.3f  %6.3f" % (yb, len(res[(yb, "mean")]), *[np.mean(res[(yb, a)]) for a in ("mean", "nearest", "slerp")]))
    allv = {a: [x for (y, aa), v in res.items() if aa == a for x in v] for a in ("mean", "nearest", "slerp")}
    print("     ALL  %4d  %6.3f  %7.3f  %6.3f" % (len(allv["mean"]), *[np.mean(allv[a]) for a in ("mean", "nearest", "slerp")]))
    for a, b in (("slerp", "mean"), ("nearest", "mean"), ("slerp", "nearest")):
        d = np.array([x - y for x, y in zip(allv[a], allv[b])])
        print("%-8s - %-8s paired: mean %+.4f  sd %.4f  wins %d/%d" % (a, b, d.mean(), d.std(), int((d > 0).sum()), len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=("gates", "source"), required=True)
    args = ap.parse_args()
    {"gates": section_gates, "source": section_source}[args.section]()


if __name__ == "__main__":
    main()
