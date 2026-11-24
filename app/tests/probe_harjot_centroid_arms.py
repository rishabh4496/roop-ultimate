"""Does a different centroid for the harjot SOURCE faceset swap closer to harjot?

    env/Scripts/python.exe tests/probe_harjot_centroid_arms.py [--source harjot] [--max-targets 12]

Measurement only. Same method as probe_pose_gate_and_source_embedding.py --section source
(leave-one-out, AdaFace scored, swapper latent cache deleted per arm) but the SOURCE is one
bank (harjot) swapped onto plates of OTHER people, and the arms are centroid variants built
from the kept photos:

  mean      production: unit mean of all kept photos' embeddings
  nonprof   mean of kept photos with |yaw| <= 50 (profiles excluded); mean if none
  weighted  mean weighted by cos(e_i, kept mean) ** 4 (outliers such as profiles count less)

For each target plate the photo nearest the target's yaw is HELD OUT of the arms and is the
AdaFace score reference (primary). Secondary: AdaFace cosine to the unit mean of ALL source
photos (pose independent, but the held-out photo leaks into it equally for every arm).
"""
import argparse
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures                                                      # noqa: E402

BANK_DIR = "faceset-v1-backup-2026-09-03"
ARMS = ("mean", "nonprof", "weighted")


def unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def yaw_of(face, jaw_pose, plain_pose):
    k = np.asarray(face.kps, np.float64)
    p = jaw_pose(k) or plain_pose(k)
    return float(p[0])


def build_arms(raw, yaws):
    us = [unit(e) for e in raw]
    mean = unit(np.mean(us, axis=0))
    keep = [u for u, y in zip(us, yaws) if abs(y) <= 50]
    nonprof = unit(np.mean(keep, axis=0)) if keep else mean
    w = np.array([max(float(u @ mean), 0.0) ** 4 for u in us])
    weighted = unit(np.sum([wi * u for wi, u in zip(w, us)], axis=0))
    return {"mean": mean, "nonprof": nonprof, "weighted": weighted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="harjot")
    ap.add_argument("--max-targets", type=int, default=12)
    ap.add_argument("--source-dir", default=os.path.join(APP, "facesets"))
    args = ap.parse_args()

    import angle_bench as ab
    from roop import recognizer_adaface as rf
    from roop.core import live_swap
    from roop.face_util import solve_pose_jaw_5pt, solve_pose_5pt
    ab.init_pipeline("cuda", "realswap", "None", "None")
    g = __import__("roop.globals", fromlist=["x"])
    options = ab.build_options(g, "realswap", "None", None)
    try:
        rf.download()
    except Exception as exc:                                         # pragma: no cover
        print("adaface download:", exc)

    src_path = os.path.join(args.source_dir, args.source + ".fsz")
    fs = ab.load_faceset(src_path)
    raw = [np.asarray(fs.embeddings_backup if i == 0 and fs.embeddings_backup is not None else f.embedding,
                      np.float64) for i, f in enumerate(fs.faces)]
    ada = []
    for img in fs.ref_images:
        f0 = ab.biggest_face(img)
        e = rf.face_embedding(f0, img) if f0 is not None else None
        ada.append(unit(e) if e is not None else None)
    yaws = [yaw_of(f, solve_pose_jaw_5pt, solve_pose_5pt) for f in fs.faces]
    print("source %s: %d photos, yaws %s, adaface ok %s" % (
        args.source, len(raw), [round(y, 1) for y in yaws], [a is not None for a in ada]), flush=True)
    ada_all = unit(np.mean([a for a in ada if a is not None], axis=0))

    root = fixtures.clip_dir(BANK_DIR, required=True)
    names = sorted(n[:-4] for n in os.listdir(root) if n.endswith(".fsz")
                   and not n.startswith(("harjot", "gargee_v2")) and n[:-4] != args.source)
    names = names[:args.max_targets]
    print("targets:", names, flush=True)

    held = collections.defaultdict(list)      # (yaw bin, arm) -> cosine to held-out photo
    allp = collections.defaultdict(list)      # (yaw bin, arm) -> cosine to all-photo mean
    for tn in names:
        for plate in ab.plates(os.path.join(root, tn + ".fsz")):
            square, bg = ab.prepare_plate(plate)
            if square is None:
                continue
            rolled = ab.roll_frame(square, bg, 0)
            pf = ab.biggest_face(rolled)
            if pf is None:
                continue
            ty = yaw_of(pf, solve_pose_jaw_5pt, solve_pose_5pt)
            yw = np.asarray(yaws)
            h = int(np.argmin(np.abs(yw - ty)))
            if ada[h] is None:
                continue
            keep = [i for i in range(len(yw)) if i != h]
            vecs = build_arms([raw[i] for i in keep], [yaws[i] for i in keep])
            yb = abs(round(ty / 45) * 45)
            for arm, v in vecs.items():
                fs.faces[0].embedding = v
                for k in [k for k in list(fs.faces[0].keys()) if str(k).startswith("_latent_")]:
                    del fs.faces[0][k]
                out = live_swap(rolled.copy(), options, input_facesets=[fs])
                of = ab.biggest_face(out) if out is not None else None
                e = rf.face_embedding(of, out) if of is not None else None
                if e is not None:
                    held[(yb, arm)].append(float(unit(e) @ ada[h]))
                    allp[(yb, arm)].append(float(unit(e) @ ada_all))
        print("done", tn, flush=True)

    for title, res in (("PRIMARY: AdaFace cosine to the HELD-OUT pose-matched harjot photo", held),
                       ("SECONDARY: AdaFace cosine to the mean of ALL harjot photos", allp)):
        print("\n" + title)
        print("target |yaw|    n    " + "  ".join("%8s" % a for a in ARMS))
        for yb in (0, 45, 90):
            n = len(res[(yb, "mean")])
            if n:
                print("%8d  %4d  " % (yb, n) + "  ".join("%8.3f" % np.mean(res[(yb, a)]) for a in ARMS))
        allv = {a: [x for (y, aa), v in res.items() if aa == a for x in v] for a in ARMS}
        print("     ALL  %4d  " % len(allv["mean"]) + "  ".join("%8.3f" % np.mean(allv[a]) for a in ARMS))
        for a in ("nonprof", "weighted"):
            d = np.array([x - y for x, y in zip(allv[a], allv["mean"])])
            if len(d):
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
                print("%-9s - mean paired: %+.4f  se %.4f  wins %d/%d" % (a, d.mean(), se, int((d > 0).sum()), len(d)))


if __name__ == "__main__":
    main()
