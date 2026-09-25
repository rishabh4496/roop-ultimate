"""Fit the ArcFace attribute directions shipped in roop/assets/identity_directions.npz.

WHAT IS FITTED. For every face in the corpus, buffalo_l gives the w600k_r50
identity vector (the one the swappers consume) plus labels:

    age         genderage's predicted age (years)
    gender      genderage's predicted sex (1 = male)
    jawline     68-landmark jaw squareness: |p4-p12| / |p0-p16|, near-frontal only
    expression  within-identity mouth opening + smile width (z-scored), |yaw| < 30

``roop.identity_algebra.fit_directions`` regresses each label on the
embeddings (identity-balanced for person-level labels, identity-centred for
expression), orthonormalises in that priority order and re-reads each label's
slope along the orthogonalised vector.  Held-out scores use identity-disjoint
folds.  THE LABELS ARE MODEL PREDICTIONS, not ground truth: the age direction
is "the direction genderage reads as older", which is the calibration the
Age Shift dial promises and no more.

IDENTITIES. A faceset is one identity.  Clip faces are clustered per clip
(greedy, cosine >= --cluster-cos to the running cluster mean) into pseudo-
identities, so frames of one person cannot land on both sides of a fold.

The corpus statistics (faces, identities, age range, sex balance) are stored in
the file's ``meta`` and printed: a direction fitted on a narrow population
(e.g. one age band) is extrapolating at the dial's ends, and the UI shows the
held-out score next to each dial for that reason.

    app/env/Scripts/python.exe tools/fit_identity_directions.py
    app/env/Scripts/python.exe tools/fit_identity_directions.py --frames-per-clip 80 --extra D:/faces
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time
import zipfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(REPO, "app")
for p in (APP, os.path.join(APP, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _analyser(det_size: int):
    """buffalo_l with every module this fit reads.  prepare_environment puts
    the CUDA/TensorRT DLLs on PATH first; the providers ORT actually gave each
    session are printed, because ORT falls back to CPU silently."""
    os.chdir(APP)
    import roop.globals as g
    from settings import Settings
    g.CFG = Settings("config.yaml")
    from ui.main import prepare_environment
    prepare_environment()
    import insightface
    fa = insightface.app.FaceAnalysis(
        name="buffalo_l", root=os.path.join(APP, ".."),
        allowed_modules=["detection", "recognition", "genderage", "landmark_3d_68"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=0.6)
    for name, model in fa.models.items():
        try:
            print(f"  {name}: {model.session.get_providers()[0]}")
        except Exception:
            pass
    return fa


def _labels(face) -> dict:
    out = {"age": np.nan, "gender": np.nan, "jaw": np.nan, "mouth_open": np.nan,
           "smile": np.nan, "yaw": np.nan}
    age = getattr(face, "age", None)
    if age is not None:
        out["age"] = float(age)
    sex = getattr(face, "gender", None)
    if sex is not None:
        out["gender"] = float(int(sex))
    pose = getattr(face, "pose", None)
    yaw = pitch = np.nan
    if pose is not None and len(pose) >= 2:
        pitch, yaw = float(pose[0]), float(pose[1])
        out["yaw"] = yaw
    lm = getattr(face, "landmark_3d_68", None)
    if lm is None:
        return out
    lm = np.asarray(lm, dtype=np.float64)[:, :2]
    d = lambda a, b: float(np.linalg.norm(lm[a] - lm[b]))
    iod = d(36, 45) or 1.0
    if np.isfinite(yaw) and abs(yaw) < 20 and abs(pitch) < 20:
        out["jaw"] = d(4, 12) / (d(0, 16) or 1.0)
    if np.isfinite(yaw) and abs(yaw) < 30:
        out["mouth_open"] = d(62, 66) / iod
        out["smile"] = d(48, 54) / iod
    return out


def _faces_in(fa, image, min_width: int):
    faces = fa.get(image) or []
    keep = []
    for f in faces:
        x0, y0, x1, y1 = [float(v) for v in f.bbox]
        if x1 - x0 < min_width or float(getattr(f, "det_score", 0.0)) < 0.6:
            continue
        if getattr(f, "normed_embedding", None) is None:
            continue
        keep.append(f)
    return keep


def _cluster(embs: list[np.ndarray], threshold: float) -> list[int]:
    """Greedy online clustering by cosine to each cluster's running mean."""
    means: list[np.ndarray] = []
    counts: list[int] = []
    out = []
    for e in embs:
        best, best_cos = -1, threshold
        for i, m in enumerate(means):
            c = float(e @ (m / np.linalg.norm(m)))
            if c >= best_cos:
                best, best_cos = i, c
        if best < 0:
            means.append(e.astype(np.float64).copy())
            counts.append(1)
            out.append(len(means) - 1)
        else:
            means[best] += e
            counts[best] += 1
            out.append(best)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--media", default=None, help="clip root (default: resolved roop-keep)")
    ap.add_argument("--facesets", nargs="*", default=None,
                    help=".fsz folders (default: <media>/faceset-v1-backup-*, <repo>/facesets)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra image folders; each SUBFOLDER is one identity")
    ap.add_argument("--frames-per-clip", type=int, default=60)
    ap.add_argument("--cluster-cos", type=float, default=0.45)
    ap.add_argument("--min-width", type=int, default=72)
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(APP, "roop", "assets", "identity_directions.npz"))
    ap.add_argument("--dump", default=None, help="also save the raw corpus (npz) here")
    ap.add_argument("--from-dump", default=None,
                    help="re-fit from a corpus saved with --dump (no model, no GPU)")
    args = ap.parse_args()
    if args.from_dump:
        d = np.load(args.from_dump, allow_pickle=False)
        labels = {k[len("label_"):]: d[k] for k in d.files if k.startswith("label_")}
        origin = [str(o) for o in d["origin"]]
        clips = sorted({o for o in origin if o.endswith(".mp4")})
        return _fit_and_save(d["Z"], d["groups"], labels, len(clips), args)

    import cv2
    from fixtures import clip_roots

    media = args.media or next((r for r in clip_roots() if os.path.isdir(r)), None)
    fs_dirs = args.facesets
    if fs_dirs is None:
        fs_dirs = sorted(glob.glob(os.path.join(media or "", "faceset-v1-backup-*")))
        fs_dirs.append(os.path.join(REPO, "facesets"))
    print(f"media={media}\nfacesets={fs_dirs}")
    fa = _analyser(args.det_size)

    embs, labs, groups, origin = [], [], [], []
    t0 = time.time()

    def add(face, group, where):
        embs.append(np.asarray(face.normed_embedding, dtype=np.float32))
        labs.append(_labels(face))
        groups.append(group)
        origin.append(where)

    # Facesets: one identity each; duplicate names across folders are one person.
    seen_names = set()
    for folder in fs_dirs:
        for path in sorted(glob.glob(os.path.join(folder, "*.fsz"))):
            name = os.path.splitext(os.path.basename(path))[0].replace("_v2", "")
            key = name.casefold()
            if key in seen_names:
                continue
            seen_names.add(key)
            try:
                zf = zipfile.ZipFile(path)
            except zipfile.BadZipFile:
                continue
            for info in zf.infolist():
                if not info.filename.lower().endswith(IMAGE_EXT):
                    continue
                img = cv2.imdecode(np.frombuffer(zf.read(info), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                # Faceset crops are tight: pad so the detector sees context.
                img = cv2.copyMakeBorder(img, img.shape[0] // 3, img.shape[0] // 3,
                                         img.shape[1] // 3, img.shape[1] // 3, cv2.BORDER_REPLICATE)
                faces = _faces_in(fa, img, 40)
                if faces:
                    add(max(faces, key=lambda f: f.bbox[2] - f.bbox[0]), f"fs:{key}", path)
    print(f"facesets: {len(embs)} faces / {len(seen_names)} identities ({time.time()-t0:.0f}s)")

    for folder in args.extra:
        for sub in sorted(d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d)):
            for path in sorted(glob.glob(os.path.join(sub, "*"))):
                if not path.lower().endswith(IMAGE_EXT):
                    continue
                img = cv2.imread(path)
                faces = _faces_in(fa, img, 40) if img is not None else []
                if faces:
                    add(max(faces, key=lambda f: f.bbox[2] - f.bbox[0]), f"x:{sub}", path)

    clips = []
    if media and os.path.isdir(media):
        for pattern in ("single/*.mp4", "double/*.mp4", "expression/*.mp4", "final/*.mp4", "*.mp4"):
            clips += sorted(glob.glob(os.path.join(media, pattern)))
    for path in clips:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            continue
        idx = np.linspace(0, total - 1, args.frames_per_clip).astype(int)
        clip_faces = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            if max(frame.shape[:2]) > 1920:
                s = 1920.0 / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            clip_faces += _faces_in(fa, frame, args.min_width)
        cap.release()
        ids = _cluster([np.asarray(f.normed_embedding, np.float32) for f in clip_faces], args.cluster_cos)
        base = os.path.relpath(path, media).replace("\\", "/")
        for f, cid in zip(clip_faces, ids):
            add(f, f"clip:{base}#{cid}", base)
        print(f"  {base}: {len(clip_faces)} faces, {len(set(ids))} pseudo-identities "
              f"({time.time()-t0:.0f}s)")

    if len(embs) < 50:
        print("corpus too small to fit anything meaningful")
        return 1
    Z = np.stack(embs)
    G = np.asarray(groups)
    col = lambda k: np.asarray([d[k] for d in labs], dtype=np.float64)
    mouth, smile = col("mouth_open"), col("smile")
    zscore = lambda a: (a - np.nanmean(a)) / (np.nanstd(a) or 1.0)
    labels = {"age": col("age"), "gender": col("gender"), "jawline": col("jaw"),
              "expression": zscore(mouth) + zscore(smile)}
    if args.dump:
        np.savez(args.dump, Z=Z, groups=G, origin=np.asarray(origin),
                 **{f"label_{k}": v for k, v in labels.items()})
    return _fit_and_save(Z, G, labels, len(clips), args)


def _fit_and_save(Z, G, labels, n_clips, args) -> int:
    from roop.identity_algebra import fit_directions
    dirs = fit_directions(Z, labels, G, ridge=args.ridge)

    age, sex = labels["age"], labels["gender"]
    ident_age = {}
    for g, a in zip(G, age):
        if np.isfinite(a):
            ident_age.setdefault(g, []).append(a)
    ident_means = np.asarray([np.mean(v) for v in ident_age.values()])
    ident_sex = {}
    for g, s in zip(G, sex):
        if np.isfinite(s):
            ident_sex.setdefault(g, []).append(s)
    male_ids = sum(1 for v in ident_sex.values() if np.mean(v) >= 0.5)
    faces = int(Z.shape[0])
    dirs.meta.update({
        "embedding": "buffalo_l/w600k_r50 normed_embedding",
        "labels": "genderage age/sex; landmark_3d_68 geometry (model predictions, not ground truth)",
        "corpus": {
            "faces": faces,
            "identities": int(np.unique(G).size),
            "faceset_identities": int(sum(1 for g in np.unique(G) if g.startswith("fs:"))),
            "clips": int(n_clips),
            "lfw_identities": int(sum(1 for g in np.unique(G) if "lfw" in str(g).lower())),
            "identity_age_p5_p50_p95": [float(x) for x in np.percentile(ident_means, [5, 50, 95])],
            "identities_male": int(male_ids),
            "identities_female": int(len(ident_sex) - male_ids),
        },
        "fitted": time.strftime("%Y-%m-%d"),
        "tool": "tools/fit_identity_directions.py",
    })
    dirs.save(args.out)

    print("\n direction | held-out | per unit step | spread (sd) | n / identities")
    for i, name in enumerate(dirs.names):
        fit = dirs.meta["fit"][name]
        print(f" {name:10s}| {fit['metric']} {dirs.heldout[name]:+.3f} | "
              f"{dirs.units_per_step[i]:+9.2f} | {dirs.spread[i]:.4f} | "
              f"{fit['n']} / {fit['identities']}")
    if "age" in dirs.names:
        i = dirs.names.index("age")
        from roop.identity_algebra import max_tangent_norm, DEFAULT_MIN_IDENTITY_COSINE
        reach = max_tangent_norm(DEFAULT_MIN_IDENTITY_COSINE) * abs(dirs.units_per_step[i])
        print(f"\n age reach inside the identity guard (cos >= {DEFAULT_MIN_IDENTITY_COSINE}): "
              f"+/-{reach:.1f} years (linear read of the fit)")
    print(json.dumps(dirs.meta["corpus"], indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
