"""Does the detector keep a face that the frame edge cuts, or that is rolled? Measured on real frames.

    env/Scripts/python.exe tests/probe_edge_roll_recall.py --section edges
    env/Scripts/python.exe tests/probe_edge_roll_recall.py --section replay

Measurement only: it never edits the code under test. A real detection of a large, clean face is
ground truth; the frame is then cut or rolled so the same face is partly outside the picture or
sideways, and the pipeline is asked for it again. Ground truth moves with the transform.

  edges   40% of the face beyond each of the four frame edges. Reports production recall, how many
          of its misses returned NOTHING (the only case a rescue ladder runs on), whether a re-run
          alone recovers any (detector noise control), what a 25% padded-frame detect recovers with
          black / reflect borders, and its false-positive rate on frames production finds no face in.
  roll    the same face rolled +90 / -90 / 180, single frame: full-frame, the ROI path as it is, and
          the ROI path stood upright with the true roll (an upper bound).
  replay  the real `TemporalFaceTracker` driven over 150 consecutive frames rolled +90 / -90 / 180,
          derotation off vs on (counterbalanced). The hint comes from the tracker's own history.

Results of 2026-10-19 (RTX 4070, retinaface_r50, Love / Weeds / Monica Bellucci / b1, 76 faces per cell):

  edges   production 271/304 (89%) recall at 40% clipped; 33 misses, 28 of them empty.
          re-run alone recovers 0/33. 25% padded detect recovers black 11, reflect 14 (13 of the 28
          empty ones) -- but fires on 9/78 (black) and 13/78 (reflect) frames that have NO face.
          Gated to "the face is >=10% beyond the frame AND score >= 0.8" it is 6 of 13 true
          recoveries at 0/78 false positives. NOT shipped: ~2 points of clipped faces for one more full
          detect on every empty frame, and 0/78 only bounds the false-positive rate below ~4%.
  replay  recall 89.5 -> 96.1 (-90), 91.4 -> 95.9 (+90), 82.7 -> 92.0 (180). SHIPPED (ROI_DEROTATE).
"""
import argparse
import collections
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures                                                      # noqa: E402

CLIPS = [("Love.mp4", 1200), ("Weeds.mp4", 900), ("Monica Bellucci .mp4", 3300), ("b1.mp4", 2000)]
ROT = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, -90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180}


def biggest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), default=None)


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def hit(faces, gt):
    best = max(faces, key=lambda f: iou(f.bbox, gt), default=None)
    return best is not None and iou(best.bbox, gt) >= 0.3


def clip_variants(frame, f, frac=0.4):
    """Sub-windows of `frame` whose edge cuts `frac` of the face off. -> {edge: (img, gt_bbox)}"""
    x1, y1, x2, y2 = [float(v) for v in f.bbox]
    bw, bh = x2 - x1, y2 - y1
    H, W = frame.shape[:2]
    X0, X1 = int(max(0, x1 - 1.5 * bw)), int(min(W, x2 + 1.5 * bw))
    Y0, Y1 = int(max(0, y1 - 1.5 * bh)), int(min(H, y2 + 1.5 * bh))
    wins = {"left": (int(x1 + frac * bw), Y0, X1, Y1), "right": (X0, Y0, int(x2 - frac * bw), Y1),
            "top": (X0, int(y1 + frac * bh), X1, Y1), "bottom": (X0, Y0, X1, int(y2 - frac * bh))}
    out = {}
    for k, (a, b, c, d) in wins.items():
        if c - a >= 64 and d - b >= 64:
            out[k] = (frame[b:d, a:c].copy(), np.array([x1 - a, y1 - b, x2 - a, y2 - b], np.float32))
    return out


def rot_box(b, W, H, a):
    pts = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], np.float32)
    if a == 180:
        q = np.stack([W - 1 - pts[:, 0], H - 1 - pts[:, 1]], 1)
    elif a == 90:
        q = np.stack([pts[:, 1], W - 1 - pts[:, 0]], 1)
    else:
        q = np.stack([H - 1 - pts[:, 1], pts[:, 0]], 1)
    return np.array([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()], np.float32)


def samples(n_per_clip, stride, fu):
    """(clip, frame, biggest face, all-faces-empty flag) for clear, large faces; plus faceless frames."""
    good, faceless = [], []
    for name, _ in CLIPS:
        if not fixtures.available(name):
            print("[probe] clip %r not available here, skipped" % name)
            continue
        cap, idx, n = cv2.VideoCapture(fixtures.clip(name)), 0, 0
        while idx < 12000 and n < n_per_clip:
            ok, fr = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            faces = fu.get_all_faces(fr) or []
            if not faces:
                if len(faceless) < 120:
                    faceless.append(fr)
                continue
            f = biggest(faces)
            if f.det_score >= 0.8 and (f.bbox[3] - f.bbox[1]) >= 160:
                good.append((name, fr, f))
                n += 1
    return good, faceless


def padded_dets(fu, img, mode, frac=0.25):
    p = int(max(img.shape[:2]) * frac)
    border = {"black": dict(borderType=cv2.BORDER_CONSTANT, value=0),
              "reflect": dict(borderType=cv2.BORDER_REFLECT_101)}[mode]
    out = cv2.copyMakeBorder(img, p, p, p, p, **border)
    H, W = img.shape[:2]
    res = []
    for f in fu.get_all_faces(out) or []:
        b = np.asarray(f.bbox, np.float32) - p
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if 0 <= cx < W and 0 <= cy < H:                       # drop mirror ghosts centred outside
            over = max(0.0, -b[0], -b[1], b[2] - W, b[3] - H) / max(b[2] - b[0], b[3] - b[1], 1.0)
            res.append((float(f.det_score), float(over), b))
    return res


def section_edges(fu, n):
    good, faceless = samples(n, 100, fu)
    R = collections.defaultdict(collections.Counter)
    tp = []
    for _, fr, f in good:
        for k, (img, gt) in clip_variants(fr, f).items():
            r = R[k]
            r["n"] += 1
            first = fu.get_all_faces(img) or []
            if hit(first, gt):
                r["prod"] += 1
                continue
            r["miss"] += 1
            r["empty"] += (not first)
            r["rerun"] += hit(fu.get_all_faces(img) or [], gt)
            for mode in ("black", "reflect"):
                r[mode] += any(iou(b, gt) >= 0.3 for _, _, b in padded_dets(fu, img, mode))
            if not first:
                tp += [(s, o) for s, o, b in padded_dets(fu, img, "reflect") if iou(b, gt) >= 0.3]
    print("edge    n  prod miss empty | recovered of the misses: rerun black25 reflect25")
    tot = collections.Counter()
    for k, r in sorted(R.items()):
        tot.update(r)
        print("%-7s %3d %4d %4d %5d | %24d %6d %9d" % (k, r["n"], r["prod"], r["miss"], r["empty"], r["rerun"], r["black"], r["reflect"]))
    print("TOTAL   %3d %4d %4d %5d | %24d %6d %9d" % (tot["n"], tot["prod"], tot["miss"], tot["empty"], tot["rerun"], tot["black"], tot["reflect"]))
    fp = collections.Counter()
    fp_det = []
    for fr in faceless:
        fp["frames"] += 1
        fp["rerun"] += bool(fu.get_all_faces(fr))
        for mode in ("black", "reflect"):
            d = padded_dets(fu, fr, mode)
            fp[mode] += bool(d)
            if mode == "reflect":
                fp_det += [(s, o) for s, o, _ in d]
    print("FACELESS frames %d: spurious detections  rerun %d  black25 %d  reflect25 %d"
          % (fp["frames"], fp["rerun"], fp["black"], fp["reflect"]))
    print("gate (face >= over of its size beyond the frame, score >= s): true recoveries %d, spurious frames' dets %d"
          % (len(tp), len(fp_det)))
    for over in (0.0, 0.10):
        for smin in (0.5, 0.8):
            print("  over>=%.2f score>=%.1f : TP %2d / %d   FP dets %2d on %d faceless frames" % (
                over, smin, sum(1 for s, o in tp if o >= over and s >= smin), len(tp),
                sum(1 for s, o in fp_det if o >= over and s >= smin), fp["frames"]))


def section_roll(fu, n):
    from roop.face_util import (_roi_window, _unrotate_face_coords, _offset_face_coords,
                                rotate_clockwise, rotate_anticlockwise, rotate_image_180)
    turn = {90: ("clockwise", rotate_clockwise), -90: ("anticlockwise", rotate_anticlockwise),
            180: ("180", rotate_image_180)}
    good, _ = samples(n, 100, fu)
    res = collections.defaultdict(collections.Counter)
    for _, fr, f in good:
        H, W = fr.shape[:2]
        for a in (90, -90, 180):
            img = cv2.rotate(fr, ROT[a])
            gt = rot_box(f.bbox, W, H, a)
            r = res[a]
            r["n"] += 1
            r["full"] += hit(fu.get_all_faces(img) or [], gt)
            r["roi_now"] += hit(fu.get_all_faces_in_roi(img, gt.tolist(), 1.0, 160) or [], gt)
            win = _roi_window(img, gt.tolist(), 1.0, 160)
            if win:
                crop, cx1, cy1 = win
                name, fn = turn[a]
                ch, cw = crop.shape[:2]
                faces = fu.get_all_faces(fn(crop)) or []
                for ff in faces:
                    _unrotate_face_coords(ff, cw, ch, name)
                    _offset_face_coords(ff, cx1, cy1)
                r["roi_known_roll"] += hit(faces, gt)
    print("roll     n | full  roi_now  roi_known_roll   (hits)")
    for a, r in sorted(res.items()):
        print("%+4d %5d | %4d %8d %15d" % (a, r["n"], r["full"], r["roi_now"], r["roi_known_roll"]))


def section_replay(fu, n):
    import time
    from roop import temporal_tracker as tt
    from roop.temporal_tracker import TemporalFaceTracker

    def run(frames, gts, derotate):
        tt.ROI_DEROTATE = derotate
        tr, hits, modes, hinted, t0 = TemporalFaceTracker.from_env(), 0, collections.Counter(), 0, time.time()
        for i, fr in enumerate(frames):
            plan = tr.plan(i, fr.shape)
            if plan.mode == "coast":
                faces, mode = [], "coast"
            elif plan.mode == "roi":
                hint = {"rotation_action": plan.rotation_action} if plan.rotation_action else {}
                hinted += bool(hint)
                faces, mode = fu.get_all_faces_in_roi(fr, plan.roi, **hint) or [], "roi"
                if not faces:
                    faces, mode = fu.get_all_faces(fr) or [], "roi_fallback_full"
            else:
                faces, mode = fu.get_all_faces(fr) or [], "full"
            modes[mode] += 1
            tr.update(faces, i, fr.shape, detection_mode=mode)
            hits += bool(gts[i] is not None and faces and hit(faces, gts[i]))
        return hits, modes, hinted, time.time() - t0

    out = collections.defaultdict(collections.Counter)
    for name, start in CLIPS:
        if not fixtures.available(name):
            continue
        cap = cv2.VideoCapture(fixtures.clip(name))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = [fr for fr in (cap.read()[1] for _ in range(n)) if fr is not None]
        base = [biggest(fu.get_all_faces(fr) or []) for fr in frames]
        for a in (90, -90, 180):
            rf = [cv2.rotate(fr, ROT[a]) for fr in frames]
            gts = [rot_box(b.bbox, fr.shape[1], fr.shape[0], a) if (b is not None and b.det_score >= 0.7) else None
                   for b, fr in zip(base, frames)]
            for arm in (False, True, False, True):               # counterbalanced
                h, modes, hinted, secs = run(rf, gts, arm)
                o = out[(a, arm)]
                o["gt"] += sum(g is not None for g in gts)
                o["hit"] += h
                o["fallback"] += modes["roi_fallback_full"]
                o["hinted"] += hinted
                o["secs"] += secs
                o["runs"] += 1
    print("roll  derotate | frames-with-GT  hit  recall | roi_fallback_full | hinted ROIs | s/run")
    for (a, arm), o in sorted(out.items()):
        print("%+4d  %-8s | %5d %5d %6.1f%% | %5d | %5d | %.1f" % (
            a, arm, o["gt"], o["hit"], 100.0 * o["hit"] / max(1, o["gt"]), o["fallback"], o["hinted"], o["secs"] / o["runs"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=("edges", "roll", "replay"), required=True)
    ap.add_argument("--faces-per-clip", type=int, default=30)
    ap.add_argument("--frames", type=int, default=150, help="replay length")
    args = ap.parse_args()
    import angle_bench as ab
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "None", sync_config=True)
    from roop import face_util as fu
    print("[probe] detector=%s" % fu.roop.globals.detector_engine)
    {"edges": lambda: section_edges(fu, args.faces_per_clip),
     "roll": lambda: section_roll(fu, args.faces_per_clip),
     "replay": lambda: section_replay(fu, args.frames)}[args.section]()


if __name__ == "__main__":
    main()
