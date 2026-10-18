"""Does the swapped face's skin tone need a mask-isolated, temporally smoothed LAB transfer?

    env/Scripts/python.exe tests/probe_skin_tone_transfer.py --section static
    env/Scripts/python.exe tests/probe_skin_tone_transfer.py --section temporal

Measurement only: it never edits the code under test. For real single-face frames, a source
person from a real bank is swapped on by the production pipeline (realswap) three ways --
colour transfer off, the production `lct`, and a CANDIDATE built here offline from the
transfer-off output: Reinhard LAB mean/std taken only over BiSeNet skin+nose pixels (eroded)
of the swapped crop and of the plate, chroma std ratio bounded 0.8..1.2 as `rct` does. The
number is the CIE76 distance between the mean skin colour of the output and of the plate, over
the plate's skin pixels in the aligned 256 px crop -- what a viewer sees at the jaw/neck seam.

  static    per face: how far is the swapped skin from the plate's, and does the error follow the
            share of the crop that is hair / background / glasses / cloth / hat (the claim that
            whole-crop statistics are skewed by non-skin pixels)?
  temporal  70 consecutive frames per clip: how much does that error move frame to frame ("colour
            pumping"), with the candidate per frame and with its coefficients EMA-smoothed
            (alpha 0.15, per tracked face).

Results of 2026-10-19 (RTX 4070, realswap, Love / Weeds / Monica Bellucci / b1; JND ~ 2.3 dE):

  static    91 faces, skin-mean dE vs plate:        median   p90   > 2.3   > 4   max
            transfer off                              1.30   2.11     7      0   2.96
            production lct                            1.80   2.42    12      0   2.91
            skin-masked candidate                     1.06   2.14     8      2   6.93
            corr(lct error, non-skin share of crop) = -0.00; no face is left above 3 dE by lct.
  temporal  91 frames, mean |frame-to-frame change| of that error (dE): off 0.27, lct 0.20,
            skin-masked per frame 0.68, skin-masked + EMA 0.44. Mean |error|: 1.21 / 1.73 / 1.50 / 1.51.
  Verdict   the swapper already lands within the JND of the plate's skin tone; the whole-crop
            statistics are not skewed (r = -0.00); per-frame segmentation-driven statistics PUMP more
            than the whole-crop ones, and EMA does not bring them back. NOT shipped.
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

CLIPS = [("Love.mp4", 600), ("Weeds.mp4", 300), ("Monica Bellucci .mp4", 1500), ("b1.mp4", 800)]
BANKS = ["sadaf", "anshita", "mahima"]
BANK_DIR = "faceset-v1-backup-2026-09-03"
ALPHA = 0.15


def lab(img):
    return (cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) * np.array([100 / 255.0, 1, 1], np.float32)
            - np.array([0, 128, 128], np.float32))


def apply_stats(S, sm, ss, tm, ts):
    k = ts / ss
    k[1:] = np.clip(k[1:], 0.8, 1.2)
    k[0] = np.clip(k[0], 0.7, 1.4)
    out = (lab(S) - sm) * k + tm + np.array([0, 128, 128], np.float32)
    out = out * np.array([255 / 100.0, 1, 1], np.float32)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def stats(img, mask):
    v = lab(img)[mask]
    return v.mean(0), np.maximum(v.std(0), 1.0)


class Rig:
    def __init__(self):
        import angle_bench as ab
        from roop.core import live_swap
        from roop.face_util import align_crop, get_all_faces
        from roop.processors.Mask_FaceParser import Mask_FaceParser
        from roop.utilities import get_device
        import roop.globals as rg
        self.ab, self.live_swap, self.align_crop, self.get_all_faces, self.rg = ab, live_swap, align_crop, get_all_faces, rg
        ab.init_pipeline("cuda", "realswap", "None", "None")
        self.options = ab.build_options(rg, "realswap", "None", None)
        self.parser = Mask_FaceParser()
        self.parser.Initialize({"devicename": get_device()})
        root = fixtures.clip_dir(BANK_DIR, required=True)
        self.fsets = {b: ab.load_faceset(os.path.join(root, b + ".fsz")) for b in BANKS}

    def labels(self, crop):
        return cv2.resize(self.parser.RunLabels(crop).astype(np.uint8), (crop.shape[1], crop.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def skin(labels):
        return cv2.erode(np.isin(labels, [1, 10]).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0

    def swap(self, frame, fs, mode):
        self.rg.color_transfer_mode = mode
        return self.live_swap(frame.copy(), self.options, input_facesets=[fs])

    def one_face(self, frame):
        faces = self.get_all_faces(frame) or []
        if len(faces) == 1 and faces[0].det_score >= 0.7 and (faces[0].bbox[3] - faces[0].bbox[1]) >= 120:
            return faces[0]
        return None


def section_static(rig, per_clip):
    rows = []
    for ci, (clip, start) in enumerate(CLIPS):
        if not fixtures.available(clip):
            print("[probe] clip %r not available here, skipped" % clip)
            continue
        cap = cv2.VideoCapture(fixtures.clip(clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        taken = 0
        for i in range(3000):
            ok, fr = cap.read()
            if not ok or taken >= per_clip:
                break
            if i % 25:
                continue
            f = rig.one_face(fr)
            if f is None:
                continue
            plate, M = rig.align_crop(fr, f.kps, 256)
            pl = rig.labels(plate)
            t_mask = rig.skin(pl)
            if t_mask.sum() < 400:
                continue
            fs = rig.fsets[BANKS[taken % len(BANKS)]]
            outs = {m: rig.swap(fr, fs, m) for m in ("none", "lct")}
            if any(o is None for o in outs.values()):
                continue
            crops = {m: cv2.warpAffine(o, M, (256, 256), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                     for m, o in outs.items()}
            S = crops["none"]
            s_mask = rig.skin(rig.labels(S))
            if s_mask.sum() < 400:
                continue
            tm, ts = stats(plate, t_mask)
            sm, ss = stats(S, s_mask)
            crops["skin-masked"] = apply_stats(S, sm, ss, tm, ts)
            ref = lab(plate)[t_mask].mean(0)
            e = {m: float(np.linalg.norm(lab(c)[t_mask].mean(0) - ref)) for m, c in crops.items()}
            rows.append((e["none"], e["lct"], e["skin-masked"], float(np.isin(pl, [0, 6, 16, 17, 18]).mean())))
            taken += 1
    a = np.array(rows)
    print("faces", len(a))
    print("arm          | dE76 skin-mean vs plate: median  mean   p75   p90   max | > 2.3 (JND)   > 4")
    for j, name in enumerate(("none", "lct", "skin-masked")):
        v = a[:, j]
        print("%-12s | %24.2f %5.2f %5.2f %5.2f %5.2f | %9d %8d" % (
            name, np.median(v), v.mean(), np.percentile(v, 75), np.percentile(v, 90), v.max(), int((v > 2.3).sum()), int((v > 4).sum())))
    print("corr(lct error, non-skin share of crop) = %.2f ; faces left > 3 dE by lct: %d"
          % (np.corrcoef(a[:, 1], a[:, 3])[0, 1], int((a[:, 1] > 3).sum())))


def section_temporal(rig, frames):
    series = collections.defaultdict(lambda: collections.defaultdict(list))
    for clip, start in CLIPS:
        if not fixtures.available(clip):
            continue
        cap = cv2.VideoCapture(fixtures.clip(clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ema, fs = None, rig.fsets[BANKS[0]]
        for _ in range(frames):
            ok, fr = cap.read()
            if not ok:
                break
            f = rig.one_face(fr)
            if f is None:
                ema = None
                continue
            plate, M = rig.align_crop(fr, f.kps, 256)
            t_mask = rig.skin(rig.labels(plate))
            outs = {m: rig.swap(fr, fs, m) for m in ("none", "lct")}
            if t_mask.sum() < 400 or any(o is None for o in outs.values()):
                ema = None
                continue
            crops = {m: cv2.warpAffine(o, M, (256, 256), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                     for m, o in outs.items()}
            S = crops["none"]
            s_mask = rig.skin(rig.labels(S))
            if s_mask.sum() < 400:
                ema = None
                continue
            sm, ss = stats(S, s_mask)
            tm, ts = stats(plate, t_mask)
            cur = np.concatenate([sm, ss, tm, ts])
            ema = cur if ema is None else (1 - ALPHA) * ema + ALPHA * cur
            arms = {"none": S, "lct": crops["lct"], "skin-masked": apply_stats(S, sm, ss, tm, ts),
                    "skin-masked+EMA": apply_stats(S, ema[0:3], ema[3:6], ema[6:9], ema[9:12])}
            ref = lab(plate)[t_mask].mean(0)
            for name, c in arms.items():
                series[clip][name].append(lab(c)[t_mask].mean(0) - ref)
    agg = collections.defaultdict(lambda: [[], [], 0])
    for d in series.values():
        for name, v in d.items():
            v = np.array(v)
            if len(v) >= 6:
                agg[name][0].append(np.linalg.norm(np.diff(v, axis=0), axis=1).mean())
                agg[name][1].append(np.linalg.norm(v, axis=1).mean())
                agg[name][2] += len(v)
    print("arm              | mean |frame-to-frame change of the skin-tone error| (dE) | mean |error| | frames")
    for name in ("none", "lct", "skin-masked", "skin-masked+EMA"):
        a = agg[name]
        print("%-16s | %56.3f | %13.3f | %d" % (name, np.mean(a[0]), np.mean(a[1]), a[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=("static", "temporal"), required=True)
    ap.add_argument("--faces-per-clip", type=int, default=24)
    ap.add_argument("--frames", type=int, default=70)
    args = ap.parse_args()
    rig = Rig()
    section_static(rig, args.faces_per_clip) if args.section == "static" else section_temporal(rig, args.frames)


if __name__ == "__main__":
    main()
