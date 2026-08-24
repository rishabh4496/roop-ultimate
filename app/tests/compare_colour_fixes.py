"""Side-by-side of the two 2026-08-24 colour fixes, through the REAL pipeline.

Renders the same frames twice with `live_swap` -- once with both fixes turned
off via their env knobs (the old output) and once with the shipped defaults --
and writes a panel plus face zooms so the difference can be judged on footage
rather than on a metric.

    old = ROOP_ULTRAMAX_CHROMA=1  (CodeFormer's own, pale, chrominance)
          ROOP_REALSWAP_LIP_COLOUR=0  (hyperswap's lips, untouched)
    new = the defaults

Both arms run the production stack read from config.yaml, and both are rendered
in ONE process so nothing about model state or engine builds differs between
them.

    env/Scripts/python.exe tests/compare_colour_fixes.py [--clip X] [--fsz Y]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('--clip', default=r'G:/pinokio/roop-keep/inverted/s1.mp4')
ap.add_argument('--fsz', default='facesets/harjot.fsz')
ap.add_argument('--frames', type=int, default=4)
ap.add_argument('--enhancer', default='UltraMax',
                help="default UltraMax -- the processor the chroma fix is in. "
                     "config.yaml's own value is used for everything else.")
ap.add_argument('--out', default='output/colour_fixes')
args = ap.parse_args()

import angle_bench as ab

# The stack the user actually renders, read live -- not this tool's defaults.
import roop.globals as _g0
from settings import Settings

_cfg = Settings('config.yaml')
# get_processing_plugins keys mask engines by their INTERNAL name; config.yaml
# and the UI use the display name. Same translation api.py's map_mask_engine
# does for the real /api/swap path -- without it initialize() raises KeyError
# on the display name, which is how this tool failed first.
_MASK_MAP = {"Clip2Seg": "mask_clip2seg", "DFL XSeg": "mask_xseg",
             "Face Parser (BiSeNet)": "mask_faceparser",
             "RealityUX": "mask_realityux", "Face Occluder": "mask_occluder",
             "Face Occluder v3 (XSeg-3)": "mask_xseg3",
             "Segment Anything (MobileSAM)": "mask_mobilesam",
             "Segment Anything (FastSAM)": "mask_fastsam",
             "Segment Anything 2 (tracked)": "mask_sam2"}
_mask = _MASK_MAP.get(_cfg.mask_engine, _cfg.mask_engine)
g = ab.init_pipeline(_cfg.provider, _cfg.swap_model, args.enhancer,
                     _mask, float(_cfg.swap_model_mask_strength))
g.codeformer_fidelity = float(_cfg.codeformer_fidelity)
print(f"stack: swap={_cfg.swap_model} enhancer={args.enhancer} "
      f"mask={_mask} provider={_cfg.provider} "
      f"blend={_cfg.blend_ratio} clarity={_cfg.merger_clarity}")
if args.enhancer != _cfg.selected_enhancer:
    print(f"  NOTE config.yaml currently selects '{_cfg.selected_enhancer}'; "
          f"this comparison forces '{args.enhancer}'.")

from roop.core import live_swap
from roop.face_util import get_all_faces

src_fs = ab.load_faceset(args.fsz)
g.INPUT_FACESETS = [src_fs]
g.TARGET_FACES = []
options = ab.build_options(g, _cfg.swap_model, _mask)

cap = cv2.VideoCapture(args.clip)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
picks, step = [], max(1, total // (args.frames + 2))
for i in range(1, args.frames + 1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
    ok, frm = cap.read()
    if ok and get_all_faces(frm):
        picks.append((i * step, frm))
cap.release()
print(f"{len(picks)} frames from {os.path.basename(args.clip)}")


def render(frames, chroma, lipc, label):
    os.environ['ROOP_ULTRAMAX_CHROMA'] = chroma
    os.environ['ROOP_REALSWAP_LIP_COLOUR'] = lipc
    # Both are read per call, but the swap model caches its constant at import,
    # so the class attribute is the one that decides and it is set here too.
    from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace as SW
    SW._LIP_COLOUR = float(lipc)
    out = []
    for fno, f in frames:
        out.append(live_swap(f.copy(), options))
    print(f"  rendered {label}")
    return out


old = render(picks, '1', '0.0', 'OLD (CodeFormer chroma, hyperswap lips)')
new = render(picks, '0', '1.0', 'NEW (crop chroma, hififace lip colour)')


def face_box(frame, pad=0.55):
    faces = get_all_faces(frame)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    x0, y0, x1, y1 = [int(v) for v in f.bbox]
    w, h = x1 - x0, y1 - y0
    cx, cy = x0 + w // 2, y0 + h // 2
    r = int(max(w, h) * (0.5 + pad))
    H, W = frame.shape[:2]
    return (max(0, cx - r), max(0, cy - r), min(W, cx + r), min(H, cy + r))


def mouth_box(frame):
    faces = get_all_faces(frame)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    kps = np.asarray(f.kps, np.float32)
    c = (kps[3] + kps[4]) / 2.0
    mw = float(np.linalg.norm(kps[4] - kps[3])) or 30.0
    r = int(mw * 1.3)
    H, W = frame.shape[:2]
    return (max(0, int(c[0]) - r), max(0, int(c[1]) - r),
            min(W, int(c[0]) + r), min(H, int(c[1]) + r))


def tile(img, box, size, label):
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        crop = img
    h, w = crop.shape[:2]
    s = size / max(h, w)
    crop = cv2.resize(crop, (int(w * s), int(h * s)),
                      interpolation=cv2.INTER_LANCZOS4)
    pad = np.zeros((size, size, 3), np.uint8)
    yo, xo = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    pad[yo:yo + crop.shape[0], xo:xo + crop.shape[1]] = crop
    cv2.rectangle(pad, (0, size - 26), (size, size), (0, 0, 0), -1)
    cv2.putText(pad, label, (7, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    return pad


os.makedirs(os.path.join(APP, args.out), exist_ok=True)
S = 380
rows = []
for i, ((fno, plate), a, b) in enumerate(zip(picks, old, new)):
    fb = face_box(plate) or (0, 0, plate.shape[1], plate.shape[0])
    mb = mouth_box(plate) or fb
    row = [tile(plate, fb, S, f"f{fno} ORIGINAL"),
           tile(a, fb, S, "OLD"),
           tile(b, fb, S, "NEW"),
           tile(a, mb, S, "OLD lips"),
           tile(b, mb, S, "NEW lips")]
    rows.append(np.hstack(row))
panel = np.vstack(rows)
p1 = os.path.join(APP, args.out, 'colour_fixes_panel.png')
cv2.imwrite(p1, panel)
print(f"\nwrote {p1}  ({panel.shape[1]}x{panel.shape[0]})")


# ── the numbers behind it, on the RENDERED frames ────────────────────────────
def face_stats(img, plate):
    """Skin chroma/saturation inside the face, and lip-vs-skin chroma."""
    faces = get_all_faces(plate)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    kps = np.asarray(f.kps, np.float32)
    eye = float(np.linalg.norm(kps[1] - kps[0])) or 30.0
    H, W = img.shape[:2]
    cheeks = np.zeros((H, W), np.uint8)
    for pt in (kps[0] * 0.35 + kps[3] * 0.65, kps[1] * 0.35 + kps[4] * 0.65):
        cv2.circle(cheeks, (int(pt[0]), int(pt[1])), int(eye * 0.22), 255, -1)
    mc = (kps[3] + kps[4]) / 2.0
    mw = float(np.linalg.norm(kps[4] - kps[3])) or 30.0
    lips = np.zeros((H, W), np.uint8)
    cv2.ellipse(lips, (int(mc[0]), int(mc[1])),
                (int(mw * 0.55), int(mw * 0.30)), 0, 0, 360, 255, -1)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    sk = cheeks > 0
    lp = lips > 0
    if sk.sum() < 20 or lp.sum() < 20:
        return None
    return dict(
        skin_sat=float(sat[sk].mean()),
        skin_a=float(lab[:, :, 1][sk].mean()),
        lip_minus_skin=float(np.hypot(lab[:, :, 1][lp].mean() - lab[:, :, 1][sk].mean(),
                                      lab[:, :, 2][lp].mean() - lab[:, :, 2][sk].mean())),
    )


acc = {'ORIGINAL': [], 'OLD': [], 'NEW': []}
for (fno, plate), a, b in zip(picks, old, new):
    for name, img in (('ORIGINAL', plate), ('OLD', a), ('NEW', b)):
        st = face_stats(img, plate)
        if st:
            acc[name].append(st)
print(f"\n==== on the rendered frames (cheeks from the plate's landmarks) ====")
print(f"  {'':10s} {'skin sat':>10s} {'skin LAB-a':>11s} {'lip-vs-skin':>12s}")
for name in ('ORIGINAL', 'OLD', 'NEW'):
    r = acc[name]
    if not r:
        continue
    print(f"  {name:10s} {np.mean([x['skin_sat'] for x in r]):10.2f} "
          f"{np.mean([x['skin_a'] for x in r]):11.2f} "
          f"{np.mean([x['lip_minus_skin'] for x in r]):12.2f}")
print("\n  ORIGINAL is the footage itself -- the reference both arms are trying "
      "to sit near.")
