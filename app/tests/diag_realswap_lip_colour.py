"""How far apart are hyperswap's and hififace's LIPS, and is it only colour?

RealSwap's base is hyperswap; the user reports hififace's lip colour as the
better one and wants that colour alone, nothing else. Before building a
transfer, measure the population it would read:

  - chroma difference on the LIPS (what the transfer is for)
  - chroma difference on the SKIN AROUND the mouth (what it must not disturb)
  - LUMINANCE difference on the lips (if this is large the two nets disagree
    about lip STRUCTURE, and a colour-only transfer is the wrong instrument)

Both nets emit [3,H,W] float32 RGB in [-1,1] (procmgr_tiling.prepare_crop_frame:
BGR->RGB, mean 0.5, std 0.5, denormalize True for both), and `_run_secondary`
has already resampled hififace into hyperswap's own crop space -- so the two
arrays are directly comparable pixel for pixel.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import fixtures
os.environ.setdefault('ROOP_TRT_POOL', '1')

import numpy as np
import cv2
import angle_bench as ab

g = ab.init_pipeline('tensorrt', 'realswap', 'None', 'None', 0.0)
import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.face_util import get_all_faces, swap_template_points

FSZ = os.environ.get('FSZ', 'facesets/harjot.fsz')
CLIP = os.environ.get('PROF_CLIP', fixtures.clip('inverted/s1.mp4'))
src_fs = ab.load_faceset(FSZ)
opts = ab.build_options(g, 'realswap', 'None')
pm = ProcessMgr(None)
pm.initialize([src_fs], [], opts)
swap_p = pm.processors[0]
print(f"primary={swap_p.loaded_model_key} secondary="
      f"{getattr(swap_p.secondary, 'loaded_model_key', None)}")

# ── the two regions, in crop space, from the template ───────────────────────
SIZE = int(swap_p.model_output_size)
pts = np.asarray(swap_template_points(SIZE, swap_p.model_template), np.float32)
mL, mR = pts[3], pts[4]
mw = float(np.linalg.norm(mR - mL))
cx, cy = (mL + mR) / 2.0
print(f"crop {SIZE}px  mouth corners {mL} {mR}  width {mw:.1f}px")

lips = np.zeros((SIZE, SIZE), np.uint8)
cv2.ellipse(lips, (int(round(cx)), int(round(cy))),
            (int(round(0.62 * mw)), int(round(0.42 * mw))), 0, 0, 360, 255, -1)
ring = np.zeros((SIZE, SIZE), np.uint8)
cv2.ellipse(ring, (int(round(cx)), int(round(cy))),
            (int(round(1.15 * mw)), int(round(0.85 * mw))), 0, 0, 360, 255, -1)
ring = cv2.subtract(ring, lips)
print(f"lip region {lips.mean() / 255 * 100:.1f}% of crop, "
      f"perioral ring {ring.mean() / 255 * 100:.1f}%")


def to_bgr8(chw):
    """[3,H,W] RGB [-1,1] -> HWC BGR uint8, the way normalize_swap_frame does."""
    hwc = np.asarray(chw, np.float32).transpose(1, 2, 0)
    hwc = ((hwc + 1.0) / 2.0 * 255.0).clip(0, 255)
    return hwc[:, :, ::-1].astype(np.uint8)


def lab_stats(img, m):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    sel = m > 0
    return (float(lab[:, :, 0][sel].mean()), float(lab[:, :, 1][sel].mean()),
            float(lab[:, :, 2][sel].mean()))


cap = cv2.VideoCapture(CLIP)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
rows = []
for fno in range(120, min(total, 1700), 40):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ok, frm = cap.read()
    if not ok:
        continue
    faces = get_all_faces(frm)
    if not faces:
        continue
    tf = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    from roop.face_util import align_crop
    aligned, M = align_crop(frm, tf.kps, SIZE, swap_p.model_template)
    blob = pm.prepare_crop_frame(aligned, swap_p)
    sf = src_fs.faces[0]
    latent = swap_p._compute_source_input(sf)
    if latent is None:
        continue
    outs = swap_p._infer({swap_p.image_input_name: blob,
                          swap_p.embed_input_name: latent})
    swap_p._stash_masks(outs, 1)
    prim = outs[0][0]
    sec = swap_p._run_secondary(sf, tf, blob)
    if sec is None:
        continue
    a, b = to_bgr8(prim), to_bgr8(sec)
    plate = to_bgr8(blob[0] if blob.ndim == 4 else blob)
    for name, img in (('hyperswap', a), ('hififace', b), ('target plate', plate)):
        Ll, Al, Bl = lab_stats(img, lips)
        Ls, As, Bs = lab_stats(img, ring)
        # LIP MINUS THAT NET'S OWN SKIN: independent of any global cast, so it
        # isolates "how much do the lips stand out from the face" -- which is
        # what reads as pale-vs-rosy lips and what a transfer must move.
        rows.append((name, Ll - Ls, Al - As, Bl - Bs))
cap.release()

print(f"\n==== LIP minus that image's OWN perioral skin, {len(rows) // 3} frames "
      f"({os.path.basename(CLIP)}, {os.path.basename(FSZ)}) ====")
print("  Independent of any global cast between the nets, so this is how far the "
      "lips sit from")
print("  the face around them -- which is what reads as pale vs rosy lips.")
print(f"  {'image':14s} {'dL':>10s} {'dLAB-a':>10s} {'dLAB-b':>10s} {'lip/skin':>11s}")
for name in ('target plate', 'hyperswap', 'hififace'):
    r = np.array([x[1:] for x in rows if x[0] == name], np.float32)
    if not len(r):
        continue
    print(f"  {name:14s} {r[:, 0].mean():10.2f} {r[:, 1].mean():10.2f} "
          f"{r[:, 2].mean():10.2f} {np.hypot(r[:, 1], r[:, 2]).mean():11.2f}")
    print(f"  {'':14s} {'(sd %.2f)' % r[:, 0].std():>10s} "
          f"{'(sd %.2f)' % r[:, 1].std():>10s} {'(sd %.2f)' % r[:, 2].std():>10s}")

ha = np.array([x[1:] for x in rows if x[0] == 'hyperswap'], np.float32)
hf = np.array([x[1:] for x in rows if x[0] == 'hififace'], np.float32)
n = min(len(ha), len(hf))
if n:
    d = np.hypot(hf[:n, 1], hf[:n, 2]) - np.hypot(ha[:n, 1], ha[:n, 2])
    print(f"\n  hififace lip/skin chroma contrast is higher on "
          f"{100.0 * (d > 0).mean():.1f}% of {n} frames, mean {d.mean():+.3f}")
