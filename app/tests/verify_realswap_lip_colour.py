"""Does the lip-colour transfer do what it claims, and ONLY that?

Four checks on real frames rather than synthetic arrays, because three of them
are about what the operator must NOT do:

  1. the lips take hififace's lip-vs-skin colour relationship;
  2. NOTHING outside the lip mask changes -- exactly zero, not approximately;
  3. no structure crosses over: the transferred quantity is three numbers, so
     the high-frequency content of the output must still be hyperswap's;
  4. the change is small and confined -- a mean offset, not a repaint.

NOT asserted, deliberately: that luminance is untouched. That constraint was
imposed on the first build and MEASURED to destroy the effect -- projecting the
luminance out of the offset closed 0.5% of the gap to hififace instead of 93%,
and pushed the LAB-b axis the wrong way. See `_lip_colour`'s docstring. The
transfer moves the lip/skin luminance relationship by ~1.1 LAB-L, which is the
tone half of "richer lips" and is checked for magnitude here rather than
forbidden.

Run: env/Scripts/python.exe tests/verify_realswap_lip_colour.py
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
from roop.ProcessMgr import ProcessMgr
from roop.face_util import get_all_faces, align_crop

FSZ = os.environ.get('FSZ', 'facesets/harjot.fsz')
CLIP = os.environ.get('PROF_CLIP', fixtures.clip('inverted/s1.mp4'))
src_fs = ab.load_faceset(FSZ)
pm = ProcessMgr(None)
pm.initialize([src_fs], [], ab.build_options(g, 'realswap', 'None'))
swap_p = pm.processors[0]
CLS = type(swap_p)
SIZE = int(swap_p.model_output_size)
TMPL = swap_p.model_template
lip, ring, _ = CLS._lip_masks(SIZE, TMPL)
print(f"crop {SIZE} {TMPL}: lip mask {100 * lip.mean():.1f}% of crop, "
      f"perioral ring {100 * ring.mean():.1f}%, strength {CLS._LIP_COLOUR:g}")


def to_bgr8(chw):
    hwc = np.asarray(chw, np.float32).transpose(1, 2, 0)
    return (((hwc + 1.0) / 2.0 * 255.0).clip(0, 255))[:, :, ::-1].astype(np.uint8)


def lipskin(img):
    """(lips - perioral skin) in LAB, for this image alone -- independent of any
    global cast between the two nets."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.array([float((lab[:, :, c] * lip).sum() / lip.sum()
                           - (lab[:, :, c] * ring).sum() / ring.sum())
                     for c in range(3)], np.float32)


cap = cv2.VideoCapture(CLIP)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
rows = []
outside_err, hf_err, inside_max = 0.0, 0.0, 0.0
n = 0
for fno in range(120, min(total, 1700), 40):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ok, frm = cap.read()
    if not ok:
        continue
    faces = get_all_faces(frm)
    if not faces:
        continue
    tf = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    aligned, _M = align_crop(frm, tf.kps, SIZE, TMPL)
    blob = pm.prepare_crop_frame(aligned, swap_p)
    sf = src_fs.faces[0]
    latent = swap_p._compute_source_input(sf)
    if latent is None:
        continue
    outs = swap_p._infer({swap_p.image_input_name: blob,
                          swap_p.embed_input_name: latent})
    swap_p._stash_masks(outs, 1)
    prim = np.asarray(outs[0][0], np.float32)
    sec = swap_p._run_secondary(sf, tf, blob)
    if sec is None:
        continue
    sec = np.asarray(sec, np.float32)
    fixed = CLS._lip_colour(prim, sec, SIZE, TMPL)

    rows.append((lipskin(to_bgr8(prim)), lipskin(to_bgr8(sec)),
                 lipskin(to_bgr8(fixed))))

    # 2. nothing outside the lip mask changes
    outside = (lip <= 0.0)
    outside_err = max(outside_err, float(np.abs(fixed - prim)[:, outside].max()))
    # 4. and the change inside it is a small offset, not a repaint
    inside_max = max(inside_max, float(np.abs(fixed - prim).max()))
    # 3. no structure crossed over
    for ch in range(3):
        hp_a = prim[ch] - cv2.GaussianBlur(prim[ch], (0, 0), 2.0)
        hp_c = fixed[ch] - cv2.GaussianBlur(fixed[ch], (0, 0), 2.0)
        hf_err = max(hf_err, float(np.abs(hp_c - hp_a).max()))
    n += 1
cap.release()

A = np.array([r[0] for r in rows])
B = np.array([r[1] for r in rows])
C = np.array([r[2] for r in rows])
chroma = lambda X: np.hypot(X[:, 1], X[:, 2])

print(f"\n==== lip MINUS that image's own perioral skin, {n} frames ====")
print(f"  {'image':22s} {'dL':>8s} {'dLAB-a':>8s} {'dLAB-b':>8s} {'lip/skin':>10s}")
for name, X in (('hyperswap (was)', A), ('hififace (the target)', B),
                ('realswap (now)', C)):
    print(f"  {name:22s} {X[:, 0].mean():8.2f} {X[:, 1].mean():8.2f} "
          f"{X[:, 2].mean():8.2f} {chroma(X).mean():10.2f}")

gap0 = np.abs(chroma(B) - chroma(A))
gap1 = np.abs(chroma(B) - chroma(C))
closed = 100 * (1 - gap1.mean() / max(1e-9, gap0.mean()))
print(f"\n  distance to hififace's lip/skin chroma: {gap0.mean():.3f} -> "
      f"{gap1.mean():.3f}  ({closed:.1f}% closed)")
print(f"  closer on {100.0 * (gap1 < gap0).mean():.1f}% of frames")

print(f"\n==== and ONLY that ({SIZE} crop, values in the models' [-1,1]) ====")
print(f"  max change OUTSIDE the lip mask       {outside_err:.2e}   "
      f"(must be exactly 0)")
print(f"  max high-frequency change (sigma 2)   {hf_err:.2e}   "
      f"(= {hf_err * 127.5:.4f}/255)")
print(f"  max change anywhere                   {inside_max:.2e}   "
      f"(= {inside_max * 127.5:.2f}/255)")

checks = [
    ('lips move toward hififace', closed > 75.0),
    ('on most frames', (gap1 < gap0).mean() > 0.75),
    ('nothing outside the mask moves', outside_err == 0.0),
    ('no visible edge from the mask', hf_err * 127.5 < 0.5),
    # A tint, not a repaint. 20/255 is ~8% of full range: above that the
    # lips would be being recoloured rather than nudged. Observed on this
    # material: 12.5/255 at the mask's peak, 0 at its edge.
    ('the change is a tint, not a repaint', inside_max * 127.5 < 20.0),
]
for name, ok in checks:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}")
print(f"\n  {'PASS' if all(c[1] for c in checks) else 'FAIL'}")
sys.exit(0 if all(c[1] for c in checks) else 1)
