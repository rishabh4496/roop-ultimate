"""Where UltraMax's per-face time goes, and how far its COLOUR drifts.

Two questions in one process, because both need the pipeline's real init
(TensorRT DLLs on PATH; without it a 4 ms model reports as 210 ms).

1. COST BREAKDOWN. The lean host path is documented as ~28.7 ms/face against
   CodeFormer's 35.2. The live ROOP_PROFILE reads 61.94 ms/call for the enhance
   stage. This splits Run() into resize / pre / infer / post / guards so the
   difference between "the network" and "everything else" is visible.

2. COLOUR. GPEN Realistic and GPEN 256 Pro both keep the SWAPPER's chrominance
   and take only luminance from the restorer. UltraMax takes CodeFormer's
   output whole. The user reports UltraMax's skin as pale. Measured here as:
     - LAB chroma drift (mean |da|, |db|) against the crop the restorer was handed
     - mean saturation (HSV S) and its ratio
     - mean L, because "pale" can be either desaturation or a lift in luminance
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('ROOP_TRT_POOL', '2')

import numpy as np
import cv2
import angle_bench as ab

g = ab.init_pipeline('tensorrt', 'realswap', 'UltraMax', 'mask_realityux', 0.0)
g.codeformer_fidelity = float(g.CFG.codeformer_fidelity)
print(f"fidelity {g.codeformer_fidelity}")

from roop.processors.Enhance_UltraMax import Enhance_UltraMax
from roop.processors.Enhance_GPENRealistic import Enhance_GPENRealistic
from roop.processors.Enhance_GPEN256Pro import Enhance_GPEN256Pro

CLIP = os.environ.get('PROF_CLIP', r'G:/pinokio/roop-keep/inverted/s1.mp4')
cap = cv2.VideoCapture(CLIP)
crops = []
for f in (200, 400, 700, 1000, 1400):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frm = cap.read()
    if ok:
        h, w = frm.shape[:2]
        crops.append(cv2.resize(frm[h // 6:h * 5 // 6, w // 4:w * 3 // 4], (256, 256)))
cap.release()
print(f"{len(crops)} crops from {CLIP}")

um = Enhance_UltraMax(); um.Initialize({'devicename': 'cuda', 'pool_size': 1})
gr = Enhance_GPENRealistic(); gr.Initialize({'devicename': 'cuda', 'pool_size': 1})
gp = Enhance_GPEN256Pro(); gp.Initialize({'devicename': 'cuda', 'pool_size': 1})

crop = crops[0]
for _ in range(8):
    um.Run(None, None, crop); gr.Run(None, None, crop); gp.Run(None, None, crop)

# ── 1. cost breakdown, by re-running the pieces of Run() ────────────────────
S = 512
N = 60
def t(fn, n=N):
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
    return (time.perf_counter() - t0) / n * 1000, r

ms_resize, src512 = t(lambda: cv2.resize(crop, (S, S), interpolation=cv2.INTER_CUBIC))
ms_pre, x = t(lambda: um._lut[src512.transpose(2, 0, 1)[::-1]][None])
w_fid = np.array([g.codeformer_fidelity], dtype=np.float64)
in0, in1 = um.model_inputs[0].name, um.model_inputs[1].name
def _inf():
    um.io_binding.bind_cpu_input(in0, x)
    um.io_binding.bind_cpu_input(in1, w_fid)
    um.session.run_with_iobinding(um.io_binding)
    return um.io_binding.copy_outputs_to_cpu()
ms_infer, outs = t(_inf)
ms_post, hwc = t(lambda: np.ascontiguousarray(outs[0][0][::-1].transpose(1, 2, 0), dtype=np.float32))
ms_fin, _ = t(lambda: (np.isfinite(hwc.sum()), cv2.convertScaleAbs(np.maximum(hwc, -1.0), alpha=127.5, beta=127.5)))
restored = cv2.convertScaleAbs(np.maximum(hwc, -1.0), alpha=127.5, beta=127.5)
from roop.processors.enhance_common import looks_collapsed
ms_coll, _ = t(lambda: looks_collapsed(restored, src512))
ms_sz, _ = t(lambda: cv2.resize(restored, (256, 256), interpolation=cv2.INTER_AREA))
ms_full, _ = t(lambda: um.Run(None, None, crop), 40)

print("\n==== UltraMax per-face cost, 256 crop in (ms) ====")
for name, v in (('resize 256->512 (CUBIC)', ms_resize), ('pre  LUT gather', ms_pre),
                ('INFER (network)', ms_infer), ('post CHW->HWC f32', ms_post),
                ('finite + convertScaleAbs', ms_fin), ('looks_collapsed', ms_coll),
                ('sized() back to 256', ms_sz)):
    print(f"  {name:28s} {v:7.3f}   {100.0 * v / ms_full:5.1f}%")
print(f"  {'-- Run() total':28s} {ms_full:7.3f}")

# ── 2. colour ───────────────────────────────────────────────────────────────
def colour(out, src):
    """out/src both uint8 BGR, same size. Face-centre window so background,
    which no restorer touches, cannot dilute the numbers."""
    h, w = src.shape[:2]
    y0, y1, x0, x1 = int(h * .25), int(h * .85), int(w * .25), int(w * .75)
    o = cv2.cvtColor(out[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    s = cv2.cvtColor(src[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    so = cv2.cvtColor(out[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    ss = cv2.cvtColor(src[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    return dict(
        drift=float(np.mean(np.abs(o[:, :, 1] - s[:, :, 1])) + np.mean(np.abs(o[:, :, 2] - s[:, :, 2]))) / 2.0,
        da=float(o[:, :, 1].mean() - s[:, :, 1].mean()),
        db=float(o[:, :, 2].mean() - s[:, :, 2].mean()),
        dL=float(o[:, :, 0].mean() - s[:, :, 0].mean()),
        sat=float(so.mean()), sat_src=float(ss.mean()),
    )

print("\n==== colour against the crop the restorer was handed ====")
print(f"  {'enhancer':18s} {'chroma drift':>12s} {'dLAB-a':>8s} {'dLAB-b':>8s} "
      f"{'dL':>7s} {'sat':>7s} {'sat src':>8s} {'sat ratio':>10s}")
for name, p in (('UltraMax', um), ('GPEN Realistic', gr), ('GPEN 256 Pro', gp)):
    acc = []
    for c in crops:
        out, sf = p.Run(None, None, c)
        if sf != 1:
            out = cv2.resize(out, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_AREA)
        acc.append(colour(out, c))
    m = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
    print(f"  {name:18s} {m['drift']:12.2f} {m['da']:8.2f} {m['db']:8.2f} "
          f"{m['dL']:7.2f} {m['sat']:7.2f} {m['sat_src']:8.2f} "
          f"{m['sat'] / max(1e-6, m['sat_src']):10.3f}")
