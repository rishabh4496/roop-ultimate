"""Per-face cost of UltraMax against the `Codeformer (fp16)` it runs inside.

INTERLEAVED, because end-to-end render times on this machine vary by ~18%
run to run — larger than the effect being measured. Two full renders of s1
gave 1.13x and 1.30x for the same pair of builds. Alternating the arms within
one process, several rounds, is what makes the number mean something.

Measured through angle_bench.init_pipeline so TensorRT's DLLs are on PATH and
ORT does not silently fall back to CPU — a 4 ms model reports as 210 ms
otherwise (see the bench-the-model-the-user-runs rule).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import fixtures
os.environ.setdefault('ROOP_TRT_POOL', '2')

import numpy as np
import cv2
import angle_bench as ab

g = ab.init_pipeline('tensorrt', 'realswap', 'Codeformer (fp16)', 'mask_realityux', 0.0)
g.codeformer_fidelity = float(g.CFG.codeformer_fidelity)

from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer
from roop.processors.Enhance_UltraMax import Enhance_UltraMax

CLIP = os.environ.get('PROF_CLIP', fixtures.clip('inverted/s1.mp4'))
cap = cv2.VideoCapture(CLIP)
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
ok, frm = cap.read()
cap.release()
crop = cv2.resize(frm[200:700, 100:600], (256, 256))

cf = Enhance_CodeFormer()
cf.Initialize({'devicename': 'cuda', 'fp16': True, 'pool_size': 1})
um = Enhance_UltraMax()
um.Initialize({'devicename': 'cuda', 'pool_size': 1})

N, ROUNDS = 40, 5
for _ in range(8):
    cf.Run(None, None, crop)
    um.Run(None, None, crop)


def timed(proc):
    t0 = time.perf_counter()
    for _ in range(N):
        out, _ = proc.Run(None, None, crop)
    return (time.perf_counter() - t0) / N * 1000, out


cf_ms, um_ms = [], []
for r in range(ROUNDS):
    a, cf_out = timed(cf)
    b, um_out = timed(um)
    cf_ms.append(a)
    um_ms.append(b)
    print(f"  round {r + 1}: codeformer {a:6.2f} ms   ultramax {b:6.2f} ms   "
          f"{a / b:.3f}x")

cf_a, um_a = np.array(cf_ms), np.array(um_ms)
print()
print(f"  Codeformer (fp16)  {cf_a.mean():6.2f} +- {cf_a.std():.2f} ms/face")
print(f"  UltraMax           {um_a.mean():6.2f} +- {um_a.std():.2f} ms/face")
print(f"  speedup            {cf_a.mean() / um_a.mean():.3f}x  "
      f"({(1 - um_a.mean() / cf_a.mean()) * 100:+.1f}% wall), "
      f"per-round min {min(cf_ms[i] / um_ms[i] for i in range(ROUNDS)):.3f}x "
      f"max {max(cf_ms[i] / um_ms[i] for i in range(ROUNDS)):.3f}x")

# With the texture restore off AND CodeFormer's own chrominance kept, the two
# must be BIT-identical: same weights, same fidelity, and a host path that only
# reorders arithmetic. If this ever prints a nonzero max, the lean pre/post
# changed the picture and every claim above it is about a different image.
#
# ROOP_ULTRAMAX_CHROMA=1 is what makes that comparison still meaningful. Since
# 2026-08-24 UltraMax DEFAULTS to keeping the swapper's chrominance instead of
# CodeFormer's, which is a deliberate difference — CodeFormer's own colour is
# what the user reported as pale skin — so bit-identity is now a property of the
# HOST PATH only, and this asserts exactly that and nothing more.
os.environ['ROOP_ULTRAMAX_TEXTURE'] = '0'
os.environ['ROOP_ULTRAMAX_CHROMA'] = '1'
bare_ms, bare = timed(um)
os.environ.pop('ROOP_ULTRAMAX_TEXTURE')
os.environ.pop('ROOP_ULTRAMAX_CHROMA')
d = np.abs(bare.astype(np.int16) - cf_out.astype(np.int16))
print()
print(f"  lean host path vs the reference implementation: mean |diff| "
      f"{d.mean():.4f}/255, max {d.max()}")
print(f"  texture restore costs {um_a.mean() - bare_ms:.2f} ms/face "
      f"(UltraMax without it: {bare_ms:.2f} ms)")

um.Release()
cf.Release()
