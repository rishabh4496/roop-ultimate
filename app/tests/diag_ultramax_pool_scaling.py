"""Enhancer throughput against the number of TensorRT contexts.

The live ROOP_PROFILE reads enhance = 61.94 ms/call while UltraMax's own Run()
measures 33.5 ms on an idle GPU. The gap is queueing: `_gpu_guard` lets the
stage run lock-free only because UltraMax owns a SessionPool, but that pool is
`ROOP_TRT_POOL` deep — 2 on a 12GB card — and the render dispatches 10 worker
threads at it. This measures faces/sec against pool depth, with VRAM, so the
choice is made on the curve rather than on the reasoning.
"""
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import fixtures
POOL = os.environ.get('POOL', '2')
os.environ['ROOP_TRT_POOL'] = POOL
THREADS = int(os.environ.get('THREADS', '10'))

import numpy as np
import cv2
import angle_bench as ab

g = ab.init_pipeline('tensorrt', 'realswap', 'UltraMax', 'mask_realityux', 0.0)
g.codeformer_fidelity = float(g.CFG.codeformer_fidelity)

from roop.processors.Enhance_UltraMax import Enhance_UltraMax


def vram_mb():
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


base = vram_mb()
um = Enhance_UltraMax()
t0 = time.perf_counter()
um.Initialize({'devicename': 'cuda'})
build = time.perf_counter() - t0
n_ctx = len(um.pool._items) if um.pool is not None else 1

CLIP = os.environ.get('PROF_CLIP', fixtures.clip('inverted/s1.mp4'))
cap = cv2.VideoCapture(CLIP)
crops = []
for f in range(200, 200 + THREADS * 60, 60):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frm = cap.read()
    h, w = frm.shape[:2]
    crops.append(cv2.resize(frm[h // 6:h * 5 // 6, w // 4:w * 3 // 4], (256, 256)))
cap.release()
for _ in range(4):
    for c in crops:
        um.Run(None, None, c)
after = vram_mb()

PER = 24
done = [0] * THREADS
barrier = threading.Barrier(THREADS + 1)


def worker(i):
    c = crops[i]
    barrier.wait()
    for _ in range(PER):
        um.Run(None, None, c)
        done[i] += 1


ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(THREADS)]
for t in ths:
    t.start()
barrier.wait()
t0 = time.perf_counter()
for t in ths:
    t.join()
el = time.perf_counter() - t0
n = sum(done)
print(f"POOL={POOL} contexts={n_ctx} threads={THREADS} build={build:.1f}s "
      f"VRAM {base} -> {after} MB (+{after - base}) | "
      f"{n} faces in {el:.2f}s = {n / el:6.1f} faces/s, "
      f"{el / n * 1000:6.2f} ms/call wall")
