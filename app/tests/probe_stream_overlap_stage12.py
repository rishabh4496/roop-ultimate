"""Can XSeg and the swap overlap on the 4070, and is there anything left to overlap? Measured with the real models.

    env/Scripts/python.exe tests/probe_stream_overlap_stage12.py [--seconds 6] [--rounds 3]

Stage 12 asked for a fork-join inside one frame: XSeg on a second CUDA stream while the swapper (and enhancer)
run on the first. Production already runs the same two independent branches concurrently ACROSS faces
(20 worker threads, each leasing its own TensorRT context from the session pools), so the question this answers
is whether the device has slack beyond that.

  serial     one thread, swap.Run then xseg.Run, repeated:                     pairs/s
  fork-join  per pair, swap on one thread and xseg on another, join:           pairs/s   (the brief's DAG)
  free       one thread loops swap, one loops xseg, for N seconds:             each op's rate vs its solo rate
             overlap efficiency = swap_free/swap_solo + xseg_free/xseg_solo
             (1.0 = pure time-slicing, 2.0 = fully free overlap)
  scale      T threads, each doing swap then xseg per pair:                    pairs/s for T = 1, 2, 3, 4, 6

  scale-fj   T frame workers, each forking swap and XSeg onto a private second thread (2T ops in flight)

Arms are interleaved over `--rounds` rounds (the first round pays engine builds and clock ramp and is reported
but not averaged). Both models run through the app's own init, so they are TensorRT where the live config says so.

Result of 2026-10-19 (RTX 4070, realswap = hyperswap_1a_256 + hififace TRT, DFL XSeg TRT fp32, 6 s arms, rounds 2-3 of 3):

  solo     swap 78.0/s  xseg 289.5/s        serial pair 62.3/s
  fork-join one pair at a time              72.2/s  (+15.9% vs serial)   overlap efficiency 1.46 (of 2.0)
  frame workers   T=1   2     3     4     6
  plain           61.2  82.9  96.1  100.5 105.9   pairs/s
  each forking    67.7  85.0  99.0  102.7  -      pairs/s   (+2.5% at T=2, +3.0% at T=3, +2.2% at T=4)

The device DOES overlap the two models, so a fork-join helps the one-owner path (T=1). It adds nothing measurable once
there are 2+ frames in flight, which is how production runs (6.2 in flight in the 600-frame regression render);
the plateau is ~100-106 pairs/s however the work is split. Round-to-round spread is ~3%.
"""
import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures                                   # noqa: E402
import angle_bench as ab                          # noqa: E402


def build():
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "DFL XSeg", sync_config=True)
    from roop.procmgr_tiling import PixelBoostMixin
    from roop.face_util import get_all_faces, align_crop
    from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace
    from roop.processors.Mask_XSeg import Mask_XSeg
    from roop.utilities import get_device
    swap = FaceSwapInsightFace()
    swap.Initialize({"devicename": get_device(), "swap_model": cfg.swap_model})
    xseg = Mask_XSeg()
    xseg.Initialize({"devicename": "cuda"})
    # Timing only: any real detected face with an embedding is a valid swap source (no faceset ships with the repo).
    def first_face(rel, limit=300, step=10):
        c = cv2.VideoCapture(fixtures.clip(rel))
        try:
            for i in range(limit):
                ok, fr = c.read()
                if not ok:
                    break
                if i % step == 0:
                    fs = get_all_faces(fr) or []
                    if fs:
                        return fr, fs[0]
        finally:
            c.release()
        raise AssertionError("no face in the first %d frames of %s" % (limit, rel))

    _, source = first_face("Love.mp4")
    frame, faces_ = first_face("single/s1.mp4")
    faces = [faces_]
    target = faces[0]
    crop, _ = align_crop(frame, target.kps, swap.model_output_size, swap.model_template)
    blob = PixelBoostMixin.prepare_crop_frame(None, crop, swap)
    xcrop, _ = align_crop(frame, np.asarray(target.kps, np.float32), 512, mode="arcface")
    print("[probe] swap_model=%s provider=%s xseg_pool=%s" % (cfg.swap_model, cfg.provider,
                                                              getattr(xseg, "pool", None) is not None), flush=True)
    do_swap = lambda: swap.Run(source, target, blob.copy())       # noqa: E731
    do_xseg = lambda: xseg.Run(xcrop.copy(), "")                  # noqa: E731
    return do_swap, do_xseg


def timed_loop(fn, seconds):
    n, end = 0, time.perf_counter() + seconds
    t0 = time.perf_counter()
    while time.perf_counter() < end:
        fn()
        n += 1
    return n / (time.perf_counter() - t0)


def arm_solo(do_swap, do_xseg, seconds):
    return timed_loop(do_swap, seconds), timed_loop(do_xseg, seconds)


def arm_serial(do_swap, do_xseg, seconds):
    def pair():
        do_swap()
        do_xseg()
    return timed_loop(pair, seconds)


def arm_forkjoin(do_swap, do_xseg, seconds):
    with ThreadPoolExecutor(max_workers=2) as ex:
        def pair():
            a = ex.submit(do_swap)
            b = ex.submit(do_xseg)
            a.result()
            b.result()
        return timed_loop(pair, seconds)


def arm_free(do_swap, do_xseg, seconds):
    out = {}

    def run(name, fn):
        out[name] = timed_loop(fn, seconds)
    ts = [threading.Thread(target=run, args=("swap", do_swap)), threading.Thread(target=run, args=("xseg", do_xseg))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return out["swap"], out["xseg"]


def arm_scale(do_swap, do_xseg, seconds, threads):
    counts = [0] * threads
    stop = time.perf_counter() + seconds

    def work(i):
        while time.perf_counter() < stop:
            do_swap()
            do_xseg()
            counts[i] += 1
    t0 = time.perf_counter()
    ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return sum(counts) / (time.perf_counter() - t0)


def arm_scale_forkjoin(do_swap, do_xseg, seconds, threads):
    """`threads` frame workers, each forking its swap and XSeg onto a private second thread (2x ops in flight)."""
    counts = [0] * threads
    stop = time.perf_counter() + seconds

    def work(i):
        with ThreadPoolExecutor(max_workers=2) as ex:
            while time.perf_counter() < stop:
                a = ex.submit(do_swap)
                b = ex.submit(do_xseg)
                a.result()
                b.result()
                counts[i] += 1
    t0 = time.perf_counter()
    ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return sum(counts) / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    do_swap, do_xseg = build()
    for _ in range(20):                       # warm-up: engine build, clock ramp
        do_swap()
        do_xseg()
    rows = []
    for r in range(args.rounds):
        s_solo, x_solo = arm_solo(do_swap, do_xseg, args.seconds)
        serial = arm_serial(do_swap, do_xseg, args.seconds)
        fork = arm_forkjoin(do_swap, do_xseg, args.seconds)
        s_free, x_free = arm_free(do_swap, do_xseg, args.seconds)
        scale = {t: arm_scale(do_swap, do_xseg, args.seconds, t) for t in (1, 2, 3, 4, 6)}
        scale_fj = {t: arm_scale_forkjoin(do_swap, do_xseg, args.seconds, t) for t in (1, 2, 3, 4)}
        rows.append((s_solo, x_solo, serial, fork, s_free, x_free, scale, scale_fj))
        print("[round %d] solo swap %.1f/s xseg %.1f/s | serial %.1f pairs/s | fork-join %.1f pairs/s | free swap %.1f xseg %.1f"
              " | efficiency %.2f | scale %s | scale-forkjoin %s" % (
                  r + 1, s_solo, x_solo, serial, fork, s_free, x_free, s_free / s_solo + x_free / x_solo,
                  " ".join("%d:%.1f" % kv for kv in scale.items()),
                  " ".join("%d:%.1f" % kv for kv in scale_fj.items())), flush=True)
    use = rows[1:] if len(rows) > 1 else rows
    m = lambda i: sum(r[i] for r in use) / len(use)               # noqa: E731
    print("\nmean of rounds %s (round 1 excluded when >1):" % ("2.." + str(len(rows)) if len(rows) > 1 else "1"))
    print("  solo     swap %.1f/s   xseg %.1f/s   (serial pair ceiling %.1f/s)" % (m(0), m(1), 1.0 / (1.0 / m(0) + 1.0 / m(1))))
    print("  serial   %.1f pairs/s" % m(2))
    print("  fork-join %.1f pairs/s  = %+.1f%% vs serial" % (m(3), 100.0 * (m(3) / m(2) - 1.0)))
    print("  free     swap %.1f/s (%.2fx solo)  xseg %.1f/s (%.2fx solo)  efficiency %.2f" % (
        m(4), m(4) / m(0), m(5), m(5) / m(1), m(4) / m(0) + m(5) / m(1)))
    for t in (1, 2, 3, 4, 6):
        print("  scale T=%d  %.1f pairs/s" % (t, sum(r[6][t] for r in use) / len(use)))
    for t in (1, 2, 3, 4):
        print("  scale-forkjoin T=%d (%d ops in flight)  %.1f pairs/s" % (t, 2 * t, sum(r[7][t] for r in use) / len(use)))


if __name__ == "__main__":
    main()
