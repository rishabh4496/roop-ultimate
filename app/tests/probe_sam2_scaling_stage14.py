"""Does the production SAM 2 pre-pass slow down, or grow VRAM, as the clip gets longer? What grows with it?

    env/Scripts/python.exe tests/probe_sam2_scaling_stage14.py [--frames 600] [--clip double/d4.mp4] [--size tiny]

Stage 14 assumed an unbounded temporal memory bank ("quadratic attention slowdown and VRAM creep"). This drives the app's own
`Mask_SAM2.precompute` (the SAM2FrameBuffer + `propagate_in_video` that `ProcessMgr._precompute_sam2` runs) over N real frames and
records, per window of frames: ms/frame, process RSS, CUDA allocated.

The predictor is built from the config with NO checkpoint (`ckpt_path=None`, random weights): compute, memory and the attention window do
not depend on the weights, so the scaling answer needs no download. The masks it produces are meaningless and are not looked at.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

KEEP = os.environ.get("ROOP_MEDIA_DIR") or os.path.normpath(os.path.join(APP, "..", "..", "..", "roop-keep"))
CFG = {"tiny": "configs/sam2.1/sam2.1_hiera_t.yaml", "small": "configs/sam2.1/sam2.1_hiera_s.yaml"}


def rss_gb():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--clip", default="double/d4.mp4")
    ap.add_argument("--size", default="tiny", choices=sorted(CFG))
    ap.add_argument("--window", type=int, default=100)
    args = ap.parse_args()

    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from roop.processors.Mask_SAM2 import Mask_SAM2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = Mask_SAM2()
    m.predictor = build_sam2_video_predictor(CFG[args.size], ckpt_path=None, device=dev)
    m.device = dev
    print("[probe] predictor=%s image_size=%d num_maskmem=%s device=%s" % (
        args.size, m.predictor.image_size, getattr(m.predictor, "num_maskmem", "?"), dev), flush=True)

    path = os.path.join(KEEP, args.clip)
    cap = cv2.VideoCapture(path)
    buf = m.new_frame_buffer(args.frames)
    t0 = time.perf_counter()
    first = None
    while buf.count < args.frames:
        ok, fr = cap.read()
        if not ok:
            break
        first = fr if first is None else first
        buf.add(fr)
    cap.release()
    h, w = first.shape[:2]
    print("[probe] buffered %d frames of %dx%d in %.1fs | RSS %.2f GB | %.1f MB/frame of SAM2 input (%.2f GB total)" % (
        buf.count, w, h, time.perf_counter() - t0, rss_gb(),
        buf._images.element_size() * buf._images[0].numel() / 1e6, buf._images.element_size() * buf._images.numel() / 1e9), flush=True)

    marks = []
    orig = m.predictor.propagate_in_video

    def timed(*a, **k):
        last = time.perf_counter()
        n = 0
        for item in orig(*a, **k):
            n += 1
            if n % args.window == 0:
                now = time.perf_counter()
                marks.append((n, (now - last) * 1000.0 / args.window, rss_gb(),
                              torch.cuda.memory_allocated() / 1e9 if dev == "cuda" else 0.0,
                              torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0))
                last = now
            yield item
    m.predictor.propagate_in_video = timed

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    box = [w * 0.3, h * 0.2, w * 0.7, h * 0.8]
    t1 = time.perf_counter()
    m.precompute(buf, [box], (h, w))
    total = time.perf_counter() - t1
    print("\nframes   ms/frame   RSS GB   CUDA alloc GB   CUDA peak GB")
    for n, ms, r, a, p in marks:
        print("%6d   %8.1f   %6.2f   %13.2f   %12.2f" % (n, ms, r, a, p))
    print("total %.1fs for %d frames = %.1f ms/frame; RSS after precompute %.2f GB" % (total, buf.count, 1000 * total / buf.count, rss_gb()))
    if len(marks) >= 2:
        first_w, last_w = marks[0][1], marks[-1][1]
        print("last window / first window ms per frame = %.2f" % (last_w / first_w))


if __name__ == "__main__":
    main()
