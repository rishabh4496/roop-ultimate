#!/usr/bin/env python3
"""Stage 3 blend step in isolation: the production CPU paste vs the repo's existing GPU batched paste.

The "VRAM-resident Stage 3" brief moves warp / feather / blend onto the GPU. `AGENTS.md`: a stage is not
closed on a reading, so this prices exactly that step on a real frame instead of arguing about it.

  CPU arm  the REAL `MaskingMixin.paste_upscale` (ellipse matte, landmark hull, `blur_area`, bounded-ROI cubic
           warp, blend) driven face by face into one accumulating destination, as `process_face` does.
           cv2 is pinned to one thread: production runs ~20 workers and each face's blend is one worker's.
  GPU arm  `CudaAffineBatch.paste_faces` (bicubic grid_sample over the FULL frame per face, box matte). Two
           timings: `with_host_boundary` = upload the BGR frame, paste, download BGR (what every current
           consumer needs, see Phase 13), and `vram_resident` = frame already on the device, result stays there.

NOT pixel-equivalent: the GPU arm has no landmark hull, no adaptive feather, no temporal matte EMA and no
overlap trim, so it does less work than the CPU arm and flatters the GPU. This is a ceiling on the step's win,
not a measurement of a port. It says nothing about the render clock; for that see `docs/perf/stage3_gpu_blend/`.

    app/env/Scripts/python.exe app/tests/probe_stage3_blend_gpu_vs_cpu.py [--video NAME_OR_PATH] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import cv2                                                          # noqa: E402
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

cv2.setNumThreads(1)

from fixtures import resolve_target                                 # noqa: E402
from roop.optimized_processor import CudaAffineBatch                # noqa: E402
from roop.procmgr_masking import MaskingMixin                       # noqa: E402

DEFAULT_CLIP = "Love.mp4"
CROP = 512                      # the restorer-sized crop a swapped face is pasted from
FACE_PX = 230.0                 # face width in the frame; ~0.8 of the crop is face


class _Stub(MaskingMixin):
    def __init__(self):
        self.options = types.SimpleNamespace(show_face_area_overlay=False, blend_ratio=1.0)
        self._tls = threading.local()


def face_matrix(i, n, w, h):
    """frame -> crop similarity (the `M` ProcessMgr carries), faces spread across the frame with a little roll."""
    s = CROP * 0.8 / FACE_PX
    cx, cy = w * (i + 1) / (n + 1), h * 0.45
    th = np.deg2rad(6 * (i - n / 2))
    c, sn = np.cos(th) * s, np.sin(th) * s
    lin = np.array([[c, -sn], [sn, c]])
    t = np.array([CROP / 2.0, CROP / 2.0]) - lin @ np.array([cx, cy])
    return np.hstack([lin, t[:, None]]).astype(np.float32), (cx, cy)


def hull_points(cx, cy):
    a = np.linspace(0, 2 * np.pi, 106, endpoint=False)
    return np.stack([cx + FACE_PX * 0.46 * np.cos(a), cy + FACE_PX * 0.55 * np.sin(a)], 1).astype(np.float32)


def timeit(fn, reps, warm=5, cuda=False):
    for _ in range(warm):
        fn()
    if cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", default=None, help="clip name (searched in the fixture roots) or a full path")
    ap.add_argument("--frame", type=int, default=120)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("[PENDING] CUDA not available; nothing to compare", file=sys.stderr)
        return 2
    path = resolve_target(args.video, DEFAULT_CLIP)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"[PENDING] could not read frame {args.frame} of {path}", file=sys.stderr)
        return 2
    h, w = frame.shape[:2]
    dev = torch.device("cuda")
    crop = cv2.resize(frame[h // 8:h // 2, w // 4:w // 2], (CROP, CROP))
    crop_t = torch.from_numpy(crop).permute(2, 0, 1).float().div(255).to(dev)[None]
    frame_dev = torch.from_numpy(frame).to(dev).permute(2, 0, 1).float().div_(255)[None]
    stub = _Stub()

    def cpu_blend(n):
        out = frame.copy()
        for i in range(n):
            m, (cx, cy) = face_matrix(i, n, w, h)
            out = stub.paste_upscale(crop, crop, m, out, 1, [0, 0, 0, 0, 20.0, 10.0],
                                     face_landmarks=hull_points(cx, cy), inplace=True)
        return out

    def gpu_blend(n, resident):
        mats = torch.from_numpy(np.stack([face_matrix(i, n, w, h)[0] for i in range(n)])).to(dev)
        faces = crop_t.expand(n, -1, -1, -1).contiguous()
        src = frame_dev if resident else \
            torch.from_numpy(frame).to(dev).permute(2, 0, 1).float().div_(255)[None]
        comp = CudaAffineBatch.paste_faces(src, faces, [0] * n, mats, masks=None, feather=True)
        if resident:
            return comp
        return comp[0].clamp_(0, 1).mul_(255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

    report = {"clip": os.path.basename(path), "frame_size": [w, h], "gpu": torch.cuda.get_device_name(0),
              "reps": args.reps, "cv2_threads": 1,
              "caveat": "GPU arm omits landmark hull / adaptive feather / temporal EMA / overlap trim; "
                        "ceiling on the step, not a port",
              "faces": {}}
    for n in (1, 2, 4):
        row = {"cpu_paste_upscale": timeit(lambda: cpu_blend(n), args.reps),
               "gpu_with_host_boundary": timeit(lambda: gpu_blend(n, False), args.reps, cuda=True),
               "gpu_vram_resident": timeit(lambda: gpu_blend(n, True), args.reps, cuda=True)}
        report["faces"][str(n)] = row
        print("faces=%d  CPU paste_upscale %.2f ms | GPU with host boundary %.2f ms | GPU VRAM-resident %.2f ms"
              % (n, row["cpu_paste_upscale"]["median_ms"], row["gpu_with_host_boundary"]["median_ms"],
                 row["gpu_vram_resident"]["median_ms"]), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
