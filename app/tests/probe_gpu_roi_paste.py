"""What does the ROI sub-tensor GPU paste cost next to the production CPU paste, at 720p / 1080p / 4K?

    app\\env\\Scripts\\python.exe app\\tests\\probe_gpu_roi_paste.py [--faces 1,2,4] [--out report.json]

Three arms per resolution, one 230 px face per slot spread across the frame (so the face is the same size at every
resolution and only the FRAME grows):

  cpu        the REAL ``MaskingMixin.paste_upscale`` (canvas matte, cv2 on one thread: production runs ~20 workers and one
             face's paste is one worker's)
  gpu full   ``CudaAffineBatch.paste_faces``      grid + grid_sample over the WHOLE frame per face, ``frames.clone()``
  gpu roi    ``CudaAffineBatch.paste_faces_roi``  grid + grid_sample over the face's box only

GPU arms are timed twice: ``host`` = upload the BGR frame, paste, download BGR (what every current consumer needs, Phase 13)
and ``vram`` = the frame already on the device and the result left there.

NOT pixel-equivalent to the CPU arm (no landmark hull, adaptive feather, temporal matte EMA, overlap trim, rim
illumination): the CPU arm does MORE, so this flatters the GPU. It prices the step; it says nothing about the render clock,
and in a GPU-bound pipeline (AGENTS.md) moving CPU work onto the GPU adds GPU work. See docs/perf/gpu_roi_paste/.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import cv2                                                          # noqa: E402
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

cv2.setNumThreads(1)
os.environ.setdefault("ROOP_GRAIN_MATCH", "0")

from roop.optimized_processor import CudaAffineBatch                # noqa: E402
from roop.procmgr_masking import MaskingMixin                       # noqa: E402

CROP, FACE = 512, 230.0
RESOLUTIONS = ((1280, 720), (1920, 1080), (3840, 2160))


class _Stub(MaskingMixin):
    def __init__(self):
        self.options = types.SimpleNamespace(show_face_area_overlay=False, blend_ratio=1.0)
        self._tls = threading.local()


def face_matrix(i, n, w, h):
    s = CROP * 0.8 / FACE
    cx, cy = w * (i + 1) / (n + 1), h * 0.45
    th = np.deg2rad(6 * (i - n / 2))
    c, sn = np.cos(th) * s, np.sin(th) * s
    lin = np.array([[c, -sn], [sn, c]])
    t = np.array([CROP / 2.0, CROP / 2.0]) - lin @ np.array([cx, cy])
    return np.hstack([lin, t[:, None]]).astype(np.float32), (cx, cy)


def hull(cx, cy):
    a = np.linspace(0, 2 * np.pi, 106, endpoint=False)
    return np.stack([cx + FACE * 0.46 * np.cos(a), cy + FACE * 0.55 * np.sin(a)], 1).astype(np.float32)


def timeit(fn, reps, warm=4, cuda=False):
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
        samples.append((time.perf_counter() - t0) * 1000.0)
    return dict(median=statistics.median(samples), p95=sorted(samples)[int(0.95 * (len(samples) - 1))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", default="1,2,4")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out:
        args.out = os.path.abspath(args.out)
    if not torch.cuda.is_available():
        print("[PENDING] CUDA is not available; the GPU arms cannot run", file=sys.stderr)
        return 2
    dev = torch.device("cuda")
    stub = _Stub()
    rows = []
    for w, h in RESOLUTIONS:
        rng = np.random.default_rng(1)
        frame = (rng.random((h, w, 3)) * 80 + 90).astype(np.uint8)
        crop = (rng.random((CROP, CROP, 3)) * 60 + 100).astype(np.uint8)
        crop_t = torch.from_numpy(crop).permute(2, 0, 1).float().div(255).to(dev)[None]
        frame_dev = torch.from_numpy(frame).to(dev).permute(2, 0, 1).float().div_(255)[None]
        for n in [int(x) for x in args.faces.split(",")]:
            mats = [face_matrix(i, n, w, h) for i in range(n)]
            m_t = torch.from_numpy(np.stack([m for m, _ in mats])).to(dev)
            faces = crop_t.expand(n, -1, -1, -1).contiguous()

            def cpu():
                out = frame.copy()
                for m, (cx, cy) in mats:
                    out = stub.paste_upscale(crop, crop, m, out, 1, [0, 0, 0, 0, 20.0, 10.0],
                                             face_landmarks=hull(cx, cy), inplace=True)
                return out

            def gpu(method, resident):
                def run():
                    src = frame_dev if resident else torch.from_numpy(frame).to(dev).permute(2, 0, 1).float().div_(255)[None]
                    comp = method(src, faces, [0] * n, m_t, masks=None, feather=True)
                    if not resident:
                        comp.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu()
                return run

            row = dict(resolution=f"{w}x{h}", faces=n,
                       cpu=timeit(cpu, args.reps),
                       gpu_full_host=timeit(gpu(CudaAffineBatch.paste_faces, False), args.reps, cuda=True),
                       gpu_full_vram=timeit(gpu(CudaAffineBatch.paste_faces, True), args.reps, cuda=True),
                       gpu_roi_host=timeit(gpu(CudaAffineBatch.paste_faces_roi, False), args.reps, cuda=True),
                       gpu_roi_vram=timeit(gpu(CudaAffineBatch.paste_faces_roi, True), args.reps, cuda=True))
            rows.append(row)
            print("%-10s %d face(s)  cpu %7.1f ms | gpu full host %7.2f vram %7.2f | gpu ROI host %7.2f vram %7.2f"
                  % (row["resolution"], n, row["cpu"]["median"], row["gpu_full_host"]["median"], row["gpu_full_vram"]["median"],
                     row["gpu_roi_host"]["median"], row["gpu_roi_vram"]["median"]), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(dict(rows=rows, device=torch.cuda.get_device_name(0)), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
