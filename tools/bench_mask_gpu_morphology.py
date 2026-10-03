"""cv2 vs torch-CUDA for the mask feather chain: speed and exactness (Stage 7 probe).

The brief: replace cv2.dilate / cv2.erode / cv2.GaussianBlur on the matte with
``F.max_pool2d`` dilation, ``-max_pool2d(-x)`` erosion and a separable ``F.conv2d`` Gaussian,
and match OpenCV to an absolute error under 0.01 (on a 0..1 mask).

Measured 2026-10-03, RTX 4070, single-threaded cv2 (how the app runs it), a face-shaped
uint8 matte, kernel sizes from the shipped ``face_mask_blend`` (20). Two runs (the cv2 numbers
swing ~2x with the machine's clocks; the ratios and the errors do not)::

    side   blur k   erode k   cv2 chain          torch resident   torch + H2D/D2H
     320     51       13      2.0 - 4.1 ms       0.2 - 0.6 ms     0.4 - 0.9 ms
     600     97       25     16.3 - 25.7 ms      0.8 - 1.8 ms     1.0 - 1.4 ms
    1000    161       41     72.4 - 133.6 ms      3.5 - 3.8 ms     4.1 - 4.2 ms

    Gaussian (separable conv2d, reflect pad) vs cv2: max error 0.0035-0.0044   <- meets 0.01
    erosion by a SQUARE max-pool vs cv2's ELLIPSE kernel: max error 0.94-1.00    <- does not
    erosion by a square max-pool vs cv2 RECT kernel: 6e-8 (exact)
    whole chain, square pools vs the shipped ellipse chain: max error 0.13-0.14

The shipped erosion (``blur_area``) and landmark dilation (``create_landmark_mask``) use
``MORPH_ELLIPSE``; a square pool is a different shape, so the brief's "max-pool" recipe cannot
meet its own 0.01 bound against the production masks (it would need the ellipse decomposed
into per-row 1-D pools). And the app is GPU-bound: moving this CPU work to the GPU adds GPU
work to a render that waits on the GPU. See docs/CHANGELOG.md (2026-10-03, mask ROI).

Run: ``app\\env\\Scripts\\python.exe tools/bench_mask_gpu_morphology.py``
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def gauss_t(x, k):
    if k <= 1:
        return x
    ker = torch.tensor(cv2.getGaussianKernel(k, 0).ravel(), dtype=torch.float32, device=x.device)
    p = k // 2
    x = F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), ker.view(1, 1, k, 1))
    return F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), ker.view(1, 1, 1, k))


def dilate_t(x, k):
    return F.max_pool2d(x, k, 1, k // 2)


def erode_t(x, k):
    return -F.max_pool2d(-x, k, 1, k // 2)


def bench(fn, n=100, sync=False):
    for _ in range(10):
        fn()
    if sync:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    cv2.setNumThreads(1)                     # the app: ProcessMgr cv2.setNumThreads(1)
    dev = "cuda"
    face_mask_blend = 20
    for side in (320, 600, 1000):
        mask = np.zeros((side, side), np.uint8)
        cv2.ellipse(mask, (side // 2, side // 2), (int(side * .38), int(side * .46)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        face = int(side * 0.8)
        blend_px = max(1, int(face * face_mask_blend / 200))
        blur_k, ero_k = blend_px * 2 + 1, max(1, blend_px // 4) * 2 + 1
        ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero_k, ero_k))
        rect = cv2.getStructuringElement(cv2.MORPH_RECT, (ero_k, ero_k))
        mt = torch.from_numpy(mask).to(dev).float().div(255)[None, None]

        def cpu_chain():
            return cv2.GaussianBlur(cv2.erode(cv2.GaussianBlur(mask, (3, 3), 0), ellipse),
                                    (blur_k, blur_k), 0)

        def gpu_resident():
            return gauss_t(erode_t(gauss_t(mt, 3), ero_k), blur_k)

        def gpu_transfer():
            x = torch.from_numpy(mask).to(dev).float().div_(255)[None, None]
            y = gauss_t(erode_t(gauss_t(x, 3), ero_k), blur_k)
            return (y * 255).round().clamp_(0, 255).byte().cpu().numpy()

        ref = cpu_chain().astype(np.float32) / 255
        gb = gauss_t(mt, blur_k).cpu().numpy()[0, 0]
        gc = cv2.GaussianBlur(mask, (blur_k, blur_k), 0).astype(np.float32) / 255
        e = erode_t(mt, ero_k).cpu().numpy()[0, 0]
        row = {
            "side": side, "blur_k": blur_k, "erode_k": ero_k,
            "cv2_chain_ms": bench(cpu_chain, 30),
            "torch_resident_ms": bench(gpu_resident, 100, True),
            "torch_with_transfer_ms": bench(gpu_transfer),
            "err_gaussian_max": float(np.abs(gb - gc).max()),
            "err_erode_square_vs_ellipse_max": float(
                np.abs(e - cv2.erode(mask, ellipse).astype(np.float32) / 255).max()),
            "err_erode_square_vs_rect_max": float(
                np.abs(e - cv2.erode(mask, rect).astype(np.float32) / 255).max()),
            "err_chain_vs_shipped_max": float(np.abs(gpu_resident().cpu().numpy()[0, 0] - ref).max()),
        }
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
