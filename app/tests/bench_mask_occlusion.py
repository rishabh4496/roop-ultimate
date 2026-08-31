"""Small real-runtime benchmark for the opt-in temporal mask policy.

This deliberately measures only the CPU mask-filter operation. It is not a
video quality claim and does not substitute for the required real-footage A/B.
"""

import argparse
import os
import sys
import time

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.one_euro import MaskStabilizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--frames', type=int, default=2000)
    args = ap.parse_args()

    kps = np.array([[220, 210], [292, 210], [256, 256], [228, 310],
                    [284, 310]], dtype=np.float32)
    empty = np.zeros((args.size, args.size), dtype=np.float32)
    occluded = np.ones_like(empty)
    results = []
    for name, alpha in (('normal', 0.0), ('fast_restore', 0.85)):
        stab = MaskStabilizer(strength=0.5, motion_beta=0.0,
                              fast_restore_alpha=alpha)
        stab.apply(empty, kps, 0)
        start = time.perf_counter()
        entering = None
        for frame in range(1, args.frames + 1):
            mask = occluded if frame == 1 else empty
            out = stab.apply(mask, kps, frame)
            if entering is None:
                entering = float(out.mean())
        elapsed = time.perf_counter() - start
        calls_per_second = args.frames / elapsed if elapsed else float('inf')
        results.append((name, entering, calls_per_second))

    print('mask_occlusion_microbenchmark')
    print(f'  size={args.size}x{args.size} calls={args.frames}')
    for name, entering, rate in results:
        print(f'  {name:12s} entering_restore_mean={entering:.6f} '
              f'calls_per_second={rate:.2f}')


if __name__ == '__main__':
    main()
