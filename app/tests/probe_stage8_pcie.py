#!/usr/bin/env python3
"""What do the host<->device hops of ONE face cost, against what the face costs?

Phase 8 asks for the aligned crop to stay in VRAM from the swapper through XSeg and the restorer to the paste.
Today each model hop crosses PCIe (host array in, host array out). This prices exactly those hops with the real
tensor sizes, pageable and pinned, on this machine, and prints them next to the per-face stage times of the
regression benchmark (docs/perf/stage7/regression_benchmark.txt: swap 93.7 + mask 2 x 33.7 + enhance 84.9 ms/face).

    app/env/Scripts/python.exe app/tests/probe_stage8_pcie.py [--reps 300]
"""
import argparse
import json
import time

import numpy as np
import torch

# (name, shape, dtype) of every array that crosses today: model input down, model output back
HOPS = [
    ('swapper in  128x128x3 + latent 512', (1, 3, 128, 128), np.float32),
    ('swapper out 128x128x3', (1, 3, 128, 128), np.float32),
    ('xseg in     256x256x3 NHWC', (1, 256, 256, 3), np.float32),
    ('xseg out    256x256x1', (1, 256, 256, 1), np.float32),
    ('restorer in 512x512x3', (1, 3, 512, 512), np.float32),
    ('restorer out 512x512x3', (1, 3, 512, 512), np.float32),
]
# per-face stage cost from the regression benchmark (ms/call, calls/face): swap 93.744 x1, mask 33.686 x2, enhance 84.913 x1
FACE_MS = 93.744 + 2 * 33.686 + 84.913


def timeit(fn, reps):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=300)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    dev = torch.device('cuda')
    rows, total_pageable, total_pinned = [], 0.0, 0.0
    for name, shape, dt in HOPS:
        host = np.random.rand(*shape).astype(dt)
        ht = torch.from_numpy(host)
        pinned = ht.pin_memory()
        gpu = torch.empty(shape, dtype=torch.float32, device=dev)
        d_page = timeit(lambda: gpu.copy_(ht), args.reps)
        d_pin = timeit(lambda: gpu.copy_(pinned), args.reps)
        up = gpu.cpu()
        u_page = timeit(lambda: gpu.cpu(), args.reps)
        u_pin_buf = torch.empty(shape, dtype=torch.float32).pin_memory()
        u_pin = timeit(lambda: u_pin_buf.copy_(gpu), args.reps)
        direction_page = d_page if ' in' in name else u_page
        direction_pin = d_pin if ' in' in name else u_pin
        rows.append(dict(hop=name, mb=round(host.nbytes / 2**20, 3), pageable_ms=round(direction_page, 4),
                         pinned_ms=round(direction_pin, 4)))
        total_pageable += direction_page
        total_pinned += direction_pin
    for r in rows:
        print('%-38s %7.3f MB  pageable %.4f ms  pinned %.4f ms' % (r['hop'], r['mb'], r['pageable_ms'], r['pinned_ms']))
    print('one face, all six hops: pageable %.3f ms, pinned %.3f ms  vs %.1f ms of stage time per face -> %.2f%% / %.2f%%'
          % (total_pageable, total_pinned, FACE_MS, 100 * total_pageable / FACE_MS, 100 * total_pinned / FACE_MS))
    out = dict(rows=rows, face_stage_ms=FACE_MS, pageable_total_ms=total_pageable, pinned_total_ms=total_pinned,
               pageable_pct=100 * total_pageable / FACE_MS, pinned_pct=100 * total_pinned / FACE_MS,
               gpu=torch.cuda.get_device_name(0))
    if args.out:
        json.dump(out, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
