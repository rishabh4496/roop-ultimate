"""Microbenchmark the Phase 11 temporal-identity hot paths.

This is component evidence only; it is not an end-to-end video FPS claim.
Run from the repository root, for example::

    app/env/Scripts/python.exe app/tests/bench_temporal_identity_cost.py
    app/env/Scripts/python.exe app/tests/bench_temporal_identity_cost.py --lowpass-size 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.temporal_identity import TemporalIdentityStabilizer  # noqa: E402


def _bench_blend(layer, frames):
    layer.blend_output(1, np.full_like(frames[0], 110), confidence=0.8)
    for frame in frames[:100]:
        layer.blend_output(1, frame, confidence=0.8, motion=0.05)
    started = time.perf_counter()
    for frame in frames:
        layer.blend_output(1, frame, confidence=0.8, motion=0.05)
    return len(frames) / max(1e-9, time.perf_counter() - started)


def _bench_mask(layer, masks):
    layer.stabilize_mask(1, masks[0], confidence=0.8)
    for mask in masks[:100]:
        layer.stabilize_mask(1, mask, confidence=0.8)
    started = time.perf_counter()
    for mask in masks:
        layer.stabilize_mask(1, mask, confidence=0.8)
    return len(masks) / max(1e-9, time.perf_counter() - started)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--lowpass-size", type=int, default=128)
    args = parser.parse_args(argv)
    size = max(64, args.size)
    count = max(101, args.frames)
    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
              for _ in range(count)]
    masks = [rng.random((size, size), dtype=np.float32) for _ in range(count)]
    layer = TemporalIdentityStabilizer(
        enabled=True, lowpass_size=args.lowpass_size)
    result = {
        "size": size,
        "frames": count,
        "lowpass_size": max(0, args.lowpass_size),
        "blend_calls_s": round(_bench_blend(layer, frames), 3),
        "mask_calls_s": round(_bench_mask(layer, masks), 3),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
