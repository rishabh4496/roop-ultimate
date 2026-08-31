"""Benchmark the Phase 3 scheduler/tracker without model or GPU claims.

This measures the policy overhead and the full-frame-call reduction for a
deterministic two-face motion script. Use baseline_controlled.py for the real
end-to-end FPS measurement; this harness intentionally does not pretend that
synthetic tracker calls are detector or pipeline throughput.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.temporal_tracker import TemporalFaceTracker  # noqa: E402


def _face(x, identity, y=160.0, size=96.0):
    return {
        "bbox": np.array([x, y, x + size, y + size], dtype=np.float32),
        "embedding": np.eye(2, dtype=np.float32)[identity],
        "det_score": 0.95,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--full-interval", type=int, default=8)
    args = ap.parse_args()

    tracker = TemporalFaceTracker(full_interval=args.full_interval)
    t0 = time.perf_counter()
    roi_area = []
    for frame in range(max(1, args.frames)):
        ax = 80.0 + 1.1 * frame
        bx = 560.0 - 1.0 * frame
        plan = tracker.plan(frame, (720, 1280, 3))
        if plan.mode == "roi":
            roi_area.append(max(0.0, (plan.roi[2] - plan.roi[0]) *
                                (plan.roi[3] - plan.roi[1])))
        tracker.update([_face(ax, 0), _face(bx, 1)], frame,
                       (720, 1280, 3), detection_mode=plan.mode)
    elapsed = max(1e-9, time.perf_counter() - t0)
    stats = tracker.stats
    full = stats["full_detections"]
    roi = stats["roi_detections"]
    print(f"frames={args.frames} full_interval={args.full_interval}")
    print(f"full_frame_calls={full} roi_calls={roi} "
          f"full_call_reduction={(1.0 - full / max(1, args.frames)) * 100:.2f}%")
    print(f"tracker_total_fps={args.frames / elapsed:.2f}")
    if roi_area:
        print(f"mean_roi_area_fraction={np.mean(roi_area) / (720.0 * 1280.0):.4f}")


if __name__ == "__main__":
    main()
