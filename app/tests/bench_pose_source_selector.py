"""Microbenchmark for the Phase 5 classical pose/source selector.

This measures selection overhead and pose-plan throughput only. It is not a
video FPS, detector, GPU, or quality benchmark.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.face_3d_recon import pose_warp_plan
from roop.pose_source_selector import PoseEstimate, select_pose_aware_source


PROPS = {"face_width_height": 0.78, "eye_distance_face_width": 0.36,
         "eye_mouth_distance_face_height": 0.34, "mouth_width_face_width": 0.30}


def entry(yaw, pitch=0.0, roll=0.0):
    return {"geometry": {"yaw": yaw, "pitch": pitch, "roll": roll,
                          "face_scale": {"relative_height": 0.25},
                          "facial_proportions": PROPS},
            "quality": {"score": 0.95},
            "identity": {"quality_confidence": 0.95},
            "expression": {"descriptor": [0.35, 0.35, 0.10, 0.30]},
            "appearance": {"luminance": {"mean": 0.5},
                           "color_temperature": 1.0}}


def target(yaw, pitch=0.0, roll=0.0):
    return PoseEstimate(yaw=yaw, pitch=pitch, roll=roll, face_scale=180.0,
                        relative_scale=0.25, proportions=PROPS,
                        expression={"descriptor": [0.35, 0.35, 0.10, 0.30]},
                        confidence=0.95, off_axis=abs(yaw), available=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()
    metadata = {"sources": [entry(y) for y in (0, -30, 30, -60, 60, -85, 85)]}
    poses = [target(y, p, r) for y in (0, 30, 45, 60, 75, 85)
             for p in (-30, 0, 30) for r in (-20, 0, 20)]
    start = time.perf_counter()
    last = None
    for i in range(args.iterations):
        last = select_pose_aware_source(metadata, poses[i % len(poses)],
                                        previous_index=last).index
    elapsed = time.perf_counter() - start
    plan_start = time.perf_counter()
    plans = 0
    for i in range(args.iterations):
        y = (-85 + (i % 171))
        pose_warp_plan(y, -30, -y, 30)
        plans += 1
    plan_elapsed = time.perf_counter() - plan_start
    print(f"sources={len(metadata['sources'])} iterations={args.iterations}")
    print(f"selector_seconds={elapsed:.6f} selector_per_sec={args.iterations / max(elapsed, 1e-9):.2f}")
    print(f"warp_plan_seconds={plan_elapsed:.6f} warp_plan_per_sec={plans / max(plan_elapsed, 1e-9):.2f}")
    print(f"last_index={last}")


if __name__ == "__main__":
    main()
