"""Synthetic Phase 12 quality/cost benchmark for final paste-back blending.

This isolates the compositor so a missing video/GPU cannot be mistaken for a
completed quality result.  The matrix covers the requested geometry, hair,
occlusion and illumination cases and reports runtime plus measurable boundary,
detail and temporal metrics.  Use ``phase12_benchmark.py`` for the real video
pipeline A/B after this component benchmark.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.temporal_compositing import (TemporalCompositeController,
                                       composite_linear, composite_multiband,
                                       refine_alpha, technique_report)

CONDITIONS = ("frontal", "lateral", "profile", "hair", "glasses",
              "hand_occlusion", "dark_scene", "bright_scene")


def _scene(condition, size=256):
    h = w = size
    yy, xx = np.mgrid[:h, :w]
    target = np.zeros((h, w, 3), np.float32)
    target[:, :, 0] = 52 + 0.20 * xx
    target[:, :, 1] = 74 + 0.16 * yy
    target[:, :, 2] = 108 + 0.10 * xx
    alpha = np.zeros((h, w), np.float32)
    axes = (70, 86)
    if condition == "lateral":
        axes = (58, 86)
    elif condition == "profile":
        axes = (42, 84)
    cv2.ellipse(alpha, (w // 2, h // 2), axes, 0, 0, 360, 1.0, -1)
    paste = np.empty_like(target)
    paste[:, :, 0] = 92 + 0.04 * yy
    paste[:, :, 1] = 118 + 0.03 * xx
    paste[:, :, 2] = 146 + 0.05 * yy
    # Persistent identity marks are on the generated face, not target texture.
    cv2.circle(paste, (w // 2 + 18, h // 2 - 20), 3, (26, 30, 32), -1)
    cv2.line(paste, (w // 2 - 25, h // 2 + 28),
             (w // 2 + 2, h // 2 + 22), (68, 75, 80), 2)
    if condition == "hair":
        cv2.ellipse(target, (w // 2, h // 2 - 70), (92, 38), 0, 0, 360,
                    (20, 22, 25), -1)
    if condition == "glasses":
        cv2.rectangle(target, (w // 2 - 58, h // 2 - 20),
                      (w // 2 - 5, h // 2 + 12), (26, 28, 30), 2)
        cv2.rectangle(target, (w // 2 + 5, h // 2 - 20),
                      (w // 2 + 58, h // 2 + 12), (26, 28, 30), 2)
    if condition == "hand_occlusion":
        cv2.ellipse(target, (w // 2 - 45, h // 2 + 8), (28, 72), -18, 0,
                    360, (164, 142, 126), -1)
        alpha[:, :w // 2 - 20] *= 0.55
    if condition == "dark_scene":
        target *= 0.28
    elif condition == "bright_scene":
        target = np.clip(target * 1.75, 0, 255)
    return np.clip(paste, 0, 255).astype(np.uint8), np.clip(target, 0, 255).astype(np.uint8), alpha


def _metrics(output, paste, target, alpha, previous=None):
    band = cv2.morphologyEx((alpha > 0.05).astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((5, 5), np.uint8)) > 0
    interior = alpha > 0.85
    pgray = cv2.cvtColor(paste, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ogray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tgray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    pg = np.hypot(cv2.Sobel(pgray, cv2.CV_32F, 1, 0, 3),
                  cv2.Sobel(pgray, cv2.CV_32F, 0, 1, 3))
    og = np.hypot(cv2.Sobel(ogray, cv2.CV_32F, 1, 0, 3),
                  cv2.Sobel(ogray, cv2.CV_32F, 0, 1, 3))
    tg = np.hypot(cv2.Sobel(tgray, cv2.CV_32F, 1, 0, 3),
                  cv2.Sobel(tgray, cv2.CV_32F, 0, 1, 3))
    seam = float(np.mean(np.abs(og[band] - tg[band]))) if np.any(band) else 0.0
    detail = float(np.mean(np.abs(og[interior] - pg[interior]))) if np.any(interior) else 0.0
    target_detail = float(np.mean(np.abs(og[interior] - tg[interior]))) if np.any(interior) else 0.0
    temporal = (float(np.mean(np.abs(output.astype(np.float32) - previous.astype(np.float32))))
                if previous is not None else 0.0)
    return {"boundary_gradient_error": round(seam, 4),
            "identity_detail_error": round(detail, 4),
            "target_texture_error": round(target_detail, 4),
            "temporal_delta": round(temporal, 4)}


def run(condition, frames=24, size=256):
    paste, target, raw_alpha = _scene(condition, size)
    controller = TemporalCompositeController(enabled=True, strength=0.65,
                                              alpha=0.30)
    face = {"_adaptive_yaw": {"frontal": 0, "lateral": 38, "profile": 78}.get(condition, 8),
            "_adaptive_pitch": 12 if condition in ("lateral", "profile") else 0,
            "_temporal_confidence": 0.78}
    appearance = {"tier": "VERY_DARK" if condition == "dark_scene" else "NORMAL"}
    linear_outputs, adaptive_outputs = [], []
    linear_start = time.perf_counter()
    for _ in range(frames):
        linear_outputs.append(composite_linear(paste, target, raw_alpha))
    linear_ms = (time.perf_counter() - linear_start) * 1000.0 / frames
    adaptive_start = time.perf_counter()
    previous = None
    for frame in range(frames):
        # Small detector wobble simulates mask chatter at the jaw/cheek edge.
        raw = np.roll(raw_alpha, 1 if frame % 2 else 0, axis=1)
        stable = controller.stabilize_mask(3, raw, frame, confidence=0.78,
                                           motion=0.08,
                                           occlusion=float(np.mean(1.0 - raw)))
        plan = controller.plan(face, appearance,
                                local_contrast=18 if condition in ("hair", "glasses", "bright_scene") else 6,
                                occlusion=0.35 if condition == "hand_occlusion" else 0.0)
        alpha = refine_alpha(stable, plan)
        adaptive_outputs.append(composite_multiband(paste, target, alpha, plan))
        previous = adaptive_outputs[-1]
    adaptive_ms = (time.perf_counter() - adaptive_start) * 1000.0 / frames
    linear_metric = _metrics(linear_outputs[-1], paste, target, raw_alpha)
    adaptive_metric = _metrics(adaptive_outputs[-1], paste, target, alpha,
                               adaptive_outputs[-2] if frames > 1 else None)
    adaptive_metric["temporal_edge_shimmer"] = adaptive_metric.pop("temporal_delta")
    return {"condition": condition, "frames": frames,
            "linear_ms_per_frame": round(linear_ms, 4),
            "adaptive_ms_per_frame": round(adaptive_ms, 4),
            "cost_ratio": round(adaptive_ms / max(linear_ms, 1e-6), 4),
            "linear": linear_metric, "adaptive": adaptive_metric}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    report = {"techniques": technique_report(),
              "conditions": [run(c, max(2, args.frames), args.size)
                             for c in CONDITIONS]}
    report["summary"] = {"conditions": list(CONDITIONS),
                          "production": report["techniques"]["production"]}
    text = json.dumps(report, indent=2)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
