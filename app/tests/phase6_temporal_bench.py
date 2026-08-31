"""Real-input benchmark for Phase 6 temporal identity stabilization.

This tool intentionally requires a user-provided video. It measures the
low-frequency aligned-face stabilizer in isolation from the full swap model,
so its numbers are temporal-difference evidence for this layer, not end-to-end
video FPS or identity-quality claims. The montage is emitted for manual visual
inspection; the tool never marks that review as passed automatically.

Example (run from ``app``):

    env/Scripts/python.exe tests/phase6_temporal_bench.py \
        --video path/to/clip.mp4 --box 420,160,900,640 \
        --scenario talking --tag phase6_talking_4070
"""

from __future__ import annotations

import argparse
import csv
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


SCENARIOS = ("static", "talking", "rapid_rotation", "blinking", "motion", "lighting")


def _box(value, width, height):
    if value:
        parts = [int(float(item.strip())) for item in value.split(",")]
        if len(parts) != 4:
            raise ValueError("--box must be x0,y0,x1,y1")
        x0, y0, x1, y1 = parts
    else:
        side = int(min(width, height) * 0.55)
        x0, y0 = (width - side) // 2, (height - side) // 2
        x1, y1 = x0 + side, y0 + side
    x0, x1 = max(0, min(width - 2, x0)), max(2, min(width, x1))
    y0, y1 = max(0, min(height - 2, y0)), max(2, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("--box is outside the video frame")
    return x0, y0, x1, y1


def _delta(previous, current):
    if previous is None or current is None or previous.shape != current.shape:
        return None
    return float(np.mean(np.abs(current.astype(np.float32)
                             - previous.astype(np.float32))) / 255.0)


def _write_montage(path, samples):
    if not samples:
        return False
    rows = []
    for frame_index, raw, stabilized in samples:
        raw = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_AREA)
        stabilized = cv2.resize(stabilized, (256, 256), interpolation=cv2.INTER_AREA)
        label = np.zeros((28, 512, 3), dtype=np.uint8)
        cv2.putText(label, f"frame {frame_index}  raw | stabilized", (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1,
                    cv2.LINE_AA)
        rows.append(np.vstack((np.hstack((raw, stabilized)), label)))
    montage = np.vstack(rows)
    return bool(cv2.imwrite(path, montage))


def run(video, box_text, scenario, tag, out_root, max_frames):
    from roop.temporal_identity import TemporalIdentityStabilizer

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    box = _box(box_text, width, height)
    outdir = os.path.join(out_root, tag)
    os.makedirs(outdir, exist_ok=True)
    layer = TemporalIdentityStabilizer.from_env()
    # A benchmark must exercise the layer even when the caller did not export
    # its production feature flag; production integration remains opt-in.
    layer.enabled = True

    raw_previous = stabilized_previous = None
    raw_deltas, stabilized_deltas = [], []
    samples = []
    frame_index = 0
    started = time.perf_counter()
    while frame_index < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        x0, y0, x1, y1 = box
        crop = frame[y0:y1, x0:x1].copy()
        stabilized = layer.blend_output(0, crop, confidence=0.6)
        raw_delta = _delta(raw_previous, crop)
        stabilized_delta = _delta(stabilized_previous, stabilized)
        if raw_delta is not None:
            raw_deltas.append(raw_delta)
            stabilized_deltas.append(stabilized_delta)
        if frame_index in np.linspace(0, max(0, max_frames - 1), 6, dtype=int):
            samples.append((frame_index, crop.copy(), stabilized.copy()))
        raw_previous, stabilized_previous = crop, stabilized
        frame_index += 1
    cap.release()
    elapsed = time.perf_counter() - started

    def mean(values):
        return round(float(np.mean(values)), 8) if values else None

    report = {
        "version": 1,
        "phase": 6,
        "scenario": scenario,
        "video": os.path.abspath(video),
        "box": list(box),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frame_index,
        "elapsed_seconds": round(elapsed, 3),
        "raw_mean_temporal_delta": mean(raw_deltas),
        "stabilized_mean_temporal_delta": mean(stabilized_deltas),
        "raw_p95_temporal_delta": round(float(np.percentile(raw_deltas, 95)), 8)
        if raw_deltas else None,
        "stabilized_p95_temporal_delta": round(float(np.percentile(stabilized_deltas, 95)), 8)
        if stabilized_deltas else None,
        "visual_review": "pending_manual_review",
        "interpretation": (
            "This is aligned-crop temporal-difference evidence for the Phase 6 "
            "layer. It is not end-to-end swap FPS or proof of identity quality."
        ),
    }
    _write_montage(os.path.join(outdir, "before_after_montage.png"), samples)
    with open(os.path.join(outdir, "temporal_delta.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("frame_delta_index", "raw_delta", "stabilized_delta"))
        writer.writerows(zip(range(1, len(raw_deltas) + 1), raw_deltas, stabilized_deltas))
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--video")
    parser.add_argument("--box", help="face ROI x0,y0,x1,y1; required for reliable measurement")
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--out", default=os.path.join(APP, "output", "phase6_temporal"))
    args = parser.parse_args(argv)
    outdir = os.path.join(os.path.abspath(args.out), args.tag)
    os.makedirs(outdir, exist_ok=True)
    if not args.video or not os.path.isfile(args.video):
        report = {
            "version": 1,
            "phase": 6,
            "scenario": args.scenario,
            "status": "pending",
            "pending_reason": "a real video fixture is required",
            "visual_review": "pending_manual_review",
        }
        with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(json.dumps(report, indent=2))
        return 0
    report = run(args.video, args.box, args.scenario, args.tag,
                 os.path.abspath(args.out), max(1, args.max_frames))
    report["status"] = "complete"
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
