"""Real-input benchmark entry point for Phase 7 occlusion cases.

The production renderer can export one restore-mask image/NPY per face crop;
this harness consumes those real masks and measures raw versus causal mask
temporal differences. It never fabricates a hand, hair, glasses, microphone,
or interacting-face fixture. Without a real video and mask export it writes a
pending report rather than claiming quality or performance.

Examples (run from ``app``)::

    env/Scripts/python.exe tests/phase7_occlusion_bench.py \
        --video path/to/hand_over_eye.mp4 --box 420,160,900,640 \
        --mask-dir path/to/hand_eye_masks --scenario hand_eye \
        --tag phase7_hand_eye_4070
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import date

import cv2
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

SCENARIOS = (
    "hand_eye", "hand_cheek", "hand_mouth", "hair", "glasses",
    "microphone", "two_faces_touching", "two_faces_crossing",
    "partially_hidden",
)


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


def _mask_path(root, index):
    for stem in (f"{index:06d}", f"{index:05d}", str(index)):
        for suffix in (".npy", ".png", ".jpg", ".jpeg"):
            path = os.path.join(root, stem + suffix)
            if os.path.isfile(path):
                return path
    return None


def _read_mask(path, shape):
    if path.lower().endswith(".npy"):
        mask = np.load(path)
    else:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.size == 0:
        return None
    if float(np.nanmax(mask)) > 1.0:
        mask /= 255.0
    mask = np.nan_to_num(mask, nan=1.0, posinf=1.0, neginf=0.0)
    mask = np.clip(mask, 0.0, 1.0)
    if mask.shape != tuple(shape[:2]):
        mask = cv2.resize(mask, (int(shape[1]), int(shape[0])),
                          interpolation=cv2.INTER_LINEAR)
    return mask.astype(np.float32)


def _delta(previous, current):
    if previous is None or current is None or previous.shape != current.shape:
        return None
    return float(np.mean(np.abs(current.astype(np.float32)
                             - previous.astype(np.float32))))


def _write_montage(path, samples):
    if not samples:
        return False
    rows = []
    for frame_index, raw, stable in samples:
        raw = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_NEAREST)
        stable = cv2.resize(stable, (256, 256), interpolation=cv2.INTER_NEAREST)
        raw = cv2.cvtColor((raw * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        stable = cv2.cvtColor((stable * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        label = np.zeros((28, 512, 3), dtype=np.uint8)
        cv2.putText(label, f"frame {frame_index}  raw | stabilized", (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1,
                    cv2.LINE_AA)
        rows.append(np.vstack((np.hstack((raw, stable)), label)))
    return bool(cv2.imwrite(path, np.vstack(rows)))


def run(video, mask_dir, box_text, scenario, tag, outdir, max_frames):
    from roop.temporal_occlusion import TemporalOcclusionEngine

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    box = _box(box_text, width, height)
    engine = TemporalOcclusionEngine.from_env()
    engine.enabled = True
    engine.set_ordered(True)
    support = np.ones((box[3] - box[1], box[2] - box[0]), dtype=np.float32)
    previous_raw = previous_stable = None
    raw_deltas, stable_deltas = [], []
    samples, modes = [], {}
    index = 0
    started = time.perf_counter()
    while index < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        path = _mask_path(mask_dir, index)
        if path is None:
            break
        crop = frame[box[1]:box[3], box[0]:box[2]]
        raw = _read_mask(path, support.shape)
        if raw is None:
            break
        decision = engine.prepare(1, index, support, observation=crop,
                                  confidence=1.0)
        modes[decision.mode] = modes.get(decision.mode, 0) + 1
        if decision.mode == "propagate":
            stable = engine.propagate(1, index, decision, confidence=1.0)
        else:
            stable = engine.observe(1, index, support, raw, confidence=1.0,
                                    analysis_mode=decision.reason)
        if stable is None:
            stable = raw
        raw_delta = _delta(previous_raw, raw)
        stable_delta = _delta(previous_stable, stable)
        if raw_delta is not None:
            raw_deltas.append(raw_delta)
            stable_deltas.append(stable_delta)
        if index in np.linspace(0, max(0, max_frames - 1), 6, dtype=int):
            samples.append((index, raw.copy(), stable.copy()))
        previous_raw, previous_stable = raw, stable
        index += 1
    cap.release()
    elapsed = time.perf_counter() - started
    report = {
        "version": 1, "phase": 7, "date": str(date.today()),
        "scenario": scenario, "tag": tag, "status": "complete",
        "video": os.path.abspath(video), "mask_dir": os.path.abspath(mask_dir),
        "box": list(box), "frames": index, "fps_source": fps,
        "elapsed_seconds": round(elapsed, 6),
        "measured_mask_fps": round(index / max(elapsed, 1e-9), 4),
        "raw_mask_mean_temporal_difference": round(float(np.mean(raw_deltas)), 8)
        if raw_deltas else None,
        "stabilized_mask_mean_temporal_difference": round(float(np.mean(stable_deltas)), 8)
        if stable_deltas else None,
        "raw_mask_p95_temporal_difference": round(float(np.percentile(raw_deltas, 95)), 8)
        if raw_deltas else None,
        "stabilized_mask_p95_temporal_difference": round(float(np.percentile(stable_deltas, 95)), 8)
        if stable_deltas else None,
        "path_counts": modes,
        "visual_review": "pending_manual_review",
        "performance_scope": "mask-state benchmark only; not end-to-end swap FPS",
        "quality_scope": "mask temporal difference only; review output video manually",
    }
    _write_montage(os.path.join(outdir, "mask_before_after_montage.png"), samples)
    with open(os.path.join(outdir, "mask_temporal_difference.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("frame_delta_index", "raw_mask_delta", "stabilized_mask_delta"))
        writer.writerows(zip(range(1, len(raw_deltas) + 1), raw_deltas, stable_deltas))
    return report


def _pending_report(scenario, tag, video, mask_dir=None):
    return {
        "version": 1, "phase": 7, "date": str(date.today()),
        "scenario": scenario, "tag": tag,
        "video": os.path.abspath(video) if video else None,
        "mask_dir": os.path.abspath(mask_dir) if mask_dir else None,
        "status": "pending",
        "pending_reason": "a real video and production mask export are required",
        "metrics": {"flicker_temporal_difference": None, "mask_temporal_difference": None,
                     "fps": None, "seconds_per_frame": None, "vram": None,
                     "gpu_utilization": None, "cpu_usage": None},
        "visual_review": "pending_manual_review",
        "required_review": [
            "object pixels remain visible while crossing the face",
            "mask restoration is smooth when the object leaves",
            "Track A and Track B remain separate during contact/crossing",
        ],
        "interpretation": "No synthetic quality or performance number is claimed.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--video")
    parser.add_argument("--mask-dir", help="real per-frame restore masks (.npy/.png)")
    parser.add_argument("--box", help="face ROI x0,y0,x1,y1")
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--out", default=os.path.join(APP, "output", "phase7_occlusion"))
    args = parser.parse_args(argv)
    outdir = os.path.join(os.path.abspath(args.out), args.tag)
    os.makedirs(outdir, exist_ok=True)
    if (args.video and os.path.isfile(args.video)
            and args.mask_dir and os.path.isdir(args.mask_dir)):
        report = run(args.video, args.mask_dir, args.box, args.scenario, args.tag,
                     outdir, max(1, args.max_frames))
    else:
        report = _pending_report(args.scenario, args.tag, args.video, args.mask_dir)
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
