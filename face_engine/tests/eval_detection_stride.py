"""Measure detection stride on a real clip: landmark accuracy and time per frame.

Not collected by pytest (no ``test_`` prefix). Usage::

    python -m face_engine.tests.eval_detection_stride CLIP [--frames 600] [--stride 3]

For every frame the detector runs as ground truth. On the frames the stride
would NOT detect, the tracked landmarks are compared with the detector's
(matched by box IoU) and with the naive alternative, holding the last
detection. Errors are normalised by the inter-ocular distance (IOD).
Timing runs the tracker over the same frames, separately, with a device
sync around each update.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch

from face_engine.core.config import EngineConfig, Provider
from face_engine.core.execution import ExecutionEngine
from face_engine.models.zoo import build_default_registry
from face_engine.pipeline.detector import SCRFDDetector
from face_engine.pipeline.tracker import StridedFaceTracker, TrackerConfig


def read_frames(path: str, count: int, start: int = 0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    index = 0
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        if index >= start:
            frames.append(frame)
        index += 1
    cap.release()
    return frames


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def nme(pred: np.ndarray, truth: np.ndarray, box: np.ndarray) -> float:
    """Mean landmark error / sqrt(box area). Not inter-ocular distance: on a
    profile the eyes nearly coincide and IOD-normalised error explodes (the
    bad cases on Love.mp4 had a 6 px IOD, the good ones 33 px)."""
    size = float(np.sqrt(max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)))
    return float(np.linalg.norm(pred - truth, axis=1).mean() / size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    registry = build_default_registry()
    engine = ExecutionEngine(EngineConfig(providers=[Provider.CUDA, Provider.CPU], strict=True))
    detector = SCRFDDetector(engine, registry.ensure("scrfd_10g_bnkps", show_progress=False))
    frames = read_frames(args.clip, args.frames, args.start)
    tensors = [torch.from_numpy(f).cuda().permute(2, 0, 1).float()[None] for f in frames]
    print(f"{len(frames)} frames {frames[0].shape[1]}x{frames[0].shape[0]} stride {args.stride}")

    truth = [detector.detect_cuda(t).to_faces()[0] for t in tensors]

    tracker = StridedFaceTracker(detector, TrackerConfig(detection_stride=args.stride))
    tracked_err, held_err, unmatched, compared = [], [], 0, 0
    held = None
    for i, t in enumerate(tensors):
        out = tracker.update(t)
        faces = out.detections.to_faces()[0]
        if out.detected:
            held = faces
            continue
        gt = [f for f in truth[i] if f.score >= 0.5]
        if not gt or not faces:
            unmatched += len(gt)
            continue
        m = iou(np.stack([f.bbox for f in gt]), np.stack([f.bbox for f in faces]))
        for g, row in zip(gt, m):
            j = int(row.argmax())
            if row[j] < 0.3:
                unmatched += 1
                continue
            compared += 1
            tracked_err.append(nme(faces[j].kps, g.kps, g.bbox))
            if held:
                h = iou(g.bbox[None], np.stack([f.bbox for f in held]))[0]
                held_err.append(nme(held[int(h.argmax())].kps, g.kps, g.bbox))
    s = tracker.stats
    print(f"sources {s.by_source}  detector saving {s.detector_saving:.1%}")
    for name, e in (("tracked", tracked_err), ("held last detection", held_err)):
        e = np.array(e)
        if e.size:
            print(f"{name:>20}: NME mean {e.mean():.4f}  median {np.median(e):.4f}  "
                  f"p95 {np.percentile(e, 95):.4f}  >0.05: {(e > 0.05).mean():.1%}  (n={e.size})")
    print(f"ground-truth faces with no tracked face (IoU<0.3): {unmatched} (compared {compared})")

    # timing
    for stride in (1, args.stride):
        tracker = StridedFaceTracker(detector, TrackerConfig(detection_stride=stride))
        for t in tensors[:30]:
            tracker.update(t)  # warm-up
        tracker.reset()
        per: dict[bool, list[float]] = {True: [], False: []}
        torch.cuda.synchronize()
        start = time.perf_counter()
        for t in tensors:
            t0 = time.perf_counter()
            out = tracker.update(t)
            torch.cuda.synchronize()
            per[out.detected].append((time.perf_counter() - t0) * 1000)
        total = (time.perf_counter() - start) * 1000
        print(f"stride {stride}: {total / len(tensors):.2f} ms/frame  "
              f"detect frames {np.mean(per[True]) if per[True] else 0:.2f} ms (n={len(per[True])})  "
              f"tracked frames {np.mean(per[False]) if per[False] else 0:.2f} ms (n={len(per[False])})")


if __name__ == "__main__":
    main()
