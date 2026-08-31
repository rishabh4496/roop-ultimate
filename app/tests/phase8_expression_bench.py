"""Real-video Phase 8 expression accuracy benchmark.

This harness never fabricates frames or scores.  Give it the original target
clip and the corresponding rendered output clip.  It compares target and
output landmark time series for the requested expression case.

Cases intentionally match the Phase 8 acceptance list:
slow_blink, fast_blink, asymmetric_blink, wink, half_open_eyes, talking,
smiling, mouth_wide_open, teeth_visible, frowning, fast_transitions.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.temporal_expression import measure_expression  # noqa: E402


CASES = (
    "slow_blink", "fast_blink", "asymmetric_blink", "wink",
    "half_open_eyes", "talking", "smiling", "mouth_wide_open",
    "teeth_visible", "frowning", "fast_transitions",
)
CHANNELS = ("left_eye_openness", "right_eye_openness", "mouth_openness",
            "mouth_aspect_ratio")


def _finite_series(values):
    return np.asarray(values, dtype=np.float64)


def _corr(a, b):
    if len(a) < 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _channel_stats(target, output):
    target, output = _finite_series(target), _finite_series(output)
    ok = np.isfinite(target) & np.isfinite(output)
    target, output = target[ok], output[ok]
    if len(target) < 4:
        return {"n": int(len(target))}
    target_range = float(np.ptp(target))
    output_range = float(np.ptp(output))
    target_delta = np.diff(target)
    output_delta = np.diff(output)
    return {
        "n": int(len(target)),
        "mae": float(np.mean(np.abs(target - output))),
        "correlation": _corr(target, output),
        "target_range": target_range,
        "output_range": output_range,
        "range_ratio": (output_range / target_range
                        if target_range > 1e-9 else None),
        "delta_correlation": _corr(target_delta, output_delta),
        "delta_mae": (float(np.mean(np.abs(target_delta - output_delta)))
                      if len(target_delta) else None),
    }


def _face(faces, reference=None):
    if not faces:
        return None
    if reference is None:
        return max(faces, key=lambda face: float(
            (getattr(face, "bbox", [0, 0, 0, 0])[2]
             - getattr(face, "bbox", [0, 0, 0, 0])[0])))
    rb = np.asarray(getattr(reference, "bbox", [0, 0, 0, 0]), dtype=np.float64)
    best, best_score = None, -1.0
    for face in faces:
        try:
            b = np.asarray(face.bbox, dtype=np.float64)
            iw = max(0.0, min(rb[2], b[2]) - max(rb[0], b[0]))
            ih = max(0.0, min(rb[3], b[3]) - max(rb[1], b[1]))
            inter = iw * ih
            union = max(1.0, (rb[2] - rb[0]) * (rb[3] - rb[1])
                        + (b[2] - b[0]) * (b[3] - b[1]) - inter)
            score = inter / union
            if score > best_score:
                best, best_score = face, score
        except (TypeError, ValueError, IndexError):
            continue
    return best or max(faces, key=lambda face: 0.0)


def grade(target_video, output_video, max_frames=0):
    from roop.face_util import get_all_faces

    target_cap, output_cap = cv2.VideoCapture(target_video), cv2.VideoCapture(output_video)
    if not target_cap.isOpened() or not output_cap.isOpened():
        target_cap.release()
        output_cap.release()
        return {"status": "error", "reason": "could_not_open_video"}
    target_values = {key: [] for key in CHANNELS}
    output_values = {key: [] for key in CHANNELS}
    total, paired, detected = 0, 0, 0
    reference = None
    while True:
        ok_target, target_frame = target_cap.read()
        ok_output, output_frame = output_cap.read()
        if not (ok_target and ok_output):
            break
        total += 1
        if max_frames and total > max_frames:
            break
        target_faces = get_all_faces(target_frame) or []
        output_faces = get_all_faces(output_frame) or []
        target_face = _face(target_faces, reference)
        if target_face is not None:
            reference = target_face
        output_face = _face(output_faces, target_face)
        if target_face is None or output_face is None:
            continue
        paired += 1
        target_measurement = measure_expression(
            getattr(target_face, "landmark_2d_106", None),
            kps=getattr(target_face, "kps", None),
            bbox=getattr(target_face, "bbox", None),
            detection_confidence=getattr(target_face, "det_score", 1.0))
        output_measurement = measure_expression(
            getattr(output_face, "landmark_2d_106", None),
            kps=getattr(output_face, "kps", None),
            bbox=getattr(output_face, "bbox", None),
            detection_confidence=getattr(output_face, "det_score", 1.0))
        if (target_measurement.get("confidence", 0.0) <= 0.0
                or output_measurement.get("confidence", 0.0) <= 0.0):
            continue
        detected += 1
        for key in CHANNELS:
            target_values[key].append(target_measurement[key])
            output_values[key].append(output_measurement[key])
    target_cap.release()
    output_cap.release()
    return {
        "status": "complete" if detected else "insufficient_detections",
        "total_frames": total,
        "paired_frames": paired,
        "graded_frames": detected,
        "coverage": float(detected / max(1, total)),
        "channels": {key: _channel_stats(target_values[key], output_values[key])
                     for key in CHANNELS},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-video", help="original target video")
    parser.add_argument("--output-video", help="rendered swapped video")
    parser.add_argument("--scenario", choices=("all",) + CASES, default="all")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--json", dest="json_path", default="")
    args = parser.parse_args()
    if not args.target_video or not args.output_video:
        result = {"status": "pending", "reason": "real_target_and_output_required",
                  "cases": list(CASES)}
    else:
        result = {"scenario": args.scenario,
                  "target_video": os.path.abspath(args.target_video),
                  "output_video": os.path.abspath(args.output_video),
                  "metrics": grade(args.target_video, args.output_video,
                                    args.max_frames)}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    return 0 if result.get("status") in ("pending", "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
