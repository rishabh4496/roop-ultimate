# SPDX-License-Identifier: MIT
"""Per-track temporal landmark, affine, and pose stabilization.

The filters in this module are NumPy/OpenCV only.  They are deliberately kept
outside inference sessions so the CUDA/TensorRT frame path and its zero-copy
buffers remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Optional

import cv2
import numpy as np


def _alpha(dt: float, cutoff: float) -> float:
    cutoff = max(1e-6, float(cutoff))
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / max(float(dt), 1e-6))


class OneEuroFilter:
    """One Euro low-pass filter for scalar or arbitrary-shaped NumPy signals.

    ``timestamp`` is in seconds.  ``beta`` may be overridden for one sample;
    FaceTrackStabilizer uses this to open the filter only when landmark motion
    is geometrically coherent (detector jitter alone does not increase cutoff).
    """

    def __init__(self, min_cutoff: float = 0.05, beta: float = 0.5,
                 d_cutoff: float = 4.0):
        self.min_cutoff = max(1e-6, float(min_cutoff))
        self.beta = max(0.0, float(beta))
        self.d_cutoff = max(1e-6, float(d_cutoff))
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    def reset(self) -> None:
        self.x_prev = self.dx_prev = self.t_prev = None

    def __call__(self, value: Any, timestamp: float, *, beta: Optional[float] = None):
        scalar = np.isscalar(value)
        x = np.asarray(value, dtype=np.float64)
        t = float(timestamp)
        if not np.isfinite(x).all() or not math.isfinite(t):
            return float(value) if scalar else np.asarray(value)
        if self.x_prev is None or self.x_prev.shape != x.shape:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return float(x) if scalar else x.copy()
        dt = t - float(self.t_prev)
        if dt <= 0.0:
            self.reset()
            return self.__call__(value, timestamp, beta=beta)
        derivative = (x - self.x_prev) / dt
        da = _alpha(dt, self.d_cutoff)
        dx_hat = da * derivative + (1.0 - da) * self.dx_prev
        b = self.beta if beta is None else max(0.0, float(beta))
        cutoff = self.min_cutoff + b * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt)
        result = a * x + (1.0 - a) * self.x_prev
        self.x_prev = result
        self.dx_prev = dx_hat
        self.t_prev = t
        return float(result) if scalar else result.copy()


def forward_backward_one_euro(values: Any, timestamps: Any, *, min_cutoff: float = 0.05,
                              beta: float = 0.25, d_cutoff: float = 8.0) -> np.ndarray:
    """Smooth a complete ordered sequence without leaving causal filter state in workers.

    The first pass runs in presentation order. The second pass runs over the first
    pass's output in reverse presentation order, with increasing synthetic times
    whose intervals match the source PTS intervals. This is an offline operation:
    callers must finish it during pre-scan and publish the returned array before
    any render worker starts.
    """
    data = np.asarray(values, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if data.ndim < 1 or len(data) != len(times):
        raise ValueError("values and timestamps must have the same non-zero leading dimension")
    if len(data) == 0:
        return data.copy()
    if not np.isfinite(data).all() or not np.isfinite(times).all():
        return data.copy()
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")

    forward = np.empty_like(data)
    filt = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    for i, (value, timestamp) in enumerate(zip(data, times)):
        forward[i] = filt(value, float(timestamp))

    reverse_times = np.zeros(len(times), dtype=np.float64)
    if len(times) > 1:
        reverse_times[1:] = np.cumsum(np.diff(times)[::-1])
    backward_reversed = np.empty_like(data)
    filt.reset()
    for i, (value, timestamp) in enumerate(zip(forward[::-1], reverse_times)):
        backward_reversed[i] = filt(value, float(timestamp))
    original = np.asarray(values)
    result = backward_reversed[::-1]
    return result.astype(original.dtype, copy=False) if np.issubdtype(original.dtype, np.floating) else result


def _get(face: Any, key: str, default=None):
    if isinstance(face, dict):
        return face.get(key, default)
    try:
        value = getattr(face, key, default)
    except (AttributeError, TypeError):
        return default
    return default if value is None else value


def _similarity_parameters(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    a, b = m[0, 0], m[0, 1]
    c, d = m[1, 0], m[1, 1]
    # The input is an alignment similarity. Averaging its two scale estimates
    # removes small numerical anisotropy before it can become affine shear.
    scale = max(1e-8, 0.5 * (math.hypot(a, c) + math.hypot(b, d)))
    angle = math.atan2(c - b, a + d)
    return np.asarray([m[0, 2], m[1, 2], math.log(scale), angle], dtype=np.float64)


def _similarity_matrix(params: np.ndarray) -> np.ndarray:
    tx, ty, log_scale, angle = np.asarray(params, dtype=np.float64).reshape(4)
    scale = math.exp(float(np.clip(log_scale, -12.0, 12.0)))
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.asarray([[scale * c, -scale * s, tx],
                       [scale * s, scale * c, ty]], dtype=np.float32)


def _fit_similarity(previous: np.ndarray, current: np.ndarray):
    """Return robust face-wide translation/rotation/scale motion evidence."""
    p = np.asarray(previous, dtype=np.float64).reshape(-1, 2)
    q = np.asarray(current, dtype=np.float64).reshape(-1, 2)
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    x, y = p - pc, q - qc
    denom = float(np.sum(x * x))
    if denom <= 1e-8:
        return 0.0, 1.0, np.zeros(4, dtype=np.float64)
    dot = float(np.sum(x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]))
    cross = float(np.sum(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]))
    angle = math.atan2(cross, dot)
    scale = math.hypot(dot, cross) / denom
    c, s = math.cos(angle), math.sin(angle)
    predicted = (x @ np.asarray([[c, s], [-s, c]], dtype=np.float64)) * scale + qc
    residual = float(np.sqrt(np.mean(np.sum((predicted - q) ** 2, axis=1))))
    displacement = float(np.sqrt(np.mean(np.sum((q - p) ** 2, axis=1))))
    coherence = 1.0 - residual / max(displacement, 1e-6)
    face_size = max(1.0, float(np.sqrt(denom / max(1, len(p)))))
    motion = np.asarray([qc[0] - pc[0], qc[1] - pc[1],
                         angle * face_size,
                         math.log(max(scale, 1e-6)) * face_size], dtype=np.float64)
    speed = float(np.linalg.norm(motion))
    return speed, max(0.0, min(1.0, coherence)), motion


@dataclass
class _TrackState:
    last_frame: Optional[int] = None
    last_timestamp: Optional[float] = None
    last_raw_kps: Optional[np.ndarray] = None
    motion_direction: Optional[np.ndarray] = None
    motion_streak: int = 0
    landmarks_filter: OneEuroFilter = field(default_factory=OneEuroFilter)
    dense_filter: OneEuroFilter = field(default_factory=OneEuroFilter)
    affine_filter: OneEuroFilter = field(default_factory=OneEuroFilter)
    bbox_filter: OneEuroFilter = field(default_factory=OneEuroFilter)
    kalman: Any = None
    missed: int = 0
    last_bbox: Optional[np.ndarray] = None
    last_pose: Optional[np.ndarray] = None
    last_affine: Optional[np.ndarray] = None


class FaceTrackStabilizer:
    """Stateful stabilizer keyed by tracked face ID.

    Landmark and affine filtering is causal and velocity-adaptive. The affine is
    filtered as ``tx, ty, log(scale), angle`` and reconstructed as a similarity
    matrix, so the output cannot acquire shear. A 12-state OpenCV Kalman filter
    tracks center/velocity, log scale, and yaw/pitch/roll with their velocities;
    callers may request at most eight prediction frames.
    """

    def __init__(self, *, strength: float = 0.5, min_cutoff: float = 0.05,
                 beta: float = 0.5, d_cutoff: float = 8.0,
                 dropout_prediction_frames: int = 8, fps: float = 30.0):
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.min_cutoff = max(0.005, float(min_cutoff))
        self.beta = max(0.0, float(beta)) * self.strength
        self.d_cutoff = max(0.1, float(d_cutoff))
        self.dropout_prediction_frames = int(np.clip(dropout_prediction_frames, 0, 8))
        self.fps = max(1.0, float(fps))
        self._tracks: dict[Any, _TrackState] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def set_strength(self, strength: float) -> None:
        """Apply an animated strength to existing filters without losing time."""
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.beta = 0.5 * self.strength
        self.min_cutoff = max(0.005, 0.5 - 0.45 * self.strength)
        for state in self._tracks.values():
            for name in ('landmarks_filter', 'dense_filter', 'affine_filter', 'bbox_filter'):
                getattr(state, name).min_cutoff = self.min_cutoff
        if self.strength <= 0.0:
            self.reset()

    def reset_track(self, track_id: Any) -> None:
        self._tracks.pop(track_id, None)

    def _new_state(self) -> _TrackState:
        return _TrackState(
            landmarks_filter=OneEuroFilter(self.min_cutoff, 0.0, self.d_cutoff),
            dense_filter=OneEuroFilter(self.min_cutoff, 0.0, self.d_cutoff),
            affine_filter=OneEuroFilter(self.min_cutoff, 0.0, self.d_cutoff),
            bbox_filter=OneEuroFilter(self.min_cutoff, 0.0, self.d_cutoff),
        )

    @staticmethod
    def _init_kalman(bbox, pose):
        box = np.asarray(bbox, dtype=np.float64).reshape(4)
        center = ((box[:2] + box[2:]) * 0.5)
        scale = max(1.0, math.sqrt(max(1.0, (box[2] - box[0]) * (box[3] - box[1]))))
        p = np.zeros(3, dtype=np.float64) if pose is None else np.asarray(pose, dtype=np.float64).reshape(3)
        kf = cv2.KalmanFilter(12, 6, 0, cv2.CV_32F)
        kf.statePost = np.asarray([[center[0]], [center[1]], [0.0], [0.0],
                                   [math.log(scale)], [0.0], [p[0]], [p[1]], [p[2]],
                                   [0.0], [0.0], [0.0]], dtype=np.float32)
        kf.errorCovPost = np.eye(12, dtype=np.float32) * 4.0
        kf.measurementMatrix = np.zeros((6, 12), dtype=np.float32)
        for row, col in enumerate((0, 1, 4, 6, 7, 8)):
            kf.measurementMatrix[row, col] = 1.0
        kf.measurementNoiseCov = np.diag([2.0, 2.0, 0.025, 4.0, 4.0, 4.0]).astype(np.float32)
        kf.processNoiseCov = np.eye(12, dtype=np.float32) * 0.05
        return kf

    def _set_transition(self, kf, dt):
        f = np.eye(12, dtype=np.float32)
        for pos, vel in ((0, 2), (1, 3), (4, 5), (6, 9), (7, 10), (8, 11)):
            f[pos, vel] = float(dt)
        kf.transitionMatrix = f

    def update(self, track_id: Any, *, frame_index: int, landmarks: Any,
               dense_landmarks: Any = None, bbox: Any = None, pose: Any = None,
               affine: Any = None, scene_cut: bool = False,
               detected: bool = True) -> dict:
        """Filter one face observation or apply its bounded dropout prediction."""
        fi = int(frame_index)
        timestamp = fi / self.fps
        state = self._tracks.get(track_id)
        if (scene_cut or (state is not None and state.last_frame is not None
                          and fi != state.last_frame + 1)):
            self.reset_track(track_id)
            state = None
        if state is None:
            state = self._tracks[track_id] = self._new_state()
        if not detected:
            return self._predict_state(track_id, state, fi, timestamp, landmarks, bbox, pose)

        kps = np.asarray(landmarks, dtype=np.float64)
        if kps.ndim != 2 or kps.shape[0] < 5 or kps.shape[1] < 2:
            kps = np.asarray(landmarks)
            return {"landmarks": kps, "dense_landmarks": dense_landmarks,
                    "bbox": bbox, "pose": pose, "affine": affine, "predicted": False}
        kps = kps[:, :2]
        coherent_speed, coherence, motion = (0.0, 0.0, np.zeros(4, dtype=np.float64))
        if state.last_raw_kps is not None and state.last_raw_kps.shape == kps.shape:
            coherent_speed, coherence, motion = _fit_similarity(state.last_raw_kps, kps)
        direction = motion / max(float(np.linalg.norm(motion)), 1e-8)
        if coherence >= 0.45 and coherent_speed * self.fps >= 8.0:
            if (state.motion_direction is not None
                    and float(np.dot(direction, state.motion_direction)) >= 0.55):
                state.motion_streak += 1
            else:
                state.motion_streak = 1
            state.motion_direction = direction
        else:
            state.motion_streak = 0
            state.motion_direction = None

        # Require a second coherent sample before opening the adaptive cutoff.
        # Independent ±4px point noise can fit an accidental similarity once,
        # but it does not preserve a common velocity direction across frames.
        motion_beta = self.beta if state.motion_streak >= 2 else 0.0

        filtered = np.asarray(state.landmarks_filter(kps, timestamp, beta=motion_beta), dtype=np.float32)
        state.last_raw_kps = kps.copy()
        dense_out = dense_landmarks
        if dense_landmarks is not None:
            dense = np.asarray(dense_landmarks, dtype=np.float64)
            if dense.ndim == 2 and dense.shape[1] >= 2 and np.isfinite(dense).all():
                dense_out = np.asarray(state.dense_filter(dense, timestamp, beta=motion_beta), dtype=np.float32)

        affine_out = affine
        if affine is not None:
            raw_affine = np.asarray(affine, dtype=np.float64).reshape(2, 3)
            params = _similarity_parameters(raw_affine)
            if state.affine_filter.x_prev is not None:
                previous_angle = float(state.affine_filter.x_prev[3])
                params[3] = previous_angle + ((params[3] - previous_angle + math.pi) % (2 * math.pi) - math.pi)
            smooth = state.affine_filter(params, timestamp, beta=motion_beta)
            affine_out = _similarity_matrix(smooth)
            state.last_affine = np.asarray(affine_out, dtype=np.float32)

        box = None if bbox is None else np.asarray(bbox, dtype=np.float32).reshape(4)
        box_out = box
        p = None if pose is None else np.asarray(pose, dtype=np.float32).reshape(3)
        if box is not None:
            box_smooth = np.asarray(state.bbox_filter(box, timestamp, beta=motion_beta), dtype=np.float32)
            box_out = box_smooth
            if state.kalman is None:
                state.kalman = self._init_kalman(box_smooth, p)
            else:
                self._set_transition(state.kalman, 1.0 / self.fps)
                state.kalman.predict()
                center = (box_smooth[:2] + box_smooth[2:]) * 0.5
                scale = max(1.0, math.sqrt(max(1.0, float((box_smooth[2] - box_smooth[0]) * (box_smooth[3] - box_smooth[1])))))
                pv = np.zeros(3, dtype=np.float32) if p is None else p
                measurement = np.asarray([[center[0]], [center[1]], [math.log(scale)],
                                          [pv[0]], [pv[1]], [pv[2]]], dtype=np.float32)
                state.kalman.correct(measurement)
            state.last_bbox, state.last_pose = box_smooth.copy(), p
        state.missed = 0
        state.last_frame, state.last_timestamp = fi, timestamp
        return {"landmarks": filtered, "dense_landmarks": dense_out,
                "bbox": box_out if box_out is not None else bbox, "pose": p if p is not None else pose,
                "affine": affine_out, "predicted": False}

    def _predict_state(self, track_id, state, frame_index, timestamp, landmarks, bbox, pose):
        state.missed += 1
        expired = state.missed > self.dropout_prediction_frames
        if state.kalman is None or expired or self.dropout_prediction_frames <= 0:
            if expired:
                # A later reappearance is a fresh observation, not a continuation
                # of the old face position. Drop landmark and affine history too.
                self.reset_track(track_id)
            state.last_frame, state.last_timestamp = frame_index, timestamp
            return {"landmarks": landmarks, "dense_landmarks": None,
                    "bbox": bbox, "pose": pose, "affine": None,
                    "predicted": False}
        self._set_transition(state.kalman, 1.0 / self.fps)
        predicted = state.kalman.predict().reshape(-1)
        center = predicted[:2]
        scale = math.exp(float(np.clip(predicted[4], -8.0, 12.0)))
        old = state.last_bbox if state.last_bbox is not None else np.asarray(bbox, dtype=np.float32).reshape(4)
        width = max(1.0, float(old[2] - old[0]))
        height = max(1.0, float(old[3] - old[1]))
        old_scale = max(1.0, math.sqrt(width * height))
        ratio = scale / old_scale
        new_w, new_h = width * ratio, height * ratio
        predicted_box = np.asarray([center[0] - new_w * 0.5, center[1] - new_h * 0.5,
                                    center[0] + new_w * 0.5, center[1] + new_h * 0.5], dtype=np.float32)
        old_center = (old[:2] + old[2:]) * 0.5
        pts = None if landmarks is None else np.asarray(landmarks, dtype=np.float32).copy()
        if pts is not None and pts.ndim == 2 and pts.shape[1] >= 2:
            pts[:, :2] = (pts[:, :2] - old_center) * ratio + center
        pv = np.asarray(predicted[[6, 7, 8]], dtype=np.float32)
        if state.last_affine is not None:
            old_matrix = np.asarray(state.last_affine, dtype=np.float64).reshape(2, 3)
            crop_center = old_matrix[:, :2] @ old_center + old_matrix[:, 2]
            new_linear = old_matrix[:, :2] / max(1e-6, ratio)
            state.last_affine = np.column_stack((
                new_linear, crop_center - new_linear @ center)).astype(np.float32)
        state.last_bbox = predicted_box
        state.last_pose = pv
        state.last_frame, state.last_timestamp = frame_index, timestamp
        return {"landmarks": pts if pts is not None else landmarks,
                "dense_landmarks": None, "bbox": predicted_box, "pose": pv,
                "affine": state.last_affine, "predicted": True}


from roop.one_euro import AdaptiveGeometryStabilizer

__all__ = ["OneEuroFilter", "FaceTrackStabilizer", "forward_backward_one_euro", "AdaptiveGeometryStabilizer"]
