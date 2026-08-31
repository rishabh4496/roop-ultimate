"""Lightweight target-expression measurements and temporal continuity.

The swapper owns identity and appearance.  This module owns only the target's
expression state: eyelid aperture, mouth aperture/shape, brow movement and jaw
motion.  It is deliberately model-free and opt-in via
``ROOP_TEMPORAL_EXPRESSION=1``.  State is bounded per track and is consumed by
the ordered tracking replay, so worker order cannot make a blink or wink
change state randomly.
"""

from dataclasses import dataclass
import math
import os
from threading import RLock

import numpy as np


# These are the same InsightFace 106-point indices used by expression_bench.py.
LEFT_EYE = (33, 35, 36, 37, 39, 42)
RIGHT_EYE = (87, 89, 90, 91, 93, 96)
MOUTH_VERTICAL = (52, 61)
MOUTH_HORIZONTAL = (53, 59)

# The existing DMDNet map is the repository's trusted 106 -> 68 convention.
# 68 brow points are 17..26, hence these two 5-point brow groups.
LEFT_BROW = (43, 48, 49, 51, 50)
RIGHT_BROW = (102, 103, 104, 105, 101)


def _array(value, minimum=1):
    if value is None:
        return None
    try:
        out = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if out.ndim != 2 or out.shape[0] < minimum or out.shape[1] < 2:
        return None
    if not np.isfinite(out[:, :2]).all():
        return None
    return out[:, :2].copy()


def _scalar(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _clip(value, low=0.0, high=1.0):
    return max(low, min(high, _scalar(value, low)))


def _distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32)
                                - np.asarray(b, dtype=np.float32)))


def _ear(points, indices):
    p = points[list(indices)]
    width = max(1e-5, _distance(p[0], p[3]))
    return (_distance(p[1], p[5]) + _distance(p[2], p[4])) / (2.0 * width)


def _bbox(points, bbox=None):
    if bbox is not None:
        try:
            value = np.asarray(bbox, dtype=np.float32).reshape(-1)
            if value.size >= 4 and np.isfinite(value[:4]).all():
                return value[:4]
        except (TypeError, ValueError):
            pass
    lo, hi = points.min(axis=0), points.max(axis=0)
    return np.asarray([lo[0], lo[1], hi[0], hi[1]], dtype=np.float32)


def measure_expression(landmarks, kps=None, bbox=None, landmarks68=None,
                       detection_confidence=1.0):
    """Measure target expression without invoking an inference model.

    ``eye_openness`` is EAR per eye. ``mouth_openness`` is the normalized
    vertical aperture (the same ratio as MAR here); it is intentionally not a
    source-face value. Brow movement is signed relative to the eye line when
    106 landmarks are present, and jaw movement is a normalized motion proxy
    filled by the temporal engine on the next frame.
    """
    points = _array(landmarks, minimum=106)
    if points is None:
        return {"confidence": 0.0}
    try:
        left_eye = _ear(points, LEFT_EYE)
        right_eye = _ear(points, RIGHT_EYE)
        mouth_w = max(1e-5, _distance(points[MOUTH_HORIZONTAL[0]],
                                       points[MOUTH_HORIZONTAL[1]]))
        mouth_open = _distance(points[MOUTH_VERTICAL[0]],
                               points[MOUTH_VERTICAL[1]]) / mouth_w
        face_box = _bbox(points, bbox)
        scale = max(1.0, float(np.linalg.norm(
            points[list(LEFT_EYE)].mean(axis=0)
            - points[list(RIGHT_EYE)].mean(axis=0))))
        face_height = max(1.0, float(face_box[3] - face_box[1]))

        brow_y = None
        eye_y = (float(points[list(LEFT_EYE)].mean(axis=0)[1])
                 + float(points[list(RIGHT_EYE)].mean(axis=0)[1])) / 2.0
        brow_indices = LEFT_BROW + RIGHT_BROW
        if max(brow_indices) < points.shape[0]:
            brow_y = float(points[list(brow_indices)].mean(axis=0)[1])

        confidence_parts = [1.0, 1.0, 1.0]
        if not np.isfinite([left_eye, right_eye, mouth_open]).all():
            return {"confidence": 0.0}
        if brow_y is None:
            confidence_parts.append(0.45)
        if kps is None:
            confidence_parts.append(0.75)
        confidence = _clip(_scalar(detection_confidence, 1.0))
        confidence *= float(np.mean(confidence_parts))
        result = {
            "left_eye_openness": float(max(0.0, left_eye)),
            "right_eye_openness": float(max(0.0, right_eye)),
            "mouth_openness": float(max(0.0, mouth_open)),
            "mouth_aspect_ratio": float(max(0.0, mouth_open)),
            "brow_position": (float((brow_y - eye_y) / face_height)
                              if brow_y is not None else None),
            "jaw_position": float(points[:33, 1].mean() / scale),
            "scale": float(scale),
            "confidence": float(_clip(confidence)),
        }
        if landmarks68 is not None:
            # Optional analysis output is accepted for callers that have it,
            # but no second model is required by this engine.
            lm68 = _array(landmarks68, minimum=68)
            if lm68 is not None:
                result["analysis68_available"] = True
        return result
    except (IndexError, TypeError, ValueError, FloatingPointError):
        return {"confidence": 0.0}


@dataclass
class TemporalExpressionState:
    track_id: int
    left_eye_openness: float = 0.0
    right_eye_openness: float = 0.0
    raw_left_eye_openness: float = 0.0
    raw_right_eye_openness: float = 0.0
    blink_state: str = "unknown"
    left_blink_state: str = "unknown"
    right_blink_state: str = "unknown"
    mouth_openness: float = 0.0
    mouth_aspect_ratio: float = 0.0
    eyebrow_movement: float = 0.0
    jaw_movement: float = 0.0
    expression_confidence: float = 0.0
    last_brow_position: object = None
    last_jaw_position: object = None
    left_open_reference: float = 0.0
    right_open_reference: float = 0.0
    mouth_closed_reference: object = None
    last_frame_index: int = -1
    left_transition_count: int = 0
    right_transition_count: int = 0

    def snapshot(self):
        return {
            "track_id": int(self.track_id),
            "left_eye_openness": float(self.left_eye_openness),
            "right_eye_openness": float(self.right_eye_openness),
            "raw_left_eye_openness": float(self.raw_left_eye_openness),
            "raw_right_eye_openness": float(self.raw_right_eye_openness),
            "blink_state": self.blink_state,
            "left_blink_state": self.left_blink_state,
            "right_blink_state": self.right_blink_state,
            "mouth_openness": float(self.mouth_openness),
            "mouth_aspect_ratio": float(self.mouth_aspect_ratio),
            "eyebrow_movement": float(self.eyebrow_movement),
            "jaw_movement": float(self.jaw_movement),
            "expression_confidence": float(self.expression_confidence),
            "last_frame_index": int(self.last_frame_index),
        }


class TemporalExpressionEngine:
    """Per-track, confidence-aware expression filtering and event gating."""

    def __init__(self, enabled=False, alpha=0.28, motion_alpha=0.82,
                 closed_ratio=0.48, open_ratio=0.70, event_strength=0.86,
                 cache_size=256):
        self.enabled = bool(enabled)
        self.alpha = _clip(alpha, 0.05, 0.8)
        self.motion_alpha = _clip(motion_alpha, self.alpha, 1.0)
        self.closed_ratio = _clip(closed_ratio, 0.20, 0.75)
        self.open_ratio = _clip(open_ratio, 0.55, 0.95)
        self.event_strength = _clip(event_strength)
        self.cache_size = max(64, int(cache_size))
        self.states = {}
        self.ordered = True
        self._lock = RLock()

    @classmethod
    def from_env(cls):
        def boolean(name, default=False):
            return str(os.environ.get(name, "1" if default else "0")).strip().lower() in (
                "1", "true", "yes", "on")

        def number(name, default):
            try:
                return float(os.environ.get(name, default) or default)
            except (TypeError, ValueError):
                return default

        def integer(name, default):
            try:
                return int(os.environ.get(name, default) or default)
            except (TypeError, ValueError):
                return default

        return cls(
            enabled=boolean("ROOP_TEMPORAL_EXPRESSION"),
            alpha=number("ROOP_TEMPORAL_EXPRESSION_ALPHA", 0.28),
            motion_alpha=number("ROOP_TEMPORAL_EXPRESSION_MOTION_ALPHA", 0.82),
            closed_ratio=number("ROOP_TEMPORAL_EXPRESSION_CLOSED_RATIO", 0.48),
            open_ratio=number("ROOP_TEMPORAL_EXPRESSION_OPEN_RATIO", 0.70),
            event_strength=number("ROOP_TEMPORAL_EXPRESSION_EVENT_STRENGTH", 0.86),
            cache_size=integer("ROOP_TEMPORAL_EXPRESSION_CACHE_SIZE", 256),
        )

    def reset(self):
        with self._lock:
            self.states.clear()

    def set_ordered(self, ordered):
        # Kept parallel to the other temporal engines.  The tracking replay is
        # the authority for order; this flag makes the contract explicit for
        # callers and future schedulers without adding a hot-loop branch.
        self.ordered = bool(ordered)

    def _state(self, track_id):
        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            return None
        state = self.states.get(track_id)
        if state is None:
            if len(self.states) >= self.cache_size:
                self.states.pop(next(iter(self.states)))
            state = self.states[track_id] = TemporalExpressionState(track_id)
        return state

    def _filter(self, previous, current, alpha, confidence):
        if previous is None:
            return float(current)
        delta = abs(float(current) - float(previous))
        # A large real expression change gets through immediately. Small
        # detector noise gets the low-pass treatment. Confidence only reduces
        # the noise response; it never prevents a genuine transition.
        adaptive = alpha + ((self.motion_alpha - alpha)
                            * min(1.0, delta / 0.16))
        adaptive = min(1.0, max(alpha, adaptive))
        if confidence < 0.35:
            adaptive *= 0.65
        return float((1.0 - adaptive) * previous + adaptive * current)

    def _eye_state(self, value, reference, previous, confidence):
        if reference <= 1e-5:
            # A first observation may itself be a closed eye.  A conservative
            # open-face prior lets that first blink/wink become an event while
            # later frames calibrate the actual person's aperture.
            reference = max(float(value), 0.28)
        # Slowly learn the open state only from frames that are not already
        # closing. This avoids teaching the filter that a blink is "normal".
        if value > reference * self.open_ratio:
            reference = 0.92 * reference + 0.08 * float(value)
        closed = reference * self.closed_ratio
        opened = reference * self.open_ratio
        if previous in ("closed", "closing"):
            return ("opening" if value >= opened else "closed"), reference
        if previous in ("open", "opening", "unknown"):
            return ("closing" if value <= closed else "open"), reference
        return previous, reference

    def update(self, track_id, frame_index, landmarks, kps=None, bbox=None,
               confidence=1.0, landmarks68=None):
        if not self.enabled:
            return None
        measurement = measure_expression(landmarks, kps=kps, bbox=bbox,
                                         landmarks68=landmarks68,
                                         detection_confidence=confidence)
        if measurement.get("confidence", 0.0) <= 0.0:
            return None
        with self._lock:
            state = self._state(track_id)
            if state is None:
                return None
            conf = _clip(measurement.get("confidence"))
            raw_l = float(measurement["left_eye_openness"])
            raw_r = float(measurement["right_eye_openness"])
            raw_m = float(measurement["mouth_openness"])
            first = state.last_frame_index < 0
            state.raw_left_eye_openness = raw_l
            state.raw_right_eye_openness = raw_r
            state.left_eye_openness = (raw_l if first else self._filter(
                state.left_eye_openness, raw_l, self.alpha, conf))
            state.right_eye_openness = (raw_r if first else self._filter(
                state.right_eye_openness, raw_r, self.alpha, conf))
            state.mouth_openness = (raw_m if first else self._filter(
                state.mouth_openness, raw_m, self.alpha, conf))
            state.mouth_aspect_ratio = state.mouth_openness
            left, state.left_open_reference = self._eye_state(
                raw_l, state.left_open_reference, state.left_blink_state, conf)
            right, state.right_open_reference = self._eye_state(
                raw_r, state.right_open_reference, state.right_blink_state, conf)
            state.left_blink_state = left
            state.right_blink_state = right
            if left in ("closing", "opening"):
                state.left_transition_count += 1
            else:
                state.left_transition_count = 0
            if right in ("closing", "opening"):
                state.right_transition_count += 1
            else:
                state.right_transition_count = 0
            if left == "closed" and right == "closed":
                state.blink_state = "closed"
            elif left == "closed" and right != "closed":
                state.blink_state = "wink_left"
            elif right == "closed" and left != "closed":
                state.blink_state = "wink_right"
            elif left == "closing" or right == "closing":
                state.blink_state = "closing"
            elif left == "opening" or right == "opening":
                state.blink_state = "opening"
            else:
                state.blink_state = "open"

            brow = measurement.get("brow_position")
            jaw = measurement.get("jaw_position")
            if state.last_brow_position is not None and brow is not None:
                state.eyebrow_movement = float(brow - state.last_brow_position)
            else:
                state.eyebrow_movement = 0.0
            if state.last_jaw_position is not None and jaw is not None:
                state.jaw_movement = float(jaw - state.last_jaw_position)
            else:
                state.jaw_movement = 0.0
            state.last_brow_position = brow
            state.last_jaw_position = jaw
            state.expression_confidence = float(
                conf if first else 0.72 * state.expression_confidence + 0.28 * conf)
            if state.mouth_closed_reference is None:
                state.mouth_closed_reference = raw_m
            elif raw_m < state.mouth_closed_reference:
                state.mouth_closed_reference = 0.92 * state.mouth_closed_reference + 0.08 * raw_m
            state.last_frame_index = int(frame_index)
            return state

    def plan(self, track_id):
        """Return regional target-preservation strengths for the current state."""
        if not self.enabled:
            return None
        with self._lock:
            state = self.states.get(int(track_id)) if track_id is not None else None
            if state is None or state.last_frame_index < 0:
                return None
            confidence = _clip(state.expression_confidence)
            left_event = state.left_blink_state in ("closing", "closed", "opening")
            right_event = state.right_blink_state in ("closing", "closed", "opening")
            left_low = state.left_eye_openness < state.left_open_reference * self.open_ratio
            right_low = state.right_eye_openness < state.right_open_reference * self.open_ratio
            eye_base = self.event_strength * confidence
            eye_strengths = (
                float(_clip(eye_base if left_event or left_low else 0.0)),
                float(_clip(eye_base if right_event or right_low else 0.0)),
            )
            mouth_base = _scalar(state.mouth_closed_reference, state.mouth_openness)
            mouth_event = (state.mouth_openness > max(0.12, mouth_base + 0.035)
                           or abs(state.jaw_movement) > 0.012
                           or abs(state.eyebrow_movement) > 0.012)
            return {
                "eye_strengths": eye_strengths,
                "mouth_strength": float(_clip(self.event_strength * confidence
                                                if mouth_event else 0.0)),
                "expression_confidence": float(confidence),
                "blink_state": state.blink_state,
            }

    def snapshot(self, track_id):
        with self._lock:
            state = self.states.get(int(track_id)) if track_id is not None else None
            return None if state is None else state.snapshot()
