"""Opt-in temporal occlusion reasoning for tracked face crops.

Mask processors in this project use ``1 = restore the original``.  This module
keeps that convention and adds a causal per-track record around it.  It does
not add a segmentation model: ordinary frames use the configured mask engine,
stable occlusion frames propagate the last trusted mask, and a change in the
occlusion evidence re-enters the configured engine.
"""

from dataclasses import dataclass
import os
from threading import RLock

import cv2
import numpy as np


def _array(value):
    if value is None:
        return None
    try:
        out = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if out.size == 0 or not np.all(np.isfinite(out)):
        return None
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]
    if out.ndim != 2:
        return None
    return np.clip(out, 0.0, 1.0).copy()


def _confidence(value, default=0.0):
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _resize(mask, shape):
    mask = _array(mask)
    if mask is None:
        return None
    if mask.shape == tuple(shape[:2]):
        return mask
    return cv2.resize(mask, (int(shape[1]), int(shape[0])),
                      interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _observation(value):
    """Make a small host-side observation for event detection only."""
    try:
        image = np.asarray(value)
        if image.ndim == 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        return image.astype(np.float32) / 255.0
    except Exception:
        return None


def build_face_support(landmarks=None, kps=None, matrix=None, shape=None):
    """Return a soft 0..1 face support in aligned-crop coordinates.

    The mask processors return a *restore* mask, while this support only says
    where a track is allowed to own pixels.  Keeping the two separate is what
    lets an occluder restore the plate without temporally blurring the swapped
    face itself.  ``matrix`` maps full-frame points into the aligned crop.
    """
    if shape is None or matrix is None:
        return None
    try:
        h, w = int(shape[0]), int(shape[1])
        points = landmarks if landmarks is not None else kps
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < 3:
            return None
        M = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
        crop_points = points @ M[:, :2].T + M[:, 2]
        crop_points = crop_points[np.all(np.isfinite(crop_points), axis=1)]
        if crop_points.shape[0] < 3:
            return None
        hull = cv2.convexHull(crop_points.astype(np.float32)).astype(np.int32)
        support = np.zeros((h, w), dtype=np.float32)
        cv2.fillConvexPoly(support, hull, 1.0)
        # A small feather stabilizes rasterisation at the boundary only. It is
        # not a temporal blur and does not touch pixels outside the face hull.
        k = max(1, int(round(min(h, w) * 0.01)) | 1)
        if k > 1:
            support = cv2.GaussianBlur(support, (k, k), 0)
        return np.clip(support, 0.0, 1.0)
    except Exception:
        return None


@dataclass
class OcclusionDecision:
    mode: str
    mask: object = None
    reason: str = ""


@dataclass
class TemporalOcclusionState:
    """Per-track occlusion state; masks are canonical crop-sized arrays."""

    track_id: int
    face_mask: object = None
    visible_face_mask: object = None
    occlusion_mask: object = None
    previous_mask: object = None
    predicted_mask: object = None
    confidence: float = 0.0
    previous_observation: object = None
    event: str = "normal"
    event_confidence: float = 0.0
    event_count: int = 0
    stable_frames: int = 0
    last_frame_index: int = -1
    last_analysis_frame: int = -1
    analysis_mode: str = "normal"
    interaction_score: float = 0.0
    other_track_ids: object = None

    def snapshot(self):
        def copy(value):
            if value is None:
                return None
            if isinstance(value, list):
                return list(value)
            try:
                return np.asarray(value).copy()
            except Exception:
                return value

        return {
            "track_id": int(self.track_id),
            "face_mask": copy(self.face_mask),
            "visible_face_mask": copy(self.visible_face_mask),
            "occlusion_mask": copy(self.occlusion_mask),
            "previous_mask": copy(self.previous_mask),
            "predicted_mask": copy(self.predicted_mask),
            "confidence": float(self.confidence),
            "previous_observation": copy(self.previous_observation),
            "event": self.event,
            "event_confidence": float(self.event_confidence),
            "event_count": int(self.event_count),
            "stable_frames": int(self.stable_frames),
            "last_frame_index": int(self.last_frame_index),
            "last_analysis_frame": int(self.last_analysis_frame),
            "analysis_mode": self.analysis_mode,
            "interaction_score": float(self.interaction_score),
            "other_track_ids": list(self.other_track_ids or []),
        }


class TemporalOcclusionEngine:
    """Causal event detector and mask propagator for one or more face tracks."""

    def __init__(self, enabled=False, event_threshold=0.12,
                 interaction_threshold=0.08, stable_frames=3,
                 refresh_frames=5, appearance_threshold=0.16,
                 leave_alpha=0.35, enter_alpha=0.90, cache_size=256):
        self.enabled = bool(enabled)
        self.event_threshold = min(1.0, max(0.01, float(event_threshold)))
        self.interaction_threshold = min(1.0, max(0.0, float(interaction_threshold)))
        self.stable_frames = max(1, int(stable_frames))
        self.refresh_frames = max(1, int(refresh_frames))
        self.appearance_threshold = min(1.0, max(0.01, float(appearance_threshold)))
        self.leave_alpha = min(1.0, max(0.01, float(leave_alpha)))
        self.enter_alpha = min(1.0, max(self.leave_alpha, float(enter_alpha)))
        self.cache_size = max(64, int(cache_size))
        self.states = {}
        self.ordered = True
        self._lock = RLock()

    @classmethod
    def from_env(cls):
        def boolean(name, default=False):
            return str(os.environ.get(name, "1" if default else "0")).strip().lower() in (
                "1", "true", "yes", "on")

        def number(name, default, convert=float):
            try:
                return convert(os.environ.get(name, default) or default)
            except (TypeError, ValueError):
                return default

        return cls(
            enabled=boolean("ROOP_TEMPORAL_OCCLUSION", False),
            event_threshold=number("ROOP_OCCLUSION_EVENT_THRESHOLD", 0.12),
            interaction_threshold=number("ROOP_OCCLUSION_INTERACTION_THRESHOLD", 0.08),
            stable_frames=number("ROOP_OCCLUSION_STABLE_FRAMES", 3, int),
            refresh_frames=number("ROOP_OCCLUSION_REFRESH_FRAMES", 5, int),
            appearance_threshold=number("ROOP_OCCLUSION_APPEARANCE_THRESHOLD", 0.16),
            leave_alpha=number("ROOP_OCCLUSION_LEAVE_ALPHA", 0.35),
            enter_alpha=number("ROOP_OCCLUSION_ENTER_ALPHA", 0.90),
            cache_size=number("ROOP_OCCLUSION_CACHE_SIZE", 256, int),
        )

    def set_ordered(self, ordered):
        self.ordered = bool(ordered)

    def reset(self):
        with self._lock:
            self.states.clear()

    def _state(self, track_id):
        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            return None
        state = self.states.get(track_id)
        if state is None:
            state = self.states[track_id] = TemporalOcclusionState(track_id)
        return state

    @staticmethod
    def _fraction(mask, face_mask):
        face = _array(face_mask)
        current = _resize(mask, face.shape) if face is not None else _array(mask)
        if face is None or current is None:
            return 0.0
        denom = float(np.sum(face))
        return float(np.sum(face * current) / max(1.0, denom))

    def prepare(self, track_id, frame_index, face_mask, observation=None,
                confidence=0.0, motion=0.0, interaction_score=0.0,
                other_track_ids=None):
        """Choose normal mask inference or cheap stable-occlusion propagation."""
        if not self.enabled or not self.ordered:
            return OcclusionDecision("disabled", None, "disabled")
        with self._lock:
            state = self._state(track_id)
            support = _array(face_mask)
            if state is None or support is None:
                return OcclusionDecision("analyze", None, "missing_face_support")
            state.face_mask = support
            state.confidence = (0.75 * state.confidence
                                + 0.25 * _confidence(confidence))
            state.interaction_score = _confidence(interaction_score)
            state.other_track_ids = list(other_track_ids or [])
            current_obs = _observation(observation)
            change = 0.0
            if current_obs is not None and state.previous_observation is not None:
                change = float(np.mean(np.abs(current_obs - state.previous_observation)))
            state.previous_observation = current_obs

            previous_fraction = self._fraction(state.previous_mask, support)
            stable_occlusion = (state.event == "stable"
                                and state.stable_frames >= self.stable_frames
                                and previous_fraction >= self.event_threshold)
            refresh_due = (state.last_analysis_frame < 0
                           or int(frame_index) - state.last_analysis_frame >= self.refresh_frames)
            motion_event = _confidence(motion) >= 0.70
            if stable_occlusion and not refresh_due and not motion_event \
                    and change < self.appearance_threshold:
                source = (state.predicted_mask if state.predicted_mask is not None
                          else state.previous_mask)
                predicted = _resize(source, support.shape)
                if predicted is not None:
                    state.analysis_mode = "cheap_propagation"
                    return OcclusionDecision("propagate", predicted,
                                             "occlusion_stable")
            reason = "initial_analysis" if state.last_analysis_frame < 0 else "normal_analysis"
            if stable_occlusion and (refresh_due or motion_event
                                     or change >= self.appearance_threshold):
                reason = "occlusion_event_reanalysis"
            state.analysis_mode = "enhanced" if reason == "occlusion_event_reanalysis" else "normal"
            return OcclusionDecision("analyze", None, reason)

    def observe(self, track_id, frame_index, face_mask, restore_mask,
                confidence=0.0, motion=0.0, interaction_score=0.0,
                other_track_ids=None, analysis_mode="normal"):
        """Consume a newly analyzed restore mask and return its causal result."""
        if not self.enabled:
            return restore_mask
        with self._lock:
            state = self._state(track_id)
            support = _array(face_mask)
            current = _resize(restore_mask, support.shape) if support is not None else _array(restore_mask)
            if state is None or support is None or current is None:
                return restore_mask
            previous = _resize(state.previous_mask, current.shape)
            previous_fraction = self._fraction(previous, support)
            current_occ = np.clip(support * current, 0.0, 1.0)
            current_fraction = float(np.sum(current_occ) / max(1.0, np.sum(support)))
            interaction = _confidence(interaction_score)
            confidence_value = _confidence(confidence)
            retained_previous = (previous * (0.5 + 0.5 * confidence_value)
                                 if previous is not None else None)
            has_event = (current_fraction >= self.event_threshold
                         or (interaction >= self.interaction_threshold
                             and current_fraction >= self.event_threshold * 0.5))
            entering = has_event and previous_fraction < self.event_threshold
            leaving = (previous_fraction >= self.event_threshold
                       and current_fraction < self.event_threshold)
            if entering:
                state.event = "entering"
                state.event_count += 1
                state.stable_frames = 0
                # Trust the event mask immediately so the object's original
                # pixels are not painted over by the swap.
                result = np.maximum(current, previous * self.enter_alpha) if previous is not None else current
            elif leaving:
                state.event = "leaving"
                state.event_count += 1
                state.stable_frames = 0
                # Restore the swapped region over several frames; never pop it
                # back in on the first frame after an object exits.
                result = np.maximum(current, previous * (1.0 - self.leave_alpha)) \
                    if previous is not None else current
            elif has_event:
                state.event = "stable"
                state.stable_frames += 1
                # Re-analysis is authoritative for a moving object. A maximum
                # retains a trusted object pixel when the fresh matte is weak.
                result = (np.maximum(current, retained_previous)
                          if retained_previous is not None else current)
            else:
                state.event = "normal"
                state.stable_frames = 0
                result = current

            result = np.clip(result, 0.0, 1.0).astype(np.float32)
            state.face_mask = support.copy()
            state.occlusion_mask = np.clip(support * result, 0.0, 1.0)
            state.visible_face_mask = np.clip(support * (1.0 - result), 0.0, 1.0)
            state.previous_mask = result.copy()
            state.predicted_mask = result.copy()
            state.confidence = (0.70 * state.confidence
                                + 0.30 * _confidence(confidence))
            state.event_confidence = min(
                1.0, max(current_fraction / max(self.event_threshold, 1e-6), interaction)
                * (0.5 + 0.5 * _confidence(confidence)))
            state.last_frame_index = int(frame_index)
            state.last_analysis_frame = int(frame_index)
            state.analysis_mode = analysis_mode
            state.interaction_score = interaction
            state.other_track_ids = list(other_track_ids or [])
            return result

    def propagate(self, track_id, frame_index, decision, confidence=0.0):
        """Advance a stable event without running a segmentation model."""
        if not self.enabled or decision is None or decision.mask is None:
            return None
        with self._lock:
            state = self._state(track_id)
            mask = _array(decision.mask)
            if state is None or mask is None:
                return None
            support = _resize(state.face_mask, mask.shape)
            if support is None:
                support = np.ones_like(mask)
            state.face_mask = support
            state.occlusion_mask = np.clip(support * mask, 0.0, 1.0)
            state.visible_face_mask = np.clip(support * (1.0 - mask), 0.0, 1.0)
            state.previous_mask = mask.copy()
            state.predicted_mask = mask.copy()
            state.confidence = (0.85 * state.confidence
                                + 0.15 * _confidence(confidence))
            state.stable_frames += 1
            state.event = "stable"
            state.analysis_mode = "cheap_propagation"
            state.last_frame_index = int(frame_index)
            return mask.copy()

    def snapshot(self, track_id):
        with self._lock:
            state = self._state(track_id)
            return None if state is None else state.snapshot()


__all__ = ["OcclusionDecision", "TemporalOcclusionState",
           "TemporalOcclusionEngine", "build_face_support"]
