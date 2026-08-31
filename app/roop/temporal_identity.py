"""Opt-in per-track temporal identity stabilization.

The existing temporal tracker owns detector scheduling and track lifecycle, while
the pose selector owns the meaning of a source-bank score.  This module is the
small bridge between them: it keeps one bounded state record per track and
applies confidence-aware, motion-aware hysteresis to the parts of a face that
should persist across frames.

It deliberately does not blur a whole face.  Output stabilization operates on
the low-frequency colour/identity field of an already aligned crop; the current
frame's high-frequency texture, eyes, mouth, and expression remain current.
The feature is disabled unless ``ROOP_TEMPORAL_IDENTITY=1``.
"""

import copy as _copy_module
from dataclasses import dataclass
import math
import os
from threading import RLock

import cv2
import numpy as np


def _copy(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    try:
        return np.asarray(value).copy()
    except Exception:
        return value


def _array(value, dtype=np.float32):
    if value is None:
        return None
    try:
        out = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return None
    if out.size == 0 or not np.all(np.isfinite(out)):
        return None
    return out.copy()


def _normalise(value):
    out = _array(value)
    if out is None or out.ndim != 1:
        return None
    norm = float(np.linalg.norm(out))
    return (out / norm).astype(np.float32) if norm > 1e-7 else None


def _angle_delta(current, previous):
    return (float(current) - float(previous) + 180.0) % 360.0 - 180.0


def _pose_dict(pose):
    if pose is None:
        return None
    if hasattr(pose, "as_dict"):
        try:
            pose = pose.as_dict()
        except Exception:
            return None
    if not isinstance(pose, dict):
        return None
    out = {}
    for key, value in pose.items():
        if isinstance(value, (int, float, np.number)):
            value = float(value)
        elif isinstance(value, np.ndarray):
            value = value.copy()
        out[key] = value
    return out


@dataclass
class TemporalIdentityState:
    """Bounded temporal state for one tracked physical face.

    ``previous_output`` is a canonical crop, never a full video frame.  This
    keeps memory proportional to face count and avoids retaining 4K buffers.
    """

    track_id: int
    source_identity: object = None
    selected_source_index: object = None
    identity_embedding: object = None
    target_embedding: object = None
    pose: object = None
    landmarks: object = None
    alignment_transform: object = None
    swap_confidence: float = 0.0
    output_face_confidence: float = 0.0
    previous_output: object = None
    previous_mask: object = None
    previous_detail: object = None
    previous_detail_source: object = None
    previous_lighting: object = None
    last_frame_index: int = -1
    motion: float = 0.0
    major_pose_transition: bool = False
    pending_source_index: object = None
    pending_source_count: int = 0
    pending_source_identity: object = None
    pending_source_identity_count: int = 0
    transition_from_index: object = None
    transition_remaining: int = 0
    transition_total: int = 0

    def snapshot(self):
        return {
            "track_id": int(self.track_id),
            "source_identity": self.source_identity,
            "selected_source_index": self.selected_source_index,
            "identity_embedding": _copy(self.identity_embedding),
            "target_embedding": _copy(self.target_embedding),
            "pose": _copy(self.pose),
            "landmarks": _copy(self.landmarks),
            "alignment_transform": _copy(self.alignment_transform),
            "swap_confidence": float(self.swap_confidence),
            "output_face_confidence": float(self.output_face_confidence),
            "previous_output": _copy(self.previous_output),
            "previous_mask": _copy(self.previous_mask),
            "previous_detail": _copy(self.previous_detail),
            "previous_detail_source": self.previous_detail_source,
            "previous_lighting": _copy(self.previous_lighting),
            "last_frame_index": int(self.last_frame_index),
            "motion": float(self.motion),
            "major_pose_transition": bool(self.major_pose_transition),
            "pending_source_index": self.pending_source_index,
            "pending_source_count": int(self.pending_source_count),
            "pending_source_identity": self.pending_source_identity,
            "pending_source_identity_count": int(self.pending_source_identity_count),
            "transition_from_index": self.transition_from_index,
            "transition_remaining": int(self.transition_remaining),
            "transition_total": int(self.transition_total),
        }


class TemporalIdentityStabilizer:
    """Per-track state, confidence weighting, and source transition control."""

    def __init__(self, enabled=False, switch_frames=3, transition_frames=4,
                 geometry_alpha=0.35, output_strength=0.35, mask_strength=0.45,
                 major_yaw=32.0, major_pitch=24.0, major_roll=30.0,
                 cache_size=256, lowpass_size=128):
        self.enabled = bool(enabled)
        self.switch_frames = max(1, int(switch_frames))
        self.transition_frames = max(1, int(transition_frames))
        self.geometry_alpha = min(1.0, max(0.01, float(geometry_alpha)))
        self.output_strength = min(1.0, max(0.0, float(output_strength)))
        self.mask_strength = min(1.0, max(0.0, float(mask_strength)))
        self.major_yaw = abs(float(major_yaw))
        self.major_pitch = abs(float(major_pitch))
        self.major_roll = abs(float(major_roll))
        self.cache_size = max(64, int(cache_size))
        # The identity correction is deliberately low-frequency.  Keep the
        # expensive Gaussian work bounded to a smaller working crop while
        # restoring the correction at the original crop size.  ``0`` is the
        # full-resolution reference path used by the equivalence tests and
        # remains available for diagnostics.
        self.lowpass_size = max(0, int(lowpass_size))
        self.states = {}
        self.ordered = True
        self._lock = RLock()

    @classmethod
    def from_env(cls):
        def _bool(name, default=False):
            return str(os.environ.get(name, "1" if default else "0")).strip().lower() in (
                "1", "true", "yes", "on")

        def _int(name, default):
            try:
                return int(os.environ.get(name, default) or default)
            except (TypeError, ValueError):
                return default

        def _float(name, default):
            try:
                return float(os.environ.get(name, default) or default)
            except (TypeError, ValueError):
                return default

        return cls(
            enabled=_bool("ROOP_TEMPORAL_IDENTITY", False),
            switch_frames=_int("ROOP_TEMPORAL_IDENTITY_SWITCH_FRAMES", 3),
            transition_frames=_int("ROOP_TEMPORAL_IDENTITY_TRANSITION_FRAMES", 4),
            geometry_alpha=_float("ROOP_TEMPORAL_IDENTITY_GEOMETRY_ALPHA", 0.35),
            output_strength=_float("ROOP_TEMPORAL_IDENTITY_OUTPUT_STRENGTH", 0.35),
            mask_strength=_float("ROOP_TEMPORAL_IDENTITY_MASK_STRENGTH", 0.45),
            major_yaw=_float("ROOP_TEMPORAL_IDENTITY_MAJOR_YAW", 32.0),
            major_pitch=_float("ROOP_TEMPORAL_IDENTITY_MAJOR_PITCH", 24.0),
            major_roll=_float("ROOP_TEMPORAL_IDENTITY_MAJOR_ROLL", 30.0),
            cache_size=_int("ROOP_TEMPORAL_IDENTITY_CACHE_SIZE", 256),
            lowpass_size=_int("ROOP_TEMPORAL_IDENTITY_LOWPASS_SIZE", 128),
        )

    def set_ordered(self, ordered):
        self.ordered = bool(ordered)

    def warmup_frames(self, eps=0.01):
        """Frames a fresh block must discard before its output history is sound.

        Asked of the recurrences rather than hardcoded, so the parallel-block
        geometry tracks the user's strength settings the way every other
        stabilizer here already does.

        Two recurrences carry a seed across frames, and the SLOWEST to forget
        sets the boundary:

        * `stabilize_mask` is `out = (1-a)*previous + a*current` with
          `a = mask_strength * (0.60 + 0.40*(1-confidence))`. Confidence only
          ever RAISES `a`, so a fully-confident track -- `mask_strength * 0.60`
          -- is the worst case. The `entering` branch doubles `a`, which is
          faster still and therefore not binding.
        * `blend_output` retains `prior_weight` of the previous low band, and
          `prior_weight` is bounded above by `output_strength` once a source
          transition has finished. (During a transition it reaches 1.0 for at
          most `transition_frames`, which is a bounded event, not a sustained
          filter, and no finite warm-up can cover a factor of 1.0 anyway.)
        """
        from roop.one_euro import ema_warmup_frames
        mask_alpha = max(0.0, min(1.0, self.mask_strength * 0.60))
        # `output_strength` is the RETAINED weight, so the EMA factor -- the
        # fraction of the current frame admitted -- is its complement.
        output_alpha = max(0.0, min(1.0, 1.0 - self.output_strength))
        return max(ema_warmup_frames(mask_alpha, eps),
                   ema_warmup_frames(output_alpha, eps))

    def clone_for_block(self):
        """An instance for one contiguous parallel block: same configuration and
        the same prepass-derived identity, with its own OUTPUT HISTORY.

        The split matters and is not arbitrary. `update_geometry`, `update_pose`,
        `propose_identity` and `propose_source` are called from the tracking
        pre-pass, which runs sequentially over the whole clip and is finished
        before any block starts; everything they wrote is read-only here, so it
        is carried in. The swap phase mutates exactly three fields --
        `previous_output`, `previous_mask` and `swap_confidence` -- and those are
        per-frame history, so they are cleared and re-primed from the block's own
        warm-up frames. Sharing them would let two blocks advance one track's
        history out of order, which is the whole reason this path used to be
        pinned to a single worker.

        `copy.copy` rather than a re-listed constructor call on purpose: a
        parameter added later is carried automatically instead of being silently
        dropped by a copy that drifted from `__init__`.
        """
        clone = _copy_module.copy(self)
        clone._lock = RLock()
        clone.states = {}
        for track_id, state in self.states.items():
            fresh = _copy_module.copy(state)
            fresh.previous_output = None
            fresh.previous_mask = None
            fresh.previous_detail = None
            fresh.previous_detail_source = None
            fresh.swap_confidence = 0.0
            clone.states[track_id] = fresh
        return clone

    def reset(self):
        with self._lock:
            self.states.clear()

    def _state(self, track_id):
        if track_id is None:
            return None
        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            return None
        state = self.states.get(track_id)
        if state is None:
            state = self.states[track_id] = TemporalIdentityState(track_id)
        return state

    @staticmethod
    def _confidence(value, default=0.0):
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return default

    def update_geometry(self, track_id, frame_index, landmarks=None,
                        target_embedding=None, confidence=0.0,
                        source_identity=None, identity_embedding=None):
        """Filter identity/landmark observations and return the live state."""
        if not self.enabled:
            return None
        with self._lock:
            state = self._state(track_id)
            if state is None:
                return None
            current_lm = _array(landmarks)
            det_conf = self._confidence(confidence)
            if source_identity is not None and state.source_identity is None:
                state.source_identity = int(source_identity)
            source_emb = _normalise(identity_embedding)
            if source_emb is not None:
                state.identity_embedding = source_emb

            if current_lm is not None:
                if state.landmarks is None or np.shape(state.landmarks) != np.shape(current_lm):
                    state.landmarks = current_lm
                    state.motion = 0.0
                else:
                    size = max(1.0, float(np.linalg.norm(
                        np.ptp(state.landmarks[..., :2], axis=0))))
                    displacement = float(np.mean(np.linalg.norm(
                        current_lm[..., :2] - state.landmarks[..., :2], axis=-1))) / size
                    state.motion = min(1.0, displacement)
                    # Low confidence observations inherit more prior geometry;
                    # fast genuine motion gets a larger current-frame alpha.
                    alpha = self.geometry_alpha + 0.50 * state.motion
                    alpha = min(1.0, max(0.08, alpha + 0.35 * det_conf))
                    state.landmarks = ((1.0 - alpha) * state.landmarks
                                       + alpha * current_lm).astype(np.float32)

            emb = _normalise(target_embedding)
            if emb is not None:
                if state.target_embedding is None or np.shape(state.target_embedding) != np.shape(emb):
                    state.target_embedding = emb
                else:
                    alpha = min(1.0, max(0.08, 0.18 + 0.55 * det_conf
                                         + 0.40 * state.motion))
                    state.target_embedding = ((1.0 - alpha) * state.target_embedding
                                               + alpha * emb).astype(np.float32)
                    state.target_embedding = _normalise(state.target_embedding)
            state.output_face_confidence = float(
                0.70 * state.output_face_confidence + 0.30 * det_conf
                if state.last_frame_index >= 0 else det_conf)
            state.last_frame_index = int(frame_index)
            return state

    def update_pose(self, track_id, pose):
        """Filter pose angles while retaining the current expression fields."""
        if not self.enabled:
            return None
        with self._lock:
            state = self._state(track_id)
            current = _pose_dict(pose)
            if state is None or current is None:
                return state
            previous = _pose_dict(state.pose)
            if previous is None:
                state.pose = current
                state.major_pose_transition = False
                return state
            yaw_delta = abs(_angle_delta(current.get("yaw", 0.0), previous.get("yaw", 0.0)))
            pitch_delta = abs(float(current.get("pitch", 0.0)) - float(previous.get("pitch", 0.0)))
            roll_delta = abs(_angle_delta(current.get("roll", 0.0), previous.get("roll", 0.0)))
            state.major_pose_transition = (
                yaw_delta >= self.major_yaw or pitch_delta >= self.major_pitch
                or roll_delta >= self.major_roll)
            alpha = min(1.0, max(0.12, self.geometry_alpha + 0.50 * state.motion))
            blended = dict(current)
            for key, delta in (("yaw", yaw_delta), ("pitch", pitch_delta), ("roll", roll_delta)):
                if key not in current or key not in previous:
                    continue
                if key in ("yaw", "roll"):
                    blended[key] = float(previous[key]) + alpha * _angle_delta(
                        current[key], previous[key])
                else:
                    blended[key] = (1.0 - alpha) * float(previous[key]) + alpha * float(current[key])
            if "confidence" in current and "confidence" in previous:
                blended["confidence"] = max(float(current["confidence"]),
                                             (1.0 - alpha) * float(previous["confidence"])
                                             + alpha * float(current["confidence"]))
            state.pose = blended
            return state

    def propose_identity(self, track_id, candidate_identity, major_pose=False):
        """Keep a track's source identity stable across weak reassignment hints.

        The whole-clip tracker normally supplies one durable source binding. This
        second guard is for recovered/fractured tracks and future per-frame
        identity candidates: an ordinary candidate must persist for the same
        hysteresis window as a bank entry, while a major pose transition may
        commit immediately. It never changes the source-bank entry itself.
        """
        if not self.enabled or candidate_identity is None:
            return candidate_identity
        with self._lock:
            state = self._state(track_id)
            if state is None:
                return candidate_identity
            try:
                candidate_identity = int(candidate_identity)
            except (TypeError, ValueError):
                return state.source_identity
            if state.source_identity is None:
                state.source_identity = candidate_identity
                state.pending_source_identity = None
                state.pending_source_identity_count = 0
            elif candidate_identity == state.source_identity:
                state.pending_source_identity = None
                state.pending_source_identity_count = 0
            elif bool(major_pose):
                state.source_identity = candidate_identity
                state.pending_source_identity = None
                state.pending_source_identity_count = 0
            else:
                if state.pending_source_identity == candidate_identity:
                    state.pending_source_identity_count += 1
                else:
                    state.pending_source_identity = candidate_identity
                    state.pending_source_identity_count = 1
                if state.pending_source_identity_count >= self.switch_frames:
                    state.source_identity = candidate_identity
                    state.pending_source_identity = None
                    state.pending_source_identity_count = 0
            return state.source_identity

    def update_alignment(self, track_id, transform):
        if not self.enabled:
            return None
        with self._lock:
            state = self._state(track_id)
            matrix = _array(transform, np.float32)
            if state is not None and matrix is not None and matrix.shape == (2, 3):
                # Alignment is stabilized by filtered landmarks before align_crop
                # runs. Keep the exact transform used for this frame for audit and
                # future consumers; do not blend a matrix after the crop is made.
                state.alignment_transform = matrix
            return state

    def update_lighting(self, track_id, lighting):
        if not self.enabled or lighting is None:
            return None
        with self._lock:
            state = self._state(track_id)
            if state is None:
                return None
            current = _copy(lighting)
            if state.previous_lighting is None:
                state.previous_lighting = current
            elif isinstance(current, dict) and isinstance(state.previous_lighting, dict):
                merged = dict(current)
                for key, value in current.items():
                    old = state.previous_lighting.get(key)
                    if isinstance(value, (int, float)) and isinstance(old, (int, float)):
                        merged[key] = 0.25 * float(value) + 0.75 * float(old)
                state.previous_lighting = merged
            else:
                state.previous_lighting = current
            return state.previous_lighting

    def propose_source(self, track_id, candidate_index, major_pose=None):
        """Commit a bank entry only after persistence or a major pose change.

        Returns ``(selected_index, transition_alpha)``.  The alpha is the
        current-source weight for a representation transition and is stored on
        each replayed face, so parallel main-pass workers never mutate this
        ordered decision state.
        """
        if not self.enabled or candidate_index is None:
            return candidate_index, 1.0
        with self._lock:
            state = self._state(track_id)
            if state is None:
                return candidate_index, 1.0
            candidate_index = int(candidate_index)
            if major_pose is None:
                major_pose = state.major_pose_transition
            if state.selected_source_index is None:
                state.selected_source_index = candidate_index
                state.pending_source_index = None
                state.pending_source_count = 0
            elif candidate_index == state.selected_source_index:
                state.pending_source_index = None
                state.pending_source_count = 0
            elif bool(major_pose):
                self._commit_source(state, candidate_index)
            else:
                if state.pending_source_index == candidate_index:
                    state.pending_source_count += 1
                else:
                    state.pending_source_index = candidate_index
                    state.pending_source_count = 1
                if state.pending_source_count >= self.switch_frames:
                    self._commit_source(state, candidate_index)
            return state.selected_source_index, self._transition_alpha(state)

    def _commit_source(self, state, candidate_index):
        old = state.selected_source_index
        state.selected_source_index = int(candidate_index)
        state.pending_source_index = None
        state.pending_source_count = 0
        if old is not None and int(old) != int(candidate_index):
            state.transition_from_index = int(old)
            state.transition_total = self.transition_frames
            state.transition_remaining = self.transition_frames

    @staticmethod
    def _transition_alpha(state):
        if state.transition_remaining <= 0 or state.transition_total <= 0:
            return 1.0
        alpha = 1.0 - (float(state.transition_remaining - 1)
                       / float(state.transition_total))
        state.transition_remaining -= 1
        if state.transition_remaining <= 0:
            state.transition_from_index = None
            state.transition_total = 0
        return min(1.0, max(0.0, alpha))

    def stabilize_mask(self, track_id, mask, confidence=0.0):
        """Damp mask-edge noise, revealing new occluders faster than they leave."""
        if not self.enabled or mask is None:
            return mask
        with self._lock:
            state = self._state(track_id)
            # ``mask`` is consumed read-only below.  _array() always copies,
            # which made this hot path copy both the input and the already
            # owned history on every face/frame.
            try:
                current = np.asarray(mask, dtype=np.float32)
            except (TypeError, ValueError):
                current = None
            if (current is not None and
                    (current.size == 0 or not np.all(np.isfinite(current)))):
                current = None
            if state is None or current is None:
                return mask
            previous = state.previous_mask
            if previous is not None:
                try:
                    previous = np.asarray(previous, dtype=np.float32)
                except (TypeError, ValueError):
                    previous = None
            if previous is None or previous.shape != current.shape:
                state.previous_mask = current.copy()
                return current.copy()
            alpha = self.mask_strength * (0.60 + 0.40 * (1.0 - self._confidence(confidence)))
            # Mask convention: 1 restores the original. Let an entering
            # occluder reveal quickly, but keep the return path smooth.
            delta = current - previous
            out = previous + alpha * delta
            entering = delta > 0.0
            out[entering] = (max(alpha, min(1.0, alpha * 2.0)) * current[entering]
                             + (1.0 - max(alpha, min(1.0, alpha * 2.0)))
                             * previous[entering])
            state.previous_mask = np.clip(out, 0.0, 1.0).astype(np.float32)
            return state.previous_mask.copy()

    @staticmethod
    def _resize_like(image, shape):
        if image is None:
            return None
        if tuple(image.shape[:2]) == tuple(shape[:2]):
            return image.astype(np.float32, copy=False)
        return cv2.resize(image, (int(shape[1]), int(shape[0])),
                          interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def blend_output(self, track_id, output, confidence=0.0,
                     transition_alpha=1.0, motion=None):
        """Blend only aligned low-frequency output components, never whole detail.

        The correction is intentionally computed on a reduced working crop by
        default.  This preserves the current frame's high-frequency residual
        while avoiding two full-resolution Gaussian passes per face/frame.
        Set ``lowpass_size=0`` for the old full-resolution reference path.
        """
        if not self.enabled or output is None:
            return output
        with self._lock:
            state = self._state(track_id)
            try:
                current = np.asarray(output, dtype=np.float32)
            except (TypeError, ValueError):
                current = None
            if (current is not None and
                    (current.size == 0 or not np.all(np.isfinite(current)))):
                current = None
            if state is None or current is None or current.ndim != 3:
                return output
            previous = state.previous_output
            if previous is None or previous.shape != current.shape:
                previous_for_blend = None
            else:
                previous_for_blend = previous
            if previous_for_blend is None and state.previous_output is not None:
                try:
                    previous_for_blend = np.asarray(state.previous_output,
                                                    dtype=np.float32)
                except (TypeError, ValueError):
                    previous_for_blend = None
            if previous_for_blend is None:
                result = current
            else:
                motion_value = state.motion if motion is None else float(motion)
                motion_release = min(1.0, max(0.0, motion_value * 2.5))
                conf = self._confidence(confidence)
                base = self.output_strength * (1.0 - motion_release)
                base *= 0.35 + 0.65 * (1.0 - conf)
                transition_weight = min(1.0, max(0.0, 1.0 - float(transition_alpha)))
                prior_weight = max(base, transition_weight)
                # Low-pass only: eyes, mouth, pores, and expression live in the
                # residual and are taken from the current crop unchanged.
                work_h, work_w = current.shape[:2]
                work_scale = 1.0
                if self.lowpass_size > 0 and max(work_h, work_w) > self.lowpass_size:
                    work_scale = float(self.lowpass_size) / float(max(work_h, work_w))
                    work_w = max(1, int(round(work_w * work_scale)))
                    work_h = max(1, int(round(work_h * work_scale)))
                    current_work = cv2.resize(
                        current, (work_w, work_h), interpolation=cv2.INTER_AREA)
                    previous_work = cv2.resize(
                        previous_for_blend, (work_w, work_h),
                        interpolation=cv2.INTER_LINEAR)
                else:
                    current_work = current
                    previous_work = self._resize_like(
                        previous_for_blend, current.shape)
                sigma = max(0.5, 4.0 * work_scale)
                previous_low = cv2.GaussianBlur(previous_work, (0, 0), sigma)
                current_low = cv2.GaussianBlur(current_work, (0, 0), sigma)
                correction = previous_low - current_low
                if correction.shape[:2] != current.shape[:2]:
                    correction = cv2.resize(
                        correction, (current.shape[1], current.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
                result = current + prior_weight * correction
            result = np.clip(result, 0.0, 255.0).astype(np.uint8)
            if result.shape[:2] != (self.cache_size, self.cache_size):
                state.previous_output = cv2.resize(
                    result, (self.cache_size, self.cache_size),
                    interpolation=cv2.INTER_AREA)
            else:
                state.previous_output = result.copy()
            state.swap_confidence = self._confidence(confidence)
            return result

    def record_output_confidence(self, track_id, confidence):
        if not self.enabled:
            return None
        with self._lock:
            state = self._state(track_id)
            if state is not None:
                state.swap_confidence = self._confidence(confidence)
            return state

    def blend_detail(self, track_id, detail, confidence=0.0, motion=0.0,
                     source_index=None, transition_alpha=1.0):
        """Temporally damp the canonical identity-detail residual.

        This is intentionally separate from ``blend_output``: the latter
        preserves only low-frequency colour/identity fields, while this state
        smooths the small source-detail residual after enhancers and merger
        operations have finished. A source switch never ghosts the old person's
        marks into the new source; the transition alpha gates the old history.
        """
        if not self.enabled or detail is None:
            return detail
        with self._lock:
            state = self._state(track_id)
            try:
                current = np.asarray(detail, dtype=np.float32)
            except (TypeError, ValueError):
                current = None
            if (state is None or current is None or current.size == 0
                    or not np.isfinite(current).all() or current.ndim != 2):
                return detail
            previous = state.previous_detail
            same_source = (source_index is None
                            or state.previous_detail_source is None
                            or int(source_index) == int(state.previous_detail_source))
            if previous is None or not same_source:
                result = current
            else:
                try:
                    previous = np.asarray(previous, dtype=np.float32)
                except (TypeError, ValueError):
                    previous = None
                if previous is None:
                    result = current
                else:
                    if previous.shape != current.shape:
                        previous = cv2.resize(
                            previous, (current.shape[1], current.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
                    motion_value = min(1.0, max(0.0, float(motion)))
                    conf = self._confidence(confidence)
                    prior = self.output_strength * (1.0 - motion_value)
                    prior *= 0.25 + 0.50 * (1.0 - conf)
                    prior *= min(1.0, max(0.0, float(transition_alpha)))
                    result = (1.0 - prior) * current + prior * previous
            # Detail is always a bounded small map in the caller, but retaining
            # a downsampled copy keeps this state safe if a custom caller sends
            # a larger crop.
            if max(result.shape) > 64:
                state.previous_detail = cv2.resize(
                    result, (64, 64), interpolation=cv2.INTER_AREA)
            else:
                state.previous_detail = result.copy()
            state.previous_detail_source = source_index
            return result.astype(np.float32)

    def snapshot(self, track_id):
        with self._lock:
            state = self._state(track_id)
            return None if state is None else state.snapshot()


__all__ = ["TemporalIdentityState", "TemporalIdentityStabilizer"]
