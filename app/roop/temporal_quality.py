"""Lightweight event-driven temporal quality control.

The controller deliberately observes values that the swap pipeline already has
(``matrix``, masks, aligned crops, appearance metrics and enhancer output).
It does not run a detector, restorer, or optical-flow pass.  A normal frame is
therefore a cheap pass-through; corrections are only requested for an event.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
import copy
import os
from threading import RLock

import cv2
import numpy as np


_ANOMALIES = (
    "identity_drift", "mask_popping", "face_brightness_jump",
    "skin_color_jump", "geometry_jump", "enhancer_hallucination",
    "detail_disappearance", "eye_state_discontinuity",
    "jawline_movement_discontinuity", "face_flicker",
)

_CORRECTIONS = {
    "identity_drift": ("reselect_source",),
    "mask_popping": ("reblend_previous_mask", "rerun_occlusion_analysis"),
    "face_brightness_jump": ("reuse_stable_color",),
    "skin_color_jump": ("reuse_stable_color",),
    "geometry_jump": ("reuse_prior_transform", "rerun_alignment"),
    "enhancer_hallucination": ("reduce_enhancer_strength",),
    "detail_disappearance": ("restore_stable_detail",),
    # A changing eye/jaw shape can be real expression.  The correction below is
    # only emitted at low motion; at high motion the event is logged and the
    # current expression is preserved rather than frozen or blurred.
    "eye_state_discontinuity": ("reuse_prior_transform",),
    "jawline_movement_discontinuity": ("reuse_prior_transform",),
    "face_flicker": ("reblend_previous_mask",),
}


def _finite(value, default=None):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _face_value(face, name, default=None):
    if face is None:
        return default
    try:
        value = face.get(name, default)
    except AttributeError:
        value = getattr(face, name, default)
    return value


def _resize_gray(image, size=64):
    if image is None:
        return None
    try:
        arr = np.asarray(image)
        if arr.size == 0:
            return None
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
        return arr.astype(np.float32)
    except Exception:
        return None


def _high_frequency(image):
    gray = _resize_gray(image)
    if gray is None:
        return None
    low = cv2.GaussianBlur(gray, (0, 0), 1.4)
    return _finite(np.mean(np.abs(gray - low)), None)


def _appearance_chroma(appearance):
    if not isinstance(appearance, dict):
        return None
    for key in ("chroma", "skin_chroma", "color", "mean_chroma"):
        value = appearance.get(key)
        if value is not None:
            try:
                if isinstance(value, dict):
                    value = value.get("mean", value.get("a"))
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                    return [float(arr[0]), float(arr[1])]
            except Exception:
                pass
    return None


def _landmark_states(face):
    """Return stable scalar proxies when the caller did not provide metrics."""
    points = _face_value(face, "landmark_2d_106")
    if points is None:
        return None, None
    try:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(points) < 33:
            return None, None
        # Reuse the repository's established 106-point eye/jaw convention when
        # landmarks are available. This remains model-free and avoids making
        # temporal QC disagree with the existing expression stabilizer.
        from roop.temporal_expression import measure_expression
        metrics = measure_expression(
            points, kps=_face_value(face, "kps"),
            bbox=_face_value(face, "bbox"),
            detection_confidence=_face_value(face, "det_score", 1.0))
        eye = None
        if metrics.get("left_eye_openness") is not None and metrics.get("right_eye_openness") is not None:
            eye = (float(metrics["left_eye_openness"]) +
                   float(metrics["right_eye_openness"])) * 0.5
        jaw = _finite(metrics.get("jaw_position"), None)
        return eye, jaw
    except Exception:
        return None, None


def make_observation(face=None, image=None, matrix=None, mask=None, output=None,
                     appearance=None, source_index=None, motion=None,
                     confidence=None, identity_similarity=None,
                     detail_metrics=None, **values):
    """Build a compact, serialisable temporal observation.

    ``values`` is intentionally accepted so tests and future pipeline stages
    can provide stronger measurements (for example ``eye_state`` or
    ``jawline``) without making this utility depend on a particular detector.
    """
    eye_state, jawline = _landmark_states(face)
    if motion is None:
        motion = _face_value(face, "_temporal_motion", 0.0)
    if confidence is None:
        confidence = _face_value(face, "_temporal_confidence", None)
    if confidence is None:
        confidence = _face_value(face, "confidence", 1.0)
    if identity_similarity is None:
        identity_similarity = _face_value(face, "_identity_similarity", None)
        if identity_similarity is None:
            identity_similarity = _face_value(face, "_temporal_identity_similarity", None)

    obs = {
        "luma": _finite(values.get("luma"), None),
        "chroma": _appearance_chroma(appearance),
        "transform": None,
        "bbox": None,
        "mask_area": None,
        "mask_shape": None,
        "input_detail": _high_frequency(image),
        "output_detail": _high_frequency(output),
        "detail_energy": None,
        "eye_state": _finite(values.get("eye_state", eye_state), None),
        "jawline": _finite(values.get("jawline", jawline), None),
        "identity_similarity": _finite(identity_similarity, None),
        "source_index": source_index,
        "motion": max(0.0, _finite(motion, 0.0) or 0.0),
        "confidence": max(0.0, min(1.0, _finite(confidence, 1.0) or 1.0)),
        "appearance": copy.deepcopy(appearance) if isinstance(appearance, dict) else None,
    }
    if image is not None:
        gray = _resize_gray(image)
        if gray is not None:
            obs["luma"] = float(np.mean(gray) / 255.0)
    if output is not None and obs["luma"] is None:
        gray = _resize_gray(output)
        if gray is not None:
            obs["luma"] = float(np.mean(gray) / 255.0)
    if matrix is not None:
        try:
            arr = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
            obs["transform"] = arr.tolist()
        except Exception:
            pass
    bbox = _face_value(face, "bbox")
    if bbox is not None:
        try:
            arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
            if arr.size >= 4:
                obs["bbox"] = [float(x) for x in arr[:4]]
        except Exception:
            pass
    if mask is not None:
        try:
            arr = np.asarray(mask, dtype=np.float32)
            if arr.size:
                small = cv2.resize(arr, (32, 32), interpolation=cv2.INTER_AREA)
                small = np.clip(small, 0.0, 1.0)
                obs["mask_area"] = float(np.mean(small))
                obs["mask_shape"] = small.tolist()
        except Exception:
            pass
    if isinstance(detail_metrics, dict):
        for key in ("energy", "detail_energy", "identity_detail_energy"):
            if key in detail_metrics:
                obs["detail_energy"] = _finite(detail_metrics[key], None)
                break
    if obs["detail_energy"] is None:
        obs["detail_energy"] = obs["output_detail"]
    for key in ("luma", "chroma", "eye_state", "jawline", "identity_similarity",
                "input_detail", "output_detail", "detail_energy"):
        if key in values and values[key] is not None:
            if key == "chroma":
                try:
                    obs[key] = [float(x) for x in np.asarray(values[key]).reshape(-1)[:2]]
                except Exception:
                    pass
            else:
                obs[key] = _finite(values[key], obs[key])
    return obs


def _clone_observation(observation):
    result = dict(observation or {})
    if result.get("mask_shape") is not None:
        result["mask_shape"] = copy.deepcopy(result["mask_shape"])
    if result.get("transform") is not None:
        result["transform"] = copy.deepcopy(result["transform"])
    if result.get("chroma") is not None:
        result["chroma"] = list(result["chroma"])
    return result


def _mask_delta(a, b):
    try:
        if a is None or b is None:
            return None
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        if aa.shape != bb.shape:
            return None
        return float(np.mean(np.abs(aa - bb)))
    except Exception:
        return None


def _geometry_delta(current, previous, bbox=None, previous_bbox=None):
    values = []
    try:
        if current is not None and previous is not None:
            a = np.asarray(current, dtype=np.float32).reshape(2, 3)
            b = np.asarray(previous, dtype=np.float32).reshape(2, 3)
            scale = max(1.0, abs(float(a[0, 0])) + abs(float(a[1, 1])))
            values.append(float(np.linalg.norm(a[:, :2] - b[:, :2]) / scale))
            values.append(float(np.linalg.norm(a[:, 2] - b[:, 2]) / 64.0))
    except Exception:
        pass
    try:
        if bbox is not None and previous_bbox is not None:
            a = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
            b = np.asarray(previous_bbox, dtype=np.float32).reshape(-1)[:4]
            scale = max(1.0, float(max(a[2] - a[0], a[3] - a[1])))
            values.append(float(np.linalg.norm((a[:2] - b[:2])) / scale))
            values.append(float(np.linalg.norm((a[2:] - a[:2]) - (b[2:] - b[:2])) / scale))
    except Exception:
        pass
    return max(values) if values else None


@dataclass
class QualityDecision:
    track_id: object
    frame_index: object
    anomalies: list = field(default_factory=list)
    corrections: list = field(default_factory=list)
    confidence: float = 0.0
    metrics: dict = field(default_factory=dict)
    previous_source_index: object = None
    stable_transform: object = None

    @property
    def normal(self):
        return not self.anomalies

    def as_dict(self):
        return {
            "anomaly_type": list(self.anomalies),
            "track": self.track_id,
            "frame_index": self.frame_index,
            "confidence": float(self.confidence),
            "correction_applied": list(self.corrections),
            "normal": self.normal,
            "metrics": dict(self.metrics),
        }


def merge_decisions(*decisions):
    """Combine pre/post-stage events into one auditable frame decision."""
    valid = [d for d in decisions if d is not None]
    if not valid:
        return None
    anomalies = []
    corrections = []
    metrics = {}
    for decision in valid:
        for value in decision.anomalies:
            if value not in anomalies:
                anomalies.append(value)
        for value in decision.corrections:
            if value not in corrections:
                corrections.append(value)
        metrics.update(decision.metrics)
    last = valid[-1]
    stable = next((d.stable_transform for d in valid
                   if d.stable_transform is not None), None)
    previous_source = next((d.previous_source_index for d in valid
                            if d.previous_source_index is not None), None)
    merged = QualityDecision(last.track_id, last.frame_index, anomalies, corrections,
                             max(d.confidence for d in valid), metrics,
                             previous_source, stable)
    merged._raw_anomalies = set().union(*(getattr(d, "_raw_anomalies", set())
                                          for d in valid))
    return merged


class TemporalQualityController:
    """Per-track short-history anomaly detector with bounded state."""

    def __init__(self, enabled=False, logging=False, history_size=4,
                 cache_size=256):
        self.enabled = bool(enabled)
        self.logging = bool(logging)
        self.history_size = max(2, int(history_size or 4))
        self.cache_size = max(1, int(cache_size or 256))
        self._tracks = {}
        self._logs = deque(maxlen=1024)
        self._counts = Counter()
        self._lock = RLock()
        self.ordered = False

    @classmethod
    def from_config(cls, config):
        enabled = bool(getattr(config, "temporal_quality_control", False))
        logging = bool(getattr(config, "temporal_quality_logging", False))
        env = os.environ.get("ROOP_TEMPORAL_QC", "").strip().lower()
        if env in ("1", "true", "yes", "on"):
            enabled = True
        env_log = os.environ.get("ROOP_TEMPORAL_QC_LOG", "").strip().lower()
        if env_log in ("1", "true", "yes", "on"):
            logging = True
        return cls(enabled=enabled, logging=logging,
                   history_size=getattr(config, "temporal_quality_history", 4),
                   cache_size=getattr(config, "temporal_quality_cache_size", 256))

    def clone_for_block(self):
        return TemporalQualityController(self.enabled, self.logging,
                                         self.history_size, self.cache_size)

    def warmup_frames(self):
        return max(0, self.history_size - 1) if self.enabled else 0

    def set_ordered(self, ordered=True):
        self.ordered = bool(ordered)

    def reset(self):
        with self._lock:
            self._tracks.clear()
            self._logs.clear()
            self._counts.clear()

    def _state(self, track_id):
        key = track_id if track_id is not None else "__default__"
        with self._lock:
            if key not in self._tracks:
                if len(self._tracks) >= self.cache_size:
                    self._tracks.pop(next(iter(self._tracks)))
                self._tracks[key] = {"history": deque(maxlen=self.history_size),
                                     "active_anomalies": set()}
            return self._tracks[key]

    @staticmethod
    def _append_unique(items, value):
        if value not in items:
            items.append(value)

    def inspect(self, track_id, frame_index, observation):
        """Compare against history without changing it."""
        if not self.enabled:
            return QualityDecision(track_id, frame_index)
        current = _clone_observation(observation)
        state = self._state(track_id)
        with self._lock:
            previous = state["history"][-1] if state["history"] else None
            if previous is None:
                decision = QualityDecision(
                    track_id, frame_index,
                    confidence=float(current.get("confidence", 0.0)))
                decision._raw_anomalies = set()
                return decision
            anomalies = []
            metrics = {}
            motion = float(current.get("motion") or 0.0)
            confidence = float(current.get("confidence") or 0.0)

            def add(name, value):
                if value is not None:
                    metrics[name] = float(value)
                    self._append_unique(anomalies, name)

            luma_delta = None
            if current.get("luma") is not None and previous.get("luma") is not None:
                luma_delta = abs(float(current["luma"]) - float(previous["luma"]))
            if luma_delta is not None and luma_delta > max(0.10, 0.055 + motion * 0.012):
                add("face_brightness_jump", luma_delta)

            chroma_delta = None
            if current.get("chroma") is not None and previous.get("chroma") is not None:
                chroma_delta = float(np.linalg.norm(np.asarray(current["chroma"]) -
                                                    np.asarray(previous["chroma"])))
            if chroma_delta is not None and chroma_delta > 10.0:
                add("skin_color_jump", chroma_delta)

            geometry = _geometry_delta(current.get("transform"), previous.get("transform"),
                                       current.get("bbox"), previous.get("bbox"))
            if geometry is not None and geometry > max(0.20, 0.14 + motion * 0.012):
                add("geometry_jump", geometry)

            mask_delta = _mask_delta(current.get("mask_shape"), previous.get("mask_shape"))
            area_delta = None
            if current.get("mask_area") is not None and previous.get("mask_area") is not None:
                area_delta = abs(float(current["mask_area"]) - float(previous["mask_area"]))
            mask_signal = max(x for x in (mask_delta, area_delta) if x is not None) if (mask_delta is not None or area_delta is not None) else None
            if mask_signal is not None and mask_signal > 0.16:
                add("mask_popping", mask_signal)

            input_detail = current.get("input_detail")
            output_detail = current.get("output_detail")
            if output_detail is not None and input_detail is not None:
                if output_detail > input_detail * 2.35 + 2.0:
                    add("enhancer_hallucination", output_detail / max(input_detail, 1e-3))
            prev_detail = previous.get("detail_energy")
            current_detail = current.get("detail_energy")
            if prev_detail is not None and current_detail is not None:
                if prev_detail > 2.0 and current_detail < prev_detail * 0.48:
                    add("detail_disappearance", current_detail / max(prev_detail, 1e-3))

            for key, name, threshold in (
                    ("eye_state", "eye_state_discontinuity", 0.24),
                    ("jawline", "jawline_movement_discontinuity", 0.18)):
                if current.get(key) is not None and previous.get(key) is not None:
                    delta = abs(float(current[key]) - float(previous[key]))
                    if delta > threshold + motion * 0.03:
                        add(name, delta)

            identity = current.get("identity_similarity")
            prev_identity = previous.get("identity_similarity")
            source_changed = (current.get("source_index") is not None and
                              previous.get("source_index") is not None and
                              current.get("source_index") != previous.get("source_index"))
            if ((identity is not None and identity < 0.55) or
                    (identity is not None and prev_identity is not None and
                     identity < prev_identity - 0.22) or
                    (source_changed and motion < 0.35 and confidence > 0.35)):
                signal = (1.0 - float(identity)) if identity is not None else 1.0
                add("identity_drift", signal)

            if luma_delta is not None and luma_delta > max(0.14, 0.08 + motion * 0.015):
                add("face_flicker", luma_delta)

            # Event edge-triggering is important: a persistent bad enhancer
            # output must not invoke a correction on every following frame.
            # The active set is advanced by record(), after the caller has
            # applied any event correction. A normal observation clears it.
            raw_anomalies = set(anomalies)
            active = state.get("active_anomalies", set())
            anomalies = [name for name in anomalies if name not in active]
            metrics = {name: value for name, value in metrics.items()
                       if name in anomalies}
            corrections = []
            for anomaly in anomalies:
                for correction in _CORRECTIONS.get(anomaly, ()):
                    # Real fast expression/jaw motion must survive.  Detection is
                    # retained, but a transform reuse is unsafe in this case.
                    if (anomaly in ("eye_state_discontinuity",
                                    "jawline_movement_discontinuity") and motion >= 0.20):
                        continue
                    self._append_unique(corrections, correction)
            stable_transform = (copy.deepcopy(previous.get("transform"))
                                if "reuse_prior_transform" in corrections else None)
            previous_source = (previous.get("source_index")
                               if "reselect_source" in corrections else None)
            decision_confidence = max(0.0, min(1.0, confidence *
                                               (0.65 if anomalies else 1.0)))
            decision = QualityDecision(track_id, frame_index, anomalies, corrections,
                                       decision_confidence, metrics, previous_source,
                                       stable_transform)
            decision._raw_anomalies = raw_anomalies
            return decision

    def record(self, track_id, frame_index, observation, decision=None):
        if not self.enabled:
            return
        obs = _clone_observation(observation)
        state = self._state(track_id)
        with self._lock:
            state["history"].append(obs)
            state["active_anomalies"] = (set(getattr(decision, "_raw_anomalies",
                                                     decision.anomalies))
                                          if decision else set())
            if decision is not None and decision.anomalies:
                for anomaly in decision.anomalies:
                    self._counts[anomaly] += 1
                if self.logging:
                    entry = decision.as_dict()
                    self._logs.append(entry)

    def correct_mask(self, track_id, current_mask, strength=0.65):
        """Use the prior alpha only for a mask-pop event; never blur pixels."""
        if not self.enabled or current_mask is None:
            return current_mask
        state = self._state(track_id)
        with self._lock:
            if not state["history"]:
                return current_mask
            previous = state["history"][-1].get("mask_shape")
        if previous is None:
            return current_mask
        try:
            current = np.asarray(current_mask, dtype=np.float32)
            old = cv2.resize(np.asarray(previous, dtype=np.float32),
                             (current.shape[1], current.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
            alpha = max(0.0, min(1.0, float(strength)))
            return np.clip(old * alpha + current * (1.0 - alpha), 0.0, 1.0)
        except Exception:
            return current_mask

    def stable_transform(self, track_id):
        state = self._state(track_id)
        with self._lock:
            if not state["history"]:
                return None
            return copy.deepcopy(state["history"][-1].get("transform"))

    def stable_source_index(self, track_id):
        state = self._state(track_id)
        with self._lock:
            return state["history"][-1].get("source_index") if state["history"] else None

    def stable_appearance(self, track_id):
        state = self._state(track_id)
        with self._lock:
            return copy.deepcopy(state["history"][-1].get("appearance")) if state["history"] else None

    def telemetry(self):
        with self._lock:
            return {"enabled": self.enabled, "logging": self.logging,
                    "anomaly_counts": dict(self._counts),
                    "recent": list(self._logs)}
