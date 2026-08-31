"""Low-overhead temporal face tracking and detection scheduling.

This module deliberately does not own source-to-face assignment.  The existing
``procmgr_tracking`` pass remains the authority for that decision.  The class
here owns the shorter-lived, frame-order state needed to decide whether the
next detector call can be an ROI call and to keep a face's geometry coherent
between detector observations.

The tracker is detector-agnostic: a caller supplies the same Face objects it
already gets from the configured detector/pool.  No model sessions, providers,
or GPU buffers are created here.
"""

from dataclasses import dataclass, field
from functools import lru_cache
import os
import math

import numpy as np


def _value(face, name, default=None):
    """Read an InsightFace Face or a dict without invoking fragile __getattr__."""
    if isinstance(face, dict):
        return face.get(name, default)
    try:
        return getattr(face, name, default)
    except Exception:
        return default


def _array(value, dtype=np.float32):
    if value is None:
        return None
    try:
        out = np.asarray(value, dtype=dtype)
        return out.copy() if out.size else None
    except (TypeError, ValueError):
        return None


def _bbox(face):
    out = _array(_value(face, "bbox"))
    if out is None or out.size != 4:
        return None
    out = out.reshape(4)
    if not np.all(np.isfinite(out)) or out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def _landmarks(face):
    # 106 points are preferred because the downstream mask/mouth paths use
    # them; five-point keypoints remain a useful detector-only fallback.
    return _array(_value(face, "landmark_2d_106")
                  if _value(face, "landmark_2d_106") is not None
                  else _value(face, "kps"))


def _pose(face):
    value = _value(face, "pose")
    if value is None:
        return None
    out = _array(value, np.float32)
    if out is None or out.size == 0 or not np.all(np.isfinite(out)):
        return None
    return out.reshape(-1)


def _embedding(face):
    value = _value(face, "embedding")
    if value is None:
        value = _value(face, "normed_embedding")
    out = _array(value, np.float32)
    if out is None or out.ndim != 1 or out.size == 0:
        return None
    norm = float(np.linalg.norm(out))
    return (out / norm).astype(np.float32) if norm > 1e-7 else None


def _mask(face):
    for name in ("mask", "face_mask", "segmentation", "previous_mask"):
        value = _value(face, name)
        if value is not None:
            return _array(value)
    return None


def _centre(box):
    return np.asarray(((box[0] + box[2]) * 0.5,
                       (box[1] + box[3]) * 0.5), dtype=np.float32)


def _size(box):
    return np.asarray((max(1.0, float(box[2] - box[0])),
                       max(1.0, float(box[3] - box[1]))), dtype=np.float32)


def _iou(a, b):
    ix0, iy0 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix1, iy1 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = aa + bb - inter
    return inter / union if union > 0.0 else 0.0


def _cosine_distance(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-7:
        return None
    return float(1.0 - np.dot(a, b) / den)


@dataclass
class DetectionPlan:
    """The next safe detector operation."""

    mode: str
    roi: object = None
    reason: str = ""
    track_ids: tuple = ()


@dataclass
class TemporalTrack:
    """Persistent state for one physical face across detector observations."""

    track_id: int
    bbox: np.ndarray
    landmarks: object = None
    pose: object = None
    identity_embedding: object = None
    confidence: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    previous_mask: object = None
    previous_frame_index: int = -1
    last_frame_index: int = -1
    predicted_bbox: object = None
    predicted_landmarks: object = None
    misses: int = 0
    hits: int = 1
    status: str = "uncertain"
    current_mask: object = None

    def snapshot(self):
        """Return a detached, JSON/debug-friendly state snapshot."""
        return {
            "track_id": int(self.track_id),
            "bbox": self.bbox.copy(),
            "landmarks": None if self.landmarks is None else self.landmarks.copy(),
            "pose": None if self.pose is None else self.pose.copy(),
            "identity_embedding": (None if self.identity_embedding is None
                                    else self.identity_embedding.copy()),
            "confidence": float(self.confidence),
            "velocity": self.velocity.copy(),
            "previous_mask": (None if self.previous_mask is None
                               else self.previous_mask.copy()),
            "previous_frame_index": int(self.previous_frame_index),
            "last_frame_index": int(self.last_frame_index),
            "predicted_bbox": (None if self.predicted_bbox is None
                                else self.predicted_bbox.copy()),
            "misses": int(self.misses),
            "hits": int(self.hits),
            "status": self.status,
        }


class TemporalFaceTracker:
    """Confidence-aware persistent tracker used by the temporal pre-pass.

    Detection scheduling is intentionally conservative: a full-frame recovery
    is required at startup, periodically, and whenever a track is lost.  In
    between, one union ROI covers all predicted live tracks.  A stable track
    gets a tighter ROI; an uncertain track expands the ROI.  The existing ROI
    helper still performs the actual configured detector call and falls back to
    full-frame detection when the crop returns no faces.

    Assignment is global for the small number of faces normally in a frame.
    Appearance is the dominant term when an embedding is available, while
    predicted motion and IoU resolve close/crossing geometry.  Ambiguous or
    impossible matches are left unmatched and become a new track instead of
    silently changing an existing identity.
    """

    def __init__(self, full_interval=8, stable_hits=2, max_misses=3,
                 reid_age=45, stable_pad=0.55, uncertain_pad=1.35,
                 min_roi=160):
        self.full_interval = max(1, int(full_interval))
        self.stable_hits = max(1, int(stable_hits))
        self.max_misses = max(1, int(max_misses))
        self.reid_age = max(self.max_misses, int(reid_age))
        self.stable_pad = max(0.1, float(stable_pad))
        self.uncertain_pad = max(self.stable_pad, float(uncertain_pad))
        self.min_roi = max(32, int(min_roi))
        self.tracks = []
        self._next_id = 0
        self._last_full_frame = -self.full_interval
        self._full_pending_frame = None
        self._lost_recovery_wait = False
        self._last_plan = None
        self.events = []
        self.stats = {
            "frames": 0,
            "full_detections": 0,
            "roi_detections": 0,
            "roi_fallback_full": 0,
            "matched": 0,
            "new_tracks": 0,
            "lost_tracks": 0,
            "recovered_tracks": 0,
        }

    @classmethod
    def from_env(cls):
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
            full_interval=_int("ROOP_TEMPORAL_FULL_DETECT_INTERVAL", 8),
            stable_hits=_int("ROOP_TEMPORAL_STABLE_HITS", 2),
            max_misses=_int("ROOP_TEMPORAL_MAX_MISSES", 3),
            reid_age=_int("ROOP_TEMPORAL_REID_AGE", 45),
            stable_pad=_float("ROOP_TEMPORAL_STABLE_ROI_PAD", 0.55),
            uncertain_pad=_float("ROOP_TEMPORAL_UNCERTAIN_ROI_PAD", 1.35),
            min_roi=_int("ROOP_TEMPORAL_MIN_ROI", 160),
        )

    @staticmethod
    def _predict(track, frame_index):
        dt = max(0, int(frame_index) - int(track.last_frame_index))
        box = track.bbox.copy()
        if dt and dt <= 8 and np.any(track.velocity):
            projected = box + track.velocity * float(dt)
            if projected[2] > projected[0] and projected[3] > projected[1]:
                box = projected.astype(np.float32)
        track.predicted_bbox = box
        if track.landmarks is not None:
            delta = box[:2] - track.bbox[:2]
            predicted = track.landmarks.copy()
            if predicted.ndim >= 2 and predicted.shape[-1] >= 2:
                predicted[..., :2] += delta
            track.predicted_landmarks = predicted
        return box

    def plan(self, frame_index, frame_shape):
        """Choose full-frame recovery or a confidence-sized union ROI."""
        frame_index = int(frame_index)
        h, w = int(frame_shape[0]), int(frame_shape[1])
        live = [t for t in self.tracks
                if t.status != "lost" and t.misses <= self.max_misses]
        def _reserve_full(frame):
            self._last_full_frame = frame
            self._full_pending_frame = frame

        if not self.tracks:
            if self._last_full_frame < 0 or frame_index - self._last_full_frame >= self.full_interval:
                plan = DetectionPlan("full", reason="initialization")
                # Reserve the seed/recovery while detector-pool futures are in
                # flight; pending frames coast until the result arrives.
                _reserve_full(frame_index)
            else:
                plan = DetectionPlan("coast", reason="initial_detection_pending")
        elif any(t.status == "lost" for t in self.tracks):
            if (self._full_pending_frame is None
                    and not (self._lost_recovery_wait
                             and frame_index - self._last_full_frame < self.full_interval)):
                plan = DetectionPlan("full", reason="track_lost",
                                     track_ids=tuple(t.track_id for t in live))
                _reserve_full(frame_index)
            else:
                plan = DetectionPlan("coast", reason="lost_recovery_pending")
        elif frame_index - self._last_full_frame >= self.full_interval:
            plan = DetectionPlan("full", reason="periodic_recovery",
                                 track_ids=tuple(t.track_id for t in live))
            # Reserve the recovery at planning time as well as at update time.
            # The temporal pre-pass may have several detector futures in flight;
            # without this reservation each pending frame sees the old
            # last-full value and schedules the same expensive full pass again.
            _reserve_full(frame_index)
        elif not live:
            plan = DetectionPlan("coast", reason="no_live_tracks")
        else:
            boxes = []
            uncertain = False
            ids = []
            for track in live:
                box = self._predict(track, frame_index)
                boxes.append(box)
                ids.append(track.track_id)
                uncertain |= track.status != "stable" or track.misses > 0
            pad_ratio = self.uncertain_pad if uncertain else self.stable_pad
            x0 = min(float(b[0]) for b in boxes)
            y0 = min(float(b[1]) for b in boxes)
            x1 = max(float(b[2]) for b in boxes)
            y1 = max(float(b[3]) for b in boxes)
            bw = max(1.0, x1 - x0)
            bh = max(1.0, y1 - y0)
            roi = np.asarray((
                max(0.0, x0 - bw * pad_ratio),
                max(0.0, y0 - bh * pad_ratio),
                min(float(w), x1 + bw * pad_ratio),
                min(float(h), y1 + bh * pad_ratio),
            ), dtype=np.float32)
            if roi[2] - roi[0] < self.min_roi:
                extra = (self.min_roi - (roi[2] - roi[0])) * 0.5
                roi[0], roi[2] = max(0.0, roi[0] - extra), min(float(w), roi[2] + extra)
            if roi[3] - roi[1] < self.min_roi:
                extra = (self.min_roi - (roi[3] - roi[1])) * 0.5
                roi[1], roi[3] = max(0.0, roi[1] - extra), min(float(h), roi[3] + extra)
            plan = DetectionPlan("roi", tuple(float(v) for v in roi),
                                 reason="uncertain_roi" if uncertain else "stable_roi",
                                 track_ids=tuple(ids))
        self._last_plan = plan
        return plan

    @staticmethod
    def _candidate(track, det, predicted):
        box = _bbox(det)
        if box is None:
            return None
        emb = _embedding(det)
        emb_dist = _cosine_distance(track.identity_embedding, emb)
        size = _size(predicted)
        center_norm = float(np.linalg.norm(_centre(box) - _centre(predicted)) /
                            max(1.0, float(np.linalg.norm(size))))
        overlap = _iou(predicted, box)
        scale_ratio = max(float(np.max(_size(box) / size)),
                          float(np.max(size / _size(box))))
        # An active track has temporal evidence; a lost track may return at a
        # new location, so re-entry is appearance-only and intentionally tight.
        if track.status == "lost":
            if emb_dist is None or emb_dist > 0.35:
                return None
            return 1.0 - emb_dist
        if emb_dist is not None and emb_dist > 0.82 and overlap < 0.08:
            return None
        if center_norm > 3.5 and overlap < 0.02 and (emb_dist is None or emb_dist > 0.35):
            return None
        motion = math.exp(-min(center_norm, 6.0) / 2.0)
        appearance = 1.0 - min(1.2, max(0.0, emb_dist if emb_dist is not None else 0.45))
        scale = math.exp(-abs(math.log(max(1.0, scale_ratio))))
        # Embedding dominates crossings; motion and IoU prevent a noisy
        # embedding from stealing a nearby track during contact.
        return 0.62 * appearance + 0.23 * motion + 0.10 * overlap + 0.05 * scale

    def _assign(self, detections, live):
        if not live or not detections:
            return {}, set(range(len(detections)))
        predicted = [self._predict(t, self._frame_index) for t in live]
        scores = [[self._candidate(t, d, p) for d in detections]
                  for t, p in zip(live, predicted)]
        n_tracks, n_dets = len(live), len(detections)

        if n_tracks <= 8 and n_dets <= 8:
            # Maximise total evidence, with an unmatched option at every track.
            # DP keeps the crossing case global without importing scipy.
            @lru_cache(maxsize=None)
            def best(i, used):
                if i >= n_tracks:
                    return 0.0, ()
                score, path = best(i + 1, used)
                result = (score, (-1,) + path)
                for j in range(n_dets):
                    value = scores[i][j]
                    if value is None or used & (1 << j):
                        continue
                    tail_score, tail_path = best(i + 1, used | (1 << j))
                    candidate = (tail_score + value, (j,) + tail_path)
                    if candidate[0] > result[0] + 1e-9:
                        result = candidate
                return result
            path = best(0, 0)[1]
        else:
            edges = sorted(
                ((value, i, j)
                 for i, row in enumerate(scores)
                 for j, value in enumerate(row) if value is not None),
                reverse=True)
            chosen_t, chosen_d = set(), set()
            path = [-1] * n_tracks
            for _value_, i, j in edges:
                if i not in chosen_t and j not in chosen_d:
                    path[i] = j
                    chosen_t.add(i)
                    chosen_d.add(j)

        assignments = {}
        used = set()
        for i, j in enumerate(path):
            if j < 0:
                continue
            # Require a real margin when the top two tracks are nearly tied for
            # the same detection. Leaving it unmatched is safer than an ID swap.
            alternatives = sorted((scores[k][j] for k in range(n_tracks)
                                   if k != i and scores[k][j] is not None), reverse=True)
            if alternatives and scores[i][j] - alternatives[0] < 0.035:
                continue
            assignments[i] = j
            used.add(j)
        return assignments, set(range(n_dets)) - used

    @staticmethod
    def _smooth(previous, current, alpha):
        if previous is None:
            return None if current is None else current.copy()
        if current is None or np.shape(previous) != np.shape(current):
            return previous.copy()
        return ((1.0 - alpha) * previous + alpha * current).astype(np.float32)

    def _new_track(self, det, frame_index):
        box = _bbox(det)
        track = TemporalTrack(
            track_id=self._next_id,
            bbox=box,
            landmarks=_landmarks(det),
            pose=_pose(det),
            identity_embedding=_embedding(det),
            confidence=float(_value(det, "det_score", 0.0) or 0.0),
            previous_mask=None,
            previous_frame_index=-1,
            last_frame_index=int(frame_index),
            predicted_bbox=box.copy(),
            current_mask=_mask(det),
        )
        self._next_id += 1
        self.tracks.append(track)
        self.stats["new_tracks"] += 1
        self.events.append({"type": "appeared", "track_id": track.track_id,
                            "frame_index": int(frame_index)})
        return track

    def _update_matched(self, track, det, frame_index):
        raw_box = _bbox(det)
        old_box = track.bbox.copy()
        old_frame = track.last_frame_index
        dt = max(1, int(frame_index) - int(old_frame))
        raw_velocity = (raw_box - old_box) / float(dt)
        speed = float(np.linalg.norm(raw_velocity[:2]) /
                      max(1.0, float(np.linalg.norm(_size(old_box)))))
        # Low-pass jitter but release quickly for genuine motion or recovery.
        alpha = 0.78 if speed > 0.28 or track.misses else 0.38
        track.previous_frame_index = old_frame
        track.previous_mask = (None if track.current_mask is None
                               else track.current_mask.copy())
        track.bbox = self._smooth(track.bbox, raw_box, alpha)
        track.velocity = (0.55 * track.velocity + 0.45 * raw_velocity).astype(np.float32)
        track.landmarks = self._smooth(track.landmarks, _landmarks(det), alpha)
        track.pose = self._smooth(track.pose, _pose(det), alpha)
        emb = _embedding(det)
        if emb is not None:
            if track.identity_embedding is None:
                track.identity_embedding = emb
            else:
                track.identity_embedding = (0.75 * track.identity_embedding + 0.25 * emb).astype(np.float32)
                norm = float(np.linalg.norm(track.identity_embedding))
                if norm > 1e-7:
                    track.identity_embedding /= norm
        det_conf = float(_value(det, "det_score", track.confidence) or track.confidence)
        track.confidence = float(0.70 * track.confidence + 0.30 * max(0.0, min(1.0, det_conf)))
        track.current_mask = _mask(det)
        track.last_frame_index = int(frame_index)
        track.predicted_bbox = track.bbox.copy()
        track.predicted_landmarks = None if track.landmarks is None else track.landmarks.copy()
        was_lost = track.status == "lost"
        track.misses = 0
        track.hits += 1
        track.status = ("stable" if track.hits >= self.stable_hits and track.confidence >= 0.35
                        else "uncertain")
        if was_lost:
            self.stats["recovered_tracks"] += 1
            self.events.append({"type": "recovered", "track_id": track.track_id,
                                "frame_index": int(frame_index)})

    def _mark_missed(self, track, frame_index):
        if track.last_frame_index < 0 or int(frame_index) <= track.last_frame_index:
            return
        track.previous_frame_index = track.last_frame_index
        track.misses = int(frame_index) - int(track.last_frame_index)
        self._predict(track, frame_index)
        if track.misses > self.max_misses:
            if track.status != "lost":
                track.status = "lost"
                self.stats["lost_tracks"] += 1
                self.events.append({"type": "left", "track_id": track.track_id,
                                    "frame_index": int(frame_index)})
        else:
            track.status = "uncertain"

    def update(self, detections, frame_index, frame_shape=None, detection_mode="full"):
        """Consume one detector result and return frame assignments/events."""
        detections = [d for d in (detections or []) if _bbox(d) is not None]
        self._frame_index = int(frame_index)
        self.events = []
        had_lost = any(t.status == "lost" for t in self.tracks)
        self.stats["frames"] += 1
        if detection_mode == "full":
            self.stats["full_detections"] += 1
            self._last_full_frame = int(frame_index)
            self._full_pending_frame = None
        elif detection_mode == "roi":
            self.stats["roi_detections"] += 1
        elif detection_mode == "roi_fallback_full":
            self.stats["roi_fallback_full"] += 1
            self.stats["full_detections"] += 1
            self._last_full_frame = int(frame_index)
            self._full_pending_frame = None

        live = [t for t in self.tracks
                if t.status != "lost" or int(frame_index) - t.last_frame_index <= self.reid_age]
        assignments, unmatched = self._assign(detections, live)
        matched_tracks = set()
        for live_index, det_index in assignments.items():
            track = live[live_index]
            self._update_matched(track, detections[det_index], frame_index)
            matched_tracks.add(track.track_id)
            self.stats["matched"] += 1
        for track in self.tracks:
            if track.track_id not in matched_tracks:
                self._mark_missed(track, frame_index)
        new_detection_indices = []
        for det_index in sorted(unmatched):
            track = self._new_track(detections[det_index], frame_index)
            # Every accepted detection receives an ID, including a face that
            # has just entered the frame. This makes the per-frame contract
            # explicit and avoids a caller having to infer new IDs from events.
            assignments[-(det_index + 1)] = track.track_id
            new_detection_indices.append(int(det_index))

        if had_lost:
            self._lost_recovery_wait = not any(
                event.get("type") == "recovered" for event in self.events)
        elif detection_mode in ("full", "roi_fallback_full"):
            self._lost_recovery_wait = False

        # Retain lost tracks for appearance-only re-entry, but bound memory and
        # future assignment work on very long clips.
        self.tracks = [t for t in self.tracks
                       if t.status != "lost" or int(frame_index) - t.last_frame_index <= self.reid_age]
        active = [t.snapshot() for t in self.tracks if t.status != "lost"]
        return {
            "assignments": {
                (int(det_index) if live_index >= 0 else int(-live_index - 1)):
                int(live[live_index].track_id if live_index >= 0 else det_index)
                for live_index, det_index in assignments.items()
            },
            # All valid detections are assigned either to an existing track or
            # to a newly-created one before returning. Keep the field for API
            # compatibility, but make the invariant explicit and expose the
            # useful subset separately.
            "unmatched_detection_indices": (),
            "new_detection_indices": tuple(new_detection_indices),
            "tracks": active,
            "events": list(self.events),
            "detection_mode": detection_mode,
        }

    def export(self):
        return {
            "tracks": [t.snapshot() for t in self.tracks],
            "stats": dict(self.stats),
            "events": list(self.events),
        }


__all__ = ["DetectionPlan", "TemporalTrack", "TemporalFaceTracker"]
