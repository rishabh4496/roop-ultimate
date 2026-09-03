"""Persistent, occlusion-aware face tracking.

WHAT THIS OWNS, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------
This module owns one thing: a face's IDENTITY and GEOMETRY across frames, and
whether that face is currently VISIBLE, PARTIALLY OCCLUDED or COASTING on a
prediction.  It creates no model sessions, allocates no GPU memory, and reads
no provider policy -- callers hand it whatever Face objects their configured
detector already produced.

It does NOT own source-to-face assignment.  `procmgr_tracking._assign_track_
sources` remains the authority on which faceset a person gets, and nothing here
may be read as an identity decision.

THE PROBLEM IT EXISTS FOR
-------------------------
A hand, a mug, a microphone or a strand of hair crossing a face takes RetinaFace
below its confidence threshold for a handful of frames.  Frame-by-frame
detection then reports "no face", the swap is skipped, and the viewer sees the
original face flash back for 3-10 frames.  That flash is the reported bug.

Three separate mechanisms are needed and only the first was present:

  1. ASSOCIATION.  Keep the right track attached to the right person when boxes
     overlap.  A Kalman/Hungarian solve over predicted-box IoU AND ArcFace
     cosine already did this (it moved here from `face_analyser`, unchanged in
     behaviour -- see BACKWARDS COMPATIBILITY below).

  2. PERSISTENCE.  When no detection arrives for a track, keep the face alive on
     the Kalman prediction

         x_{t|t-1} = F x_{t-1} + B u

     and hand the pipeline a synthetic Face built from that prediction, so the
     swap keeps running through the occlusion instead of blinking off.  This is
     what was missing: the old tracker predicted purely to score the assignment
     and returned only the detections it was given, so a frame with no detection
     produced no face no matter how confident the track was a frame earlier.

  3. TERMINATION.  Stop coasting before the prediction becomes fiction.  Two
     separate limits, because they answer different questions:

       MAX_COAST_FRAMES (15)  -- how long a *swap* may ride a prediction.  Past
                                 this the geometry is extrapolated far enough
                                 that pasting a face on it is a guess.
       MAX_LOST_FRAMES  (30)  -- how long the *track* survives for re-association.
                                 Longer, because a track that stops coasting is
                                 still the cheapest correct answer if the same
                                 person reappears; it costs nothing but a slot.

WHY COASTING IS GUARDED RATHER THAN TRUSTED
-------------------------------------------
A coasted face is invented.  It carries the track's mean embedding by
construction, so it passes every downstream identity gate automatically -- the
same property that makes `procmgr_tracking`'s gap-fill dangerous and the reason
that path has `_bridgeable` and `_interp_collides`.  A prediction that runs off
the end of a track paints a swapped face onto the background, which is a worse
artefact than the flicker it was meant to fix.

So every coasted face is:

  * capped at MAX_COAST_FRAMES consecutive frames;
  * refused unless the track was actually established (`min_hits_to_coast`);
  * refused once the predicted box leaves the frame by more than `max_outside`;
  * refused if it would land on another track's REAL detection (the caller
    supplies those; see `coast`);
  * stamped `_coasted=True`, `_interpolated=True` and `occlusion_state`, so the
    swap audit counts it as gap-filled rather than as a real detection.

`_interpolated` is set on purpose: this project's audit already treats that flag
as "landmarks nobody detected", and a new flag would have made coasted faces
invisible to a report that exists precisely to surface them.

BACKWARDS COMPATIBILITY
-----------------------
`FaceTrack` and `FaceTracker` were defined in `roop.face_analyser`, which
re-exports them from here.  `FaceTracker.update()` has the identical signature
and the identical return value -- the detections it was handed, with `_track_id`
stamped.  Coasting is a SEPARATE call (`coast`, or `update_with_coasting`), so
no existing caller changes behaviour by upgrading.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                        # scipy is a hard dep of the app, but
    from scipy.optimize import linear_sum_assignment
except Exception:                           # keep the module importable for the
    linear_sum_assignment = None            # pure-geometry helpers below.


# -- lifecycle constants ------------------------------------------------------
# Named at module scope because they are the two numbers a reviewer will look
# for, and because the harnesses import them rather than restating them.
MAX_LOST_FRAMES = 30
MAX_COAST_FRAMES = 15
MIN_HITS_TO_COAST = 3
LOW_CONFIDENCE = 0.5

# Occlusion states stamped on a face as `occlusion_state`.
STATE_VISIBLE = 'visible'
STATE_PARTIAL = 'partial'
STATE_COASTED = 'coasted'


# -- face-object access -------------------------------------------------------
# insightface's Face is a dict subclass whose __getattr__ returns None for any
# missing key, so `getattr(face, 'x', default)` silently yields None rather than
# the default. Every read goes through these two.

def _face_field(face: Any, name: str, default: Any = None) -> Any:
    """Read an InsightFace ``Face``, mapping, or small test-double uniformly."""
    if isinstance(face, dict):
        return face.get(name, default)
    try:
        value = face.get(name, default)
    except (AttributeError, TypeError):
        value = getattr(face, name, default)
    return default if value is None else value


def _set_face_field(face: Any, name: str, value: Any) -> None:
    """Stamp metadata without imposing a concrete detector-face type."""
    try:
        face[name] = value
        return
    except (TypeError, AttributeError):
        pass
    setattr(face, name, value)


def _bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection-over-union for ``[left, top, right, bottom]`` boxes."""
    ax0, ay0, ax1, ay1 = (float(v) for v in first)
    bx0, by0, bx1, by1 = (float(v) for v in second)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    second_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = first_area + second_area - intersection
    return intersection / union if union > 1e-6 else 0.0


def _cosine_similarity(first: Optional[np.ndarray],
                       second: Optional[np.ndarray]) -> float:
    """Return a bounded ArcFace cosine similarity; missing embeddings are neutral."""
    if first is None or second is None:
        return 0.0
    a = np.asarray(first, dtype=np.float32).reshape(-1)
    b = np.asarray(second, dtype=np.float32).reshape(-1)
    if a.shape != b.shape or not len(a):
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-6:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def _array(value, dtype=np.float32) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        out = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return None
    if out.size == 0 or not np.all(np.isfinite(out)):
        return None
    return out.copy()


# -- the tracklet -------------------------------------------------------------

@dataclass
class FaceTrack:
    """One constant-velocity face track in centre/aspect/height space.

    `state` is exactly ``[x, y, a, h, dx, dy, da, dh]``.  The trajectory stores
    are bounded deques: they exist so a coasted face can carry plausible
    landmarks and so symmetry inpainting has a reference pose, NOT as a general
    history buffer -- an unbounded one on a 60k-frame render is the host-RAM
    leak this project already fixed once.
    """

    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    embedding: Optional[np.ndarray]
    hits: int = 1
    missed: int = 0

    # Occlusion / persistence bookkeeping.
    coasted_run: int = 0
    last_seen_frame: int = -1
    last_frame_index: int = -1
    occlusion_state: str = STATE_VISIBLE

    # Last-known observation, carried onto coasted faces.
    template: Any = None
    landmarks: Optional[np.ndarray] = None
    kps: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None
    confidence: float = 0.0

    # Bounded trajectories.
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=16))
    landmark_history: deque = field(default_factory=lambda: deque(maxlen=16))

    @property
    def velocity(self) -> np.ndarray:
        """``[dx, dy, da, dh]`` per frame, from the Kalman state."""
        return np.asarray(self.state[4:8], dtype=np.float32).copy()

    def snapshot(self) -> Dict[str, Any]:
        """A plain dict for diagnostics and harnesses; no live references."""
        return {
            'id': int(self.track_id),
            'bbox': FaceTracker.measurement_to_bbox(self.state[:4]).tolist(),
            'velocity': self.velocity.tolist(),
            'hits': int(self.hits),
            'missed': int(self.missed),
            'coasted_run': int(self.coasted_run),
            'last_seen_frame': int(self.last_seen_frame),
            'confidence': float(self.confidence),
            'occlusion_state': self.occlusion_state,
            'has_embedding': self.embedding is not None,
            'landmark_history': len(self.landmark_history),
        }


class FaceTracker:
    """Deterministic Kalman/Hungarian tracker with occlusion coasting.

    ArcFace appearance and predicted-box IoU are solved together, preventing the
    left-to-right detector ordering from exchanging sources when people cross.
    """

    _MOTION_WEIGHT = 0.6
    _IDENTITY_WEIGHT = 0.4

    def __init__(self, max_age: int = MAX_LOST_FRAMES, max_cost: float = 0.85,
                 process_noise: float = 1.0, measurement_noise: float = 4.0,
                 max_coast: int = MAX_COAST_FRAMES,
                 min_hits_to_coast: int = MIN_HITS_TO_COAST,
                 history: int = 16, max_outside: float = 0.5):
        self.max_age = max(0, int(max_age))
        self.max_cost = float(max_cost)
        self.process_noise = max(1e-6, float(process_noise))
        self.measurement_noise = max(1e-6, float(measurement_noise))
        # Coasting can never outlive the track it rides on: a coast that
        # survived retirement would emit faces for a track that no longer
        # exists on the next frame, which reads as a one-frame flash.
        self.max_coast = max(0, min(int(max_coast), self.max_age))
        self.min_hits_to_coast = max(1, int(min_hits_to_coast))
        self.history = max(2, int(history))
        self.max_outside = float(max_outside)
        self.tracks: Dict[int, FaceTrack] = {}
        self._next_track_id = 0
        self._last_frame_index: Optional[int] = None
        self._lock = RLock()
        self.stats = {'coasted': 0, 'coast_refused_outside': 0,
                      'coast_refused_collide': 0, 'coast_refused_young': 0,
                      'coast_expired': 0, 'retired': 0}

    # -- geometry -------------------------------------------------------------

    @staticmethod
    def bbox_to_measurement(bbox: Sequence[float]) -> np.ndarray:
        """Convert an xyxy box to the Kalman measurement ``[x, y, a, h]``."""
        x0, y0, x1, y1 = (float(v) for v in bbox)
        height = max(1e-3, y1 - y0)
        width = max(1e-3, x1 - x0)
        return np.asarray(((x0 + x1) * 0.5, (y0 + y1) * 0.5,
                           width / height, height), dtype=np.float32)

    @staticmethod
    def measurement_to_bbox(measurement: Sequence[float]) -> np.ndarray:
        """Convert ``[x, y, a, h]`` back to an xyxy box."""
        x, y, aspect, height = (float(v) for v in measurement)
        height = max(1e-3, height)
        width = max(1e-3, aspect * height)
        return np.asarray((x - width * 0.5, y - height * 0.5,
                           x + width * 0.5, y + height * 0.5), dtype=np.float32)

    @staticmethod
    def _embedding(face: Any) -> Optional[np.ndarray]:
        embedding = _face_field(face, 'embedding')
        if embedding is None:
            return None
        try:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            return vector.copy() if vector.size and np.isfinite(vector).all() else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        transition = np.eye(8, dtype=np.float32)
        transition[0, 4] = transition[1, 5] = dt
        transition[2, 6] = transition[3, 7] = dt
        return transition

    def _predict(self, track: FaceTrack, dt: float) -> None:
        transition = self._transition(dt)
        # Velocity uncertainty should grow faster than measurement uncertainty.
        q = self.process_noise * np.diag((dt * dt, dt * dt, 0.01 * dt * dt,
                                          dt * dt, dt, dt, 0.01 * dt, dt)).astype(np.float32)
        track.state = transition @ track.state
        track.state[2] = max(track.state[2], 1e-3)
        track.state[3] = max(track.state[3], 1e-3)
        track.covariance = transition @ track.covariance @ transition.T + q

    def _update(self, track: FaceTrack, measurement: np.ndarray,
                embedding: Optional[np.ndarray]) -> None:
        observation = np.zeros((4, 8), dtype=np.float32)
        observation[np.arange(4), np.arange(4)] = 1.0
        innovation = measurement - observation @ track.state
        innovation_covariance = (observation @ track.covariance @ observation.T +
                                 np.eye(4, dtype=np.float32) * self.measurement_noise)
        gain = track.covariance @ observation.T @ np.linalg.pinv(innovation_covariance)
        track.state = track.state + gain @ innovation
        track.state[2] = max(track.state[2], 1e-3)
        track.state[3] = max(track.state[3], 1e-3)
        track.covariance = (np.eye(8, dtype=np.float32) - gain @ observation) @ track.covariance
        track.hits += 1
        track.missed = 0
        if embedding is not None:
            # Never blend an identity discontinuity into a stable ArcFace mean.
            if track.embedding is None or _cosine_similarity(track.embedding, embedding) >= 0.5:
                base = embedding if track.embedding is None else 0.9 * track.embedding + 0.1 * embedding
                norm = float(np.linalg.norm(base))
                track.embedding = (base / norm if norm > 1e-6 else base).astype(np.float32)

    def _new_track(self, measurement: np.ndarray,
                   embedding: Optional[np.ndarray]) -> FaceTrack:
        track = FaceTrack(
            track_id=self._next_track_id,
            state=np.concatenate((measurement, np.zeros(4, dtype=np.float32))).astype(np.float32),
            covariance=np.diag((16.0, 16.0, 0.04, 16.0,
                                64.0, 64.0, 0.16, 64.0)).astype(np.float32),
            embedding=embedding,
        )
        track.bbox_history = deque(maxlen=self.history)
        track.landmark_history = deque(maxlen=self.history)
        self.tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    # -- observation bookkeeping ----------------------------------------------

    def _observe(self, track: FaceTrack, face: Any, frame_index: int) -> None:
        """Record what a real detection told us, for later coasting."""
        track.template = face
        track.last_seen_frame = int(frame_index)
        track.coasted_run = 0
        track.occlusion_state = STATE_VISIBLE
        track.confidence = float(_face_field(face, 'det_score', 0.0) or 0.0)
        bbox = _array(_face_field(face, 'bbox'))
        if bbox is not None and bbox.size == 4:
            track.bbox_history.append(bbox.reshape(4))
        kps = _array(_face_field(face, 'kps'))
        if kps is not None:
            track.kps = kps
        landmarks = _array(_face_field(face, 'landmark_2d_106'))
        if landmarks is None:
            landmarks = kps
        if landmarks is not None:
            track.landmarks = landmarks
            track.landmark_history.append(landmarks)
        pose = _array(_face_field(face, 'pose'))
        if pose is not None:
            track.pose = pose.reshape(-1)

    # -- association ----------------------------------------------------------

    def association_cost_matrix(self, detections: Sequence[Any]) -> np.ndarray:
        """Return the specified IoU/ArcFace cost for active tracks and detections."""
        active = [self.tracks[track_id] for track_id in sorted(self.tracks)]
        matrix = np.empty((len(active), len(detections)), dtype=np.float32)
        for row, track in enumerate(active):
            predicted_bbox = self.measurement_to_bbox(track.state[:4])
            for column, face in enumerate(detections):
                detection_bbox = _face_field(face, 'bbox')
                if detection_bbox is None:
                    matrix[row, column] = np.inf
                    continue
                iou = _bbox_iou(predicted_bbox, detection_bbox)
                similarity = _cosine_similarity(track.embedding, self._embedding(face))
                matrix[row, column] = (self._MOTION_WEIGHT * (1.0 - iou) +
                                       self._IDENTITY_WEIGHT * (1.0 - similarity))
        return matrix

    def update(self, detections: Sequence[Any],
               frame_index: Optional[int] = None) -> List[Any]:
        """Associate detections, stamp ``_track_id``, and expire stale tracks.

        Signature and return value are unchanged from the pre-coasting tracker:
        the list handed in, with `_track_id` stamped on each face.  Coasted
        faces are NOT returned here -- see `coast` / `update_with_coasting` --
        so upgrading cannot change an existing caller's output.
        """
        faces = list(detections or ())
        with self._lock:
            if frame_index is None:
                frame_index = 0 if self._last_frame_index is None else self._last_frame_index + 1
            frame_index = int(frame_index)
            if self._last_frame_index is None:
                dt = 1.0
            else:
                # A worker may finish an old frame after a newer one.  It may use
                # the current prediction, but must not rewind the persistent state.
                dt = max(1.0, float(frame_index - self._last_frame_index))
            for track in self.tracks.values():
                self._predict(track, dt)
                track.missed += 1
                track.last_frame_index = frame_index

            cost = self.association_cost_matrix(faces)
            matched_detections = set()
            if cost.size and linear_sum_assignment is not None:
                rows, columns = linear_sum_assignment(cost)
                active_ids = sorted(self.tracks)
                for row, column in zip(rows, columns):
                    value = float(cost[row, column])
                    if not np.isfinite(value) or value > self.max_cost:
                        continue
                    track = self.tracks[active_ids[int(row)]]
                    face = faces[int(column)]
                    bbox = _face_field(face, 'bbox')
                    self._update(track, self.bbox_to_measurement(bbox),
                                 self._embedding(face))
                    self._observe(track, face, frame_index)
                    _set_face_field(face, '_track_id', track.track_id)
                    matched_detections.add(int(column))

            for index, face in enumerate(faces):
                if index in matched_detections:
                    continue
                bbox = _face_field(face, 'bbox')
                if bbox is None:
                    continue
                track = self._new_track(self.bbox_to_measurement(bbox),
                                        self._embedding(face))
                track.last_frame_index = frame_index
                self._observe(track, face, frame_index)
                _set_face_field(face, '_track_id', track.track_id)

            for track_id in tuple(self.tracks):
                if self.tracks[track_id].missed > self.max_age:
                    del self.tracks[track_id]
                    self.stats['retired'] += 1
            if self._last_frame_index is None or frame_index > self._last_frame_index:
                self._last_frame_index = frame_index
        return faces

    # -- persistence through occlusion ----------------------------------------

    def coast(self, frame_index: Optional[int] = None,
              frame_shape: Optional[Tuple[int, ...]] = None,
              occupied: Optional[Sequence[Any]] = None,
              collide_frac: float = 0.35) -> List[Any]:
        """Synthetic faces for tracks that had no detection on this frame.

        Call AFTER `update` for the same frame.  Returns a list (possibly empty)
        of Face-like objects built from each unmatched track's Kalman prediction,
        stamped `_coasted`, `_interpolated` and `occlusion_state`.

        `occupied` is the list of REAL faces already accepted for this frame; a
        prediction overlapping one of them by `collide_frac` of its own area is
        refused, because that is the shape of the bug where an invented face
        lands on the neighbour a close pair was occluding each other with.
        """
        produced: List[Any] = []
        if self.max_coast <= 0:
            return produced
        with self._lock:
            if frame_index is None:
                frame_index = self._last_frame_index
            frame_index = int(0 if frame_index is None else frame_index)
            real_boxes = []
            for other in (occupied or ()):
                box = _array(_face_field(other, 'bbox'))
                if box is not None and box.size == 4:
                    real_boxes.append(box.reshape(4))

            for track_id in sorted(self.tracks):
                track = self.tracks[track_id]
                if track.missed <= 0 or track.template is None:
                    continue                    # matched this frame, or never seen
                if track.coasted_run >= self.max_coast:
                    self.stats['coast_expired'] += 1
                    continue
                if track.hits < self.min_hits_to_coast:
                    self.stats['coast_refused_young'] += 1
                    continue
                bbox = self.measurement_to_bbox(track.state[:4])
                if not self._inside_frame(bbox, frame_shape, self.max_outside):
                    self.stats['coast_refused_outside'] += 1
                    continue
                if self._collides(bbox, real_boxes, collide_frac):
                    self.stats['coast_refused_collide'] += 1
                    continue
                face = self._coasted_face(track, bbox, frame_index)
                if face is None:
                    continue
                track.coasted_run += 1
                track.occlusion_state = STATE_COASTED
                self.stats['coasted'] += 1
                produced.append(face)
        return produced

    def update_with_coasting(self, detections: Sequence[Any],
                             frame_index: Optional[int] = None,
                             frame_shape: Optional[Tuple[int, ...]] = None,
                             collide_frac: float = 0.35) -> Tuple[List[Any], List[Any]]:
        """`update` then `coast`; returns ``(all_faces, coasted_only)``."""
        faces = self.update(detections, frame_index)
        if frame_index is None:
            frame_index = self._last_frame_index
        coasted = self.coast(frame_index, frame_shape, occupied=faces,
                             collide_frac=collide_frac)
        return list(faces) + list(coasted), coasted

    @staticmethod
    def _inside_frame(bbox: np.ndarray, frame_shape: Optional[Tuple[int, ...]],
                      max_outside: float) -> bool:
        """Is enough of the predicted box still on screen to be worth swapping?

        A constant-velocity prediction on a face that WALKED OUT of frame keeps
        marching off the edge, and every one of those frames would otherwise be
        a swap painted on nothing.  With no frame shape the caller cannot be
        judged, so nothing is refused.
        """
        if frame_shape is None or len(frame_shape) < 2:
            return True
        height, width = float(frame_shape[0]), float(frame_shape[1])
        area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        ix = max(0.0, min(bbox[2], width) - max(bbox[0], 0.0))
        iy = max(0.0, min(bbox[3], height) - max(bbox[1], 0.0))
        return (1.0 - (ix * iy) / area) <= max_outside

    @staticmethod
    def _collides(bbox: np.ndarray, real_boxes: Sequence[np.ndarray],
                  collide_frac: float) -> bool:
        if collide_frac <= 0 or not real_boxes:
            return False
        area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
        for other in real_boxes:
            iw = max(0.0, min(bbox[2], other[2]) - max(bbox[0], other[0]))
            ih = max(0.0, min(bbox[3], other[3]) - max(bbox[1], other[1]))
            if (iw * ih) / area >= collide_frac:
                return True
        return False

    def _coasted_face(self, track: FaceTrack, bbox: np.ndarray,
                      frame_index: int) -> Optional[Any]:
        """Build a Face from the prediction, translating the last known geometry.

        The landmarks are the last observed ones RIGID-SHIFTED and SCALED onto
        the predicted box.  They are not re-predicted point by point: a
        per-landmark Kalman on an occluded face fits noise, and the swap only
        needs the five alignment keypoints to be in the right place -- the whole
        crop is a similarity transform of them.
        """
        template = track.template
        try:
            face = type(template)(template)     # insightface Face is a dict subclass
        except Exception:
            try:
                face = dict(template)
            except Exception:
                return None

        last = track.bbox_history[-1] if track.bbox_history else None
        if last is None:
            return None
        lw = max(1e-3, float(last[2] - last[0]))
        lh = max(1e-3, float(last[3] - last[1]))
        sx = float(bbox[2] - bbox[0]) / lw
        sy = float(bbox[3] - bbox[1]) / lh
        lcx, lcy = (float(last[0] + last[2]) * 0.5, float(last[1] + last[3]) * 0.5)
        pcx, pcy = (float(bbox[0] + bbox[2]) * 0.5, float(bbox[1] + bbox[3]) * 0.5)

        def _carry(points):
            pts = np.asarray(points, dtype=np.float64)
            out = pts.copy()
            out[..., 0] = (pts[..., 0] - lcx) * sx + pcx
            out[..., 1] = (pts[..., 1] - lcy) * sy + pcy
            return out.astype(np.float32)

        _set_face_field(face, 'bbox', bbox.astype(np.float32))
        for key in ('kps', 'landmark_2d_106', 'landmark_3d_68'):
            value = _array(_face_field(template, key))
            if value is None:
                continue
            if key == 'landmark_3d_68' and value.ndim == 2 and value.shape[1] == 3:
                moved = value.copy()
                moved[:, :2] = _carry(value[:, :2])
                _set_face_field(face, key, moved.astype(np.float32))
            else:
                _set_face_field(face, key, _carry(value))
        if track.embedding is not None:
            # The track mean, not the last frame's: the last frame before an
            # occlusion is the most contaminated one there is.
            _set_face_field(face, 'embedding', track.embedding.astype(np.float32))
        # Confidence decays with the length of the coast so anything reading
        # det_score sees a prediction weakening rather than a fresh detection.
        decay = 1.0 - (track.coasted_run + 1) / float(self.max_coast + 1)
        _set_face_field(face, 'det_score',
                        np.float32(max(0.05, track.confidence * decay)))
        _set_face_field(face, '_track_id', int(track.track_id))
        _set_face_field(face, '_coasted', True)
        _set_face_field(face, '_coast_age', int(track.coasted_run + 1))
        _set_face_field(face, '_interpolated', True)
        _set_face_field(face, 'occlusion_state', STATE_COASTED)
        return face

    # -- diagnostics ----------------------------------------------------------

    def tracklets(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self.tracks[i].snapshot() for i in sorted(self.tracks)]

    def summary_line(self) -> str:
        s = self.stats
        return ('[Tracker] %d coasted face(s); refused %d off-frame, %d colliding, '
                '%d too-young; %d coast(s) expired at %d frames; %d track(s) retired '
                'after %d lost frames.'
                % (s['coasted'], s['coast_refused_outside'], s['coast_refused_collide'],
                   s['coast_refused_young'], s['coast_expired'], self.max_coast,
                   s['retired'], self.max_age))

    def reset(self) -> None:
        with self._lock:
            self.tracks.clear()
            self._last_frame_index = None
            for key in self.stats:
                self.stats[key] = 0


# =============================================================================
# Occlusion-aware landmark symmetry inpainting
# =============================================================================
#
# WHAT IS BEING REPAIRED.  When a hand covers half a face the detector still
# returns a full landmark set -- it does not report "these twelve points are
# behind something".  The points under the occluder are then wherever the
# regressor guessed, and because the swap's alignment is a similarity fit over
# the five keypoints, a hallucinated eye or mouth corner drags the whole crop.
#
# THE AXIS.  A face is bilaterally symmetric about the plane through the nasal
# bridge.  Its image is the line from the midpoint of the eyes to the midpoint
# of the mouth corners -- the same axis `orientation.roll_from_face` uses, and
# the reason that one was chosen over the detector's box: it survives roll and
# in-plane rotation because it is DERIVED from the landmarks rather than from
# the axis-aligned box.
#
# Under yaw the two halves are no longer mirror images in the image plane (the
# far half is foreshortened), so the reflection is corrected by the measured
# half-widths of the VISIBLE points before it is used.  Under extreme yaw the
# far half is not visible at all and there is nothing to mirror FROM, which is
# why `symmetry_inpaint_landmarks` refuses past `max_yaw` rather than inventing.
#
# THE PAIRING.  No landmark-index table is hardcoded.  Mirror partners are
# derived from a complete, well-conditioned observation of the same landmark
# set by canonicalising it onto the axis and matching each point to the one
# nearest its own reflection.  The derivation is only accepted when it is a
# proper involution with a small residual, so a bad reference is refused rather
# than silently producing a wrong permutation.  This is why the tracker keeps a
# landmark trajectory: the reference comes from the track's own best frame.

def symmetry_axis(points: Optional[np.ndarray] = None,
                  kps: Optional[np.ndarray] = None
                  ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(origin, unit_axis)`` for the nasal-bridge midline, or None.

    `kps` is the detector's five keypoints ``[eye_l, eye_r, nose, mouth_l,
    mouth_r]``; it is preferred because those five are the points the alignment
    itself uses.  With only a dense set available, the axis is taken from its
    principal direction, which for a face is the vertical midline.
    """
    five = _array(kps)
    if five is not None and five.ndim == 2 and five.shape[0] >= 5:
        eyes = (five[0, :2] + five[1, :2]) * 0.5
        mouth = (five[3, :2] + five[4, :2]) * 0.5
        axis = mouth - eyes
        norm = float(np.linalg.norm(axis))
        if norm > 1e-6:
            return eyes.astype(np.float64), (axis / norm).astype(np.float64)

    dense = _array(points)
    if dense is None or dense.ndim != 2 or dense.shape[0] < 3:
        return None
    flat = dense[:, :2].astype(np.float64)
    origin = flat.mean(axis=0)
    centred = flat - origin
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    axis = vt[0]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        return None
    return origin, (axis / norm)


def _canonical(points: np.ndarray, origin: np.ndarray,
               axis: np.ndarray) -> np.ndarray:
    """Rotate/translate so the midline is +y through the origin.

    Returns ``[across, along]`` per point: `across` is signed distance from the
    midline (its sign is the face's left/right), `along` runs eyes -> mouth.
    """
    normal = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    delta = points[:, :2].astype(np.float64) - origin
    return np.stack((delta @ normal, delta @ axis), axis=1)


def _uncanonical(canonical: np.ndarray, origin: np.ndarray,
                 axis: np.ndarray) -> np.ndarray:
    normal = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    return (origin
            + canonical[:, 0:1] * normal[None, :]
            + canonical[:, 1:2] * axis[None, :])


def derive_mirror_map(points: np.ndarray, kps: Optional[np.ndarray] = None,
                      tolerance: float = 0.12) -> Optional[np.ndarray]:
    """Mirror-partner index per landmark, derived from one complete observation.

    Returns an integer array `m` where `m[i]` is the index of `i`'s reflection
    (a self-symmetric point on the midline maps to itself), or None when the
    observation is too asymmetric to trust.  `tolerance` is the largest accepted
    residual as a fraction of the face's own span, so it is scale-free.

    REFUSAL IS THE POINT.  A permutation derived from a yawed or half-occluded
    face is wrong in a way that is invisible downstream -- it would mirror the
    chin onto the brow.  Two conditions must hold: the mapping must be an
    involution (`m[m[i]] == i`), and every residual must be inside tolerance.
    """
    dense = _array(points, np.float64)
    if dense is None or dense.ndim != 2 or dense.shape[0] < 5:
        return None
    frame = symmetry_axis(dense, kps)
    if frame is None:
        return None
    origin, axis = frame
    canonical = _canonical(dense, origin, axis)
    span = float(np.linalg.norm(canonical.max(axis=0) - canonical.min(axis=0)))
    if span <= 1e-6:
        return None

    reflected = canonical.copy()
    reflected[:, 0] *= -1.0
    # Full pairwise distance from each reflected point to every real point.
    diff = reflected[:, None, :] - canonical[None, :, :]
    distance = np.sqrt((diff * diff).sum(axis=2))
    partner = np.argmin(distance, axis=1).astype(np.int32)
    residual = distance[np.arange(len(partner)), partner]
    if float(residual.max()) > tolerance * span:
        return None
    if not np.array_equal(partner[partner], np.arange(len(partner), dtype=np.int32)):
        return None
    return partner


def landmark_visibility(points: np.ndarray, occluder_mask: Optional[np.ndarray],
                        threshold: float = 0.5,
                        origin: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Boolean visibility per landmark, read off a foreground-occluder mask.

    `occluder_mask` is full-frame, float in [0, 1], HIGH where a foreground
    object sits in front of the face -- i.e. exactly what `Mask_Occluder`
    produces and what `M_composite = M_face * (1 - M_occluder)` subtracts.
    `origin` shifts the landmarks into the mask's coordinate frame when the mask
    covers a crop rather than the whole frame.

    With no mask everything is reported visible: this function must never invent
    an occlusion, because inpainting a landmark that was fine replaces a
    measurement with an estimate.
    """
    dense = _array(points, np.float64)
    if dense is None or dense.ndim != 2:
        return np.ones(0, dtype=bool)
    visible = np.ones(len(dense), dtype=bool)
    mask = _array(occluder_mask, np.float32)
    if mask is None or mask.ndim < 2:
        return visible
    if mask.ndim == 3:
        mask = mask[..., 0]
    height, width = mask.shape[:2]
    xs = np.round(dense[:, 0] - float(origin[0])).astype(np.int64)
    ys = np.round(dense[:, 1] - float(origin[1])).astype(np.int64)
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    sampled = np.zeros(len(dense), dtype=np.float32)
    sampled[inside] = mask[ys[inside], xs[inside]]
    visible[inside] = sampled[inside] < float(threshold)
    return visible


def symmetry_inpaint_landmarks(points: np.ndarray, visible: Optional[np.ndarray],
                               kps: Optional[np.ndarray] = None,
                               mirror_map: Optional[np.ndarray] = None,
                               reference: Optional[np.ndarray] = None,
                               pose: Optional[Sequence[float]] = None,
                               max_yaw: float = 55.0
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Replace occluded landmarks with their mirrored partners.

    Returns ``(points, filled)`` -- a copy with the repairs applied, and a
    boolean array marking which indices were estimated rather than measured.
    The input is returned unchanged (and `filled` all-False) whenever the repair
    cannot be justified:

      * nothing is occluded;
      * no usable mirror map (see `derive_mirror_map` -- refusal is deliberate);
      * `|yaw| > max_yaw`, where the far half is foreshortened past the point
        that a planar reflection describes it.  This project has measured
        `solve_pose_5pt` to be 15-20 degrees off per person, so the limit is set
        well inside where the geometry actually breaks rather than at it;
      * the partner of an occluded point is itself occluded -- two hidden halves
        leave nothing to mirror from, and a midline point is its own partner so
        it can never be repaired this way.
    """
    dense = _array(points, np.float64)
    if dense is None or dense.ndim != 2 or dense.shape[0] == 0:
        return _array(points, np.float32), np.zeros(0, dtype=bool)
    count = len(dense)
    filled = np.zeros(count, dtype=bool)
    if visible is None:
        return dense.astype(np.float32), filled
    seen = np.asarray(visible, dtype=bool).reshape(-1)
    if seen.size != count or seen.all():
        return dense.astype(np.float32), filled

    if pose is not None:
        try:
            yaw = float(np.asarray(pose, dtype=np.float64).reshape(-1)[1])
            if abs(yaw) > float(max_yaw):
                return dense.astype(np.float32), filled
        except (TypeError, ValueError, IndexError):
            pass

    partner = mirror_map
    if partner is None:
        source = reference if reference is not None else dense
        partner = derive_mirror_map(source, kps)
    if partner is None or len(partner) != count:
        return dense.astype(np.float32), filled

    frame = symmetry_axis(dense[seen] if seen.any() else dense, kps)
    if frame is None:
        return dense.astype(np.float32), filled
    origin, axis = frame
    canonical = _canonical(dense, origin, axis)

    # Foreshortening correction. Under yaw the visible half is wider or narrower
    # in the image than the hidden one; a raw reflection would inherit the
    # WRONG half-width. Scale by the ratio the visible pairs actually show.
    ratios = []
    for i in range(count):
        j = int(partner[i])
        if j == i or not (seen[i] and seen[j]):
            continue
        near, far = abs(canonical[i, 0]), abs(canonical[j, 0])
        if near > 1e-3 and far > 1e-3:
            ratios.append(near / far)
    scale = float(np.median(ratios)) if ratios else 1.0
    scale = float(np.clip(scale, 0.5, 2.0))

    repaired = canonical.copy()
    for i in range(count):
        if seen[i]:
            continue
        j = int(partner[i])
        if j == i or not seen[j]:
            continue                    # midline point, or both halves hidden
        repaired[i, 0] = -canonical[j, 0] * scale
        repaired[i, 1] = canonical[j, 1]
        filled[i] = True

    if not filled.any():
        return dense.astype(np.float32), filled
    out = dense.copy()
    out[filled, :2] = _uncanonical(repaired[filled], origin, axis)
    return out.astype(np.float32), filled


def occlusion_state_for(visible: Optional[np.ndarray],
                        coasted: bool = False,
                        partial_frac: float = 0.08) -> str:
    """Name this face's occlusion state for the `occlusion_state` stamp.

    `partial_frac` is the share of landmarks that must be behind something
    before a face is called partially occluded.  It is not zero because a
    single landmark grazing a mask edge is segmentation noise, and flagging on
    it would put most frames of most clips into the partial state -- a flag
    that is always on carries no information.
    """
    if coasted:
        return STATE_COASTED
    if visible is None:
        return STATE_VISIBLE
    seen = np.asarray(visible, dtype=bool).reshape(-1)
    if seen.size == 0:
        return STATE_VISIBLE
    hidden = 1.0 - float(seen.mean())
    return STATE_PARTIAL if hidden >= float(partial_frac) else STATE_VISIBLE


__all__ = [
    'MAX_LOST_FRAMES', 'MAX_COAST_FRAMES', 'MIN_HITS_TO_COAST', 'LOW_CONFIDENCE',
    'STATE_VISIBLE', 'STATE_PARTIAL', 'STATE_COASTED',
    'FaceTrack', 'FaceTracker',
    'symmetry_axis', 'derive_mirror_map', 'landmark_visibility',
    'symmetry_inpaint_landmarks', 'occlusion_state_for',
]
