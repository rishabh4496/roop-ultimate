# SPDX-License-Identifier: MIT
"""Clean ByteTrack-style two-pass association; no upstream tracker code/weights.

Reuses this application's constant-velocity Kalman implementation. SciPy's
Hungarian solver is BSD-3-Clause. Low-score observations update geometry only.
"""
from types import MappingProxyType

import numpy as np
from scipy.optimize import linear_sum_assignment

from roop.tracker import (FaceTracker, _bbox_iou, _face_field, _set_face_field,
                          _cosine_similarity)


def locked_source(bindings, face, allowed_sources):
    """Read a frozen pre-pass binding, restricted to this run's selection.

    Guessed geometry is not permission to paste. Mutable legacy maps retain
    their original appearance-gated semantics.
    """
    if not isinstance(bindings, MappingProxyType):
        return None
    if _face_field(face, '_coasted', False) or _face_field(face, '_interpolated', False):
        return None
    binding = bindings.get(_face_field(face, '_track_id'))
    source = binding[0] if isinstance(binding, (tuple, list)) and binding else binding
    return source if source is not None and source in allowed_sources else None


class MultiFaceTracker(FaceTracker):
    _MOTION_WEIGHT = .55
    _IDENTITY_WEIGHT = .45

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bindings = {}

    @staticmethod
    def _embedding(face):
        vector = FaceTracker._embedding(face)
        if vector is None:
            return None
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-6 else None

    @property
    def bindings(self):
        return MappingProxyType(self._bindings)

    def bind(self, track_id, faceset_id):
        """First reference/manual assignment wins until retirement; no stealing."""
        with self._lock:
            if track_id not in self.tracks:
                raise KeyError(track_id)
            if track_id in self._bindings and self._bindings[track_id] != faceset_id:
                raise ValueError('track already bound to another faceset')
            self._bindings[track_id] = faceset_id

    def reset(self):
        with self._lock:
            super().reset()
            self._bindings.clear()

    def update(self, detections, frame_index=None):
        with self._lock:
            index = (0 if self._last_frame_index is None else self._last_frame_index + 1) if frame_index is None else int(frame_index)
            if self._last_frame_index is not None and index <= self._last_frame_index:
                return []  # unordered workers cannot advance temporal state
            dt = 1 if self._last_frame_index is None else index - self._last_frame_index
            for tid in list(self.tracks):
                track = self.tracks[tid]
                if track.missed + dt > self.max_age + 1:
                    del self.tracks[tid]
                    self._bindings.pop(tid, None)
                    continue
                self._predict(track, dt)
                track.missed += dt
                track.last_frame_index = index
            faces = []
            for f in detections or ():
                bb = np.asarray(_face_field(f, 'bbox', []))
                if bb.shape == (4,) and np.isfinite(bb).all() and np.all(bb[2:] > bb[:2]):
                    faces.append(f)
            high = [i for i, f in enumerate(faces) if float(_face_field(f, 'det_score', 1.) or 0.) >= .55]
            low = [i for i, f in enumerate(faces) if .20 <= float(_face_field(f, 'det_score', 1.) or 0.) < .55]
            remaining = sorted(self.tracks)
            matched = set()

            def associate(indices, geometry_only):
                nonlocal remaining
                cost = np.full((len(remaining), len(indices)), 1e5)
                for r, tid in enumerate(remaining):
                    t = self.tracks[tid]
                    for c, fi in enumerate(indices):
                        f = faces[fi]
                        iou = _bbox_iou(self.measurement_to_bbox(t.state[:4]), _face_field(f, 'bbox'))
                        embedding = self._embedding(f)
                        sim = _cosine_similarity(t.embedding, embedding)
                        dirty = bool(_face_field(f, '_emb_contam', 0) > .4)
                        if geometry_only or dirty:
                            if iou >= .10:
                                cost[r, c] = 1 - iou
                        elif t.embedding is None or embedding is None:
                            if iou >= .35:
                                cost[r, c] = 1 - iou
                        else:
                            # Strong geometric continuity retains a track through a noisy
                            # profile. A weak spatial claim needs stronger identity evidence.
                            threshold = .24 if iou >= .45 and t.missed <= 2 else .44
                            if iou >= .01 and sim >= threshold:
                                cost[r, c] = self._MOTION_WEIGHT * (1 - iou) + self._IDENTITY_WEIGHT * (1 - sim)
                used = set()
                if cost.size:
                    # Gate BEFORE assignment; invalid edges cannot steal valid ones.
                    limit = .90 if geometry_only else self.max_cost
                    cost[cost > limit] = 1e5
                    for r, c in zip(*linear_sum_assignment(cost)):
                        if cost[r, c] >= 1e4:
                            continue
                        tid, fi = remaining[r], indices[c]
                        t, f = self.tracks[tid], faces[fi]
                        emb = None if geometry_only or _face_field(f, '_emb_contam', 0) > .4 else self._embedding(f)
                        self._update(t, self.bbox_to_measurement(_face_field(f, 'bbox')), emb)
                        self._observe(t, f, index)
                        _set_face_field(f, '_track_id', tid)
                        matched.add(fi)
                        used.add(tid)
                remaining = [tid for tid in remaining if tid not in used]

            associate(high, False)
            associate(low, True)
            for fi in high:
                if fi not in matched:
                    f = faces[fi]
                    t = self._new_track(self.bbox_to_measurement(_face_field(f, 'bbox')), self._embedding(f))
                    self._observe(t, f, index)
                    _set_face_field(f, '_track_id', t.track_id)
                    matched.add(fi)
            for tid in list(self.tracks):
                if self.tracks[tid].missed > self.max_age:
                    del self.tracks[tid]
                    self._bindings.pop(tid, None)
                    self.stats['retired'] += 1
            self._last_frame_index = index
            return [f for i, f in enumerate(faces) if i in matched]
