"""Ordered, bounded state for one video render worker or overlap block."""
from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from roop.temporal_stabilizer import OneEuroFilter
from roop.nonfrontal import NonFrontalRouter


MISS_GRACE = 3


from roop.scene_detector import ShotBoundaryDetector


class SourceSlots:
    """Keep each visible track's source slot when another face leaves or crosses it."""

    def __init__(self):
        self.active = {}

    def update(self, faces, index, capacity):
        self.active = {tid: entry for tid, entry in self.active.items()
                       if index - entry[1] <= MISS_GRACE + 1}
        used = {slot for slot, _ in self.active.values()}
        available = iter(i for i in range(capacity) if i not in used)
        for face in sorted(faces, key=lambda f: float(f.bbox[0])):
            tid = face.get('_track_id')
            if tid is None:
                continue
            entry = self.active.get(tid)
            slot = entry[0] if entry is not None else next(available, None)
            if slot is not None:
                self.active[tid] = (slot, index)
                face['_temporal_source_slot'] = slot


def _geometry_filter():
    return OneEuroFilter(min_cutoff=0.08, beta=5.0, d_cutoff=6.0)


def _adaptive_beta(value, filt, jitter, fast):
    """Open one cutoff for all coordinates, based on displacement above jitter."""
    if filt.x_prev is None:
        return 0.0
    displacement = np.linalg.norm((value - filt.x_prev).reshape(-1, 2), axis=1).mean()
    return filt.beta * float(np.clip((displacement - jitter) / (fast - jitter), 0.0, 1.0))


@dataclass
class _TrackHistory:
    index: int
    landmarks: OneEuroFilter = field(default_factory=_geometry_filter)
    bbox: OneEuroFilter = field(default_factory=_geometry_filter)
    affine: OneEuroFilter = field(default_factory=_geometry_filter)
    template: tuple | None = None
    motion: float = 0.0
    values: dict = field(default_factory=dict)


class VideoTemporalState:
    def __init__(self, fps=30.0):
        self.fps = max(1.0, float(fps))
        self.shots = ShotBoundaryDetector()
        self.index = None
        self.tracks = {}
        self.slots = SourceSlots()
        self.patches = {}
        self.selected_sources = {}
        self.nonfrontal_router = NonFrontalRouter()
        self.enhancer_gate = {}
        self.last_swapped_frame = None
        self.num_frames_no_face = 0

    def reset(self):
        self.tracks.clear()
        self.slots = SourceSlots()
        self.patches.clear()
        self.selected_sources.clear()
        self.nonfrontal_router.reset()
        self.enhancer_gate.clear()
        self.last_swapped_frame = None
        self.num_frames_no_face = 0

    def begin(self, frame, index, known_cut=False):
        discontinuity = self.index is not None and index != self.index + 1
        if discontinuity:
            self.shots.previous = None
        content_cut = self.shots.observe(frame)
        self.cut = bool(known_cut or content_cut or discontinuity)
        self.index = index
        if self.cut:
            self.reset()
        else:
            for tid in [tid for tid, state in self.tracks.items() if index - state.index > MISS_GRACE + 1]:
                del self.tracks[tid]
                self.patches.pop(tid, None)
                self.selected_sources.pop(tid, None)
                self.enhancer_gate.pop(tid, None)
        return self.cut

    def track(self, tid):
        state = self.tracks.get(tid)
        if state is None or self.index - state.index > 1:
            state = self.tracks[tid] = _TrackHistory(self.index)
        state.index = self.index
        return state

    def geometry(self, face, tid):
        points = np.asarray(face.kps, dtype=np.float64)
        box = np.asarray(face.bbox, dtype=np.float64)
        if points.shape != (5, 2) or box.shape != (4,) or not np.isfinite(points).all() or not np.isfinite(box).all():
            return
        state = self.track(tid)
        if state.landmarks.t_prev == self.index / self.fps:
            return
        extent = max(16.0, float(np.max(box[2:] - box[:2])))
        if state.landmarks.x_prev is not None:
            state.motion = float(np.linalg.norm(points - state.landmarks.x_prev, axis=1).mean() / extent)
        timestamp = self.index / self.fps
        jitter, fast = max(.55, .004 * extent), max(3., .03 * extent)
        coast = bool(face.get('_coasted'))
        beta = state.landmarks.beta if coast else _adaptive_beta(points, state.landmarks, jitter, fast)
        filtered = state.landmarks(points, timestamp, beta=beta)
        face.kps = filtered.astype(np.float32)
        face.bbox = state.bbox(box, timestamp, beta=beta).astype(np.float32)
        # Move the dense landmarks with the same fit that moved the five points.
        raw = points - points.mean(axis=0)
        smooth = filtered - filtered.mean(axis=0)
        norm = np.sum(raw * raw)
        if norm > 1e-6:
            a = np.sum(raw * smooth) / norm
            b = np.sum(raw[:, 0] * smooth[:, 1] - raw[:, 1] * smooth[:, 0]) / norm
            rotation = np.array([[a, -b], [b, a]])
            for name in ('landmark_2d_106', 'landmark_3d_68'):
                dense = face.get(name)
                if dense is not None:
                    dense = np.array(dense, dtype=np.float32, copy=True)
                    if dense.ndim == 2 and dense.shape[1] >= 2:
                        dense[:, :2] = (dense[:, :2] - points.mean(axis=0)) @ rotation.T + filtered.mean(axis=0)
                        face[name] = dense

    def matrix(self, tid, matrix, size, template):
        state = self.track(tid)
        if state.template != template:
            state.affine.reset()
            state.template = template
        filt = state.affine
        timestamp = self.index / self.fps
        scale = math.hypot(matrix[0, 0], matrix[1, 0])
        angle = math.atan2(matrix[1, 0], matrix[0, 0])
        # Express similarity at its canonical centre to keep translation, scale
        # and roll in comparable units, even for a face near the frame edge.
        inverse = cv2.invertAffineTransform(matrix)
        centre = inverse[:, :2] @ np.full(2, size * .5) + inverse[:, 2]
        value = np.array([*centre, size / scale, angle * size / scale])
        if filt.x_prev is not None:
            displacement = np.linalg.norm(centre - filt.x_prev[:2]) / max(16., value[2] * .5)
            state.motion = max(state.motion, float(displacement))
            previous_angle = filt.x_prev[3] / filt.x_prev[2]
            angle = previous_angle + (angle - previous_angle + math.pi) % (2 * math.pi) - math.pi
            value[3] = angle * value[2]
        if filt.t_prev != timestamp:
            value = filt(value, timestamp, beta=_adaptive_beta(value, filt, .35, 2.0))
        else:
            value = filt.x_prev
        angle = value[3] / value[2]
        a, b = (size / value[2]) * np.array([math.cos(angle), math.sin(angle)])
        linear = np.array([[a, -b], [b, a]])
        matrix[:, :2] = linear
        matrix[:, 2] = size * .5 - linear @ value[:2]

    def _previous(self, tid, key, value):
        state = self.track(tid)
        previous = state.values.get(key)
        state.values[key] = (self.index, value)
        return (previous[1] if previous is not None and previous[0] == self.index - 1 else None), state

    def mask(self, tid, alpha, guide, coast=False, source_slot=0, occluder=None):
        size = (96, 96)
        small = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        gray = _guide(guide, size)
        key = ('alpha', source_slot)
        previous, state = self._previous(tid, key, (small, gray))
        if previous is None:
            if coast:
                state.values.pop(key, None)
                return alpha * 0.75
            return alpha
        old_alpha, old_gray = previous
        visible = np.clip(1.0 - cv2.absdiff(gray, old_gray).astype(np.float32) / 18., 0., 1.)
        motion = max(0., 1. - state.motion / .07)
        # A strong current occlusion vetoes history. A predicted face may
        # reuse the matte only where its current plate agrees with the last.
        stable_edge = np.clip(1.0 - np.abs(old_alpha - small) / .55, 0., 1.)
        weight = .66 * motion * visible * stable_edge
        delta = (old_alpha - small) * weight
        if coast:
            delta = (np.minimum(old_alpha, small) * visible - small) * .7
        state.values[key] = (self.index, (small + delta, gray))
        correction = cv2.resize(delta, (alpha.shape[1], alpha.shape[0]))
        if occluder is not None:
            correction = np.where(occluder > .35, np.minimum(correction, 0.), correction)
        return np.where(alpha > .03, np.clip(alpha + correction, 0., 1.), alpha)

    def color(self, tid, key, stats):
        stats = tuple(np.array(item, copy=True) for item in stats)
        previous, state = self._previous(tid, key, (stats[2], stats))
        if previous is None:
            return stats
        old_raw, old = previous
        lighting_change = float(np.max(np.abs(stats[2] - old_raw)))
        weight = .55 * max(0., 1. - state.motion / .06) * max(0., 1. - lighting_change / 12.)
        smoothed = tuple(now + weight * (then - now) for now, then in zip(stats, old))
        state.values[key] = (self.index, (stats[2], smoothed))
        return smoothed

    def texture(self, tid, key, crop, guide, keep):
        size = (crop.shape[1], crop.shape[0])
        crop_f = crop.astype(np.float32)
        low = cv2.GaussianBlur(crop_f, (0, 0), 1.25)
        detail = crop_f - low
        gray = _guide(guide, size)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        previous, state = self._previous(tid, key, (detail, gray, crop_gray))
        if previous is None:
            return crop
        if len(previous) == 2:
            old_detail, old_gray = previous
            old_crop_gray = old_gray
        else:
            old_detail, old_gray, old_crop_gray = previous
        if old_detail.ndim == 2 and detail.ndim == 3:
            old_detail = old_detail[:, :, None]
        change = cv2.dilate(cv2.absdiff(gray, old_gray), np.ones((5, 5), np.uint8))
        # Macro change on low-frequency band avoids triggering on restorer's own pore noise
        crop_macro = cv2.GaussianBlur(crop_gray, (0, 0), 3.0)
        old_crop_macro = cv2.GaussianBlur(old_crop_gray, (0, 0), 3.0)
        crop_change = cv2.dilate(cv2.absdiff(crop_macro, old_crop_macro), np.ones((5, 5), np.uint8))
        total_change = np.maximum(change, crop_change)
        gate = np.clip(1. - total_change.astype(np.float32) / 14., 0., 1.)
        if keep is not None:
            gate *= cv2.resize(keep, size, interpolation=cv2.INTER_LINEAR)
        delta = (old_detail - detail) * gate[:, :, None] * (.75 * max(0., 1. - state.motion / .08))
        state.values[key] = (self.index, (detail + delta, gray, crop_gray))
        return np.clip(np.rint(crop_f + delta), 0, 255).astype(np.uint8)


def _guide(crop, size):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
