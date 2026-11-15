"""Small CPU geometry guards; no inference/provider or device-buffer ownership."""
import threading
from collections import OrderedDict
import numpy as np


class InvalidAlignment(ValueError):
    """A face cannot safely be sampled; retain its original pixels."""


def landmarks5(points, bbox=None):
    if hasattr(points, 'detach'):
        points = points.detach().cpu().numpy()
    p = np.asarray(points, dtype=np.float32)
    if p.shape != (5, 2) or not np.isfinite(p).all():
        raise InvalidAlignment("nonfinite or malformed five-point landmarks")
    eye = p[1] - p[0]
    down = (p[3] + p[4] - p[0] - p[1]) * .5
    extent = max(float(np.ptp(p, axis=0).max()), 1e-6)
    iod = float(np.linalg.norm(eye))
    depth = float(np.linalg.norm(down))
    # Rotation invariant semantic ordering: do not use screen x/y ordering.
    cross = float(eye[0] * down[1] - eye[1] * down[0])
    if iod < max(.5, .015 * extent) or depth < max(.5, .05 * extent) or cross <= .01 * iod * depth:
        raise InvalidAlignment("collapsed or reversed landmark baselines")
    if bbox is not None:
        if hasattr(bbox, 'detach'):
            bbox = bbox.detach().cpu().numpy()
        b = np.asarray(bbox, dtype=np.float32)
        if b.shape != (4,) or not np.isfinite(b).all() or np.any(b[2:] <= b[:2]):
            raise InvalidAlignment("invalid bounding box")
        size = b[2:] - b[:2]
        if np.any(p < b[:2] - .25 * size) or np.any(p > b[2:] + .25 * size):
            raise InvalidAlignment("landmarks outside face bounding box")
        if extent < .08 * float(size.max()) or extent > 1.8 * float(size.max()):
            raise InvalidAlignment("landmark/bounding-box scale mismatch")
    return p


def affine(matrix, points=None, template=None, sar=1.):
    if hasattr(matrix, 'detach'):
        matrix = matrix.detach().cpu().numpy()
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (2, 3) or not np.isfinite(m).all():
        raise InvalidAlignment("nonfinite affine")
    a = m[:, :2].copy()
    try:
        sar_f = float(sar)
    except (TypeError, ValueError):
        sar_f = 1.0
    if not np.isfinite(sar_f) or sar_f <= 0.0:
        sar_f = 1.0
    a[:, 0] /= sar_f
    det = float(np.linalg.det(a))
    s = np.linalg.svd(a, compute_uv=False)
    if det <= 1e-10 or s[-1] < 1e-5 or s[0] > 1e4 or (s[0] / max(float(s[-1]), 1e-12)) > 1.05:
        raise InvalidAlignment("reflected, collapsed or anisotropic affine")
    if points is not None and template is not None:
        pts = np.asarray(points, np.float64)
        tmpl = np.asarray(template, np.float64)
        mapped = pts @ m[:, :2].T + m[:, 2]
        residual = np.sqrt(np.mean(np.sum((mapped - tmpl) ** 2, axis=1)))
        extent = max(float(np.ptp(tmpl, axis=0).max()), 1.)
        if residual / extent > .43:
            raise InvalidAlignment("five-point fit residual too large")
    return m


def unit_latent(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    v = np.asarray(value, dtype=np.float32)
    if not np.isfinite(v).all():
        raise ValueError("nonfinite identity latent")
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("zero or invalid identity latent norm")
    return v / norm


class TrackGeometry:
    """Bounded run-local EMA, with explicit frame keys and isolated track state.

    Late/replayed frames never update newer state. Gaps, cuts and large motion
    reset continuity; a lock binds each snapshot atomically to its frame index.
    """
    def __init__(self, capacity=256):
        self._state = OrderedDict()
        self._lock = threading.Lock()
        self.capacity = capacity
        self._matrices = OrderedDict()
        self._resets = OrderedDict()

    def update(self, track_id, frame_index, points, bbox, cut=False, smooth=True):
        p = landmarks5(points, bbox).copy()
        if track_id is None or frame_index is None:
            return p
        with self._lock:
            prev = self._state.get(track_id)
            reset = prev is None or cut
            if prev is not None:
                idx, old, old_box = prev
                if frame_index <= idx:
                    return p
                span = max(float(np.ptp(p, axis=0).max()), 1.)
                motion = float(np.linalg.norm(p.mean(0) - old.mean(0))) / span
                reset = cut or frame_index != idx + 1 or motion >= .5
                if not reset:
                    ratio = float(np.linalg.norm(p[3:].mean(0)-p[:2].mean(0))) / max(float(np.linalg.norm(old[3:].mean(0)-old[:2].mean(0))), 1e-6)
                    if not .5 <= ratio <= 2.:
                        # Do not contaminate subsequent state with an outlier.
                        raise InvalidAlignment("consecutive-frame scale jump")
                    if smooth:
                        p = .65 * p + .35 * old
            if reset:
                self._resets[track_id] = frame_index
                self._resets.move_to_end(track_id)
                while len(self._resets) > self.capacity:
                    self._resets.popitem(last=False)
            self._state[track_id] = (frame_index, p.copy(), np.asarray(bbox).copy() if bbox is not None else None)
            self._state.move_to_end(track_id)
            while len(self._state) > self.capacity:
                self._state.popitem(last=False)
        return p

    def validate_matrix(self, track_id, frame_index, matrix, cut=False, sar=1.):
        m = affine(matrix, sar=sar)
        if track_id is None or frame_index is None:
            return
        scale = float(np.sqrt(max(float(np.linalg.det(m[:,:2])), 1e-12)))
        with self._lock:
            cut = cut or self._resets.get(track_id) == frame_index
            previous = self._matrices.get(track_id)
            if previous is not None:
                idx, old_scale = previous
                if frame_index <= idx:
                    return
                ratio = scale / max(old_scale, 1e-12)
                if frame_index == idx + 1 and not cut and not .5 <= ratio <= 2.:
                    raise InvalidAlignment("consecutive affine scale jump")
            self._matrices[track_id] = (frame_index, scale)
            self._matrices.move_to_end(track_id)
            while len(self._matrices) > self.capacity:
                self._matrices.popitem(last=False)
