"""One validator for every Face the analyser hands to the pipeline.

InsightFace's ``Face.__getattr__`` returns ``None`` for a missing key instead of raising, so
``getattr(face, 'det_score', 1.0)`` quietly yields ``None`` and the first ``None >= 0.65`` is a
TypeError - which is what a8df052 fixed at three call sites one at a time. A non-finite value
is worse than a missing one: ``float('nan') < floor`` is False, so a NaN ``det_score`` passes a
confidence gate it should fail, and a NaN embedding becomes a stray one-frame track that is
never given a source.

Applied once, at the end of ``face_util._enrich_detected_faces`` (the exit shared by
``get_all_faces``, the rescue chain and ``get_all_faces_hires``, which is also what the
region-of-interest detector calls), so everything downstream reads a face whose fields are
either usable or absent:

    det_score                         None / NaN / inf / non-numeric  ->  0.0
    bbox, kps                         wrong shape, non-finite, empty box  ->  the face is DROPPED
                                      (nothing can align, track or paste a face with no box)
    landmark_2d_106, landmark_3d_68,
    pose, embedding, normed_embedding present but unusable  ->  set to None (what "absent"
                                      already means everywhere)

A face with no embedding at all (recognition not loaded) is left alone: that is a mode, not
damage. Every action is counted, and ``report_once`` prints the totals the first time any
action happened, so a detector producing garbage is never silent.
"""
from __future__ import annotations

import math
import threading
from collections import Counter
from typing import Any, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from roop.degrade import swallowed as _swallowed

_SHAPES = {
    'bbox': (4,),
    'kps': (5, 2),
    'landmark_2d_106': (106, 2),
    'landmark_3d_68': (68, 3),
    'pose': (3,),
}
_EMBEDDINGS = ('embedding', 'normed_embedding')
_MIN_EMBEDDING_DIM = 64
_EMBEDDING_NORM_FLOOR = 1e-6

COUNTS: Counter = Counter()
_LOCK = threading.Lock()
_REPORTED = False


def _get(face: Any, name: str) -> Any:
    if isinstance(face, Mapping):
        return face.get(name)
    return getattr(face, name, None)


def _is_stored(face: Any, name: str) -> bool:
    """`name` is data on the face, not a property computed from other data."""
    if isinstance(face, Mapping):
        return name in face
    return name in getattr(face, '__dict__', {})


def _set(face: Any, name: str, value: Any) -> None:
    if isinstance(face, dict):
        face[name] = value
    else:
        setattr(face, name, value)


def _usable(value: Any, shape: Optional[Tuple[int, ...]]) -> bool:
    """A finite numeric array of exactly `shape` (a trailing dimension may be -1 = any)."""
    if value is None:
        return False
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if arr.dtype == object or (shape is not None and arr.shape != shape):
        return False
    return bool(arr.size) and bool(np.isfinite(arr).all())


def _usable_embedding(value: Any) -> bool:
    if not _usable(value, None):
        return False
    arr = np.asarray(value, dtype=np.float64)
    return (arr.ndim == 1 and arr.shape[0] >= _MIN_EMBEDDING_DIM
            and float(np.linalg.norm(arr)) > _EMBEDDING_NORM_FLOOR)


def _usable_box(value: Any) -> bool:
    if not _usable(value, _SHAPES['bbox']):
        return False
    x1, y1, x2, y2 = (float(v) for v in np.asarray(value, dtype=np.float64))
    return x2 > x1 and y2 > y1


def _score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def sanitize_face_attributes(face: Any) -> bool:
    """Repair `face` in place. Returns False when the face must be dropped (no usable
    box or keypoints); True otherwise. Never raises."""
    try:
        raw = _get(face, 'det_score')
        fixed = _score(raw)
        if raw is None:
            COUNTS['det_score_missing'] += 1
            _set(face, 'det_score', fixed)
        elif not isinstance(raw, (int, float, np.floating, np.integer)) or raw != fixed:
            COUNTS['det_score_invalid'] += 1
            _set(face, 'det_score', fixed)

        if not _usable_box(_get(face, 'bbox')):
            COUNTS['dropped_bbox'] += 1
            return False
        if not _usable(_get(face, 'kps'), _SHAPES['kps']):
            COUNTS['dropped_kps'] += 1
            return False

        for name in ('landmark_2d_106', 'landmark_3d_68', 'pose'):
            value = _get(face, name)
            if value is not None and not _usable(value, _SHAPES[name]):
                COUNTS['nulled_' + name] += 1
                _set(face, name, None)
        for name in _EMBEDDINGS:
            if not _is_stored(face, name):
                continue            # InsightFace's normed_embedding is a property of `embedding`
            value = _get(face, name)
            if value is not None and not _usable_embedding(value):
                COUNTS['nulled_' + name] += 1
                _set(face, name, None)
        return True
    except Exception as exc:                           # a validator must not be the new crash
        COUNTS['validator_error'] += 1
        _swallowed('roop/face_sanity.py:sanitize_face_attributes', exc, 'face kept as it was')
        return True


def sanitize_faces(faces: Iterable[Any]) -> List[Any]:
    """`faces` without the ones that cannot be used, the rest repaired in place."""
    out = [f for f in faces if sanitize_face_attributes(f)]
    if COUNTS:
        report_once()
    return out


def snapshot() -> dict:
    with _LOCK:
        return dict(COUNTS)


def reset() -> None:
    """Clear the counters and the once-per-process report (tests, new job)."""
    global _REPORTED
    with _LOCK:
        COUNTS.clear()
        _REPORTED = False


def report_once() -> None:
    global _REPORTED
    with _LOCK:
        if _REPORTED or not COUNTS:
            return
        _REPORTED = True
        detail = ', '.join(f'{k}={v}' for k, v in sorted(COUNTS.items()))
    print(f'[FaceSanity] the analyser returned damaged face data ({detail}); fields were repaired '
          f'or the face dropped. Counts keep accumulating - see face_sanity.snapshot().',
          flush=True)
