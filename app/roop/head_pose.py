"""One head-pose type, and the one place a pose is converted into it.

WHY THIS EXISTS
---------------
Four pose representations are live in this tree, and two of them disagree about
which element is which:

    face_util.solve_pose_5pt / solve_pose_jaw_5pt   (yaw, pitch, roll)   degrees
    insightface ``face.pose``                       (pitch, yaw, roll)   degrees
    pose_source_selector.PoseEstimate / as_dict()   object / dict with .yaw .pitch .roll
    FaceSet.face_poses                              (yaw, pitch) -- but V1 sets hold the
        EPnP fit's angles (frontal ~ +-170, see below) and V2 sets hold solve_pose_5pt's

The permutation used to be written inline at each site (``faceset_v2._pose``,
``pose_source_selector.estimate_target_pose``), and the places that did NOT
convert were wrong silently: ``procmgr_tracking`` read ``getattr(dict, 'yaw', 0.0)``
off the temporal-identity pose dict, which is always 0.0.

CONVENTION (pinned, not chosen here)
------------------------------------
Angles are in ``solve_pose_5pt``'s convention, which `angle_portfolio` and the
Stage 2 tests pin: **+yaw = nose toward the viewer's RIGHT, +pitch = the face
tilted DOWN, +roll = clockwise in the image**. A dead-frontal head is (0, 0, 0).
Changing a sign here would flip every gate that keys on it, so this module
adapts INTO that convention and never out of it. ``pitch_up`` is provided for
callers that think in the physical direction.

THE ORDER OF A BARE SEQUENCE IS NOT GUESSABLE
---------------------------------------------
``(30.0, 5.0, 0.0)`` is a valid pose in either order. This module therefore
never infers the order from magnitudes: a bare tuple / ndarray / list MUST be
declared with ``order``. Dicts, ``HeadPose`` and objects with ``yaw``/``pitch``
attributes name their fields and need no declaration.

    CANONICAL_ORDER   "ypr"   solve_pose_5pt, HeadPose
    INSIGHTFACE_ORDER "pyr"   ``face.pose`` from the buffalo_l 3D-68 model

EPnP (face_3d_recon.decompose_yaw_pitch) angles are NOT in this convention --
a frontal head reads yaw ~ +-170 and pitch ~ -50 -- and cannot be adapted by a
permutation. Do not pass them in; take the pose from the 5 keypoints with
``head_pose_from_kps``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np

from roop.degrade import swallowed as _swallowed

CANONICAL_ORDER = "ypr"
INSIGHTFACE_ORDER = "pyr"

_FIELDS = ("yaw", "pitch", "roll")
_INITIALS = ("y", "p", "r")


@dataclass
class HeadPose:
    """Yaw / pitch / roll in degrees (solve_pose_5pt convention) with provenance.

    Iterates and indexes as ``(yaw, pitch, roll)`` so it drops in wherever the
    solver's tuple was consumed; string keys work too for dict-style callers.
    """

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    confidence: float = 1.0
    landmarks_5pt: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __iter__(self) -> Iterator[float]:
        yield self.yaw
        yield self.pitch
        yield self.roll

    def __len__(self) -> int:
        return 3

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in _FIELDS or key == "confidence":
                return getattr(self, key)
            raise KeyError(key)
        return (self.yaw, self.pitch, self.roll)[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError, TypeError):
            return default

    @property
    def pitch_up(self) -> float:
        """Pitch as the physical direction: positive = looking UP."""
        return -self.pitch

    def as_tuple(self):
        return (self.yaw, self.pitch, self.roll)

    def as_dict(self) -> dict:
        return {"yaw": self.yaw, "pitch": self.pitch, "roll": self.roll,
                "confidence": self.confidence}


def _finite(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _from_fields(yaw, pitch, roll, confidence, landmarks) -> Optional[HeadPose]:
    yaw, pitch = _finite(yaw), _finite(pitch)
    if yaw is None or pitch is None:
        return None
    roll = _finite(roll)
    conf = _finite(confidence)
    return HeadPose(yaw, pitch, 0.0 if roll is None else roll,
                    1.0 if conf is None else conf, landmarks)


def normalize_pose_tuple(raw: Any, order: str = CANONICAL_ORDER,
                         confidence: float = 1.0,
                         landmarks_5pt: Optional[np.ndarray] = None) -> Optional[HeadPose]:
    """Adapt any pose representation in this tree to a ``HeadPose``, or None.

    ``raw`` may be: ``HeadPose`` (returned as is); a dict or an object exposing
    ``yaw`` / ``pitch`` (+ optional ``roll`` / ``confidence``) -- including
    ``PoseEstimate`` and the temporal-identity pose dict; or a bare sequence /
    ndarray of 2 or 3 numbers, whose meaning is given by ``order`` (a
    permutation of "ypr"; for a 2-sequence the first two letters apply and roll
    is 0). Anything unusable -- None, NaN, too short, an unknown order -- is
    None, never an exception and never a guess.
    """
    if raw is None:
        return None
    if isinstance(raw, HeadPose):
        return raw
    if isinstance(raw, dict) or hasattr(raw, "yaw"):
        get = raw.get if isinstance(raw, dict) else (lambda k, d=None: getattr(raw, k, d))
        if get("yaw") is None and get("pitch") is None:
            return None
        return _from_fields(get("yaw"), get("pitch"), get("roll", 0.0),
                            get("confidence", confidence), landmarks_5pt)
    if hasattr(raw, "as_dict") and not isinstance(raw, (list, tuple, np.ndarray)):
        try:
            return normalize_pose_tuple(raw.as_dict(), order, confidence, landmarks_5pt)
        except Exception as _degrade_error:
            _swallowed("roop/head_pose.py:as_dict", _degrade_error, "pose treated as unusable")
            return None
    if sorted(str(order)) != sorted(_INITIALS):
        return None
    try:
        flat = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if flat.size < 2:
        return None
    # A 4-tuple (solve_pose_jaw_5pt appends the jaw fraction) keeps its first 3.
    n = 3 if flat.size >= 3 else 2
    named = {order[i]: flat[i] for i in range(n)}
    return _from_fields(named.get("y"), named.get("p"), named.get("r", 0.0),
                        confidence, landmarks_5pt)


def head_pose_from_kps(kps, jaw_aware: bool = True) -> Optional[HeadPose]:
    """HeadPose from the 5 ArcFace keypoints, or None.

    ``jaw_aware`` first tries ``solve_pose_jaw_5pt`` (the head's own angles,
    mouth opening factored out) and falls back to ``solve_pose_5pt`` -- the same
    order ``pose_source_selector`` uses for a TARGET face, so a bank face and a
    target are compared like for like.
    """
    if kps is None:
        return None
    from roop.face_util import solve_pose_5pt, solve_pose_jaw_5pt
    value = solve_pose_jaw_5pt(kps) if jaw_aware else None
    if value is None:
        value = solve_pose_5pt(kps)
    if value is None:
        return None
    try:
        pts = np.asarray(kps, dtype=np.float32).reshape(5, 2)
    except (TypeError, ValueError):
        pts = None
    return normalize_pose_tuple(tuple(value)[:3], CANONICAL_ORDER, 1.0, pts)
