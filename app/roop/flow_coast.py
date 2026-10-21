"""Optical-flow landmark propagation through a detector dropout.

THE GAP THIS FILLS
------------------
When a hand, a shoulder or another head covers most of a face, the detector
returns nothing for a few frames. The pre-pass has two ways to fill that hole
(`procmgr_tracking._build_temporal_faces`): a Hermite blend between the real
observations on either side, and -- for holes it cannot bridge -- a
constant-velocity Kalman prediction (`roop/tracker.py`). Both are GEOMETRY
ONLY: neither looks at a pixel while the face is hidden. A head that turns or
ducks behind the occluder is placed by extrapolation.

This module looks at the pixels that are still visible. Pyramidal Lucas-Kanade
tracks a cloud of points anchored on the face at its last real detection; the
points the occluder has not swallowed move rigidly with the head, so a robust
similarity fit over them moves EVERY landmark -- including the ones under the
occluder -- by the head's actual motion.

WHY A SIMILARITY FIT AND NOT PER-LANDMARK FLOW
----------------------------------------------
A point under the occluder tracks the occluder, not the face. Tracking the 106
landmarks individually would drag exactly the hidden ones off the head. The fit
asks the visible majority how the head moved and applies that single motion to
all of them, which is what `tracker._coasted_face` already does with the box --
only with measured motion instead of predicted.

HOW IT KNOWS IT IS WRONG
------------------------
Every step reports `consistent`. A step is consistent only if enough points
survive a forward-backward check, a RANSAC similarity explains most of them
within a pixel or two, and the recovered scale and rotation are physically
plausible for one frame. An inconsistent step is not used: the caller falls
back to the geometry-only path, so flow can only ever improve a hole, not
poison one. When a caller can supply the occluder mask for the frame, points
inside it are dropped before the fit.

This module owns no model session and no frames: callers hand it grayscale
arrays. It is bounded in memory (one ROI-sized gray patch per coaster).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:                                 # pragma: no cover - cv2 is a hard dep
    cv2 = None

# One step of LK is a statement about ~1 frame of motion. These bounds reject a
# fit that "explains" the points by a motion no head makes between two frames.
MAX_SCALE_STEP = 0.18           # |s - 1| per step
MAX_ROT_STEP_DEG = 25.0
FB_TOLERANCE_PX = 1.5           # forward-backward round trip, in ROI pixels
RANSAC_PX = 2.0
MIN_INLIERS = 6
MIN_INLIER_RATIO = 0.6
MAX_MEDIAN_RESIDUAL_PX = 1.5
# LK's per-point error is the mean absolute grey difference between the patch
# around the old point and the patch around the new one. A point that has gone
# under an occluder lands on a DIFFERENT texture, so its patch error is large
# even when the forward-backward round trip happens to close. Keep a point only
# if its error is under both an absolute ceiling and a multiple of the cloud's
# own median (a blurry, low-contrast face has a high floor; the median adapts).
MAX_PATCH_ERROR = 30.0
PATCH_ERROR_MEDIAN_MULT = 3.0
# A flow path is believed only if it arrives where the closing detection is,
# to within this fraction of the inter-ocular distance (mean over landmarks).
BRIDGE_TOLERANCE_IOD = 0.30
# After the dominant motion is fitted, the points it rejected are fitted once
# more. If THEY form a rigid group nearly as large, the scene holds two
# competing motions (the head and the thing in front of it) and the evidence
# does not say which is the head -- the step is refused rather than guessed.
AMBIGUITY_RATIO = 0.7


@dataclass
class FlowResult:
    consistent: bool
    landmarks: Optional[np.ndarray]        # (N, 2) in full-frame pixels
    kps: Optional[np.ndarray]              # (5, 2)
    bbox: Optional[np.ndarray]             # (4,)
    transform: Optional[np.ndarray]        # 2x3, previous step -> this step
    n_points: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    median_residual: float = float('inf')
    reason: str = ''
    # Fraction of the ANCHOR cloud still proven rigid. An occluder covering part
    # of the face takes the points under it out of the chain (they fail the
    # forward-backward / patch checks), so this falls with the covered fraction:
    # an occlusion estimate that needs no mask. 1.0 for a failed step (unknown).
    survival: float = 1.0


def _as_points(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        a = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.shape[1] < 2 or a.shape[0] == 0 or not np.isfinite(a).all():
        return None
    return a[:, :2].copy()


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (pts @ m[:, :2].T + m[:, 2]).astype(np.float32)


def _compose(m2: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """The transform that applies `m1` and then `m2` (both 2x3)."""
    a = np.vstack([m1, [0, 0, 1]])
    b = np.vstack([m2, [0, 0, 1]])
    return (b @ a)[:2].astype(np.float32)


def _decompose(m: np.ndarray) -> Tuple[float, float]:
    """(scale, rotation in degrees) of a similarity 2x3."""
    s = float(np.hypot(m[0, 0], m[1, 0]))
    r = float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
    return s, r


class FlowCoaster:
    """Lucas-Kanade propagation for ONE face.

    ``anchor`` is called with each REAL detection (cheap: one ROI and a corner
    search); ``step`` is called for each frame the detector missed and returns
    where the face has moved to since the previous call. The coaster keeps the
    cumulative geometry itself, so consecutive ``step`` calls chain.
    """

    def __init__(self, max_side: int = 192, win: int = 15, levels: int = 3,
                 max_features: int = 60, roi_scale: float = 1.7, tracker: Any = None):
        self.max_side = int(max_side)
        self.win = int(win)
        self.levels = int(levels)
        self.max_features = int(max_features)
        self.roi_scale = float(roi_scale)
        # Where the points go between two crops is the only pixel work a step does; a back end supplies it
        # (roop/flow_backend.py). The default is the CPU call this class always made, so a coaster built
        # without one behaves exactly as before.
        self._tracker = tracker
        # Only a CROP of the last frame is kept (never the frame): a coaster per
        # track on 4K footage would otherwise pin tens of MB each.
        self._crop: Optional[np.ndarray] = None
        self._origin: Tuple[int, int] = (0, 0)
        self._crop_hw: Tuple[int, int] = (0, 0)           # full-resolution crop size
        # LK is tuned for a face about `max_side` px across. Footage is not: a
        # 4K face is ~1000 px, where one frame of head motion is several times
        # the LK window and the chain fails. The crop is tracked at a fixed
        # scale chosen when the chain is anchored; every pixel threshold below is
        # in that flow space and is divided by the scale when it is applied to
        # full-resolution coordinates.
        self._scale: float = 1.0
        self._n0: int = 0
        self._shape: Optional[Tuple[int, int]] = None
        self._pts: Optional[np.ndarray] = None          # tracked cloud, full-frame px
        self._landmarks: Optional[np.ndarray] = None
        self._kps: Optional[np.ndarray] = None
        self._bbox: Optional[np.ndarray] = None
        self.steps = 0

    @property
    def anchored(self) -> bool:
        return self._crop is not None and self._pts is not None

    # -- geometry helpers -----------------------------------------------------

    def _crop_around(self, gray: np.ndarray, pts: np.ndarray):
        """A copy of `gray` around the cloud (room for one frame of motion),
        resized into flow space. Returns ``(crop, (x0, y0), (full_h, full_w))``."""
        h, w = gray.shape
        size = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])))
        margin = int(24.0 / self._scale + 0.35 * size)
        x0 = int(max(0, np.floor(pts[:, 0].min() - margin)))
        y0 = int(max(0, np.floor(pts[:, 1].min() - margin)))
        x1 = int(min(w, np.ceil(pts[:, 0].max() + margin)))
        y1 = int(min(h, np.ceil(pts[:, 1].max() + margin)))
        crop = gray[y0:y1, x0:x1]
        if self._scale < 1.0:
            crop = cv2.resize(crop, None, fx=self._scale, fy=self._scale,
                              interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(crop), (x0, y0), (y1 - y0, x1 - x0)

    def _roi(self, bbox: np.ndarray, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        h, w = shape
        cx, cy = (bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5
        half = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.5 * self.roi_scale
        x0 = int(max(0, np.floor(cx - half)))
        y0 = int(max(0, np.floor(cy - half)))
        x1 = int(min(w, np.ceil(cx + half)))
        y1 = int(min(h, np.ceil(cy + half)))
        return x0, y0, x1, y1

    # -- anchor ---------------------------------------------------------------

    def anchor(self, gray: np.ndarray, bbox: Any, landmarks: Any = None,
               kps: Any = None) -> bool:
        """Re-seed from a real detection. Returns False when it cannot (then
        ``step`` returns None until the next successful anchor)."""
        self._crop = None
        self._pts = None
        if cv2 is None or gray is None or gray.ndim != 2:
            return False
        box = _as_points(np.asarray(bbox, np.float32).reshape(2, 2)) if bbox is not None else None
        if box is None:
            return False
        bb = np.array([box[0, 0], box[0, 1], box[1, 0], box[1, 1]], np.float32)
        lm = _as_points(landmarks)
        k5 = _as_points(kps)
        if lm is None and k5 is None:
            return False
        x0, y0, x1, y1 = self._roi(bb, gray.shape)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return False
        base = lm if lm is not None else k5
        # Interior mask, built on the ROI only (a full-frame mask per face per
        # frame would cost more on 4K than the repair is worth): the landmark
        # hull, eroded. Features on the hull edge sit on the background and
        # slide with it; the interior is the head.
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        hull = cv2.convexHull((base - np.array([x0, y0], np.float32)).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        k = max(3, int(0.06 * max(bb[2] - bb[0], bb[3] - bb[1])) | 1)
        mask = cv2.erode(mask, np.ones((k, k), np.uint8))
        feats = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1], maxCorners=self.max_features,
                                        qualityLevel=0.01, minDistance=3,
                                        mask=mask, blockSize=5)
        cloud: List[np.ndarray] = [base]
        if feats is not None and len(feats):
            cloud.append(feats.reshape(-1, 2) + np.array([x0, y0], np.float32))
        pts = np.concatenate(cloud, axis=0).astype(np.float32)
        if len(pts) < MIN_INLIERS:
            return False
        self._scale = float(min(1.0, self.max_side / max(bb[2] - bb[0], bb[3] - bb[1], 1.0)))
        self._crop, self._origin, self._crop_hw = self._crop_around(gray, pts)
        self._shape = gray.shape
        self._pts = pts
        self._n0 = len(pts)
        self._landmarks = lm
        self._kps = k5
        self._bbox = bb
        self.steps = 0
        return True

    # -- step -----------------------------------------------------------------

    def step(self, gray: np.ndarray,
             occluder: Optional[np.ndarray] = None) -> Optional[FlowResult]:
        """Track the cloud into ``gray``.

        ``occluder`` (optional, same HxW as ``gray``): non-zero where something
        in front of the face is known to be. Points landing there are dropped
        before the fit.
        """
        if cv2 is None or not self.anchored or gray is None or gray.shape != self._shape:
            return None
        prev_c, pts = self._crop, self._pts
        ys, xs = pts[:, 1], pts[:, 0]
        h, w = gray.shape
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if int(inside.sum()) < MIN_INLIERS:
            return self._fail('cloud left the frame')
        if self._tracker is None:
            from roop.flow_backend import CvLK
            self._tracker = CvLK(win=self.win, levels=self.levels)
        # Track inside the stored crop, not the whole frame: LK builds an image
        # pyramid per call, and on 4K that would cost more than the dropout it
        # repairs. The current frame is cut at the SAME bounds.
        cx0, cy0 = self._origin
        fh, fw = self._crop_hw
        sc = self._scale
        off = np.array([cx0, cy0], np.float32)
        gray_c = np.ascontiguousarray(gray[cy0:cy0 + fh, cx0:cx0 + fw])
        if sc < 1.0:
            gray_c = cv2.resize(gray_c, (prev_c.shape[1], prev_c.shape[0]),
                                interpolation=cv2.INTER_AREA)
        if gray_c.shape != prev_c.shape:
            return self._fail('crop shape changed')
        p0 = ((pts - off) * sc).reshape(-1, 1, 2)
        p1, st, err = self._tracker.track(prev_c, gray_c, p0)
        if p1 is None:
            return self._fail('lk failed')
        back, stb, _ = self._tracker.track(gray_c, prev_c, p1)
        if back is None:
            return self._fail('lk failed')
        # back to full-resolution coordinates
        p1 = p1 / sc + off.reshape(1, 1, 2)
        p0 = p0 / sc + off.reshape(1, 1, 2)
        back = back / sc + off.reshape(1, 1, 2)
        px = 1.0 / sc                              # one flow-space pixel, in full-res pixels
        ok = (st.reshape(-1) == 1) & (stb.reshape(-1) == 1) & inside
        fb = np.linalg.norm((back - p0).reshape(-1, 2), axis=1)
        ok &= fb < FB_TOLERANCE_PX * px
        if err is not None:
            e = err.reshape(-1)
            if ok.any():
                ceiling = min(MAX_PATCH_ERROR,
                              max(4.0, PATCH_ERROR_MEDIAN_MULT * float(np.median(e[ok]))))
                ok &= e <= ceiling
        new = p1.reshape(-1, 2)
        if occluder is not None and occluder.shape == gray.shape:
            nx = np.clip(new[:, 0].round().astype(int), 0, w - 1)
            ny = np.clip(new[:, 1].round().astype(int), 0, h - 1)
            ok &= occluder[ny, nx] == 0
        n_ok = int(ok.sum())
        if n_ok < MIN_INLIERS:
            return self._fail('too few points survive the forward-backward check',
                              n_points=len(pts), n_inliers=n_ok)
        m, inl = cv2.estimateAffinePartial2D(
            pts[ok], new[ok], method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX * px,
            maxIters=200, confidence=0.99)
        if m is None or inl is None or not np.isfinite(m).all():
            return self._fail('no similarity fit', n_points=len(pts), n_inliers=n_ok)
        inl = inl.reshape(-1).astype(bool)
        n_in = int(inl.sum())
        residual = np.linalg.norm(_apply(m, pts[ok][inl]) - new[ok][inl], axis=1)
        med = float(np.median(residual)) if n_in else float('inf')
        scale, rot = _decompose(m)
        ratio = n_in / float(n_ok)
        rival = 0
        leftover = np.where(ok)[0][~inl]
        if len(leftover) >= MIN_INLIERS:
            m2, inl2 = cv2.estimateAffinePartial2D(
                pts[leftover], new[leftover], method=cv2.RANSAC,
                ransacReprojThreshold=RANSAC_PX * px, maxIters=100, confidence=0.9)
            if m2 is not None and inl2 is not None:
                rival = int(inl2.sum())
        reason = ''
        if n_in < MIN_INLIERS:
            reason = 'too few inliers'
        elif rival >= max(MIN_INLIERS, AMBIGUITY_RATIO * n_in):
            reason = 'two competing motions'
        elif ratio < MIN_INLIER_RATIO:
            reason = 'inlier ratio too low (motion is not one rigid head)'
        elif med > MAX_MEDIAN_RESIDUAL_PX * px:
            reason = 'residual too large'
        elif abs(scale - 1.0) > MAX_SCALE_STEP:
            reason = 'implausible scale step'
        elif abs(rot) > MAX_ROT_STEP_DEG:
            reason = 'implausible rotation step'
        if reason:
            return self._fail(reason, n_points=len(pts), n_inliers=n_in,
                              ratio=ratio, median=med)

        self._pts = _apply(m, pts[ok][inl])          # carry only what proved rigid
        self._crop, self._origin, self._crop_hw = self._crop_around(gray, self._pts)
        if self._landmarks is not None:
            self._landmarks = _apply(m, self._landmarks)
        if self._kps is not None:
            self._kps = _apply(m, self._kps)
        corners = np.array([[self._bbox[0], self._bbox[1]], [self._bbox[2], self._bbox[1]],
                            [self._bbox[2], self._bbox[3]], [self._bbox[0], self._bbox[3]]],
                           np.float32)
        moved = _apply(m, corners)
        self._bbox = np.array([moved[:, 0].min(), moved[:, 1].min(),
                               moved[:, 0].max(), moved[:, 1].max()], np.float32)
        self.steps += 1
        return FlowResult(True, None if self._landmarks is None else self._landmarks.copy(),
                          None if self._kps is None else self._kps.copy(),
                          self._bbox.copy(), m.astype(np.float32), len(pts), n_in,
                          ratio, med, 'ok',
                          survival=float(len(self._pts)) / float(max(1, self._n0)))

    def _fail(self, reason: str, n_points: int = 0, n_inliers: int = 0,
              ratio: float = 0.0, median: float = float('inf')) -> FlowResult:
        # A failed step ends the chain: a cloud we no longer believe must not be
        # tracked onward as if we did.
        self._crop = None
        self._pts = None
        return FlowResult(False, None, None, None, None, n_points, n_inliers,
                          ratio, median, reason)


def face_from_flow(template: Any, result: FlowResult, *, coast_age: int = 1) -> Optional[Any]:
    """A Face-like copy of `template` carrying the flow-propagated geometry.

    Stamped like `tracker.FaceTracker._coasted_face` (so the swap audit counts it
    as gap-filled rather than detected) plus ``_flow_coasted``.
    """
    if result is None or not result.consistent or result.bbox is None:
        return None
    try:
        face = type(template)(template)     # insightface Face is a dict subclass
    except (TypeError, ValueError):
        try:
            face = dict(template)
        except (TypeError, ValueError):
            return None
    face['bbox'] = result.bbox.astype(np.float32)
    if result.kps is not None:
        face['kps'] = result.kps.astype(np.float32)
    old_lm = template.get('landmark_2d_106') if hasattr(template, 'get') else None
    if result.landmarks is not None and old_lm is not None \
            and np.asarray(old_lm).shape[0] == result.landmarks.shape[0]:
        face['landmark_2d_106'] = result.landmarks.astype(np.float32)
    face['_coasted'] = True
    face['_interpolated'] = True
    face['_flow_coasted'] = True
    face['_coast_age'] = int(coast_age)
    face['occlusion_state'] = 'coasted'
    return face


def spread_residual(path: Sequence[np.ndarray], end: np.ndarray,
                    closing: np.ndarray) -> List[np.ndarray]:
    """Spread ``closing - end`` linearly over ``path`` (frame j gets (j+1)/(L+1))."""
    residual = np.asarray(closing, np.float32)[:, :2] - np.asarray(end, np.float32)[:, :2]
    n = len(path)
    return [np.asarray(p, np.float32)[:, :2] + ((j + 1) / float(n + 1)) * residual
            for j, p in enumerate(path)]


def bridge_gap(flow_path: Sequence[np.ndarray], flow_end: Optional[np.ndarray],
               closing: np.ndarray, iod: float,
               tolerance: float = BRIDGE_TOLERANCE_IOD) -> Optional[List[np.ndarray]]:
    """Pin a flow path to the detection that closes the gap.

    ``flow_path[j]`` is the flow-propagated landmark set for the j-th dropped
    frame (j = 0 .. L-1); ``flow_end`` is the same cloud flowed onto the closing
    frame, where ``closing`` is the real detection. The shortfall
    ``closing - flow_end`` is spread linearly over the gap, so the path keeps the
    SHAPE flow measured and ends exactly on the anchor.

    Returns None -- and the caller keeps the geometry-only fill -- when flow did
    not arrive within ``tolerance`` x ``iod`` of the anchor. That is the guard
    against a path that followed an occluder: a real detection on the far side
    of the gap is the one piece of evidence flow cannot fake.
    """
    if flow_end is None or closing is None or not len(flow_path):
        return None
    residual = np.asarray(closing, np.float32)[:, :2] - np.asarray(flow_end, np.float32)[:, :2]
    if float(np.mean(np.linalg.norm(residual, axis=1))) > tolerance * max(iod, 1e-6):
        return None
    return spread_residual(flow_path, flow_end, closing)


# ── Alignment through an occluder: the per-frame policy for a persistent chain ─
#
# Stage 17. A chain anchored on the last TRUSTED detection is stepped on every
# frame, detected or not. `FlowResult.survival` (the fraction of the anchor cloud
# still proven rigid) is a mask-free occlusion estimate: measured on real footage
# with a pasted occluder (`tests/probe_alignment_chain_stage17.py`) it stays high on
# clean frames (median 0.84-1.00; under 0.7 in 0-4.3% of chains) and falls under a
# covering occluder (median 0.23-0.71). The keypoints the chain carries then beat
# the detector's, which an occluder drags by several pixels.
#
# A detector-vs-flow DISAGREEMENT gate was measured and rejected: on clean frames
# flow disagrees with a good detection by > 0.05 x IOD in 23-81% of s5/s7 chains,
# so it cannot tell "detector hijacked" from "flow wrong".
OCCLUDED_BELOW = 0.70       # survival under this: the face is partly covered
REFRESH_ABOVE = 0.90        # survival over this (and old enough): safe to re-anchor
REFRESH_AGE = 4             # frames before a clean chain is re-anchored
# The probe validated chains of AT MOST 8 frames from a clean anchor (clean
# survival < 0.7 in 0-4.3%). Longer chains lose anchor points to blur and
# rotation until clean frames read as occluded, and points lost under an
# occluder never come back after it leaves: the first end-to-end version let
# chains run to 24 frames and replaced 99 of 156 CLEAN frames on s6.
MAX_AGE = 8                 # a chain this old is re-anchored whatever it says
MAX_SUBSTITUTED = 8         # consecutive occluded frames carried before the detector is trusted again

SUBSTITUTE = 'substitute'   # use the chain's geometry for this detected face
REFRESH = 'refresh'         # re-anchor on this detection
CONTINUE = 'continue'       # keep the chain, keep the detector's geometry
REANCHOR = 'reanchor'       # chain unusable: start over from this detection


def chain_policy(consistent: bool, survival: float, age: int, substituted: int,
                 occluded_below: float = OCCLUDED_BELOW, refresh_above: float = REFRESH_ABOVE,
                 refresh_age: int = REFRESH_AGE, max_age: int = MAX_AGE,
                 max_substituted: int = MAX_SUBSTITUTED) -> str:
    """What to do with a DETECTED face whose chain has just been stepped onto it.

    ``consistent``: the LK step passed every guard. ``survival``: fraction of the
    anchor cloud still rigid. ``age``: steps since the anchor. ``substituted``:
    consecutive frames already carried on the chain's geometry.
    """
    if not consistent:
        return REANCHOR
    if survival < occluded_below and substituted < max_substituted:
        return SUBSTITUTE
    if age >= max_age:
        return REANCHOR
    if survival >= refresh_above and age >= refresh_age:
        return REFRESH
    return CONTINUE
