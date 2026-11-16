"""Frame-indexed face geometry used by the trajectory HUD and render workers.

WHAT IT HOLDS. One row per face per frame, sorted by (frame, track), in ONE structured array so a single
`np.load(..., mmap_mode='r')` maps the whole thing and a lookup by frame is a binary search. `TrajectoryPlan` rows
also carry forward/backward-smoothed landmarks, pose-warped canonical templates, silhouette anchor weights, and the
exact source PTS index. Render workers read those rows by `frame_idx`; they do not run filters or route track IDs.

    frame   int32   ABSOLUTE decoder frame (frame_start + the scan's frame index), so a resumed run that
                    starts later and a fresh run that starts earlier index the same rows
    track   int32   the pre-pass track id (`face['_track_id']`)
    flags   uint8   1 gap-filled / interpolated, 2 placed by optical flow, 4 Kalman-coasted
    kps     f4 5x2  the SMOOTHED alignment keypoints the swap will use
    bbox    f4 4
    source_kps f4 5x2 the Stage 2 landmarks, used to validate a resumed plan
    pose    f4 3    yaw, pitch, roll in degrees, solved FROM THE PLAN landmarks (pose_router.py's convention)
    M       f8 2x3  legacy/diagnostic frame -> 112x112 similarity; Plan rows use the yaw-warped 112 template
    M_inv   f8 2x3  template -> frame (what paste-back uses)
    template_{112,128,256,512} f4 5x2 yaw-warped destination landmarks for the common swap crop sizes
    silhouette f4 3x2 source near-ear/nose/chin anchors, plus pose-derived destination anchors and weights

The existing 112-pixel matrices remain diagnostic/HUD data and are not substituted for the selected swap crop's
alignment. The render path consumes the plan's keypoints and the matching-size warped template, then solves the
similarity for the current frame and sample-aspect ratio. The head-pose HUD (`GET /api/trajectory/pose`), project
checkpoint, render workers, accurate VFR seek, and `/api/preview/frame` share this frame-indexed record.

THE SAME LEAST SQUARES, NOT THE SAME BITS. The legacy cache matrices are solved in closed form (the 2-D similarity
that minimises squared landmark error is a complex ratio; `estimate_norm`'s SVD solves the same problem).
`tests/test_trajectory_cache.py` pins that base-cache path against `skimage.SimilarityTransform` and `estimate_norm`.
Plan matrices are diagnostic for their warped 112 template; Stage 3 independently fits the selected crop size and
its optional silhouette anchors. Nothing that needs bit-identity with the frontal `estimate_norm` may use Plan M.

THE ACTIVE CACHE. `set_active` / `active` hold the cache of the job that is running (or the last one), cleared
when a job begins, so the HUD never shows the previous clip's heads.
"""
from __future__ import annotations

import hashlib
import base64
import binascii
import json
import math
import os
import tempfile
import threading
import zlib
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from roop.degrade import swallowed as _swallowed

FORMAT_NAME = "roop-trajectory"
FORMAT_VERSION = 2

FLAG_INTERPOLATED = 1
FLAG_FLOW = 2
FLAG_COASTED = 4
FLAG_REUSE_FLOW_WARP = 8

ROW = np.dtype([
    ("frame", "<i4"), ("track", "<i4"), ("flags", "u1"),
    ("kps", "<f4", (5, 2)), ("bbox", "<f4", (4,)), ("pose", "<f4", (3,)),
    ("M", "<f8", (2, 3)), ("M_inv", "<f8", (2, 3)),
    ("source_kps", "<f4", (5, 2)),
    ("template_112", "<f4", (5, 2)), ("template_128", "<f4", (5, 2)),
    ("template_256", "<f4", (5, 2)), ("template_512", "<f4", (5, 2)),
    ("silhouette", "<f4", (3, 2)),
    ("silhouette_112", "<f4", (3, 2)), ("silhouette_128", "<f4", (3, 2)),
    ("silhouette_256", "<f4", (3, 2)), ("silhouette_512", "<f4", (3, 2)),
    ("anchor_weights", "<f4", (3,)),
])

TEMPLATE = "arcface"
TEMPLATE_SIZE = 112
POSE_CONVENTION = "solve_pose_5pt: +yaw nose toward the viewer's right, +pitch tilted down, +roll clockwise (degrees)"


# ── vectorized alignment ─────────────────────────────────────────────────────────────────────────────
def template_points(size: int = TEMPLATE_SIZE, mode: str = TEMPLATE) -> np.ndarray:
    from roop.face_util import swap_template_points
    return np.asarray(swap_template_points(int(size), mode), dtype=np.float64)


def similarity_matrices(kps: np.ndarray, dst: np.ndarray, sar: float = 1.0) -> np.ndarray:
    """(N, 2, 3) float64: the least-squares similarity (scale, rotation, translation; no shear, no reflection)
    taking each row of `kps` (N, 5, 2) onto the template `dst` (5, 2).

    With z = x + iy the optimum of  sum |a z_src + t - z_dst|^2  is  a = sum conj(zs) zd / sum |zs|^2  after
    centring, t = mean(zd) - a mean(zs): the same answer `skimage.SimilarityTransform.estimate` reaches by SVD.
    A clip with non-square pixels (`sar` != 1) is fitted in display space and mapped back to the pixel raster,
    exactly as `face_util.estimate_norm` does. Rows with non-finite or coincident points come back NaN.
    """
    kps = np.asarray(kps, dtype=np.float64).reshape(-1, 5, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(5, 2)
    sar = float(sar) if sar and math.isfinite(float(sar)) else 1.0
    src = kps.copy()
    use_sar = abs(sar - 1.0) > 0.005
    if use_sar:
        src[:, :, 0] *= sar
    zs = src[:, :, 0] + 1j * src[:, :, 1]
    zd = dst[:, 0] + 1j * dst[:, 1]
    zs_c = zs - zs.mean(axis=1, keepdims=True)
    zd_c = zd - zd.mean()
    denom = (np.abs(zs_c) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (np.conj(zs_c) * zd_c[None, :]).sum(axis=1) / denom
        t = zd.mean() - a * zs.mean(axis=1)
    out = np.empty((len(kps), 2, 3), dtype=np.float64)
    out[:, 0, 0], out[:, 0, 1], out[:, 0, 2] = a.real, -a.imag, t.real
    out[:, 1, 0], out[:, 1, 1], out[:, 1, 2] = a.imag, a.real, t.imag
    if use_sar:
        out[:, :, 0] *= sar              # back to the pixel raster, as estimate_norm does
    bad = ~np.isfinite(kps).all(axis=(1, 2)) | ~(denom > 1e-12)
    out[bad] = np.nan
    return out


def invert_matrices(m: np.ndarray) -> np.ndarray:
    """(N, 2, 3) -> (N, 2, 3): the inverse affine of each row (what `cv2.invertAffineTransform` returns)."""
    m = np.asarray(m, dtype=np.float64)
    a = m[:, :, :2]
    with np.errstate(divide="ignore", invalid="ignore"):
        det = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
        inv = np.empty_like(a)
        inv[:, 0, 0], inv[:, 0, 1] = a[:, 1, 1] / det, -a[:, 0, 1] / det
        inv[:, 1, 0], inv[:, 1, 1] = -a[:, 1, 0] / det, a[:, 0, 0] / det
    out = np.empty_like(m)
    out[:, :, :2] = inv
    out[:, :, 2] = -np.einsum("nij,nj->ni", inv, m[:, :, 2])
    return out


def _flags_of(face: Any) -> int:
    def has(name: str) -> bool:
        try:
            return bool(face.get(name))
        except AttributeError:
            return bool(getattr(face, name, False))
    return (FLAG_INTERPOLATED if has("_interpolated") else 0) | (FLAG_FLOW if has("_flow_bridged") else 0) \
        | (FLAG_COASTED if has("_coasted") else 0) \
        | (FLAG_REUSE_FLOW_WARP if (has("_reuse_flow_warp") or has("reuse_flow_warp")) else 0)


def _get(face: Any, name: str):
    try:
        return face.get(name)
    except AttributeError:
        return getattr(face, name, None)


# ── the cache ───────────────────────────────────────────────────────────────────────────────────────
class TrajectoryCache:
    """Rows (see the module docstring) plus the facts needed to read them."""

    def __init__(self, rows: np.ndarray, meta: Optional[Mapping[str, Any]] = None):
        if rows.dtype != ROW:
            raise ValueError("trajectory rows have the wrong dtype")
        self.rows = rows
        self.meta: Dict[str, Any] = dict(meta or {})
        self.meta.setdefault("format", FORMAT_NAME)
        self.meta.setdefault("version", FORMAT_VERSION)
        self.meta.setdefault("template", TEMPLATE)
        self.meta.setdefault("template_size", TEMPLATE_SIZE)
        self.meta.setdefault("pose_convention", POSE_CONVENTION)
        self.meta.setdefault("frame_start", 0)
        self.meta.setdefault("sar", 1.0)

    # -- construction --------------------------------------------------------------------------------
    @staticmethod
    def _collect(faces_by_frame: Mapping[int, List[Any]], frame_start: int):
        """The faces a trajectory can hold, sorted by (frame, track): (frame, track, flags, kps, bbox, faces).
        A face without a track id or 5 finite keypoints is left out (the swap pass cannot align it either).
        `build` and `window_for` both go through here, so what is stored and what is compared cannot differ."""
        keys: List[Tuple[int, int]] = []
        kps_l, bbox_l, flag_l, face_l = [], [], [], []
        for f_idx in sorted(faces_by_frame):
            for face in faces_by_frame[f_idx] or ():
                tid = _get(face, "_track_id")
                kps = _get(face, "kps")
                if tid is None or kps is None:
                    continue
                k = np.asarray(kps, dtype=np.float32)
                if k.shape != (5, 2) or not np.isfinite(k).all():
                    continue
                bb = _get(face, "bbox")
                bb = (np.asarray(bb, dtype=np.float32).reshape(-1)[:4]
                      if bb is not None else np.full(4, np.nan, np.float32))
                if bb.shape != (4,):
                    bb = np.full(4, np.nan, np.float32)
                keys.append((int(f_idx) + int(frame_start), int(tid)))
                kps_l.append(k)
                bbox_l.append(bb)
                flag_l.append(_flags_of(face))
                face_l.append(face)
        if not keys:
            return (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.uint8),
                    np.zeros((0, 5, 2), np.float32), np.zeros((0, 4), np.float32), [])
        order = np.lexsort(([k[1] for k in keys], [k[0] for k in keys]))          # frame, then track
        return (np.asarray([keys[i][0] for i in order], dtype=np.int32),
                np.asarray([keys[i][1] for i in order], dtype=np.int32),
                np.asarray([flag_l[i] for i in order], dtype=np.uint8),
                np.stack([kps_l[i] for i in order]), np.stack([bbox_l[i] for i in order]),
                [face_l[i] for i in order])

    @classmethod
    def build(cls, faces_by_frame: Mapping[int, List[Any]], *, frame_start: int = 0, sar: float = 1.0,
              fps: Optional[float] = None, total_frames: Optional[int] = None) -> "TrajectoryCache":
        """From `ProcessMgr._temporal_faces` ({scan frame: [Face]})."""
        from roop.head_pose import head_pose_from_kps
        frames, tracks, flags, kps_arr, bbox_arr, _faces = cls._collect(faces_by_frame, frame_start)
        n = len(frames)
        rows = np.zeros(n, dtype=ROW)
        if n:
            poses = np.full((n, 3), np.nan, dtype=np.float32)
            for i in range(n):
                try:
                    hp = head_pose_from_kps(kps_arr[i])
                except Exception as exc:                  # a pose is a hint, never a reason to drop a row
                    _swallowed("roop/trajectory_cache.py:pose", exc, "pose left NaN for this face")
                    hp = None
                if hp is not None:
                    poses[i] = (hp.yaw, hp.pitch, hp.roll)
            rows["frame"], rows["track"], rows["flags"] = frames, tracks, flags
            rows["kps"], rows["source_kps"], rows["bbox"], rows["pose"] = kps_arr, kps_arr, bbox_arr, poses
            m = similarity_matrices(kps_arr, template_points(), sar)
            rows["M"] = m
            rows["M_inv"] = invert_matrices(m)
        meta = dict(frame_start=int(frame_start), sar=float(sar), faces=int(n),
                    frames=int(len(np.unique(frames))) if n else 0,
                    tracks=int(len(np.unique(tracks))) if n else 0)
        if fps:
            meta["fps"] = float(fps)
        if total_frames is not None:
            meta["total_frames"] = int(total_frames)
        return cls(rows, meta)

    def window_for(self, faces_by_frame: Mapping[int, List[Any]], *, frame_start: int = 0
                   ) -> Optional["TrajectoryCache"]:
        """This cache restricted to the frames of `faces_by_frame`, IF its rows there are exactly those faces
        (same frames, tracks and keypoints, bit for bit); None otherwise. A resumed run restores Stage 2's
        faces for its own window and the saved file may cover more; a file that disagrees about even one
        keypoint is not the trajectory of these faces and must not drive the HUD."""
        frames, tracks, _flags, kps_arr, _bbox, _faces = self._collect(faces_by_frame, frame_start)
        if not len(frames) or not len(self.rows):
            return None
        lo = int(np.searchsorted(self.rows["frame"], int(frames[0]), side="left"))
        hi = int(np.searchsorted(self.rows["frame"], int(frames[-1]), side="right"))
        sub = self.rows[lo:hi]
        if (len(sub) != len(frames) or not np.array_equal(sub["frame"], frames)
                or not np.array_equal(sub["track"], tracks)
                or not np.array_equal(sub["source_kps"], kps_arr)):
            return None
        meta = dict(self.meta, frame_start=int(frame_start), faces=int(len(sub)),
                    frames=int(len(np.unique(frames))), tracks=int(len(np.unique(tracks))))
        return type(self)(sub, meta)

    # -- persistence ---------------------------------------------------------------------------------
    def save(self, path: str) -> Dict[str, Any]:
        """Write atomically (temp file + replace). Returns the record entry: size, SHA-256 and the meta."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".traj-", suffix=".npy", dir=directory)
        try:
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, np.ascontiguousarray(self.rows), allow_pickle=False)
            digest = hashlib.sha256()
            with open(tmp, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(block)
            size = os.path.getsize(tmp)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return {"sidecar": {"size": int(size), "sha256": digest.hexdigest()}, **self.meta}

    @classmethod
    def open(cls, path: str, meta: Optional[Mapping[str, Any]] = None, mmap: bool = True) -> "TrajectoryCache":
        """Map a saved cache. `allow_pickle=False`: a tampered file cannot execute code."""
        rows = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        if rows.dtype != ROW:
            raise ValueError("%s is not a trajectory cache (dtype %s)" % (path, rows.dtype))
        return cls(rows, meta)

    # -- queries -------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return int(len(self.rows))

    @property
    def nbytes(self) -> int:
        return int(self.rows.nbytes)

    @property
    def frame_range(self) -> Optional[Tuple[int, int]]:
        if not len(self.rows):
            return None
        return int(self.rows["frame"][0]), int(self.rows["frame"][-1])

    def _span(self, frame: int) -> Tuple[int, int]:
        col = self.rows["frame"]
        return int(np.searchsorted(col, frame, side="left")), int(np.searchsorted(col, frame, side="right"))

    def __contains__(self, frame_idx: Any) -> bool:
        """Bounds check: whether frame_idx is within the timeline trim range of this cache.
        Enforces that frame_idx is within valid bounds whether indexed canonically as 0-based
        [0, total_frames) or as absolute frame [frame_start, frame_start + total_frames).
        """
        try:
            f = int(frame_idx)
        except (TypeError, ValueError):
            return False
        total = self.meta.get("total_frames")
        start = int(self.meta.get("frame_start", 0) or 0)
        if total is not None and total > 0:
            if 0 <= f < total:
                return True
            if start <= f < start + total:
                return True
            return False
        rng = self.frame_range
        if rng is not None:
            if rng[0] <= f <= rng[1]:
                return True
            if 0 <= f <= (rng[1] - start):
                return True
        return False

    def get_frame(self, canonical_frame_idx: int) -> np.ndarray:
        """Enforce canonical indexing contract with explicit bounds assertion.
        Whether trim_in is frame 0 or frame 1500, Stage 2 and Stage 3 address the exact
        same canonical frame index with an explicit bounds assertion:
            assert canonical_frame_idx in self
        Returns the structured array view of rows for canonical_frame_idx.
        """
        assert canonical_frame_idx in self, (
            f"Frame index {canonical_frame_idx} out of bounds for trajectory cache "
            f"(frame_start={self.meta.get('frame_start')}, total_frames={self.meta.get('total_frames')})"
        )
        start = int(self.meta.get("frame_start", 0) or 0)
        total = self.meta.get("total_frames")
        f = int(canonical_frame_idx)
        # If given as 0-based relative index within [0, total), map to absolute frame
        if start > 0 and 0 <= f < (total if total else float("inf")):
            abs_frame = f + start
        else:
            abs_frame = f
        lo, hi = self._span(abs_frame)
        return self.rows[lo:hi]

    def at(self, frame: int, absolute: bool = True) -> np.ndarray:
        """The rows of one frame (a view). `absolute=False` takes a frame counted from the job's first frame."""
        f = int(frame) + (0 if absolute else int(self.meta.get("frame_start", 0)))
        lo, hi = self._span(f)
        return self.rows[lo:hi]

    def nearest(self, frame: int, absolute: bool = True, window: int = 12) -> Tuple[int, np.ndarray]:
        """(frame actually used, rows) - the closest frame within `window` that holds a face, so a HUD polled
        at an approximate frame number still reads a head. (frame, empty) when none is near."""
        f = int(frame) + (0 if absolute else int(self.meta.get("frame_start", 0)))
        col = self.rows["frame"]
        if not len(col):
            return f, self.rows[0:0]
        i = int(np.searchsorted(col, f))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(col) and (best is None or abs(int(col[j]) - f) < abs(int(col[best]) - f)):
                best = j
        if best is None or abs(int(col[best]) - f) > window:
            return f, self.rows[0:0]
        used = int(col[best])
        lo, hi = self._span(used)
        return used, self.rows[lo:hi]

    def series(self, track: int, max_points: int = 600) -> Dict[str, List[float]]:
        """One track's yaw/pitch/roll against frame, decimated to at most `max_points` (for a sparkline)."""
        sel = np.nonzero(self.rows["track"] == int(track))[0]
        if len(sel) > max_points:
            sel = sel[np.linspace(0, len(sel) - 1, max_points).astype(int)]
        sub = self.rows[sel]
        return {"frame": [int(v) for v in sub["frame"]],
                "yaw": [_num(v) for v in sub["pose"][:, 0]],
                "pitch": [_num(v) for v in sub["pose"][:, 1]],
                "roll": [_num(v) for v in sub["pose"][:, 2]]}

    def identify_delta_skips(
        self,
        cuts: Optional[set[int]] = None,
        keyframes: Optional[set[int]] = None,
        max_consecutive: int = 2,
        disp_threshold: float = 0.15,
        illum_threshold: float = 0.005,
        roi_lumas: Optional[Mapping[Tuple[int, int], float]] = None,
    ) -> List[Tuple[int, int]]:
        """Identify frames in this trajectory cache where inner landmark displacement
        (delta_eyes_mouth_px) is < disp_threshold (0.15 px) and local illumination
        change is < illum_threshold (0.5%), capped at max_consecutive skips, never on
        keyframes or scene cuts.

        Flags matching rows with FLAG_REUSE_FLOW_WARP and returns list of (frame, track).
        """
        cuts = cuts or set()
        keyframes = keyframes or set()
        if len(self.rows) < 2:
            return []

        from roop.temporal_smart_skip import compute_inner_landmark_displacement

        tracks = np.unique(self.rows["track"])
        flagged: List[Tuple[int, int]] = []

        for tid in tracks:
            mask = self.rows["track"] == tid
            track_indices = np.where(mask)[0]
            if len(track_indices) < 2:
                continue

            order = np.argsort(self.rows["frame"][track_indices])
            sorted_indices = track_indices[order]

            consecutive_skips = 0
            for idx_pos in range(len(sorted_indices)):
                curr_idx = sorted_indices[idx_pos]
                curr_frame = int(self.rows["frame"][curr_idx])

                if idx_pos == 0:
                    self.rows["flags"][curr_idx] &= ~FLAG_REUSE_FLOW_WARP
                    consecutive_skips = 0
                    continue

                prev_idx = sorted_indices[idx_pos - 1]
                prev_frame = int(self.rows["frame"][prev_idx])

                if curr_frame != prev_frame + 1:
                    self.rows["flags"][curr_idx] &= ~FLAG_REUSE_FLOW_WARP
                    consecutive_skips = 0
                    continue

                if curr_frame in cuts or curr_frame in keyframes:
                    self.rows["flags"][curr_idx] &= ~FLAG_REUSE_FLOW_WARP
                    consecutive_skips = 0
                    continue

                if consecutive_skips >= max_consecutive:
                    self.rows["flags"][curr_idx] &= ~FLAG_REUSE_FLOW_WARP
                    consecutive_skips = 0
                    continue

                curr_kps = self.rows["kps"][curr_idx]
                prev_kps = self.rows["kps"][prev_idx]
                disp = compute_inner_landmark_displacement(curr_kps, prev_kps)

                delta_illum = 0.0
                if roi_lumas is not None:
                    curr_luma = float(roi_lumas.get((curr_frame, int(tid)), 0.5))
                    prev_luma = float(roi_lumas.get((prev_frame, int(tid)), 0.5))
                    delta_illum = abs(curr_luma - prev_luma)

                if disp < disp_threshold and delta_illum < illum_threshold:
                    self.rows["flags"][curr_idx] |= FLAG_REUSE_FLOW_WARP
                    consecutive_skips += 1
                    flagged.append((curr_frame, int(tid)))
                else:
                    self.rows["flags"][curr_idx] &= ~FLAG_REUSE_FLOW_WARP
                    consecutive_skips = 0

        return flagged

    def summary(self) -> Dict[str, Any]:
        out = dict(self.meta)
        out["faces"], out["bytes"] = len(self), self.nbytes
        rng = self.frame_range
        out["first_frame"], out["last_frame"] = (rng if rng else (None, None))
        out["track_ids"] = sorted({int(t) for t in np.unique(self.rows["track"])}) if len(self.rows) else []
        return out


_TEMPLATE_FIELDS = {112: ("template_112", "silhouette_112"),
                    128: ("template_128", "silhouette_128"),
                    256: ("template_256", "silhouette_256"),
                    512: ("template_512", "silhouette_512")}


def pose_template_from_row(row: Any, size: int, mode: str = TEMPLATE):
    """Rehydrate a read-only, crop-specific template carried by one plan row."""
    from roop.pose_template import PoseTemplate, warp_template
    size = int(size)
    fields = _TEMPLATE_FIELDS.get(size)
    if fields is None:
        return warp_template(row["pose"], size, mode, source_anchors=row["silhouette"])
    points_field, silhouette_field = fields
    return PoseTemplate(row[points_field], row[silhouette_field], row["anchor_weights"],
                        row["silhouette"], *map(float, row["pose"]), size, mode)


def _encode_pts_table(pts: Any) -> str:
    exact = []
    for frame_idx, value in enumerate(() if pts is None else pts):
        value = value if isinstance(value, Fraction) else Fraction(value)
        exact.append([int(frame_idx), int(value.numerator), int(value.denominator)])
    packed = zlib.compress(json.dumps(exact, separators=(",", ":")).encode("ascii"), level=9)
    return base64.b64encode(packed).decode("ascii")


def _decode_pts_table(blob: str) -> Tuple[Tuple[int, Fraction], ...]:
    try:
        raw = zlib.decompress(base64.b64decode(str(blob).encode("ascii")))
        entries = json.loads(raw.decode("ascii"))
        return tuple((int(frame), Fraction(int(num), int(den))) for frame, num, den in entries)
    except (ValueError, TypeError, KeyError, ZeroDivisionError, binascii.Error,
            zlib.error, json.JSONDecodeError):
        return ()


class TrajectoryPlan(TrajectoryCache):
    """Deterministic, frame-indexed output of the sequential video pre-scan.

    Detection, association, Kalman/causal filtering, offline forward/backward
    landmark smoothing, pose estimation, and template projection finish before
    workers start. The render and preview paths only read rows by frame index.
    """

    def __init__(self, rows: np.ndarray, meta: Optional[Mapping[str, Any]] = None):
        super().__init__(rows, meta)
        self._pts_table_cache = _decode_pts_table(self.meta.get("pts_table", ""))

    @classmethod
    def build(cls, faces_by_frame: Mapping[int, List[Any]], *, frame_start: int = 0,
              sar: float = 1.0, fps: Optional[float] = None,
              total_frames: Optional[int] = None, source_video: Optional[str] = None,
              frame_shape: Any = None, cuts: Optional[set[int]] = None,
              mode: str = TEMPLATE, pts: Any = None,
              min_cutoff: float = 0.05, beta: float = 0.25,
              d_cutoff: float = 8.0) -> "TrajectoryPlan":
        from roop.pose_router import estimate_head_pose
        from roop.pose_template import source_silhouette_anchors, warp_template
        from roop.temporal_stabilizer import forward_backward_one_euro

        frames, tracks, flags, source_kps, bbox_arr, faces = cls._collect(faces_by_frame, frame_start)
        n = len(frames)
        rows = np.zeros(n, dtype=ROW)
        rows["frame"], rows["track"], rows["flags"] = frames, tracks, flags
        rows["source_kps"], rows["bbox"] = source_kps, bbox_arr

        source_timing = None
        if pts is None and source_video:
            try:
                from roop.vfr_timing import probe_source_timing
                source_timing = probe_source_timing(source_video)
                pts = source_timing.pts if source_timing is not None else None
            except Exception as exc:
                _swallowed("roop/trajectory_cache.py:pts", exc,
                           "the trajectory will use frame-rate timestamps")
        pts = tuple(() if pts is None else pts)
        rate = float(fps or 0.0)
        if not math.isfinite(rate) or rate <= 0.0:
            rate = 30.0
        min_cutoff = max(0.005, float(min_cutoff))
        beta = max(0.0, float(beta))
        d_cutoff = max(0.005, float(d_cutoff))

        def timestamp(abs_frame: int) -> float:
            if 0 <= abs_frame < len(pts):
                value = pts[abs_frame]
                return float(value if isinstance(value, Fraction) else Fraction(value))
            return float(abs_frame) / rate

        smoothed = source_kps.copy()
        cuts_abs = {int(frame_start) + int(c) for c in (cuts or ())}
        # Split each track at a frame gap or shot cut so filter state never
        # crosses a discontinuity. Each chunk is filtered in timestamp order.
        for tid in np.unique(tracks):
            inds = np.flatnonzero(tracks == tid)
            if not len(inds):
                continue
            chunks: List[List[int]] = [[]]
            for idx in inds:
                if chunks[-1]:
                    prev = int(frames[chunks[-1][-1]])
                    cur = int(frames[idx])
                    if cur != prev + 1 or cur in cuts_abs:
                        chunks.append([])
                chunks[-1].append(int(idx))
            for chunk in chunks:
                if not chunk:
                    continue
                chunk_frames = frames[chunk].astype(np.int64)
                times = np.asarray([timestamp(int(v)) for v in chunk_frames], dtype=np.float64)
                if len(times) > 1 and np.any(np.diff(times) <= 0.0):
                    times = np.arange(len(chunk), dtype=np.float64) / rate
                smoothed[chunk] = forward_backward_one_euro(
                    source_kps[chunk], times, min_cutoff=min_cutoff,
                    beta=beta, d_cutoff=d_cutoff)
        rows["kps"] = smoothed

        poses = np.full((n, 3), np.nan, dtype=np.float32)
        source_anchors = np.full((n, 3, 2), np.nan, dtype=np.float32)
        for i in range(n):
            try:
                pose = estimate_head_pose(smoothed[i], frame_shape=frame_shape)
            except Exception as exc:
                _swallowed("roop/trajectory_cache.py:plan_pose", exc,
                           "stored face pose used when available")
                pose = None
            if pose is None:
                fallback = _get(faces[i], "_pose_pnp")
                if fallback is not None:
                    try:
                        pose = tuple(float(v) for v in np.asarray(fallback).reshape(3))
                    except (TypeError, ValueError):
                        pose = None
            if pose is None:
                pose = (0.0, 0.0, 0.0)
            poses[i] = np.asarray(pose, dtype=np.float32)
            source_anchors[i] = source_silhouette_anchors(faces[i], float(pose[0]))
        rows["pose"] = poses
        rows["silhouette"] = source_anchors

        templates_112 = np.empty((n, 5, 2), dtype=np.float32)
        for i, pose in enumerate(poses):
            for size in (112, 128, 256, 512):
                warped = warp_template(pose, size, mode, source_anchors=source_anchors[i])
                rows[_TEMPLATE_FIELDS[size][0]][i] = warped.points
                rows[_TEMPLATE_FIELDS[size][1]][i] = warped.silhouette
                if size == 112:
                    rows["anchor_weights"][i] = warped.anchor_weights
                    templates_112[i] = warped.points
        m = np.zeros((n, 2, 3), dtype=np.float64)
        if n:
            for i in range(n):
                m[i] = similarity_matrices(smoothed[i:i + 1], templates_112[i], sar)[0]
        rows["M"] = m
        rows["M_inv"] = invert_matrices(m) if n else np.zeros((0, 2, 3), dtype=np.float64)

        meta: Dict[str, Any] = dict(frame_start=int(frame_start), sar=float(sar), faces=int(n),
                                    frames=int(len(np.unique(frames))) if n else 0,
                                    tracks=int(len(np.unique(tracks))) if n else 0,
                                    plan_kind="forward_backward_one_euro",
                                    template_mode=str(mode or TEMPLATE),
                                    filter_min_cutoff=min_cutoff, filter_beta=beta,
                                    filter_d_cutoff=d_cutoff)
        if fps:
            meta["fps"] = float(fps)
        if total_frames is not None:
            meta["total_frames"] = int(total_frames)
        if source_video:
            try:
                stat = os.stat(source_video)
                meta.update(source_path=os.path.abspath(source_video), source_size=int(stat.st_size),
                            source_mtime_ns=int(stat.st_mtime_ns))
            except OSError:
                meta["source_path"] = os.path.abspath(source_video)
        if frame_shape is not None:
            meta["frame_shape"] = [int(v) for v in tuple(frame_shape)[:2]]
        if pts:
            meta["pts_table"] = _encode_pts_table(pts)
            meta["pts_count"] = len(pts)
            meta["pts_time_base"] = "seconds_as_exact_fraction"
            meta["is_vfr"] = bool(source_timing.is_vfr) if source_timing is not None else False
        return cls(rows, meta)

    @property
    def pts_table(self) -> Tuple[Tuple[int, Fraction], ...]:
        return self._pts_table_cache

    def pts_for_frame(self, frame_idx: int) -> Optional[Fraction]:
        index = int(frame_idx)
        table = self.pts_table
        if 0 <= index < len(table) and table[index][0] == index:
            return table[index][1]
        return None

    def frame_for_pts(self, pts: Any) -> Optional[int]:
        try:
            target = pts if isinstance(pts, Fraction) else Fraction(str(pts))
        except (ValueError, ZeroDivisionError, TypeError):
            return None
        for frame_idx, value in self.pts_table:
            if value == target:
                return frame_idx
        return None

    def matches_source(self, video_path: str) -> bool:
        source = self.meta.get("source_path")
        if not source or not video_path:
            return False
        try:
            path = os.path.abspath(video_path)
            stat = os.stat(path)
            return (os.path.normcase(path) == os.path.normcase(os.path.abspath(source))
                    and int(stat.st_size) == int(self.meta.get("source_size", -1))
                    and int(stat.st_mtime_ns) == int(self.meta.get("source_mtime_ns", -1)))
        except OSError:
            return False

    def seek_time(self, video_path: str, frame_idx: int) -> Optional[float]:
        if not self.meta.get("is_vfr") or not self.matches_source(video_path):
            return None
        k = int(frame_idx)
        here, previous = self.pts_for_frame(k), self.pts_for_frame(k - 1)
        if here is None or previous is None:
            return None
        return float((previous + here) / 2)

    def faces_for_frame(self, frame_idx: int, faces: Optional[List[Any]], *, absolute: bool = False) -> List[Any]:
        """Clone faces and stamp plan geometry; no shared face/filter state reaches workers."""
        rows = self.at(frame_idx, absolute=absolute)
        by_track = {int(row["track"]): row for row in rows}
        assigned: Dict[int, Any] = {}
        used_tracks = set()

        def overlap(a, b):
            a, b = np.asarray(a, dtype=np.float64).reshape(4), np.asarray(b, dtype=np.float64).reshape(4)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                return 0.0
            w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = w * h
            union = max(0.0, (a[2] - a[0]) * (a[3] - a[1])) + max(
                0.0, (b[2] - b[0]) * (b[3] - b[1])) - inter
            return inter / union if union > 1e-9 else 0.0

        faces = list(faces or ())
        candidates = []
        for i, face in enumerate(faces):
            track = _get(face, "_track_id")
            row = by_track.get(int(track)) if track is not None else None
            if row is not None and overlap(_get(face, "bbox"), row["bbox"]) >= 0.15:
                assigned[i] = row
                used_tracks.add(int(row["track"]))
                continue
            for candidate in rows:
                tid = int(candidate["track"])
                score = overlap(_get(face, "bbox"), candidate["bbox"])
                if tid not in used_tracks and score >= 0.30:
                    candidates.append((-score, i, tid, candidate))
        for _negative_score, i, tid, row in sorted(candidates, key=lambda item: (item[0], item[1], item[2])):
            if i not in assigned and tid not in used_tracks:
                assigned[i] = row
                used_tracks.add(tid)

        output = []
        for i, face in enumerate(faces):
            try:
                clone = type(face)(face)
            except Exception:
                clone = dict(face) if isinstance(face, dict) else face
            if isinstance(clone, dict):
                for key, value in tuple(clone.items()):
                    if isinstance(value, np.ndarray):
                        clone[key] = value.copy()
            row = assigned.get(i)
            if row is not None:
                assignments = {"kps": np.array(row["kps"], copy=True),
                               "bbox": np.array(row["bbox"], copy=True),
                               "_track_id": int(row["track"]),
                               "_pose_pnp": tuple(float(v) for v in row["pose"]),
                               "_adaptive_yaw": float(row["pose"][0]),
                               "_adaptive_pitch": float(row["pose"][1]),
                               "_adaptive_roll": float(row["pose"][2])}
                for key, value in assignments.items():
                    try:
                        clone[key] = value
                    except (TypeError, AttributeError):
                        setattr(clone, key, value)
                row_copy = row.copy()
                row_copy.setflags(write=False)
                try:
                    clone["_trajectory_plan_row"] = row_copy
                except (TypeError, AttributeError):
                    setattr(clone, "_trajectory_plan_row", row_copy)
            output.append(clone)
        return output


def _num(v) -> Optional[float]:
    v = float(v)
    return round(v, 2) if math.isfinite(v) else None


def pose_payload(cache: Optional["TrajectoryCache"], frame: int, absolute: bool = False) -> Dict[str, Any]:
    """What `GET /api/trajectory/pose` returns for one frame: every face there with its pose, or why not."""
    if cache is None:
        return {"available": False, "reason": "no trajectory yet (the pre-pass has not finished, or this job has none)"}
    used, rows = cache.nearest(frame, absolute=absolute)
    faces = []
    for r in rows:
        faces.append({"track": int(r["track"]), "yaw": _num(r["pose"][0]), "pitch": _num(r["pose"][1]),
                      "roll": _num(r["pose"][2]), "flags": int(r["flags"]),
                      "bbox": [round(float(x), 1) for x in r["bbox"]],
                      "scale": round(float(math.hypot(r["M"][0, 0], r["M"][1, 0])), 4)})
    return {"available": True, "frame": used, "requested": int(frame), "faces": faces,
            "frame_start": int(cache.meta.get("frame_start", 0)), "pose_convention": POSE_CONVENTION,
            "first_frame": cache.frame_range[0] if cache.frame_range else None,
            "last_frame": cache.frame_range[1] if cache.frame_range else None}


# ── the cache of the job that is running ─────────────────────────────────────────────────────────────
_LOCK = threading.Lock()
_ACTIVE: Dict[str, Any] = {"cache": None, "project_id": None}


def set_active(cache: Optional[TrajectoryCache], project_id: Optional[str] = None) -> None:
    with _LOCK:
        _ACTIVE["cache"], _ACTIVE["project_id"] = cache, project_id


def active() -> Optional[TrajectoryCache]:
    with _LOCK:
        return _ACTIVE["cache"]


def active_project() -> Optional[str]:
    with _LOCK:
        return _ACTIVE["project_id"]


def clear_active() -> None:
    set_active(None, None)
