"""The Stage 2 smoothed face trajectory: alignment matrices and head pose for every face of every frame.

WHAT IT HOLDS. One row per face per frame, sorted by (frame, track), in ONE structured array so a single
`np.load(..., mmap_mode='r')` maps the whole thing and a lookup by frame is a binary search:

    frame   int32   ABSOLUTE decoder frame (frame_start + the scan's frame index), so a resumed run that
                    starts later and a fresh run that starts earlier index the same rows
    track   int32   the pre-pass track id (`face['_track_id']`)
    flags   uint8   1 gap-filled / interpolated, 2 placed by optical flow, 4 Kalman-coasted
    kps     f4 5x2  the SMOOTHED alignment keypoints the swap will use
    bbox    f4 4
    pose    f4 3    yaw, pitch, roll in degrees, solved FROM THOSE SMOOTHED keypoints (head_pose.py's
                    convention: +yaw = nose toward the viewer's right, +pitch = tilted DOWN, +roll clockwise)
    M       f8 2x3  frame -> 112x112 ArcFace template (what `face_util.estimate_norm(kps, 112)` returns)
    M_inv   f8 2x3  template -> frame (what paste-back uses)

WHAT IT IS NOT. It is not a hand-off: Stage 3 still reads `ProcessMgr._temporal_faces`, and it does not read
this file's matrices. Stage 3's own alignment depends on the swapper's crop size and template, the restorers'
sizes and the clip's SAR (`estimate_norm(lmk, size, mode, sar)`), none of which one 112-px matrix can stand in
for without changing pixels, and Phase 11 measured the CPU affine math at < 0.5% of a render. The consumers are
the head-pose HUD (`GET /api/trajectory/pose`), the project checkpoint (so a resumed project can show its
trajectory without the pre-pass) and any tool that wants the alignment of a frame without re-running a model.

THE SAME LEAST SQUARES, NOT THE SAME BITS. The matrices are solved in closed form for every row at once (the 2-D
similarity that minimises the squared landmark error is a complex ratio; `estimate_norm`'s SVD solves the same
problem). `tests/test_trajectory_cache.py` pins them against `skimage.SimilarityTransform` to 1e-8 and against
`estimate_norm` to 5e-5 px - which is `estimate_norm`'s own float32 noise (measured: it moves by 2.1e-5 between a
float32 and a float64 call with the SAME keypoint values) - and `M_inv` against `cv2.invertAffineTransform` to 1e-10.
Nothing that needs bit-identity with `estimate_norm` may be fed from here.

THE ACTIVE CACHE. `set_active` / `active` hold the cache of the job that is running (or the last one), cleared
when a job begins, so the HUD never shows the previous clip's heads.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from roop.degrade import swallowed as _swallowed

FORMAT_NAME = "roop-trajectory"
FORMAT_VERSION = 1

FLAG_INTERPOLATED = 1
FLAG_FLOW = 2
FLAG_COASTED = 4
FLAG_REUSE_FLOW_WARP = 8

ROW = np.dtype([
    ("frame", "<i4"), ("track", "<i4"), ("flags", "u1"),
    ("kps", "<f4", (5, 2)), ("bbox", "<f4", (4,)), ("pose", "<f4", (3,)),
    ("M", "<f8", (2, 3)), ("M_inv", "<f8", (2, 3)),
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
            rows["kps"], rows["bbox"], rows["pose"] = kps_arr, bbox_arr, poses
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
                or not np.array_equal(sub["track"], tracks) or not np.array_equal(sub["kps"], kps_arr)):
            return None
        meta = dict(self.meta, frame_start=int(frame_start), faces=int(len(sub)),
                    frames=int(len(np.unique(frames))), tracks=int(len(np.unique(tracks))))
        return TrajectoryCache(sub, meta)

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
