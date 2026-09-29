"""Pose-binned reference portfolio: the best frame per head angle, and a fused
identity embedding from the near-frontal ones.

Consumes Stage-2 rows (``pose_quality.CandidateFaceMetric``), sorts the valid
ones into a 9-bin yaw/pitch lattice, keeps the highest composite score per bin,
exports a 512x512 aligned JPEG per chosen frame, and fuses the frontal and
quarter-angle embeddings into one L2-normalised reference vector.

Sign convention -- read before touching the bins
------------------------------------------------
Angles arrive in ``face_util.solve_pose_5pt``'s convention (Stage 2 is pinned to
it), which is: **+yaw = the face turned toward the VIEWER'S RIGHT** (nose toward
image right) and **+pitch = the face tilted DOWN** (nose tip toward the mouth).
Verified from the reference head itself: ``_REF3D_68`` is y-up with the nose at
+z (toward the viewer), and ``_project_reference(0, +30)`` -- a head whose
forward axis points at the floor -- solves to pitch +30. (Some prose in the tree
says the opposite: ``face_3d_recon.decompose_yaw_pitch`` documents its OWN
decomposer as up-positive, and ``nonfrontal.py`` calls pitch +30 "tilted up".)

The bins below are defined on the physical direction, so ``BIN_7_PITCH_UP``
holds faces looking UP: binning reads ``pitch_up = -pitch``. Yaw is used as is,
so LEFT means facing the viewer's left.

The lattice has gaps (e.g. yaw 0 at pitch 12, or yaw 30 at pitch 20): a
candidate there belongs to no bin but can fill an empty one as its nearest
neighbour.
"""
from __future__ import annotations

import base64
import enum
import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from roop.degrade import swallowed as _swallowed
from roop.pose_quality import CandidateFaceMetric
from roop.scanner import open_capture

logger = logging.getLogger("roop.angle_portfolio")

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE_ROOT = os.path.join(_APP_DIR, "output", "cache", "target_angles")
EXPORT_SIZE = 512
EXPORT_TEMPLATE = "ffhq_512"      # the full-face template the enhancers align to
# An empty bin may borrow a candidate at most this far (degrees) outside it.
MAX_FILL_DEG = float(os.environ.get("ROOP_PORTFOLIO_MAX_FILL_DEG", "10") or 10)
_UNBOUNDED = 90.0


class AngleBin(enum.IntEnum):
    BIN_0_FRONTAL = 0
    BIN_1_QUARTER_LEFT = 1
    BIN_2_QUARTER_RIGHT = 2
    BIN_3_HALF_PROFILE_LEFT = 3
    BIN_4_HALF_PROFILE_RIGHT = 4
    BIN_5_PROFILE_LEFT = 5
    BIN_6_PROFILE_RIGHT = 6
    BIN_7_PITCH_UP = 7
    BIN_8_PITCH_DOWN = 8

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS = {
    AngleBin.BIN_0_FRONTAL: "Frontal",
    AngleBin.BIN_1_QUARTER_LEFT: "Quarter left",
    AngleBin.BIN_2_QUARTER_RIGHT: "Quarter right",
    AngleBin.BIN_3_HALF_PROFILE_LEFT: "Half profile left",
    AngleBin.BIN_4_HALF_PROFILE_RIGHT: "Half profile right",
    AngleBin.BIN_5_PROFILE_LEFT: "Profile left",
    AngleBin.BIN_6_PROFILE_RIGHT: "Profile right",
    AngleBin.BIN_7_PITCH_UP: "Looking up",
    AngleBin.BIN_8_PITCH_DOWN: "Looking down",
}

FUSION_BINS = (AngleBin.BIN_0_FRONTAL, AngleBin.BIN_1_QUARTER_LEFT, AngleBin.BIN_2_QUARTER_RIGHT)

# (yaw_lo, yaw_hi, pitch_up_lo, pitch_up_hi) as CLOSED rectangles, used only for
# the nearest-neighbour distance. Membership uses the brief's exact open/closed
# edges in angle_bin() below.
_REGIONS = {
    AngleBin.BIN_0_FRONTAL: (-10, 10, -10, 10),
    AngleBin.BIN_1_QUARTER_LEFT: (-25, -10, -10, 10),
    AngleBin.BIN_2_QUARTER_RIGHT: (10, 25, -10, 10),
    AngleBin.BIN_3_HALF_PROFILE_LEFT: (-45, -25, -15, 15),
    AngleBin.BIN_4_HALF_PROFILE_RIGHT: (25, 45, -15, 15),
    AngleBin.BIN_5_PROFILE_LEFT: (-_UNBOUNDED, -45, -_UNBOUNDED, _UNBOUNDED),
    AngleBin.BIN_6_PROFILE_RIGHT: (45, _UNBOUNDED, -_UNBOUNDED, _UNBOUNDED),
    AngleBin.BIN_7_PITCH_UP: (-20, 20, 15, _UNBOUNDED),
    AngleBin.BIN_8_PITCH_DOWN: (-20, 20, -_UNBOUNDED, -15),
}


def pitch_up(pitch: float) -> float:
    """Project pitch (+ = down) to the bins' up-positive pitch."""
    return -float(pitch)


def angle_bin(yaw: float, pitch: float) -> Optional[AngleBin]:
    """The bin for a (yaw, pitch) in solve_pose_5pt's convention, or None for
    a pose in one of the lattice's gaps."""
    y, p = float(yaw), pitch_up(pitch)
    if not (math.isfinite(y) and math.isfinite(p)):
        return None
    if y < -45:
        return AngleBin.BIN_5_PROFILE_LEFT
    if y > 45:
        return AngleBin.BIN_6_PROFILE_RIGHT
    if -10 <= p <= 10:
        if -10 <= y <= 10:
            return AngleBin.BIN_0_FRONTAL
        if -25 <= y < -10:
            return AngleBin.BIN_1_QUARTER_LEFT
        if 10 < y <= 25:
            return AngleBin.BIN_2_QUARTER_RIGHT
    if -15 <= p <= 15:
        if -45 <= y < -25:
            return AngleBin.BIN_3_HALF_PROFILE_LEFT
        if 25 < y <= 45:
            return AngleBin.BIN_4_HALF_PROFILE_RIGHT
    if -20 <= y <= 20:
        if p > 15:
            return AngleBin.BIN_7_PITCH_UP
        if p < -15:
            return AngleBin.BIN_8_PITCH_DOWN
    return None


# Faces this close to straight-on in yaw define a person's NEUTRAL pitch.
NEUTRAL_YAW_DEG = 25.0


def neutral_pitch(poses: Iterable[Tuple[float, float]]) -> float:
    """A person's neutral pitch: the median pitch of their near-frontal faces
    ((yaw, pitch) pairs, |yaw| <= 25), 0 when there are none.

    5-point pitch is read against ONE reference head, so a person whose eye /
    nose / mouth proportions differ from it reads a constant offset: a level,
    straight-on passport photo (app/facesets/anshita.png) solves to 19 deg "up"
    with the mouth closed (solve_pose_jaw_5pt: jaw -0.09, still 23 deg). The
    offset is anatomy, so it is the same in every frame of that person;
    subtracting their median removes it. Bins read pitch RELATIVE to this."""
    vals = sorted(float(p) for y, p in poses
                  if y is not None and p is not None and math.isfinite(float(y)) and math.isfinite(float(p))
                  and abs(float(y)) <= NEUTRAL_YAW_DEG)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def distance_to_bin(yaw: float, pitch: float, bin_: AngleBin) -> float:
    """Degrees from a pose to the bin's region (0 inside it)."""
    y0, y1, p0, p1 = _REGIONS[bin_]
    y, p = float(yaw), pitch_up(pitch)
    dy = max(y0 - y, 0.0, y - y1)
    dp = max(p0 - p, 0.0, p - p1)
    return math.hypot(dy, dp)


@dataclass
class BinSlot:
    """Why a bin holds what it holds."""
    status: str                           # "selected" | "nearest" | "override" | "missing"
    candidates: int = 0                   # valid candidates that fell in the bin
    distance_deg: float = 0.0             # > 0 only for "nearest"
    # The bin a "nearest" fill or an "override" frame actually sits in by its
    # own pose (None = a lattice gap, or no pose).
    source_bin: Optional[str] = None


def _rank_key(m: CandidateFaceMetric):
    sim = m.id_similarity if m.id_similarity is not None else -2.0
    return (m.composite_score, sim, -m.frame_idx)


def _identity(m: CandidateFaceMetric) -> Tuple[int, Optional[int]]:
    return (m.frame_idx, m.track_id)


class AnglePortfolioSelector:
    def __init__(self, max_fill_deg: float = MAX_FILL_DEG):
        self.max_fill_deg = float(max_fill_deg)
        self.report: Dict[AngleBin, BinSlot] = {}
        # track_id -> neutral pitch subtracted before binning (see neutral_pitch)
        self.neutral: Dict[Optional[int], float] = {}

    def relative_pitch(self, m: CandidateFaceMetric) -> Optional[float]:
        if m.pitch is None:
            return None
        return float(m.pitch) - self.neutral.get(m.track_id, self.neutral.get(None, 0.0))

    def select_portfolio(self, metrics: List[CandidateFaceMetric],
                         overrides: Optional[Dict[AngleBin, CandidateFaceMetric]] = None
                         ) -> Dict[AngleBin, CandidateFaceMetric]:
        """Best valid candidate per bin. An empty bin takes the nearest unused
        candidate within ``max_fill_deg`` of it (flagged "nearest" in
        ``self.report``) or stays out of the result (flagged "missing").

        ``overrides`` are the user's own picks: they win their bin whatever
        their pose or validity (flagged "override", with the bin their pose
        really falls in), and are not lent to other bins."""
        overrides = dict(overrides or {})
        usable = [m for m in metrics
                  if m.is_valid and m.yaw is not None and m.pitch is not None
                  and math.isfinite(m.yaw) and math.isfinite(m.pitch)]
        # Per person (track), from every candidate that has a pose, valid or not:
        # the neutral describes the person, not what survived the gates.
        by_track: Dict[Optional[int], List[Tuple[float, float]]] = {}
        for m in metrics:
            if m.yaw is not None and m.pitch is not None:
                by_track.setdefault(m.track_id, []).append((m.yaw, m.pitch))
        self.neutral = {t: neutral_pitch(v) for t, v in by_track.items()}
        self.neutral[None] = neutral_pitch(pose for v in by_track.values() for pose in v)
        groups: Dict[AngleBin, List[CandidateFaceMetric]] = {b: [] for b in AngleBin}
        home: Dict[Tuple[int, Optional[int]], Optional[AngleBin]] = {}
        for m in usable:
            b = angle_bin(m.yaw, self.relative_pitch(m))
            home[_identity(m)] = b
            if b is not None:
                groups[b].append(m)

        portfolio: Dict[AngleBin, CandidateFaceMetric] = {}
        report: Dict[AngleBin, BinSlot] = {}
        used = set()
        for b, m in overrides.items():
            b = AngleBin(b)
            portfolio[b] = m
            used.add(_identity(m))
            natural = (angle_bin(m.yaw, self.relative_pitch(m))
                       if m.yaw is not None and m.pitch is not None else None)
            report[b] = BinSlot("override", candidates=len(groups[b]),
                                source_bin=None if natural is None else natural.name)
        for b in AngleBin:
            if b in portfolio:
                continue
            if groups[b]:
                best = max(groups[b], key=_rank_key)
                portfolio[b] = best
                used.add(_identity(best))
                report[b] = BinSlot("selected", candidates=len(groups[b]))

        # Nearest-neighbour fill, globally closest pairs first, so one candidate
        # sitting between two empty bins goes to the one it is closer to.
        empty = [b for b in AngleBin if b not in portfolio]
        pairs = []
        for b in empty:
            for m in usable:
                if _identity(m) in used:
                    continue
                d = distance_to_bin(m.yaw, self.relative_pitch(m), b)
                if d <= self.max_fill_deg:
                    pairs.append((d, -m.composite_score, int(b), m))
        pairs.sort(key=lambda t: t[:3])
        for d, _neg_score, b_int, m in pairs:
            b = AngleBin(b_int)
            if b in portfolio or _identity(m) in used:
                continue
            portfolio[b] = m
            used.add(_identity(m))
            src = home.get(_identity(m))
            report[b] = BinSlot("nearest", distance_deg=round(d, 2),
                                source_bin=None if src is None else src.name)
        for b in AngleBin:
            report.setdefault(b, BinSlot("missing"))
        self.report = report
        return {b: portfolio[b] for b in AngleBin if b in portfolio}


def synthesize_fused_embedding(portfolio: Dict[AngleBin, CandidateFaceMetric],
                               embeddings: Dict[int, np.ndarray]) -> np.ndarray:
    """Composite-score-weighted mean of the BIN_0/1/2 embeddings (each
    L2-normalised first, so the weight - not the vector's norm - decides its
    pull), L2-normalised. ``embeddings`` is keyed by frame_idx. Raises
    ValueError when none of those bins has an embedding: fusing profiles
    instead would change what the reference means."""
    vecs, weights = [], []
    for b in FUSION_BINS:
        m = portfolio.get(b)
        if m is None or m.frame_idx not in embeddings:
            continue
        v = np.asarray(embeddings[m.frame_idx], dtype=np.float64).reshape(-1)
        n = float(np.linalg.norm(v))
        if not (n > 1e-9 and np.isfinite(v).all()):
            continue
        vecs.append(v / n)
        weights.append(max(0.0, float(m.composite_score)))
    if not vecs:
        raise ValueError("no frontal or quarter-angle embedding to fuse")
    if len({v.shape for v in vecs}) != 1:
        raise ValueError("embeddings have different dimensions")
    w = np.asarray(weights)
    if not w.sum() > 0:
        w = np.ones_like(w)
    fused = (np.stack(vecs) * w[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(fused))
    if not norm > 1e-9:
        raise ValueError("fused embedding cancelled out")
    return (fused / norm).astype(np.float32)


def shortlist_candidates(detections: List[Dict[str, Any]], frame_shape: Tuple[int, int],
                         per_bin: int = 40) -> List[Dict[str, Any]]:
    """At most ``per_bin`` Stage-1 detections per pose bin (gaps grouped by
    10-degree cell, a quarter of that each), spread evenly over time.

    Stage 2 decodes and runs the landmark model per candidate; a long clip's
    member tracklets hold tens of thousands of detections that are mostly the
    same pose. The pose is solved here from the keypoints alone (no decode), so
    thinning never costs a rare angle its only frames. Evenly spaced in time
    rather than top-scored because sharpness is not known yet and a burst of
    consecutive frames tends to share its motion blur."""
    from roop.pose_quality import estimate_head_pose
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    posed = []
    for det in detections:
        kps = det.get("kps")
        if kps is None:
            continue
        yaw, pitch, _roll = estimate_head_pose(np.asarray(kps, dtype=np.float64), frame_shape)
        if math.isfinite(yaw) and math.isfinite(pitch):
            posed.append((det, yaw, pitch))
    neutral: Dict[Any, float] = {}
    for track in {d.get("track_id") for d, _, _ in posed}:
        neutral[track] = neutral_pitch((y, p) for d, y, p in posed if d.get("track_id") == track)
    for det, yaw, pitch in posed:
        b = angle_bin(yaw, pitch - neutral.get(det.get("track_id"), 0.0))
        key = b if b is not None else ("gap", int(round(yaw / 10.0)), int(round(pitch / 10.0)))
        groups.setdefault(key, []).append(det)
    out: List[Dict[str, Any]] = []
    for key, dets in groups.items():
        cap = per_bin if isinstance(key, AngleBin) else max(1, per_bin // 4)
        dets = sorted(dets, key=lambda d: int(d["frame_idx"]))
        if len(dets) > cap:
            picks = np.linspace(0, len(dets) - 1, cap).round().astype(int)
            dets = [dets[i] for i in sorted(set(picks.tolist()))]
        out.extend(dets)
    return sorted(out, key=lambda d: int(d["frame_idx"]))


# ── Frame export ─────────────────────────────────────────────────────────────
def app_embedding(frame: np.ndarray, kps: Any) -> Optional[np.ndarray]:
    """The app's own recognizer (the analyser's 'recognition' model, i.e. the
    embedding every other identity decision uses) on the aligned face."""
    from insightface.app.common import Face
    from roop.face_util import lease_face_analyser
    with lease_face_analyser() as fa:
        model = getattr(fa, "models", {}).get("recognition")
        if model is None:
            return None
        face = Face(kps=np.asarray(kps, dtype=np.float32))
        emb = model.get(frame, face)
    return None if emb is None else np.asarray(emb, dtype=np.float32).reshape(-1)


def aligned_crop(frame: np.ndarray, kps: Any, size: int = EXPORT_SIZE) -> np.ndarray:
    from roop.face_util import align_crop
    crop, _ = align_crop(frame, np.asarray(kps, dtype=np.float32).reshape(5, 2), size, mode=EXPORT_TEMPLATE)
    return crop


def _media_key(media_path: str, portfolio: Dict[AngleBin, CandidateFaceMetric]) -> str:
    st = os.stat(media_path)
    sel = [(int(b), m.frame_idx, m.track_id, m.kps) for b, m in sorted(portfolio.items())]
    raw = json.dumps([os.path.abspath(media_path), st.st_size, int(st.st_mtime), sel], sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _dataurl(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def _decode_frames(media_path: str, wanted: Iterable[int]) -> Iterable[Tuple[int, np.ndarray]]:
    want = sorted(set(int(i) for i in wanted))
    if not want:
        return
    with open_capture(media_path) as capture:
        idx, pos = -1, 0
        while pos < len(want):
            idx += 1
            if not capture.grab():
                return
            if idx != want[pos]:
                continue
            pos += 1
            ok, frame = capture.retrieve()
            if ok and frame is not None:
                yield idx, frame


def build_target_angle_payload(media_path: str, metrics: List[CandidateFaceMetric],
                               selector: Optional[AnglePortfolioSelector] = None,
                               embed_fn: Optional[Callable] = app_embedding,
                               cache_root: Optional[str] = None,
                               jpeg_quality: int = 92,
                               overrides: Optional[Dict[AngleBin, CandidateFaceMetric]] = None,
                               inline_images: bool = True) -> Dict[str, Any]:
    """Select, export and fuse; returns the React payload.

    Writes ``<cache_root>/<key>/bin_<n>.jpg``, ``fused_embedding.npy`` and
    ``manifest.json``; the key covers the file's identity and the exact
    selection, so a repeat call with the same selection is served from disk
    without decoding. ``cache_root`` defaults to ``app/output/cache/target_angles``
    (``ROOP_TARGET_ANGLE_CACHE`` overrides). ``inline_images=False`` leaves the
    base64 ``image`` fields out (a caller serving the files by URL).
    """
    selector = selector or AnglePortfolioSelector()
    portfolio = selector.select_portfolio(metrics, overrides=overrides)
    root = cache_root or os.environ.get("ROOP_TARGET_ANGLE_CACHE") or DEFAULT_CACHE_ROOT
    key = _media_key(media_path, portfolio)
    out_dir = os.path.join(root, key)
    manifest_path = os.path.join(out_dir, "manifest.json")

    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                payload = json.load(fh)
            for entry in payload["bins"]:
                if entry.get("file"):
                    with open(os.path.join(out_dir, entry["file"]), "rb") as fh:
                        data = fh.read()
                    if inline_images:
                        entry["image"] = _dataurl(data)
            payload["cached"] = True
            return payload
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("angle_portfolio: cache %s unreadable (%s); rebuilding", out_dir, exc)

    os.makedirs(out_dir, exist_ok=True)
    by_frame: Dict[int, List[AngleBin]] = {}
    for b, m in portfolio.items():
        if m.kps is not None:
            by_frame.setdefault(m.frame_idx, []).append(b)
    images: Dict[AngleBin, bytes] = {}
    embeddings: Dict[int, np.ndarray] = {}
    embed_failures = 0
    for idx, frame in _decode_frames(media_path, by_frame):
        for b in by_frame[idx]:
            m = portfolio[b]
            ok, buf = cv2.imencode(".jpg", aligned_crop(frame, m.kps),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if ok:
                images[b] = buf.tobytes()
            if b in FUSION_BINS and embed_fn is not None:
                try:
                    emb = embed_fn(frame, m.kps)
                except Exception as exc:  # one failed embedding must not lose the export
                    _swallowed("roop/angle_portfolio.py:embed_fn", exc, "fused without this frame")
                    emb = None
                if emb is None:
                    embed_failures += 1
                else:
                    embeddings[m.frame_idx] = emb
        del frame

    fused = None
    fused_error = None
    try:
        fused = synthesize_fused_embedding(portfolio, embeddings)
        np.save(os.path.join(out_dir, "fused_embedding.npy"), fused)
    except ValueError as exc:
        fused_error = str(exc)

    bins = []
    for b in AngleBin:
        slot = selector.report.get(b, BinSlot("missing"))
        entry: Dict[str, Any] = {"bin": b.name, "index": int(b), "label": b.label, **asdict(slot)}
        m = portfolio.get(b)
        if m is not None:
            entry.update({
                "frame_idx": m.frame_idx, "track_id": m.track_id,
                # pitch = as measured; pitch_up = relative to the person's
                # neutral (what the bins read), up-positive.
                "yaw": m.yaw, "pitch": m.pitch,
                "pitch_up": None if m.pitch is None else pitch_up(selector.relative_pitch(m)),
                "pitch_neutral": None if m.pitch is None else round(
                    selector.neutral.get(m.track_id, selector.neutral.get(None, 0.0)), 2),
                "roll": m.roll, "composite_score": m.composite_score, "sharpness": m.sharpness,
                "id_similarity": m.id_similarity, "ear": m.ear,
            })
            if b in images:
                name = f"bin_{int(b)}.jpg"
                with open(os.path.join(out_dir, name), "wb") as fh:
                    fh.write(images[b])
                entry["file"] = name
            else:
                entry["file"] = None
                entry["export_error"] = "frame not decodable"
        bins.append(entry)

    payload = {
        "media_path": media_path,
        "cache_key": key,
        "cache_dir": out_dir,
        "coverage": {"selected": sum(1 for s in selector.report.values() if s.status == "selected"),
                     "nearest": sum(1 for s in selector.report.values() if s.status == "nearest"),
                     "missing": sum(1 for s in selector.report.values() if s.status == "missing"),
                     "total": len(AngleBin)},
        "fused_embedding": {"available": fused is not None,
                            "file": "fused_embedding.npy" if fused is not None else None,
                            "dim": None if fused is None else int(fused.size),
                            "sources": [b.name for b in FUSION_BINS
                                        if b in portfolio and portfolio[b].frame_idx in embeddings],
                            "embed_failures": embed_failures,
                            "error": fused_error},
        "bins": bins,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    for entry in bins:
        if entry.get("file") and inline_images:
            entry["image"] = _dataurl(images[AngleBin(entry["index"])])
    payload["cached"] = False
    return payload


__all__ = ["AngleBin", "AnglePortfolioSelector", "BinSlot", "FUSION_BINS", "angle_bin", "neutral_pitch",
           "distance_to_bin", "pitch_up", "synthesize_fused_embedding", "app_embedding",
           "aligned_crop", "build_target_angle_payload", "shortlist_candidates",
           "DEFAULT_CACHE_ROOT"]
