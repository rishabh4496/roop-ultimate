# SPDX-License-Identifier: MIT
"""Timeline state: keyframed render parameters and face-track bindings.

This is the one place a React timeline's automation lives on the backend. The same ``evaluate_keyframes`` answer is
used by the live preview (``/api/preview/frame``) and by every render worker (``procmgr_masking.paste_upscale``), so
a scrubbed frame and the frame the render later writes were produced from the same numbers.

FRAME CONVENTION (read this before wiring a client): every frame number in this module - a keyframe's ``frame``, a
binding's ``active_ranges``, the ``frame_idx`` a caller evaluates at - is a **0-based ABSOLUTE decoder frame** of the
target media, the same numbering as ``TrajectoryPlan`` rows and ``X-Frame-Index``. It is NOT the 1-based UI frame
the older ``/api/preview`` ``frame`` field and the roto polygons use (that is ``frame_idx + 1``). Ranges are inclusive
on both ends. A render converts its clip-relative index with ``roto_frame_base - 1`` (the clip's ``frame_start``).

PARAMETERS. Each keyframe keys any subset of five numbers; a parameter nobody keys is left exactly as the run's own
settings say (``evaluate_keyframes`` returns only keyed parameters), so a timeline with no keyframes is a no-op and the
render is byte-for-byte what it was before this module existed.

    swap_opacity          0..1   alpha scale of the pasted face over the original plate (1 = unchanged)
    mask_feather          0..100 the inward feather, in the units of the ``face_mask_blend`` slider (absolute)
    yaw_erosion_bias     -1..1   scales the yaw-driven far-side inset of the semantic matte: factor = 1 + bias
                                 (0 = unchanged, -1 = none, +1 = double). Only the semantic matte has that inset.
    color_match_strength  0..1   strength of the skin-class Lab match (absolute, replaces ``skin_lab_match``)
    occlusion_sensitivity 0..2   scales the residual-occluder thresholds (1 = unchanged, 0 = residual detector off).
                                 Label-based occluders (hair/hat/cloth, from the parse) are not scaled.

INTERPOLATION belongs to the LEFT keyframe of a segment and describes how the value travels to the NEXT keyframe of
that same parameter: ``linear``, ``ease_in_out`` (smoothstep) or ``hold`` (the left value until the next key). Before
the first key the first value holds; after the last key the last value holds.

BINDINGS pin a track to a faceset. ``locked`` bindings override the pre-pass's appearance match; unlocked ones only
fill a track the matcher left without a source. ``active_ranges`` limit where the track is swapped at all (empty = the
whole clip). A binding's ``faceset_path`` is matched against the run's loaded gallery - it is never opened from disk.
"""
from __future__ import annotations

import base64
import binascii
import bisect
import math
import os
import threading
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = 1
INTERPOLATIONS = ("linear", "ease_in_out", "hold")

# name -> (lo, hi). Every keyed value is validated against its own range; nothing is silently clamped.
PARAMETER_RANGES: Dict[str, Tuple[float, float]] = {
    "swap_opacity": (0.0, 1.0),
    "mask_feather": (0.0, 100.0),
    "yaw_erosion_bias": (-1.0, 1.0),
    "color_match_strength": (0.0, 1.0),
    "occlusion_sensitivity": (0.0, 2.0),
}
PARAMETERS = tuple(PARAMETER_RANGES)

MAX_KEYFRAMES = 20000
MAX_BINDINGS = 256
MAX_RANGES = 4096
MAX_THUMBNAIL_CHARS = 64 * 1024       # a 96x96 JPEG is ~5 KB; this keeps a checkpoint from carrying megabytes of art


class TimelineError(ValueError):
    """A timeline payload that cannot be used. The message names the field, for a 422 body."""


def _int(value: Any, what: str, lo: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or isinstance(value, float) and not value.is_integer():
        raise TimelineError(f"{what} must be an integer, got {value!r}")
    value = int(value)
    if value < lo:
        raise TimelineError(f"{what} must be >= {lo}, got {value}")
    return value


def _float(value: Any, what: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimelineError(f"{what} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise TimelineError(f"{what} must be finite")
    if not lo <= value <= hi:
        raise TimelineError(f"{what} must be within [{lo:g}, {hi:g}], got {value:g}")
    return value


# ── schema ───────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ParameterKeyframe:
    frame: int
    interpolation: str = "linear"
    swap_opacity: Optional[float] = None
    mask_feather: Optional[float] = None
    yaw_erosion_bias: Optional[float] = None
    color_match_strength: Optional[float] = None
    occlusion_sensitivity: Optional[float] = None

    def values(self) -> Dict[str, float]:
        return {n: getattr(self, n) for n in PARAMETERS if getattr(self, n) is not None}

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"frame": self.frame, "interpolation": self.interpolation}
        out.update(self.values())
        return out

    @classmethod
    def from_dict(cls, data: Any, where: str = "keyframe") -> "ParameterKeyframe":
        if isinstance(data, cls):
            return data
        if not isinstance(data, Mapping):
            raise TimelineError(f"{where} must be an object")
        known = {"frame", "interpolation", *PARAMETERS}
        extra = sorted(set(data) - known)
        if extra:
            raise TimelineError(f"{where} has unknown field(s): {', '.join(map(str, extra))}")
        if "frame" not in data:
            raise TimelineError(f"{where} needs a frame")
        interpolation = data.get("interpolation", "linear")
        if interpolation not in INTERPOLATIONS:
            raise TimelineError(f"{where}.interpolation must be one of {', '.join(INTERPOLATIONS)}, got {interpolation!r}")
        values: Dict[str, Optional[float]] = {}
        for name, (lo, hi) in PARAMETER_RANGES.items():
            raw = data.get(name)
            values[name] = None if raw is None else _float(raw, f"{where}.{name}", lo, hi)
        return cls(frame=_int(data["frame"], f"{where}.frame"), interpolation=interpolation, **values)


def _normalise_ranges(raw: Any, where: str) -> Tuple[Tuple[int, int], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise TimelineError(f"{where} must be a list of [start_frame, end_frame] pairs")
    if len(raw) > MAX_RANGES:
        raise TimelineError(f"{where} has {len(raw)} ranges (max {MAX_RANGES})")
    pairs: List[Tuple[int, int]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TimelineError(f"{where}[{i}] must be [start_frame, end_frame]")
        start, end = _int(item[0], f"{where}[{i}][0]"), _int(item[1], f"{where}[{i}][1]")
        if end < start:
            raise TimelineError(f"{where}[{i}] ends before it starts ({start} > {end})")
        pairs.append((start, end))
    pairs.sort()
    merged: List[List[int]] = []
    for s, e in pairs:
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return tuple((s, e) for s, e in merged)


@dataclass(frozen=True)
class FaceTrackBinding:
    track_id: int
    label: str = ""
    faceset_path: str = ""
    thumbnail_b64: str = ""
    active_ranges: Tuple[Tuple[int, int], ...] = ()
    locked: bool = False

    def is_active(self, frame: int) -> bool:
        """Empty ``active_ranges`` means the whole clip."""
        if not self.active_ranges:
            return True
        i = bisect.bisect_right(self.active_ranges, (frame, math.inf)) - 1
        return i >= 0 and self.active_ranges[i][0] <= frame <= self.active_ranges[i][1]

    def to_dict(self) -> Dict[str, Any]:
        return {"track_id": self.track_id, "label": self.label, "faceset_path": self.faceset_path,
                "thumbnail_b64": self.thumbnail_b64,
                "active_ranges": [[s, e] for s, e in self.active_ranges], "locked": self.locked}

    @classmethod
    def from_dict(cls, data: Any, where: str = "binding") -> "FaceTrackBinding":
        if isinstance(data, cls):
            return data
        if not isinstance(data, Mapping):
            raise TimelineError(f"{where} must be an object")
        known = {f.name for f in fields(cls)}
        extra = sorted(set(data) - known)
        if extra:
            raise TimelineError(f"{where} has unknown field(s): {', '.join(map(str, extra))}")
        if "track_id" not in data:
            raise TimelineError(f"{where} needs a track_id")
        label = data.get("label", "")
        path = data.get("faceset_path", "")
        thumb = data.get("thumbnail_b64", "")
        for name, val in (("label", label), ("faceset_path", path), ("thumbnail_b64", thumb)):
            if not isinstance(val, str):
                raise TimelineError(f"{where}.{name} must be a string")
        if "\x00" in path or len(path) > 1024:
            raise TimelineError(f"{where}.faceset_path is not a usable path")
        if len(label) > 200:
            raise TimelineError(f"{where}.label is longer than 200 characters")
        if thumb:
            if len(thumb) > MAX_THUMBNAIL_CHARS:
                raise TimelineError(f"{where}.thumbnail_b64 is {len(thumb)} characters (max {MAX_THUMBNAIL_CHARS})")
            body = thumb.split(",", 1)[1] if thumb.startswith("data:") and "," in thumb else thumb
            try:
                base64.b64decode(body, validate=True)
            except (binascii.Error, ValueError):
                raise TimelineError(f"{where}.thumbnail_b64 is not valid base64") from None
        locked = data.get("locked", False)
        if not isinstance(locked, bool):
            raise TimelineError(f"{where}.locked must be true or false")
        return cls(track_id=_int(data["track_id"], f"{where}.track_id", lo=-(2 ** 31)), label=label,
                   faceset_path=path, thumbnail_b64=thumb,
                   active_ranges=_normalise_ranges(data.get("active_ranges"), f"{where}.active_ranges"),
                   locked=locked)


@dataclass(frozen=True)
class TimelineState:
    bindings: Tuple[FaceTrackBinding, ...] = ()
    keyframes: Tuple[ParameterKeyframe, ...] = ()
    version: int = SCHEMA_VERSION

    @property
    def empty(self) -> bool:
        return not self.bindings and not self.keyframes

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "bindings": [b.to_dict() for b in self.bindings],
                "keyframes": [k.to_dict() for k in self.keyframes]}

    @classmethod
    def from_dict(cls, data: Any) -> "TimelineState":
        if isinstance(data, cls):
            return data
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TimelineError("timeline must be an object")
        extra = sorted(set(data) - {"version", "bindings", "keyframes", "track_bindings"})
        if extra:
            raise TimelineError(f"timeline has unknown field(s): {', '.join(map(str, extra))}")
        version = data.get("version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise TimelineError(f"timeline version {version!r} is not supported (this server reads {SCHEMA_VERSION})")
        raw_b = data.get("bindings", data.get("track_bindings")) or []
        raw_k = data.get("keyframes") or []
        if not isinstance(raw_b, (list, tuple)) or not isinstance(raw_k, (list, tuple)):
            raise TimelineError("bindings and keyframes must be lists")
        if len(raw_b) > MAX_BINDINGS:
            raise TimelineError(f"{len(raw_b)} bindings (max {MAX_BINDINGS})")
        if len(raw_k) > MAX_KEYFRAMES:
            raise TimelineError(f"{len(raw_k)} keyframes (max {MAX_KEYFRAMES})")
        bindings = tuple(FaceTrackBinding.from_dict(b, f"bindings[{i}]") for i, b in enumerate(raw_b))
        keyframes = tuple(sorted((ParameterKeyframe.from_dict(k, f"keyframes[{i}]") for i, k in enumerate(raw_k)),
                                 key=lambda k: k.frame))
        seen = set()
        for b in bindings:
            if b.track_id in seen:
                raise TimelineError(f"track {b.track_id} is bound twice")
            seen.add(b.track_id)
        for a, b in zip(keyframes, keyframes[1:]):
            if a.frame == b.frame:
                raise TimelineError(f"two keyframes at frame {a.frame}")
        return cls(bindings=bindings, keyframes=keyframes)

    @classmethod
    def from_payload(cls, payload: Optional[Mapping[str, Any]]) -> Optional["TimelineState"]:
        """The timeline a request carries, or None when it carries none (so the caller changes nothing).

        Accepts the nested form ``{"timeline": {"keyframes", "bindings"}}`` and the flat form used by the preview
        endpoint (``keyframes`` / ``track_bindings`` at the top level). Both forms present: the nested one wins."""
        if not isinstance(payload, Mapping):
            return None
        if payload.get("timeline") is not None:
            return cls.from_dict(payload["timeline"])
        if payload.get("keyframes") is not None or payload.get("track_bindings") is not None:
            return cls.from_dict({"keyframes": payload.get("keyframes"), "bindings": payload.get("track_bindings")})
        return None


# ── evaluation ───────────────────────────────────────────────────────────────────────────────────────────
def _ease(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


class KeyframeCurves:
    """Per-parameter sorted key arrays, built once and read by every worker thread without a lock."""

    __slots__ = ("_curves",)

    def __init__(self, keyframes: Iterable[ParameterKeyframe]):
        raw: Dict[str, List[Tuple[int, float, str]]] = {n: [] for n in PARAMETERS}
        for kf in sorted((ParameterKeyframe.from_dict(k) for k in keyframes), key=lambda k: k.frame):
            for name, value in kf.values().items():
                raw[name].append((kf.frame, value, kf.interpolation))
        self._curves = {n: (tuple(f for f, _, _ in pts), tuple(v for _, v, _ in pts), tuple(i for _, _, i in pts))
                        for n, pts in raw.items() if pts}

    def __bool__(self) -> bool:
        return bool(self._curves)

    def evaluate(self, frame_idx: int) -> Dict[str, float]:
        frame = int(frame_idx)
        out: Dict[str, float] = {}
        for name, (frames, values, interps) in self._curves.items():
            i = bisect.bisect_right(frames, frame) - 1
            if i < 0:
                out[name] = values[0]
            elif frames[i] == frame or i == len(frames) - 1:
                out[name] = values[i]
            elif interps[i] == "hold":
                out[name] = values[i]
            else:
                t = (frame - frames[i]) / float(frames[i + 1] - frames[i])
                if interps[i] == "ease_in_out":
                    t = _ease(t)
                out[name] = values[i] + (values[i + 1] - values[i]) * t
        return out


def evaluate_keyframes(keyframes: Iterable[Any], frame_idx: int) -> Dict[str, float]:
    """Exact interpolated value of every KEYED parameter at ``frame_idx`` (0-based absolute frame).

    Unkeyed parameters are absent from the result: the caller keeps its own setting for them. ``keyframes`` may be
    ``ParameterKeyframe`` objects or their dicts; invalid ones raise ``TimelineError``."""
    return KeyframeCurves(keyframes).evaluate(frame_idx)


# ── bindings -> the pre-pass source map ──────────────────────────────────────────────────────────────────
def _entry_for(binding: FaceTrackBinding, entries: Sequence[Mapping[str, Any]]):
    from roop import batch_matrix as _bm
    if not binding.faceset_path:
        return None, "binding has no faceset_path"
    entry, _via, reason = _bm._match_ref(binding.faceset_path, entries)
    if entry is not None and not entry.get("faces", 1):
        return None, "that faceset is empty"
    return entry, reason


def apply_bindings_to_source_map(source_map: Mapping[Any, Any], state: Optional[TimelineState],
                                 entries: Sequence[Mapping[str, Any]]):
    """``(new_map, report)``: the pre-pass ``{track: (source, embedding)}`` with the timeline's bindings applied.

    Locked bindings replace the matcher's choice; unlocked ones only fill a track that has no source. A binding whose
    track is not in the map (a track id from another run), or whose faceset is not in this run's gallery, is left
    out of the map and says so in ``report`` - it is never guessed. The input mapping is not modified; the result has
    the same type (a frozen ``MappingProxyType`` stays frozen, which ``multi_face_tracker.locked_source`` requires)."""
    report: List[Dict[str, Any]] = []
    if state is None or not state.bindings:
        return source_map, report
    new = dict(source_map)
    for b in state.bindings:
        row = {"track_id": b.track_id, "label": b.label, "faceset_path": b.faceset_path, "locked": b.locked,
               "applied": False, "changed": False, "source_index": None, "reason": None}
        report.append(row)
        if b.track_id not in new:
            row["reason"] = "this run has no such track"
            continue
        entry, why = _entry_for(b, entries)
        if entry is None:
            row["reason"] = why or "faceset not in this run's gallery"
            continue
        current = new[b.track_id]
        current_src = current[0] if isinstance(current, (tuple, list)) and current else current
        emb = current[1] if isinstance(current, (tuple, list)) and len(current) > 1 else None
        row["source_index"] = int(entry["index"])
        if current_src is not None and not b.locked:
            row["reason"] = "the matcher already chose source %s and this binding is not locked" % current_src
            continue
        new[b.track_id] = (int(entry["index"]), emb)
        row["applied"] = True
        row["changed"] = current_src is None or int(current_src) != int(entry["index"])
    out = MappingProxyType(new) if isinstance(source_map, MappingProxyType) else new
    return out, report


def bindings_fingerprint(state: Optional[TimelineState] = None) -> List[List[Any]]:
    """What of the bindings decides the track -> source map, as JSON-comparable lists (``active_ranges`` do not: they
    gate pasting per frame and never change the assignment). Stored in the stage cache meta and compared on restore."""
    state = _active if state is None else state
    if state is None:
        return []
    return [[b.track_id, b.faceset_path, b.locked] for b in sorted(state.bindings, key=lambda b: b.track_id)]


# ── per-frame gating ─────────────────────────────────────────────────────────────────────────────────────
def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _face_field(face: Any, key: str):
    if isinstance(face, Mapping):
        return face.get(key)
    return getattr(face, key, None)


def track_of(face: Any, absolute_frame: Optional[int] = None) -> Optional[int]:
    """The track a face belongs to: its own ``_track_id`` (a render), else the best-overlapping row the preview
    registered for this frame (a single-frame preview has no tracker, only the sealed plan's rows)."""
    tid = _face_field(face, "_track_id")
    if tid is not None:
        try:
            return int(tid)
        except (TypeError, ValueError):
            return None
    rows = _preview_tracks
    box = _face_field(face, "bbox")
    if not rows or box is None:
        return None
    try:
        box = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    best, best_iou = None, 0.3
    for row_box, row_tid in rows:
        v = _iou(box, row_box)
        if v > best_iou:
            best, best_iou = row_tid, v
    return best


def gate_pending(pending: List[Tuple[Any, Any]], absolute_frame: int) -> Tuple[List[Tuple[Any, Any]], List[int]]:
    """Drop the (source, face) pairs whose track is bound but not active at ``absolute_frame``.

    Returns ``(kept, dropped_track_ids)``. A face with no resolvable track, or a track with no binding, is kept: a
    binding restricts only what it names."""
    state = _active
    if state is None or not state.bindings:
        return pending, []
    by_track = {b.track_id: b for b in state.bindings if b.active_ranges}
    if not by_track:
        return pending, []
    kept, dropped = [], []
    for item in pending:
        tid = track_of(item[1], absolute_frame)
        b = by_track.get(tid) if tid is not None else None
        if b is not None and not b.is_active(absolute_frame):
            dropped.append(tid)
        else:
            kept.append(item)
    return kept, dropped


# ── the active timeline (immutable once set, so workers read it without a lock) ───────────────────────────
_lock = threading.Lock()
_active: Optional[TimelineState] = None
_curves: Optional[KeyframeCurves] = None
_preview_tracks: Tuple[Tuple[Tuple[float, ...], int], ...] = ()


def set_active(state: Optional[TimelineState]) -> None:
    global _active, _curves
    with _lock:
        if state is None or state.empty:
            _active, _curves = None, None
        else:
            _active = state
            curves = KeyframeCurves(state.keyframes)
            _curves = curves if curves else None


def active() -> Optional[TimelineState]:
    return _active


def clear_active() -> None:
    set_active(None)
    set_preview_tracks(())


def set_preview_tracks(rows: Iterable[Tuple[Sequence[float], int]]) -> None:
    """``[(bbox_xyxy_px, track_id)]`` for the frame a preview is drawing (from the sealed trajectory plan)."""
    global _preview_tracks
    _preview_tracks = tuple((tuple(float(v) for v in box), int(tid)) for box, tid in rows)


def params_at(absolute_frame: Optional[int]) -> Dict[str, float]:
    """The active timeline's keyed parameters at a 0-based absolute frame; ``{}`` when nothing is keyed."""
    curves = _curves
    if curves is None or absolute_frame is None:
        return {}
    return curves.evaluate(int(absolute_frame))


# ── what a preview wants to know about the faces the paste actually saw ──────────────────────────────────
_report_lock = threading.Lock()
_report: Optional[Dict[str, Any]] = None


def begin_report(*, frame_channels: Sequence[str] = ()) -> None:
    """Start collecting per-face facts from ``paste_upscale``. ``frame_channels`` names the frame-space overlays the
    paste should also build (``"bisenet"``, ``"occlusion"``); they cost a warp each, so only a preview that asks pays."""
    global _report
    with _report_lock:
        _report = {"faces": [], "channels": {}, "want": tuple(frame_channels)}


def reporting() -> bool:
    return _report is not None


def wants(channel: str) -> bool:
    rep = _report
    return rep is not None and channel in rep["want"]


def report_face(**fields_: Any) -> None:
    with _report_lock:
        if _report is not None:
            _report["faces"].append(fields_)


def report_channel(name: str, plane) -> None:
    """Max-accumulate a frame-space float32 plane (0..1, or a label-colour BGR image for ``bisenet``)."""
    import numpy as np
    with _report_lock:
        if _report is None:
            return
        have = _report["channels"].get(name)
        _report["channels"][name] = plane if have is None or have.shape != plane.shape else np.maximum(have, plane)


def end_report() -> Dict[str, Any]:
    global _report
    with _report_lock:
        rep, _report = _report, None
    return rep or {"faces": [], "channels": {}, "want": ()}
