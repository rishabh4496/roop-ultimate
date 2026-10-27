"""What the React timeline's lanes read from the backend.

Three independent pieces, each of which answers "no data" honestly rather than inventing any:

``presence_intervals``  where each person is on screen, from the Face Bank's SAMPLED detections.
                        This is a sampled estimate (one look every ``stride`` frames), and the API
                        says so; a person who walks through a gap between two samples is not seen.
``waveform``            audio amplitude peaks for the filmstrip, built once per file in the
                        background by one ffmpeg audio decode and cached by (path, mtime, size).
``render maps``         the last run's Stage 1 face-free ranges and Stage 2 planned smart-skip reuse
                        frames, published by the pipeline AFTER those stages finish and read back by
                        ``/api/target/timeline_tracks``. Keyed to the video they came from.

Coordinates. The pipeline works in 0-based, clip-relative frame indices (0 = first frame of the
trimmed window; absolute = relative + ``frame_start``). The UI numbers frames from 1 and absolute.
Every function here that crosses that boundary says so in its name or docstring: ``ui_ranges`` and
``presence_intervals`` return UI numbering, everything that takes a ``*_idx`` takes pipeline numbering.
"""
from __future__ import annotations

import base64
import bisect
import os
import subprocess
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from roop.degrade import swallowed as _swallowed

# ── interval helpers (pipeline numbering in, UI numbering out) ───────────────

Range = Tuple[int, int]


def merge_ranges(ranges: Iterable[Sequence[int]]) -> List[Range]:
    """Sorted, merged inclusive ranges; touching ranges (end + 1 == start) merge too."""
    rows = sorted((int(min(a, b)), int(max(a, b))) for a, b in ranges)
    out: List[Range] = []
    for s, e in rows:
        if out and s <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def complement_ranges(ranges: Iterable[Sequence[int]], lo: int, hi: int) -> List[Range]:
    """The parts of [lo, hi] (inclusive) that ``ranges`` do not cover."""
    out: List[Range] = []
    cursor = lo
    for s, e in merge_ranges(ranges):
        if e < lo or s > hi:
            continue
        if s > cursor:
            out.append((cursor, s - 1))
        cursor = max(cursor, e + 1)
    if cursor <= hi:
        out.append((cursor, hi))
    return out


def ui_ranges(ranges: Iterable[Sequence[int]], frame_start: int) -> List[List[int]]:
    """Clip-relative 0-based inclusive ranges -> absolute 1-based inclusive UI ranges."""
    base = int(frame_start) + 1
    return [[int(s) + base, int(e) + base] for s, e in ranges]


def frames_to_ranges(frames: Iterable[int]) -> List[Range]:
    """Individual frame indices -> maximal runs of consecutive indices."""
    return merge_ranges((int(f), int(f)) for f in frames)


# ── presence ─────────────────────────────────────────────────────────────────

def median_stride(sample_frames: Sequence[int]) -> int:
    """Typical gap between looks, 0 when there are fewer than two."""
    s = sorted({int(f) for f in sample_frames})
    if len(s) < 2:
        return 0
    return int(np.median(np.diff(s)))


def presence_intervals(hit_frames: Iterable[int], sample_frames: Iterable[int], total_frames: int = 0,
                       frame_base: int = 0) -> List[List[int]]:
    """Where one person is on screen, from the frames they were SEEN in.

    ``hit_frames`` are the sampled frames (0-based) where the person was detected;
    ``sample_frames`` are ALL the frames that were looked at, with or without them. Two hits belong
    to one stretch only if no sampled frame lies between them, because a sample between them where
    the person was not seen is evidence they left. A stretch is then widened by half the gap to the
    neighbouring sample on each side (we only know the person was in frame at the hit, and not in
    frame at the next look, so the boundary is somewhere in between).

    Returns absolute 1-based inclusive UI ranges (``frame_base`` is the 0-based frame the sample
    indices count from; pass 0 when they are already absolute).
    """
    samples = sorted({int(f) for f in sample_frames})
    hits = sorted({int(f) for f in hit_frames})
    if not hits:
        return []
    if not samples:
        samples = hits
    pos = {f: i for i, f in enumerate(samples)}
    stride = median_stride(samples) or 1
    last_index = (int(total_frames) - 1) if total_frames and total_frames > 0 else None

    runs: List[List[int]] = []
    for h in hits:
        i = pos.get(h)
        if i is None:                                  # a hit that was never a sample: its own island
            i = bisect.bisect_left(samples, h)
        if runs and runs[-1][3] + 1 == i:
            runs[-1][1] = h
            runs[-1][3] = i
        else:
            runs.append([h, h, i, i])

    out: List[Range] = []
    for first, last, i_first, i_last in runs:
        prev_f = samples[i_first - 1] if i_first - 1 >= 0 and samples[i_first - 1] < first else None
        next_f = samples[i_last + 1] if i_last + 1 < len(samples) and samples[i_last + 1] > last else None
        start = first - ((first - prev_f) // 2 if prev_f is not None else stride // 2)
        end = last + ((next_f - last) // 2 if next_f is not None else stride // 2)
        start = max(0, start)
        if last_index is not None:
            end = min(last_index, end)
        out.append((start, max(start, end)))
    return ui_ranges(merge_ranges(out), frame_base)


# ── Stage 1 / Stage 2 maps ──────────────────────────────────────────────────

_maps_lock = threading.Lock()
_maps: Dict[str, Any] = {}


def begin_run(video: Optional[str], frame_start: int, frame_count: int) -> None:
    """A new render: forget the previous run's maps so they cannot be shown against this one."""
    with _maps_lock:
        _maps.clear()
        _maps.update(video=os.path.abspath(str(video)) if video else None, frame_start=int(frame_start or 0),
                     frame_count=int(frame_count or 0), active=None, reuse=[], ts=time.time())


def publish_active_ranges(active: Optional[Iterable[Sequence[int]]]) -> None:
    """Stage 1's verdict: the clip-relative ranges that hold a face. ``None`` = Stage 1 did not run."""
    with _maps_lock:
        if not _maps:
            return
        _maps["active"] = None if active is None else merge_ranges(active)


def reuse_frames(faces_by_frame: Any) -> List[int]:
    """Clip-relative frames whose EVERY face carries the smart-skip mark (``reuse_flow_warp``).

    A frame with one marked face and one unmarked face is not a reuse frame: the model still has to
    run on the second. Faces are dict-like (``Face`` supports ``.get``)."""
    out: List[int] = []
    for fi, faces in faces_by_frame.items():
        if faces and all(f.get("reuse_flow_warp") for f in faces):
            out.append(int(fi))
    return out


def publish_reuse_frames(frames: Iterable[int]) -> None:
    """Stage 2's PLAN: clip-relative frames whose every face is marked for smart-skip reuse.

    A plan, not an outcome - whether a marked face is actually reused is decided per face at swap
    time (``temporal_smart_skip.patch_is_reusable``) - and the API labels it that way."""
    with _maps_lock:
        if not _maps:
            return
        _maps["reuse"] = frames_to_ranges(frames)


def render_maps(video: Optional[str]) -> Dict[str, Any]:
    """The last run's maps in UI numbering, and whether they belong to ``video``."""
    with _maps_lock:
        snap = dict(_maps)
    out: Dict[str, Any] = {"video_matches": False, "range": None, "face_free": [], "reuse": [],
                           "stage1_ran": False}
    if not snap or not snap.get("video") or not video:
        return out
    if os.path.normcase(snap["video"]) != os.path.normcase(os.path.abspath(str(video))):
        return out
    count = int(snap.get("frame_count") or 0)
    start = int(snap.get("frame_start") or 0)
    out["video_matches"] = True
    if count > 0:
        out["range"] = [start + 1, start + count]
    if snap.get("active") is not None and count > 0:
        out["stage1_ran"] = True
        out["face_free"] = ui_ranges(complement_ranges(snap["active"], 0, count - 1), start)
    out["reuse"] = ui_ranges(snap.get("reuse") or [], start)
    return out


# ── waveform ────────────────────────────────────────────────────────────────

ENVELOPE_HZ = 100            # one peak per 10 ms
_DECODE_HZ = 2000            # 20 samples per envelope bin
_CHUNK_BYTES = _DECODE_HZ * 2 * 5

_env_lock = threading.Lock()
_envelopes: Dict[Tuple[str, int, int], Dict[str, Any]] = {}


def _ffmpeg() -> str:
    from roop.ffmpeg_path import ffmpeg_binary
    return ffmpeg_binary()


def envelope_from_pcm(pcm: np.ndarray) -> np.ndarray:
    """int16 mono at 2 kHz -> uint8 peaks at ``ENVELOPE_HZ``.

    Peak absolute value per 10 ms, on a square-root scale: speech sits 20-40 dB below full scale,
    and a linear scale would draw it as a flat line next to the occasional loud transient."""
    per = _DECODE_HZ // ENVELOPE_HZ
    usable = (len(pcm) // per) * per
    if usable <= 0:
        return np.zeros(0, dtype=np.uint8)
    peaks = np.abs(pcm[:usable].astype(np.int32)).reshape(-1, per).max(axis=1)
    scaled = np.sqrt(np.minimum(peaks, 32767) / 32767.0) * 255.0
    return np.round(scaled).astype(np.uint8)


def _build_envelope(path: str, key: Tuple[str, int, int]) -> None:
    state: Dict[str, Any] = {"state": "error", "env": None}
    try:
        cmd = [_ffmpeg(), "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", str(_DECODE_HZ),
               "-f", "s16le", "-"]
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=creation)
        pieces: List[np.ndarray] = []
        carry = b""
        try:
            while True:
                chunk = proc.stdout.read(_CHUNK_BYTES)
                if not chunk:
                    break
                buf = carry + chunk
                usable = (len(buf) // 2 // (_DECODE_HZ // ENVELOPE_HZ)) * (_DECODE_HZ // ENVELOPE_HZ) * 2
                carry = buf[usable:]
                if usable:
                    pieces.append(envelope_from_pcm(np.frombuffer(buf[:usable], dtype="<i2")))
        finally:
            try:
                proc.stdout.close()
            except Exception as exc:
                _swallowed("timeline_tracks.py:close_stdout", exc, "ffmpeg pipe already closed")
            proc.wait(timeout=30)
        env = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.uint8)
        # ffmpeg exits 0 with an empty stream for a file with no audio track.
        state = {"state": "ready" if len(env) else "none", "env": env}
    except Exception as exc:
        _swallowed("timeline_tracks.py:build_envelope", exc, "waveform unavailable for this file")
        state = {"state": "error", "env": None}
    with _env_lock:
        _envelopes[key] = state


def envelope_state(path: str) -> Tuple[str, Optional[np.ndarray]]:
    """('ready'|'building'|'none'|'error', envelope). Starts the background build on first ask."""
    try:
        st = os.stat(path)
    except OSError:
        return "error", None
    key = (os.path.abspath(path), int(st.st_mtime_ns), int(st.st_size))
    with _env_lock:
        cur = _envelopes.get(key)
        if cur is None:
            _envelopes[key] = {"state": "building", "env": None}
            threading.Thread(target=_build_envelope, args=(path, key), daemon=True, name="waveform-envelope").start()
            return "building", None
        # An "error" is retried on the next ask rather than cached for the life of the process.
        if cur["state"] == "error":
            _envelopes[key] = {"state": "building", "env": None}
            threading.Thread(target=_build_envelope, args=(path, key), daemon=True, name="waveform-envelope").start()
            return "building", None
        return cur["state"], cur["env"]


def slice_peaks(env: np.ndarray, t0: float, t1: float, bins: int) -> np.ndarray:
    """Max-pool the envelope over [t0, t1] seconds down to ``bins`` values (uint8)."""
    bins = int(max(1, min(bins, 8192)))
    if env is None or len(env) == 0 or t1 <= t0:
        return np.zeros(bins, dtype=np.uint8)
    i0 = int(max(0, np.floor(t0 * ENVELOPE_HZ)))
    i1 = int(min(len(env), np.ceil(t1 * ENVELOPE_HZ)))
    seg = env[i0:i1]
    if len(seg) == 0:
        return np.zeros(bins, dtype=np.uint8)
    if len(seg) <= bins:
        # Zoomed past the envelope's own resolution: stretch, do not invent detail.
        idx = np.minimum((np.arange(bins) * len(seg)) // bins, len(seg) - 1)
        return seg[idx].astype(np.uint8)
    edges = np.linspace(0, len(seg), bins + 1).astype(np.int64)[:-1]
    return np.maximum.reduceat(seg, edges).astype(np.uint8)


def peaks_b64(peaks: np.ndarray) -> str:
    return base64.b64encode(np.asarray(peaks, dtype=np.uint8).tobytes()).decode("ascii")
