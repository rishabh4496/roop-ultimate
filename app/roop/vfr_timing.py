"""Variable-frame-rate sources: detect them, and keep every frame at its own presentation time through the render.

The render path is constant-rate by construction: frames travel one per source frame down a raw pipe and the encoder
stamps them on a uniform grid at the clip's rate, while the audio is stream-copied from the source on the source's own
clock. On a VFR clip (a phone recording: the camera stretches or drops frames with exposure and load) the two clocks
disagree by however far the real timestamps wander from the grid - measured -42.6 .. +24.0 ms per frame, audio 23 ms
short, on a mild synthetic VFR clip (20 ms steps every 37 frames, 13 ms wobble). A real recording wanders further.

Stage 1 (the source probe, before any frame is decoded) reads the packet timestamps of the source:

* ``is_vfr``: ``r_frame_rate != avg_frame_rate`` beyond 0.1 % (the brief's signal), or any frame more than 1 ms off the
  grid its average rate implies (catches a VFR file whose two rates happen to agree).
* The exact timestamps are kept as ``Fraction`` seconds - nothing is rounded until the muxer's own timescale.

At finalisation ``roop.finalize_output`` calls ``retime_intermediate``: the intermediate's video samples are re-stamped
at the source's timestamps (``roop.mp4_timing``, to 1/90000 s), and ``audio_window`` gives the audio cut the true
start/length of a trimmed range instead of ``frame / constant_fps``. Frames are not duplicated or dropped: output frame k
IS source frame ``trim_start + k`` (verified against the decoded count), so the render's frame indices, plan and
trajectory cache are untouched.

Not covered, and said so in the report: a trimmed audio cut is exact to one AAC packet (~21-23 ms) because the audio is
stream-copied; the un-trimmed case is sample-exact.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

from roop.degrade import swallowed as _swallowed

RATE_TOLERANCE = 1e-3          # r_frame_rate vs avg_frame_rate, relative
GRID_TOLERANCE_S = 0.001       # a frame this far off the average-rate grid makes a clip VFR


@dataclass(frozen=True)
class SourceTiming:
    path: str
    pts: Tuple[Fraction, ...]          # presentation time of every video frame, in presentation order, seconds
    r_frame_rate: Fraction
    avg_frame_rate: Fraction
    is_vfr: bool
    reason: str
    max_grid_deviation_s: float

    @property
    def frames(self) -> int:
        return len(self.pts)

    def summary(self) -> Dict[str, Any]:
        steps = [float(b - a) * 1000.0 for a, b in zip(self.pts, self.pts[1:])]
        return {"vfr": self.is_vfr, "reason": self.reason, "frames": self.frames,
                "r_frame_rate": str(self.r_frame_rate), "avg_frame_rate": str(self.avg_frame_rate),
                "max_grid_deviation_ms": round(self.max_grid_deviation_s * 1000.0, 3),
                "min_step_ms": round(min(steps), 3) if steps else None,
                "max_step_ms": round(max(steps), 3) if steps else None}


def _frac(text: Any) -> Fraction:
    try:
        f = Fraction(str(text))
        return f if f > 0 else Fraction(0)
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def classify(r_rate: Fraction, avg_rate: Fraction, pts: Sequence[Fraction]) -> Tuple[bool, str, float]:
    """(is_vfr, why, worst deviation of any frame from the grid the average rate implies)."""
    dev = 0.0
    if avg_rate > 0 and len(pts) > 2:
        step = 1 / avg_rate
        dev = max(abs(float(p - (pts[0] + k * step))) for k, p in enumerate(pts))
    if r_rate > 0 and avg_rate > 0 and abs(float(r_rate - avg_rate)) / float(r_rate) > RATE_TOLERANCE:
        return True, f"r_frame_rate {r_rate} != avg_frame_rate {avg_rate}", dev
    if dev > GRID_TOLERANCE_S:
        return True, f"a frame is {dev * 1000:.1f} ms off its constant-rate grid", dev
    return False, "constant frame rate", dev


_cache: Dict[Tuple[str, int, int], Optional[SourceTiming]] = {}


def probe_source_timing(path: str) -> Optional[SourceTiming]:
    """Packet timestamps of the first video stream, or None when they cannot be read. Cached per (path, size, mtime)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), st.st_size, int(st.st_mtime_ns))
    if key in _cache:
        return _cache[key]
    timing = None
    try:
        from roop.ffmpeg_path import ffprobe_binary
        out = subprocess.run(
            [ffprobe_binary(), "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate,avg_frame_rate,time_base:packet=pts,dts", "-of", "json", path],
            capture_output=True, text=True, timeout=120).stdout
        blob = json.loads(out or "{}")
        stream = (blob.get("streams") or [{}])[0]
        tb = _frac(stream.get("time_base"))
        ticks: List[int] = []
        for pkt in blob.get("packets") or []:
            v = pkt.get("pts")
            if v in (None, "N/A"):
                v = pkt.get("dts")
            if v not in (None, "N/A"):
                ticks.append(int(v))
        if tb > 0 and ticks:
            pts = tuple(t * tb for t in sorted(ticks))
            r, avg = _frac(stream.get("r_frame_rate")), _frac(stream.get("avg_frame_rate"))
            if avg == 0 and len(pts) > 1 and pts[-1] > pts[0]:
                avg = Fraction(len(pts) - 1) / (pts[-1] - pts[0])
            vfr, why, dev = classify(r, avg, pts)
            timing = SourceTiming(os.path.abspath(path), pts, r, avg, vfr, why, dev)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        _swallowed("roop/vfr_timing.py:probe", exc, "source timestamps unreadable; the render keeps constant-rate timing")
    _cache[key] = timing
    return timing


def seek_time(video_path: str, frame: int) -> Optional[float]:
    """Where an accurate input seek must aim to land on source frame ``frame`` of a variable-frame-rate clip: midway
    between that frame's timestamp and the previous one's. None = not VFR (or unknown): use ``(frame - 0.5) / fps``.

    ``frame / constant_fps`` is wrong on VFR: with the clip's average rate (800/27 = 29.63 fps) a seek to frame 67 aimed
    at 2.244 s, past that frame's 2.233 s, and the whole trimmed render started one frame late (frame 68; measured by
    stamping each frame with its number). A constant-rate clip of the same length starts on frame 67."""
    timing = probe_source_timing(video_path)
    if timing is None or not timing.is_vfr:
        return None
    k = int(frame)
    if k <= 0 or k >= timing.frames:
        return None
    return float((timing.pts[k - 1] + timing.pts[k]) / 2)


@dataclass(frozen=True)
class AudioWindow:
    start_s: float
    duration_s: Optional[float]


def _slice(timing: SourceTiming, start: Optional[int], end: Optional[int]) -> Optional[Tuple[int, int]]:
    lo = int(start or 0)
    hi = timing.frames if end is None else int(end)
    if lo < 0 or hi <= lo or hi > timing.frames:
        return None
    return lo, hi


def audio_window(video_path: str, trim_start: Optional[int], trim_end: Optional[int]) -> Optional[AudioWindow]:
    """Where a trimmed audio cut must start and how long it runs, from the TRUE timestamps. None = not a VFR clip (or
    the range does not fit the timestamps): the caller keeps its ``frame / constant_fps`` arithmetic."""
    timing = probe_source_timing(video_path)
    if timing is None or not timing.is_vfr:
        return None
    span = _slice(timing, trim_start, trim_end)
    if span is None:
        return None
    lo, hi = span
    last_step = (timing.pts[hi] - timing.pts[hi - 1]) if hi < timing.frames else (
        (timing.pts[hi - 1] - timing.pts[hi - 2]) if hi - lo > 1 else 1 / timing.avg_frame_rate)
    start = float(timing.pts[lo])
    duration = float(timing.pts[hi - 1] + last_step - timing.pts[lo]) if trim_end is not None else None
    return AudioWindow(start_s=start, duration_s=duration)


def retime_intermediate(intermediate: str, video_source: str, trim_start: Optional[int], frames: int) -> Dict[str, Any]:
    """Re-stamp the intermediate's ``frames`` video samples at the source's timestamps (in place, atomically).

    Always returns a report dict; never raises. ``retimed`` is False with a ``reason`` whenever the intermediate was
    left exactly as it was."""
    report: Dict[str, Any] = {"retimed": False, "vfr": False}
    try:
        timing = probe_source_timing(video_source)
        if timing is None:
            report["reason"] = "the source's timestamps could not be read"
            return report
        report.update(timing.summary())
        if not timing.is_vfr:
            report["reason"] = "constant frame rate: nothing to re-stamp"
            return report
        lo = int(trim_start or 0)
        if lo < 0 or lo + frames > timing.frames:
            report["reason"] = f"the range ({lo}..{lo + frames}) does not fit the source's {timing.frames} timestamps"
            return report
        from roop import mp4_timing
        window = timing.pts[lo:lo + frames]
        # A trimmed render starts its own timeline at its first frame (its audio is cut from there); an untrimmed one
        # keeps the source's start offset so the A/V relation is the source's.
        base = window[0] if lo > 0 else Fraction(0)
        targets = [p - base for p in window]
        have = mp4_timing.read_track(intermediate)[2].samples
        if have != frames:
            report["reason"] = f"the intermediate has {have} video samples, expected {frames}"
            return report
        tmp = intermediate + ".vfr.mp4"
        try:
            mp4_timing.retime(intermediate, tmp, targets)
            got = mp4_timing.presentation_seconds(tmp)
            worst = max(abs(float(g - t)) for g, t in zip(got, targets))
            if len(got) != len(targets) or worst > 1e-4:
                raise mp4_timing.Mp4Error(f"re-read timestamps differ from the targets by {worst * 1000:.4f} ms")
            os.replace(tmp, intermediate)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        report.update(retimed=True, max_error_ms=round(worst * 1000.0, 5), timescale=mp4_timing.TIMESCALE,
                      first_frame_s=float(targets[0]))
    except Exception as exc:                                  # a failed re-stamp must never fail the render
        report["reason"] = f"{type(exc).__name__}: {exc}"
        _swallowed("roop/vfr_timing.py:retime", exc, "the intermediate keeps its constant-rate timing")
    return report
