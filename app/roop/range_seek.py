"""Frame-exact positioning of a capture at the start of each Stage 1 active range.

The Stage 2 pre-pass reads only ``[a_i, b_i]`` ranges. Its observations are stored by
frame index and consumed by the swap pass by frame index, so the frame that comes out of
the capture after "go to a_i" has to BE frame a_i - one frame late is a persistent
one-frame lag on every face of the range.

What was measured (``tests/probe_gop_seek_stage2.py``, ``docs/perf/stage2_seek/``; frame
identity = 32x18 luma thumbnail against a sequential decode of the same reader):

* Constant-frame-rate H.264 / HEVC (4 real clips, 2 synthetic), cv2 and the ffmpeg pipe:
  ``cap.set(POS_FRAMES)`` landed on the right frame 160/160 and 72/72 times, at 12-30 ms
  (cv2) / 70-88 ms (pipe) whatever the distance. Walking there costs the gap: cv2 ~0.1 ms
  per frame, the pipe ~3 ms per frame (a bgr24 copy each). The walk is only cheaper below
  ~100 frames (cv2) / ~25 frames (pipe); at 250 it is 2x and 10x SLOWER. Stage 1 coalesces
  ranges closer than MERGE_GAP = 45, so that regime barely occurs. Seek on CFR.
* Variable-frame-rate H.264: both readers land one frame LATE on 30-35% of seeks (the seek
  is a TIME, (frame - 0.5) / fps, and the timestamps do not sit on that grid). Walking from
  a seek inherits the error, so a hybrid that seeks for large gaps is not exact either.
* ``get(POS_FRAMES)`` cannot catch it: cv2 returns the REQUESTED position (0 error on every
  trial) and the pipe reader returned a constant 0.0. A "step forward while actual < wanted"
  check is therefore blind on cv2 and, on the pipe, would decode ``wanted`` frames.

So: CFR sources seek; a source whose timestamps do not sit on the frame-rate grid is
positioned by decoding from frame 0, which is exact by construction (a frame's index is how
many frames were decoded before it - the same definition the render pass uses).
"""
from __future__ import annotations

import os
import subprocess
import threading
from typing import Any, Dict, Optional, Tuple

import cv2

from roop.degrade import swallowed as _swallowed

# A time seek lands on frame k when the frame's timestamp is within half a frame of
# (k / fps); require well inside that so jitter in the other direction cannot flip it.
GRID_TOLERANCE = 0.2            # of one frame period
_PROBE_TIMEOUT_S = 120

# Sequential catch-up for small gaps on a CFR source. 0 = always seek. The measured
# crossover is ~100 frames on cv2 and ~25 on the pipe reader with a few ms at stake either
# way, so the default keeps the single code path that was verified exact.
# ROOP_RANGE_SEEK_SEQ_GAP overrides it (frames).
SEQ_GAP_DEFAULT = 0

_CFR_CACHE: Dict[Tuple[str, int, int], Optional[bool]] = {}
_CFR_LOCK = threading.Lock()


def _seq_gap_max() -> int:
    try:
        return max(0, int(os.environ.get('ROOP_RANGE_SEEK_SEQ_GAP', SEQ_GAP_DEFAULT)))
    except ValueError:
        return SEQ_GAP_DEFAULT


def timestamps_on_grid(pts, fps: float, tolerance: float = GRID_TOLERANCE) -> bool:
    """True when every presentation time sits within `tolerance` of a frame period of
    ``pts[0] + k / fps`` - the condition under which a time seek to (k - 0.5) / fps lands
    on frame k. `pts` need not be sorted (decode order has B-frames out of order)."""
    if fps is None or fps <= 0 or len(pts) < 2:
        return True
    ordered = sorted(float(p) for p in pts)
    t0 = ordered[0]
    period = 1.0 / float(fps)
    limit = tolerance * period
    return all(abs((t - t0) - k * period) <= limit for k, t in enumerate(ordered))


def _probe_pts(path: str):
    from roop.ffmpeg_path import ffprobe_binary
    cmd = [ffprobe_binary(), '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'packet=pts_time', '-of', 'csv=p=0', path]
    kw: Dict[str, Any] = {}
    if os.name == 'nt':
        kw['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S,
                          check=False, **kw)
    out = []
    for line in proc.stdout.splitlines():
        line = line.strip().rstrip(',')
        if not line or line == 'N/A':
            continue
        try:
            out.append(float(line))
        except ValueError:
            continue
    return out


def source_is_cfr(path: Optional[str], fps: float) -> Optional[bool]:
    """True: a time seek is frame-exact. False: it is not (variable frame rate).
    None: could not tell (unreadable, no ffprobe, no timestamps) - callers keep the seek,
    which is what they did before this existed. Cached per (path, mtime, size)."""
    if not path or not os.path.isfile(path):
        return None
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return None
    with _CFR_LOCK:
        if key in _CFR_CACHE:
            return _CFR_CACHE[key]
    verdict: Optional[bool]
    try:
        pts = _probe_pts(path)
        verdict = timestamps_on_grid(pts, fps) if len(pts) >= 2 else None
    except Exception as exc:                      # ffprobe missing / timed out
        _swallowed('roop/range_seek.py:source_is_cfr', exc, 'seek kept, frame rate unverified')
        verdict = None
    with _CFR_LOCK:
        _CFR_CACHE[key] = verdict
    return verdict


# Process-wide totals over every RangeSeeker, for the per-job report (stage_telemetry): `set_calls` are
# cap.set(POS_FRAMES) seeks, `grab_calls` frames advanced by cap.grab() (or the read fallback), `seek_walks`
# the forward walks that replaced a seek. The Stage 2 caller zeroes them on entry and reads them on exit.
_TOTALS = {'set_calls': 0, 'grab_calls': 0, 'seek_walks': 0}
_TOTALS_LOCK = threading.Lock()


def reset_counts() -> None:
    with _TOTALS_LOCK:
        for k in _TOTALS:
            _TOTALS[k] = 0


def counts() -> dict:
    with _TOTALS_LOCK:
        return dict(_TOTALS)


def _count(key: str, n: int = 1) -> None:
    if n:
        with _TOTALS_LOCK:
            _TOTALS[key] += n


class RangeSeeker:
    """Moves a capture to absolute frame indices and counts what it has delivered.

    ``pos`` is the index of the frame the next ``read()`` returns - tracked here, never
    asked of the capture. Always read through ``seeker.read()`` so the count stays true.

    sequential=True positions by decoding (exact on any source, cost = the distance from
    the current position; a backward move re-opens at 0 by seeking to 0, which every
    reader does exactly). Otherwise a gap of at most ``seq_gap_max`` frames is walked and
    anything longer is a ``cap.set``.
    """

    def __init__(self, cap: Any, sequential: bool = False,
                 seq_gap_max: Optional[int] = None, start: int = 0):
        self.cap = cap
        self.sequential = bool(sequential)
        self.seq_gap_max = _seq_gap_max() if seq_gap_max is None else int(seq_gap_max)
        self.pos = int(start)
        self.stats = {'seeks': 0, 'walked': 0, 'walks': 0, 'failed': 0}

    def read(self):
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.pos += 1
        return ok, frame

    def _advance(self, n: int) -> bool:
        grab = getattr(self.cap, 'grab', None)
        step = grab if callable(grab) else (lambda: self.cap.read()[0])
        walked = 0
        try:
            for _ in range(n):
                if not step():
                    return False
                self.pos += 1
                self.stats['walked'] += 1
                walked += 1
        finally:
            _count('grab_calls', walked)
        return True

    def goto(self, frame: int) -> bool:
        """Position the capture so the next read() returns `frame`. False = the stream
        ended first (the caller stops, exactly as for a failed read)."""
        frame = int(frame)
        if frame == self.pos:
            return True
        gap = frame - self.pos
        if gap < 0 and (self.sequential or frame <= self.seq_gap_max):
            # Walking forward from 0 is the exact way to a nearby frame; going backwards
            # then needs the one seek every reader does exactly.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.stats['seeks'] += 1
            _count('set_calls')
            self.pos = 0
            gap = frame
        if gap > 0 and (self.sequential or gap <= self.seq_gap_max):
            self.stats['walks'] += 1
            _count('seek_walks')
            ok = self._advance(gap)
            if not ok:
                self.stats['failed'] += 1
            return ok
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        self.stats['seeks'] += 1
        _count('set_calls')
        self.pos = frame
        return True


def seeker_for(cap: Any, source_path: Optional[str], fps: float = 0.0,
               n_ranges: int = 1, first_start: int = 0) -> RangeSeeker:
    """A RangeSeeker for `cap`, walking instead of seeking when the source is VFR.

    The frame-rate probe reads every packet header (~0.1-1 s), so it is skipped when
    nothing will be seeked: a single range that starts at 0."""
    sequential = False
    if n_ranges > 1 or first_start > 0:
        if fps <= 0:
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            except Exception as exc:
                _swallowed('roop/range_seek.py:seeker_for', exc, 'frame rate unknown, seek kept')
                fps = 0.0
        sequential = source_is_cfr(source_path, fps) is False
        if sequential:
            print(f'[Track] {os.path.basename(str(source_path))}: timestamps are not on the '
                  f'{fps:.3f} fps grid (variable frame rate) - positioning ranges by decoding '
                  f'instead of by time, so every range starts on the right frame.', flush=True)
    return RangeSeeker(cap, sequential=sequential)
