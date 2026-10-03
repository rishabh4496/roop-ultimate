"""Live pipeline monitor: FPS and queue state every N frames, with a verdict.

The render pipeline is already decoupled -- ffmpeg decode and NVENC encode run in
child processes (outside the GIL), a reader thread and a writer thread bound the
workers with queues -- but until now nothing said, WHILE it ran, whether the GPU
side was waiting on decode or the encoder was holding it back. The scheduler's
bottleneck verdict only printed once, at the end, and only with diagnostics on.

This logs one line per window of ``every`` finished frames (default 100,
``ROOP_PIPELINE_LOG_EVERY``; 0 turns it off):

    [Pipeline] frames 300 | 7.31 fps (avg 7.29) | in 14.2/20 (empty 0%) | out 0.3/20 (full 0%) | OK | 1.20 faces/frame, 8.8 faces/s

and a closing summary. The trailing faces figure appears once the renderer reports
how many faces it painted into a finished frame (``frame_done(work=n)``). It is there
because frames/s alone misreads a render: the cost of a frame is its face count, so
fps falls whenever the footage gets busier even though faces/s -- the real
throughput -- stays flat (measured 2026-10-04 on a 54,714-frame render: 33 fps at 0.54
faces/frame, 17 fps at 1.2, both ~18-21 faces/s, reproduced from a fresh process).
Queue state is SAMPLED on every finished frame rather than
read once at the tick, because a single instantaneous read of a queue that
oscillates says nothing; the verdict is the share of samples in the window where
the input side was empty (workers had nothing to do: decode-starved) or the
output side was full (the writer was not draining: encoder-bound).

An input queue that is FULL is not a problem -- a GPU-bound pipeline should have a
reader that is ahead of it -- so only emptiness of the input and fullness of the
output are treated as faults.
"""

from __future__ import annotations
from roop.degrade import swallowed as _swallowed

import os
import threading
import time
from typing import Callable, Dict, Optional, Tuple

# Share of samples in a window at which a side is called out.
STARVE_SHARE = 0.25
# "Full" for a multi-queue side: at least this fraction of its total capacity.
FULL_FRACTION = 0.90

# 'input' / 'output' -> (depth, capacity); an optional 'unit' (str) names what one
# queue slot holds when it is not a single frame (the stabilizer queues hold chunks).
QueueState = Dict[str, object]


def log_every_from_env(default: int = 100) -> int:
    try:
        return max(0, int(os.environ.get('ROOP_PIPELINE_LOG_EVERY', '') or default))
    except ValueError:
        return default


class PipelineMonitor:

    def __init__(self, every: int, state: Callable[[], QueueState],
                 emit: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.perf_counter):
        self.every = max(0, int(every))
        self._state = state
        self._emit = emit
        self._clock = clock
        self._lock = threading.Lock()
        self._sampling = True
        self._unit = ''
        self.frames = 0
        self._t0: Optional[float] = None
        self._reset_window(None)
        self._tot_samples = 0
        self._tot_starved = 0
        self._tot_backed_up = 0
        self._tot_work = 0       # faces painted into finished frames, whole run
        self.lines = []          # every emitted line, for tests and diagnostics

    # -- window bookkeeping -------------------------------------------------
    def _reset_window(self, now):
        self._w_t0 = now
        self._w_frames = 0
        self._w_work = 0
        self._w_samples = 0
        self._w_in_empty = 0
        self._w_out_full = 0
        self._w_in_depth = 0.0
        self._w_out_depth = 0.0
        self._in_cap = 0
        self._out_cap = 0

    def start(self):
        with self._lock:
            if self._t0 is None:
                self._t0 = self._clock()
                self._reset_window(self._t0)

    def _sample(self):
        if not self._sampling:
            return
        try:
            state = self._state() or {}
            d_in, c_in = state.get('input', (0, 0))
            d_out, c_out = state.get('output', (0, 0))
            self._unit = str(state.get('unit', '') or '')
        except Exception as _degrade_error:
            # A monitor must never be able to stop a render.
            _swallowed("roop/pipeline_monitor.py:sample", _degrade_error,
                       "monitor sampling disabled")
            self._sampling = False
            return
        self._w_samples += 1
        self._w_in_depth += d_in
        self._w_out_depth += d_out
        self._in_cap, self._out_cap = c_in, c_out
        if c_in > 0 and d_in <= 0:
            self._w_in_empty += 1
        if c_out > 0 and d_out >= FULL_FRACTION * c_out:
            self._w_out_full += 1

    # -- the hook -----------------------------------------------------------
    def frame_done(self, work: int = 0):
        """Call once per finished frame. Safe from any thread.

        ``work`` is the number of faces painted into that frame (0 when the path
        cannot say); it only feeds the faces/frame and faces/s figures."""
        if self.every <= 0:
            return
        with self._lock:
            if self._t0 is None:
                self._t0 = self._clock()
                self._reset_window(self._t0)
            self.frames += 1
            self._w_frames += 1
            if work > 0:
                self._w_work += work
                self._tot_work += work
            self._sample()
            if self._w_frames >= self.every:
                self._tick()

    def _verdict(self):
        if self._w_samples == 0 or not self._sampling:
            return 'queues n/a'
        starved = self._w_in_empty / self._w_samples
        backed_up = self._w_out_full / self._w_samples
        if starved >= STARVE_SHARE and starved >= backed_up:
            return 'DECODE-STARVED: workers waited on input'
        if backed_up >= STARVE_SHARE:
            return 'ENCODER-BOUND: output queue backed up'
        return 'OK'

    def _tick(self):
        now = self._clock()
        win = max(1e-9, now - self._w_t0)
        total = max(1e-9, now - self._t0)
        n = max(1, self._w_samples)
        if self._w_samples and self._sampling:
            unit = (' ' + self._unit) if self._unit else ''
            queues = ('in %.1f/%d%s (empty %.0f%%) | out %.1f/%d%s (full %.0f%%)' % (
                self._w_in_depth / n, self._in_cap, unit, 100.0 * self._w_in_empty / n,
                self._w_out_depth / n, self._out_cap, unit, 100.0 * self._w_out_full / n))
        else:
            queues = 'queues n/a'
        line = '[Pipeline] frames %d | %.2f fps (avg %.2f) | %s | %s' % (
            self.frames, self._w_frames / win, self.frames / total, queues,
            self._verdict())
        if self._tot_work > 0:
            line += ' | %.2f faces/frame, %.1f faces/s' % (
                self._w_work / max(1, self._w_frames), self._w_work / win)
        self._tot_samples += self._w_samples
        self._tot_starved += self._w_in_empty
        self._tot_backed_up += self._w_out_full
        self._emit_line(line)
        self._reset_window(now)

    def _emit_line(self, line):
        self.lines.append(line)
        try:
            self._emit(line)
        except Exception as _degrade_error:
            _swallowed("roop/pipeline_monitor.py:emit", _degrade_error,
                       "monitor line dropped")

    def finish(self):
        """Closing summary over the whole run; returns it (None if nothing ran)."""
        if self.every <= 0:
            return None
        with self._lock:
            if self._t0 is None or self.frames == 0:
                return None
            now = self._clock()
            # Fold in the partial window the last tick did not cover.
            samples = self._tot_samples + self._w_samples
            starved = self._tot_starved + self._w_in_empty
            backed_up = self._tot_backed_up + self._w_out_full
            elapsed = max(1e-9, now - self._t0)
            if samples and self._sampling:
                tail = 'decode-starved %.0f%% | encoder-bound %.0f%% of samples' % (
                    100.0 * starved / samples, 100.0 * backed_up / samples)
            else:
                tail = 'queues n/a'
            line = '[Pipeline] done: %d frames in %.1fs = %.2f fps | %s' % (
                self.frames, elapsed, self.frames / elapsed, tail)
            if self._tot_work > 0:
                line += ' | %.2f faces/frame, %.1f faces/s' % (
                    self._tot_work / max(1, self.frames), self._tot_work / elapsed)
            self._emit_line(line)
            return line
