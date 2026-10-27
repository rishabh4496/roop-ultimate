"""Peak device VRAM of one render, sampled by a daemon thread.

`torch.cuda.max_memory_allocated` sees only torch's allocator: the swapper, detector and enhancer run in
ONNX Runtime / TensorRT, whose memory is invisible to it (AGENTS.md, "VRAM": every byte that matters is
`torch.cuda.mem_get_info`, which sees the whole device). So this polls `mem_get_info` and records the
largest USED value (total - free), per stage, together with the used value when sampling began so a
reader can tell the render's own footprint from the desktop's.

The probe is injectable, and the default returns None until torch is already imported and a CUDA device is
present, so importing this module (or constructing the sampler under a test) never loads a GPU stack.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Callable, Dict, Optional, Tuple

from roop.degrade import swallowed as _swallowed

GIB = 1024.0 ** 3
Probe = Callable[[], Optional[Tuple[int, int]]]          # () -> (free_bytes, total_bytes) | None


def default_probe() -> Optional[Tuple[int, int]]:
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return int(free), int(total)
    except Exception as exc:                            # a probe must never take the run down
        _swallowed("roop/vram_peak.py:probe", exc, "peak VRAM is not sampled")
        return None


class PeakSampler:
    def __init__(self, probe: Probe = default_probe, stage_of: Callable[[], Optional[str]] = lambda: None,
                 interval_s: float = 0.25):
        self._probe, self._stage_of, self._interval = probe, stage_of, float(interval_s)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.total_bytes: Optional[int] = None
            self.base_used: Optional[int] = None
            self.last_used: Optional[int] = None        # the newest sample: what the terminal dashboard calls "allocated"
            self.peak_used: Optional[int] = None
            self.by_stage: Dict[str, int] = {}
            self.samples = 0

    def sample(self) -> None:
        got = self._probe()
        if got is None:
            return
        free, total = got
        used = int(total) - int(free)
        stage = self._stage_of()
        with self._lock:
            self.samples += 1
            self.last_used = used
            self.total_bytes = int(total)
            if self.base_used is None:
                self.base_used = used
            if self.peak_used is None or used > self.peak_used:
                self.peak_used = used
            if stage and used > self.by_stage.get(stage, -1):
                self.by_stage[stage] = used

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self._interval)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.reset()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vram_peak", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Take one last sample (the end of a stage is when memory is highest), then stop the thread."""
        self.sample()
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None

    def current_gb(self) -> Optional[float]:
        """Device-wide used VRAM at the newest sample (GiB), or None before one exists."""
        with self._lock:
            return None if self.last_used is None else round(self.last_used / GIB, 3)

    def report(self) -> Dict[str, Optional[float]]:
        """GiB, or None when no sample was ever taken (no CUDA device / torch not loaded): never a 0."""
        def gib(v: Optional[int]) -> Optional[float]:
            return None if v is None else round(v / GIB, 3)
        with self._lock:
            return {"peak_gb": gib(self.peak_used), "base_gb": gib(self.base_used),
                    "total_gb": gib(self.total_bytes), "samples": self.samples,
                    "by_stage_gb": {k: gib(v) for k, v in self.by_stage.items()}}
