"""``/ws/telemetry``: render progress and GPU health, broadcast at 4 Hz.

Payload (one JSON object per tick)::

    {"type": "telemetry", "time": 1727500000.25, "interval_s": 0.25,
     "render": {"state": "rendering", "fps": 43.8, "frames_done": 120,
                "frames_total": 418, "progress": 0.287, "elapsed_s": 3.1, "eta_s": 6.8},
     "gpu": {"name": "NVIDIA GeForce RTX 4070", "utilization_pct": 91,
             "vram_used_mb": 7310, "vram_total_mb": 12282, "temperature_c": 64,
             "source": "nvml", "stale": false, "age_s": 0.0},
     "job": {...}}                          # the full job snapshot (older clients)

GPU numbers come from NVML (<1 ms per query on the RTX 4070), else
``nvidia-smi``, else ``null``, never invented. Every query runs on ONE
dedicated thread behind a timeout, so a stalled driver call cannot freeze the
event loop or exhaust the default thread pool: on a timeout the last good
sample is re-sent marked ``stale`` and queries back off exponentially (1 s,
2 s, ... 60 s). roop-ultimate's HUD went dark for a whole session after one
``nvidia-smi`` timeout before it learned to back off.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()

TICK_S = 0.25  # 4 Hz
QUERY_TIMEOUT_S = 0.5
MAX_BACKOFF_S = 60.0


class GpuSampler:
    """GPU statistics from NVML, else ``nvidia-smi``; ``None`` when neither works."""

    def __init__(self, device: int = 0) -> None:
        self.device = device
        self._nvml: Any = None
        self._handle: Any = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        except Exception:  # noqa: BLE001 - no NVML: nvidia-smi instead
            self._nvml = None

    @property
    def source(self) -> str:
        return "nvml" if self._nvml is not None else "nvidia-smi"

    def query(self) -> dict[str, Any] | None:
        """One blocking query (called on the sampler thread)."""
        if self._nvml is not None:
            n, h = self._nvml, self._handle
            mem = n.nvmlDeviceGetMemoryInfo(h)
            name = n.nvmlDeviceGetName(h)
            return {"name": name.decode() if isinstance(name, bytes) else name,
                    "temperature_c": int(n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)),
                    "utilization_pct": int(n.nvmlDeviceGetUtilizationRates(h).gpu),
                    "vram_used_mb": int(mem.used // 2 ** 20),
                    "vram_total_mb": int(mem.total // 2 ** 20), "source": "nvml"}
        out = subprocess.run(
            ["nvidia-smi", f"--id={self.device}", "--format=csv,noheader,nounits",
             "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total"],
            capture_output=True, text=True, timeout=2.0, check=True).stdout
        name, temp, util, used, total = [v.strip() for v in out.strip().split(",")]
        return {"name": name, "temperature_c": int(temp), "utilization_pct": int(util),
                "vram_used_mb": int(used), "vram_total_mb": int(total), "source": "nvidia-smi"}


class GuardedSampler:
    """Runs :meth:`GpuSampler.query` on a dedicated thread with timeout and back-off."""

    def __init__(self, sampler: GpuSampler | None = None, timeout: float = QUERY_TIMEOUT_S) -> None:
        self.sampler = sampler if sampler is not None else GpuSampler()
        self.timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-telemetry")
        self._pending: Future[Any] | None = None
        self._last: dict[str, Any] | None = None
        self._last_time = 0.0
        self._backoff = 0.0
        self._next_try = 0.0
        self.failures = 0

    def _stale(self, now: float) -> dict[str, Any] | None:
        if self._last is None:
            return None
        return {**self._last, "stale": True, "age_s": round(now - self._last_time, 1)}

    def _fail(self, now: float, why: str) -> dict[str, Any] | None:
        self.failures += 1
        self._backoff = min(max(self._backoff * 2, 1.0), MAX_BACKOFF_S)
        self._next_try = now + self._backoff
        logger.debug("GPU telemetry %s; next query in %.0f s", why, self._backoff)
        return self._stale(now)

    async def sample(self) -> dict[str, Any] | None:
        now = time.monotonic()
        if now < self._next_try:
            return self._stale(now)
        if self._pending is not None and not self._pending.done():
            # The previous query is still stuck on the thread: do not queue another.
            return self._fail(now, "query still stalled")
        self._pending = self._executor.submit(self.sampler.query)
        try:
            result = await asyncio.wait_for(asyncio.wrap_future(self._pending), self.timeout)
        except TimeoutError:
            return self._fail(now, f"query exceeded {self.timeout} s")
        except Exception as exc:  # noqa: BLE001 - NVML / nvidia-smi error
            return self._fail(now, f"query failed: {exc}")
        if result is None:
            return self._fail(now, "no GPU data")
        self._backoff = 0.0
        self._last, self._last_time = result, now
        return {**result, "stale": False, "age_s": 0.0}

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class TelemetryHub:
    """Samples every :data:`TICK_S` and fans the snapshot out to all subscribers."""

    def __init__(self, state: Any, interval: float = TICK_S,
                 sampler: GuardedSampler | None = None) -> None:
        self.state = state
        self.interval = interval
        self.gpu = sampler if sampler is not None else GuardedSampler()
        self._clients: set[WebSocket] = set()
        self._task: asyncio.Task[None] | None = None
        self.last: dict[str, Any] = {}

    @staticmethod
    def _render(job: Any) -> dict[str, Any] | None:
        if job is None:
            return None
        total = job.frames_total or 0
        return {"state": job.state, "fps": round(job.fps, 2), "frames_done": job.frames_done,
                "frames_total": total,
                "progress": round(job.frames_done / total, 4) if total else 0.0,
                "elapsed_s": round(job.elapsed, 2),
                "eta_s": None if job.eta is None else round(job.eta, 1)}

    async def snapshot(self) -> dict[str, Any]:
        job = self.state.job
        return {"type": "telemetry", "time": time.time(), "interval_s": self.interval,
                "render": self._render(job), "gpu": await self.gpu.sample(),
                "job": None if job is None else job.snapshot()}

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self.gpu.close()

    async def _loop(self) -> None:
        while True:
            started = time.monotonic()
            try:
                self.last = await self.snapshot()
            except Exception:
                logger.exception("telemetry snapshot failed")
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send_json(self.last)
                except Exception:  # noqa: BLE001 - client went away
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)
            await asyncio.sleep(max(0.0, self.interval - (time.monotonic() - started)))

    async def serve(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        try:
            await ws.send_json(self.last or await self.snapshot())
            while True:  # keep the socket open; clients may send pings
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            self._clients.discard(ws)


@router.websocket("/ws/telemetry")
async def telemetry(ws: WebSocket) -> None:
    await ws.app.state.hub.serve(ws)
