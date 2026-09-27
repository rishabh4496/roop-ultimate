"""Live telemetry (``/ws/telemetry``) and single-frame preview (``/api/preview/frame``).

Telemetry: one snapshot per ``interval`` seconds, broadcast to every connected
WebSocket: render state, processing fps, per-frame latency, elapsed, ETA, and
the GPU's name, temperature, utilisation and VRAM. GPU numbers come from NVML
(<1 ms per query on the RTX 4070); if NVML is unavailable, ``nvidia-smi`` is
polled with a timeout and exponential back-off (roop-ultimate's HUD went dark
for the session after one ``nvidia-smi`` timeout until it learned to back off);
with neither, the GPU block is ``null`` rather than invented numbers.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import subprocess
import time
from typing import Any

import cv2
from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from face_engine.server.processing import RenderParams
from face_engine.server.state import AppState, ProjectError

logger = logging.getLogger(__name__)
router = APIRouter()


class GpuSampler:
    """GPU statistics via NVML, else nvidia-smi (with back-off), else None."""

    def __init__(self, device: int = 0) -> None:
        self.device = device
        self._nvml: Any = None
        self._handle: Any = None
        self._smi_backoff_until = 0.0
        self._smi_backoff = 5.0
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        except Exception:  # noqa: BLE001 - no NVML: fall back to nvidia-smi
            self._nvml = None

    def sample(self) -> dict[str, Any] | None:
        if self._nvml is not None:
            try:
                n, h = self._nvml, self._handle
                mem = n.nvmlDeviceGetMemoryInfo(h)
                name = n.nvmlDeviceGetName(h)
                return {"name": name.decode() if isinstance(name, bytes) else name,
                        "temperature_c": n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
                        "utilization_pct": n.nvmlDeviceGetUtilizationRates(h).gpu,
                        "vram_used_mb": mem.used // 2 ** 20, "vram_total_mb": mem.total // 2 ** 20,
                        "source": "nvml"}
            except Exception as exc:  # noqa: BLE001
                logger.debug("NVML query failed: %s", exc)
        return self._smi()

    def _smi(self) -> dict[str, Any] | None:
        now = time.monotonic()
        if now < self._smi_backoff_until:
            return None
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--id={self.device}", "--format=csv,noheader,nounits",
                 "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total"],
                capture_output=True, text=True, timeout=2.0, check=True).stdout
            name, temp, util, used, total = [v.strip() for v in out.strip().split(",")]
            self._smi_backoff = 5.0
            return {"name": name, "temperature_c": int(temp), "utilization_pct": int(util),
                    "vram_used_mb": int(used), "vram_total_mb": int(total), "source": "nvidia-smi"}
        except (OSError, subprocess.SubprocessError, ValueError):
            self._smi_backoff_until = now + self._smi_backoff
            self._smi_backoff = min(self._smi_backoff * 2, 300.0)
            return None


class TelemetryHub:
    """Samples once per interval and fans out to all subscribers."""

    def __init__(self, state: AppState, interval: float = 1.0) -> None:
        self.state = state
        self.interval = interval
        self.gpu = GpuSampler()
        self._clients: set[WebSocket] = set()
        self._task: asyncio.Task[None] | None = None
        self.last: dict[str, Any] = {}

    def snapshot(self) -> dict[str, Any]:
        job = self.state.job
        return {"type": "telemetry", "time": time.time(), "gpu": self.gpu.sample(),
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

    async def _loop(self) -> None:
        while True:
            self.last = await run_in_threadpool(self.snapshot)
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send_json(self.last)
                except Exception:  # noqa: BLE001 - client went away
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)
            await asyncio.sleep(self.interval)

    async def serve(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        try:
            await ws.send_json(self.last or await run_in_threadpool(self.snapshot))
            while True:  # keep the socket open; clients may send pings
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            self._clients.discard(ws)


@router.websocket("/ws/telemetry")
async def telemetry(ws: WebSocket) -> None:
    await ws.app.state.hub.serve(ws)


@router.get("/api/preview/frame")
async def preview_frame(request: Request, frame: int = Query(0, ge=0),
                        format: str = Query("jpeg", pattern="^(jpeg|base64)$"),
                        mode: str = Query("swapped", pattern="^(swapped|original)$"),
                        params: str | None = Query(None, description="RenderParams as JSON")) -> Response:
    """Render one frame at ``frame`` with the given (or default) parameters."""
    state: AppState = request.app.state.app_state
    try:
        render = RenderParams.model_validate_json(params) if params else RenderParams()
    except ValueError as exc:
        raise HTTPException(422, f"invalid params: {exc}") from None
    t0 = time.perf_counter()
    try:
        if mode == "original":
            image, stats = await run_in_threadpool(state.read_frame, frame), {}
        else:
            image, stats = await run_in_threadpool(state.preview, frame, render)
    except ProjectError as exc:
        raise HTTPException(422, str(exc)) from None
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(500, "JPEG encoding failed")
    ms = round((time.perf_counter() - t0) * 1000, 1)
    headers = {"X-Render-Ms": str(ms), "X-Faces": str(stats.get("faces", "")),
               "X-Swapped": str(stats.get("swapped", ""))}
    if format == "base64":
        return JSONResponse({"frame": frame, "mode": mode, "render_ms": ms, **stats,
                             "image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()})
    return Response(jpeg.tobytes(), media_type="image/jpeg", headers=headers)
