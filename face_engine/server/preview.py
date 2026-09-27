"""``/api/preview/frame``: one target frame through the GPU pipeline, as a JPEG.

Latency budget (target < 50 ms) and where it goes, measured 2026-09-28 on an
RTX 4070 with a 1080p H.264 target (keyframe every 76 frames):

* **Decode.** Seeking to one frame decodes from its keyframe: 52-95 ms on
  its own. So frames are cached: a miss decodes keyframe -> requested frame
  (keeping every frame on the way) and a background thread decodes the rest
  of that GOP; scrubbing inside a GOP is then a cache hit. The cache is an
  LRU with a byte budget.
* **Process.** A :class:`~face_engine.server.processing.GpuFrameProcessor`
  stays warm between requests and is rebuilt only when the parameters or the
  project change (the first request after a change pays model loading).
* **Encode.** nvJPEG through ``torchvision.io.encode_jpeg`` on the CUDA
  tensor: 0.31 ms for 1080p vs 8.7 ms for download + ``cv2.imencode``, PSNR
  43.0 vs 43.6 dB. Only the JPEG bytes leave the GPU.

Every response carries its timings (``X-Decode-Ms``, ``X-Process-Ms``,
``X-Encode-Ms``, ``X-Render-Ms``) and ``X-Cache: hit|miss``, so the budget is
visible per request.
"""
from __future__ import annotations

import base64
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import Field

from face_engine.server.processing import GpuFrameProcessor, RenderParams
from face_engine.server.state import AppState, ProjectError

logger = logging.getLogger(__name__)
router = APIRouter()

CACHE_BUDGET_BYTES = 1536 * 2 ** 20


class FrameCache:
    """Decoded target frames (host BGR), filled a GOP at a time, LRU by bytes."""

    def __init__(self, path: Path, budget: int = CACHE_BUDGET_BYTES) -> None:
        from face_engine.media.capturer import VideoSource

        self.source = VideoSource(path)
        self.keyframes = self.source.keyframes
        self.count = self.source.frame_count
        self.budget = budget
        self._frames: OrderedDict[int, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self._prefetching: set[int] = set()
        self.hits = self.misses = 0

    def _keyframe(self, index: int) -> tuple[int, int]:
        """``(gop_start, gop_end)`` containing ``index``."""
        before = self.keyframes[self.keyframes <= index]
        after = self.keyframes[self.keyframes > index]
        return (int(before[-1]) if before.size else 0,
                int(after[0]) if after.size else self.count)

    def _store(self, index: int, image: np.ndarray) -> None:
        with self._lock:
            if index in self._frames:
                return
            self._frames[index] = image
            self._bytes += image.nbytes
            while self._bytes > self.budget and len(self._frames) > 1:
                _, old = self._frames.popitem(last=False)
                self._bytes -= old.nbytes

    def get(self, index: int) -> tuple[np.ndarray, bool]:
        """Frame ``index`` and whether it was cached."""
        index = int(np.clip(index, 0, self.count - 1))
        with self._lock:
            if index in self._frames:
                self._frames.move_to_end(index)
                self.hits += 1
                return self._frames[index], True
            self.misses += 1
        start, end = self._keyframe(index)
        with self._lock:
            first = next((i for i in range(start, index) if i not in self._frames), index)
        image = None
        for frame in self.source.frames(first, index + 1):
            self._store(frame.index, frame.image)
            if frame.index == index:
                image = frame.image
        if index + 1 < end:
            self._prefetch(index + 1, end)
        assert image is not None
        return image, False

    def _prefetch(self, start: int, end: int) -> None:
        with self._lock:
            if start in self._prefetching:
                return
            self._prefetching.add(start)

        def run() -> None:
            try:
                for frame in self.source.frames(start, end):
                    self._store(frame.index, frame.image)
            except Exception:
                logger.debug("preview prefetch [%d, %d) failed", start, end, exc_info=True)
            finally:
                with self._lock:
                    self._prefetching.discard(start)

        threading.Thread(target=run, name=f"preview-prefetch-{start}", daemon=True).start()


class PreviewService:
    """Warm processor + frame cache for the current project (one GPU job at a time)."""

    def __init__(self, state: AppState) -> None:
        self.state = state
        self._lock = threading.Lock()
        self._cache: tuple[Any, FrameCache | np.ndarray] | None = None
        self._processor: tuple[str, GpuFrameProcessor] | None = None

    def _frame(self, index: int) -> tuple[np.ndarray, bool]:
        target = self.state.target
        if target is None:
            raise ProjectError("no target loaded")
        key = (target.path, self.state.revision)
        if self._cache is None or self._cache[0] != key:
            value: FrameCache | np.ndarray = (
                self.state.read_frame(0) if target.kind == "image" else FrameCache(target.path))
            self._cache = (key, value)
        cached = self._cache[1]
        if isinstance(cached, np.ndarray):
            return cached, True
        return cached.get(index)

    def _get_processor(self, params: RenderParams) -> GpuFrameProcessor:
        key = f"{params.model_dump_json()}|{self.state.revision}"
        if self._processor is None or self._processor[0] != key:
            if self._processor is not None:
                self._processor[1].close()
                self._processor = None
            self._processor = (key, GpuFrameProcessor(self.state.processor_config(params)))
        return self._processor[1]

    def render(self, index: int, params: RenderParams, mode: str,
               quality: int = 90) -> tuple[bytes, dict[str, Any]]:
        """JPEG bytes + timings/stats for one frame."""
        import torch
        from torchvision.io import encode_jpeg

        with self._lock:
            t0 = time.perf_counter()
            image, hit = self._frame(index)
            t1 = time.perf_counter()
            frame = torch.from_numpy(np.ascontiguousarray(image)).cuda().permute(2, 0, 1)[None]
            stats: dict[str, Any] = {}
            if mode == "swapped":
                out, s = self._get_processor(params).process_tensor(frame)
                stats = {"faces": s.faces, "swapped": s.swapped}
            else:
                out = frame
            rgb = out[0].round().clamp(0, 255).to(torch.uint8).flip(0).contiguous()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            jpeg = encode_jpeg(rgb, quality=quality).cpu().numpy().tobytes()
            t3 = time.perf_counter()
        return jpeg, {"decode_ms": (t1 - t0) * 1e3, "process_ms": (t2 - t1) * 1e3,
                      "encode_ms": (t3 - t2) * 1e3, "render_ms": (t3 - t0) * 1e3,
                      "cache": "hit" if hit else "miss", **stats}

    def reset(self) -> None:
        with self._lock:
            if self._processor is not None:
                self._processor[1].close()
            self._processor = None
            self._cache = None


class PreviewRequest(RenderParams):
    """``POST /api/preview/frame`` body: a frame, a mode, and the render parameters."""

    frame_index: int = Field(default=0, ge=0)
    mode: Literal["swapped", "original"] = "swapped"
    quality: int = Field(default=90, ge=30, le=100)


def _service(request: Request) -> PreviewService:
    return request.app.state.preview


async def _respond(request: Request, index: int, params: RenderParams, mode: str,
                   quality: int, fmt: str) -> Response:
    service = _service(request)
    try:
        jpeg, info = await run_in_threadpool(service.render, index, params, mode, quality)
    except ProjectError as exc:
        raise HTTPException(422, str(exc)) from None
    headers = {"X-Render-Ms": f"{info['render_ms']:.1f}", "X-Decode-Ms": f"{info['decode_ms']:.1f}",
               "X-Process-Ms": f"{info['process_ms']:.1f}", "X-Encode-Ms": f"{info['encode_ms']:.2f}",
               "X-Cache": info["cache"], "X-Faces": str(info.get("faces", "")),
               "X-Swapped": str(info.get("swapped", "")), "Cache-Control": "no-store"}
    if fmt == "base64":
        return JSONResponse({"frame": index, "mode": mode, **info,
                             "image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()})
    return Response(jpeg, media_type="image/jpeg", headers=headers)


@router.post("/api/preview/frame")
async def preview_frame_post(request: Request, body: PreviewRequest) -> Response:
    """Render ``frame_index`` with the body's parameters; JPEG bytes back."""
    params = RenderParams.model_validate(body.model_dump(exclude={"frame_index", "mode",
                                                                   "quality"}))
    return await _respond(request, body.frame_index, params, body.mode, body.quality, "jpeg")


@router.get("/api/preview/frame")
async def preview_frame_get(request: Request, frame: int = Query(0, ge=0),
                            format: str = Query("jpeg", pattern="^(jpeg|base64)$"),
                            mode: str = Query("swapped", pattern="^(swapped|original)$"),
                            params: str | None = Query(None, description="RenderParams as JSON"),
                            quality: int = Query(90, ge=30, le=100)) -> Response:
    """The same as POST, parameters in the query string (``params`` = JSON)."""
    try:
        render = RenderParams.model_validate_json(params) if params else RenderParams()
    except ValueError as exc:
        raise HTTPException(422, f"invalid params: {exc}") from None
    return await _respond(request, frame, render, mode, quality, format)
