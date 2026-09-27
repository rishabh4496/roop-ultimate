"""FastAPI application: project, detection, pipeline, media and outputs.

Endpoints (JSON unless noted):

=======  ===========================  ===============================================
POST     /api/project/load            multipart ``sources`` (1-8 images) + ``target``
GET      /api/project                 current project + job
POST     /api/project/assign          ``{person_id: source_id | null}``
GET      /api/detect/faces            people found on ``frames`` sampled frames
GET      /api/options                 accepted parameter values + availability
POST     /api/pipeline/start          :class:`RenderParams` -> job
POST     /api/pipeline/stop           cancel, terminate workers, free shared memory
GET      /api/pipeline/status         job snapshot
GET      /api/outputs/{filename}      rendered file, HTTP 206 ranges, CORS
GET      /api/media/target            the target file, HTTP 206 ranges
GET      /api/thumbnails/{name}       face thumbnails (JPEG)
POST     /api/preview/frame           one frame through the GPU pipeline -> JPEG (``preview.py``)
GET      /api/preview/frame           the same, parameters in the query string
WS       /ws/telemetry                render + GPU metrics at 4 Hz (``telemetry.py``)
=======  ===========================  ===============================================

The server binds 127.0.0.1 by default. Uploads are streamed to disk with a
size cap, re-validated by decoding (not by extension alone), and every file
endpoint resolves names inside its own directory only.
"""
from __future__ import annotations

import contextlib
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from face_engine.server import preview, telemetry
from face_engine.server.processing import PRESETS, RenderParams
from face_engine.server.state import (
    IMAGE_EXTS,
    VIDEO_EXTS,
    AppState,
    ProjectError,
    ServerSettings,
    safe_name,
)

logger = logging.getLogger(__name__)

MEDIA_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska",
               ".webm": "video/webm", ".m4v": "video/mp4", ".avi": "video/x-msvideo",
               ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".webp": "image/webp", ".bmp": "image/bmp"}


def _inside(root: Path, name: str) -> Path:
    """``root/name`` if it is a file directly inside ``root``, else 404."""
    if not name or name != Path(name).name or name.startswith("."):
        raise HTTPException(404, "not found")
    path = (root / name).resolve()
    if path.parent != root.resolve() or not path.is_file():
        raise HTTPException(404, "not found")
    return path


def _save_upload(upload: UploadFile, dest_dir: Path, allowed: frozenset[str],
                 limit: int) -> Path:
    """Copy an upload to disk with a size cap. Blocking: call via ``run_in_threadpool``
    (a multi-GB copy on the event loop would stall telemetry for every client)."""
    name = safe_name(upload.filename or "upload")
    ext = Path(name).suffix.lower()
    if ext not in allowed:
        raise ProjectError(f"{upload.filename!r}: unsupported file type {ext or '(none)'}")
    path = dest_dir / name
    written = 0
    upload.file.seek(0)
    with open(path, "wb") as fh:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                fh.close()
                path.unlink(missing_ok=True)
                raise ProjectError(f"{upload.filename!r} is larger than {limit // 2**20} MB")
            fh.write(chunk)
    if written == 0:
        path.unlink(missing_ok=True)
        raise ProjectError(f"{upload.filename!r} is empty")
    return path


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings()
    state = AppState(settings)
    hub = telemetry.TelemetryHub(state)
    previews = preview.PreviewService(state)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        from face_engine.media.ipc_pool import install_cleanup_handlers

        install_cleanup_handlers()
        await hub.start()
        try:
            yield
        finally:
            await hub.stop()
            await run_in_threadpool(previews.reset)
            await run_in_threadpool(state.shutdown)

    app = FastAPI(title="face_engine", version="0.1.0", lifespan=lifespan)
    app.state.app_state = state
    app.state.hub = hub
    app.state.preview = previews
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
                       expose_headers=["Content-Range", "Accept-Ranges", "Content-Length",
                                       "X-Render-Ms", "X-Decode-Ms", "X-Process-Ms",
                                       "X-Encode-Ms", "X-Cache", "X-Faces", "X-Swapped"])

    @app.exception_handler(ProjectError)
    async def project_error(_request: Request, exc: ProjectError) -> Any:
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": str(exc)}, status_code=422)

    # ------------------------------------------------------------------ project
    @app.post("/api/project/load")
    async def load_project(sources: Annotated[list[UploadFile], File()],
                           target: Annotated[UploadFile, File()]) -> dict[str, Any]:
        if not 1 <= len(sources) <= settings.max_sources:
            raise ProjectError(f"upload 1-{settings.max_sources} source images")
        folder = state.root / "uploads" / uuid.uuid4().hex[:10]
        folder.mkdir(parents=True)
        saved = []
        for upload in sources:
            saved.append((upload.filename or "source",
                          await run_in_threadpool(_save_upload, upload, folder, IMAGE_EXTS,
                                                   settings.max_image_bytes)))
        target_ext = Path(target.filename or "").suffix.lower()
        limit = settings.max_image_bytes if target_ext in IMAGE_EXTS else settings.max_video_bytes
        target_path = await run_in_threadpool(_save_upload, target, folder,
                                              IMAGE_EXTS | VIDEO_EXTS, limit)
        await run_in_threadpool(state.load_project, saved, (target.filename or "target", target_path))
        return state.project_json()

    @app.get("/api/project")
    async def project() -> dict[str, Any]:
        return state.project_json()

    @app.post("/api/project/assign")
    async def assign(mapping: dict[str, str | None]) -> dict[str, Any]:
        return {"assignments": state.assign(mapping)}

    @app.get("/api/detect/faces")
    async def detect_faces(frames: int = 8) -> dict[str, Any]:
        if not 1 <= frames <= 64:
            raise HTTPException(422, "frames must be 1-64")
        people = await run_in_threadpool(state.detect_people, frames)
        return {"faces": [{"id": p.id, "count": p.count, "frame": p.frame, "bbox": p.bbox,
                           "score": p.score, "thumbnail_url": p.thumbnail} for p in people]}

    @app.get("/api/options")
    async def options() -> dict[str, Any]:
        from face_engine.core.execution import ExecutionEngine
        from face_engine.processors.swapper import SWAP_MODELS

        providers = ExecutionEngine.available_providers()
        schema = RenderParams.model_json_schema()["properties"]
        return {
            "params": schema,
            "defaults": RenderParams().model_dump(),
            "unavailable": {name: "no public model release" for name, spec in SWAP_MODELS.items()
                            if not spec.available},
            "providers": {"tensorrt": "TensorrtExecutionProvider" in providers,
                          "cuda": "CUDAExecutionProvider" in providers, "cpu": True},
            "presets": PRESETS,
        }

    # ------------------------------------------------------------------ pipeline
    @app.post("/api/pipeline/start")
    async def start(params: RenderParams) -> dict[str, Any]:
        try:
            job = await run_in_threadpool(state.start_job, params)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None
        return job.snapshot()

    @app.post("/api/pipeline/stop")
    async def stop() -> dict[str, Any]:
        from face_engine.media import ipc_pool

        job = await run_in_threadpool(state.stop_job)
        return {"job": None if job is None else job.snapshot(),
                "shared_memory_segments": len(ipc_pool._OWNED) + len(ipc_pool._ATTACHED)}

    @app.get("/api/pipeline/status")
    async def status() -> dict[str, Any]:
        return {"job": None if state.job is None else state.job.snapshot()}

    # ------------------------------------------------------------------ files
    @app.get("/api/outputs/{filename}")
    async def outputs(filename: str) -> FileResponse:
        path = _inside(state.outputs_dir(), filename)
        return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix.lower(),
                                                             "application/octet-stream"))

    @app.get("/api/media/target")
    async def target_media() -> FileResponse:
        if state.target is None:
            raise HTTPException(404, "no target loaded")
        return FileResponse(state.target.path,
                            media_type=MEDIA_TYPES.get(state.target.path.suffix.lower(),
                                                       "application/octet-stream"))

    @app.get("/api/thumbnails/{name}")
    async def thumbnails(name: str) -> FileResponse:
        return FileResponse(_inside(state.root / "thumbnails", name), media_type="image/jpeg")

    app.include_router(preview.router)
    app.include_router(telemetry.router)

    ui = settings.ui_dist
    if ui is not None and (ui / "index.html").is_file():
        app.mount("/", StaticFiles(directory=ui, html=True), name="ui")
    return app
