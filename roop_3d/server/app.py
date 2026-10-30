"""Minimal local service; no model, diffusion or restricted-weight imports."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles

from roop_3d import __version__
from roop_3d.core.hardware import Hardware, probe_hardware
from roop_3d.core.integrity import isolation_status
from roop_3d.core.settings import Settings
from roop_3d.server.middleware import IntakeMiddleware


def create_app(settings: Settings | None = None, *, repo_root: Path | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    root = repo_root or Path(__file__).absolute().parents[2]
    started = time.monotonic()
    app = FastAPI(title="Roop 3D", version=__version__, docs_url=None, redoc_url=None)
    app.state.settings = settings

    def envelope(request: Request, code: str, message: str, status: int) -> JSONResponse:
        return JSONResponse({"error": {"code": code, "message": message,
                                       "request_id": getattr(request.state, "request_id", None)}}, status_code=status)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        return envelope(request, f"HTTP_{error.status_code}", str(error.detail), error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        return envelope(request, "VALIDATION_ERROR", "Request does not match the endpoint schema", 422)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception) -> JSONResponse:
        return envelope(request, "INTERNAL_ERROR", "The local service could not complete the request", 500)

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "application": "roop_3d", "version": __version__,
                "uptime_seconds": max(0.0, time.monotonic() - started), "isolation": isolation_status(root)}

    @app.get("/api/v1/system/hardware", response_model=Hardware)
    def hardware() -> Hardware:
        return probe_hardware()

    @app.get("/", response_class=HTMLResponse)
    def landing() -> str:
        return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Roop 3D</title><style>body{{background:#10141c;color:#edf2fa;font:17px system-ui;max-width:760px;margin:12vh auto;padding:24px}}a{{color:#80d3ff}}small{{color:#aab6c8}}</style>
<h1>Roop 3D</h1><p>Local development service · v{__version__}</p>
<p>Stage 01 is ready for the next implementation stage. Face replacement is not available yet.</p>
<p><a href="/api/v1/health">Health &amp; isolation</a> · <a href="/api/v1/system/hardware">Live hardware</a> · <a href="/openapi.json">API schema</a></p>
<small>API: 127.0.0.1:{settings.port} · Vite development port: 5174 · Local processing by default</small></html>"""

    ui = root / "roop_3d_ui/dist"
    if ui.is_dir():
        from roop_3d.audit.isolation import safe_path
        safe_path(root, "roop_3d_ui/dist")
        # API routes remain earlier in the route table. Static assets share the API origin.
        app.mount("/ui", StaticFiles(directory=str(ui), html=True), name="ui")
    if settings.dev_cors:
        app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5174"], allow_methods=["GET"],
                           allow_headers=["Content-Type"])
    app.add_middleware(IntakeMiddleware, settings=settings)
    return app
