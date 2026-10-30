"""Bounded ASGI intake, explicit origins, and JSON request logging."""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from roop_3d.core.settings import Settings

log = logging.getLogger("roop3d.requests")


class IntakeMiddleware:
    def __init__(self, app: ASGIApp, settings: Settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        start = time.monotonic()
        headers = {k.lower(): v for k, v in scope.get("headers", [])}

        async def reject(code: str, message: str, status: int) -> None:
            response = JSONResponse({"error": {"code": code, "message": message, "request_id": request_id}},
                                    status_code=status, headers={"X-Request-ID": request_id})
            await response(scope, receive, send)

        origin = headers.get(b"origin", b"").decode("latin1")
        host = headers.get(b"host", b"").decode("latin1").split(":", 1)[0]
        if host and host not in {"127.0.0.1", "localhost"}:
            await reject("HOST_DENIED", "Host must identify the local service", 400)
            return
        allowed = {f"http://127.0.0.1:{self.settings.port}"}
        if self.settings.dev_cors:
            allowed.add("http://127.0.0.1:5174")
        if origin and origin not in allowed:
            await reject("ORIGIN_DENIED", "Origin is not enabled for this local service", 403)
            return
        try:
            length = int(headers[b"content-length"]) if b"content-length" in headers else None
            if length is not None and length < 0:
                raise ValueError()
        except ValueError:
            await reject("INVALID_CONTENT_LENGTH", "Content-Length must be a nonnegative integer", 400)
            return
        if length is not None and length > self.settings.max_body_bytes:
            await reject("REQUEST_TOO_LARGE", "Request exceeds ROOP3D_MAX_BODY_BYTES", 413)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(chunk) > self.settings.max_body_bytes - len(body):
                await reject("REQUEST_TOO_LARGE", "Request exceeds ROOP3D_MAX_BODY_BYTES", 413)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        if length is not None and len(body) != length:
            await reject("INVALID_CONTENT_LENGTH", "Body size does not match Content-Length", 400)
            return
        delivered = False

        async def replay() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        status = 500

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, replay, tracked_send)
        finally:
            log.info(json.dumps({"event": "request", "request_id": request_id, "method": scope["method"],
                                 "path": scope["path"], "status": status,
                                 "duration_ms": round((time.monotonic() - start) * 1000, 3)}))
