"""REST + WebSocket view of `roop.pipeline_status`.

    GET /api/v1/pipeline/status          one snapshot (cheap; poll it while a request is slow)
    WS  /ws/v1/pipeline/status           a snapshot on connect, then one per change

A render's first minutes (engine build, canary) have no frame counter, so the React client's 15 s
request deadline can fire on a healthy backend. This is the channel that says "compiling", not
"dead". The socket SAMPLES the registry's sequence number (the pipeline runs on plain threads and
never awaits), so an idle backend sends nothing after the first frame.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from roop import pipeline_status

router = APIRouter()

SAMPLE_INTERVAL_S = 0.25


@router.get("/api/v1/pipeline/status")
def pipeline_status_snapshot(events: int = 50):
    return pipeline_status.snapshot(events=max(0, min(int(events), 200)))


@router.get("/api/v1/pipeline/dashboard")
def pipeline_dashboard_snapshot():
    """The terminal dashboard's snapshot WITH the swap profile: the same dict the console paints from and
    `/ws/telemetry` carries under `dashboard` (which leaves the profile out - it is constant per job)."""
    from roop import terminal_dashboard
    return terminal_dashboard.snapshot()


@router.websocket("/ws/v1/pipeline/status")
async def pipeline_status_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    last = -1
    try:
        while True:
            current = pipeline_status.seq()
            if current != last:
                last = current
                await websocket.send_json(pipeline_status.snapshot())
            await asyncio.sleep(SAMPLE_INTERVAL_S)
    except (WebSocketDisconnect, RuntimeError):
        return
