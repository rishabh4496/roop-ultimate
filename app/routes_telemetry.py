"""WebSocket telemetry: push live run state instead of polling for it.

WHY THIS EXISTS
---------------
The React client polls `GET /api/progress` once a second for the entire length
of a render. Measured on this repo (`app/tests/probe_progress_cost.py`), that
response is 3.9 KB idle and **17.4 KB mid-render**, because it carries the
250-line rolling log, the parts snapshot and the whole nested `runtime` block
on every tick. A telemetry frame carrying the numbers that actually change is
108 bytes -- **161x smaller**. Over a one-hour render the poll moves ~60 MB and
3,600 requests to deliver ~0.4 MB of new information.

Worse, the interesting number is the one the poll cannot see: a 1 Hz sample of a
counter that updates per frame shows a 24 fps render as a value that jumps in
steps, and a browser that backgrounds the tab throttles that 1 Hz to as little
as one sample a minute.

WHAT THIS IS NOT
----------------
This is deliberately NOT a second FastAPI application. `app/api.py` is already
the server: 12 routers, CORS, the settings surface and the job endpoints
(`/api/swap`, `/api/stop`, `/api/pause`, `/api/resume`). A parallel `app =
FastAPI()` would bind a second port, split that state in half and leave two
servers disagreeing about whether a job is running. This module is a router
that `api.py` includes like every other route module.

THE LOAD-BEARING CONSTRAINT
---------------------------
The swap pipeline is NOT async. It runs in a plain `threading.Thread`
(`api.py`: `threading.Thread(target=_run_swap, ...)`) and reports progress by
mutating the `_progress` dict in place. A worker thread therefore has no event
loop and CANNOT await anything. `asyncio.run_coroutine_threadsafe` against the
loop captured at startup is the only supported bridge, and it is verified
end-to-end in `app/tests/probe_ws_feasibility.py`.

That constraint is why this module SAMPLES rather than asking the pipeline to
push. Nothing in the pipeline calls into here, so there is no new coupling and
no pipeline code to change: a single asyncio task reads the same `_progress`
dict the poll reads, and broadcasts only when a value actually changed. The
pipeline stays exactly as it is.

DELIVERY GUARANTEES
-------------------
Telemetry is lossy ON PURPOSE. A slow or wedged client must never apply
backpressure to a render, so a send that would block is dropped rather than
awaited, and a client whose socket errors is disconnected. Every frame carries
the complete current state (not a delta), so a client that misses frames is
fully correct again on the next one.

`/api/progress` REMAINS the source of truth and is unchanged. This is an
accelerator, not a replacement: the client keeps polling as a fallback whenever
the socket is not connected, so a proxy that blocks WebSocket upgrades degrades
to exactly today's behaviour instead of breaking.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any, Callable, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Every broad handler here reports through this. A telemetry fault must not
# break a render, but it must not be invisible either: a socket that fails
# silently just degrades the UI to polling with nothing anywhere saying why.
from roop.degrade import swallowed

router = APIRouter()

# Sampling cadence. 4 Hz is chosen, not arbitrary: it is fast enough that an fps
# readout and a progress bar look continuous to the eye, and slow enough that it
# costs far less than the 1 Hz poll it replaces (4 x 108 B = 432 B/s against
# 1 x 17.4 KB = 17.4 KB/s, still ~40x cheaper). The sampler only emits on
# CHANGE, so an idle backend sends nothing at all.
SAMPLE_HZ = 4.0
SAMPLE_INTERVAL = 1.0 / SAMPLE_HZ

# A heartbeat is sent when nothing has changed for this long, so a client can
# tell "no news" apart from "the connection is dead" without waiting for a TCP
# timeout that may never come on a half-open socket.
HEARTBEAT_S = 15.0


class TelemetryHub:
    """Fan-out to connected clients, safe to poke from a non-async thread.

    The set of connections is only ever mutated on the event loop, so it needs
    no lock. `broadcast_threadsafe` is the door for worker threads and is the
    only method that may be called from outside the loop.
    """

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._sampler: Optional[asyncio.Task] = None
        # Diagnostics, surfaced by /api/telemetry/status so the transport can be
        # inspected without attaching a debugger to a live render.
        self.stats = {"connected": 0, "total_connections": 0, "frames_sent": 0,
                      "frames_dropped": 0, "last_send_ts": 0.0}

    # -- lifecycle ---------------------------------------------------------
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the serving loop. Called once, from a startup hook."""
        self._loop = loop

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    def client_count(self) -> int:
        return len(self._clients)

    # -- connection handling (event loop only) -----------------------------
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        self.stats["connected"] = len(self._clients)
        self.stats["total_connections"] += 1

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)
        self.stats["connected"] = len(self._clients)

    async def broadcast(self, message: dict[str, Any]) -> int:
        """Send to every client. Returns the number that received it.

        A client that raises is dropped immediately: a render must not be held
        up by a browser that stopped reading, and a half-open socket would
        otherwise accumulate sends forever.
        """
        if not self._clients:
            return 0
        payload = json.dumps(message, default=str)
        dead: list[WebSocket] = []
        sent = 0
        for client in list(self._clients):
            try:
                await client.send_text(payload)
                sent += 1
            except Exception as error:
                # Covers WebSocketDisconnect, a closed transport and any
                # serialisation/transport error. Telemetry is lossy by design;
                # losing a client is never worth raising into a render.
                swallowed('telemetry.send', error, 'dropped this client')
                dead.append(client)
        for client in dead:
            self.disconnect(client)
            self.stats["frames_dropped"] += 1
        if sent:
            self.stats["frames_sent"] += 1
            self.stats["last_send_ts"] = time.time()
        return sent

    def broadcast_threadsafe(self, message: dict[str, Any], timeout: float = 0.5) -> bool:
        """Broadcast from a thread that has no event loop.

        This is the ONLY entry point the swap pipeline's worker thread may use.
        Returns False rather than raising if the loop is gone or the send did
        not complete in `timeout` -- a telemetry failure must never surface as a
        render failure.
        """
        loop = self._loop
        if loop is None or loop.is_closed() or not self._clients:
            return False
        try:
            future = asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)
            return bool(future.result(timeout=timeout))
        except Exception as error:
            # Includes the timeout: a client that stopped reading must not
            # be able to stall the pipeline thread that is rendering.
            swallowed('telemetry.broadcast_threadsafe', error,
                      'telemetry frame dropped; the render is unaffected')
            return False

    # -- sampler -----------------------------------------------------------
    def start_sampler(self, snapshot: Callable[[], dict[str, Any]]) -> None:
        """Begin sampling `snapshot()` and broadcasting on change.

        `snapshot` is injected rather than imported so this module never has to
        import `api.py` (which imports it) -- and so the tests can drive the hub
        with a fake source.
        """
        if self._loop is None:
            return
        if self._sampler is not None and not self._sampler.done():
            return
        self._sampler = self._loop.create_task(self._sample_loop(snapshot))

    async def stop_sampler(self) -> None:
        if self._sampler is None:
            return
        self._sampler.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._sampler
        self._sampler = None

    async def _sample_loop(self, snapshot: Callable[[], dict[str, Any]]) -> None:
        last: Optional[dict[str, Any]] = None
        last_sent = 0.0
        while True:
            try:
                await asyncio.sleep(SAMPLE_INTERVAL)
                if not self._clients:
                    # Nobody attached: skip the snapshot entirely rather than
                    # doing the work and discarding it.
                    last = None
                    continue
                try:
                    current = snapshot()
                except Exception as error:
                    # A telemetry sample must never take the server down.
                    swallowed('telemetry.snapshot', error, 'skipped one sample')
                    continue
                now = time.time()
                changed = current != last
                stale = (now - last_sent) >= HEARTBEAT_S
                if not changed and not stale:
                    continue
                frame = dict(current)
                frame["event"] = "progress" if changed else "heartbeat"
                frame["ts"] = now
                await self.broadcast(frame)
                last = current
                last_sent = now
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Keep the loop alive across unexpected faults; a dead sampler
                # would silently degrade every client to the polling fallback.
                swallowed('telemetry.sampler', error, 'sampler continued')
                await asyncio.sleep(SAMPLE_INTERVAL)


hub = TelemetryHub()

# Injected by api.py at import time (see the wiring block there). Kept as a
# module attribute rather than an import so this module has no dependency on
# api.py, which would be circular.
progress_snapshot: Optional[Callable[[], dict[str, Any]]] = None


@router.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket) -> None:
    """Live run telemetry.

    Protocol, kept deliberately small:
      server -> client  {"event": "progress"|"heartbeat"|"hello", ...state}
      client -> server  "ping"  (any text is accepted and answered with a frame)

    The client never needs to send anything; the receive loop exists so a
    client CAN prove liveness through an intermediary that would otherwise time
    out an idle upgrade.
    """
    await hub.connect(websocket)
    try:
        # Send current state immediately so a client that attaches mid-render
        # renders the right thing without waiting for the first change.
        if progress_snapshot is not None:
            try:
                hello = dict(progress_snapshot())
            except Exception as error:
                swallowed('telemetry.hello', error, 'greeted with empty state')
                hello = {}
            hello["event"] = "hello"
            hello["ts"] = time.time()
            await websocket.send_text(json.dumps(hello, default=str))

        while True:
            # Any inbound text is a liveness ping. Answering keeps proxies from
            # reaping an idle tunnel and lets the client measure round-trip.
            await websocket.receive_text()
            await websocket.send_text(json.dumps({"event": "pong", "ts": time.time()}))
    except WebSocketDisconnect:
        pass
    except Exception as error:
        # Any transport fault ends this connection only.
        swallowed('telemetry.connection', error, 'closed one client')
    finally:
        hub.disconnect(websocket)


@router.get("/api/telemetry/status")
def telemetry_status() -> dict[str, Any]:
    """Transport health, for diagnostics and for the tests.

    Deliberately cheap and free of side effects: unlike /api/progress this does
    not sync pause state or append to the log, so it is safe to call at any
    frequency.
    """
    return {
        "enabled": hub.loop is not None,
        "sample_hz": SAMPLE_HZ,
        "heartbeat_s": HEARTBEAT_S,
        **hub.stats,
    }
