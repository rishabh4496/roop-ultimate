"""Binary WebSocket frame transport: ``/ws/frames``.

WHAT IT CARRIES
---------------
Two kinds of picture the React UI shows as moving video:

  * LIVE -- the newest frame of a running render (roop/live_preview.py). The
    client used to learn a new one existed from ``live_seq`` in a telemetry
    frame or a poll, and only THEN issue ``GET /api/live_frame``: two hops, a
    full HTTP request per picture, and an ``<img>`` decode on the main thread.
    Here the server pushes the already-encoded JPEG bytes the moment ``seq``
    moves.
  * PLAY -- timeline playback of the target clip, a sequential run of frames.
    ``GET /api/target/preview_seq`` delivers those in chunks of 16-48, so the
    buffer refills in bursts with a round trip between every chunk. A PLAY
    stream is one long sequential decode the client meters with CREDIT.

Neither is base64 and neither goes through JSON: every picture is one binary
message, a fixed header followed by the encoded bytes, which the browser hands
to ``createImageBitmap`` as an ``ArrayBuffer``.

WHAT IT IS NOT
--------------
It does NOT raise the live preview's publish rate. The render is GPU-bound and
the preview is encoded on the render's own process (3.7 ms per publish at
1080p, see live_preview.py); this transport removes client-side cost and
latency, it does not ask the pipeline for more frames. ROOP_LIVE_PREVIEW_MS
remains the one knob for that, and the watched-gating is kept: a subscribed
client counts as a viewer exactly like a poller of /api/live_frame does.

It is an ACCELERATOR, like /ws/telemetry. Every HTTP endpoint it shadows stays
and is unchanged; a client whose socket is down uses them exactly as before.

WIRE FORMAT (all little-endian)
-------------------------------
server -> client, binary::

    offset size
      0     u8   version        (1)
      1     u8   kind           1 LIVE, 2 PLAY, 3 END (stream finished)
      2     u16  flags          END: bit0 = ended on an error, not the out point
      4     u32  stream         PLAY/END: the client's stream id; LIVE: 0
      8     u32  frame          LIVE: publish seq; PLAY: 1-based frame number;
                                END: the first frame NOT sent
     12     u32  width          LIVE: SOURCE size (for the caption);
     16     u32  height         PLAY: encoded size
     20     ...  payload        JPEG bytes (empty for END)

client -> server, text (JSON), each optional::

    {"op":"live","on":true|false}
    {"op":"play","stream":7,"index":0,"start":1,"end":600,
     "width":960,"quality":82,"credit":24}
    {"op":"credit","stream":7,"n":24}
    {"op":"stop","stream":7}
    "ping"                                   -> {"event":"pong"} (text)

FLOW CONTROL
------------
A PLAY stream sends only while it holds credit, one frame per unit. The client
grants credit as its buffer drains, so a slow client (or a hidden tab) stops
the decode instead of queueing megabytes in a socket buffer -- the same "a
wedged client must never cost the server" rule the telemetry hub follows.
LIVE has no queue at all: the loop sends whatever is newest when it gets
round to it, so a slow client just sees fewer frames.

One PLAY stream per connection; a new ``play`` replaces the old one. The
decode runs in a worker thread through ``frame_source`` and so goes through the
same capture lock and frame cache every raw-frame endpoint uses.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import struct
from typing import Any, Callable, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from roop import live_preview
from roop.degrade import swallowed

router = APIRouter()

VERSION = 1
KIND_LIVE = 1
KIND_PLAY = 2
KIND_END = 3
FLAG_ERROR = 1

HEADER = struct.Struct("<BBHIIII")
assert HEADER.size == 20

# How often the LIVE loop looks at live_preview.seq(). It is an int read, so
# polling it is nothing; 50 ms bounds the added latency well under the 500 ms
# publish interval without spinning.
LIVE_POLL_S = 0.05

# Hard caps on what a client may ask for. Credit bounds the frames in flight
# (and so the memory one connection can pin in the socket buffer); width and
# quality mirror /api/target/preview_seq's clamps.
MAX_CREDIT = 240
MAX_WIDTH = 3840
QUALITY_RANGE = (30, 95)

# Injected by api.py (like routes_telemetry.progress_snapshot), so this module
# never imports api.py. Signature: (index, frame, width, quality) ->
# (jpeg_bytes, width, height) or None when the frame does not exist (past the
# end of the clip). Raises ValueError for a target that is not a video.
frame_source: Optional[Callable[[int, int, int, int], Optional[tuple]]] = None

stats = {"connected": 0, "total_connections": 0, "live_sent": 0,
         "play_sent": 0, "streams": 0}


def pack_header(kind: int, stream: int, frame: int, width: int, height: int,
                flags: int = 0) -> bytes:
    return HEADER.pack(VERSION, kind, flags & 0xFFFF, stream & 0xFFFFFFFF,
                       max(0, frame) & 0xFFFFFFFF, max(0, width) & 0xFFFFFFFF,
                       max(0, height) & 0xFFFFFFFF)


def unpack_header(message: bytes) -> dict:
    """Inverse of pack_header, for tests and diagnostics."""
    version, kind, flags, stream, frame, width, height = HEADER.unpack_from(message, 0)
    return {"version": version, "kind": kind, "flags": flags, "stream": stream,
            "frame": frame, "width": width, "height": height,
            "payload": bytes(message[HEADER.size:])}


def _int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


class _Connection:
    """One client: an optional LIVE subscription and at most one PLAY stream."""

    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        # Two tasks can send on one socket; a frame must never be interleaved
        # with another's bytes, so every send takes this.
        self.send_lock = asyncio.Lock()
        self.live_task: Optional[asyncio.Task] = None
        self.play_task: Optional[asyncio.Task] = None
        self.play_stream = 0
        self.credit = 0
        self.credit_event = asyncio.Event()

    async def send(self, data: bytes) -> None:
        async with self.send_lock:
            await self.ws.send_bytes(data)

    # ── LIVE ──────────────────────────────────────────────────────────────
    def set_live(self, on: bool) -> None:
        if on and self.live_task is None:
            self.live_task = asyncio.create_task(self._live_loop())
        elif not on and self.live_task is not None:
            self.live_task.cancel()
            self.live_task = None

    async def _live_loop(self) -> None:
        last = -1
        while True:
            # A subscribed client is a viewer. Without this the pipeline drops
            # to its idle cadence (one frame per 3 s) because nobody polls
            # /api/live_frame any more -- the socket would be "live" at 0.3 Hz.
            live_preview.note_fetch()
            seq = live_preview.seq()
            if seq != last:
                data, cur, size = live_preview.snapshot()
                if data and cur != last:
                    await self.send(pack_header(KIND_LIVE, 0, cur, size[0], size[1]) + data)
                    stats["live_sent"] += 1
                last = cur if data else seq
            await asyncio.sleep(LIVE_POLL_S)

    # ── PLAY ──────────────────────────────────────────────────────────────
    def start_play(self, msg: dict) -> None:
        self.stop_play()
        stream = _int(msg.get("stream"), 0, 0, 0xFFFFFFFF)
        index = _int(msg.get("index"), -1, -1, 1 << 20)
        start = _int(msg.get("start"), 1, 1, 1 << 30)
        end = _int(msg.get("end"), start, start, 1 << 30)
        width = _int(msg.get("width"), 960, 0, MAX_WIDTH)
        quality = _int(msg.get("quality"), 82, *QUALITY_RANGE)
        self.play_stream = stream
        self.credit = _int(msg.get("credit"), 0, 0, MAX_CREDIT)
        self.credit_event.set()
        stats["streams"] += 1
        self.play_task = asyncio.create_task(
            self._play_loop(stream, index, start, end, width, quality))

    def add_credit(self, msg: dict) -> None:
        if _int(msg.get("stream"), -1, -1, 0xFFFFFFFF) != self.play_stream:
            return  # credit for a stream that has already been replaced
        self.credit = min(MAX_CREDIT, self.credit + _int(msg.get("n"), 0, 0, MAX_CREDIT))
        self.credit_event.set()

    def stop_play(self, stream: Optional[int] = None) -> None:
        if stream is not None and stream != self.play_stream:
            return
        if self.play_task is not None:
            self.play_task.cancel()
            self.play_task = None
        self.credit = 0

    async def _play_loop(self, stream: int, index: int, start: int, end: int,
                         width: int, quality: int) -> None:
        frame = start
        flags = 0
        try:
            while frame <= end:
                while self.credit <= 0:
                    self.credit_event.clear()
                    await self.credit_event.wait()
                if frame_source is None:
                    flags = FLAG_ERROR
                    break
                try:
                    # Off the event loop: a decode is a video seek + JPEG
                    # encode, and the loop also serves every other client.
                    result = await asyncio.to_thread(frame_source, index, frame, width, quality)
                except ValueError:
                    flags = FLAG_ERROR      # not a video / no such target
                    break
                if result is None:
                    break                   # past the last frame: a clean end
                data, w, h = result
                self.credit -= 1
                await self.send(pack_header(KIND_PLAY, stream, frame, w, h) + data)
                stats["play_sent"] += 1
                frame += 1
        except asyncio.CancelledError:
            raise
        except Exception as error:
            swallowed('frames.play', error, 'stream ended early')
            flags = FLAG_ERROR
        with contextlib.suppress(Exception):
            await self.send(pack_header(KIND_END, stream, frame, 0, 0, flags))

    def close(self) -> None:
        self.set_live(False)
        self.stop_play()


@router.websocket("/ws/frames")
async def websocket_frames(websocket: WebSocket) -> None:
    await websocket.accept()
    stats["connected"] += 1
    stats["total_connections"] += 1
    conn = _Connection(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            if text == "ping":
                async with conn.send_lock:
                    await websocket.send_text('{"event":"pong"}')
                continue
            try:
                msg = json.loads(text)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            op = msg.get("op")
            if op == "live":
                conn.set_live(bool(msg.get("on")))
            elif op == "play":
                conn.start_play(msg)
            elif op == "credit":
                conn.add_credit(msg)
            elif op == "stop":
                conn.stop_play(_int(msg.get("stream"), -1, -1, 0xFFFFFFFF))
    except WebSocketDisconnect:
        pass
    except Exception as error:
        swallowed('frames.connection', error, 'closed one client')
    finally:
        conn.close()
        stats["connected"] -= 1


@router.get("/api/frames/status")
def frames_status() -> dict[str, Any]:
    """Transport counters, so a test (or a person) can prove frames flowed."""
    return {"enabled": frame_source is not None, "header_bytes": HEADER.size,
            "live_poll_s": LIVE_POLL_S, "max_credit": MAX_CREDIT, **stats}
