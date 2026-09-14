"""Probe the three facts that decide the Stage 5 telemetry design.

Run:  app/env/Scripts/python.exe app/tests/probe_ws_feasibility.py

1. Does uvicorn in THIS env actually serve a WebSocket? (needs a ws impl)
2. Can a plain `threading.Thread` -- which is how the swap pipeline runs --
   push a message into the asyncio loop and reach a connected client?
   `asyncio.run_coroutine_threadsafe` is the only supported bridge; if it
   works, a broadcast can be driven from the pipeline thread.
3. What does a 1 Hz poll of /api/progress actually cost vs a pushed frame?
   Measured, so "polling is wasteful" is a number and not a slogan.
"""
import asyncio
import json
import threading
import time
import socket
import sys
import urllib.request


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main() -> int:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    import uvicorn

    app = FastAPI()
    loop_box = {}
    clients = set()

    @app.on_event("startup")  # noqa: DeprecationWarning (probe only)
    async def _capture_loop():
        loop_box["loop"] = asyncio.get_running_loop()

    @app.get("/api/progress")
    def progress():
        # Shaped like the real one: it is the payload SIZE that makes polling
        # expensive, and the real endpoint ships a 250-line rolling log.
        return {"processing": True, "progress": 0.5, "desc": "12 / 300",
                "log": ["frame %d" % i for i in range(250)]}

    @app.websocket("/ws/telemetry")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(websocket)

    port = _free_port()
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    for _ in range(100):
        if getattr(server, "started", False):
            break
        time.sleep(0.05)
    if not server.started:
        print("FAIL: server did not start")
        return 1

    results = {}

    # --- 1 + 2: websocket serves, and a NON-async thread can broadcast to it.
    import websockets.sync.client as wsc

    try:
        conn = wsc.connect(f"ws://127.0.0.1:{port}/ws/telemetry", open_timeout=5)
    except Exception as e:
        print(f"FAIL: websocket handshake rejected: {e!r}")
        return 1
    results["handshake"] = True

    # Give the server a moment to register the client.
    time.sleep(0.2)

    # This is the crux: the swap pipeline is a plain threading.Thread with no
    # event loop. Can it reach the client?
    def pipeline_thread():
        loop = loop_box["loop"]

        async def send_all():
            for c in list(clients):
                await c.send_json({"event": "progress", "frame": 42, "fps": 23.5})

        fut = asyncio.run_coroutine_threadsafe(send_all(), loop)
        fut.result(timeout=5)

    pt = threading.Thread(target=pipeline_thread)
    pt.start()
    pt.join(timeout=10)

    msg = json.loads(conn.recv(timeout=5))
    results["thread_to_client"] = (msg.get("frame") == 42)
    print(f"  handshake                : {'OK' if results['handshake'] else 'FAIL'}")
    print(f"  worker-thread -> client  : {'OK' if results['thread_to_client'] else 'FAIL'}  payload={msg}")

    # --- 3: cost of the current 1 Hz poll vs one pushed frame.
    url = f"http://127.0.0.1:{port}/api/progress"
    n = 30
    t0 = time.perf_counter()
    total_bytes = 0
    for _ in range(n):
        with urllib.request.urlopen(url, timeout=5) as r:
            total_bytes += len(r.read())
    poll_ms = (time.perf_counter() - t0) / n * 1000
    poll_bytes = total_bytes / n

    push_payload = json.dumps({"event": "progress", "current_frame": 12,
                               "total_frames": 300, "progress": 4.0, "fps": 23.5}).encode()
    print(f"  poll  /api/progress      : {poll_ms:.2f} ms/req, {poll_bytes:,.0f} bytes/req")
    print(f"  push  telemetry frame    : {len(push_payload):,} bytes")
    print(f"  ratio                    : {poll_bytes / len(push_payload):.0f}x more bytes per update when polling")

    conn.close()
    server.should_exit = True
    t.join(timeout=5)

    ok = results.get("handshake") and results.get("thread_to_client")
    print("\n" + ("ALL PROBES PASSED" if ok else "PROBE FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
