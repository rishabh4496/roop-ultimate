"""WebSocket telemetry latency and backend memory across real renders, through the live API.

Starts the real backend (what start_react.js runs), seeds a library faceset and a clip, then
drives `--jobs` consecutive renders through ``/api/swap`` while a client:

* pings ``/ws/telemetry`` at 20 Hz and times the pong (the route answers any text with a
  ``pong``): the round trip of the event loop that also serves the UI;
* samples the backend's resident memory (the Windows venv launcher is a 5 MB shim; the real
  backend is its child, so the child tree is what is reported);
* optionally, when a ping is outstanding for > 300 ms, takes a ``py-spy dump`` of the backend
  right then (``--pyspy path\\to\\py-spy.exe``; ``pip install --target <dir> py-spy``), which
  shows which thread holds the GIL while the loop is blocked.

Measured 2026-10-03, RTX 4070, hyperswap + Restore Ultra, s3.mp4 frames 0-200, 3 renders::

    ping round trip          p50     p95      p99      max
    idle                     0.6    7.0      7.5      7.6 ms
    model-load phase         1.0    120      1249     5200 ms   <- first ~25 s of every job
    frame-processing phase   0.9    5-11     13-97    145 ms

All of the multi-second stalls were in model loading: the stack dumps put
``onnxruntime InferenceSession`` creation (``_create_inference_session``) on the GIL, with
``onnx.shape_inference``, ``release_face_analyser`` and a hardware-profile call alongside. A C
call that holds the GIL for seconds freezes the event loop, the WebSocket and the progress
sampler with it. Moving the work to another PROCESS is the only fix for that.

Run (app stopped; ~4 min)::

    app\\env\\Scripts\\python.exe tools/probe_telemetry_latency.py --jobs 3 [--pyspy py-spy.exe]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"
for p in (str(APP / "tests"), str(APP)):
    if p not in sys.path:
        sys.path.insert(0, p)


def media_dir() -> Path:
    env = os.environ.get("ROOP_KEEP_DIR")
    return Path(env) if env else REPO.parents[1] / "roop-keep"


def _post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode() or "{}")


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return json.loads(r.read().decode())


def _pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))] if values else float("nan")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clip", default=str(media_dir() / "single" / "s3.mp4"))
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--end-frame", type=int, default=200)
    parser.add_argument("--pyspy", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import psutil
    import websockets
    from browser_driver import free_port, wait_for_http
    from frame_transport_ab import seed, start_backend, stop

    work = Path(tempfile.mkdtemp(prefix="telemetry_"))
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    backend, _ = start_backend(port, str(work / "backend.log"))
    shim = psutil.Process(backend.pid)
    rtt, rss, dumps = [], [], []
    phase = {"name": "boot"}
    t0 = time.monotonic()
    stop_flag = {"v": False}
    try:
        wait_for_http(base + "/api/meta", timeout=900)
        await seed(base, args.clip)
        _post(base, "/api/target/set_frame", {"which": "end", "frame": args.end_frame})
        _post(base, "/api/target/set_frame", {"which": "start", "frame": 0})

        def real_pid():
            kids = shim.children(recursive=True)
            return max(kids, key=lambda c: c.memory_info().rss).pid if kids else shim.pid

        async def pinger():
            uri = f"ws://127.0.0.1:{port}/ws/telemetry"
            async with websockets.connect(uri, max_size=None) as ws:
                await ws.recv()                                      # hello
                while not stop_flag["v"]:
                    t = time.perf_counter()
                    await ws.send("ping")
                    dumped = False
                    while True:
                        try:
                            msg = json.loads(await asyncio.wait_for(ws.recv(), 0.3))
                        except asyncio.TimeoutError:
                            if args.pyspy and not dumped:
                                dumped = True
                                out = subprocess.run([args.pyspy, "dump", "--pid", str(real_pid())],
                                                     capture_output=True, text=True, timeout=60).stdout
                                dumps.append({"t": time.monotonic() - t0, "dump": out})
                            continue
                        if msg.get("event") == "pong":
                            break
                    rtt.append((time.monotonic() - t0, (time.perf_counter() - t) * 1e3, phase["name"]))
                    await asyncio.sleep(0.05)

        async def sampler():
            while not stop_flag["v"]:
                try:
                    kids = shim.children(recursive=True)
                    rss.append((time.monotonic() - t0, sum(c.memory_info().rss for c in kids) / 2 ** 20,
                                phase["name"]))
                except psutil.Error:
                    pass
                await asyncio.sleep(1.0)

        tasks = [asyncio.create_task(pinger()), asyncio.create_task(sampler())]
        phase["name"] = "idle"
        await asyncio.sleep(10)
        for j in range(args.jobs):
            phase["name"] = f"job{j}"
            _post(base, "/api/swap", {"target_index": 0, "detection": "All faces"})
            for _ in range(60):
                if _get(base, "/api/jobs/active").get("processing"):
                    break
                await asyncio.sleep(0.5)
            while _get(base, "/api/jobs/active").get("processing"):
                await asyncio.sleep(2)
            phase["name"] = f"after{j}"
            await asyncio.sleep(5)
        stop_flag["v"] = True
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        stop(backend)

    def line(name, values):
        print(f"{name:24s} n={len(values):4d} p50={_pct(values, .5):8.2f} p95={_pct(values, .95):8.2f} "
              f"p99={_pct(values, .99):9.2f} max={max(values) if values else float('nan'):9.1f} ms")
    line("idle", [r[1] for r in rtt if r[2] == "idle"])
    for j in range(args.jobs):
        rows = [r for r in rtt if r[2] == f"job{j}"]
        if not rows:
            continue
        stalls = [r[0] for r in rows if r[1] > 100]
        cut = (max(stalls) + 1) if stalls else rows[0][0]
        line(f"job{j} load phase", [r[1] for r in rows if r[0] <= cut])
        line(f"job{j} frame phase", [r[1] for r in rows if r[0] > cut])
    for j in range(args.jobs):
        r = [x for x in rss if x[2] == f"after{j}"]
        if r:
            print(f"backend tree RSS after job{j}: {r[-1][1]:.0f} MB")
    if args.out:
        Path(args.out).write_text(json.dumps({"rtt": rtt, "rss": rss, "dumps": dumps}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
