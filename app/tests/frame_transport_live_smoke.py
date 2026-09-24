"""Real render, real browser: the LIVE frame socket and the output compare view.

frame_transport_ab.py covers timeline playback. This covers the other two
paths, which only exist while (and after) a render actually runs:

  * LIVE: while rendering, the Processing tab must receive the pipeline's
    frames over /ws/frames (window.__roopFrameSocket.stats.live grows), show
    them on the FastCanvasPlayer, and NOT fall back to polling
    /api/live_frame while the socket is open.
  * COMPARE: once finished, /api/progress must carry the output's `version`
    and `source`; /api/output/source must answer a Range request with 206;
    and the Split / Side-by-side views must draw (screenshots are saved for a
    person to look at — a pixel count cannot say the two sides line up).

The client is the production build served by the backend itself (same-origin,
exactly what Pinokio loads). The render is 120 frames of the clip.

    app/env/Scripts/python.exe tests/frame_transport_live_smoke.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from frame_transport_ab import (  # noqa: E402
    APP, INSTRUMENT, GpuBrowser, _clip, build_new, seed, start_backend, stop,
)
from browser_driver import find_browser, free_port, wait_for_http  # noqa: E402
if APP not in sys.path:
    sys.path.insert(0, APP)

OUT = os.path.join(APP, "output", "frame_transport_live")
RENDER_TIMEOUT_S = float(os.environ.get("ROOP_LIVE_RENDER_TIMEOUT", "1500"))
START, END = 200, 320


def _post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode() or "{}")


def _get(base, path, headers=None):
    req = urllib.request.Request(base + path, headers=headers or {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, dict(r.headers), r.read()


async def main():
    os.makedirs(OUT, exist_ok=True)
    report = []

    def ok(name, cond, detail=""):
        report.append((name, bool(cond), detail))
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)

    clip = _clip()
    build_new()
    port = free_port()
    backend, blog = start_backend(port, os.path.join(OUT, "backend.log"))
    base = f"http://127.0.0.1:{port}"
    try:
        wait_for_http(base + "/api/meta", timeout=900)
        await seed(base, clip)
        _post(base, "/api/target/set_frame", {"which": "end", "frame": END})
        _post(base, "/api/target/set_frame", {"which": "start", "frame": START})

        async with GpuBrowser(find_browser()) as page:
            await page.send("Page.addScriptToEvaluateOnNewDocument", source=INSTRUMENT)
            # Start the render through the API, NOT the Start button: the button
            # first POSTs the panel's settings to /api/settings, which rewrites
            # the operator's config.yaml. `detection` is a per-run override
            # ("All faces"), so no target face needs capturing either.
            cfg_before = os.path.getmtime(os.path.join(APP, "config.yaml"))
            await page.goto(base + "/#/processing", settle=4.0)
            try:
                answer = _post(base, "/api/swap", {"target_index": 0, "detection": "All faces"})
            except urllib.error.HTTPError as err:
                answer = {"http": err.code, "body": err.read().decode(errors="replace")[:300]}
            running = False
            for _ in range(60):
                _, _, body = _get(base, "/api/jobs/active")
                if json.loads(body).get("processing"):
                    running = True
                    break
                await asyncio.sleep(0.5)
            ok("render started", running, json.dumps(answer)[:300])
            live = await page.wait_for(
                "(window.__roopFrameSocket?.stats.live || 0) >= 3", timeout=RENDER_TIMEOUT_S)
            stats = await page.evaluate(
                "const fs = window.__roopFrameSocket;"
                " const polls = performance.getEntriesByType('resource')"
                "   .filter(e => e.name.includes('/api/live_frame')).length;"
                " const c = document.querySelector(\"canvas[aria-label='Latest processed frame']\");"
                " return { socket: fs ? { ...fs.stats, open: fs.isOpen() } : null, polls,"
                "   canvas: !!c, shown: !!c && !c.classList.contains('invisible'),"
                "   tab: location.hash };")
            ok("live frames arrive over /ws/frames", live, json.dumps(stats))
            ok("the live canvas is showing them", stats and stats["shown"])
            ok("no /api/live_frame polling while the socket is open",
               stats and stats["polls"] == 0, f"polls={stats and stats['polls']}")
            await page.screenshot(os.path.join(OUT, "live.png"))

            deadline = time.monotonic() + RENDER_TIMEOUT_S
            while time.monotonic() < deadline:
                _, _, body = _get(base, "/api/jobs/active")
                if not json.loads(body).get("processing"):
                    break
                await asyncio.sleep(3)
            _, _, body = _get(base, "/api/progress")
            prog = json.loads(body)
            out = prog.get("output") or {}
            ok("output carries a content version", bool(out.get("version")), out.get("version", ""))
            src = out.get("source") or {}
            ok("output carries its source", bool(src.get("url")), json.dumps(src))
            if src.get("url"):
                status, headers, data = _get(base, src["url"], {"Range": "bytes=0-99"})
                ok("source answers a Range request with 206", status == 206 and len(data) == 100,
                   headers.get("Content-Range", ""))
            if out.get("absolute_path"):
                # The delivered file itself: a trimmed render must start its
                # video with its audio and keep every rendered frame (see
                # util_ffmpeg.restore_audio).
                import subprocess
                from roop.ffmpeg_path import ffmpeg_binary
                probe = os.path.join(os.path.dirname(ffmpeg_binary()), "ffprobe.exe" if os.name == "nt" else "ffprobe")
                rows = subprocess.run([probe, "-v", "error", "-count_packets", "-show_entries",
                                       "stream=codec_type,start_time,nb_read_packets", "-of", "csv=p=0",
                                       out["absolute_path"]], capture_output=True, text=True).stdout.split()
                info = {r.split(",")[0]: r.split(",")[1:] for r in rows}
                v, a = info.get("video"), info.get("audio")
                ok("output video starts with its audio",
                   v and a and abs(float(v[0]) - float(a[0])) < 0.05, json.dumps(info))
                ok("output keeps every rendered frame", v and int(v[1]) == END - START, json.dumps(info))
            if out.get("url"):
                status, headers, data = _get(base, out["url"], {"Range": "bytes=-64"})
                ok("output answers a suffix range with its tail", status == 206 and len(data) == 64,
                   headers.get("Content-Range", ""))

            await page.goto(base + "/#/processing", settle=5.0)
            for view in ("Split", "Side by side"):
                clicked = await page.click_text("button[role=radio]", view, settle=4.0)
                drawn = await page.evaluate(
                    "const c = document.querySelector(\"canvas[aria-label*='compared']\");"
                    " const v = Array.from(document.querySelectorAll('video'));"
                    " return { canvas: !!c, w: c?.width || 0,"
                    "   ready: v.map(x => x.readyState) };")
                ok(f"{view} view mounts a WebGL stage", clicked and drawn and drawn["canvas"] and drawn["w"] > 0,
                   json.dumps(drawn))
                await page.screenshot(os.path.join(OUT, f"compare_{view.replace(' ', '_')}.png"))
            # Play a moment so both clocks run, then check they agree.
            await page.click("button[aria-label='Play']", settle=3.0)
            sync = await page.evaluate(
                "const v = Array.from(document.querySelectorAll('video'));"
                " return v.map(x => ({ t: x.currentTime, paused: x.paused, ready: x.readyState }));")
            await page.screenshot(os.path.join(OUT, "compare_playing.png"))
            if sync and len(sync) == 2:
                offset = START / float(src.get("fps") or 1)
                drift = abs((sync[0]["t"]) - (sync[1]["t"] + offset))
                ok("original and result play in step", drift < 0.25 and not sync[1]["paused"],
                   f"follower={sync[0]['t']:.3f}s master={sync[1]['t']:.3f}s offset={offset:.3f}s drift={drift:.3f}s")
            else:
                ok("original and result play in step", False, json.dumps(sync))
        ok("config.yaml untouched", os.path.getmtime(os.path.join(APP, "config.yaml")) == cfg_before)
    finally:
        stop(backend)
        blog.close()
    failed = [r for r in report if not r[1]]
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as fh:
        json.dump([{"check": n, "ok": o, "detail": d} for n, o, d in report], fh, indent=2)
    print(f"\n{len(report) - len(failed)}/{len(report)} checks passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
