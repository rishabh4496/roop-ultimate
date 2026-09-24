"""Real-browser A/B of timeline playback: the pre-2026-09-24 client vs. the
binary-socket / FastCanvasPlayer / 10 Hz-playhead client.

WHY A HARNESS AND NOT A NUMBER IN A COMMIT MESSAGE
This project's recurring failure is a feature that reports success while not
running. So this does not assume the new path runs: it asks the page. For the
new build it reads `window.__roopFrameSocket.stats` (frames received over
/ws/frames) and which transport fed playback; for both builds it counts React
commits (a devtools hook installed before the app loads), main-thread script
and task time (CDP Performance.getMetrics), long tasks, delivered animation
frames, and how many target frames the playhead actually advanced.

ONE backend (run.py, the real app) serves both builds through two
`vite preview` servers (proxying /api and /ws), so the only difference between
arms is the client. Arms run A/B/B/A (counterbalanced, AGENTS.md) in fresh
browser profiles. Chromium runs headless WITH the GPU: `--disable-gpu` would
push the new renderer onto its 2D fallback and measure something the Pinokio
webview never runs.

    app/env/Scripts/python.exe tests/frame_transport_ab.py
    env: ROOP_AB_CLIP (default <PINOKIO_HOME>/roop-keep/b1.mp4),
         ROOP_AB_SECONDS (default 10), ROOP_AB_KEEP_WORKTREE=1
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
sys.path.insert(0, HERE)

import fixtures  # noqa: E402
from browser_driver import Browser, Page, find_browser, free_port, wait_for_http  # noqa: E402

import websockets  # noqa: E402

SECONDS = float(os.environ.get("ROOP_AB_SECONDS", "10"))
WARM_S = 2.0
PREVIEW_TIMEOUT_S = float(os.environ.get("ROOP_AB_PREVIEW_TIMEOUT", "900"))
OUT = os.path.join(APP, "output", "frame_transport_ab")


def _clip():
    env = os.environ.get("ROOP_AB_CLIP")
    if env:
        return env
    home = fixtures.pinokio_home()
    return os.path.join(home, "roop-keep", "b1.mp4") if home else ""


def _node_dir():
    found = shutil.which("node")
    if found:
        return os.path.dirname(found)
    home = fixtures.pinokio_home() or ""
    for sub in ("miniforge", "miniconda"):
        cand = os.path.join(home, "bin", sub, "node.exe")
        if os.path.exists(cand):
            return os.path.dirname(cand)
    raise RuntimeError("node not found")


def _run(cmd, cwd, env=None):
    subprocess.run(cmd, cwd=cwd, env=env, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT, text=True)


def build_old(workdir):
    """The client as of HEAD (before this change), built into its own dist/."""
    wt = os.path.join(workdir, "ab_old")
    _run(["git", "worktree", "add", "--detach", wt, "HEAD"], cwd=ROOT)
    # One node_modules for both: a junction, not a copy of 300 MB.
    subprocess.run(["cmd", "/c", "mklink", "/J",
                    os.path.join(wt, "react-ui", "node_modules"),
                    os.path.join(ROOT, "react-ui", "node_modules")],
                   check=True, stdout=subprocess.DEVNULL)
    node = os.path.join(_node_dir(), "node.exe")
    _run([node, "node_modules/vite/bin/vite.js", "build"], cwd=os.path.join(wt, "react-ui"))
    return wt


def build_new():
    node = os.path.join(_node_dir(), "node.exe")
    _run([node, "node_modules/vite/bin/vite.js", "build"], cwd=os.path.join(ROOT, "react-ui"))


def start_preview(react_dir, api_port, port, log_path):
    env = os.environ.copy()
    env.update({"ROOP_API_PORT": str(api_port), "PORT": str(port)})
    env["PATH"] = _node_dir() + os.pathsep + env.get("PATH", "")
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    node = os.path.join(_node_dir(), "node.exe")
    proc = subprocess.Popen([node, "node_modules/vite/bin/vite.js", "preview",
                             "--host", "127.0.0.1", "--port", str(port), "--strictPort"],
                            cwd=react_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, log


def start_backend(api_port, log_path):
    env = os.environ.copy()
    # Exactly what start_react.js runs (the backend the React client talks to).
    env.update({"ROOP_API_PORT": str(api_port), "ROOP_GRADIO_PORT": str(api_port + 1),
                "ROOP_REACT_CLIENT": "1", "NO_ALBUMENTATIONS_UPDATE": "1",
                "ROOP_TEMPORAL_STEP": "1", "PYTHONUNBUFFERED": "1"})
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([os.path.join(APP, "env", "Scripts", "python.exe"), "run.py", "--ui", "react"],
                            cwd=APP, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, log


def stop(proc):
    if proc is None:
        return
    with contextlib.suppress(Exception):
        proc.terminate()
        proc.wait(timeout=20)
    if proc.poll() is None:
        with contextlib.suppress(Exception):
            proc.kill()


class GpuBrowser(Browser):
    """browser_driver.Browser, but with the GPU on (see module docstring)."""

    async def __aenter__(self):
        self.port = free_port()
        self._profile = tempfile.mkdtemp(prefix="roop-ab-")
        self._process = subprocess.Popen(
            [self.binary, "--headless=new", "--enable-gpu", "--ignore-gpu-blocklist",
             "--no-first-run", "--no-default-browser-check", "--disable-extensions",
             "--disable-background-networking", "--disable-background-timer-throttling",
             "--disable-renderer-backgrounding", "--disable-features=Translate,MediaRouter",
             "--autoplay-policy=no-user-gesture-required",
             f"--window-size={self.width},{self.height}",
             f"--remote-debugging-port={self.port}",
             f"--user-data-dir={self._profile}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        target = await self._await_target()
        self._socket = await websockets.connect(target, max_size=64 * 1024 * 1024)
        self.page = Page(self._socket)
        await self.page.start()
        await self.page.set_viewport(self.width, self.height)
        return self.page


# Installed before the app's scripts run: counts React commits through the
# devtools hook React looks for at startup, and records long tasks.
INSTRUMENT = r"""
window.__commits = 0;
window.__long = [];
window.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
  supportsFiber: true, isDisabled: false, renderers: new Map(),
  inject() { return 1; }, checkDCE() {},
  onCommitFiberRoot() { window.__commits += 1; },
  onCommitFiberUnmount() {}, onPostCommitFiberRoot() {}, setStrictMode() {},
};
try {
  new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__long.push(e.duration); })
    .observe({ type: 'longtask', buffered: true });
} catch (e) {}
"""

FRAME_INPUT = "input[type=number][title^='Type a frame number']"


async def metrics(page):
    raw = await page.send("Performance.getMetrics")
    return {m["name"]: m["value"] for m in raw.get("metrics", [])}


async def measure_arm(label, base, clip_fps, shots):
    async with GpuBrowser(find_browser()) as page:
        await page.send("Page.addScriptToEvaluateOnNewDocument", source=INSTRUMENT)
        await page.send("Performance.enable", timeMode="threadTicks")
        await page.goto(base + "/#/faceswap", settle=4.0)
        gl = await page.evaluate(
            "const c = document.createElement('canvas');"
            " return !!(c.getContext('webgl2') || c.getContext('webgl'));")
        mounted = await page.wait_for(
            "document.querySelectorAll(\"canvas[aria-label='Preview frame']\").length > 0",
            timeout=PREVIEW_TIMEOUT_S)
        if not mounted:
            with contextlib.suppress(Exception):
                await page.click_text("button", "Refresh", settle=1.0)
            mounted = await page.wait_for(
                "document.querySelectorAll(\"canvas[aria-label='Preview frame']\").length > 0",
                timeout=PREVIEW_TIMEOUT_S)
        if not mounted:
            return {"arm": label, "error": "the preview stage never mounted"}
        await asyncio.sleep(2.0)
        # Park at frame 1 so every arm plays the same stretch of the clip.
        await page.evaluate(
            f"const i = document.querySelector({json.dumps(FRAME_INPUT)});"
            " if (!i) return false; i.focus();"
            " const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;"
            " set.call(i, '1'); i.dispatchEvent(new Event('input', { bubbles: true }));"
            " i.blur(); return true;")
        await asyncio.sleep(3.0)
        f0 = await page.evaluate(f"return Number(document.querySelector({json.dumps(FRAME_INPUT)})?.value || 0);")

        played = await page.click("button[title='Play (Space)']", settle=0.0)
        if not played:
            return {"arm": label, "error": "no Play button"}
        t_play = time.monotonic()
        await asyncio.sleep(WARM_S)

        await page.evaluate(
            "window.__raf = 0; window.__rafOn = true;"
            " const loop = () => { if (!window.__rafOn) return; window.__raf += 1; requestAnimationFrame(loop); };"
            " requestAnimationFrame(loop);"
            " window.__c0 = window.__commits; window.__l0 = window.__long.length; return true;")
        m0 = await metrics(page)
        w0 = await page.evaluate(f"return Number(document.querySelector({json.dumps(FRAME_INPUT)})?.value || 0);")
        t0 = time.monotonic()
        await asyncio.sleep(SECONDS)
        m1 = await metrics(page)
        w1 = await page.evaluate(f"return Number(document.querySelector({json.dumps(FRAME_INPUT)})?.value || 0);")
        t1 = time.monotonic()
        window = await page.evaluate(
            "window.__rafOn = false;"
            " const fs = window.__roopFrameSocket;"
            " return { commits: window.__commits - window.__c0,"
            "   longs: window.__long.slice(window.__l0), raf: window.__raf,"
            "   socket: fs ? { ...fs.stats, open: fs.isOpen() } : null,"
            "   playback: window.__roopPlayback ? window.__roopPlayback.stats : null,"
            "   http_chunks: performance.getEntriesByType('resource')"
            "     .filter(e => e.name.includes('/api/target/preview_seq')).length,"
            "   http_stills: performance.getEntriesByType('resource')"
            "     .filter(e => e.name.includes('/api/target/preview?')).length };")
        os.makedirs(shots, exist_ok=True)
        await page.screenshot(os.path.join(shots, f"{label}.png"))

        await page.click("button[title='Pause (Space)']", settle=1.5)
        t_pause = time.monotonic()
        f1 = await page.evaluate(f"return Number(document.querySelector({json.dumps(FRAME_INPUT)})?.value || 0);")

        dt = t1 - t0
        d = lambda k: (m1.get(k, 0.0) - m0.get(k, 0.0))  # noqa: E731
        span = t_pause - t_play
        return {
            "arm": label,
            "webgl": gl,
            "window_s": round(dt, 2),
            "react_commits_per_s": round(window["commits"] / dt, 1),
            "script_ms_per_s": round(d("ScriptDuration") * 1000 / dt, 1),
            "task_ms_per_s": round(d("TaskDuration") * 1000 / dt, 1),
            "style_layout_ms_per_s": round((d("RecalcStyleDuration") + d("LayoutDuration")) * 1000 / dt, 1),
            "long_tasks": len(window["longs"]),
            "long_task_ms": round(sum(window["longs"]), 1),
            "raf_fps": round(window["raf"] / dt, 1),
            "frames_played": int(f1 - f0),
            # Playhead advance over the measured window only. The new client
            # writes the playhead at 10 Hz, so either end can lag the picture
            # by <= 100 ms (~2.4 frames at 24 fps, ~1% of the window).
            "window_play_fps": round((w1 - w0) / dt, 2),
            "play_fps_incl_start": round((f1 - f0) / span, 2) if span > 0 else 0,
            "clip_fps": clip_fps,
            "socket": window["socket"],
            "playback": window.get("playback"),
            "http_chunks": window.get("http_chunks"),
            "http_stills": window.get("http_stills"),
        }


async def seed(api_base, clip):
    """Faces from the library, the clip by path: no uploads."""
    import urllib.request

    def post(path, body):
        req = urllib.request.Request(api_base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode() or "{}")

    with urllib.request.urlopen(api_base + "/api/faceset/library", timeout=60) as r:
        lib = json.loads(r.read().decode())
    entry = (lib.get("entries") or [None])[0]
    if not entry:
        raise RuntimeError("no faceset in the library to seed a source face")
    post("/api/faceset/library/load", {"filename": entry["filename"]})
    added = post("/api/target/add_path", {"paths": [clip]})
    return entry["filename"], added


async def main():
    clip = _clip()
    if not clip or not os.path.isfile(clip):
        print(f"clip not found: {clip!r} (set ROOP_AB_CLIP)")
        return 2
    os.makedirs(OUT, exist_ok=True)
    work = tempfile.mkdtemp(prefix="roop_ab_")
    procs, logs = [], []
    wt = None
    try:
        print("building old (HEAD) and new clients…", flush=True)
        wt = build_old(work) if "old" in os.environ.get("ROOP_AB_ARMS", "old") else None
        build_new()
        api_port, p_old, p_new = free_port(), free_port(), free_port()
        backend, blog = start_backend(api_port, os.path.join(OUT, "backend.log"))
        procs.append(backend); logs.append(blog)
        wait_for_http(f"http://127.0.0.1:{api_port}/api/meta", timeout=600)
        face, added = await seed(f"http://127.0.0.1:{api_port}", clip)
        print(f"seeded: face={face} target={added}", flush=True)
        servers = [(os.path.join(ROOT, "react-ui"), p_new, "new")]
        if wt:
            servers.append((os.path.join(wt, "react-ui"), p_old, "old"))
        for react_dir, port, name in servers:
            proc, log = start_preview(react_dir, api_port, port, os.path.join(OUT, f"preview_{name}.log"))
            procs.append(proc); logs.append(log)
            wait_for_http(f"http://127.0.0.1:{port}/", timeout=120)
        import cv2
        cap = cv2.VideoCapture(clip)
        clip_fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 0), 3)
        cap.release()

        bases = {"old": f"http://127.0.0.1:{p_old}", "new": f"http://127.0.0.1:{p_new}"}
        results = []
        # ROOP_AB_ARMS=new (comma list) for a diagnostic run of one client.
        arms = tuple(a for a in os.environ.get("ROOP_AB_ARMS", "new,old,old,new").split(",") if a)
        for i, arm in enumerate(arms):
            label = f"{i + 1}_{arm}"
            print(f"arm {label}…", flush=True)
            r = await measure_arm(label, bases[arm], clip_fps, os.path.join(OUT, "shots"))
            print(json.dumps(r), flush=True)
            results.append(r)
        with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
            json.dump({"clip": os.path.basename(clip), "seconds": SECONDS, "results": results}, fh, indent=2)
        return 0
    finally:
        for p in procs:
            stop(p)
        for lg in logs:
            with contextlib.suppress(Exception):
                lg.close()
        if wt and not os.environ.get("ROOP_AB_KEEP_WORKTREE"):
            junction = os.path.join(wt, "react-ui", "node_modules")
            with contextlib.suppress(Exception):
                os.rmdir(junction)   # removes the junction itself, never its target
            if os.path.lexists(junction):
                # Never recurse a delete through a live junction: it points at
                # the REAL react-ui/node_modules.
                print(f"junction still present, worktree left in place: {wt}")
            else:
                subprocess.run(["git", "worktree", "remove", "--force", wt], cwd=ROOT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
