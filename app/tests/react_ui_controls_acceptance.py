"""Live Chromium audit for React tabs and processing controls.

This intentionally drives the UI against a real, already-running backend. It is
not a source-only check: tab clicks, pause/resume/stop buttons, the Vite proxy,
and progress transitions all travel through the same browser path users use.
Set ROOP_CONTROL_API_PORT to the warm backend port (default 42003).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from browser_driver import Browser, find_browser, free_port, wait_for_http  # noqa: E402
from preview_pipeline_smoke import start_ui_only  # noqa: E402
from runtime_lifecycle import Api, payload_for  # noqa: E402

API_PORT = int(os.environ.get("ROOP_CONTROL_API_PORT", "42003"))
TABS = ("Home", "Face Swap", "Batch Matrix", "Processing", "Face Manager", "Editor", "Outputs", "History", "Settings")

# Pinokio commonly runs Python with the Windows cp1252 console encoding. Real
# uploads can contain emoji and other Unicode characters, so diagnostics must
# never crash before the browser acceptance path gets a verdict.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def stop(proc):
    if proc is None:
        return
    with contextlib.suppress(Exception):
        proc.terminate()
        proc.wait(timeout=20)
    if proc.poll() is None:
        with contextlib.suppress(Exception):
            proc.kill()


def check(condition, name, detail=""):
    state = "PASS" if condition else "FAIL"
    print(f"[control] {state:<5} {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return bool(condition)


def ensure_live_render():
    """Start a bounded real render when the warm backend is idle.

    The media and faceset are already part of this workstation's normal
    Pinokio state. A five-thousand-frame run is long enough to click pause,
    resume, and stop, but bounded so this acceptance check never becomes a
    full-clip job.
    """
    api = Api(API_PORT)
    progress = api.get('/api/progress')
    if progress.get('processing'):
        return True
    state = api.get('/api/state')
    targets = state.get('targets') or []
    if not state.get('source_faces') or not targets:
        print('[control] FAIL  cannot start bounded render -- source faces or target media missing', flush=True)
        return False
    index = min(1, len(targets) - 1)
    api.post('/api/target/select', {'index': index})
    api.post('/api/target/set_frame', {'which': 'start', 'frame': 1})
    api.post('/api/target/set_frame', {'which': 'end', 'frame': 5000})
    state = api.get('/api/state')
    body = payload_for(api, state)
    body.update({'upscale_after_swap': False, 'interp_after_swap': 'off'})
    api.post('/api/settings', api.get('/api/settings'))
    started = api.post('/api/swap', body)
    if started.get('message') and started.get('_status', 200) >= 400:
        print(f"[control] FAIL  bounded render start -- {started}", flush=True)
        return False
    for _ in range(120):
        progress = api.get('/api/progress')
        if progress.get('processing') or progress.get('error'):
            print('[control] setup -- bounded render started', flush=True)
            return bool(progress.get('processing'))
        import time
        time.sleep(1)
    print('[control] FAIL  bounded render start -- timeout', flush=True)
    return False


async def main():
    browser = find_browser()
    if not check(bool(browser), "browser runtime", browser or "not found"):
        return 2
    ui_port = free_port()
    log_dir = os.path.join(APP, "output", "react_controls_acceptance")
    vite, log = start_ui_only("react-ui", API_PORT, ui_port, log_dir)
    base = f"http://127.0.0.1:{ui_port}"
    failures = 0
    try:
        failures += not check(ensure_live_render(), "bounded real render is available")
        failures += not check(wait_for_http(base, timeout=60), "Vite dev server", base)
        async with Browser(browser, width=1440, height=900) as page:
            await page.goto(base + "#/home", settle=4.0)
            failures += not check(await page.wait_for("document.querySelector('#root')?.children.length > 0", 30), "React app mounts")
            failures += not check(await page.evaluate("return (await fetch('/api/meta')).status === 200"), "Vite proxy reaches backend")

            layout = {}
            for label in TABS:
                clicked = await page.click_text("button", label, settle=1.2)
                await page.wait_for("document.querySelector('main') && document.querySelector('main').textContent.trim().length > 0", 20)
                result = await page.evaluate("""
                    const root = document.documentElement;
                    const main = document.querySelector('main');
                    return {
                      hash: location.hash,
                      body: (document.body.innerText || '').slice(0, 180),
                      viewport: {w: innerWidth, h: innerHeight},
                      overflowX: root.scrollWidth > root.clientWidth + 1,
                      overflowPx: root.scrollWidth - root.clientWidth,
                      scrollHeight: root.scrollHeight,
                      clientHeight: root.clientHeight,
                      mainHeight: main ? Math.round(main.getBoundingClientRect().height) : 0,
                      buttons: document.querySelectorAll('button').length,
                    };
                """)
                layout[label] = result
                failures += not check(clicked and not result.get("overflowX"), f"desktop {label} layout", json.dumps(result, ensure_ascii=True))

            await page.set_viewport(390, 844)
            await asyncio.sleep(1.0)
            for label in TABS:
                clicked = await page.click_text("button", label, settle=0.7)
                await page.wait_for("document.querySelector('main') && document.querySelector('main').textContent.trim().length > 0", 20)
                result = await page.evaluate("""
                    const root = document.documentElement;
                    return {hash: location.hash, overflowX: root.scrollWidth > root.clientWidth + 1,
                            overflowPx: root.scrollWidth - root.clientWidth,
                            scrollHeight: root.scrollHeight, clientHeight: root.clientHeight,
                            buttons: document.querySelectorAll('button').length};
                """)
                failures += not check(clicked and not result.get("overflowX"), f"mobile {label} layout", json.dumps(result, ensure_ascii=True))

            # Return to the live run screen. The current backend is expected to
            # be processing, because this is a real-control acceptance path.
            await page.set_viewport(1440, 900)
            await page.click_text("button", "Processing", settle=1.0)
            before = await page.evaluate("return await (await fetch('/api/progress')).json();")
            failures += not check(bool(before.get("processing")), "live render available for controls", json.dumps({k: before.get(k) for k in ('processing','paused','pause_requested','progress','desc')}))

            if before.get("processing"):
                pause = await page.click("button[aria-label='Pause the run']", settle=0.8)
                failures += not check(pause, "Processing Pause button is clickable")
                pause_state = await page.wait_for("(async()=>{const r=await fetch('/api/progress'); const p=await r.json(); return p.pause_requested || p.paused;})()", 30)
                after_pause = await page.evaluate("return await (await fetch('/api/progress')).json();")
                failures += not check(pause_state, "pause request reaches backend", json.dumps({k: after_pause.get(k) for k in ('processing','paused','pause_requested','progress','desc')}))
                failures += not check(await page.wait_for("document.body.innerText.includes('Pause Requested') || document.body.innerText.includes('Paused')", 15), "pause state is reflected in React")

                resume = await page.click("button[aria-label='Resume the run']", settle=0.8)
                failures += not check(resume, "Processing Resume button is clickable")
                resume_state = await page.wait_for("(async()=>{const r=await fetch('/api/progress'); const p=await r.json(); return !p.paused && !p.pause_requested;})()", 30)
                after_resume = await page.evaluate("return await (await fetch('/api/progress')).json();")
                failures += not check(resume_state, "resume request reaches backend", json.dumps({k: after_resume.get(k) for k in ('processing','paused','pause_requested','progress','desc')}))
                failures += not check(await page.wait_for("document.querySelector(\"button[aria-label='Pause the run']\") !== null", 15), "resume state restores Pause control")

                stop_clicked = await page.click("button[aria-label='Stop the run']", settle=0.8)
                failures += not check(stop_clicked, "Processing Stop button is clickable")
                stop_confirmed = await page.click("div[role='dialog'] button:last-child", settle=0.8)
                failures += not check(stop_confirmed, "Stop confirmation is accepted")
                failures += not check(await page.wait_for("document.body.innerText.includes('Stopping')", 10), "stop state is reflected immediately in React")
                stopped = await page.wait_for("(async()=>{const r=await fetch('/api/progress'); const p=await r.json(); return !p.processing;})()", 45)
                after_stop = await page.evaluate("return await (await fetch('/api/progress')).json();")
                failures += not check(stopped, "stop request ends the active job", json.dumps({k: after_stop.get(k) for k in ('processing','paused','pause_requested','progress','desc','error')}))
                failures += not check(await page.wait_for("document.body.innerText.includes('Run stopped') || document.body.innerText.includes('Run complete') || document.body.innerText.includes('Run failed')", 30), "stop completion is reflected in Processing tab")

            failures += not check(not page.errors, "browser has no uncaught errors", "; ".join(e.get('text','')[:160] for e in page.errors[:4]) or "none")
            with open(os.path.join(log_dir, "layout.json"), "w", encoding="utf-8") as handle:
                json.dump(layout, handle, indent=2)
    finally:
        stop(vite)
        with contextlib.suppress(Exception):
            log.close()
    print(f"[control] result: {failures} failure(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
