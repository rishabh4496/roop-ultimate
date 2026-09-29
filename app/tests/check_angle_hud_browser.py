"""Drive the Biometric Angle HUD in a real headless Chromium.

Serves the HUD harness page (react-ui/.render-check/angle-hud) next to the REAL
routes_angle_scan router, whose model hooks are the same stand-ins
test_routes_angle_scan.py uses: a synthetic clip of one head turning from -60
to +60 degrees, a detector that reads the head out of the frame, no GPU. The
real hook, components, socket, HTTP endpoints, crop serving and pipeline run.

Checks, each the way the feature could look fine while not working:
  * the scan starts BY ITSELF when the page opens (auto-trigger) and the
    progress block is shown while it runs;
  * the matrix renders 9 cells, filled ones load their 512 px crops over HTTP;
  * JUMP moves the timeline to the card's frame + 1 (1-based timeline);
  * the min-eye-distance slider re-gates LIVE (no rescan) and the coverage
    warning appears when the frontal bin empties, then clears;
  * RETAKE + scrub + Assign pins the scrubbed frame (frame - 1 on the server);
  * "Add to angle bank" reaches apply_to_person and hands its payload back;
  * a reload rehydrates the session WITHOUT a second scan;
  * no console errors.

Not a pytest test (needs Node + Chromium). Run from app/:
    env/Scripts/python.exe tests/check_angle_hud_browser.py [--screenshot PATH]
Exit status 0 = every check passed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REACT_UI = os.path.join(os.path.dirname(APP), "react-ui")
sys.path.insert(0, APP)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import routes_angle_scan as ras  # noqa: E402
from browser_driver import Browser, free_port  # noqa: E402
import test_routes_angle_scan as fx  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    results.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"\n          {detail}" if detail and not cond else ""))
    return bool(cond)


def node_binary() -> str:
    found = shutil.which("node")
    if found:
        return found
    raise SystemExit("node is not on PATH (Pinokio ships one; add its folder to PATH)")


def build_harness(out_dir: str) -> None:
    subprocess.run([node_binary(), os.path.join(".render-check", "angle-hud", "build.mjs"), out_dir],
                   cwd=REACT_UI, check=True)


def make_app(clip: str, dist: str, cache: str, counters: dict) -> FastAPI:
    os.environ["ROOP_TARGET_ANGLE_CACHE"] = cache

    def resolve(payload):
        counters["scans"] += 1
        return {"media_path": clip, "media_id": payload.get("target_media_id") or "m1",
                "person_id": payload.get("target_person_id") or "p1",
                "references": np.stack([fx._emb(0)])}

    def slow_detect(frame):
        time.sleep(0.06)                  # long enough for the progress block to be seen
        return fx._detect(frame)

    def read(path, idx):
        import cv2
        cap = cv2.VideoCapture(path)
        try:
            frame = None
            for _ in range(idx + 1):
                ok, frame = cap.read()
                if not ok:
                    return None
            return frame
        finally:
            cap.release()

    def apply(person_id, media_id, media_path, picks):
        counters["applied"].append([p["bin"] for p in picks])
        return {"added": len(picks), "added_bins": picks, "skipped": [], "target_person_ids": [person_id]}

    ras.resolve_target = resolve
    ras.is_busy = lambda: False
    ras.detect_faces = slow_detect
    ras.read_frame = read
    ras.landmarks_fn = None
    ras.embed_fn = lambda frame, kps: fx._emb(0)
    ras.apply_to_person = apply
    ras._session = None
    # A SOURCE faceset with a frontal and both profiles (the clip's own head
    # stands in for a different person here: routing only reads its geometry
    # and vectors).
    import types
    source = types.SimpleNamespace(faces=[fx._face_at(30), fx._face_at(0), fx._face_at(60)])
    counters["source"] = source
    ras.get_source_faceset = lambda i: [source][i]

    app = FastAPI()
    app.include_router(ras.router)
    app.mount("/", StaticFiles(directory=dist, html=True), name="harness")
    return app


async def drive(url: str, counters: dict, screenshot: str | None) -> None:
    async with Browser(width=900, height=1400) as page:
        await page.goto(url, settle=0.2)

        saw_progress = await page.wait_for("document.querySelector('[data-testid=angle-progress]')", timeout=15)
        check("auto-scan starts on its own and shows progress", saw_progress)
        done = await page.wait_for(
            "document.querySelector('[data-testid=angle-matrix] [data-bin=BIN_0_FRONTAL][data-status=selected]') !== null"
            " && !document.querySelector('[data-testid=angle-progress]')", timeout=60)
        check("scan finishes with a frontal pick", done)
        check("exactly one scan ran", counters["scans"] == 1, f"scans={counters['scans']}")

        cells = await page.evaluate(
            "return Array.from(document.querySelectorAll('[data-testid=angle-matrix] [data-bin]'))"
            ".map(n => [n.dataset.bin, n.dataset.status]);")
        check("matrix renders 9 cells", len(cells) == 9, str(cells))
        filled = [b for b, s in cells if s != "missing"]
        check("both profiles and the frontal cell are filled",
              {"BIN_0_FRONTAL", "BIN_5_PROFILE_LEFT", "BIN_6_PROFILE_RIGHT"} <= set(filled), str(cells))
        missing_cls = await page.evaluate(
            "return Array.from(document.querySelectorAll('[data-testid=angle-matrix] [data-status=missing]'))"
            ".every(n => n.className.includes('border-dashed'));")
        check("empty bins render the warning border", missing_cls)

        imgs_ok = await page.wait_for(
            "Array.from(document.querySelectorAll('[data-testid=angle-matrix] img'))"
            ".every(i => i.complete && i.naturalWidth === 512)", timeout=15)
        n_imgs = await page.evaluate("return document.querySelectorAll('[data-testid=angle-matrix] img').length;")
        check("every filled card loads its 512 px crop", imgs_ok and n_imgs == len(filled),
              f"imgs={n_imgs} filled={len(filled)}")
        radar = await page.evaluate(
            "return Array.from(document.querySelectorAll('[data-testid=angle-radar] [data-bin]'))"
            ".map(n => [n.dataset.bin, n.dataset.status]);")
        check("radar agrees with the matrix", sorted(radar) == sorted(cells), f"{radar} vs {cells}")

        sess = await page.evaluate("return (await (await fetch('/api/angle-scan/session')).json()).session;")
        front = next(b for b in sess["bins"] if b["bin"] == "BIN_0_FRONTAL")
        await page.evaluate(
            "const c = document.querySelector('[data-testid=angle-matrix] [data-bin=BIN_0_FRONTAL]');"
            " Array.from(c.querySelectorAll('button')).find(b => b.textContent.trim() === 'JUMP').click(); return true;")
        frame = await page.evaluate("return Number(document.querySelector('[data-testid=frame]').textContent);")
        check("JUMP moves the timeline to frame_idx + 1", frame == front["frame_idx"] + 1,
              f"timeline={frame} frame_idx={front['frame_idx']}")

        set_range = ("const el = document.querySelector({sel});"
                     " const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;"
                     " setter.call(el, '{val}'); el.dispatchEvent(new Event('input', {{bubbles: true}}));"
                     " el.dispatchEvent(new Event('change', {{bubbles: true}})); return true;")
        await page.evaluate(set_range.format(sel=json.dumps("input[aria-label='Minimum eye distance in pixels']"), val=120))
        emptied = await page.wait_for(
            "document.querySelectorAll('[data-testid=angle-matrix] [data-status=missing]').length === 9", timeout=10)
        warned = await page.wait_for("document.querySelector('[data-testid=angle-warning]') !== null", timeout=5)
        check("min eye distance slider re-gates live (all bins empty at 120 px)", emptied)
        check("coverage warning appears when the frontal bin empties", warned)
        await page.evaluate(set_range.format(sel=json.dumps("input[aria-label='Minimum eye distance in pixels']"), val=45))
        restored = await page.wait_for(
            "document.querySelector('[data-testid=angle-matrix] [data-bin=BIN_0_FRONTAL][data-status=selected]') !== null"
            " && document.querySelector('[data-testid=angle-warning]') === null", timeout=10)
        check("moving it back restores the picks and clears the warning", restored)
        check("the sliders did not rescan", counters["scans"] == 1, f"scans={counters['scans']}")

        await page.evaluate(
            "const c = document.querySelector('[data-testid=angle-matrix] [data-bin=BIN_7_PITCH_UP]');"
            " Array.from(c.querySelectorAll('button')).find(b => b.textContent.trim() === 'RETAKE').click(); return true;")
        await page.evaluate(set_range.format(sel=json.dumps("[data-testid=scrub]"), val=31))
        assign_label = await page.evaluate(
            "const b = Array.from(document.querySelectorAll('[data-testid=angle-matrix] [data-bin=BIN_7_PITCH_UP] button'))"
            ".find(x => x.textContent.includes('Assign')); return b ? b.textContent.trim() : null;")
        check("RETAKE offers to assign the scrubbed frame", assign_label == "Assign frame 31", str(assign_label))
        await page.click_text("[data-testid=angle-matrix] [data-bin=BIN_7_PITCH_UP] button", "Assign")
        pinned = await page.wait_for("document.querySelector('[data-testid=angle-matrix] [data-bin=BIN_7_PITCH_UP][data-status=override]') !== null",
                                     timeout=10)
        sess = await page.evaluate("return (await (await fetch('/api/angle-scan/session')).json()).session;")
        b7 = next(b for b in sess["bins"] if b["bin"] == "BIN_7_PITCH_UP")
        check("Assign pins timeline frame 31 = server frame_idx 30", pinned and b7["frame_idx"] == 30,
              json.dumps({k: b7.get(k) for k in ("status", "frame_idx")}))
        toasts = json.loads(await page.evaluate("return document.querySelector('[data-testid=toasts]').textContent;"))
        check("a pose mismatch is reported (frame 31 is a frontal head, pinned as Looking up)",
              any("Looking up set to frame 31" in t["message"] and "reads as frontal" in t["message"]
                  and t["type"] == "warning" for t in toasts), str(toasts))

        await page.click_text("button", "Add to angle bank")
        applied = await page.wait_for("document.querySelector('[data-testid=bank]').textContent.includes('added')",
                                      timeout=10)
        bank = await page.evaluate("return document.querySelector('[data-testid=bank]').textContent;")
        check("Add to angle bank reaches apply_to_person and returns its payload",
              applied and counters["applied"] and "BIN_7_PITCH_UP" in counters["applied"][-1], bank[:200])
        toasts = json.loads(await page.evaluate("return document.querySelector('[data-testid=toasts]').textContent;"))
        check("the result is announced", any("Added" in t["message"] for t in toasts), str(toasts))

        # ── pose routing (Stage 5): the SOURCE portfolio ────────────────────
        off = await page.evaluate("return document.querySelector('[data-testid=source-routing]').dataset.active;")
        check("pose routing starts off", off == "false", str(off))
        lut_line = await page.evaluate(
            "return document.querySelector('[data-testid=source-routing]').textContent;")
        check("the target scan published a frame LUT", "scanned frames" in lut_line, lut_line[-120:])
        await page.click_text("[data-testid=source-routing] button", "Enable from source faceset")
        on = await page.wait_for("document.querySelector('[data-testid=source-routing]').dataset.active === 'true'",
                                 timeout=10)
        cells = await page.evaluate(
            "return Array.from(document.querySelectorAll('[data-testid=source-routing] [data-bin]'))"
            ".filter(n => n.dataset.has === 'true').map(n => n.dataset.bin);")
        check("Enable builds the source portfolio and attaches it to the faceset",
              on and getattr(counters["source"], "angle_portfolio", None) is not None, str(cells))
        check("the source's frontal and both profiles are listed",
              {"BIN_0_FRONTAL", "BIN_5_PROFILE_LEFT", "BIN_6_PROFILE_RIGHT"} <= set(cells), str(cells))
        text = await page.evaluate("return document.querySelector('[data-testid=source-routing]').textContent;")
        check("the routing rule is stated", "0.7 × source profile + 0.3 × fused" in text, text[:200])

        if screenshot:
            await page.screenshot(screenshot)

        await page.click_text("[data-testid=source-routing] button", "Off")
        cleared = await page.wait_for(
            "document.querySelector('[data-testid=source-routing]').dataset.active === 'false'", timeout=10)
        check("Off detaches it", cleared and counters["source"].angle_portfolio is None)
        await page.click_text("[data-testid=source-routing] button", "Enable from source faceset")
        await page.wait_for("document.querySelector('[data-testid=source-routing]').dataset.active === 'true'",
                            timeout=10)

        await page.goto(url, settle=2.0)
        rehydrated = await page.wait_for(
            "document.querySelector('[data-testid=angle-matrix] [data-bin=BIN_7_PITCH_UP][data-status=override]') !== null", timeout=10)
        routing_kept = await page.wait_for(
            "document.querySelector('[data-testid=source-routing]').dataset.active === 'true'", timeout=10)
        check("reload shows the source portfolio still attached", routing_kept)
        await asyncio.sleep(1.5)                  # past the auto-trigger's settle delay
        check("reload rehydrates the session", rehydrated)
        check("reload does not rescan", counters["scans"] == 1, f"scans={counters['scans']}")

        errors = [e for e in page.errors if "favicon" not in (e.get("url") or e.get("text", ""))]
        check("no console errors", not errors, json.dumps(errors)[:500])


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # cp1252 consoles
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshot", default=None)
    args = ap.parse_args()
    work = tempfile.mkdtemp(prefix="angle-hud-")
    try:
        dist = os.path.join(work, "dist")
        build_harness(dist)
        clip = fx.make_clip(os.path.join(work, "clip.avi"))
        counters = {"scans": 0, "applied": []}
        app = make_app(clip, dist, os.path.join(work, "cache"), counters)
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 20
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        print("── Biometric Angle HUD in Chromium ───────────────────────")
        try:
            asyncio.run(drive(f"http://127.0.0.1:{port}/?person=p1&media=m1&frames={fx.N}", counters,
                              args.screenshot))
        finally:
            server.should_exit = True
            thread.join(timeout=10)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
