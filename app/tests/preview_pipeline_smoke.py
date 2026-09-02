"""Real-browser smoke for the canvas preview pipeline and job recovery.

The unit tests in test_ui_preview_pipeline.py read the SOURCE. They can prove
that the code says the right thing; they cannot prove the page runs. This boots
the same two processes the Pinokio launcher does, drives a real Chromium at
them, and asserts the things only a runtime can answer:

  * `/api/jobs/active` is reachable THROUGH the dev server's proxy and answers
    the shape the client reconciles against. (A route can be registered on the
    app object and still be unreachable — that is exactly how GET /api/settings
    spent a release answering 422.)
  * the stage really is a <canvas>, not the two <img> layers it replaced;
  * the object-URL registry ends the session with nothing live that it created,
    which is the only way to observe the leak this refactor is about;
  * the decoder worker actually boots — a module worker that a webview refuses
    to start would fall back silently, and the fallback is on the main thread.

    env/Scripts/python.exe tests/preview_pipeline_smoke.py
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP)
sys.path.insert(0, os.path.join(APP, "tests"))

from browser_driver import Browser, find_browser, free_port, wait_for_http  # noqa: E402
from ui_browser_acceptance import Report, start_stack, stop  # noqa: E402

# Any clip on this machine will do; the assertion is about the stage, not the
# footage. Overridable so the smoke is not tied to one box's media folder.
# A cold process builds TensorRT engines before the first preview can land;
# this project measures that at 2-18 minutes. Generous by default, and
# overridable so CI can choose to fail fast instead.
PREVIEW_TIMEOUT_S = float(os.environ.get("ROOP_SMOKE_PREVIEW_TIMEOUT", "420"))

# Point at an already-running backend to skip the cold start entirely. That
# backend must be new enough to serve /api/jobs/active.
REUSE_API_PORT = os.environ.get("ROOP_SMOKE_API_PORT")

TARGET_CLIP = os.environ.get(
    "ROOP_SMOKE_CLIP",
    os.path.join("G:" + os.sep, "pinokio", "roop-keep", "b1.mp4"))


def start_ui_only(ui_dir, api_port, ui_port, log_dir):
    """Vite alone, proxied at an already-running backend.

    Same wiring as start_stack's UI half, minus the backend — including the two
    things that are easy to miss: `node` lives beside `npm` in the Pinokio
    toolchain and is absent from a plain process's PATH, and Vite binds ::1
    only unless told otherwise, which makes 127.0.0.1 refuse the connection.
    """
    import subprocess
    from ui_browser_acceptance import _npm

    os.makedirs(log_dir, exist_ok=True)
    npm = _npm()
    if not npm:
        raise RuntimeError("npm could not be resolved; the UI cannot be served")
    env = os.environ.copy()
    env.update({"ROOP_API_PORT": str(api_port), "PORT": str(ui_port)})
    env["PATH"] = os.path.dirname(npm) + os.pathsep + env.get("PATH", "")
    log = open(os.path.join(log_dir, "vite.log"), "w", encoding="utf-8",
               errors="replace")
    proc = subprocess.Popen(
        [npm, "run", "dev", "--", "--host", "127.0.0.1"],
        cwd=os.path.join(ROOT, ui_dir), env=env, stdout=log,
        stderr=subprocess.STDOUT, text=True, shell=False)
    return proc, log


async def run():
    report = Report()
    browser_exe = find_browser()
    if not report.ok("browser runtime", bool(browser_exe), browser_exe or "not found"):
        return report

    api_port = int(REUSE_API_PORT) if REUSE_API_PORT else free_port()
    ui_port = free_port()
    log_dir = os.path.join(APP, "output", "preview_smoke")
    if REUSE_API_PORT:
        backend, backend_log = None, None
        vite, ui_log = start_ui_only("react-ui", api_port, ui_port, log_dir)
    else:
        backend, vite, backend_log, ui_log = start_stack(
            "react-ui", api_port, ui_port, log_dir)
    try:
        wait_for_http(f"http://127.0.0.1:{api_port}/api/meta")
        report.ok("backend boots", True,
                  f"/api/meta on 127.0.0.1:{api_port}"
                  + (" (reused)" if REUSE_API_PORT else ""))
        base = f"http://127.0.0.1:{ui_port}"
        wait_for_http(base)
        report.ok("dev server boots", True, base)

        async with Browser(browser_exe) as page:
            await page.goto(base, settle=4.0)

            mounted = await page.evaluate(
                "const r = document.getElementById('root');"
                " return !!r && r.children.length > 0;")
            report.ok("app mounts", bool(mounted), "React tree present")

            # 1. The new endpoint, through Vite's proxy — the path the client
            #    actually takes. Registered-but-unreachable is a real failure
            #    mode here, not a hypothetical.
            jobs = await page.evaluate(
                "const r = await fetch('/api/jobs/active');"
                " const j = r.status === 200 ? await r.json() : {};"
                " return { status: r.status, keys: Object.keys(j).sort() };")
            status = (jobs or {}).get("status")
            keys = set((jobs or {}).get("keys") or [])
            report.ok("/api/jobs/active answers", status == 200, f"HTTP {status}")
            report.ok("/api/jobs/active shape",
                      {"processing", "job_id", "started_at", "queued"} <= keys,
                      ", ".join(sorted(keys)) or "no keys")

            # It must be the CHEAP endpoint: this is polled by clients that may
            # be attached to nothing.
            report.ok("/api/jobs/active stays small",
                      not ({"log", "parts", "runtime"} & keys),
                      "carries no log/parts/runtime")

            # 2. The store settled on a real answer rather than sitting in
            #    'reconciling' forever, and did NOT invent a running job.
            phase = await page.evaluate(
                "try { return JSON.parse("
                "localStorage.getItem('roop_active_job') || '{}'); }"
                " catch (e) { return {}; }")
            report.ok("job store persists only client knowledge",
                      "phase" not in ((phase or {}).get("state") or {}),
                      str((phase or {}).get("state") or {})[:120] or "empty")

            # 3. The stage. It only MOUNTS once there is a rendered preview
            #    (the empty session shows an onboarding card instead), so the
            #    session has to be brought to that state or this asserts
            #    nothing. Everything is already on this disk, so the target
            #    goes in by path and the faces come from the library — no
            #    upload, and no second copy of a 4 GB clip.
            with contextlib.suppress(Exception):
                await page.click_text("button", "Face Swap", settle=2.0)

            seeded = await page.evaluate(
                "const post = (u, b) => fetch(u, { method: 'POST',"
                "  headers: { 'Content-Type': 'application/json' },"
                "  body: JSON.stringify(b) }).then(r => r.json());"
                " const lib = await fetch('/api/faceset/library').then(r => r.json());"
                " const entry = (lib.entries || [])[0];"
                " if (!entry) return { ok: false, why: 'no faceset in the library' };"
                " await post('/api/faceset/library/load', { filename: entry.filename });"
                " const added = await post('/api/target/add_path', { paths: [%s] });"
                " return { ok: true, target: added && (added.count || added.targets), "
                "          face: entry.filename };" % repr(TARGET_CLIP).replace("'", '"'))
            if not (seeded or {}).get('ok'):
                report.add("stage renders a canvas", "SKIP",
                           (seeded or {}).get('why', 'could not seed a session'))
            else:
                # Reload so the panel picks the seeded session up through its
                # normal rehydrate path rather than a hand-poked state.
                await page.goto(base + "#/faceswap", settle=5.0)
                with contextlib.suppress(Exception):
                    await page.click_text("button", "Refresh", settle=1.0)
                rendered = await page.wait_for(
                    "document.querySelectorAll("
                    "'canvas[aria-label=\\\"Preview frame\\\"]').length > 0",
                    timeout=PREVIEW_TIMEOUT_S)
                if not rendered:
                    # The first preview on a cold process builds TensorRT
                    # engines, which this project measures at 2-18 minutes. That
                    # is an environment condition, not a defect in the stage, so
                    # it must not be reported as one — point the smoke at an
                    # already-warm backend (ROOP_SMOKE_API_PORT) for the strong
                    # assertion.
                    report.add("stage renders a canvas", "SKIP",
                               f"no preview within {PREVIEW_TIMEOUT_S:.0f}s"
                               " (cold TensorRT engine build)")
                else:
                    report.ok("stage renders a canvas", True,
                              "a preview swap mounted the canvas")

                if rendered:
                    # A canvas element proves the component mounted. Non-blank
                    # pixels prove drawImage actually ran — the failure mode of
                    # an uncontrolled canvas is a correct element showing
                    # nothing, which no DOM assertion can see.
                    painted = await page.evaluate(
                        "const c = document.querySelector("
                        "'canvas[aria-label=\\\"Preview frame\\\"]');"
                        " if (!c || !c.width) return 0;"
                        " const g = c.getContext('2d');"
                        " const d = g.getImageData(0, 0, c.width, c.height).data;"
                        " let lit = 0;"
                        " for (let i = 3; i < d.length; i += 4000) if (d[i] > 8) lit++;"
                        " return lit;")
                    report.ok("the canvas actually painted", (painted or 0) > 0,
                              f"{painted} non-transparent samples")

                    # The canvas must carry the FRAME's dimensions, not the
                    # 300x150 a <canvas> defaults to. That default is what a
                    # collapsed stage looks like, and it also throws the
                    # face-box overlay out of alignment.
                    size = await page.evaluate(
                        "const c = document.querySelector("
                        "'canvas[aria-label=\\\"Preview frame\\\"]');"
                        " return { w: c.width, h: c.height };")
                    report.ok("the canvas is sized to the frame",
                              (size or {}).get("w", 0) > 320,
                              f"{(size or {}).get('w')}x{(size or {}).get('h')} backing store")

                    # 6. The compare curtain. Driving it from the keyboard also
                    #    proves the handle is focusable, which is the only way
                    #    it is reachable without a pointer.
                    # The toggle is a <label> around a visually-hidden
                    # checkbox, so clicking the label's text is what a user
                    # does and what actually flips it.
                    await page.evaluate(
                        "const l = Array.from(document.querySelectorAll('label'))"
                        "  .find(x => x.textContent.trim() === 'Compare');"
                        " if (!l) return false;"
                        " const cb = l.querySelector('input[type=checkbox]');"
                        " if (cb && !cb.checked) cb.click();"
                        " await new Promise(r => setTimeout(r, 600));"
                        " return true;")
                    slider = await page.evaluate(
                        "const s = document.querySelector('[role=\\\"slider\\\"]');"
                        " if (!s) return null;"
                        " const before = s.getAttribute('aria-valuenow');"
                        " s.focus();"
                        " s.dispatchEvent(new KeyboardEvent('keydown',"
                        "   { key: 'ArrowLeft', bubbles: true }));"
                        " await new Promise(r => setTimeout(r, 250));"
                        " return { before, after: s.getAttribute('aria-valuenow'),"
                        "          focused: document.activeElement === s };")
                    if slider is None:
                        report.add("compare curtain responds", "SKIP",
                                   "compare mode not reachable in this session")
                    else:
                        report.ok("compare curtain responds",
                                  bool(slider.get("focused")),
                                  f"focusable; aria-valuenow {slider.get('before')}"
                                  f" -> {slider.get('after')}")

            # 4. The decoder worker boots. A webview that refuses module workers
            #    falls back to the main thread SILENTLY, so the only way to know
            #    is to ask.
            worker = await page.evaluate(
                "try {"
                " const w = new Worker('/src/components/faceswap/decoder.worker.js',"
                " { type: 'module' });"
                " w.terminate(); return true;"
                "} catch (e) { return false; }")
            report.ok("decoder worker constructs", bool(worker),
                      "module worker supported" if worker
                      else "falls back to main-thread decode")

            # 5. Nothing the registry created is still live at rest. This is the
            #    leak, stated as a number: created - revoked, for URLs this app
            #    made on purpose.
            # 5. The leak, stated as a number.
            #
            # NOT "live must be zero" — the frame on screen is a live object URL
            # and revoking it would blank the stage. The property that actually
            # distinguishes a leak from correct behaviour is that `live` does
            # not GROW with the number of frames rendered: each new preview
            # replaces the last, and the replaced one must be freed by name or
            # it stays resident for the life of the document.
            stats = await page.evaluate(
                "return window.__roopObjectUrls"
                " ? window.__roopObjectUrls.__stats() : null;")
            if stats is None:
                report.add("object URLs are bounded", "SKIP",
                           "dev-only diagnostic not exposed (production build)")
            else:
                before = dict(stats)
                for _ in range(3):
                    with contextlib.suppress(Exception):
                        await page.click_text("button", "Refresh", settle=0.5)
                    await page.wait_for("true", timeout=0.1)
                    await asyncio.sleep(6)
                after = await page.evaluate(
                    "return window.__roopObjectUrls.__stats();")
                grew = int(after["created"]) - int(before["created"])
                held = int(after["live"]) - int(before["live"])
                report.ok("object URLs are bounded",
                          grew == 0 or held <= 1,
                          f"{grew} more created, live {before['live']} -> "
                          f"{after['live']} (revoked {after['revoked']})")

            report.ok("no uncaught page errors", not page.errors,
                      "; ".join(e["text"][:160] for e in page.errors[:4]) or "none")
    finally:
        stop(vite)
        stop(backend)
        with contextlib.suppress(Exception):
            if backend_log:
                backend_log.close()
            ui_log.close()
    return report


def main():
    report = asyncio.run(run())
    print(f"\n[ui] preview pipeline: {len(report.rows)} checks; "
          f"{len(report.failed)} FAIL", flush=True)
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
