"""The built web UI, driven in a real headless Chromium against the real server.

Unit tests render components in jsdom; this renders the production bundle in
a real engine and clicks through it like a user: load a project, detect,
assign one person, preview, render, watch the output play, then start and
stop a second render. Any console error or uncaught exception fails the test
(roop-ultimate once shipped a UI that passed build, unit tests and lint but
crashed on first render).

Needs: ``web_ui/dist`` (``npm run build`` in ``web_ui``), a Chromium-based
browser, the models. Uses roop-ultimate's dependency-free DevTools driver
(``app/tests/browser_driver.py``). Screenshots land in ``web_ui/e2e-shots``.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# The backend test's fixture clip (a one-person source crop + a six-person video).
try:
    from face_engine.tests.test_server import media  # noqa: F401
except (ImportError, Exception):
    media = None

REPO = Path(__file__).resolve().parents[2]
DIST = REPO / "web_ui" / "dist"
SHOTS = REPO / "web_ui" / "e2e-shots"
sys.path.insert(0, str(REPO / "app" / "tests"))

pytestmark = pytest.mark.gpu


def _driver():  # type: ignore[no-untyped-def]
    import browser_driver

    return browser_driver


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    if not (DIST / "index.html").is_file():
        pytest.fail("web_ui/dist is missing: run `npm run build` in web_ui first")
    drv = _driver()
    if drv.find_browser() is None:
        pytest.skip("no Chromium-based browser on this host")
    # The live preview starts a TensorRT build of the swapper when no compiled engine exists
    # (~4 min for hyperswap_1a_256 on an RTX 4070), longer than this test waits for the preview.
    if not list((REPO / ".cache" / "trt_engines").glob("hyperswap_1a_256_sm*_fp16_b*.engine")):
        pytest.skip("no compiled hyperswap_1a_256 engine; run "
                    "`python tools/compile_engines.py --models swapper` once (minutes)")
    from face_engine.tests.test_server import (
        _analysis,  # noqa: F401 - ensures models resolve
    )
    workspace = tmp_path_factory.mktemp("e2e_ws")
    port = drv.free_port()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    # A log FILE, not a pipe: nobody drains a pipe here, and once ONNX Runtime's
    # warnings fill it (~64 KB) the server blocks on its next write.
    log_path = workspace / "server.log"
    log = open(log_path, "wb")  # noqa: SIM115 - closed below
    proc = subprocess.Popen([sys.executable, "-m", "face_engine.server", "--port", str(port),
                             "--workspace", str(workspace), "--ui", str(DIST)],
                            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    if not drv.wait_for_http(url + "/api/options", timeout=120):
        proc.kill()
        log.close()
        pytest.fail("server did not start: " + log_path.read_text(errors="replace")[-2000:])
    yield url
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


async def _set_files(page, selector: str, paths: list[Path]) -> None:  # type: ignore[no-untyped-def]
    doc = await page.send("DOM.getDocument", depth=-1)
    node = await page.send("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector=selector)
    await page.send("DOM.setFileInputFiles", nodeId=node["nodeId"], files=[str(p) for p in paths])


async def _select(page, selector: str, value: str) -> bool:  # type: ignore[no-untyped-def]
    """Set a <select> the way a user does: value + a real 'change' event React sees."""
    return await page.evaluate(
        f"const el = document.querySelector({json.dumps(selector)}); if (!el) return false;"
        " const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;"
        f" set.call(el, {json.dumps(value)});"
        " el.dispatchEvent(new Event('change', { bubbles: true })); return true;")


async def _flow(url: str, media: dict[str, Path]) -> dict[str, object]:  # noqa: F811
    drv = _driver()
    found: dict[str, object] = {}
    async with drv.Browser(width=1600, height=1000) as page:
        await page.goto(url, settle=1.5)
        assert await page.wait_for("document.querySelector('h1')?.textContent === 'face_engine'")
        assert await page.wait_for(
            "document.querySelector('[data-testid=telemetry-status]')?.textContent === 'live'", 20)

        await _set_files(page, "input[name=sources]", [media["source"]])
        await _set_files(page, "input[name=target]", [media["target"]])
        assert await page.click_text("button", "Load project")
        assert await page.wait_for("document.querySelectorAll('li[data-person]').length === 6", 180), \
            await page.evaluate("return document.body.innerText.slice(0, 2000);")
        found["people"] = 6

        # Assign the source to the leftmost person (people are sorted by count, so
        # find the leftmost by the thumbnail's bbox order is not exposed: use the API).
        people = await page.evaluate(
            "const r = await fetch('/api/project'); return (await r.json()).people;")
        leftmost = min(people, key=lambda p: p["bbox"][0])
        src = (await page.evaluate("const r = await fetch('/api/project'); return (await r.json()).sources;"))[0]
        assert await _select(page, f"select[aria-label='Source for person {leftmost['id']}']", src["id"])
        assert await page.wait_for(
            "document.querySelector('[data-testid=swap-mode]')?.textContent.includes('1 of 6')", 20)
        assert await _select(page, "#provider", "cuda")

        # Live previews already run (after load, the assignment and the provider
        # change); wait until the button is idle again, then ask explicitly.
        assert await page.wait_for(
            "[...document.querySelectorAll('button')].some("
            "b => b.textContent === 'Preview frame' && !b.disabled)", 180)
        assert await page.click_text("button", "Preview frame")
        assert await page.wait_for(
            "(() => { const i = document.querySelector('[data-testid=right-media]');"
            " return i && i.complete && i.naturalWidth > 0; })()", 180)
        await page.screenshot(str(SHOTS / "1-preview.png"))

        # Scrub the timeline: a debounced POST /api/preview/frame renders frame 20.
        await page.evaluate(
            "const r = document.querySelector('input[aria-label=Frame]');"
            " const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;"
            " set.call(r, '20'); r.dispatchEvent(new Event('input', {bubbles: true}));"
            " return true;")
        assert await page.wait_for(
            "document.querySelector('[data-testid=player-badge]')?.textContent.includes('frame 21')", 120),             await page.evaluate("return document.body.innerText.slice(0, 2000);")
        found["scrub_badge"] = await page.evaluate(
            "return document.querySelector('[data-testid=player-badge]').textContent;")

        assert await page.click_text("button", "Start render")
        assert await page.wait_for(
            "document.querySelector('[data-testid=header-job-state]')?.textContent === 'completed'", 600), \
            await page.evaluate("return document.body.innerText.slice(0, 3000);")
        # The output video is loaded and drawn into the comparison canvas.
        assert await page.wait_for(
            "(() => { const v = document.querySelector('video[data-testid=right-media]');"
            " return v && v.readyState >= 2 && v.videoWidth === 960; })()", 60)
        await page.click_text("button", "Play")
        await asyncio.sleep(1.5)
        drawn = await page.evaluate(
            "const c = document.querySelector('[data-testid=split-canvas]');"
            " const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;"
            " let lit = 0; for (let i = 0; i < d.length; i += 4000) lit += d[i] + d[i+1] + d[i+2] > 30;"
            " return {w: c.width, h: c.height, lit};")
        assert drawn["w"] == 960 and drawn["lit"] > 50, drawn
        found["canvas"] = drawn
        await page.click_text("button", "Pause")
        await page.click("[aria-label='Comparison mode'] button[aria-pressed=false]")  # side-by-side
        await asyncio.sleep(0.5)
        await page.screenshot(str(SHOTS / "2-rendered.png"))

        # Start again and stop from the UI.
        await page.click("[aria-label='Comparison mode'] button[aria-pressed=false]")  # back to split
        assert await _select(page, "#enhancer", "gpen_bfr_512")
        assert await page.click_text("button", "Start render")
        assert await page.wait_for(
            "document.querySelector('[data-testid=header-job-state]')?.textContent === 'rendering'", 300)
        assert await page.click_text("button", "Stop render")
        assert await page.wait_for(
            "document.querySelector('[data-testid=header-job-state]')?.textContent === 'cancelled'", 120)
        await page.screenshot(str(SHOTS / "3-stopped.png"))
        found["errors"] = list(page.errors)
    return found


def test_web_ui_end_to_end(server: str, media: dict[str, Path]) -> None:  # noqa: F811
    result = asyncio.run(_flow(server, media))
    assert result["errors"] == [], result["errors"]
