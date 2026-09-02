"""Drive a real Chromium over the DevTools Protocol, with no new dependencies.

WHY THIS EXISTS.  Every acceptance gate from Stage 13 onward records the same
sentence -- "browser interaction could not be verified because no browser
runtime was available" -- and on that basis marks V2 launch, routing, controls,
themes, preview, telemetry, queue and pause/resume as BLOCKED.  That is the
single largest unverified block in the whole matrix.

The premise turned out to be false.  Windows ships Edge, this host also has
Chrome, and `websockets` is already installed in `app/env`.  A DevTools
Protocol client is therefore reachable without installing Playwright or
Selenium -- which matters, because the validated environment is itself under
test and adding packages to it would invalidate the dependency check that
`update_health.py` performs.

WHAT THIS IS NOT.  It is not a human looking at the screen.  It renders real
pages in a real engine, executes the real JavaScript, dispatches real clicks
and captures real screenshots, so it closes "does the UI run and respond".  It
does not close visual/aesthetic review, which stays NOT TESTED.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import websockets

# Ordered by preference.  Edge is present on every Windows 11 install, so the
# harness must never depend on Chrome specifically.
_BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


def find_browser():
    """Absolute path to a usable Chromium, or None."""
    for name in ("chrome", "msedge", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    for path in _BROWSERS:
        if os.path.isfile(path):
            return path
    return None


def free_port():
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


class Page:
    """One DevTools page target with a single multiplexed connection."""

    def __init__(self, socket_):
        self._socket = socket_
        self._next = 0
        self._pending = {}
        self._pump = None
        # Everything the page said about itself, kept for the report.
        self.console = []
        self.errors = []

    async def _read(self):
        try:
            async for raw in self._socket:
                message = json.loads(raw)
                if "id" in message:
                    future = self._pending.pop(message["id"], None)
                    if future is not None and not future.done():
                        future.set_result(message)
                    continue
                self._on_event(message)
        except Exception:
            pass

    def _on_event(self, message):
        method = message.get("method")
        params = message.get("params") or {}
        if method == "Runtime.consoleAPICalled":
            text = " ".join(
                str(arg.get("value", arg.get("description", "")))
                for arg in params.get("args", []))
            entry = {"level": params.get("type"), "text": text}
            self.console.append(entry)
            if params.get("type") in ("error", "assert"):
                self.errors.append(entry)
        elif method == "Runtime.exceptionThrown":
            detail = params.get("exceptionDetails", {})
            text = (detail.get("exception", {}).get("description")
                    or detail.get("text") or "uncaught exception")
            entry = {"level": "exception", "text": text}
            self.console.append(entry)
            self.errors.append(entry)
        elif method == "Log.entryAdded":
            entry_ = params.get("entry", {})
            # The URL is the whole diagnosis for a network entry: "404" alone
            # cannot distinguish a missing favicon from a missing API route.
            where = entry_.get("url") or ""
            entry = {"level": entry_.get("level"), "url": where,
                     "text": f"{entry_.get('source')}: {entry_.get('text')}"
                             + (f" [{where}]" if where else "")}
            self.console.append(entry)
            if entry_.get("level") == "error":
                self.errors.append(entry)

    async def send(self, method, **params):
        self._next += 1
        message_id = self._next
        future = asyncio.get_running_loop().create_future()
        self._pending[message_id] = future
        await self._socket.send(json.dumps(
            {"id": message_id, "method": method, "params": params}))
        reply = await asyncio.wait_for(future, timeout=60)
        if "error" in reply:
            raise RuntimeError(f"{method}: {reply['error']}")
        return reply.get("result", {})

    async def start(self):
        self._pump = asyncio.create_task(self._read())
        await self.send("Page.enable")
        await self.send("Runtime.enable")
        await self.send("Log.enable")

    async def goto(self, url, settle=1.0):
        await self.send("Page.navigate", url=url)
        await self.wait_for("document.readyState === 'complete'", timeout=45)
        await asyncio.sleep(settle)

    async def evaluate(self, expression):
        """Evaluate and return a JSON-able value.

        The wrapper is an ASYNC IIFE so callers may `await` inside it -- a
        fetch against the app's own origin, or a settle delay between clicks.
        `awaitPromise` then resolves the returned promise before serialising.
        A synchronous body is unaffected.
        """
        result = await self.send(
            "Runtime.evaluate", expression=f"(async function(){{ {expression} }})()",
            returnByValue=True, awaitPromise=True)
        details = result.get("exceptionDetails")
        if details:
            raise RuntimeError(details.get("text") or "evaluation failed")
        return result.get("result", {}).get("value")

    async def wait_for(self, condition, timeout=20.0, interval=0.2):
        """Poll a JS boolean expression until true; False on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                if await self.evaluate(f"return !!({condition});"):
                    return True
            await asyncio.sleep(interval)
        return False

    async def click(self, selector, settle=0.4):
        """Click via a real DOM event on the first matching element."""
        clicked = await self.evaluate(
            f"const el = document.querySelector({json.dumps(selector)});"
            " if (!el) return false; el.click(); return true;")
        if clicked:
            await asyncio.sleep(settle)
        return bool(clicked)

    async def click_text(self, selector, text, settle=0.4):
        """Click the first element matching `selector` whose text contains `text`."""
        clicked = await self.evaluate(
            f"const nodes = Array.from(document.querySelectorAll({json.dumps(selector)}));"
            f" const hit = nodes.find((n) => (n.textContent || '').trim().toLowerCase()"
            f".includes({json.dumps(text.lower())}));"
            " if (!hit) return false; hit.click(); return true;")
        if clicked:
            await asyncio.sleep(settle)
        return bool(clicked)

    async def screenshot(self, path):
        result = await self.send("Page.captureScreenshot", format="png")
        data = result.get("data")
        if not data:
            return False
        import base64
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(base64.b64decode(data))
        return True

    async def set_viewport(self, width, height):
        await self.send("Emulation.setDeviceMetricsOverride", width=int(width),
                        height=int(height), deviceScaleFactor=1, mobile=False)


class Browser:
    """A headless Chromium child process, torn down on exit."""

    def __init__(self, binary=None, width=1440, height=900):
        self.binary = binary or find_browser()
        self.width = width
        self.height = height
        self._process = None
        self._profile = None
        self._socket = None
        self.port = None

    async def __aenter__(self):
        if not self.binary:
            raise RuntimeError("no Chromium-based browser was found on this host")
        self.port = free_port()
        self._profile = tempfile.mkdtemp(prefix="roop-cdp-")
        self._process = subprocess.Popen(
            [self.binary, "--headless=new", "--disable-gpu",
             "--no-first-run", "--no-default-browser-check",
             "--disable-extensions", "--disable-background-networking",
             "--disable-features=Translate,MediaRouter",
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

    async def _await_target(self, timeout=45.0):
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError("the browser process exited during startup")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/json/list", timeout=2) as response:
                    targets = json.loads(response.read().decode("utf-8"))
                for entry in targets:
                    if entry.get("type") == "page" and entry.get("webSocketDebuggerUrl"):
                        return entry["webSocketDebuggerUrl"]
            except (OSError, urllib.error.URLError, ValueError) as exc:
                last = str(exc)
            await asyncio.sleep(0.25)
        raise RuntimeError(f"the browser never exposed a page target: {last}")

    async def __aexit__(self, *_):
        with contextlib.suppress(Exception):
            await self._socket.close()
        if self._process is not None:
            with contextlib.suppress(Exception):
                self._process.terminate()
                self._process.wait(timeout=15)
            if self._process.poll() is None:
                with contextlib.suppress(Exception):
                    self._process.kill()
        if self._profile:
            shutil.rmtree(self._profile, ignore_errors=True)
        return False


def wait_for_http(url, timeout=180.0):
    """Block until `url` answers, or return False."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if int(response.status) < 500:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.5)
    return False
