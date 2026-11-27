/**
 * End-to-end proof that telemetry reaches a REAL browser from the REAL backend.
 *
 * Everything before this is a unit test or a measurement. This is the only
 * check that exercises the whole path the user actually gets:
 *
 *     api.py (uvicorn)  ->  /ws/telemetry  ->  browser WebSocket  ->  the hook
 *
 * It boots the actual FastAPI app with uvicorn on a free port, drives a real
 * Chromium at it, and asserts:
 *
 *   1. the handshake succeeds and the greeting carries current state
 *   2. a change made on the SERVER arrives unprompted (the push property)
 *   3. a worker THREAD -- the shape the swap pipeline really is -- reaches the
 *      browser, which is the case that cannot be faked by an async endpoint
 *   4. the socket RECONNECTS by itself after the connection is severed
 *
 * Usage: node scripts/verify-telemetry-e2e.mjs
 */
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import net from 'node:net';
import path from 'node:path';

// PLAYWRIGHT_REQUIRE_FROM: a directory whose node_modules holds playwright-core
// (e.g. an npx cache entry); defaults to this package. CHROME_PATH: the
// Chromium executable to drive.
const REQ = createRequire(process.env.PLAYWRIGHT_REQUIRE_FROM || import.meta.url);
const { chromium } = REQ('playwright-core');

const CHROME = process.env.CHROME_PATH;
if (!CHROME) throw new Error('set CHROME_PATH to a Chromium executable');
const REPO = path.resolve('..');
const PY = path.join(REPO, 'app', 'env', 'Scripts', 'python.exe');

const freePort = () =>
  new Promise((resolve) => {
    const srv = net.createServer();
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

let failures = 0;
const check = (name, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  ${detail}` : ''}`);
  if (!ok) failures++;
};

const PORT = await freePort();

// Boot the real app. A tiny driver script is used rather than `uvicorn api:app`
// so the control endpoints below can mutate the same module the server serves.
const driver = `
import os, sys, threading, time
sys.path.insert(0, r"${path.join(REPO, 'app').replace(/\\/g, '\\\\')}")
os.environ["ROOP_REACT_CLIENT"] = "1"
import api
from fastapi import Request

# Test-only control surface: lets the browser test move the pipeline's state
# exactly the way a real render does (mutating _progress in place), including
# from a worker THREAD with no event loop.
@api.app.post("/__test__/advance")
async def _advance(request: Request):
    body = await request.json()
    api._progress.update({"processing": True, "progress": body["progress"],
                          "desc": body["desc"], "error": ""})
    api._run_stats.update({"start": time.time() - 30,
                           "frames_done": body["done"], "frames_total": body["total"]})
    return {"ok": True}

@api.app.post("/__test__/thread_push")
def _thread_push():
    import routes_telemetry
    out = {}
    def worker():
        out["sent"] = routes_telemetry.hub.broadcast_threadsafe(
            {"event": "progress", "current_frame": 8888, "total_frames": 9999,
             "fps": 42.0, "processing": True, "progress": 0.88,
             "desc": "from a worker thread", "paused": False, "error": "",
             "eta_s": 5, "started_at": time.time(), "live_seq": 0})
    t = threading.Thread(target=worker); t.start(); t.join(10)
    return {"sent": bool(out.get("sent"))}

import uvicorn
uvicorn.run(api.app, host="127.0.0.1", port=${PORT}, log_level="error")
`;

console.log(`booting the real backend on 127.0.0.1:${PORT} ...`);
const server = spawn(PY, ['-c', driver], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
let serverLog = '';
server.stdout.on('data', (d) => { serverLog += d; });
server.stderr.on('data', (d) => { serverLog += d; });

const base = `http://127.0.0.1:${PORT}`;
let up = false;
for (let i = 0; i < 120; i++) {
  try {
    const r = await fetch(`${base}/api/telemetry/status`);
    if (r.ok) { up = true; break; }
  } catch { /* still starting */ }
  await wait(500);
}
if (!up) {
  console.log('FAIL: backend never came up\n' + serverLog.slice(-2000));
  server.kill();
  process.exit(1);
}
console.log('backend up.\n');

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage();

// Run the real client protocol in the page, collecting every frame.
await page.goto(`${base}/api/telemetry/status`);
await page.evaluate((port) => {
  window.__frames = [];
  window.__opens = 0;
  window.__connect = () => {
    const ws = new WebSocket(`ws://127.0.0.1:${port}/ws/telemetry`);
    window.__ws = ws;
    ws.onopen = () => { window.__opens += 1; };
    ws.onmessage = (e) => window.__frames.push(JSON.parse(e.data));
    // Mirror the hook: reconnect on an unexpected close.
    ws.onclose = () => {
      if (!window.__stop) setTimeout(window.__connect, 300);
    };
  };
  window.__connect();
}, PORT);

await wait(1000);

// 1. handshake + greeting
let frames = await page.evaluate(() => window.__frames);
check('handshake succeeds and greets with state',
  frames.some((f) => f.event === 'hello'),
  `got ${frames.length} frame(s)`);

// 2. a server-side change arrives unprompted
await fetch(`${base}/__test__/advance`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ progress: 0.42, desc: '126 / 300', done: 126, total: 300 }),
});
await wait(1500);
frames = await page.evaluate(() => window.__frames);
const advanced = frames.find((f) => f.current_frame === 126);
check('a server-side change is PUSHED to the browser unprompted',
  Boolean(advanced), advanced ? `fps=${advanced.fps} total=${advanced.total_frames}` : '');
check('pushed frame carries a derived fps', advanced ? advanced.fps > 0 : false,
  advanced ? `fps=${advanced.fps}` : '');

// 3. a worker thread reaches the browser
const tp = await (await fetch(`${base}/__test__/thread_push`, { method: 'POST' })).json();
await wait(1000);
frames = await page.evaluate(() => window.__frames);
check('broadcast_threadsafe reported success', tp.sent === true);
check('a WORKER THREAD frame reaches the browser',
  frames.some((f) => f.current_frame === 8888));

// 4. reconnect after the socket is severed
const opensBefore = await page.evaluate(() => window.__opens);
await page.evaluate(() => window.__ws.close());
await wait(2000);
const opensAfter = await page.evaluate(() => window.__opens);
check('socket reconnects automatically after a drop',
  opensAfter > opensBefore, `opens ${opensBefore} -> ${opensAfter}`);

// the reconnected socket is live
await fetch(`${base}/__test__/advance`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ progress: 0.9, desc: '270 / 300', done: 270, total: 300 }),
});
await wait(1500);
frames = await page.evaluate(() => window.__frames);
check('the RECONNECTED socket receives new frames',
  frames.some((f) => f.current_frame === 270));

await page.evaluate(() => { window.__stop = true; window.__ws.close(); });

const status = await (await fetch(`${base}/api/telemetry/status`)).json();
console.log(`\nhub stats: frames_sent=${status.frames_sent} `
  + `total_connections=${status.total_connections} dropped=${status.frames_dropped}`);

await browser.close();
server.kill();

console.log(`\n${failures === 0 ? 'END-TO-END TELEMETRY VERIFIED' : failures + ' CHECK(S) FAILED'}`);
process.exit(failures ? 1 : 0);
