/**
 * Acceptance check: does the shipped UI load and work when served by the
 * BACKEND, with no Vite process anywhere?
 *
 * This is the cross-device regression this file exists for. The launcher used
 * to run `vite preview` as a second server and proxy /api and /ws back to the
 * backend, which put a Node toolchain on the runtime path: if `vite build`
 * failed on a machine -- wrong Node major, missing per-platform rolldown
 * binary, no `dist/` on a fresh clone -- `vite preview` refused to start, the
 * launcher's URL matcher never fired, and the user saw a Vite error and no UI.
 *
 * app/api.py now serves react-ui/dist itself, so there is exactly one server
 * and one origin. That removes the proxy entirely, which is the part most
 * worth proving: /api and /ws/telemetry have to keep working with no proxy in
 * front of them, SPA deep links have to fall back to index.html, and a /api
 * miss must still be a real 404 rather than a 200 full of HTML.
 *
 * Runs the exact production path from start_react.js:
 *
 *     uvicorn api:app     (the real backend, serving the real built client)
 *     Chromium            (the real browser)
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
    const s = net.createServer();
    s.listen(0, '127.0.0.1', () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
  });

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

let failures = 0;
const check = (name, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  ${detail}` : ''}`);
  if (!ok) failures++;
};

const PORT = await freePort();

const driver = `
import os, sys, time
sys.path.insert(0, r"${path.join(REPO, 'app').replace(/\\/g, '\\\\')}")
os.environ["ROOP_REACT_CLIENT"] = "1"
import api
from fastapi import Request

@api.app.post("/api/__test__/advance")
async def _advance(request: Request):
    b = await request.json()
    api._progress.update({"processing": True, "progress": b["progress"],
                          "desc": b["desc"], "error": ""})
    api._run_stats.update({"start": time.time() - 30,
                           "frames_done": b["done"], "frames_total": b["total"]})
    return {"ok": True}

import uvicorn
uvicorn.run(api.app, host="127.0.0.1", port=${PORT}, log_level="error")
`;

console.log(`single server -> 127.0.0.1:${PORT} (backend serves API + UI)`);
const server = spawn(PY, ['-c', driver], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
let log = '';
server.stdout.on('data', (d) => { log += d; });
server.stderr.on('data', (d) => { log += d; });

const base = `http://127.0.0.1:${PORT}`;
let up = false;
for (let i = 0; i < 180; i++) {
  try {
    if ((await fetch(`${base}/api/telemetry/status`)).ok) { up = true; break; }
  } catch { /* booting */ }
  await wait(500);
}
if (!up) { console.log('FAIL: backend never came up\n' + log.slice(-2000)); server.kill(); process.exit(1); }
console.log('up.\n');

// --- routing contract, with no proxy in front of anything ----------------
const uiStatus = await (await fetch(`${base}/api/ui/status`)).json();
check('the backend reports a production build on disk', uiStatus.built === true, uiStatus.dist);

const root = await fetch(`${base}/`);
check('the backend serves the SPA at /', root.ok && /text\/html/.test(root.headers.get('content-type')));

const deep = await fetch(`${base}/some/deep/link`);
check('an SPA deep link falls back to index.html',
  deep.ok && /text\/html/.test(deep.headers.get('content-type')), `status=${deep.status}`);

const missingApi = await fetch(`${base}/api/definitely_not_a_route`);
check('an unknown /api route is still a real 404, not index.html',
  missingApi.status === 404, `status=${missingApi.status}`);

const meta = await fetch(`${base}/api/meta`);
check('a real API route is not shadowed by the SPA mount',
  meta.ok && /application\/json/.test(meta.headers.get('content-type')));

// --- the real browser, on the real single origin -------------------------
const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage();

const sockets = [];
page.on('websocket', (ws) => {
  const rec = { url: ws.url(), frames: [] };
  sockets.push(rec);
  ws.on('framereceived', (d) => {
    try { rec.frames.push(JSON.parse(d.payload)); } catch { /* non-JSON */ }
  });
});
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
const failedRequests = [];
page.on('requestfailed', (r) => failedRequests.push(`${r.url()} ${r.failure()?.errorText}`));

await page.goto(`${base}/`, { waitUntil: 'load' });
await wait(4000);

// The app must actually MOUNT. A served-but-broken bundle still returns 200 for
// index.html, so the HTTP checks above cannot tell a working UI from a blank
// page -- only the rendered DOM can.
const rootHtml = await page.evaluate(() => document.getElementById('root')?.innerHTML.length ?? 0);
check('the React app mounts (#root is populated)', rootHtml > 0, `${rootHtml} chars`);

const bodyText = await page.evaluate(() => document.body.innerText);
check('the UI renders its real chrome', /Face Swap/.test(bodyText),
  bodyText.slice(0, 80).replace(/\s+/g, ' '));

check('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));
check('no failed asset requests', failedRequests.length === 0, failedRequests.slice(0, 2).join(' | '));

// The lazy route chunk is a separate HTTP request; a broken static mount breaks
// code-splitting specifically, so exercise a lazily-loaded tab.
const settingsTab = page.getByRole('button', { name: 'Settings', exact: true });
check('the Settings tab is present', await settingsTab.count() === 1);
await settingsTab.click();
let settingsText = '';
for (let i = 0; i < 50; i++) {
  settingsText = await page.evaluate(() => document.body.innerText);
  if (/Apply Settings/.test(settingsText)) break;
  await wait(200);
}
check('a lazily code-split route loads from the static mount',
  /Apply Settings/.test(settingsText));

await page.getByRole('button', { name: 'Face Swap', exact: true }).click();
await wait(800);

// --- telemetry, with no /ws proxy ---------------------------------------
check('the shipped app opens a telemetry socket by itself',
  sockets.some((s) => s.url.includes('/ws/telemetry')),
  sockets.length ? sockets.map((s) => s.url).join(', ') : 'no sockets opened');

const tele = sockets.find((s) => s.url.includes('/ws/telemetry'));
check('the WebSocket upgrade succeeds same-origin (frames arrive)',
  Boolean(tele && tele.frames.length), tele ? `${tele.frames.length} frame(s)` : '');
check('first frame is the hello greeting',
  Boolean(tele && tele.frames[0] && tele.frames[0].event === 'hello'));

await fetch(`${base}/api/__test__/advance`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ progress: 0.55, desc: '165 / 300', done: 165, total: 300 }),
});
let advanced = null;
for (let i = 0; i < 60; i++) {
  // Look across EVERY socket the page opened, not just the first. The hook
  // reconnects on its own schedule, so pinning this to one socket makes the
  // check flaky in a way that has nothing to do with what it is testing.
  for (const s of sockets) {
    advanced = s.frames.find((f) => f.current_frame === 165);
    if (advanced) break;
  }
  if (advanced) break;
  await wait(100);
}
check('a server-side change reaches the shipped client', Boolean(advanced));

let shown = '';
for (let i = 0; i < 30; i++) {
  shown = await page.evaluate(() => document.body.innerText);
  if (/165\s*\/\s*300|55\s*%/.test(shown)) break;
  await wait(100);
}
check('the UI renders the pushed progress', /165\s*\/\s*300|55\s*%|5[0-9]%/.test(shown),
  shown.match(/\d+\s*\/\s*300|\d+%/)?.[0] ?? '(no counter found in DOM)');

await browser.close();

const killTree = (child) =>
  new Promise((resolve) => {
    if (!child.pid) return resolve();
    const k = spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
    k.on('close', resolve);
    k.on('error', resolve);
  });
await killTree(server);

console.log(`\n${failures === 0 ? 'BACKEND-SERVED UI VERIFIED END TO END' : failures + ' CHECK(S) FAILED'}`);
process.exit(failures ? 1 : 0);
