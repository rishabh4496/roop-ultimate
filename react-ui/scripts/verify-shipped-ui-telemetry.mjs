/**
 * Acceptance check: does the REAL shipped UI connect and receive telemetry?
 *
 * verify-telemetry-e2e.mjs proves the SERVER pushes correctly, but it drives a
 * hand-written socket in the page. That leaves the actual question unanswered:
 * does the built `useTelemetrySocket` bundle -- the code the user really runs,
 * through the real Vite proxy -- connect, filter frames and update the UI?
 *
 * This runs the exact production path from start_react.js:
 *
 *     uvicorn api:app            (the real backend)
 *     vite preview --proxy       (the real built client, real /ws proxy)
 *     Chromium                   (the real browser)
 *
 * and asserts the page's own WebSocket -- opened by the shipped hook, not by
 * this script -- carries a server-side change through to the DOM.
 */
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import net from 'node:net';
import path from 'node:path';

const REQ = createRequire('G:/pinokio/cache/npm_config_cache/_npx/e41f203b7505f1fb/');
const { chromium } = REQ('playwright-core');

const CHROME = 'C:/Users/rishr/AppData/Local/ms-playwright/chromium-1234/chrome-win64/chrome.exe';
const UI = path.resolve('.');
const REPO = path.resolve('..');
const PY = path.join(REPO, 'app', 'env', 'Scripts', 'python.exe');
// npm.cmd ships inside Pinokio's miniforge env, not in bin/npm (which holds
// only pterm/bun/claude shims). Verified with a directory listing.
const NPM = 'G:\\pinokio\\bin\\miniforge\\npm.cmd';

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

const API_PORT = await freePort();
const UI_PORT = await freePort();

// --- real backend --------------------------------------------------------
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
uvicorn.run(api.app, host="127.0.0.1", port=${API_PORT}, log_level="error")
`;

console.log(`backend  -> 127.0.0.1:${API_PORT}`);
const server = spawn(PY, ['-c', driver], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
let log = '';
server.stdout.on('data', (d) => { log += d; });
server.stderr.on('data', (d) => { log += d; });

let up = false;
for (let i = 0; i < 120; i++) {
  try {
    if ((await fetch(`http://127.0.0.1:${API_PORT}/api/telemetry/status`)).ok) { up = true; break; }
  } catch { /* booting */ }
  await wait(500);
}
if (!up) { console.log('FAIL: backend never came up\n' + log.slice(-1500)); server.kill(); process.exit(1); }

// --- real built client, served exactly as the launcher serves it ---------
console.log(`ui       -> 127.0.0.1:${UI_PORT} (vite preview, real /ws proxy)`);
const ui = spawn(NPM, ['run', 'preview', '--', '--host', '127.0.0.1', '--port', String(UI_PORT)], {
  cwd: UI,
  env: { ...process.env, ROOP_API_PORT: String(API_PORT), PORT: String(UI_PORT) },
  stdio: ['ignore', 'pipe', 'pipe'],
  shell: true,
});
let uiLog = '';
ui.stdout.on('data', (d) => { uiLog += d; });
ui.stderr.on('data', (d) => { uiLog += d; });

let uiUp = false;
for (let i = 0; i < 120; i++) {
  try {
    if ((await fetch(`http://127.0.0.1:${UI_PORT}/`)).ok) { uiUp = true; break; }
  } catch { /* booting */ }
  await wait(500);
}
if (!uiUp) { console.log('FAIL: UI never came up\n' + uiLog.slice(-1500)); server.kill(); ui.kill(); process.exit(1); }
console.log('both up.\n');

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage();

// Observe the socket the PAGE opens. Nothing here creates one.
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

await page.goto(`http://127.0.0.1:${UI_PORT}/`, { waitUntil: 'load' });
await wait(4000);

check('the shipped app opens a telemetry socket by itself',
  sockets.some((s) => s.url.includes('/ws/telemetry')),
  sockets.length ? sockets.map((s) => s.url).join(', ') : 'no sockets opened');

const tele = sockets.find((s) => s.url.includes('/ws/telemetry'));

check('the /ws proxy forwards the upgrade (frames arrive)',
  Boolean(tele && tele.frames.length),
  tele ? `${tele.frames.length} frame(s)` : '');

check('first frame is the hello greeting',
  Boolean(tele && tele.frames[0] && tele.frames[0].event === 'hello'),
  tele && tele.frames[0] ? `event=${tele.frames[0].event}` : '');

// A real server-side change must reach the real client.
await fetch(`http://127.0.0.1:${API_PORT}/api/__test__/advance`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ progress: 0.55, desc: '165 / 300', done: 165, total: 300 }),
});
await wait(2500);

const advanced = tele && tele.frames.find((f) => f.current_frame === 165);
check('a server-side change reaches the shipped client',
  Boolean(advanced), advanced ? `fps=${advanced.fps}` : '');

// And the app must actually render it, not just receive it.
const shown = await page.evaluate(() => document.body.innerText);
check('the UI renders the pushed progress', /165\s*\/\s*300|55\s*%|5[0-9]%/.test(shown),
  shown.match(/\d+\s*\/\s*300|\d+%/)?.[0] ?? '(no counter found in DOM)');

check('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));

const status = await (await fetch(`http://127.0.0.1:${API_PORT}/api/telemetry/status`)).json();
console.log(`\nhub: connected=${status.connected} total=${status.total_connections} sent=${status.frames_sent}`);
check('the backend saw a real client attach', status.total_connections >= 1);

await browser.close();
// `vite preview` and the python driver each spawn children of their own, so
// killing only the shell we started leaves a server holding the port and this
// process alive forever -- the run would print every PASS and then hang.
// taskkill /T kills the whole tree; await it so the exit below does not race it.
const killTree = (child) =>
  new Promise((resolve) => {
    if (!child.pid) return resolve();
    const k = spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
    k.on('close', resolve);
    k.on('error', resolve);
  });
await killTree(server);
await killTree(ui);

console.log(`\n${failures === 0 ? 'SHIPPED UI VERIFIED END TO END' : failures + ' CHECK(S) FAILED'}`);
process.exit(failures ? 1 : 0);
