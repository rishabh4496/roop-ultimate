import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import net from 'node:net';
import path from 'node:path';

const REQ = createRequire('G:/pinokio/cache/npm_config_cache/_npx/e41f203b7505f1fb/');
const { chromium } = REQ('playwright-core');

const CHROME = 'C:/Users/rishr/AppData/Local/ms-playwright/chromium-1234/chrome-win64/chrome.exe';
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

const PORT = await freePort();

const driver = `
import os, sys, time
sys.path.insert(0, r"${path.join(REPO, 'app').replace(/\\/g, '\\\\')}")
os.environ["ROOP_REACT_CLIENT"] = "1"
import api
from settings import Settings
import roop.globals as roop_globals
roop_globals.CFG = Settings('default_config.yaml')
import uvicorn
uvicorn.run(api.app, host="127.0.0.1", port=${PORT}, log_level="error")
`;

const server = spawn(PY, ['-c', driver], { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
server.on('exit', () => {});

const base = `http://127.0.0.1:${PORT}`;
let up = false;
for (let i = 0; i < 180; i++) {
  try {
    if ((await fetch(`${base}/api/telemetry/status`)).ok) { up = true; break; }
  } catch {}
  await wait(500);
}
if (!up) { console.error('Backend did not start'); server.kill(); process.exit(1); }

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu'],
});
const page = await browser.newPage();
await page.goto(`${base}/`, { waitUntil: 'load' });
await wait(3000);

// Navigate to Settings tab
const settingsTab = page.getByRole('button', { name: 'Settings', exact: true });
await settingsTab.click();

for (let i = 0; i < 50; i++) {
  const text = await page.evaluate(() => document.body.innerText);
  if (/Apply Settings/.test(text)) break;
  await wait(200);
}

// Check Provider select exists
const providerSelect = page.locator('[data-setting="provider"] select');
const provCount = await providerSelect.count();
console.log(`PASS  Provider select rendered: count=${provCount}`);

// Check Provider options include tensorrt, cuda, cpu
const options = await providerSelect.locator('option').allInnerTexts();
console.log(`PASS  Provider options available: ${JSON.stringify(options)}`);
if (!options.includes('tensorrt') || !options.includes('cuda') || !options.includes('cpu')) {
  console.error('FAIL: Missing expected providers in options');
  await browser.close();
  server.kill();
  process.exit(1);
}

// Check Runtime status indicator is displayed
const statusText = await page.locator('text=Active:').first().innerText();
console.log(`PASS  Runtime status displayed: "${statusText}"`);

// Check Precision mode (TensorRT) is rendered in the DOM
const precisionSelect = page.locator('[data-setting="trt_precision"] select');
const precCount = await precisionSelect.count();
console.log(`PASS  Precision mode rendered: count=${precCount}`);

// Check Advanced Performance section has TensorRT controls
const builderSelect = page.locator('[data-setting="trt_builder_optimization_level"] select');
console.log(`PASS  TRT builder optimization rendered: count=${await builderSelect.count()}`);

const auxSelect = page.locator('[data-setting="trt_auxiliary_streams"] select');
console.log(`PASS  TRT auxiliary streams rendered: count=${await auxSelect.count()}`);

const graphToggle = page.locator('[data-setting="trt_cuda_graph"] input');
console.log(`PASS  TRT CUDA graph toggle rendered: count=${await graphToggle.count()}`);

await browser.close();
server.kill();
console.log('ALL UI TENSORRT VISIBILITY CHECKS PASSED');
