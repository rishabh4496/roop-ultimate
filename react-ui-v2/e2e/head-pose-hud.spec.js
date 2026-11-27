import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { openTab } from './helpers.js';

// The head-pose HUD in a real browser: on the Processing tab's live panels (where the render is watched) and in Face
// Swap's collapsible diagnostics section. Its states, its numbers, its request budget, and axe on it while it shows
// live angles. The numbers come from the mock's GET /api/trajectory/pose (the real one answers from
// roop/trajectory_cache.py).

const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'];

const frame = (extra) => ({
  event: 'progress', processing: true, paused: false, progress: 0.4, desc: 'Processing frame 144 / 360',
  error: '', current_frame: 144, total_frames: 360, fps: 11.8, fps_now: 14.4, frame_ms: 69, eta_s: 7,
  started_at: Date.now() / 1000 - 20, live_seq: 0, engine_state: 'IDLE', engine_label: '', engine_model: '',
  engine_precision: '', engine_since: null, ...extra,
});

async function scriptTelemetry(page, frames, gap = 700) {
  // The app also polls GET /api/progress (the socket's fallback) and merges both into one store. The mock answers
  // that poll as "idle" (0 frames), which would make the frame number flap between the script's and 0; against the
  // real backend the two channels agree, so the poll is answered with the script's own counters.
  const live = { processing: !!frames[0].processing, current_frame: frames[0].current_frame,
    total_frames: frames[0].total_frames };
  await page.route('**/api/progress', async (route) => {
    const res = await route.fetch();
    let body = {};
    try { body = await res.json(); } catch { body = {}; }
    await route.fulfill({ response: res, json: { ...body, ...live } });
  });
  await page.routeWebSocket(/\/ws\/telemetry/, (ws) => {
    let i = 0;
    const send = () => {
      const f = frames[Math.min(i, frames.length - 1)];
      live.processing = !!f.processing;
      live.current_frame = f.current_frame;
      live.total_frames = f.total_frames;
      ws.send(JSON.stringify(f));
      i += 1;
    };
    send();
    const timer = setInterval(send, gap);
    ws.onClose(() => clearInterval(timer));
  });
}

const openFaceSwapSection = async (page) => {
  await openTab(page, 'faceswap');
  const toggle = page.getByRole('button', { name: /Live Telemetry & Diagnostics/ });
  await toggle.scrollIntoViewIfNeeded();
  if ((await toggle.getAttribute('aria-expanded')) !== 'true') await toggle.click();
};

test('idle: the HUD says it appears with a job and asks the backend for nothing', async ({ page }) => {
  const asked = [];
  page.on('request', (r) => { if (r.url().includes('/api/trajectory/')) asked.push(r.url()); });
  await openFaceSwapSection(page);
  const hud = page.getByTestId('head-pose-hud');
  await expect(hud).toBeVisible();
  await expect(hud).toHaveAttribute('data-status', 'idle');
  await expect(hud).toContainText('Appears while a job runs');
  await page.waitForTimeout(1500);
  expect(asked, 'an idle app must not poll the trajectory').toEqual([]);
});

test('a running job: live yaw / pitch / roll for each face, flagged when not detected, axe clean', async ({ page }) => {
  const asked = [];
  page.on('request', (r) => { if (r.url().includes('/api/trajectory/pose')) asked.push(r.url()); });
  await scriptTelemetry(page, [frame({ event: 'hello' }), frame({})], 500);
  await openTab(page, 'processing');

  const hud = page.getByTestId('head-pose-hud');
  await expect(hud).toHaveAttribute('data-status', 'live');
  const faces = hud.getByTestId('head-pose-face');
  await expect(faces).toHaveCount(2);
  await expect(faces.nth(0)).toContainText('Face 0');
  // frame 144 -> yaw 40 * sin(12) = -21.46 (the mock's own formula), shown to one decimal: -21.5
  await expect(faces.nth(0).locator('[data-dial="yaw"]')).toContainText('-21.5°');
  await expect(faces.nth(0).locator('[data-dial="pitch"]')).toContainText('-8.5°');
  await expect(faces.nth(0).locator('[data-dial="roll"]')).toContainText('+2.3°');
  await expect(faces.nth(1)).toContainText('Face 1');
  await expect(hud).toContainText('+yaw: nose toward your right');
  await expect(hud).toContainText('Frame 144');

  const results = await new AxeBuilder({ page }).include('[data-testid="head-pose-hud"]').withTags(TAGS).analyze();
  expect(results.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(' | ')}`)).toEqual([]);

  // The frame does not move (the script repeats it): after the first answer there is nothing new to ask.
  const settled = asked.length;
  await page.waitForTimeout(2500);
  expect(asked.length, 'one request per CHANGE of frame, not one per tick').toBe(settled);
  expect(settled).toBeGreaterThan(0);
  expect(settled).toBeLessThanOrEqual(2);
});

test('a frame that moves on is asked for, at no more than 2 Hz', async ({ page }) => {
  const times = [];
  page.on('request', (r) => { if (r.url().includes('/api/trajectory/pose')) times.push(Date.now()); });
  const script = [frame({ event: 'hello', current_frame: 10 })];
  for (let i = 1; i <= 40; i += 1) script.push(frame({ current_frame: 10 + i * 3 }));
  await scriptTelemetry(page, script, 100);          // 10 frames a second, each a new frame number
  await openTab(page, 'processing');
  await expect(page.getByTestId('head-pose-hud')).toHaveAttribute('data-status', 'live');
  await page.waitForTimeout(3000);
  const recent = times.filter((t) => t > Date.now() - 3000);
  expect(recent.length, 'the HUD follows the frame at <= 2 Hz even when frames arrive at 10 Hz').toBeLessThanOrEqual(8);
  expect(recent.length).toBeGreaterThanOrEqual(2);
});
