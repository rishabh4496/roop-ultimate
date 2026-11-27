import { test, expect } from '@playwright/test';
import { openTab } from './helpers.js';

test('render progress follows the active phase through tracking, swap, encoding and upscale', async ({ page }) => {
  const startedAt = Date.now() / 1000 - 600;
  let sample = { processing: true, paused: false, started_at: startedAt, progress: 0.81, desc: 'Analyzing faces…' };
  let socket;
  await page.route('**/api/progress', (route) => route.fulfill({ json: sample }));
  await page.routeWebSocket(/\/ws\/telemetry/, (ws) => {
    socket = ws;
    ws.send(JSON.stringify({ event: 'hello', ...sample }));
  });
  const update = (patch) => {
    sample = { ...sample, ...patch };
    socket.send(JSON.stringify({ event: 'progress', ...sample }));
  };
  await openTab(page, 'processing');
  const bar = page.getByRole('progressbar', { name: 'Render progress' });

  update({ dashboard: { active: true, stages: { S2: {
    state: 'RUNNING', detail: { phase: 'Tracking' }, pct: 62, done: 62, total: 100, unit: 'frames', eta_s: 42,
  } } } });
  await expect(bar).toHaveAttribute('aria-valuenow', '62');
  await expect(bar).toHaveAttribute('aria-valuetext', /Tracking.*62 of 100 frames/);

  update({ progress: 0.81, desc: 'Processing frame 81 / 100', current_frame: 72, total_frames: 100, fps_now: 16,
    dashboard: { active: false }, eta_s: 12 });
  await expect(bar).toHaveAttribute('aria-valuenow', '72');
  await expect(bar).toHaveAttribute('aria-valuetext', /Swap.*72 of 100 frames/);
  await expect(page.getByText('16.0 fps', { exact: true })).toBeVisible();

  update({ progress: 1, desc: 'Combining (encode + audio)…', current_frame: 100, eta_s: null });
  await expect(bar).not.toHaveAttribute('aria-valuenow');
  await expect(bar).toHaveAttribute('aria-valuetext', /Finalizing/);

  update({ progress: 0.1, desc: 'Upscaling frame 10 / 100', current_frame: 10 });
  await expect(bar).toHaveAttribute('aria-valuenow', '10');
  await expect(bar).toHaveAttribute('aria-valuetext', /Upscale.*10 of 100 frames/);

  update({ paused: true, desc: 'Paused', eta_s: 15 });
  await expect(bar).toHaveAttribute('aria-valuenow', '10');
  await expect(bar).toHaveAttribute('aria-valuetext', /Paused.*Upscale/);
  await expect(bar).not.toHaveAttribute('aria-valuetext', /remaining/);
});

test('a delayed poll cannot replace newer socket progress', async ({ page }) => {
  const sample = { processing: true, paused: false, started_at: Date.now() / 1000,
    progress: 0.2, desc: 'Processing frame 20 / 100', current_frame: 20, total_frames: 100 };
  let socket;
  let holdNext = false;
  let sawPoll;
  let releasePoll;
  const pending = new Promise((resolve) => { sawPoll = resolve; });
  const release = new Promise((resolve) => { releasePoll = resolve; });
  await page.route('**/api/progress', async (route) => {
    let snapshot = sample;
    if (holdNext) {
      holdNext = false;
      snapshot = { ...sample, log: [{ seq: 1, msg: 'A delayed console line' }] };
      sawPoll();
      await release;
    }
    await route.fulfill({ json: snapshot });
  });
  await page.routeWebSocket(/\/ws\/telemetry/, (ws) => {
    socket = ws;
    ws.send(JSON.stringify({ event: 'hello', ...sample }));
  });
  await openTab(page, 'processing');
  const bar = page.getByRole('progressbar', { name: 'Render progress' });
  await expect(bar).toHaveAttribute('aria-valuenow', '20');

  holdNext = true;
  await page.evaluate(() => window.dispatchEvent(new Event('focus')));
  await pending;
  socket.send(JSON.stringify({ event: 'progress', ...sample, progress: 1, desc: 'Combining video…' }));
  await expect(bar).toHaveAttribute('aria-valuetext', 'Finalizing');
  releasePoll();
  await expect(page.getByText('A delayed console line')).toBeVisible();
  await expect(bar).toHaveAttribute('aria-valuetext', 'Finalizing');
});

test('progress falls back to polled written-frame counts when the socket is unavailable', async ({ page }) => {
  const startedAt = Date.now() / 1000 - 30;
  await page.routeWebSocket(/\/ws\/telemetry/, (ws) => ws.close());
  await page.route('**/api/progress', (route) => route.fulfill({ json: {
    processing: true, started_at: startedAt, progress: 0.67, desc: 'Processing frame 67 / 100',
    runtime: { frame_progress: { done: 60, total: 100 } },
  } }));
  await openTab(page, 'processing');
  await expect(page.getByRole('progressbar', { name: 'Render progress' })).toHaveAttribute('aria-valuenow', '60');
});

test('preview height supports drag, keyboard, presets and reload', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openTab(page, 'faceswap');
  const preview = page.locator('#face-swap-preview');
  await expect(preview.locator('.preview-stage')).toBeVisible();
  const size = page.getByRole('combobox', { name: 'Preview size' });
  const handle = page.getByRole('separator', { name: 'Resize preview' });
  await size.selectOption('compact');
  const compact = await preview.evaluate((el) => el.offsetHeight);
  await handle.scrollIntoViewIfNeeded();
  const box = await handle.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2 + 60, { steps: 4 });
  await page.mouse.up();
  await expect.poll(() => preview.evaluate((el) => el.offsetHeight)).toBe(compact + 60);

  await handle.focus();
  await handle.press('ArrowDown');
  await expect.poll(() => preview.evaluate((el) => el.offsetHeight)).toBeGreaterThan(compact + 60);
  const resized = await preview.evaluate((el) => el.offsetHeight);
  await page.reload();
  await expect.poll(() => preview.evaluate((el) => el.offsetHeight)).toBe(resized);

  await size.selectOption('large');
  await expect.poll(() => preview.evaluate((el) => el.offsetHeight)).toBeGreaterThan(resized);
  await handle.press('Enter');
  await expect(size).toHaveValue('fit');

  await page.setViewportSize({ width: 390, height: 844 });
  await size.selectOption('compact');
  const mobileStage = await preview.locator('.preview-stage').boundingBox();
  expect(mobileStage.height).toBe(compact);
  expect(mobileStage.x + mobileStage.width).toBeLessThanOrEqual(390);
});
