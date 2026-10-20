import { test, expect } from '@playwright/test';
import { PNG_1X1 } from './helpers.js';

// While the pipeline's pre-pass (Stage 1 sparse pre-scan / Stage 2 active-range pre-pass) has the
// GPU flat out, the backend answers an UNCACHED frame with a fast
//   503 {"busy": true, "stage": "STAGE_2_ACTIVE_PREPASS", "retry_after_ms": N}  + Retry-After
// (api._preview_busy_response). That is "ask again in N ms", not a failure: it must not spend the
// 250 / 500 / 1000 ms failure budget (a786b33 / retryPolicy.js) and put "Frame unavailable" over a
// healthy render, and it must come back by itself when the stage ends.
const isStageFrame = (url) => {
  const u = new URL(url);
  return u.pathname === '/api/target/preview' && !u.searchParams.has('width');
};

const BUSY = {
  status: 503,
  contentType: 'application/json',
  headers: { 'Retry-After': '1' },
  body: JSON.stringify({ status: 'busy', busy: true, stage: 'STAGE_2_ACTIVE_PREPASS', retry_after_ms: 300 }),
};

test('busy frame: waits as told, never shows "Frame unavailable", recovers when the stage ends', async ({ page }) => {
  test.setTimeout(60_000);
  let busyLeft = 8;                                  // twice the failure budget (4 attempts)
  const stage = [];
  await page.route('**/api/target/preview*', (route) => {
    if (!isStageFrame(route.request().url())) return route.fallback();
    stage.push(Date.now());
    if (busyLeft > 0) { busyLeft -= 1; return route.fulfill(BUSY); }
    return route.fulfill({ status: 200, contentType: 'image/png', body: PNG_1X1 });
  });
  await page.route('**/api/preview', (route) => (route.request().method() === 'POST' && busyLeft > 0
    ? route.fulfill(BUSY)
    : route.fallback()));

  await page.goto('/#/faceswap');
  const banner = page.getByTestId('frame-unavailable');

  // 8 busy answers x 300 ms is ~2.4 s: well past the 1.75 s in which a FAILURE would have given up.
  await expect.poll(() => stage.length, { timeout: 20_000, message: 'busy answers were retried' })
    .toBeGreaterThan(8);
  await expect(banner, 'a busy backend is not an unavailable frame').toHaveCount(0);

  // Each retry waited at least what it was told (300 ms, minus timer slop) -- not hammering.
  const gaps = stage.slice(1, 9).map((t, i) => t - stage[i]);
  for (const [i, gap] of gaps.entries()) {
    expect(gap, `gap before busy retry ${i + 1}`).toBeGreaterThanOrEqual(250);
  }

  // The stage ended: the next answer is a picture and the page settles with no message.
  const settled = stage.length;
  await page.waitForTimeout(3_000);
  expect(stage.length - settled, 'it stopped asking once it had the frame').toBeLessThanOrEqual(1);
  await expect(banner).toHaveCount(0);
});

test('busy frame: a real failure still gives up after 1 + 3 attempts (busy did not loosen the budget)', async ({ page }) => {
  test.setTimeout(60_000);
  const stage = [];
  await page.route('**/api/target/preview*', (route) => {
    if (!isStageFrame(route.request().url())) return route.fallback();
    stage.push(Date.now());
    return route.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"boom"}' });
  });
  await page.route('**/api/preview', (route) => (route.request().method() === 'POST'
    ? route.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"boom"}' })
    : route.fallback()));
  await page.goto('/#/faceswap');
  await expect(page.getByTestId('frame-unavailable')).toBeVisible({ timeout: 15_000 });
  expect(stage.length).toBe(4);
});

test('busy playback chunk: waits as told and playback is not stopped', async ({ page }) => {
  test.setTimeout(60_000);
  let busyLeft = 7;                                  // more than the 4 attempts a failure is allowed
  const chunks = [];
  await page.route('**/api/target/preview_seq*', (route) => {
    chunks.push(Date.now());
    if (busyLeft > 0) { busyLeft -= 1; return route.fulfill(BUSY); }
    return route.fulfill({ status: 200, contentType: 'application/octet-stream', body: Buffer.alloc(0) });
  });
  await page.goto('/#/faceswap');
  const play = page.getByRole('button', { name: /^Play \(Space\)$/ });
  await expect(play).toBeVisible({ timeout: 15_000 });
  await play.click();

  await expect.poll(() => chunks.length, { timeout: 25_000, message: 'busy chunk answers were retried' })
    .toBeGreaterThan(7);
  const toast = page.getByRole('button', { name: /Dismiss notification: Playback stopped/ });
  await expect(toast, 'a busy backend must not stop playback with an error').toHaveCount(0);
  const gaps = chunks.slice(1, 8).map((t, i) => t - chunks[i]);
  for (const [i, gap] of gaps.entries()) {
    expect(gap, `gap before busy chunk retry ${i + 1}`).toBeGreaterThanOrEqual(250);
  }
});
