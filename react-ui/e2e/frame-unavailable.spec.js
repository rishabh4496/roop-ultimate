import { test, expect } from '@playwright/test';
import { PNG_1X1 } from './helpers.js';

// When the stage's frame cannot be fetched, the preview says so and offers a
// retry, instead of silently keeping the previous picture on screen (or an empty
// placeholder).
const isStageFrame = (url) => {
  const u = new URL(url);
  return u.pathname === '/api/target/preview' && !u.searchParams.has('width');
};

const BOOM = { status: 500, contentType: 'application/json', body: '{"detail":"boom"}' };

test('frame unavailable: shown once retries are spent, Retry recovers', async ({ page }) => {
  test.setTimeout(90_000);
  // Everything fails, the swapped-preview POST included: no preview was ever
  // produced, so the stage shows its placeholder.
  let healthy = false;
  const stage = [];
  await page.route('**/api/target/preview*', (route) => {
    if (isStageFrame(route.request().url())) stage.push(Date.now());
    return healthy
      ? route.fulfill({ status: 200, contentType: 'image/png', body: PNG_1X1 })
      : route.fulfill(BOOM);
  });
  await page.route('**/api/preview', (route) => (route.request().method() === 'POST' && !healthy
    ? route.fulfill(BOOM)
    : route.fallback()));

  await page.goto('/#/faceswap');
  const banner = page.getByTestId('frame-unavailable');

  // Five attempts at ~0.5/1/2/4 s gaps is ~8 s; not before.
  await expect(banner).toBeVisible({ timeout: 25_000 });
  await expect(banner).toHaveText(/Frame unavailable\s*-\s*Retry/);
  await expect(banner.getByRole('button', { name: 'Retry' })).toBeVisible();

  // ...and it has actually STOPPED asking.
  const spent = stage.length;
  expect(spent, 'stage-frame attempts before giving up').toBe(5);
  await page.waitForTimeout(5_000);
  expect(stage.length, 'requests after giving up').toBe(spent);

  // Backend recovers; Retry asks again and the message goes away.
  healthy = true;
  await banner.getByRole('button', { name: 'Retry' }).click();
  await expect(banner).toBeHidden({ timeout: 10_000 });
  expect(stage.length, 'Retry made a request').toBeGreaterThan(spent);
});

test('frame unavailable: over a stale swap, stepping to a frame that fails', async ({ page }) => {
  test.setTimeout(90_000);
  // The swap for frame 1 renders (the mock's own /api/preview), then the backend
  // starts failing: stepping to frame 2 leaves frame 1's swap on the stage, which
  // is exactly the picture that must not pass for the current frame.
  let failing = false;
  await page.route('**/api/target/preview*', (route) => (failing
    ? route.fulfill(BOOM)
    : route.fulfill({ status: 200, contentType: 'image/png', body: PNG_1X1 })));
  await page.route('**/api/preview', (route) => (route.request().method() === 'POST' && failing
    ? route.fulfill(BOOM)
    : route.fallback()));

  await page.goto('/#/faceswap');
  const stageEl = page.locator('.preview-stage');
  const banner = page.getByTestId('frame-unavailable');
  await expect(stageEl).toBeVisible({ timeout: 15_000 });    // frame 1's swap is up
  await expect(banner).toHaveCount(0);

  failing = true;
  await page.getByRole('button', { name: 'Next frame (→)' }).click();   // the timeline's; the HUD has its own
  await expect(banner).toBeVisible({ timeout: 25_000 });
  await expect(stageEl, 'the stale stage is still there under the message').toBeVisible();
  await expect(banner.getByRole('button', { name: 'Retry' })).toBeEnabled();
});
