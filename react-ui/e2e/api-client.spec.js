import { test, expect } from '@playwright/test';
import { openTab } from './helpers.js';

// The API client's three behaviours that only mean something in a browser (the Node
// checks in .render-check/render-api-client.mjs cover the client's logic itself):
//
//   * a failure that used to be swallowed (`.catch(() => {})`) now reaches the user,
//     ONCE per distinct failure, with the status and path in the message;
//   * a request is cancelled when the component that made it unmounts;
//   * the 15 s default deadline really cuts a hung request off (checked on a fake clock,
//     not by waiting 15 seconds).

const toast = (page, text) => page.getByRole('button', { name: new RegExp(`Dismiss notification: ${text}`) });

test('a failed settings save is reported once, with its status and path, not swallowed', async ({ page }) => {
  const logs = [];
  page.on('console', (m) => { if (m.type() === 'warning' && /\[roop-ui\]/.test(m.text())) logs.push(m.text()); });
  let posts = 0;
  await page.route('**/api/settings', async (route) => {
    if (route.request().method() !== 'POST') return route.continue();
    posts += 1;
    return route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ message: 'disk full' }) });
  });

  await openTab(page, 'faceswap');
  const slider = page.locator('input[type="range"][aria-label="Original / Enhanced Blend"]');
  await slider.focus();
  await slider.press('ArrowRight');                                   // changes a setting -> debounced POST /api/settings

  const first = toast(page, 'Saving settings failed: disk full \\(HTTP 500 POST /api/settings\\)');
  await expect(first, 'the failure reaches the user, with status and path').toBeVisible({ timeout: 10_000 });

  await slider.press('ArrowRight');                                   // the same failure again
  await expect.poll(() => posts, { message: 'a second save was attempted (and failed the same way)' }).toBeGreaterThanOrEqual(2);
  await page.waitForTimeout(400);
  await expect(toast(page, 'Saving settings failed'), 'but it is not announced twice').toHaveCount(1);
  expect(logs.filter((l) => /Saving settings failed/.test(l)), 'and it is logged once').toHaveLength(1);
});

test('a request is cancelled when the tab that made it unmounts', async ({ page }) => {
  const aborted = [];
  page.on('requestfailed', (r) => { if (/\/api\/history/.test(r.url()) && /ABORTED/i.test(r.failure()?.errorText || '')) aborted.push(r.url()); });
  let started = false;
  await page.route('**/api/history*', () => { started = true; return new Promise(() => {}); });    // never answers

  await page.goto('/#/home');
  await page.locator('header nav button[aria-current="page"]').waitFor();
  await expect.poll(() => started, { message: 'Home asked for the run history' }).toBe(true);
  expect(aborted, 'still in flight while Home is on screen').toHaveLength(0);

  await page.locator('header nav > div:first-child button').last().click();       // Settings: Home unmounts
  await expect(page).toHaveURL(/#\/settings$/);
  await expect.poll(() => aborted.length, { message: 'the history request was aborted by the unmount', timeout: 10_000 }).toBeGreaterThan(0);
  // An abort is not a failure: nothing about it is shown to the user.
  await expect(page.getByRole('button', { name: /Dismiss notification: .*history/i })).toHaveCount(0);
});

test('the default deadline is 15 s: a hung GET is cut off and says which request', async ({ page }) => {
  const logs = [];
  page.on('console', (m) => { if (/\[roop-ui\]/.test(m.text())) logs.push(m.text()); });
  await page.clock.install();
  await page.route('**/api/settings/defaults', () => new Promise(() => {}));       // never answers

  const asked = page.waitForRequest(/\/api\/settings\/defaults/);
  await page.goto('/#/settings');
  await page.locator('header nav button[aria-current="page"]').waitFor();
  await asked;                  // the deadline's timer starts when the request is made, not before
  await page.clock.fastForward(14_000);
  expect(logs.filter((l) => /defaults/.test(l)), 'not yet: 14 s is inside the deadline').toHaveLength(0);
  await page.clock.fastForward(2_000);
  await expect.poll(() => logs.find((l) => /Loading the setting defaults failed/.test(l)), { timeout: 10_000 })
    .toMatch(/Request timed out after 15 s \(GET \/api\/settings\/defaults\)/);
});
