import { test, expect } from '@playwright/test';
import { PNG_1X1, loadAllowlist, saveAllowlist, UPDATE_BASELINE } from './helpers.js';

// An idle Face Swap tab must not keep hitting /api/target/preview.
//
//   * preview answers 500  -> at most 6 requests in the 30 s window
//   * preview answers a PNG -> none once the first load has settled
//
// The 500 window starts at the nav being up (so it includes the initial load and
// every retry); the PNG window starts 3 s after that, because the first load
// legitimately asks once and the claim is about what happens AFTER it.
// Every URL variant counts (full frame, 96 px thumbs, timeline thumbs).
const WINDOW_MS = 30_000;
const SETTLE_MS = 3_000;
const PREVIEW = '**/api/target/preview*';

async function countPreviewRequests(page, fulfill, { settle }) {
  const stamps = [];
  await page.route(PREVIEW, (route) => {
    stamps.push(Date.now());
    return fulfill(route);
  });
  await page.goto('/#/faceswap');
  await page.locator('header nav button[aria-current="page"]').waitFor();
  const t0 = Date.now();
  await page.waitForTimeout(settle + WINDOW_MS);
  const from = t0 + settle;
  return { total: stamps.length, inWindow: stamps.filter((s) => s >= from).length };
}

const CASES = [
  {
    key: 'stub500',
    title: 'preview 500 -> <= 6 requests in 30 s',
    budget: 6,
    settle: 0,
    fulfill: (route) => route.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"boom"}' }),
  },
  {
    key: 'stubPng',
    title: 'preview PNG -> 0 requests in 30 s after load',
    budget: 0,
    settle: SETTLE_MS,
    fulfill: (route) => route.fulfill({ status: 200, contentType: 'image/png', body: PNG_1X1 }),
  },
];

for (const c of CASES) {
  test(`idle requests: ${c.title}`, async ({ page }, testInfo) => {
    test.setTimeout(WINDOW_MS + 45_000);
    const { total, inWindow } = await countPreviewRequests(page, c.fulfill, { settle: c.settle });
    testInfo.annotations.push({ type: 'observed', description: `${inWindow} in window (${total} since load)` });

    if (UPDATE_BASELINE) {
      saveAllowlist((a) => {
        if (inWindow > c.budget) a.idleRequests[c.key] = inWindow;
        else delete a.idleRequests[c.key];
      });
      return;
    }

    const ceiling = Math.max(c.budget, loadAllowlist().idleRequests[c.key] ?? 0);
    if (inWindow < ceiling && ceiling > c.budget) {
      testInfo.annotations.push({
        type: 'tighten allowlist',
        description: `${c.key}: ${inWindow} < baseline ${ceiling} -- run npm run test:e2e:baseline`,
      });
    }
    expect(inWindow, `/api/target/preview requests in ${WINDOW_MS / 1000}s (budget ${c.budget}, baseline ceiling ${ceiling})`)
      .toBeLessThanOrEqual(ceiling);
  });
}
