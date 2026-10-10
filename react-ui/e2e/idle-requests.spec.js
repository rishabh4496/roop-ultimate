import { test, expect } from '@playwright/test';
import { PNG_1X1, loadAllowlist, saveAllowlist, UPDATE_BASELINE } from './helpers.js';

// An idle Face Swap tab must not keep hitting /api/target/preview.
//
//   * preview answers 500   -> at most 6 STAGE-FRAME requests in the 30 s window
//   * preview answers a PNG -> no preview request at all once the first load has
//                              settled
//
// "Stage frame" is /api/target/preview without a `width` parameter -- the full
// frame for the playhead, the one request a retry loop can spin on. The other
// preview URLs are one-shot page-load requests that scale with the library, not
// with time: a 96 px thumbnail per target (width=96) and the filmstrip
// (/api/target/preview_grid). They do not count towards the 500 budget (the mock
// has two targets, so they are three requests that no retry policy could remove)
// but they are NOT exempt: no URL of any kind may be requested more than
// MAX_PER_URL times, which is what catches a loop on one of them.
//
// The 500 window starts when the nav is up, so it includes the initial load and
// every retry; the PNG window starts 3 s later, because the first load
// legitimately asks once and the claim is about what happens AFTER it. The PNG
// case counts every preview URL.
const WINDOW_MS = 30_000;
const SETTLE_MS = 3_000;
const MAX_PER_URL = 6;
const PREVIEW = '**/api/target/preview*';

const isStageFrame = (url) => {
  const u = new URL(url);
  return u.pathname === '/api/target/preview' && !u.searchParams.has('width');
};

async function watchPreviewRequests(page, fulfill, { settle }) {
  const seen = [];
  const t0 = Date.now();
  await page.route(PREVIEW, (route) => {
    const url = route.request().url();
    seen.push({ at: Date.now(), url, stage: isStageFrame(url) });
    return fulfill(route);
  });
  await page.goto('/#/faceswap');
  await page.locator('header nav button[aria-current="page"]').waitFor();
  const from = Date.now() + settle;
  await page.waitForTimeout(settle + WINDOW_MS);
  return {
    inWindow: seen.filter((r) => r.at >= from),
    all: seen,
    // What a failure message needs first: which URLs, and when.
    log: seen.map((r) => `${((r.at - t0) / 1000).toFixed(1)}s ${r.stage ? '[stage] ' : ''}${r.url.replace(/^https?:\/\/[^/]+/, '')}`),
  };
}

const CASES = [
  {
    key: 'stub500',
    title: 'preview 500 -> <= 6 requests in 30 s',
    budget: 6,
    settle: 0,
    scope: (r) => r.stage,
    fulfill: (route) => route.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"boom"}' }),
  },
  {
    key: 'stubPng',
    title: 'preview PNG -> 0 requests in 30 s after load',
    budget: 0,
    settle: SETTLE_MS,
    scope: () => true,
    fulfill: (route) => route.fulfill({ status: 200, contentType: 'image/png', body: PNG_1X1 }),
  },
];

for (const c of CASES) {
  test(`idle requests: ${c.title}`, async ({ page }, testInfo) => {
    test.setTimeout(WINDOW_MS + 45_000);
    const { inWindow, all, log } = await watchPreviewRequests(page, c.fulfill, { settle: c.settle });
    const counted = inWindow.filter(c.scope).length;
    testInfo.annotations.push({
      type: 'observed',
      description: `${counted} counted in window (${inWindow.length} preview requests in window, ${all.length} since load)`,
    });

    if (UPDATE_BASELINE) {
      saveAllowlist((a) => {
        if (counted > c.budget) a.idleRequests[c.key] = counted;
        else delete a.idleRequests[c.key];
      });
      return;
    }

    const ceiling = Math.max(c.budget, loadAllowlist().idleRequests[c.key] ?? 0);
    if (counted < ceiling && ceiling > c.budget) {
      testInfo.annotations.push({
        type: 'tighten allowlist',
        description: `${c.key}: ${counted} < baseline ${ceiling} -- run npm run test:e2e:baseline`,
      });
    }
    expect(counted, `/api/target/preview requests in ${WINDOW_MS / 1000}s (budget ${c.budget}, baseline ceiling ${ceiling}):\n${log.join('\n')}`)
      .toBeLessThanOrEqual(ceiling);

    const perUrl = new Map();
    for (const r of all) perUrl.set(r.url, (perUrl.get(r.url) || 0) + 1);
    const loops = [...perUrl].filter(([, n]) => n > MAX_PER_URL);
    expect(loops, `a URL requested more than ${MAX_PER_URL} times -- a retry loop:\n${log.join('\n')}`).toEqual([]);
  });
}
