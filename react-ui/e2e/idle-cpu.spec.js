import { test, expect } from '@playwright/test';
import { PNG_1X1 } from './helpers.js';

// An idle Face Swap tab whose backend is FAILING must not cost much more CPU
// than one whose frames load: within 2x. Before the retry policy a 500 made the
// tab re-request, re-fail and re-render ~6.5 times a second, forever.
//
// CPU is the browser's own accounting of cumulative CPU time across ALL its
// processes (renderer, GPU, browser), sampled over a window that starts after the
// failing case has used up its retries (~8 s), so what is compared is the steady
// state -- which is the only thing "idle" can mean. Two floors keep it honest on a
// quiet machine, where both numbers are tiny and a ratio of tiny numbers is noise:
// the bound is `failing <= 2 x valid + ABS_SLACK` (a fraction of one core).
const SETTLE_MS = 12_000;
const MEASURE_MS = 15_000;
const RATIO = 2;
const ABS_SLACK = 0.02;

async function idleCpu(browser, healthy) {
  const context = await browser.newContext({ reducedMotion: 'reduce' });
  const page = await context.newPage();
  const cdp = await browser.newBrowserCDPSession();
  try {
    await page.route('**/api/target/preview*', (route) => (healthy
      ? route.fulfill({ status: 200, contentType: 'image/png', body: PNG_1X1 })
      : route.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"boom"}' })));
    await page.route('**/api/preview', (route) => (route.request().method() === 'POST' && !healthy
      ? route.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"boom"}' })
      : route.fallback()));
    await page.goto('/#/faceswap');
    await page.locator('header nav button[aria-current="page"]').waitFor();
    await page.waitForTimeout(SETTLE_MS);

    const cpuSeconds = async () => {
      const { processInfo } = await cdp.send('SystemInfo.getProcessInfo');
      return processInfo.reduce((s, p) => s + p.cpuTime, 0);
    };
    const c0 = await cpuSeconds();
    const t0 = Date.now();
    await page.waitForTimeout(MEASURE_MS);
    const c1 = await cpuSeconds();
    return (c1 - c0) / ((Date.now() - t0) / 1000);
  } finally {
    await cdp.detach().catch(() => {});
    await context.close();
  }
}

test('idle CPU: failing backend within 2x of valid frames', async ({ browser }, testInfo) => {
  test.setTimeout(150_000);
  const valid = await idleCpu(browser, true);
  const failing = await idleCpu(browser, false);
  const bound = RATIO * valid + ABS_SLACK;
  const observed = `valid ${(valid * 100).toFixed(1)}% of a core, failing ${(failing * 100).toFixed(1)}%, bound ${(bound * 100).toFixed(1)}%`;
  testInfo.annotations.push({ type: 'observed', description: observed });
  console.log(`idle CPU: ${observed}`);
  expect(failing, `failing ${failing.toFixed(4)} vs ${RATIO}x valid ${valid.toFixed(4)} + ${ABS_SLACK}`).toBeLessThanOrEqual(bound);
});
