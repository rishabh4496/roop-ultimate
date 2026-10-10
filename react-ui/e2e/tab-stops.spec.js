import { test, expect } from '@playwright/test';
import { openTab, loadAllowlist, saveAllowlist, UPDATE_BASELINE } from './helpers.js';

// How many Tab presses it takes to walk the Face Swap tab once. Counted by
// pressing Tab for real (so display:none, disabled, inert and tabindex all count
// the way the browser counts them) until focus returns to an element it has
// already visited or leaves the document. The number is a ceiling taken from the
// allowlist: a new control that joins the tab order fails the run until someone
// decides it should, then records it.
const MAX_PRESSES = 1500;

test('tab stops: faceswap', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  await openTab(page, 'faceswap');

  await page.evaluate(() => {
    document.activeElement?.blur();
    window.scrollTo(0, 0);
    window.__stops = [];
    window.__wrapped = false;
    document.addEventListener('focusin', (e) => {
      if (e.target === document.body) return;
      if (window.__stops.includes(e.target)) window.__wrapped = true;
      else window.__stops.push(e.target);
    }, true);
  });

  let presses = 0;
  while (presses < MAX_PRESSES) {
    await page.keyboard.press('Tab');
    presses += 1;
    const state = await page.evaluate(() => ({
      wrapped: window.__wrapped,
      onBody: document.activeElement === document.body || document.activeElement === null,
      stops: window.__stops.length,
    }));
    // Focus on <body> after at least one stop means it fell off the end.
    if (state.wrapped || (state.onBody && state.stops > 0)) break;
  }

  const count = await page.evaluate(() => window.__stops.length);
  testInfo.annotations.push({ type: 'observed', description: `${count} tab stops` });
  expect(presses, 'tab walk never terminated').toBeLessThan(MAX_PRESSES);
  expect(count, 'no tab stops found at all').toBeGreaterThan(0);

  if (UPDATE_BASELINE) {
    saveAllowlist((a) => { a.tabStops.faceswap = count; });
    return;
  }

  const ceiling = loadAllowlist().tabStops.faceswap;
  expect(ceiling, 'no tabStops.faceswap baseline -- run npm run test:e2e:baseline').toBeDefined();
  if (count < ceiling) {
    testInfo.annotations.push({
      type: 'tighten allowlist',
      description: `faceswap: ${count} < baseline ${ceiling} -- run npm run test:e2e:baseline`,
    });
  }
  expect(count, `Face Swap tab stops (baseline ${ceiling})`).toBeLessThanOrEqual(ceiling);
});
