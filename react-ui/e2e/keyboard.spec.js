import { test, expect } from '@playwright/test';
import { openTab } from './helpers.js';

// A keyboard-only run of the core job: refresh the preview, then start a swap.
// No mouse event is issued anywhere in this file -- focus moves with Tab, the
// controls are operated with Enter/Space. If a control in that path loses its
// tab stop, its name, or its keyboard activation, this is the test that stops
// being able to finish.
//
// The mock backend answers the requests; what is asserted is what the UI SENT and
// what it DID with the answer (it follows a started run to the Processing tab).
const MAX_TABS = 600;

const accessibleName = () => {
  const e = document.activeElement;
  if (!e || e === document.body) return '';
  const label = e.getAttribute('aria-label')
    || (e.labels && e.labels[0] && e.labels[0].textContent)
    || e.innerText || e.textContent || '';
  return label.replace(/\s+/g, ' ').trim();
};

/** Press Tab until the focused control's name matches; returns the presses taken. */
async function tabTo(page, pattern) {
  for (let presses = 1; presses <= MAX_TABS; presses++) {
    await page.keyboard.press('Tab');
    const name = await page.evaluate(accessibleName);
    if (pattern.test(name)) return presses;
  }
  throw new Error(`Tab never reached a control named ${pattern} in ${MAX_TABS} presses`);
}

// This test STARTS A RUN in the shared mock, and a run outlives the test: the next
// test would load with the transient Processing tab in the nav (that broke the
// nav-visibility spec, which runs straight after). So leave the mock idle.
async function leaveMockIdle(page) {
  await page.request.post('/api/stop', { data: {} });
  await expect.poll(async () => (await (await page.request.get('/api/progress')).json()).processing,
    { message: 'the mock run to stop', timeout: 15_000 }).toBe(false);
}

// afterEach, so it also runs when the test fails half way through the swap.
test.afterEach(async ({ page }) => { await leaveMockIdle(page); });

test('keyboard only: refresh the preview, then start a swap', async ({ page }) => {
  test.setTimeout(90_000);
  const calls = { preview: 0, swap: 0 };
  page.on('request', (req) => {
    if (req.method() !== 'POST') return;
    const { pathname } = new URL(req.url());
    if (pathname === '/api/preview') calls.preview += 1;
    if (pathname === '/api/swap') calls.swap += 1;
  });

  await openTab(page, 'faceswap');
  await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); });

  // ── Preview ──────────────────────────────────────────────────────────────
  // The auto-preview at load has already asked once; an explicit Refresh must ask
  // again (it forces past the cache), so the count has to move.
  await expect.poll(() => calls.preview, { message: 'the load-time preview' }).toBeGreaterThan(0);
  const before = calls.preview;
  const toRefresh = await tabTo(page, /^Refresh Preview$/);
  await page.keyboard.press('Enter');
  await expect.poll(() => calls.preview, { message: 'Enter on Refresh Preview must request a preview' })
    .toBeGreaterThan(before);
  await expect(page.locator('.preview-stage'), 'a preview is on the stage').toBeVisible();

  // ── Swap ─────────────────────────────────────────────────────────────────
  // Tab onward from where focus is, then Space (the other activation key) on the
  // real start button.
  const toStart = await tabTo(page, /Start Swapping/);
  await expect(page.getByRole('button', { name: /Start Swapping/ })).toBeFocused();
  await page.keyboard.press('Space');
  await expect.poll(() => calls.swap, { message: 'Space on Start Swapping must POST /api/swap' }).toBe(1);

  // The UI follows a started run to its own tab.
  await expect(page).toHaveURL(/#\/processing$/, { timeout: 15_000 });
  await expect(page.locator('header nav button[aria-current="page"]')).toContainText('Processing');

  test.info().annotations.push({
    type: 'tab presses',
    description: `Refresh Preview after ${toRefresh}, Start Swapping after a further ${toStart}`,
  });
});
