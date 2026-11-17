import { test, expect } from '@playwright/test';
import { openTab } from './helpers.js';

// The Face Swap centre column: the run bar and the preview + timeline "deck" are pinned, and the
// stage's wheel zoom is proportional. Each is a place that "looked fine" in a screenshot taken at
// scroll 0 and was wrong one scroll later:
//
//   * the run bar's `top` was a guessed 5rem under a 55px header, so a strip of scrolling content
//     showed between them;
//   * the stage sat ~2300px down the page under a one-column slider grid, with the timeline below it,
//     so the picture and the timeline that drives it were never on screen together;
//   * the timeline's resize divider saved a height that nothing read;
//   * one wheel EVENT was a fixed 15% zoom step, so a trackpad swipe (dozens of tiny events) flew from
//     1x to 16x.

async function openFaceSwap(page, width, height) {
  await page.setViewportSize({ width, height });
  await openTab(page, 'faceswap');
  await page.locator('[data-testid="stage-deck"]').waitFor();
  await page.locator('[data-testid="timeline-deck"]').waitFor();
}

const rect = (page, testid) => page.locator(`[data-testid="${testid}"]`).evaluate((el) => {
  const r = el.getBoundingClientRect();
  return { top: r.top, bottom: r.bottom, left: r.left, right: r.right, height: r.height };
});

for (const [w, h] of [[1440, 900], [1920, 1080], [1280, 720]]) {
  test(`run bar, stage and timeline stay pinned and on screen together @ ${w}x${h}`, async ({ page }) => {
    await openFaceSwap(page, w, h);
    const headerBottom = await page.locator('header').first().evaluate((el) => el.getBoundingClientRect().bottom);

    // Scroll positions inside the Preview section: a sticky deck is released, correctly, where its
    // section ends (the Output settings section below it scrolls up normally).
    for (const y of [300, 600, 900]) {
      await page.evaluate((to) => window.scrollTo(0, to), y);
      await page.waitForTimeout(250);
      const bar = await rect(page, 'run-bar');
      const deck = await rect(page, 'stage-deck');
      const stage = await page.locator('.preview-stage').first().evaluate((el) => {
        const r = el.getBoundingClientRect();
        return { top: r.top, bottom: r.bottom };
      });
      const tl = await rect(page, 'timeline-deck');

      // flush under the header, and the deck flush under the bar: no strip of content between them
      expect(Math.abs(bar.top - headerBottom), `run bar sits on the header (scrollY ${y})`).toBeLessThanOrEqual(1);
      expect(Math.abs(deck.top - bar.bottom), `deck sits on the run bar (scrollY ${y})`).toBeLessThanOrEqual(1);
      // the picture and the timeline are both inside the window, with the sliders' room left below them
      expect(stage.top, 'stage below the pinned bars').toBeGreaterThanOrEqual(deck.top - 1);
      expect(tl.bottom, 'timeline inside the window').toBeLessThanOrEqual(h);
      expect(stage.bottom, 'stage above the timeline').toBeLessThanOrEqual(tl.top + 1);
      expect(h - deck.bottom, 'at least 100px is left under the deck for the controls that scroll beneath it').toBeGreaterThanOrEqual(100);
      // The deck must never scroll INSIDE itself: if its content is taller than its cap, a wheel over the
      // preview scrolls the picture under the run bar instead of the page ("the preview is not fixed").
      // --stage-max-h is measured from the deck's real overhead for exactly this.
      const overflow = await page.locator('[data-testid="stage-deck"]').evaluate((el) => el.scrollHeight - el.clientHeight);
      expect(overflow, `the deck does not scroll internally (scrollY ${y})`).toBeLessThanOrEqual(1);
    }
  });
}

test('the timeline divider changes the timeline height (it used to save a number nothing read)', async ({ page }) => {
  await openFaceSwap(page, 1920, 1080);
  const before = (await rect(page, 'timeline-deck')).height;
  // the divider is the horizontal separator directly above the timeline deck; dragging UP makes it taller
  const divider = page.locator('[data-testid="timeline-deck"]').locator('xpath=preceding-sibling::*[1]');
  const box = await divider.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2 - 90, { steps: 6 });
  await page.mouse.up();
  await page.waitForTimeout(200);
  const after = (await rect(page, 'timeline-deck')).height;
  expect(after, `timeline grew when its divider was dragged up (${before} -> ${after})`).toBeGreaterThan(before + 20);
});

test('wheel zoom is proportional to the wheel: a tick moves it a little, a notch ~15%, never a jump', async ({ page }) => {
  await openFaceSwap(page, 1920, 1080);
  const stage = page.locator('.preview-stage').first();
  const readout = stage.locator('span.font-mono.tabular-nums').first();
  const pct = async () => parseInt((await readout.textContent()) || '0', 10);
  const sb = await stage.boundingBox();
  await page.mouse.move(sb.x + sb.width / 2, sb.y + sb.height / 2);
  expect(await pct(), 'starts at fit').toBe(100);

  // a trackpad swipe: forty 3px events. The old fixed step compounded these to the 1600% ceiling.
  for (let i = 0; i < 40; i++) await page.mouse.wheel(0, -3);
  await page.waitForTimeout(200);
  const swipe = await pct();
  expect(swipe, 'a 40 x 3px swipe zooms in').toBeGreaterThan(100);
  expect(swipe, 'a 40 x 3px swipe is a modest zoom, not 16x').toBeLessThan(250);

  // zooming back out crosses fit exactly: it lands ON 100%, it does not skip past it
  for (let i = 0; i < 60; i++) await page.mouse.wheel(0, 3);
  await page.waitForTimeout(200);
  expect(await pct(), 'the swipe back out stops exactly on fit').toBe(100);
});

// Pro Studio swaps the stage for PreviewViewport, which used to size itself (min-h-[360px], no cap) and so
// ignored the deck's height budget: the deck was ~94px taller than its cap at 1583x823, scrolled INSIDE
// itself, and the preview slid under the run bar on every wheel tick instead of staying pinned.
for (const [w, h] of [[1583, 823], [1440, 900], [1920, 1080]]) {
  test(`Pro Studio: the pinned deck never scrolls inside itself @ ${w}x${h}`, async ({ page }) => {
    await openFaceSwap(page, w, h);
    await page.locator('[title="Toggle Pro Studio NLE View & Viewport"]').first().click();
    await page.locator('[data-testid="preview-viewport-container"]').waitFor();
    await page.waitForTimeout(500);
    const deck = page.locator('[data-testid="stage-deck"]');
    const overflow = () => deck.evaluate((el) => el.scrollHeight - el.clientHeight);
    expect(await overflow(), 'the deck holds its content without an inner scrollbar').toBeLessThanOrEqual(1);
    // and it stays pinned under the run bar as the page scrolls
    for (const y of [300, 600]) {
      await page.evaluate((to) => window.scrollTo(0, to), y);
      await page.waitForTimeout(250);
      const bar = await rect(page, 'run-bar');
      const d = await rect(page, 'stage-deck');
      expect(Math.abs(d.top - bar.bottom), `deck sits on the run bar (scrollY ${y})`).toBeLessThanOrEqual(1);
      expect(await overflow(), `still no inner scroll (scrollY ${y})`).toBeLessThanOrEqual(1);
    }
  });
}
