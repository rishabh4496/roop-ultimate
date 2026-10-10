import { test, expect } from '@playwright/test';
import { openTab } from './helpers.js';

// Face Swap's slider grid, type scale and run controls, measured in the real
// layout at 1440px (the width the slider labels used to truncate at).
//
//   * no slider-tracker label is cut off (they wrap to two lines; the title holds
//     the whole label);
//   * no BODY text is under 12px. Body text is anything that reads as a sentence
//     or a label/value. Exempt, because they are not read that way: a glyph or
//     tick of 1-3 characters, a <kbd>, an uppercase letter-spaced tag, a pill
//     badge -- the sizes `nano`/`micro` exist for -- and anything marked
//     `data-small-ok` (use sparingly, with a reason beside it);
//   * every range input has a >=24px hit area;
//   * there is ONE run control, with the reason in words when it is unavailable;
//   * the floating dock covers nothing once the page is scrolled to its end.
const MIN_BODY_PX = 12;
const MIN_HIT_PX = 24;

async function openFaceSwap(page, width = 1440) {
  await page.setViewportSize({ width, height: 900 });
  await openTab(page, 'faceswap');
}

test('slider tracker: no label is truncated @ 1440px, and each carries its full label as a title', async ({ page }) => {
  await openFaceSwap(page);
  const report = await page.evaluate(() => {
    const labels = [...document.querySelectorAll('[class~="group/card"] > div:first-child > span:first-child')];
    return labels.map((el) => {
      const cs = getComputedStyle(el);
      const lineH = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.3;
      return {
        text: el.textContent.trim(),
        title: el.getAttribute('title') || '',
        // text-overflow: ellipsis on a single line, or a clamp that actually cut text
        ellipsis: cs.textOverflow === 'ellipsis' && el.scrollWidth > el.clientWidth + 1,
        cut: el.scrollHeight > el.clientHeight + 1,
        lines: Math.round(el.getBoundingClientRect().height / lineH),
        cardW: Math.round(el.closest('[class~="group/card"]').getBoundingClientRect().width),
      };
    });
  });
  expect(report.length, 'the tracker rendered its sliders').toBeGreaterThan(8);
  expect(report.filter((r) => r.ellipsis || r.cut).map((r) => r.text), 'truncated labels').toEqual([]);
  expect(report.filter((r) => r.lines > 2).map((r) => r.text), 'labels over two lines').toEqual([]);
  expect(report.filter((r) => !r.title.includes(r.text)).map((r) => r.text), 'labels without a full-label title').toEqual([]);
  expect(Math.min(...report.map((r) => r.cardW)), 'auto-fit keeps each card at 220px or more').toBeGreaterThanOrEqual(219);
});

test(`type scale: no body text under ${MIN_BODY_PX}px on Face Swap @ 1440px`, async ({ page }) => {
  await openFaceSwap(page);
  const offenders = await page.evaluate((min) => {
    const isPill = (el) => {
      for (let e = el, i = 0; e && i < 3; e = e.parentElement, i++) {
        const cs = getComputedStyle(e);
        const h = e.getBoundingClientRect().height;
        const filled = parseFloat(cs.borderTopWidth) > 0
          || (cs.backgroundColor && cs.backgroundColor !== 'rgba(0, 0, 0, 0)');
        if (parseFloat(cs.borderTopLeftRadius) >= h / 2 - 1 && h < 40 && filled) return true;
      }
      return false;
    };
    const out = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let n;
    while ((n = walker.nextNode())) {
      const t = n.textContent.trim();
      const el = n.parentElement;
      if (!t || !el || el.closest('script,style,noscript,[data-small-ok]')) continue;
      const cs = getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden') continue;
      const b = el.getBoundingClientRect();
      if (!b.width || !b.height) continue;
      const fs = parseFloat(cs.fontSize);
      if (fs >= min) continue;
      if (t.length <= 3) continue;                                              // glyph / tick
      if (el.closest('kbd')) continue;
      if (cs.textTransform === 'uppercase' && parseFloat(cs.letterSpacing) > 0) continue; // tag
      if (isPill(el)) continue;                                                 // badge
      out.push(`${fs}px "${t.slice(0, 40)}" <${el.tagName.toLowerCase()} class="${String(el.className).slice(0, 60)}">`);
    }
    return out;
  }, MIN_BODY_PX);
  expect(offenders, 'body text under the 12px floor').toEqual([]);
});

test(`range inputs have a >=${MIN_HIT_PX}px hit area`, async ({ page }) => {
  await openFaceSwap(page);
  const sizes = await page.evaluate(() => [...document.querySelectorAll('input[type="range"]')]
    .map((el) => el.getBoundingClientRect())
    .filter((r) => r.width && r.height)
    .map((r) => ({ w: Math.round(r.width), h: Math.round(r.height * 10) / 10 })));
  expect(sizes.length, 'range inputs are on the page').toBeGreaterThan(20);
  expect(sizes.filter((s) => s.h < MIN_HIT_PX || s.w < MIN_HIT_PX), 'range inputs under 24px').toEqual([]);

  // And the 24px box really is the target: a press 8px above the rail's centre
  // (inside the box, outside the 6px rail) still moves the thumb.
  const slider = page.locator('input[type="range"][aria-label="Original / Enhanced Blend"]');
  await slider.scrollIntoViewIfNeeded();
  const box = await slider.boundingBox();
  const before = await slider.inputValue();
  await page.mouse.click(box.x + box.width * 0.9, box.y + box.height / 2 - 8);
  expect(await slider.inputValue(), 'a click 8px off the rail still sets the value').not.toBe(before);
});

test('one run control: Start lives in the run bar only, and says why it is unavailable', async ({ page }) => {
  await openFaceSwap(page);
  const starts = page.getByRole('button', { name: /^(▶\s*)?Start( Swap(ping)?)?$/ });
  await expect(starts, 'exactly one Start control on the tab').toHaveCount(1);
  const start = page.getByRole('button', { name: /Start Swapping/ });
  await expect(start).toBeEnabled();
  await expect(start).toHaveAccessibleDescription(/Ready to swap/);
  // The dock keeps its own tools, but no second Start.
  await expect(page.locator('.fixed.bottom-6 button', { hasText: /Start/ })).toHaveCount(0);
  await expect(page.locator('.fixed.bottom-6 button', { hasText: /Preview/ })).toHaveCount(1);
});

test('with no media loaded Start is disabled and the reason is on screen beside it', async ({ page }) => {
  const withoutMedia = async (route) => {
    const res = await route.fetch();
    const json = await res.json();
    await route.fulfill({ response: res, json: { ...json, targets: [], target_faces: [], target_groups: [], target_faces_info: [], target_names: [] } });
  };
  await page.route('**/api/state', withoutMedia);
  await openFaceSwap(page);
  const start = page.getByRole('button', { name: /Start Swapping/ });
  await expect(start).toBeDisabled();
  const reason = page.locator('#run-status');
  await expect(reason).toBeVisible();
  await expect(reason).toHaveText(/Add a target video or image to start\./);
  await expect(start).toHaveAccessibleDescription(/Add a target video or image to start\./);
  // Visible text, not truncated, and next to the button (same bar).
  const [r, b] = await Promise.all([reason.boundingBox(), start.boundingBox()]);
  expect(Math.abs((r.y + r.height / 2) - (b.y + b.height / 2)), 'on the same row as the button').toBeLessThan(40);
  expect(await reason.evaluate((el) => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
});

test('the floating dock covers no control once the page is scrolled to the end', async ({ page }) => {
  await openFaceSwap(page);
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
  await page.waitForTimeout(400);
  const covered = await page.evaluate(() => {
    const dock = document.querySelector('.fixed.bottom-6');
    if (!dock) return ['(no dock found)'];
    const d = dock.getBoundingClientRect();
    return [...document.querySelectorAll('button, input, select, textarea, a[href]')]
      .filter((el) => !dock.contains(el) && !el.closest('header'))
      .filter((el) => {
        const r = el.getBoundingClientRect();
        if (!r.width || !r.height || getComputedStyle(el).visibility === 'hidden') return false;
        const cx = r.left + r.width / 2;
        const cy = r.top + r.height / 2;
        return cx >= d.left && cx <= d.right && cy >= d.top && cy <= d.bottom;
      })
      .map((el) => (el.getAttribute('aria-label') || el.textContent || el.tagName).trim().slice(0, 40));
  });
  expect(covered, 'controls under the dock at the end of the page').toEqual([]);
});

test('keyboard focus is not left under the dock (scroll-padding on the page)', async ({ page }) => {
  await openFaceSwap(page);
  const pad = await page.evaluate(() => parseFloat(getComputedStyle(document.documentElement).scrollPaddingBottom));
  expect(pad, 'bottom scroll-padding reserved for the dock').toBeGreaterThanOrEqual(96);
});
