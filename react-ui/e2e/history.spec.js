import { test, expect } from '@playwright/test';
import { openTab, makePng } from './helpers.js';

// Stage 23 in a real browser: the undo/redo history, the History popover, mask-stroke undo, and the
// A/B/C/D snapshot slots including the ImageBitmaps they keep.
//
// The mock backend is replaced for /api/preview by a handler whose swapped frame is a function of
// the posted `face_mask_blend`: two recipes therefore look different, so "the slot flipped the
// picture" is a pixel assertion and not a hope. The swapped frame is a different size from the raw
// one (333x222 vs 640x360) so ImageBitmap.close() calls can be attributed to it.

const RAW = { w: 640, h: 360 };
const SWAP = { w: 333, h: 222 };
const SLIDER = 'Face Mask Edge Blend';

// Count close() on bitmaps of the swapped frame's size. Installed before any page script runs.
const countCloses = () => {
  const proto = window.ImageBitmap && window.ImageBitmap.prototype;
  if (!proto || proto.__closeCounted) return;
  const orig = proto.close;
  window.__swapCloses = 0;
  proto.close = function close() {
    if (this.width === 333 && this.height === 222) window.__swapCloses += 1;
    return orig.call(this);
  };
  proto.__closeCounted = true;
};

async function openStage(page, { slowAfter = Infinity, slowMs = 0 } = {}) {
  const raw = makePng(RAW.w, RAW.h, () => [200, 40, 40]);
  let served = 0;
  await page.addInitScript(countCloses);
  await page.route('**/api/target/preview*', (route) => route.fulfill({ status: 200, contentType: 'image/png', body: raw }));
  await page.route('**/api/preview', async (route) => {
    if (route.request().method() !== 'POST') return route.fallback();
    let blend = 20;
    try { blend = Number(JSON.parse(route.request().postData() || '{}').face_mask_blend ?? 20); } catch { /* not JSON */ }
    served += 1;
    if (served > slowAfter) await new Promise((r) => setTimeout(r, slowMs));
    const png = makePng(SWAP.w, SWAP.h, () => [20, Math.min(255, Math.round(blend)), 200]);
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ image: `data:image/png;base64,${png.toString('base64')}`, faces: [], person_ids: [], kps: [], pose: [] }),
    });
  });
  await openTab(page, 'faceswap');
  const stage = page.locator('.preview-stage').first();
  await expect(stage).toBeVisible({ timeout: 15_000 });
  await stage.hover();
  return stage;
}


const slider = (page) => page.getByRole('slider', { name: SLIDER, exact: true });

/** The swapped frame's green channel at the stage centre: our stand-in for "which recipe is on screen". */
const shown = (stage) => stage.locator('canvas').first().evaluate((c) => {
  const d = c.getContext('2d').getImageData(Math.floor(c.width / 2), Math.floor(c.height / 2), 1, 1).data;
  return d[1];
});

const actionCount = async (page) => {
  const label = await page.getByTestId('history-toggle').getAttribute('aria-label');
  return Number(/(\d+) actions/.exec(label)[1]);
};

const setBlendByDrag = async (page, fractions) => {
  const box = await slider(page).boundingBox();
  const y = box.y + box.height / 2;
  await page.mouse.move(box.x + box.width * fractions[0], y);
  await page.mouse.down();
  for (const f of fractions.slice(1)) await page.mouse.move(box.x + box.width * f, y, { steps: 4 });
  await page.mouse.up();
  await page.waitForTimeout(700);                 // slider commit debounce + transaction settle
};

test('a slider drag is ONE history entry; Ctrl+Z / Ctrl+Y restore it; the popover lists and time-travels', async ({ page }) => {
  await openStage(page);
  await slider(page).scrollIntoViewIfNeeded();
  const base = await actionCount(page);
  const startValue = await slider(page).inputValue();

  await setBlendByDrag(page, [0.2, 0.4, 0.6, 0.8]);
  const dragged = Number(await slider(page).inputValue());
  expect(dragged, 'the drag moved the slider').toBeGreaterThan(Number(startValue));
  expect(await actionCount(page), 'sixteen pointer moves, one entry').toBe(base + 1);

  await page.getByTestId('history-toggle').click();
  const top = page.locator('[data-testid="history-panel"] li').first();
  await expect(top).toContainText('Changed');
  await expect(top).toContainText('->');
  await page.keyboard.press('Escape');

  await page.mouse.move(5, 5);
  await page.keyboard.press('Control+z');
  await expect.poll(async () => Number(await slider(page).inputValue())).toBeLessThan(dragged);
  expect(await actionCount(page), 'undo is not an action').toBe(base + 1);
  await page.keyboard.press('Control+y');
  await expect.poll(async () => Number(await slider(page).inputValue())).toBe(dragged);
  await page.keyboard.press('Control+z');
  await page.keyboard.press('Control+Shift+z');
  await expect.poll(async () => Number(await slider(page).inputValue())).toBe(dragged);

  // time travel: click the oldest row ("Initial state") and the slider is back where it began
  await page.getByTestId('history-toggle').click();
  await page.locator('[data-history-row="0"]').click();
  await expect.poll(async () => Number(await slider(page).inputValue())).toBeLessThan(dragged);
});

test('Ctrl+Z inside a text field is left to the field', async ({ page }) => {
  await openStage(page);
  const base = await actionCount(page);
  await slider(page).scrollIntoViewIfNeeded();
  await setBlendByDrag(page, [0.3, 0.5]);
  const moved = Number(await slider(page).inputValue());
  const field = page.locator('#history-probe');
  await page.evaluate(() => { const i = document.createElement('input'); i.type = 'text'; i.id = 'history-probe'; document.body.appendChild(i); });
  await field.focus();
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(150);
  expect(Number(await slider(page).inputValue()), 'history did not fire').toBe(moved);
  expect(await actionCount(page)).toBe(base + 1);
});

test('mask brush: each stroke is an undo step, undo erases exactly that stroke, redo repaints it', async ({ page }) => {
  const stage = await openStage(page);
  await stage.getByRole('button', { name: 'Mask Brush' }).click();
  const canvas = stage.locator('canvas.cursor-none');
  await expect(canvas).toBeVisible();
  const dims = await canvas.evaluate((c) => ({ w: c.width, h: c.height }));
  const box = await canvas.boundingBox();
  const at = (fx, fy) => ({ x: box.x + box.width * fx, y: box.y + box.height * fy });
  const alpha = (fx, fy) => canvas.evaluate((c, [x, y]) => c.getContext('2d').getImageData(Math.round(c.width * x), Math.round(c.height * y), 1, 1).data[3], [fx, fy]);
  const stroke = async (fx, fy) => {
    const p = at(fx, fy);
    await page.mouse.move(p.x, p.y);
    await page.mouse.down();
    await page.mouse.move(p.x + 6, p.y + 2, { steps: 3 });
    await page.mouse.up();
    await page.waitForTimeout(150);
  };

  await stroke(0.3, 0.4);
  await stroke(0.7, 0.6);
  expect(await alpha(0.3, 0.4)).toBeGreaterThan(0);
  expect(await alpha(0.7, 0.6)).toBeGreaterThan(0);
  expect(dims.w).toBeGreaterThan(0);

  await page.getByTestId('history-toggle').click();
  await expect(page.locator('[data-testid="history-panel"] li').first()).toContainText('Painted Occlusion Mask Stroke');
  await page.keyboard.press('Escape');

  await page.mouse.move(5, 5);
  await page.keyboard.press('Control+z');
  await expect.poll(() => alpha(0.7, 0.6)).toBe(0);
  expect(await alpha(0.3, 0.4), 'the earlier stroke survives').toBeGreaterThan(0);
  await page.keyboard.press('Control+z');
  await expect.poll(() => alpha(0.3, 0.4)).toBe(0);
  await page.keyboard.press('Control+y');
  await expect.poll(() => alpha(0.3, 0.4)).toBeGreaterThan(0);
  await page.keyboard.press('Control+y');
  await expect.poll(() => alpha(0.7, 0.6)).toBeGreaterThan(0);
});

test('snapshot slots: capture, instant flip while the real render is slow, key 1-4, Ctrl+Z, and bitmaps are closed', async ({ page }) => {
  // The first two renders are fast (initial + the one after the first edit); everything later takes 2.5 s.
  const stage = await openStage(page, { slowAfter: 2, slowMs: 2500 });
  await slider(page).scrollIntoViewIfNeeded();
  const START = Number(await slider(page).inputValue());
  await expect.poll(() => shown(stage), { timeout: 10_000 }).toBe(START);

  // Recipe A: blend 20 (initial). Capture it.
  await page.getByRole('button', { name: /^Slot A, empty/ }).click();
  await expect(page.getByRole('button', { name: /^Slot A, frame kept/ })).toBeVisible();

  // Move to a different recipe and let its (fast) render land.
  await slider(page).focus();
  await page.keyboard.press('End');
  const TOP = Number(await slider(page).inputValue());
  await expect.poll(() => shown(stage), { timeout: 10_000 }).toBe(TOP);
  await page.getByRole('button', { name: /^Slot B, empty/ }).click();
  await expect(page.getByRole('button', { name: /^Slot B, frame kept/ })).toBeVisible();

  // Recall A: its render is now slow, so what appears at once can only be the kept bitmap.
  await page.getByRole('button', { name: /^Slot A,/ }).click();
  await expect.poll(() => shown(stage), { timeout: 1500 }).toBe(START);
  expect(Number(await slider(page).inputValue()), "the recipe came back with the picture").toBe(START);

  // Key 2 flips to B the same way, with the stage (not a text field) focused.
  await page.mouse.move(5, 5);
  await page.keyboard.press('2');
  await expect.poll(() => shown(stage), { timeout: 1500 }).toBe(TOP);
  await page.keyboard.press('1');
  await expect.poll(() => shown(stage), { timeout: 1500 }).toBe(START);

  // A recall is an undoable action.
  await page.keyboard.press('Control+z');
  await expect.poll(async () => Number(await slider(page).inputValue()), { timeout: 3000 }).toBe(TOP);

  // Bitmaps: overwriting a slot closes the bitmap it held; clearing closes the other.
  const closes = () => page.evaluate(() => window.__swapCloses);
  await page.waitForTimeout(3500);                   // let the slow renders land so nothing else is closing
  const c0 = await closes();
  await page.getByRole('button', { name: /^Slot B,/ }).click({ modifiers: ['Shift'] });
  await page.waitForTimeout(700);
  const c1 = await closes();
  expect(c1 - c0, 'overwriting Slot B closed its old bitmap exactly once').toBe(1);
  await page.getByRole('button', { name: /^Slot A,/ }).click({ button: 'right' });
  await page.getByRole('menuitem', { name: 'Clear slot' }).click();
  await page.waitForTimeout(700);
  expect((await closes()) - c1, 'clearing Slot A closed its bitmap exactly once').toBe(1);
  await expect(page.getByRole('button', { name: /^Slot A, empty/ })).toBeVisible();
});

test('split-wipe reference: Slot A on one side of the wipe, the live result on the other', async ({ page }) => {
  const stage = await openStage(page);
  await slider(page).scrollIntoViewIfNeeded();
  const START = Number(await slider(page).inputValue());
  await expect.poll(() => shown(stage), { timeout: 10_000 }).toBe(START);
  await page.getByRole('button', { name: /^Slot A, empty/ }).click();
  await expect(page.getByRole('button', { name: /^Slot A, frame kept/ })).toBeVisible();
  await slider(page).focus();
  await page.keyboard.press('End');
  const TOP = Number(await slider(page).inputValue());
  await expect.poll(() => shown(stage), { timeout: 10_000 }).toBe(TOP);

  await page.getByRole('button', { name: /^Slot A,/ }).click({ button: 'right' });
  await page.getByRole('menuitem', { name: 'Assign as Split-Wipe Reference' }).click();
  await expect(page.getByRole('button', { name: /split-wipe reference/ })).toBeVisible();
  // Reference assigned => the compare wipe is on, labelled with the slot. Slot A is 20, the live result 200.
  await expect(stage.getByRole('slider', { name: 'Compare Slot A against Live result' })).toBeVisible();
  const sides = await stage.locator('canvas').first().evaluate((c) => {
    const row = Math.floor(c.height / 2);
    const g = (x) => c.getContext('2d').getImageData(Math.floor(c.width * x), row, 1, 1).data[1];
    return [g(0.05), g(0.95)];
  });
  expect(sides.some((v) => Math.abs(v - START) <= 3), `one side shows Slot A (${START}): ${sides}`).toBe(true);
  expect(sides.some((v) => Math.abs(v - TOP) <= 3), `the other shows the live result (${TOP}): ${sides}`).toBe(true);
});
