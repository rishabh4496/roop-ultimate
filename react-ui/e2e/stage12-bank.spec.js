import { test, expect } from '@playwright/test';
import { openTab, makePng } from './helpers.js';

// Stage 12, part 2: the snapshot bank docked in the viewport header, the Quad-slot split compare, and the history stack with
// non-destructive branching - in a real Chromium against the production build.
//
// /api/preview is replaced by a handler whose swapped picture is a function of the posted `face_mask_blend` (the same device
// history.spec.js uses), so "this quadrant shows that slot's picture" is a pixel assertion, and a slider move is a real
// recipe change with a visible consequence. The mock's detections are stand-ins; nothing here says what a real swap looks like.

const RAW = { w: 640, h: 360 };
const SWAP = { w: 333, h: 222 };
const SLIDER = 'Face Mask Edge Blend';

async function openStage(page) {
  const raw = makePng(RAW.w, RAW.h, () => [200, 40, 40]);
  await page.route('**/api/target/preview*', (route) => route.fulfill({ status: 200, contentType: 'image/png', body: raw }));
  await page.route('**/api/preview', async (route) => {
    if (route.request().method() !== 'POST') return route.fallback();
    let blend = 20;
    try { blend = Number(JSON.parse(route.request().postData() || '{}').face_mask_blend ?? 20); } catch { /* not JSON */ }
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
/** The green channel at the centre of an <img> (a quadrant's picture). */
const imgGreen = (img) => img.evaluate((el) => {
  const c = document.createElement('canvas');
  c.width = el.naturalWidth; c.height = el.naturalHeight;
  c.getContext('2d').drawImage(el, 0, 0);
  return c.getContext('2d').getImageData(Math.floor(c.width / 2), Math.floor(c.height / 2), 1, 1).data[1];
});

test.describe('snapshot bank + Quad compare', () => {
  test('the bank is docked above the stage; a capture shows a thumbnail and the time', async ({ page }) => {
    const stage = await openStage(page);
    await slider(page).scrollIntoViewIfNeeded();
    const START = Number(await slider(page).inputValue());
    await expect.poll(() => shown(stage), { timeout: 10_000 }).toBe(START);

    const dock = page.getByTestId('viewport-header-dock');
    await expect(dock).toBeVisible();
    // it is the viewport's header: docked at the top edge of the stage, not down in the controls
    const d = await dock.boundingBox();
    const s = await stage.boundingBox();
    expect(d.y).toBeGreaterThanOrEqual(s.y);
    expect(d.y - s.y, "at the top of the stage").toBeLessThan(30);
    expect(d.x + d.width, "inside the stage").toBeLessThanOrEqual(s.x + s.width + 1);

    await expect(page.getByTestId('slot-time-A')).toHaveCount(0);
    await dock.getByRole('button', { name: /^Slot A, empty/ }).click();
    await expect(dock.getByRole('button', { name: /^Slot A, frame kept/ })).toBeVisible();
    await expect(page.getByTestId('slot-time-A')).toHaveText(/\d\d:\d\d:\d\d/);
    await expect(page.getByTestId('slot-time-B')).toHaveCount(0);
    // the badge's thumbnail really holds the kept frame (its green channel is the recipe's)
    const thumbGreen = await dock.locator('[data-slot="A"] canvas').evaluate((c) => c.getContext('2d').getImageData(Math.floor(c.width / 2), Math.floor(c.height / 2), 1, 1).data[1]);
    expect(Math.abs(thumbGreen - START)).toBeLessThan(6);
  });

  test('Quad needs two slots; with two it shows each slot its own picture and exactly how they differ', async ({ page }) => {
    const stage = await openStage(page);
    await slider(page).scrollIntoViewIfNeeded();
    const START = Number(await slider(page).inputValue());
    await expect.poll(() => shown(stage), { timeout: 10_000 }).toBe(START);
    const quadBtn = page.getByTestId('slot-quad-toggle');
    await expect(quadBtn).toBeDisabled();                                    // nothing captured

    await page.getByRole('button', { name: /^Slot A, empty/ }).click();
    await expect(quadBtn).toBeDisabled();                                    // one slot is not a comparison

    await slider(page).focus();
    await page.keyboard.press(START >= 100 ? 'Home' : 'End');            // the mock backend is shared: START may already be at an end
    const TOP = Number(await slider(page).inputValue());
    expect(TOP).not.toBe(START);
    await expect.poll(() => shown(stage), { timeout: 10_000 }).toBe(TOP);
    await page.getByRole('button', { name: /^Slot B, empty/ }).click();
    await expect(page.getByRole('button', { name: /^Slot B, frame kept/ })).toBeVisible();
    await expect(quadBtn).toBeEnabled();

    await quadBtn.click();
    const quad = page.getByTestId('quad-slot-compare');
    await expect(quad).toBeVisible();
    await expect(page.locator('.preview-stage')).toHaveCount(0);            // the quad takes the stage over
    await expect(quad.locator('img')).toHaveCount(2);                        // A and B have pictures; C and D are empty
    await expect(page.getByTestId('quad-empty-C')).toBeVisible();
    await expect(page.getByTestId('quad-empty-D')).toBeVisible();

    // the two pictures are the two slots' own: A is the start recipe, B the top one
    const imgs = quad.locator('img');
    const greens = [await imgGreen(imgs.nth(0)), await imgGreen(imgs.nth(1))];
    expect(Math.abs(greens[0] - START)).toBeLessThan(6);
    expect(Math.abs(greens[1] - TOP)).toBeLessThan(6);
    expect(Math.abs(greens[0] - greens[1])).toBeGreaterThan(20);

    // A is the baseline; B names the one slider that changed, with both values
    await expect(page.getByTestId('quad-baseline-A')).toBeVisible();
    const info = page.getByTestId('quad-info-B');
    await expect(info).toContainText('1 difference from A');
    await expect(info).toContainText('Mask feather');                          // the label the history stack uses for this setting
    await expect(info).toContainText(`${START}`);
    await expect(info).toContainText(`${TOP}`);
    // and the table lists it once, with the differing cell marked
    const row = page.locator('[data-testid^="quad-row-"]');
    await expect(row).toHaveCount(1);
    await expect(row.first().locator('td[data-differs="1"]')).toHaveCount(1);

    // move the baseline to B: now A is the one that differs
    await page.getByTestId('quad-set-baseline-B').click();
    await expect(page.getByTestId('quad-baseline-B')).toBeVisible();
    await expect(page.getByTestId('quad-info-A')).toContainText('1 difference from B');

    // all four share ONE zoom: scrolling over one quadrant zooms the others too
    const cells = quad.locator('[data-cmp-cell]');
    const tf = (n) => cells.nth(n).locator('img').evaluate((el) => el.parentElement.style.transform);
    const before = await tf(0);
    await cells.nth(1).scrollIntoViewIfNeeded();                             // a wheel at coordinates outside the window goes nowhere
    const box = await cells.nth(1).boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.wheel(0, -300);
    await expect.poll(() => tf(0)).not.toBe(before);
    expect(await tf(1)).toBe(await tf(0));

    // an empty quadrant can capture the live look
    await page.getByTestId('quad-capture-C').click();
    await expect(page.getByTestId('quad-empty-C')).toHaveCount(0);
    await expect(quad.locator('img')).toHaveCount(3);

    // Recall returns to the single viewport with that recipe in force
    await page.getByTestId('quad-recall-A').click();
    await expect(quad).toHaveCount(0);
    await expect.poll(async () => Number(await slider(page).inputValue()), { timeout: 3000 }).toBe(START);
  });

  test('Back to the viewport closes it; two slots of one recipe say so', async ({ page }) => {
    const stage = await openStage(page);
    await slider(page).scrollIntoViewIfNeeded();
    await expect.poll(() => shown(stage), { timeout: 10_000 }).toBeGreaterThan(0);
    await page.getByRole('button', { name: /^Slot A, empty/ }).click();
    await page.getByRole('button', { name: /^Slot B, empty/ }).click();       // same recipe twice: still two filled slots
    await page.getByTestId('slot-quad-toggle').click();
    await expect(page.getByTestId('quad-slot-compare')).toBeVisible();
    await expect(page.getByTestId('quad-info-B')).toContainText('Same recipe as A');
    await page.getByTestId('quad-close').click();
    await expect(page.getByTestId('quad-slot-compare')).toHaveCount(0);
    await expect(page.locator('.preview-stage').first()).toBeVisible();
  });
});

test.describe('history stack', () => {
  const group = (page) => page.getByRole('heading', { name: /^History/ }).getByRole('button');

  test('every change is a clickable state; an edit after an undo keeps the old path as a branch you can switch back to', async ({ page }) => {
    const stage = await openStage(page);
    await slider(page).scrollIntoViewIfNeeded();
    await expect.poll(() => shown(stage), { timeout: 10_000 }).toBeGreaterThan(0);
    const START = Number(await slider(page).inputValue());

    await slider(page).focus();
    const first = START >= 100 ? 'Home' : 'End';
    await page.keyboard.press(first);
    const TOP = Number(await slider(page).inputValue());
    await page.waitForTimeout(900);                                          // past the 700 ms coalescing window
    await page.keyboard.press(first === 'End' ? 'Home' : 'End');
    const LOW = Number(await slider(page).inputValue());
    await page.waitForTimeout(900);
    expect(new Set([START, TOP, LOW]).size).toBeGreaterThan(1);

    await group(page).click();
    const panel = page.getByTestId('history-stack');
    await expect(panel).toBeVisible();
    const rows = panel.locator('[data-history-stack-row]');
    await expect(rows.first()).toHaveAttribute('data-kind', 'parameter');
    await expect(rows.first()).toHaveAttribute('aria-current', 'step');
    expect(await rows.count()).toBeGreaterThanOrEqual(3);                    // Start + two edits
    await expect(page.getByTestId('history-stack-position')).toContainText(/Step (\d+) of \1/);   // the cursor is on the newest step

    // undo one step: the slider goes to the earlier edit and Redo is available
    await page.getByTestId('history-stack-undo').click();
    await expect.poll(async () => Number(await slider(page).inputValue())).toBe(TOP);
    await expect(page.getByTestId('history-stack-redo')).toBeEnabled();
    await expect(panel.locator('[data-history-branch]')).toHaveCount(0);

    // a NEW edit from here: the step we undid (the move to LOW) must survive as a branch, not be deleted
    await slider(page).focus();
    await page.keyboard.press(TOP <= 0 ? 'ArrowRight' : 'ArrowLeft');
    const NEW = Number(await slider(page).inputValue());
    expect(NEW).not.toBe(TOP);
    await page.waitForTimeout(900);
    await expect(page.getByTestId('history-stack-redo')).toBeDisabled();     // the redo PATH is gone...
    const branch = panel.locator('[data-history-branch]');
    await expect(branch).toHaveCount(1);                                     // ...but the branch is listed
    await expect(branch).toContainText('1 step');

    // switch to it: the slider is back at LOW, and the path we left is now the branch
    await branch.click();
    await expect.poll(async () => Number(await slider(page).inputValue()), { timeout: 3000 }).toBe(LOW);
    await expect(branch).toHaveCount(1);
    // and switching back restores the new edit
    await branch.click();
    await expect.poll(async () => Number(await slider(page).inputValue()), { timeout: 3000 }).toBe(NEW);

    // time-travel by clicking a row: the one just under the newest is the move to TOP (the first row is not: the page
    // may record a settings load before the first edit, and the mock backend is shared between tests)
    const last = (await rows.count()) - 1;
    await panel.locator(`[data-history-stack-row="${last - 1}"]`).click();
    await expect.poll(async () => Number(await slider(page).inputValue()), { timeout: 3000 }).toBe(TOP);
    await expect(panel.locator(`[data-history-stack-row="${last - 1}"]`)).toHaveAttribute('aria-current', 'step');
  });

  test('the badge counts the steps, and the group collapses again', async ({ page }) => {
    const stage = await openStage(page);
    await slider(page).scrollIntoViewIfNeeded();
    await expect.poll(() => shown(stage), { timeout: 10_000 }).toBeGreaterThan(0);
    await slider(page).focus();
    await page.keyboard.press((Number(await slider(page).inputValue()) >= 100) ? 'Home' : 'End');
    await page.waitForTimeout(900);
    const badge = Number((await group(page).textContent()).replace(/\D/g, ''));
    expect(badge, 'the closed group already says how many steps it holds').toBeGreaterThanOrEqual(1);
    await group(page).click();
    await expect(page.getByTestId('history-stack-position')).toContainText(`of ${badge}`);
    await group(page).click();                                              // collapsible: closes again
    await expect(page.getByTestId('history-stack')).toHaveCount(0);
  });
});
