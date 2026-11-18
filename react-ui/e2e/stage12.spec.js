import { test, expect } from '@playwright/test';
import { openTab } from './helpers.js';

// Stage 12, in a real Chromium against the production build with the mock server standing in for app/api.py.
//
//   curve drawer     overlay of several curves, Bezier tangent handles, value / frame grid snapping, box selection and a
//                    group drag that is ONE undo step, the drawer's expand toggle
//   (appended below as each part lands: face menu + role badges, snapshot bank + quad compare, history stack)
//
// The mock's pictures and detections are stand-ins, so none of this says what a real swap looks like. What it does prove is
// that the pointer gestures reach the model: every assertion reads the curve the editor PERSISTED, not the pixels it drew.

const TL = '[data-testid="timeline"]';
const GRAPH_H = 168;
const PAD = 14;

async function resizeClip(request, frames) {
  const res = await request.post('/api/mock/target_frames', { data: { index: 0, frames } });
  expect(res.ok(), 'the mock must accept the clip resize').toBeTruthy();
}

async function openTimeline(page) {
  await openTab(page, 'faceswap');
  const tl = page.locator(TL);
  await tl.waitFor({ state: 'visible', timeout: 20_000 });
  await tl.scrollIntoViewIfNeeded();
  return tl;
}

async function openCurves(page) {
  const tl = await openTimeline(page);
  await page.getByTestId('curve-editor-toggle').click();
  await expect(page.getByTestId('curve-editor')).toBeVisible();
  await page.getByTestId('curve-graph').evaluate((el) => el.scrollIntoView({ block: 'center' }));
  return tl;
}

/** Page coordinates of (frame, value) on the graph, for a parameter range [min, max] and the drawer's current height. */
async function graphPoint(page, tl, frame, value, { min = 0, max = 1, h = GRAPH_H } = {}) {
  // Playwright scrolls a clicked tab (above the graph) into view, which can push the graph out of the timeline's visible
  // area: a click on its coordinates then lands on whatever is there instead. Bring the graph back first.
  await page.getByTestId('curve-graph').evaluate((el) => el.scrollIntoView({ block: 'center' }));
  const g = await page.getByTestId('curve-graph').boundingBox();
  const vs = Number(await tl.getAttribute('data-view-start'));
  const ve = Number(await tl.getAttribute('data-view-end'));
  const yPx = PAD + (1 - (value - min) / (max - min)) * (h - 2 * PAD);
  return { x: g.x + ((frame - vs) / (ve - vs)) * g.width, y: g.y + (yPx / h) * g.height };
}

/** The curves exactly as the editor persisted them for this clip: { paramId: [{frame, value, interp, bz?}] }. */
const tracksOf = (page) => page.evaluate(() => {
  const k = Object.keys(localStorage).find((x) => /^roop_curves_(?!enabled_)/.test(x));
  return k ? JSON.parse(localStorage.getItem(k)) : {};
});
const framesOf = async (page, id) => ((await tracksOf(page))[id] || []).map((k) => k.frame);

async function dbl(page, tl, frame, value, range) {
  const p = await graphPoint(page, tl, frame, value, range);
  await page.mouse.dblclick(p.x, p.y);
}

async function drag(page, from, to, steps = 12) {
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps });
  await page.mouse.up();
}

test.describe('curve drawer (300-frame clip)', () => {
  test.beforeEach(async ({ request }) => { await resizeClip(request, 300); });
  test.afterEach(async ({ request }) => {
    await request.post('/api/queue/stop').catch(() => {});
    await request.post('/api/queue/clear', { data: { keep_running: false } }).catch(() => {});
    await resizeClip(request, 180);
  });

  test('dragging a tangent handle makes the segment Custom Bezier, as ONE undo step', async ({ page }) => {
    const tl = await openCurves(page);
    await dbl(page, tl, 60, 0.3);
    await dbl(page, tl, 200, 0.9);
    expect((await framesOf(page, 'swap_opacity')).length).toBe(2);
    expect((await tracksOf(page)).swap_opacity[0].interp).toBe('smooth');

    await page.getByTestId('curve-key-60').click();                         // select the key: its segment shows handles
    const h1 = page.getByTestId('curve-handle-60-1');
    await expect(h1).toBeVisible();
    await expect(page.getByTestId('curve-handle-60-2')).toBeVisible();

    const from = await h1.boundingBox();
    const to = await graphPoint(page, tl, 150, 0.6);                        // x = (150-60)/140 = 0.64, y = (0.6-0.3)/0.6 = 0.5
    await drag(page, { x: from.x + from.width / 2, y: from.y + from.height / 2 }, to);
    await expect.poll(async () => (await tracksOf(page)).swap_opacity[0].interp).toBe('bezier');
    const k = (await tracksOf(page)).swap_opacity[0];
    expect(k.bz.length).toBe(4);
    expect(Math.abs(k.bz[0] - 0.643)).toBeLessThan(0.06);
    expect(Math.abs(k.bz[1] - 0.5)).toBeLessThan(0.08);
    expect(Math.abs(k.bz[2] - 2 / 3)).toBeLessThan(0.01);                    // the other handle did not move
    expect(k.bz[3]).toBe(1);
    await expect(page.getByTestId('curve-interp-bezier')).toHaveAttribute('aria-pressed', 'true');

    // one drag, one history entry
    await page.keyboard.press('Control+z');
    await expect.poll(async () => (await tracksOf(page)).swap_opacity[0].interp).toBe('smooth');
    await page.keyboard.press('Control+y');
    await expect.poll(async () => (await tracksOf(page)).swap_opacity[0].interp).toBe('bezier');
  });

  test('the Custom Bezier button starts from the shape the segment already has', async ({ page }) => {
    const tl = await openCurves(page);
    await dbl(page, tl, 60, 0.3);
    await dbl(page, tl, 200, 0.9);
    await page.getByTestId('curve-key-60').click();
    await page.getByTestId('curve-interp-linear').click();
    await page.getByTestId('curve-interp-bezier').click();
    const k = (await tracksOf(page)).swap_opacity[0];
    expect(k.interp).toBe('bezier');
    expect(Math.abs(k.bz[0] - 1 / 3)).toBeLessThan(0.01);                    // a linear segment's diagonal
    expect(Math.abs(k.bz[1] - 1 / 3)).toBeLessThan(0.01);
    await page.getByTestId('curve-interp-smooth').click();
    expect((await tracksOf(page)).swap_opacity[0].bz).toBeUndefined();
  });

  test('a box selects keys, the group drags as one block and one Ctrl+Z puts every key back', async ({ page }) => {
    const tl = await openCurves(page);
    await dbl(page, tl, 60, 0.3);
    await dbl(page, tl, 100, 0.5);
    await dbl(page, tl, 140, 0.7);
    await dbl(page, tl, 260, 0.9);                                           // outside the box
    expect(await framesOf(page, 'swap_opacity')).toEqual([60, 100, 140, 260]);

    const a = await graphPoint(page, tl, 40, 0.95);
    const b = await graphPoint(page, tl, 160, 0.2);
    await page.mouse.move(a.x, a.y);
    await page.mouse.down();
    await page.mouse.move((a.x + b.x) / 2, (a.y + b.y) / 2, { steps: 6 });
    await expect(page.getByTestId('curve-marquee')).toBeVisible();           // the rubber band is on screen while dragging
    await page.mouse.move(b.x, b.y, { steps: 6 });
    await page.mouse.up();
    await expect(page.getByTestId('curve-sel-text')).toContainText('3 keys selected');
    await expect(page.getByTestId('curve-key-100')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('curve-key-260')).toHaveAttribute('aria-pressed', 'false');

    // drag the middle key 30 frames right and up: all three move rigidly
    const k100 = await page.getByTestId('curve-key-100').boundingBox();
    const to = await graphPoint(page, tl, 130, 0.6);
    await drag(page, { x: k100.x + k100.width / 2, y: k100.y + k100.height / 2 }, to);
    await expect.poll(() => framesOf(page, 'swap_opacity')).toEqual([90, 130, 170, 260]);
    const after = (await tracksOf(page)).swap_opacity;
    expect(Math.abs(after[0].value - 0.4)).toBeLessThan(0.03);               // 0.3 + 0.1
    expect(Math.abs(after[1].value - 0.6)).toBeLessThan(0.03);
    expect(Math.abs(after[2].value - 0.8)).toBeLessThan(0.03);
    expect(after[3].value).toBe(0.9);                                        // untouched

    await page.keyboard.press('Control+z');
    await expect.poll(() => framesOf(page, 'swap_opacity')).toEqual([60, 100, 140, 260]);
  });

  test('a group cannot be dragged through an unselected key', async ({ page }) => {
    const tl = await openCurves(page);
    await dbl(page, tl, 60, 0.3);
    await dbl(page, tl, 100, 0.5);
    await dbl(page, tl, 200, 0.8);
    const a = await graphPoint(page, tl, 40, 0.95);
    const b = await graphPoint(page, tl, 110, 0.1);
    await drag(page, a, b);                                                   // box over the first two
    await expect(page.getByTestId('curve-sel-text')).toContainText('2 keys selected');
    const k100 = await page.getByTestId('curve-key-100').boundingBox();
    const to = await graphPoint(page, tl, 280, 0.5);
    await drag(page, { x: k100.x + k100.width / 2, y: k100.y + k100.height / 2 }, to);
    const frames = await framesOf(page, 'swap_opacity');
    expect(frames[2]).toBe(200);
    expect(frames[1]).toBe(199);                                              // one frame short of the neighbour
    expect(frames[0]).toBe(159);                                              // and the group kept its 40-frame spacing
  });

  test('Delete removes the whole selection; Escape clears it first', async ({ page }) => {
    const tl = await openCurves(page);
    await dbl(page, tl, 60, 0.3);
    await dbl(page, tl, 120, 0.5);
    await dbl(page, tl, 180, 0.7);
    await page.getByTestId('curve-select-all').click();
    await expect(page.getByTestId('curve-sel-text')).toContainText('3 keys selected');
    await page.getByTestId('curve-key-120').focus();
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('curve-sel-text')).not.toContainText('3 keys selected');
    await page.getByTestId('curve-select-all').click();
    await page.getByTestId('curve-key-120').focus();
    await page.keyboard.press('Delete');
    await expect.poll(() => framesOf(page, 'swap_opacity')).toEqual([]);
    await page.keyboard.press('Control+z');
    await expect.poll(() => framesOf(page, 'swap_opacity')).toEqual([60, 120, 180]);   // one step brings all three back
  });

  test('grid snapping: a double-click lands on the grid, Alt lands free', async ({ page }) => {
    const tl = await openCurves(page);
    await page.getByTestId('curve-snap-value').selectOption('4');
    await page.getByTestId('curve-snap-frame').selectOption('10');
    await dbl(page, tl, 63, 0.3);                                             // nearest grid point: frame 60, value 0.25 (0.3 is 0.075 clear of the 0.375 boundary)
    let k = ((await tracksOf(page)).swap_opacity || [])[0];
    expect(k, 'a key was added').toBeTruthy();
    expect(k.frame).toBe(60);
    expect(k.value).toBe(0.25);
    // the choice is remembered
    await page.reload();
    await openTimeline(page);
    await page.getByTestId('curve-editor-toggle').click();
    await expect(page.getByTestId('curve-snap-value')).toHaveValue('4');
    await expect(page.getByTestId('curve-snap-frame')).toHaveValue('10');
    // drag it free with Alt held
    const to = await graphPoint(page, tl, 97, 0.37);                          // first: it scrolls the graph into view
    const kb = await page.getByTestId('curve-key-60').boundingBox();
    await page.keyboard.down('Alt');
    await drag(page, { x: kb.x + kb.width / 2, y: kb.y + kb.height / 2 }, to);
    await page.keyboard.up('Alt');
    k = (await tracksOf(page)).swap_opacity[0];
    expect(Math.abs(k.frame - 97)).toBeLessThanOrEqual(1);
    expect(Math.abs(k.value - 0.37)).toBeLessThan(0.05);
    // and snapped without it
    const to2 = await graphPoint(page, tl, 143, 0.7);
    const kc = await page.getByTestId(`curve-key-${k.frame}`).boundingBox();
    await drag(page, { x: kc.x + kc.width / 2, y: kc.y + kc.height / 2 }, to2);
    k = (await tracksOf(page)).swap_opacity[0];
    expect(k.frame).toBe(140);
    expect(k.value).toBe(0.75);
  });

  test('several curves are drawn at once; pressing another curve\'s dot makes it the active one', async ({ page }) => {
    const tl = await openCurves(page);
    await dbl(page, tl, 60, 0.3);
    await dbl(page, tl, 200, 0.9);
    await page.getByTestId('curve-param-mask_feather').click();              // mask feather: 0..200 px
    await dbl(page, tl, 100, 40, { min: 0, max: 200 });
    await dbl(page, tl, 220, 160, { min: 0, max: 200 });
    const t = await tracksOf(page);
    expect(Object.keys(t).sort()).toEqual(['mask_feather', 'swap_opacity']);
    // both curves are on screen: the active one's keys are big buttons, the other's are dots
    await expect(page.getByTestId('curve-key-100')).toBeVisible();
    await expect(page.getByTestId('curve-dot-swap_opacity-60')).toBeVisible();
    await expect(page.getByTestId('curve-dot-swap_opacity-200')).toBeVisible();
    // the overlay is drawn: the canvas holds pixels of BOTH curve colours (#34d399 and #f59e0b)
    const colours = await page.getByTestId('curve-graph').locator('canvas').evaluate((c) => {
      const { data } = c.getContext('2d').getImageData(0, 0, c.width, c.height);
      const near = (i, r, g, b) => data[i + 3] > 40 && Math.abs(data[i] - r) < 40 && Math.abs(data[i + 1] - g) < 40 && Math.abs(data[i + 2] - b) < 40;
      let a = 0; let b = 0;
      for (let i = 0; i < data.length; i += 4) { if (near(i, 52, 211, 153)) a++; if (near(i, 245, 158, 11)) b++; }
      return { opacity: a, feather: b };
    });
    expect(colours.opacity, 'the inactive curve is still drawn').toBeGreaterThan(100);
    expect(colours.feather, 'the active curve is drawn').toBeGreaterThan(100);
    // hide it with its eye
    await page.getByTestId('curve-eye-swap_opacity').click();
    await expect(page.getByTestId('curve-dot-swap_opacity-60')).toHaveCount(0);
    await page.getByTestId('curve-eye-swap_opacity').click();
    await expect(page.getByTestId('curve-dot-swap_opacity-60')).toBeVisible();
    // press a dot: that curve becomes the active one
    await page.getByTestId('curve-dot-swap_opacity-60').click();
    await expect(page.getByTestId('curve-param-swap_opacity')).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByTestId('curve-key-60')).toBeVisible();
    await expect(page.getByTestId('curve-dot-mask_feather-100')).toBeVisible();
  });

  test('Expand makes the drawer taller, and the keys still land where the pointer is', async ({ page }) => {
    const tl = await openCurves(page);
    const before = (await page.getByTestId('curve-graph').boundingBox()).height;
    await page.getByTestId('curve-expand').click();
    await expect(page.getByTestId('curve-expand')).toHaveAttribute('aria-pressed', 'true');
    const after = (await page.getByTestId('curve-graph').boundingBox()).height;
    expect(after).toBeGreaterThan(before * 1.8);
    await dbl(page, tl, 150, 0.8, { h: 340 });
    const k = (await tracksOf(page)).swap_opacity[0];
    expect(Math.abs(k.frame - 150)).toBeLessThanOrEqual(1);
    expect(Math.abs(k.value - 0.8)).toBeLessThan(0.05);
  });
});
