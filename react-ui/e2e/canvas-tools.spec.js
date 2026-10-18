import { test, expect } from '@playwright/test';
import { openTab, makePng } from './helpers.js';

// The Media Canvas's pointer tools, checked on real pixels in a real browser:
//   * the Mask Brush paints under the pointer (sub-pixel, at 100% and at the stage's 2.5x zoom),
//     at a radius that is the on-screen brush diameter, and Erase removes it;
//   * the 3.5x Lens shows the pixel under the pointer, magnified 3.5x on top of the stage zoom.
// The mapping's arithmetic is pinned by .render-check/brush-mapping-check.mjs; this is the DOM
// contract: that the canvas the user paints on is the one the maths describes.

// A frame with real dimensions: the mock's placeholders are SVGs, which never decode into a canvas
// frame, so without this the stage never learns its size and mounts no mask canvas at all.
const FRAME = { w: 640, h: 360 };

async function openStage(page) {
  const png = makePng(FRAME.w, FRAME.h);
  await page.route('**/api/target/preview*', (route) => route.fulfill({ status: 200, contentType: 'image/png', body: png }));
  await openTab(page, 'faceswap');
  const stage = page.locator('.preview-stage').first();
  await expect(stage).toBeVisible({ timeout: 15_000 });
  await stage.hover();                                   // the HUD is `group-hover`
  return stage;
}

const brushCanvas = (stage) => stage.locator('canvas.cursor-crosshair');

/** Where an object-contain picture of w x h sits in the canvas's box (independent of the app code). */
const contained = (box, w, h) => {
  const s = Math.min(box.width / w, box.height / h);
  const cw = w * s; const ch = h * s;
  return { left: box.x + (box.width - cw) / 2, top: box.y + (box.height - ch) / 2, width: cw, height: ch };
};

const alphaAt = (canvas, x, y) => canvas.evaluate((c, [px, py]) => {
  const d = c.getContext('2d').getImageData(Math.round(px), Math.round(py), 1, 1).data;
  return d[3];
}, [x, y]);

async function paint(page, stage, mode, fx, fy, sizePx = 30, at = null) {
  await stage.getByRole('button', { name: 'Mask Brush' }).click();
  const canvas = brushCanvas(stage);
  await expect(canvas).toBeVisible();
  const dims = await canvas.evaluate((c) => ({ w: c.width, h: c.height }));
  const box = await canvas.boundingBox();
  const pic = contained(box, dims.w, dims.h);
  // `at`: an absolute client point (when the picture is zoomed past the stage, a fraction of it may be off screen).
  const px = at ? at.x : pic.left + pic.width * fx;
  const py = at ? at.y : pic.top + pic.height * fy;
  if (mode === 'erase') await stage.getByRole('button', { name: /erase/i }).first().click();
  await page.mouse.move(px, py);
  await page.mouse.down();
  await page.mouse.up();
  const scale = dims.w / pic.width;
  return { canvas, dims, pic, scale, px, py, cx: (px - pic.left) * scale, cy: (py - pic.top) * scale, rCanvas: (sizePx / 2) * scale };
}

test('mask brush: paints at the pointer (sub-pixel, 100%), with the on-screen radius', async ({ page }) => {
  const stage = await openStage(page);
  const s = await paint(page, stage, 'paint', 0.3125, 0.4375);       // fractional client coordinates
  expect(await alphaAt(s.canvas, s.cx, s.cy), 'painted under the pointer').toBeGreaterThan(0);
  expect(await alphaAt(s.canvas, s.cx + s.rCanvas * 0.8, s.cy), 'inside the radius').toBeGreaterThan(0);
  expect(await alphaAt(s.canvas, s.cx, s.cy - s.rCanvas * 0.8), 'inside the radius (above)').toBeGreaterThan(0);
  expect(await alphaAt(s.canvas, s.cx + s.rCanvas * 1.4, s.cy), 'outside the radius').toBe(0);
  expect(await alphaAt(s.canvas, s.cx, s.cy + s.rCanvas * 1.4), 'outside the radius (below)').toBe(0);
  expect(await alphaAt(s.canvas, s.dims.w - 3, s.dims.h - 3), 'far away is untouched').toBe(0);
});

test('mask brush: still lands under the pointer after the stage is zoomed 2.5x', async ({ page }) => {
  const stage = await openStage(page);
  const sb = await stage.boundingBox();
  const at = { x: sb.x + sb.width * 0.62 + 0.37, y: sb.y + sb.height * 0.58 + 0.21 };   // fractional, clear of the HUD
  const unzoomed = await stage.locator('canvas').first().boundingBox();
  await page.mouse.dblclick(at.x, at.y);                              // 2.5x zoom (handleDoubleClick)
  await page.waitForTimeout(600);
  const zoomed = await stage.locator('canvas').first().boundingBox();
  expect(zoomed.width, 'the stage is zoomed ~2.5x').toBeGreaterThan(unzoomed.width * 2);
  const s = await paint(page, stage, 'paint', 0, 0, 30, at);
  expect(await alphaAt(s.canvas, s.cx, s.cy), 'painted under the pointer when zoomed').toBeGreaterThan(0);
  expect(await alphaAt(s.canvas, s.cx + s.rCanvas * 1.4, s.cy), 'a zoomed 30 px brush is still 30 SCREEN px').toBe(0);
});

test('mask brush: erase removes what paint put down', async ({ page }) => {
  const stage = await openStage(page);
  const s = await paint(page, stage, 'paint', 0.5, 0.5);
  expect(await alphaAt(s.canvas, s.cx, s.cy)).toBeGreaterThan(0);
  await stage.getByRole('button', { name: /erase/i }).first().click();
  await page.mouse.move(s.px, s.py);
  await page.mouse.down();
  await page.mouse.up();
  expect(await alphaAt(s.canvas, s.cx, s.cy), 'erased').toBe(0);
});

// ── Compare modes, on pixels ────────────────────────────────────────────────
// The raw frame is solid RED, the swapped frame solid BLUE, so every mode's output is a number we
// can compute independently: wipe = a hard edge at the divider, alpha blend = red + p*(blue-red),
// diff = |red - blue|.
const RED = [200, 40, 40];
const BLUE = [40, 40, 200];

async function openCompare(page) {
  const red = makePng(FRAME.w, FRAME.h, () => RED);
  const blue = makePng(FRAME.w, FRAME.h, () => BLUE);
  await page.route('**/api/target/preview*', (route) => route.fulfill({ status: 200, contentType: 'image/png', body: red }));
  await page.route('**/api/preview', (route) => (route.request().method() === 'POST'
    ? route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image: `data:image/png;base64,${blue.toString('base64')}`,
        faces: [], person_ids: [], kps: [], pose: [],
      }),
    })
    : route.fallback()));
  await openTab(page, 'faceswap');
  const stage = page.locator('.preview-stage').first();
  await expect(stage).toBeVisible({ timeout: 15_000 });
  await stage.hover();
  await stage.getByRole('button', { name: /Compare Faces/ }).first().click();
  return stage;
}

/** RGB of the stage's picture canvas at a fraction of its width/height. */
const pixel = (stage, fx, fy) => stage.locator('canvas').first().evaluate((c, [x, y]) => {
  const d = c.getContext('2d').getImageData(Math.floor(c.width * x), Math.floor(c.height * y), 1, 1).data;
  return [d[0], d[1], d[2]];
}, [fx, fy]);

const close = (got, want, tol = 3) => got.every((v, i) => Math.abs(v - want[i]) <= tol);

test('compare: the alpha blend at 25% / 50% / 75% is red + p * (blue - red)', async ({ page }) => {
  const stage = await openCompare(page);
  await stage.getByRole('button', { name: 'Blend', exact: true }).first().click();
  for (const pct of [25, 50, 75]) {
    await stage.getByRole('button', { name: `Move the compare split to ${pct}%` }).click();
    await page.waitForTimeout(400);
    const want = RED.map((r, i) => Math.round(r + (pct / 100) * (BLUE[i] - r)));
    expect(close(await pixel(stage, 0.5, 0.5), want), `blend ${pct}%: want ${want}, got ${await pixel(stage, 0.5, 0.5)}`).toBe(true);
  }
});

test('compare: the difference map is |raw - swap| per channel', async ({ page }) => {
  const stage = await openCompare(page);
  await stage.getByRole('button', { name: 'Diff', exact: true }).first().click();
  await page.waitForTimeout(500);
  const want = RED.map((r, i) => Math.abs(r - BLUE[i]));
  expect(close(await pixel(stage, 0.5, 0.5), want), `diff: want ${want}, got ${await pixel(stage, 0.5, 0.5)}`).toBe(true);
});

test('compare: the wipe shows the swap on one side of the divider and the raw frame on the other', async ({ page }) => {
  const stage = await openCompare(page);
  await stage.getByRole('button', { name: 'Move the compare split to 50%' }).click();
  await page.waitForTimeout(500);
  const a = await pixel(stage, 0.1, 0.5);
  const b = await pixel(stage, 0.9, 0.5);
  const isRed = (p) => close(p, RED); const isBlue = (p) => close(p, BLUE);
  expect((isRed(a) && isBlue(b)) || (isBlue(a) && isRed(b)), `wipe halves differ: ${a} | ${b}`).toBe(true);
});

// ── The 3.5x Lens ───────────────────────────────────────────────────────────
test('lens: shows the picture at 3.5x, centred on the pixel under the pointer (100% and 2.5x stage zoom)', async ({ page }) => {
  const stage = await openStage(page);
  const lensBtn = stage.getByRole('button', { name: 'Lens', exact: true });
  await lensBtn.click();

  const check = async (label) => {
    const pic = await stage.locator('canvas').first().boundingBox();
    const sb = await stage.boundingBox();
    // A point on the picture, fractional, away from the HUD.
    const at = { x: Math.min(pic.x + pic.width * 0.42, sb.x + sb.width * 0.7) + 0.31, y: Math.max(pic.y + pic.height * 0.55, sb.y + sb.height * 0.45) + 0.17 };
    await page.mouse.move(at.x - 20, at.y - 20);
    await page.mouse.move(at.x, at.y, { steps: 3 });
    const img = stage.getByAltText('Magnifier lens view');
    await expect(img, `${label}: the lens is up`).toBeVisible();
    const ib = await img.boundingBox();
    const lens = await img.locator('xpath=..').boundingBox();
    expect(ib.width / pic.width, `${label}: lens image width / picture width`).toBeGreaterThan(3.45);
    expect(ib.width / pic.width, `${label}: lens image width / picture width`).toBeLessThan(3.55);
    expect(ib.height / pic.height, `${label}: and the same for height`).toBeGreaterThan(3.45);
    expect(ib.height / pic.height, `${label}: and the same for height`).toBeLessThan(3.55);
    // The glass centre must sample the same FRACTION of the picture the pointer is over.
    const fracPointer = { x: (at.x - pic.x) / pic.width, y: (at.y - pic.y) / pic.height };
    const fracLens = { x: (lens.x + lens.width / 2 - ib.x) / ib.width, y: (lens.y + lens.height / 2 - ib.y) / ib.height };
    expect(Math.abs(fracPointer.x - fracLens.x), `${label}: horizontal sample point`).toBeLessThan(0.004);
    expect(Math.abs(fracPointer.y - fracLens.y), `${label}: vertical sample point`).toBeLessThan(0.004);
    return ib.width;
  };

  await check('100%');
  const sb = await stage.boundingBox();
  await page.mouse.dblclick(sb.x + sb.width * 0.62, sb.y + sb.height * 0.58);   // 2.5x stage zoom
  await page.waitForTimeout(600);
  await check('2.5x');
});
