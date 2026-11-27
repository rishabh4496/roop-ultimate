import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { openTab, makePng } from './helpers.js';

// Stage 11, in a real browser against the mock backend:
//   * the Pen tool (P): click-to-place, click-the-first-point-to-close, press-drag for a Bezier handle,
//     exclude / include, feather, frame range, and the 2-point landmark pin - and that what is drawn LEAVES
//     the browser as `roto_polygons` in the preview request, in frame pixels;
//   * Expression & Gaze Fidelity: the three controls and the identity lock reach the request;
//   * the identity confidence strip: amber cells, 1-click seek, prev / next amber, and the user's match
//     threshold travelling to the backend.
// The geometry itself is pinned to the Python module by .render-check/render-stage11-logic.mjs; the
// backend half by app/tests/test_roto_polygons.py / test_stage11_backend.py. These say nothing about the
// real pipeline: the mock answers with a flat picture.

// A realistic desktop. At the default 1280 px the three-column layout leaves the stage 422 px wide and the
// app's own HUD rows (not this feature's) cover most of the picture - the Mask Brush has the same limit.
test.use({ viewport: { width: 1920, height: 1080 } });

const FRAME = { w: 640, h: 360 };
const FACE = [250, 90, 390, 270];                  // x0, y0, x1, y1 in frame pixels
const ANCHORS = [[320, 165], [320, 250]];          // nose, chin

// The mock keeps what POST /api/settings writes for the life of its process, and these specs change Stage 11
// settings through the real UI. Reset them around every test, or the next test (or the next spec file)
// starts from a state it did not choose - one stray `identity_lock: true` flipped a toggle test's click
// into an un-click.
const STAGE11_DEFAULTS = {
  pupil_gaze_retention: 0, micro_expression_restore: 0, inner_mouth_preserve: false,
  identity_lock: false, identity_lock_margin: 0.08, identity_lock_hold: 3,
};
test.beforeEach(({ request }) => request.post('/api/settings', { data: STAGE11_DEFAULTS }));
test.afterEach(({ request }) => request.post('/api/settings', { data: STAGE11_DEFAULTS }));

// Every /api/preview body the page sent, newest last. Captured in the route handler: page.route answers the
// request locally, so the mock server never sees it.
const sent = new WeakMap();

async function openStage(page, { anchors = ANCHORS } = {}) {
  const png = makePng(FRAME.w, FRAME.h);
  sent.set(page, []);
  await page.route('**/api/target/preview*', (route) => route.fulfill({ status: 200, contentType: 'image/png', body: png }));
  await page.route('**/api/preview', (route) => {
    if (route.request().method() !== 'POST') return route.fallback();
    try { sent.get(page).push(JSON.parse(route.request().postData() || '{}')); } catch { /* not JSON */ }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image: `data:image/png;base64,${png.toString('base64')}`,
        faces: [FACE], person_ids: [0],
        kps: [[[290, 140], [350, 140], [320, 165], [296, 215], [344, 215]]],
        anchors: [anchors],
        pose: [],
      }),
    });
  });
  await openTab(page, 'faceswap');
  const stage = page.locator('.preview-stage').first();
  await expect(stage).toBeVisible({ timeout: 15_000 });
  await stage.hover();
  return stage;
}

const lastPreview = async (page) => sent.get(page).at(-1) ?? null;

/** The Pen layer's screen box and its frame size (the viewBox), independent of the app's own maths. */
async function penGeometry(page) {
  const layer = page.getByTestId('roto-layer');
  const box = await layer.boundingBox();
  const vb = (await layer.getAttribute('viewBox')).split(/\s+/).map(Number);
  return { layer, box, w: vb[2], h: vb[3], at: (fx, fy) => ({ x: box.x + box.width * fx, y: box.y + box.height * fy }) };
}

async function click(page, p) {
  await page.mouse.move(p.x, p.y);
  await page.mouse.down();
  await page.mouse.up();
}

/** Draw a closed four-point zone around the middle of the picture. */
async function drawSquare(page, g, { fx0 = 0.2, fy0 = 0.45, fx1 = 0.4, fy1 = 0.75 } = {}) {
  const pts = [g.at(fx0, fy0), g.at(fx1, fy0), g.at(fx1, fy1), g.at(fx0, fy1)];
  for (const p of pts) await click(page, p);
  await click(page, pts[0]);                                    // back onto the first point closes it
  return pts;
}

async function openPen(page, stage) {
  await stage.hover();
  await page.keyboard.press('p');
  await expect(page.getByTestId('roto-toolbar')).toBeVisible();
  return penGeometry(page);
}

test('pen: P opens the tool, a zone is drawn, closed on its first point, and sent in frame pixels', async ({ page }) => {
  const stage = await openStage(page);
  const g = await openPen(page, stage);
  const pts = await drawSquare(page, g);

  await expect(page.locator('[data-zone-id][data-closed="true"]')).toHaveCount(1);
  await expect(page.getByTestId('roto-toggle')).toContainText('(1)');

  await expect.poll(async () => (await lastPreview(page))?.roto_polygons?.length, { timeout: 15_000 }).toBe(1);
  const z = (await lastPreview(page)).roto_polygons[0];
  expect(z.mode).toBe('exclude');
  expect(z.points).toHaveLength(4);
  // Each vertex landed where the pointer was, converted to FRAME pixels (not screen pixels).
  pts.forEach((p, i) => {
    const wantX = ((p.x - g.box.x) / g.box.width) * g.w;
    const wantY = ((p.y - g.box.y) / g.box.height) * g.h;
    expect(Math.abs(z.points[i].x - wantX), `x of point ${i}`).toBeLessThan(1.5);
    expect(Math.abs(z.points[i].y - wantY), `y of point ${i}`).toBeLessThan(1.5);
  });
  expect(z.pin).toBeNull();
  expect(z.frames).toBeNull();
});

test('pen: an open zone is not sent; Escape cancels it', async ({ page }) => {
  const stage = await openStage(page);
  const g = await openPen(page, stage);
  for (const p of [g.at(0.2, 0.5), g.at(0.4, 0.5), g.at(0.4, 0.7)]) await click(page, p);
  await expect(page.locator('[data-zone-id][data-closed="false"]')).toHaveCount(1);
  await page.waitForTimeout(800);
  expect((await lastPreview(page))?.roto_polygons ?? []).toHaveLength(0);
  await page.keyboard.press('Escape');
  await expect(page.locator('[data-zone-id]')).toHaveCount(0);
});

test('pen: Enter closes an open zone with three points', async ({ page }) => {
  const stage = await openStage(page);
  const g = await openPen(page, stage);
  for (const p of [g.at(0.2, 0.5), g.at(0.45, 0.5), g.at(0.3, 0.8)]) await click(page, p);
  await page.keyboard.press('Enter');
  await expect(page.locator('[data-zone-id][data-closed="true"]')).toHaveCount(1);
  await expect.poll(async () => (await lastPreview(page))?.roto_polygons?.length).toBe(1);
});

test('pen: press-and-drag after a click pulls out a Bezier handle', async ({ page }) => {
  const stage = await openStage(page);
  const g = await openPen(page, stage);
  const a = g.at(0.15, 0.5); const b = g.at(0.4, 0.5); const c = g.at(0.25, 0.8);
  await click(page, a);
  await page.mouse.move(b.x, b.y);
  await page.mouse.down();
  await page.mouse.move(b.x + 40, b.y + 30, { steps: 6 });         // drag out a handle
  await page.mouse.up();
  await click(page, c);
  await page.keyboard.press('Enter');
  await expect.poll(async () => (await lastPreview(page))?.roto_polygons?.length).toBe(1);
  const pts = (await lastPreview(page)).roto_polygons[0].points;
  expect(pts[0].hout).toBeNull();
  expect(pts[1].hout, 'the dragged point has a handle').not.toBeNull();
  expect(pts[1].hin[0]).toBeCloseTo(-pts[1].hout[0], 1);           // smooth: the other handle mirrors it
  expect(Math.hypot(...pts[1].hout)).toBeGreaterThan(20);
});

test('pen: dragging a vertex moves it and commits once', async ({ page }) => {
  const stage = await openStage(page);
  const g = await openPen(page, stage);
  const pts = await drawSquare(page, g);
  await expect.poll(async () => (await lastPreview(page))?.roto_polygons?.length).toBe(1);
  const before = (await lastPreview(page)).roto_polygons[0].points[2];
  await page.mouse.move(pts[2].x, pts[2].y);
  await page.mouse.down();
  await page.mouse.move(pts[2].x + 50, pts[2].y + 30, { steps: 5 });
  await page.mouse.up();
  await expect.poll(async () => (await lastPreview(page)).roto_polygons[0].points[2].x).not.toBe(before.x);
  const after = (await lastPreview(page)).roto_polygons[0].points[2];
  expect(after.x - before.x).toBeCloseTo((50 / g.box.width) * g.w, 0);
  expect(after.y - before.y).toBeCloseTo((30 / g.box.height) * g.h, 0);
});

test('pen: include mode, feather and a frame range reach the request', async ({ page }) => {
  const stage = await openStage(page);
  const g = await openPen(page, stage);
  await page.getByTestId('roto-mode-include').click();
  await drawSquare(page, g);
  await expect(page.locator('[data-zone-id][data-mode="include"]')).toHaveCount(1);
  await page.getByTestId('roto-feather').fill('40');
  await page.getByTestId('roto-range').selectOption('from');
  await expect.poll(async () => (await lastPreview(page))?.roto_polygons?.[0]?.feather).toBe(40);
  const z = (await lastPreview(page)).roto_polygons[0];
  expect(z.mode).toBe('include');
  expect(z.frames[0]).toBeGreaterThanOrEqual(1);
  expect(z.frames[1]).toBeGreaterThanOrEqual(z.frames[0]);
  await expect(page.getByTestId('roto-feather-band')).toHaveCount(1);
});

test('pen: pinning follows the face - the reference anchors are sent, and the zone moves when the face does', async ({ page }) => {
  const stage = await openStage(page);
  const g = await openPen(page, stage);
  await drawSquare(page, g, { fx0: 0.55, fy0: 0.45, fx1: 0.68, fy1: 0.7 });
  await expect(page.getByTestId('roto-pin')).toBeEnabled({ timeout: 10_000 });
  await page.getByTestId('roto-pin').click();
  await expect.poll(async () => (await lastPreview(page))?.roto_polygons?.[0]?.pin).toBeTruthy();
  const pin = (await lastPreview(page)).roto_polygons[0].pin;
  expect(pin.ref_nose).toEqual(ANCHORS[0]);
  expect(pin.ref_chin).toEqual(ANCHORS[1]);
  expect(pin.enabled).toBe(true);
  await expect(page.locator('[data-zone-id]').first()).toHaveAttribute('data-tracked', 'true');

  // The face moves 60 frame-px right: the next preview reports shifted anchors and the zone follows.
  const outline = () => page.locator('[data-zone-id] path').first().getAttribute('d');
  const before = await outline();
  const png = makePng(FRAME.w, FRAME.h);
  await page.unroute('**/api/preview');
  await page.route('**/api/preview', (route) => {
    if (route.request().method() !== 'POST') return route.fallback();
    try { sent.get(page).push(JSON.parse(route.request().postData() || '{}')); } catch { /* not JSON */ }
    return route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({
        image: `data:image/png;base64,${png.toString('base64')}`,
        faces: [[310, 90, 450, 270]], person_ids: [0],
        kps: [[[350, 140], [410, 140], [380, 165], [356, 215], [404, 215]]],
        anchors: [[[380, 165], [380, 250]]], pose: [],
      }),
    });
  });
  await page.getByTestId('roto-feather').fill('41');                  // any edit re-requests the preview
  await expect.poll(outline, { timeout: 15_000 }).not.toBe(before);
  const moved = await outline();
  const firstX = (d) => Number(/^M([\d.-]+)/.exec(d)[1]);
  expect(firstX(moved) - firstX(before)).toBeCloseTo(60, 0);
});

test('pen: Delete removes the selected zone, and the next request carries none', async ({ page }) => {
  const stage = await openStage(page);
  const g = await openPen(page, stage);
  await drawSquare(page, g);
  await expect.poll(async () => (await lastPreview(page))?.roto_polygons?.length).toBe(1);
  await page.keyboard.press('Delete');
  await expect(page.locator('[data-zone-id]')).toHaveCount(0);
  // Deleting returns the request to a signature the preview cache already holds, so the UI is served from
  // cache and sends NOTHING (by design: the signature is the request body). Whatever is sent next must not
  // still carry the zone, so make the next request happen by changing an unrelated control.
  await page.getByRole('button', { name: /Expression & Gaze Fidelity/ }).click();
  await page.locator('[data-setting="pupil_gaze_retention"] input[type="range"]').fill('0.3');
  await expect.poll(async () => (await lastPreview(page))?.pupil_gaze_retention).toBe(0.3);
  expect((await lastPreview(page)).roto_polygons).toEqual([]);
});

test('pen: P toggles the tool off and the brush and the pen are mutually exclusive', async ({ page }) => {
  const stage = await openStage(page);
  await stage.hover();
  await page.keyboard.press('p');
  await expect(page.getByTestId('roto-toggle')).toHaveAttribute('aria-pressed', 'true');
  await page.keyboard.press('b');
  await expect(page.getByTestId('roto-toggle')).toHaveAttribute('aria-pressed', 'false');
  await expect(page.getByTestId('roto-toolbar')).toHaveCount(0);
  await page.keyboard.press('p');
  await expect(page.getByTestId('roto-toggle')).toHaveAttribute('aria-pressed', 'true');
  await page.keyboard.press('p');
  await expect(page.getByTestId('roto-toolbar')).toHaveCount(0);
});

// ── Expression & Gaze Fidelity ──────────────────────────────────────────────

test('fidelity: the three controls and the identity lock leave the browser', async ({ page }) => {
  await openStage(page);
  await page.getByRole('button', { name: /Expression & Gaze Fidelity/ }).click();
  const slider = (key) => page.locator(`[data-setting="${key}"] input[type="range"]`);
  await slider('pupil_gaze_retention').fill('0.6');
  await slider('micro_expression_restore').fill('0.4');
  // Click the label TEXT: the middle of a Toggle row can be its info badge, which opens a tooltip instead.
  await page.locator('[data-setting="inner_mouth_preserve"]').getByText('Inner-mouth & teeth preservation', { exact: true }).click();
  await page.locator('[data-setting="identity_lock"]').getByText('Identity crossover lock', { exact: true }).click();
  await slider('identity_lock_hold').fill('6');
  await expect.poll(async () => (await lastPreview(page))?.identity_lock_hold).toBe(6);
  const body = await lastPreview(page);
  expect(body.pupil_gaze_retention).toBe(0.6);
  expect(body.micro_expression_restore).toBe(0.4);
  expect(body.inner_mouth_preserve).toBe(true);
  expect(body.identity_lock).toBe(true);
  expect(body.identity_lock_margin).toBeCloseTo(0.08, 5);
});

test('fidelity: every control defaults to a no-op and the lock sliders stay hidden until it is on', async ({ page }) => {
  await openStage(page);
  await page.getByRole('button', { name: /Expression & Gaze Fidelity/ }).click();
  await expect.poll(async () => (await lastPreview(page))?.pupil_gaze_retention).toBe(0);
  const body = await lastPreview(page);
  expect([body.micro_expression_restore, body.inner_mouth_preserve, body.identity_lock]).toEqual([0, false, false]);
  expect(body.roto_polygons).toEqual([]);
  expect(body.identity_pins).toEqual([]);
  await expect(page.locator('[data-setting="identity_lock_hold"]')).toHaveCount(0);
});

// ── Identity confidence strip ───────────────────────────────────────────────

async function openTimeline(page) {
  await openStage(page);
  const lane = page.getByTestId('lane-identity');
  await lane.scrollIntoViewIfNeeded();
  return lane;
}

/** The frame the stage's HUD reads (the "n/total" counter), independent of the app's state. */
const hudFrame = (stage) => stage.evaluate((el) => {
  const m = [...el.querySelectorAll('span')].map((n) => n.textContent.replace(/\s+/g, '')).find((t) => /^[\d,]+\/[\d,]+$/.test(t));
  return m ? Number(m.split('/')[0].replace(/,/g, '')) : null;
});

const amberFrames = (page) => page.evaluate(async () => {
  const j = await (await fetch('/api/target/timeline_tracks?index=0&threshold=0.75')).json();
  return [...new Set(j.identity_confidence.flatMap((l) => l.amber))].sort((x, y) => x - y);
});

test('identity strip: amber samples are counted and Next / Prev amber walk them in order, wrapping', async ({ page }) => {
  const lane = await openTimeline(page);
  await expect(lane).toBeVisible();
  const amber = await amberFrames(page);
  expect(amber.length).toBeGreaterThanOrEqual(2);
  await expect(page.getByTestId('identity-amber-count')).toHaveText(`${amber.length} amber`);

  const stage = page.locator('.preview-stage').first();
  await stage.hover();
  expect(await hudFrame(stage)).toBe(1);
  // The rule, stated independently of the app: "next" is the first amber frame AFTER the playhead, else the
  // first of all (wrapping); "previous" is the last one BEFORE it, else the last of all. The playhead starts
  // on frame 1, which is itself amber, so the first "next" must skip past it.
  const nextOf = (f) => amber.find((a) => a > f) ?? amber[0];
  const prevOf = (f) => [...amber].reverse().find((a) => a < f) ?? amber.at(-1);
  let at = 1;
  const visited = [];
  for (let i = 0; i < amber.length + 1; i++) {                        // a full lap and the wrap
    at = nextOf(at);
    await page.getByTestId('identity-next').click();
    await expect.poll(() => hudFrame(stage)).toBe(at);
    visited.push(at);
  }
  expect([...new Set(visited)].sort((x, y) => x - y)).toEqual(amber); // every amber frame was reached
  expect(visited.at(-1)).toBe(visited[0]);                            // and the lap closed on itself (wrapped)
  for (let i = 0; i < 2; i++) {
    at = prevOf(at);
    await page.getByTestId('identity-prev').click();
    await expect.poll(() => hudFrame(stage)).toBe(at);
  }
});

test('identity strip: the user\'s match threshold is what the backend is asked about', async ({ page }) => {
  const asked = [];
  page.on('request', (r) => { if (r.url().includes('/api/target/timeline_tracks')) asked.push(r.url()); });
  await openTimeline(page);
  await expect.poll(() => asked.some((u) => /threshold=0\.75/.test(u))).toBe(true);
});

test('identity strip: clicking an amber cell seeks to that frame', async ({ page }) => {
  const lane = await openTimeline(page);
  const stage = page.locator('.preview-stage').first();
  const amber = await amberFrames(page);
  const target = amber[1];
  const box = await lane.boundingBox();
  const view = await page.getByTestId('timeline').evaluate((el) => ({ s: Number(el.dataset.viewStart), e: Number(el.dataset.viewEnd) }));
  // the centre of that sample's cell, in the lane's own coordinates
  const x = box.x + ((target - view.s + 0.5) / (view.e - view.s + 1)) * box.width;
  await page.mouse.click(x, box.y + 4);
  await stage.hover();
  await expect.poll(() => hudFrame(stage)).toBe(target);
});

// ── accessibility of the new surfaces ───────────────────────────────────────

test('a11y: the Pen toolbar, the fidelity group and the strip have no axe violations', async ({ page }) => {
  const stage = await openStage(page);
  await openPen(page, stage);
  await drawSquare(page, await penGeometry(page));
  await page.getByRole('button', { name: /Expression & Gaze Fidelity/ }).click();
  const results = await new AxeBuilder({ page })
    .include('[data-testid="roto-toolbar"]')
    .include('[data-testid="lane-identity"]')
    .analyze();
  const bad = results.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(' | ')}`);
  expect(bad).toEqual([]);
});

// ── what a RENDER sends, and per-clip isolation ─────────────────────────────

test('render: the zones, pins and controls are in the /api/swap payload, not only the preview', async ({ page }) => {
  const swaps = [];
  await page.route('**/api/swap', (route) => {
    if (route.request().method() !== 'POST') return route.fallback();
    try { swaps.push(JSON.parse(route.request().postData() || '{}')); } catch { /* not JSON */ }
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'started' }) });
  });
  const stage = await openStage(page);
  const g = await openPen(page, stage);
  await page.getByTestId('roto-mode-include').click();
  await drawSquare(page, g);
  await page.getByTestId('roto-done').click();
  await page.getByRole('button', { name: /Expression & Gaze Fidelity/ }).click();
  await page.locator('[data-setting="micro_expression_restore"] input[type="range"]').fill('0.5');
  await page.getByRole('button', { name: /Start Swapping/ }).click();
  await expect.poll(() => swaps.length, { timeout: 15_000 }).toBeGreaterThan(0);
  const body = swaps.at(-1);
  expect(body.roto_polygons).toHaveLength(1);
  expect(body.roto_polygons[0].mode).toBe('include');
  expect(body.roto_polygons[0].points).toHaveLength(4);
  expect(body.micro_expression_restore).toBe(0.5);
  expect(body.pupil_gaze_retention).toBe(0);
  expect(body).toHaveProperty('identity_pins');
});

test('pins: "Pin Identity to Track #N" locks this person on this frame, and the pin travels in the request', async ({ page }) => {
  const stage = await openStage(page);
  const pin = page.getByTestId('pin-identity').first();
  await expect(pin).toBeVisible({ timeout: 15_000 });
  await expect(pin).toHaveText('Pin Identity to Track #1');
  await pin.click();
  await expect(page.getByTestId('pin-identity').first()).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: /Expression & Gaze Fidelity/ }).click();
  await expect(page.getByTestId('pin-list')).toContainText('Track #1');
  await expect.poll(async () => (await lastPreview(page))?.identity_pins?.length).toBe(1);
  const p = (await lastPreview(page)).identity_pins[0];
  expect(p.frame).toBe(1);
  expect(p.source_index).toBe(0);
  expect(p.person_id).toBe('tp_0');
  expect(p.bbox).toEqual(FACE);                                   // the box the user clicked, in frame pixels
  await stage.hover();
  await page.getByTestId('pin-identity').first().click();          // clicking a pinned face unpins it
  await expect(page.getByTestId('pin-list')).toHaveCount(0);
});

test('per clip: a zone drawn on one target is not on the next, and is still there when you come back', async ({ page }) => {
  const stage = await openStage(page);
  const g = await openPen(page, stage);
  await drawSquare(page, g);
  await expect(page.locator('[data-zone-id][data-closed="true"]')).toHaveCount(1);
  await page.getByTestId('roto-done').click();
  await page.getByText('commercial_portrait.png', { exact: true }).first().click();
  await expect(page.getByTestId('roto-toggle')).not.toContainText('(1)');      // the still has no zones
  await expect(page.locator('[data-zone-id]')).toHaveCount(0);
  await page.getByText('interview_scene_1080p.mp4', { exact: true }).first().click();
  await expect(page.getByTestId('roto-toggle')).toContainText('(1)');           // the video's zone is back
  await expect(page.locator('[data-zone-id][data-closed="true"]')).toHaveCount(1);
});
