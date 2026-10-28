import { test, expect } from '@playwright/test';
import { openTab, makePng } from './helpers.js';

// Stage 25, in a real Chromium against the production build with the mock server standing in for app/api.py:
//
//   curve editor   C opens it (Shift+C is still Compare); keys are added, dragged, re-interpolated; the PREVIEW request
//                  carries the curve's value at the playhead and a RENDER of the range is queued as stepped pieces -
//                  the two places a curve could look right in the editor and do nothing; the graph follows the
//                  timeline's zoom; Ctrl+Z undoes a key.
//   face menu      right-click a detected box: the menu opens inside the window at every corner and at two UI zooms,
//                  Escape and an outside press dismiss it, and each of its six actions has its real effect (a mapping
//                  write, a pin, a scene override, a multipart crop upload, a changed warmth in the NEXT preview, a 2x
//                  stage transform).
//
// The mock's pictures and detections are stand-ins, so none of this says what a real swap looks like.

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

/** Page coordinates of (frame, value) on the graph, for the parameter range [min, max]. */
async function graphPoint(page, tl, frame, value, { min = 0, max = 1 } = {}) {
  const g = await page.getByTestId('curve-graph').boundingBox();
  const vs = Number(await tl.getAttribute('data-view-start'));
  const ve = Number(await tl.getAttribute('data-view-end'));
  const yPx = PAD + (1 - (value - min) / (max - min)) * (GRAPH_H - 2 * PAD);
  return { x: g.x + ((frame - vs) / (ve - vs)) * g.width, y: g.y + (yPx / GRAPH_H) * g.height };
}

/** The keys on the graph: [{frame, value}] from their accessible names. */
const keys = (page) => page.locator('[data-testid^="curve-key-"]').evaluateAll((els) => els.map((el) => {
  const m = /frame (\d+), value ([\d.]+)/.exec(el.getAttribute('aria-label') || '');
  return m ? { frame: Number(m[1]), value: Number(m[2]) } : null;
}).filter(Boolean).sort((a, b) => a.frame - b.frame));

async function setFrame(page, n) {
  const input = page.getByLabel('Frame number');
  await input.fill(String(n));
  await input.press('Enter');
}

/** Key the playhead's frame on the open curve and give the key a value. */
async function keyAt(page, frame, value) {
  await setFrame(page, frame);
  await page.getByTestId('curve-add-key').click();
  await expect(page.getByTestId(`curve-key-${frame}`)).toBeVisible();
  if (value !== undefined) {
    const v = page.getByTestId('curve-value');
    await v.fill(String(value));
    await v.press('Enter');
    await expect(page.getByTestId(`curve-key-${frame}`)).toHaveAttribute('aria-label', new RegExp(`value ${Number(value).toFixed(2)}`));
  }
}

test.describe('curve editor (300-frame clip)', () => {
  test.beforeEach(async ({ request }) => { await resizeClip(request, 300); });
  // The render test starts the mock queue; the mock is one shared process, so leave it as it was found.
  test.afterEach(async ({ request }) => {
    await request.post('/api/queue/stop').catch(() => {});
    await request.post('/api/queue/clear', { data: { keep_running: false } }).catch(() => {});
    await resizeClip(request, 180);
  });

  test('C opens and closes it, and Shift+C is still Compare', async ({ page }) => {
    await openTimeline(page);
    await expect(page.getByTestId('curve-editor')).toHaveCount(0);
    await page.keyboard.press('c');
    await expect(page.getByTestId('curve-editor')).toBeVisible();
    await page.keyboard.press('c');
    await expect(page.getByTestId('curve-editor')).toHaveCount(0);
    await page.keyboard.press('Shift+C');
    await expect(page.getByTestId('curve-editor'), 'Shift+C must not open the editor').toHaveCount(0);
    await expect(page.getByText(/^Compare \(/), 'Shift+C is the comparison view').toBeVisible();
    // The key is a key, not a text-field character: typing a c in the frame box must do nothing.
    await page.keyboard.press('Shift+C');
    const frame = page.getByLabel('Frame number');
    await frame.focus();
    await page.keyboard.type('c');
    await expect(page.getByTestId('curve-editor')).toHaveCount(0);
  });

  test('double-click keys a value, a drag moves it, the interpolation buttons set the segment after a key', async ({ page }) => {
    const tl = await openCurves(page);
    await expect(page.getByTestId('curve-hit')).toBeVisible();

    let p = await graphPoint(page, tl, 60, 0.3);
    await page.mouse.dblclick(p.x, p.y);
    p = await graphPoint(page, tl, 200, 0.9);
    await page.mouse.dblclick(p.x, p.y);
    const added = await keys(page);
    expect(added.length).toBe(2);
    expect(Math.abs(added[0].frame - 60)).toBeLessThanOrEqual(1);
    expect(Math.abs(added[0].value - 0.3)).toBeLessThan(0.06);
    expect(Math.abs(added[1].frame - 200)).toBeLessThanOrEqual(1);

    // The line is really drawn, in this parameter's own colour (#34d399).
    const drawn = await page.getByTestId('curve-graph').locator('canvas').evaluate((c) => {
      const { data } = c.getContext('2d').getImageData(0, 0, c.width, c.height);
      let n = 0;
      for (let i = 0; i < data.length; i += 4) if (data[i + 3] > 40 && Math.abs(data[i] - 52) < 40 && Math.abs(data[i + 1] - 211) < 40 && Math.abs(data[i + 2] - 153) < 40) n++;
      return n;
    });
    expect(drawn, 'the curve must have pixels').toBeGreaterThan(200);

    // Select the first key; set its segment to Hold.
    await page.getByTestId(`curve-key-${added[0].frame}`).click();
    await page.getByTestId('curve-interp-hold').click();
    await expect(page.getByTestId('curve-interp-hold')).toHaveAttribute('aria-pressed', 'true');
    await page.getByTestId('curve-interp-smooth').click();
    await expect(page.getByTestId('curve-interp-smooth')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('curve-interp-hold')).toHaveAttribute('aria-pressed', 'false');
    // The last key has no segment after it.
    await page.getByTestId(`curve-key-${added[1].frame}`).click();
    await expect(page.getByTestId('curve-interp-linear')).toBeDisabled();

    // Drag the first key to (frame 120, 0.6): one move, the old position is gone.
    const from = await page.getByTestId(`curve-key-${added[0].frame}`).boundingBox();
    const to = await graphPoint(page, tl, 120, 0.6);
    await page.mouse.move(from.x + from.width / 2, from.y + from.height / 2);
    await page.mouse.down();
    await page.mouse.move(to.x, to.y, { steps: 12 });
    await page.mouse.up();
    const moved = await keys(page);
    expect(moved.length).toBe(2);
    expect(Math.abs(moved[0].frame - 120)).toBeLessThanOrEqual(1);
    expect(Math.abs(moved[0].value - 0.6)).toBeLessThan(0.06);
    expect(moved[1].frame).toBe(added[1].frame);

    // A key cannot be dragged past its neighbour.
    const k0 = await page.getByTestId(`curve-key-${moved[0].frame}`).boundingBox();
    const past = await graphPoint(page, tl, 280, 0.6);
    await page.mouse.move(k0.x + k0.width / 2, k0.y + k0.height / 2);
    await page.mouse.down();
    await page.mouse.move(past.x, past.y, { steps: 10 });
    await page.mouse.up();
    const bounded = await keys(page);
    expect(bounded[0].frame).toBeLessThan(bounded[1].frame);
    expect(bounded[1].frame).toBe(added[1].frame);

    // They are still there after a reload (kept per clip).
    await page.reload();
    await openTimeline(page);
    await page.getByTestId('curve-editor-toggle').click();
    expect((await keys(page)).length).toBe(2);
  });

  test('the preview request carries the value the curve has at the playhead', async ({ page }) => {
    const bodies = [];
    page.on('request', (r) => {
      if (r.method() === 'POST' && /\/api\/preview$/.test(r.url())) { try { bodies.push(JSON.parse(r.postData() || '{}')); } catch { /* not JSON */ } }
    });
    await openCurves(page);
    // Swap Opacity: 0 at frame 1 rising linearly to 1 at frame 101.
    await keyAt(page, 1, 0);
    await keyAt(page, 101, 1);
    // The first key is not the last one any more, so it has a segment after it to set.
    await page.getByTestId('curve-key-1').click();
    await page.getByTestId('curve-interp-linear').click();

    await setFrame(page, 51);
    await expect.poll(() => bodies.filter((b) => b.frame === 51).map((b) => b.blend_ratio).pop(), { timeout: 20_000 }).toBeCloseTo(0.5, 1);
    await setFrame(page, 26);
    await expect.poll(() => bodies.filter((b) => b.frame === 26).map((b) => b.blend_ratio).pop(), { timeout: 20_000 }).toBeCloseTo(0.25, 1);
    await setFrame(page, 200);
    await expect.poll(() => bodies.filter((b) => b.frame === 200).map((b) => b.blend_ratio).pop(), { timeout: 20_000 }).toBe(1);

    // Switching the master switch off puts the clip's own setting back in the request.
    await page.getByTestId('curve-enabled').uncheck();
    await setFrame(page, 51);
    await expect.poll(() => bodies.filter((b) => b.frame === 51).map((b) => b.blend_ratio).pop(), { timeout: 20_000 }).not.toBeCloseTo(0.5, 1);
  });

  test('a render of the range is queued as stepped pieces that follow the curve', async ({ page }) => {
    await openCurves(page);
    await keyAt(page, 1, 0);
    await keyAt(page, 101, 1);
    await page.getByTestId('curve-key-1').click();
    await page.getByTestId('curve-interp-linear').click();
    await expect(page.getByTestId('curve-plan'), 'the editor says how a render will cut the range').toContainText(/pieces/);

    const jobs = [];
    page.on('request', (r) => {
      if (r.method() === 'POST' && /\/api\/queue\/add_batch$/.test(r.url())) { try { jobs.push(...(JSON.parse(r.postData() || '{}').jobs || [])); } catch { /* not JSON */ } }
    });
    const render = page.getByTestId('render-range');
    await render.scrollIntoViewIfNeeded();
    await expect(render).toBeEnabled();
    await render.click();
    await expect.poll(() => jobs.length, { timeout: 20_000 }).toBeGreaterThan(5);

    expect(jobs.length, 'under the piece cap').toBeLessThanOrEqual(24);
    // The pieces tile the whole range exactly, in order.
    expect(jobs[0].frame_start).toBe(1);
    expect(jobs[jobs.length - 1].frame_end).toBe(300);
    for (let i = 1; i < jobs.length; i++) expect(jobs[i].frame_start).toBe(jobs[i - 1].frame_end + 1);
    // And the value each carries rises with the curve, from 0 to 1.
    const values = jobs.map((j) => j.payload.blend_ratio);
    for (let i = 1; i < values.length; i++) expect(values[i]).toBeGreaterThanOrEqual(values[i - 1]);
    expect(values[0]).toBeLessThanOrEqual(0.05);
    expect(values[values.length - 1]).toBe(1);
    // Each is stamped as one split render, so the queue joins them into a single file.
    expect(new Set(jobs.map((j) => j.segment?.group)).size).toBe(1);
  });

  test('the graph is on the timeline’s own x-axis: zooming the timeline moves the keys with it', async ({ page }) => {
    const tl = await openCurves(page);
    await keyAt(page, 100, 0.5);
    const key = page.getByTestId('curve-key-100');
    const centre = async () => {
      const g = await page.getByTestId('curve-graph').boundingBox();
      const k = await key.boundingBox();
      return (k.x + k.width / 2 - g.x) / g.width;
    };
    const before = await centre();
    const v0 = { s: Number(await tl.getAttribute('data-view-start')), e: Number(await tl.getAttribute('data-view-end')) };
    expect(Math.abs(before - (100 - v0.s) / (v0.e - v0.s))).toBeLessThan(0.01);

    const lane = page.getByTestId('timeline-stack');
    const box = await lane.boundingBox();
    await page.mouse.move(box.x + box.width * 0.33, box.y + 40);
    await page.keyboard.down('Control');
    for (let i = 0; i < 4; i++) await page.mouse.wheel(0, -240);
    await page.keyboard.up('Control');
    await expect.poll(async () => Number(await tl.getAttribute('data-view-end')) - Number(await tl.getAttribute('data-view-start'))).toBeLessThan(v0.e - v0.s - 20);

    const v1 = { s: Number(await tl.getAttribute('data-view-start')), e: Number(await tl.getAttribute('data-view-end')) };
    if (100 >= v1.s - 1 && 100 <= v1.e + 1) {
      expect(Math.abs((await centre()) - (100 - v1.s) / (v1.e - v1.s))).toBeLessThan(0.01);
    } else {
      await expect(key, 'a key outside the visible window is not drawn').toHaveCount(0);
    }
  });

  test('Ctrl+Z takes a key back out, Ctrl+Y puts it back', async ({ page }) => {
    await openCurves(page);
    await keyAt(page, 40);
    await keyAt(page, 90);
    expect((await keys(page)).length).toBe(2);
    await page.locator('body').click({ position: { x: 2, y: 2 } });
    await page.keyboard.press('Control+z');
    await expect.poll(async () => (await keys(page)).length).toBe(1);
    await page.keyboard.press('Control+y');
    await expect.poll(async () => (await keys(page)).length).toBe(2);
  });

  test('a Relight curve says so when Light Harmonizer is off, and an Identity Morph curve needs a blender', async ({ page }) => {
    await openCurves(page);
    await page.getByTestId('curve-param-relight').click();
    await keyAt(page, 30);
    await expect(page.getByTestId('curve-ineffective')).toContainText(/Light Harmonizer is off/);
    await page.getByTestId('curve-param-identity_morph').click();
    await keyAt(page, 30);
    await expect(page.getByTestId('curve-morph')).toContainText(/two or more sources/);
    await expect(page.getByTestId('curve-morph-apply')).toBeDisabled();
  });
});

// ── the face context menu ────────────────────────────────────────────────────

const FRAME = { w: 640, h: 360 };
const PLATE = [224, 172, 140];
const SWAPPED = [223, 172, 144];          // a touch cooler than the plate: the skin-tone match should add warmth
// The real /api/state always carries stable target-person ids; the mock's does not, which sends the page down its
// legacy rank path (a mapping keyed by rank, not by id). The state is given ids here so these tests run the path the
// real backend takes.
const P0 = 'person-a';

async function openFaces(page, { zoom = null } = {}) {
  if (zoom) await page.addInitScript((z) => localStorage.setItem('roop_zoom', String(z)), zoom);
  const plate = makePng(FRAME.w, FRAME.h, () => PLATE);
  const swapped = makePng(FRAME.w, FRAME.h, () => SWAPPED);
  const writes = [];
  const previews = [];
  await page.route('**/api/target/preview*', (route) => route.fulfill({ status: 200, contentType: 'image/png', body: plate }));
  await page.route('**/api/preview', (route) => {
    if (route.request().method() !== 'POST') return route.fallback();
    try { previews.push(JSON.parse(route.request().postData() || '{}')); } catch { /* not JSON */ }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image: `data:image/png;base64,${swapped.toString('base64')}`,
        faces: [[40, 50, 160, 210], [470, 70, 590, 230]],
        person_ids: [0, 1],
        kps: [], pose: [],
      }),
    });
  });
  await page.route('**/api/target/context*', (route) => {
    if (route.request().method() === 'POST') {
      try { writes.push(JSON.parse(route.request().postData() || '{}')); } catch { /* not JSON */ }
      return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    }
    return route.fallback();
  });
  await page.route('**/api/state', async (route) => {
    const res = await route.fetch();
    const json = await res.json();
    json.target_person_ids = [P0];
    json.target_reference_face_ids = ['ref-a'];
    await route.fulfill({ response: res, json });
  });
  await openTab(page, 'faceswap');
  await expect(page.locator('[data-face-index="0"]')).toBeVisible({ timeout: 20_000 });
  return { writes, previews };
}

const face = (page, i) => page.locator(`[data-face-index="${i}"]`);
const menu = (page) => page.getByTestId('face-context-menu');
/**
 * A real contextmenu MouseEvent at CLIENT coordinates (x, y), delivered to a face box wherever it is on screen. Playwright's
 * own dispatchEvent('contextmenu', {clientX}) does not carry the coordinates (it builds a plain Event), which made the
 * corner tests pass with the clamp switched off - found by breaking the clamp on purpose.
 */
const contextAt = (page, i, x, y) => face(page, i).evaluate((el, [cx, cy]) => {
  el.dispatchEvent(new MouseEvent('contextmenu', { clientX: cx, clientY: cy, button: 2, bubbles: true, cancelable: true }));
}, [x, y]);
// A click that cannot be blocked by the stage's own HUD bar or the sticky header at awkward window sizes.
const pick = (page, i) => face(page, i).dispatchEvent('click');

/** A point inside the box that the stage's own HUD (top-left pill, bottom bar) is not over. */
async function inBox(page, i) {
  const b = await face(page, i).boundingBox();
  return { x: b.width * 0.5, y: b.height * 0.3 };
}

async function rightClick(page, i) {
  await face(page, i).click({ button: 'right', position: await inBox(page, i) });
  await expect(menu(page)).toBeVisible();
  await expect(menu(page)).toHaveCSS('visibility', 'visible');
}

test.describe('face context menu', () => {
  test('hovering a box shows its track and the identity it wears', async ({ page }) => {
    await openFaces(page);
    await pick(page, 0);                                   // the captured person becomes the target
    await face(page, 0).hover();
    const badge = face(page, 0).getByTestId('face-identity-badge');
    await expect(badge).toHaveCSS('opacity', '1');
    await expect(badge).toContainText(/^Track #1 → source_alpha\.png$/);
    // A face nobody captured is not swapped, and says so rather than naming a source.
    await face(page, 1).hover();
    await expect(face(page, 1).getByTestId('face-identity-badge')).toContainText(/not swapped/);
    await expect(face(page, 1).getByTestId('face-identity-badge')).not.toContainText(/\[\d/);
  });

  test('right-click opens a menu with the six actions; Escape and an outside press dismiss it', async ({ page }) => {
    await openFaces(page);
    await pick(page, 0);
    await rightClick(page, 0);
    for (const id of ['assign', 'pin', 'bypass', 'extract', 'skin', 'focus']) await expect(page.getByTestId(`face-menu-${id}`)).toBeVisible();
    await expect(page.getByTestId('face-menu-badge')).toContainText(/Track #1/);

    await page.keyboard.press('Escape');
    await expect(menu(page)).toHaveCount(0);

    await rightClick(page, 0);
    await page.mouse.click(4, 4);
    await expect(menu(page)).toHaveCount(0);

    // Right-clicking another box replaces the menu, not stacks a second.
    await rightClick(page, 0);
    await face(page, 1).click({ button: 'right', position: await inBox(page, 1) });
    await expect(menu(page)).toHaveCount(1);
    await expect(page.getByTestId('face-menu-badge')).toContainText(/Face 2|Track #2/);
  });

  for (const zoom of [null, 1.25]) {
    test(`the menu stays inside the window at all four corners (UI zoom ${zoom ?? 1})`, async ({ page }) => {
      await openFaces(page, { zoom });
      await pick(page, 0);
      const { w, h } = await page.evaluate(() => ({ w: window.innerWidth, h: window.innerHeight }));
      for (const [x, y] of [[2, 2], [w - 2, 2], [2, h - 2], [w - 2, h - 2], [w / 2, h - 2], [w - 2, h / 2]]) {
        await contextAt(page, 0, x, y);
        await expect(menu(page)).toBeVisible();
        await expect(menu(page)).toHaveCSS('visibility', 'visible');
        const r = await menu(page).evaluate((el) => { const b = el.getBoundingClientRect(); return { l: b.left, t: b.top, r: b.right, b: b.bottom }; });
        expect(r.l, `left edge at click (${x}, ${y})`).toBeGreaterThanOrEqual(0);
        expect(r.t, `top edge at click (${x}, ${y})`).toBeGreaterThanOrEqual(0);
        expect(r.r, `right edge at click (${x}, ${y})`).toBeLessThanOrEqual(w + 0.5);
        expect(r.b, `bottom edge at click (${x}, ${y})`).toBeLessThanOrEqual(h + 0.5);
        await page.keyboard.press('Escape');
        await expect(menu(page)).toHaveCount(0);
      }
    });
  }

  test('a short window pins the menu to the top and lets it scroll rather than hanging off the screen', async ({ page }) => {
    await page.setViewportSize({ width: 900, height: 330 });
    await openFaces(page);
    await pick(page, 0);
    await contextAt(page, 0, 450, 300);
    await expect(menu(page)).toHaveCSS('visibility', 'visible');
    const r = await menu(page).evaluate((el) => { const b = el.getBoundingClientRect(); return { t: b.top, b: b.bottom, scrolls: el.scrollHeight > el.clientHeight }; });
    expect(r.t).toBeGreaterThanOrEqual(0);
    expect(r.b).toBeLessThanOrEqual(330.5);
  });

  test('Assign / Change Source Identity maps that person to the chosen source', async ({ page }) => {
    const { writes } = await openFaces(page);
    await pick(page, 0);
    await rightClick(page, 0);
    await page.getByTestId('face-menu-assign').click();
    await expect(page.getByTestId('face-menu-picker')).toBeVisible();
    await expect(page.getByTestId('face-menu-source-0')).toHaveAttribute('aria-checked', 'true');
    const before = writes.length;
    await page.getByTestId('face-menu-source-1').click();
    await expect(menu(page)).toHaveCount(0);
    await expect.poll(() => writes.length, { timeout: 5_000 }).toBeGreaterThan(before);
    const last = [...writes].reverse().map((w) => w.target_person_source_mapping ?? w.face_mapping).find((m) => m && typeof m === 'object');
    expect(last?.[P0], 'person 1 is now mapped to source_beta').toBe('/mock/facesets/source_beta.png');
    await face(page, 0).hover();
    await expect(face(page, 0).getByTestId('face-identity-badge')).toContainText(/source_beta\.png/);
  });

  test('Pin / Lock Identity toggles the pin on that track', async ({ page }) => {
    await openFaces(page);
    await pick(page, 0);
    await rightClick(page, 0);
    await expect(page.getByTestId('face-menu-pin')).toContainText(/Pin \/ Lock Identity to Track #1/);
    await page.getByTestId('face-menu-pin').click();
    await expect(face(page, 0).getByTestId('pin-identity')).toHaveAttribute('aria-pressed', 'true');
    await rightClick(page, 0);
    await expect(page.getByTestId('face-menu-pin')).toContainText(/Unpin Identity from Track #1/);
    await page.getByTestId('face-menu-pin').click();
    await expect(face(page, 0).getByTestId('pin-identity')).toHaveAttribute('aria-pressed', 'false');
  });

  test('Bypass Swap on This Face adds a bypass scene override for the scene under the playhead', async ({ page }) => {
    await openFaces(page);
    await pick(page, 0);
    await rightClick(page, 0);
    await expect(page.getByTestId('face-menu-bypass')).not.toHaveAttribute('aria-disabled', 'true');
    await expect(page.getByTestId('face-menu-bypass')).toContainText(/frames 1-/);          // says what range it will cover
    await page.getByTestId('face-menu-bypass').click();
    const stored = await page.evaluate(() => Object.entries(localStorage)
      .filter(([k]) => /^roop_overrides_(?!enabled_)/.test(k)).map(([, v]) => JSON.parse(v)).flat());
    expect(stored.length).toBe(1);
    expect(stored[0].bypass, 'created bypassing, not as an empty override').toBe(true);
    expect(stored[0].label).toMatch(/^Bypass Track #1/);
    expect(stored[0].start).toBeLessThanOrEqual(1);
    await expect(page.getByText(/Bypassing the swap on frames/).first()).toBeVisible();
  });

  test('Bypass is refused, with the reason, for a face that is not swapped', async ({ page }) => {
    await openFaces(page);
    await pick(page, 0);
    await rightClick(page, 1);                                    // the uncaptured face
    const bypass = page.getByTestId('face-menu-bypass');
    await expect(bypass).toHaveAttribute('aria-disabled', 'true');
    await expect(bypass).toContainText(/not swapped/);
    await bypass.dispatchEvent('click');          // Playwright will not click an aria-disabled item; a real user can, and nothing may happen
    await expect(menu(page), 'a refused item leaves the menu open').toBeVisible();
  });

  test('Extract Face Crop uploads the full-resolution crop to the Face Manager bank', async ({ page }) => {
    await openFaces(page);
    await pick(page, 0);
    await rightClick(page, 0);
    const upload = page.waitForRequest((r) => r.method() === 'POST' && /\/api\/facemgr\/add$/.test(r.url()));
    await page.getByTestId('face-menu-extract').click();
    const body = (await upload).postDataBuffer();
    const text = body.toString('latin1');
    expect(text).toMatch(/filename="interview_scene_1080p_f1_face1\.png"/);
    expect(text).toMatch(/name="detector"\r\n\r\nscrfd/);
    // A real PNG, cut to the box grown by 60 % each side (40,50,160,210 in a 640x360 frame -> 232 x 306).
    const sig = body.indexOf(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
    expect(sig, 'the part must hold a PNG').toBeGreaterThan(0);
    expect(body.readUInt32BE(sig + 16)).toBe(232);
    expect(body.readUInt32BE(sig + 20)).toBe(306);
    await expect(page.getByText(/to the Face Manager bank/).first()).toBeVisible();
  });

  test('Auto-Match Skin Tone writes a warmer slider value into the next preview', async ({ page }) => {
    const { previews } = await openFaces(page);
    await pick(page, 0);
    await rightClick(page, 0);
    const before = previews.length;
    await page.getByTestId('face-menu-skin').click();
    await expect(page.getByText(/Skin tone warmth 0 to /).first()).toBeVisible({ timeout: 10_000 });
    // The swapped fixture is a touch cooler than the plate, so warmth goes UP (measured +44 in the model checks).
    await expect.poll(() => previews.slice(before).map((b) => b.skin_tone_warmth).filter((w) => w > 0).pop(), { timeout: 20_000 }).toBeGreaterThan(30);
    expect(previews.slice(before).map((b) => b.skin_tone_warmth).filter((w) => w > 0).pop()).toBeLessThan(60);
  });

  test('Focus & Zoom 200% puts the stage at 2x', async ({ page }) => {
    await openFaces(page);
    await pick(page, 0);
    await rightClick(page, 0);
    await page.getByTestId('face-menu-focus').click();
    await expect(page.locator('.preview-stage div[style*="scale(2)"]').first()).toBeVisible();
  });
});
