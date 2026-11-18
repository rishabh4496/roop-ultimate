import { test, expect } from '@playwright/test';
import { openTab, PNG_1X1 } from './helpers.js';

// Stage 4: the multi-lane NLE timeline in the Face Swap workstation, in a real browser.
//
// What the pure models cannot show (.render-check/nle-stage4-check.mjs covers those): that the canvases
// paint, that a pointer lands on the frame the playhead is drawn on, that the toggles are wired to the
// lanes, that the deck drives the same state the keyboard does. Everything runs against the production
// bundle + the mock backend (180-frame, 30 fps clip), so it says nothing about real decoding; it checks
// what the SHELL does with what it is given.
//
// The timeline lives in a scrolling dock (260 px by default). Most tests seed a taller dock so a lane is
// on screen when the test aims at it; one test keeps the DEFAULT dock to prove the controls you scrub
// with (transport, overview, ruler) are sticky and stay on screen while the lanes scroll under them.

test.use({ viewport: { width: 1600, height: 1300 } });

const NLE = '[data-testid="nle-timeline"]';
const TALL_DOCK = { panelSizes: { bottomHeight: 600 } };

test.beforeEach(async ({ page }) => {
  await page.addInitScript((ws) => { try { localStorage.setItem('roop_workspace', JSON.stringify(ws)); } catch { /* blocked */ } }, TALL_DOCK);
});

const colorCount = (canvas, rgb, tol = 48) => canvas.evaluate((c, [r, g, b, t]) => {
  const { data } = c.getContext('2d').getImageData(0, 0, c.width, c.height);
  let n = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] > 40 && Math.abs(data[i] - r) < t && Math.abs(data[i + 1] - g) < t && Math.abs(data[i + 2] - b) < t) n++;
  }
  return n;
}, [...rgb, tol]);

/** Open Face Swap, switch on the Pro Studio view and (optionally) run the mock Pre-Scan. */
async function openNle(page, { prescan = true, spans = null, beforeToggle = null } = {}) {
  if (spans) await page.route('**/api/trajectory/spans*', (route) => route.fulfill({ json: spans }));
  await openTab(page, 'faceswap');
  beforeToggle?.();
  await page.getByTitle('Toggle Pro Studio NLE View & Viewport').click();
  const tl = page.locator(NLE);
  await tl.waitFor();
  if (prescan) {
    await page.getByRole('button', { name: /Pre-Scan/ }).last().click();
    await page.getByTestId('lane-track-0').waitFor();
  }
  await page.getByTestId('nle-ruler').waitFor();
  return tl;
}

/**
 * Scroll `locator` to the middle of its scrolling dock and prove it is the topmost thing at its centre
 * (the sticky controls and the page's own sticky bars could be covering it), then return its box. A mouse
 * event aimed at coordinates something else covers lands on THAT element and the test "fails" for a reason
 * that has nothing to do with the timeline (the same trap timeline.spec.js documents).
 */
async function reveal(page, locator) {
  await locator.evaluate((el) => el.scrollIntoView({ block: 'center' }));
  await page.waitForTimeout(120);
  const box = await locator.boundingBox();
  const clear = await locator.evaluate((el, [x, y]) => {
    const hit = document.elementFromPoint(x, y);
    return !!hit && (hit === el || el.contains(hit) || hit.contains(el));
  }, [box.x + box.width / 2, box.y + box.height / 2]);
  expect(clear, 'the lane must be the topmost element at the point the test aims at').toBeTruthy();
  return box;
}

/** The lane area's rect (the ruler layer): the one element every pixel <-> frame mapping reads. */
const areaBox = (page) => page.getByTestId('nle-lane-area').boundingBox();
const rulerBox = (page) => page.getByTestId('nle-ruler').boundingBox();
const playheadFrame = async (page) => Number(await page.getByTestId('nle-playhead').getAttribute('data-frame'));
const playheadX = async (page) => (await page.getByTestId('nle-playhead').boundingBox()).x;
const view = async (tl) => ({ start: Number(await tl.getAttribute('data-view-start')), end: Number(await tl.getAttribute('data-view-end')) });
const span = (v) => v.end - v.start;
const readoutFrame = async (page) => Number(((await page.getByTestId('nle-readout-frame').textContent()) || '').match(/#(\d+)/)?.[1]);

/** Ctrl+wheel over the ruler: zoom the track about that point. */
async function ctrlWheel(page, fraction, notches, dy = -240) {
  const area = await areaBox(page);
  const rb = await rulerBox(page);
  await page.mouse.move(area.x + area.width * fraction, rb.y + rb.height / 2);
  await page.keyboard.down('Control');
  for (let i = 0; i < notches; i++) await page.mouse.wheel(0, dy);
  await page.keyboard.up('Control');
}

/** A spans payload with a dropout (frames 50-79) on every track, so the coverage lane has all three states. */
const GAP_SPANS = {
  available: true,
  relative: false,
  tracks: [0, 1].map((track) => ({ track, first: 0, last: 179, frames: 150, ranges: [[0, 49], [80, 179]], source_index: track, faceset_id: null, faceset_path: null, via: 'mock' })),
};

test.describe('NLE timeline: ruler, playhead, scrub', () => {
  test('a click lands on the frame the playhead is drawn on, at the left edge, the middle and the right', async ({ page }) => {
    await openNle(page, { prescan: false });
    const rb = await rulerBox(page);
    const area = await areaBox(page);
    expect(Math.abs(rb.x - area.x), 'the ruler lane and the overlay layer share a left edge').toBeLessThan(1.5);
    const pxPerFrame = area.width / 179;
    // The pre-fix mapping divided by the whole row (header included), so a click at the LEFT edge read as
    // frame ~28 of 180 and the playhead jumped away from the pointer.
    // (ratios avoid a half-frame rounding boundary, and the frame is allowed +-1: Chromium may truncate a fractional clientX)
    for (const r of [0.04, 0.503, 0.97]) {
      const x = area.x + r * area.width;
      await page.mouse.click(x, rb.y + rb.height - 8);
      const expected = Math.round(r * 179);
      await expect.poll(async () => Math.abs((await playheadFrame(page)) - expected), { message: `click at ${r * 100}%` }).toBeLessThanOrEqual(1);
      const drawn = await playheadX(page);
      expect(Math.abs(drawn - x), `the playhead must be drawn where it was clicked (${r * 100}%)`).toBeLessThanOrEqual(pxPerFrame / 2 + 1.5);
    }
  });

  test('the ruler paints SMPTE timecode and #frame numbers, and its density follows the zoom', async ({ page }) => {
    const tl = await openNle(page, { prescan: false });
    const canvas = page.getByTestId('nle-ruler-canvas');
    await expect.poll(() => colorCount(canvas, [0, 240, 255], 60), { message: 'frame-number labels (cyan)' }).toBeGreaterThan(30);
    await expect.poll(() => colorCount(canvas, [212, 212, 212], 40), { message: 'timecode labels' }).toBeGreaterThan(30);
    await expect(canvas).toHaveAttribute('data-first-label', /^#\d{5,}$/);
    const before = { major: Number(await canvas.getAttribute('data-major-step')), view: await view(tl) };

    const appZoom = await page.evaluate(() => document.documentElement.style.zoom);
    await ctrlWheel(page, 0.6, 6);
    await expect.poll(async () => span(await view(tl)), { message: 'Ctrl+wheel zooms the track' }).toBeLessThan(span(before.view) / 2);
    expect(Number(await canvas.getAttribute('data-major-step')), 'tick spacing gets finer when zoomed in').toBeLessThan(before.major);
    expect(await page.evaluate(() => document.documentElement.style.zoom), 'one wheel notch zooms the track, never the app').toBe(appZoom);
  });

  test('the zoom slider zooms the track and agrees with the wheel', async ({ page }) => {
    const tl = await openNle(page, { prescan: false });
    const slider = page.getByTestId('nle-zoom-slider');
    await expect(slider).toHaveValue('0');
    await slider.fill('60');
    await expect.poll(async () => span(await view(tl))).toBeLessThan(60);
    await page.getByTestId('nle-zoom-fit').click();
    await expect.poll(async () => span(await view(tl))).toBe(179);
    await expect(slider).toHaveValue('0');
  });

  test('the playhead is the cyan token and its cap scrubs, with a floating still + frame + timecode card', async ({ page }) => {
    const previews = [];
    page.on('request', (req) => { if (/\/api\/target\/preview\?/.test(req.url())) previews.push(req.url()); });
    // a real, decodable image for the exact-frame request (the mock serves SVG, which createImageBitmap refuses)
    await page.route(/\/api\/target\/preview\?.*width=240/, (route) => route.fulfill({ status: 200, contentType: 'image/png', body: PNG_1X1 }));
    await openNle(page, { prescan: false });

    expect(await page.getByTestId('nle-playhead-line').evaluate((el) => getComputedStyle(el).backgroundColor), 'the playhead line is --accent-cyan').toBe('rgb(0, 240, 255)');
    expect(await page.getByTestId('nle-playhead').locator('div').first().evaluate((el) => getComputedStyle(el).backgroundColor), 'and so is its continuation through the lanes').toBe('rgb(0, 240, 255)');

    const cb = await page.getByTestId('nle-playhead-cap').boundingBox();
    const area = await areaBox(page);
    await page.mouse.move(cb.x + cb.width / 2, cb.y + cb.height / 2);
    await page.mouse.down();
    await page.mouse.move(area.x + area.width * 0.4, cb.y + 6, { steps: 6 });

    const tip = page.getByTestId('nle-scrub-tooltip');
    await expect(tip).toBeVisible();
    const f = await playheadFrame(page);
    await expect(tip).toHaveAttribute('data-frame', String(f));
    await expect(page.getByTestId('nle-scrub-frame')).toHaveText(`#${String(f).padStart(5, '0')}`);
    await expect(page.getByTestId('nle-scrub-timecode')).toHaveText(/^\d\d:\d\d:\d\d:\d\d$/);
    const tb = await tip.boundingBox();
    expect(tb.y + tb.height, 'the card sits above the ruler, not inside the clipped timeline').toBeLessThanOrEqual(area.y + 2);
    expect(tb.x).toBeGreaterThanOrEqual(0);
    // the exact frame is requested 1-based, at the card's size, and replaces the nearest filmstrip still
    await expect.poll(() => previews.some((u) => u.includes(`frame=${f + 1}&`) && u.includes('width=240')), { message: 'exact-frame request' }).toBeTruthy();
    await expect(page.getByTestId('nle-scrub-thumb')).toHaveAttribute('data-approximate', '0');
    await page.mouse.up();
    await expect(tip).toBeHidden();
  });

  test('the filmstrip asks for 1-based frames, gets denser when zoomed in, and never re-asks for what it has', async ({ page }) => {
    const grid = [];
    const tl = await openNle(page, {
      prescan: false,
      // collect only from the moment the NLE opens: the classic timeline underneath asks for its own stills
      beforeToggle: () => page.on('request', (req) => { if (req.url().includes('/api/target/preview_grid') && req.url().includes('width=160')) grid.push(new URL(req.url()).searchParams.get('frames')); }),
    });
    const canvas = page.getByTestId('nle-filmstrip-canvas');
    await expect.poll(() => grid.length, { message: 'a still request' }).toBeGreaterThan(0);
    // 0-based frame 0 is UI frame 1: the old strip sent 0 and so showed every still one frame early
    expect(grid[0].split(',').map(Number)[0], 'frame 0 is requested as 1').toBe(1);
    expect(Math.min(...grid.flatMap((g) => g.split(',').map(Number))), 'no still is requested as frame 0').toBeGreaterThanOrEqual(1);
    // The mock's stills are 1x1 JPEGs. Whether or not they decode, the strip must settle: a fetch that
    // brought nothing back used to re-run the request effect forever (an unbounded stream of identical GETs).
    await page.waitForTimeout(2500);
    const settled = grid.length;
    await page.waitForTimeout(2500);
    expect(grid.length, 'no further requests once settled').toBe(settled);
    expect(settled, 'and not a runaway before settling').toBeLessThan(6);

    const wide = Number(await canvas.getAttribute('data-slot-stride'));
    await ctrlWheel(page, 0.5, 8);
    await expect.poll(async () => Number(await canvas.getAttribute('data-slot-stride'))).toBeLessThan(wide);
    expect(span(await view(tl))).toBeLessThan(60);
  });

  test('the controls you scrub with stay on screen in the default 260 px dock while the lanes scroll', async ({ page }) => {
    await page.addInitScript(() => { try { localStorage.setItem('roop_workspace', JSON.stringify({ panelSizes: { bottomHeight: 260 } })); } catch { /* blocked */ } });
    await openNle(page);
    const deck = page.getByTestId('timeline-deck');
    const db = await deck.boundingBox();
    expect(db.height, 'this test exists for the SHORT dock').toBeLessThan(300);
    // scroll to the very bottom: the face-track lanes
    await deck.evaluate((el) => { el.scrollTop = el.scrollHeight; });
    await page.waitForTimeout(150);
    expect(await deck.evaluate((el) => el.scrollTop), 'the dock really scrolled').toBeGreaterThan(100);
    for (const id of ['nle-transport', 'nle-minimap', 'nle-ruler', 'nle-playhead-cap']) {
      const el = page.getByTestId(id);
      const b = await el.boundingBox();
      const inside = b.y >= db.y - 1 && b.y + b.height <= db.y + db.height + 1;
      const covered = await el.evaluate((node, [x, y]) => { const hit = document.elementFromPoint(x, y); return !(hit && (hit === node || node.contains(hit) || hit.contains(node))); }, [b.x + b.width / 2, b.y + b.height / 2]);
      expect(inside && !covered, `${id} must stay visible inside the dock when scrolled to the bottom`).toBeTruthy();
    }
    // and they still work from there: clicking the ruler moves the playhead
    const rb = await rulerBox(page);
    const area = await areaBox(page);
    await page.mouse.click(area.x + area.width * 0.7, rb.y + rb.height - 8);
    await expect.poll(async () => Math.abs((await playheadFrame(page)) - Math.round(0.7 * 179))).toBeLessThanOrEqual(1);
  });
});

test.describe('NLE timeline: lanes', () => {
  test('hide, height and lock are real, and the layout survives a reload', async ({ page }) => {
    await openNle(page);
    const v1 = page.getByTestId('lane-v1-filmstrip');
    await reveal(page, v1);
    const h = async () => (await v1.boundingBox()).height;
    const normal = await h();
    await page.getByTestId('lane-v1-height').click();
    const tall = await h();
    await page.getByTestId('lane-v1-height').click();
    const compact = await h();
    expect(compact).toBeLessThan(normal);
    expect(normal).toBeLessThan(tall);
    await page.getByTestId('lane-v1-height').click(); // back to normal

    await page.getByTestId('lane-v1-hide').click();
    await expect(v1).toHaveAttribute('data-hidden', '1');
    await expect(page.getByTestId('nle-filmstrip-canvas')).toHaveCount(0);
    expect(await h(), 'a hidden lane collapses to its header strip').toBeLessThan(30);
    await expect(page.getByTestId('lane-v1-hide')).toHaveAttribute('aria-pressed', 'true');

    // lock refuses edits: the keyframe lane cannot take a keyframe while locked
    const add = page.getByRole('button', { name: /^Add .* keyframe$/ });
    await reveal(page, page.getByTestId('lane-kf-editor'));
    await expect(add).toBeEnabled();
    await page.getByTestId('lane-kf-lock').click();
    await expect(add).toBeDisabled();
    await expect(page.getByTestId('lane-kf-lock')).toHaveAttribute('aria-pressed', 'true');

    await page.reload();
    await page.getByTestId('lane-v1-filmstrip').waitFor();
    await expect(page.getByTestId('lane-v1-filmstrip')).toHaveAttribute('data-hidden', '1');
    await expect(page.getByTestId('lane-kf-editor')).toHaveAttribute('data-locked', '1');
  });

  test('a locked keyframe lane refuses edits to an existing keyframe, but still lets you select it', async ({ page }) => {
    await openNle(page);
    await reveal(page, page.getByTestId('lane-kf-editor'));
    await page.getByRole('button', { name: /^Add .* keyframe$/ }).click();
    const kf = page.getByTestId('nle-keyframe');
    await expect(kf).toHaveCount(1);
    await page.getByTestId('lane-kf-lock').click();
    await kf.click();
    const value = page.getByRole('spinbutton', { name: 'Keyframe value' });
    await expect(value).toBeVisible();
    await expect(value, 'locked: the value cannot be edited').toBeDisabled();
    await expect(page.getByRole('button', { name: 'Delete keyframe' })).toBeDisabled();
  });

  test('the coverage heatmap paints emerald / amber / rose from real data, and a click on a problem jumps to it', async ({ page }) => {
    await openNle(page, { spans: GAP_SPANS });
    const canvas = page.getByTestId('nle-health-canvas');
    await reveal(page, page.getByTestId('lane-health-body'));
    await expect.poll(() => colorCount(canvas, [16, 185, 129]), { message: 'emerald: face tracked' }).toBeGreaterThan(100);
    await expect.poll(() => colorCount(canvas, [245, 158, 11]), { message: 'amber: extreme pose (from /api/trajectory/series)' }).toBeGreaterThan(50);
    await expect.poll(() => colorCount(canvas, [244, 63, 94]), { message: 'rose: the 50-79 dropout' }).toBeGreaterThan(50);
    await expect(page.getByTestId('nle-health-status')).toContainText(/\d+% ok · \d+% risk · \d+% lost/);

    // frames 50..79 are the dropout: a click anywhere in it jumps to its first frame
    const area = await areaBox(page);
    const body = await page.getByTestId('lane-health-body').boundingBox();
    await page.mouse.click(area.x + (65 / 179) * area.width, body.y + body.height / 2);
    await expect.poll(() => playheadFrame(page)).toBe(50);

    // prev / next problem walk the amber and rose runs in order, from the start of the clip
    await page.getByTestId('transport-start').click();
    await expect.poll(() => playheadFrame(page)).toBe(0);
    const seen = [];
    for (let i = 0; i < 3; i++) {
      await page.getByTestId('nle-health-next').click();
      await expect.poll(() => playheadFrame(page)).toBeGreaterThan(seen.at(-1) ?? -1);
      seen.push(await playheadFrame(page));
    }
    expect(seen[0], 'first problem: the first amber run (|yaw| > 45 from frame 20)').toBe(20);
    expect(seen[1], 'then the dropout').toBe(50);
    await page.getByTestId('nle-health-prev').click();
    await expect.poll(() => playheadFrame(page)).toBe(seen[1]);
  });

  test('a track header follows the project: the swap mute and the faceset assignment show their new value at once', async ({ page }) => {
    await openNle(page);
    await reveal(page, page.getByTestId('lane-track-0'));
    const mute = page.getByTestId('lane-track-0-mute');
    await expect(mute).toHaveAttribute('aria-pressed', 'false');
    await mute.click();
    await expect(mute, 'muting must reach the lane (a stale stable-reference key once hid it)').toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('lane-track-0')).toHaveClass(/opacity-60/);
    await mute.click();
    await expect(mute).toHaveAttribute('aria-pressed', 'false');
    const select = page.getByRole('combobox', { name: 'Faceset for Track #1' });
    const before = await select.inputValue();
    const other = (await select.locator('option').evaluateAll((os) => os.map((o) => o.value))).find((v) => v && v !== before);
    await select.selectOption(other);
    await expect(select).toHaveValue(other);
  });

  test('a sustained head turn is ONE run on its track lane, not a dot per sample', async ({ page }) => {
    await openNle(page);
    await reveal(page, page.getByTestId('lane-track-0'));
    const runs = page.getByTestId('lane-track-0').getByTestId('nle-warning-run');
    await expect.poll(() => runs.count()).toBeGreaterThan(0);
    // the mock series samples every frame (180 samples in view) and 62*sin(f/24) crosses 45 deg a handful of times
    expect(await runs.count(), 'runs, not samples').toBeLessThan(15);
    await expect(runs.first()).toHaveAttribute('data-run-type', 'pose');
    expect(await runs.first().evaluate((el) => getComputedStyle(el).backgroundColor), 'a pose run is amber, never the occlusion red').toBe('rgb(245, 158, 11)');
    await expect(runs.first()).toHaveAttribute('title', /Hard Yaw \(>50°\)· ?\d+ samples|Hard Yaw \(>50°\) · \d+ samples/);
  });

  test('an unscanned clip is a hatched "not scanned" strip, never rose', async ({ page }) => {
    await openNle(page, { prescan: false });
    await reveal(page, page.getByTestId('lane-health-body'));
    await expect(page.getByTestId('lane-health-header')).toContainText('not scanned');
    const canvas = page.getByTestId('nle-health-canvas');
    expect(await colorCount(canvas, [244, 63, 94])).toBe(0);
    expect(await colorCount(canvas, [16, 185, 129])).toBe(0);
  });

  test('the waveform lane paints peaks from the mock envelope', async ({ page }) => {
    await openNle(page, { prescan: false });
    await reveal(page, page.getByTestId('lane-a1-waveform'));
    await expect.poll(() => colorCount(page.getByTestId('nle-waveform-canvas'), [94, 234, 212], 40), { message: 'waveform body (teal)' }).toBeGreaterThan(50);
  });
});

test.describe('NLE timeline: markers', () => {
  test('one marker list: + Marker and the M key both write it, drag moves it, the popover edits colour + note, and it survives a reload', async ({ page }) => {
    const tl = await openNle(page, { prescan: false });
    const flags = page.getByTestId('nle-marker-flag');
    await page.getByTestId('nle-add-marker').click();
    await expect(flags).toHaveCount(1);

    // the M key writes the SAME list (it used to feed a second, invisible one)
    const rb = await rulerBox(page);
    const area = await areaBox(page);
    await page.mouse.click(area.x + area.width * 0.5, rb.y + rb.height - 8);
    await tl.focus();
    await page.keyboard.press('m');
    await expect(flags).toHaveCount(2);
    await expect(page.getByTestId('nle-add-marker')).toContainText('(2)');

    // drag the first flag across the lane
    await reveal(page, page.getByTestId('lane-segments'));
    const first = flags.first();
    const startFrame = Number(await first.getAttribute('data-frame'));
    const fb = await first.boundingBox();
    await page.mouse.move(fb.x + 3, fb.y + 6);
    await page.mouse.down();
    await page.mouse.move(fb.x + 3 + area.width * 0.2, fb.y + 6, { steps: 8 });
    await page.mouse.up();
    await expect.poll(async () => Number(await flags.first().getAttribute('data-frame'))).toBeGreaterThan(startFrame + 10);
    await expect(flags).toHaveCount(2);

    // click (no drag): popover; tag it red and write a note
    const moved = flags.first();
    const mb = await moved.boundingBox();
    await page.mouse.click(mb.x + 3, mb.y + 6);
    const pop = page.getByTestId('nle-marker-popover');
    await expect(pop).toBeVisible();
    await page.getByTestId('marker-color-rose').click();
    await page.getByTestId('marker-note').fill('Hand covers mouth\nretake');
    await expect(moved).toHaveAttribute('title', /Hand covers mouth/);
    await page.keyboard.press('Escape');
    await expect(pop).toBeHidden();

    // storage: the legacy list is still bare numbers; colour + note live beside it
    const stored = await page.evaluate(() => {
      const k = Object.keys(localStorage).find((x) => x.startsWith('roop_markers_'));
      const m = Object.keys(localStorage).find((x) => x.startsWith('roop_marker_meta_'));
      return { frames: JSON.parse(localStorage.getItem(k)), meta: JSON.parse(localStorage.getItem(m)) };
    });
    expect(stored.frames.every((n) => typeof n === 'number')).toBeTruthy();
    expect(Object.values(stored.meta).some((v) => v.note.startsWith('Hand covers mouth') && v.color === '#fb7185')).toBeTruthy();

    await page.reload();
    await page.locator('nav[aria-label="Main"] button[aria-current="page"]').waitFor();
    await page.locator('.deferred-fallback').waitFor({ state: 'detached' });
    await page.locator(NLE).waitFor();
    await expect(page.getByTestId('nle-marker-flag')).toHaveCount(2);
    await expect(page.getByTestId('nle-marker-flag').filter({ hasText: 'Hand covers mouth' })).toHaveCount(1);
  });

  test('a locked markers lane refuses a drag and shows the popover read-only', async ({ page }) => {
    await openNle(page, { prescan: false });
    await page.getByTestId('nle-add-marker').click();
    await reveal(page, page.getByTestId('lane-segments'));
    const flag = page.getByTestId('nle-marker-flag').first();
    const at = await flag.getAttribute('data-frame');
    await page.getByTestId('lane-segments-lock').click();
    await expect(page.getByTestId('nle-add-marker')).toBeDisabled();
    const area = await areaBox(page);
    const fb = await flag.boundingBox();
    await page.mouse.move(fb.x + 3, fb.y + 6);
    await page.mouse.down();
    await page.mouse.move(fb.x + 3 + area.width * 0.3, fb.y + 6, { steps: 6 });
    await page.mouse.up();
    await expect(flag, 'a locked lane does not move its markers').toHaveAttribute('data-frame', at);
    const fb2 = await flag.boundingBox();
    await page.mouse.click(fb2.x + 3, fb2.y + 6);
    await expect(page.getByTestId('marker-note')).toHaveAttribute('readonly', '');
    await expect(page.getByTestId('marker-delete')).toBeDisabled();
  });
});

test.describe('NLE timeline: transport deck', () => {
  test('play / step / loop / speed are wired to the playback state the keyboard drives', async ({ page }) => {
    await openNle(page, { prescan: false });
    await expect(page.getByTestId('nle-transport')).toBeVisible();

    // step buttons move exactly one frame
    const rb = await rulerBox(page);
    const area = await areaBox(page);
    await page.mouse.click(area.x + area.width * 0.3, rb.y + rb.height - 8);
    await expect.poll(() => readoutFrame(page)).toBeGreaterThan(5);
    const f0 = await readoutFrame(page);
    await page.getByTestId('transport-step-fwd').click();
    await expect.poll(() => readoutFrame(page)).toBe(f0 + 1);
    await page.getByTestId('transport-step-back').click();
    await page.getByTestId('transport-step-back').click();
    await expect.poll(() => readoutFrame(page)).toBe(f0 - 1);
    await page.getByTestId('transport-start').click();
    await expect.poll(() => readoutFrame(page)).toBe(0);
    await page.getByTestId('transport-end').click();
    await expect.poll(() => readoutFrame(page)).toBe(179);

    // loop is a toggle
    const loop = page.getByTestId('transport-loop');
    const loopBefore = await loop.getAttribute('aria-pressed');
    await loop.click();
    await expect(loop).toHaveAttribute('aria-pressed', loopBefore === 'true' ? 'false' : 'true');

    // speed: the pick SURVIVES pressing Play (the shortcut hook used to reset it to 1x on every play/pause)
    await page.getByTestId('transport-start').click();
    const rate = (r) => page.getByTestId(`transport-rate-${r}`);
    await expect(rate(1)).toHaveAttribute('aria-checked', 'true');
    await rate(0.5).click();
    await expect(rate(0.5)).toHaveAttribute('aria-checked', 'true');
    const play = page.getByTestId('transport-play');
    await play.click();
    await expect(play).toHaveAttribute('aria-pressed', 'true');
    await page.waitForTimeout(500);
    await expect(rate(0.5), 'pressing Play must not reset the chosen speed to 1x').toHaveAttribute('aria-checked', 'true');
    await play.click();
    await expect(play).toHaveAttribute('aria-pressed', 'false');
    await expect(rate(0.5)).toHaveAttribute('aria-checked', 'true');
  });

  test('Space and the Play button are one state; J / K / L shuttle through the deck and the keys alike', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await openNle(page, { prescan: false });
    const play = page.getByTestId('transport-play');
    const rb = await rulerBox(page);
    const area = await areaBox(page);
    await page.mouse.click(area.x + area.width * 0.1, rb.y + rb.height - 8);
    // Regression: the scrub's preventDefault used to leave focus on the Pro Studio toggle that was clicked
    // earlier, so this Space switched the whole NLE view OFF instead of playing.
    expect(await page.evaluate(() => document.activeElement?.getAttribute('data-testid')), 'scrubbing takes focus').toBe('nle-timeline');
    await page.keyboard.press('Space');
    await expect(play).toHaveAttribute('aria-pressed', 'true');
    await page.keyboard.press('Space');
    await expect(play).toHaveAttribute('aria-pressed', 'false');

    // the L button and the L key climb the same 1x -> 2x -> 4x ladder and the deck reflects both
    await page.getByTestId('transport-shuttle-l').click();
    await expect(page.getByTestId('transport-shuttle-label')).toContainText('1×');
    await page.keyboard.press('l');
    await expect(page.getByTestId('transport-shuttle-label')).toContainText('2×');
    await expect(page.getByTestId('transport-rate-2')).toHaveAttribute('aria-checked', 'true');
    await page.getByTestId('transport-shuttle-k').click();
    await expect(page.getByTestId('transport-shuttle-label')).toHaveText('');
    await expect(page.getByTestId('transport-rate-1'), 'ending a shuttle puts the player back at 1x').toHaveAttribute('aria-checked', 'true');
    expect(errors).toEqual([]);
  });

  test('previous / next marker jump between markers and disable at the ends', async ({ page }) => {
    await openNle(page, { prescan: false });
    const next = page.getByTestId('transport-next-marker');
    const prev = page.getByTestId('transport-prev-marker');
    await expect(next).toBeDisabled();
    for (const x of [0.2, 0.5, 0.8]) {
      const area = await areaBox(page);
      const rb = await rulerBox(page);
      await page.mouse.click(area.x + area.width * x, rb.y + rb.height - 8);
      await page.getByTestId('nle-add-marker').click();
    }
    await page.getByTestId('transport-start').click();
    await expect(prev).toBeDisabled();
    const hops = [];
    for (let i = 0; i < 3; i++) {
      await next.click();
      await expect.poll(() => readoutFrame(page)).toBeGreaterThan(hops.at(-1) ?? 0);
      hops.push(await readoutFrame(page));
    }
    await expect(next, 'no marker beyond the last').toBeDisabled();
    await prev.click();
    await expect.poll(() => readoutFrame(page)).toBe(hops[1]);
  });
});

test.describe('NLE timeline: overview navigator and trim', () => {
  test('the overview window pans, resizes and recentres; the keyboard pans it', async ({ page }) => {
    const tl = await openNle(page, { prescan: false });
    await ctrlWheel(page, 0.5, 5);
    const zoomed = await view(tl);
    expect(span(zoomed)).toBeLessThan(120);

    const nav = page.getByTestId('nle-minimap-area');
    const win = page.getByTestId('nle-minimap-window');
    await expect(win).toBeVisible();
    const wb = await win.boundingBox();
    const y = wb.y + wb.height / 2;
    // drag the window body: same span, later start
    await page.mouse.move(wb.x + wb.width / 2, y);
    await page.mouse.down();
    await page.mouse.move(wb.x + wb.width / 2 + 120, y, { steps: 6 });
    await page.mouse.up();
    const panned = await view(tl);
    expect(panned.start).toBeGreaterThan(zoomed.start);
    expect(span(panned), 'panning keeps the zoom').toBe(span(zoomed));

    // drag the right edge: wider window = zoomed out
    const wb2 = await win.boundingBox();
    await page.mouse.move(wb2.x + wb2.width, y);
    await page.mouse.down();
    await page.mouse.move(wb2.x + wb2.width + 80, y, { steps: 6 });
    await page.mouse.up();
    const wider = await view(tl);
    expect(span(wider)).toBeGreaterThan(span(panned));

    // click far outside: recentre
    const nb = await nav.boundingBox();
    await page.mouse.click(nb.x + 8, y);
    expect((await view(tl)).start).toBeLessThan(wider.start);

    // keyboard
    await nav.focus();
    const k0 = (await view(tl)).start;
    await page.keyboard.press('ArrowRight');
    expect((await view(tl)).start, 'ArrowRight on the focused overview pans the window').toBeGreaterThan(k0);
    await page.keyboard.press('Home');
    expect((await view(tl)).start).toBe(0);
  });

  test('wheel over the overview zooms it, and a plain wheel elsewhere leaves the track alone', async ({ page }) => {
    const tl = await openNle(page, { prescan: false });
    const nb = await page.getByTestId('nle-minimap-area').boundingBox();
    await page.mouse.move(nb.x + nb.width / 2, nb.y + nb.height / 2);
    await page.mouse.wheel(0, -240);
    await page.mouse.wheel(0, -240);
    await expect.poll(async () => span(await view(tl)), { message: 'wheel over the overview zooms' }).toBeLessThan(179);
    const before = await view(tl);
    // a plain vertical wheel over the ruler is the dock's scroll: it must not pan or zoom the track
    const rb = await rulerBox(page);
    await page.mouse.move(nb.x + nb.width / 2, rb.y + rb.height / 2);
    await page.mouse.wheel(0, 120);
    await page.waitForTimeout(150);
    expect(await view(tl)).toEqual(before);
  });

  test('dragging the Out handle moves the trim, dims the clip outside it and updates the duration readout', async ({ page }) => {
    await openNle(page, { prescan: false });
    const dur0 = await page.getByTestId('nle-region-duration').textContent();
    const handle = page.getByTestId('nle-trim-out');
    const hb = await handle.boundingBox();
    await page.mouse.move(hb.x + hb.width / 2, hb.y + hb.height / 2);
    await page.mouse.down();
    await page.mouse.move(hb.x - 300, hb.y + hb.height / 2, { steps: 8 });
    await page.mouse.up();
    await expect.poll(async () => await page.getByTestId('nle-region-duration').textContent(), { message: 'duration readout follows Out' }).not.toBe(dur0);
    await expect(page.getByTestId('nle-dim-after'), 'the clip after Out is dimmed').toBeVisible();
    const dim = await page.getByTestId('nle-dim-after').boundingBox();
    expect(dim.width).toBeGreaterThan(100);
  });
});

test.describe('NLE timeline: hygiene', () => {
  test('no page errors, no remote requests, and no canvas repaint on playhead moves', async ({ page }) => {
    const errors = [];
    const remote = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('request', (req) => {
      const url = new URL(req.url());
      if (['http:', 'https:', 'ws:', 'wss:'].includes(url.protocol) && !['localhost', '127.0.0.1', '::1'].includes(url.hostname)) remote.push(req.url());
    });
    await openNle(page);
    await page.waitForTimeout(800);
    // The waveform canvas is repainted only when its inputs change. Count real drawing calls on its
    // context while the playhead moves frame by frame.
    await page.evaluate(() => {
      const c = document.querySelector('[data-testid="nle-waveform-canvas"]');
      const ctx = c.getContext('2d');
      window.__wavePaints = 0;
      const orig = ctx.fillRect.bind(ctx);
      ctx.fillRect = (...a) => { window.__wavePaints++; return orig(...a); };
    });
    const step = page.getByTestId('transport-step-fwd');
    for (let i = 0; i < 12; i++) await step.click();
    expect(await page.evaluate(() => window.__wavePaints), 'stepping the playhead must not repaint the waveform').toBe(0);
    expect(errors).toEqual([]);
    expect(remote).toEqual([]);
  });
});
