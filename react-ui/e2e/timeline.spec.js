import { test, expect } from '@playwright/test';
import { openTab } from './helpers.js';

// The multi-track timeline, in a real Chromium against the production build with the mock server
// standing in for app/api.py.
//
// What each group proves, because each is a place the timeline can look right and be wrong:
//   lanes      every lane has PIXELS from the data the mock sent (cut colour, a person's colour, the
//              cache-strip states, the waveform) - a lane that is mounted but never painted, or
//              painted from an empty array, fails here;
//   ruler      its label mode really switches (seconds -> timecode -> exact frame numbers) as you zoom;
//   gestures   Ctrl+wheel zooms the track about the cursor and does NOT also zoom the app; Shift+wheel,
//              a middle-button drag and the minimap handles all move the window;
//   keys       J/K/L shuttle, I/O, comma/period, Up/Down jump to cuts - each read back from the UI;
//   overrides  right-click -> "+ Add Scene Override" creates one, it persists, and it appears as a pill;
//   scale      100 frames and 50,000 frames: wheel-zoom and scrub with no main-thread block.
//
// Geometry of the mock clip (react-ui/mock-server/server.ts): 30 fps, a cut every 210 frames starting at
// 211, three people, a face-free stretch at 40-50 % of the clip and a smart-skip stretch at 60-70 %.

const FRAMES = 1200;
const TL = '[data-testid="timeline"]';

test.beforeEach(async ({ page }) => {
  // Long-task + rAF accounting for the scale tests (same approach as frame-budget.spec.js).
  await page.addInitScript(() => {
    const perf = { long: [], ts: [], on: false };
    window.__tlperf = perf;
    const tick = (t) => { if (perf.on) perf.ts.push(t); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
    try {
      new PerformanceObserver((list) => { if (perf.on) for (const e of list.getEntries()) perf.long.push(e.duration); })
        .observe({ entryTypes: ['longtask'] });
    } catch { /* unsupported: the count stays 0 */ }
  });
});

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

/** Number of pixels in a canvas within `tol` of an RGB colour (alpha ignored below 40). */
const colorCount = (canvas, rgb, tol = 48) => canvas.evaluate((c, [r, g, b, t]) => {
  const ctx = c.getContext('2d');
  const { data } = ctx.getImageData(0, 0, c.width, c.height);
  let n = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] > 40 && Math.abs(data[i] - r) < t && Math.abs(data[i + 1] - g) < t && Math.abs(data[i + 2] - b) < t) n++;
  }
  return n;
}, [...rgb, tol]);

const view = async (tl) => ({
  start: Number(await tl.getAttribute('data-view-start')),
  end: Number(await tl.getAttribute('data-view-end')),
});
const span = (v) => v.end - v.start;

test.describe('multi-track timeline (1,200-frame clip)', () => {
  test.beforeEach(async ({ request }) => { await resizeClip(request, FRAMES); });
  test.afterEach(async ({ request }) => { await resizeClip(request, 180); });

  test('every lane is painted from real data', async ({ page }) => {
    const tl = await openTimeline(page);
    for (const id of ['timeline-minimap', 'lane-ruler', 'lane-cache', 'lane-film', 'lane-scenes', 'lane-overrides']) {
      await expect(page.getByTestId(id), `${id} must be on screen`).toBeVisible();
    }
    const scenes = page.getByTestId('lane-scenes');
    const cache = page.getByTestId('lane-cache');
    const film = page.getByTestId('lane-film');
    // scene cuts: the cut colour (#fb7a3c)
    await expect.poll(() => colorCount(scenes, [251, 122, 60]), { message: 'scene-cut flags drawn' }).toBeGreaterThan(20);
    // person 1 (#38bdf8) and person 2 (#f472b6): two different people, two different colours
    await expect.poll(() => colorCount(scenes, [56, 189, 248]), { message: 'person 1 pills' }).toBeGreaterThan(200);
    await expect.poll(() => colorCount(scenes, [244, 114, 182]), { message: 'person 2 pills' }).toBeGreaterThan(50);
    // cache strip: dirty (amber), face-free bypass (purple) and smart-skip reuse (cyan) from the mock's render map
    await expect.poll(() => colorCount(cache, [245, 158, 11], 70), { message: 'dirty / unrendered' }).toBeGreaterThan(50);
    await expect.poll(() => colorCount(cache, [168, 85, 247]), { message: 'face-free bypass' }).toBeGreaterThan(20);
    await expect.poll(() => colorCount(cache, [34, 211, 238]), { message: 'smart-skip reuse' }).toBeGreaterThan(20);
    // waveform over the filmstrip (#7dd3fc at ~0.6 alpha over a dark scrim)
    await expect.poll(() => colorCount(film, [90, 150, 180], 60), { message: 'audio waveform' }).toBeGreaterThan(100);
    // the legend names every state the strip can show
    const legend = page.getByTestId('timeline-legend');
    for (const word of ['Rendered', 'Smart-skip reuse', 'Face-free bypass', 'Dirty']) await expect(legend).toContainText(word);
    await expect(tl).toHaveAttribute('data-zoomed', 'false');
  });

  test('the ruler switches label mode as you zoom, down to exact frame numbers', async ({ page }) => {
    const tl = await openTimeline(page);
    const ruler = page.getByTestId('lane-ruler');
    await expect(ruler).toHaveAttribute('data-mode', 'seconds');
    const slider = page.getByTestId('timeline-zoom');
    await slider.fill('0.6');
    await expect.poll(async () => span(await view(tl))).toBeLessThan(FRAMES / 2);
    await slider.fill('1');
    await expect(ruler).toHaveAttribute('data-mode', 'frames');
    // frame-by-frame: about one frame per still's width, never fewer than 4 frames on screen
    const v = await view(tl);
    expect(span(v)).toBeGreaterThanOrEqual(4);
    expect(span(v)).toBeLessThanOrEqual(14);
  });

  test('Ctrl+wheel zooms the track about the cursor and not the app', async ({ page }) => {
    const tl = await openTimeline(page);
    const film = page.locator('div.cursor-ew-resize').first();
    const box = await film.boundingBox();
    const appZoomBefore = await page.evaluate(() => document.documentElement.style.zoom);
    await page.mouse.move(box.x + box.width * 0.75, box.y + box.height / 2);
    // the frame under the cursor before and after: it must not move
    const frameAt = (v, ratio) => v.start + ratio * span(v);
    const before = await view(tl);
    const anchor = frameAt(before, 0.75);
    await page.keyboard.down('Control');
    for (let i = 0; i < 6; i++) await page.mouse.wheel(0, -120);
    await page.keyboard.up('Control');
    await expect.poll(async () => span(await view(tl))).toBeLessThan(span(before) * 0.6);
    const after = await view(tl);
    expect(Math.abs(frameAt(after, 0.75) - anchor), 'the frame under the cursor stays under the cursor').toBeLessThanOrEqual(3);
    expect(await page.evaluate(() => document.documentElement.style.zoom), 'the app UI zoom must not change').toBe(appZoomBefore);
    // ... and wheeling back out returns the whole clip
    await page.keyboard.down('Control');
    for (let i = 0; i < 30; i++) await page.mouse.wheel(0, 120);
    await page.keyboard.up('Control');
    await expect.poll(async () => span(await view(tl))).toBeGreaterThanOrEqual(FRAMES - 2);
  });

  test('Shift+wheel and a middle-button drag scroll the zoomed window', async ({ page }) => {
    const tl = await openTimeline(page);
    await page.getByTestId('timeline-zoom').fill('0.55');
    await expect(tl).toHaveAttribute('data-zoomed', 'true');
    const film = page.locator('div.cursor-ew-resize').first();
    const box = await film.boundingBox();
    const v0 = await view(tl);
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.keyboard.down('Shift');
    await page.mouse.wheel(0, 240);
    await page.keyboard.up('Shift');
    await expect.poll(async () => (await view(tl)).start).toBeGreaterThan(v0.start);
    const v1 = await view(tl);
    expect(span(v1), 'scrolling keeps the zoom').toBe(span(v0));
    // middle-button drag left drags the content, so the window moves right
    await page.mouse.move(box.x + box.width * 0.6, box.y + box.height / 2);
    await page.mouse.down({ button: 'middle' });
    await page.mouse.move(box.x + box.width * 0.3, box.y + box.height / 2, { steps: 8 });
    await page.mouse.up({ button: 'middle' });
    const v2 = await view(tl);
    expect(v2.start, 'a middle-button drag scrolls').not.toBe(v1.start);
    expect(span(v2)).toBe(span(v0));
  });

  test('the minimap is always on screen and its handles zoom, its window scrolls', async ({ page }) => {
    const tl = await openTimeline(page);
    const map = page.getByTestId('timeline-minimap');
    await expect(map, 'the minimap shows even at full zoom-out').toBeVisible();
    const mb = await map.boundingBox();
    // full zoom-out: the window fills the bar
    const win = page.getByTestId('timeline-minimap-window');
    expect((await win.boundingBox()).width).toBeGreaterThan(mb.width * 0.95);
    // drag the left handle inwards: zoom
    const hl = await page.getByTestId('timeline-minimap-handle-left').boundingBox();
    await page.mouse.move(hl.x + hl.width / 2, hl.y + hl.height / 2);
    await page.mouse.down();
    await page.mouse.move(mb.x + mb.width * 0.4, hl.y + hl.height / 2, { steps: 6 });
    await page.mouse.up();
    const zoomed = await view(tl);
    expect(zoomed.start).toBeGreaterThan(FRAMES * 0.3);
    expect(zoomed.end).toBe(FRAMES);
    // drag the window left: scroll, width unchanged
    const w = await win.boundingBox();
    await page.mouse.move(w.x + w.width / 2, w.y + w.height / 2);
    await page.mouse.down();
    await page.mouse.move(w.x + w.width / 2 - mb.width * 0.25, w.y + w.height / 2, { steps: 6 });
    await page.mouse.up();
    const scrolled = await view(tl);
    expect(scrolled.start).toBeLessThan(zoomed.start);
    expect(span(scrolled)).toBe(span(zoomed));
    // click elsewhere on the bar: centre the window there
    await page.mouse.click(mb.x + mb.width * 0.9, mb.y + mb.height / 2);
    const jumped = await view(tl);
    expect(jumped.start).toBeGreaterThan(scrolled.start);
  });

  test('J / K / L shuttle, I / O, comma / period and Up / Down jump to cuts', async ({ page }) => {
    await openTimeline(page);
    const frameBox = page.getByLabel('Frame number');
    const readout = page.getByTestId('shuttle-readout');
    const setFrame = async (n) => { await frameBox.fill(String(n)); await frameBox.press('Enter'); await page.mouse.click(2, 2); };

    await setFrame(600);
    await expect(frameBox).toHaveValue('600');

    // . and , step one frame; Shift steps ten
    await page.keyboard.press('.');
    await expect(frameBox).toHaveValue('601');
    await page.keyboard.press(',');
    await expect(frameBox).toHaveValue('600');
    await page.keyboard.press('Shift+.');
    await expect(frameBox).toHaveValue('610');
    await page.keyboard.press('Shift+,');
    await expect(frameBox).toHaveValue('600');

    // I / O set In and Out at the playhead (read back through the header fields)
    await setFrame(100);
    await page.keyboard.press('i');
    await expect(page.locator('label[title^="In point"] input')).toHaveValue('100');
    await setFrame(900);
    await page.keyboard.press('o');
    await expect(page.locator('label[title^="Out point"] input')).toHaveValue('900');

    // Up / Down: previous / next scene cut (cuts at 211, 421, 631, ...)
    await setFrame(500);
    await page.keyboard.press('ArrowDown');
    await expect(frameBox).toHaveValue('631');
    await page.keyboard.press('ArrowUp');
    await expect(frameBox).toHaveValue('421');
    await page.keyboard.press('ArrowUp');
    await expect(frameBox).toHaveValue('211');

    // shuttle ladder: L 1x -> 2x -> 4x, K pauses
    await expect(readout).toHaveCount(0);
    await page.keyboard.press('l');
    await expect(readout).toHaveText('▶ 1×');
    await page.keyboard.press('l');
    await expect(readout).toHaveText('▶ 2×');
    await page.keyboard.press('l');
    await expect(readout).toHaveText('▶ 4×');
    await page.keyboard.press('k');
    await expect(readout).toHaveCount(0);

    // J runs the playhead backwards on a clock
    await setFrame(800);
    await page.keyboard.press('j');
    await expect(readout).toHaveText('◀ 1×');
    await expect.poll(async () => Number(await frameBox.inputValue()), { timeout: 4000 }).toBeLessThan(790);
    await page.keyboard.press('j');
    await expect(readout).toHaveText('◀ 2×');
    await page.keyboard.press('k');
    await expect(readout).toHaveCount(0);
    const stopped = Number(await frameBox.inputValue());
    await page.waitForTimeout(400);
    expect(Number(await frameBox.inputValue()), 'K really stops the reverse clock').toBe(stopped);
  });

  test('keys are ignored while typing in a field', async ({ page }) => {
    await openTimeline(page);
    const frameBox = page.getByLabel('Frame number');
    await frameBox.click();
    await frameBox.press('l');
    await expect(page.getByTestId('shuttle-readout')).toHaveCount(0);
  });

  test('right-click adds a scene override, it persists, and its fields are editable', async ({ page }) => {
    await openTimeline(page);
    const scenes = page.getByTestId('lane-scenes');
    const sb = await scenes.boundingBox();
    // third shot (frames 421-630 of 1200): ~42 % across
    await page.mouse.click(sb.x + sb.width * 0.42, sb.y + sb.height - 3, { button: 'right' });
    const menu = page.getByTestId('timeline-menu');
    await expect(menu).toBeVisible();
    const add = menu.getByRole('menuitem', { name: /\+ Add Scene Override.*Scene 3.*421-630/ });
    await expect(add).toBeVisible();
    await add.click();
    await expect(menu).toBeHidden();

    const inspector = page.getByTestId('override-inspector');
    await expect(inspector).toBeVisible();
    await expect(inspector.getByLabel('From')).toHaveValue('421');
    await expect(inspector.getByLabel('To')).toHaveValue('630');

    // opt in to the feather field and move it
    await inspector.getByLabel('Override Mask feather').check();
    await inspector.getByLabel('Mask feather', { exact: true }).fill('64');
    // the override lane has pixels now (amber pill)
    const lane = page.getByTestId('lane-overrides');
    await expect.poll(() => colorCount(lane, [217, 119, 6], 60), { message: 'override pill drawn' }).toBeGreaterThan(200);

    // it survives a reload (per-clip localStorage)
    const stored = await page.evaluate(() => Object.entries(localStorage).filter(([k]) => k.startsWith('roop_overrides_') && !k.includes('enabled')));
    expect(stored.length).toBe(1);
    const list = JSON.parse(stored[0][1]);
    expect(list[0]).toMatchObject({ start: 421, end: 630 });
    expect(list[0].params.mask_feather).toBe(64);
    await page.reload();
    await page.locator(TL).waitFor({ state: 'visible' });
    await page.getByTestId('lane-overrides').scrollIntoViewIfNeeded();
    await expect.poll(() => colorCount(page.getByTestId('lane-overrides'), [217, 119, 6], 60)).toBeGreaterThan(200);

    // remove it from the inspector
    await page.getByTestId('lane-overrides').click({ position: { x: (await page.getByTestId('lane-overrides').boundingBox()).width * 0.42, y: 12 } });
    await expect(page.getByTestId('override-inspector')).toBeVisible();
    await page.getByTestId('override-inspector').getByRole('button', { name: 'Remove' }).click();
    await expect(page.getByTestId('override-inspector')).toBeHidden();
  });

  test('the bypass switch is withheld unless the detection mode can honour it', async ({ page }) => {
    await openTimeline(page);
    const sb = await page.getByTestId('lane-scenes').boundingBox();
    await page.mouse.click(sb.x + sb.width * 0.2, sb.y + sb.height - 3, { button: 'right' });
    await page.getByTestId('timeline-menu').getByRole('menuitem', { name: /Add Scene Override/ }).first().click();
    const bypass = page.getByTestId('override-inspector').getByLabel('Bypass swap for this range');
    // The mock's default detection mode is not "Selected face", so bypass must say why it is off.
    const mode = await page.evaluate(() => { try { return JSON.parse(localStorage.getItem('roop_settings') || '{}').face_detection_mode; } catch { return null; } });
    test.info().annotations.push({ type: 'note', description: `detection mode seen: ${mode}` });
    if (await bypass.isDisabled()) {
      await expect(page.getByTestId('override-inspector')).toContainText(/needs Selected-face/i);
    } else {
      await bypass.check();
      await expect(page.getByTestId('override-inspector').getByLabel('Override Mask feather')).toBeDisabled();
    }
  });

  test('clicking a presence pill sets In/Out to that stretch', async ({ page }) => {
    await openTimeline(page);
    const scenes = page.getByTestId('lane-scenes');
    const sb = await scenes.boundingBox();
    // Person 1 (row 0) is in every shot; click inside shot 2 (frames 211-420): ~25 % across, row 0 = y 16..29
    await page.mouse.click(sb.x + sb.width * 0.25, sb.y + 16 + 6);
    await expect(page.locator('label[title^="In point"] input')).toHaveValue('211');
    await expect(page.locator('label[title^="Out point"] input')).toHaveValue('420');
    await expect(page.getByLabel('Frame number')).toHaveValue('211');
  });

  test('Render In→Out is offered and disabled for the reason Start is', async ({ page }) => {
    await openTimeline(page);
    const btn = page.getByTestId('render-range');
    await expect(btn).toBeVisible();
    await expect(btn).toHaveText(/Render In→Out Range Only/);
  });
});

// ── scale: 100 frames and 50,000 frames ─────────────────────────────────────

async function windowStats(page, during) {
  await page.evaluate(() => { const p = window.__tlperf; p.long = []; p.ts = []; p.on = true; });
  const t0 = Date.now();
  await during();
  const secs = (Date.now() - t0) / 1000;
  const raw = await page.evaluate(() => { const p = window.__tlperf; p.on = false; return { long: p.long, ts: p.ts }; });
  const d = raw.ts.slice(1).map((t, i) => t - raw.ts[i]);
  return {
    secs,
    longTasks: raw.long.length,
    longMax: Math.max(0, ...raw.long),
    frameMax: Math.max(0, ...d),
    over50: d.filter((x) => x > 50).length,
    frames: d.length,
  };
}

for (const frames of [100, 50_000]) {
  test(`scale: wheel-zoom and scrub a ${frames.toLocaleString()}-frame clip without blocking the main thread`, async ({ page, request }) => {
    await resizeClip(request, frames);
    try {
      const tl = await openTimeline(page);
      const film = page.locator('div.cursor-ew-resize').first();
      await film.scrollIntoViewIfNeeded();
      const box = await film.boundingBox();
      const frameBox = page.getByLabel('Frame number');

      const zoom = await windowStats(page, async () => {
        await page.mouse.move(box.x + box.width * 0.5, box.y + box.height / 2);
        await page.keyboard.down('Control');
        for (let i = 0; i < 40; i++) await page.mouse.wheel(0, -100);
        for (let i = 0; i < 40; i++) await page.mouse.wheel(0, 100);
        await page.keyboard.up('Control');
      });
      const afterZoom = await view(tl);
      expect(span(afterZoom), 'the zoom ran: it came back out to the whole clip').toBeGreaterThanOrEqual(frames - 2);

      const before = await frameBox.inputValue();
      const scrub = await windowStats(page, async () => {
        await page.mouse.move(box.x + box.width * 0.1, box.y + box.height / 2);
        await page.mouse.down();
        for (let i = 0; i <= 200; i++) await page.mouse.move(box.x + box.width * (0.1 + 0.8 * (i / 200)), box.y + box.height / 2);
        await page.mouse.up();
      });
      expect(await frameBox.inputValue(), 'the scrub moved the playhead').not.toBe(before);

      // zoom to the floor on a long clip, then pan across the whole thing with the minimap
      await page.getByTestId('timeline-zoom').fill('1');
      const map = await page.getByTestId('timeline-minimap').boundingBox();
      const pan = await windowStats(page, async () => {
        await page.mouse.move(map.x + 4, map.y + map.height / 2);
        await page.mouse.down();
        for (let i = 0; i <= 120; i++) await page.mouse.move(map.x + 4 + (map.width - 8) * (i / 120), map.y + map.height / 2);
        await page.mouse.up();
      });

      for (const [name, s] of [['wheel zoom', zoom], ['scrub', scrub], ['minimap pan', pan]]) {
        // eslint-disable-next-line no-console
        console.log(`[timeline ${frames}f] ${name.padEnd(12)} ${s.secs.toFixed(1)}s | long tasks ${s.longTasks} (worst ${s.longMax.toFixed(0)} ms) | worst frame ${s.frameMax.toFixed(0)} ms | frames >50 ms: ${s.over50}/${s.frames}`);
        test.info().annotations.push({ type: 'perf', description: JSON.stringify({ frames, name, ...s }) });
        expect(s.longTasks, `${name}: main-thread blocks of 50 ms or more`).toBe(0);
      }
    } finally {
      await resizeClip(request, 180);
    }
  });
}
