// Stage 8: virtualization, containment and route loading.
//
//   #/gallery   2,000 outputs  -> a few dozen cards, ZERO <video> until hover, one while hovered, zero after
//   #/history   1,500 runs     -> a few dozen rows
//   #/facemgr   600 face crops -> a few dozen tiles, every one still reachable
//   #/processing a 10,000-line log -> a 2,000-line ring, ~30 mounted rows, instant severity filtering
//   routes      hovering a rail icon fetches that route's chunk before any click; Face Swap/Processing ship eager
//   containment `contain: layout style paint` on the preview, timeline and inspector groups, and the keyboard
//               focus ring is NOT clipped by it
//
// These run against the MOCK backend with large stubbed payloads (page.route): they test what the SHELL does with
// a big list, not how fast the real backend produces one. Headless Chromium cannot decode H.264, so a hovered
// <video> errors out; what is asserted is whether the element is MOUNTED, which is the point (a mounted video is a
// decoder).
import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { PNG_1X1, openTab } from './helpers.js';

// `contain: layout style paint` is serialized by getComputedStyle as the shorthand `content`.
const isContained = (c) => c === 'content' || ['layout', 'style', 'paint'].every((w) => c.includes(w));
const MAX_TILES_MOUNTED = 80;     // a 4-column grid, 72vh tall, overscan 3 rows is ~30; 80 leaves slack for wide windows

const stubJson = (page, pattern, make) => page.route(pattern, async (route) => {
  if (route.request().method() !== 'GET') { await route.continue(); return; }
  const real = await route.fetch();
  const body = await real.json().catch(() => ({}));
  await route.fulfill({ response: real, json: make(body) });
});

const outputs = (n) => Array.from({ length: n }, (_, i) => ({
  name: `clip_${String(i + 1).padStart(4, '0')}${i % 5 === 4 ? '.png' : '.mp4'}`,
  kind: i % 5 === 4 ? 'image' : 'video',
  mtime: 1_800_000_000 - i * 60,           // clip_0001 is the newest
  size: 1_000_000 + i,
}));

/** Scroll a virtualized scroller to its true end (its height keeps being re-measured as rows mount). */
async function scrollToEnd(scroller) {
  let last = -1;
  for (let i = 0; i < 40; i += 1) {
    const h = await scroller.evaluate((el) => { el.scrollTop = el.scrollHeight; return el.scrollHeight; });
    if (h === last) return;
    last = h;
    await scroller.page().waitForTimeout(60);
  }
}

test.describe('Outputs', () => {
  test.beforeEach(async ({ page }) => {
    await stubJson(page, '**/api/output', (b) => ({ ...b, files: outputs(2000) }));
  });

  test('2,000 files mount a few dozen cards and no video decoder', async ({ page }) => {
    await openTab(page, 'gallery');
    const grid = page.getByTestId('gallery-grid');
    await expect(grid).toBeVisible();
    const cards = await grid.locator('div.group\\/card').count();
    expect(cards, 'cards in the DOM').toBeGreaterThan(0);
    expect(cards, 'cards in the DOM').toBeLessThan(MAX_TILES_MOUNTED);
    expect(await page.locator('video').count(), 'a <video> per tile is a decoder per tile').toBe(0);
    test.info().annotations.push({ type: 'perf', description: `2000 files -> ${cards} cards, 0 videos mounted` });
  });

  test('a <video> mounts on pointer-enter and is gone, src and all, on pointer-leave', async ({ page }) => {
    await openTab(page, 'gallery');
    const surface = page.getByTestId('scrub-surface').first();
    await surface.scrollIntoViewIfNeeded();
    const b = await surface.boundingBox();
    expect(await page.getByTestId('hover-video').count()).toBe(0);

    await page.mouse.move(b.x + b.width * 0.5, b.y + b.height * 0.5);
    await expect(page.getByTestId('hover-video'), 'exactly one decoder while hovered').toHaveCount(1);
    expect(await page.locator('video').count()).toBe(1);

    // sweep across several cards: never more than the one under the pointer
    const peak = await page.evaluate(() => new Promise((resolve) => {
      let max = 0; let n = 0;
      const t = setInterval(() => { max = Math.max(max, document.querySelectorAll('video').length); if (++n > 20) { clearInterval(t); resolve(max); } }, 25);
    }));
    expect(peak).toBeLessThanOrEqual(1);

    await page.mouse.move(1, 1);
    await expect(page.getByTestId('hover-video')).toHaveCount(0);
    expect(await page.locator('video').count()).toBe(0);
  });

  test('every file is reachable: scrolling to the end shows the last one', async ({ page }) => {
    await openTab(page, 'gallery');
    const grid = page.getByTestId('gallery-grid');
    await scrollToEnd(grid);
    await expect(grid.getByTitle('clip_2000.png')).toBeVisible();
  });

  test('the same holds at 150% UI zoom (CSS zoom changes what getBoundingClientRect means)', async ({ page }) => {
    await page.addInitScript(() => { try { localStorage.setItem('roop_zoom', '1.5'); } catch { /* ignore */ } });
    await openTab(page, 'gallery');
    const grid = page.getByTestId('gallery-grid');
    expect(await page.evaluate(() => document.documentElement.style.zoom)).toBe('1.5');
    await scrollToEnd(grid);
    await expect(grid.getByTitle('clip_2000.png')).toBeVisible();
    expect(await grid.locator('div.group\\/card').count()).toBeLessThan(MAX_TILES_MOUNTED);
  });

  test('list view: windowed rows, reachable end, and a valid table structure', async ({ page }) => {
    await openTab(page, 'gallery');
    await page.getByRole('button', { name: /List/ }).click();
    const list = page.getByTestId('gallery-list');
    await expect(list).toBeVisible();
    const rows = await list.getByRole('row').count();
    expect(rows).toBeGreaterThan(0);
    expect(rows).toBeLessThan(60);
    await scrollToEnd(list);
    await expect(list.getByText('clip_2000.png', { exact: true })).toBeVisible();
    const { violations } = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'best-practice']).analyze();
    expect(violations.map((v) => `${v.id} x${v.nodes.length}`)).toEqual([]);
  });

  test('a first-frame still stands in for the video, and the capture queue is bounded', async ({ page }) => {
    await openTab(page, 'gallery');
    await page.waitForTimeout(1500);
    const stats = await page.evaluate(() => window.__roopVideoThumbs.__thumbStats());
    expect(stats.peakRunning, 'detached capture elements alive at once').toBeLessThanOrEqual(2);
    expect(await page.locator('video').count(), 'no capture element is left in the document').toBe(0);
  });
});

test.describe('History', () => {
  test('1,500 runs mount a few dozen rows; the oldest is reachable', async ({ page }) => {
    await stubJson(page, '**/api/history', (b) => ({
      ...b,
      entries: Array.from({ length: 1500 }, (_, i) => ({
        id: 1_700_000_000_000 + i, time: 1_800_000_000 - i * 600,
        outputs: [`run_${String(i + 1).padStart(4, '0')}.mp4`], settings: {}, duration_s: 30 + i % 7, frames: 180, fps: 7.4,
      })),
    }));
    await openTab(page, 'history');
    const list = page.getByTestId('history-list');
    await expect(list).toBeVisible();
    const mounted = Number(await list.getAttribute('data-virtual-mounted'));
    expect(mounted).toBeGreaterThan(0);
    expect(mounted).toBeLessThan(40);
    expect(await page.locator('video').count()).toBe(0);
    await scrollToEnd(list);
    await expect(list.getByText('run_1500.mp4', { exact: true })).toBeVisible();
  });
});

test.describe('Face Manager', () => {
  test('600 crops: a windowed grid in which every face can still be reached and selected', async ({ page }) => {
    const face = (i) => `data:image/svg+xml;utf8,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="hsl(${i % 360},40%,30%)"/><text x="6" y="36" fill="white" font-size="14">${i + 1}</text></svg>`)}`;
    await page.route('**/api/facemgr/add', (route) => route.fulfill({
      json: {
        faces: Array.from({ length: 600 }, (_, i) => face(i)),
        scores: Array.from({ length: 600 }, (_, i) => 0.3 + (i % 7) / 10),
        meta: Array.from({ length: 600 }, () => ({ face_px: 128, sharp_var: 200, frontality: 0.8 })),
      },
    }));
    await openTab(page, 'facemgr');
    await page.locator('input[type=file][accept="image/*,video/*"]').setInputFiles({ name: 'a.png', mimeType: 'image/png', buffer: PNG_1X1 });
    const grid = page.getByTestId('face-grid');
    await expect(grid).toBeVisible();
    const tiles = await grid.getByRole('button').count();
    expect(tiles, 'face tiles in the DOM').toBeGreaterThan(0);
    expect(tiles, 'face tiles in the DOM').toBeLessThan(MAX_TILES_MOUNTED);
    expect(Number(await grid.getAttribute('data-virtual-total'))).toBeLessThan(600);   // rows, not faces

    await scrollToEnd(grid);
    const last = grid.getByRole('button', { name: 'face 600' });
    await expect(last).toBeVisible();
    await last.click();
    await expect(last, 'the selection ring follows the click').toHaveClass(/ring-2/);
  });
});

test.describe('Processing terminal', () => {
  const LINES = 10_000;
  const MIX = [
    { msg: 'Loading models', category: 'PROCESSING', level: 'INFO' },
    { msg: 'warning: low VRAM headroom', category: 'WARNINGS', level: 'WARNING' },
    { msg: 'throughput 21.4 faces/s', category: 'PERFORMANCE', level: 'INFO' },
    { msg: 'frame 12 failed to swap', category: 'ERRORS', level: 'ERROR' },
  ];
  const log = Array.from({ length: LINES }, (_, i) => ({ t: '12:00:00', seq: i + 1, part: 0, event: 'terminal_line', ...MIX[i % 4], msg: `${MIX[i % 4].msg} #${i + 1}` }));

  test.beforeEach(async ({ page }) => {
    await stubJson(page, '**/api/progress', (b) => ({ ...b, log }));
    await openTab(page, 'processing');
  });

  test('a 10,000-line log is a 2,000-line ring and ~30 mounted rows', async ({ page }) => {
    const body = page.getByTestId('terminal-body');
    await expect(body).toHaveAttribute('data-ring-size', '2000');
    await expect(body).toHaveAttribute('data-shown', '2000');
    const mounted = Number(await body.getAttribute('data-mounted'));
    expect(mounted).toBeGreaterThan(0);
    expect(mounted).toBeLessThan(80);
    expect(await body.locator('[data-index]').count()).toBe(mounted);
    // following the tail: the newest line is the one in view
    await expect(body.getByText('#10000', { exact: false })).toBeVisible();
    await expect(page.getByTestId('terminal-follow')).toHaveText('Following');
  });

  test('severity chips filter instantly, with exact counts', async ({ page }) => {
    const body = page.getByTestId('terminal-body');
    // 2,000 newest lines of a 4-way repeating mix: 500 of each
    for (const [id, n] of [['sev-warn', 500], ['sev-error', 500], ['sev-perf', 500]]) {
      const t = await page.evaluate(async (testid) => {
        const el = document.querySelector(`[data-testid="${testid}"]`);
        const t0 = performance.now();
        el.click();
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        return performance.now() - t0;
      }, id);
      await expect(body).toHaveAttribute('data-shown', String(n));
      expect(await page.getByTestId(id).getAttribute('aria-pressed')).toBe('true');
      test.info().annotations.push({ type: 'perf', description: `${id}: click -> painted ${t.toFixed(1)} ms` });
      expect(t, `${id} click to two animation frames`).toBeLessThan(120);
      expect(Number(await body.getAttribute('data-mounted'))).toBeLessThan(80);
    }
    // each chip shows only its own kind
    await page.getByTestId('sev-error').click();
    await expect(body.getByText('failed to swap').first()).toBeVisible();
    await expect(body.getByText('Loading models')).toHaveCount(0);
    await page.getByTestId('sev-all').click();
    await expect(body).toHaveAttribute('data-shown', '2000');
  });

  test('scrolling up unlocks the tail; Follow (or reaching the bottom) locks it again', async ({ page }) => {
    const body = page.getByTestId('terminal-body');
    const follow = page.getByTestId('terminal-follow');
    await expect(follow).toHaveText('Following');
    await body.evaluate((el) => { el.scrollTop = Math.max(0, el.scrollHeight / 2); });
    await expect(follow).toHaveText('Paused');
    await expect(page.getByRole('button', { name: 'Scroll to latest output' })).toBeVisible();
    await follow.click();
    await expect(follow).toHaveText('Following');
    await expect(body.getByText('#10000')).toBeVisible();
  });
});

test.describe('Route loading', () => {
  // The idle sweep warms every chunk ~a frame after first paint; stop it so what is measured is the hover.
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => { window.requestIdleCallback = () => 0; window.cancelIdleCallback = () => {}; });
  });

  const LAZY = [
    ['Face Manager', /\/assets\/FaceManager-[^/]+\.js$/],
    ['Editor', /\/assets\/Extras-[^/]+\.js$/],
    ['Outputs', /\/assets\/Gallery-[^/]+\.js$/],
    ['History', /\/assets\/RunHistory-[^/]+\.js$/],
    ['Settings', /\/assets\/Settings-[^/]+\.js$/],
  ];

  test('hovering a rail icon fetches that route\'s chunk before any click', async ({ page }) => {
    const seen = [];
    page.on('request', (r) => { if (r.url().endsWith('.js')) seen.push(r.url()); });
    await openTab(page, 'faceswap');
    const nav = page.locator('nav[aria-label="Main"]');
    for (const [label, re] of LAZY) {
      expect(seen.some((u) => re.test(u)), `${label} chunk requested before it was hovered`).toBe(false);
      await nav.getByRole('button', { name: label, exact: true }).hover();
      await expect.poll(() => seen.some((u) => re.test(u)), { message: `${label} chunk after hover` }).toBe(true);
    }
    expect(await page.locator('nav[aria-label="Main"] button[aria-current="page"]').textContent()).toContain('Face Swap');
  });

  test('Face Swap and Processing are in the entry chunk: there is no chunk of their own to fetch', async ({ page }) => {
    const seen = [];
    page.on('request', (r) => { if (r.url().endsWith('.js')) seen.push(r.url()); });
    await openTab(page, 'faceswap');
    await page.goto('/#/processing');
    await page.locator('.deferred-fallback').waitFor({ state: 'detached' });
    expect(seen.filter((u) => /\/assets\/(FaceSwap|Processing)-[^/]+\.js$/.test(u))).toEqual([]);
    await expect(page.getByTestId('terminal-body')).toBeVisible();
  });
});

test.describe('Containment', () => {
  test('preview, timeline and every inspector group are layout/style/paint contained', async ({ page }) => {
    await openTab(page, 'faceswap');
    const contain = (loc) => loc.first().evaluate((el) => getComputedStyle(el).contain);
    // Every section of the inspector column: the quick-recipes bar and each accordion group.
    const sections = page.getByTestId('inspector').locator('section');
    const n = await sections.count();
    expect(n, 'inspector sections').toBeGreaterThan(1);
    for (let i = 0; i < n; i += 1) {
      const c = await sections.nth(i).evaluate((el) => getComputedStyle(el).contain);
      expect(isContained(c), `inspector section ${i}: contain: ${c}`).toBe(true);
    }
    const stage = page.locator('.preview-stage');
    if (await stage.count()) {       // no media loaded in this fixture -> no preview stage; the others still run
      const c = await contain(stage);
      expect(isContained(c), `preview: contain: ${c}`).toBe(true);
    }
    const tl = page.getByTestId('timeline');
    if (await tl.count()) expect(isContained(await contain(tl))).toBe(true);
  });

  test('paint containment does not clip the keyboard focus ring (and the check can tell when it does)', async ({ page }) => {
    await openTab(page, 'faceswap');
    // An accordion group (its header is the first button), not the recipes bar.
    const section = page.getByTestId('inspector').locator('section:has(> h2 > button[aria-expanded])').first();
    const header = section.locator('> h2 > button');
    const box = await section.boundingBox();
    // A strip just OUTSIDE the section's left edge, where an outline on a flush control would be drawn.
    const strip = { x: box.x - 5, y: box.y + 2, width: 4, height: Math.max(8, Math.min(box.height - 4, 30)) };

    await page.keyboard.press('Shift');                 // keyboard modality, so :focus-visible applies
    await header.blur();
    const before = await page.screenshot({ clip: strip });
    await header.focus();
    const focused = await page.screenshot({ clip: strip });
    expect(Buffer.compare(before, focused), 'the focus ring is painted outside the section edge').not.toBe(0);

    // Negative control: take the clip margin away and the same measurement must see NOTHING drawn out there.
    await section.evaluate((el) => { el.style.overflowClipMargin = '0px'; });
    await header.blur();
    const before0 = await page.screenshot({ clip: strip });
    await header.focus();
    const focused0 = await page.screenshot({ clip: strip });
    expect(Buffer.compare(before0, focused0), 'without overflow-clip-margin the ring IS clipped (so the test above means something)').toBe(0);
  });
});
