import { test, expect } from '@playwright/test';
import { openTab, TABS, makePng } from './helpers.js';

for (const [width, height] of [[1920, 1080], [3840, 2160], [3440, 1440], [5120, 1440]]) {
  test(`all nine routes fill ${width}x${height} without document scrolling`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await openTab(page, 'home');
    for (const { id } of TABS) {
      await page.locator('.studio-routes').getByRole('link', { name: TABS.find((t) => t.id === id).label, exact: true }).click();
      await expect(page.locator('main')).toHaveAttribute('data-route', id);
      await expect(page.locator('.studio-route')).toBeVisible();
      const box = await page.evaluate(() => {
        const rect = (s) => { const r = document.querySelector(s).getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; };
        return { shell: rect('.studio-shell'), header: rect('.studio-header'), main: rect('main'), scrollW: document.documentElement.scrollWidth, scrollH: document.documentElement.scrollHeight };
      });
      expect(box.shell).toEqual({ x: 0, y: 0, w: width, h: height });
      expect(box.header).toEqual({ x: 0, y: 0, w: width, h: 38 });
      expect(box.main.x + box.main.w).toBe(width);
      expect(box.main.y + box.main.h).toBe(height);
      expect(box.scrollW).toBe(width);
      expect(box.scrollH).toBe(height);
    }
    await page.reload();
    await expect(page.locator('main')).toHaveAttribute('data-route', 'settings');
  });
}

test('pane resizing, reset, collapse, presets and reload persist real geometry', async ({ page }) => {
  await page.setViewportSize({ width: 1920, height: 1080 });
  await openTab(page, 'faceswap');
  const left = page.getByRole('region', { name: 'Assets / Facesets', exact: true });
  const right = page.getByRole('region', { name: 'Parameters / Diagnostics', exact: true });
  await page.getByRole('button', { name: 'Standard NLE', exact: true }).click();
  expect((await left.boundingBox()).width).toBe(260);
  expect((await right.boundingBox()).width).toBe(340);
  const divider = page.getByRole('separator', { name: 'Resize Assets panel', exact: true });
  const r = await divider.boundingBox();
  await page.mouse.move(r.x, r.y + 80);
  await page.mouse.down(); await page.mouse.move(r.x + 80, r.y + 80); await page.mouse.up();
  expect((await left.boundingBox()).width).toBe(340);
  await page.reload(); await expect(left).toBeVisible();
  expect((await left.boundingBox()).width).toBe(340);
  await divider.dblclick({ force: true });
  expect((await left.boundingBox()).width).toBe(260);
  await divider.focus(); await page.keyboard.press('ArrowRight');
  expect((await left.boundingBox()).width).toBe(270);
  await page.getByRole('button', { name: 'Collapse Assets / Facesets', exact: true }).click();
  expect((await left.boundingBox()).width).toBe(32);
  await page.getByRole('button', { name: 'Expand Assets / Facesets', exact: true }).click();
  expect((await left.boundingBox()).width).toBe(270);
  await page.getByRole('button', { name: 'Collapse Assets / Facesets', exact: true }).click();
  await page.getByRole('button', { name: 'Standard NLE', exact: true }).click();
  expect((await left.boundingBox()).width).toBe(260);
  await page.getByRole('button', { name: 'Cinema Preview', exact: true }).click();
  expect((await left.boundingBox()).width).toBe(32);
  expect((await right.boundingBox()).width).toBe(32);
  await expect(page.getByRole('region', { name: 'Timeline', exact: true })).toBeVisible();
  await page.setViewportSize({ width: 3440, height: 1440 });
  await page.getByRole('button', { name: 'Ultra-Wide Dual Stage', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Live Queue', exact: true })).toBeVisible();
  await expect(page.getByTestId('workspace-split')).toHaveAttribute('data-layout', 'dual');
  await page.screenshot({ path: 'e2e/test-results/stage2-ultrawide.png' });
});

test('wide first launch selects dual stage and browser history preserves routes', async ({ page }) => {
  await page.setViewportSize({ width: 3440, height: 1440 });
  await openTab(page, 'faceswap');
  await expect(page.getByTestId('workspace-split')).toHaveAttribute('data-layout', 'dual');
  await page.locator('.studio-routes a[href="#/gallery"]').click();
  await expect(page.locator('main')).toHaveAttribute('data-route', 'gallery');
  await page.goBack();
  await expect(page.locator('main')).toHaveAttribute('data-route', 'faceswap');
});

test('dual stage decodes the source image, paints the preview canvas and scrolls only its panes', async ({ page }) => {
  await page.setViewportSize({ width: 3440, height: 1440 });
  // The shared mock serves SVG thumbnails. Use an actual raster frame to drive
  // the production worker/bitmap/canvas path, then read its painted pixels.
  const png = makePng(1280, 720, () => [22, 125, 183]);
  await page.route('**/api/target/preview?**', (r) => r.fulfill({ contentType: 'image/png', body: png }));
  await openTab(page, 'faceswap');
  const canvas = page.locator('.workspace-ab .preview-stage canvas').first();
  await expect.poll(() => canvas.evaluate((c) => {
    const ctx = c.getContext('2d'); const p = ctx.getImageData(c.width / 2, c.height / 2, 1, 1).data;
    return [p[0], p[1], p[2], p[3]];
  })).toEqual([22, 125, 183, 255]);
  await expect(page.locator('.workspace-ab > figure img')).toBeVisible();
  await expect.poll(() => page.locator('.workspace-ab > figure img').evaluate((img) => img.complete && img.naturalWidth === 1280)).toBe(true);
  const dimensions = await page.getByTestId('stage-deck').evaluate((el) => ({ height: el.clientHeight, scrollHeight: el.scrollHeight }));
  expect(dimensions.scrollHeight).toBeLessThanOrEqual(dimensions.height + 1);
  const inspector = page.locator('.workspace-right .workspace-pane-scroll');
  expect(await inspector.evaluate((el) => getComputedStyle(el, '::-webkit-scrollbar').width)).toBe('4px');
  await inspector.evaluate((el) => el.scrollTo(0, 500));
  expect(await inspector.evaluate((el) => el.scrollTop)).toBeGreaterThan(0);
  expect(await page.evaluate(() => window.scrollY)).toBe(0);
  await page.screenshot({ path: 'e2e/test-results/stage2-dual-raster.png' });
});

test('checkpoint selection restores the saved recipe and the dirty indicator reads actual edits', async ({ page }) => {
  const settings = await (await page.request.get('/api/settings')).json();
  await page.addInitScript((recipe) => {
    localStorage.setItem('roop_session_snapshots', JSON.stringify([{ id: 42, name: 'Review checkpoint', settings: recipe, tab: 'settings' }]));
    localStorage.setItem('roop_checkpoint', '42');
  }, settings);
  // Keep this test's recipe edits local to its browser, independent of later tests.
  await page.route('**/api/settings', (route) => route.request().method() === 'POST'
    ? route.fulfill({ json: { status: 'ok' } }) : route.continue());
  await openTab(page, 'settings');
  const checkpoint = page.getByRole('combobox', { name: 'Project checkpoint', exact: true });
  await expect(checkpoint).toHaveValue('42');
  await expect(page.getByRole('img', { name: 'Checkpoint saved', exact: true })).toBeVisible();
  const provider = page.locator('[data-setting="provider"] select');
  const before = await provider.inputValue();
  await provider.selectOption(before === 'cpu' ? 'cuda' : 'cpu');
  await expect(page.getByRole('img', { name: 'Unsaved checkpoint changes', exact: true })).toBeVisible();
  await checkpoint.selectOption('42');
  await expect(provider).toHaveValue(before);
  await expect(page.getByRole('img', { name: 'Checkpoint saved', exact: true })).toBeVisible();
});
