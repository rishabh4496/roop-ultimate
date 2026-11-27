import { test, expect } from '@playwright/test';
import { VIEWS, openTab } from './helpers.js';

const WIDTHS = [390, 768, 1440];

async function openActiveProcessing(page) {
  const run = { processing: true, started_at: Date.now() / 1000 - 10, progress: 0.4,
    desc: 'Processing frame 40 / 100', current_frame: 40, total_frames: 100, fps_now: 12 };
  await page.route('**/api/progress', (route) => route.fulfill({ json: run }));
  await page.routeWebSocket(/\/ws\/telemetry/, (ws) => ws.send(JSON.stringify({ event: 'hello', ...run })));
  await openTab(page, 'processing');
  await expect(page.getByRole('progressbar', { name: 'Render progress' })).toBeVisible();
}

const VIEWS_WITH_WORKFLOWS = [
  ...VIEWS,
  { id: 'facemgr/morph', open: async (page) => {
    await openTab(page, 'facemgr');
    await page.getByRole('tab', { name: 'Morph Lab' }).click();
    await expect(page.getByRole('tabpanel', { name: 'Morph Lab' })).toBeVisible();
  } },
  { id: 'gallery/list', open: async (page) => {
    await openTab(page, 'gallery');
    await page.getByTitle('List View').click();
    await expect(page.getByRole('table', { name: 'Output files' })).toBeVisible();
  } },
  { id: 'settings/shortcuts', open: async (page) => {
    await openTab(page, 'settings');
    await page.getByRole('tab', { name: /Shortcuts/ }).click();
    await expect(page.getByRole('tabpanel', { name: /Shortcuts/ })).toBeVisible();
  } },
  { id: 'processing/active', open: openActiveProcessing },
  { id: 'faceswap/active', open: async (page) => {
    await openActiveProcessing(page);
    await page.getByRole('navigation', { name: 'Main' }).getByRole('button', { name: 'Face Swap' }).click();
    await expect(page.getByRole('button', { name: /View progress/ })).toBeVisible();
    await expect(page.getByRole('button', { name: /Render 5s Preview/ })).toBeVisible();
  } },
];

for (const width of WIDTHS) {
  for (const view of VIEWS_WITH_WORKFLOWS) {
    test(`${view.id} controls fit at ${width}px`, async ({ page }) => {
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await page.setViewportSize({ width, height: 900 });
      await view.open(page);

      const clipped = await page.evaluate(() => {
        const screenWidth = document.documentElement.clientWidth;
        const controls = document.querySelectorAll(':is(header, main) :is(button, summary, a[href], input, select, textarea, [role="separator"])');
        const overflows = [];
        for (const control of controls) {
          if (!control.checkVisibility({ opacityProperty: true, visibilityProperty: true })
              || control.closest('[inert],[aria-hidden="true"],.sr-only')) continue;
          const rect = control.getBoundingClientRect();
          if (!rect.width || !rect.height || (rect.left >= -1 && rect.right <= screenWidth + 1)) continue;

          // A toolbar or tab strip may scroll inside the page. Its controls can
          // extend past the viewport as long as its scrollport fits on screen.
          let scrollable = false;
          for (let el = control.parentElement; el && el.tagName !== 'MAIN'; el = el.parentElement) {
            if (!['auto', 'scroll'].includes(getComputedStyle(el).overflowX)) continue;
            const port = el.getBoundingClientRect();
            if (port.left >= -1 && port.right <= screenWidth + 1) { scrollable = true; break; }
          }
          if (!scrollable) overflows.push({
            label: control.getAttribute('aria-label') || control.getAttribute('title')
              || control.textContent.trim().slice(0, 70) || control.tagName,
            left: Math.round(rect.left), right: Math.round(rect.right),
          });
        }
        return overflows;
      });

      expect.soft(errors, 'uncaught page errors').toEqual([]);
      expect(clipped, 'controls outside the viewport, including overflow hidden by the shell').toEqual([]);
    });
  }
}

test('Face Swap preview controls can be used at phone width', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openTab(page, 'faceswap');
  const stage = page.locator('.preview-stage');
  const presets = stage.getByLabel('Viewport zoom presets');
  await presets.click();
  const lock = stage.getByTitle('Lock to Target Face Auto-Follow camera (T)');
  await lock.click();
  await expect(lock).toHaveAttribute('aria-pressed', 'true');
  await lock.press('Escape');
  await expect(presets).toBeFocused();
  await stage.getByRole('button', { name: /Compare Faces/ }).click();
  await expect(stage.getByText('Compare (SLIDER)')).toBeVisible();
  const deck = page.getByRole('button', { name: 'Toggle the timeline and logs deck' });
  await deck.click();
  await expect(deck).toHaveAttribute('aria-pressed', 'false');
});

test('Batch Matrix opens and imports a preset from the keyboard at tablet width', async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 900 });
  await openTab(page, 'batch');
  const importButton = page.getByRole('button', { name: 'Import Preset' });
  const chooser = page.waitForEvent('filechooser');
  await importButton.press('Enter');
  await (await chooser).setFiles({
    name: 'matrix.json', mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({ type: 'roop-batch-preset', batchMode: 'matrix' })),
  });
  await expect(page.getByRole('heading', { name: 'Strategy 3: Per-File Custom Mapping Matrix' })).toBeVisible();
  const quickSlot = page.getByRole('button', { name: 'Save Quick Slot #5' });
  await quickSlot.click();
  await expect(page.getByRole('button', { name: 'Slot 5', exact: true })).toHaveAttribute('title', 'Load Quick Slot #5');
});

test('Face Manager can switch between its panels with the arrow keys', async ({ page }) => {
  await openTab(page, 'facemgr');
  const tabs = page.getByRole('tablist', { name: 'Face Manager' });
  const builder = tabs.getByRole('tab').first();
  const morph = tabs.getByRole('tab', { name: 'Morph Lab' });
  await builder.press('ArrowRight');
  await expect(morph).toBeFocused({ timeout: 1500 });
  await expect(morph).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('tabpanel', { name: 'Morph Lab' })).toBeVisible();
  await morph.press('Home');
  await expect(builder).toBeFocused();
  await expect(builder).toHaveAttribute('aria-selected', 'true');
});

test('the shared Workspace Snapshots dialog traps focus and returns it on Escape', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 580 });
  await openTab(page, 'home');
  const opener = page.getByRole('button', { name: 'Session snapshots' });
  await opener.press('Enter');
  const dialog = page.getByRole('dialog', { name: 'Workspace Session Snapshots' });
  const close = dialog.getByRole('button', { name: 'Close workspace snapshots modal' });
  const save = dialog.getByRole('button', { name: /Save Current State/ });
  await expect(close).toBeFocused({ timeout: 1500 });
  await page.keyboard.press('Shift+Tab');
  await expect(save).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(opener).toBeFocused();
});
