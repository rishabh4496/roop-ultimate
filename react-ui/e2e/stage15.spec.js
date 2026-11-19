import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { openTab, makePng } from './helpers.js';

// Stage 15, in a real Chromium against the production build with the mock server standing in for app/api.py:
//
//   expression pill   typing `*2` / `/2` / `min(v*10, 0.3)` into a numeric field shows `Formula -> Result` BEFORE Enter, and
//                     Enter stores exactly what the pill said; a mistake gets an error pill, a plain number gets none
//   layouts           save / rename / hotkey-bind (Alt+1..5) / switch split-pane proportions at 16:9, 21:9 and 32:9 window
//                     shapes, with no page overflow and a centre stage that keeps its room
//   remapper          the ANSI keyboard lights keys by family, shows modifier layers, flags a clash, exports and imports a
//                     profile; the palette opens it
//   floating preview  on Processing too, with the A/B wipe (before | after, drag + keys) and Play/Pause
//
// What these do NOT prove: a render's LIVE frames in the window (the mock publishes none - that path is covered by the node
// checks in .render-check/render-stage15-logic.mjs and has not been seen against a real render), and anything about GPU cost.

const ERODE = 'Face detection threshold';
const field = (page, label) => page.locator('label.block').filter({ has: page.getByText(label, { exact: true }) }).first();
const bar = (page, label) => field(page, label).getByTestId('scrubber-bar');
const value = async (page, label) => Number((await field(page, label).getByTestId('scrubber-value').textContent()).replace(/[^0-9.\-]/g, ''));

// ── expression pill ──────────────────────────────────────────────────────────
test.describe('expression preview pill', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/settings', (route) => (route.request().method() === 'POST'
      ? route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }) : route.fallback()));
    await openTab(page, 'faceswap');
    await field(page, ERODE).scrollIntoViewIfNeeded();
  });

  const typeInto = async (page, text) => {
    await bar(page, ERODE).dblclick();
    const input = field(page, ERODE).getByLabel(`${ERODE} value`);
    await input.fill(text);
    return input;
  };
  const pill = (page) => field(page, ERODE).getByTestId('expr-preview');

  test('shows Formula -> Result before Enter, and Enter stores what it said', async ({ page }) => {
    expect(await value(page, ERODE)).toBe(0.5);
    const input = await typeInto(page, '/2');
    await expect(pill(page)).toBeVisible();
    await expect(pill(page)).toContainText('/2');
    await expect(pill(page)).toContainText('→');
    await expect(pill(page)).toContainText('0.25');
    expect(await field(page, ERODE).getByRole('slider').inputValue(), 'nothing is committed while typing').toBe('0.5');
    await input.press('Enter');
    await expect(pill(page), 'the pill goes with the editor').toHaveCount(0);
    expect(await value(page, ERODE)).toBe(0.25);
  });

  test('relative forms and min()/max() each preview and commit the same number', async ({ page }) => {
    const cases = [['*0.5', 0.25], ['+0.05', 0.3], ['-10%', 0.27], ['min(v*10, 0.4)', 0.4], ['max(v/2, 0.3)', 0.3]];
    for (const [text, want] of cases) {
      const before = await value(page, ERODE);
      const input = await typeInto(page, text);
      await expect(pill(page), `${text}: pill`).toContainText(String(want));
      await input.press('Enter');
      const got = await value(page, ERODE);
      expect(Math.abs(got - want), `${text} on ${before}: wanted ${want}, got ${got}`).toBeLessThan(0.0051);
    }
  });

  test('a mistake is an error pill; a plain number has none; Escape leaves the value alone', async ({ page }) => {
    const input = await typeInto(page, 'min(1,');
    await expect(pill(page)).toHaveAttribute('data-state', 'error');
    await input.fill('0.4');
    await expect(pill(page), 'a plain number would only repeat the field').toHaveCount(0);
    await input.fill('*3');
    await expect(pill(page)).toBeVisible();
    await input.press('Escape');
    await expect(pill(page)).toHaveCount(0);
    expect(await value(page, ERODE)).toBe(0.5);
  });

  test('the pill is announced (status) and axe has nothing to say about it', async ({ page }) => {
    await typeInto(page, '*2');
    await expect(pill(page)).toHaveAttribute('role', 'status');
    const res = await new AxeBuilder({ page }).include('[data-testid="expr-preview"]').analyze();
    expect(res.violations.map((v) => v.id)).toEqual([]);
  });
});

// ── saveable workspace layouts across monitor shapes ──────────────────────────
const SHAPES = [
  { name: '16:9', w: 1920, h: 1080 },
  { name: '21:9', w: 2560, h: 1080 },
  { name: '32:9', w: 3840, h: 1080 },
];
const openMenu = async (page) => {
  if (await page.getByTestId('layout-menu').count() === 0) await page.getByTestId('layout-menu-toggle').click();
  await expect(page.getByTestId('layout-menu')).toBeVisible();
};
const leftCol = (page) => page.getByTestId('workspace-split').evaluate((el) => parseFloat(getComputedStyle(el).gridTemplateColumns.split(' ')[0]));
const noOverflow = (page) => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1);
const centreWidth = (page) => page.locator('.workspace-center').first().evaluate((el) => el.getBoundingClientRect().width);

/** Panel sizes by seeding what the store itself persists, so a test starts from known proportions. */
const seedLayouts = (page, { sizes, layouts }) => page.addInitScript(({ sizes: s, layouts: l }) => {
  try {
    if (!window.localStorage.getItem('roop_workspace')) {
      window.localStorage.setItem('roop_workspace', JSON.stringify({ panelSizes: s, customLayouts: l, activeLayoutId: 'standard', railCollapsed: true }));
    }
  } catch { /* storage blocked */ }
}, { sizes, layouts });

for (const shape of SHAPES) {
  test.describe(`workspace layouts on a ${shape.name} window (${shape.w}x${shape.h})`, () => {
    test.use({ viewport: { width: shape.w, height: shape.h } });

    test('the window is classed, there is no page overflow, and the centre stage keeps its room', async ({ page }) => {
      await openTab(page, 'faceswap');
      await openMenu(page);
      await expect(page.getByTestId('layout-window-aspect')).toContainText(shape.name);
      await expect(page.getByTestId('layout-window-aspect')).toContainText(`${shape.w}×${shape.h}`);
      expect(await noOverflow(page), 'no horizontal page scroll').toBe(true);
      expect(await centreWidth(page), 'the centre stage keeps >= 360 px').toBeGreaterThanOrEqual(360);
    });

    test('save (tagged with this shape) -> rename -> bind Alt+1 -> the key switches to it', async ({ page }) => {
      await openTab(page, 'faceswap');
      const startLeft = await leftCol(page);
      await openMenu(page);
      await page.getByTestId('layout-save-open').click();
      await page.getByLabel('New layout name').fill(`Roto ${shape.name}`);
      await page.getByRole('button', { name: 'Save', exact: true }).click();
      await openMenu(page);
      const row = page.getByTestId('layout-row').filter({ hasText: `Roto ${shape.name}` });
      await expect(row).toHaveCount(1);
      await expect(row.getByTestId('layout-aspect-badge')).toHaveText(shape.name);
      await expect(page.getByTestId('active-custom-layout')).toContainText(`Roto ${shape.name}`);

      // Rename
      await row.getByRole('button', { name: /^Edit layout/ }).click();
      await row.getByLabel('Layout name').fill(`Desk ${shape.name}`);
      await row.getByTestId('layout-edit').getByRole('button', { name: 'Rename', exact: true }).click();
      await expect(page.getByTestId('layout-row').filter({ hasText: `Desk ${shape.name}` })).toHaveCount(1);
      // Bind slot 1
      const renamed = page.getByTestId('layout-row').filter({ hasText: `Desk ${shape.name}` });
      await renamed.getByLabel('Hotkey slot').selectOption('1');
      await expect(renamed.getByTestId('layout-hotkey-badge')).toHaveText(/Alt \+ 1|Alt\+1/);

      // Change the panels, then Alt+1 brings the saved proportions back.
      await page.keyboard.press('Escape');
      await page.getByTestId('workspace-split').getByRole('separator').first().focus().catch(() => {});
      await page.evaluate(() => {
        const w = JSON.parse(localStorage.getItem('roop_workspace') || '{}');
        w.panelSizes = { ...(w.panelSizes || {}), leftWidth: 520 };
        localStorage.setItem('roop_workspace', JSON.stringify(w));
      });
      await page.reload();
      await page.locator('nav[aria-label="Main"] button[aria-current="page"]').waitFor();
      await page.waitForTimeout(1200);
      expect(await leftCol(page), 'the reload picked up the changed panel').toBeGreaterThan(startLeft + 100);
      await page.locator('body').click({ position: { x: 4, y: 4 } });
      await page.keyboard.press('Alt+1');
      await expect.poll(() => leftCol(page), { message: 'Alt+1 restores the saved left panel' }).toBeLessThan(startLeft + 40);
      expect(await noOverflow(page)).toBe(true);
    });

    test('an empty slot says so instead of doing nothing', async ({ page }) => {
      await openTab(page, 'faceswap');
      await page.locator('body').click({ position: { x: 4, y: 4 } });
      await page.keyboard.press('Alt+4');
      await expect(page.getByRole('button', { name: /Dismiss notification: Nothing is bound to Alt\+4/ })).toBeVisible();
    });
  });
}

test.describe('a layout keeps its proportions on another monitor of the same shape', () => {
  test('saved on 3440x1440, switched to on 2560x1080: the side panels scale with the width', async ({ page }) => {
    const saved = [{
      id: 'custom_e2e_1', name: 'Wide roto', label: 'Wide roto', mode: 'default', aspect: '21:9', hotkey: 1,
      sizes: { leftWidth: 460, rightWidth: 400, bottomHeight: 300, queueWidth: 280 }, viewport: { w: 3440, h: 1440 },
    }];
    await seedLayouts(page, { sizes: { leftWidth: 260, rightWidth: 340, bottomHeight: 260, queueWidth: 280 }, layouts: saved });
    await page.setViewportSize({ width: 2560, height: 1080 });
    await openTab(page, 'faceswap');
    await page.locator('body').click({ position: { x: 4, y: 4 } });
    await page.keyboard.press('Alt+1');
    const want = Math.round(460 * 2560 / 3440);
    await expect.poll(() => leftCol(page), { message: `460 px at 3440 wide is ${want} px at 2560 wide` }).toBeGreaterThan(want - 6);
    expect(await leftCol(page)).toBeLessThan(want + 6);
  });

  test('layouts made for this shape are listed first, and none are hidden', async ({ page }) => {
    const mk = (id, name, aspect) => ({ id, name, label: name, mode: 'default', aspect, hotkey: null, sizes: { leftWidth: 300, rightWidth: 340, bottomHeight: 260 }, viewport: null });
    await seedLayouts(page, {
      sizes: { leftWidth: 260, rightWidth: 340, bottomHeight: 260, queueWidth: 280 },
      layouts: [mk('a', 'for sixteen', '16:9'), mk('b', 'for ultra', '21:9'), mk('c', 'for anything', 'any'), mk('d', 'for super', '32:9')],
    });
    await page.setViewportSize({ width: 3840, height: 1080 });
    await openTab(page, 'faceswap');
    await openMenu(page);
    const names = await page.getByTestId('layout-row').locator('button[aria-label^="Switch to layout"]').evaluateAll((els) => els.map((e) => e.getAttribute('aria-label').replace('Switch to layout ', '')));
    expect(names).toEqual(['for super', 'for anything', 'for sixteen', 'for ultra']);
  });

  test('the layout menu has no axe violations', async ({ page }) => {
    await openTab(page, 'faceswap');
    await openMenu(page);
    const res = await new AxeBuilder({ page }).include('[data-testid="layout-menu"]').analyze();
    expect(res.violations.map((v) => `${v.id}: ${v.nodes.map((n) => `${n.html.slice(0, 130)} -> ${(n.any[0]?.message || n.failureSummary || '').slice(0, 110)}`).join(' || ')}`)).toEqual([]);
  });
});

// ── the key remapper ──────────────────────────────────────────────────────────
async function openShortcutsTab(page) {
  await openTab(page, 'settings');
  await page.getByRole('tab', { name: /Keyboard Shortcuts/ }).click();
  await expect(page.getByTestId('keyboard-shortcuts-tab')).toBeVisible();
  await page.waitForTimeout(900);                       // the tab fades in; axe must not sample a half-faded colour
}
const key = (page, token) => page.getByTestId('ansi-keyboard').locator(`[data-key="${token}"]`).first();

test.describe('key remapper', () => {
  test('lights keys by family, per modifier layer', async ({ page }) => {
    await openShortcutsTab(page);
    await expect(page.getByTestId('ansi-keyboard')).toBeVisible();
    await expect(key(page, 'Space')).toHaveAttribute('data-group', 'transport');
    await expect(key(page, 'B')).toHaveAttribute('data-group', 'masking');
    await expect(key(page, 'F')).toHaveAttribute('data-group', 'viewport');
    await expect(key(page, '1'), '1 is a snapshot slot AND a flow-bar solo: Pipeline Soloing leads').toHaveAttribute('data-group', 'soloing');
    await expect(key(page, '5')).toHaveAttribute('data-group', 'soloing');
    await expect(key(page, '9'), '9 is only the contact sheet cell').toHaveAttribute('data-lit', 'yes');
    await expect(key(page, 'Q'), 'Q is Add-to-queue: a Workspace key').toHaveAttribute('data-group', 'workspace');
    await expect(key(page, 'Y'), 'Y is free').toHaveAttribute('data-lit', 'no');
    await page.locator('[data-layer-tab="alt"]').click();
    for (const n of ['1', '2', '3', '4', '5']) await expect(key(page, n), `Alt+${n}`).toHaveAttribute('data-group', 'workspace');
    await expect(key(page, 'Space'), 'nothing under Alt+Space').toHaveAttribute('data-lit', 'no');
    await page.locator('[data-layer-tab="shift"]').click();
    await expect(key(page, '/'), '? is Shift+/').toHaveAttribute('data-lit', 'yes');
    await page.getByTestId('ansi-keyboard').locator('[data-group-chip="transport"]').click();
    await page.locator('[data-layer-tab="plain"]').click();
    await expect(key(page, 'Space'), 'a family switched off in the legend goes dark').toHaveAttribute('data-lit', 'no');
  });

  test('a clash is flagged on the key and listed, and fixing it clears it', async ({ page }) => {
    await page.addInitScript(() => {
      try { localStorage.setItem('roop_keybinds_v1', JSON.stringify({ v: 1, overrides: { 'viewport.fit': ['S'] } })); } catch { /* blocked */ }
    });
    await openShortcutsTab(page);
    await expect(key(page, 'S')).toHaveAttribute('data-conflict', 'yes');
    await expect(page.getByTestId('keymap-conflicts')).toContainText('would fire two actions at once');
    await expect(page.locator('[data-conflict-combo="S"]')).toBeVisible();
    await page.locator('[data-action="viewport.fit"]').getByRole('button', { name: /Reset shortcut/ }).click();
    await expect(page.getByTestId('keymap-conflicts')).toHaveCount(0);
    await expect(key(page, 'S')).not.toHaveAttribute('data-conflict', 'yes');
  });

  test('a clicked key narrows the list to its actions', async ({ page }) => {
    await openShortcutsTab(page);
    await key(page, 'B').click();
    await expect(page.getByTestId('selected-key-filter')).toBeVisible();
    await expect(page.locator('[data-action="mask.brush"]')).toBeVisible();
    await expect(page.locator('[data-action="transport.playPause"]')).toHaveCount(0);
    await page.getByRole('button', { name: 'Show all' }).click();
    await expect(page.locator('[data-action="transport.playPause"]')).toBeVisible();
  });

  test('export writes a profile file; import replaces the keymap and reports clashes', async ({ page }) => {
    await page.addInitScript(() => {
      try { localStorage.setItem('roop_keybinds_v1', JSON.stringify({ v: 1, overrides: { 'viewport.fit': ['Y'] } })); } catch { /* blocked */ }
    });
    await openShortcutsTab(page);
    await page.getByLabel('Hotkey profile name').fill('Roto desk');
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByTestId('keymap-export').click()]);
    expect(download.suggestedFilename()).toBe('roop-keymap-roto-desk.json');
    const text = await (await import('node:fs')).promises.readFile(await download.path(), 'utf8');
    const profile = JSON.parse(text);
    expect(profile.format).toBe('roop-ultimate-keymap');
    expect(profile.name).toBe('Roto desk');
    expect(profile.overrides['viewport.fit']).toEqual(['Y']);
    await expect(page.getByTestId('keymap-notice')).toContainText('Exported');

    // A bad file says why and changes nothing.
    await page.getByTestId('keymap-import-file').setInputFiles({ name: 'x.json', mimeType: 'application/json', buffer: Buffer.from('{"nope":1}') });
    await expect(page.getByTestId('keymap-notice')).toContainText('not a Roop Ultimate keymap profile');
    await expect(page.locator('[data-action="viewport.fit"]')).toContainText('Y');

    // A clashing profile replaces it and the clash shows at once.
    const clashing = { format: 'roop-ultimate-keymap', v: 1, name: 'Clashy', overrides: { 'viewport.fit': ['S'], 'not.an.action': ['Q'] } };
    await page.getByTestId('keymap-import-file').setInputFiles({ name: 'c.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(clashing)) });
    await expect(page.getByTestId('keymap-notice')).toContainText('Imported “Clashy”');
    await expect(page.getByTestId('keymap-notice')).toContainText('1 this version does not know ignored');
    await expect(key(page, 'S')).toHaveAttribute('data-conflict', 'yes');
    await expect(page.locator('[data-action="viewport.fit"]'), 'the import REPLACED the old Y').not.toContainText('Y');

    // The defaults import clears it.
    const defaults = { format: 'roop-ultimate-keymap', v: 1, name: 'Defaults', overrides: {} };
    await page.getByTestId('keymap-import-file').setInputFiles({ name: 'd.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(defaults)) });
    await expect(page.getByTestId('keymap-notice')).toContainText('No conflicts');
    await expect(page.getByTestId('keymap-conflicts')).toHaveCount(0);
  });

  test('Ctrl+K "key remapper" opens it', async ({ page }) => {
    await openTab(page, 'faceswap');
    await page.locator('body').click({ position: { x: 4, y: 4 } });
    await page.keyboard.press('Control+k');
    const box = page.locator('input').filter({ visible: true }).last();
    await box.fill('key remapper');
    await page.getByText('Open the key remapper').first().click();
    await expect(page).toHaveURL(/#\/settings/);
    await expect(page.getByTestId('keyboard-shortcuts-tab')).toBeVisible();
    await expect(page.getByTestId('ansi-keyboard')).toBeVisible();
  });

  test('the keyboard tab has no axe violations', async ({ page }) => {
    await openShortcutsTab(page);
    const res = await new AxeBuilder({ page }).include('[data-testid="keyboard-shortcuts-tab"]').analyze();
    expect(res.violations.map((v) => `${v.id}: ${v.nodes.map((n) => `${n.html.slice(0, 130)} -> ${(n.any[0]?.message || n.failureSummary || '').slice(0, 110)}`).join(' || ')}`)).toEqual([]);
  });

  test('a saved layout is listed under the Alt+n actions with its slot reachable from the remapper', async ({ page }) => {
    await openShortcutsTab(page);
    for (let n = 1; n <= 5; n += 1) await expect(page.locator(`[data-action="workspace.layout${n}"]`)).toBeVisible();
  });
});

// ── the floating preview: Processing, A/B wipe, Play/Pause ────────────────────
const FRAME = { w: 640, h: 360 };
const PLATE = [40, 40, 200];       // the "before": blue
const SWAPPED = [200, 40, 40];     // the "after": red

async function openFaceSwapWithPictures(page) {
  const plate = makePng(FRAME.w, FRAME.h, () => PLATE);
  const swapped = makePng(FRAME.w, FRAME.h, () => SWAPPED);
  const previews = [];
  await page.route('**/api/target/preview*', (route) => route.fulfill({ status: 200, contentType: 'image/png', body: plate }));
  await page.route('**/api/preview', (route) => {
    if (route.request().method() !== 'POST') return route.fallback();
    try { previews.push(JSON.parse(route.request().postData() || '{}')); } catch { /* not JSON */ }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ image: `data:image/png;base64,${swapped.toString('base64')}`, faces: [], person_ids: [], kps: [], pose: [] }),
    });
  });
  await openTab(page, 'faceswap');
  await expect.poll(() => previews.length, { message: 'Face Swap rendered a preview', timeout: 20_000 }).toBeGreaterThan(0);
  return { previews };
}
const gotoTab = async (page, id) => {
  await page.goto(`/#/${id}`);
  await page.locator('nav[aria-label="Main"] button[aria-current="page"]').waitFor();
};
/** The RGB at (fx, fy) of the PiP canvas, fractions of its size. */
const pipAt = (page, fx, fy = 0.5) => page.getByTestId('pip-canvas').evaluate((c, [x, y]) => {
  const d = c.getContext('2d').getImageData(Math.floor(c.width * x), Math.floor(c.height * y), 1, 1).data;
  return [d[0], d[1], d[2]];
}, [fx, fy]);
const near = (a, b, tol = 24) => a.every((v, i) => Math.abs(v - b[i]) <= tol);

test.describe('floating preview: A/B, Play/Pause, Processing', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      new MutationObserver(() => {
        const d = document.querySelector('details.studio-tools');
        if (d && !d.open) d.open = true;
      }).observe(document, { childList: true, subtree: true });
    });
  });

  test('it follows you to Processing, and the glass window is named and axe-clean', async ({ page }) => {
    await openFaceSwapWithPictures(page);
    await page.getByTestId('pip-toggle').click();
    await gotoTab(page, 'processing');
    await expect(page.getByTestId('pip-window')).toBeVisible();
    await expect.poll(() => pipAt(page, 0.5).then((p) => near(p, SWAPPED)), { message: 'it shows the stage picture' }).toBe(true);
    await expect(page.getByTestId('pip-window')).toHaveClass(/pip-glass/);
    for (const name of [/Pause|Play/, /Before \/ after/]) await expect(page.getByTestId('pip-transport').getByRole('button', { name }).first()).toBeVisible();
    const res = await new AxeBuilder({ page }).include('[data-testid="pip-window"]').analyze();
    expect(res.violations.map((v) => `${v.id}: ${v.nodes.map((n) => `${n.html.slice(0, 130)} -> ${(n.any[0]?.message || n.failureSummary || '').slice(0, 110)}`).join(' || ')}`)).toEqual([]);
  });

  test('A/B: After -> wipe (before left, after right) -> Before -> After; the divider moves by key and by drag', async ({ page }) => {
    await openFaceSwapWithPictures(page);
    await page.getByTestId('pip-toggle').click();
    await gotoTab(page, 'batch');
    await expect(page.getByTestId('pip-window')).toBeVisible();
    await expect(page.getByTestId('pip-ab'), 'the stage offered its "before"').toBeEnabled({ timeout: 10_000 });
    await expect(page.getByTestId('pip-window')).toHaveAttribute('data-mode', 'after');
    expect(near(await pipAt(page, 0.5), SWAPPED)).toBe(true);

    await page.getByTestId('pip-ab').click();
    await expect(page.getByTestId('pip-window')).toHaveAttribute('data-mode', 'wipe');
    await expect(page.getByTestId('pip-wipe')).toHaveAttribute('aria-valuenow', '50');
    await expect.poll(async () => near(await pipAt(page, 0.25), PLATE) && near(await pipAt(page, 0.75), SWAPPED), { message: 'before on the left, after on the right' }).toBe(true);

    await page.getByTestId('pip-wipe').focus();
    await page.keyboard.press('End');
    await expect(page.getByTestId('pip-wipe')).toHaveAttribute('aria-valuenow', '100');
    await expect.poll(async () => near(await pipAt(page, 0.75), PLATE), { message: 'divider at the far right: all before' }).toBe(true);
    await page.keyboard.press('Home');
    await expect.poll(async () => near(await pipAt(page, 0.25), SWAPPED), { message: 'divider at the far left: all after' }).toBe(true);

    const handle = await page.getByTestId('pip-wipe').boundingBox();
    const picture = await page.getByTestId('pip-canvas').boundingBox();
    await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2);
    await page.mouse.down();
    await page.mouse.move(picture.x + picture.width * 0.8, handle.y + handle.height / 2, { steps: 6 });
    await page.mouse.up();
    const dragged = Number(await page.getByTestId('pip-wipe').getAttribute('aria-valuenow'));
    expect(dragged, 'dragging to 80% of the picture').toBeGreaterThan(70);
    expect(dragged).toBeLessThan(90);

    await page.getByTestId('pip-ab').click();
    await expect(page.getByTestId('pip-window')).toHaveAttribute('data-mode', 'before');
    await expect.poll(async () => near(await pipAt(page, 0.5), PLATE)).toBe(true);
    await page.getByTestId('pip-ab').click();
    await expect(page.getByTestId('pip-window')).toHaveAttribute('data-mode', 'after');
  });

  test('Play/Pause: steps the clip itself when Face Swap is not mounted, one request at a time, and stops on Pause', async ({ page }) => {
    const { previews } = await openFaceSwapWithPictures(page);
    await page.getByTestId('pip-toggle').click();
    await gotoTab(page, 'facemgr');
    await expect(page.getByTestId('pip-window')).toBeVisible();
    const play = page.getByTestId('pip-play');
    if (await play.isDisabled()) {
      // A one-frame target has nothing to step through: the button must say so rather than look broken.
      await expect(play).toHaveAttribute('title', /Nothing to play/);
      test.info().annotations.push({ type: 'note', description: 'the mock target has one frame; stepping was not exercised' });
      return;
    }
    const before = previews.length;
    await play.click();
    await expect(play).toHaveAttribute('aria-pressed', 'true');
    await expect.poll(() => previews.length, { message: 'it requested further frames' }).toBeGreaterThan(before + 1);
    const frames = previews.slice(before).map((b) => b.frame);
    expect(frames.every((f, i) => i === 0 || f > frames[i - 1]), `frames only move forward: ${frames.join(',')}`).toBe(true);
    await play.click();
    await expect(play).toHaveAttribute('aria-pressed', 'false');
    const stopped = previews.length;
    await page.waitForTimeout(1200);
    expect(previews.length - stopped, 'at most the request that was already in flight').toBeLessThanOrEqual(1);
  });
});
