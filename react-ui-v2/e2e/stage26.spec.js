import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { openTab, makePng } from './helpers.js';

// Stage 26, in a real Chromium against the production build with the mock server standing in for app/api.py:
//
//   scrub + math   dragging a label / a number scrubs it (Shift x10, Alt x0.1), typing takes an expression, a changed value
//                  gets a dot that resets it, and a whole scrub is ONE undo step
//   shortcuts      ? opens exactly one overlay; a remap made in Settings changes what a KEY DOES (not just a label), survives
//                  a reload, refuses a clash until told to replace, and the old key stops working
//   floating view  follows you to Face Manager, docks, resizes, persists, re-renders when the live source changes, and issues
//                  no request of its own just for the move
//
// The mock's pictures and detections are stand-ins, so none of this says what a real swap looks like. The expected scrub
// values come straight from numeric/scrubMath.js: 1 px = 0.25 on the -32..32 integer slider used below (max(step/4, range/600)).

const ERODE = 'Face detection threshold';       // visible on load, has a default (0.50); range and step are read from the DOM

const field = (page, label) => page.locator('label.block').filter({ has: page.getByText(label, { exact: true }) }).first();
const labelOf = (page, label) => field(page, label).getByText(label, { exact: true });
const readout = (page, label) => field(page, label).locator('button[title^="Click to type"]');
const dotOf = (page, label) => field(page, label).getByTestId('modified-dot');

/** The slider's own min / max / step, so expectations follow the control rather than a number copied from it. */
const spec = async (page, label) => field(page, label).getByRole('slider').evaluate((el) => ({
  min: Number(el.min), max: Number(el.max), step: Number(el.step),
}));
/** What dragging `px` pixels with a modifier does to `from`: numeric/scrubMath.js, restated (the maths is proven in node). */
const scrubbed = ({ min, max, step }, from, px, { shift = false, alt = false } = {}) => {
  const perPx = Math.max(step / 4, (max - min) / 600);
  const mult = shift && alt ? 1 : shift ? 10 : alt ? 0.1 : 1;
  const grid = alt && !shift && !(Number.isInteger(step) && step >= 1) ? Math.min(step, 0.001) : step;
  const acc = Math.min(max, Math.max(min, from + px * perPx * mult));
  return Number((min + Math.round((acc - min) / grid) * grid).toFixed(6));
};

/** Press on the element, hold the modifiers, nudge 3 px to cross the click threshold (that travel is not counted), then drag `dx`. */
async function scrubBy(page, locator, dx, { shift = false, alt = false } = {}) {
  await locator.scrollIntoViewIfNeeded();
  const box = await locator.boundingBox();
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;
  if (shift) await page.keyboard.down('Shift');
  if (alt) await page.keyboard.down('Alt');
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + Math.sign(dx) * 3, y);
  await page.mouse.move(x + Math.sign(dx) * 3 + dx, y);
  await page.mouse.up();
  if (alt) await page.keyboard.up('Alt');
  if (shift) await page.keyboard.up('Shift');
}

const value = async (page, label) => Number((await readout(page, label).innerText()).trim());
const near = (a, b) => Math.abs(a - b) < 1e-6;

test.describe('scrub, expressions, the modified dot', () => {
  test.beforeEach(async ({ page }) => {
    // The mock keeps what the app saves, so a value left by one test (the clamp test ends at the maximum) would be the next
    // test's starting point. These tests are about the control, not about persistence: the save goes nowhere.
    await page.route('**/api/settings', (route) => (route.request().method() === 'POST'
      ? route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }) : route.fallback()));
    await openTab(page, 'faceswap');
    await field(page, ERODE).scrollIntoViewIfNeeded();
  });

  test('dragging the label scrubs: plain, Shift x10, Alt x0.1', async ({ page }) => {
    const label = labelOf(page, ERODE);
    const sp = await spec(page, ERODE);
    expect(await label.evaluate((el) => getComputedStyle(el).cursor), 'the label shows the resize cursor').toBe('ew-resize');
    let v = await value(page, ERODE);
    expect(v).toBe(0.5);

    await scrubBy(page, label, 40);
    v = scrubbed(sp, v, 40);
    expect(await value(page, ERODE), 'plain drag').toBe(v);
    expect(v, 'the drag moved it').not.toBe(0.5);
    const afterPlain = v;
    await scrubBy(page, label, 4, { shift: true });
    v = scrubbed(sp, v, 4, { shift: true });
    expect(await value(page, ERODE), 'Shift is x10').toBe(v);
    await scrubBy(page, label, 40, { alt: true });
    const alt = scrubbed(sp, v, 40, { alt: true });
    expect(await value(page, ERODE), 'Alt is x0.1, on the 0.001 grid').toBe(alt);
    expect(Math.abs(alt - v), 'Alt moves a tenth as far as the same drag plain').toBeLessThan(Math.abs(afterPlain - 0.5) / 5);
    v = alt;
    await scrubBy(page, label, -40);
    expect(await value(page, ERODE), 'dragging left lowers it').toBe(scrubbed(sp, v, -40));
  });

  test('the number scrubs too, and a plain click on it still opens the editor', async ({ page }) => {
    const num = readout(page, ERODE);
    const sp = await spec(page, ERODE);
    await scrubBy(page, num, 20);
    expect(await value(page, ERODE)).toBe(scrubbed(sp, 0.5, 20));
    await expect(field(page, ERODE).getByLabel(`${ERODE} value`), 'a drag must not leave the editor open').toHaveCount(0);
    await num.click();
    await expect(field(page, ERODE).getByLabel(`${ERODE} value`)).toBeVisible();
  });

  test('the cursor stays at ew-resize for the whole drag, even off the label, and is released after', async ({ page }) => {
    const label = labelOf(page, ERODE);
    const box = await label.boundingBox();
    await page.mouse.move(box.x + 4, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + 60, box.y + box.height / 2);
    await page.mouse.move(box.x + 400, box.y + 200);       // far off the label
    expect(await page.evaluate(() => document.body.style.cursor)).toBe('ew-resize');
    await page.mouse.up();
    expect(await page.evaluate(() => document.body.style.cursor), 'released after the drag').toBe('');
  });

  test('a whole scrub is ONE undo step', async ({ page }) => {
    const label = labelOf(page, ERODE);
    await scrubBy(page, label, 40);
    expect(await value(page, ERODE)).not.toBe(0.5);
    await page.locator('body').click({ position: { x: 2, y: 2 } });     // focus nowhere in particular
    await page.keyboard.press('Control+z');
    await expect.poll(() => value(page, ERODE), { message: 'one Ctrl+Z undoes the whole drag' }).toBe(0.5);
  });

  test('typing takes expressions: arithmetic, +10%, a share of the range, and clamps', async ({ page }) => {
    const { min, max } = await spec(page, ERODE);
    const type = async (text) => {
      await readout(page, ERODE).click();
      const input = field(page, ERODE).getByLabel(`${ERODE} value`);
      await input.fill(text);
      await input.press('Enter');
    };
    await type('0.2*2');
    expect(await value(page, ERODE), '0.2*2').toBe(0.4);
    await type('50%');
    expect(near(await value(page, ERODE), Math.round((min + 0.5 * (max - min)) * 100) / 100), '50% is half the RANGE').toBe(true);
    await type('0.4');
    await type('+10%');
    expect(await value(page, ERODE), '+10% scales the current value (0.40 -> 0.44)').toBe(0.44);
    await type('-5');
    expect(await value(page, ERODE), '-5 is the number -5, clamped to the minimum').toBe(min);
    await type('9999');
    expect(await value(page, ERODE), 'clamped to the maximum').toBe(max);
    await type('(0.1+0.2)*2^2/4');
    expect(await value(page, ERODE), 'precedence and grouping').toBe(0.3);
  });

  test('an expression that does not parse keeps the editor open and says why; Escape restores the value', async ({ page }) => {
    await readout(page, ERODE).click();
    const input = field(page, ERODE).getByLabel(`${ERODE} value`);
    await input.fill('1/0');
    await input.press('Enter');
    await expect(input, 'Enter on a bad expression must not drop it').toBeVisible();
    await expect(input).toHaveAttribute('aria-invalid', 'true');
    await expect(page.getByRole('alert').filter({ hasText: /zero/i })).toHaveCount(1);
    await input.fill('alert(1)');
    await input.press('Enter');
    await expect(input).toHaveAttribute('aria-invalid', 'true');
    await input.press('Escape');
    await expect(input).toHaveCount(0);
    expect(await value(page, ERODE), 'the old value is back').toBe(0.5);
  });

  test('leaving the field on something unparseable puts the old value back (no silent garbage)', async ({ page }) => {
    await readout(page, ERODE).click();
    const input = field(page, ERODE).getByLabel(`${ERODE} value`);
    await input.fill('banana');
    await page.locator('body').click({ position: { x: 2, y: 2 } });
    await expect(input).toHaveCount(0);
    expect(await value(page, ERODE)).toBe(0.5);
  });

  test('a changed value shows the cyan dot; clicking it resets just that control', async ({ page }) => {
    await expect(dotOf(page, ERODE), 'at the default there is no dot').toHaveCount(0);
    await scrubBy(page, labelOf(page, ERODE), 40);
    await expect(dotOf(page, ERODE)).toHaveCount(1);
    const colour = await dotOf(page, ERODE).locator('span').evaluate((el) => getComputedStyle(el).backgroundColor);
    expect(colour, 'cyan').toBe('rgb(34, 211, 238)');
    await dotOf(page, ERODE).click();
    expect(await value(page, ERODE)).toBe(0.5);
    await expect(dotOf(page, ERODE)).toHaveCount(0);
  });

  test('the dot is not a tab stop and is hidden from the accessibility tree (Backspace on the slider still resets)', async ({ page }) => {
    await scrubBy(page, labelOf(page, ERODE), 40);
    const dot = dotOf(page, ERODE);
    await expect(dot).toHaveAttribute('tabindex', '-1');
    await expect(dot).toHaveAttribute('aria-hidden', 'true');
    const slider = field(page, ERODE).getByRole('slider');
    await slider.focus();
    await page.keyboard.press('Backspace');
    await expect.poll(() => value(page, ERODE)).toBe(0.5);
  });

  test('double-clicking the track still resets (unchanged)', async ({ page }) => {
    await scrubBy(page, labelOf(page, ERODE), 40);
    await field(page, ERODE).getByRole('slider').dblclick();
    await expect.poll(() => value(page, ERODE)).toBe(0.5);
  });

  test('Shift+drag on the TRACK is still the fine adjustment it always was', async ({ page }) => {
    const slider = field(page, ERODE).getByRole('slider');
    const box = await slider.boundingBox();
    await page.keyboard.down('Shift');
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 40, box.y + box.height / 2, { steps: 5 });
    await page.mouse.up();
    await page.keyboard.up('Shift');
    const v = await value(page, ERODE);
    expect(Math.abs(v - 0.5), 'a tenth-speed track drag moves far less than the track width would').toBeLessThan(0.1);
  });
});

// ── keyboard shortcuts ───────────────────────────────────────────────────────

const overlay = (page) => page.getByRole('dialog', { name: 'Keyboard shortcuts' });
const row = (page, id) => page.locator(`[data-action="${id}"]`);
const railWidth = (page) => page.locator('nav[aria-label="Main"]').evaluate((el) => el.getBoundingClientRect().width);

async function openShortcutsTab(page) {
  await openTab(page, 'settings');
  await page.getByRole('tab', { name: /Keyboard Shortcuts/ }).click();
  await expect(page.getByTestId('keyboard-shortcuts-tab')).toBeVisible();
}

test.describe('keyboard shortcuts', () => {
  test('? opens exactly ONE overlay, with every category; search filters; Esc closes; Ctrl+/ opens it too', async ({ page }) => {
    await openTab(page, 'faceswap');
    await page.locator('body').click({ position: { x: 2, y: 2 } });
    await page.keyboard.press('?');
    await expect(overlay(page)).toBeVisible();
    // Face Swap used to carry a second, hand-typed copy that also answered to "?" - two overlays at once.
    await expect(page.getByRole('dialog'), 'one dialog, not two').toHaveCount(1);
    for (const cat of ['Transport', 'Viewport & Scopes', 'Mask & Roto', 'Timeline', 'Snapshots']) {
      await expect(overlay(page).getByRole('region', { name: cat }), cat).toBeVisible();
    }
    await expect(row(page, 'transport.playPause').locator('kbd')).toHaveText('Space');
    await overlay(page).getByLabel('Search shortcuts').fill('split');
    await expect(row(page, 'viewport.splitView')).toBeVisible();
    await expect(row(page, 'transport.addToQueue')).toHaveCount(0);
    await overlay(page).getByLabel('Search shortcuts').fill('ctrl');
    await expect(row(page, 'app.palette'), 'searching by key text').toBeVisible();
    await page.keyboard.press('Escape');
    await expect(overlay(page)).toHaveCount(0);
    await page.keyboard.press('Control+/');
    await expect(overlay(page)).toBeVisible();
    await page.keyboard.press('Control+/');
    await expect(overlay(page), 'the same key toggles it closed').toHaveCount(0);
  });

  test('typing "?" into a text field does not open it', async ({ page }) => {
    await openTab(page, 'settings');
    const box = page.getByRole('searchbox').first().or(page.getByPlaceholder(/search/i).first());
    await box.click();
    await page.keyboard.type('what?');
    await expect(overlay(page)).toHaveCount(0);
  });

  test('G opens the loupe, and M no longer does (it used to open it AND set a timeline marker)', async ({ page }) => {
    await openTab(page, 'faceswap');
    await page.locator('body').click({ position: { x: 2, y: 2 } });
    // The toolbar button always reads "Loupe (..x)"; it is lit (cyan) only while the loupe is on.
    const loupe = page.getByTitle(/Floating Magnifier Loupe/);
    await expect(loupe).not.toHaveClass(/text-cyan-300/);
    await page.keyboard.press('m');
    await page.waitForTimeout(300);
    await expect(loupe, 'M must not switch the loupe on').not.toHaveClass(/text-cyan-300/);
    await page.keyboard.press('g');
    await expect(loupe, 'G is the loupe').toHaveClass(/text-cyan-300/);
    await page.keyboard.press('g');
    await expect(loupe).not.toHaveClass(/text-cyan-300/);
    await expect(loupe, 'the tooltip no longer promises M').toHaveAttribute('title', /\(G\)/);
  });

  test('a remap changes what a KEY DOES: the new key works, the old one stops, and it survives a reload', async ({ page }) => {
    await openShortcutsTab(page);
    await row(page, 'app.toggleRail').getByRole('button', { name: /^Change shortcut for/ }).click();
    await expect(page.getByTestId('shortcut-recording')).toBeVisible();
    await page.keyboard.press('Control+Shift+L');
    await expect(row(page, 'app.toggleRail').locator('kbd')).toHaveText('Ctrl + Shift + L');
    await expect(page.getByTestId('shortcut-changed-count')).toHaveText('1 changed');

    await page.goto('/#/home');
    await page.locator('nav[aria-label="Main"] button[aria-current="page"]').waitFor();
    await page.locator('body').click({ position: { x: 300, y: 300 } });
    const w0 = await railWidth(page);
    await page.keyboard.press('Control+b');
    await page.waitForTimeout(500);
    expect(await railWidth(page), 'the OLD key (Ctrl+B) no longer toggles the rail').toBe(w0);
    await page.keyboard.press('Control+Shift+L');
    await expect.poll(() => railWidth(page), { message: 'the NEW key toggles it' }).not.toBe(w0);

    await page.reload();
    await page.locator('nav[aria-label="Main"] button[aria-current="page"]').waitFor();
    await page.locator('body').click({ position: { x: 300, y: 300 } });
    const w1 = await railWidth(page);
    await page.keyboard.press('Control+Shift+L');
    await expect.poll(() => railWidth(page), { message: 'still remapped after a reload' }).not.toBe(w1);

    // back to defaults, through the UI
    await openShortcutsTab(page);
    await row(page, 'app.toggleRail').getByRole('button', { name: /^Reset shortcut for/ }).click();
    await expect(row(page, 'app.toggleRail').locator('kbd')).toHaveText('Ctrl + B');
    await expect(page.getByTestId('shortcut-changed-count')).toHaveText('All defaults');
  });

  test('a key another action already uses is refused until you choose to replace it', async ({ page }) => {
    await openShortcutsTab(page);
    await row(page, 'viewport.splitView').getByRole('button', { name: /^Change shortcut for/ }).click();
    await page.keyboard.press('q');
    const conflict = page.getByTestId('shortcut-conflict');
    await expect(conflict).toBeVisible();
    await expect(conflict).toContainText('Add to the batch queue');
    await expect(row(page, 'viewport.splitView').locator('kbd'), 'nothing is bound while the clash stands').toHaveText('S');
    await page.getByTestId('shortcut-conflict-replace').click();
    await expect(row(page, 'viewport.splitView').locator('kbd')).toHaveText('Q');
    await expect(row(page, 'transport.addToQueue'), 'the other action gave the key up').toContainText('unbound');
    await expect(page.getByTestId('shortcut-changed-count')).toHaveText('2 changed');
    await page.getByRole('button', { name: 'Reset all' }).click();
    await expect(page.getByTestId('shortcut-changed-count')).toHaveText('All defaults');
    await expect(row(page, 'transport.addToQueue').locator('kbd')).toHaveText('Q');
  });

  test('reserved keys are refused with the reason, and a recording swallows the key (Ctrl+K does not open the palette)', async ({ page }) => {
    await openShortcutsTab(page);
    await row(page, 'viewport.splitView').getByRole('button', { name: /^Change shortcut for/ }).click();
    await page.keyboard.press('Tab');
    await expect(page.getByTestId('shortcut-recording')).toContainText('Reserved');
    await page.keyboard.press('Control+k');
    await expect(page.getByTestId('shortcut-conflict'), 'Ctrl+K is the palette: a clash, not a palette').toContainText('Command palette');
    await expect(page.getByRole('dialog'), 'no palette opened behind the recording').toHaveCount(0);
    await page.getByRole('button', { name: 'Cancel' }).click();
    await expect(page.getByTestId('shortcut-conflict')).toHaveCount(0);
  });

  test('a stored keymap with a bad entry does not break the app', async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem('roop_keybinds_v1', JSON.stringify({ v: 1, overrides: { 'nope.gone': ['X'], 'viewport.fit': ['Escape', 'Mod+'], 'viewport.splitView': 7 } }));
    });
    await openTab(page, 'faceswap');
    await page.locator('body').click({ position: { x: 2, y: 2 } });
    await page.keyboard.press('?');
    await expect(row(page, 'viewport.fit').locator('kbd'), 'junk is dropped and the default stands').toHaveText('F');
  });
});

// ── the floating preview ─────────────────────────────────────────────────────

const FRAME = { w: 640, h: 360 };
const PLATE = [40, 40, 200];                // blue: the raw frame
const SWAPPED = [200, 40, 40];              // red: the swapped picture the PiP must show

/** Face Swap with real PNG pictures (the mock's SVGs never decode into a bitmap); records every /api/preview body. */
async function openFaceSwapWithPictures(page) {
  const plate = makePng(FRAME.w, FRAME.h, () => PLATE);
  const swapped = makePng(FRAME.w, FRAME.h, () => SWAPPED);
  const previews = [];
  const stateReads = [];
  await page.route('**/api/target/preview*', (route) => route.fulfill({ status: 200, contentType: 'image/png', body: plate }));
  await page.route('**/api/state', (route) => { stateReads.push(Date.now()); return route.fallback(); });
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
  return { previews, stateReads };
}

const pip = (page) => page.getByTestId('pip-window');
const gotoTab = async (page, id) => {
  await page.goto(`/#/${id}`);
  await page.locator('nav[aria-label="Main"] button[aria-current="page"]').waitFor();
};
/** The RGB of the pixel at the middle of the PiP's canvas. */
const pipCentre = (page) => page.getByTestId('pip-canvas').evaluate((c) => {
  const d = c.getContext('2d').getImageData(Math.floor(c.width / 2), Math.floor(c.height / 2), 1, 1).data;
  return [d[0], d[1], d[2]];
});

test.describe('floating preview (PiP)', () => {
  test('it is off until switched on; on Face Swap it shows only while the stage is scrolled out of view', async ({ page }) => {
    await openFaceSwapWithPictures(page);
    const stage = page.locator('.preview-stage').first();
    await stage.scrollIntoViewIfNeeded();                  // in this layout the stage starts well below the fold
    await expect(pip(page)).toHaveCount(0);
    await expect(page.getByTestId('pip-toggle')).toHaveAttribute('aria-pressed', 'false');
    await page.getByTestId('pip-toggle').click();
    await expect(page.getByTestId('pip-toggle')).toHaveAttribute('aria-pressed', 'true');
    await page.waitForTimeout(400);
    await expect(pip(page), 'the stage is on screen, so no second copy').toHaveCount(0);
    await page.evaluate(() => window.scrollTo(0, 0));
    await expect(pip(page), 'scrolled away from the stage: the corner copy appears').toBeVisible();
    await expect.poll(() => pipCentre(page)).toEqual([200, 40, 40]);
    await stage.scrollIntoViewIfNeeded();
    await expect(pip(page), 'and goes again when the stage is back in view').toHaveCount(0);
  });

  test('it follows you to Face Manager / Batch / Settings with the stage\'s own picture, and not to Home', async ({ page }) => {
    const { previews } = await openFaceSwapWithPictures(page);
    await page.getByTestId('pip-toggle').click();
    const before = previews.length;
    await gotoTab(page, 'facemgr');
    await expect(pip(page)).toBeVisible();
    // the picture is the SWAPPED one (red), not the raw plate (blue)
    await expect.poll(() => pipCentre(page)).toEqual([200, 40, 40]);
    const box = await pip(page).boundingBox();
    expect(Math.round(box.width), '280 wide').toBe(280);
    expect(Math.round(box.height), '158 of picture + the 28 px bar').toBe(186);
    await page.waitForTimeout(1500);
    expect(previews.length, 'moving to another tab asked the backend for nothing').toBe(before);

    await gotoTab(page, 'batch');
    await expect(pip(page)).toBeVisible();
    await gotoTab(page, 'settings');
    await expect(pip(page)).toBeVisible();
    await gotoTab(page, 'home');
    await expect(pip(page), 'not on Home').toHaveCount(0);
    await gotoTab(page, 'facemgr');
    await expect(pip(page), 'and back again').toBeVisible();
  });

  test('it docks to corners, drags, resizes keeping the picture shape, and remembers all of it', async ({ page }) => {
    await openFaceSwapWithPictures(page);
    await page.getByTestId('pip-toggle').click();
    await gotoTab(page, 'facemgr');
    await expect(pip(page)).toBeVisible();
    await expect(pip(page)).toHaveAttribute('data-corner', 'bottom-right');
    const vp = page.viewportSize();
    let box = await pip(page).boundingBox();
    expect(Math.round(box.x + box.width), '16 px from the right edge').toBe(vp.width - 16);
    expect(Math.round(box.y + box.height), '16 px from the bottom edge').toBe(vp.height - 16);

    // drag the bar to the top-left: dropped near the corner, it docks there
    const cap = await page.getByTestId('pip-caption').boundingBox();
    await page.mouse.move(cap.x + 10, cap.y + 4);
    await page.mouse.down();
    await page.mouse.move(200, 120, { steps: 6 });
    await page.mouse.move(60, 40, { steps: 6 });
    await page.mouse.up();
    await expect(pip(page)).toHaveAttribute('data-corner', 'top-left');
    box = await pip(page).boundingBox();
    expect([Math.round(box.x), Math.round(box.y)]).toEqual([16, 16]);

    // the dock button walks the corners
    await page.getByTestId('pip-dock').click();
    await expect(pip(page)).toHaveAttribute('data-corner', 'top-right');
    await page.getByTestId('pip-dock').click();
    await expect(pip(page)).toHaveAttribute('data-corner', 'bottom-right');

    // dropped away from every corner it floats where it was put
    const cap2 = await page.getByTestId('pip-caption').boundingBox();
    await page.mouse.move(cap2.x + 10, cap2.y + 4);
    await page.mouse.down();
    await page.mouse.move(vp.width / 2, vp.height / 2, { steps: 8 });
    await page.mouse.up();
    await expect(pip(page)).toHaveAttribute('data-corner', 'free');

    // dock top-left again, then resize from the grip (bottom-right of a left-docked window)
    await page.getByTestId('pip-dock').click();
    const corner = await pip(page).getAttribute('data-corner');
    expect(corner).not.toBe('free');
    while ((await pip(page).getAttribute('data-corner')) !== 'top-left') await page.getByTestId('pip-dock').click();
    const grip = await page.getByTestId('pip-resize').boundingBox();
    const gx = grip.x + grip.width / 2;
    const gy = grip.y + grip.height / 2;
    await page.mouse.move(gx, gy);
    await page.mouse.down();
    await page.mouse.move(gx + 60, gy + 30, { steps: 5 });
    await page.mouse.move(gx + 120, gy + 60, { steps: 5 });
    await page.mouse.up();
    box = await pip(page).boundingBox();
    expect(Math.round(box.width), 'grew by the drag').toBe(400);
    expect(Math.abs((box.height - 28) / box.width - 9 / 16), 'the picture keeps its 16:9 shape').toBeLessThan(0.01);

    // keyboard: the resize grip takes arrows
    await page.getByTestId('pip-resize').focus();
    await page.keyboard.press('ArrowLeft');
    box = await pip(page).boundingBox();
    expect(Math.round(box.width), 'one arrow = 16 px').toBe(384);

    // and it is remembered
    const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('roop_pip_v1')));
    expect(saved).toMatchObject({ enabled: true, w: 384, corner: 'top-left' });
    await page.reload();
    await page.locator('nav[aria-label="Main"] button[aria-current="page"]').waitFor();
    await expect(page.getByTestId('pip-toggle'), 'still switched on after a reload').toHaveAttribute('aria-pressed', 'true');
  });

  test('the close button hides it and switches it off; the header button brings it back', async ({ page }) => {
    await openFaceSwapWithPictures(page);
    await page.getByTestId('pip-toggle').click();
    await gotoTab(page, 'facemgr');
    await page.getByTestId('pip-close').click();
    await expect(pip(page)).toHaveCount(0);
    await expect(page.getByTestId('pip-toggle')).toHaveAttribute('aria-pressed', 'false');
  });

  test('Morph Lab Preview with the PiP on stays in Face Manager and the PiP re-renders: fresh request id, later version, the live source', async ({ page }) => {
    const { previews, stateReads } = await openFaceSwapWithPictures(page);
    await page.getByTestId('pip-toggle').click();
    await gotoTab(page, 'facemgr');
    await expect(pip(page)).toBeVisible();
    const first = previews.at(-1);
    const reads0 = stateReads.length;
    const n0 = previews.length;

    await page.getByTestId('facemgr-tab-morph').click();
    await page.getByTestId('morph-pick-lib:Alice.fsz').click();
    await page.getByTestId('morph-pick-lib:Bob.fsz').click();
    await page.getByTestId('morph-preview').click();

    await expect.poll(() => previews.length, { message: 'the PiP asked for the picture again', timeout: 10_000 }).toBe(n0 + 1);
    await expect(page, 'it did not jump to Face Swap').toHaveURL(/#\/facemgr$/);
    const again = previews.at(-1);
    expect(again.processing_selection.request_id, 'a fresh request id').not.toBe(first.processing_selection.request_id);
    expect(again.processing_selection.selection_version, 'a later selection version').toBeGreaterThan(first.processing_selection.selection_version);
    expect(again.frame, 'the same frame').toBe(first.frame);
    expect(again.index, 'the same target').toBe(first.index);
    expect(stateReads.length, 'it read /api/state for the live source').toBeGreaterThan(reads0);
    await page.waitForTimeout(1800);
    expect(previews.length, 'one refresh, not a loop').toBe(n0 + 1);
    await expect(page.getByTestId('pip-error')).toHaveCount(0);
  });

  test('without the PiP, Morph Lab Preview still opens Face Swap (unchanged)', async ({ page }) => {
    await openFaceSwapWithPictures(page);
    await gotoTab(page, 'facemgr');
    await page.getByTestId('facemgr-tab-morph').click();
    await page.getByTestId('morph-pick-lib:Alice.fsz').click();
    await page.getByTestId('morph-pick-lib:Bob.fsz').click();
    await page.getByTestId('morph-preview').click();
    await expect(page).toHaveURL(/#\/faceswap$/);
  });

  test('a refresh that fails says so inside the window and leaves the picture alone', async ({ page }) => {
    await openFaceSwapWithPictures(page);
    await page.getByTestId('pip-toggle').click();
    await gotoTab(page, 'facemgr');
    await expect(pip(page)).toBeVisible();
    await page.unroute('**/api/preview');
    await page.route('**/api/preview', (route) => (route.request().method() === 'POST'
      ? route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ message: 'a render is running' }) })
      : route.fallback()));
    await page.getByTestId('pip-refresh').click();
    await expect(page.getByTestId('pip-error')).toContainText('a render is running');
    expect(await pipCentre(page), 'the last good picture is still there').toEqual([200, 40, 40]);
  });

  test('the PiP has no axe violations, and its controls are named', async ({ page }) => {
    await openFaceSwapWithPictures(page);
    await page.getByTestId('pip-toggle').click();
    await gotoTab(page, 'facemgr');
    await expect(pip(page)).toBeVisible();
    for (const name of [/Move the floating preview/, /Re-render the floating preview/, /Dock to the next corner/, /Back to Face Swap/, /Close the floating preview/, /Resize the floating preview/]) {
      await expect(pip(page).getByRole('button', { name }), String(name)).toHaveCount(1);
    }
    const res = await new AxeBuilder({ page }).include('[data-testid="pip-window"]').analyze();
    expect(res.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(', ')}`)).toEqual([]);
  });

  test('Back to Face Swap returns; from there the same button brings the stage into view and the window hides', async ({ page }) => {
    await openFaceSwapWithPictures(page);
    await page.getByTestId('pip-toggle').click();
    await gotoTab(page, 'facemgr');
    await page.getByTestId('pip-return').click();
    await expect(page).toHaveURL(/#\/faceswap$/);
    // back on Face Swap at the top of the page the stage is still below the fold, so the window stays - until asked
    await expect(pip(page)).toBeVisible();
    await page.getByTestId('pip-return').click();
    await expect(pip(page), 'the stage is in view, so the corner copy goes').toHaveCount(0);
  });
});
